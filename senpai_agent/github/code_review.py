"""Review the exact proposed merge with a fresh smart-profile agent."""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    model_validator,
)

from senpai_agent.git_transport import (
    github_repository_url,
    isolated_bare_repository,
    run_git,
)
from senpai_agent.github.pull_requests import PullRequest, fetch_pull_request
from senpai_agent.github.rendering import clip_text, render_review_context
from senpai_agent.github.workflow.errors import WorkflowPreconditionError
from senpai_agent.github.workflow.responses import PullRequestSnapshot
from senpai_agent.PROMPTS import (
    CODE_QUALITY_REVIEW_PROMPT,
    CODE_QUALITY_REVIEW_RETRY_PROMPT,
    render_prompt,
)

if TYPE_CHECKING:
    from senpai_agent.openhands_runner import RunnerConfig


MERGE_COMPLETION_RESERVE_SECONDS = 60
# The first review gets this share of its window; a compact retry gets the rest.
FULL_REVIEW_TIME_SHARE = 2 / 3
# A compact retry sees this share of the first review's context.
COMPACT_REVIEW_CONTEXT_SHARE = 1 / 4
# PR discussion gets this share of a review's context; the diff gets the rest.
DISCUSSION_CONTEXT_SHARE = 1 / 4
DIFF_CLIPPED_NOTICE = (
    "\n[Diff truncated to fit the review budget. The summary above lists every "
    "changed file; inspect the rest with git.]"
)
ReviewText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CodeReviewRejected(WorkflowPreconditionError):
    """A completed, valid review found actionable code-quality problems."""


class CodeQualityVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approved: bool = Field(
        description=(
            "Approve when the review is complete and no blocking finding remains. "
            "Notes never block."
        ),
    )
    review_summary: ReviewText = Field(
        description=(
            "A concise conclusion with clear, actionable feedback where appropriate. "
            "If blocked, explain what must change and why."
        ),
    )
    findings: list[ReviewText] = Field(
        description=(
            "Blocking issues only, each with an actionable fix and file or line "
            "references where useful. Include any reason the review is incomplete. "
            "Empty on approval."
        ),
    )
    notes: list[ReviewText] = Field(
        description=(
            "Non-blocking suggestions, such as documentation length or wording, "
            "naming, CLI flags, formatting, or minor simplifications."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_findings(self) -> Self:
        if self.approved and self.findings:
            raise ValueError("an approval must have no blocking findings")
        if not self.approved and not self.findings:
            raise ValueError("a rejection must explain at least one blocking finding")
        return self

    def report(self) -> str:
        sections = [self.review_summary]
        for title, items in (
            ("Blocking findings", self.findings),
            ("Non-blocking notes", self.notes),
        ):
            if items:
                sections.append(
                    f"**{title}**\n\n" + "\n".join(f"- {item}" for item in items)
                )
        return "\n\n".join(sections)


def review_code_quality(
    pull: PullRequestSnapshot,
    *,
    base_sha: str,
    token: SecretStr,
    config: RunnerConfig,
) -> CodeQualityVerdict:
    from senpai_agent.openhands_runner import run_openhands, supervisor_config

    started_at = time.time()
    review_deadline = min(
        started_at + config.timeout_seconds,
        config.delegation_deadline_epoch or float("inf"),
    ) - MERGE_COMPLETION_RESERVE_SECONDS
    if review_deadline <= started_at:
        raise WorkflowPreconditionError(
            "insufficient time remains for code quality review and merge feedback"
        )

    record = fetch_pull_request(config.github_repo, pull.number, token=token)
    if record.manifest_entry.head_sha != pull.head_sha:
        raise WorkflowPreconditionError(
            "PR head changed while loading code review context"
        )

    with isolated_bare_repository() as repository:
        run_git(
            repository,
            "fetch", "--no-tags", github_repository_url(config.github_repo),
            base_sha, pull.head_sha,
            token=token,
        )
        for commit in (base_sha, pull.head_sha):
            if run_git(repository, "rev-parse", f"{commit}^{{commit}}") != commit:
                raise WorkflowPreconditionError(
                    "review source does not match the requested commits"
                )
        merge_base = run_git(repository, "merge-base", base_sha, pull.head_sha)
        diff = run_git(
            repository, "diff", "--stat=200", "--patch", merge_base, pull.head_sha,
        )
        reviewer = replace(
            supervisor_config(config), workspace=repository, delegation_task_id=None,
        )

        def attempt(
            context_chars: int, deadline: float, *, retry: bool,
        ) -> CodeQualityVerdict | None:
            prompt = review_prompt(
                pull, base_sha=base_sha, merge_base=merge_base, record=record,
                diff=diff, context_chars=context_chars,
                minutes=max(1, int((deadline - time.time()) // 60)), retry=retry,
            )
            results: list[CodeQualityVerdict] = []
            status = run_openhands(
                prompt,
                replace(
                    reviewer,
                    delegation_deadline_epoch=deadline,
                    conversation_id=uuid.uuid4() if retry else reviewer.conversation_id,
                ),
                response_schema=CodeQualityVerdict,
                on_structured_result=results.append,
            )
            return results[0] if status == 0 and results else None

        # The compaction trigger bounds the model's usable window in tokens; a
        # matching character budget keeps the embedded context to about a third.
        context_chars = config.compaction_trigger_tokens
        first_deadline = (
            started_at + (review_deadline - started_at) * FULL_REVIEW_TIME_SHARE
        )
        verdict = attempt(context_chars, first_deadline, retry=False)
        if verdict is None:
            print(
                "SENPAI_CODE_REVIEW_RETRY "
                f"pr={pull.number} elapsed_seconds={time.time() - started_at:.0f}",
                file=sys.stderr,
                flush=True,
            )
            verdict = attempt(
                int(context_chars * COMPACT_REVIEW_CONTEXT_SHARE),
                review_deadline,
                retry=True,
            )
        if verdict is None:
            raise WorkflowPreconditionError(
                "code quality reviewer did not complete with a verdict in "
                f"{(time.time() - started_at) / 60:.0f} minutes, including a "
                "compact retry"
            )
        if not verdict.approved:
            raise CodeReviewRejected(
                f"Code quality review blocked PR #{pull.number}.\n\n{verdict.report()}"
            )
        return verdict


def review_prompt(
    pull: PullRequestSnapshot,
    *,
    base_sha: str,
    merge_base: str,
    record: PullRequest,
    diff: str,
    context_chars: int,
    minutes: int,
    retry: bool,
) -> str:
    discussion = render_review_context(
        record, int(context_chars * DISCUSSION_CONTEXT_SHARE),
    )
    prompt = render_prompt(
        CODE_QUALITY_REVIEW_PROMPT,
        BASE_SHA=base_sha,
        HEAD_SHA=pull.head_sha,
        MERGE_BASE_SHA=merge_base,
        DIFF=clip_text(diff, context_chars - len(discussion), DIFF_CLIPPED_NOTICE),
        PR_CONTEXT=discussion,
        MINUTES=str(minutes),
    )
    if retry:
        prompt += "\n\n" + CODE_QUALITY_REVIEW_RETRY_PROMPT
    return prompt + "\n"
