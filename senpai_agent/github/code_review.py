"""Review the exact proposed merge with a fresh smart-profile agent."""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from fnmatch import fnmatchcase
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
# Merge reviews may reject one PR this many times; later findings become notes.
DEFAULT_MAX_BLOCKING_REVIEW_ROUNDS = 2
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
MERGE_REVIEW_MARKER_PREFIX = "<!-- senpai-merge-review:"
REJECTED_VERDICT_LINE = "**Verdict: Rejected**"
ReviewText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CodeReviewRejected(WorkflowPreconditionError):
    """A completed, valid review found actionable code-quality problems."""


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str | None = Field(
        description=(
            "Repository path of the file the finding is about, or null when it "
            "concerns the change as a whole."
        ),
    )
    finding: ReviewText = Field(
        description="The problem and its actionable fix, with line references where useful.",
    )
    safety: bool = Field(
        default=False,
        description=(
            "True only for exposed credentials, destructive commands, or a "
            "violation of the target program's rules or file boundaries. Such "
            "findings block wherever they are and are never waived."
        ),
    )

    def render(self) -> str:
        return f"{self.path}: {self.finding}" if self.path else self.finding


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
    findings: list[ReviewFinding] = Field(
        description=(
            "Blocking issues only, each with its file or line reference and an "
            "actionable fix. Include any reason the review is incomplete. "
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
            ("Blocking findings", [finding.render() for finding in self.findings]),
            ("Non-blocking notes", self.notes),
        ):
            if items:
                sections.append(
                    f"**{title}**\n\n" + "\n".join(f"- {item}" for item in items)
                )
        return "\n\n".join(sections)


def in_deliverable(path: str | None, deliverable_paths: Sequence[str]) -> bool:
    """A finding with no path concerns the whole change and stays in scope."""

    if not deliverable_paths or path is None:
        return True
    return any(
        fnmatchcase(path, pattern) or path == pattern.rstrip("/")
        or path.startswith(pattern.rstrip("/") + "/")
        for pattern in deliverable_paths
    )


def scope_verdict(
    verdict: CodeQualityVerdict,
    *,
    deliverable_paths: Sequence[str],
    prior_blocking_rounds: int,
    max_blocking_rounds: int,
) -> CodeQualityVerdict:
    """Keep blocking findings to the deliverable and to a bounded number of rounds.

    Findings on files the assignment did not declare as its deliverable become
    notes. Once a PR has been rejected ``max_blocking_rounds`` times, the
    remaining findings become notes too and the merge proceeds. Safety
    findings are exempt from both: they block wherever they are.
    """

    if verdict.approved:
        return verdict
    blocking: list[ReviewFinding] = []
    notes = list(verdict.notes)
    for finding in verdict.findings:
        if finding.safety or in_deliverable(finding.path, deliverable_paths):
            blocking.append(finding)
        else:
            notes.append(f"Outside the deliverable: {finding.render()}")
    summary = verdict.review_summary
    if blocking and prior_blocking_rounds >= max_blocking_rounds and not any(
        finding.safety for finding in blocking
    ):
        notes.extend(
            f"After {prior_blocking_rounds} blocking rounds: {finding.render()}"
            for finding in blocking
        )
        summary += (
            f"\n\nThis PR has had {prior_blocking_rounds} blocking review rounds, "
            f"the configured limit of {max_blocking_rounds}; the remaining findings "
            "are posted as notes and the merge proceeds."
        )
        blocking = []
    if not blocking and verdict.findings:
        summary = (
            summary
            if "the merge proceeds" in summary
            else summary + "\n\nEvery finding lies outside the declared deliverable; "
            "they are posted as notes and the merge proceeds."
        )
    return CodeQualityVerdict(
        approved=not blocking,
        review_summary=summary,
        findings=blocking,
        notes=notes,
    )


def count_blocking_rounds(record: PullRequest) -> int:
    """Count the merge reviews that rejected this PR so far."""

    return sum(
        1
        for entry in record.issue_comments
        if MERGE_REVIEW_MARKER_PREFIX in str(entry.get("body") or "")
        and REJECTED_VERDICT_LINE in str(entry.get("body") or "")
    )


def review_code_quality(
    pull: PullRequestSnapshot,
    *,
    base_sha: str,
    token: SecretStr,
    config: RunnerConfig,
    deliverable_paths: Sequence[str] = (),
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
                deliverable_paths=deliverable_paths,
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
        verdict = scope_verdict(
            verdict,
            deliverable_paths=deliverable_paths,
            prior_blocking_rounds=count_blocking_rounds(record),
            max_blocking_rounds=config.max_blocking_review_rounds,
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
    deliverable_paths: Sequence[str] = (),
) -> str:
    discussion = render_review_context(
        record, int(context_chars * DISCUSSION_CONTEXT_SHARE),
    )
    deliverable = (
        "\n".join(f"- `{path}`" for path in deliverable_paths)
        if deliverable_paths
        else "The assignment declares no deliverable paths; every changed file is in scope."
    )
    prompt = render_prompt(
        CODE_QUALITY_REVIEW_PROMPT,
        BASE_SHA=base_sha,
        HEAD_SHA=pull.head_sha,
        MERGE_BASE_SHA=merge_base,
        DIFF=clip_text(diff, context_chars - len(discussion), DIFF_CLIPPED_NOTICE),
        PR_CONTEXT=discussion,
        DELIVERABLE=deliverable,
        MINUTES=str(minutes),
    )
    if retry:
        prompt += "\n\n" + CODE_QUALITY_REVIEW_RETRY_PROMPT
    return prompt + "\n"
