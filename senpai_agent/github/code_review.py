"""Review the exact proposed merge with a fresh smart-profile agent."""

from __future__ import annotations

import time
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
from senpai_agent.github.pull_requests import get_prs
from senpai_agent.github.workflow.errors import WorkflowPreconditionError
from senpai_agent.github.workflow.responses import PullRequestSnapshot
from senpai_agent.PROMPTS import (
    CODE_QUALITY_REVIEW_PROMPT,
    render_prompt,
)

if TYPE_CHECKING:
    from senpai_agent.openhands_runner import RunnerConfig


MERGE_COMPLETION_RESERVE_SECONDS = 60
ReviewText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CodeReviewRejected(WorkflowPreconditionError):
    """A completed, valid review found actionable code-quality problems."""


class CodeQualityVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approved: bool = Field(
        description="Approve only when the review is complete with no blocking issues.",
    )
    review_summary: ReviewText = Field(
        description=(
            "A concise conclusion with clear, actionable feedback where appropriate. "
            "If blocked, explain what must change and why."
        ),
    )
    findings: list[ReviewText] = Field(
        description=(
            "Blocking issues and actionable fixes, with file or line references where "
            "useful. Include any reason the review is incomplete. Empty on approval."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_findings(self) -> Self:
        if self.approved and self.findings:
            raise ValueError("an approval must have no blocking findings")
        if not self.approved and not self.findings:
            raise ValueError("a rejection must explain at least one blocking finding")
        return self


def review_code_quality(
    pull: PullRequestSnapshot,
    *,
    base_sha: str,
    token: SecretStr,
    config: RunnerConfig,
) -> None:
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

    context = get_prs(config.github_repo, numbers=(pull.number,), token=token)
    if context.manifest[0].head_sha != pull.head_sha:
        raise WorkflowPreconditionError(
            "PR head changed while loading code review context"
        )
    assert context.markdown is not None  # A single PR is always returned inline.

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
        prompt = review_prompt(
            pull, base_sha=base_sha, merge_base=merge_base,
            pr_context=context.markdown,
        )
        results: list[CodeQualityVerdict] = []
        status = run_openhands(
            prompt,
            replace(
                supervisor_config(config),
                workspace=repository,
                delegation_task_id=None,
                delegation_deadline_epoch=review_deadline,
            ),
            response_schema=CodeQualityVerdict,
            on_structured_result=results.append,
        )
        if status != 0 or not results:
            raise WorkflowPreconditionError(
                "code quality reviewer did not complete with a verdict"
            )
        verdict = results[0]
        if not verdict.approved:
            findings = "\n".join(f"- {finding}" for finding in verdict.findings)
            raise CodeReviewRejected(
                f"Code quality review blocked PR #{pull.number}: "
                f"{verdict.review_summary}\n{findings}"
            )


def review_prompt(
    pull: PullRequestSnapshot, *, base_sha: str, merge_base: str, pr_context: str,
) -> str:
    return render_prompt(
        CODE_QUALITY_REVIEW_PROMPT,
        BASE_SHA=base_sha,
        HEAD_SHA=pull.head_sha,
        MERGE_BASE_SHA=merge_base,
        PR_CONTEXT=pr_context,
    ) + "\n"
