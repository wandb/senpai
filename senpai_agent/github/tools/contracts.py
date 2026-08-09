"""Model-facing contracts for GitHub workflow mutations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Self

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import Action, Observation
from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.github.workflow import MutationResult
from senpai_agent.models import ExperimentResult


class AssignmentVersion(BaseModel):
    """Exact assignment revision and pull-request head a mutation may change."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pr_number: int = Field(
        gt=0,
        description="Pull-request number containing the assignment.",
    )
    assignment_id: str = Field(
        min_length=1,
        description="Stable assignment ID from the trusted assignment marker.",
    )
    revision_id: str = Field(
        min_length=1,
        description="Current revision ID from the trusted assignment marker.",
    )
    expected_pr_head_sha: str = Field(
        min_length=1,
        description="Current pull-request head SHA. The mutation fails if the PR moved.",
    )


class CreateAssignmentAction(Action):
    """Create one isolated student branch and its typed draft assignment PR."""

    assignment_id: str = Field(
        min_length=1,
        description="New stable assignment ID; reuse it only to replay this assignment.",
    )
    revision_id: str = Field(
        min_length=1,
        description="Initial revision ID for this assignment.",
    )
    student: str = Field(
        min_length=1,
        description="Exact configured student name that will own the assignment.",
    )
    expected_base_sha: str = Field(
        min_length=1,
        description=(
            "Exact current SHA of the runtime-configured advisor branch, used as "
            "the assignment creation precondition."
        ),
    )
    head_branch: str = Field(
        min_length=1,
        description="New remote branch dedicated to this student assignment.",
    )
    title: str = Field(
        min_length=1,
        max_length=256,
        description="Concise falsifiable experiment title for the pull request.",
    )
    body: str = Field(
        min_length=1,
        max_length=50_000,
        description="Complete experiment brief, evidence contract, and stopping rule.",
    )


class PublishAdvisorBranchAction(Action):
    """Lease-publish the configured advisor branch at one exact local commit."""

    remote_branch_sha_before_push: str = Field(
        min_length=1,
        description="Current remote advisor-branch SHA used as the push lease.",
    )
    local_commit_sha: str = Field(
        min_length=1,
        description="Exact local commit to publish; the worktree HEAD must equal it.",
    )


class RepairAssignmentRoutingAction(Action):
    """Restore the desired protocol state for one current assignment revision."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    working_state: Literal["wip", "review"] = Field(
        description=(
            "Desired productive state: wip while the student works, or review "
            "after a terminal result is ready."
        ),
    )
    blockers: set[Literal["blocked", "hold", "needs-rebase"]] = Field(
        default_factory=set,
        description="Exact protocol blockers that should remain on the assignment.",
    )


class SendAssignmentFeedbackAction(Action):
    """Send idempotent guidance without starting a new assignment revision."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    feedback_id: str = Field(
        min_length=1,
        max_length=256,
        description=(
            "Stable ID for this guidance item. Replay is a no-op; changed guidance "
            "must use a new ID."
        ),
    )
    comment: str = Field(
        min_length=1,
        max_length=50_000,
        description="Actionable guidance that does not require a fresh revision.",
    )


class RequestAssignmentRevisionAction(Action):
    """Start a new revision of an existing assignment on an exact research base."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    new_revision_id: str = Field(
        min_length=1,
        description="Fresh revision ID that has never identified another revision.",
    )
    required_base_sha: str = Field(
        min_length=1,
        description="Exact live base-branch SHA against which the new revision must run.",
    )
    comment: str = Field(
        min_length=1,
        max_length=50_000,
        description="Concrete reason and changed evidence requested for the revision.",
    )


class AcceptResultOnCurrentBaseAction(Action):
    """Record that one exact result remains valid on the current research base."""

    assignment: AssignmentVersion = Field(
        description="Submitted result revision and PR-head precondition.",
    )
    expected_current_base_sha: str = Field(
        min_length=1,
        description=(
            "Exact live SHA of the assignment's recorded base branch after review."
        ),
    )
    reason: str = Field(
        min_length=1,
        max_length=50_000,
        description="Scientific reason the existing result remains valid on that base.",
    )


class MergeExperimentAction(Action):
    """Merge one reviewed result after exact head and research-base validation."""

    assignment: AssignmentVersion = Field(
        description="Submitted result revision and PR-head precondition.",
    )
    expected_current_base_sha: str = Field(
        min_length=1,
        description="Exact live base-branch SHA immediately expected for the merge.",
    )
    merge_method: Literal["merge", "squash", "rebase"] = Field(
        default="squash",
        description="GitHub merge method to apply after all workflow checks pass.",
    )


class CloseExperimentAction(Action):
    """Close one reviewed assignment as a durable non-winning experiment."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    reason: str = Field(
        min_length=1,
        max_length=50_000,
        description="Evidence-backed reason this experiment should close unmerged.",
    )


class RespondToHumanIssueAction(Action):
    """Respond once to one authenticated human-authored issue message."""

    issue_number: int = Field(
        gt=0,
        description=(
            "GitHub issue number containing a human message addressed to this "
            "configured advisor or student."
        ),
    )
    human_message_id: int = Field(
        gt=0,
        description="Exact numeric ID of the human-authored body or comment answered.",
    )
    response: str = Field(
        min_length=1,
        max_length=50_000,
        description="Response text; the runtime adds the authenticated role prefix.",
    )


class SubmitExperimentResultAction(Action):
    """Validate, publish, and submit one student's terminal experiment result."""

    branch: str = Field(
        min_length=1,
        description="Assignment branch named by the current pull request.",
    )
    remote_branch_sha_before_push: str = Field(
        min_length=1,
        description="Current remote branch SHA used as the force-with-lease guard.",
    )
    result: ExperimentResult = Field(
        description=(
            "Complete terminal result. Its assignment PR number and commit SHA are "
            "the publication target and local commit; do not repeat them elsewhere."
        ),
    )


class GitHubMutationObservation(Observation):
    """Verified durable state reached by one GitHub workflow mutation."""

    changed: bool
    resource_url: str
    state: str
    version: str | None = None

    @classmethod
    def from_result(cls, result: MutationResult) -> Self:
        return cls(
            changed=result.changed,
            resource_url=result.resource_url,
            state=result.state,
            version=result.version,
        )

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=json.dumps(
                    self.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ]
