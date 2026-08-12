"""Operation-specific GitHub tool definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from openhands.sdk.tool import ToolDefinition, ToolExecutor

from .advisor import (
    AcceptResultOnCurrentBaseExecutor,
    AdoptAssignmentExecutor,
    CloseExperimentExecutor,
    CreateAssignmentExecutor,
    MergeExperimentExecutor,
    PublishAdvisorBranchExecutor,
    RepairAssignmentRoutingExecutor,
    RequestAssignmentRevisionExecutor,
    SendAssignmentFeedbackExecutor,
)
from .contracts import (
    AcceptResultOnCurrentBaseAction,
    AdoptAssignmentAction,
    CloseExperimentAction,
    CreateAssignmentAction,
    GitHubMutationObservation,
    MergeExperimentAction,
    PublishAdvisorBranchAction,
    RepairAssignmentRoutingAction,
    RequestAssignmentRevisionAction,
    RespondToHumanIssueAction,
    SendAssignmentFeedbackAction,
    SubmitExperimentResultAction,
)
from .runtime import (
    GitHubToolRuntime,
    SubmitExperimentResultExecutor,
    tool_annotations,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


def _tool(cls, action_type, title: str, description: str, executor):
    """Build one operation-specific tool without repeating SDK wiring."""

    return [
        cls(
            description=description,
            action_type=action_type,
            observation_type=GitHubMutationObservation,
            annotations=tool_annotations(title),
            executor=executor,
        )
    ]


class RespondToHumanIssueExecutor(
    ToolExecutor[RespondToHumanIssueAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: RespondToHumanIssueAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        result = self.runtime.workflow.respond_to_issue(
            action.issue_number,
            human_message_id=action.human_message_id,
            response=action.response,
            audience_labels=self.runtime.human_issue_audience(),
            responder=self.runtime.human_issue_responder(),
        )
        return GitHubMutationObservation.from_result(result)


class CreateAssignmentTool(
    ToolDefinition[CreateAssignmentAction, GitHubMutationObservation]
):
    """Create one assignment for a configured student."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, CreateAssignmentAction, "Create assignment",
            "Create or exactly replay one student's typed draft assignment PR from "
            "the configured advisor branch and a lease-checked base commit. The "
            "student must belong to this launch.",
            CreateAssignmentExecutor(runtime),
        )


class AdoptAssignmentTool(
    ToolDefinition[AdoptAssignmentAction, GitHubMutationObservation]
):
    """Adopt one exact existing assignment PR for a configured student."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, AdoptAssignmentAction, "Adopt assignment PR",
            "Attach typed assignment identity to one exact markerless draft PR "
            "after verifying its configured student, routing, and Git history. "
            "This never creates a branch or infers identity from PR prose.",
            AdoptAssignmentExecutor(runtime),
        )


class PublishAdvisorBranchTool(
    ToolDefinition[PublishAdvisorBranchAction, GitHubMutationObservation]
):
    """Publish only the configured advisor branch with a lease."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, PublishAdvisorBranchAction, "Publish advisor branch",
            "Publish the configured advisor branch with force-with-lease. Supply its "
            "current remote SHA and the exact local commit to push.",
            PublishAdvisorBranchExecutor(runtime),
        )


class RepairAssignmentRoutingTool(
    ToolDefinition[RepairAssignmentRoutingAction, GitHubMutationObservation]
):
    """Repair protocol-owned assignment routing state."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, RepairAssignmentRoutingAction, "Repair assignment labels and draft",
            "Repair labels and draft state for an assignment that already has a "
            "valid marker. This cannot create or adopt assignment identity.",
            RepairAssignmentRoutingExecutor(runtime.workflow),
        )


class SendAssignmentFeedbackTool(
    ToolDefinition[SendAssignmentFeedbackAction, GitHubMutationObservation]
):
    """Send guidance for one exact assignment version."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, SendAssignmentFeedbackAction, "Send assignment feedback",
            "Send a clarification, hold, question, or nudge to the current assignment "
            "without changing its revision or routing state.",
            SendAssignmentFeedbackExecutor(runtime.workflow),
        )


class RequestAssignmentRevisionTool(
    ToolDefinition[RequestAssignmentRevisionAction, GitHubMutationObservation]
):
    """Request a fresh assignment revision on an exact research base."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, RequestAssignmentRevisionAction, "Request assignment revision",
            "Request a scientifically meaningful rerun as a fresh assignment "
            "revision, bound to the exact live base commit it must evaluate.",
            RequestAssignmentRevisionExecutor(runtime.workflow),
        )


class AcceptResultOnCurrentBaseTool(
    ToolDefinition[AcceptResultOnCurrentBaseAction, GitHubMutationObservation]
):
    """Accept one exact result on the current research base."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, AcceptResultOnCurrentBaseAction, "Accept result on current base",
            "After comparing the exact submitted result with a changed research base, "
            "record why that result remains valid. This does not merge the PR.",
            AcceptResultOnCurrentBaseExecutor(runtime.workflow),
        )


class MergeExperimentTool(
    ToolDefinition[MergeExperimentAction, GitHubMutationObservation]
):
    """Merge one verified experiment result."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, MergeExperimentAction, "Merge experiment",
            "Merge one review-ready experiment at the expected PR head and current "
            "research base. Changed-base results require a prior "
            "accept_result_on_current_base call.",
            MergeExperimentExecutor(runtime.workflow),
        )


class CloseExperimentTool(
    ToolDefinition[CloseExperimentAction, GitHubMutationObservation]
):
    """Close one exact non-winning experiment."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, CloseExperimentAction, "Close experiment",
            "Close one current experiment without merging it, recording an "
            "evidence-backed reason and preserving the durable result.",
            CloseExperimentExecutor(runtime.workflow),
        )


class RespondToHumanIssueTool(
    ToolDefinition[RespondToHumanIssueAction, GitHubMutationObservation]
):
    """Respond once to an authenticated human Issue message."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, RespondToHumanIssueAction, "Respond to human issue",
            "Respond once to a verified human-authored GitHub Issue body or comment "
            "delivered to this configured role. The backend rechecks the Issue's "
            "team/branch/student audience labels.",
            RespondToHumanIssueExecutor(runtime),
        )


class SubmitExperimentResultTool(
    ToolDefinition[SubmitExperimentResultAction, GitHubMutationObservation]
):
    """Publish one student's validated terminal result."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, SubmitExperimentResultAction, "Submit experiment result",
            "Validate one terminal result against its current assignment and research "
            "base, lease-push result.commit_sha, then publish the typed result and "
            "make the PR review-ready.",
            SubmitExperimentResultExecutor(runtime),
        )
