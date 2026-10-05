"""Operation-specific GitHub tool definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from openhands.sdk.tool import ToolDefinition, ToolExecutor
from senpai_agent.delegation import SpawnAgentsObservation

from .advisor import (
    AcceptResultOnCurrentBaseExecutor,
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
    CloseExperimentAction,
    CreateAssignmentAction,
    CreateHumanIssueAction,
    GitHubMutationObservation,
    MergeExperimentAction,
    PostAssignmentCommentAction,
    PublishAdvisorBranchAction,
    PushExperimentCommitAction,
    RepairAssignmentRoutingAction,
    RequestAssignmentRevisionAction,
    RespondToHumanIssueAction,
    SendAssignmentFeedbackAction,
    SubmitExperimentResultAction,
)
from .runtime import (
    GitHubToolRuntime,
    PostAssignmentCommentExecutor,
    PushExperimentCommitExecutor,
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


class RequestSupervisorTool(ToolDefinition):
    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        from senpai_agent.github import supervision

        return _tool(
            cls,
            supervision.SupervisorRequest,
            "Request Supervisor",
            "Ask a fresh Supervisor to diagnose and repair the advisor or a student. "
            "The target pod handles the request at its next safe boundary and returns "
            "feedback; other pods can continue working.",
            supervision.RequestSupervisorExecutor(runtime),
        )


class CreateHumanIssueExecutor(
    ToolExecutor[CreateHumanIssueAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: CreateHumanIssueAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        result = self.runtime.workflow.create_human_issue(
            issue_id=action.issue_id,
            title=action.title,
            body=action.body,
            audience_label=self.runtime.human_issue_audience_label(),
            creator=self.runtime.human_issue_responder(),
        )
        return GitHubMutationObservation.from_result(result)


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


class PushExperimentCommitTool(
    ToolDefinition[PushExperimentCommitAction, GitHubMutationObservation]
):
    """Push the student's exact local HEAD to the existing experiment PR branch."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, PushExperimentCommitAction, "Push experiment commit",
            "Push the exact current local commit (HEAD) to the existing GitHub branch for "
            "this student's experiment PR. There must be no uncommitted changes. "
            "The PR must be open and marked work in progress (status:wip). Supply "
            "the PR number, assignment ID, current instruction revision ID, "
            "current GitHub PR head and local Git commit identifiers (SHAs). "
            "The tool checks "
            "the assigned student, branch, base commit and current instructions; "
            "it refuses to overwrite other commits or push after a final experiment "
            "result. It does not post a comment, submit a result, change PR "
            "labels or draft state, remove a hold, or authorize training.",
            PushExperimentCommitExecutor(runtime),
        )


class RepairAssignmentRoutingTool(
    ToolDefinition[RepairAssignmentRoutingAction, GitHubMutationObservation]
):
    """Repair protocol-owned assignment routing state."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, RepairAssignmentRoutingAction, "Repair assignment routing",
            "Repair one current assignment's protocol routing after labels or draft "
            "state drift. Choose wip or review and only named blockers; do not use "
            "this for ordinary assignment decisions.",
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
            "revision. Explicitly retain its assigned base or select the exact "
            "live base. A result must contain the selected base commit.",
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
    ToolDefinition[MergeExperimentAction, SpawnAgentsObservation]
):
    """Queue an independent code review and merge for one exact experiment."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [cls(
            description=(
                "Merge one review-ready experiment, an AI agent will carry out one final "
                "code review and then merge if satisifed, otherwise it will return with "
                "review feedback."
            ),
            action_type=MergeExperimentAction,
            observation_type=SpawnAgentsObservation,
            annotations=tool_annotations("Queue experiment review and merge"),
            executor=MergeExperimentExecutor(runtime),
        )]


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


class CreateHumanIssueTool(
    ToolDefinition[CreateHumanIssueAction, GitHubMutationObservation]
):
    """Open an issue for human input once per stable issue ID."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls, CreateHumanIssueAction, "Create human issue",
            "Create or exactly replay one issue for human input. Reuse its issue_id "
            "with unchanged title and body on retries. The backend adds human and "
            "this role's audience labels, and mentions the GitHub credential "
            "owner, when available, in the initial issue body.",
            CreateHumanIssueExecutor(runtime),
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


class PostAssignmentCommentTool(
    ToolDefinition[PostAssignmentCommentAction, GitHubMutationObservation]
):
    """Post one student-authored comment to its current assignment PR."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return _tool(
            cls,
            PostAssignmentCommentAction,
            "Post assignment comment",
            "Post or exactly replay one meaningful interim progress update, "
            "question, blocker, evidence item, or response on this student's "
            "current assignment without pushing or changing workflow state.",
            PostAssignmentCommentExecutor(runtime),
        )
