"""Advisor-owned GitHub workflow executors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor

from senpai_agent import git_workflow
from senpai_agent.models import (
    AssignmentRecord,
    DispositionRecord,
    render_disposition_marker,
)

from .contracts import (
    AcceptResultOnCurrentBaseAction,
    CloseExperimentAction,
    CreateAssignmentAction,
    GitHubMutationObservation,
    MergeExperimentAction,
    PublishAdvisorBranchAction,
    RepairAssignmentRoutingAction,
    RequestAssignmentRevisionAction,
    SendAssignmentFeedbackAction,
)
from .runtime import GitHubToolRuntime

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from senpai_agent.github.workflow import GitHubWorkflow


class CreateAssignmentExecutor(
    ToolExecutor[CreateAssignmentAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: CreateAssignmentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        base_branch = self.runtime.assignment_base_branch()
        self.runtime.require_configured_student(action.student)
        with self.runtime.workflow.serialized_assignment_mutation():
            branch = git_workflow.create_assignment_branch(
                self.runtime.workspace,
                branch=action.head_branch,
                base_branch=base_branch,
                expected_base_sha=action.expected_base_sha,
                assignment_id=action.assignment_id,
                token=self.runtime.git_token,
            )
            result = self.runtime.workflow.create_assignment(
                AssignmentRecord(
                    repo=self.runtime.workflow.repo,
                    assignment_id=action.assignment_id,
                    revision_id=action.revision_id,
                    student=action.student,
                    base_ref=base_branch,
                    base_sha=action.expected_base_sha,
                    head_ref=action.head_branch,
                    head_sha=branch.head_sha,
                ),
                title=action.title,
                body=action.body,
            )
        return GitHubMutationObservation.from_result(result)


class PublishAdvisorBranchExecutor(
    ToolExecutor[PublishAdvisorBranchAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: PublishAdvisorBranchAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        if not self.runtime.advisor_branch:
            raise RuntimeError("publish_advisor_branch requires an advisor branch")
        with self.runtime.workflow.serialized_assignment_mutation():
            pushed = git_workflow.push_assignment_branch(
                self.runtime.workspace,
                branch=self.runtime.advisor_branch,
                expected_remote_sha=action.remote_branch_sha_before_push,
                expected_local_sha=action.local_commit_sha,
                token=self.runtime.git_token,
            )
        return GitHubMutationObservation(
            changed=pushed.changed,
            resource_url=f"git:origin/{pushed.branch}",
            state="branch_pushed",
            version=pushed.head_sha,
        )


class RepairAssignmentRoutingExecutor(
    ToolExecutor[RepairAssignmentRoutingAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: RepairAssignmentRoutingAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.repair_assignment_routing(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            working_state=action.working_state,
            blockers=action.blockers,
        )
        return GitHubMutationObservation.from_result(result)


class SendAssignmentFeedbackExecutor(
    ToolExecutor[SendAssignmentFeedbackAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: SendAssignmentFeedbackAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.send_assignment_feedback(
            version.pr_number,
            assignment_id=version.assignment_id,
            revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            feedback_id=action.feedback_id,
            comment=action.comment,
        )
        return GitHubMutationObservation.from_result(result)


class RequestAssignmentRevisionExecutor(
    ToolExecutor[RequestAssignmentRevisionAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: RequestAssignmentRevisionAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.request_revision(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            new_revision_id=action.new_revision_id,
            required_base_sha=action.required_base_sha,
            expected_head_sha=version.expected_pr_head_sha,
            comment=action.comment,
        )
        return GitHubMutationObservation.from_result(result)


class AcceptResultOnCurrentBaseExecutor(
    ToolExecutor[AcceptResultOnCurrentBaseAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: AcceptResultOnCurrentBaseAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.accept_result_on_current_base(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            expected_current_base_sha=action.expected_current_base_sha,
            reason=action.reason,
        )
        return GitHubMutationObservation.from_result(result)


class MergeExperimentExecutor(
    ToolExecutor[MergeExperimentAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: MergeExperimentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.merge_experiment(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            expected_current_base_sha=action.expected_current_base_sha,
            merge_method=action.merge_method,
        )
        return GitHubMutationObservation.from_result(result)


class CloseExperimentExecutor(
    ToolExecutor[CloseExperimentAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: CloseExperimentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.close_experiment(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            marker=render_disposition_marker(
                DispositionRecord(
                    repo=self.workflow.repo,
                    pr_number=version.pr_number,
                    assignment_id=version.assignment_id,
                    head_sha=version.expected_pr_head_sha,
                )
            ),
            reason=action.reason,
        )
        return GitHubMutationObservation.from_result(result)
