"""Runtime-bound student messages to own and peer assignment PRs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor

from senpai_agent.github.workflow import StaleAssignmentRevisionError

from .contracts import (
    GitHubMutationObservation,
    PostAssignmentCommentAction,
    PostPeerCommentAction,
)
from .runtime import GitHubToolRuntime, _finish_stale_assignment_turn

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class PostAssignmentCommentExecutor(
    ToolExecutor[PostAssignmentCommentAction, GitHubMutationObservation]
):
    """Post one durable interim message to the student's current assignment."""

    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: PostAssignmentCommentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        try:
            result = self.runtime.workflow.post_assignment_comment(
                version.pr_number,
                assignment_id=version.assignment_id,
                revision_id=version.revision_id,
                expected_head_sha=version.expected_pr_head_sha,
                student=self.runtime.current_student(),
                comment_id=action.comment_id,
                comment=action.comment,
            )
        except StaleAssignmentRevisionError as error:
            _finish_stale_assignment_turn(error, conversation)
        return GitHubMutationObservation.from_result(result)


class PostPeerCommentExecutor(
    ToolExecutor[PostPeerCommentAction, GitHubMutationObservation]
):
    """Bind a peer message to the runtime's student and current assignment."""

    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: PostPeerCommentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        if not self.runtime.advisor_branch:
            raise RuntimeError("peer comments require a configured advisor branch")
        try:
            result = self.runtime.workflow.post_peer_comment(
                version.pr_number,
                assignment_id=version.assignment_id,
                revision_id=version.revision_id,
                expected_head_sha=version.expected_pr_head_sha,
                student=self.runtime.current_student(),
                advisor_branch=self.runtime.advisor_branch,
                target_pr_number=action.target_pr_number,
                comment_id=action.comment_id,
                comment=action.comment,
            )
        except StaleAssignmentRevisionError as error:
            _finish_stale_assignment_turn(error, conversation)
        return GitHubMutationObservation.from_result(result)
