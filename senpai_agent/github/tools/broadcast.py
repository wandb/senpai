"""Compact student discovery broadcasts through ordinary GitHub polling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from openhands.sdk.tool import Action, ToolDefinition, ToolExecutor
from pydantic import Field

from senpai_agent.github.workflow import StaleAssignmentRevisionError
from senpai_agent.models import MAX_BROADCAST_MESSAGE_CHARS

from .contracts import AssignmentVersion, GitHubMutationObservation
from .runtime import GitHubToolRuntime, _finish_stale_assignment_turn, tool_annotations

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class BroadcastMessageAction(Action):
    """Share one concise discovery with all other students on this advisor base."""

    assignment: AssignmentVersion = Field(
        description="Your own current assignment revision and PR-head precondition.",
    )
    broadcast_id: str = Field(
        min_length=1,
        max_length=256,
        description="Stable discovery ID. Reuse the same ID and text to retry delivery.",
    )
    message: str = Field(
        min_length=1,
        max_length=MAX_BROADCAST_MESSAGE_CHARS,
        description=(
            "Compact FYI: state the discovery, evidence or uncertainty, and which "
            "work it may help. Explain terms for students outside your subproblem. "
            "Link supporting PRs, commits, or runs instead of pasting logs or code. "
            "The runtime adds your name and current PR link. This is optional "
            "context for peers' existing assignments, not a request to change focus."
        ),
    )


class BroadcastMessageObservation(GitHubMutationObservation):
    """Source discovery and the current peer PRs that received it."""

    delivered_pr_numbers: tuple[int, ...]


class BroadcastMessageExecutor(
    ToolExecutor[BroadcastMessageAction, BroadcastMessageObservation]
):
    """Bind a discovery broadcast to the student's current assignment."""

    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: BroadcastMessageAction,
        conversation: LocalConversation | None = None,
    ) -> BroadcastMessageObservation:
        version = action.assignment
        if not self.runtime.advisor_branch:
            raise RuntimeError("broadcasts require a configured advisor branch")
        try:
            result = self.runtime.workflow.broadcast_message(
                version.pr_number,
                assignment_id=version.assignment_id,
                revision_id=version.revision_id,
                expected_head_sha=version.expected_pr_head_sha,
                student=self.runtime.current_student(),
                advisor_branch=self.runtime.advisor_branch,
                broadcast_id=action.broadcast_id,
                message=action.message,
            )
        except StaleAssignmentRevisionError as error:
            _finish_stale_assignment_turn(error, conversation)
        return BroadcastMessageObservation(
            changed=result.changed,
            resource_url=result.resource_url,
            state=result.state,
            version=result.version,
            delivered_pr_numbers=result.delivered_pr_numbers,
        )


class BroadcastMessageTool(
    ToolDefinition[BroadcastMessageAction, BroadcastMessageObservation]
):
    """Broadcast a compact discovery without interrupting other students."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Publish a concise discovery FYI on your PR and all other students' "
                    "open assignment PRs on the configured advisor base, regardless of "
                    "draft or workflow status. Maximum 1,500 characters; explain the "
                    "finding, relevance, and evidence, and link details. The runtime "
                    "adds your student name and PR link. Peers receive it through "
                    "regular GitHub polling without interruption and consider it only "
                    "within their assignment. Use post_peer_comment for a targeted "
                    "question or reply. Retry partial delivery with the same ID and "
                    "text; existing copies are not reposted."
                ),
                action_type=BroadcastMessageAction,
                observation_type=BroadcastMessageObservation,
                annotations=tool_annotations("Broadcast message"),
                executor=BroadcastMessageExecutor(runtime),
            )
        ]
