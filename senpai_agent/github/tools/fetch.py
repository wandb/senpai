"""Read-only fetch of one assignment-repository branch into the checkout.

The student shell holds no GitHub credential, so `git fetch` cannot reach the
repository and a peer's branch, the advisor branch or a commit cited by the
advisor stays invisible. This tool fetches one branch with the controller's
token through the same two-hop copy as the advisor's sync: authenticated Git
runs in a throwaway repository and the checkout receives the objects over a
local file fetch, so the token never reaches the worktree or the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor
from pydantic import Field

from senpai_agent import git_workflow
from senpai_agent.git_transport import GitWorkflowPreconditionError, github_repository_url

from .runtime import GitHubToolRuntime, tool_annotations

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class FetchBranchAction(Action):
    """Fetch one branch of the assignment repository, read-only."""

    branch: str = Field(
        min_length=1,
        max_length=255,
        description=(
            "Branch name in the assignment repository, such as a peer PR's head "
            "branch from get_prs or the advisor branch. Not a URL, a SHA or a "
            "refs/ path."
        ),
    )


class FetchBranchObservation(Observation):
    branch: str
    ref: str
    head_sha: str
    changed: bool

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        objects = "new objects were copied" if self.changed else "nothing had changed"
        return [
            TextContent(
                text=(
                    f"Fetched branch {self.branch} into {self.ref} at {self.head_sha}; "
                    f"{objects}. Inspect it with `git log {self.ref}`, "
                    f"`git diff HEAD...{self.ref}` or `git show <sha>`, and cite its "
                    "commits by the SHAs git prints. The checkout, its branch and "
                    "its files are unchanged."
                )
            )
        ]


class FetchBranchExecutor(ToolExecutor[FetchBranchAction, FetchBranchObservation]):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: FetchBranchAction,
        conversation: LocalConversation | None = None,
    ) -> FetchBranchObservation:
        try:
            synced = git_workflow.sync_remote_branch(
                self.runtime.workspace,
                branch=action.branch,
                authenticated_remote=github_repository_url(self.runtime.workflow.repo),
                token=self.runtime.git_token,
            )
        except GitWorkflowPreconditionError as error:
            raise GitWorkflowPreconditionError(
                f"fetch_branch could not fetch {action.branch!r} from "
                f"{self.runtime.workflow.repo}: {error}. Branch names come from "
                "get_prs (a PR's head branch) or the advisor branch; a commit on no "
                "branch cannot be fetched."
            ) from error
        return FetchBranchObservation(
            branch=synced.branch,
            ref=f"refs/remotes/origin/{synced.branch}",
            head_sha=synced.head_sha,
            changed=synced.changed,
        )


class FetchBranchTool(ToolDefinition[FetchBranchAction, FetchBranchObservation]):
    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Fetch one branch of the assignment repository into "
                    "refs/remotes/origin/<branch> without changing the checkout, "
                    "so another student's PR branch, the advisor branch or a commit "
                    "the advisor cites can be read with git log, diff and show. The "
                    "terminal has no GitHub credential, so `git fetch` cannot do this."
                ),
                action_type=FetchBranchAction,
                observation_type=FetchBranchObservation,
                annotations=tool_annotations("Fetch branch", read_only=True),
                executor=FetchBranchExecutor(runtime),
            )
        ]
