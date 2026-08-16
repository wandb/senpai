"""Runtime-owned GitHub authority and role context."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn

from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.tool import ToolAnnotations, ToolExecutor
from pydantic import SecretStr

from senpai_agent import git_workflow
from senpai_agent.github.workflow import (
    GitHubWorkflow,
    MutationResult,
    PullHeadMismatchError,
    StaleAssignmentRevisionError,
)
from senpai_agent.models import ExperimentResult

from .contracts import (
    GitHubMutationObservation,
    PostAssignmentCommentAction,
    SubmitExperimentResultAction,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


_POST_PUSH_HEAD_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class GitHubCredentials:
    """GitHub authority held by the runtime and never exposed to the model."""

    repo: str
    token: SecretStr
    trusted_actor: str | None = None


_credentials: GitHubCredentials | None = None


def configure_github_credentials(
    repo: str,
    token: SecretStr,
    *,
    trusted_actor: str | None = None,
) -> None:
    """Hold write auth outside model-facing tool specs and terminal secrets."""

    global _credentials
    if len(repo.split("/")) != 2 or not all(repo.split("/")):
        raise ValueError("repo must use owner/name form")
    if not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if not token.get_secret_value().strip():
        raise ValueError("token must not be empty")
    if trusted_actor is not None and not trusted_actor.strip():
        raise ValueError("trusted actor must not be empty")
    _credentials = GitHubCredentials(
        repo=repo,
        token=token,
        trusted_actor=trusted_actor,
    )


def clear_github_credentials() -> None:
    """Remove the process-local GitHub authority after a conversation turn."""

    global _credentials
    _credentials = None


def current_github_credentials() -> GitHubCredentials | None:
    """Return the authority configured for this process."""

    return _credentials


@dataclass(frozen=True)
class GitHubToolRuntime:
    """Shared non-model runtime state for one role's GitHub tools."""

    workflow: GitHubWorkflow
    workspace: Path
    git_token: SecretStr | None
    role: Literal["advisor", "student"]
    advisor_branch: str | None
    student_names: frozenset[str]
    student_name: str | None

    def assignment_base_branch(self) -> str:
        """Return the configured advisor branch or fail before a mutation."""

        if self.role != "advisor" or not self.advisor_branch:
            raise RuntimeError("advisor GitHub tools require an advisor branch")
        return self.advisor_branch

    def require_configured_student(self, student: str) -> None:
        """Reject assignment names outside this launch before touching GitHub."""

        if self.role != "advisor" or not self.student_names:
            raise RuntimeError("create_assignment requires configured student names")
        if student not in self.student_names:
            allowed = ", ".join(sorted(self.student_names))
            raise PermissionError(
                f"student {student!r} is outside this launch; choose one of: {allowed}"
            )

    def require_current_student(self, student: str) -> None:
        """Bind a submitted result to this student runtime."""

        current = self.current_student()
        if student != current:
            raise PermissionError(
                f"result student {student!r} does not match this runtime's "
                f"student {current!r}"
            )

    def current_student(self) -> str:
        """Return the configured identity for a student-owned mutation."""

        if self.role != "student" or not self.student_name:
            raise RuntimeError("student GitHub tools require a student name")
        return self.student_name

    def human_issue_audience(self) -> set[str]:
        """Return the only Issue audience labels this role may answer."""

        if self.role == "advisor":
            return {"team", self.assignment_base_branch()}
        return {"team", f"student:{self.current_student()}"}

    def human_issue_responder(self) -> str:
        """Return the role or pod identity used to key one Issue reply."""

        if self.role == "advisor":
            return "advisor"
        return self.current_student()


class SubmitExperimentResultExecutor(
    ToolExecutor[SubmitExperimentResultAction, GitHubMutationObservation]
):
    """Validate, publish, and record one terminal student result."""

    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: SubmitExperimentResultAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        self.runtime.require_current_student(action.result.assignment.student)
        number = action.result.assignment.pr_number
        commit_sha = action.result.commit_sha
        with self.runtime.workflow.serialized_assignment_mutation():
            try:
                preflight = self.runtime.workflow.preflight_submit_result(
                    number,
                    branch=action.branch,
                    current_head_sha=action.remote_branch_sha_before_push,
                    expected_result_head_sha=commit_sha,
                    result=action.result,
                )
                git_workflow.require_commit_contains_base(
                    self.runtime.workspace,
                    commit_sha=commit_sha,
                    base_sha=preflight.assignment.base_sha,
                )
                git_workflow.push_assignment_branch(
                    self.runtime.workspace,
                    branch=action.branch,
                    expected_remote_sha=action.remote_branch_sha_before_push,
                    expected_local_sha=commit_sha,
                    token=self.runtime.git_token,
                )
                result = self._submit_after_push(number, action.result)
            except StaleAssignmentRevisionError as error:
                _finish_stale_assignment_turn(error, conversation)
            return GitHubMutationObservation.from_result(result)

    def _submit_after_push(
        self,
        number: int,
        result: ExperimentResult,
    ) -> MutationResult:
        for delay in _POST_PUSH_HEAD_RETRY_DELAYS:
            try:
                return self.runtime.workflow.submit_result(
                    number,
                    expected_head_sha=result.commit_sha,
                    result=result,
                )
            except PullHeadMismatchError:
                time.sleep(delay)
        return self.runtime.workflow.submit_result(
            number,
            expected_head_sha=result.commit_sha,
            result=result,
        )


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


def _finish_stale_assignment_turn(
    error: StaleAssignmentRevisionError,
    conversation: LocalConversation | None,
) -> NoReturn:
    if conversation is not None:
        conversation.state.execution_status = ConversationExecutionStatus.FINISHED
    raise ValueError(
        f"{error} Ending this stale turn so the controller can resume "
        "the current assignment revision."
    ) from error


def configured_student_names(value: Sequence[str] | str | None) -> frozenset[str]:
    """Normalize an explicit or environment-provided launch allowlist."""

    if value is None:
        value = os.environ.get("STUDENT_NAMES", "")
    items = value.split(",") if isinstance(value, str) else value
    return frozenset(name for item in items if (name := item.strip()))


def tool_annotations(title: str, *, read_only: bool = False) -> ToolAnnotations:
    """Describe one GitHub tool's side-effect contract."""

    return ToolAnnotations(
        title=title,
        readOnlyHint=read_only,
        destructiveHint=not read_only,
        idempotentHint=True,
        openWorldHint=True,
    )
