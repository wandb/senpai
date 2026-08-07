"""Guarded assignment-branch publication."""

from __future__ import annotations

import base64
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from senpai_agent.secrets import scrub_github_credentials


class GitWorkflowPreconditionError(RuntimeError):
    """Local or remote Git state does not permit a safe push."""


@dataclass(frozen=True, slots=True)
class PushResult:
    changed: bool
    branch: str
    head_sha: str


def require_clean_training_worktree(workspace: Path) -> None:
    """Require every tracked and untracked assignment change to be committed."""

    workspace = Path(workspace).resolve()
    _git(workspace, "rev-parse", "--is-inside-work-tree")
    if _git(workspace, "status", "--porcelain", "--untracked-files=all"):
        raise GitWorkflowPreconditionError(
            "assignment worktree must be clean before training"
        )


def require_commit_contains_base(
    workspace: Path,
    *,
    commit_sha: str,
    base_sha: str,
) -> None:
    """Require an exact result commit to contain its assigned research base."""

    workspace = Path(workspace).resolve()
    if not commit_sha.strip() or not base_sha.strip():
        raise ValueError("commit_sha and base_sha must not be empty")
    commit = _git(workspace, "rev-parse", f"{commit_sha}^{{commit}}")
    base = _git(workspace, "rev-parse", f"{base_sha}^{{commit}}")
    try:
        _git(workspace, "merge-base", "--is-ancestor", base, commit)
    except GitWorkflowPreconditionError as error:
        raise GitWorkflowPreconditionError(
            f"result commit {commit} does not contain assigned research base {base}"
        ) from error


def push_assignment_branch(
    workspace: Path,
    *,
    branch: str,
    expected_remote_sha: str,
    expected_local_sha: str | None = None,
    remote: str = "origin",
    token: SecretStr | None = None,
) -> PushResult:
    """Push one clean assignment branch with exact local and remote head leases."""

    workspace = Path(workspace).resolve()
    _validate_token(token)
    _git(workspace, "rev-parse", "--is-inside-work-tree")
    _git(workspace, "check-ref-format", "--branch", branch)
    current_branch = _git(workspace, "branch", "--show-current")
    if current_branch != branch:
        raise GitWorkflowPreconditionError(
            f"current branch is {current_branch!r}, expected {branch!r}"
        )
    if _git(workspace, "status", "--porcelain"):
        raise GitWorkflowPreconditionError(
            "assignment worktree must be clean before push"
        )

    local_sha = _git(workspace, "rev-parse", "HEAD")
    if expected_local_sha is not None and local_sha != expected_local_sha:
        raise GitWorkflowPreconditionError(
            f"local head is {local_sha}, expected {expected_local_sha}"
        )
    remote_sha = _remote_head(workspace, remote, branch, token=token)
    if remote_sha == local_sha:
        return PushResult(False, branch, local_sha)
    if remote_sha != expected_remote_sha:
        raise GitWorkflowPreconditionError(
            f"remote head is {remote_sha or '<missing>'}, "
            f"expected {expected_remote_sha}"
        )
    _git(
        workspace,
        "fetch",
        "--no-tags",
        remote,
        f"refs/heads/{branch}",
        token=token,
    )
    if _git(workspace, "rev-parse", "FETCH_HEAD") != remote_sha:
        raise GitWorkflowPreconditionError("remote head moved while preparing the push")
    try:
        _git(workspace, "merge-base", "--is-ancestor", remote_sha, local_sha)
    except GitWorkflowPreconditionError as error:
        raise GitWorkflowPreconditionError(
            f"local head {local_sha} does not fast-forward remote head {remote_sha}"
        ) from error

    _git(
        workspace,
        "push",
        f"--force-with-lease=refs/heads/{branch}:{expected_remote_sha}",
        remote,
        f"{local_sha}:refs/heads/{branch}",
        token=token,
    )
    if _remote_head(workspace, remote, branch, token=token) != local_sha:
        raise RuntimeError("remote branch did not reach the pushed commit")
    return PushResult(True, branch, local_sha)


def create_assignment_branch(
    workspace: Path,
    *,
    branch: str,
    base_branch: str,
    expected_base_sha: str,
    assignment_id: str,
    remote: str = "origin",
    token: SecretStr | None = None,
) -> PushResult:
    """Publish one empty assignment commit without changing the worktree."""

    workspace = Path(workspace).resolve()
    _validate_token(token)
    _git(workspace, "rev-parse", "--is-inside-work-tree")
    _git(workspace, "check-ref-format", "--branch", branch)
    _git(workspace, "check-ref-format", "--branch", base_branch)
    if branch == base_branch:
        raise GitWorkflowPreconditionError(
            "assignment branch must differ from the base branch"
        )
    if not assignment_id.strip():
        raise ValueError("assignment_id must not be empty")

    base_sha = _remote_head(workspace, remote, base_branch, token=token)
    if base_sha != expected_base_sha:
        raise GitWorkflowPreconditionError(
            f"remote base head is {base_sha or '<missing>'}, "
            f"expected {expected_base_sha}"
        )
    _git(
        workspace,
        "fetch",
        "--no-tags",
        remote,
        f"refs/heads/{base_branch}",
        token=token,
    )
    if _git(workspace, "rev-parse", "FETCH_HEAD") != expected_base_sha:
        raise GitWorkflowPreconditionError(
            "fetched base does not match the expected base SHA"
        )

    message = f"senpai assignment: {assignment_id}"
    tree_sha = _git(
        workspace,
        "rev-parse",
        f"{expected_base_sha}^{{tree}}",
    )
    remote_sha = _remote_head(workspace, remote, branch, token=token)
    if remote_sha:
        _git(
            workspace,
            "fetch",
            "--no-tags",
            remote,
            f"refs/heads/{branch}",
            token=token,
        )
        if (
            _git(workspace, "rev-parse", f"{remote_sha}^") != expected_base_sha
            or _git(
                workspace,
                "rev-parse",
                f"{remote_sha}^{{tree}}",
            )
            != tree_sha
            or _git(
                workspace,
                "show",
                "-s",
                "--format=%B",
                remote_sha,
            )
            != message
        ):
            raise GitWorkflowPreconditionError(
                f"assignment branch {branch!r} already exists with foreign history"
            )
        return PushResult(False, branch, remote_sha)

    head_sha = _git(
        workspace,
        "commit-tree",
        tree_sha,
        "-p",
        expected_base_sha,
        input_text=f"{message}\n",
    )
    _git(
        workspace,
        "push",
        f"--force-with-lease=refs/heads/{branch}:",
        remote,
        f"{head_sha}:refs/heads/{branch}",
        token=token,
    )
    if _remote_head(workspace, remote, branch, token=token) != head_sha:
        raise RuntimeError("remote assignment branch did not reach the new commit")
    return PushResult(True, branch, head_sha)


def _remote_head(
    workspace: Path,
    remote: str,
    branch: str,
    *,
    token: SecretStr | None,
) -> str:
    result = _git(
        workspace,
        "ls-remote",
        "--refs",
        remote,
        f"refs/heads/{branch}",
        token=token,
    )
    return result.split(maxsplit=1)[0] if result else ""


def _git(
    workspace: Path,
    *arguments: str,
    input_text: str | None = None,
    token: SecretStr | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=workspace,
        text=True,
        input=input_text,
        capture_output=True,
        env=_git_process_env(token),
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitWorkflowPreconditionError(
            f"git {' '.join(arguments[:2])} failed: {detail[:1000]}"
        )
    return completed.stdout.strip()


def _validate_token(token: SecretStr | None) -> None:
    if token is not None and not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if token is not None and not token.get_secret_value().strip():
        raise ValueError("token must not be empty")


def _git_process_env(token: SecretStr | None) -> dict[str, str]:
    env = dict(os.environ)
    scrub_github_credentials(env)
    if token is None:
        return env

    credential = base64.b64encode(
        f"x-access-token:{token.get_secret_value()}".encode()
    ).decode()
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credential}",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env
