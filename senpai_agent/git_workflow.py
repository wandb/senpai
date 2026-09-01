"""Guarded assignment-branch publication."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import SecretStr

from senpai_agent.git_transport import (
    GIT_EXECUTABLE,
    GitWorkflowPreconditionError,
    git_process_env,
    github_repository_url,
    isolated_bare_repository as _isolated_bare_repository,
    remote_head as _remote_head,
    run_git as _git,
    staged_commit as _staged_commit,
)


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
    authenticated_remote: str | None = None,
    token: SecretStr | None = None,
) -> PushResult:
    """Push one validated commit with exact local and remote head leases."""

    workspace = Path(workspace).resolve()
    _validate_token(token)
    _git(workspace, "rev-parse", "--is-inside-work-tree")
    _git(workspace, "check-ref-format", "--branch", branch)
    current_branch = _git(workspace, "branch", "--show-current")
    if current_branch != branch:
        raise GitWorkflowPreconditionError(
            f"current branch is {current_branch!r}, expected {branch!r}"
        )
    local_sha = _git(workspace, "rev-parse", "HEAD")
    if expected_local_sha is not None and local_sha != expected_local_sha:
        raise GitWorkflowPreconditionError(
            f"local head is {local_sha}, expected {expected_local_sha}"
        )
    network_remote = remote
    if token is not None:
        if authenticated_remote is None:
            raise ValueError("authenticated_remote is required for a credentialed push")
        network_remote = authenticated_remote
    repository = (
        _staged_commit(workspace, local_sha)
        if token is not None
        else nullcontext(workspace)
    )
    with repository as network_workspace:
        return _push_validated_commit(
            network_workspace,
            branch=branch,
            expected_remote_sha=expected_remote_sha,
            local_sha=local_sha,
            remote=network_remote,
            token=token,
        )


def create_assignment_branch(
    workspace: Path,
    *,
    branch: str,
    base_branch: str,
    expected_base_sha: str,
    assignment_id: str,
    remote: str = "origin",
    authenticated_remote: str | None = None,
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

    network_remote = remote
    if token is not None:
        if authenticated_remote is None:
            raise ValueError(
                "authenticated_remote is required for credentialed branch creation"
            )
        network_remote = authenticated_remote
    repository = (
        _isolated_bare_repository() if token is not None else nullcontext(workspace)
    )
    with repository as network_workspace:
        return _create_assignment_commit(
            network_workspace,
            branch=branch,
            base_branch=base_branch,
            expected_base_sha=expected_base_sha,
            assignment_id=assignment_id,
            remote=network_remote,
            token=token,
        )


def _push_validated_commit(
    repository: Path,
    *,
    branch: str,
    expected_remote_sha: str,
    local_sha: str,
    remote: str,
    token: SecretStr | None,
) -> PushResult:
    remote_sha = _remote_head(repository, remote, branch, token=token)
    if remote_sha == local_sha:
        return PushResult(False, branch, local_sha)
    if remote_sha != expected_remote_sha:
        raise GitWorkflowPreconditionError(
            f"remote head is {remote_sha or '<missing>'}, "
            f"expected {expected_remote_sha}"
        )
    _git(
        repository,
        "fetch",
        "--no-tags",
        remote,
        f"refs/heads/{branch}",
        token=token,
    )
    if _git(repository, "rev-parse", "FETCH_HEAD") != remote_sha:
        raise GitWorkflowPreconditionError("remote head moved while preparing the push")
    try:
        _git(repository, "merge-base", "--is-ancestor", remote_sha, local_sha)
    except GitWorkflowPreconditionError as error:
        raise GitWorkflowPreconditionError(
            f"local head {local_sha} does not fast-forward remote head {remote_sha}"
        ) from error
    _require_program_policy_unchanged(repository, remote_sha, local_sha)

    _git(
        repository,
        "push",
        f"--force-with-lease=refs/heads/{branch}:{expected_remote_sha}",
        remote,
        f"{local_sha}:refs/heads/{branch}",
        token=token,
    )
    if _remote_head(repository, remote, branch, token=token) != local_sha:
        raise RuntimeError("remote branch did not reach the pushed commit")
    return PushResult(True, branch, local_sha)


def _require_program_policy_unchanged(
    repository: Path,
    base_sha: str,
    head_sha: str,
) -> None:
    changed_paths = _git(
        repository,
        "diff",
        "--name-only",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "-z",
        base_sha,
        head_sha,
        "--",
    ).removesuffix("\0")
    if any(
        PurePosixPath(path).name == "program.md"
        for path in changed_paths.split("\0")
        if path
    ):
        raise GitWorkflowPreconditionError(
            "program.md changes require explicit operator publication"
        )


def _create_assignment_commit(
    repository: Path,
    *,
    branch: str,
    base_branch: str,
    expected_base_sha: str,
    assignment_id: str,
    remote: str,
    token: SecretStr | None,
) -> PushResult:
    base_sha = _remote_head(repository, remote, base_branch, token=token)
    if base_sha != expected_base_sha:
        raise GitWorkflowPreconditionError(
            f"remote base head is {base_sha or '<missing>'}, "
            f"expected {expected_base_sha}"
        )
    _git(
        repository,
        "fetch",
        "--no-tags",
        remote,
        f"refs/heads/{base_branch}",
        token=token,
    )
    if _git(repository, "rev-parse", "FETCH_HEAD") != expected_base_sha:
        raise GitWorkflowPreconditionError(
            "fetched base does not match the expected base SHA"
        )

    message = f"senpai assignment: {assignment_id}"
    tree_sha = _git(repository, "rev-parse", f"{expected_base_sha}^{{tree}}")
    remote_sha = _remote_head(repository, remote, branch, token=token)
    if remote_sha:
        _git(
            repository,
            "fetch",
            "--no-tags",
            remote,
            f"refs/heads/{branch}",
            token=token,
        )
        if (
            _git(repository, "rev-parse", f"{remote_sha}^") != expected_base_sha
            or _git(repository, "rev-parse", f"{remote_sha}^{{tree}}") != tree_sha
            or _git(repository, "show", "-s", "--format=%B", remote_sha)
            != message
        ):
            raise GitWorkflowPreconditionError(
                f"assignment branch {branch!r} already exists with foreign history"
            )
        return PushResult(False, branch, remote_sha)

    head_sha = _git(
        repository,
        "commit-tree",
        tree_sha,
        "-p",
        expected_base_sha,
        input_text=f"{message}\n",
        extra_env={
            "GIT_AUTHOR_NAME": "Senpai",
            "GIT_AUTHOR_EMAIL": "senpai@localhost",
            "GIT_COMMITTER_NAME": "Senpai",
            "GIT_COMMITTER_EMAIL": "senpai@localhost",
        },
    )
    _git(
        repository,
        "push",
        f"--force-with-lease=refs/heads/{branch}:",
        remote,
        f"{head_sha}:refs/heads/{branch}",
        token=token,
    )
    if _remote_head(repository, remote, branch, token=token) != head_sha:
        raise RuntimeError("remote assignment branch did not reach the new commit")
    return PushResult(True, branch, head_sha)


def _validate_token(token: SecretStr | None) -> None:
    if token is not None and not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if token is not None and not token.get_secret_value().strip():
        raise ValueError("token must not be empty")
