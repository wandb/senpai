"""Read-only verification for adopting an existing remote assignment branch."""

from pathlib import Path

from pydantic import SecretStr

from senpai_agent.git_workflow import (
    GitWorkflowPreconditionError,
    _git,
    _remote_head,
    _validate_token,
)


def require_remote_assignment_history(
    workspace: Path,
    *,
    branch: str,
    expected_head_sha: str,
    base_branch: str,
    expected_base_sha: str,
    remote: str = "origin",
    token: SecretStr | None = None,
) -> None:
    """Verify that an existing remote branch contains its declared research base."""

    workspace = Path(workspace).resolve()
    _validate_token(token)
    _git(workspace, "rev-parse", "--is-inside-work-tree")
    _git(workspace, "check-ref-format", "--branch", branch)
    _git(workspace, "check-ref-format", "--branch", base_branch)
    if branch == base_branch:
        raise GitWorkflowPreconditionError(
            "assignment branch must differ from the base branch"
        )
    if not expected_head_sha.strip() or not expected_base_sha.strip():
        raise ValueError("expected head and base SHAs must not be empty")

    remote_head = _remote_head(workspace, remote, branch, token=token)
    if not remote_head:
        raise GitWorkflowPreconditionError(
            f"remote assignment branch {branch!r} does not exist"
        )
    remote_base = _remote_head(workspace, remote, base_branch, token=token)
    if not remote_base:
        raise GitWorkflowPreconditionError(
            f"remote base branch {base_branch!r} does not exist"
        )

    for ref, expected in ((base_branch, remote_base), (branch, remote_head)):
        _git(
            workspace,
            "fetch",
            "--no-tags",
            remote,
            f"refs/heads/{ref}",
            token=token,
        )
        if _git(workspace, "rev-parse", "FETCH_HEAD") != expected:
            raise GitWorkflowPreconditionError(
                f"remote branch {ref!r} moved while verifying assignment history"
            )

    try:
        base = _git(workspace, "rev-parse", f"{expected_base_sha}^{{commit}}")
        head = _git(workspace, "rev-parse", f"{expected_head_sha}^{{commit}}")
    except GitWorkflowPreconditionError as error:
        raise GitWorkflowPreconditionError(
            "expected assignment head and base must be fetched Git commits"
        ) from error
    if base != expected_base_sha or head != expected_head_sha:
        raise GitWorkflowPreconditionError(
            "assignment SHAs must be full Git commit object IDs"
        )
    try:
        _git(workspace, "merge-base", "--is-ancestor", head, remote_head)
    except GitWorkflowPreconditionError as error:
        raise GitWorkflowPreconditionError(
            f"remote head {remote_head} does not contain expected head {head}"
        ) from error
    try:
        branch_point = _git(workspace, "merge-base", remote_base, remote_head)
    except GitWorkflowPreconditionError as error:
        raise GitWorkflowPreconditionError(
            f"assignment head {remote_head} does not share research-base history"
        ) from error
    if branch_point != base:
        raise GitWorkflowPreconditionError(
            f"assignment branch diverges at {branch_point}, expected base {base}"
        )

    if (
        _remote_head(workspace, remote, branch, token=token) != remote_head
        or _remote_head(workspace, remote, base_branch, token=token) != remote_base
    ):
        raise GitWorkflowPreconditionError(
            "remote assignment history moved while it was being verified"
        )
