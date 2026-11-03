"""Classify an assignment worktree before a launch or a push.

Lane output (results, plots, scratch scripts) is untracked and does not
change what a commit reproduces, so it never blocks. A modified, staged,
renamed or deleted tracked file does: no commit would reproduce the run.
Untracked files are recorded in a snapshot commit beside HEAD so the exact
inputs of a run stay inspectable without joining the assignment branch.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from senpai_agent.git_transport import GitWorkflowPreconditionError, run_git as _git


# Untracked files up to this size are stored in a worktree snapshot. Larger
# files, such as checkpoints, profiles and built extensions, are named only.
SNAPSHOT_FILE_LIMIT_BYTES = 1 << 20
SNAPSHOT_REF_PREFIX = "refs/senpai/snapshots/"
_NAMED_PATHS = 20


@dataclass(frozen=True, slots=True)
class WorktreeSnapshot:
    """What the assignment worktree held when a run or push started."""

    head_sha: str
    # Every untracked file, whether or not the snapshot stores its content.
    untracked_paths: tuple[str, ...]
    # Commit of HEAD plus the stored untracked files, kept under
    # SNAPSHOT_REF_PREFIX; None when the worktree had no untracked files.
    snapshot_sha: str | None


def require_tracked_changes_committed(workspace: Path) -> None:
    """Refuse a worktree whose tracked files differ from HEAD.

    A modified, staged, renamed or deleted tracked file means no commit
    reproduces what would run or be published. Untracked files, such as
    results, plots and scratch scripts written beside the code, do not change
    what HEAD reproduces, so they do not block.
    """

    workspace = Path(workspace).resolve()
    _git(workspace, "rev-parse", "--is-inside-work-tree")
    status = _git(workspace, "status", "--porcelain", "--untracked-files=no")
    if status:
        # Each line is "XY path"; the first line's leading space is stripped.
        paths = [line.split(maxsplit=1)[-1] for line in status.splitlines()]
        raise GitWorkflowPreconditionError(
            "assignment worktree has uncommitted changes to tracked files: "
            f"{_name_paths(paths)}. Commit the exact implementation that will "
            "run, or discard an incidental change with `git checkout -- <path>`. "
            "Untracked files do not block."
        )


def snapshot_training_worktree(workspace: Path) -> WorktreeSnapshot:
    """Require committed tracked files and record the untracked ones.

    Untracked files are stored in a commit object whose parent is HEAD and
    whose ref is ``refs/senpai/snapshots/<sha>``, so the exact inputs of a run
    can be inspected later with ``git show --stat <sha>``. The commit never
    joins the assignment branch. Identical worktrees produce the same
    snapshot commit.
    """

    workspace = Path(workspace).resolve()
    require_tracked_changes_committed(workspace)
    head = _git(workspace, "rev-parse", "--verify", "HEAD^{commit}")
    untracked = tuple(
        path
        for path in _git(
            workspace, "ls-files", "--others", "--exclude-standard", "-z"
        ).split("\0")
        if path
    )
    if not untracked:
        return WorktreeSnapshot(head, (), None)
    # A path ending in "/" is a nested repository, which git cannot add as
    # files; a file a running process has already removed cannot be stored.
    stored = [
        path
        for path in untracked
        if not path.endswith("/") and _stored_size(workspace / path) is not None
    ]
    named_only = [path for path in untracked if path not in stored]
    with tempfile.NamedTemporaryFile(prefix="senpai-snapshot-index-") as index:
        index_env = {"GIT_INDEX_FILE": index.name}
        _git(workspace, "read-tree", head, extra_env=index_env)
        if stored:
            _git(
                workspace,
                "--literal-pathspecs",
                "add",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
                input_text="".join(f"{path}\0" for path in stored),
                extra_env=index_env,
            )
        tree = _git(workspace, "write-tree", extra_env=index_env)
    message = "Senpai worktree snapshot: HEAD plus the untracked files present at launch\n"
    if stored:
        message += "\nStored:\n" + "".join(f"  {path}\n" for path in stored)
    if named_only:
        message += (
            f"\nNamed only (nested repository or over {SNAPSHOT_FILE_LIMIT_BYTES} bytes):\n"
            + "".join(f"  {path}\n" for path in named_only)
        )
    date = _git(workspace, "log", "-1", "--format=%cI", head)
    identity = {
        "GIT_AUTHOR_NAME": "senpai",
        "GIT_AUTHOR_EMAIL": "senpai@senpai",
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_NAME": "senpai",
        "GIT_COMMITTER_EMAIL": "senpai@senpai",
        "GIT_COMMITTER_DATE": date,
    }
    snapshot = _git(
        workspace, "commit-tree", tree, "-p", head, "-m", message, extra_env=identity
    )
    _git(workspace, "update-ref", f"{SNAPSHOT_REF_PREFIX}{snapshot}", snapshot)
    return WorktreeSnapshot(head, untracked, snapshot)


def _stored_size(path: Path) -> int | None:
    try:
        size = path.lstat().st_size
    except FileNotFoundError:
        return None
    return size if size <= SNAPSHOT_FILE_LIMIT_BYTES else None


def _name_paths(paths: list[str]) -> str:
    named = ", ".join(paths[:_NAMED_PATHS])
    if len(paths) > _NAMED_PATHS:
        named += f" and {len(paths) - _NAMED_PATHS} more"
    return named
