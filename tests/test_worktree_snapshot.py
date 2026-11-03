import subprocess
from pathlib import Path

import pytest
from git_workflow_support import commit_file, git, repository

from senpai_agent.git_transport import GitWorkflowPreconditionError
from senpai_agent.worktree_snapshot import (
    SNAPSHOT_FILE_LIMIT_BYTES,
    SNAPSHOT_REF_PREFIX,
    require_tracked_changes_committed,
    snapshot_training_worktree,
)


def test_a_clean_worktree_snapshots_nothing(tmp_path: Path):
    workspace, _remote, head = repository(tmp_path)

    snapshot = snapshot_training_worktree(workspace)

    assert snapshot.head_sha == head
    assert snapshot.untracked_paths == ()
    assert snapshot.snapshot_sha is None
    assert git(workspace, "for-each-ref", SNAPSHOT_REF_PREFIX) == ""


def test_untracked_lane_output_is_stored_beside_head_and_named(tmp_path: Path):
    workspace, _remote, head = repository(tmp_path)
    lane = workspace / "research" / "lanes" / "fold4"
    lane.mkdir(parents=True)
    (lane / "ab.py").write_text("print('twin')\n")
    (lane / "results.json").write_text('{"gain": 0.2}\n')

    snapshot = snapshot_training_worktree(workspace)

    assert snapshot.head_sha == head
    assert snapshot.untracked_paths == (
        "research/lanes/fold4/ab.py",
        "research/lanes/fold4/results.json",
    )
    assert snapshot.snapshot_sha is not None
    ref = f"{SNAPSHOT_REF_PREFIX}{snapshot.snapshot_sha}"
    assert git(workspace, "rev-parse", ref) == snapshot.snapshot_sha
    assert git(workspace, "rev-parse", f"{snapshot.snapshot_sha}^") == head
    assert git(workspace, "show", f"{snapshot.snapshot_sha}:research/lanes/fold4/ab.py") == (
        "print('twin')"
    )
    # The snapshot never joins the branch and leaves the index untouched.
    assert git(workspace, "rev-parse", "HEAD") == head
    assert git(workspace, "status", "--porcelain", "--untracked-files=all") == (
        "?? research/lanes/fold4/ab.py\n?? research/lanes/fold4/results.json"
    )
    assert git(workspace, "diff", "--cached", "--stat") == ""


def test_identical_worktrees_share_one_snapshot_commit(tmp_path: Path):
    workspace, _remote, _head = repository(tmp_path)
    (workspace / "notes.txt").write_text("same\n")

    first = snapshot_training_worktree(workspace)
    second = snapshot_training_worktree(workspace)
    (workspace / "notes.txt").write_text("changed\n")
    third = snapshot_training_worktree(workspace)

    assert first.snapshot_sha == second.snapshot_sha
    assert third.snapshot_sha != first.snapshot_sha
    assert git(workspace, "for-each-ref", "--format=%(refname)", SNAPSHOT_REF_PREFIX).split() == sorted(
        f"{SNAPSHOT_REF_PREFIX}{sha}" for sha in {first.snapshot_sha, third.snapshot_sha}
    )


def test_large_files_and_nested_repositories_are_named_but_not_stored(tmp_path: Path):
    workspace, _remote, _head = repository(tmp_path)
    (workspace / "small.txt").write_text("kept\n")
    (workspace / "checkpoint.bin").write_bytes(b"\0" * (SNAPSHOT_FILE_LIMIT_BYTES + 1))
    nested = workspace / "vendor"
    git(workspace, "init", "--quiet", str(nested))

    snapshot = snapshot_training_worktree(workspace)

    assert snapshot.untracked_paths == ("checkpoint.bin", "small.txt", "vendor/")
    stored = git(workspace, "ls-tree", "-r", "--name-only", snapshot.snapshot_sha).split()
    assert "small.txt" in stored
    assert "checkpoint.bin" not in stored
    assert not any(path.startswith("vendor") for path in stored)
    message = git(workspace, "log", "-1", "--format=%B", snapshot.snapshot_sha)
    assert "Stored:\n  small.txt" in message
    assert "checkpoint.bin" in message.split("Named only")[1]
    assert "vendor/" in message.split("Named only")[1]


@pytest.mark.parametrize(
    "dirty",
    [
        lambda workspace: (workspace / "model.py").write_text("baseline = 2\n"),
        lambda workspace: (
            (workspace / "new.py").write_text("x = 1\n"),
            git(workspace, "add", "new.py"),
        ),
        lambda workspace: git(workspace, "rm", "--quiet", "model.py"),
    ],
    ids=["modified", "staged", "deleted"],
)
def test_changes_to_tracked_files_refuse_and_name_the_path(tmp_path: Path, dirty):
    workspace, _remote, _head = repository(tmp_path)
    (workspace / "plot.png").write_bytes(b"png")
    dirty(workspace)

    with pytest.raises(GitWorkflowPreconditionError) as error:
        snapshot_training_worktree(workspace)

    message = str(error.value)
    assert "uncommitted changes to tracked files" in message
    assert ("model.py" in message) or ("new.py" in message)
    assert "plot.png" not in message
    assert "Untracked files do not block" in message
    assert git(workspace, "for-each-ref", SNAPSHOT_REF_PREFIX) == ""


def test_tracked_change_check_names_at_most_twenty_paths(tmp_path: Path):
    workspace, _remote, _head = repository(tmp_path)
    for index in range(25):
        commit_file(workspace, f"file{index:02d}.py", "a\n", f"file {index}")
    for index in range(25):
        (workspace / f"file{index:02d}.py").write_text("b\n")

    with pytest.raises(GitWorkflowPreconditionError, match="file19.py and 5 more"):
        require_tracked_changes_committed(workspace)


def test_snapshot_requires_a_git_worktree(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(GitWorkflowPreconditionError):
        snapshot_training_worktree(plain)
    assert subprocess.run(
        ["git", "-C", str(plain), "rev-parse"], capture_output=True
    ).returncode != 0
