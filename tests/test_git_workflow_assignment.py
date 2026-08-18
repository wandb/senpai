from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.git_workflow as git_workflow
from senpai_agent.git_workflow import (
    GitWorkflowPreconditionError,
    create_assignment_branch,
)

from git_workflow_support import commit_file, detached_commit, git, repository


def advisor_repository(tmp_path: Path) -> tuple[Path, Path, str]:
    workspace, remote, base_sha = repository(tmp_path)
    git(workspace, "branch", "-M", "schmidhuber")
    git(workspace, "push", "origin", "schmidhuber")
    return workspace, remote, base_sha


def test_create_assignment_branch_is_empty_idempotent_and_worktree_safe(
    tmp_path: Path,
):
    workspace, remote, base_sha = advisor_repository(tmp_path)
    (workspace / "advisor-notes.md").write_text("uncommitted research\n")
    original_status = git(workspace, "status", "--porcelain")

    first = create_assignment_branch(
        workspace,
        branch="student-one/lower-lr",
        base_branch="schmidhuber",
        expected_base_sha=base_sha,
        assignment_id="assignment-7",
    )
    repeated = create_assignment_branch(
        workspace,
        branch="student-one/lower-lr",
        base_branch="schmidhuber",
        expected_base_sha=base_sha,
        assignment_id="assignment-7",
    )

    assert first.changed is True
    assert repeated.changed is False
    assert first.head_sha == repeated.head_sha
    assert git(workspace, "branch", "--show-current") == "schmidhuber"
    assert git(workspace, "status", "--porcelain") == original_status
    assert (workspace / "advisor-notes.md").read_text() == "uncommitted research\n"
    assert git(remote, "rev-parse", f"{first.head_sha}^") == base_sha
    assert git(remote, "rev-parse", f"{first.head_sha}^{{tree}}") == git(
        remote,
        "rev-parse",
        f"{base_sha}^{{tree}}",
    )


def test_credentialed_assignment_creation_uses_an_isolated_repository(tmp_path: Path):
    workspace, remote, base_sha = advisor_repository(tmp_path)
    attacker_remote = tmp_path / "attacker.git"
    git(tmp_path, "init", "--bare", str(attacker_remote))
    git(workspace, "remote", "set-url", "--push", "origin", str(attacker_remote))

    created = create_assignment_branch(
        workspace,
        branch="student-one/lower-lr",
        base_branch="schmidhuber",
        expected_base_sha=base_sha,
        assignment_id="assignment-7",
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )

    assert git(remote, "rev-parse", f"refs/heads/{created.branch}") == created.head_sha
    assert git(attacker_remote, "branch", "--list", created.branch) == ""


def test_create_assignment_branch_rejects_foreign_existing_history(
    tmp_path: Path,
):
    workspace, remote, base_sha = advisor_repository(tmp_path)
    git(workspace, "checkout", "-b", "student-one/lower-lr")
    foreign_sha = commit_file(
        workspace,
        "foreign.py",
        "foreign = True\n",
        "foreign work",
    )
    git(workspace, "push", "origin", "student-one/lower-lr")

    with pytest.raises(GitWorkflowPreconditionError):
        create_assignment_branch(
            workspace,
            branch="student-one/lower-lr",
            base_branch="schmidhuber",
            expected_base_sha=base_sha,
            assignment_id="assignment-7",
        )

    assert git(remote, "rev-parse", "refs/heads/student-one/lower-lr") == foreign_sha


def test_create_assignment_branch_rejects_a_base_that_moves_during_fetch(
    tmp_path: Path,
    monkeypatch,
):
    workspace, remote, base_sha = advisor_repository(tmp_path)
    real_remote_head = git_workflow._remote_head
    advanced = False

    def advance_base_after_read(*args, **kwargs):
        nonlocal advanced
        remote_sha = real_remote_head(*args, **kwargs)
        if not advanced:
            advanced = True
            next_sha = detached_commit(workspace, base_sha, "advance base")
            git(workspace, "push", str(remote), f"{next_sha}:refs/heads/schmidhuber")
        return remote_sha

    monkeypatch.setattr(git_workflow, "_remote_head", advance_base_after_read)

    with pytest.raises(GitWorkflowPreconditionError):
        create_assignment_branch(
            workspace,
            branch="student-one/lower-lr",
            base_branch="schmidhuber",
            expected_base_sha=base_sha,
            assignment_id="assignment-7",
        )

    assert git(remote, "branch", "--list", "student-one/lower-lr") == ""
