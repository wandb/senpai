from pathlib import Path

import pytest

import senpai_agent.git_workflow as git_workflow
from senpai_agent.git_assignment import require_remote_assignment_history
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


def test_require_remote_assignment_history_accepts_work_after_an_older_base(
    tmp_path: Path,
):
    workspace, remote, base_sha = advisor_repository(tmp_path)
    git(workspace, "checkout", "-b", "student-one/lower-lr")
    head_sha = commit_file(workspace, "candidate.py", "candidate = True\n", "candidate")
    git(workspace, "push", "origin", "student-one/lower-lr")
    advanced_base = detached_commit(workspace, base_sha, "advance base")
    git(workspace, "push", str(remote), f"{advanced_base}:refs/heads/schmidhuber")

    require_remote_assignment_history(
        workspace,
        branch="student-one/lower-lr",
        expected_head_sha=head_sha,
        base_branch="schmidhuber",
        expected_base_sha=base_sha,
    )


def test_require_remote_assignment_history_rejects_an_unknown_head(tmp_path: Path):
    workspace, _remote, base_sha = advisor_repository(tmp_path)
    git(workspace, "checkout", "-b", "student-one/lower-lr")
    commit_file(workspace, "candidate.py", "candidate = True\n", "candidate")
    git(workspace, "push", "origin", "student-one/lower-lr")

    with pytest.raises(GitWorkflowPreconditionError, match="fetched Git commits"):
        require_remote_assignment_history(
            workspace,
            branch="student-one/lower-lr",
            expected_head_sha="f" * 40,
            base_branch="schmidhuber",
            expected_base_sha=base_sha,
        )


def test_require_remote_assignment_history_rejects_foreign_history(tmp_path: Path):
    workspace, _remote, base_sha = advisor_repository(tmp_path)
    tree = git(workspace, "rev-parse", f"{base_sha}^{{tree}}")
    unrelated = git(workspace, "commit-tree", tree, "-m", "unrelated root")
    git(workspace, "checkout", "--detach", unrelated)
    head_sha = commit_file(workspace, "candidate.py", "candidate = True\n", "candidate")
    git(workspace, "push", "origin", f"{head_sha}:refs/heads/student-one/lower-lr")

    with pytest.raises(GitWorkflowPreconditionError, match="does not share"):
        require_remote_assignment_history(
            workspace,
            branch="student-one/lower-lr",
            expected_head_sha=head_sha,
            base_branch="schmidhuber",
            expected_base_sha=base_sha,
        )


def test_require_remote_assignment_history_accepts_a_replayed_ancestor_head(
    tmp_path: Path,
):
    workspace, _remote, base_sha = advisor_repository(tmp_path)
    git(workspace, "checkout", "-b", "student-one/lower-lr")
    expected_head = commit_file(
        workspace, "candidate.py", "candidate = True\n", "candidate"
    )
    commit_file(workspace, "notes.md", "continued work\n", "continue")
    git(workspace, "push", "origin", "student-one/lower-lr")

    require_remote_assignment_history(
        workspace,
        branch="student-one/lower-lr",
        expected_head_sha=expected_head,
        base_branch="schmidhuber",
        expected_base_sha=base_sha,
    )


def test_require_remote_assignment_history_rejects_an_older_ancestral_base(
    tmp_path: Path,
):
    workspace, remote, old_base = advisor_repository(tmp_path)
    current_base = detached_commit(workspace, old_base, "advance base")
    git(workspace, "push", str(remote), f"{current_base}:refs/heads/schmidhuber")
    git(workspace, "checkout", "--detach", current_base)
    git(workspace, "checkout", "-b", "student-one/lower-lr")
    head_sha = commit_file(workspace, "candidate.py", "candidate = True\n", "candidate")
    git(workspace, "push", "origin", "student-one/lower-lr")

    with pytest.raises(GitWorkflowPreconditionError, match="diverges at"):
        require_remote_assignment_history(
            workspace,
            branch="student-one/lower-lr",
            expected_head_sha=head_sha,
            base_branch="schmidhuber",
            expected_base_sha=old_base,
        )
