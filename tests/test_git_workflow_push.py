import base64
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.git_workflow as git_workflow
from senpai_agent.git_workflow import (
    GitWorkflowPreconditionError,
    push_assignment_branch,
    require_commit_contains_base,
)

from git_workflow_support import commit_file, detached_commit, git, repository


def test_result_commit_must_contain_its_assigned_research_base(tmp_path: Path):
    workspace, _remote, base_sha = repository(tmp_path)
    result_sha = commit_file(
        workspace,
        "model.py",
        "baseline = 2\n",
        "candidate",
    )

    require_commit_contains_base(
        workspace,
        commit_sha=result_sha,
        base_sha=base_sha,
    )


def test_result_commit_rejects_an_unrelated_research_base(tmp_path: Path):
    workspace, _remote, _base_sha = repository(tmp_path)
    result_sha = commit_file(
        workspace,
        "model.py",
        "baseline = 2\n",
        "candidate",
    )
    tree = git(workspace, "rev-parse", f"{result_sha}^{{tree}}")
    unrelated_base = git(workspace, "commit-tree", tree, "-m", "unrelated base")

    with pytest.raises(
        GitWorkflowPreconditionError,
        match="does not contain assigned research base",
    ):
        require_commit_contains_base(
            workspace,
            commit_sha=result_sha,
            base_sha=unrelated_base,
        )


def test_push_is_lease_guarded_verified_and_idempotent(tmp_path: Path):
    workspace, remote, previous_sha = repository(tmp_path)
    candidate_sha = commit_file(
        workspace,
        "model.py",
        "baseline = 2\n",
        "candidate",
    )

    first = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
    )
    repeated = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
    )

    assert first.changed is True
    assert first.head_sha == candidate_sha
    assert repeated.changed is False
    assert repeated.head_sha == candidate_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == candidate_sha


@pytest.mark.parametrize(
    ("role", "branch"),
    [
        ("advisor", "experiment-7"),
        ("student", "student-one/experiment-7"),
    ],
)
def test_push_ref_is_accepted_by_the_role_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    branch: str,
):
    workspace, remote, previous_sha = repository(tmp_path)
    if branch != "experiment-7":
        git(workspace, "branch", "-m", branch)
        git(workspace, "push", "-u", "origin", branch)
    guard = (
        Path(__file__).parents[1] / "plugins" / "senpai" / "scripts" / "git-guard.sh"
    )
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; install_senpai_target_git_guard "$2"',
            "install-guard",
            str(guard),
            str(workspace),
        ],
        check=True,
    )
    monkeypatch.setenv("SENPAI_ROLE", role)
    monkeypatch.setenv("ADVISOR_BRANCH", "experiment-7")
    monkeypatch.setenv("STUDENT_NAME", "student-one")
    monkeypatch.setenv("STUDENT_NAMES", "student-one")
    commit_file(workspace, "model.py", "baseline = 2\n", "candidate")

    pushed = push_assignment_branch(
        workspace,
        branch=branch,
        expected_remote_sha=previous_sha,
    )

    assert git(remote, "rev-parse", f"refs/heads/{branch}") == pushed.head_sha


def test_push_publishes_only_the_validated_commit(tmp_path: Path, monkeypatch):
    workspace, remote, previous_sha = repository(tmp_path)
    validated_sha = commit_file(
        workspace,
        "model.py",
        "baseline = 2\n",
        "validated candidate",
    )
    real_remote_head = git_workflow._remote_head
    advanced = False

    def advance_local_head_after_validation(*args, **kwargs):
        nonlocal advanced
        remote_sha = real_remote_head(*args, **kwargs)
        if not advanced:
            advanced = True
            commit_file(
                workspace,
                "model.py",
                "baseline = 3\n",
                "unvalidated candidate",
            )
        return remote_sha

    monkeypatch.setattr(
        git_workflow,
        "_remote_head",
        advance_local_head_after_validation,
    )

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        expected_local_sha=validated_sha,
    )

    assert pushed.head_sha == validated_sha
    assert git(workspace, "rev-parse", "HEAD") != validated_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == validated_sha


def test_push_rejects_a_dirty_worktree_without_publishing(tmp_path: Path):
    workspace, remote, remote_sha = repository(tmp_path)
    (workspace / "untracked.txt").write_text("dirty")

    with pytest.raises(GitWorkflowPreconditionError):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=remote_sha,
        )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == remote_sha


@pytest.mark.parametrize(
    ("lease", "error"),
    [
        ("stale", "remote head"),
        ("current", "fast-forward"),
    ],
)
def test_push_rejects_remote_divergence_without_publishing(
    tmp_path: Path,
    lease: str,
    error: str,
):
    workspace, remote, previous_sha = repository(tmp_path)
    remote_sha = detached_commit(workspace, previous_sha, "remote update")
    git(workspace, "push", str(remote), f"{remote_sha}:refs/heads/experiment-7")
    commit_file(workspace, "model.py", "baseline = 3\n", "local update")

    expected_remote_sha = previous_sha if lease == "stale" else remote_sha
    with pytest.raises(GitWorkflowPreconditionError, match=error):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=expected_remote_sha,
        )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == remote_sha


@pytest.mark.parametrize("mismatch", ["branch", "head"])
def test_push_rejects_the_wrong_branch_or_head(tmp_path: Path, mismatch: str):
    workspace, remote, remote_sha = repository(tmp_path)
    commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    branch = "experiment-7"
    expected_local_sha = "f" * 40
    if mismatch == "branch":
        git(workspace, "branch", "-m", "wrong-branch")
        expected_local_sha = None

    with pytest.raises(GitWorkflowPreconditionError):
        push_assignment_branch(
            workspace,
            branch=branch,
            expected_remote_sha=remote_sha,
            expected_local_sha=expected_local_sha,
        )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == remote_sha


def test_typed_push_auth_is_confined_to_network_git_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workspace, _remote, previous_sha = repository(tmp_path)
    commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    real_run = subprocess.run

    def guarded_run(command, **kwargs):
        env = kwargs["env"]
        assert all("typed-write-token" not in argument for argument in command)
        assert "ambient-write-token" not in env.values()
        assert "ambient-gh-token" not in env.values()
        assert "GITHUB_TOKEN" not in env
        assert "GH_TOKEN" not in env
        if command[1] in {"ls-remote", "fetch", "push"}:
            encoded = env["GIT_CONFIG_VALUE_0"].removeprefix("Authorization: Basic ")
            assert base64.b64decode(encoded).decode() == (
                "x-access-token:typed-write-token"
            )
        else:
            assert "GIT_CONFIG_VALUE_0" not in env
        return real_run(command, **kwargs)

    monkeypatch.setattr("senpai_agent.git_workflow.subprocess.run", guarded_run)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")
    monkeypatch.setenv("GH_TOKEN", "ambient-gh-token")

    push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        token=SecretStr("typed-write-token"),
    )
