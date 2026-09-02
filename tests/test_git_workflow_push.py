import base64
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.git_workflow as git_workflow
from senpai_agent.git_workflow import (
    GitWorkflowPreconditionError,
    git_process_env,
    push_assignment_branch,
    require_commit_contains_base,
)

from git_workflow_support import commit_file, detached_commit, git, repository


def test_authenticated_git_environment_keeps_only_the_scoped_header(tmp_path: Path):
    environment = git_process_env(SecretStr("typed-write-token"))

    result = subprocess.run(
        [
            git_workflow.GIT_EXECUTABLE,
            "config",
            "--get-urlmatch",
            "http.extraHeader",
            "https://github.com/acme/widgets.git",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    encoded = result.stdout.strip().removeprefix("Authorization: Basic ")
    assert base64.b64decode(encoded).decode() == "x-access-token:typed-write-token"


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


def test_push_publishes_only_head_when_the_worktree_is_dirty(tmp_path: Path):
    workspace, remote, remote_sha = repository(tmp_path)
    head_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    (workspace / "model.py").write_text("uncommitted = True\n")
    (workspace / "untracked.txt").write_text("dirty")

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=remote_sha,
        expected_local_sha=head_sha,
    )

    assert pushed.head_sha == head_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == head_sha
    assert (workspace / "model.py").read_text() == "uncommitted = True\n"


@pytest.mark.parametrize("change", ["add", "modify", "delete"])
def test_typed_push_rejects_program_policy_changes(tmp_path: Path, change: str):
    workspace, remote, remote_sha = repository(tmp_path)
    if change != "add":
        remote_sha = commit_file(
            workspace,
            "program.md",
            "Operator-reviewed policy.\n",
            "add program policy",
        )
        git(workspace, "push", "origin", "experiment-7")

    if change == "add":
        policy = workspace / "policy" / "program.md"
        policy.parent.mkdir()
        policy.write_text("Agent-authored policy.\n")
        git(workspace, "add", str(policy.relative_to(workspace)))
    elif change == "modify":
        (workspace / "program.md").write_text("Agent-authored policy.\n")
        git(workspace, "add", "program.md")
    else:
        git(workspace, "rm", "program.md")
    git(workspace, "commit", "-m", f"{change} program policy")

    with pytest.raises(
        GitWorkflowPreconditionError,
        match="explicit operator publication",
    ):
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
    workspace, remote, previous_sha = repository(tmp_path)
    commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    real_run = subprocess.run

    def guarded_run(command, **kwargs):
        env = kwargs["env"]
        assert all("typed-write-token" not in argument for argument in command)
        assert "ambient-write-token" not in env.values()
        assert "ambient-gh-token" not in env.values()
        assert "GITHUB_TOKEN" not in env
        assert "GH_TOKEN" not in env
        assert command[0] == git_workflow.GIT_EXECUTABLE
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_CONFIG_SYSTEM"] == os.devnull
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        configuration = {
            env[f"GIT_CONFIG_KEY_{index}"]: env[f"GIT_CONFIG_VALUE_{index}"]
            for index in range(int(env["GIT_CONFIG_COUNT"]))
        }
        assert configuration["core.hooksPath"] == os.devnull
        assert configuration["credential.helper"] == ""
        assert configuration["http.proxy"] == ""
        assert configuration["http.sslVerify"] == "true"
        assert configuration["http.followRedirects"] == "false"
        authorization = next(
            (
                value
                for value in configuration.values()
                if value.startswith("Authorization: Basic ")
            ),
            None,
        )
        if authorization is not None:
            encoded = authorization.removeprefix("Authorization: Basic ")
            assert base64.b64decode(encoded).decode() == (
                "x-access-token:typed-write-token"
            )
            assert Path(kwargs["cwd"]) != workspace
        else:
            assert not any(
                value.startswith("Authorization: Basic ")
                for value in configuration.values()
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr("senpai_agent.git_transport.subprocess.run", guarded_run)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")
    monkeypatch.setenv("GH_TOKEN", "ambient-gh-token")

    push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )


def test_typed_push_ignores_agent_controlled_path_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workspace, remote, previous_sha = repository(tmp_path)
    commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    wrapper_marker = tmp_path / "wrapper-ran"
    hook_marker = tmp_path / "hook-ran"
    wrapper_dir = tmp_path / "agent-bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "git"
    wrapper.write_text(
        f"#!/bin/sh\nprintf ran > {wrapper_marker}\nexit 99\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    hook = workspace / ".git" / "hooks" / "pre-push"
    hook.write_text(
        f"#!/bin/sh\nprintf '%s' \"$GIT_CONFIG_VALUE_0\" > {hook_marker}\nexit 99\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrapper_dir}:{os.environ['PATH']}")

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )

    assert pushed.changed is True
    assert not wrapper_marker.exists()
    assert not hook_marker.exists()


def test_credentialed_push_ignores_checkout_remote_and_http_configuration(
    tmp_path: Path,
):
    workspace, remote, previous_sha = repository(tmp_path)
    attacker_remote = tmp_path / "attacker.git"
    git(tmp_path, "init", "--bare", str(attacker_remote))
    commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    trusted_url = remote.resolve().as_uri()
    git(workspace, "remote", "set-url", "--push", "origin", str(attacker_remote))
    git(workspace, "config", f"http.{trusted_url}.proxy", "http://127.0.0.1:1")
    git(workspace, "config", f"http.{trusted_url}.sslVerify", "false")

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=trusted_url,
        token=SecretStr("typed-write-token"),
    )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == pushed.head_sha
    assert git(attacker_remote, "branch", "--list", "experiment-7") == ""


def test_credentialed_push_does_not_run_checkout_status_helpers(tmp_path: Path):
    workspace, remote, previous_sha = repository(tmp_path)
    fsmonitor_marker = tmp_path / "fsmonitor-ran"
    filter_marker = tmp_path / "filter-ran"
    fsmonitor = tmp_path / "fsmonitor.sh"
    clean_filter = tmp_path / "clean-filter.sh"
    fsmonitor.write_text(
        f"#!/bin/sh\nprintf ran > {fsmonitor_marker}\n",
        encoding="utf-8",
    )
    clean_filter.write_text(
        f"#!/bin/sh\nprintf ran > {filter_marker}\ncat\n",
        encoding="utf-8",
    )
    fsmonitor.chmod(0o755)
    clean_filter.chmod(0o755)
    git(workspace, "config", "core.fsmonitor", str(fsmonitor))
    git(workspace, "config", "filter.evil.clean", str(clean_filter))
    (workspace / ".gitattributes").write_text("model.py filter=evil\n")
    candidate_sha = commit_file(
        workspace,
        "model.py",
        "baseline = 2\n",
        "candidate",
    )
    fsmonitor_marker.unlink(missing_ok=True)
    filter_marker.unlink(missing_ok=True)

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        expected_local_sha=candidate_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )

    assert pushed.head_sha == candidate_sha
    assert not fsmonitor_marker.exists()
    assert not filter_marker.exists()
