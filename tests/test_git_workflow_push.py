import base64
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.git_workflow as git_workflow
from senpai_agent.git_transport import (
    GIT_EXECUTABLE,
    git_process_env,
    isolated_bare_repository,
    run_git,
)
from senpai_agent.git_workflow import (
    GitWorkflowPreconditionError,
    push_assignment_branch,
    require_clean_training_worktree,
    require_commit_contains_base,
)

from git_workflow_support import commit_file, detached_commit, git, repository


def test_authenticated_git_environment_keeps_only_the_scoped_header(tmp_path: Path):
    environment = git_process_env(SecretStr("typed-write-token"))

    result = subprocess.run(
        [
            GIT_EXECUTABLE,
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


def test_git_trusts_only_its_resolved_workspace_when_mount_owner_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workspace, remote, _base_sha = repository(tmp_path)
    alias = tmp_path / "mounted-workspace"
    alias.symlink_to(workspace, target_is_directory=True)
    hook_marker = tmp_path / "hook-ran"
    hook = workspace / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch '{hook_marker}'\n")
    hook.chmod(0o755)
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text("[safe]\n\tdirectory = *\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    ownership_check = {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}

    run_git(alias, "checkout", "-b", "mounted", extra_env=ownership_check)

    assert git(workspace, "branch", "--show-current") == "mounted"
    assert not hook_marker.exists()
    with pytest.raises(GitWorkflowPreconditionError, match="dubious ownership"):
        run_git(
            alias, "-C", str(remote), "rev-parse", "--git-dir",
            extra_env=ownership_check,
        )
    with isolated_bare_repository() as staging:
        assert run_git(
            staging, "rev-parse", "--is-bare-repository",
            extra_env=ownership_check,
        ) == "true"


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


@pytest.mark.parametrize("credentialed", [False, True])
def test_push_is_lease_guarded_verified_and_idempotent(
    tmp_path: Path, credentialed: bool
):
    workspace, remote, previous_sha = repository(tmp_path)
    candidate_sha = commit_file(
        workspace,
        "model.py",
        "baseline = 2\n",
        "candidate",
    )

    # ls-remote patterns also match ref-name suffixes. This decoy must not
    # make the real assignment look already published.
    git(
        workspace,
        "push",
        str(remote),
        f"{candidate_sha}:refs/heads/decoy/refs/heads/experiment-7",
    )

    first = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token") if credentialed else None,
    )
    assert git(workspace, "status", "--short", "--branch") == (
        "## experiment-7...origin/experiment-7"
    )
    repeated = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token") if credentialed else None,
    )

    assert first.changed is True
    assert first.head_sha == candidate_sha
    assert repeated.changed is False
    assert repeated.head_sha == candidate_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == candidate_sha
    assert git(workspace, "rev-parse", "origin/experiment-7") == candidate_sha


@pytest.mark.parametrize("credentialed", [False, True])
def test_push_publishes_only_the_validated_commit(
    tmp_path: Path, monkeypatch, credentialed: bool
):
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
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token") if credentialed else None,
    )

    assert pushed.head_sha == validated_sha
    assert git(workspace, "rev-parse", "HEAD") != validated_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == validated_sha
    assert git(workspace, "rev-parse", "origin/experiment-7") == validated_sha


@pytest.mark.parametrize("credentialed", [False, True])
def test_push_publishes_only_head_when_the_worktree_is_dirty(
    tmp_path: Path, credentialed: bool
):
    workspace, remote, remote_sha = repository(tmp_path)
    head_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    (workspace / "model.py").write_text("uncommitted = True\n")
    (workspace / "untracked.txt").write_text("dirty")

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=remote_sha,
        expected_local_sha=head_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token") if credentialed else None,
    )

    assert pushed.head_sha == head_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == head_sha
    assert (workspace / "model.py").read_text() == "uncommitted = True\n"
    assert (workspace / "untracked.txt").read_text() == "dirty"
    assert git(workspace, "branch", "--show-current") == "experiment-7"
    assert git(workspace, "rev-parse", "HEAD") == head_sha
    assert git(workspace, "rev-parse", "origin/experiment-7") == head_sha


@pytest.mark.parametrize("tracking_state", ["stale", "missing", "symbolic"])
def test_credentialed_push_repairs_tracking_on_retry(
    tmp_path: Path, tracking_state: str
):
    workspace, remote, previous_sha = repository(tmp_path)
    candidate_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    # Simulate a successful push followed by interruption before local bookkeeping.
    git(workspace, "push", remote.resolve().as_uri(), "HEAD:refs/heads/experiment-7")
    tracking_ref = "refs/remotes/origin/experiment-7"
    if tracking_state == "missing":
        git(workspace, "update-ref", "-d", tracking_ref)
    elif tracking_state == "symbolic":
        git(workspace, "branch", "unrelated", previous_sha)
        git(workspace, "symbolic-ref", tracking_ref, "refs/heads/unrelated")
    else:
        git(workspace, "update-ref", tracking_ref, previous_sha)

    result = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        expected_local_sha=candidate_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )

    assert result.changed is False
    assert git(workspace, "rev-parse", tracking_ref) == candidate_sha
    assert git(workspace, "status", "--short", "--branch") == (
        "## experiment-7...origin/experiment-7"
    )
    if tracking_state == "symbolic":
        assert git(workspace, "rev-parse", "refs/heads/unrelated") == previous_sha


@pytest.mark.parametrize("concurrent_change", ["advance", "create", "delete"])
def test_credentialed_push_preserves_concurrent_tracking_changes(
    tmp_path: Path, monkeypatch, concurrent_change: str
):
    workspace, remote, previous_sha = repository(tmp_path)
    candidate_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    newer_sha = detached_commit(workspace, candidate_sha, "another publisher")
    tracking_ref = "refs/remotes/origin/experiment-7"
    if concurrent_change == "create":
        git(workspace, "update-ref", "-d", tracking_ref)
    real_run = subprocess.run
    changed = False

    def update_tracking_concurrently(command, **kwargs):
        nonlocal changed
        during_network = concurrent_change == "advance" and command[1] == "ls-remote"
        during_update = (
            Path(kwargs["cwd"]) == workspace
            and command[1] == "update-ref"
            and tracking_ref in command[2:]
        )
        if not changed and (during_network or during_update):
            changed = True
            if concurrent_change == "delete":
                git(workspace, "update-ref", "-d", tracking_ref)
            else:
                git(workspace, "update-ref", tracking_ref, newer_sha)
        return real_run(command, **kwargs)

    monkeypatch.setattr(
        "senpai_agent.git_transport.subprocess.run",
        update_tracking_concurrently,
    )
    result = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )

    assert result.changed is True
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == candidate_sha
    if concurrent_change == "delete":
        assert git(workspace, "for-each-ref", tracking_ref) == ""
    else:
        assert git(workspace, "rev-parse", tracking_ref) == newer_sha


def test_credentialed_push_reports_tracking_failure_and_can_retry(tmp_path: Path):
    workspace, remote, previous_sha = repository(tmp_path)
    candidate_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    tracking_ref = "refs/remotes/origin/experiment-7"
    lock = workspace / ".git" / f"{tracking_ref}.lock"
    lock.touch()

    with pytest.raises(GitWorkflowPreconditionError, match="was published.*tracking"):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=previous_sha,
            authenticated_remote=remote.resolve().as_uri(),
            token=SecretStr("typed-write-token"),
        )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == candidate_sha
    assert git(workspace, "rev-parse", tracking_ref) == previous_sha
    lock.unlink()
    retried = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=remote.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )
    assert retried.changed is False
    assert git(workspace, "rev-parse", tracking_ref) == candidate_sha


def test_failed_push_verification_leaves_tracking_unchanged(tmp_path: Path, monkeypatch):
    workspace, remote, previous_sha = repository(tmp_path)
    candidate_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    real_remote_head = git_workflow._remote_head
    reads = 0

    def move_remote_before_verification(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            git(remote, "update-ref", "refs/heads/experiment-7", previous_sha)
        return real_remote_head(*args, **kwargs)

    monkeypatch.setattr(git_workflow, "_remote_head", move_remote_before_verification)
    with pytest.raises(RuntimeError, match="did not reach the pushed commit"):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=previous_sha,
            authenticated_remote=remote.resolve().as_uri(),
            token=SecretStr("typed-write-token"),
        )

    assert git(workspace, "rev-parse", "origin/experiment-7") == previous_sha
    assert git(workspace, "rev-parse", "HEAD") == candidate_sha


@pytest.mark.parametrize("credentialed", [False, True])
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
    credentialed: bool,
):
    workspace, remote, previous_sha = repository(tmp_path)
    remote_sha = detached_commit(workspace, previous_sha, "remote update")
    git(workspace, "push", str(remote), f"{remote_sha}:refs/heads/experiment-7")
    commit_file(workspace, "model.py", "baseline = 3\n", "local update")
    tracking_sha = git(workspace, "rev-parse", "origin/experiment-7")

    expected_remote_sha = previous_sha if lease == "stale" else remote_sha
    with pytest.raises(GitWorkflowPreconditionError, match=error):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=expected_remote_sha,
            authenticated_remote=remote.resolve().as_uri(),
            token=SecretStr("typed-write-token") if credentialed else None,
        )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == remote_sha
    if credentialed:
        assert git(workspace, "rev-parse", "origin/experiment-7") == tracking_sha


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
        assert command[0] == GIT_EXECUTABLE
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_CONFIG_SYSTEM"] == os.devnull
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        configuration = {
            env[f"GIT_CONFIG_KEY_{index}"]: env[f"GIT_CONFIG_VALUE_{index}"]
            for index in range(int(env["GIT_CONFIG_COUNT"]))
        }
        assert configuration["core.hooksPath"] == os.devnull
        assert configuration["credential.helper"] == ""
        assert configuration["core.fsmonitor"] == "false"
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
        if {"ls-remote", "fetch", "push"}.intersection(command):
            assert authorization is not None
            encoded = authorization.removeprefix("Authorization: Basic ")
            assert base64.b64decode(encoded).decode() == (
                "x-access-token:typed-write-token"
            )
            assert Path(kwargs["cwd"]) != workspace
        else:
            assert authorization is None
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


@pytest.mark.parametrize("hook_name", ["pre-push", "reference-transaction"])
def test_typed_push_ignores_agent_controlled_path_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hook_name: str,
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
    hook = workspace / ".git" / "hooks" / hook_name
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


def test_push_from_a_shallow_checkout_keeps_the_history_boundary(tmp_path: Path):
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    git(tmp_path, "init", "--bare", str(origin))
    git(tmp_path, "init", str(seed))
    git(seed, "config", "user.name", "Student")
    git(seed, "config", "user.email", "student@example.com")
    commit_file(seed, "model.py", "baseline = 0\n", "root")
    previous_sha = commit_file(seed, "model.py", "baseline = 1\n", "baseline")
    git(seed, "branch", "-M", "experiment-7")
    git(seed, "push", str(origin), "experiment-7")
    workspace = tmp_path / "workspace"
    git(
        tmp_path,
        "clone",
        "--depth",
        "1",
        "--branch",
        "experiment-7",
        origin.resolve().as_uri(),
        str(workspace),
    )
    git(workspace, "config", "user.name", "Student")
    git(workspace, "config", "user.email", "student@example.com")
    candidate_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    assert (workspace / ".git" / "shallow").is_file()

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        authenticated_remote=origin.resolve().as_uri(),
        token=SecretStr("typed-write-token"),
    )

    assert pushed.head_sha == candidate_sha
    assert git(origin, "rev-parse", "refs/heads/experiment-7") == candidate_sha


def test_clean_worktree_check_ignores_a_lying_fsmonitor_hook(tmp_path: Path):
    workspace, _remote, _sha = repository(tmp_path)
    fsmonitor = tmp_path / "fsmonitor.sh"
    fsmonitor.write_text("#!/bin/sh\nprintf 'token\\0'\n", encoding="utf-8")
    fsmonitor.chmod(0o755)
    git(workspace, "config", "core.fsmonitor", str(fsmonitor))
    git(workspace, "status", "--porcelain")
    (workspace / "model.py").write_text("baseline = 2\n")

    with pytest.raises(GitWorkflowPreconditionError, match="clean before training"):
        require_clean_training_worktree(workspace)
