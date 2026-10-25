import base64
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.git_workflow as git_workflow
from senpai_agent.git_transport import GitWorkflowPreconditionError, git_process_env
from senpai_agent.git_workflow import PushResult, sync_remote_branch

from git_workflow_support import commit_file, detached_commit, git, repository


def advance_branch_elsewhere(tmp_path: Path, remote: Path, branch: str) -> str:
    """Move the remote branch the way a merge through the GitHub API does.

    The checkout under test takes no part, so it ends up holding neither the new
    tip nor its objects.
    """

    elsewhere = tmp_path / "elsewhere"
    git(tmp_path, "init", str(elsewhere))
    git(elsewhere, "config", "user.name", "Advisor")
    git(elsewhere, "config", "user.email", "advisor@example.com")
    git(elsewhere, "remote", "add", "origin", str(remote))
    git(elsewhere, "fetch", "origin", branch)
    git(elsewhere, "checkout", "-B", branch, "FETCH_HEAD")
    head = commit_file(elsewhere, "merged.py", "merged = 1\n", "merge experiment")
    git(elsewhere, "push", "origin", f"{branch}:{branch}")
    return head


def has_object(workspace: Path, sha: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=workspace,
        capture_output=True,
    ).returncode == 0


def test_sync_imports_commits_the_checkout_never_pushed(tmp_path: Path):
    workspace, remote, base_sha = repository(tmp_path)
    merged = advance_branch_elsewhere(tmp_path, remote, "experiment-7")
    assert git(workspace, "rev-parse", "refs/remotes/origin/experiment-7") == base_sha
    assert not has_object(workspace, merged)

    synced = sync_remote_branch(workspace, branch="experiment-7")

    assert synced == PushResult(True, "experiment-7", merged)
    assert git(workspace, "rev-parse", "refs/remotes/origin/experiment-7") == merged
    # The pointer alone is useless; the objects decide whether a rebase can run.
    assert has_object(workspace, merged)
    assert git(workspace, "merge-base", base_sha, merged) == base_sha
    assert git(workspace, "rev-parse", "HEAD") == base_sha
    assert git(workspace, "status", "--porcelain") == ""
    assert git(workspace, "for-each-ref", "refs/senpai/") == ""


def test_sync_is_repeatable_and_reports_an_unmoved_branch(tmp_path: Path):
    workspace, remote, base_sha = repository(tmp_path)
    merged = advance_branch_elsewhere(tmp_path, remote, "experiment-7")

    assert sync_remote_branch(workspace, branch="experiment-7").changed
    repeated = sync_remote_branch(workspace, branch="experiment-7")

    assert repeated == PushResult(False, "experiment-7", merged)
    assert git(workspace, "rev-parse", "refs/remotes/origin/experiment-7") == merged
    assert base_sha != merged


def test_sync_never_moves_a_branch_behind_a_symbolic_tracking_ref(tmp_path: Path):
    workspace, remote, _base_sha = repository(tmp_path)
    local = commit_file(workspace, "notes.md", "unpublished\n", "research notes")
    merged = advance_branch_elsewhere(tmp_path, remote, "experiment-7")
    git(
        workspace,
        "symbolic-ref",
        "refs/remotes/origin/experiment-7",
        "refs/heads/experiment-7",
    )

    synced = sync_remote_branch(workspace, branch="experiment-7")

    assert synced.head_sha == merged
    assert git(workspace, "rev-parse", "refs/heads/experiment-7") == local
    assert git(workspace, "rev-parse", "refs/remotes/origin/experiment-7") == merged
    assert git(workspace, "status", "--porcelain") == ""


def test_sync_keeps_a_newer_concurrent_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, remote, base_sha = repository(tmp_path)
    advance_branch_elsewhere(tmp_path, remote, "experiment-7")
    published = detached_commit(workspace, base_sha, "published concurrently")
    real_fetch = git_workflow._fetch_refs

    def fetch_then_publish(*args, **kwargs):
        real_fetch(*args, **kwargs)
        git(workspace, "update-ref", "refs/remotes/origin/experiment-7", published)

    monkeypatch.setattr(git_workflow, "_fetch_refs", fetch_then_publish)

    with pytest.raises(GitWorkflowPreconditionError, match="moved during the sync"):
        sync_remote_branch(workspace, branch="experiment-7")

    assert git(workspace, "rev-parse", "refs/remotes/origin/experiment-7") == published


def test_sync_fails_clearly_when_the_branch_does_not_exist(tmp_path: Path):
    workspace, _remote, _base_sha = repository(tmp_path)

    with pytest.raises(GitWorkflowPreconditionError):
        sync_remote_branch(workspace, branch="never-created")


def test_sync_requires_an_authenticated_remote_for_a_credentialed_fetch(
    tmp_path: Path,
):
    workspace, _remote, _base_sha = repository(tmp_path)

    with pytest.raises(ValueError):
        sync_remote_branch(
            workspace,
            branch="experiment-7",
            token=SecretStr("typed-token"),
        )


def test_sync_keeps_the_credential_out_of_the_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, remote, base_sha = repository(tmp_path)
    merged = advance_branch_elsewhere(tmp_path, remote, "experiment-7")
    github_remote = "https://github.com/acme/widgets.git"
    real_run = subprocess.run
    hops: list[str] = []
    workspace_objects = (workspace / ".git" / "objects").resolve()

    def guarded_run(command, **kwargs):
        if "env" not in kwargs:  # this test's own assertions, not code under test
            return real_run(command, **kwargs)
        env = kwargs["env"]
        assert "typed-token" not in command
        if github_remote in command:
            hops.append("authenticated")
            assert Path(kwargs["cwd"]).name == "objects.git"
            assert env["GIT_ALLOW_PROTOCOL"] == "https"
            configuration = {
                env[f"GIT_CONFIG_KEY_{index}"]: env[f"GIT_CONFIG_VALUE_{index}"]
                for index in range(int(env["GIT_CONFIG_COUNT"]))
            }
            encoded = configuration[
                "http.https://github.com/.extraHeader"
            ].removeprefix("Authorization: Basic ")
            assert base64.b64decode(encoded).decode() == "x-access-token:typed-token"
            # Negotiate from the stale tip using the checkout's objects, without
            # letting Git run anything inside the checkout under this header.
            staging = Path(kwargs["cwd"])
            alternates = staging / "objects" / "info" / "alternates"
            assert alternates.read_text().strip() == str(workspace_objects)
            assert git(staging, "rev-parse", "refs/senpai/local") == base_sha
            assert configuration["core.alternateRefsCommand"] == "true"
            command = [*command]
            command[command.index(github_remote)] = str(remote)
            replacement = git_process_env(None)
            for name in tuple(replacement):
                if name.startswith("GIT_"):
                    replacement.pop(name)
            replacement.update(
                {
                    "GIT_ALLOW_PROTOCOL": "file",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": os.devnull,
                }
            )
            kwargs["env"] = replacement
        else:
            assert not any("typed-token" in value for value in env.values())
            assert not any("Authorization:" in value for value in env.values())
            if Path(kwargs["cwd"]) == workspace and "fetch" in command:
                hops.append("file")
                assert env["GIT_ALLOW_PROTOCOL"] == "file"
                assert any(
                    argument.endswith("/objects.git") for argument in command
                )
        return real_run(command, **kwargs)

    monkeypatch.setattr("senpai_agent.git_transport.subprocess.run", guarded_run)

    synced = sync_remote_branch(
        workspace,
        branch="experiment-7",
        authenticated_remote=github_remote,
        token=SecretStr("typed-token"),
    )

    assert synced.head_sha == merged
    assert hops == ["authenticated", "file"]
    assert has_object(workspace, merged)
