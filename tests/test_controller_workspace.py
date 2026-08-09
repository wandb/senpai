import base64
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

from senpai_agent.git_workflow import git_process_env
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.workspace import (
    StudentWorkspaceReconciler,
    WorkspaceDivergence,
    WorkspaceJobRunning,
)


def git(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def assigned_workspace(tmp_path: Path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    workspace = tmp_path / "student"
    git("init", "--bare", str(remote))
    git("init", str(seed))
    git("config", "user.name", "test", cwd=seed)
    git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "program.py").write_text("baseline\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "baseline", cwd=seed)
    git("branch", "-M", "research", cwd=seed)
    git("remote", "add", "origin", str(remote), cwd=seed)
    git("push", "origin", "research", cwd=seed)
    base_sha = git("rev-parse", "HEAD", cwd=seed)
    git("switch", "-c", "student/candidate", cwd=seed)
    git("push", "origin", "student/candidate", cwd=seed)
    assigned_head = git("rev-parse", "HEAD", cwd=seed)
    git(
        "clone",
        "--branch",
        "student/candidate",
        str(remote),
        str(workspace),
    )
    git("config", "user.name", "student", cwd=workspace)
    git("config", "user.email", "student@example.com", cwd=workspace)
    return remote, seed, workspace, base_sha, assigned_head


def assignment_event(head_sha: str, base_sha: str):
    return ControllerEvent(
        kind="student_assignment",
        dedupe_key="assignment:restart",
        payload={
            "head_ref": "student/candidate",
            "head_sha": head_sha,
            "base_ref": "research",
            "base_sha": base_sha,
        },
    )


def test_reconciliation_preserves_unpushed_commits_and_dirty_files(
    tmp_path: Path,
):
    _remote, _seed, workspace, base_sha, assigned_head = assigned_workspace(tmp_path)
    (workspace / "program.py").write_text("candidate\n")
    git("commit", "-am", "candidate", cwd=workspace)
    local_head = git("rev-parse", "HEAD", cwd=workspace)
    (workspace / "notes.txt").write_text("dirty but recoverable\n")

    StudentWorkspaceReconciler(workspace)((assignment_event(assigned_head, base_sha),))

    assert git("rev-parse", "HEAD", cwd=workspace) == local_head
    assert (workspace / "notes.txt").read_text() == "dirty but recoverable\n"


def test_reconciliation_does_not_touch_checkout_while_mutable_job_is_active(
    tmp_path: Path,
):
    _remote, _seed, workspace, base_sha, assigned_head = assigned_workspace(tmp_path)
    original_head = git("rev-parse", "HEAD", cwd=workspace)
    reconciler = StudentWorkspaceReconciler(
        workspace,
        active_mutable_job_ids=lambda: ("job-17",),
    )

    with pytest.raises(WorkspaceJobRunning, match="job-17"):
        reconciler((assignment_event(assigned_head, base_sha),))

    assert git("rev-parse", "HEAD", cwd=workspace) == original_head


def test_reconciliation_rejects_a_remote_head_newer_than_the_assignment(
    tmp_path: Path,
):
    _remote, seed, workspace, base_sha, assigned_head = assigned_workspace(tmp_path)
    (seed / "program.py").write_text("moved after assignment\n")
    git("commit", "-am", "move assignment branch", cwd=seed)
    git("push", "origin", "student/candidate", cwd=seed)

    with pytest.raises(RuntimeError, match="assignment head moved"):
        StudentWorkspaceReconciler(workspace)(
            (assignment_event(assigned_head, base_sha),)
        )

    assert git("rev-parse", "HEAD", cwd=workspace) == assigned_head


def test_reconciliation_surfaces_and_preserves_a_diverged_active_branch(
    tmp_path: Path,
):
    _remote, seed, workspace, base_sha, _assigned_head = assigned_workspace(tmp_path)
    (workspace / "program.py").write_text("rebased experiment\n")
    git("commit", "-am", "local experiment", cwd=workspace)
    local_head = git("rev-parse", "HEAD", cwd=workspace)
    (workspace / "notes.txt").write_text("dirty measurements\n")

    git("checkout", "--orphan", "replacement", cwd=seed)
    git("rm", "-f", "program.py", cwd=seed)
    (seed / "program.py").write_text("new advisor base\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "replace assignment base", cwd=seed)
    git("push", "--force", "origin", "HEAD:student/candidate", cwd=seed)
    expected_head = git("rev-parse", "HEAD", cwd=seed)

    with pytest.raises(WorkspaceDivergence) as raised:
        StudentWorkspaceReconciler(workspace)(
            (assignment_event(expected_head, base_sha),)
        )

    assert raised.value.event.kind == "workspace_diverged"
    assert raised.value.event.payload["preserved_local_head"] == local_head
    assert git("rev-parse", "HEAD", cwd=workspace) == local_head
    assert (workspace / "notes.txt").read_text() == "dirty measurements\n"


def test_reconciliation_surfaces_divergence_when_assignment_is_not_checked_out(
    tmp_path: Path,
):
    _remote, seed, workspace, base_sha, _assigned_head = assigned_workspace(tmp_path)
    git("checkout", "-b", "other-work", cwd=workspace)
    git("checkout", "--orphan", "replacement", cwd=seed)
    git("rm", "-f", "program.py", cwd=seed)
    (seed / "program.py").write_text("new advisor base\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "replace assignment base", cwd=seed)
    git("push", "--force", "origin", "HEAD:student/candidate", cwd=seed)
    expected_head = git("rev-parse", "HEAD", cwd=seed)

    with pytest.raises(WorkspaceDivergence) as raised:
        StudentWorkspaceReconciler(workspace)(
            (assignment_event(expected_head, base_sha),)
        )

    assert raised.value.event.payload["current_branch"] == "other-work"
    assert git("branch", "--show-current", cwd=workspace) == "other-work"


@pytest.mark.parametrize("assignment_branch_exists", [True, False])
def test_reconciliation_never_carries_dirty_work_across_branches(
    tmp_path: Path,
    assignment_branch_exists: bool,
):
    _remote, _seed, workspace, base_sha, assigned_head = assigned_workspace(tmp_path)
    git("switch", "-c", "other-work", cwd=workspace)
    if not assignment_branch_exists:
        git("branch", "-D", "student/candidate", cwd=workspace)
    (workspace / "program.py").write_text("unrelated dirty work\n")
    before = git("rev-parse", "HEAD", cwd=workspace)

    with pytest.raises(WorkspaceDivergence) as raised:
        StudentWorkspaceReconciler(workspace)(
            (assignment_event(assigned_head, base_sha),)
        )

    assert raised.value.event.payload["current_branch"] == "other-work"
    assert git("branch", "--show-current", cwd=workspace) == "other-work"
    assert git("rev-parse", "HEAD", cwd=workspace) == before
    assert (workspace / "program.py").read_text() == "unrelated dirty work\n"


def test_reconciliation_hydrates_exact_controller_owned_head_and_base_refs(
    tmp_path: Path,
):
    _remote, _seed, workspace, base_sha, assigned_head = assigned_workspace(tmp_path)

    StudentWorkspaceReconciler(workspace)((assignment_event(assigned_head, base_sha),))

    assert (
        git(
            "rev-parse",
            "refs/senpai/assignment/head",
            cwd=workspace,
        )
        == assigned_head
    )
    assert (
        git(
            "rev-parse",
            "refs/senpai/assignment/base",
            cwd=workspace,
        )
        == base_sha
    )


def test_reconciliation_rejects_an_unavailable_exact_base_without_mutation(
    tmp_path: Path,
):
    _remote, _seed, workspace, _base_sha, assigned_head = assigned_workspace(tmp_path)
    before = git("rev-parse", "HEAD", cwd=workspace)
    (workspace / "notes.txt").write_text("preserve me\n")

    with pytest.raises(RuntimeError, match="assignment base.*unavailable"):
        StudentWorkspaceReconciler(workspace)(
            (assignment_event(assigned_head, "f" * 40),)
        )

    assert git("rev-parse", "HEAD", cwd=workspace) == before
    assert (workspace / "notes.txt").read_text() == "preserve me\n"


def test_divergence_identity_changes_when_dirty_work_changes(tmp_path: Path):
    _remote, seed, workspace, base_sha, _assigned_head = assigned_workspace(tmp_path)
    (workspace / "program.py").write_text("local experiment\n")
    git("commit", "-am", "local experiment", cwd=workspace)
    git("checkout", "--orphan", "replacement", cwd=seed)
    git("rm", "-f", "program.py", cwd=seed)
    (seed / "program.py").write_text("replacement\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "replace branch", cwd=seed)
    git("push", "--force", "origin", "HEAD:student/candidate", cwd=seed)
    expected_head = git("rev-parse", "HEAD", cwd=seed)
    reconciler = StudentWorkspaceReconciler(workspace)
    (workspace / "program.py").write_text("first dirty version\n")

    with pytest.raises(WorkspaceDivergence) as first:
        reconciler((assignment_event(expected_head, base_sha),))
    (workspace / "program.py").write_text("second dirty version\n")
    with pytest.raises(WorkspaceDivergence) as second:
        reconciler((assignment_event(expected_head, base_sha),))

    assert first.value.event.dedupe_key != second.value.event.dedupe_key


def test_divergence_identity_changes_when_untracked_content_changes(tmp_path: Path):
    _remote, seed, workspace, base_sha, _assigned_head = assigned_workspace(tmp_path)
    (workspace / "program.py").write_text("local experiment\n")
    git("commit", "-am", "local experiment", cwd=workspace)
    git("checkout", "--orphan", "replacement", cwd=seed)
    git("rm", "-f", "program.py", cwd=seed)
    (seed / "program.py").write_text("replacement\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "replace branch", cwd=seed)
    git("push", "--force", "origin", "HEAD:student/candidate", cwd=seed)
    expected_head = git("rev-parse", "HEAD", cwd=seed)
    reconciler = StudentWorkspaceReconciler(workspace)
    note = workspace / "notes.txt"
    note.write_text("first untracked version\n")

    with pytest.raises(WorkspaceDivergence) as first:
        reconciler((assignment_event(expected_head, base_sha),))
    note.write_text("second untracked version\n")
    with pytest.raises(WorkspaceDivergence) as second:
        reconciler((assignment_event(expected_head, base_sha),))

    assert first.value.event.dedupe_key != second.value.event.dedupe_key


def test_reconciliation_isolates_typed_auth_from_hostile_git_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    remote, _seed, workspace, base_sha, assigned_head = assigned_workspace(tmp_path)
    real_run = subprocess.run
    git(
        "remote",
        "set-url",
        "origin",
        "https://attacker.invalid/redirect.git",
        cwd=workspace,
    )
    git("config", "http.proxy", "http://attacker.invalid:8080", cwd=workspace)
    git("config", "http.sslVerify", "false", cwd=workspace)
    git("config", "http.followRedirects", "true", cwd=workspace)
    git("config", "credential.helper", "!false", cwd=workspace)
    git(
        "config",
        "url.https://attacker.invalid/rewrite.git.insteadOf",
        "https://github.com/acme/widgets.git",
        cwd=workspace,
    )
    global_config = tmp_path / "hostile-global.gitconfig"
    global_config.write_text(
        '[url "https://attacker.invalid/global.git"]\n'
        "\tinsteadOf = https://github.com/acme/widgets.git\n"
        "[http]\n\tsslVerify = false\n"
    )
    system_config = tmp_path / "hostile-system.gitconfig"
    system_config.write_text("[http]\n\tproxy = http://attacker.invalid:8080\n")
    github_remote = "https://github.com/acme/widgets.git"

    def guarded_run(command, **kwargs):
        env = kwargs["env"]
        assert "typed-token" not in command
        assert "ambient-token" not in env.values()
        assert "GITHUB_TOKEN" not in env
        if github_remote in command:
            assert Path(kwargs["cwd"]) != workspace
            assert Path(kwargs["cwd"]).name == "objects.git"
            assert env["GIT_ALLOW_PROTOCOL"] == "https"
            assert env["GIT_CONFIG_GLOBAL"] == os.devnull
            assert env["GIT_CONFIG_SYSTEM"] == os.devnull
            assert env["GIT_CONFIG_NOSYSTEM"] == "1"
            assert "HTTPS_PROXY" not in env
            assert "GIT_SSL_NO_VERIFY" not in env
            configuration = {
                env[f"GIT_CONFIG_KEY_{index}"]: env[f"GIT_CONFIG_VALUE_{index}"]
                for index in range(int(env["GIT_CONFIG_COUNT"]))
            }
            assert configuration["credential.helper"] == ""
            assert configuration["http.proxy"] == ""
            assert configuration["http.https://github.com/.proxy"] == ""
            assert configuration["http.sslVerify"] == "true"
            assert configuration["http.https://github.com/.sslVerify"] == "true"
            assert configuration["http.followRedirects"] == "false"
            assert configuration["http.https://github.com/.followRedirects"] == "false"
            assert configuration["http.extraHeader"] == ""
            encoded = configuration[
                "http.https://github.com/.extraHeader"
            ].removeprefix("Authorization: Basic ")
            assert base64.b64decode(encoded).decode() == ("x-access-token:typed-token")
            command = [*command]
            command[command.index(github_remote)] = str(remote)
            replacement_environment = git_process_env(None)
            for name in tuple(replacement_environment):
                if name.startswith("GIT_"):
                    replacement_environment.pop(name)
            replacement_environment.update(
                {
                    "GIT_ALLOW_PROTOCOL": "file",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": os.devnull,
                }
            )
            kwargs["env"] = replacement_environment
        elif command[1] == "fetch":
            assert Path(kwargs["cwd"]) == workspace
            assert env["GIT_ALLOW_PROTOCOL"] == "file"
            assert any(argument.endswith("/objects.git") for argument in command)
            assert not any("Authorization:" in value for value in env.values())
        else:
            assert not any("typed-token" in value for value in env.values())
        return real_run(command, **kwargs)

    monkeypatch.setattr("senpai_agent.workspace.subprocess.run", guarded_run)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-token")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system_config))
    monkeypatch.setenv("GIT_SSL_NO_VERIFY", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid:8080")

    StudentWorkspaceReconciler(
        workspace,
        repo="acme/widgets",
        token=SecretStr("typed-token"),
    )((assignment_event(assigned_head, base_sha),))
