"""fetch_branch copies one repository branch into the student checkout, read-only."""

import subprocess
from pathlib import Path

import pytest
from git_workflow_support import git, repository
from pydantic import SecretStr
from test_git_workflow_sync import advance_branch_elsewhere, has_object
from test_tools_github_submission import RecordingWorkflow

from senpai_agent import git_workflow
from senpai_agent.git_transport import GitWorkflowPreconditionError
from senpai_agent.github.tools import (
    FetchBranchAction,
    FetchBranchTool,
    GitHubToolRuntime,
)
from senpai_agent.github.tools import fetch as fetch_module


def student_runtime(workspace: Path, token: SecretStr | None = None) -> GitHubToolRuntime:
    return GitHubToolRuntime(
        workflow=RecordingWorkflow(),
        workspace=workspace,
        git_token=token,
        role="student",
        advisor_branch="research",
        student_names=frozenset(),
        student_name="student-one",
    )


def test_fetch_branch_copies_a_peer_branch_without_touching_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace, remote, base_sha = repository(tmp_path)
    git(remote, "update-ref", "refs/heads/student-two/fold4", base_sha)
    peer_head = advance_branch_elsewhere(tmp_path, remote, "student-two/fold4")
    monkeypatch.setattr(fetch_module, "github_repository_url", lambda repo: str(remote))
    tool = FetchBranchTool.create(student_runtime(workspace))[0]
    assert not has_object(workspace, peer_head)

    fetched = tool(FetchBranchAction(branch="student-two/fold4"))

    assert fetched.head_sha == peer_head
    assert fetched.ref == "refs/remotes/origin/student-two/fold4"
    assert fetched.changed is True
    assert git(workspace, "rev-parse", fetched.ref) == peer_head
    assert has_object(workspace, peer_head)
    text = fetched.to_llm_content[0].text
    assert f"git log {fetched.ref}" in text
    assert peer_head in text
    # The checkout, its branch and its files are unchanged.
    assert git(workspace, "rev-parse", "HEAD") == base_sha
    assert git(workspace, "rev-parse", "--abbrev-ref", "HEAD") == "experiment-7"
    assert git(workspace, "status", "--porcelain") == ""

    repeated = tool(FetchBranchAction(branch="student-two/fold4"))
    assert repeated.changed is False
    assert "nothing had changed" in repeated.to_llm_content[0].text


def test_fetch_branch_names_a_missing_branch_and_where_names_come_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace, remote, _base_sha = repository(tmp_path)
    monkeypatch.setattr(fetch_module, "github_repository_url", lambda repo: str(remote))
    tool = FetchBranchTool.create(student_runtime(workspace))[0]

    with pytest.raises(GitWorkflowPreconditionError, match="'nope'.*acme/widgets.*get_prs"):
        tool(FetchBranchAction(branch="nope"))
    assert git(workspace, "for-each-ref", "refs/remotes/origin/nope") == ""


def test_fetch_branch_rejects_refs_and_urls_as_branch_names(tmp_path: Path):
    workspace, _remote, _base_sha = repository(tmp_path)
    tool = FetchBranchTool.create(student_runtime(workspace))[0]

    for name in ("refs/heads/main", "https://github.com/acme/widgets", "-x", "a..b"):
        with pytest.raises(GitWorkflowPreconditionError):
            tool(FetchBranchAction(branch=name))


def test_fetch_branch_runs_the_credentialed_hop_outside_the_checkout_from_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace, remote, base_sha = repository(tmp_path)
    git(remote, "update-ref", "refs/heads/student-two/fold4", base_sha)
    peer_head = advance_branch_elsewhere(tmp_path, remote, "student-two/fold4")
    github_remote = "https://github.com/acme/widgets.git"
    monkeypatch.setattr(fetch_module, "github_repository_url", lambda repo: github_remote)
    real_run = subprocess.run
    hops: list[str] = []

    def guarded_run(command, **kwargs):
        if "env" not in kwargs:  # this test's own assertions, not code under test
            return real_run(command, **kwargs)
        assert "typed-token" not in command
        if github_remote in command:
            hops.append("authenticated")
            staging = Path(kwargs["cwd"])
            assert staging.name == "objects.git"
            # A branch never fetched before negotiates from the checkout HEAD,
            # with the checkout's objects borrowed, so only new history crosses.
            alternates = staging / "objects" / "info" / "alternates"
            assert alternates.read_text().strip() == str(
                (workspace / ".git" / "objects").resolve()
            )
            assert git(staging, "rev-parse", "refs/senpai/local") == base_sha
            command = [*command]
            command[command.index(github_remote)] = str(remote)
            kwargs["env"] = {
                **{k: v for k, v in kwargs["env"].items() if not k.startswith("GIT_")},
                "GIT_ALLOW_PROTOCOL": "file",
            }
        else:
            assert not any("typed-token" in value for value in kwargs["env"].values())
            if Path(kwargs["cwd"]) == workspace and "fetch" in command:
                hops.append("file")
        return real_run(command, **kwargs)

    monkeypatch.setattr("senpai_agent.git_transport.subprocess.run", guarded_run)
    tool = FetchBranchTool.create(student_runtime(workspace, SecretStr("typed-token")))[0]

    fetched = tool(FetchBranchAction(branch="student-two/fold4"))

    assert fetched.head_sha == peer_head
    assert hops == ["authenticated", "file"]
    assert has_object(workspace, peer_head)
    git_dir = workspace / ".git"
    assert not any(
        "typed-token" in path.read_text(errors="ignore")
        for path in git_dir.rglob("*")
        if path.is_file() and "objects" not in path.parts
    )
    assert "senpai sync student-two/fold4" in git(
        workspace, "reflog", "show", "refs/remotes/origin/student-two/fold4"
    )


def test_fetch_branch_tells_a_transient_failure_from_a_bad_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace, _remote, _base_sha = repository(tmp_path)
    tool = FetchBranchTool.create(student_runtime(workspace))[0]

    def time_out(*args, **kwargs):
        raise GitWorkflowPreconditionError("git fetch --no-tags timed out after 300 s")

    monkeypatch.setattr(git_workflow, "sync_remote_branch", time_out)
    with pytest.raises(GitWorkflowPreconditionError) as error:
        tool(FetchBranchAction(branch="student-two/fold4"))
    assert "Retry once" in str(error.value)
    assert "get_prs" not in str(error.value)
