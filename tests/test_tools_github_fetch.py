"""fetch_branch copies one repository branch into the student checkout, read-only."""

from pathlib import Path

import pytest
from git_workflow_support import git, repository
from test_git_workflow_sync import advance_branch_elsewhere, has_object
from test_tools_github_submission import RecordingWorkflow

from senpai_agent.git_transport import GitWorkflowPreconditionError
from senpai_agent.github.tools import (
    FetchBranchAction,
    FetchBranchTool,
    GitHubToolRuntime,
)
from senpai_agent.github.tools import fetch as fetch_module


def student_runtime(workspace: Path) -> GitHubToolRuntime:
    return GitHubToolRuntime(
        workflow=RecordingWorkflow(),
        workspace=workspace,
        git_token=None,
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
