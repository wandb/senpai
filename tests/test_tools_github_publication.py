import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from git_workflow_support import commit_file, git, repository
from github_workflow_support import (
    FakeGitHub,
    assignment_record,
    comment,
    experiment_result,
    pull_request,
    workflow,
)
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.conversation.event_store import EventLog
from openhands.sdk.event import ActionEvent
from openhands.sdk.io import LocalFileStore
from pydantic import SecretStr

from senpai_agent.git_workflow import GitWorkflowPreconditionError
from senpai_agent.github.tools import (
    GitHubWorkflowToolSet,
    clear_github_credentials,
    configure_github_credentials,
)
from senpai_agent.github.workflow import (
    GitHubTransportError,
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker, render_result_comment


@pytest.fixture
def publication(tmp_path: Path):
    workspace, remote, _ = repository(tmp_path)
    base_sha = commit_file(
        workspace, "program.md", "Keep the scientific contract.\n", "program"
    )
    branch = "student-one/lower-lr"
    git(workspace, "branch", "-m", branch)
    git(workspace, "push", "origin", branch)
    local_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    assignment = assignment_record(base_sha=base_sha, head_sha=base_sha)

    class RepositoryGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            self.pr["head_sha"] = git(
                remote, "rev-parse", f"refs/heads/{self.pr['head_ref']}"
            )
            return super().request(method, url, headers=headers, json_body=json_body)

    fake = RepositoryGitHub(
        pull_request(
            head_sha=base_sha,
            body=render_assignment_marker(assignment),
            labels={"student:student-one", "status:wip", "status:hold", "keep"},
            draft=True,
        )
    )
    return SimpleNamespace(
        workspace=workspace,
        remote=remote,
        base_sha=base_sha,
        local_sha=local_sha,
        branch=branch,
        assignment=assignment,
        fake=fake,
        tmp_path=tmp_path,
    )


def publication_tool(case):
    configure_github_credentials(
        "acme/widgets", SecretStr("github-secret"), trusted_actor="senpai-bot"
    )
    try:
        tools = GitHubWorkflowToolSet.create(
            workflow=workflow(case.fake, role="student"),
            role="student",
            student_name="student-one",
            workspace=case.workspace,
            state_dir=case.tmp_path / "state",
        )
    finally:
        clear_github_credentials()
    return next(tool for tool in tools if tool.name == "push_experiment_commit")


def publication_action(tool, case):
    return tool.action_type.model_validate(
        {
            "assignment": {
                "pr_number": 7,
                "assignment_id": case.assignment.assignment_id,
                "revision_id": case.assignment.revision_id,
                "expected_pr_head_sha": case.base_sha,
            },
            "local_commit_sha": case.local_sha,
        }
    )


def test_push_experiment_commit_without_submitting_or_releasing_hold(publication):
    case = publication
    before = deepcopy(case.fake.pr)
    tool = publication_tool(case)
    action = publication_action(tool, case)

    first = tool(action)
    replay = tool(action)

    assert first.changed is True
    assert replay.changed is False
    assert first.state == replay.state == "experiment_commit_pushed"
    assert first.version == replay.version == case.local_sha
    assert git(case.remote, "rev-parse", f"refs/heads/{case.branch}") == case.local_sha
    assert case.fake.pr == {**before, "head_sha": case.local_sha}
    assert case.fake.comments == []
    assert case.fake.mutations == []


@pytest.mark.parametrize(
    ("guard", "message"),
    [
        ("student", "runtime's student"),
        ("revision", "stale turn"),
        ("closed", "open and unmerged"),
        ("review", "status:wip"),
        ("dirty", "worktree must be clean"),
        ("base", "does not contain assigned research base"),
        ("head", "pull request head SHA"),
        ("foreign-branch", "branch must belong to student"),
        ("base-branch", "branch must differ from"),
    ],
)
def test_push_experiment_commit_rejects_invalid_current_source_before_push(
    publication, guard, message
):
    case = publication
    tool = publication_tool(case)
    action = publication_action(tool, case)
    if guard == "student":
        case.fake.pr["body"] = render_assignment_marker(
            case.assignment.model_copy(update={"student": "student-two"})
        )
    elif guard == "revision":
        case.fake.pr["body"] = render_assignment_marker(
            case.assignment.model_copy(update={"revision_id": "revision-2"})
        )
    elif guard == "closed":
        case.fake.pr["state"] = "closed"
    elif guard == "review":
        case.fake.pr["labels"] = {"student:student-one", "status:review"}
    elif guard == "dirty":
        (case.workspace / "uncommitted.py").write_text("not published\n")
    elif guard == "base":
        unrelated = git(case.workspace, "commit-tree", "HEAD^{tree}", "-m", "unrelated")
        case.fake.pr["body"] = render_assignment_marker(
            case.assignment.model_copy(update={"base_sha": unrelated})
        )
    elif guard == "head":
        action = action.model_copy(
            update={
                "assignment": action.assignment.model_copy(
                    update={"expected_pr_head_sha": "c" * 40}
                )
            }
        )
    elif guard == "foreign-branch":
        case.branch = "student-two/candidate"
        git(case.workspace, "branch", "-m", case.branch)
        git(case.remote, "update-ref", f"refs/heads/{case.branch}", case.base_sha)
        case.fake.pr["head_ref"] = case.branch
        case.fake.pr["body"] = render_assignment_marker(
            case.assignment.model_copy(update={"head_ref": case.branch})
        )
    elif guard == "base-branch":
        case.fake.pr["base_ref"] = case.branch
        case.fake.pr["body"] = render_assignment_marker(
            case.assignment.model_copy(update={"base_ref": case.branch})
        )

    with pytest.raises(
        (WorkflowPreconditionError, GitWorkflowPreconditionError, ValueError),
        match=message,
    ):
        tool(action)

    assert git(case.remote, "rev-parse", f"refs/heads/{case.branch}") == case.base_sha
    assert case.fake.mutations == []


@pytest.mark.parametrize("revision", ["revision-1", "older-revision"])
def test_push_experiment_commit_respects_trusted_current_revision_terminal_result(
    publication, revision
):
    case = publication
    result = experiment_result(
        commit_sha=case.base_sha, expected_head_sha=case.base_sha
    )
    result = result.model_copy(
        update={
            "assignment": result.assignment.model_copy(update={"revision_id": revision})
        }
    )
    case.fake.comments.append(comment(91, render_result_comment(result)))
    tool = publication_tool(case)
    action = publication_action(tool, case)

    if revision == "revision-1":
        with pytest.raises(
            WorkflowPreconditionError, match="already has a final experiment result"
        ):
            tool(action)
        expected_head = case.base_sha
    else:
        assert tool(action).state == "experiment_commit_pushed"
        expected_head = case.local_sha

    assert git(case.remote, "rev-parse", f"refs/heads/{case.branch}") == expected_head
    assert len(case.fake.comments) == 1
    assert case.fake.mutations == []


@pytest.mark.parametrize("race", ["revision", "base", "terminal", "transport"])
def test_push_experiment_commit_reports_race_after_push_without_rewriting_workflow(
    publication, monkeypatch, race
):
    case = publication
    request = case.fake.request
    changed = False

    def racing_request(*args, **kwargs):
        nonlocal changed
        if (
            not changed
            and git(case.remote, "rev-parse", f"refs/heads/{case.branch}")
            == case.local_sha
        ):
            changed = True
            if race == "transport":
                raise GitHubTransportError(
                    "GET", "https://api.github.test/repos/acme/widgets/pulls/7"
                )
            if race == "terminal":
                result = experiment_result(
                    commit_sha=case.local_sha, expected_head_sha=case.local_sha
                )
                case.fake.comments.append(comment(91, render_result_comment(result)))
            else:
                update = (
                    {"revision_id": "revision-2"}
                    if race == "revision"
                    else {"base_sha": "b" * 40}
                )
                case.fake.pr["body"] = render_assignment_marker(
                    case.assignment.model_copy(update=update)
                )
        return request(*args, **kwargs)

    monkeypatch.setattr(case.fake, "request", racing_request)
    tool = publication_tool(case)
    conversation = SimpleNamespace(
        state=SimpleNamespace(execution_status=ConversationExecutionStatus.RUNNING)
    )
    with pytest.raises(
        (ValueError, ReconciliationError), match=f"{case.local_sha} was pushed"
    ):
        tool(publication_action(tool, case), conversation=conversation)

    assert git(case.remote, "rev-parse", f"refs/heads/{case.branch}") == case.local_sha
    assert case.fake.mutations == []
    assert case.fake.pr["labels"] == {
        "student:student-one",
        "status:wip",
        "status:hold",
        "keep",
    }
    if race == "revision":
        assert (
            conversation.state.execution_status == ConversationExecutionStatus.FINISHED
        )


def test_saved_push_action_loads_after_tool_rename(tmp_path):
    arguments = {
        "assignment": {
            "pr_number": 7,
            "assignment_id": "assignment-one",
            "revision_id": "revision-1",
            "expected_pr_head_sha": "a" * 40,
        },
        "local_commit_sha": "b" * 40,
    }
    saved = json.dumps({
        "kind": "ActionEvent",
        "id": "11111111-1111-4111-8111-111111111111",
        "timestamp": "2026-09-27T12:00:00",
        "source": "agent",
        "thought": [],
        "action": {"kind": "PublishAssignmentBranchAction", **arguments},
        "tool_name": "publish_assignment_branch",
        "tool_call_id": "call-old-push",
        "tool_call": {
            "id": "call-old-push",
            "name": "publish_assignment_branch",
            "arguments": json.dumps(arguments),
            "origin": "responses",
        },
        "llm_response_id": "resp-before-rename",
    })
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    event_file = events_dir / "event-00000-11111111-1111-4111-8111-111111111111.json"
    event_file.write_text(saved)

    history = EventLog(LocalFileStore(str(tmp_path)))
    event = history[0]

    assert isinstance(event, ActionEvent)
    assert event.action.assignment.assignment_id == "assignment-one"
    assert event.action.assignment.revision_id == "revision-1"
    assert event.action.local_commit_sha == "b" * 40
    assert event.tool_call.name == "publish_assignment_branch"
    assert event.tool_call.id == "call-old-push"
    assert event.llm_response_id == "resp-before-rename"
    assert event_file.read_text() == saved
