import json
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from github_workflow_support import (
    API_URL,
    ASSIGNMENT_ID,
    HEAD_SHA,
    REPO,
    FakeGitHub,
    assignment_record,
    pull_request,
    workflow,
)
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands_support import runtime_config
from pydantic import SecretStr

from senpai_agent.github import http as github_http
from senpai_agent.github.mailbox import GitHubMailbox
from senpai_agent.github.supervision import (
    RequestSupervisorTool,
    SupervisorEnvelope,
    SupervisorGateway,
    SupervisorRequest,
    render_request,
)
from senpai_agent.github.tools.contracts import AssignmentVersion
from senpai_agent.github.tools.runtime import GitHubToolRuntime
from senpai_agent.github.workflow import HttpResponse, WorkflowPreconditionError
from senpai_agent.models import render_assignment_marker

BRANCH = "schmidhuber"
STUDENT = "student-one"
TASK = "Inspect the stalled run and recommend the smallest corrective action."


def request(*, target=STUDENT, assignment=True, **changes):
    values = {
        "request_id": "inspect-stalled-run",
        "target": target,
        "assignment": AssignmentVersion(
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-1",
            expected_pr_head_sha=HEAD_SHA,
        )
        if assignment
        else None,
        "task": TASK,
        "context_prs": [7],
    }
    return SupervisorRequest(**(values | changes))


@pytest.fixture
def supervision_case(tmp_path, monkeypatch):
    fake = FakeGitHub(pull_request(labels={"status:wip", f"student:{STUDENT}"}))
    original_request = fake.request

    def transport(method, url, *, headers, json_body=None):
        path = urlsplit(url).path
        if method == "PATCH" and path == f"/repos/{REPO}/issues/7":
            fake.requests.append((method, path, json_body, dict(headers)))
            fake.issue.update(json_body)
            return HttpResponse(200, fake.issue)
        return original_request(method, url, headers=headers, json_body=json_body)

    monkeypatch.setattr(fake, "request", transport)
    monkeypatch.setattr(
        "senpai_agent.github.workflow.core.UrllibTransport", lambda: fake
    )

    def make_tool(*, role="advisor", student=STUDENT, branch=BRANCH):
        runtime = GitHubToolRuntime(
            workflow=workflow(fake, role=role),
            workspace=tmp_path,
            git_token=SecretStr("github-secret"),
            role=role,
            advisor_branch=branch,
            student_names=frozenset({STUDENT, "student-two"}),
            student_name=student if role == "student" else None,
        )
        return RequestSupervisorTool.create(runtime)[0]

    def urlopen(github_request, timeout):
        assert github_request.get_method() == "GET"
        assert github_request.headers["Authorization"] == "Bearer github-secret"
        path = urlsplit(github_request.full_url).path
        if path == f"/repos/{REPO}/pulls":
            payload = []
        elif path == f"/repos/{REPO}/issues":
            # Return candidates without applying the API's label filter, so the
            # consumer must validate the actual issue before dispatching it.
            payload = [fake.issue] if fake.issue is not None else []
        else:
            payload = fake.request(
                "GET",
                github_request.full_url,
                headers=github_request.headers,
            ).json_body
        response = BytesIO(json.dumps(payload).encode())
        response.headers = {}
        return response

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)

    def mailbox(*, role="student", student=STUDENT, branch=BRANCH):
        return GitHubMailbox(
            repo=REPO,
            token=SecretStr("github-secret"),
            role=role,
            advisor_branch=branch,
            student_name=student if role == "student" else None,
            api_url=API_URL,
            trusted_actor="senpai-bot",
            human_issues_enabled=False,
        )

    def gateway(*, role="advisor"):
        return SupervisorGateway(
            runtime_config(
                tmp_path,
                role=role,
                advisor_branch=BRANCH,
                student_name=STUDENT if role == "student" else None,
                github_repo=REPO,
                github_token=SecretStr("github-secret"),
                github_trusted_actor="senpai-bot",
            )
        )

    return SimpleNamespace(fake=fake, tool=make_tool, mailbox=mailbox, gateway=gateway)


@pytest.mark.parametrize(
    ("role", "target", "with_assignment"),
    [
        ("advisor", STUDENT, True),
        ("advisor", "advisor", False),
        ("student", STUDENT, True),
        ("student", "advisor", True),
    ],
)
def test_supervisor_request_replays_the_same_durable_issue(
    supervision_case,
    role,
    target,
    with_assignment,
):
    case = supervision_case
    action = request(target=target, assignment=with_assignment)
    tool = case.tool(role=role)

    first = tool(action)
    assert first.changed is True
    assert first.resource_url == f"https://github.com/{REPO}/issues/7"
    assert case.fake.issue is not None
    assert TASK in case.fake.issue["body"]
    assert "supervisor" in {label["name"] for label in case.fake.issue["labels"]}
    issue_body = case.fake.issue["body"]
    case.fake.issue["state"] = "closed"

    replay = tool(action)

    assert replay.changed is False
    assert replay.resource_url == first.resource_url
    assert case.fake.issue["body"] == issue_body
    assert case.fake.issue["state"] == "closed"
    assert [(method, path) for method, path, _ in case.fake.mutations] == [
        ("POST", f"/repos/{REPO}/issues"),
    ]

    with pytest.raises(WorkflowPreconditionError):
        tool(action.model_copy(update={"task": "Run a different investigation."}))
    assert len(case.fake.mutations) == 1


@pytest.mark.parametrize(
    "invalid",
    [
        "head",
        "revision",
        "branch",
        "student",
        "hold",
        "missing-assignment",
        "student-missing-assignment",
        "unconfigured-target",
        "student-targets-peer",
    ],
)
def test_supervisor_request_rejects_stale_or_unowned_work_before_posting(
    supervision_case,
    invalid,
):
    case = supervision_case
    action = request()
    role = "advisor"
    request_changes = None
    if invalid == "head":
        case.fake.pr["head_sha"] = "c" * 40
    elif invalid == "revision":
        case.fake.pr["body"] = render_assignment_marker(
            assignment_record(revision_id="revision-2"),
        )
    elif invalid == "branch":
        case.fake.pr["base_ref"] = "other-advisor"
        case.fake.pr["body"] = render_assignment_marker(
            assignment_record(base_ref="other-advisor"),
        )
    elif invalid == "student":
        case.fake.pr["body"] = render_assignment_marker(
            assignment_record(student="student-two"),
        )
    elif invalid == "hold":
        case.fake.pr["labels"].add("status:hold")
    elif invalid == "missing-assignment":
        request_changes = {"assignment": False}
    elif invalid == "student-missing-assignment":
        action = request(target="advisor", assignment=False)
        role = "student"
    elif invalid == "unconfigured-target":
        action = request(target="outside-this-launch")
    else:
        action = request(target="student-two")
        role = "student"

    with pytest.raises((WorkflowPreconditionError, PermissionError, ValueError)):
        if request_changes is not None:
            action = request(**request_changes)
        case.tool(role=role)(action)

    assert case.fake.mutations == []


@pytest.mark.parametrize("target", [STUDENT, "advisor"])
def test_supervisor_request_routes_only_to_its_exact_branch_and_recipient(
    supervision_case,
    target,
):
    case = supervision_case
    case.tool()(request(target=target))
    role = "advisor" if target == "advisor" else "student"
    mailbox = case.mailbox(role=role)

    events = mailbox.poll()

    assert len(events) == 1
    assert events[0].kind == "supervisor_requested"
    assert events[0].payload["number"] == 7
    assert events[0].payload["url"] == f"https://github.com/{REPO}/issues/7"
    assert TASK in events[0].to_prompt()
    assert mailbox.poll() == events
    assert case.mailbox(role=role, branch="other-advisor").poll() == ()
    assert case.mailbox(role="student", student="student-two").poll() == ()
    other_role = "student" if role == "advisor" else "advisor"
    assert case.mailbox(role=other_role).poll() == ()


@pytest.mark.parametrize(
    "invalid",
    ["author", "label", "malformed", "repo", "pull-request", "closed"],
)
def test_supervisor_mailbox_rejects_untrusted_or_invalid_request_issues(
    supervision_case,
    invalid,
):
    case = supervision_case
    case.tool()(request())
    issue = case.fake.issue
    if invalid == "author":
        issue["user"]["login"] = "untrusted-user"
        issue["author_association"] = "OWNER"
    elif invalid == "label":
        issue["labels"] = []
    elif invalid == "malformed":
        issue["body"] = "<!-- senpai-supervisor:v1 not-json -->"
    elif invalid == "repo":
        issue["body"] = render_request(
            SupervisorEnvelope(
                repo="other/repository",
                advisor_branch=BRANCH,
                requester="advisor",
                request=request(),
            )
        )
    elif invalid == "pull-request":
        issue["pull_request"] = {"url": f"https://api.github.com/repos/{REPO}/pulls/7"}
    else:
        issue["state"] = "closed"

    assert case.mailbox().poll() == ()


@pytest.mark.parametrize("resolved", [True, False])
def test_supervisor_completion_persists_feedback_for_the_requesting_conversation(
    supervision_case,
    resolved,
):
    case = supervision_case
    action = request(target="advisor")
    conversation_id = uuid4()
    conversation = SimpleNamespace(
        id=conversation_id,
        state=SimpleNamespace(secret_registry=SecretRegistry()),
    )
    case.tool(role="student")(action, conversation)
    gateway = case.gateway()
    summary = "Checked the failure. " + (
        "Repaired the configuration." if resolved else "Needs an operator decision."
    )

    gateway.complete(7, action, summary, resolved)

    assert case.fake.issue["state"] == "closed"
    assert summary in case.fake.issue["body"]
    requester = case.mailbox()
    events = requester.poll()
    assert len(events) == 1
    assert events[0].kind == "supervisor_completed"
    assert events[0].payload["result"] == {
        "resolved": resolved,
        "repair_summary": summary,
    }
    assert events[0].payload["parent_conversation_id"] == str(conversation_id)
    assert requester.poll() == events
    assert case.mailbox(role="advisor").poll() == ()
    assert case.mailbox(student="student-two").poll() == ()
    assert case.mailbox(branch="other-advisor").poll() == ()

    gateway.complete(7, action, summary, resolved)
    assert requester.poll() == events


@pytest.mark.parametrize("changed", ["request", "author"])
def test_supervisor_completion_rejects_an_issue_changed_during_the_repair(
    supervision_case,
    changed,
):
    case = supervision_case
    action = request(target="advisor")
    case.tool(role="student")(action)
    if changed == "request":
        case.fake.issue["body"] = render_request(
            SupervisorEnvelope(
                repo=REPO,
                advisor_branch=BRANCH,
                requester=STUDENT,
                request=action.model_copy(update={"task": "A different task."}),
            )
        )
    else:
        case.fake.issue["user"]["login"] = "untrusted-user"

    with pytest.raises(WorkflowPreconditionError):
        case.gateway().complete(7, action, "Resolved the original request.", True)

    assert case.fake.issue["state"] == "open"
    assert [(method, path) for method, path, _ in case.fake.mutations] == [
        ("POST", f"/repos/{REPO}/issues"),
    ]
