import json
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
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
    SupervisorResult,
    parse_request,
    render_request,
)
from senpai_agent.github.tools.contracts import AssignmentVersion
from senpai_agent.github.tools.runtime import GitHubToolRuntime
from senpai_agent.github.workflow import (
    GitHubTransportError,
    HttpResponse,
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker

BRANCH = "schmidhuber"
STUDENT = "student-one"
TASK = "Inspect the stalled run and recommend the smallest corrective action."


def request(*, target=STUDENT, assignment=True):
    return SupervisorRequest(
        request_id="inspect-stalled-run",
        target=target,
        assignment=AssignmentVersion(
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-1",
            expected_pr_head_sha=HEAD_SHA,
        )
        if assignment
        else None,
        task=TASK,
        context_prs=[7],
    )


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

    def make_tool(*, role="advisor"):
        runtime = GitHubToolRuntime(
            workflow=workflow(fake, role=role),
            workspace=tmp_path,
            git_token=SecretStr("github-secret"),
            role=role,
            advisor_branch=BRANCH,
            student_names=frozenset({STUDENT, "student-two"}),
            student_name=STUDENT if role == "student" else None,
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


@pytest.mark.parametrize(
    ("role", "target", "recipient", "resolved"),
    [
        ("student", "advisor", "remote", True),
        ("student", "advisor", "remote", False),
        ("advisor", "advisor", "same", True),
        ("student", STUDENT, "same", False),
        ("student", STUDENT, "other-conversation", True),
        ("advisor", "advisor", "legacy", True),
    ],
)
def test_supervisor_completion_persists_feedback_for_the_requesting_conversation(
    supervision_case,
    role,
    target,
    recipient,
    resolved,
):
    case = supervision_case
    action = request(target=target)
    conversation_id = uuid4()
    conversation = SimpleNamespace(
        id=conversation_id,
        state=SimpleNamespace(secret_registry=SecretRegistry()),
    )
    if recipient == "legacy":
        conversation = None
    tool = case.tool(role=role)
    tool(action, conversation)
    gateway = case.gateway(role="advisor" if target == "advisor" else "student")
    delivered_to = conversation_id if recipient == "same" else uuid4()
    summary = "Checked the failure. " + (
        "Repaired the configuration." if resolved else "Needs an operator decision."
    )

    gateway.complete(7, action, summary, resolved, delivered_to=delivered_to)

    local_receipt = recipient in {"same", "legacy"}
    assert case.fake.issue["state"] == ("closed" if local_receipt else "open")
    assert summary in case.fake.issue["body"]
    assert parse_request(case.fake.issue["body"]).result == SupervisorResult(
        resolved=resolved,
        repair_summary=summary,
    )
    assert tool(action, conversation).changed is False
    with pytest.raises(WorkflowPreconditionError, match="different result"):
        gateway.complete(
            7, action, "A different result", resolved, delivered_to=delivered_to
        )
    requester = case.mailbox(role=role)
    events = requester.poll()
    if local_receipt:
        assert events == ()
    else:
        assert len(events) == 1
        assert events[0].kind == "supervisor_completed"
        assert events[0].payload["result"] == {
            "resolved": resolved,
            "repair_summary": summary,
        }
        assert events[0].payload["parent_conversation_id"] == str(conversation_id)
    assert requester.poll() == events
    if recipient == "remote":
        assert case.mailbox(role="advisor").poll() == ()
    assert case.mailbox(student="student-two").poll() == ()
    assert case.mailbox(branch="other-advisor").poll() == ()

    gateway.complete(7, action, summary, resolved, delivered_to=delivered_to)
    assert requester.poll() == events
    keys = [event.dedupe_key for event in events]
    requester.acknowledge(keys)
    assert case.fake.issue["state"] == "closed"
    assert requester.poll() == ()
    mutations = list(case.fake.mutations)
    requester.acknowledge(keys)
    gateway.complete(7, action, summary, resolved, delivered_to=delivered_to)
    assert case.fake.mutations == mutations


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
        case.gateway().complete(
            7,
            action,
            "Resolved the original request.",
            True,
            delivered_to=uuid4(),
        )

    assert case.fake.issue["state"] == "open"
    assert [(method, path) for method, path, _ in case.fake.mutations] == [
        ("POST", f"/repos/{REPO}/issues"),
    ]


def test_supervisor_poll_retains_old_pending_requests_and_unread_results_across_pages(
    supervision_case,
    monkeypatch,
):
    case = supervision_case
    case.tool()(request())
    template = case.fake.issue
    original = parse_request(template["body"])

    def issue(number, request_id, *, target=STUDENT, result=None):
        candidate = deepcopy(template)
        candidate.update(
            number=number,
            html_url=f"https://github.com/{REPO}/issues/{number}",
            body=render_request(
                original.model_copy(
                    update={
                        "request": original.request.model_copy(
                            update={
                                "request_id": request_id,
                                "target": target,
                            }
                        ),
                        "result": result,
                    }
                )
            ),
        )
        return candidate

    recent = [issue(1000 + index, f"recent-{index}") for index in range(100)]
    pending = issue(7, "old-pending", target="advisor")
    unread = issue(
        8,
        "old-unread",
        result=SupervisorResult(
            resolved=True,
            repair_summary="Verified the earlier repair.",
        ),
    )
    pages = []

    def urlopen(github_request, timeout):
        parsed = urlsplit(github_request.full_url)
        links = {}
        if parsed.path == f"/repos/{REPO}/pulls":
            payload = []
        else:
            assert parsed.path == f"/repos/{REPO}/issues"
            query = parse_qs(parsed.query)
            assert query["state"] == ["open"]
            page = int(query.get("page", ["1"])[0])
            pages.append(page)
            payload = recent if page == 1 else [pending, unread]
            if page == 1:
                links["Link"] = f'<{github_request.full_url}&page=2>; rel="next"'
        response = BytesIO(json.dumps(payload).encode())
        response.headers = links
        return response

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)

    events = case.mailbox(role="advisor").poll()

    assert pages == [1, 2]
    assert [(event.kind, event.payload["number"]) for event in events] == [
        ("supervisor_requested", 7),
        ("supervisor_completed", 8),
    ]


@pytest.mark.parametrize("changed", ["requester", "result", "label"])
def test_supervisor_acknowledgement_never_closes_changed_or_other_requester_feedback(
    supervision_case,
    changed,
):
    case = supervision_case
    action = request(target="advisor")
    case.tool(role="student")(action)
    case.gateway().complete(7, action, "Repaired.", True, delivered_to=uuid4())
    mailbox = case.mailbox()
    keys = [event.dedupe_key for event in mailbox.poll()]
    mutations = list(case.fake.mutations)
    if changed == "requester":
        mailbox = case.mailbox(role="advisor")
    elif changed == "result":
        envelope = parse_request(case.fake.issue["body"])
        case.fake.issue["body"] = render_request(
            envelope.model_copy(
                update={
                    "result": SupervisorResult(
                        resolved=False, repair_summary="New feedback."
                    ),
                }
            )
        )
    else:
        case.fake.issue["labels"] = []

    mailbox.acknowledge(keys)

    assert case.fake.issue["state"] == "open"
    assert case.fake.mutations == mutations


def test_supervisor_acknowledgement_retries_an_unconfirmed_close(
    supervision_case,
    monkeypatch,
):
    case = supervision_case
    action = request(target="advisor")
    case.tool(role="student")(action)
    case.gateway().complete(7, action, "Repaired.", True, delivered_to=uuid4())
    mailbox = case.mailbox()
    keys = [event.dedupe_key for event in mailbox.poll()]
    original_request = case.fake.request

    def unavailable(method, url, **kwargs):
        if method == "PATCH":
            raise GitHubTransportError(method, url)
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(case.fake, "request", unavailable)
    with pytest.raises(ReconciliationError, match="did not acknowledge"):
        mailbox.acknowledge(keys)
    assert case.fake.issue["state"] == "open"

    monkeypatch.setattr(case.fake, "request", original_request)
    mailbox.acknowledge(keys)
    assert case.fake.issue["state"] == "closed"
