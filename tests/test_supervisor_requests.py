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
from test_controller import Turns, controller

from senpai_agent.github import http as github_http
from senpai_agent.github.mailbox import GitHubMailbox
from senpai_agent.github.supervision import (
    SupervisorEnvelope,
    SupervisorGateway,
    SupervisorRequest,
    SupervisorResult,
    parse_request,
    render_request,
)
from senpai_agent.github.tools.contracts import AssignmentVersion
from senpai_agent.github.tools.definitions import RequestSupervisorTool
from senpai_agent.github.tools.runtime import GitHubToolRuntime
from senpai_agent.github.workflow import (
    GitHubTransportError,
    HttpResponse,
    WorkflowPreconditionError,
)
from senpai_agent.inbox import PersistentInbox
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import CompositeMailbox, LocalMailbox, SupervisedMailbox
from senpai_agent.models import render_assignment_marker
from senpai_agent.openhands_runner import local_event_db_path
from senpai_agent.state import (
    AssignmentConversationRegistry,
    StudentConversationSelector,
)
from senpai_agent.supervision import SupervisorHandler

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


def assert_reply(payload, envelope):
    copied = deepcopy(payload)
    assert copied.pop("number") == 7
    assert copied.pop("url") == f"https://github.com/{REPO}/issues/7"
    assert SupervisorEnvelope.model_validate(copied) == envelope


@pytest.fixture
def supervision_case(tmp_path, monkeypatch):
    fake = FakeGitHub(pull_request(labels={"status:wip", f"student:{STUDENT}"}))
    conversation = SimpleNamespace(
        id=uuid4(), state=SimpleNamespace(secret_registry=SecretRegistry())
    )
    original_request = fake.request

    def transport(method, url, *, headers, json_body=None):
        path = urlsplit(url).path
        if method == "GET" and path == f"/repos/{REPO}/issues":
            fake.requests.append((method, path, json_body, dict(headers)))
            return HttpResponse(200, [fake.issue] if fake.issue is not None else [])
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

    def config(role, *, student=STUDENT, branch=BRANCH):
        return runtime_config(
            tmp_path,
            role=role,
            advisor_branch=branch,
            student_name=student if role == "student" else None,
            github_repo=REPO,
            github_token=SecretStr("github-secret"),
            github_trusted_actor="senpai-bot",
        )

    def gateway(*, role="advisor", student=STUDENT, branch=BRANCH):
        return SupervisorGateway(config(role, student=student, branch=branch))

    def recipient(*, role="student"):
        runtime = config(role)
        inbox = PersistentInbox(runtime.state_dir / f"{role}-inbox.sqlite3")
        handler = SupervisorHandler(runtime, inbox)
        local = LocalMailbox(local_event_db_path(runtime))
        return SimpleNamespace(
            handler=handler,
            local=local,
            inbox=inbox,
            config=runtime,
            mailbox=SupervisedMailbox(
                CompositeMailbox(mailbox(role=role), local), handler
            ),
        )

    return SimpleNamespace(
        fake=fake,
        tool=make_tool,
        mailbox=mailbox,
        gateway=gateway,
        conversation=conversation,
        recipient=recipient,
    )


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
    action = request(target=target, assignment=with_assignment).model_copy(
        update={"task": TASK + " The diagnostic contains <!-- a note -->."}
    )
    tool = case.tool(role=role)

    first = tool(action, case.conversation)
    assert first.changed is True
    assert first.resource_url == f"https://github.com/{REPO}/issues/7"
    assert case.fake.issue is not None
    assert TASK in case.fake.issue["body"]
    assert "supervisor" in {label["name"] for label in case.fake.issue["labels"]}
    issue_body = case.fake.issue["body"]
    assert issue_body.splitlines()[0].count("-->") == 1
    assert parse_request(issue_body).request == action
    case.fake.issue["state"] = "closed"

    replay = tool(action, case.conversation)

    assert replay.changed is False
    assert replay.resource_url == first.resource_url
    assert case.fake.issue["body"] == issue_body
    assert case.fake.issue["state"] == "closed"
    assert [(method, path) for method, path, _ in case.fake.mutations] == [
        ("POST", f"/repos/{REPO}/issues"),
    ]

    with pytest.raises(WorkflowPreconditionError):
        tool(
            action.model_copy(update={"task": "Run a different investigation."}),
            case.conversation,
        )
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
        "missing-conversation",
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
    elif invalid == "student-targets-peer":
        action = request(target="student-two")
        role = "student"

    with pytest.raises(
        RuntimeError
        if invalid == "missing-conversation"
        else (WorkflowPreconditionError, PermissionError, ValueError),
        match="require a conversation" if invalid == "missing-conversation" else None,
    ):
        if request_changes is not None:
            action = request(**request_changes)
        case.tool(role=role)(
            action, None if invalid == "missing-conversation" else case.conversation
        )

    assert case.fake.mutations == []


@pytest.mark.parametrize("target", [STUDENT, "advisor"])
def test_supervisor_request_routes_only_to_its_exact_branch_and_recipient(
    supervision_case,
    target,
):
    case = supervision_case
    case.tool()(request(target=target), case.conversation)
    role = "advisor" if target == "advisor" else "student"
    gateway = case.gateway(role=role)

    pending = tuple(gateway.pending())

    assert len(pending) == 1
    issue, envelope = pending[0]
    assert issue["number"] == 7
    assert issue["html_url"] == f"https://github.com/{REPO}/issues/7"
    assert envelope.request == request(target=target)
    assert tuple(gateway.pending()) == pending
    assert tuple(case.gateway(role=role, branch="other-advisor").pending()) == ()
    assert tuple(case.gateway(role="student", student="student-two").pending()) == ()
    other_role = "student" if role == "advisor" else "advisor"
    assert tuple(case.gateway(role=other_role).pending()) == ()


@pytest.mark.parametrize(
    "invalid",
    ["author", "label", "malformed", "repo", "pull-request", "closed"],
)
def test_supervisor_gateway_rejects_untrusted_or_invalid_request_issues(
    supervision_case,
    invalid,
):
    case = supervision_case
    case.tool()(request(), case.conversation)
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
                parent_conversation_id=case.conversation.id,
            )
        )
    elif invalid == "pull-request":
        issue["pull_request"] = {"url": f"https://api.github.com/repos/{REPO}/pulls/7"}
    else:
        issue["state"] = "closed"

    assert tuple(case.gateway(role="student").pending()) == ()


@pytest.mark.parametrize(
    ("role", "target", "recipient", "resolved"),
    [
        ("student", "advisor", "remote", True),
        ("student", "advisor", "remote", False),
        ("advisor", "advisor", "same", True),
        ("student", STUDENT, "same", False),
        ("student", STUDENT, "other-conversation", True),
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
    tool = case.tool(role=role)
    tool(action, conversation)
    original = parse_request(case.fake.issue["body"])
    gateway = case.gateway(role="advisor" if target == "advisor" else "student")
    delivered_to = conversation_id if recipient == "same" else uuid4()
    summary = "Checked the failure. " + (
        "Repaired the configuration." if resolved else "Needs an operator decision."
    )

    gateway.complete(7, original, summary, resolved, delivered_to=delivered_to)

    local_receipt = recipient == "same"
    assert case.fake.issue["state"] == ("closed" if local_receipt else "open")
    assert summary in case.fake.issue["body"]
    assert parse_request(case.fake.issue["body"]).result == SupervisorResult(
        resolved=resolved,
        repair_summary=summary,
    )
    assert tool(action, conversation).changed is False
    with pytest.raises(WorkflowPreconditionError, match="different result"):
        gateway.complete(
            7, original, "A different result", resolved, delivered_to=delivered_to
        )
    requester = case.gateway(role=role)
    pending = tuple(requester.pending())
    if local_receipt:
        assert pending == ()
    else:
        assert len(pending) == 1
        assert pending[0][1].result == SupervisorResult(
            resolved=resolved, repair_summary=summary
        )
        assert pending[0][1].parent_conversation_id == conversation_id
    assert tuple(requester.pending()) == pending
    if recipient == "remote":
        assert tuple(case.gateway(role="advisor").pending()) == ()
    assert tuple(case.gateway(role="student", student="student-two").pending()) == ()
    assert tuple(case.gateway(branch="other-advisor").pending()) == ()

    gateway.complete(7, original, summary, resolved, delivered_to=delivered_to)
    assert tuple(requester.pending()) == pending
    delivery = case.recipient(role=role)
    receipts = delivery.mailbox.poll()
    assert [event.kind for event in receipts] == (
        [] if local_receipt else ["supervisor_completed"]
    )
    keys = [event.dedupe_key for event in receipts]
    for event, (_, envelope) in zip(receipts, pending, strict=True):
        assert_reply(event.payload, envelope)
    assert case.fake.issue["state"] == "closed"
    assert tuple(requester.pending()) == ()
    mutations = list(case.fake.mutations)
    delivery.mailbox.acknowledge(keys)
    delivery.handler()
    delivery.mailbox.acknowledge(keys)
    assert delivery.mailbox.poll() == ()
    gateway.complete(7, original, summary, resolved, delivered_to=delivered_to)
    assert case.fake.mutations == mutations


@pytest.mark.parametrize(
    "changed", ["request", "author", "requester", "parent", "repo", "branch"]
)
def test_supervisor_completion_rejects_an_issue_changed_during_the_repair(
    supervision_case,
    changed,
):
    case = supervision_case
    action = request(target="advisor")
    case.tool(role="student")(action, case.conversation)
    original = parse_request(case.fake.issue["body"])
    if changed == "author":
        case.fake.issue["user"]["login"] = "untrusted-user"
    else:
        field, value = {
            "request": (
                "request",
                action.model_copy(update={"task": "A different task."}),
            ),
            "requester": ("requester", "advisor"),
            "parent": ("parent_conversation_id", uuid4()),
            "repo": ("repo", "other/repository"),
            "branch": ("advisor_branch", "other-advisor"),
        }[changed]
        case.fake.issue["body"] = render_request(
            original.model_copy(update={field: value})
        )

    with pytest.raises(WorkflowPreconditionError):
        case.gateway().complete(
            7,
            original,
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
    case.tool()(request(), case.conversation)
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
    recent[0] = issue(
        1000,
        "recent-reply",
        result=SupervisorResult(
            resolved=True, repair_summary="Verified recent repair."
        ),
    )
    pending_issue = issue(7, "old-pending", target="advisor")
    unread = issue(
        8,
        "old-unread",
        result=SupervisorResult(
            resolved=True,
            repair_summary="Verified the earlier repair.",
        ),
    )
    candidates = [*recent, pending_issue, unread]
    by_number = {item["number"]: item for item in candidates}
    pages = []
    original_request = case.fake.request

    def paginated(method, url, **kwargs):
        parsed = urlsplit(url)
        if method == "GET" and parsed.path == f"/repos/{REPO}/issues":
            query = parse_qs(parsed.query)
            assert query["state"] == ["open"]
            page = int(query.get("page", ["1"])[0])
            pages.append(page)
            opened = [item for item in candidates if item["state"] == "open"]
            payload = opened[(page - 1) * 100 : page * 100]
            headers = (("Link", f'<{url}&page=2>; rel="next"'),) if page == 1 else ()
            return HttpResponse(200, payload, headers=headers)
        if parsed.path.startswith(f"/repos/{REPO}/issues/"):
            item = by_number[int(parsed.path.rsplit("/", 1)[-1])]
            if method == "PATCH":
                item.update(kwargs["json_body"])
            else:
                assert method == "GET"
            return HttpResponse(200, item)
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(case.fake, "request", paginated)
    repairs = []

    def repair(_self, _gateway, envelope, *_args):
        repairs.append(envelope.request.request_id)
        return SupervisorResult(resolved=False, repair_summary="Needs operator help.")

    monkeypatch.setattr(SupervisorHandler, "_repair", repair)
    delivery = case.recipient(role="advisor")
    receipts = delivery.mailbox.poll()

    assert pages == [1, 2]
    assert repairs == ["old-pending"]
    assert by_number[1000]["state"] == by_number[8]["state"] == "closed"
    assert parse_request(by_number[7]["body"]).result is not None
    assert len(receipts) == 3
    replies = [event.payload for event in receipts if "number" in event.payload]
    assert {reply["request"]["request_id"] for reply in replies} == {
        "recent-reply",
        "old-unread",
    }


@pytest.mark.parametrize(
    "changed", ["requester", "parent", "request", "result", "label", "author"]
)
def test_supervisor_reply_handoff_never_closes_changed_feedback(
    supervision_case,
    monkeypatch,
    changed,
):
    case = supervision_case
    action = request(target="advisor")
    case.tool(role="student")(action, case.conversation)
    original = parse_request(case.fake.issue["body"])
    case.gateway().complete(7, original, "Repaired.", True, delivered_to=uuid4())
    original = parse_request(case.fake.issue["body"])
    delivery = case.recipient()
    mutations = list(case.fake.mutations)
    original_request = case.fake.request

    def edit_before_close(method, url, **kwargs):
        if method == "GET" and urlsplit(url).path == f"/repos/{REPO}/issues/7":
            envelope = parse_request(case.fake.issue["body"])
            if changed == "label":
                case.fake.issue["labels"] = []
            elif changed == "author":
                case.fake.issue["user"]["login"] = "untrusted-user"
            else:
                updates = {
                    "requester": {"requester": "advisor"},
                    "parent": {"parent_conversation_id": uuid4()},
                    "request": {
                        "request": envelope.request.model_copy(
                            update={"task": "Changed task."}
                        )
                    },
                    "result": {
                        "result": SupervisorResult(
                            resolved=False, repair_summary="Changed feedback."
                        )
                    },
                }
                case.fake.issue["body"] = render_request(
                    envelope.model_copy(update=updates[changed])
                )
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(case.fake, "request", edit_before_close)
    receipts = delivery.mailbox.poll()
    assert case.fake.issue["state"] == "open"
    assert case.fake.mutations == mutations
    with LocalEventStore(delivery.local.store_path) as store:
        [saved] = store.pending()
        assert_reply(saved.payload, original)
    assert len(receipts) == 1
    assert_reply(receipts[0].payload, original)


def test_supervisor_reply_is_durable_before_close_and_retries_without_redelivery(
    supervision_case,
    monkeypatch,
):
    case = supervision_case
    action = request(target="advisor")
    case.tool(role="student")(action, case.conversation)
    original = parse_request(case.fake.issue["body"])
    case.gateway().complete(7, original, "Repaired.", True, delivered_to=uuid4())
    original = parse_request(case.fake.issue["body"])
    delivery = case.recipient()
    original_request = case.fake.request
    patches = []

    def unavailable(method, url, **kwargs):
        if method == "PATCH":
            with LocalEventStore(delivery.local.store_path) as store:
                [saved] = store.pending()
                assert saved.kind == "supervisor_completed"
                assert_reply(saved.payload, original)
            patches.append(url)
            raise GitHubTransportError(method, url)
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(case.fake, "request", unavailable)
    receipts = delivery.mailbox.poll()
    assert len(receipts) == 1
    assert_reply(receipts[0].payload, original)
    assert receipts[0].kind == "supervisor_completed"
    assert patches
    assert case.fake.issue["state"] == "open"
    delivery.mailbox.acknowledge([receipts[0].dedupe_key])

    monkeypatch.setattr(case.fake, "request", original_request)
    assert delivery.mailbox.poll() == ()
    assert case.fake.issue["state"] == "closed"
    mutations = list(case.fake.mutations)
    delivery.handler()
    assert delivery.mailbox.poll() == ()
    assert case.fake.mutations == mutations


@pytest.mark.parametrize("failure", ["outage", "deleted-issue"])
def test_copied_supervisor_reply_does_not_need_github_to_acknowledge_or_continue(
    supervision_case,
    monkeypatch,
    failure,
):
    case = supervision_case
    action = request(target="advisor")
    case.tool(role="student")(action, case.conversation)
    original = parse_request(case.fake.issue["body"])
    case.gateway().complete(7, original, "Repaired.", True, delivered_to=uuid4())
    reply = parse_request(case.fake.issue["body"])
    delivery = case.recipient()
    original_request = case.fake.request
    disconnected = False

    def unavailable(method, url, **kwargs):
        if disconnected:
            if failure == "outage":
                raise GitHubTransportError(method, url)
            if urlsplit(url).path == f"/repos/{REPO}/issues/7":
                return HttpResponse(404, {"message": "Not Found"})
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(case.fake, "request", unavailable)

    class DisconnectAfterProcessing(Turns):
        def run(self, *args, **kwargs):
            nonlocal disconnected
            result = super().run(*args, **kwargs)
            if not disconnected:
                disconnected = True
                if failure == "deleted-issue":
                    case.fake.issue = None
                with LocalEventStore(delivery.local.store_path) as store:
                    store.enqueue(
                        LocalEvent(
                            kind="agent_result",
                            dedupe_key="other-local-work",
                            payload={
                                "parent_conversation_id": str(case.conversation.id),
                                "result": "Other work finished.",
                            },
                        )
                    )
            return result

    turns = DisconnectAfterProcessing()
    runtime = controller(
        delivery.mailbox,
        turns,
        role="student",
        inbox=delivery.inbox,
        conversation_id=case.conversation.id,
        conversation_for_events=StudentConversationSelector(
            AssignmentConversationRegistry(
                delivery.config.state_dir / "student-conversations.json"
            )
        ),
    )
    runtime.run(max_cycles=1)

    assert len(turns.calls[0][2]) == 1
    receipt_key = next(iter(turns.calls[0][2]))
    assert [call[2] for call in turns.calls] == [
        frozenset({receipt_key}),
        frozenset({"other-local-work"}),
    ]
    with LocalEventStore(delivery.local.store_path) as store:
        assert_reply(store.get(receipt_key).payload, reply)
        assert store.acknowledged((receipt_key, "other-local-work")) == {
            receipt_key,
            "other-local-work",
        }
    assert delivery.inbox.processed_turns() == ()
