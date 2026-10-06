from types import SimpleNamespace

import pytest
from openhands_support import runtime_config
from test_controller import Mailbox, Turns, controller

from senpai_agent import supervision
from senpai_agent.delegation import AgentTask, DelegationManager
from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.supervision import SupervisorEnvelope, SupervisorRequest
from senpai_agent.github.tools.contracts import AssignmentVersion
from senpai_agent.github.workflow import GitHubAPIError, GitHubTransportError
from senpai_agent.inbox import PersistentInbox
from senpai_agent.local_events import LocalEventStore
from senpai_agent.mailbox import (
    CompositeMailbox,
    ControllerEvent,
    LocalMailbox,
    LocalStudentMailbox,
    SupervisedMailbox,
)
from senpai_agent.openhands_runner import delegation_config
from senpai_agent.state import (
    AssignmentConversationRegistry,
    StudentConversationSelector,
)
from senpai_agent.supervisor import ProgressLease, WorkerLease


@pytest.fixture
def supervisor_case(tmp_path, monkeypatch):
    def build(role="advisor"):
        config = runtime_config(
            tmp_path,
            role=role,
            advisor_branch="research",
            student_name="student-one",
        )
        inbox = PersistentInbox(config.state_dir / "delivery-inbox.sqlite3")
        assignment = AssignmentVersion(
            pr_number=7,
            assignment_id="experiment-7",
            revision_id="revision-1",
            expected_pr_head_sha="a" * 40,
        )
        parent_id = config.conversation_id
        if role == "student":
            parent_id = AssignmentConversationRegistry(
                config.state_dir / "student-conversations.json"
            ).for_assignment(assignment.assignment_id, assignment.revision_id)
        inbox.enqueue(parent_id, "original", "Original assignment")
        turn = inbox.next_turn(parent_id, "Continue the original assignment")
        for message in turn.messages:
            inbox.record_delivered(message.delivery_id, message.body)
        inbox.quarantine(turn.turn_id, "recovery budget exhausted")
        request = SupervisorRequest(
            request_id="repair-runtime",
            target="advisor" if role == "advisor" else "student-one",
            assignment=assignment if role == "student" else None,
            task="Repair the broken local module.",
        )
        case = SimpleNamespace(
            config=config,
            inbox=inbox,
            parent_id=parent_id,
            turn=turn,
            request=request,
            children=[],
            prompts=[],
            completed=[],
            closed=False,
            completion_failures=0,
            completion_attempts=[],
            completion_envelopes=[],
            collected_on_completion=[],
            preflight_error=None,
            postflight_error=False,
            result='{"resolved":true,"repair_summary":"Repaired and checked."}',
        )

        class Gateway:
            def pending(self):
                pending, case.pending = case.pending, []
                return pending

            def validate(self, request):
                if case.preflight_error is not None:
                    if isinstance(case.preflight_error, Exception):
                        raise case.preflight_error
                    raise RuntimeError(case.preflight_error)
                if (
                    case.postflight_error
                    and case.children
                    and case.children[-1].finished
                ):
                    if isinstance(case.postflight_error, Exception):
                        raise case.postflight_error
                    raise RuntimeError("assignment changed during repair")

            def prepare(self, envelope):
                self.validate(envelope.request)
                return "Full PR discussion and explicit repair request."

            def complete(self, number, envelope, summary, resolved, *, delivered_to):
                assert delivered_to == case.parent_id
                case.completion_attempts.append((number, summary, resolved))
                case.completion_envelopes.append(envelope)
                case.collected_on_completion.append(
                    [
                        row["collected_at"]
                        for row in case.manager.registry.rows()
                        if row["agent"] == "supervisor"
                    ]
                )
                if case.completion_failures:
                    case.completion_failures -= 1
                    raise RuntimeError("GitHub unavailable")
                case.completed.append((number, summary, resolved))
                case.closed = True

        class Child:
            finished = False
            interrupted = False

            def start(self, task, timeout, complete):
                case.prompts.append(task)
                self.complete = complete

            def interrupt(self):
                self.interrupted = True

        def factory(request):
            child = Child()
            case.children.append(child)
            return child

        def finish(_seconds):
            child = case.children[-1]
            child.finished = True
            child.complete(case.result, None)

        monkeypatch.setattr(supervision, "SupervisorGateway", lambda _config: Gateway())
        monkeypatch.setattr(
            supervision,
            "OpenHandsChildProcess",
            lambda _config, request: factory(request),
        )
        monkeypatch.setattr(supervision, "time", SimpleNamespace(sleep=finish))
        case.manager = DelegationManager(delegation_config(config), factory)
        case.lease_path = tmp_path / "lease.json"
        case.handler = supervision.SupervisorHandler(
            config,
            inbox,
            progress=ProgressLease(case.lease_path),
        )
        case.issue = {
            "number": 23,
            "html_url": "https://github.com/acme/widgets/issues/23",
        }
        case.envelope = SupervisorEnvelope(
            repo=config.github_repo,
            advisor_branch=config.advisor_branch,
            requester=request.target,
            parent_conversation_id=parent_id,
            request=request,
        )
        case.pending = [(case.issue, case.envelope)]
        mailbox_type = LocalMailbox if role == "advisor" else LocalStudentMailbox
        case.local_mailbox = mailbox_type(config.state_dir / f"{role}-events.sqlite3")
        case.delivery = controller(
            case.local_mailbox,
            Turns(),
            role=role,
            conversation_id=parent_id,
            inbox=inbox,
        )
        return case

    return build


@pytest.mark.parametrize("role", ["advisor", "student"])
@pytest.mark.parametrize("outcome", ["resolved", "unresolved", "malformed", "stale"])
def test_supervisor_repairs_before_resuming_the_original_quarantined_conversation(
    supervisor_case,
    role,
    outcome,
):
    case = supervisor_case(role)
    if outcome == "unresolved":
        case.result = (
            '{"resolved":false,"repair_summary":"Dependency still unavailable."}'
        )
    elif outcome == "malformed":
        case.result = "The code looks fixed."
    case.postflight_error = outcome == "stale"
    turns = Turns()
    reconciled = []
    stale = ControllerEvent(kind="old_assignment", dedupe_key="old", payload={})

    def reconcile(events):
        assert case.closed
        assert all(child.finished for child in case.children)
        reconciled.extend(event.kind for event in events)

    class CurrentSnapshotMailbox(Mailbox):
        def poll(self):
            return () if case.closed else (stale,)

    runtime = controller(
        SupervisedMailbox(
            CompositeMailbox(CurrentSnapshotMailbox(()), case.local_mailbox),
            case.handler,
        ),
        turns,
        role=role,
        conversation_id=case.parent_id,
        inbox=case.inbox,
        reconcile=reconcile,
    )
    runtime.run(max_cycles=1)

    assert len(case.children) == 1
    assert case.turn.turn_id in case.prompts[0]
    assert "recovery budget exhausted" in case.prompts[0]
    assert "Original assignment" not in case.prompts[0]
    assert "old_assignment" not in reconciled
    assert case.completed[-1][2] is (outcome == "resolved")
    assert case.completion_envelopes == [case.envelope]
    assert WorkerLease.read(case.lease_path).phase == "supervisor-complete"
    assert not case.manager.registry.active_rows()
    if outcome == "resolved":
        assert len(turns.calls) == 1
        assert turns.calls[0][1] == case.parent_id
        assert turns.calls[0][2] == frozenset({"original", "supervisor:23"})
        assert case.inbox.turn(case.turn.turn_id).quarantine_reason is None
    else:
        assert turns.calls == []
        assert (
            case.inbox.turn(case.turn.turn_id).quarantine_reason
            == "recovery budget exhausted"
        )
        assert case.inbox.pending_count(case.parent_id) == 1


@pytest.mark.parametrize("assignment_changed", [False, True])
def test_request_replay_reuses_child_and_cannot_reset_a_later_quarantine(
    supervisor_case,
    assignment_changed,
    capsys,
):
    case = supervisor_case("student")
    case.completion_failures = 1
    turns = Turns()
    controller(
        SupervisedMailbox(
            CompositeMailbox(Mailbox(()), case.local_mailbox),
            case.handler,
        ),
        turns,
        role="student",
        conversation_id=case.parent_id,
        inbox=case.inbox,
    ).run(max_cycles=1)
    assert "SENPAI_SUPERVISOR_ERROR" in capsys.readouterr().err
    assert [call[2] for call in turns.calls] == [
        frozenset({"original", "supervisor:23"}),
    ]
    assert case.completed == []
    assert all(value is not None for value in case.collected_on_completion[0])
    with LocalEventStore(case.local_mailbox.store_path) as store:
        assert store.acknowledged(("supervisor:23",)) == {"supervisor:23"}
        assert store.pending() == []

    case.inbox.enqueue(case.parent_id, "later", "The next assignment step")
    later_turn = case.inbox.next_turn(case.parent_id, "Continue the assignment")
    case.inbox.quarantine(later_turn.turn_id, "a different failure")
    case.preflight_error = "PR head changed" if assignment_changed else None

    restarted = supervision.SupervisorHandler(case.config, case.inbox)
    case.pending = [(case.issue, case.envelope)]
    replay_turns = Turns()
    runtime = controller(
        SupervisedMailbox(CompositeMailbox(Mailbox(()), case.local_mailbox), restarted),
        replay_turns,
        role="student",
        conversation_id=case.parent_id,
        inbox=case.inbox,
    )
    runtime.run(max_cycles=1)
    assert len(case.children) == 1
    assert (
        case.inbox.turn(later_turn.turn_id).quarantine_reason == "a different failure"
    )
    assert replay_turns.calls == []
    assert case.completed == [(23, "Repaired and checked.", True)]
    assert case.completion_attempts[0] == case.completion_attempts[1]
    assert case.inbox.turn(case.turn.turn_id).event_keys == (
        "original",
        "supervisor:23",
    )
    assert case.inbox.pending_count(case.parent_id) == 0

    # Different requesters may reuse the same request_id; Issues identify attempts.
    case.preflight_error = None
    next_issue = {
        "number": 24,
        "html_url": "https://github.com/acme/widgets/issues/24",
    }
    case.pending = [(next_issue, case.envelope)]
    runtime.run(max_cycles=1)
    assert len(case.children) == 2
    assert case.inbox.turn(later_turn.turn_id).quarantine_reason is None
    assert [call[2] for call in replay_turns.calls] == [
        frozenset({"later", "supervisor:24"}),
    ]


def test_blocked_request_keeps_its_outcome_when_issue_completion_must_retry(
    supervisor_case,
):
    case = supervisor_case("student")
    case.preflight_error = "assignment is on hold"
    case.completion_failures = 1
    case.handler()
    assert case.completed == []
    case.delivery._poll_into_inbox()

    # The restriction is gone, but this request already finished without a repair.
    case.preflight_error = None
    restarted = supervision.SupervisorHandler(case.config, case.inbox)
    case.pending = [(case.issue, case.envelope)]
    restarted()
    case.delivery._poll_into_inbox()
    assert case.children == []
    assert case.completion_attempts[0] == case.completion_attempts[1]
    assert case.completed[-1][2] is False
    assert case.inbox.pending_count(case.parent_id) == 1
    assert (
        case.inbox.turn(case.turn.turn_id).quarantine_reason
        == "recovery budget exhausted"
    )


@pytest.mark.parametrize("edit", [None, "task", "assignment", "requester", "parent"])
def test_unfinished_repair_replay_rejects_changed_requests(
    supervisor_case,
    monkeypatch,
    edit,
):
    case = supervisor_case("student")

    def killed(*_args, **_kwargs):
        raise SystemExit("Controller killed before cleanup")

    # An abrupt process exit leaves its independent child and registry row alive.
    with monkeypatch.context() as crash:
        crash.setattr(supervision.time, "sleep", killed)
        crash.setattr(DelegationManager, "cancel", killed)
        with pytest.raises(SystemExit, match="Controller killed"):
            case.handler()
    assert len(case.manager.registry.active_rows()) == 1
    assert not case.local_mailbox.poll()

    changed = case.envelope
    registry = AssignmentConversationRegistry(
        case.config.state_dir / "student-conversations.json"
    )
    if edit == "task":
        request = case.request.model_copy(update={"task": "Repair a different module."})
        changed = changed.model_copy(update={"request": request})
    elif edit == "assignment":
        assignment = case.request.assignment.model_copy(
            update={"revision_id": "revision-2"}
        )
        request = case.request.model_copy(update={"assignment": assignment})
        changed = changed.model_copy(update={"request": request})
        case.parent_id = registry.for_assignment(
            assignment.assignment_id,
            assignment.revision_id,
        )
    elif edit == "requester":
        changed = changed.model_copy(update={"requester": "advisor"})
    elif edit == "parent":
        changed = changed.model_copy(
            update={"parent_conversation_id": case.config.conversation_id}
        )
    case.pending = [(case.issue, changed)]
    restarted = supervision.SupervisorHandler(case.config, case.inbox)
    restarted()
    case.delivery.conversation_for_events = StudentConversationSelector(registry)
    case.delivery._poll_into_inbox()

    assert len(case.children) == 1
    assert not case.manager.registry.active_rows()
    assert all(value is not None for value in case.collected_on_completion[-1])
    assert case.completed[-1][2] is (edit is None)
    assert case.completion_envelopes == [changed]
    if edit is None:
        assert not case.children[0].interrupted
        assert case.inbox.turn(case.turn.turn_id).quarantine_reason is None
    else:
        assert case.children[0].interrupted
        assert "new Supervisor request" in case.completed[-1][1]
        assert (
            case.inbox.turn(case.turn.turn_id).quarantine_reason
            == "recovery budget exhausted"
        )


@pytest.mark.parametrize("initially_resolved", [True, False])
@pytest.mark.parametrize("edit", ["task", "requester", "parent"])
def test_edited_request_cannot_relaunch_or_prevent_other_controller_work(
    supervisor_case,
    capsys,
    initially_resolved,
    edit,
):
    case = supervisor_case("student")
    if not initially_resolved:
        case.result = '{"resolved":false,"repair_summary":"Repair was unsuccessful."}'
    case.completion_failures = 1
    case.handler()
    assert case.completed == []
    case.delivery._poll_into_inbox()
    if initially_resolved:
        case.inbox.quarantine(case.turn.turn_id, "later failure")
    reason = case.inbox.turn(case.turn.turn_id).quarantine_reason
    if edit == "task":
        changed = case.request.model_copy(update={"task": "A different repair."})
        edited = case.envelope.model_copy(update={"request": changed})
    elif edit == "requester":
        edited = case.envelope.model_copy(update={"requester": "advisor"})
    else:
        edited = case.envelope.model_copy(
            update={"parent_conversation_id": case.config.conversation_id}
        )
    case.pending = [(case.issue, edited)]
    # A transient failure publishing this rejection must not restart the controller.
    case.completion_failures = 1
    other = ControllerEvent(
        kind="human_issue",
        dedupe_key="human_issue:99:501",
        payload={
            "number": 99,
            "parent_conversation_id": str(case.config.conversation_id),
        },
    )
    turns = Turns()
    runtime = controller(
        SupervisedMailbox(Mailbox(((other,), ())), case.handler),
        turns,
        role="student",
        inbox=case.inbox,
        conversation_id=case.parent_id,
        conversation_for_events=StudentConversationSelector(
            AssignmentConversationRegistry(
                case.config.state_dir / "student-conversations.json"
            )
        ),
    )
    runtime.run(max_cycles=1)
    assert "SENPAI_SUPERVISOR_ERROR" in capsys.readouterr().err
    assert len(case.children) == 1
    assert len(turns.calls) == 1
    assert turns.calls[0][1] == case.config.conversation_id
    assert case.inbox.turn(case.turn.turn_id).quarantine_reason == reason
    case.pending = [(case.issue, edited)]
    case.handler()
    assert case.completed[-1][2] is False
    assert "new Supervisor request" in case.completed[-1][1]
    assert case.completion_envelopes == [case.envelope, edited, edited]


@pytest.mark.parametrize("busy", ["subagent", "training"])
def test_supervisor_refuses_a_workspace_with_an_active_writer(supervisor_case, busy):
    case = supervisor_case("student")
    if busy == "subagent":
        case.manager.spawn(
            "work",
            [AgentTask(task="Edit the target", model="smart")],
            str(case.parent_id),
        )
    else:
        case.handler.monitor_store = SimpleNamespace(
            active=lambda: [SimpleNamespace(training_id="run-1")]
        )
        case.handler.training = SimpleNamespace(
            get_training_status=lambda _id: SimpleNamespace(state="running"),
        )
    started = len(case.children)
    case.handler()
    assert len(case.children) == started
    assert case.completed[-1][2] is False
    assert busy in case.completed[-1][1]
    assert "new request_id" in case.completed[-1][1]
    assert (
        case.inbox.turn(case.turn.turn_id).quarantine_reason
        == "recovery budget exhausted"
    )
    for row in case.manager.registry.active_rows():
        case.manager.cancel([row["task_id"]], str(case.parent_id))


def test_controller_shutdown_cancels_supervisor_before_returning(
    supervisor_case, monkeypatch
):
    case = supervisor_case()

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(supervision.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        case.handler()
    case.delivery._poll_into_inbox()
    assert case.children[0].interrupted
    assert not case.manager.registry.active_rows()
    assert not case.completed
    [saved] = case.manager.registry.rows()
    assert "Controller shutdown" in saved["error"]
    assert "new request_id" in saved["error"]
    assert (
        case.inbox.turn(case.turn.turn_id).quarantine_reason
        == "recovery budget exhausted"
    )


@pytest.mark.parametrize("stage", ["prepare", "postflight"])
@pytest.mark.parametrize(
    "error",
    [
        GitHubTransportError(
            "GET", "https://api.github.com/repos/acme/widgets/pulls/7"
        ),
        GitHubAPIError("GET", "https://api.github.com/repos/acme/widgets/pulls/7", 503),
        GitHubReadError("GitHub is rate-limited", status_code=429),
    ],
    ids=["transport", "server", "reader"],
)
def test_transient_github_read_retries_without_consuming_the_repair(
    supervisor_case, stage, error
):
    case = supervisor_case()
    if stage == "prepare":
        case.preflight_error = error
    else:
        case.postflight_error = error
    case.handler()
    assert not case.completed
    assert case.local_mailbox.poll() == ()
    assert case.inbox.turn(case.turn.turn_id).quarantine_reason
    case.preflight_error = None
    case.postflight_error = False
    case.pending = [(case.issue, case.envelope)]
    case.handler()
    assert len(case.children) == 1
    assert case.completed == [(23, "Repaired and checked.", True)]
