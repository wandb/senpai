from types import SimpleNamespace

import pytest
from openhands_support import runtime_config
from test_controller import Mailbox, Turns, controller

from senpai_agent import supervision
from senpai_agent.delegation import AgentTask, DelegationManager, SpawnAgentsAction
from senpai_agent.github.supervision import SupervisorRequest
from senpai_agent.github.tools.contracts import AssignmentVersion
from senpai_agent.inbox import PersistentInbox
from senpai_agent.mailbox import ControllerEvent
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
            collected_on_completion=[],
            preflight_error=None,
            postflight_error=False,
            result='{"resolved":true,"repair_summary":"Repaired and checked."}',
        )

        class Gateway:
            def validate(self, request):
                if case.preflight_error is not None:
                    raise RuntimeError(case.preflight_error)
                if (
                    case.postflight_error
                    and case.children
                    and case.children[-1].finished
                ):
                    raise RuntimeError("assignment changed during repair")

            def prepare(self, request):
                return "Full PR discussion and explicit repair request."

            def complete(self, number, request, summary, resolved):
                case.completion_attempts.append((number, summary, resolved))
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

        def manager(config, **_kwargs):
            return DelegationManager(delegation_config(config), factory)

        def finish(_seconds):
            child = case.children[-1]
            child.finished = True
            child.complete(case.result, None)

        monkeypatch.setattr(supervision, "SupervisorGateway", lambda _config: Gateway())
        monkeypatch.setattr(supervision, "make_supervisor_manager", manager)
        monkeypatch.setattr(supervision, "time", SimpleNamespace(sleep=finish))
        case.manager = manager(config)
        case.lease_path = tmp_path / "lease.json"
        case.handler = supervision.SupervisorHandler(
            config,
            inbox,
            progress=ProgressLease(case.lease_path),
        )
        case.event = ControllerEvent(
            kind="supervisor_requested",
            dedupe_key="supervisor_requested:23",
            payload={
                "number": 23,
                "url": "https://github.com/acme/widgets/issues/23",
                "request": request.model_dump(mode="json"),
            },
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

    runtime = controller(
        Mailbox(((case.event, stale), (), ())),
        turns,
        role=role,
        conversation_id=case.parent_id,
        inbox=case.inbox,
        supervise=case.handler,
        reconcile=reconcile,
    )
    runtime.run(max_cycles=1)

    assert len(case.children) == 1
    assert case.turn.turn_id in case.prompts[0]
    assert "recovery budget exhausted" in case.prompts[0]
    assert "Original assignment" not in case.prompts[0]
    assert "old_assignment" not in reconciled
    assert case.completed[-1][2] is (outcome == "resolved")
    assert WorkerLease.read(case.lease_path).phase == "supervisor-complete"
    assert not case.manager.registry.active_rows()
    if outcome == "resolved":
        assert len(turns.calls) == 1
        assert turns.calls[0][1] == case.parent_id
        assert turns.calls[0][2] == frozenset({"original", "supervisor_recovered:23"})
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
):
    case = supervisor_case()
    case.completion_failures = 1
    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        case.handler(case.event)
    assert all(value is not None for value in case.collected_on_completion[0])
    assert case.inbox.turn(case.turn.turn_id).quarantine_reason is None
    case.inbox.quarantine(case.turn.turn_id, "a different failure")
    case.preflight_error = "assignment changed" if assignment_changed else None

    restarted = supervision.SupervisorHandler(case.config, case.inbox)
    restarted(case.event)
    assert len(case.children) == 1
    assert case.inbox.turn(case.turn.turn_id).quarantine_reason == "a different failure"
    assert case.completed[-1][2] is (not assignment_changed)

    # Different requesters may reuse the same request_id; Issues identify attempts.
    case.preflight_error = None
    next_issue = ControllerEvent(
        kind="supervisor_requested",
        dedupe_key="supervisor_requested:24",
        payload={**case.event.payload, "number": 24},
    )
    restarted(next_issue)
    assert len(case.children) == 2
    assert case.inbox.turn(case.turn.turn_id).quarantine_reason is None


def test_blocked_request_keeps_its_outcome_when_issue_completion_must_retry(
    supervisor_case,
):
    case = supervisor_case("student")
    case.preflight_error = "assignment is on hold"
    case.completion_failures = 1
    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        case.handler(case.event)

    # The restriction is gone, but this request already finished without a repair.
    case.preflight_error = None
    restarted = supervision.SupervisorHandler(case.config, case.inbox)
    restarted(case.event)
    assert case.children == []
    assert case.completion_attempts[0] == case.completion_attempts[1]
    assert case.completed[-1][2] is False
    assert case.inbox.pending_count(case.parent_id) == 1
    assert (
        case.inbox.turn(case.turn.turn_id).quarantine_reason
        == "recovery budget exhausted"
    )


@pytest.mark.parametrize("initially_resolved", [True, False])
def test_edited_request_cannot_relaunch_or_prevent_other_controller_work(
    supervisor_case,
    capsys,
    initially_resolved,
):
    case = supervisor_case("student")
    if not initially_resolved:
        case.result = '{"resolved":false,"repair_summary":"Repair was unsuccessful."}'
    case.completion_failures = 1
    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        case.handler(case.event)
    if initially_resolved:
        case.inbox.quarantine(case.turn.turn_id, "later failure")
    reason = case.inbox.turn(case.turn.turn_id).quarantine_reason
    changed = case.request.model_copy(update={"task": "A different repair."})
    edited = ControllerEvent(
        kind=case.event.kind,
        dedupe_key=case.event.dedupe_key,
        payload={**case.event.payload, "request": changed.model_dump(mode="json")},
    )
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
        Mailbox(((edited,), (other,), ())),
        turns,
        role="student",
        inbox=case.inbox,
        conversation_id=case.parent_id,
        supervise=case.handler,
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
    case.handler(edited)
    assert case.completed[-1][2] is False
    assert "new Supervisor request" in case.completed[-1][1]


@pytest.mark.parametrize("busy", ["subagent", "training"])
def test_supervisor_refuses_a_workspace_with_an_active_writer(supervisor_case, busy):
    case = supervisor_case("student")
    if busy == "subagent":
        case.manager.spawn_for_owner(
            SpawnAgentsAction(
                batch_key="work",
                tasks=[AgentTask(task="Edit the target", model="smart")],
            ),
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
    case.handler(case.event)
    assert len(case.children) == started
    assert case.completed[-1][2] is False
    assert busy in case.completed[-1][1]
    assert (
        case.inbox.turn(case.turn.turn_id).quarantine_reason
        == "recovery budget exhausted"
    )
    for row in case.manager.registry.active_rows():
        case.manager.cancel_for_owner([row["task_id"]], str(case.parent_id))


def test_controller_shutdown_cancels_supervisor_before_returning(
    supervisor_case, monkeypatch
):
    case = supervisor_case()

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(supervision.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        case.handler(case.event)
    assert case.children[0].interrupted
    assert not case.manager.registry.active_rows()
    assert not case.completed
    assert (
        case.inbox.turn(case.turn.turn_id).quarantine_reason
        == "recovery budget exhausted"
    )
