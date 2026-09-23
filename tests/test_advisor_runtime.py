import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.llm import MessageToolCall

from senpai_agent.advisor import (
    AdvisorEventPump,
    advisor_conversation_id,
    advisor_main,
    deliver_pending_events,
)
from senpai_agent.delegation import AgentStatusAction, AgentStatusObservation
from senpai_agent.hooks import queued_feedback_marker
from senpai_agent.inbox import PersistentInbox, deliver_turn_messages
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import ControllerEvent


class ConversationStateStub:
    def __init__(
        self,
        events=(),
        execution_status=ConversationExecutionStatus.FINISHED,
    ):
        self.events = list(events)
        self.execution_status = execution_status
        self.inspected = threading.Event()
        self._lock = threading.RLock()

    def active_branch(self):
        self.inspected.set()
        return list(self.events)

    def append(self, event) -> None:
        with self:
            self.events.append(event)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self._lock.release()


def pending_tool_action() -> ActionEvent:
    action = AgentStatusAction()
    return ActionEvent(
        thought=[],
        action=action,
        tool_name="agent_status",
        tool_call_id="status-call",
        tool_call=MessageToolCall(
            id="status-call",
            name="agent_status",
            arguments=json.dumps(action.model_dump(mode="json")),
            origin="completion",
        ),
        llm_response_id="status-response",
    )


def completed_tool_action(action: ActionEvent) -> ObservationEvent:
    return ObservationEvent(
        tool_name="agent_status",
        tool_call_id=action.tool_call_id,
        action_id=action.id,
        observation=AgentStatusObservation(tasks=[]),
    )


STEERING_CONVERSATION_ID = "00000000-0000-0000-0000-000000000017"


class SteeringConversation:
    def __init__(
        self,
        status: ConversationExecutionStatus = ConversationExecutionStatus.RUNNING,
    ):
        self.state = ConversationStateStub(execution_status=status)
        self.messages: list[str] = []
        self.interrupted = threading.Event()
        self.paused = threading.Event()
        self.interrupts = 0
        self.pauses = 0

    def send_message(self, message: str, sender: str | None = None) -> None:
        self.messages.append(message)
        self.state.append(SimpleNamespace(message=message, sender=sender))

    def interrupt(self) -> None:
        self.interrupts += 1
        self.interrupted.set()
        self.state.execution_status = ConversationExecutionStatus.PAUSED

    def pause(self) -> None:
        with self.state:
            self.pauses += 1
            self.paused.set()
            self.state.execution_status = ConversationExecutionStatus.PAUSED


def active_steering_turn(tmp_path: Path):
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(STEERING_CONVERSATION_ID, "controller:event", "controller event")
    turn = inbox.next_turn(STEERING_CONVERSATION_ID, "controller prompt")
    assert turn is not None
    conversation = SteeringConversation()
    deliver_turn_messages(conversation, inbox, turn.turn_id)
    return inbox, turn, conversation


def test_advisor_conversation_id_is_stable_and_persisted(tmp_path: Path):
    first = advisor_conversation_id(tmp_path)
    second = advisor_conversation_id(tmp_path)

    assert first == second
    assert (tmp_path / "advisor-conversation-id").read_text() == f"{first}\n"


def test_advisor_conversation_id_write_preserves_the_previous_value_on_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original = advisor_conversation_id(tmp_path)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        advisor_conversation_id(
            tmp_path,
            "00000000-0000-0000-0000-000000000099",
        )

    assert (tmp_path / "advisor-conversation-id").read_text() == f"{original}\n"


def test_event_store_deduplicates_and_survives_reopen(tmp_path: Path):
    database = tmp_path / "advisor-events.sqlite3"
    event = LocalEvent(
        kind="review_ready",
        dedupe_key="review_ready:3467:abc123",
        payload={"pr": 3467, "head_sha": "abc123"},
    )

    with LocalEventStore(database) as store:
        assert store.enqueue(event) is True
        assert store.pending_count() == 1
        assert store.enqueue(event) is False
        with pytest.raises(RuntimeError, match="reused with a different payload"):
            store.enqueue(event.model_copy(update={"payload": {"pr": 999}}))

    with LocalEventStore(database) as reopened:
        pending = reopened.pending()
        assert pending == [event]
        reopened.acknowledge(event.dedupe_key)
        assert reopened.pending_count() == 0
        assert reopened.pending() == []


def test_event_store_discards_absent_level_triggers(tmp_path: Path):
    database = tmp_path / "advisor-events.sqlite3"
    stale = LocalEvent(
        kind="student_available_for_assignment",
        dedupe_key="student_available_for_assignment:Fern",
        payload={"student": "Fern"},
    )
    retained = stale.model_copy(
        update={
            "dedupe_key": "student_available_for_assignment:Frieren",
            "payload": {"student": "Frieren"},
        }
    )

    with LocalEventStore(database) as store:
        assert store.enqueue(stale) is True
        assert store.enqueue(retained) is True
        store.acknowledge(stale.dedupe_key)
        assert store.discard_prefix(
            "student_available_for_assignment:",
            retained_keys=(retained.dedupe_key,),
        ) == 1
        assert store.enqueue(stale) is True
        assert store.enqueue(retained) is False


def test_event_message_renders_observation_time_and_structured_payload():
    event = LocalEvent(
        kind="review_ready",
        dedupe_key="review_ready:17:abc",
        payload={
            "pr": 17,
            "head_sha": "abc",
        },
        observed_at=datetime(2026, 7, 31, 12, 30, tzinfo=UTC),
    )

    assert event.to_user_message() == (
        "# Senpai event: review_ready\n\n"
        "Observed at (UTC): 2026-07-31T12:30:00+00:00\n\n"
        "```json\n"
        "{\n"
        '  "head_sha": "abc",\n'
        '  "pr": 17\n'
        "}\n"
        "```"
    )


def test_deliver_pending_events_acknowledges_only_messages_sent(tmp_path: Path):
    first = LocalEvent(
        kind="review_ready",
        dedupe_key="review_ready:11:ddd",
        payload={"pr": 11},
    )
    second = LocalEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()
            self.messages: list[str] = []

        def send_message(self, message: str) -> None:
            if self.messages:
                raise RuntimeError("conversation unavailable")
            self.messages.append(message)

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(first)
        store.enqueue(second)
        conversation = Conversation()

        with pytest.raises(RuntimeError, match="conversation unavailable"):
            deliver_pending_events(store, conversation)

        assert conversation.messages == [first.to_user_message()]
        assert store.pending() == [second]


def test_event_pump_keeps_events_queued_while_a_tool_action_is_unmatched(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub([pending_tool_action()])
            self.messages: list[str] = []

        def send_message(self, message: str) -> None:
            self.messages.append(message)

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        conversation = Conversation()

        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            assert conversation.state.inspected.wait(1)

        assert conversation.messages == []
        assert store.pending() == [event]


def test_event_pump_delivers_queued_event_after_the_tool_boundary_is_safe(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )
    action = pending_tool_action()

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub([action])
            self.messages: list[str] = []
            self.received = threading.Event()

        def send_message(self, message: str) -> None:
            self.messages.append(message)
            self.received.set()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        conversation = Conversation()
        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            assert conversation.state.inspected.wait(1)
            assert conversation.messages == []
            assert store.pending() == [event]

            conversation.state.append(completed_tool_action(action))
            assert conversation.received.wait(1)

        assert conversation.messages == [event.to_user_message()]
        assert store.pending() == []


def test_event_pump_injects_new_events_while_conversation_is_running(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="review_ready",
        dedupe_key="review_ready:12:eee",
        payload={"pr": 12},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()
            self.messages: list[str] = []

        def send_message(self, message: str) -> None:
            self.messages.append(message)

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        conversation = Conversation()
        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            store.enqueue(event)
            deadline = time.monotonic() + 1
            while not conversation.messages and time.monotonic() < deadline:
                time.sleep(0.01)
            assert store.pending() == [event]

        assert conversation.messages == [event.to_user_message()]
        assert store.pending() == []


def test_event_pump_queues_into_the_controller_inbox_without_mid_turn_injection(
    tmp_path: Path,
):
    """
    Requirement: controller and event-pump messages use one durable inbox.
    Interface: AdvisorEventPump, PersistentInbox, and the active conversation.
    """
    event = LocalEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )
    conversation_id = "00000000-0000-0000-0000-000000000017"

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()
            self.messages: list[str] = []
            self.received = threading.Event()

        def send_message(self, message: str) -> None:
            self.messages.append(message)
            self.received.set()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        conversation = Conversation()
        inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
        inbox.enqueue(conversation_id, "controller:event", "controller event")
        active = inbox.next_turn(conversation_id, "controller prompt")
        assert active is not None
        for message in active.messages:
            inbox.record_delivered(message.delivery_id, message.body)
        with AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=conversation_id,
        ):
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)

        assert conversation.messages == []
        assert store.pending() == []
        retry = inbox.next_turn(conversation_id, "retry prompt")
        assert retry is not None and retry.turn_id == active.turn_id
        assert [message.body for message in retry.events] == ["controller event"]
        assert inbox.pending_count(conversation_id) == 1

        inbox.record_processed(active.turn_id)
        inbox.acknowledge(active.turn_id)
        next_turn = inbox.next_turn(conversation_id, "next prompt")
        assert next_turn is not None
        assert [message.body for message in next_turn.events] == [
            event.to_inbox_message()
        ]
        assert next_turn.acknowledgement_keys == (event.dedupe_key,)


def test_event_pump_accepts_an_in_memory_inbox(tmp_path: Path):
    with (
        LocalEventStore(tmp_path / "events.sqlite3") as store,
        PersistentInbox() as inbox,
    ):
        AdvisorEventPump(
            store,
            SteeringConversation(),
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )


@pytest.mark.parametrize(
    "kind",
    ["human_issue", "human_pr_comment"],
    ids=["human-issue", "human-pr-comment"],
)
def test_human_instruction_steers_the_active_turn_after_the_tool_boundary(
    tmp_path: Path,
    kind: str,
):
    event = LocalEvent(
        kind=kind,
        dedupe_key=f"{kind}:1",
        payload={"message": "Change direction."},
    )
    inbox, active, conversation = active_steering_turn(tmp_path)
    conversation.state.acquire()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
            steering_grace_seconds=0.2,
        )
        assert pump.prepare_run()
        pump.run_started()
        with pump:
            store.enqueue(event)
            assert not conversation.interrupted.wait(0.03)
            conversation.state.release()
            assert conversation.interrupted.wait(1)
            assert pump.finish_run()

        assert store.pending() == []
    steered = inbox.turn(active.turn_id)
    assert steered.event_keys == ("controller:event", event.dedupe_key)
    assert conversation.messages[-1] == event.to_inbox_message()


@pytest.mark.parametrize(
    "kind",
    ["human_issue", "human_pr_comment"],
    ids=["human-issue", "human-pr-comment"],
)
def test_event_pump_drops_an_acknowledged_human_instruction(
    tmp_path: Path,
    kind: str,
):
    event = LocalEvent(
        kind=kind,
        dedupe_key=f"{kind}:1",
        payload={"message": "Change direction."},
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    conversation = SteeringConversation()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert store.enqueue(event)
        assert pump._transfer_to_inbox() == 1
        first = inbox.next_turn(STEERING_CONVERSATION_ID, "controller prompt")
        assert first is not None
        deliver_turn_messages(conversation, inbox, first.turn_id)
        inbox.record_processed(first.turn_id)
        inbox.acknowledge(first.turn_id)

        store.discard_prefix(event.dedupe_key)
        assert store.enqueue(event)
        assert pump._transfer_to_inbox() == 1

        assert store.pending() == []
        assert inbox.next_turn(STEERING_CONVERSATION_ID, "controller prompt") is None


def test_student_feedback_waits_for_the_step_and_marks_a_clean_unwind(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:1",
        payload={"message": "Try the narrower experiment next."},
    )
    inbox, active, conversation = active_steering_turn(tmp_path)
    marker = queued_feedback_marker(inbox.path.parent)
    conversation.state.acquire()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert pump.prepare_run()
        pump.run_started()
        marker.touch()
        with pump:
            assert not marker.exists()
            store.enqueue(event)
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert store.pending() == []
            assert not conversation.paused.wait(0.03)
            assert conversation.interrupts == 0
            conversation.state.release()
            assert conversation.paused.wait(1)
            assert marker.is_file()
            assert pump.finish_run()

    assert not marker.exists()
    assert conversation.pauses == 1
    assert conversation.interrupts == 0
    assert inbox.turn(active.turn_id).event_keys == (
        "controller:event",
        event.dedupe_key,
    )
    assert conversation.messages[-1] == event.to_inbox_message()


def test_review_ready_waits_for_the_step_without_interrupting_current_work(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="review_ready",
        dedupe_key="review_ready:29:abc123",
        payload={"pr": 29, "head_sha": "abc123"},
    )
    inbox, active, conversation = active_steering_turn(tmp_path)
    conversation.state.acquire()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert pump.prepare_run()
        pump.run_started()
        with pump:
            store.enqueue(event)
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert store.pending() == []
            assert not conversation.paused.wait(0.03)
            assert conversation.interrupts == 0
            conversation.state.release()
            assert conversation.paused.wait(1)
            assert pump.finish_run()

    assert conversation.pauses == 1
    assert conversation.interrupts == 0
    assert inbox.turn(active.turn_id).event_keys == (
        "controller:event",
        event.dedupe_key,
    )
    assert conversation.messages[-1] == event.to_inbox_message()


def test_student_feedback_waits_for_a_starting_run_to_leave_idle(tmp_path: Path):
    event = LocalEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:1",
        payload={"message": "Try the narrower experiment next."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)
    conversation.state.execution_status = ConversationExecutionStatus.IDLE

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert pump.prepare_run()
        pump.run_started()
        with pump:
            store.enqueue(event)
            assert not conversation.paused.wait(0.03)
            conversation.state.execution_status = ConversationExecutionStatus.RUNNING
            assert conversation.paused.wait(1)
            assert pump.finish_run()

    assert conversation.pauses == 1
    assert conversation.interrupts == 0


def test_human_issue_upgrades_queued_feedback_to_an_interrupt(tmp_path: Path):
    feedback = LocalEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:1",
        payload={"message": "Try the narrower experiment next."},
    )
    instruction = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:1",
        payload={"message": "Stop and inspect the latest frontier now."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)
    marker = queued_feedback_marker(inbox.path.parent)
    conversation.state.acquire()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
            steering_grace_seconds=0.03,
        )
        assert pump.prepare_run()
        pump.run_started()
        with pump:
            store.enqueue(feedback)
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            store.enqueue(instruction)
            assert conversation.interrupted.wait(1)
            conversation.state.release()
            assert pump.finish_run()

    assert conversation.pauses == 0
    assert conversation.interrupts == 1
    assert not marker.exists()
    assert "interrupted the active run" in conversation.messages[-3]
    assert conversation.messages[-2:] == [
        feedback.to_inbox_message(),
        instruction.to_inbox_message(),
    ]


def test_human_issue_interrupts_a_tool_after_the_steering_grace(tmp_path: Path):
    event = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:1",
        payload={"message": "Change direction now."},
    )
    paired = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:paired",
        payload={"message": "Keep this paired instruction too."},
    )
    followup = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:2",
        payload={"message": "And preserve the current workspace."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)
    conversation.state.acquire()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
            steering_grace_seconds=0.03,
        )
        assert pump.prepare_run()
        pump.run_started()
        store.enqueue(event)
        store.enqueue(paired)
        with pump:
            assert not conversation.interrupted.wait(0.01)
            assert conversation.interrupted.wait(1)
            store.enqueue(followup)
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            conversation.state.release()
            assert pump.finish_run()

    assert conversation.interrupts == 1
    assert conversation.messages[-3:] == [
        event.to_inbox_message(),
        paired.to_inbox_message(),
        followup.to_inbox_message(),
    ]
    assert "Trusted human steering" in conversation.messages[-4]


def test_human_issue_before_run_is_delivered_without_starting_then_cancelling(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:1",
        payload={"message": "Use this direction first."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        with pump:
            store.enqueue(event)
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not pump.prepare_run()

    assert conversation.interrupts == 0
    assert conversation.messages[-1] == event.to_inbox_message()


def test_student_feedback_does_not_resume_an_unrelated_pause(tmp_path: Path):
    event = LocalEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:1",
        payload={"message": "Use this after restart."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert pump.prepare_run()
        pump.run_started()
        conversation.state.execution_status = ConversationExecutionStatus.PAUSED
        store.enqueue(event)
        with pump:
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not pump.finish_run()

    assert conversation.pauses == 0
    assert conversation.interrupts == 0
    assert conversation.messages[-1] == event.to_inbox_message()


@pytest.mark.parametrize(
    "status",
    [ConversationExecutionStatus.PAUSED, ConversationExecutionStatus.ERROR],
)
def test_human_issue_interrupts_a_run_with_stale_persisted_status(
    tmp_path: Path,
    status: ConversationExecutionStatus,
):
    event = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:1",
        payload={"message": "Stop the newly starting run."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert pump.prepare_run()
        conversation.state.execution_status = status
        pump.run_started()
        with pump:
            store.enqueue(event)
            assert conversation.interrupted.wait(1)
            assert pump.finish_run()

    assert conversation.interrupts == 1
    assert conversation.messages[-1] == event.to_inbox_message()
    assert all("Trusted human steering" not in message for message in conversation.messages)


@pytest.mark.parametrize(
    "status",
    [
        ConversationExecutionStatus.IDLE,
        ConversationExecutionStatus.WAITING_FOR_CONFIRMATION,
        ConversationExecutionStatus.FINISHED,
        ConversationExecutionStatus.ERROR,
        ConversationExecutionStatus.STUCK,
    ],
)
def test_human_issue_does_not_claim_a_cleanly_ended_run_was_interrupted(
    tmp_path: Path,
    status: ConversationExecutionStatus,
):
    event = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:1",
        payload={"message": "Apply this next."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)
    conversation.state.execution_status = status
    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert pump.prepare_run()
        pump.run_started()
        with pump:
            store.enqueue(event)
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert pump.finish_run()

    assert conversation.messages[-1] == event.to_inbox_message()
    assert all("Trusted human steering" not in message for message in conversation.messages)


def test_human_issue_joins_a_failed_turn_for_its_retry(tmp_path: Path):
    event = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:1",
        payload={"message": "Recover with this direction."},
    )
    conversation_id = "00000000-0000-0000-0000-000000000017"

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub(
                execution_status=ConversationExecutionStatus.ERROR
            )
            self.messages: list[str] = []

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)

    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(conversation_id, "controller:event", "controller event")
    active = inbox.next_turn(conversation_id, "controller prompt")
    assert active is not None
    conversation = Conversation()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        with AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=conversation_id,
        ):
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
        assert store.pending() == []

    assert inbox.turn(active.turn_id).event_keys == (
        "controller:event",
        event.dedupe_key,
    )
    assert conversation.messages == []


def test_non_finished_turn_leaves_delivered_child_result_pending(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub(
                execution_status=ConversationExecutionStatus.PAUSED
            )
            self.messages: list[str] = []
            self.received = threading.Event()

        def send_message(self, message: str) -> None:
            self.messages.append(message)
            self.received.set()

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        conversation = Conversation()

        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            assert conversation.received.wait(1)

        assert conversation.messages == [event.to_user_message()]
        assert store.pending() == [event]


def test_event_pump_routes_child_results_to_their_parent_conversation(
    tmp_path: Path,
):
    first_parent = "00000000-0000-0000-0000-000000000001"
    second_parent = "00000000-0000-0000-0000-000000000002"
    first = LocalEvent(
        kind="agent_result",
        dedupe_key="agent_result:first",
        payload={"parent_conversation_id": first_parent},
    )
    second = LocalEvent(
        kind="agent_result",
        dedupe_key="agent_result:second",
        payload={"parent_conversation_id": second_parent},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()
            self.messages: list[str] = []

        def send_message(self, message: str) -> None:
            self.messages.append(message)

    with LocalEventStore(tmp_path / "student-events.sqlite3") as store:
        store.enqueue(first)
        store.enqueue(second)
        conversation = Conversation()
        with AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            parent_conversation_id=first_parent,
        ):
            deadline = time.monotonic() + 1
            while not conversation.messages and time.monotonic() < deadline:
                time.sleep(0.01)

        assert conversation.messages == [first.to_user_message()]
        assert store.pending() == [second]


def test_event_pump_surfaces_delivery_failure_and_leaves_event_pending(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="review_ready",
        dedupe_key="review_ready:13:fff",
        payload={"pr": 13},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()

        def send_message(self, _message: str) -> None:
            raise RuntimeError("conversation rejected event")

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        with (
            pytest.raises(RuntimeError, match="conversation rejected event"),
            AdvisorEventPump(store, Conversation(), poll_interval=0.01) as pump,
        ):
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            pump.prepare_run()

        assert store.pending() == [event]


def test_event_pump_failure_after_arming_interrupts_again_when_the_run_starts(
    tmp_path: Path,
):
    conversation = SteeringConversation()
    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(store, conversation)
        assert pump.prepare_run()
        pump._fail(RuntimeError("pump failed before run startup"))

        assert conversation.interrupts == 0
        pump.run_started()

    assert conversation.interrupts == 1


def test_external_event_source_failure_interrupts_the_active_run(tmp_path: Path):
    conversation = SteeringConversation()
    failure: list[BaseException] = []
    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            failure_source=lambda: failure[-1] if failure else None,
        )
        assert pump.prepare_run()
        with pytest.raises(RuntimeError, match="watcher failed"):
            with pump:
                pump.run_started()
                failure.append(RuntimeError("watcher failed"))
                assert conversation.interrupted.wait(1)

    assert conversation.interrupts == 1


def test_external_event_source_failure_does_not_mask_a_primary_error(
    tmp_path: Path,
):
    conversation = SteeringConversation()
    failure = RuntimeError("watcher failed")
    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            failure_source=lambda: failure,
        )
        assert pump.prepare_run()
        with pytest.raises(ValueError, match="primary turn failure"):
            with pump:
                pump.run_started()
                assert conversation.interrupted.wait(1)
                raise ValueError("primary turn failure")


def test_event_pump_surfaces_a_mid_batch_failure_without_boundary_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    event = LocalEvent(
        kind="human_issue",
        dedupe_key="human_issue:1",
        payload={"message": "Change direction."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)

    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        def fail_acknowledgement(_dedupe_key: str) -> None:
            raise RuntimeError("inbox acknowledgement failed")

        monkeypatch.setattr(store, "acknowledge", fail_acknowledgement)
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
            steering_grace_seconds=1,
        )
        assert pump.prepare_run()
        pump.run_started()
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="inbox acknowledgement failed"):
            with pump:
                store.enqueue(event)
                assert pump._stop.wait(1)

    assert time.monotonic() - started < 1
    assert conversation.interrupts == 1


def test_event_pump_surfaces_a_queue_boundary_failure_and_unwinds_the_run(
    tmp_path: Path,
):
    event = LocalEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:1",
        payload={"message": "Apply this next."},
    )
    inbox, _active, conversation = active_steering_turn(tmp_path)

    def fail_pause() -> None:
        raise RuntimeError("conversation pause failed")

    conversation.pause = fail_pause
    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
            steering_grace_seconds=1,
        )
        assert pump.prepare_run()
        pump.run_started()
        with pytest.raises(RuntimeError, match="conversation pause failed"):
            with pump:
                store.enqueue(event)
                assert pump._stop.wait(1)

    assert conversation.interrupts == 1


def test_advisor_cli_reports_the_pending_event_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    with LocalEventStore(tmp_path / "advisor-events.sqlite3") as store:
        store.enqueue(
            LocalEvent(
                kind="review_ready",
                dedupe_key="review_ready:1:a",
                payload={"pr": 1},
            )
        )

    assert (
        advisor_main(
            [
                "pending-count",
                "--state-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "1"


def test_inbox_rendering_excludes_transport_only_parent_conversation_id():
    """Watcher and controller paths must render one payload for one event key."""
    payload = {"assignment_id": "a-17", "revision_id": "r-2"}
    controller_event = ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key="feedback:17",
        payload=payload,
    )
    watcher_event = LocalEvent(
        kind=controller_event.kind,
        dedupe_key=controller_event.dedupe_key,
        payload={**payload, "parent_conversation_id": "conversation-17"},
    )

    assert watcher_event.to_inbox_message() == controller_event.to_prompt()


def test_student_availability_rendering_matches_controller_and_watcher_paths():
    controller_event = ControllerEvent(
        kind="student_available_for_assignment",
        dedupe_key="student_available_for_assignment:qwen-edward",
        payload={"student": "qwen-edward"},
    )
    watcher_event = LocalEvent(
        kind=controller_event.kind,
        dedupe_key=controller_event.dedupe_key,
        payload=controller_event.payload,
    )

    assert watcher_event.to_inbox_message() == controller_event.to_prompt()
