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
from sqlite_test_support import assert_repeated_concurrent_first_open

from senpai_agent.advisor import (
    AdvisorEvent,
    AdvisorEventPump,
    AdvisorEventStore,
    advisor_conversation_id,
    advisor_main,
    deliver_pending_events,
)
from senpai_agent.delegation import AgentStatusAction, AgentStatusObservation
from senpai_agent.inbox import PersistentInbox
from senpai_agent.mailbox import ControllerEvent


class ConversationStateStub:
    def __init__(
        self,
        events=(),
        execution_status=ConversationExecutionStatus.RUNNING,
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


def test_advisor_conversation_id_is_persisted(tmp_path: Path):
    first = advisor_conversation_id(tmp_path)
    second = advisor_conversation_id(tmp_path)

    assert first == second
    assert (tmp_path / "advisor-conversation-id").read_text() == f"{first}\n"


def test_event_store_allows_bounded_concurrent_first_open(tmp_path: Path):
    """
    Requirement: controller, watcher, delegation, and role-control processes may
    first discover the event queue concurrently without losing the queue.
    Interface: AdvisorEventStore construction and its pending-event view.
    """

    assert_repeated_concurrent_first_open(
        lambda attempt: AdvisorEventStore(
            tmp_path / f"advisor-events-{attempt}.sqlite3"
        ),
        attempts=100,
    )

    database = tmp_path / "advisor-events-reopen.sqlite3"
    with AdvisorEventStore(database) as reopened:
        assert reopened.pending() == []


def test_event_store_deduplicates_and_survives_reopen(tmp_path: Path):
    database = tmp_path / "advisor-events.sqlite3"
    event = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:3467:abc123",
        payload={"pr": 3467, "head_sha": "abc123"},
    )

    with AdvisorEventStore(database) as store:
        assert store.enqueue(event) is True
        assert store.pending_count() == 1
        assert store.enqueue(event) is False
        with pytest.raises(RuntimeError, match="reused with a different payload"):
            store.enqueue(event.model_copy(update={"payload": {"pr": 999}}))

    with AdvisorEventStore(database) as reopened:
        pending = reopened.pending()
        assert pending == [event]
        reopened.acknowledge(event.dedupe_key)
        assert reopened.pending_count() == 0
        assert reopened.pending() == []


def test_event_message_renders_observation_time_and_structured_payload():
    event = AdvisorEvent(
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
    first = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:11:ddd",
        payload={"pr": 11},
    )
    second = AdvisorEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()
            self.messages: list[str] = []

        def send_message(self, message: str, sender: str | None = None) -> None:
            if self.messages:
                raise RuntimeError("conversation unavailable")
            self.messages.append(message)

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
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
    event = AdvisorEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub([pending_tool_action()])
            self.messages: list[str] = []

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)
            self.state.append(SimpleNamespace(sender=sender))

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        conversation = Conversation()

        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            assert conversation.state.inspected.wait(1)

        assert conversation.messages == []
        assert store.pending() == [event]


def test_event_pump_delivers_queued_event_after_the_tool_boundary_is_safe(
    tmp_path: Path,
):
    event = AdvisorEvent(
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

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)
            self.state.append(SimpleNamespace(sender=sender))
            self.received.set()

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        conversation = Conversation()
        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            assert conversation.state.inspected.wait(1)
            assert conversation.messages == []
            assert store.pending() == [event]

            conversation.state.append(completed_tool_action(action))
            assert conversation.received.wait(1)
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED

        assert conversation.messages == [event.to_user_message()]
        assert store.pending() == []


def test_event_pump_injects_new_events_while_conversation_is_running(
    tmp_path: Path,
):
    event = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:12:eee",
        payload={"pr": 12},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()
            self.messages: list[str] = []

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)
            self.state.append(SimpleNamespace(sender=sender))

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        conversation = Conversation()
        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            store.enqueue(event)
            deadline = time.monotonic() + 1
            while not conversation.messages and time.monotonic() < deadline:
                time.sleep(0.01)
            assert store.pending() == [event]
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED

        assert conversation.messages == [event.to_user_message()]
        assert store.pending() == []


def test_event_pump_queues_into_the_controller_inbox_without_mid_turn_injection(
    tmp_path: Path,
):
    """
    Requirement: controller and event-pump messages use one durable inbox.
    Interface: AdvisorEventPump, PersistentInbox, and the active conversation.
    """
    event = AdvisorEvent(
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

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)
            self.state.append(SimpleNamespace(sender=sender))
            self.received.set()

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
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


def test_non_finished_turn_leaves_delivered_child_result_pending(
    tmp_path: Path,
):
    event = AdvisorEvent(
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

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)
            self.state.append(SimpleNamespace(sender=sender))
            self.received.set()

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
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
    first = AdvisorEvent(
        kind="agent_result",
        dedupe_key="agent_result:first",
        payload={"parent_conversation_id": first_parent},
    )
    second = AdvisorEvent(
        kind="agent_result",
        dedupe_key="agent_result:second",
        payload={"parent_conversation_id": second_parent},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()
            self.messages: list[str] = []

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)
            self.state.append(SimpleNamespace(sender=sender))

    with AdvisorEventStore(tmp_path / "student-events.sqlite3") as store:
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
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED

        assert conversation.messages == [first.to_user_message()]
        assert store.pending() == [second]


def test_event_pump_leaves_late_event_pending_after_conversation_finishes(
    tmp_path: Path,
):
    event = AdvisorEvent(
        kind="submission_slot_free",
        dedupe_key="submission_slot_free:cedar",
        payload={"campaign": "cedar"},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub(
                execution_status=ConversationExecutionStatus.FINISHED
            )
            self.messages: list[str] = []

        def send_message(self, message: str, sender: str | None = None) -> None:
            self.messages.append(message)

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        conversation = Conversation()

        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            time.sleep(0.03)

        assert conversation.messages == []
        assert store.pending() == [event]


def test_event_pump_surfaces_delivery_failure_and_leaves_event_pending(
    tmp_path: Path,
):
    event = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:13:fff",
        payload={"pr": 13},
    )

    class Conversation:
        def __init__(self):
            self.state = ConversationStateStub()

        def send_message(
            self,
            _message: str,
            sender: str | None = None,
        ) -> None:
            raise RuntimeError("conversation rejected event")

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        with (
            pytest.raises(RuntimeError, match="conversation rejected event"),
            AdvisorEventPump(store, Conversation(), poll_interval=0.01),
        ):
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)

        assert store.pending() == [event]


def test_advisor_cli_reports_the_pending_event_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    with AdvisorEventStore(tmp_path / "advisor-events.sqlite3") as store:
        store.enqueue(
            AdvisorEvent(
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
    watcher_event = AdvisorEvent(
        kind=controller_event.kind,
        dedupe_key=controller_event.dedupe_key,
        payload={**payload, "parent_conversation_id": "conversation-17"},
    )

    assert watcher_event.to_inbox_message() == controller_event.to_prompt()
