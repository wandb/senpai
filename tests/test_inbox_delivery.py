import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from senpai_agent.inbox import (
    DeliveryState,
    PersistentInbox,
    deliver_turn_messages,
    turn_has_finished_response,
)


CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000017")


class Conversation:
    def __init__(self, events=()):
        self.events = list(events)
        self.sent: list[tuple[str, str]] = []
        self.state = SimpleNamespace(active_branch=lambda: list(self.events))

    def send_message(self, message: str, sender: str | None = None) -> None:
        assert sender is not None
        self.sent.append((message, sender))
        self.events.append(SimpleNamespace(message=message, sender=sender))


def event_message(index: int, size: int = 0) -> str:
    return f"event-{index}:" + ("x" * size)


def delivery_sender(delivery_id: str) -> str:
    return "senpai-delivery:" + hashlib.sha256(delivery_id.encode()).hexdigest()


def test_delivery_state_is_durable_and_monotonic_across_restarts(tmp_path: Path):
    """
    Requirement: each model-visible message moves pending -> delivered -> processed.
    Interface: the persistent inbox reopened by a restarted Senpai process.
    """
    path = tmp_path / "inbox.sqlite3"
    inbox = PersistentInbox(path)
    assert inbox.enqueue(CONVERSATION_ID, "event:1", "first event") is True
    turn = inbox.next_turn(CONVERSATION_ID, "controller prompt")
    assert turn is not None
    assert [message.state for message in turn.messages] == [
        DeliveryState.PENDING,
        DeliveryState.PENDING,
    ]

    event = turn.events[0]
    inbox.record_delivered(event.delivery_id, event.body)
    reopened = PersistentInbox(path)
    resumed = reopened.next_turn(CONVERSATION_ID, "different retry prompt")
    assert resumed is not None
    assert resumed.turn_id == turn.turn_id
    assert resumed.prompt.body == "controller prompt"
    assert resumed.events[0].state is DeliveryState.DELIVERED

    with pytest.raises(ValueError, match="cannot move backwards"):
        reopened.record_pending(event.delivery_id)

    reopened.record_delivered(resumed.prompt.delivery_id, resumed.prompt.body)
    reopened.record_processed(turn.turn_id)
    processed = PersistentInbox(path).turn(turn.turn_id)
    assert processed.state is DeliveryState.PROCESSED
    assert all(
        message.state is DeliveryState.PROCESSED for message in processed.messages
    )


def test_stable_sender_verifies_payload_after_crash_between_append_and_receipt(
    tmp_path: Path,
):
    """
    Requirement: recovery recognizes an already-appended message and verifies it.
    Interface: the sender and body visible on the active OpenHands branch.
    """
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(CONVERSATION_ID, "event:1", "canonical event")
    turn = inbox.next_turn(CONVERSATION_ID, "canonical prompt")
    assert turn is not None

    appended_before_crash = [
        SimpleNamespace(message=message.body, sender=message.sender)
        for message in turn.messages
    ]
    conversation = Conversation(appended_before_crash)
    deliver_turn_messages(conversation, PersistentInbox(inbox.path), turn.turn_id)

    assert conversation.sent == []
    assert PersistentInbox(inbox.path).turn(turn.turn_id).state is (
        DeliveryState.DELIVERED
    )

    tampered = Conversation(
        [SimpleNamespace(message="changed payload", sender=turn.events[0].sender)]
    )
    with pytest.raises(RuntimeError, match="payload mismatch"):
        deliver_turn_messages(tampered, PersistentInbox(inbox.path), turn.turn_id)


def test_event_identity_cannot_hide_a_changed_payload(tmp_path: Path):
    """
    Requirement: one event identity always denotes one canonical payload.
    Interface: repeated enqueue calls against the persistent inbox.
    """
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(CONVERSATION_ID, "event:1", "canonical event")

    with pytest.raises(RuntimeError, match="reused with a different payload"):
        inbox.enqueue(CONVERSATION_ID, "event:1", "changed event")

    turn = inbox.next_turn(CONVERSATION_ID, "controller prompt")
    assert turn is not None
    deliver_turn_messages(Conversation(), inbox, turn.turn_id)
    inbox.record_processed(turn.turn_id)
    inbox.acknowledge(turn.turn_id)

    with pytest.raises(RuntimeError, match="reused with a different payload"):
        inbox.enqueue(CONVERSATION_ID, "event:1", "changed after acknowledgement")


def test_crash_before_append_reuses_turn_and_appends_each_message_once(tmp_path: Path):
    """
    Requirement: a crash before append retries the same durable delivery normally.
    Interface: persistent turn identity and model-visible conversation messages.
    """
    path = tmp_path / "inbox.sqlite3"
    inbox = PersistentInbox(path)
    inbox.enqueue(CONVERSATION_ID, "event:1", "first event")
    before_crash = inbox.next_turn(CONVERSATION_ID, "controller prompt")
    assert before_crash is not None

    after_restart = PersistentInbox(path).next_turn(
        CONVERSATION_ID,
        "new prompt that must not replace the active turn",
    )
    assert after_restart is not None
    assert after_restart.turn_id == before_crash.turn_id
    assert [message.sender for message in after_restart.messages] == [
        message.sender for message in before_crash.messages
    ]

    conversation = Conversation()
    deliver_turn_messages(conversation, PersistentInbox(path), after_restart.turn_id)
    deliver_turn_messages(conversation, PersistentInbox(path), after_restart.turn_id)
    assert [message for message, _sender in conversation.sent] == [
        "controller prompt",
        "first event",
    ]


def test_fifo_drain_is_bounded_by_event_count_and_bytes(tmp_path: Path):
    """
    Requirement: each inference turn receives at most 16 events or 64 KiB.
    Interface: events returned by the persistent inbox in FIFO order.
    """
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    for index in range(20):
        inbox.enqueue(CONVERSATION_ID, f"event:{index}", event_message(index))

    first = inbox.next_turn(CONVERSATION_ID, "first prompt")
    assert first is not None
    assert [event.event_key for event in first.events] == [
        f"event:{index}" for index in range(16)
    ]
    deliver_turn_messages(Conversation(), inbox, first.turn_id)
    inbox.record_processed(first.turn_id)
    inbox.acknowledge(first.turn_id)

    second = inbox.next_turn(CONVERSATION_ID, "second prompt")
    assert second is not None
    assert [event.event_key for event in second.events] == [
        f"event:{index}" for index in range(16, 20)
    ]

    large_conversation = UUID("00000000-0000-0000-0000-000000000064")
    inbox.enqueue(large_conversation, "oversized", event_message(0, 70 * 1024))
    inbox.enqueue(large_conversation, "next", "small")
    oversized = inbox.next_turn(large_conversation, "large prompt")
    assert oversized is not None
    assert [event.event_key for event in oversized.events] == ["oversized"]


def test_new_events_wait_behind_an_unresolved_delivery(tmp_path: Path):
    """
    Requirement: retries never admit newly observed events into an unresolved turn.
    Interface: next_turn while a delivered turn has not been processed.
    """
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(CONVERSATION_ID, "event:first", "first")
    first = inbox.next_turn(CONVERSATION_ID, "first prompt")
    assert first is not None
    deliver_turn_messages(Conversation(), inbox, first.turn_id)

    inbox.enqueue(CONVERSATION_ID, "event:later", "later")
    retry = inbox.next_turn(CONVERSATION_ID, "retry prompt")
    assert retry is not None
    assert retry.turn_id == first.turn_id
    assert [event.event_key for event in retry.events] == ["event:first"]
    assert inbox.pending_count(CONVERSATION_ID) == 1


def test_context_reset_preserves_old_turn_and_requeues_one_canonical_copy(
    tmp_path: Path,
):
    """
    Requirement: context reset preserves audit history and creates one recovery copy.
    Interface: reset_turn plus the active and historical persistent turns.
    """
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(CONVERSATION_ID, "event:1", "canonical event")
    old = inbox.next_turn(CONVERSATION_ID, "old branch prompt")
    assert old is not None
    deliver_turn_messages(Conversation(), inbox, old.turn_id)

    recovery = inbox.reset_turn(old.turn_id, "fresh branch prompt")
    repeated = PersistentInbox(inbox.path).reset_turn(
        old.turn_id,
        "a second reset request must not create another copy",
    )

    assert recovery.turn_id == repeated.turn_id
    assert recovery.turn_id != old.turn_id
    assert recovery.prompt.body == "fresh branch prompt"
    assert [event.body for event in recovery.events] == ["canonical event"]
    assert recovery.events[0].delivery_id != old.events[0].delivery_id
    assert inbox.turn(old.turn_id).superseded_by == recovery.turn_id
    assert inbox.turn(old.turn_id).state is DeliveryState.DELIVERED


def test_deliberate_reminder_gets_a_fresh_delivery_identity(tmp_path: Path):
    """
    Requirement: a later reminder is a new message, not a replay of an old attempt.
    Interface: enqueueing the same event after processing and acknowledgement.
    """
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(CONVERSATION_ID, "event:1", "still actionable")
    first = inbox.next_turn(CONVERSATION_ID, "first prompt")
    assert first is not None
    deliver_turn_messages(Conversation(), inbox, first.turn_id)
    inbox.record_processed(first.turn_id)
    inbox.acknowledge(first.turn_id)

    assert inbox.enqueue(CONVERSATION_ID, "event:1", "still actionable") is True
    reminder = inbox.next_turn(CONVERSATION_ID, "reminder prompt")
    assert reminder is not None
    assert reminder.events[0].delivery_id != first.events[0].delivery_id
    assert reminder.events[0].sender != first.events[0].sender


def test_108_events_and_four_failed_resumes_leave_one_visible_copy_per_event(
    tmp_path: Path,
):
    """
    Requirement: repeated failed inference cannot multiply model-visible events.
    Interface: persistent bounded turns delivered into one OpenHands conversation.
    """
    path = tmp_path / "inbox.sqlite3"
    inbox = PersistentInbox(path)
    for index in range(108):
        inbox.enqueue(CONVERSATION_ID, f"event:{index}", event_message(index))

    conversation = Conversation()
    first = inbox.next_turn(CONVERSATION_ID, "prompt-0")
    assert first is not None
    for _failure in range(4):
        deliver_turn_messages(conversation, PersistentInbox(path), first.turn_id)
        resumed = PersistentInbox(path).next_turn(CONVERSATION_ID, "changed retry")
        assert resumed is not None and resumed.turn_id == first.turn_id

    deliver_turn_messages(conversation, inbox, first.turn_id)
    inbox.record_processed(first.turn_id)
    inbox.acknowledge(first.turn_id)

    turn_number = 1
    while inbox.pending_count(CONVERSATION_ID):
        turn = inbox.next_turn(CONVERSATION_ID, f"prompt-{turn_number}")
        assert turn is not None
        deliver_turn_messages(conversation, inbox, turn.turn_id)
        inbox.record_processed(turn.turn_id)
        inbox.acknowledge(turn.turn_id)
        turn_number += 1

    visible = Counter(message for message, _sender in conversation.sent)
    assert all(visible[event_message(index)] == 1 for index in range(108))


@pytest.mark.parametrize(
    "prompt_identity",
    (
        "initial:full historical prompt",
        "system-context:historical system context",
        "turn:00000000-0000-0000-0000-000000000117",
    ),
)
def test_legacy_pr3472_delivery_is_adopted_without_replay(
    tmp_path: Path,
    prompt_identity: str,
):
    """All old prompt modes and verbose pump payloads survive the cutover."""
    event_key = "review_ready:17:abc"
    event_body = "canonical compact event"
    historical_event_body = "# Senpai event: review_ready\n\nverbose historical body"
    legacy_event_id = "00000000-0000-0000-0000-000000000117"
    legacy_path = tmp_path / "pending-message-deliveries.json"
    legacy_value = {
        str(CONVERSATION_ID): {event_key: legacy_event_id},
    }
    legacy_path.write_text(json.dumps(legacy_value), encoding="utf-8")

    inbox = PersistentInbox(
        tmp_path / "inbox.sqlite3",
        legacy_path=legacy_path,
    )
    inbox.enqueue(CONVERSATION_ID, event_key, event_body)
    turn = inbox.next_turn(
        CONVERSATION_ID,
        "new controller prompt",
        legacy_prompt_identity=prompt_identity,
    )
    assert turn is not None

    legacy_prompt_id = (
        "controller-prompt:"
        + hashlib.sha256(prompt_identity.encode()).hexdigest()
    )

    branch = [
        SimpleNamespace(
            message="old controller prompt",
            sender=delivery_sender(legacy_prompt_id),
        ),
        SimpleNamespace(
            message=historical_event_body,
            sender=delivery_sender(legacy_event_id),
        ),
        SimpleNamespace(message="completed answer", sender="agent"),
    ]
    conversation = Conversation(branch)

    recovered = deliver_turn_messages(conversation, inbox, turn.turn_id)

    assert conversation.sent == []
    assert recovered.prompt.delivery_id == legacy_prompt_id
    assert recovered.prompt.body == "old controller prompt"
    assert recovered.events[0].body == historical_event_body
    assert turn_has_finished_response(conversation, recovered)
    inbox.record_processed(turn.turn_id)
    inbox.acknowledge(turn.turn_id)

    assert inbox.enqueue(CONVERSATION_ID, event_key, event_body) is True
    reminder = inbox.next_turn(CONVERSATION_ID, "later reminder")
    assert reminder is not None
    assert reminder.events[0].delivery_id != legacy_event_id
    assert json.loads(legacy_path.read_text(encoding="utf-8")) == legacy_value


def test_reset_preserves_legacy_provenance_for_a_later_compact_reminder(
    tmp_path: Path,
):
    event_key = "idle_student:Fern"
    compact_body = "compact event"
    legacy_id = str(UUID(int=119))
    legacy_path = tmp_path / "pending-message-deliveries.json"
    legacy_path.write_text(
        json.dumps({str(CONVERSATION_ID): {event_key: legacy_id}}),
        encoding="utf-8",
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3", legacy_path=legacy_path)
    inbox.enqueue(CONVERSATION_ID, event_key, compact_body)
    original = inbox.next_turn(CONVERSATION_ID, "new prompt")
    assert original is not None
    branch = [
        SimpleNamespace(
            message="old prompt",
            sender=delivery_sender(original.legacy_prompt_delivery_id),
        ),
        SimpleNamespace(
            message="verbose legacy event",
            sender=delivery_sender(legacy_id),
        ),
    ]
    deliver_turn_messages(Conversation(branch), inbox, original.turn_id)

    recovery = inbox.reset_turn(original.turn_id, "recovery prompt")
    inbox.record_context_reset(recovery.turn_id)
    deliver_turn_messages(Conversation(), inbox, recovery.turn_id)
    inbox.record_processed(recovery.turn_id)
    inbox.acknowledge(recovery.turn_id)

    assert inbox.enqueue(CONVERSATION_ID, event_key, compact_body) is True


def test_adopted_legacy_prompt_is_stable_when_migration_reenters(tmp_path: Path):
    """Retrying a migrated turn must verify the same nearest historical prompt."""
    event_key = "job:finished"
    event_id = str(UUID(int=118))
    legacy_path = tmp_path / "pending-message-deliveries.json"
    legacy_path.write_text(
        json.dumps({str(CONVERSATION_ID): {event_key: event_id}}),
        encoding="utf-8",
    )
    inbox_path = tmp_path / "inbox.sqlite3"
    inbox = PersistentInbox(
        inbox_path,
        legacy_path=legacy_path,
    )
    inbox.enqueue(CONVERSATION_ID, event_key, "new compact event")
    turn = inbox.next_turn(
        CONVERSATION_ID,
        "new controller prompt",
        legacy_prompt_identity="initial:new-controller-prompt",
    )
    assert turn is not None

    current_prompt_id = "historical-current-prompt"
    branch = [
        SimpleNamespace(
            message="older controller prompt",
            sender=delivery_sender("older-controller-prompt"),
        ),
        SimpleNamespace(
            message="current controller prompt",
            sender=delivery_sender(current_prompt_id),
        ),
        SimpleNamespace(
            message="historical event body",
            sender=delivery_sender(event_id),
        ),
        SimpleNamespace(message="completed answer", sender="agent"),
    ]
    conversation = Conversation(branch)

    first = inbox.prepare_legacy_turn(turn.turn_id, branch)
    assert first.state is DeliveryState.DELIVERED
    assert all(
        message.state is DeliveryState.DELIVERED for message in first.messages
    )
    inbox.close()
    reopened = PersistentInbox(inbox_path, legacy_path=legacy_path)
    second = deliver_turn_messages(conversation, reopened, turn.turn_id)

    assert conversation.sent == []
    assert first.prompt.delivery_id == second.prompt.delivery_id == (
        "legacy-prompt:"
        + delivery_sender(current_prompt_id).removeprefix("senpai-delivery:")
    )
    assert second.prompt.body == "current controller prompt"
    assert second.state is DeliveryState.DELIVERED
    assert turn_has_finished_response(conversation, second)

    tampered = Conversation(
        [
            branch[0],
            SimpleNamespace(
                message="tampered current controller prompt",
                sender=delivery_sender(current_prompt_id),
            ),
            *branch[2:],
        ]
    )
    with pytest.raises(RuntimeError, match="payload mismatch"):
        deliver_turn_messages(tampered, reopened, turn.turn_id)

    tampered_event = Conversation(
        [
            *branch[:2],
            SimpleNamespace(
                message="tampered historical event body",
                sender=delivery_sender(event_id),
            ),
            branch[3],
        ]
    )
    with pytest.raises(RuntimeError, match="payload mismatch"):
        deliver_turn_messages(tampered_event, reopened, turn.turn_id)


def test_unappended_legacy_backlog_obeys_normal_count_and_byte_limits(
    tmp_path: Path,
):
    """Migrating the old JSON ledger cannot recreate one oversized injection."""
    legacy_path = tmp_path / "pending-message-deliveries.json"
    legacy_path.write_text(
        json.dumps(
            {
                str(CONVERSATION_ID): {
                    f"event:{index}": str(UUID(int=1000 + index))
                    for index in range(20)
                }
            }
        ),
        encoding="utf-8",
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3", legacy_path=legacy_path)
    for index in range(20):
        inbox.enqueue(CONVERSATION_ID, f"event:{index}", event_message(index))

    turn = inbox.next_turn(CONVERSATION_ID, "bounded migration prompt")
    assert turn is not None
    assert len(turn.events) == 16

    conversation = Conversation()
    deliver_turn_messages(conversation, inbox, turn.turn_id)
    assert len(conversation.sent) == 17
    assert inbox.pending_count(CONVERSATION_ID) == 4

    byte_conversation = UUID(int=65)
    byte_ledger = tmp_path / "byte-pending-message-deliveries.json"
    byte_ledger.write_text(
        json.dumps(
            {
                str(byte_conversation): {
                    "large": str(UUID(int=2000)),
                    "next": str(UUID(int=2001)),
                }
            }
        ),
        encoding="utf-8",
    )
    byte_inbox = PersistentInbox(
        tmp_path / "byte-inbox.sqlite3",
        legacy_path=byte_ledger,
    )
    byte_inbox.enqueue(
        byte_conversation,
        "large",
        event_message(0, 70 * 1024),
    )
    byte_inbox.enqueue(byte_conversation, "next", "small")
    oversized = byte_inbox.next_turn(byte_conversation, "byte limit")
    assert oversized is not None
    assert [event.event_key for event in oversized.events] == ["large"]


def test_already_visible_legacy_batch_is_absorbed_without_replay(tmp_path: Path):
    """Old model-visible events are reconciled together without a new injection."""
    legacy_path = tmp_path / "pending-message-deliveries.json"
    deliveries = {
        f"event:{index}": str(UUID(int=3000 + index)) for index in range(20)
    }
    legacy_path.write_text(
        json.dumps({str(CONVERSATION_ID): deliveries}),
        encoding="utf-8",
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3", legacy_path=legacy_path)
    for index in range(20):
        inbox.enqueue(CONVERSATION_ID, f"event:{index}", event_message(index))
    selected = inbox.next_turn(CONVERSATION_ID, "new bounded prompt")
    assert selected is not None
    assert len(selected.events) == 16

    old_prompt_id = "controller-prompt:historical-batch"
    branch = [
        SimpleNamespace(
            message="old batch prompt",
            sender=delivery_sender(old_prompt_id),
        ),
        *(
            SimpleNamespace(
                message=f"historical-{index}",
                sender=delivery_sender(deliveries[f"event:{index}"]),
            )
            for index in range(20)
        ),
        SimpleNamespace(message="completed answer", sender="agent"),
    ]
    conversation = Conversation(branch)

    recovered = deliver_turn_messages(conversation, inbox, selected.turn_id)

    assert conversation.sent == []
    assert len(recovered.events) == 20
    assert [event.body for event in recovered.events] == [
        f"historical-{index}" for index in range(20)
    ]
    assert inbox.pending_count(CONVERSATION_ID) == 0
    assert turn_has_finished_response(conversation, recovered)
