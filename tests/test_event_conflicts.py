"""A contradictory notification must not crash or silently advance a programme."""

import json
import time
from pathlib import Path
from uuid import uuid4

import pytest

from senpai_agent.advisor import AdvisorEventPump
from senpai_agent.event_conflicts import DeliveryBlocked, EventIdentityConflict
from senpai_agent.inbox import DeliveryState, PersistentInbox
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import CompositeMailbox, ControllerEvent, LocalMailbox
from senpai_agent.state import ConversationBatch
from senpai_agent.supervisor import ProgressLease, WorkerLease
from test_advisor_runtime import STEERING_CONVERSATION_ID, active_steering_turn
from test_controller import Mailbox, Turns, controller


def test_conflicting_sources_block_only_their_conversation_across_restart(
    tmp_path: Path,
):
    affected, healthy = uuid4(), uuid4()
    first = ControllerEvent(
        "training_monitor", "run:gate:0", {"conversation_id": str(affected), "value": 1}
    )
    changed = ControllerEvent(
        first.kind, first.dedupe_key, {**first.payload, "value": 2}
    )
    other = ControllerEvent(
        "training_monitor", "other:status:finished", {"conversation_id": str(healthy)}
    )
    inbox_path = tmp_path / "inbox.sqlite3"
    lease = ProgressLease(tmp_path / "lease.json")

    def batches(events):
        return tuple(
            ConversationBatch(
                owner,
                tuple(e for e in events if e.payload["conversation_id"] == str(owner)),
            )
            for owner in (affected, healthy)
        )

    # A composite poll must not hide the second source's conflicting candidate.
    mailbox = CompositeMailbox(Mailbox([(first, other)]), Mailbox([(changed,)]))
    turns = Turns()
    controller(
        mailbox,
        turns,
        inbox=PersistentInbox(inbox_path),
        conversation_for_events=batches,
        progress=lease,
    ).run(max_cycles=1)
    assert [call[1] for call in turns.calls] == [healthy]
    assert WorkerLease.read(lease.path).delivery_conflicts == 1
    reopened = PersistentInbox(inbox_path)
    (conflict,) = reopened.conflicts()
    assert (conflict.original_body, conflict.candidate_body) == (
        first.to_prompt(),
        changed.to_prompt(),
    )
    assert reopened.next_turn(affected, "continue") is None
    assert reopened.acknowledged_event_keys() == frozenset({other.dedupe_key})

    # Explicit rejection retains evidence and does not accept or acknowledge the candidate.
    reopened.resolve_conflict(
        conflict.identity, "Bad producer repaired; retain the first alert"
    )
    reopened.resolve_conflict(
        conflict.identity, "Bad producer repaired; retain the first alert"
    )
    with pytest.raises(ValueError, match="different resolution"):
        reopened.resolve_conflict(conflict.identity, "overwrite the decision")
    turns = Turns()
    controller(
        Mailbox([(first,)]), turns, inbox=reopened, conversation_for_events=batches
    ).run(max_cycles=1)
    assert [call[1] for call in turns.calls] == [affected]
    assert reopened.conflicts() == ()


@pytest.mark.parametrize(
    "local_collision,pause_fails",
    [(False, False), (True, False), (False, True)],
    ids=["inbox-history", "local-staging", "pause-failure"],
)
def test_active_delivery_conflict_waits_for_tools_without_false_receipt(
    tmp_path: Path, local_collision: bool, pause_fails: bool, monkeypatch
):
    inbox, active, conversation = active_steering_turn(tmp_path)
    if pause_fails:

        def fail_pause():
            raise RuntimeError("SDK pause failed")

        monkeypatch.setattr(conversation, "pause", fail_pause)
    original = LocalEvent(
        kind="child_finished", dedupe_key="child:one", payload={"result": "first"}
    )
    changed = original.model_copy(update={"payload": {"result": "contradictory"}})
    if not local_collision:
        inbox.enqueue(
            STEERING_CONVERSATION_ID, original.dedupe_key, original.to_inbox_message()
        )
    path = tmp_path / "events.sqlite3"
    with LocalEventStore(path) as store:
        if local_collision:
            store.enqueue(original)
            with pytest.raises(EventIdentityConflict):
                store.enqueue(changed)
        else:
            store.enqueue(changed)
        pump = AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            inbox=inbox,
            conversation_id=STEERING_CONVERSATION_ID,
        )
        assert pump.prepare_run()
        pump.run_started()
        # Executing SDK tools hold this lock. A hard interrupt would cancel them.
        conversation.state.acquire()
        locked = True
        try:
            with pytest.raises(
                RuntimeError if pause_fails else DeliveryBlocked,
                match="SDK pause failed" if pause_fails else None,
            ), pump:
                deadline = time.monotonic() + 2
                while not inbox.conflicts() and time.monotonic() < deadline:
                    time.sleep(0.01)
                conflict_recorded = bool(inbox.conflicts())
                tool_was_left_running = not (
                    conversation.interrupted.is_set() or conversation.paused.is_set()
                )
                conversation.state.release()
                locked = False
                if pause_fails:
                    boundary_reached = conversation.interrupted.wait(2)
                else:
                    boundary_reached = conversation.paused.wait(2)
                pump.finish_run()
        finally:
            if locked:
                conversation.state.release()
        assert conflict_recorded
        assert tool_was_left_running
        assert boundary_reached
        assert conversation.interrupted.is_set() is pause_fails
        assert store.acknowledged([original.dedupe_key]) == set()
        assert inbox.turn(active.turn_id).state is DeliveryState.DELIVERED
        with pytest.raises(DeliveryBlocked):
            inbox.record_processed(active.turn_id)
        assert not inbox.processed_turns()
        assert len(PersistentInbox(inbox.path).conflicts()) == 1
        assert len(conversation.messages) == len(active.messages)
    if local_collision:
        # The foreground path also carries the durable conflict when no pump is running.
        events = LocalMailbox(path).poll()
        assert len(events) == 1
        assert events[0].conflict is not None
        assert (
            json.loads(events[0].conflict.candidate_body)["payload"] == changed.payload
        )


def test_global_once_conflict_holds_candidate_conversation(tmp_path: Path):
    original, candidate = uuid4(), uuid4()
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(original, "human:1", "original", once=True)
    with pytest.raises(EventIdentityConflict):
        inbox.enqueue(candidate, "human:1", "changed", once=True)
    assert inbox.conflicts()[0].conversation_id == str(candidate)
    assert inbox.next_turn(original, "read original") is not None
    assert inbox.next_turn(candidate, "read candidate") is None


def test_controller_survives_active_conversation_hold(tmp_path: Path):
    affected, healthy = uuid4(), uuid4()
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(affected, "first", "first")
    inbox.enqueue(healthy, "second", "second")

    class ConflictingTurn(Turns):
        def run(self, *args, conversation_id, **kwargs):
            if conversation_id == affected:
                inbox.record_conflict(
                    EventIdentityConflict(str(affected), "first", "first", "changed")
                )
                inbox.require_delivery_ready(affected)
            return super().run(*args, conversation_id=conversation_id, **kwargs)

    turns = ConflictingTurn()
    controller(Mailbox([]), turns, inbox=inbox).run(max_cycles=2)
    assert [call[1] for call in turns.calls] == [healthy]
    assert len(PersistentInbox(inbox.path).conflicts()) == 1
    assert inbox.acknowledged_event_keys() == frozenset({"second"})


def test_concurrent_local_producers_retain_the_winner_and_conflicting_candidate(
    tmp_path: Path,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "events.sqlite3"
    first = LocalEvent(
        kind="child_finished", dedupe_key="result:1", payload={"result": "first"}
    )
    second = first.model_copy(update={"payload": {"result": "second"}})
    barrier = Barrier(2)

    def enqueue(event):
        with LocalEventStore(path) as store:
            barrier.wait(timeout=2)
            try:
                return store.enqueue(event)
            except EventIdentityConflict:
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(enqueue, (first, second)))
    assert set(outcomes) == {True, "conflict"}
    with LocalEventStore(path) as store:
        (conflict,) = store.conflicts()
        assert {
            json.loads(body)["payload"]["result"]
            for body in (conflict.original_body, conflict.candidate_body)
        } == {"first", "second"}
        assert (
            json.loads(conflict.original_body)["payload"]
            == store.get(first.dedupe_key).payload
        )
        assert store.pending_count() == 1
        assert store.pending() == []


def test_operator_resolution_preserves_history_and_never_creates_a_store(
    tmp_path: Path, monkeypatch, capsys
):
    import sqlite3
    from senpai_agent.event_conflicts import main

    path = tmp_path / "inbox.sqlite3"
    inbox = PersistentInbox(path)
    owner = uuid4()
    inbox.enqueue(owner, "event", "first")
    with pytest.raises(EventIdentityConflict):
        inbox.enqueue(owner, "event", "second")
    (conflict,) = inbox.conflicts()
    monkeypatch.setattr(
        "sys.argv",
        [
            "event_conflicts",
            str(path),
            "reject",
            conflict.identity,
            "--reason",
            "Producer corrected",
        ],
    )
    main()
    monkeypatch.setattr("sys.argv", ["event_conflicts", str(path), "list"])
    main()
    (row,) = json.loads(capsys.readouterr().out)
    assert row["original_body"] == "first" and row["candidate_body"] == "second"
    assert row["resolution"] == "Producer corrected" and row["resolved_at"]
    assert inbox.acknowledged_event_keys() == frozenset()
    missing = tmp_path / "missing.sqlite3"
    monkeypatch.setattr("sys.argv", ["event_conflicts", str(missing), "list"])
    with pytest.raises(sqlite3.OperationalError):
        main()
    assert not missing.exists()


def test_once_only_local_event_keeps_its_first_owner_but_routes_conflicts_to_the_candidate(
    tmp_path: Path,
):
    from senpai_agent.mailbox import LocalStudentMailbox

    original_owner, candidate_owner = str(uuid4()), str(uuid4())
    path = tmp_path / "student-events.sqlite3"
    original = LocalEvent(
        kind="human_issue",
        dedupe_key="human:42",
        payload={
            "message": "Inspect the result",
            "parent_conversation_id": original_owner,
        },
    )
    rerouted = original.model_copy(
        update={
            "payload": {**original.payload, "parent_conversation_id": candidate_owner}
        }
    )
    with LocalEventStore(path) as store:
        assert store.enqueue(original)
        assert not store.enqueue(rerouted)
        assert store.conflicts() == ()
        assert store.get(original.dedupe_key) == original
        changed = rerouted.model_copy(
            update={"payload": {**rerouted.payload, "message": "Different instruction"}}
        )
        with pytest.raises(EventIdentityConflict):
            store.enqueue(changed)
        assert store.conflicts()[0].conversation_id == candidate_owner
    (event,) = LocalStudentMailbox(path).poll()
    assert event.payload["parent_conversation_id"] == candidate_owner
    assert event.conflict is not None


def test_obsolete_monitor_candidate_can_be_withdrawn_without_a_delivery_receipt(
    tmp_path: Path, monkeypatch
):
    import sqlite3
    from senpai_agent.event_conflicts import main
    from senpai_agent.monitor import (
        MonitorEvaluation,
        MonitorMailbox,
        MonitorSignal,
        MonitorStore,
        TrainingMonitorSpec,
    )
    from senpai_agent.training import TrainingState
    from types import SimpleNamespace

    owner = uuid4()
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    path = tmp_path / "monitors.sqlite3"
    spec = TrainingMonitorSpec(training_id="old-run", conversation_id=owner)
    signal = MonitorSignal(
        kind="metric_gate",
        dedupe_key="old-run:gate:0",
        training_id="old-run",
        metric="loss",
        value=2,
        state=TrainingState.RUNNING,
        detail="Later policy fired",
    )
    inbox.enqueue(owner, signal.dedupe_key, "Earlier policy already delivered")
    turn = inbox.next_turn(owner, "read earlier result")
    for message in turn.messages:
        inbox.record_delivered(message.delivery_id, message.body)
    inbox.record_processed(turn.turn_id)
    inbox.acknowledge(turn.turn_id)
    with MonitorStore(path) as store:
        store.register(spec)
        store.record_poll(spec, MonitorEvaluation(signals=(signal,)), None)
        mailbox = MonitorMailbox(SimpleNamespace(poll=lambda: None), store)
        controller(mailbox, Turns(), inbox=inbox, conversation_id=owner).run(
            max_cycles=1
        )
        (conflict,) = inbox.conflicts()
        for _ in range(2):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "event_conflicts",
                    str(path),
                    "withdraw-monitor",
                    signal.dedupe_key,
                    "--reason",
                    "Obsolete gate reviewed; retain historical evidence",
                ],
            )
            main()
        inbox.resolve_conflict(
            conflict.identity, "Source notification withdrawn after review"
        )
        turns = Turns()
        controller(mailbox, turns, inbox=inbox, conversation_id=owner).run(max_cycles=2)
        assert turns.calls == []
        assert mailbox.poll() == ()
        assert inbox.conflicts() == ()
    with sqlite3.connect(path) as database:
        body, handled, reason = database.execute(
            "SELECT signal_json, handled, superseded_reason FROM monitor_signals"
        ).fetchone()
        assert MonitorSignal.model_validate_json(body) == signal
        assert handled == 0
        assert reason == "Obsolete gate reviewed; retain historical evidence"


def test_conflict_count_does_not_wake_another_parent(tmp_path: Path):
    first, second = str(uuid4()), str(uuid4())
    event = LocalEvent(
        kind="child_finished",
        dedupe_key="child:first",
        payload={"parent_conversation_id": first},
    )
    with LocalEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        with pytest.raises(EventIdentityConflict):
            store.enqueue(event.model_copy(update={"kind": "changed"}))
        assert store.pending_count(first) == 1
        assert store.pending_count(second) == 0
        assert store.pending_count() == 1


def test_failed_conflict_persistence_remains_fatal(tmp_path: Path):
    import sqlite3

    path = tmp_path / "inbox.sqlite3"
    owner = uuid4()
    inbox = PersistentInbox(path)
    inbox.enqueue(owner, "event", "accepted")
    with sqlite3.connect(path) as database:
        database.execute("DROP TABLE event_conflicts")
    with pytest.raises(sqlite3.OperationalError, match="event_conflicts"):
        inbox.enqueue(owner, "event", "contradiction")
    with sqlite3.connect(path) as database:
        assert database.execute("SELECT body FROM inbox_messages").fetchall() == [
            ("accepted",)
        ]
