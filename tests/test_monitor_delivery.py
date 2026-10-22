from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.inbox import PersistentInbox
from senpai_agent.monitor import (
    MetricGate,
    MetricSample,
    MonitorMailbox,
    MonitorStore,
    TrainingMonitorEngine,
    TrainingMonitorSpec,
)
from senpai_agent.training import TrainingResult, TrainingState


@pytest.mark.parametrize("gates", [(), (MetricGate(operator="gte", threshold=0),)])
def test_policy_change_during_handling_survives_restart(tmp_path: Path, gates):
    """A new rule cannot replace or acknowledge the alert that woke its turn."""
    conversation = uuid4()
    spec = TrainingMonitorSpec(
        training_id="run-1",
        conversation_id=conversation,
        metric="loss",
        gates=gates,
        stale_after_seconds=1,
    )
    training = SimpleNamespace(
        get_training_status=lambda _: TrainingResult(
            training_id="run-1",
            state=TrainingState.RUNNING,
            exit_code=None,
            elapsed_seconds=10,
            log_path="run.log",
            wandb_run_ids=("run-1",),
        )
    )
    metrics = SimpleNamespace(
        latest=lambda *_: MetricSample(
            value=1,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    database = tmp_path / "monitors.sqlite3"
    inbox_path = tmp_path / "inbox.sqlite3"
    inbox = PersistentInbox(inbox_path)
    with MonitorStore(database) as store:
        store.register(spec)
        mailbox = MonitorMailbox(TrainingMonitorEngine(store, training, metrics), store)
        original = mailbox.poll()
        for event in original:
            inbox.enqueue(conversation, event.dedupe_key, event.to_prompt())
        turn = inbox.next_turn(conversation, "check training")
        for message in turn.messages:
            inbox.record_delivered(message.delivery_id, message.body)
        store.register(spec.model_copy(update={"metric": "validation"}))
        inbox.record_processed(turn.turn_id)
        mailbox.acknowledge(turn.acknowledgement_keys)
        inbox.acknowledge(turn.turn_id)

    inbox = PersistentInbox(inbox_path)
    with MonitorStore(database) as store:
        mailbox = MonitorMailbox(TrainingMonitorEngine(store, training, metrics), store)
        changed = mailbox.poll()
        assert changed
        for event in changed:
            inbox.enqueue(conversation, event.dedupe_key, event.to_prompt())
        assert set(e.dedupe_key for e in original).isdisjoint(
            e.dedupe_key for e in changed
        )
        assert [m.body for m in inbox.turn(turn.turn_id).events] == [
            e.to_prompt() for e in original
        ]
        pending = inbox.next_turn(conversation, "check the new rule")
        assert {m.event_key for m in pending.events} == {e.dedupe_key for e in changed}


def test_terminal_capture_survives_a_concurrent_policy_change(tmp_path: Path):
    spec = TrainingMonitorSpec(
        training_id="run-1", conversation_id=uuid4(), metric="loss"
    )
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)

        def terminal_status(_):
            store.register(spec.model_copy(update={"metric": "validation"}))
            return TrainingResult(
                training_id="run-1",
                state=TrainingState.FINISHED,
                exit_code=0,
                elapsed_seconds=10,
                log_path="run.log",
            )

        engine = TrainingMonitorEngine(
            store,
            SimpleNamespace(get_training_status=terminal_status),
            SimpleNamespace(latest=lambda *_: pytest.fail("terminal metric read")),
        )
        signals = engine.poll()
        assert len(signals) == 1
        assert signals[0].state is TrainingState.FINISHED
        assert store.active() == []
        assert store.spec("run-1").policy_revision == 2
        assert store.pending_events()[0].dedupe_key == "run-1:status:finished"
        assert engine.poll() == ()


def test_existing_monitor_notifications_keep_their_rendered_bytes(tmp_path: Path):
    import sqlite3

    from senpai_agent.mailbox import ControllerEvent
    from senpai_agent.monitor import MonitorSignal

    spec = TrainingMonitorSpec(
        training_id="old-run",
        conversation_id=uuid4(),
        metric="loss",
        gates=(
            MetricGate(operator="gte", threshold=0),
            MetricGate(operator="lte", threshold=2),
        ),
    )
    signal = MonitorSignal(
        kind="metric_gate",
        dedupe_key="old-run:gate:0",
        training_id="old-run",
        metric="loss",
        value=1,
        state=TrainingState.RUNNING,
        detail="Original threshold crossed.",
    )
    original = ControllerEvent(
        "training_monitor",
        signal.dedupe_key,
        {
            "conversation_id": str(spec.conversation_id),
            "training_id": signal.training_id,
            "summary": signal.detail,
            "reason": "The registered monitor policy emitted this signal.",
            "signal": signal.model_dump(mode="json"),
        },
    )
    path = tmp_path / "monitors.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE monitors (training_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1, previous_sample_json TEXT, next_poll_at REAL DEFAULT 0);
            CREATE TABLE monitor_signals (dedupe_key TEXT PRIMARY KEY, training_id TEXT NOT NULL,
                signal_json TEXT NOT NULL, handled INTEGER NOT NULL DEFAULT 0);
        """)
        db.execute(
            "INSERT INTO monitors (training_id, spec_json) VALUES (?, ?)",
            (spec.training_id, spec.model_dump_json(exclude={"policy_revision"})),
        )
        db.execute(
            "INSERT INTO monitor_signals (dedupe_key, training_id, signal_json) VALUES (?, ?, ?)",
            (signal.dedupe_key, signal.training_id, signal.model_dump_json()),
        )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(
        spec.conversation_id,
        "old-run:gate:1",
        "An earlier policy deleted from the monitor database",
    )
    with MonitorStore(path) as store:
        assert store.pending_events()[0].to_prompt() == original.to_prompt()
        training = SimpleNamespace(
            get_training_status=lambda _: TrainingResult(
                training_id="old-run",
                state=TrainingState.RUNNING,
                exit_code=None,
                elapsed_seconds=1,
                log_path="old.log",
                wandb_run_ids=("old-run",),
            )
        )
        metrics = SimpleNamespace(
            latest=lambda *_: MetricSample(value=1, observed_at=datetime.now(UTC))
        )
        signals = TrainingMonitorEngine(store, training, metrics).poll()
        assert [signal.dedupe_key for signal in signals] == ["old-run:r1:gate:1"]
        event = store.pending_events()[1]
        inbox.enqueue(spec.conversation_id, event.dedupe_key, event.to_prompt())
        assert inbox.conflicts() == ()
        store.acknowledge(event.dedupe_key)
        store.register(spec.model_copy(update={"metric": "validation"}))
        assert store.pending_events()[0].to_prompt() == original.to_prompt()
        assert store.connection.execute(
            "SELECT observed_at FROM monitor_signals"
        ).fetchone() == (None,)
        with pytest.raises(PermissionError):
            store.register(spec.model_copy(update={"conversation_id": uuid4()}))
        store.acknowledge(signal.dedupe_key)
    with MonitorStore(path) as store:
        assert store.pending_events() == ()
        assert store.spec(spec.training_id).policy_revision == 2
