import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricGate,
    MetricSample,
    MonitorEvaluation,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorSpec,
    evaluate_monitor,
)
from senpai_agent.training import TrainingState


NOW = datetime(2026, 7, 30, tzinfo=UTC)


def spec(**changes):
    values = {
        "training_id": "train-1",
        "conversation_id": uuid4(),
        "metric": "val/loss",
        "direction": "min",
        "gates": (MetricGate(operator="lte", threshold=0.2),),
        "registered_at": NOW,
    }
    values.update(changes)
    return TrainingMonitorSpec(**values)


SIGNAL = MonitorSignal(
    kind="metric_gate",
    dedupe_key="train-1:gate:0",
    training_id="train-1",
    metric="val/loss",
    value=0.19,
    state=TrainingState.RUNNING,
    detail="val/loss crossed the threshold.",
)


def test_registration_and_stop_hook_marker_survive_reopen(tmp_path: Path):
    monitor = spec()
    database = tmp_path / "monitors.sqlite3"

    with MonitorStore(database) as store:
        assert store.register(monitor) is True
        marker = store.marker_dir / "train-1.json"
        assert TrainingMonitorSpec.model_validate_json(marker.read_text()) == monitor

    with MonitorStore(database) as reopened:
        assert reopened.active() == [monitor]


def test_next_poll_deadline_tracks_active_monitor_state(tmp_path: Path):
    monitor = spec(poll_interval_seconds=60)

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        assert store.seconds_until_next_poll(NOW) == 0

        store.record_poll(monitor, MonitorEvaluation(), None, now=NOW)
        assert store.seconds_until_next_poll(NOW + timedelta(seconds=15)) == 45

        store.complete(monitor.training_id)
        assert store.seconds_until_next_poll(NOW) is None


def test_same_policy_reregistration_preserves_derived_state(tmp_path: Path):
    monitor = spec()
    repeated = monitor.model_copy(update={"registered_at": NOW + timedelta(minutes=5)})
    sample = MetricSample(value=0.19, observed_at=NOW)

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        store.record_poll(
            monitor,
            MonitorEvaluation(signals=(SIGNAL,)),
            sample,
            now=NOW,
        )

        assert store.register(repeated) is False
        assert store.spec("train-1") == monitor
        assert store.previous_sample("train-1") == sample
        assert store.pending_signals() == [SIGNAL]


def test_changed_policy_reactivates_monitor_and_clears_old_state(tmp_path: Path):
    original = spec()
    changed = original.model_copy(
        update={
            "gates": (MetricGate(operator="lte", threshold=0.1),),
            "registered_at": NOW + timedelta(minutes=5),
        }
    )

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(original)
        store.record_poll(
            original,
            MonitorEvaluation(signals=(SIGNAL,)),
            MetricSample(value=0.19, observed_at=NOW),
            now=NOW,
        )
        store.complete(original.training_id)

        assert store.register(changed) is True
        assert store.active() == [changed]
        assert store.previous_sample("train-1") is None
        assert store.baseline_sample("train-1") is None
        assert store.pending_signals() == []
        assert store.emitted("train-1") == frozenset()
        assert store.due(NOW + timedelta(minutes=5)) == [changed]
        marker = store.marker_dir / "train-1.json"
        assert TrainingMonitorSpec.model_validate_json(marker.read_text()) == changed


def test_signal_is_durable_deduplicated_and_acknowledged(tmp_path: Path):
    database = tmp_path / "monitors.sqlite3"
    monitor = spec()
    evaluation = MonitorEvaluation(signals=(SIGNAL,))

    with MonitorStore(database) as store:
        store.register(monitor)
        store.record_poll(monitor, evaluation, None, now=NOW)
        store.record_poll(monitor, evaluation, None, now=NOW)

    with MonitorStore(database) as reopened:
        assert reopened.pending_signals() == [SIGNAL]
        assert reopened.emitted("train-1") == frozenset({SIGNAL.dedupe_key})
        reopened.acknowledge(SIGNAL.dedupe_key)

    with MonitorStore(database) as reopened:
        assert reopened.pending_signals() == []
        assert reopened.emitted("train-1") == frozenset({SIGNAL.dedupe_key})


def test_first_sample_remains_the_change_gate_baseline(tmp_path: Path):
    monitor = spec(
        gates=(MetricGate(operator="improved_by", threshold=0.1),)
    )
    first = MetricSample(value=0.8, observed_at=NOW)
    later = MetricSample(value=0.75, observed_at=NOW + timedelta(minutes=1))

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        store.record_poll(monitor, MonitorEvaluation(), first, now=NOW)
        store.record_poll(
            monitor,
            MonitorEvaluation(),
            later,
            now=NOW + timedelta(minutes=1),
        )

        assert store.baseline_sample("train-1") == first
        assert store.previous_sample("train-1") == later


@pytest.mark.parametrize("baseline_column_exists", [False, True])
def test_legacy_schema_promotes_previous_sample_to_change_baseline(
    tmp_path: Path,
    baseline_column_exists: bool,
):
    monitor = spec(
        gates=(MetricGate(operator="improved_by", threshold=0.1),),
        registered_at=NOW - timedelta(minutes=1),
    )
    previous = MetricSample(value=1.0, observed_at=NOW - timedelta(minutes=1))
    current = MetricSample(value=0.89, observed_at=NOW)
    database = tmp_path / "monitors.sqlite3"
    baseline_column = "baseline_sample_json TEXT," if baseline_column_exists else ""
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            f"""
            CREATE TABLE monitors (
                training_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                previous_sample_json TEXT,
                {baseline_column}
                next_poll_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            INSERT INTO monitors (training_id, spec_json, previous_sample_json)
            VALUES (?, ?, ?)
            """,
            (
                monitor.training_id,
                monitor.model_dump_json(),
                previous.model_dump_json(),
            ),
        )
        connection.commit()

    with MonitorStore(database) as store:
        assert store.baseline_sample("train-1") == previous
        evaluation, _ = evaluate_monitor(
            monitor,
            SimpleNamespace(
                state=TrainingState.RUNNING,
                exit_code=None,
            ),
            current,
            previous=store.previous_sample("train-1"),
            baseline=store.baseline_sample("train-1"),
            emitted=frozenset(),
            now=NOW,
        )

        assert [item.kind for item in evaluation.signals] == ["metric_gate"]
