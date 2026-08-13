import json
import math
import sqlite3
import sys
import threading
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlite_test_support import assert_repeated_concurrent_first_open

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


def test_monitor_store_allows_bounded_concurrent_first_open(tmp_path: Path):
    """
    Requirement: training tools and lifecycle observers may first discover the
    monitor database concurrently without a lock error or partial schema.
    Interface: MonitorStore construction and its active-monitor view.
    """

    assert_repeated_concurrent_first_open(
        lambda attempt: MonitorStore(tmp_path / f"monitors-{attempt}.sqlite3"),
        attempts=25,
    )

    database = tmp_path / "monitors-reopen.sqlite3"
    with MonitorStore(database) as reopened:
        assert reopened.active() == []


def test_registration_survives_reopen_in_the_authoritative_database(tmp_path: Path):
    monitor = spec()
    database = tmp_path / "monitors.sqlite3"

    with MonitorStore(database) as store:
        assert store.register(monitor) is True

    with MonitorStore(database) as reopened:
        assert reopened.active() == [monitor]


def test_upgrade_migrates_legacy_intervals_without_retiring_monitors(
    tmp_path: Path,
    capsys,
):
    database = tmp_path / "monitors.sqlite3"
    conversation_id = uuid4()
    legacy_policy = {
        "training_id": "legacy-active",
        "conversation_id": str(conversation_id),
        "metric": None,
        "direction": None,
        "gates": [],
        "poll_interval_seconds": 0.5,
        "stale_after_seconds": float("inf"),
        "notify_on_status": ["finished", "failed", "timed_out", "cancelled"],
        "registered_at": NOW.isoformat(),
    }
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            """
            CREATE TABLE monitors (
                training_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                previous_sample_json TEXT,
                baseline_sample_json TEXT,
                next_poll_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO monitors (training_id, spec_json, active)
            VALUES (?, ?, ?)
            """,
            (
                ("legacy-active", json.dumps(legacy_policy), 1),
                (
                    "legacy-complete",
                    json.dumps(
                        {**legacy_policy, "training_id": "legacy-complete"}
                    ),
                    0,
                ),
            ),
        )
        connection.commit()

    with MonitorStore(database) as store:
        active = store.active()
        inactive = store.spec("legacy-complete")
        rows = store.connection.execute(
            "SELECT training_id, spec_json, active FROM monitors ORDER BY training_id"
        ).fetchall()

    assert [item.training_id for item in active] == ["legacy-active"]
    assert active[0].poll_interval_seconds == 5
    assert math.isfinite(active[0].stale_after_seconds)
    assert active[0].stale_after_seconds == sys.float_info.max
    terminal, _ = evaluate_monitor(
        active[0],
        SimpleNamespace(state=TrainingState.FINISHED, exit_code=0),
        None,
        previous=None,
        emitted=frozenset(),
        now=NOW,
    )
    assert [signal.kind for signal in terminal.signals] == ["training_status"]
    assert inactive.training_id == "legacy-complete"
    assert [(row["training_id"], row["active"]) for row in rows] == [
        ("legacy-active", 1),
        ("legacy-complete", 0),
    ]
    assert all(
        json.loads(row["spec_json"])["poll_interval_seconds"] == 5
        for row in rows
    )
    assert "SENPAI_MONITOR_QUARANTINED" not in capsys.readouterr().err


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


def test_next_poll_delay_tracks_the_earliest_active_monitor(tmp_path: Path):
    first = spec(training_id="first", poll_interval_seconds=60)
    second = spec(training_id="second", poll_interval_seconds=180)

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        assert store.seconds_until_next_poll(NOW) is None
        store.register(first)
        store.register(second)
        assert store.seconds_until_next_poll(NOW) == 0

        store.record_poll(first, MonitorEvaluation(), None, now=NOW)
        store.record_poll(second, MonitorEvaluation(), None, now=NOW)
        assert store.seconds_until_next_poll(NOW + timedelta(seconds=30)) == 30

        store.complete(first.training_id)
        assert store.seconds_until_next_poll(NOW + timedelta(seconds=30)) == 150

        store.complete(second.training_id)
        assert store.seconds_until_next_poll(NOW) is None


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
    monitor = spec(gates=(MetricGate(operator="improved_by", threshold=0.1),))
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


def test_malformed_monitor_is_quarantined_without_blocking_healthy_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    broken = spec(training_id="broken")
    healthy = spec(training_id="healthy")

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(broken)
        store.register(healthy)
        with store.connection:
            store.connection.execute(
                "UPDATE monitors SET spec_json = ? WHERE training_id = ?",
                ("not-json", broken.training_id),
            )

        assert store.due(NOW) == [healthy]
        assert store.active() == [healthy]
        assert store.seconds_until_next_poll(NOW) == 0

    assert "SENPAI_MONITOR_QUARANTINED job_id=broken" in capsys.readouterr().err


def test_nonfinite_schedule_is_quarantined_without_a_zero_sleep_loop(
    tmp_path: Path,
):
    broken = spec(training_id="broken")
    healthy = spec(training_id="healthy")

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(broken)
        store.register(healthy)
        store.record_poll(healthy, MonitorEvaluation(), None, now=NOW)
        with store.connection:
            store.connection.execute(
                "UPDATE monitors SET next_poll_at = ? WHERE training_id = ?",
                ("nan", broken.training_id),
            )

        assert store.seconds_until_next_poll(NOW) == 60
        assert store.active() == [healthy]


def test_store_operations_are_serialized_across_monitor_and_tool_threads(
    tmp_path: Path,
):
    monitor = spec()
    errors = []

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)

        def exercise_store(offset: int):
            try:
                for step in range(30):
                    sample = MetricSample(
                        value=float(offset + step),
                        observed_at=NOW + timedelta(seconds=step),
                    )
                    store.record_poll(monitor, MonitorEvaluation(), sample, now=NOW)
                    store.spec(monitor.training_id)
                    store.previous_sample(monitor.training_id)
                    store.emitted(monitor.training_id)
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        threads = [
            threading.Thread(target=exercise_store, args=(index,)) for index in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert store.active() == [monitor]
