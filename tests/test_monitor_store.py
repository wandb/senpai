from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricGate,
    MetricMonitorSpec,
    MetricSample,
    MonitorEvaluation,
    JobMonitorMailbox,
    MonitorSignal,
    JobMonitorStore,
    JobMonitorSpec,
)
from senpai_agent.jobs import JobState


NOW = datetime(2026, 7, 30, tzinfo=UTC)


def spec(**changes):
    values = {
        "job_id": "job-1",
        "conversation_id": uuid4(),
        "metrics": (
            MetricMonitorSpec(
                metric="val/loss",
                direction="min",
                gates=(MetricGate(operator="lte", threshold=0.2),),
            ),
        ),
        "registered_at": NOW,
    }
    values.update(changes)
    return JobMonitorSpec(**values)


SIGNAL = MonitorSignal(
    kind="metric_gate",
    dedupe_key="job-1:2026-07-30T00:00:00+00:00:metric:val/loss:gate:0",
    job_id="job-1",
    metric="val/loss",
    value=0.19,
    state=JobState.RUNNING,
    detail="val/loss crossed the threshold.",
)


def test_registration_survives_reopen(tmp_path: Path):
    monitor = spec()
    database = tmp_path / "monitors.sqlite3"

    with JobMonitorStore(database) as store:
        assert store.register(monitor) is True

    with JobMonitorStore(database) as reopened:
        assert reopened.active() == [monitor]


def test_same_policy_reregistration_preserves_derived_state(tmp_path: Path):
    monitor = spec()
    repeated = monitor.model_copy(update={"registered_at": NOW + timedelta(minutes=5)})
    sample = MetricSample(value=0.19, observed_at=NOW)

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        store.record_poll(
            monitor,
            MonitorEvaluation(signals=(SIGNAL,)),
            {"val/loss": sample},
            now=NOW,
        )

        assert store.register(repeated) is False
        assert store.spec("job-1") == monitor
        assert store.previous_sample("job-1", "val/loss") == sample
        assert store.pending_signals() == [SIGNAL]


def test_poll_interval_update_preserves_registration_samples_and_dedupe(
    tmp_path: Path,
):
    monitor = spec(poll_interval_seconds=60)
    rescheduled = monitor.model_copy(
        update={
            "poll_interval_seconds": 15,
            "registered_at": NOW + timedelta(minutes=5),
        }
    )
    sample = MetricSample(value=0.19, observed_at=NOW)

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        store.record_poll(
            monitor,
            MonitorEvaluation(signals=(SIGNAL,)),
            {"val/loss": sample},
            now=NOW,
        )
        store.acknowledge(SIGNAL.dedupe_key)

        assert store.register(rescheduled) is False
        stored = store.spec("job-1")
        assert stored.poll_interval_seconds == 15
        assert stored.registered_at == monitor.registered_at
        assert store.previous_sample("job-1", "val/loss") == sample
        assert store.baseline_sample("job-1", "val/loss") == sample
        assert store.emitted("job-1") == frozenset({SIGNAL.dedupe_key})
        assert store.pending_signals() == []
        assert store.due(NOW + timedelta(seconds=14)) == []
        assert store.due(NOW + timedelta(seconds=15)) == [stored]


def test_changed_policy_reactivates_monitor_and_clears_old_state(tmp_path: Path):
    original = spec()
    changed = original.model_copy(
        update={
            "metrics": (
                MetricMonitorSpec(
                    metric="val/loss",
                    direction="min",
                    gates=(MetricGate(operator="lte", threshold=0.1),),
                ),
            ),
            "registered_at": NOW + timedelta(minutes=5),
        }
    )

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(original)
        store.record_poll(
            original,
            MonitorEvaluation(signals=(SIGNAL,)),
            {"val/loss": MetricSample(value=0.19, observed_at=NOW)},
            now=NOW,
        )
        store.discard(original.job_id)

        assert store.register(changed) is True
        assert store.active() == [changed]
        assert store.previous_sample("job-1", "val/loss") is None
        assert store.baseline_sample("job-1", "val/loss") is None
        assert store.pending_signals() == []
        assert store.emitted("job-1") == frozenset()
        assert store.due(NOW + timedelta(minutes=5)) == [changed]


def test_signal_is_durable_deduplicated_and_acknowledged(tmp_path: Path):
    database = tmp_path / "monitors.sqlite3"
    monitor = spec()
    evaluation = MonitorEvaluation(signals=(SIGNAL,))

    with JobMonitorStore(database) as store:
        store.register(monitor)
        store.record_poll(monitor, evaluation, {}, now=NOW)
        store.record_poll(monitor, evaluation, {}, now=NOW)

    with JobMonitorStore(database) as reopened:
        assert reopened.pending_signals() == [SIGNAL]
        assert reopened.emitted("job-1") == frozenset({SIGNAL.dedupe_key})
        reopened.acknowledge(SIGNAL.dedupe_key)

    with JobMonitorStore(database) as reopened:
        assert reopened.pending_signals() == []
        assert reopened.emitted("job-1") == frozenset({SIGNAL.dedupe_key})


def test_terminal_observation_survives_policy_reactivation(tmp_path: Path):
    monitor = spec()
    result = SimpleNamespace(
        job_id=monitor.job_id,
        state=JobState.FAILED,
        exit_code=2,
        wandb_run_ids=(),
    )

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        signal = store.record_terminal_and_complete(result)
        assert signal is not None
        store.acknowledge(signal.dedupe_key)

        assert store.register(
            monitor.model_copy(
                update={"registered_at": NOW + timedelta(minutes=1)}
            )
        )
        assert store.record_terminal_and_complete(result) is None
        assert store.emitted(monitor.job_id) == frozenset(
            {"job-1:status:failed"}
        )


def test_first_sample_remains_the_change_gate_baseline(tmp_path: Path):
    monitor = spec(
        metrics=(
            MetricMonitorSpec(
                metric="val/loss",
                direction="min",
                gates=(MetricGate(operator="improved_by", threshold=0.1),),
            ),
        )
    )
    first = MetricSample(value=0.8, observed_at=NOW)
    later = MetricSample(value=0.75, observed_at=NOW + timedelta(minutes=1))

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        store.record_poll(
            monitor,
            MonitorEvaluation(),
            {"val/loss": first},
            now=NOW,
        )
        store.record_poll(
            monitor,
            MonitorEvaluation(),
            {"val/loss": later},
            now=NOW + timedelta(minutes=1),
        )

        assert store.baseline_sample("job-1", "val/loss") == first
        assert store.previous_sample("job-1", "val/loss") == later


def test_fresh_schema_is_job_only_without_singleton_sample_columns(tmp_path: Path):
    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        tables = store.connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        columns = {
            table: [
                row[1]
                for row in store.connection.execute(f"PRAGMA table_info({table})")
            ]
            for table, _sql in tables
            if table.startswith("monitor")
        }

    assert columns["monitors"] == [
        "job_id",
        "spec_json",
        "active",
        "next_poll_at",
        "poll_generation",
        "poll_claimed_until",
    ]
    assert columns["monitor_signals"][:2] == ["dedupe_key", "job_id"]
    assert columns["monitor_metric_samples"][:2] == ["job_id", "metric"]
    assert "training" not in " ".join(sql for _table, sql in tables).lower()


def test_late_overlapping_poll_cannot_replace_a_newer_sample(tmp_path: Path):
    database = tmp_path / "monitors.sqlite3"
    monitor = spec()
    older = MetricSample(value=0.8, observed_at=NOW)
    newer = MetricSample(value=0.7, observed_at=NOW + timedelta(seconds=2))

    with JobMonitorStore(database) as first, JobMonitorStore(database) as second:
        first.register(monitor)
        first_claim = first.claim_due(NOW, lease_seconds=1)
        second_claim = second.claim_due(
            NOW + timedelta(seconds=2),
            lease_seconds=1,
        )
        assert first_claim is not None
        assert second_claim is not None

        assert second.record_poll(
            second_claim.spec,
            MonitorEvaluation(),
            {"val/loss": newer},
            generation=second_claim.generation,
            now=NOW + timedelta(seconds=2),
        )
        assert not first.record_poll(
            first_claim.spec,
            MonitorEvaluation(),
            {"val/loss": older},
            generation=first_claim.generation,
            now=NOW,
        )
        assert first.previous_sample("job-1", "val/loss") == newer


def test_metric_samples_keep_independent_previous_and_baseline_values(
    tmp_path: Path,
):
    monitor = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(
            MetricMonitorSpec(metric="loss"),
            MetricMonitorSpec(metric="throughput"),
        ),
        registered_at=NOW,
    )
    first = {
        "loss": MetricSample(value=1.0, observed_at=NOW),
        "throughput": MetricSample(value=100, observed_at=NOW),
    }
    later = {
        "loss": MetricSample(value=0.8, observed_at=NOW + timedelta(minutes=1)),
        "throughput": MetricSample(
            value=120,
            observed_at=NOW + timedelta(minutes=1),
        ),
    }

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        store.record_poll(monitor, MonitorEvaluation(), first, now=NOW)
        store.record_poll(
            monitor,
            MonitorEvaluation(),
            later,
            now=NOW + timedelta(minutes=1),
        )

        assert store.baseline_sample("job-1", "loss") == first["loss"]
        assert store.previous_sample("job-1", "loss") == later["loss"]
        assert store.baseline_sample("job-1", "throughput") == first[
            "throughput"
        ]
        assert store.previous_sample("job-1", "throughput") == later[
            "throughput"
        ]


def test_monitor_mailbox_emits_canonical_job_event(tmp_path: Path):
    monitor = spec()

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(monitor)
        store.record_poll(
            monitor,
            MonitorEvaluation(signals=(SIGNAL,)),
            {},
            now=NOW,
        )

        events = JobMonitorMailbox(
            SimpleNamespace(poll=lambda: ()),
            store,
        ).poll()

    assert len(events) == 1
    assert events[0].kind == "job_monitor"
    assert events[0].payload["job_id"] == "job-1"
    assert events[0].payload["signal"]["job_id"] == "job-1"
