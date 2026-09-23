import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricSample,
    MonitorEvaluation,
    MonitorStore,
    TrainingMonitorEngine,
    TrainingMonitorSpec,
    WandbMetricSource,
)
from senpai_agent.training import TrainingResult, TrainingState


NOW = datetime(2026, 7, 30, tzinfo=UTC)


def result(tmp_path: Path, training_id: str, state=TrainingState.RUNNING):
    return TrainingResult(
        training_id=training_id,
        state=state,
        exit_code=0 if state is TrainingState.FINISHED else None,
        elapsed_seconds=20,
        log_path=str(tmp_path / f"{training_id}.log"),
        wandb_run_ids=(f"run-{training_id}",),
    )


def monitor(training_id="train-1", *, metric=None):
    return TrainingMonitorSpec(
        training_id=training_id,
        conversation_id=uuid4(),
        metric=metric,
        poll_interval_seconds=60,
        registered_at=NOW,
    )


@pytest.mark.parametrize("failure_site", ["status", "metric"])
def test_backend_failure_is_durable_and_does_not_block_other_monitors(
    tmp_path: Path,
    failure_site: str,
):
    bad = monitor("train-bad", metric="val/loss")
    good = monitor("train-good")

    class Training:
        def get_training_status(self, training_id):
            if training_id == bad.training_id and failure_site == "status":
                raise RuntimeError("status backend unavailable")
            state = (
                TrainingState.RUNNING
                if training_id == bad.training_id
                else TrainingState.FINISHED
            )
            return result(tmp_path, training_id, state)

    class Metrics:
        def latest(self, run_id, _metric):
            if failure_site == "metric" and run_id == "run-train-bad":
                raise ValueError("val/loss returned non-finite value nan")
            return MetricSample(value=0.2, observed_at=NOW)

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(bad)
        store.register(good)

        produced = TrainingMonitorEngine(store, Training(), Metrics()).poll(NOW)

        assert {item.kind for item in produced} == {
            "monitor_error",
            "training_status",
        }
        error = next(item for item in produced if item.kind == "monitor_error")
        assert error.training_id == bad.training_id
        assert error.hard_failure is True
        assert error.state is (
            None if failure_site == "status" else TrainingState.RUNNING
        )
        assert store.pending_signals() == list(produced)
        assert store.active() == [bad]
        assert store.due(NOW + timedelta(seconds=59)) == []


def test_repeated_backend_failure_does_not_duplicate_its_signal(tmp_path: Path):
    spec = monitor()

    class Training:
        def get_training_status(self, _training_id):
            raise RuntimeError("status backend unavailable")

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        engine = TrainingMonitorEngine(store, Training(), SimpleNamespace())

        first = engine.poll(NOW)
        repeated = engine.poll(NOW + timedelta(seconds=60))

        assert [item.kind for item in first] == ["monitor_error"]
        assert repeated == ()
        assert store.pending_signals() == [first[0]]


@pytest.mark.parametrize(
    "sample_column", ["previous_sample_json", "baseline_sample_json"]
)
def test_invalid_legacy_sample_emits_error_and_clears_sample_state(
    tmp_path: Path,
    sample_column: str,
):
    spec = monitor(metric="val/loss")

    class Training:
        def get_training_status(self, training_id):
            return result(tmp_path, training_id)

    class Metrics:
        def latest(self, _run_id, _metric):
            return MetricSample(value=0.2, observed_at=NOW)

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        store.connection.execute(
            f"UPDATE monitors SET {sample_column} = ? WHERE training_id = ?",
            (
                '{"value":null,"observed_at":"2026-07-30T00:00:00Z"}',
                spec.training_id,
            ),
        )
        store.connection.commit()

        produced = TrainingMonitorEngine(store, Training(), Metrics()).poll(NOW)

        assert [item.kind for item in produced] == ["monitor_error"]
        assert store.previous_sample("train-1") is None
        assert store.baseline_sample("train-1") is None


@pytest.mark.parametrize(
    "obstruction",
    ["metric-source", "previous_sample_json", "baseline_sample_json"],
)
def test_terminal_failure_wins_over_metric_and_sample_failures(
    tmp_path: Path,
    obstruction: str,
):
    spec = monitor(metric="val/loss")

    class Training:
        def get_training_status(self, training_id):
            return result(tmp_path, training_id, TrainingState.FAILED)

    class Metrics:
        def latest(self, _run_id, _metric):
            raise RuntimeError("W&B is unavailable")

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        if obstruction != "metric-source":
            store.connection.execute(
                f"UPDATE monitors SET {obstruction} = ? WHERE training_id = ?",
                (
                    '{"value":null,"observed_at":"2026-07-30T00:00:00Z"}',
                    spec.training_id,
                ),
            )
            store.connection.commit()

        produced = TrainingMonitorEngine(store, Training(), Metrics()).poll(NOW)

        assert [item.kind for item in produced] == ["training_status"]
        assert produced[0].state is TrainingState.FAILED
        assert produced[0].hard_failure is True
        assert store.pending_signals() == list(produced)
        assert store.active() == []


def test_terminal_hint_bypasses_future_deadline_without_fetching_metrics(
    tmp_path: Path,
):
    spec = monitor(metric="val/loss")

    class Training:
        def __init__(self):
            self.calls = []

        def get_training_status(self, training_id):
            self.calls.append(training_id)
            return result(tmp_path, training_id, TrainingState.FINISHED)

    class Metrics:
        def __init__(self):
            self.calls = []

        def latest(self, run_id, metric):
            self.calls.append((run_id, metric))
            raise AssertionError("terminal hints must not fetch metrics")

    training = Training()
    metrics = Metrics()
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        store.record_poll(spec, MonitorEvaluation(), None, now=NOW)
        assert store.due(NOW + timedelta(seconds=1)) == []

        polled = TrainingMonitorEngine(store, training, metrics).poll_terminal(
            (spec.training_id,),
            now=NOW + timedelta(seconds=1),
        )
        produced = polled.items

        assert [signal.kind for signal in produced] == ["training_status"]
        assert polled.resolved_training_ids == frozenset({spec.training_id})
        assert produced[0].state is TrainingState.FINISHED
        assert training.calls == [spec.training_id]
        assert metrics.calls == []
        assert store.active() == []


def test_running_unknown_and_inactive_terminal_hints_are_no_ops(tmp_path: Path):
    running = monitor("train-running", metric="val/loss")
    inactive = monitor("train-inactive")

    class Training:
        def __init__(self):
            self.calls = []

        def get_training_status(self, training_id):
            self.calls.append(training_id)
            if training_id != running.training_id:
                raise AssertionError("inactive and unknown hints must not be queried")
            return result(tmp_path, training_id, TrainingState.RUNNING)

    class Metrics:
        def latest(self, _run_id, _metric):
            raise AssertionError("a terminal hint for a running job must not fetch metrics")

    training = Training()
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(running)
        store.record_poll(running, MonitorEvaluation(), None, now=NOW)
        store.register(inactive)
        store.complete(inactive.training_id)

        polled = TrainingMonitorEngine(store, training, Metrics()).poll_terminal(
            ("train-unknown", inactive.training_id, running.training_id),
            now=NOW + timedelta(seconds=1),
        )
        produced = polled.items

        assert produced == ()
        assert polled.resolved_training_ids == frozenset(
            {"train-unknown", inactive.training_id, running.training_id}
        )
        assert training.calls == [running.training_id]
        assert store.active() == [running]
        assert store.due(NOW + timedelta(seconds=59)) == []
        assert store.due(NOW + timedelta(seconds=60)) == [running]


def test_terminal_hints_are_deduplicated_before_status_lookup(tmp_path: Path):
    spec = monitor("train-deduplicated")

    class Training:
        def __init__(self):
            self.calls = []

        def get_training_status(self, training_id):
            self.calls.append(training_id)
            return result(tmp_path, training_id, TrainingState.FINISHED)

    training = Training()
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)

        polled = TrainingMonitorEngine(
            store,
            training,
            SimpleNamespace(),
        ).poll_terminal(
            (spec.training_id, spec.training_id, spec.training_id),
            now=NOW,
        )
        produced = polled.items

        assert training.calls == [spec.training_id]
        assert [signal.dedupe_key for signal in produced] == [
            f"{spec.training_id}:status:finished"
        ]
        assert store.pending_signals() == list(produced)


def test_terminal_status_read_failure_remains_unresolved_until_retry(
    tmp_path: Path,
):
    spec = monitor("train-retry")

    class Training:
        def __init__(self):
            self.calls = 0

        def get_training_status(self, training_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("status backend unavailable")
            return result(tmp_path, training_id, TrainingState.FINISHED)

    training = Training()
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        engine = TrainingMonitorEngine(store, training, SimpleNamespace())

        failed = engine.poll_terminal((spec.training_id,), now=NOW)
        recovered = engine.poll_terminal(
            (spec.training_id,),
            now=NOW + timedelta(seconds=60),
        )

        assert failed.resolved_training_ids == frozenset()
        assert recovered.resolved_training_ids == frozenset({spec.training_id})
        assert [signal.kind for signal in failed.items] == ["monitor_error"]
        assert [signal.kind for signal in recovered.items] == ["training_status"]


def test_wandb_source_returns_latest_value_and_timestamp(monkeypatch):
    class Run:
        def history(self, *, keys, **_options):
            if keys != ["accuracy", "_timestamp"]:
                raise AssertionError("metric and timestamp must be requested together")
            return [
                {"accuracy": 0.6, "_timestamp": 100},
                {"accuracy": 0.7, "_timestamp": 200},
            ]

    def run(path):
        if path != "entity/project/run-1":
            raise AssertionError("wrong W&B run path")
        return Run()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=lambda **_options: SimpleNamespace(run=run)),
    )

    sample = WandbMetricSource("entity", "project").latest("run-1", "accuracy")

    assert sample == MetricSample(
        value=0.7,
        observed_at=datetime.fromtimestamp(200, UTC),
    )
