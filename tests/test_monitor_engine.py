import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricRunNotFoundError,
    MetricSample,
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


def test_wandb_source_returns_latest_value_and_timestamp(monkeypatch):
    class Run:
        def history(self, *, keys, **_options):
            if keys != ["accuracy", "_timestamp"]:
                raise AssertionError("metric and timestamp must be requested together")
            return [
                {"accuracy": 0.6, "_timestamp": 100},
                {"accuracy": 0.7, "_timestamp": 200},
            ]

    def runs(path, *, filters, per_page):
        assert path == "entity/project"
        assert filters == {"name": "run-1"}
        assert per_page == 1
        return iter([Run()])

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=lambda **_options: SimpleNamespace(runs=runs)),
    )

    sample = WandbMetricSource("entity", "project").latest("run-1", "accuracy")

    assert sample == MetricSample(
        value=0.7,
        observed_at=datetime.fromtimestamp(200, UTC),
    )


@pytest.mark.parametrize("project_accessible", [True, False])
def test_wandb_source_distinguishes_absent_run_from_inaccessible_project(
    monkeypatch, project_accessible
):
    import wandb
    from wandb.apis.public.runs import Runs

    project = {"runs": {"edges": [], "pageInfo": {"hasNextPage": False}}}
    service = SimpleNamespace(
        execute_graphql=lambda *_args: {
            "project": project if project_accessible else None
        }
    )
    monkeypatch.setattr(
        wandb,
        "Api",
        lambda **_options: SimpleNamespace(
            runs=lambda _path, **options: Runs(
                service, "entity", "project", **options
            )
        ),
    )

    error = MetricRunNotFoundError if project_accessible else ValueError
    with pytest.raises(error):
        WandbMetricSource("entity", "project").latest("run-1", "accuracy")


def test_missing_startup_run_recovers_when_first_metric_arrives(tmp_path: Path):
    spec = monitor(metric="train/global_step").model_copy(
        update={"stale_after_seconds": 900}
    )
    sample = MetricSample(value=1, observed_at=NOW + timedelta(seconds=60))

    class Metrics:
        def latest(self, _run_id, _metric):
            if not self.ready:
                raise MetricRunNotFoundError("W&B run is not initialized")
            return sample

        ready = False

    metrics = Metrics()
    training = SimpleNamespace(
        get_training_status=lambda training_id: result(tmp_path, training_id)
    )
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        engine = TrainingMonitorEngine(store, training, metrics)

        assert engine.poll(NOW) == ()
        assert store.baseline_sample(spec.training_id) is None
        metrics.ready = True
        assert engine.poll(NOW + timedelta(seconds=60)) == ()
        assert store.baseline_sample(spec.training_id) == sample
        assert store.previous_sample(spec.training_id) == sample
        assert store.pending_signals() == []


def test_missing_startup_run_still_emits_stale_signal_at_deadline(tmp_path: Path):
    spec = monitor(metric="train/global_step").model_copy(
        update={"stale_after_seconds": 900}
    )

    class Metrics:
        def latest(self, _run_id, _metric):
            raise MetricRunNotFoundError("W&B run is not initialized")

    training = SimpleNamespace(
        get_training_status=lambda training_id: result(tmp_path, training_id)
    )
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        engine = TrainingMonitorEngine(store, training, Metrics())

        assert engine.poll(NOW) == ()
        assert engine.poll(NOW + timedelta(seconds=840)) == ()
        signals = engine.poll(NOW + timedelta(seconds=900))
        assert [signal.kind for signal in signals] == ["metric_stale"]
        assert signals[0].value is None
        assert signals[0].hard_failure is False
        assert engine.poll(NOW + timedelta(seconds=960)) == ()
        assert store.pending_signals() == list(signals)


@pytest.mark.parametrize("failure", ["auth", "permission", "network"])
def test_wandb_startup_backend_errors_remain_hard_failures(
    tmp_path: Path, monkeypatch, failure: str
):
    import wandb

    errors = {
        "auth": wandb.errors.AuthenticationError("Invalid API key"),
        "permission": wandb.errors.CommError("HTTP 403: permission denied"),
        "network": wandb.errors.CommError("Connection timed out"),
    }

    def runs(*_args, **_options):
        raise errors[failure]

    monkeypatch.setattr(
        wandb, "Api", lambda **_options: SimpleNamespace(runs=runs)
    )
    spec = monitor(metric="train/global_step")
    training = SimpleNamespace(
        get_training_status=lambda training_id: result(tmp_path, training_id)
    )
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        signals = TrainingMonitorEngine(
            store, training, WandbMetricSource("entity", "project")
        ).poll(NOW)

        assert [signal.kind for signal in signals] == ["monitor_error"]
        assert signals[0].hard_failure is True
        assert str(errors[failure]) in signals[0].detail


def test_run_disappearing_after_metric_is_a_durable_hard_failure(tmp_path: Path):
    spec = monitor(metric="train/global_step")
    sample = MetricSample(value=1, observed_at=NOW)

    class Metrics:
        def latest(self, _run_id, _metric):
            if self.missing:
                raise MetricRunNotFoundError("W&B run disappeared")
            return sample

        missing = False

    metrics = Metrics()
    training = SimpleNamespace(
        get_training_status=lambda training_id: result(tmp_path, training_id)
    )
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        engine = TrainingMonitorEngine(store, training, metrics)

        assert engine.poll(NOW) == ()
        metrics.missing = True
        signals = engine.poll(NOW + timedelta(seconds=60))
        assert [signal.kind for signal in signals] == ["monitor_error"]
        assert signals[0].hard_failure is True
        assert store.baseline_sample(spec.training_id) == sample
        assert store.previous_sample(spec.training_id) == sample
        assert engine.poll(NOW + timedelta(seconds=120)) == ()
        assert store.pending_signals() == list(signals)
