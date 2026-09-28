import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricGate,
    MetricRunNotFoundError,
    MetricSample,
    MonitorEvaluation,
    MonitorSignal,
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


@pytest.mark.parametrize("cleanup_seconds,reported", [(4, False), (34, False), (34, True)])
def test_terminal_notification_waits_for_release_with_durable_bounded_followup(
    tmp_path: Path,
    cleanup_seconds: int,
    reported: bool,
):
    spec = monitor(metric="loss")
    current = result(tmp_path, spec.training_id)
    metric_reads = []
    training = SimpleNamespace(get_training_status=lambda _id: current)
    metrics = SimpleNamespace(latest=lambda *_args: metric_reads.append(True))
    database = tmp_path / "monitors.sqlite3"
    with MonitorStore(database) as store:
        store.register(spec)
        engine = TrainingMonitorEngine(store, training, metrics)
        assert engine.poll(NOW) == ()
        current = current.model_copy(
            update={
                "state": TrainingState.FAILED,
                "kubernetes_released": False,
            }
        )
        assert engine.poll(NOW + timedelta(seconds=2)) == ()
        if reported:
            store.mark_terminal_reported(current)

    with MonitorStore(database) as store:
        engine = TrainingMonitorEngine(store, training, metrics)
        if cleanup_seconds > 30:
            assert engine.poll(NOW + timedelta(seconds=31)) == ()
            pending = engine.poll(NOW + timedelta(seconds=32))
            if reported:
                assert pending == ()
            else:
                assert len(pending) == 1
                assert "cleanup is still pending" in pending[0].detail.lower()
                store.acknowledge(pending[0].dedupe_key)
            assert engine.poll(NOW + timedelta(seconds=33)) == ()
        current = current.model_copy(update={"kubernetes_released": True})
        released = engine.poll(NOW + timedelta(seconds=cleanup_seconds))
        assert len(released) == 1
        assert "released" in released[0].detail.lower()
        assert released[0].state is TrainingState.FAILED
        assert store.active() == []
        assert metric_reads == [True]

    with MonitorStore(database) as store:
        assert (
            TrainingMonitorEngine(store, training, metrics).poll(
                NOW + timedelta(seconds=90)
            )
            == ()
        )


def test_first_threshold_alert_combines_gates_and_stays_latched_after_restart(
    tmp_path: Path,
):
    spec = monitor(metric="loss").model_copy(
        update={
            "gates": (
                MetricGate(operator="lte", threshold=0.5),
                MetricGate(operator="lte", threshold=0.4),
                MetricGate(operator="lte", threshold=0.2),
            )
        }
    )
    value = 0.3
    training = SimpleNamespace(
        get_training_status=lambda _id: result(tmp_path, spec.training_id)
    )
    metrics = SimpleNamespace(
        latest=lambda *_args: MetricSample(
            value=value,
            observed_at=NOW,
        )
    )
    database = tmp_path / "monitors.sqlite3"
    with MonitorStore(database) as store:
        store.register(spec)
        signals = TrainingMonitorEngine(store, training, metrics).poll(NOW)
        assert len(signals) == 1
        assert signals[0].kind == "metric_gate"
        assert "0.5" in signals[0].detail and "0.4" in signals[0].detail
        store.acknowledge(signals[0].dedupe_key)
    with MonitorStore(database) as store:
        assert not store.register(
            spec.model_copy(
                update={
                    "registered_at": NOW + timedelta(seconds=60),
                }
            )
        )
        value = 0.1
        assert (
            TrainingMonitorEngine(store, training, metrics).poll(
                NOW + timedelta(seconds=600)
            )
            == ()
        )


@pytest.mark.parametrize("failure", [False, True])
def test_replaced_policy_discards_an_inflight_metric_result(
    tmp_path: Path, failure: bool
):
    original = monitor(metric="loss").model_copy(
        update={
            "gates": (MetricGate(operator="lte", threshold=0.5),),
        }
    )
    replacement = original.model_copy(
        update={
            "gates": (MetricGate(operator="lte", threshold=0.1),),
            "registered_at": NOW + timedelta(seconds=1),
        }
    )
    database = tmp_path / "monitors.sqlite3"

    class Metrics:
        def latest(self, *_args):
            with MonitorStore(database) as other:
                other.register(replacement)
            if failure:
                raise RuntimeError("old policy request failed")
            return MetricSample(value=0.3, observed_at=NOW)

    training = SimpleNamespace(
        get_training_status=lambda _id: result(tmp_path, original.training_id)
    )
    with MonitorStore(database) as store:
        store.register(original)
        assert TrainingMonitorEngine(store, training, Metrics()).poll(NOW) == ()
        assert store.pending_signals() == []
        assert store.previous_sample(original.training_id) is None
        assert store.due(NOW) == [replacement]


@pytest.mark.parametrize(
    "kind,suffix",
    [
        ("metric_gate", "gate:0"),
        ("metric_stale", "stale:2026-07-29T00:00:00+00:00"),
        ("monitor_error", "monitor_error:RuntimeError"),
    ],
)
def test_legacy_alert_remains_latched_after_monitor_upgrade(
    tmp_path: Path, kind, suffix
):
    spec = monitor(metric="loss").model_copy(
        update={
            "gates": (MetricGate(operator="lte", threshold=0.5),)
            if kind == "metric_gate"
            else (),
            "stale_after_seconds": 60 if kind == "metric_stale" else None,
        }
    )
    prior = MonitorSignal(
        kind=kind,
        dedupe_key=f"train-1:{suffix}",
        training_id="train-1",
        state=TrainingState.RUNNING,
        detail="Previously notified.",
    )

    class Metrics:
        def latest(self, *_args):
            if kind == "monitor_error":
                raise ValueError("A different backend error")
            return MetricSample(value=0.1, observed_at=NOW)

    database = tmp_path / "monitors.sqlite3"
    with MonitorStore(database) as store:
        store.register(spec)
        store.record_poll(spec, MonitorEvaluation(signals=(prior,)), None, now=NOW)
        store.acknowledge(prior.dedupe_key)
    with MonitorStore(database) as store:
        training = SimpleNamespace(
            get_training_status=lambda name: result(tmp_path, name)
        )
        assert (
            TrainingMonitorEngine(store, training, Metrics()).poll(
                NOW + timedelta(seconds=60)
            )
            == ()
        )
        assert store.pending_signals() == []


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
            runs=lambda _path, **options: Runs(service, "entity", "project", **options)
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

    monkeypatch.setattr(wandb, "Api", lambda **_options: SimpleNamespace(runs=runs))
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
