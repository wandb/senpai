import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricGate,
    MetricMonitorSpec,
    MetricSample,
    JobMonitorStore,
    JobMonitorEngine,
    JobMonitorSpec,
    WandbMetricSource,
    WandbJobStatusSource,
)
from senpai_agent.jobs import JobResult, JobState


NOW = datetime(2026, 7, 30, tzinfo=UTC)


def result(tmp_path: Path, job_id: str, state=JobState.RUNNING):
    return JobResult(
        job_id=job_id,
        state=state,
        exit_code=None,
        elapsed_seconds=20,
        log_path=str(tmp_path / f"{job_id}.log"),
        wandb_run_ids=(f"run-{job_id}",),
    )


def monitor(job_id="job-1", *, metric=None):
    return JobMonitorSpec(
        job_id=job_id,
        conversation_id=uuid4(),
        metrics=(MetricMonitorSpec(metric=metric),) if metric else (),
        poll_interval_seconds=60,
        registered_at=NOW,
    )


@pytest.mark.parametrize("failure_site", ["status", "metric"])
def test_backend_failure_is_durable_and_does_not_block_other_monitors(
    tmp_path: Path,
    failure_site: str,
):
    bad = monitor("job-bad", metric="val/loss")
    good = monitor("job-good")

    class Jobs:
        def get_job_status(self, job_id):
            if job_id == bad.job_id and failure_site == "status":
                raise RuntimeError("status backend unavailable")
            state = (
                JobState.RUNNING
                if job_id == bad.job_id
                else JobState.FINISHED
            )
            return result(tmp_path, job_id, state)

    class Metrics:
        def latest(self, run_id, _metric):
            if failure_site == "metric" and run_id == "run-job-bad":
                raise ValueError("val/loss returned non-finite value nan")
            return MetricSample(value=0.2, observed_at=NOW)

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(bad)
        store.register(good)

        produced = JobMonitorEngine(store, Jobs(), Metrics()).poll(NOW)

        assert {item.kind for item in produced} == {
            "monitor_error",
            "job_status",
        }
        error = next(item for item in produced if item.kind == "monitor_error")
        assert error.job_id == bad.job_id
        assert error.hard_failure is True
        assert error.state is (
            None if failure_site == "status" else JobState.RUNNING
        )
        assert store.pending_signals() == list(produced)
        assert store.active() == [bad]
        assert store.due(NOW + timedelta(seconds=59)) == []


def test_repeated_backend_failure_does_not_duplicate_its_signal(tmp_path: Path):
    spec = monitor()

    class Jobs:
        def get_job_status(self, _job_id):
            raise RuntimeError("status backend unavailable")

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        engine = JobMonitorEngine(store, Jobs(), SimpleNamespace())

        first = engine.poll(NOW)
        repeated = engine.poll(NOW + timedelta(seconds=60))

        assert [item.kind for item in first] == ["monitor_error"]
        assert repeated == ()
        assert store.pending_signals() == [first[0]]


@pytest.mark.parametrize(
    "sample_column", ["previous_sample_json", "baseline_sample_json"]
)
def test_invalid_metric_sample_emits_error_and_clears_sample_state(
    tmp_path: Path,
    sample_column: str,
):
    spec = monitor(metric="val/loss")

    class Jobs:
        def get_job_status(self, job_id):
            return result(tmp_path, job_id)

    class Metrics:
        def latest(self, _run_id, _metric):
            return MetricSample(value=0.2, observed_at=NOW)

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        store.connection.execute(
            f"""
            INSERT INTO monitor_metric_samples (
                job_id,
                metric,
                {sample_column}
            )
            VALUES (?, ?, ?)
            """,
            (
                spec.job_id,
                "val/loss",
                '{"value":null,"observed_at":"2026-07-30T00:00:00Z"}',
            ),
        )
        store.connection.commit()

        produced = JobMonitorEngine(store, Jobs(), Metrics()).poll(NOW)

        assert [item.kind for item in produced] == ["monitor_error"]
        assert store.previous_sample("job-1", "val/loss") is None
        assert store.baseline_sample("job-1", "val/loss") is None


@pytest.mark.parametrize(
    "obstruction",
    ["metric-source", "previous_sample_json", "baseline_sample_json"],
)
def test_terminal_failure_wins_over_metric_and_sample_failures(
    tmp_path: Path,
    obstruction: str,
):
    spec = monitor(metric="val/loss")

    class Jobs:
        def get_job_status(self, job_id):
            return result(tmp_path, job_id, JobState.FAILED)

    class Metrics:
        def latest(self, _run_id, _metric):
            raise RuntimeError("W&B is unavailable")

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        if obstruction != "metric-source":
            store.connection.execute(
                f"""
                INSERT INTO monitor_metric_samples (
                    job_id,
                    metric,
                    {obstruction}
                )
                VALUES (?, ?, ?)
                """,
                (
                    spec.job_id,
                    "val/loss",
                    '{"value":null,"observed_at":"2026-07-30T00:00:00Z"}',
                ),
            )
            store.connection.commit()

        produced = JobMonitorEngine(store, Jobs(), Metrics()).poll(NOW)

        assert [item.kind for item in produced] == ["job_status"]
        assert produced[0].state is JobState.FAILED
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


@pytest.mark.parametrize(
    ("wandb_state", "job_state", "exit_code"),
    [
        ("running", JobState.RUNNING, None),
        ("pending", JobState.RUNNING, None),
        ("preempting", JobState.RUNNING, None),
        ("preempted", JobState.RUNNING, None),
        ("finished", JobState.FINISHED, None),
        ("failed", JobState.FAILED, None),
        ("crashed", JobState.FAILED, None),
        ("killed", JobState.CANCELLED, None),
    ],
)
def test_wandb_status_source_maps_run_states(
    monkeypatch,
    wandb_state: str,
    job_state: JobState,
    exit_code: int | None,
):
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Api=lambda **_options: SimpleNamespace(
                run=lambda path: SimpleNamespace(state=wandb_state)
            )
        ),
    )

    status = WandbJobStatusSource("entity", "project").get_job_status(
        "run-1"
    )

    assert status.state is job_state
    assert status.exit_code == exit_code
    assert status.wandb_run_ids == ("run-1",)


def test_wandb_status_source_accepts_dotted_run_id(monkeypatch):
    requested = []
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Api=lambda **_options: SimpleNamespace(
                run=lambda path: (
                    requested.append(path) or SimpleNamespace(state="running")
                )
            )
        ),
    )

    status = WandbJobStatusSource("entity", "project").get_job_status(
        "sweep.run.7"
    )

    assert requested == ["entity/project/sweep.run.7"]
    assert status.state is JobState.RUNNING


def test_wandb_status_source_rejects_unknown_state(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Api=lambda **_options: SimpleNamespace(
                run=lambda _path: SimpleNamespace(state="unknown")
            )
        ),
    )

    with pytest.raises(ValueError, match="unsupported W&B run state 'unknown'"):
        WandbJobStatusSource("entity", "project").get_job_status(
            "run-1"
        )


def test_quiet_monitor_polls_never_create_signals(tmp_path: Path):
    spec = monitor()

    class Jobs:
        def get_job_status(self, job_id):
            return result(tmp_path, job_id)

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        engine = JobMonitorEngine(store, Jobs(), SimpleNamespace())

        for minute in range(1_000):
            assert engine.poll(NOW + timedelta(minutes=minute)) == ()

        assert store.pending_signals() == []


def test_one_metric_failure_does_not_block_another_metric_gate(tmp_path: Path):
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(
            MetricMonitorSpec(metric="broken"),
            MetricMonitorSpec(
                metric="throughput",
                gates=(MetricGate(operator="gte", threshold=100),),
            ),
        ),
        poll_interval_seconds=60,
        registered_at=NOW,
    )

    class Jobs:
        def get_job_status(self, job_id):
            return result(tmp_path, job_id)

    class Metrics:
        def latest(self, _run_id, metric):
            if metric == "broken":
                raise RuntimeError("metric unavailable")
            return MetricSample(value=101, observed_at=NOW)

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)

        produced = JobMonitorEngine(store, Jobs(), Metrics()).poll(NOW)

        assert [(signal.kind, signal.metric) for signal in produced] == [
            ("monitor_error", "broken"),
            ("metric_gate", "throughput"),
        ]
        assert store.previous_sample("job-1", "broken") is None
        assert store.previous_sample("job-1", "throughput") == MetricSample(
            value=101,
            observed_at=NOW,
        )


@pytest.mark.parametrize("poll_result", ["terminal", "metric", "metric_error"])
def test_in_flight_poll_cannot_mutate_a_replaced_policy(
    tmp_path: Path,
    poll_result: str,
):
    database = tmp_path / "monitors.sqlite3"
    old = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(
            MetricMonitorSpec(
                metric="old",
                gates=(MetricGate(operator="gte", threshold=1),),
            ),
        ),
        registered_at=NOW,
    )
    replacement = JobMonitorSpec(
        job_id=old.job_id,
        conversation_id=old.conversation_id,
        metrics=(MetricMonitorSpec(metric="new"),),
        registered_at=NOW + timedelta(seconds=1),
    )
    replaced = False

    def replace_policy():
        nonlocal replaced
        if replaced:
            return
        with JobMonitorStore(database) as tool_store:
            assert tool_store.register(replacement) is True
        replaced = True

    class Jobs:
        def get_job_status(self, job_id):
            if poll_result == "terminal":
                replace_policy()
                return result(tmp_path, job_id, JobState.FINISHED)
            return result(tmp_path, job_id)

    class Metrics:
        def latest(self, _run_id, _metric):
            replace_policy()
            if poll_result == "metric_error":
                raise RuntimeError("old policy failed")
            return MetricSample(value=2, observed_at=NOW)

    with JobMonitorStore(database) as polling_store:
        polling_store.register(old)

        produced = JobMonitorEngine(
            polling_store,
            Jobs(),
            Metrics(),
        ).poll(NOW)

        assert produced == ()
        assert polling_store.active() == [replacement]
        assert polling_store.pending_signals() == []
        assert polling_store.previous_sample(old.job_id, "old") is None
        assert polling_store.due(NOW) == [replacement]


def test_unchanged_registration_preserves_an_in_flight_poll(tmp_path: Path):
    database = tmp_path / "monitors.sqlite3"
    spec = monitor()

    class Jobs:
        def get_job_status(self, job_id):
            with JobMonitorStore(database) as tool_store:
                repeated = spec.model_copy(
                    update={"registered_at": NOW + timedelta(seconds=1)}
                )
                assert tool_store.register(repeated) is False
            return result(tmp_path, job_id, JobState.FINISHED)

    with JobMonitorStore(database) as polling_store:
        polling_store.register(spec)

        produced = JobMonitorEngine(
            polling_store,
            Jobs(),
            SimpleNamespace(),
        ).poll(NOW)

        assert [signal.kind for signal in produced] == ["job_status"]
        assert polling_store.active() == []
