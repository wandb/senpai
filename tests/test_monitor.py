from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricGate,
    MetricMonitorSpec,
    MetricSample,
    JobMonitorSpec,
    evaluate_monitor,
)
from senpai_agent.jobs import JobState


NOW = datetime(2026, 7, 30, tzinfo=UTC)


def result(state=JobState.RUNNING):
    return SimpleNamespace(
        state=state,
        exit_code=None,
    )


@pytest.mark.parametrize(
    ("operator", "before", "crossed"),
    [
        ("lte", 0.3, 0.2),
        ("gte", 0.1, 0.2),
    ],
)
def test_absolute_gate_emits_once_when_the_threshold_is_crossed(
    operator: str,
    before: float,
    crossed: float,
):
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(
            MetricMonitorSpec(
                metric="score",
                gates=(MetricGate(operator=operator, threshold=0.2),),
            ),
        ),
    )
    previous = MetricSample(value=before, observed_at=NOW)
    current = MetricSample(
        value=crossed,
        observed_at=NOW + timedelta(minutes=1),
    )

    quiet, _ = evaluate_monitor(
        spec,
        result(),
        {"score": previous},
        previous={},
        emitted=frozenset(),
        now=NOW,
    )
    fired, _ = evaluate_monitor(
        spec,
        result(),
        {"score": current},
        previous={"score": previous},
        emitted=frozenset(),
        now=NOW + timedelta(minutes=1),
    )
    duplicate, _ = evaluate_monitor(
        spec,
        result(),
        {"score": current},
        previous={"score": previous},
        emitted=frozenset(fired.dedupe_keys),
        now=NOW + timedelta(minutes=2),
    )

    assert quiet.signals == ()
    assert [signal.kind for signal in fired.signals] == ["metric_gate"]
    assert duplicate.signals == ()


@pytest.mark.parametrize(
    ("direction", "operator", "baseline_value", "middle_value", "crossed_value"),
    [
        ("min", "improved_by", 1.0, 0.96, 0.91),
        ("min", "regressed_by", 1.0, 1.04, 1.09),
        ("max", "improved_by", 0.5, 0.54, 0.59),
        ("max", "regressed_by", 0.5, 0.46, 0.41),
    ],
)
def test_change_gate_compares_with_the_first_sample_not_the_previous_one(
    direction: str,
    operator: str,
    baseline_value: float,
    middle_value: float,
    crossed_value: float,
):
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(
            MetricMonitorSpec(
                metric="score",
                direction=direction,
                gates=(MetricGate(operator=operator, threshold=0.08),),
            ),
        ),
    )
    baseline = MetricSample(value=baseline_value, observed_at=NOW)
    middle = MetricSample(
        value=middle_value,
        observed_at=NOW + timedelta(minutes=1),
    )
    crossed = MetricSample(
        value=crossed_value,
        observed_at=NOW + timedelta(minutes=2),
    )

    first, _ = evaluate_monitor(
        spec,
        result(),
        {"score": baseline},
        previous={},
        baseline={},
        emitted=frozenset(),
        now=NOW,
    )
    middle_poll, _ = evaluate_monitor(
        spec,
        result(),
        {"score": middle},
        previous={"score": baseline},
        baseline={"score": baseline},
        emitted=frozenset(),
        now=NOW + timedelta(minutes=1),
    )
    final, _ = evaluate_monitor(
        spec,
        result(),
        {"score": crossed},
        previous={"score": middle},
        baseline={"score": baseline},
        emitted=frozenset(),
        now=NOW + timedelta(minutes=2),
    )

    assert first.signals == ()
    assert middle_poll.signals == ()
    assert [signal.kind for signal in final.signals] == ["metric_gate"]


@pytest.mark.parametrize(
    ("state", "hard_failure"),
    [
        (JobState.FINISHED, False),
        (JobState.FAILED, True),
        (JobState.TIMED_OUT, True),
        (JobState.CANCELLED, True),
    ],
)
def test_terminal_status_preempts_metric_and_staleness_signals(
    state: JobState,
    hard_failure: bool,
):
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(
            MetricMonitorSpec(
                metric="accuracy",
                gates=(MetricGate(operator="gte", threshold=0.8),),
                stale_after_seconds=60,
            ),
        ),
    )
    old = MetricSample(value=0.7, observed_at=NOW - timedelta(minutes=2))

    evaluation, _ = evaluate_monitor(
        spec,
        result(state),
        {
            "accuracy": MetricSample(
                value=0.9,
                observed_at=NOW - timedelta(minutes=2),
            )
        },
        previous={"accuracy": old},
        emitted=frozenset(),
        now=NOW,
    )

    assert len(evaluation.signals) == 1
    signal = evaluation.signals[0]
    assert signal.kind == "job_status"
    assert signal.state is state
    assert signal.hard_failure is hard_failure
    assert signal.metric is None
    assert signal.value is None


def test_old_metric_sample_emits_one_stale_signal():
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(MetricMonitorSpec(metric="accuracy", stale_after_seconds=60),),
    )
    old = MetricSample(value=0.7, observed_at=NOW - timedelta(minutes=2))

    stale, _ = evaluate_monitor(
        spec,
        result(),
        {"accuracy": old},
        previous={"accuracy": old},
        emitted=frozenset(),
        now=NOW,
    )
    duplicate, _ = evaluate_monitor(
        spec,
        result(),
        {"accuracy": old},
        previous={"accuracy": old},
        emitted=frozenset(stale.dedupe_keys),
        now=NOW + timedelta(minutes=1),
    )

    assert [signal.kind for signal in stale.signals] == ["metric_stale"]
    assert duplicate.signals == ()


def test_status_only_monitor_never_emits_metric_staleness():
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        registered_at=NOW - timedelta(hours=1),
    )

    evaluation, _ = evaluate_monitor(
        spec,
        result(),
        {},
        previous={},
        emitted=frozenset(),
        now=NOW,
    )

    assert evaluation.signals == ()


def test_three_metric_policies_evaluate_independently():
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        metrics=(
            MetricMonitorSpec(
                metric="val/loss",
                direction="min",
                gates=(MetricGate(operator="lte", threshold=0.2),),
            ),
            MetricMonitorSpec(
                metric="samples_per_second",
                gates=(MetricGate(operator="gte", threshold=100),),
            ),
            MetricMonitorSpec(metric="heartbeat", stale_after_seconds=60),
        ),
        registered_at=NOW,
    )

    evaluation, latest = evaluate_monitor(
        spec,
        result(),
        {
            "val/loss": MetricSample(value=0.19, observed_at=NOW),
            "samples_per_second": MetricSample(value=101, observed_at=NOW),
            "heartbeat": MetricSample(
                value=1,
                observed_at=NOW - timedelta(minutes=2),
            ),
        },
        previous={
            "val/loss": MetricSample(value=0.21, observed_at=NOW),
            "samples_per_second": MetricSample(value=99, observed_at=NOW),
        },
        emitted=frozenset(),
        now=NOW,
    )

    assert [(signal.kind, signal.metric) for signal in evaluation.signals] == [
        ("metric_gate", "val/loss"),
        ("metric_gate", "samples_per_second"),
        ("metric_stale", "heartbeat"),
    ]
    assert set(latest) == {"val/loss", "samples_per_second", "heartbeat"}


def test_metric_policy_count_and_names_are_validated():
    conversation_id = uuid4()
    with pytest.raises(ValueError, match="at most 3 items"):
        JobMonitorSpec(
            job_id="job-1",
            conversation_id=conversation_id,
            metrics=tuple(
                MetricMonitorSpec(metric=f"metric-{index}") for index in range(4)
            ),
        )
    with pytest.raises(ValueError, match="unique metric names"):
        JobMonitorSpec(
            job_id="job-1",
            conversation_id=conversation_id,
            metrics=(
                MetricMonitorSpec(metric="loss"),
                MetricMonitorSpec(metric="loss"),
            ),
        )


@pytest.mark.parametrize("removed_field", ["metric", "training_id"])
def test_removed_monitor_fields_are_rejected(removed_field: str):
    values = {
        "job_id": "job-1",
        "conversation_id": str(uuid4()),
        removed_field: "val/loss" if removed_field == "metric" else "job-1",
    }

    with pytest.raises(ValueError, match=removed_field):
        JobMonitorSpec.model_validate(values)


def test_changed_policy_gets_distinct_signal_identity():
    conversation_id = uuid4()
    first = JobMonitorSpec(
        job_id="run-1",
        conversation_id=conversation_id,
        metrics=(
            MetricMonitorSpec(
                metric="score",
                gates=(MetricGate(operator="gte", threshold=1),),
            ),
        ),
        registered_at=NOW,
    )
    changed = JobMonitorSpec(
        job_id=first.job_id,
        conversation_id=conversation_id,
        metrics=(
            MetricMonitorSpec(
                metric="score",
                gates=(MetricGate(operator="gte", threshold=2),),
            ),
        ),
        registered_at=NOW + timedelta(seconds=1),
    )

    first_signal, _ = evaluate_monitor(
        first,
        result(),
        {"score": MetricSample(value=1, observed_at=NOW)},
        previous={},
        emitted=frozenset(),
        now=NOW,
    )
    changed_signal, _ = evaluate_monitor(
        changed,
        result(),
        {"score": MetricSample(value=2, observed_at=NOW)},
        previous={},
        emitted=frozenset(),
        now=NOW,
    )
    same_policy_signal, _ = evaluate_monitor(
        JobMonitorSpec.model_validate(first.model_dump()),
        result(),
        {"score": MetricSample(value=1, observed_at=NOW)},
        previous={},
        emitted=frozenset(),
        now=NOW,
    )

    first_key = first_signal.dedupe_keys[0]
    changed_key = changed_signal.dedupe_keys[0]

    assert first_key.startswith("run-1:2026-07-30T00:00:00+00:00:")
    assert changed_key.startswith("run-1:2026-07-30T00:00:01+00:00:")
    assert "policy:" not in first_key
    assert first_signal.dedupe_keys != changed_signal.dedupe_keys
    assert first_signal.dedupe_keys == same_policy_signal.dedupe_keys


def test_terminal_signal_identity_is_job_and_state():
    spec = JobMonitorSpec(
        job_id="job-1",
        conversation_id=uuid4(),
        registered_at=NOW,
    )

    first, _ = evaluate_monitor(
        spec,
        result(JobState.FAILED),
        {},
        previous={},
        emitted=frozenset(),
        now=NOW,
    )
    duplicate, _ = evaluate_monitor(
        spec.model_copy(update={"registered_at": NOW + timedelta(hours=1)}),
        result(JobState.FAILED),
        {},
        previous={},
        emitted=frozenset(first.dedupe_keys),
        now=NOW + timedelta(hours=1),
    )

    assert first.dedupe_keys == ("job-1:status:failed",)
    assert duplicate.signals == ()
