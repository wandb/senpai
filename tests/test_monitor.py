from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from senpai_agent.monitor import (
    MetricGate,
    MetricSample,
    TrainingMonitorSpec,
    evaluate_monitor,
)
from senpai_agent.training import TrainingState


NOW = datetime(2026, 7, 30, tzinfo=UTC)


def result(state=TrainingState.RUNNING):
    return SimpleNamespace(
        state=state,
        exit_code=0 if state is TrainingState.FINISHED else None,
        kubernetes_released=None,
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
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="score",
        gates=(MetricGate(operator=operator, threshold=0.2),),
    )
    previous = MetricSample(value=before, observed_at=NOW)
    current = MetricSample(
        value=crossed,
        observed_at=NOW + timedelta(minutes=1),
    )

    quiet, _ = evaluate_monitor(
        spec,
        result(),
        previous,
        previous=None,
        emitted=frozenset(),
        now=NOW,
    )
    fired, _ = evaluate_monitor(
        spec,
        result(),
        current,
        previous=previous,
        emitted=frozenset(),
        now=NOW + timedelta(minutes=1),
    )
    duplicate, _ = evaluate_monitor(
        spec,
        result(),
        current,
        previous=previous,
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
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="score",
        direction=direction,
        gates=(MetricGate(operator=operator, threshold=0.08),),
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
        baseline,
        previous=None,
        baseline=None,
        emitted=frozenset(),
        now=NOW,
    )
    middle_poll, _ = evaluate_monitor(
        spec,
        result(),
        middle,
        previous=baseline,
        baseline=baseline,
        emitted=frozenset(),
        now=NOW + timedelta(minutes=1),
    )
    final, _ = evaluate_monitor(
        spec,
        result(),
        crossed,
        previous=middle,
        baseline=baseline,
        emitted=frozenset(),
        now=NOW + timedelta(minutes=2),
    )

    assert first.signals == ()
    assert middle_poll.signals == ()
    assert [signal.kind for signal in final.signals] == ["metric_gate"]


@pytest.mark.parametrize(
    ("state", "hard_failure"),
    [
        (TrainingState.FINISHED, False),
        (TrainingState.FAILED, True),
        (TrainingState.TIMED_OUT, True),
        (TrainingState.CANCELLED, True),
    ],
)
def test_terminal_status_preempts_metric_and_staleness_signals(
    state: TrainingState,
    hard_failure: bool,
):
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="accuracy",
        gates=(MetricGate(operator="gte", threshold=0.8),),
        stale_after_seconds=60,
    )
    old = MetricSample(value=0.7, observed_at=NOW - timedelta(minutes=2))

    evaluation, _ = evaluate_monitor(
        spec,
        result(state),
        MetricSample(value=0.9, observed_at=NOW - timedelta(minutes=2)),
        previous=old,
        emitted=frozenset(),
        now=NOW,
    )

    assert len(evaluation.signals) == 1
    signal = evaluation.signals[0]
    assert signal.kind == "training_status"
    assert signal.state is state
    assert signal.hard_failure is hard_failure
    assert signal.metric is None
    assert signal.value is None


def test_old_metric_sample_emits_one_stale_signal():
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="accuracy",
        stale_after_seconds=60,
    )
    old = MetricSample(value=0.7, observed_at=NOW - timedelta(minutes=2))

    stale, _ = evaluate_monitor(
        spec,
        result(),
        old,
        previous=old,
        emitted=frozenset(),
        now=NOW,
    )
    duplicate, _ = evaluate_monitor(
        spec,
        result(),
        MetricSample(value=0.8, observed_at=NOW),
        previous=old,
        emitted=frozenset(stale.dedupe_keys),
        now=NOW + timedelta(minutes=1),
    )

    assert [signal.kind for signal in stale.signals] == ["metric_stale"]
    assert duplicate.signals == ()


def test_status_only_monitor_never_emits_metric_staleness():
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        registered_at=NOW - timedelta(hours=1),
    )

    evaluation, _ = evaluate_monitor(
        spec,
        result(),
        None,
        previous=None,
        emitted=frozenset(),
        now=NOW,
    )

    assert evaluation.signals == ()
