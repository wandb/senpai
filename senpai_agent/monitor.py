"""Durable, programmatic training-monitor state and signal evaluation."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self
from uuid import UUID

from pydantic import Field

from senpai_agent.event_kinds import EventKind
from senpai_agent.models import Contract
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.training import TrainingResult, TrainingState


class MetricGate(Contract):
    """One threshold or change that should be surfaced to the student."""

    operator: Literal["lte", "gte", "improved_by", "regressed_by"]
    threshold: float


class TrainingMonitorSpec(Contract):
    """Durable monitoring policy for one training process and conversation."""

    training_id: str = Field(min_length=1)
    conversation_id: UUID
    metric: str | None = None
    direction: Literal["min", "max"] | None = None
    gates: tuple[MetricGate, ...] = ()
    poll_interval_seconds: float = Field(default=60, gt=0)
    stale_after_seconds: float = Field(default=600, gt=0)
    notify_on_status: frozenset[TrainingState] = Field(
        default_factory=lambda: frozenset(
            {
                TrainingState.FINISHED,
                TrainingState.FAILED,
                TrainingState.TIMED_OUT,
                TrainingState.CANCELLED,
            }
        )
    )
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, _context: object) -> None:
        if self.gates and self.metric is None:
            raise ValueError("metric gates require a metric")
        if (
            any(gate.operator in {"improved_by", "regressed_by"} for gate in self.gates)
            and self.direction is None
        ):
            raise ValueError("change gates require metric direction")


class MetricSample(Contract):
    value: float
    observed_at: datetime


class MonitorSignal(Contract):
    """Compact event handed back to the student conversation."""

    kind: Literal[
        "metric_gate",
        "metric_stale",
        "monitor_error",
        "training_status",
    ]
    dedupe_key: str
    training_id: str
    metric: str | None = None
    value: float | None = None
    state: TrainingState | None
    detail: str = Field(min_length=1, max_length=1_000)
    hard_failure: bool = False

class MonitorEvaluation(Contract):
    signals: tuple[MonitorSignal, ...] = ()

    @property
    def dedupe_keys(self) -> tuple[str, ...]:
        return tuple(signal.dedupe_key for signal in self.signals)


def evaluate_monitor(
    spec: TrainingMonitorSpec,
    result: TrainingResult,
    sample: MetricSample | None,
    *,
    previous: MetricSample | None,
    emitted: frozenset[str],
    baseline: MetricSample | None = None,
    now: datetime | None = None,
) -> tuple[MonitorEvaluation, MetricSample | None]:
    """Evaluate one poll without invoking a model."""

    now = (now or datetime.now(UTC)).astimezone(UTC)
    signals: list[MonitorSignal] = []

    if (
        result.state in spec.notify_on_status
        and result.state is not TrainingState.RUNNING
    ):
        key = f"{spec.training_id}:status:{result.state.value}"
        if key not in emitted:
            hard_failure = result.state in {
                TrainingState.FAILED,
                TrainingState.TIMED_OUT,
                TrainingState.CANCELLED,
            }
            signals.append(
                MonitorSignal(
                    kind="training_status",
                    dedupe_key=key,
                    training_id=spec.training_id,
                    state=result.state,
                    detail=(
                        f"Training reached terminal state {result.state.value}"
                        + (
                            f" with exit code {result.exit_code}."
                            if result.exit_code is not None
                            else "."
                        )
                    ),
                    hard_failure=hard_failure,
                )
            )

    if result.state is not TrainingState.RUNNING:
        return MonitorEvaluation(signals=tuple(signals)), sample or previous

    if sample is not None:
        for index, gate in enumerate(spec.gates):
            key = f"{spec.training_id}:gate:{index}"
            if key in emitted or not _gate_crossed(
                gate,
                spec.direction,
                previous,
                baseline,
                sample,
            ):
                continue
            signals.append(
                MonitorSignal(
                    kind="metric_gate",
                    dedupe_key=key,
                    training_id=spec.training_id,
                    metric=spec.metric,
                    value=sample.value,
                    state=result.state,
                    detail=(
                        f"{spec.metric} crossed {gate.operator} "
                        f"{gate.threshold:g} at {sample.value:g}."
                    ),
                )
            )
    if spec.metric is not None:
        latest = sample or previous
        last_update = (
            latest.observed_at.astimezone(UTC)
            if latest is not None
            else spec.registered_at.astimezone(UTC)
        )
        stale_key = f"{spec.training_id}:stale:{last_update.isoformat()}"
        age = (now - last_update).total_seconds()
        if age >= spec.stale_after_seconds and stale_key not in emitted:
            signals.append(
                MonitorSignal(
                    kind="metric_stale",
                    dedupe_key=stale_key,
                    training_id=spec.training_id,
                    metric=spec.metric,
                    value=latest.value if latest is not None else None,
                    state=result.state,
                    detail=(f"{spec.metric} has not updated for {round(age)} seconds."),
                )
            )

    return MonitorEvaluation(signals=tuple(signals)), sample or previous


def _gate_crossed(
    gate: MetricGate,
    direction: Literal["min", "max"] | None,
    previous: MetricSample | None,
    baseline: MetricSample | None,
    sample: MetricSample,
) -> bool:
    if gate.operator == "lte":
        return sample.value <= gate.threshold and (
            previous is None or previous.value > gate.threshold
        )
    if gate.operator == "gte":
        return sample.value >= gate.threshold and (
            previous is None or previous.value < gate.threshold
        )
    if baseline is None or direction is None:
        return False
    improvement = (
        baseline.value - sample.value
        if direction == "min"
        else sample.value - baseline.value
    )
    if gate.operator == "improved_by":
        return improvement >= gate.threshold
    return -improvement >= gate.threshold


class MonitorStore:
    """SQLite state plus a tiny JSON presence marker used by the Stop hook."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.marker_dir = path.parent / "monitors"
        self.marker_dir.mkdir(exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitors (
                    training_id TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    previous_sample_json TEXT,
                    baseline_sample_json TEXT,
                    next_poll_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_signals (
                    dedupe_key TEXT PRIMARY KEY,
                    training_id TEXT NOT NULL,
                    signal_json TEXT NOT NULL,
                    handled INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            monitor_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(monitors)")
            }
            if "baseline_sample_json" not in monitor_columns:
                self.connection.execute(
                    "ALTER TABLE monitors ADD COLUMN baseline_sample_json TEXT"
                )
            self.connection.execute(
                """
                UPDATE monitors
                SET baseline_sample_json = previous_sample_json
                WHERE baseline_sample_json IS NULL
                  AND previous_sample_json IS NOT NULL
                """
            )
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def register(self, spec: TrainingMonitorSpec) -> bool:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT spec_json, active FROM monitors
                WHERE training_id = ?
                """,
                (spec.training_id,),
            ).fetchone()
            existing = (
                TrainingMonitorSpec.model_validate_json(row[0])
                if row is not None
                else None
            )
            unchanged = (
                existing is not None
                and bool(row[1])
                and _same_monitor_policy(existing, spec)
            )
            effective = existing if unchanged else spec
            if not unchanged:
                with self.connection:
                    self.connection.execute(
                        """
                        INSERT INTO monitors (
                            training_id,
                            spec_json,
                            active,
                            previous_sample_json,
                            baseline_sample_json,
                            next_poll_at
                        )
                        VALUES (?, ?, 1, NULL, NULL, 0)
                        ON CONFLICT(training_id) DO UPDATE SET
                            spec_json = excluded.spec_json,
                            active = 1,
                            previous_sample_json = NULL,
                            baseline_sample_json = NULL,
                            next_poll_at = 0
                        """,
                        (spec.training_id, spec.model_dump_json()),
                    )
                    self.connection.execute(
                        "DELETE FROM monitor_signals WHERE training_id = ?",
                        (spec.training_id,),
                    )
            self._write_marker(effective)
            return not unchanged

    def _write_marker(self, spec: TrainingMonitorSpec) -> None:
        marker = self.marker_dir / f"{spec.training_id}.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(marker)

    def active(self) -> list[TrainingMonitorSpec]:
        rows = self.connection.execute(
            """
            SELECT spec_json FROM monitors
            WHERE active = 1
            ORDER BY rowid
            """
        ).fetchall()
        return [TrainingMonitorSpec.model_validate_json(row[0]) for row in rows]

    def spec(self, training_id: str) -> TrainingMonitorSpec:
        row = self.connection.execute(
            "SELECT spec_json FROM monitors WHERE training_id = ?",
            (training_id,),
        ).fetchone()
        if row is None:
            raise KeyError(training_id)
        return TrainingMonitorSpec.model_validate_json(row[0])

    def due(
        self,
        now: datetime | None = None,
    ) -> list[TrainingMonitorSpec]:
        timestamp = (now or datetime.now(UTC)).timestamp()
        rows = self.connection.execute(
            """
            SELECT spec_json FROM monitors
            WHERE active = 1 AND next_poll_at <= ?
            ORDER BY rowid
            """,
            (timestamp,),
        ).fetchall()
        return [TrainingMonitorSpec.model_validate_json(row[0]) for row in rows]

    def emitted(self, training_id: str) -> frozenset[str]:
        rows = self.connection.execute(
            "SELECT dedupe_key FROM monitor_signals WHERE training_id = ?",
            (training_id,),
        ).fetchall()
        return frozenset(row[0] for row in rows)

    def previous_sample(self, training_id: str) -> MetricSample | None:
        row = self.connection.execute(
            "SELECT previous_sample_json FROM monitors WHERE training_id = ?",
            (training_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return MetricSample.model_validate_json(row[0])

    def baseline_sample(self, training_id: str) -> MetricSample | None:
        row = self.connection.execute(
            "SELECT baseline_sample_json FROM monitors WHERE training_id = ?",
            (training_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return MetricSample.model_validate_json(row[0])

    def record_poll(
        self,
        spec: TrainingMonitorSpec,
        evaluation: MonitorEvaluation,
        sample: MetricSample | None,
        *,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(UTC)
        for signal in evaluation.signals:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO monitor_signals
                (dedupe_key, training_id, signal_json)
                VALUES (?, ?, ?)
                """,
                (
                    signal.dedupe_key,
                    spec.training_id,
                    signal.model_dump_json(),
                ),
            )
        self.connection.execute(
            """
            UPDATE monitors
            SET previous_sample_json = ?,
                baseline_sample_json = COALESCE(baseline_sample_json, ?),
                next_poll_at = ?
            WHERE training_id = ?
            """,
            (
                sample.model_dump_json() if sample is not None else None,
                sample.model_dump_json() if sample is not None else None,
                now.timestamp() + spec.poll_interval_seconds,
                spec.training_id,
            ),
        )
        self.connection.commit()

    def record_poll_error(
        self,
        spec: TrainingMonitorSpec,
        error: Exception,
        *,
        state: TrainingState | None,
        clear_previous_sample: bool = False,
        now: datetime | None = None,
    ) -> MonitorSignal | None:
        now = now or datetime.now(UTC)
        signal = MonitorSignal(
            kind="monitor_error",
            dedupe_key=(f"{spec.training_id}:monitor_error:{type(error).__name__}"),
            training_id=spec.training_id,
            metric=spec.metric,
            state=state,
            detail=_monitor_error_detail(error),
            hard_failure=True,
        )
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO monitor_signals
                (dedupe_key, training_id, signal_json)
                VALUES (?, ?, ?)
                """,
                (
                    signal.dedupe_key,
                    spec.training_id,
                    signal.model_dump_json(),
                ),
            )
            self.connection.execute(
                """
                UPDATE monitors
                SET previous_sample_json = CASE WHEN ? THEN NULL
                                               ELSE previous_sample_json END,
                    baseline_sample_json = CASE WHEN ? THEN NULL
                                               ELSE baseline_sample_json END,
                    next_poll_at = ?
                WHERE training_id = ?
                """,
                (
                    clear_previous_sample,
                    clear_previous_sample,
                    now.timestamp() + spec.poll_interval_seconds,
                    spec.training_id,
                ),
            )
        return signal if cursor.rowcount == 1 else None

    def pending_signals(self) -> list[MonitorSignal]:
        rows = self.connection.execute(
            """
            SELECT signal_json FROM monitor_signals
            WHERE handled = 0
            ORDER BY rowid
            """
        ).fetchall()
        return [MonitorSignal.model_validate_json(row[0]) for row in rows]

    def acknowledge(self, dedupe_key: str) -> None:
        self.connection.execute(
            "UPDATE monitor_signals SET handled = 1 WHERE dedupe_key = ?",
            (dedupe_key,),
        )
        self.connection.commit()

    def complete(self, training_id: str) -> None:
        self.connection.execute(
            "UPDATE monitors SET active = 0 WHERE training_id = ?",
            (training_id,),
        )
        self.connection.commit()
        (self.marker_dir / f"{training_id}.json").unlink(missing_ok=True)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class TrainingStatusSource(Protocol):
    def get_training_status(self, training_id: str) -> TrainingResult: ...


class MetricSource(Protocol):
    def latest(self, run_id: str, metric: str) -> MetricSample | None: ...


class WandbMetricSource:
    """Fetch one latest metric value without carrying history into the agent."""

    def __init__(self, entity: str, project: str, timeout_seconds: int = 30):
        self.entity = entity
        self.project = project
        self.timeout_seconds = timeout_seconds

    def latest(self, run_id: str, metric: str) -> MetricSample | None:
        import wandb

        run = wandb.Api(timeout=self.timeout_seconds).run(
            f"{self.entity}/{self.project}/{run_id}"
        )
        rows = run.history(
            keys=[metric, "_timestamp"],
            samples=2,
            pandas=False,
        )
        samples = [row for row in rows if row.get(metric) is not None]
        if not samples:
            return None
        latest = samples[-1]
        value = latest[metric]
        timestamp = latest.get("_timestamp")
        observed_at = (
            datetime.fromtimestamp(float(timestamp), UTC)
            if timestamp is not None
            else datetime.now(UTC)
        )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{metric} returned non-finite value {value}")
        return MetricSample(value=value, observed_at=observed_at)


class TrainingMonitorEngine:
    """Poll due monitors and persist only compact, deduplicated signals."""

    def __init__(
        self,
        store: MonitorStore,
        training: TrainingStatusSource,
        metrics: MetricSource,
    ):
        self.store = store
        self.training = training
        self.metrics = metrics

    def poll(self, now: datetime | None = None) -> tuple[MonitorSignal, ...]:
        now = now or datetime.now(UTC)
        produced: list[MonitorSignal] = []
        for spec in self.store.due(now):
            try:
                result = self.training.get_training_status(spec.training_id)
            except Exception as error:  # noqa: BLE001
                signal = self.store.record_poll_error(
                    spec,
                    error,
                    state=None,
                    now=now,
                )
                if signal is not None:
                    produced.append(signal)
                continue
            sample = None
            try:
                if (
                    result.state is TrainingState.RUNNING
                    and spec.metric
                    and result.wandb_run_ids
                ):
                    sample = self.metrics.latest(
                        result.wandb_run_ids[-1],
                        spec.metric,
                    )
            except Exception as error:  # noqa: BLE001
                signal = self.store.record_poll_error(
                    spec,
                    error,
                    state=result.state,
                    now=now,
                )
                if signal is not None:
                    produced.append(signal)
                continue
            try:
                previous = (
                    self.store.previous_sample(spec.training_id)
                    if result.state is TrainingState.RUNNING
                    else None
                )
                baseline = (
                    self.store.baseline_sample(spec.training_id)
                    if result.state is TrainingState.RUNNING
                    else None
                )
                evaluation, latest = evaluate_monitor(
                    spec,
                    result,
                    sample,
                    previous=previous,
                    emitted=self.store.emitted(spec.training_id),
                    baseline=baseline,
                    now=now,
                )
                self.store.record_poll(spec, evaluation, latest, now=now)
                if result.state is not TrainingState.RUNNING:
                    self.store.complete(spec.training_id)
            except Exception as error:  # noqa: BLE001
                signal = self.store.record_poll_error(
                    spec,
                    error,
                    state=result.state,
                    clear_previous_sample=True,
                    now=now,
                )
                if signal is not None:
                    produced.append(signal)
                continue
            produced.extend(evaluation.signals)
        return tuple(produced)


class MonitorMailbox:
    """Resume a student for every signal its monitor policy requested."""

    def __init__(self, engine: TrainingMonitorEngine, store: MonitorStore):
        self.engine = engine
        self.store = store

    def poll(self) -> tuple[ControllerEvent, ...]:
        self.engine.poll()
        return tuple(
            ControllerEvent(
                kind=EventKind.TRAINING_MONITOR,
                dedupe_key=signal.dedupe_key,
                payload={
                    "conversation_id": str(
                        self.store.spec(signal.training_id).conversation_id
                    ),
                    "training_id": signal.training_id,
                    "summary": signal.detail,
                    "reason": "The registered monitor policy emitted this signal.",
                    "signal": signal.model_dump(mode="json"),
                },
            )
            for signal in self.store.pending_signals()
        )

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        for key in dedupe_keys:
            self.store.acknowledge(key)


def _same_monitor_policy(
    left: TrainingMonitorSpec,
    right: TrainingMonitorSpec,
) -> bool:
    return left.model_dump(exclude={"registered_at"}) == right.model_dump(
        exclude={"registered_at"}
    )


def _monitor_error_detail(error: Exception) -> str:
    message = " ".join(str(error).split())
    prefix = f"Monitor poll failed ({type(error).__name__})"
    return f"{prefix}: {message}"[:1_000] if message else prefix
