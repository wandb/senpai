"""Durable, programmatic training-monitor state and signal evaluation."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self
from uuid import UUID

from pydantic import Field

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
    policy_revision: int = Field(default=1, ge=1)

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


class MetricRunNotFoundError(LookupError):
    """The metrics project is accessible, but the requested run does not exist."""


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
            signals.append(
                _terminal_signal(spec.training_id, result.state, result.exit_code)
            )

    if result.state is not TrainingState.RUNNING:
        return MonitorEvaluation(signals=tuple(signals)), sample or previous

    if sample is not None:
        for index, gate in enumerate(spec.gates):
            key = _policy_key(spec, f"gate:{index}")
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
        stale_key = _policy_key(spec, f"stale:{last_update.isoformat()}")
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


def _terminal_signal(
    training_id: str, state: TrainingState, exit_code: int | None
) -> MonitorSignal:
    return MonitorSignal(
        kind="training_status",
        dedupe_key=f"{training_id}:status:{state.value}",
        training_id=training_id,
        state=state,
        detail=f"Training reached terminal state {state.value}"
        + (f" with exit code {exit_code}." if exit_code is not None else "."),
        hard_failure=state
        in {TrainingState.FAILED, TrainingState.TIMED_OUT, TrainingState.CANCELLED},
    )


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
        self._lock = threading.RLock()
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
                row[1] for row in self.connection.execute("PRAGMA table_info(monitors)")
            }
            if "baseline_sample_json" not in monitor_columns:
                self.connection.execute(
                    "ALTER TABLE monitors ADD COLUMN baseline_sample_json TEXT"
                )
            signal_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(monitor_signals)")
            }
            for name in ("event_json", "observed_at", "superseded_reason"):
                if name not in signal_columns:
                    self.connection.execute(
                        f"ALTER TABLE monitor_signals ADD COLUMN {name} TEXT"
                    )
            # Preserve the exact legacy prompt shape. Ownership is immutable per run.
            for key, signal_json, spec_json in self.connection.execute(
                """SELECT s.dedupe_key, s.signal_json, m.spec_json
                   FROM monitor_signals s JOIN monitors m USING (training_id)
                   WHERE s.event_json IS NULL"""
            ).fetchall():
                self.connection.execute(
                    "UPDATE monitor_signals SET event_json = ? WHERE dedupe_key = ?",
                    (
                        json.dumps(
                            _signal_payload(
                                TrainingMonitorSpec.model_validate_json(spec_json),
                                MonitorSignal.model_validate_json(signal_json),
                            ),
                            sort_keys=True,
                        ),
                        key,
                    ),
                )
            self.connection.execute(
                """UPDATE monitor_signals
                   SET superseded_reason = 'monitor was already terminal at upgrade'
                   WHERE handled = 0 AND superseded_reason IS NULL
                     AND training_id IN (SELECT training_id FROM monitors WHERE active = 0)
                     AND json_extract(signal_json, '$.kind') = 'metric_stale'
                     AND json_extract(signal_json, '$.hard_failure') = 0"""
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
        with self._transaction():
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
            if existing is not None:
                if existing.conversation_id != spec.conversation_id:
                    raise PermissionError(
                        "training belongs to a different student conversation"
                    )
                if not row[1]:
                    return False
            unchanged = existing is not None and _same_monitor_policy(existing, spec)
            effective = (
                existing.model_copy(
                    update={
                        "poll_interval_seconds": spec.poll_interval_seconds,
                        "stale_after_seconds": spec.stale_after_seconds,
                    }
                )
                if unchanged
                else spec.model_copy(
                    update={
                        "policy_revision": existing.policy_revision + 1
                        if existing
                        else 1,
                    }
                )
            )
            if not unchanged:
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
                    (effective.training_id, effective.model_dump_json()),
                )
            else:
                self.connection.execute(
                    """UPDATE monitors SET spec_json = ?, next_poll_at = MIN(next_poll_at, ?)
                       WHERE training_id = ?""",
                    (
                        effective.model_dump_json(),
                        datetime.now(UTC).timestamp() + effective.poll_interval_seconds,
                        effective.training_id,
                    ),
                )
        self._sync_marker(effective.training_id)
        return not unchanged

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def _current(self, spec: TrainingMonitorSpec) -> bool:
        row = self.connection.execute(
            "SELECT active, spec_json FROM monitors WHERE training_id = ?",
            (spec.training_id,),
        ).fetchone()
        return bool(
            row
            and row[0]
            and TrainingMonitorSpec.model_validate_json(row[1]).policy_revision
            == spec.policy_revision
        )

    def _insert_signal(
        self, spec: TrainingMonitorSpec, signal: MonitorSignal, now: datetime
    ) -> bool:
        if signal.dedupe_key in self.emitted(spec.training_id):
            return False
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO monitor_signals
               (dedupe_key, training_id, signal_json, event_json, observed_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                signal.dedupe_key,
                spec.training_id,
                signal.model_dump_json(),
                json.dumps(_signal_payload(spec, signal), sort_keys=True),
                now.isoformat(),
            ),
        )
        return cursor.rowcount == 1

    def _sync_marker(self, training_id: str) -> None:
        # Serialize the file projection against policy changes in other connections.
        with self._transaction():
            row = self.connection.execute(
                "SELECT active, spec_json FROM monitors WHERE training_id = ?",
                (training_id,),
            ).fetchone()
            if row[0]:
                self._write_marker(TrainingMonitorSpec.model_validate_json(row[1]))
            else:
                (self.marker_dir / f"{training_id}.json").unlink(missing_ok=True)

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
        keys = {row[0] for row in rows}
        # Pre-upgrade rows belong to the retained policy. Preserve their keys,
        # but remember their slots so a newly versioned poll does not emit them twice.
        for key in tuple(keys):
            suffix = key.removeprefix(f"{training_id}:")
            if suffix.startswith(("gate:", "stale:", "monitor_error:")):
                keys.add(f"{training_id}:r1:{suffix}")
        return frozenset(keys)

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
    ) -> bool:
        now = now or datetime.now(UTC)
        with self._transaction():
            if not self._current(spec):
                return False
            for signal in evaluation.signals:
                self._insert_signal(spec, signal, now)
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
                    now.timestamp() + self.spec(spec.training_id).poll_interval_seconds,
                    spec.training_id,
                ),
            )
        return True

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
            dedupe_key=_policy_key(spec, f"monitor_error:{type(error).__name__}"),
            training_id=spec.training_id,
            metric=spec.metric,
            state=state,
            detail=_monitor_error_detail(error),
            hard_failure=True,
        )
        with self._transaction():
            if not self._current(spec):
                return None
            inserted = self._insert_signal(spec, signal, now)
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
                    now.timestamp() + self.spec(spec.training_id).poll_interval_seconds,
                    spec.training_id,
                ),
            )
        return signal if inserted else None

    def pending_signals(self) -> list[MonitorSignal]:
        rows = self.connection.execute(
            """
            SELECT signal_json FROM monitor_signals
            WHERE handled = 0 AND superseded_reason IS NULL
            ORDER BY rowid
            """
        ).fetchall()
        return [MonitorSignal.model_validate_json(row[0]) for row in rows]

    def pending_events(self) -> tuple[ControllerEvent, ...]:
        rows = self.connection.execute(
            """SELECT dedupe_key, event_json FROM monitor_signals
               WHERE handled = 0 AND superseded_reason IS NULL ORDER BY rowid"""
        ).fetchall()
        return tuple(
            ControllerEvent(
                kind="training_monitor", dedupe_key=key, payload=json.loads(payload)
            )
            for key, payload in rows
        )

    def acknowledge(self, dedupe_key: str) -> None:
        with self._transaction():
            self.connection.execute(
                "UPDATE monitor_signals SET handled = 1 WHERE dedupe_key = ?",
                (dedupe_key,),
            )

    def complete(
        self,
        training_id: str,
        *,
        state: TrainingState,
        exit_code: int | None = None,
        notify: bool = True,
    ) -> MonitorSignal | None:
        if state is TrainingState.RUNNING:
            raise ValueError("cannot complete a running monitor")
        with self._transaction():
            spec = self.spec(training_id)
            signal = _terminal_signal(training_id, state, exit_code)
            inserted = (
                notify
                and state in spec.notify_on_status
                and self._insert_signal(spec, signal, datetime.now(UTC))
            )
            self.connection.execute(
                "UPDATE monitors SET active = 0 WHERE training_id = ?",
                (training_id,),
            )
            self.connection.execute(
                """UPDATE monitor_signals SET superseded_reason = ?
                   WHERE training_id = ? AND handled = 0 AND superseded_reason IS NULL
                     AND json_extract(signal_json, '$.kind') = 'metric_stale'
                     AND json_extract(signal_json, '$.hard_failure') = 0""",
                (f"training reached {state.value}", training_id),
            )
        self._sync_marker(training_id)
        return signal if inserted else None

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

        runs = wandb.Api(timeout=self.timeout_seconds).runs(
            f"{self.entity}/{self.project}",
            filters={"name": run_id},
            per_page=1,
        )
        run = next(iter(runs), None)
        if run is None:
            raise MetricRunNotFoundError(
                f"W&B run {self.entity}/{self.project}/{run_id} does not exist"
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
                    try:
                        sample = self.metrics.latest(
                            result.wandb_run_ids[-1],
                            spec.metric,
                        )
                    except MetricRunNotFoundError:
                        if self.store.baseline_sample(spec.training_id) is not None:
                            raise
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
            if result.state is not TrainingState.RUNNING:
                signal = self.store.complete(
                    spec.training_id, state=result.state, exit_code=result.exit_code
                )
                if signal is not None:
                    produced.append(signal)
                continue
            try:
                previous = self.store.previous_sample(spec.training_id)
                baseline = self.store.baseline_sample(spec.training_id)
                evaluation, latest = evaluate_monitor(
                    spec,
                    result,
                    sample,
                    previous=previous,
                    emitted=self.store.emitted(spec.training_id),
                    baseline=baseline,
                    now=now,
                )
                accepted = self.store.record_poll(spec, evaluation, latest, now=now)
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
            if accepted:
                produced.extend(evaluation.signals)
        return tuple(produced)


class MonitorMailbox:
    """Resume a student for every signal its monitor policy requested."""

    def __init__(self, engine: TrainingMonitorEngine, store: MonitorStore):
        self.engine = engine
        self.store = store

    def poll(self) -> tuple[ControllerEvent, ...]:
        self.engine.poll()
        return self.store.pending_events()

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        for key in dedupe_keys:
            self.store.acknowledge(key)


def _policy_key(spec: TrainingMonitorSpec, suffix: str) -> str:
    return f"{spec.training_id}:r{spec.policy_revision}:{suffix}"


def _signal_payload(
    spec: TrainingMonitorSpec, signal: MonitorSignal
) -> dict[str, object]:
    return {
        "conversation_id": str(spec.conversation_id),
        "training_id": signal.training_id,
        "summary": signal.detail,
        "reason": "The registered monitor policy emitted this signal.",
        "signal": signal.model_dump(mode="json"),
    }


def _same_monitor_policy(left: TrainingMonitorSpec, right: TrainingMonitorSpec) -> bool:
    cadence = {
        "registered_at",
        "policy_revision",
        "poll_interval_seconds",
        "stale_after_seconds",
    }
    return left.model_dump(exclude=cadence) == right.model_dump(exclude=cadence)


def _monitor_error_detail(error: Exception) -> str:
    message = " ".join(str(error).split())
    prefix = f"Monitor poll failed ({type(error).__name__})"
    return f"{prefix}: {message}"[:1_000] if message else prefix
