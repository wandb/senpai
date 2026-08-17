"""Durable, programmatic job-monitor state and signal evaluation."""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self
from uuid import UUID

from pydantic import Field

from senpai_agent.jobs import JobState
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import Contract


class MetricGate(Contract):
    """One W&B threshold or change surfaced to the owning conversation."""

    operator: Literal["lte", "gte", "improved_by", "regressed_by"]
    threshold: float


class MetricMonitorSpec(Contract):
    """Independent monitoring policy for one W&B metric."""

    metric: str = Field(
        min_length=1,
        description="Exact W&B metric key to monitor.",
    )
    direction: Literal["min", "max"] | None = Field(
        default=None,
        description="Optimization direction; required only for change gates.",
    )
    gates: tuple[MetricGate, ...] = Field(
        default=(),
        description="Thresholds or changes that emit one deduplicated signal.",
    )
    stale_after_seconds: float = Field(
        default=600,
        gt=0,
        description="Stale timeout; choose longer than the logging cadence.",
    )

    def model_post_init(self, _context: object) -> None:
        if (
            any(gate.operator in {"improved_by", "regressed_by"} for gate in self.gates)
            and self.direction is None
        ):
            raise ValueError("change gates require metric direction")


class JobMonitorSpec(Contract):
    """Durable monitoring policy for one job and conversation."""

    job_id: str = Field(min_length=1)
    conversation_id: UUID
    wandb_run_id: str | None = Field(
        default=None,
        min_length=1,
        description="W&B run bound to every metric policy for this registration.",
    )
    metrics: tuple[MetricMonitorSpec, ...] = Field(default=(), max_length=3)
    poll_interval_seconds: float = Field(default=60, ge=5)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, _context: object) -> None:
        names = [policy.metric for policy in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric policies must use unique metric names")
        if self.metrics and self.wandb_run_id is None:
            raise ValueError("metric policies require a W&B run ID")


@dataclass(frozen=True, slots=True)
class JobMonitorClaim:
    spec: JobMonitorSpec
    generation: int


class MetricSample(Contract):
    value: float
    observed_at: datetime


class MonitorSignal(Contract):
    """Compact internal signal handed back to the owning conversation."""

    kind: Literal[
        "metric_gate",
        "metric_stale",
        "monitor_error",
        "job_status",
    ]
    dedupe_key: str
    job_id: str
    conversation_id: UUID
    wandb_run_id: str | None = None
    metric: str | None = None
    value: float | None = None
    state: JobState | None = None
    detail: str = Field(min_length=1, max_length=1_000)
    hard_failure: bool = False


class MonitorEvaluation(Contract):
    signals: tuple[MonitorSignal, ...] = ()

    @property
    def dedupe_keys(self) -> tuple[str, ...]:
        return tuple(signal.dedupe_key for signal in self.signals)


class MonitoredJobStatus(Protocol):
    state: JobState
    exit_code: int | None


def evaluate_monitor(
    spec: JobMonitorSpec,
    result: MonitoredJobStatus,
    samples: Mapping[str, MetricSample | None],
    *,
    previous: Mapping[str, MetricSample | None],
    emitted: frozenset[str],
    baseline: Mapping[str, MetricSample | None] | None = None,
    unavailable_metrics: frozenset[str] = frozenset(),
    now: datetime | None = None,
) -> tuple[MonitorEvaluation, dict[str, MetricSample | None]]:
    """Evaluate one poll without invoking a model."""

    now = (now or datetime.now(UTC)).astimezone(UTC)
    sample_by_metric = dict(samples)
    previous_by_metric = dict(previous)
    baseline_by_metric = dict(baseline or {})
    latest_by_metric = {
        policy.metric: (
            None
            if policy.metric in unavailable_metrics
            else sample_by_metric.get(policy.metric)
            or previous_by_metric.get(policy.metric)
        )
        for policy in spec.metrics
    }

    if result.state is not JobState.RUNNING:
        signal = _terminal_signal(spec, result)
        return (
            MonitorEvaluation(
                signals=() if signal.dedupe_key in emitted else (signal,)
            ),
            latest_by_metric,
        )

    signals: list[MonitorSignal] = []
    for policy in spec.metrics:
        if policy.metric in unavailable_metrics:
            continue
        sample = sample_by_metric.get(policy.metric)
        previous_sample = previous_by_metric.get(policy.metric)
        if sample is not None:
            for index, gate in enumerate(policy.gates):
                key = _signal_key(spec, f"metric:{policy.metric}:gate:{index}")
                if key in emitted or not _gate_crossed(
                    gate,
                    policy.direction,
                    previous_sample,
                    baseline_by_metric.get(policy.metric),
                    sample,
                ):
                    continue
                signals.append(
                    MonitorSignal(
                        kind="metric_gate",
                        dedupe_key=key,
                        job_id=spec.job_id,
                        conversation_id=spec.conversation_id,
                        wandb_run_id=spec.wandb_run_id,
                        metric=policy.metric,
                        value=sample.value,
                        state=result.state,
                        detail=(
                            f"{policy.metric} crossed {gate.operator} "
                            f"{gate.threshold:g} at {sample.value:g}."
                        ),
                    )
                )

        latest = latest_by_metric[policy.metric]
        last_update = (
            latest.observed_at.astimezone(UTC)
            if latest is not None
            else spec.registered_at.astimezone(UTC)
        )
        stale_key = _signal_key(
            spec,
            f"metric:{policy.metric}:stale:{last_update.isoformat()}",
        )
        age = (now - last_update).total_seconds()
        if age >= policy.stale_after_seconds and stale_key not in emitted:
            signals.append(
                MonitorSignal(
                    kind="metric_stale",
                    dedupe_key=stale_key,
                    job_id=spec.job_id,
                    conversation_id=spec.conversation_id,
                    wandb_run_id=spec.wandb_run_id,
                    metric=policy.metric,
                    value=latest.value if latest is not None else None,
                    state=result.state,
                    detail=(
                        f"{policy.metric} has not updated for {round(age)} seconds."
                    ),
                )
            )

    return MonitorEvaluation(signals=tuple(signals)), latest_by_metric


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


def _signal_key(spec: JobMonitorSpec, suffix: str) -> str:
    registered_at = spec.registered_at.astimezone(UTC).isoformat()
    return f"{spec.job_id}:{registered_at}:{suffix}"


def _terminal_signal(
    spec: JobMonitorSpec,
    result: MonitoredJobStatus,
) -> MonitorSignal:
    hard_failure = result.state in {
        JobState.FAILED,
        JobState.TIMED_OUT,
        JobState.CANCELLED,
    }
    return MonitorSignal(
        kind="job_status",
        dedupe_key=f"{spec.job_id}:status:{result.state.value}",
        job_id=spec.job_id,
        conversation_id=spec.conversation_id,
        wandb_run_id=spec.wandb_run_id,
        state=result.state,
        detail=(
            f"Job reached terminal state {result.state.value}"
            + (
                f" with exit code {result.exit_code}."
                if result.exit_code is not None
                else "."
            )
        ),
        hard_failure=hard_failure,
    )


class JobMonitorStore:
    """Thread-safe SQLite state for durable job monitoring."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        with self._transaction():
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitors (
                    job_id TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    next_poll_at REAL NOT NULL DEFAULT 0,
                    poll_generation INTEGER NOT NULL DEFAULT 0,
                    poll_claimed_until REAL NOT NULL DEFAULT 0
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_signals (
                    dedupe_key TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    signal_json TEXT NOT NULL,
                    handled INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_metric_samples (
                    job_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    previous_sample_json TEXT,
                    baseline_sample_json TEXT,
                    PRIMARY KEY (job_id, metric)
                )
                """
            )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.connection.rollback()
                raise
            else:
                self.connection.commit()

    def register(self, spec: JobMonitorSpec) -> bool:
        """Register a policy, preserving state for cadence-only updates."""

        with self._transaction():
            row = self.connection.execute(
                """
                SELECT spec_json, active, next_poll_at FROM monitors
                WHERE job_id = ?
                """,
                (spec.job_id,),
            ).fetchone()
            existing = (
                JobMonitorSpec.model_validate_json(row[0])
                if row is not None
                else None
            )
            unchanged = (
                existing is not None
                and bool(row[1])
                and _same_monitor_policy(existing, spec)
            )
            if unchanged:
                effective = spec.model_copy(
                    update={"registered_at": existing.registered_at}
                )
                next_poll_at = _reschedule(
                    float(row[2]),
                    existing.poll_interval_seconds,
                    effective.poll_interval_seconds,
                )
                self.connection.execute(
                    """
                    UPDATE monitors
                    SET spec_json = ?, next_poll_at = ?
                    WHERE job_id = ?
                    """,
                    (effective.model_dump_json(), next_poll_at, spec.job_id),
                )
                return False

            effective = _new_registration(spec, existing)
            if row is None:
                self.connection.execute(
                    """
                    INSERT INTO monitors (job_id, spec_json, active, next_poll_at)
                    VALUES (?, ?, 1, 0)
                    """,
                    (spec.job_id, effective.model_dump_json()),
                )
            else:
                self.connection.execute(
                    """
                    UPDATE monitors
                    SET spec_json = ?,
                        active = 1,
                        next_poll_at = 0,
                        poll_generation = poll_generation + 1,
                        poll_claimed_until = 0
                    WHERE job_id = ?
                    """,
                    (effective.model_dump_json(), spec.job_id),
                )
            self.connection.execute(
                "DELETE FROM monitor_metric_samples WHERE job_id = ?",
                (spec.job_id,),
            )
            terminal_prefix = f"{spec.job_id}:status:"
            self.connection.execute(
                """
                DELETE FROM monitor_signals
                WHERE job_id = ?
                  AND substr(dedupe_key, 1, length(?)) != ?
                """,
                (spec.job_id, terminal_prefix, terminal_prefix),
            )
            return True

    def active(self) -> list[JobMonitorSpec]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT spec_json FROM monitors
                WHERE active = 1
                ORDER BY rowid
                """
            ).fetchall()
        return [JobMonitorSpec.model_validate_json(row[0]) for row in rows]

    def spec(self, job_id: str) -> JobMonitorSpec:
        with self._lock:
            row = self.connection.execute(
                "SELECT spec_json FROM monitors WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobMonitorSpec.model_validate_json(row[0])

    def due(
        self,
        now: datetime | None = None,
        *,
        limit: int = 32,
    ) -> list[JobMonitorSpec]:
        if limit <= 0:
            raise ValueError("monitor due limit must be positive")
        timestamp = (now or datetime.now(UTC)).timestamp()
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT spec_json FROM monitors
                WHERE active = 1
                  AND next_poll_at <= ?
                  AND poll_claimed_until <= ?
                ORDER BY next_poll_at, rowid
                LIMIT ?
                """,
                (timestamp, timestamp, limit),
            ).fetchall()
        return [JobMonitorSpec.model_validate_json(row[0]) for row in rows]

    def claim_due(
        self,
        now: datetime | None = None,
        *,
        lease_seconds: float = 300,
        exclude_job_ids: frozenset[str] = frozenset(),
    ) -> JobMonitorClaim | None:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("monitor poll lease must be finite and positive")
        timestamp = (now or datetime.now(UTC)).timestamp()
        excluded = ""
        parameters: list[object] = [timestamp, timestamp]
        if exclude_job_ids:
            placeholders = ",".join("?" for _ in exclude_job_ids)
            excluded = f"AND job_id NOT IN ({placeholders})"
            parameters.extend(sorted(exclude_job_ids))
        with self._transaction():
            row = self.connection.execute(
                f"""
                SELECT job_id, spec_json, poll_generation
                FROM monitors
                WHERE active = 1
                  AND next_poll_at <= ?
                  AND poll_claimed_until <= ?
                  {excluded}
                ORDER BY next_poll_at, rowid
                LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None
            generation = int(row[2]) + 1
            self.connection.execute(
                """
                UPDATE monitors
                SET poll_generation = ?, poll_claimed_until = ?
                WHERE job_id = ?
                """,
                (generation, timestamp + lease_seconds, row[0]),
            )
        return JobMonitorClaim(
            spec=JobMonitorSpec.model_validate_json(row[1]),
            generation=generation,
        )

    def seconds_until_next_poll(self, now: datetime | None = None) -> float | None:
        timestamp = (now or datetime.now(UTC)).timestamp()
        with self._lock:
            row = self.connection.execute(
                """
                SELECT MIN(MAX(next_poll_at, poll_claimed_until))
                FROM monitors
                WHERE active = 1
                """
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return max(0.0, float(row[0]) - timestamp)

    def emitted(self, job_id: str) -> frozenset[str]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT dedupe_key FROM monitor_signals WHERE job_id = ?",
                (job_id,),
            ).fetchall()
            return frozenset(row[0] for row in rows)

    def previous_sample(self, job_id: str, metric: str) -> MetricSample | None:
        return self._sample(job_id, metric, "previous_sample_json")

    def baseline_sample(self, job_id: str, metric: str) -> MetricSample | None:
        return self._sample(job_id, metric, "baseline_sample_json")

    def _sample(
        self,
        job_id: str,
        metric: str,
        column: Literal["previous_sample_json", "baseline_sample_json"],
    ) -> MetricSample | None:
        with self._lock:
            row = self.connection.execute(
                f"""
                SELECT {column} FROM monitor_metric_samples
                WHERE job_id = ? AND metric = ?
                """,
                (job_id, metric),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return MetricSample.model_validate_json(row[0])

    def record_poll(
        self,
        spec: JobMonitorSpec,
        evaluation: MonitorEvaluation,
        samples: Mapping[str, MetricSample | None],
        *,
        generation: int | None = None,
        now: datetime | None = None,
    ) -> bool:
        now = now or datetime.now(UTC)
        with self._transaction():
            current = self._current_registration(spec, generation=generation)
            if current is None:
                return False
            for signal in evaluation.signals:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO monitor_signals
                    (dedupe_key, job_id, signal_json)
                    VALUES (?, ?, ?)
                    """,
                    (signal.dedupe_key, signal.job_id, signal.model_dump_json()),
                )
            for metric, sample in samples.items():
                if sample is None:
                    continue
                sample_json = sample.model_dump_json()
                self.connection.execute(
                    """
                    INSERT INTO monitor_metric_samples (
                        job_id,
                        metric,
                        previous_sample_json,
                        baseline_sample_json
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(job_id, metric) DO UPDATE SET
                        previous_sample_json = excluded.previous_sample_json,
                        baseline_sample_json = COALESCE(
                            monitor_metric_samples.baseline_sample_json,
                            excluded.baseline_sample_json
                        )
                    """,
                    (spec.job_id, metric, sample_json, sample_json),
                )
            self.connection.execute(
                """
                UPDATE monitors
                SET next_poll_at = ?, poll_claimed_until = 0
                WHERE job_id = ?
                """,
                (
                    now.timestamp() + current.poll_interval_seconds,
                    spec.job_id,
                ),
            )
        return True

    def record_poll_error(
        self,
        spec: JobMonitorSpec,
        error: Exception,
        *,
        state: JobState | None,
        metric: str | None = None,
        clear_metric_sample: bool = False,
        generation: int | None = None,
        now: datetime | None = None,
    ) -> MonitorSignal | None:
        now = now or datetime.now(UTC)
        signal = MonitorSignal(
            kind="monitor_error",
            dedupe_key=_signal_key(
                spec,
                f"monitor_error:{metric or 'status'}:{type(error).__name__}",
            ),
            job_id=spec.job_id,
            conversation_id=spec.conversation_id,
            wandb_run_id=spec.wandb_run_id,
            metric=metric,
            state=state,
            detail=_monitor_error_detail(error),
            hard_failure=True,
        )
        with self._transaction():
            current = self._current_registration(spec, generation=generation)
            if current is None:
                return None
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO monitor_signals
                (dedupe_key, job_id, signal_json)
                VALUES (?, ?, ?)
                """,
                (
                    signal.dedupe_key,
                    spec.job_id,
                    signal.model_dump_json(),
                ),
            )
            if clear_metric_sample and metric is not None:
                self.connection.execute(
                    """
                    DELETE FROM monitor_metric_samples
                    WHERE job_id = ? AND metric = ?
                    """,
                    (spec.job_id, metric),
                )
            self.connection.execute(
                """
                UPDATE monitors
                SET next_poll_at = ?, poll_claimed_until = 0
                WHERE job_id = ?
                """,
                (
                    now.timestamp() + current.poll_interval_seconds,
                    spec.job_id,
                ),
            )
        return signal if cursor.rowcount == 1 else None

    def pending_signals(self) -> list[MonitorSignal]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT signal_json FROM monitor_signals
                WHERE handled = 0
                ORDER BY rowid
                """
            ).fetchall()
        return [MonitorSignal.model_validate_json(row[0]) for row in rows]

    def acknowledge(self, dedupe_key: str) -> None:
        with self._transaction():
            self.connection.execute(
                "UPDATE monitor_signals SET handled = 1 WHERE dedupe_key = ?",
                (dedupe_key,),
            )

    def record_terminal_and_complete(
        self,
        result: MonitoredJobStatus,
        *,
        spec: JobMonitorSpec | None = None,
        generation: int | None = None,
    ) -> MonitorSignal | None:
        """Atomically persist a terminal observation and retire its monitor."""

        if result.state is JobState.RUNNING:
            raise ValueError("cannot complete a monitor with a running job")
        job_id = getattr(result, "job_id", None) or (spec.job_id if spec else None)
        if not isinstance(job_id, str):
            raise ValueError("terminal job status has no job_id")
        with self._transaction():
            row = self.connection.execute(
                """
                SELECT spec_json, active, poll_generation
                FROM monitors
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            current = JobMonitorSpec.model_validate_json(row[0])
            if spec is not None and (
                not row[1] or not _same_registration(current, spec)
            ):
                return None
            if generation is not None and int(row[2]) != generation:
                return None
            signal = _terminal_signal(current, result)
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO monitor_signals
                (dedupe_key, job_id, signal_json)
                VALUES (?, ?, ?)
                """,
                (signal.dedupe_key, job_id, signal.model_dump_json()),
            )
            self.connection.execute(
                """
                UPDATE monitors
                SET active = 0, poll_claimed_until = 0
                WHERE job_id = ?
                """,
                (job_id,),
            )
        return signal if cursor.rowcount == 1 else None

    def discard(self, job_id: str) -> None:
        """Retire a nonterminal registration without inventing an observation."""

        with self._transaction():
            self.connection.execute(
                """
                UPDATE monitors
                SET active = 0,
                    poll_generation = poll_generation + 1,
                    poll_claimed_until = 0
                WHERE job_id = ?
                """,
                (job_id,),
            )

    def _current_registration(
        self,
        expected: JobMonitorSpec,
        *,
        generation: int | None = None,
    ) -> JobMonitorSpec | None:
        row = self.connection.execute(
            """
            SELECT spec_json, active, poll_generation
            FROM monitors
            WHERE job_id = ?
            """,
            (expected.job_id,),
        ).fetchone()
        if row is None or not row[1]:
            return None
        if generation is not None and int(row[2]) != generation:
            return None
        current = JobMonitorSpec.model_validate_json(row[0])
        return current if _same_registration(current, expected) else None

    def close(self) -> None:
        with self._lock:
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


class JobStatusSource(Protocol):
    def get_job_status(self, job_id: str) -> MonitoredJobStatus: ...


class MetricSource(Protocol):
    def latest(self, run_id: str, metric: str) -> MetricSample | None: ...


class WandbRunState(Protocol):
    state: str


@dataclass(frozen=True, slots=True)
class WandbJobStatus:
    job_id: str
    state: JobState
    exit_code: int | None


class WandbJobStatusSource:
    """Read one configured-project W&B run without process authority."""

    def __init__(
        self,
        entity: str,
        project: str,
        timeout_seconds: int = 30,
        *,
        api_key: str | None = None,
    ):
        self.entity = entity
        self.project = project
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def get_job_status(self, job_id: str) -> WandbJobStatus:
        state = str(self._run(job_id).state).lower()
        if state in {"running", "pending", "queued", "preempting", "preempted"}:
            job_state = JobState.RUNNING
        elif state == "finished":
            job_state = JobState.FINISHED
        elif state in {"failed", "crashed"}:
            job_state = JobState.FAILED
        elif state == "killed":
            job_state = JobState.CANCELLED
        else:
            raise ValueError(f"unsupported W&B run state {state!r}")
        return WandbJobStatus(
            job_id=job_id,
            state=job_state,
            exit_code=None,
        )

    def _run(self, run_id: str) -> WandbRunState:
        import wandb

        return wandb.Api(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        ).run(f"{self.entity}/{self.project}/{run_id}")


class WandbMetricSource:
    """Fetch one latest metric value without carrying history into the agent."""

    def __init__(
        self,
        entity: str,
        project: str,
        timeout_seconds: int = 20,
        *,
        api_key: str | None = None,
    ):
        self.entity = entity
        self.project = project
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def latest(self, run_id: str, metric: str) -> MetricSample | None:
        import wandb

        run = wandb.Api(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        ).run(f"{self.entity}/{self.project}/{run_id}")
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


class JobMonitorEngine:
    """Poll due job monitors and persist only compact, deduplicated signals."""

    def __init__(
        self,
        store: JobMonitorStore,
        jobs: JobStatusSource,
        metrics: MetricSource,
        *,
        max_polls_per_cycle: int = 4,
        poll_budget_seconds: float = 120,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if max_polls_per_cycle <= 0:
            raise ValueError("max_polls_per_cycle must be positive")
        if not math.isfinite(poll_budget_seconds) or poll_budget_seconds <= 0:
            raise ValueError("poll_budget_seconds must be finite and positive")
        self.store = store
        self.jobs = jobs
        self.metrics = metrics
        self.max_polls_per_cycle = max_polls_per_cycle
        self.poll_budget_seconds = poll_budget_seconds
        self.monotonic = monotonic

    def poll(self, now: datetime | None = None) -> tuple[MonitorSignal, ...]:
        now = now or datetime.now(UTC)
        produced: list[MonitorSignal] = []
        claimed_job_ids: set[str] = set()
        deadline = self.monotonic() + self.poll_budget_seconds
        for index in range(self.max_polls_per_cycle):
            if index and self.monotonic() >= deadline:
                break
            claim = self.store.claim_due(
                now,
                lease_seconds=max(300, self.poll_budget_seconds),
                exclude_job_ids=frozenset(claimed_job_ids),
            )
            if claim is None:
                break
            spec = claim.spec
            claimed_job_ids.add(spec.job_id)
            try:
                result = self.jobs.get_job_status(spec.job_id)
            except Exception as error:  # noqa: BLE001
                signal = self.store.record_poll_error(
                    spec,
                    error,
                    state=None,
                    generation=claim.generation,
                    now=now,
                )
                if signal is not None:
                    produced.append(signal)
                continue

            if result.state is not JobState.RUNNING:
                signal = self.store.record_terminal_and_complete(
                    result,
                    spec=spec,
                    generation=claim.generation,
                )
                if signal is not None:
                    produced.append(signal)
                continue

            samples: dict[str, MetricSample | None] = {}
            previous: dict[str, MetricSample | None] = {}
            baseline: dict[str, MetricSample | None] = {}
            unavailable: set[str] = set()
            wandb_run_id = spec.wandb_run_id
            if spec.metrics:
                assert wandb_run_id is not None
            for policy in spec.metrics:
                try:
                    samples[policy.metric] = self.metrics.latest(
                        wandb_run_id,
                        policy.metric,
                    )
                except Exception as error:  # noqa: BLE001
                    unavailable.add(policy.metric)
                    signal = self.store.record_poll_error(
                        spec,
                        error,
                        state=result.state,
                        metric=policy.metric,
                        generation=claim.generation,
                        now=now,
                    )
                    if signal is not None:
                        produced.append(signal)
                    continue
                try:
                    previous[policy.metric] = self.store.previous_sample(
                        spec.job_id,
                        policy.metric,
                    )
                    baseline[policy.metric] = self.store.baseline_sample(
                        spec.job_id,
                        policy.metric,
                    )
                except Exception as error:  # noqa: BLE001
                    unavailable.add(policy.metric)
                    signal = self.store.record_poll_error(
                        spec,
                        error,
                        state=result.state,
                        metric=policy.metric,
                        clear_metric_sample=True,
                        generation=claim.generation,
                        now=now,
                    )
                    if signal is not None:
                        produced.append(signal)

            try:
                evaluation, latest = evaluate_monitor(
                    spec,
                    result,
                    samples,
                    previous=previous,
                    emitted=self.store.emitted(spec.job_id),
                    baseline=baseline,
                    unavailable_metrics=frozenset(unavailable),
                    now=now,
                )
                if self.store.record_poll(
                    spec,
                    evaluation,
                    latest,
                    generation=claim.generation,
                    now=now,
                ):
                    produced.extend(evaluation.signals)
            except Exception as error:  # noqa: BLE001
                signal = self.store.record_poll_error(
                    spec,
                    error,
                    state=result.state,
                    generation=claim.generation,
                    now=now,
                )
                if signal is not None:
                    produced.append(signal)
        return tuple(
            signal
            for signal in produced
            if signal.dedupe_key in self.store.emitted(signal.job_id)
        )


class JobMonitorMailbox:
    """Resume an agent for each durable job-monitor signal."""

    def __init__(self, engine: JobMonitorEngine, store: JobMonitorStore):
        self.engine = engine
        self.store = store

    def poll(self) -> tuple[ControllerEvent, ...]:
        self.engine.poll()
        events: list[ControllerEvent] = []
        for signal in self.store.pending_signals():
            payload = {
                "conversation_id": str(signal.conversation_id),
                "job_id": signal.job_id,
                "summary": signal.detail,
                "reason": "The registered job monitor emitted this signal.",
                "signal": signal.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude={"conversation_id", "wandb_run_id"},
                ),
            }
            if signal.wandb_run_id is not None:
                payload["wandb_run_id"] = signal.wandb_run_id
            events.append(
                ControllerEvent(
                    kind="job_monitor",
                    dedupe_key=signal.dedupe_key,
                    payload=payload,
                )
            )
        return tuple(events)

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        for key in dedupe_keys:
            self.store.acknowledge(key)


def _same_monitor_policy(
    left: JobMonitorSpec,
    right: JobMonitorSpec,
) -> bool:
    excluded = {"registered_at", "poll_interval_seconds"}
    return left.model_dump(exclude=excluded) == right.model_dump(exclude=excluded)


def _same_registration(left: JobMonitorSpec, right: JobMonitorSpec) -> bool:
    return left.job_id == right.job_id and left.registered_at == right.registered_at


def _new_registration(
    requested: JobMonitorSpec,
    previous: JobMonitorSpec | None,
) -> JobMonitorSpec:
    if previous is None or requested.registered_at != previous.registered_at:
        return requested
    registered_at = datetime.now(UTC)
    if registered_at == previous.registered_at:
        registered_at += timedelta(microseconds=1)
    return requested.model_copy(update={"registered_at": registered_at})


def _reschedule(
    next_poll_at: float,
    previous_interval: float,
    new_interval: float,
) -> float:
    if next_poll_at <= 0 or previous_interval == new_interval:
        return next_poll_at
    return next_poll_at - previous_interval + new_interval


def _monitor_error_detail(error: Exception) -> str:
    message = " ".join(str(error).split())
    prefix = f"Monitor poll failed ({type(error).__name__})"
    return f"{prefix}: {message}"[:1_000] if message else prefix
