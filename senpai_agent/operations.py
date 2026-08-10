"""Typed, campaign-scoped operations for an external Senpai supervisor.

The service deliberately knows nothing about Kubernetes, SSH, or AWS.  A
backend binds the configured campaign inventory to those transports.  Callers
can name only a Senpai role from that inventory; they cannot supply a host,
deployment, working directory, environment, or credential.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Annotated, Literal, Protocol, Self, TypeAlias
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


_Text = Annotated[str, Field(min_length=1, max_length=4_000)]
_Key = Annotated[str, Field(min_length=1, max_length=200)]
AnomalyCategory: TypeAlias = Literal[
    "idle_capacity",
    "recovery_deferral",
    "stale_wip",
    "restart_churn",
    "benchmark_inactivity",
    "controller_failure",
    "context_pollution",
    "research_drift",
    "other_operational",
]


class RoleTarget(_Contract):
    """One role in one launch; never a caller-controlled machine address."""

    research_tag: _Key
    role: Literal["advisor", "student"]
    student: Annotated[str, Field(min_length=1, max_length=200)] | None = None

    @model_validator(mode="after")
    def _role_has_the_right_identity(self) -> RoleTarget:
        if self.role == "advisor" and self.student is not None:
            raise ValueError("advisor targets cannot name a student")
        if self.role == "student" and self.student is None:
            raise ValueError("student targets require a student name")
        return self

    @property
    def key(self) -> str:
        suffix = "advisor" if self.role == "advisor" else f"student:{self.student}"
        return f"{self.research_tag}:{suffix}"


class CampaignInventory(_Contract):
    """The immutable authority boundary for one supervisor instance."""

    research_tag: _Key
    repo: _Key
    advisor_branch: _Key
    students: tuple[Annotated[str, Field(min_length=1, max_length=200)], ...]

    @field_validator("repo")
    @classmethod
    def _repo_is_a_slug(cls, value: str) -> str:
        if len(value.split("/")) != 2 or not all(value.split("/")):
            raise ValueError("repo must use owner/name form")
        return value

    @field_validator("students")
    @classmethod
    def _students_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("campaign students must be unique")
        return value

    def require(self, target: RoleTarget) -> None:
        allowed = target.research_tag == self.research_tag and (
            target.role == "advisor" or target.student in self.students
        )
        if not allowed:
            raise PermissionError(
                f"target {target.key!r} is outside this campaign inventory"
            )


class RoleObservation(_Contract):
    """Backend observation used for compare-and-act safety checks."""

    target: RoleTarget
    observed_at: datetime
    control_token: _Key
    restart_control_token: _Key | None = None
    controller_alive: bool | None
    controller_phase: Annotated[str, Field(min_length=1, max_length=200)] | None
    worker_generation: Annotated[int, Field(ge=1)] | None = None
    conversation_id: UUID | None
    active_turn: bool | None
    unmatched_actions: Annotated[int, Field(ge=0)] | None
    raw_history_event_count: Annotated[int, Field(ge=0)] | None
    raw_history_digest: Annotated[str, Field(min_length=1, max_length=200)] | None
    pending_event_keys: tuple[_Key, ...] = ()
    active_delegation_count: Annotated[int, Field(ge=0, le=100)] | None = None

    @field_validator("observed_at")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observation timestamp must be timezone-aware")
        return value.astimezone(UTC)


class _Action(_Contract):
    operation_key: _Key
    target: RoleTarget


class _Mutation(_Action):
    incident_key: _Key
    anomaly_category: AnomalyCategory = "other_operational"
    reason: _Text


class CollectRole(_Action):
    kind: Literal["collect_role"] = "collect_role"


class Nudge(_Mutation):
    kind: Literal["nudge"] = "nudge"
    expected_conversation_id: UUID
    message: Annotated[str, Field(min_length=1, max_length=8_000)]


class Restart(_Mutation):
    kind: Literal["restart"] = "restart"
    expected_conversation_id: UUID


class ContextReset(_Mutation):
    kind: Literal["context_reset"] = "context_reset"
    expected_conversation_id: UUID
    recovery_prompt: Annotated[str, Field(min_length=1, max_length=16_000)]


OperationAction: TypeAlias = CollectRole | Nudge | Restart | ContextReset


class CollectRoleReceipt(_Contract):
    kind: Literal["collect_role"] = "collect_role"
    observation: RoleObservation


class NudgeReceipt(_Contract):
    kind: Literal["nudge"] = "nudge"
    target: RoleTarget
    conversation_id: UUID
    delivery_key: _Key


class RestartReceipt(_Contract):
    kind: Literal["restart"] = "restart"
    target: RoleTarget
    request_id: _Key | None = None
    status: Literal["queued"] | None = None
    conversation_id: UUID
    expected_worker_generation: Annotated[int, Field(ge=1)] | None = None
    state_preserved: bool | None = None
    compute_preserved: bool | None = None


class ContextResetReceipt(_Contract):
    kind: Literal["context_reset"] = "context_reset"
    target: RoleTarget
    request_id: _Key
    status: Literal["queued"] = "queued"
    expected_conversation_id: UUID
    expected_raw_history_event_count: Annotated[int, Field(ge=0)]
    expected_raw_history_digest: _Key
    expected_pending_event_keys: tuple[_Key, ...]


OperationReceipt: TypeAlias = Annotated[
    CollectRoleReceipt
    | NudgeReceipt
    | RestartReceipt
    | ContextResetReceipt,
    Field(discriminator="kind"),
]
_RECEIPT_ADAPTER = TypeAdapter(OperationReceipt)


class ContextResetRequest(_Contract):
    """Compare-and-reset request consumed only by the target role controller."""

    request_id: _Key
    target: RoleTarget
    expected_conversation_id: UUID
    expected_control_token: _Key
    expected_raw_history_event_count: Annotated[int, Field(ge=0)]
    expected_raw_history_digest: _Key
    expected_pending_event_keys: tuple[_Key, ...]
    recovery_prompt: Annotated[str, Field(min_length=1, max_length=16_000)]


class RestartRequest(_Contract):
    """Durable restart intent consumed only by the controller process owner."""

    request_id: _Key
    target: RoleTarget
    expected_conversation_id: UUID
    expected_restart_control_token: _Key
    expected_worker_generation: Annotated[int, Field(ge=1)]
    expected_completed_turns: Annotated[int, Field(ge=0)]


class RestartCompletion(_Contract):
    """Evidence that a newer owned worker replaced the requested generation."""

    request_id: _Key
    target: RoleTarget
    conversation_id: UUID
    source_generation: Annotated[int, Field(ge=1)]
    replacement_generation: Annotated[int, Field(ge=1)]
    state_preserved: bool
    compute_preserved: bool


class RestartRequestStatus(_Contract):
    request: RestartRequest
    status: Literal["pending", "processing", "completed", "rejected"]
    completion: RestartCompletion | None = None
    rejection_code: str | None = None
    planned_replacement_generation: Annotated[int, Field(ge=1)] | None = None


class RestartStatus(_Contract):
    """Sanitized restart queue state exposed by role diagnostics."""

    request_id: _Key
    status: Literal["queued", "processing", "completed", "rejected"]
    conversation_id: UUID
    source_generation: Annotated[int, Field(ge=1)]
    replacement_generation: Annotated[int, Field(ge=1)] | None = None
    rejection_code: Annotated[str, Field(min_length=1, max_length=200)] | None = None


class ContextResetCompletion(_Contract):
    """Evidence recorded by the role owner after reset-context execution."""

    request_id: _Key
    target: RoleTarget
    conversation_id: UUID
    raw_history_event_count_after: Annotated[int, Field(ge=0)]
    raw_history_digest: _Key
    pending_event_keys: tuple[_Key, ...]
    delivered_event_keys: tuple[_Key, ...] = ()


class ContextResetRequestStatus(_Contract):
    request: ContextResetRequest
    status: Literal["pending", "processing", "completed", "rejected"]
    completion: ContextResetCompletion | None = None
    rejection_code: str | None = None


class ContextResetStatus(_Contract):
    """Sanitized queue state exposed by a role's diagnostic endpoint."""

    request_id: _Key
    status: Literal["queued", "processing", "completed", "rejected"]
    conversation_id: UUID
    rejection_code: Annotated[str, Field(min_length=1, max_length=200)] | None = None


class OperationOutcome(_Contract):
    operation_key: _Key
    disposition: Literal["executed", "replayed", "suppressed"]
    receipt: OperationReceipt | None = None
    source_operation_key: str | None = None
    prior_status: Literal["running", "succeeded", "failed"] | None = None


class OperationAuditRecord(_Contract):
    operation_key: _Key
    action_kind: Literal[
        "collect_role", "nudge", "restart", "context_reset"
    ]
    target: RoleTarget
    incident_key: str | None
    anomaly_category: AnomalyCategory | None
    stable_incident_key: str | None
    requested_at: datetime
    completed_at: datetime | None
    status: Literal["running", "succeeded", "failed", "suppressed"]
    source_operation_key: str | None
    error_type: str | None

    @field_validator("requested_at", "completed_at")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audit timestamps must be timezone-aware")
        return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class OperationPolicy:
    mutation_cooldown_seconds: float = 3600

    def __post_init__(self) -> None:
        if self.mutation_cooldown_seconds < 0:
            raise ValueError("mutation cooldown must not be negative")


class OperationBackend(Protocol):
    """Campaign-bound management transport implemented by a deployment backend."""

    def collect_role(self, target: RoleTarget) -> RoleObservation: ...

    def nudge(
        self,
        target: RoleTarget,
        *,
        operation_key: str,
        expected_conversation_id: UUID,
        message: str,
        control_token: str,
    ) -> NudgeReceipt: ...

    def restart_controller(
        self,
        target: RoleTarget,
        *,
        expected_conversation_id: UUID,
        restart_control_token: str,
    ) -> RestartReceipt: ...

    def request_context_reset(
        self,
        target: RoleTarget,
        *,
        request: ContextResetRequest,
    ) -> ContextResetReceipt: ...

class OperationError(RuntimeError):
    """Base class for a rejected or unverifiable supervisor operation."""


class IdempotencyConflict(OperationError):
    """An operation key was reused for a different action."""


class OperationInProgress(OperationError):
    """An exact operation replay arrived before its first attempt completed."""


class RecordedOperationError(OperationError):
    """An exact replay of a durably recorded failed operation."""

    def __init__(self, operation_key: str, error_type: str | None):
        super().__init__(
            f"operation {operation_key!r} previously failed with "
            f"{error_type or 'an unknown error'}"
        )


class OperationInvariantError(OperationError):
    """A backend result could not prove the requested safety invariant."""


class UnsafeContextReset(OperationError):
    """A context reset was requested without a safe quiescent observation."""


@dataclass(frozen=True, slots=True)
class _Reservation:
    execute: bool
    outcome: OperationOutcome | None = None


class ContextResetRequestStore:
    """Durable single-owner queue for model-visible context resets.

    The management plane may enqueue requests, but only the role controller
    may claim and consume them.  A claimed request remains ``processing``
    across a crash so an ambiguous reset is never replayed implicitly.
    """

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS context_reset_requests (
                request_id TEXT PRIMARY KEY,
                research_tag TEXT NOT NULL,
                role TEXT NOT NULL,
                student TEXT,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                completion_json TEXT,
                rejection_code TEXT
            )
            """
        )
        self._connection.commit()
        self.path.chmod(0o600)

    def enqueue(self, request: ContextResetRequest) -> bool:
        encoded = request.model_dump_json()
        with self._lock:
            existing = self._connection.execute(
                """
                SELECT request_json FROM context_reset_requests
                WHERE request_id = ?
                """,
                (request.request_id,),
            ).fetchone()
            if existing is not None:
                if ContextResetRequest.model_validate_json(
                    existing["request_json"]
                ) != request:
                    raise IdempotencyConflict(
                        f"context reset request {request.request_id!r} was reused"
                    )
                return False
            self._connection.execute(
                """
                INSERT INTO context_reset_requests (
                    request_id, research_tag, role, student, request_json, status
                ) VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    request.request_id,
                    request.target.research_tag,
                    request.target.role,
                    request.target.student,
                    encoded,
                ),
            )
            self._connection.commit()
            return True

    def pending(
        self,
        target: RoleTarget | None = None,
    ) -> tuple[ContextResetRequest, ...]:
        query = (
            "SELECT request_json FROM context_reset_requests "
            "WHERE status = 'pending'"
        )
        parameters: tuple[object, ...] = ()
        if target is not None:
            query += (
                " AND research_tag = ? AND role = ? "
                "AND COALESCE(student, '') = ?"
            )
            parameters = (
                target.research_tag,
                target.role,
                target.student or "",
            )
        query += " ORDER BY rowid"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(
            ContextResetRequest.model_validate_json(row["request_json"])
            for row in rows
        )

    def claim_next(
        self,
        target: RoleTarget,
        *,
        conversation_id: UUID | None = None,
    ) -> ContextResetRequest | None:
        """Atomically claim the oldest request for this exact role."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._connection.execute(
                    """
                    SELECT request_id, request_json
                    FROM context_reset_requests
                    WHERE status = 'pending'
                      AND research_tag = ?
                      AND role = ?
                      AND COALESCE(student, '') = ?
                    ORDER BY rowid
                    """,
                    (target.research_tag, target.role, target.student or ""),
                ).fetchall()
                row = next(
                    (
                        candidate
                        for candidate in rows
                        if conversation_id is None
                        or ContextResetRequest.model_validate_json(
                            candidate["request_json"]
                        ).expected_conversation_id
                        == conversation_id
                    ),
                    None,
                )
                if row is None:
                    self._connection.commit()
                    return None
                cursor = self._connection.execute(
                    """
                    UPDATE context_reset_requests SET status = 'processing'
                    WHERE request_id = ? AND status = 'pending'
                    """,
                    (row["request_id"],),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("context reset request claim raced")
                self._connection.commit()
                return ContextResetRequest.model_validate_json(row["request_json"])
            except BaseException:
                self._connection.rollback()
                raise

    def complete(self, completion: ContextResetCompletion) -> bool:
        """Record owner evidence that UUID, trace, and pending events survived."""

        with self._lock:
            row = self._row(completion.request_id)
            request = ContextResetRequest.model_validate_json(row["request_json"])
            _validate_reset_completion(request, completion)
            if row["status"] == "completed":
                prior = ContextResetCompletion.model_validate_json(
                    row["completion_json"]
                )
                if prior != completion:
                    raise IdempotencyConflict(
                        f"context reset completion {completion.request_id!r} changed"
                    )
                return False
            if row["status"] != "processing":
                raise OperationInvariantError(
                    "only a claimed context reset request can be completed"
                )
            self._connection.execute(
                """
                UPDATE context_reset_requests
                SET status = 'completed', completion_json = ?
                WHERE request_id = ?
                """,
                (completion.model_dump_json(), completion.request_id),
            )
            self._connection.commit()
            return True

    def reject(self, request_id: str, rejection_code: str) -> bool:
        """Record a bounded reason code, never a raw backend error message."""

        rejection_code = rejection_code.strip()
        if not rejection_code or len(rejection_code) > 200:
            raise ValueError("context reset rejection code must be bounded")
        with self._lock:
            row = self._row(request_id)
            if row["status"] == "rejected":
                if row["rejection_code"] != rejection_code:
                    raise IdempotencyConflict(
                        f"context reset rejection {request_id!r} changed"
                    )
                return False
            if row["status"] == "completed":
                raise OperationInvariantError(
                    "a completed context reset request cannot be rejected"
                )
            self._connection.execute(
                """
                UPDATE context_reset_requests
                SET status = 'rejected', rejection_code = ?
                WHERE request_id = ?
                """,
                (rejection_code, request_id),
            )
            self._connection.commit()
            return True

    def result(self, request_id: str) -> ContextResetRequestStatus:
        with self._lock:
            row = self._row(request_id)
        return ContextResetRequestStatus(
            request=ContextResetRequest.model_validate_json(row["request_json"]),
            status=str(row["status"]),
            completion=(
                ContextResetCompletion.model_validate_json(row["completion_json"])
                if row["completion_json"]
                else None
            ),
            rejection_code=(
                str(row["rejection_code"]) if row["rejection_code"] else None
            ),
        )

    def statuses(
        self,
        target: RoleTarget,
        *,
        limit: int = 20,
    ) -> tuple[ContextResetStatus, ...]:
        """Return newest queue states without prompts, tokens, or trace evidence."""

        if not 1 <= limit <= 100:
            raise ValueError("context reset status limit must be between 1 and 100")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT request_id, request_json, status, rejection_code
                FROM context_reset_requests
                WHERE research_tag = ?
                  AND role = ?
                  AND COALESCE(student, '') = ?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (
                    target.research_tag,
                    target.role,
                    target.student or "",
                    limit,
                ),
            ).fetchall()
        return tuple(
            ContextResetStatus(
                request_id=str(row["request_id"]),
                status=(
                    "queued" if str(row["status"]) == "pending" else str(row["status"])
                ),
                conversation_id=ContextResetRequest.model_validate_json(
                    row["request_json"]
                ).expected_conversation_id,
                rejection_code=(
                    str(row["rejection_code"]) if row["rejection_code"] else None
                ),
            )
            for row in rows
        )

    def _row(self, request_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM context_reset_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return row

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class RestartRequestStore:
    """Durable handoff between role control and the worker-owning supervisor."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS controller_restart_requests (
                request_id TEXT PRIMARY KEY,
                research_tag TEXT NOT NULL,
                role TEXT NOT NULL,
                student TEXT,
                request_json TEXT NOT NULL,
                source_generation INTEGER NOT NULL,
                status TEXT NOT NULL,
                planned_replacement_generation INTEGER,
                completion_json TEXT,
                rejection_code TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS controller_worker_generation (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                generation INTEGER NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_controller_restart_per_generation
            ON controller_restart_requests (
                research_tag,
                role,
                COALESCE(student, ''),
                source_generation
            )
            WHERE status IN ('pending', 'processing')
            """
        )
        self._connection.commit()
        self.path.chmod(0o600)

    def allocate_worker_generation(self) -> int:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT generation FROM controller_worker_generation "
                    "WHERE singleton = 1"
                ).fetchone()
                generation = 1 if row is None else int(row["generation"]) + 1
                self._connection.execute(
                    """
                    INSERT INTO controller_worker_generation (singleton, generation)
                    VALUES (1, ?)
                    ON CONFLICT(singleton) DO UPDATE SET generation = excluded.generation
                    """,
                    (generation,),
                )
                self._connection.commit()
                return generation
            except BaseException:
                self._connection.rollback()
                raise

    def enqueue(self, request: RestartRequest) -> bool:
        encoded = request.model_dump_json()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    "SELECT request_json FROM controller_restart_requests "
                    "WHERE request_id = ?",
                    (request.request_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        RestartRequest.model_validate_json(existing["request_json"])
                        != request
                    ):
                        raise IdempotencyConflict(
                            "controller restart request "
                            f"{request.request_id!r} was reused"
                        )
                    self._connection.commit()
                    return False
                active = self._connection.execute(
                    """
                    SELECT request_id FROM controller_restart_requests
                    WHERE research_tag = ?
                      AND role = ?
                      AND COALESCE(student, '') = ?
                      AND source_generation = ?
                      AND status IN ('pending', 'processing')
                    """,
                    (
                        request.target.research_tag,
                        request.target.role,
                        request.target.student or "",
                        request.expected_worker_generation,
                    ),
                ).fetchone()
                if active is not None:
                    raise OperationInProgress(
                        "a controller restart is already active for worker generation "
                        f"{request.expected_worker_generation}"
                    )
                self._connection.execute(
                    """
                    INSERT INTO controller_restart_requests (
                        request_id, research_tag, role, student, request_json,
                        source_generation, status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        request.request_id,
                        request.target.research_tag,
                        request.target.role,
                        request.target.student,
                        encoded,
                        request.expected_worker_generation,
                    ),
                )
                self._connection.commit()
                return True
            except BaseException:
                self._connection.rollback()
                raise

    def claim_next(
        self,
        target: RoleTarget,
        *,
        worker_generation: int,
        replacement_generation: int,
    ) -> RestartRequest | None:
        if replacement_generation <= worker_generation:
            raise ValueError("replacement generation must follow the source worker")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._connection.execute(
                    """
                    SELECT request_id, request_json
                    FROM controller_restart_requests
                    WHERE status = 'pending'
                      AND research_tag = ?
                      AND role = ?
                      AND COALESCE(student, '') = ?
                    ORDER BY rowid
                    """,
                    (target.research_tag, target.role, target.student or ""),
                ).fetchall()
                row = next(
                    (
                        candidate
                        for candidate in rows
                        if RestartRequest.model_validate_json(
                            candidate["request_json"]
                        ).expected_worker_generation
                        == worker_generation
                    ),
                    None,
                )
                if row is None:
                    self._connection.commit()
                    return None
                cursor = self._connection.execute(
                    """
                    UPDATE controller_restart_requests
                    SET status = 'processing', planned_replacement_generation = ?
                    WHERE request_id = ? AND status = 'pending'
                    """,
                    (replacement_generation, row["request_id"]),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("controller restart request claim raced")
                self._connection.commit()
                return RestartRequest.model_validate_json(row["request_json"])
            except BaseException:
                self._connection.rollback()
                raise

    def awaiting_replacement(
        self,
        target: RoleTarget,
        *,
        replacement_generation: int,
    ) -> tuple[RestartRequest, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT request_json, planned_replacement_generation
                FROM controller_restart_requests
                WHERE status = 'processing'
                  AND research_tag = ?
                  AND role = ?
                  AND COALESCE(student, '') = ?
                ORDER BY rowid
                """,
                (target.research_tag, target.role, target.student or ""),
            ).fetchall()
        return tuple(
            RestartRequest.model_validate_json(row["request_json"])
            for row in rows
            if int(row["planned_replacement_generation"]) == replacement_generation
        )

    def missed_replacements(
        self,
        target: RoleTarget,
        *,
        replacement_generation: int,
    ) -> tuple[RestartRequest, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT request_json
                FROM controller_restart_requests
                WHERE status = 'processing'
                  AND research_tag = ?
                  AND role = ?
                  AND COALESCE(student, '') = ?
                  AND planned_replacement_generation < ?
                ORDER BY rowid
                """,
                (
                    target.research_tag,
                    target.role,
                    target.student or "",
                    replacement_generation,
                ),
            ).fetchall()
        return tuple(
            RestartRequest.model_validate_json(row["request_json"]) for row in rows
        )

    def missed_sources(
        self,
        target: RoleTarget,
        *,
        live_generation: int,
    ) -> tuple[RestartRequest, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT request_json
                FROM controller_restart_requests
                WHERE status = 'pending'
                  AND research_tag = ?
                  AND role = ?
                  AND COALESCE(student, '') = ?
                  AND source_generation < ?
                ORDER BY rowid
                """,
                (
                    target.research_tag,
                    target.role,
                    target.student or "",
                    live_generation,
                ),
            ).fetchall()
        return tuple(
            RestartRequest.model_validate_json(row["request_json"]) for row in rows
        )

    def complete(self, completion: RestartCompletion) -> bool:
        with self._lock:
            row = self._row(completion.request_id)
            request = RestartRequest.model_validate_json(row["request_json"])
            if (
                completion.target != request.target
                or completion.conversation_id != request.expected_conversation_id
                or completion.source_generation != request.expected_worker_generation
                or completion.replacement_generation <= completion.source_generation
                or not completion.state_preserved
                or not completion.compute_preserved
            ):
                raise OperationInvariantError(
                    "controller restart completion did not prove a safe replacement"
                )
            if row["status"] == "completed":
                prior = RestartCompletion.model_validate_json(row["completion_json"])
                if prior != completion:
                    raise IdempotencyConflict(
                        f"controller restart completion {completion.request_id!r} changed"
                    )
                return False
            if row["status"] != "processing":
                raise OperationInvariantError(
                    "only a claimed controller restart can be completed"
                )
            if (
                int(row["planned_replacement_generation"])
                != completion.replacement_generation
            ):
                raise OperationInvariantError(
                    "controller restart completion used an unplanned generation"
                )
            self._connection.execute(
                """
                UPDATE controller_restart_requests
                SET status = 'completed', completion_json = ?
                WHERE request_id = ?
                """,
                (completion.model_dump_json(), completion.request_id),
            )
            self._connection.commit()
            return True

    def reject(self, request_id: str, rejection_code: str) -> bool:
        rejection_code = rejection_code.strip()
        if not rejection_code or len(rejection_code) > 200:
            raise ValueError("controller restart rejection code must be bounded")
        with self._lock:
            row = self._row(request_id)
            if row["status"] == "rejected":
                if row["rejection_code"] != rejection_code:
                    raise IdempotencyConflict(
                        f"controller restart rejection {request_id!r} changed"
                    )
                return False
            if row["status"] == "completed":
                raise OperationInvariantError(
                    "a completed controller restart cannot be rejected"
                )
            self._connection.execute(
                """
                UPDATE controller_restart_requests
                SET status = 'rejected', rejection_code = ?
                WHERE request_id = ?
                """,
                (rejection_code, request_id),
            )
            self._connection.commit()
            return True

    def result(self, request_id: str) -> RestartRequestStatus:
        with self._lock:
            row = self._row(request_id)
        return RestartRequestStatus(
            request=RestartRequest.model_validate_json(row["request_json"]),
            status=str(row["status"]),
            completion=(
                RestartCompletion.model_validate_json(row["completion_json"])
                if row["completion_json"]
                else None
            ),
            rejection_code=(
                str(row["rejection_code"]) if row["rejection_code"] else None
            ),
            planned_replacement_generation=(
                int(row["planned_replacement_generation"])
                if row["planned_replacement_generation"] is not None
                else None
            ),
        )

    def statuses(
        self,
        target: RoleTarget,
        *,
        limit: int = 20,
    ) -> tuple[RestartStatus, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("controller restart status limit must be between 1 and 100")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT request_id, request_json, status,
                       planned_replacement_generation, completion_json,
                       rejection_code
                FROM controller_restart_requests
                WHERE research_tag = ?
                  AND role = ?
                  AND COALESCE(student, '') = ?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (
                    target.research_tag,
                    target.role,
                    target.student or "",
                    limit,
                ),
            ).fetchall()
        statuses = []
        for row in rows:
            request = RestartRequest.model_validate_json(row["request_json"])
            completion = (
                RestartCompletion.model_validate_json(row["completion_json"])
                if row["completion_json"]
                else None
            )
            statuses.append(
                RestartStatus(
                    request_id=str(row["request_id"]),
                    status=(
                        "queued" if str(row["status"]) == "pending" else row["status"]
                    ),
                    conversation_id=request.expected_conversation_id,
                    source_generation=request.expected_worker_generation,
                    replacement_generation=(
                        completion.replacement_generation
                        if completion is not None
                        else (
                            int(row["planned_replacement_generation"])
                            if row["planned_replacement_generation"] is not None
                            else None
                        )
                    ),
                    rejection_code=(
                        str(row["rejection_code"]) if row["rejection_code"] else None
                    ),
                )
            )
        return tuple(statuses)

    def _row(self, request_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM controller_restart_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return row

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class OperationLedger:
    """Durable idempotency, cooldown, and metadata-only action audit."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS operation_audit (
                operation_key TEXT PRIMARY KEY,
                action_kind TEXT NOT NULL,
                research_tag TEXT NOT NULL,
                role TEXT NOT NULL,
                student TEXT,
                incident_key TEXT,
                anomaly_category TEXT,
                cooldown_key TEXT,
                fingerprint TEXT NOT NULL,
                requested_at REAL NOT NULL,
                completed_at REAL,
                status TEXT NOT NULL,
                receipt_json TEXT,
                source_operation_key TEXT,
                error_type TEXT
            )
            """
        )
        columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(operation_audit)"
            ).fetchall()
        }
        if "anomaly_category" not in columns:
            self._connection.execute(
                "ALTER TABLE operation_audit ADD COLUMN anomaly_category TEXT"
            )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS operation_cooldown
            ON operation_audit (cooldown_key, requested_at DESC)
            """
        )
        self._migrate_legacy_mutation_incidents()
        self._connection.commit()
        self.path.chmod(0o600)

    def _migrate_legacy_mutation_incidents(self) -> None:
        """Give pre-category mutation rows the deterministic default identity."""

        rows = self._connection.execute(
            """
            SELECT operation_key, action_kind, research_tag, role, student,
                   anomaly_category
            FROM operation_audit
            WHERE action_kind != 'collect_role' AND anomaly_category IS NULL
            """
        ).fetchall()
        for row in rows:
            anomaly_category = (
                str(row["anomaly_category"])
                if row["anomaly_category"]
                else "other_operational"
            )
            stable_key = _stable_incident_key_from_parts(
                action_kind=str(row["action_kind"]),
                research_tag=str(row["research_tag"]),
                role=str(row["role"]),
                student=str(row["student"]) if row["student"] else None,
                anomaly_category=anomaly_category,
            )
            self._connection.execute(
                """
                UPDATE operation_audit
                SET anomaly_category = ?, cooldown_key = ?
                WHERE operation_key = ?
                """,
                (anomaly_category, stable_key, str(row["operation_key"])),
            )

    def reserve(
        self,
        action: OperationAction,
        *,
        fingerprint: str,
        cooldown_key: str | None,
        cooldown_seconds: float,
        now: datetime,
    ) -> _Reservation:
        timestamp = now.timestamp()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    "SELECT * FROM operation_audit WHERE operation_key = ?",
                    (action.operation_key,),
                ).fetchone()
                if existing is not None:
                    reservation = self._replay(existing, fingerprint)
                    self._connection.commit()
                    return reservation

                prior = None
                if cooldown_key is not None and cooldown_seconds > 0:
                    prior = self._connection.execute(
                        """
                        SELECT * FROM operation_audit
                        WHERE cooldown_key = ?
                          AND status != 'suppressed'
                          AND requested_at > ?
                        ORDER BY requested_at DESC
                        LIMIT 1
                        """,
                        (cooldown_key, timestamp - cooldown_seconds),
                    ).fetchone()
                if prior is not None:
                    receipt_json = prior["receipt_json"]
                    self._insert(
                        action,
                        fingerprint=fingerprint,
                        cooldown_key=cooldown_key,
                        requested_at=timestamp,
                        completed_at=timestamp,
                        status="suppressed",
                        receipt_json=receipt_json,
                        source_operation_key=str(prior["operation_key"]),
                    )
                    self._connection.commit()
                    receipt = (
                        _RECEIPT_ADAPTER.validate_json(receipt_json)
                        if receipt_json
                        else None
                    )
                    return _Reservation(
                        execute=False,
                        outcome=OperationOutcome(
                            operation_key=action.operation_key,
                            disposition="suppressed",
                            receipt=receipt,
                            source_operation_key=str(prior["operation_key"]),
                            prior_status=str(prior["status"]),
                        ),
                    )

                self._insert(
                    action,
                    fingerprint=fingerprint,
                    cooldown_key=cooldown_key,
                    requested_at=timestamp,
                    completed_at=None,
                    status="running",
                )
                self._connection.commit()
                return _Reservation(execute=True)
            except BaseException:
                self._connection.rollback()
                raise

    def _replay(self, row: sqlite3.Row, fingerprint: str) -> _Reservation:
        if row["fingerprint"] != fingerprint:
            raise IdempotencyConflict(
                f"operation key {row['operation_key']!r} was reused for a "
                "different action"
            )
        status = str(row["status"])
        if status == "running":
            raise OperationInProgress(
                f"operation {row['operation_key']!r} is already running"
            )
        if status == "failed":
            raise RecordedOperationError(
                str(row["operation_key"]),
                str(row["error_type"]) if row["error_type"] else None,
            )
        receipt = (
            _RECEIPT_ADAPTER.validate_json(row["receipt_json"])
            if row["receipt_json"]
            else None
        )
        disposition = "replayed" if status == "succeeded" else "suppressed"
        return _Reservation(
            execute=False,
            outcome=OperationOutcome(
                operation_key=str(row["operation_key"]),
                disposition=disposition,
                receipt=receipt,
                source_operation_key=(
                    str(row["source_operation_key"])
                    if row["source_operation_key"]
                    else None
                ),
            ),
        )

    def _insert(
        self,
        action: OperationAction,
        *,
        fingerprint: str,
        cooldown_key: str | None,
        requested_at: float,
        completed_at: float | None,
        status: str,
        receipt_json: str | None = None,
        source_operation_key: str | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO operation_audit (
                operation_key, action_kind, research_tag, role, student,
                incident_key, anomaly_category, cooldown_key, fingerprint,
                requested_at,
                completed_at, status, receipt_json, source_operation_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action.operation_key,
                action.kind,
                action.target.research_tag,
                action.target.role,
                action.target.student,
                getattr(action, "incident_key", None),
                getattr(action, "anomaly_category", None),
                cooldown_key,
                fingerprint,
                requested_at,
                completed_at,
                status,
                receipt_json,
                source_operation_key,
            ),
        )

    def succeed(
        self,
        operation_key: str,
        receipt: OperationReceipt,
        *,
        now: datetime,
    ) -> None:
        receipt_json = _RECEIPT_ADAPTER.dump_json(receipt).decode()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE operation_audit
                SET status = 'succeeded', completed_at = ?, receipt_json = ?
                WHERE operation_key = ? AND status = 'running'
                """,
                (now.timestamp(), receipt_json, operation_key),
            )
            if cursor.rowcount != 1:
                self._connection.rollback()
                raise RuntimeError("operation reservation is no longer active")
            self._connection.commit()

    def fail(
        self,
        operation_key: str,
        error: BaseException,
        *,
        now: datetime,
    ) -> None:
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE operation_audit
                SET status = 'failed', completed_at = ?, error_type = ?
                WHERE operation_key = ? AND status = 'running'
                """,
                (now.timestamp(), type(error).__name__, operation_key),
            )
            if cursor.rowcount != 1:
                self._connection.rollback()
                raise RuntimeError("operation reservation is no longer active")
            self._connection.commit()

    def records(self) -> list[OperationAuditRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM operation_audit ORDER BY requested_at, rowid"
            ).fetchall()
        return self._records(rows)

    def recent_mutations(self, *, limit: int = 12) -> list[OperationAuditRecord]:
        """Return a bounded newest-first mutation audit for a fresh wake."""

        if not 1 <= limit <= 50:
            raise ValueError("recent mutation audit limit must be between 1 and 50")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM operation_audit
                WHERE action_kind != 'collect_role'
                ORDER BY requested_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return self._records(rows)

    @staticmethod
    def _records(rows: list[sqlite3.Row]) -> list[OperationAuditRecord]:
        return [
            OperationAuditRecord(
                operation_key=str(row["operation_key"]),
                action_kind=str(row["action_kind"]),
                target=RoleTarget(
                    research_tag=str(row["research_tag"]),
                    role=str(row["role"]),
                    student=str(row["student"]) if row["student"] else None,
                ),
                incident_key=(
                    str(row["incident_key"]) if row["incident_key"] else None
                ),
                anomaly_category=(
                    str(row["anomaly_category"])
                    if row["anomaly_category"]
                    else None
                ),
                stable_incident_key=(
                    str(row["cooldown_key"]) if row["cooldown_key"] else None
                ),
                requested_at=datetime.fromtimestamp(row["requested_at"], UTC),
                completed_at=(
                    datetime.fromtimestamp(row["completed_at"], UTC)
                    if row["completed_at"] is not None
                    else None
                ),
                status=str(row["status"]),
                source_operation_key=(
                    str(row["source_operation_key"])
                    if row["source_operation_key"]
                    else None
                ),
                error_type=str(row["error_type"]) if row["error_type"] else None,
            )
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class OperationService:
    """Validate, deduplicate, audit, and execute one scoped operation."""

    def __init__(
        self,
        inventory: CampaignInventory,
        backend: OperationBackend,
        ledger: OperationLedger,
        *,
        policy: OperationPolicy | None = None,
    ):
        self.inventory = inventory
        self.backend = backend
        self.ledger = ledger
        self.policy = policy or OperationPolicy()

    def execute(
        self,
        action: OperationAction,
        *,
        now: datetime | None = None,
    ) -> OperationOutcome:
        fixed_timestamp = now is not None
        requested_at = now or datetime.now(UTC)
        if requested_at.tzinfo is None or requested_at.utcoffset() is None:
            raise ValueError("operation timestamp must be timezone-aware")
        requested_at = requested_at.astimezone(UTC)
        self.inventory.require(action.target)
        if isinstance(action, CollectRole):
            receipt = self._dispatch(action)
            return OperationOutcome(
                operation_key=action.operation_key,
                disposition="executed",
                receipt=receipt,
            )
        fingerprint = _fingerprint(action)
        mutation = isinstance(action, _Mutation)
        cooldown_key = _cooldown_key(action) if mutation else None
        reservation = self.ledger.reserve(
            action,
            fingerprint=fingerprint,
            cooldown_key=cooldown_key,
            cooldown_seconds=(
                self.policy.mutation_cooldown_seconds if mutation else 0
            ),
            now=requested_at,
        )
        if not reservation.execute:
            assert reservation.outcome is not None
            return reservation.outcome

        try:
            receipt = self._dispatch(action)
        except BaseException as error:
            completed_at = requested_at if fixed_timestamp else datetime.now(UTC)
            self.ledger.fail(action.operation_key, error, now=completed_at)
            raise
        completed_at = requested_at if fixed_timestamp else datetime.now(UTC)
        self.ledger.succeed(action.operation_key, receipt, now=completed_at)
        return OperationOutcome(
            operation_key=action.operation_key,
            disposition="executed",
            receipt=receipt,
        )

    def _dispatch(self, action: OperationAction) -> OperationReceipt:
        if isinstance(action, CollectRole):
            observed = self.backend.collect_role(action.target)
            _require_target(action.target, observed.target)
            return CollectRoleReceipt(observation=observed)
        if isinstance(action, Nudge):
            observed = self._observed_conversation(
                action.target,
                action.expected_conversation_id,
            )
            receipt = self.backend.nudge(
                action.target,
                operation_key=action.operation_key,
                expected_conversation_id=action.expected_conversation_id,
                message=action.message,
                control_token=observed.control_token,
            )
            _require_target(action.target, receipt.target)
            if receipt.conversation_id != observed.conversation_id:
                raise OperationInvariantError(
                    "nudge was not bound to the existing conversation"
                )
            return receipt
        if isinstance(action, Restart):
            observed = self.backend.collect_role(action.target)
            _require_target(action.target, observed.target)
            if observed.conversation_id != action.expected_conversation_id:
                raise OperationInvariantError(
                    "restart observation does not match the expected conversation"
                )
            if observed.restart_control_token is None:
                raise OperationInvariantError(
                    "role observation did not provide restart authorization"
                )
            if observed.worker_generation is None:
                raise OperationInvariantError(
                    "role observation did not provide an owned worker generation"
                )
            receipt = self.backend.restart_controller(
                action.target,
                expected_conversation_id=action.expected_conversation_id,
                restart_control_token=observed.restart_control_token,
            )
            _require_target(action.target, receipt.target)
            if (
                receipt.status != "queued"
                or receipt.request_id is None
                or receipt.expected_worker_generation is None
            ):
                raise OperationInvariantError(
                    "controller restart transport did not return an owner-queued receipt"
                )
            if receipt.conversation_id != observed.conversation_id:
                raise OperationInvariantError(
                    "controller restart did not preserve the conversation"
                )
            if receipt.expected_worker_generation != observed.worker_generation:
                raise OperationInvariantError(
                    "controller restart was queued for a different worker generation"
                )
            return receipt
        if isinstance(action, ContextReset):
            before = self._safe_reset_observation(action)
            request = ContextResetRequest(
                request_id=action.operation_key,
                target=action.target,
                expected_conversation_id=action.expected_conversation_id,
                expected_control_token=before.control_token,
                expected_raw_history_event_count=before.raw_history_event_count,
                expected_raw_history_digest=before.raw_history_digest,
                expected_pending_event_keys=before.pending_event_keys,
                recovery_prompt=action.recovery_prompt,
            )
            receipt = self.backend.request_context_reset(
                action.target,
                request=request,
            )
            _require_target(action.target, receipt.target)
            if not (
                receipt.request_id == request.request_id
                and receipt.expected_conversation_id
                == request.expected_conversation_id
                and receipt.expected_raw_history_event_count
                == request.expected_raw_history_event_count
                and receipt.expected_raw_history_digest
                == request.expected_raw_history_digest
                and receipt.expected_pending_event_keys
                == request.expected_pending_event_keys
            ):
                raise OperationInvariantError(
                    "context reset queue did not preserve the compare-and-reset "
                    "request"
                )
            return receipt
        raise TypeError(f"unsupported operation action: {type(action).__name__}")

    def _observed_conversation(
        self,
        target: RoleTarget,
        expected_conversation_id: UUID,
    ) -> RoleObservation:
        observed = self.backend.collect_role(target)
        _require_target(target, observed.target)
        if observed.conversation_id != expected_conversation_id:
            raise OperationInvariantError(
                "role observation does not match the expected conversation"
            )
        return observed

    def _safe_reset_observation(self, action: ContextReset) -> RoleObservation:
        observed = self._observed_conversation(
            action.target,
            action.expected_conversation_id,
        )
        if observed.active_turn is None:
            raise UnsafeContextReset("conversation activity is unknown")
        if observed.active_turn:
            raise UnsafeContextReset("conversation has an active turn")
        if observed.unmatched_actions is None:
            raise UnsafeContextReset("conversation tool-action state is unknown")
        if observed.unmatched_actions:
            raise UnsafeContextReset("conversation has unmatched tool actions")
        if observed.raw_history_event_count is None:
            raise UnsafeContextReset("conversation has no verifiable history count")
        if observed.raw_history_digest is None:
            raise UnsafeContextReset("conversation has no verifiable history digest")
        return observed

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def _fingerprint(action: OperationAction) -> str:
    value = action.model_dump(mode="json", exclude={"operation_key"})
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cooldown_key(action: _Mutation) -> str:
    return _stable_incident_key_from_parts(
        action_kind=action.kind,
        research_tag=action.target.research_tag,
        role=action.target.role,
        student=action.target.student,
        anomaly_category=action.anomaly_category,
    )


def _stable_incident_key_from_parts(
    *,
    action_kind: str,
    research_tag: str,
    role: str,
    student: str | None,
    anomaly_category: str,
) -> str:
    """Identify a repair class without trusting a model-provided label."""

    value: Mapping[str, str | None] = {
        "action_kind": action_kind,
        "research_tag": research_tag,
        "role": role,
        "student": student,
        "anomaly_category": anomaly_category,
    }
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"incident-{digest}"


def _require_target(expected: RoleTarget, actual: RoleTarget) -> None:
    if actual != expected:
        raise OperationInvariantError(
            f"backend returned target {actual.key!r}, expected {expected.key!r}"
        )


def _validate_reset_completion(
    request: ContextResetRequest,
    completion: ContextResetCompletion,
) -> None:
    if (
        completion.request_id != request.request_id
        or completion.target != request.target
    ):
        raise OperationInvariantError("context reset completion has the wrong identity")
    if completion.conversation_id != request.expected_conversation_id:
        raise OperationInvariantError(
            "context reset completion did not preserve the conversation UUID"
        )
    if completion.raw_history_digest != request.expected_raw_history_digest:
        raise OperationInvariantError(
            "context reset completion did not preserve the raw conversation history"
        )
    if (
        completion.raw_history_event_count_after
        < request.expected_raw_history_event_count
    ):
        raise OperationInvariantError(
            "context reset completion removed raw conversation history events"
        )
    preserved_event_keys = {
        *completion.pending_event_keys,
        *completion.delivered_event_keys,
    }
    if not set(request.expected_pending_event_keys).issubset(preserved_event_keys):
        raise OperationInvariantError(
            "context reset completion lost pending role events"
        )
