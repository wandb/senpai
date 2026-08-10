"""Durable campaign-bound repair commands for secret-free role sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sqlite3
import stat
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from senpai_agent.operations import CampaignInventory, RoleTarget
from senpai_agent.protocols import REPAIR_PROTOCOL_VERSION
from senpai_agent.repair_broker_health import (
    REPAIR_BROKER_HEALTH_PROTOCOL,
    repair_broker_health_socket,
)
from senpai_agent.repair_executor import REPAIR_STREAM_LIMIT_CHARS
from senpai_agent.socket_framing import (
    SocketFrameError,
    SocketFrameTooLarge,
    encode_json_frame,
    receive_frame,
    unix_socket_address,
)
from senpai_agent.sqlite_store import initialize_sqlite_store

_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
_REQUEST_READ_TIMEOUT_SECONDS = 5
_RESPONSE_GRACE_SECONDS = 30
_RETAINED_REPAIR_PAYLOADS = 128
_BROKER_HEALTH_FRAME_LIMIT_BYTES = 16 * 1024
_Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_OperationId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"),
]


def repair_command_fingerprint(
    *,
    target: RoleTarget,
    command: str,
    cwd: str,
    timeout_seconds: int,
) -> str:
    canonical = json.dumps(
        {
            "target": target.model_dump(mode="json"),
            "command": command,
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class RepairRequest(_Contract):
    """One idempotent command bound to an exact role and fixed mount root."""

    operation_id: _OperationId
    target: RoleTarget
    command: str = Field(min_length=1, max_length=65_536)
    command_fingerprint: _Fingerprint
    cwd: Literal["workspace", "state", "scratch"] = "workspace"
    timeout_seconds: int = Field(default=300, ge=1, le=3_600)

    @model_validator(mode="after")
    def fingerprint_matches_command(self) -> RepairRequest:
        expected = repair_command_fingerprint(
            target=self.target,
            command=self.command,
            cwd=self.cwd,
            timeout_seconds=self.timeout_seconds,
        )
        if self.command_fingerprint != expected:
            raise ValueError("repair command fingerprint does not match its payload")
        return self

    @field_validator("command")
    @classmethod
    def command_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("repair command must contain non-whitespace text")
        return value

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        target: RoleTarget,
        command: str,
        cwd: Literal["workspace", "state", "scratch"] = "workspace",
        timeout_seconds: int = 300,
    ) -> RepairRequest:
        return cls(
            operation_id=operation_id,
            target=target,
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            command_fingerprint=repair_command_fingerprint(
                target=target,
                command=command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            ),
        )


class RepairResult(_Contract):
    exit_code: int
    stdout: str = Field(max_length=REPAIR_STREAM_LIMIT_CHARS)
    stderr: str = Field(max_length=REPAIR_STREAM_LIMIT_CHARS)
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    controller_resumed: bool = True
    resume_error_type: str | None = None

    @field_validator("stdout", "stderr")
    @classmethod
    def output_is_valid_utf8(cls, value: str) -> str:
        return value.encode("utf-8", errors="replace").decode("utf-8")

    @model_validator(mode="after")
    def resume_metadata_is_consistent(self) -> RepairResult:
        if self.controller_resumed == (self.resume_error_type is not None):
            raise ValueError("repair controller-resume metadata is inconsistent")
        return self


RepairOperationState = Literal["running", "completed", "failed", "unknown"]


class RepairOperationStatus(_Contract):
    operation_id: _OperationId
    target: RoleTarget
    command_fingerprint: _Fingerprint
    cwd: Literal["workspace", "state", "scratch"]
    timeout_seconds: int = Field(ge=1, le=3_600)
    requested_at: datetime
    completed_at: datetime | None
    status: RepairOperationState
    receipt_retained: bool
    exit_code: int | None
    controller_resumed: bool | None
    resume_error_type: str | None
    payload_pruned_at: datetime | None
    result: RepairResult | None = None
    error_type: str | None = None

    @model_validator(mode="after")
    def receipt_metadata_is_consistent(self) -> RepairOperationStatus:
        if self.receipt_retained != (self.result is not None):
            raise ValueError("repair receipt retention metadata is inconsistent")
        if self.result is not None and self.result.exit_code != self.exit_code:
            raise ValueError("repair receipt exit code is inconsistent")
        if self.receipt_retained and self.payload_pruned_at is not None:
            raise ValueError("retained repair receipt cannot have a prune timestamp")
        if self.status == "completed":
            if self.controller_resumed is None:
                raise ValueError("completed repair must record controller recovery")
            if self.controller_resumed == (self.resume_error_type is not None):
                raise ValueError("repair controller-resume metadata is inconsistent")
        elif self.controller_resumed is not None or self.resume_error_type is not None:
            raise ValueError("unfinished repair cannot record controller recovery")
        return self


class RepairAuditRecord(_Contract):
    operation_id: _OperationId
    target: RoleTarget
    command_fingerprint: _Fingerprint
    cwd: Literal["workspace", "state", "scratch"]
    timeout_seconds: int = Field(ge=1, le=3_600)
    requested_at: datetime
    completed_at: datetime | None
    status: RepairOperationState
    receipt_retained: bool
    exit_code: int | None
    controller_resumed: bool | None
    resume_error_type: str | None
    payload_pruned_at: datetime | None
    error_type: str | None


class RepairBackend(Protocol):
    def run_repair(
        self,
        target: RoleTarget,
        *,
        command: str,
        cwd: str,
        timeout_seconds: int,
    ) -> RepairResult: ...


class RepairTransportError(RuntimeError):
    """The campaign repair broker was unavailable or returned bad data."""


class RepairOutcomeUnknown(RepairTransportError):
    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        super().__init__(
            f"repair operation {operation_id!r} outcome is unknown; query --status"
        )


class RepairReceiptExpired(RuntimeError):
    """A completed operation whose bounded full receipt has been pruned."""

    def __init__(self, status: RepairOperationStatus):
        self.status = status
        super().__init__(
            f"repair operation {status.operation_id!r} completed with exit code "
            f"{status.exit_code}, but its full receipt has expired"
        )


class RepairIdempotencyConflict(RuntimeError):
    pass


class RepairOperationNotFound(LookupError):
    pass


class RecordedRepairError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RepairReservation:
    execute: bool
    status: RepairOperationStatus


class RepairLedger:
    """Durable operation identity, receipt, ambiguity, and recent audit."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self._lock = threading.RLock()
        self._connection = initialize_sqlite_store(
            self.path,
            self._initialize_schema,
        )

    @staticmethod
    def _initialize_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS repair_operations (
                operation_id TEXT PRIMARY KEY,
                command_fingerprint TEXT NOT NULL,
                research_tag TEXT NOT NULL,
                role TEXT NOT NULL,
                student TEXT,
                cwd TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL,
                requested_at REAL NOT NULL,
                completed_at REAL,
                status TEXT NOT NULL,
                result_json TEXT,
                error_type TEXT,
                receipt_retained INTEGER NOT NULL DEFAULT 0,
                exit_code INTEGER,
                controller_resumed INTEGER,
                resume_error_type TEXT,
                payload_pruned_at REAL
            )
            """
        )
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(repair_operations)"
            ).fetchall()
        }
        migrations = {
            "receipt_retained": (
                "ALTER TABLE repair_operations ADD COLUMN "
                "receipt_retained INTEGER NOT NULL DEFAULT 0"
            ),
            "exit_code": ("ALTER TABLE repair_operations ADD COLUMN exit_code INTEGER"),
            "payload_pruned_at": (
                "ALTER TABLE repair_operations ADD COLUMN payload_pruned_at REAL"
            ),
            "controller_resumed": (
                "ALTER TABLE repair_operations ADD COLUMN controller_resumed INTEGER"
            ),
            "resume_error_type": (
                "ALTER TABLE repair_operations ADD COLUMN resume_error_type TEXT"
            ),
        }
        for column, statement in migrations.items():
            if column not in columns:
                connection.execute(statement)
        connection.execute(
            """
            UPDATE repair_operations
            SET receipt_retained = 1, payload_pruned_at = NULL
            WHERE result_json IS NOT NULL
            """
        )
        legacy_receipts = connection.execute(
            """
            SELECT operation_id, result_json
            FROM repair_operations
            WHERE result_json IS NOT NULL
              AND (exit_code IS NULL OR controller_resumed IS NULL)
            """
        ).fetchall()
        for row in legacy_receipts:
            result = RepairResult.model_validate_json(row["result_json"])
            connection.execute(
                """
                UPDATE repair_operations
                SET receipt_retained = 1, exit_code = ?,
                    controller_resumed = ?, resume_error_type = ?
                WHERE operation_id = ?
                """,
                (
                    result.exit_code,
                    result.controller_resumed,
                    result.resume_error_type,
                    row["operation_id"],
                ),
            )
        connection.execute(
            """
            UPDATE repair_operations
            SET controller_resumed = 1
            WHERE status = 'completed' AND controller_resumed IS NULL
            """
        )

    def recover_interrupted(self) -> None:
        """Make pre-restart executions explicitly ambiguous without replaying them."""

        now = datetime.now(UTC).timestamp()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    UPDATE repair_operations
                    SET status = 'unknown', completed_at = COALESCE(completed_at, ?),
                        error_type = COALESCE(error_type, 'BrokerInterrupted')
                    WHERE status = 'running'
                    """,
                    (now,),
                )
                self._prune_receipt_payloads(now)
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _prune_receipt_payloads(self, now: float) -> None:
        self._connection.execute(
            """
            UPDATE repair_operations
            SET result_json = NULL, receipt_retained = 0,
                payload_pruned_at = COALESCE(payload_pruned_at, ?)
            WHERE status = 'completed' AND result_json IS NOT NULL
              AND operation_id NOT IN (
                  SELECT operation_id
                  FROM repair_operations
                  WHERE status = 'completed' AND result_json IS NOT NULL
                  ORDER BY completed_at DESC, rowid DESC
                  LIMIT ?
              )
            """,
            (now, _RETAINED_REPAIR_PAYLOADS),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def reserve(self, request: RepairRequest) -> RepairReservation:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM repair_operations WHERE operation_id = ?",
                    (request.operation_id,),
                ).fetchone()
                if row is not None:
                    if row["command_fingerprint"] != request.command_fingerprint:
                        raise RepairIdempotencyConflict(
                            f"repair operation {request.operation_id!r} was reused "
                            "for a different command"
                        )
                    status = self._status_from_row(row)
                    self._connection.commit()
                    return RepairReservation(execute=False, status=status)
                self._connection.execute(
                    """
                    INSERT INTO repair_operations (
                        operation_id, command_fingerprint, research_tag, role,
                        student, cwd, timeout_seconds, requested_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
                    """,
                    (
                        request.operation_id,
                        request.command_fingerprint,
                        request.target.research_tag,
                        request.target.role,
                        request.target.student,
                        request.cwd,
                        request.timeout_seconds,
                        datetime.now(UTC).timestamp(),
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM repair_operations WHERE operation_id = ?",
                    (request.operation_id,),
                ).fetchone()
                self._connection.commit()
                assert row is not None
                return RepairReservation(execute=True, status=self._status_from_row(row))
            except BaseException:
                self._connection.rollback()
                raise

    def complete(self, operation_id: str, result: RepairResult) -> RepairOperationStatus:
        return self._finish(
            operation_id,
            status="completed",
            result=result,
            error_type=None,
        )

    def fail(self, operation_id: str, error: BaseException) -> RepairOperationStatus:
        return self._finish(
            operation_id,
            status="failed",
            result=None,
            error_type=type(error).__name__,
        )

    def unknown(self, operation_id: str, error: BaseException) -> RepairOperationStatus:
        return self._finish(
            operation_id,
            status="unknown",
            result=None,
            error_type=type(error).__name__,
        )

    def _finish(
        self,
        operation_id: str,
        *,
        status: Literal["completed", "failed", "unknown"],
        result: RepairResult | None,
        error_type: str | None,
    ) -> RepairOperationStatus:
        result_json = result.model_dump_json() if result is not None else None
        completed_at = datetime.now(UTC).timestamp()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._connection.execute(
                    """
                    UPDATE repair_operations
                    SET status = ?, completed_at = ?, result_json = ?,
                        receipt_retained = ?, exit_code = ?,
                        controller_resumed = ?, resume_error_type = ?,
                        payload_pruned_at = NULL, error_type = ?
                    WHERE operation_id = ? AND status = 'running'
                    """,
                    (
                        status,
                        completed_at,
                        result_json,
                        result is not None,
                        result.exit_code if result is not None else None,
                        result.controller_resumed if result is not None else None,
                        result.resume_error_type if result is not None else None,
                        error_type,
                        operation_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "repair operation reservation is no longer active"
                    )
                self._prune_receipt_payloads(completed_at)
                row = self._connection.execute(
                    "SELECT * FROM repair_operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        assert row is not None
        return self._status_from_row(row)

    def status(self, operation_id: str) -> RepairOperationStatus:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM repair_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        if row is None:
            raise RepairOperationNotFound(
                f"repair operation {operation_id!r} was not found"
            )
        return self._status_from_row(row)

    def recent(self, *, limit: int = 12) -> tuple[RepairAuditRecord, ...]:
        if not 1 <= limit <= 50:
            raise ValueError("recent repair audit limit must be between 1 and 50")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM repair_operations
                ORDER BY requested_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(
            RepairAuditRecord(
                operation_id=str(row["operation_id"]),
                target=self._target_from_row(row),
                command_fingerprint=str(row["command_fingerprint"]),
                cwd=str(row["cwd"]),
                timeout_seconds=int(row["timeout_seconds"]),
                requested_at=datetime.fromtimestamp(row["requested_at"], UTC),
                completed_at=(
                    datetime.fromtimestamp(row["completed_at"], UTC)
                    if row["completed_at"] is not None
                    else None
                ),
                status=str(row["status"]),
                receipt_retained=bool(row["receipt_retained"]),
                exit_code=(
                    int(row["exit_code"]) if row["exit_code"] is not None else None
                ),
                controller_resumed=(
                    bool(row["controller_resumed"])
                    if row["controller_resumed"] is not None
                    else None
                ),
                resume_error_type=(
                    str(row["resume_error_type"])
                    if row["resume_error_type"]
                    else None
                ),
                payload_pruned_at=(
                    datetime.fromtimestamp(row["payload_pruned_at"], UTC)
                    if row["payload_pruned_at"] is not None
                    else None
                ),
                error_type=str(row["error_type"]) if row["error_type"] else None,
            )
            for row in rows
        )

    @classmethod
    def _status_from_row(cls, row: sqlite3.Row) -> RepairOperationStatus:
        return RepairOperationStatus(
            operation_id=str(row["operation_id"]),
            target=cls._target_from_row(row),
            command_fingerprint=str(row["command_fingerprint"]),
            cwd=str(row["cwd"]),
            timeout_seconds=int(row["timeout_seconds"]),
            requested_at=datetime.fromtimestamp(row["requested_at"], UTC),
            completed_at=(
                datetime.fromtimestamp(row["completed_at"], UTC)
                if row["completed_at"] is not None
                else None
            ),
            status=str(row["status"]),
            receipt_retained=bool(row["receipt_retained"]),
            exit_code=(int(row["exit_code"]) if row["exit_code"] is not None else None),
            controller_resumed=(
                bool(row["controller_resumed"])
                if row["controller_resumed"] is not None
                else None
            ),
            resume_error_type=(
                str(row["resume_error_type"])
                if row["resume_error_type"]
                else None
            ),
            payload_pruned_at=(
                datetime.fromtimestamp(row["payload_pruned_at"], UTC)
                if row["payload_pruned_at"] is not None
                else None
            ),
            result=(
                RepairResult.model_validate_json(row["result_json"])
                if row["result_json"]
                else None
            ),
            error_type=str(row["error_type"]) if row["error_type"] else None,
        )

    @staticmethod
    def _target_from_row(row: sqlite3.Row) -> RoleTarget:
        return RoleTarget(
            research_tag=str(row["research_tag"]),
            role=str(row["role"]),
            student=str(row["student"]) if row["student"] else None,
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _wire_safe_result(result: RepairResult) -> RepairResult:
    stdout, stdout_byte_truncated = _bounded_utf8_tail(
        result.stdout,
        REPAIR_STREAM_LIMIT_CHARS,
    )
    stderr, stderr_byte_truncated = _bounded_utf8_tail(
        result.stderr,
        REPAIR_STREAM_LIMIT_CHARS,
    )
    stdout_truncated = result.stdout_truncated or stdout_byte_truncated
    stderr_truncated = result.stderr_truncated or stderr_byte_truncated
    while True:
        candidate = RepairResult(
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            controller_resumed=result.controller_resumed,
            resume_error_type=result.resume_error_type,
        )
        try:
            encode_json_frame(
                {"result": candidate.model_dump(mode="json")},
                max_bytes=_MAX_MESSAGE_BYTES,
            )
            return candidate
        except SocketFrameTooLarge:
            if not stdout and not stderr:
                raise
            if stdout:
                stdout = stdout[len(stdout) // 2 :]
                stdout_truncated = True
            if stderr:
                stderr = stderr[len(stderr) // 2 :]
                stderr_truncated = True


def _bounded_utf8_tail(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    tail = encoded[-limit:]
    for offset in range(min(4, len(tail) + 1)):
        try:
            return tail[offset:].decode("utf-8"), True
        except UnicodeDecodeError:
            continue
    raise RuntimeError("could not find a UTF-8 boundary in bounded repair output")


class RepairBrokerClient:
    def __init__(self, socket_path: str | Path):
        self.socket_path = Path(socket_path)

    def execute(self, request: RepairRequest) -> RepairResult:
        response = self._request(
            {
                "protocol": REPAIR_PROTOCOL_VERSION,
                "operation": "execute",
                "request": request.model_dump(mode="json"),
            },
            timeout_seconds=request.timeout_seconds + _RESPONSE_GRACE_SECONDS,
            operation_id=request.operation_id,
        )
        if "result" in response:
            return RepairResult.model_validate(response["result"])
        status = RepairOperationStatus.model_validate(response.get("status"))
        if status.status == "completed" and status.result is not None:
            return status.result
        if status.status == "completed" and not status.receipt_retained:
            raise RepairReceiptExpired(status)
        if status.status in {"running", "unknown"}:
            raise RepairOutcomeUnknown(request.operation_id)
        raise RecordedRepairError(
            f"repair operation {request.operation_id!r} previously failed "
            f"({status.error_type or 'unknown error'})"
        )

    def status(self, operation_id: str) -> RepairOperationStatus:
        response = self._request(
            {
                "protocol": REPAIR_PROTOCOL_VERSION,
                "operation": "status",
                "operation_id": operation_id,
            },
            timeout_seconds=_REQUEST_READ_TIMEOUT_SECONDS,
            operation_id=None,
        )
        return RepairOperationStatus.model_validate(response["status"])

    def _request(
        self,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        operation_id: str | None,
    ) -> dict[str, object]:
        try:
            frame = encode_json_frame(payload, max_bytes=_MAX_MESSAGE_BYTES)
        except SocketFrameError as error:
            raise RepairTransportError("repair request exceeded its byte limit") from error
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
            connection.connect(str(self.socket_path))
        except OSError as error:
            connection.close()
            raise RepairTransportError("campaign repair broker is unavailable") from error
        sent = False
        try:
            with connection:
                sent = True
                connection.sendall(frame)
                connection.settimeout(timeout_seconds)
                response = json.loads(
                    receive_frame(
                        connection,
                        max_bytes=_MAX_MESSAGE_BYTES,
                        timeout_seconds=timeout_seconds,
                    )
                )
        except (OSError, SocketFrameError, json.JSONDecodeError, UnicodeDecodeError) as error:
            if sent and operation_id is not None:
                raise RepairOutcomeUnknown(operation_id) from error
            raise RepairTransportError("campaign repair broker is unavailable") from error
        if not isinstance(response, dict):
            raise RepairTransportError("campaign repair broker returned invalid data")
        if response.get("protocol") != REPAIR_PROTOCOL_VERSION:
            if operation_id is not None:
                raise RepairOutcomeUnknown(operation_id)
            raise RepairTransportError("campaign repair broker protocol mismatch")
        if "error" in response:
            error_type = str(response.get("error_type", "RepairError"))
            message = str(response["error"])
            if error_type == "PermissionError":
                raise PermissionError(message)
            if error_type == "RepairIdempotencyConflict":
                raise RepairIdempotencyConflict(message)
            if error_type == "RepairOperationNotFound":
                raise RepairOperationNotFound(message)
            raise RepairTransportError(f"{error_type}: {message}")
        return response


class _RepairBrokerHeartbeat:
    """Serve broker-thread liveness independently of a bounded repair call."""

    def __init__(self, command_socket: str | Path):
        self.socket_path = repair_broker_health_socket(command_socket)
        self._lock = threading.Lock()
        self._state = "idle"
        self._updated_at = time.monotonic()
        self._operation_deadline: float | None = None
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="senpai-repair-broker-health",
            daemon=True,
        )

    def start(self) -> None:
        filesystem_path = (
            None if self.socket_path.startswith("@") else Path(self.socket_path)
        )
        if filesystem_path is not None:
            filesystem_path.parent.mkdir(parents=True, exist_ok=True)
            self._remove_filesystem_socket(filesystem_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(unix_socket_address(self.socket_path))
        if filesystem_path is not None:
            os.chmod(filesystem_path, 0o600)
        listener.listen(4)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread.start()

    def publish(self, state: Literal["idle", "active"], deadline: float | None = None) -> None:
        with self._lock:
            self._state = state
            self._operation_deadline = deadline
            self._updated_at = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            listener.close()
        if self._thread.ident is not None:
            self._thread.join(timeout=1)
        if not self.socket_path.startswith("@"):
            self._remove_filesystem_socket(Path(self.socket_path))

    def _serve(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection, self._lock:
                status = {
                    "protocol": REPAIR_BROKER_HEALTH_PROTOCOL,
                    "server_pid": os.getpid(),
                    "state": self._state,
                    "heartbeat_monotonic": self._updated_at,
                    "operation_deadline_monotonic": self._operation_deadline,
                }
                try:
                    connection.sendall(
                        encode_json_frame(
                            status,
                            max_bytes=_BROKER_HEALTH_FRAME_LIMIT_BYTES,
                        )
                    )
                except (OSError, SocketFrameError):
                    continue

    @staticmethod
    def _remove_filesystem_socket(socket_path: Path) -> None:
        try:
            mode = socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(
                f"refusing to replace non-socket path: {socket_path}"
            )
        socket_path.unlink()


class RepairBrokerServer:
    """Map exact RoleTarget values to durable fixed-sidecar operations."""

    def __init__(
        self,
        socket_path: str | Path,
        inventory: CampaignInventory,
        backend: RepairBackend,
        *,
        ledger_path: Path,
    ):
        self.socket_path = Path(socket_path)
        self.inventory = inventory
        self.backend = backend
        self.ledger = RepairLedger(ledger_path)
        self.ledger.recover_interrupted()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_socket()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._socket.listen(8)
        self._socket.settimeout(0.2)
        self._stop = threading.Event()
        self._heartbeat = _RepairBrokerHeartbeat(socket_path)
        self._closed = False
        self._thread = threading.Thread(
            target=self._serve,
            name="senpai-repair-broker",
            daemon=True,
        )

    def __enter__(self) -> Self:
        self._heartbeat.start()
        try:
            self._thread.start()
        except BaseException:
            self._heartbeat.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(self.socket_path))
        except OSError:
            pass
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
        self._heartbeat.close()
        self._socket.close()
        self._remove_stale_socket()
        self.ledger.close()

    def recent_audit(self, *, limit: int = 12) -> tuple[RepairAuditRecord, ...]:
        return self.ledger.recent(limit=limit)

    def _serve(self) -> None:
        while not self._stop.is_set():
            self._heartbeat.publish("idle")
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                if self._stop.is_set():
                    return
                self._heartbeat.publish(
                    "active",
                    time.monotonic()
                    + _REQUEST_READ_TIMEOUT_SECONDS
                    + _RESPONSE_GRACE_SECONDS,
                )
                connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
                raw_request = b""
                request: RepairRequest | None = None
                try:
                    raw_request = receive_frame(
                        connection,
                        max_bytes=_MAX_MESSAGE_BYTES,
                        timeout_seconds=_REQUEST_READ_TIMEOUT_SECONDS,
                    )
                    envelope = json.loads(raw_request)
                    if not isinstance(envelope, dict):
                        raise ValueError("invalid repair broker envelope")
                    if envelope.get("protocol") != REPAIR_PROTOCOL_VERSION:
                        raise ValueError("unsupported repair broker protocol")
                    if envelope.get("operation") == "status":
                        status = self.ledger.status(str(envelope["operation_id"]))
                        response = {"status": status.model_dump(mode="json")}
                    elif envelope.get("operation") == "execute":
                        request = RepairRequest.model_validate(envelope["request"])
                        self._heartbeat.publish(
                            "active",
                            time.monotonic()
                            + request.timeout_seconds
                            + _RESPONSE_GRACE_SECONDS,
                        )
                        response = self._execute(raw_request, request)
                    else:
                        raise ValueError("unsupported repair broker operation")
                except Exception as error:  # noqa: BLE001
                    if request is not None:
                        self._audit(raw_request, request, outcome="denied", error=error)
                    response = {
                        "error_type": type(error).__name__,
                        "error": str(error)[:4_096],
                    }
                response["protocol"] = REPAIR_PROTOCOL_VERSION
                try:
                    connection.sendall(
                        encode_json_frame(response, max_bytes=_MAX_MESSAGE_BYTES)
                    )
                except (OSError, SocketFrameError):
                    continue
                finally:
                    self._heartbeat.publish("idle")

    def _execute(
        self,
        raw_request: bytes,
        request: RepairRequest,
    ) -> dict[str, object]:
        self.inventory.require(request.target)
        reservation = self.ledger.reserve(request)
        if not reservation.execute:
            status = reservation.status
            self._audit(raw_request, request, outcome="replayed", status=status)
            if status.status == "completed" and status.result is not None:
                return {"result": status.result.model_dump(mode="json")}
            return {"status": status.model_dump(mode="json")}
        try:
            result = _wire_safe_result(
                self.backend.run_repair(
                    request.target,
                    command=request.command,
                    cwd=request.cwd,
                    timeout_seconds=request.timeout_seconds,
                )
            )
        except Exception as error:  # noqa: BLE001
            status = self.ledger.unknown(request.operation_id, error)
            self._audit(
                raw_request,
                request,
                outcome="unknown",
                status=status,
                error=error,
            )
            return {"status": status.model_dump(mode="json")}
        status = self.ledger.complete(request.operation_id, result)
        self._audit(raw_request, request, outcome="completed", status=status)
        return {"result": result.model_dump(mode="json")}

    @staticmethod
    def _audit(
        raw_request: bytes,
        request: RepairRequest,
        *,
        outcome: Literal["completed", "replayed", "denied", "unknown"],
        status: RepairOperationStatus | None = None,
        error: BaseException | None = None,
    ) -> None:
        record: dict[str, object] = {
            "event": "SENPAI_REPAIR_COMMAND",
            "timestamp": datetime.now(UTC).isoformat(),
            "outcome": outcome,
            "operation_id": request.operation_id,
            "request_sha256": hashlib.sha256(raw_request).hexdigest(),
            "target": request.target.key,
            "cwd": request.cwd,
            "timeout_seconds": request.timeout_seconds,
            "command_sha256": request.command_fingerprint,
        }
        if status is not None:
            record["status"] = status.status
            record["receipt_retained"] = status.receipt_retained
            record["exit_code"] = status.exit_code
            record["controller_resumed"] = status.controller_resumed
            record["resume_error_type"] = status.resume_error_type
            record["payload_pruned_at"] = (
                status.payload_pruned_at.isoformat()
                if status.payload_pruned_at is not None
                else None
            )
        if error is not None:
            record["error_type"] = type(error).__name__
        print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)

    def _remove_stale_socket(self) -> None:
        try:
            mode = self.socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(
                f"refusing to replace non-socket path: {self.socket_path}"
            )
        self.socket_path.unlink()


def repair_broker_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Use Senpai's role repair broker.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    repair = subparsers.add_parser("repair")
    repair.add_argument("--socket", type=Path)
    repair.add_argument("--research-tag")
    repair.add_argument("--role", choices=("advisor", "student"))
    repair.add_argument("--student")
    repair.add_argument(
        "--cwd", choices=("workspace", "state", "scratch"), default="workspace"
    )
    repair.add_argument("--timeout", type=int, default=300)
    repair.add_argument("--operation-id")
    repair.add_argument("--status", metavar="OPERATION_ID")
    repair.add_argument("--command")
    args = parser.parse_args(argv)

    socket_path = args.socket or Path(
        os.environ.get(
            "SENPAI_SUPERVISOR_REPAIR_SOCKET",
            "/run/senpai-repair/repair.sock",
        )
    )
    client = RepairBrokerClient(socket_path)
    if args.status:
        if any((args.operation_id, args.command, args.role, args.student)):
            parser.error("--status cannot be combined with repair command arguments")
        print(client.status(args.status).model_dump_json())
        return 0
    if not args.operation_id:
        parser.error("--operation-id is required for a repair command")
    if not args.command:
        parser.error("--command is required for a repair command")
    if not args.role:
        parser.error("--role is required for a repair command")
    research_tag = args.research_tag or os.environ.get("SENPAI_RESEARCH_TAG")
    if not research_tag:
        parser.error("--research-tag or SENPAI_RESEARCH_TAG is required")
    target = RoleTarget(
        research_tag=research_tag,
        role=args.role,
        student=args.student,
    )
    result = client.execute(
        RepairRequest.create(
            operation_id=args.operation_id,
            target=target,
            command=args.command,
            cwd=args.cwd,
            timeout_seconds=args.timeout,
        )
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if not result.controller_resumed:
        sys.stderr.write(
            "\nSENPAI_REPAIR_CONTROLLER_NOT_RESUMED "
            f"error_type={result.resume_error_type or 'UnknownError'}\n"
        )
    return result.exit_code or (75 if not result.controller_resumed else 0)


if __name__ == "__main__":
    raise SystemExit(repair_broker_main())
