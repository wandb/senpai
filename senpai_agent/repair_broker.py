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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from senpai_agent.operations import CampaignInventory, RoleTarget
from senpai_agent.repair_executor import REPAIR_STREAM_LIMIT_CHARS
from senpai_agent.socket_framing import (
    SocketFrameError,
    SocketFrameTooLarge,
    encode_json_frame,
    receive_frame,
)


_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
_REQUEST_READ_TIMEOUT_SECONDS = 5
_RESPONSE_GRACE_SECONDS = 30
_Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


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

    operation_id: Annotated[str, Field(min_length=1, max_length=200)]
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
        normalized_command = command.strip()
        return cls(
            operation_id=operation_id,
            target=target,
            command=normalized_command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            command_fingerprint=repair_command_fingerprint(
                target=target,
                command=normalized_command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            ),
        )


class RepairResult(_Contract):
    exit_code: int
    stdout: str = Field(max_length=REPAIR_STREAM_LIMIT_CHARS)
    stderr: str = Field(max_length=REPAIR_STREAM_LIMIT_CHARS)

    @field_validator("stdout", "stderr")
    @classmethod
    def output_is_valid_utf8(cls, value: str) -> str:
        return value.encode("utf-8", errors="replace").decode("utf-8")


RepairOperationState = Literal["running", "completed", "failed", "unknown"]


class RepairOperationStatus(_Contract):
    operation_id: Annotated[str, Field(min_length=1, max_length=200)]
    target: RoleTarget
    command_fingerprint: _Fingerprint
    cwd: Literal["workspace", "state", "scratch"]
    timeout_seconds: int = Field(ge=1, le=3_600)
    requested_at: datetime
    completed_at: datetime | None
    status: RepairOperationState
    result: RepairResult | None = None
    error_type: str | None = None


class RepairAuditRecord(_Contract):
    operation_id: Annotated[str, Field(min_length=1, max_length=200)]
    target: RoleTarget
    command_fingerprint: _Fingerprint
    cwd: Literal["workspace", "state", "scratch"]
    timeout_seconds: int = Field(ge=1, le=3_600)
    requested_at: datetime
    completed_at: datetime | None
    status: RepairOperationState
    exit_code: int | None
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
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
                error_type TEXT
            )
            """
        )
        now = datetime.now(UTC).timestamp()
        self._connection.execute(
            """
            UPDATE repair_operations
            SET status = 'unknown', completed_at = COALESCE(completed_at, ?),
                error_type = COALESCE(error_type, 'BrokerInterrupted')
            WHERE status = 'running'
            """,
            (now,),
        )
        self._connection.commit()
        self.path.chmod(0o600)

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
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE repair_operations
                SET status = ?, completed_at = ?, result_json = ?, error_type = ?
                WHERE operation_id = ? AND status = 'running'
                """,
                (
                    status,
                    datetime.now(UTC).timestamp(),
                    result_json,
                    error_type,
                    operation_id,
                ),
            )
            if cursor.rowcount != 1:
                self._connection.rollback()
                raise RuntimeError("repair operation reservation is no longer active")
            self._connection.commit()
            return self.status(operation_id)

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
                exit_code=(
                    RepairResult.model_validate_json(row["result_json"]).exit_code
                    if row["result_json"]
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
    stdout = result.stdout
    stderr = result.stderr
    while True:
        candidate = RepairResult(
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=stderr,
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
            stdout = stdout[len(stdout) // 2 :]
            stderr = stderr[len(stderr) // 2 :]


class RepairBrokerClient:
    def __init__(self, socket_path: str | Path):
        self.socket_path = Path(socket_path)

    def execute(self, request: RepairRequest) -> RepairResult:
        response = self._request(
            {"operation": "execute", "request": request.model_dump(mode="json")},
            timeout_seconds=request.timeout_seconds + _RESPONSE_GRACE_SECONDS,
            operation_id=request.operation_id,
        )
        if "result" in response:
            return RepairResult.model_validate(response["result"])
        status = RepairOperationStatus.model_validate(response.get("status"))
        if status.status == "completed" and status.result is not None:
            return status.result
        if status.status in {"running", "unknown"}:
            raise RepairOutcomeUnknown(request.operation_id)
        raise RecordedRepairError(
            f"repair operation {request.operation_id!r} previously failed "
            f"({status.error_type or 'unknown error'})"
        )

    def status(self, operation_id: str) -> RepairOperationStatus:
        response = self._request(
            {"operation": "status", "operation_id": operation_id},
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
                    receive_frame(connection, max_bytes=_MAX_MESSAGE_BYTES)
                )
        except (OSError, SocketFrameError, json.JSONDecodeError, UnicodeDecodeError) as error:
            if sent and operation_id is not None:
                raise RepairOutcomeUnknown(operation_id) from error
            raise RepairTransportError("campaign repair broker is unavailable") from error
        if not isinstance(response, dict):
            raise RepairTransportError("campaign repair broker returned invalid data")
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
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_socket()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._socket.listen(8)
        self._socket.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            name="senpai-repair-broker",
            daemon=True,
        )

    def __enter__(self) -> Self:
        self._thread.start()
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
        self._stop.set()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(self.socket_path))
        except OSError:
            pass
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
        self._socket.close()
        self._remove_stale_socket()
        self.ledger.close()

    def recent_audit(self, *, limit: int = 12) -> tuple[RepairAuditRecord, ...]:
        return self.ledger.recent(limit=limit)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                if self._stop.is_set():
                    return
                connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
                raw_request = b""
                request: RepairRequest | None = None
                try:
                    raw_request = receive_frame(
                        connection,
                        max_bytes=_MAX_MESSAGE_BYTES,
                    )
                    envelope = json.loads(raw_request)
                    if envelope.get("operation") == "status":
                        status = self.ledger.status(str(envelope["operation_id"]))
                        response = {"status": status.model_dump(mode="json")}
                    elif envelope.get("operation") == "execute":
                        request = RepairRequest.model_validate(envelope["request"])
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
                try:
                    connection.sendall(
                        encode_json_frame(response, max_bytes=_MAX_MESSAGE_BYTES)
                    )
                except (OSError, SocketFrameError):
                    continue

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
            if status.result is not None:
                record["exit_code"] = status.result.exit_code
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
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(repair_broker_main())
