"""Campaign-bound arbitrary repair commands for secret-free role sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import sys
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.operations import CampaignInventory, RoleTarget
from senpai_agent.repair_executor import REPAIR_STREAM_LIMIT_CHARS


_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
_REQUEST_READ_TIMEOUT_SECONDS = 5
_RESPONSE_GRACE_SECONDS = 30


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RepairRequest(_Contract):
    """One command bound to an inventory-validated role and fixed mount root."""

    target: RoleTarget
    command: str = Field(min_length=1, max_length=65_536)
    cwd: Literal["workspace", "state", "scratch"] = "workspace"
    timeout_seconds: int = Field(default=300, ge=1, le=3_600)


class RepairResult(_Contract):
    exit_code: int
    stdout: str = Field(max_length=REPAIR_STREAM_LIMIT_CHARS)
    stderr: str = Field(max_length=REPAIR_STREAM_LIMIT_CHARS)


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


def _receive_line(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(min(65_536, _MAX_MESSAGE_BYTES - size + 1))
        if not chunk:
            break
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > _MAX_MESSAGE_BYTES:
            raise RepairTransportError("repair message exceeded the size limit")
    payload = b"".join(chunks)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise RepairTransportError("repair message exceeded the size limit")
    return payload


class RepairBrokerClient:
    def __init__(self, socket_path: str | Path):
        self.socket_path = Path(socket_path)

    def execute(self, request: RepairRequest) -> RepairResult:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
                connection.connect(str(self.socket_path))
                connection.sendall(request.model_dump_json().encode() + b"\n")
                connection.settimeout(
                    request.timeout_seconds + _RESPONSE_GRACE_SECONDS
                )
                response = json.loads(_receive_line(connection))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RepairTransportError(
                "campaign repair broker is unavailable"
            ) from error
        if not isinstance(response, dict):
            raise RepairTransportError("campaign repair broker returned invalid data")
        if response.get("error_type") == "PermissionError":
            raise PermissionError(str(response.get("error", "target denied")))
        if "error" in response:
            error_type = str(response.get("error_type", "RepairError"))
            raise RepairTransportError(f"{error_type}: {response['error']}")
        try:
            return RepairResult.model_validate(response["result"])
        except (KeyError, TypeError, ValueError) as error:
            raise RepairTransportError(
                "campaign repair broker returned an invalid result"
            ) from error


class RepairBrokerServer:
    """Map exact RoleTarget values to fixed repair sidecars."""

    def __init__(
        self,
        socket_path: str | Path,
        inventory: CampaignInventory,
        backend: RepairBackend,
    ):
        self.socket_path = Path(socket_path)
        self.inventory = inventory
        self.backend = backend
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
                    raw_request = _receive_line(connection)
                    request = RepairRequest.model_validate_json(raw_request)
                    self.inventory.require(request.target)
                    result = self.backend.run_repair(
                        request.target,
                        command=request.command,
                        cwd=request.cwd,
                        timeout_seconds=request.timeout_seconds,
                    )
                    self._audit(
                        raw_request,
                        request,
                        outcome="completed",
                        result=result,
                    )
                    response = {"result": result.model_dump(mode="json")}
                except Exception as error:  # noqa: BLE001
                    self._audit(
                        raw_request,
                        request,
                        outcome=(
                            "denied" if isinstance(error, PermissionError) else "error"
                        ),
                        error=error,
                    )
                    response = {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                try:
                    connection.sendall(json.dumps(response).encode() + b"\n")
                except OSError:
                    continue

    @staticmethod
    def _audit(
        raw_request: bytes,
        request: RepairRequest | None,
        *,
        outcome: Literal["completed", "denied", "error"],
        result: RepairResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        record: dict[str, object] = {
            "event": "SENPAI_REPAIR_COMMAND",
            "timestamp": datetime.now(UTC).isoformat(),
            "outcome": outcome,
            "request_sha256": hashlib.sha256(raw_request).hexdigest(),
        }
        if request is not None:
            record.update(
                {
                    "target": request.target.key,
                    "cwd": request.cwd,
                    "timeout_seconds": request.timeout_seconds,
                    "command_sha256": hashlib.sha256(
                        request.command.encode()
                    ).hexdigest(),
                }
            )
        if result is not None:
            record["exit_code"] = result.exit_code
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
    repair.add_argument("--role", choices=("advisor", "student"), required=True)
    repair.add_argument("--student")
    repair.add_argument(
        "--cwd", choices=("workspace", "state", "scratch"), default="workspace"
    )
    repair.add_argument("--timeout", type=int, default=300)
    repair.add_argument("--command", required=True)
    args = parser.parse_args(argv)

    socket_path = args.socket or Path(
        os.environ.get(
            "SENPAI_SUPERVISOR_REPAIR_SOCKET",
            "/run/senpai-repair/repair.sock",
        )
    )
    research_tag = args.research_tag or os.environ.get("SENPAI_RESEARCH_TAG")
    if not research_tag:
        parser.error("--research-tag or SENPAI_RESEARCH_TAG is required")
    target = RoleTarget(
        research_tag=research_tag,
        role=args.role,
        student=args.student,
    )
    result = RepairBrokerClient(socket_path).execute(
        RepairRequest(
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
