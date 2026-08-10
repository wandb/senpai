"""Authenticated, crash-bounded controller quiescence for role repairs."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import stat
import struct
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Iterator, Self
from uuid import uuid4

import psutil

from senpai_agent.socket_framing import (
    SocketFrameError,
    encode_json_frame,
    receive_frame,
    unix_socket_address,
)

REPAIR_PAUSE_PROTOCOL = "senpai-controller-repair-pause/v2"
DEFAULT_REPAIR_PAUSE_SOCKET = "@senpai-controller-repair-pause-v2"
_LEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_CAPABILITY_HASH = re.compile(r"^[0-9a-f]{64}$")
_FRAME_LIMIT_BYTES = 16 * 1024
_REQUEST_TIMEOUT_SECONDS = 5.0
_MAX_WAIT_SECONDS = 120.0
_MAX_DURATION_SECONDS = 3_900.0


class RepairPauseError(RuntimeError):
    """A private controller pause could not be established or released."""


@dataclass(frozen=True, slots=True)
class RepairPause:
    protocol: str
    lease_id: str
    expires_at: float
    requester_pid: int
    requester_start_time: float
    acknowledged_at: float | None = None
    supervisor_pid: int | None = None
    resume_capability_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.protocol != REPAIR_PAUSE_PROTOCOL:
            raise ValueError("unsupported repair-pause protocol")
        if not _LEASE_ID.fullmatch(self.lease_id):
            raise ValueError("invalid repair-pause lease id")
        if (
            not self.expires_at > 0
            or self.requester_pid <= 0
            or self.requester_start_time <= 0
        ):
            raise ValueError("invalid repair-pause expiry")
        acknowledgement = (
            self.acknowledged_at,
            self.supervisor_pid,
            self.resume_capability_sha256,
        )
        if all(value is None for value in acknowledgement):
            return
        if any(value is None for value in acknowledgement):
            raise ValueError("partial repair-pause acknowledgement")
        if (
            not 0 < self.acknowledged_at < self.expires_at
            or self.supervisor_pid <= 0
            or not _CAPABILITY_HASH.fullmatch(self.resume_capability_sha256)
        ):
            raise ValueError("invalid persisted repair-pause acknowledgement")


@dataclass(frozen=True, slots=True)
class RepairPauseAcknowledgement:
    protocol: str
    lease_id: str
    expires_at: float
    acknowledged_at: float
    supervisor_pid: int
    resume_capability_sha256: str

    def __post_init__(self) -> None:
        if self.protocol != REPAIR_PAUSE_PROTOCOL:
            raise ValueError("unsupported repair-pause protocol")
        if not _LEASE_ID.fullmatch(self.lease_id):
            raise ValueError("invalid repair-pause lease id")
        if (
            not self.expires_at > self.acknowledged_at > 0
            or self.supervisor_pid <= 0
            or not _CAPABILITY_HASH.fullmatch(self.resume_capability_sha256)
        ):
            raise ValueError("invalid repair-pause acknowledgement")


@dataclass(frozen=True, slots=True)
class RepairPauseGrant:
    acknowledgement: RepairPauseAcknowledgement
    resume_capability: str

    def __post_init__(self) -> None:
        if not self.resume_capability:
            raise ValueError("repair-pause resume capability is empty")
        observed = hashlib.sha256(self.resume_capability.encode()).hexdigest()
        if not hmac.compare_digest(
            observed,
            self.acknowledgement.resume_capability_sha256,
        ):
            raise ValueError("repair-pause resume capability does not match its grant")


class RepairPauseStore:
    """Persist only the expiring request so PID 1 restarts remain paused."""

    def __init__(self, control_dir: Path):
        self.control_dir = control_dir.resolve()
        self.request_path = self.control_dir / "repair-pause.json"
        self.lock_path = self.control_dir / "repair-pause.lock"

    def request(self, lease_id: str, *, duration_seconds: float) -> RepairPause:
        if not 0 < duration_seconds <= _MAX_DURATION_SECONDS:
            raise ValueError("repair-pause duration is outside its bounded range")
        pause = RepairPause(
            protocol=REPAIR_PAUSE_PROTOCOL,
            lease_id=lease_id,
            expires_at=time.monotonic() + duration_seconds,
            requester_pid=os.getpid(),
            requester_start_time=psutil.Process().create_time(),
        )
        self._prepare_directory()
        with self._locked():
            self._discard_expired_unlocked()
            if self.request_path.exists():
                raise RepairPauseError("another role repair pause is already active")
            self._write_atomic(self.request_path, asdict(pause))
        return pause

    def current(self) -> RepairPause | None:
        try:
            pause = self._read(self.request_path)
        except FileNotFoundError:
            return None
        if pause.expires_at <= time.monotonic():
            self.remove_if_matches(pause.lease_id)
            return None
        return pause

    def record_acknowledgement(
        self,
        pause: RepairPause,
        acknowledgement: RepairPauseAcknowledgement,
    ) -> RepairPause:
        self._prepare_directory()
        with self._locked():
            current = self._current_unlocked()
            if current != pause or current.acknowledged_at is not None:
                raise RepairPauseError(
                    "repair-pause lease changed before acknowledgement"
                )
            acknowledged = RepairPause(
                **{
                    **asdict(current),
                    "acknowledged_at": acknowledgement.acknowledged_at,
                    "supervisor_pid": acknowledgement.supervisor_pid,
                    "resume_capability_sha256": (
                        acknowledgement.resume_capability_sha256
                    ),
                }
            )
            self._write_atomic(self.request_path, asdict(acknowledged))
            return acknowledged

    def release(self, lease_id: str, resume_capability: str) -> None:
        self._prepare_directory()
        with self._locked():
            pause = self._current_unlocked()
            if pause is None:
                raise RepairPauseError("repair-pause lease is not active")
            if pause.lease_id != lease_id:
                raise RepairPauseError("refusing to release another repair-pause lease")
            if pause.resume_capability_sha256 is None:
                raise RepairPauseError("repair-pause lease has not been acknowledged")
            observed = hashlib.sha256(resume_capability.encode()).hexdigest()
            if not hmac.compare_digest(observed, pause.resume_capability_sha256):
                raise RepairPauseError("repair-pause resume capability was rejected")
            self.request_path.unlink(missing_ok=True)

    def remove_if_matches(self, lease_id: str) -> None:
        self._prepare_directory()
        with self._locked():
            try:
                pause = self._read(self.request_path)
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
                return
            if pause.lease_id == lease_id:
                self.request_path.unlink(missing_ok=True)

    def _discard_expired_unlocked(self) -> None:
        try:
            pause = self._read(self.request_path)
        except FileNotFoundError:
            return
        if pause.expires_at <= time.monotonic():
            self.request_path.unlink(missing_ok=True)

    def _current_unlocked(self) -> RepairPause | None:
        try:
            pause = self._read(self.request_path)
        except FileNotFoundError:
            return None
        if pause.expires_at <= time.monotonic():
            self.request_path.unlink(missing_ok=True)
            return None
        return pause

    def _prepare_directory(self) -> None:
        self.control_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = self.control_dir.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise RepairPauseError("repair-pause control directory must be private")

    @staticmethod
    def _read(path: Path) -> RepairPause:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise RepairPauseError("repair-pause state must be a private regular file")
        return RepairPause(**json.loads(path.read_text(encoding="utf-8")))

    def _write_atomic(self, path: Path, payload: dict[str, object]) -> None:
        self._prepare_directory()
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            os.write(descriptor, json.dumps(payload, sort_keys=True).encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            temporary.replace(path)
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            directory = os.open(self.control_dir, directory_flags)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


class RepairPauseControlServer:
    """PID 1-owned rendezvous; only this live process can emit an acknowledgement."""

    def __init__(
        self,
        store: RepairPauseStore,
        socket_path: str | Path = DEFAULT_REPAIR_PAUSE_SOCKET,
    ):
        self.store = store
        self.socket_path = str(socket_path)
        self.supervisor_pid = os.getpid()
        self._condition = threading.Condition()
        self._grant: RepairPauseGrant | None = None
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="senpai-repair-pause-control",
            daemon=True,
        )
        self._handlers_lock = threading.Lock()
        self._handlers: set[threading.Thread] = set()
        self._handler_slots = threading.BoundedSemaphore(8)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        filesystem_path = (
            None if self.socket_path.startswith("@") else Path(self.socket_path)
        )
        if filesystem_path is not None:
            filesystem_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                mode = filesystem_path.lstat().st_mode
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISSOCK(mode):
                    raise RepairPauseError(
                        f"refusing to replace non-socket path: {filesystem_path}"
                    )
                filesystem_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        listener.bind(unix_socket_address(self.socket_path))
        if filesystem_path is not None:
            os.chmod(filesystem_path, 0o600)
        listener.listen(8)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        listener = self._listener
        if listener is not None:
            listener.close()
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
        with self._handlers_lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            handler.join(timeout=1)
        if not self.socket_path.startswith("@"):
            Path(self.socket_path).unlink(missing_ok=True)

    def acknowledge(self, pause: RepairPause) -> RepairPauseAcknowledgement:
        current = self.store.current()
        if current != pause:
            raise RepairPauseError("repair-pause lease changed before acknowledgement")
        if current.acknowledged_at is not None:
            acknowledgement = self._acknowledgement_from_pause(current)
            with self._condition:
                self._condition.notify_all()
            return acknowledgement
        resume_capability = secrets.token_urlsafe(32)
        acknowledgement = RepairPauseAcknowledgement(
            protocol=REPAIR_PAUSE_PROTOCOL,
            lease_id=pause.lease_id,
            expires_at=pause.expires_at,
            acknowledged_at=time.monotonic(),
            supervisor_pid=self.supervisor_pid,
            resume_capability_sha256=hashlib.sha256(
                resume_capability.encode()
            ).hexdigest(),
        )
        self.store.record_acknowledgement(pause, acknowledgement)
        with self._condition:
            self._grant = RepairPauseGrant(
                acknowledgement=acknowledgement,
                resume_capability=resume_capability,
            )
            self._condition.notify_all()
        return acknowledgement

    def acknowledgement(self) -> RepairPauseAcknowledgement | None:
        current = self.store.current()
        if current is None or current.acknowledged_at is None:
            return None
        return self._acknowledgement_from_pause(current)

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
            if not self._handler_slots.acquire(blocking=False):
                connection.close()
                continue
            connection.set_inheritable(False)
            handler = threading.Thread(
                target=self._handle,
                args=(connection,),
                name="senpai-repair-pause-request",
                daemon=True,
            )
            with self._handlers_lock:
                self._handlers.add(handler)
            handler.start()

    def _handle(self, connection: socket.socket) -> None:
        try:
            with connection:
                try:
                    request = json.loads(
                        receive_frame(
                            connection,
                            max_bytes=_FRAME_LIMIT_BYTES,
                            timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
                        )
                    )
                    response = self._dispatch(request)
                except Exception as error:  # noqa: BLE001
                    response = {
                        "protocol": REPAIR_PAUSE_PROTOCOL,
                        "error_type": type(error).__name__,
                        "error": str(error)[:4_096],
                    }
                try:
                    connection.sendall(
                        encode_json_frame(response, max_bytes=_FRAME_LIMIT_BYTES)
                    )
                except (OSError, SocketFrameError):
                    pass
        finally:
            with self._handlers_lock:
                self._handlers.discard(threading.current_thread())
            self._handler_slots.release()

    def _dispatch(self, request: object) -> dict[str, object]:
        if not isinstance(request, dict):
            raise ValueError("invalid repair-pause request")
        if request.get("protocol") != REPAIR_PAUSE_PROTOCOL:
            raise ValueError("unsupported repair-pause protocol")
        operation = request.get("operation")
        if operation == "pause":
            wait_seconds = float(request["wait_seconds"])
            if not 0 < wait_seconds <= _MAX_WAIT_SECONDS:
                raise ValueError("repair-pause wait is outside its bounded range")
            pause = self.store.request(
                str(request["lease_id"]),
                duration_seconds=float(request["duration_seconds"]),
            )
            grant = self._wait_for_acknowledgement(pause, wait_seconds)
            return {
                "protocol": REPAIR_PAUSE_PROTOCOL,
                "operation": "pause",
                "acknowledgement": asdict(grant.acknowledgement),
                "resume_capability": grant.resume_capability,
            }
        if operation == "resume":
            lease_id = str(request["lease_id"])
            resume_capability = str(request["resume_capability"])
            self.store.release(lease_id, resume_capability)
            with self._condition:
                if (
                    self._grant is not None
                    and self._grant.acknowledgement.lease_id == lease_id
                ):
                    self._grant = None
                self._condition.notify_all()
            return {
                "protocol": REPAIR_PAUSE_PROTOCOL,
                "operation": "resume",
                "lease_id": lease_id,
                "released": True,
                "supervisor_pid": self.supervisor_pid,
            }
        if operation == "status":
            acknowledgement = self.acknowledgement()
            current = self.store.current()
            state = (
                "paused"
                if acknowledgement is not None
                else "quiescing"
                if current is not None
                else "idle"
            )
            return {
                "protocol": REPAIR_PAUSE_PROTOCOL,
                "operation": "status",
                "state": state,
                "acknowledgement": (
                    asdict(acknowledgement)
                    if acknowledgement is not None
                    else None
                ),
                "supervisor_pid": self.supervisor_pid,
            }
        raise ValueError("unsupported repair-pause operation")

    def _wait_for_acknowledgement(
        self,
        pause: RepairPause,
        wait_seconds: float,
    ) -> RepairPauseGrant:
        deadline = time.monotonic() + wait_seconds
        with self._condition:
            while not self._stop.is_set():
                grant = self._grant
                if (
                    grant is not None
                    and grant.acknowledgement.lease_id == pause.lease_id
                    and grant.acknowledgement.expires_at == pause.expires_at
                ):
                    self._grant = None
                    return grant
                current = self.store.current()
                if current is None or current.lease_id != pause.lease_id:
                    raise RepairPauseError(
                        "repair-pause lease disappeared before acknowledgement"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.store.remove_if_matches(pause.lease_id)
                    raise RepairPauseError(
                        "controller did not quiesce before the repair-pause deadline"
                    )
                self._condition.wait(min(remaining, 0.1))
        raise RepairPauseError("repair-pause control server stopped")

    @staticmethod
    def _acknowledgement_from_pause(
        pause: RepairPause,
    ) -> RepairPauseAcknowledgement:
        if (
            pause.acknowledged_at is None
            or pause.supervisor_pid is None
            or pause.resume_capability_sha256 is None
        ):
            raise RepairPauseError("repair-pause lease is not acknowledged")
        return RepairPauseAcknowledgement(
            protocol=pause.protocol,
            lease_id=pause.lease_id,
            expires_at=pause.expires_at,
            acknowledged_at=pause.acknowledged_at,
            supervisor_pid=pause.supervisor_pid,
            resume_capability_sha256=pause.resume_capability_sha256,
        )


PeerPid = Callable[[socket.socket], int]


def linux_peer_pid(connection: socket.socket) -> int:
    """Return the authenticated peer PID from Linux AF_UNIX credentials."""

    if not sys.platform.startswith("linux") or not hasattr(socket, "SO_PEERCRED"):
        raise RepairPauseError("repair-pause peer credentials require Linux")
    credentials = connection.getsockopt(
        socket.SOL_SOCKET,
        socket.SO_PEERCRED,
        struct.calcsize("3i"),
    )
    pid, _uid, _gid = struct.unpack("3i", credentials)
    return pid


class RepairPauseClient:
    """Authenticate PID 1 and exchange pause, resume, or health messages."""

    def __init__(
        self,
        socket_path: str | Path = DEFAULT_REPAIR_PAUSE_SOCKET,
        *,
        expected_supervisor_pid: int = 1,
        peer_pid: PeerPid = linux_peer_pid,
    ):
        if expected_supervisor_pid <= 0:
            raise ValueError("expected supervisor PID must be positive")
        self.socket_path = str(socket_path)
        self.expected_supervisor_pid = expected_supervisor_pid
        self._peer_pid = peer_pid

    def pause(
        self,
        lease_id: str,
        *,
        duration_seconds: float,
        wait_seconds: float,
    ) -> RepairPauseGrant:
        response = self._request(
            {
                "protocol": REPAIR_PAUSE_PROTOCOL,
                "operation": "pause",
                "lease_id": lease_id,
                "duration_seconds": duration_seconds,
                "wait_seconds": wait_seconds,
            },
            timeout_seconds=wait_seconds + _REQUEST_TIMEOUT_SECONDS,
        )
        if response.get("operation") != "pause":
            raise RepairPauseError("invalid repair-pause response")
        grant = RepairPauseGrant(
            acknowledgement=RepairPauseAcknowledgement(
                **response["acknowledgement"]
            ),
            resume_capability=str(response["resume_capability"]),
        )
        if grant.acknowledgement.supervisor_pid != self.expected_supervisor_pid:
            raise RepairPauseError("repair-pause acknowledgement was not issued by PID 1")
        return grant

    def resume(self, lease_id: str, resume_capability: str) -> None:
        response = self._request(
            {
                "protocol": REPAIR_PAUSE_PROTOCOL,
                "operation": "resume",
                "lease_id": lease_id,
                "resume_capability": resume_capability,
            },
            timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
        )
        expected = {
            "protocol": REPAIR_PAUSE_PROTOCOL,
            "operation": "resume",
            "lease_id": lease_id,
            "released": True,
            "supervisor_pid": self.expected_supervisor_pid,
        }
        if response != expected:
            raise RepairPauseError("invalid repair-resume response")

    def is_paused(self) -> bool:
        return self._status() == "paused"

    def is_quiescing_or_paused(self) -> bool:
        return self._status() in {"quiescing", "paused"}

    def _status(self) -> str:
        response = self._request(
            {"protocol": REPAIR_PAUSE_PROTOCOL, "operation": "status"},
            timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
        )
        if (
            response.get("operation") != "status"
            or response.get("supervisor_pid") != self.expected_supervisor_pid
            or response.get("state") not in {"idle", "quiescing", "paused"}
        ):
            raise RepairPauseError("invalid repair-pause status response")
        if response["state"] != "paused":
            if response.get("acknowledgement") is not None:
                raise RepairPauseError("non-paused status carried an acknowledgement")
            return str(response["state"])
        acknowledgement = RepairPauseAcknowledgement(
            **response["acknowledgement"]
        )
        if acknowledgement.supervisor_pid != self.expected_supervisor_pid:
            raise RepairPauseError("repair-pause status was not issued by PID 1")
        return "paused"

    def validate_peer(self, connection: socket.socket) -> None:
        observed = self._peer_pid(connection)
        if observed != self.expected_supervisor_pid:
            raise RepairPauseError(
                f"repair-pause peer was PID {observed}; expected PID 1"
            )

    def _request(
        self,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(_REQUEST_TIMEOUT_SECONDS)
            connection.connect(unix_socket_address(self.socket_path))
            self.validate_peer(connection)
            connection.sendall(
                encode_json_frame(payload, max_bytes=_FRAME_LIMIT_BYTES)
            )
            response = json.loads(
                receive_frame(
                    connection,
                    max_bytes=_FRAME_LIMIT_BYTES,
                    timeout_seconds=timeout_seconds,
                )
            )
        except (OSError, SocketFrameError, TypeError, ValueError) as error:
            raise RepairPauseError("repair-pause control transport failed") from error
        finally:
            connection.close()
        if not isinstance(response, dict):
            raise RepairPauseError("repair-pause control returned invalid data")
        if response.get("protocol") != REPAIR_PAUSE_PROTOCOL:
            raise RepairPauseError("repair-pause protocol mismatch")
        if "error" in response:
            raise RepairPauseError(
                f"{response.get('error_type', 'RepairPauseError')}: "
                f"{response['error']}"
            )
        return response
