#!/usr/bin/env python3
"""Stdlib-only PID-1 repair executor copied into immutable role images."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

REPAIR_STREAM_LIMIT_BYTES = 128 * 1024
REPAIR_STREAM_LIMIT_CHARS = REPAIR_STREAM_LIMIT_BYTES
EXECUTOR_FRAME_LIMIT_BYTES = 2 * 1024 * 1024
EXECUTOR_REQUEST_LIMIT_BYTES = 512 * 1024
DEFAULT_EXECUTOR_SOCKET = "/run/senpai-repair-executor/executor.sock"
REPAIR_EXECUTOR_PROTOCOL = "senpai-repair-executor/v4"
_POLL_SECONDS = 0.02
_HEARTBEAT_INTERVAL_SECONDS = 0.2
_HEARTBEAT_STALE_SECONDS = 2.0
_OPERATION_DEADLINE_GRACE_SECONDS = 5.0
_TRUSTED_COMMAND_PATH = "/opt/senpai-venv/bin:/usr/local/bin:/usr/bin:/bin"
_VOLATILE_PARENT_NAME = "senpai-repair-operations"
_VOLATILE_CHILD_PREFIX = "operation-"
_VOLATILE_OWNER_FILE = ".senpai-owner.json"
_VOLATILE_OWNER_LIMIT_BYTES = 4_096
_PR_SET_CHILD_SUBREAPER = 36
_REPAIR_ENVIRONMENT_KEYS = frozenset(
    {
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SENPAI_RESEARCH_TAG",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


class RepairExecutorError(RuntimeError):
    pass


class RepairExecutorOutcomeUnknown(RepairExecutorError):
    pass


def _process_start_token(pid: int) -> str | None:
    if sys.platform != "linux":
        return None
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def _volatile_parent() -> Path:
    parent = Path(tempfile.gettempdir()) / _VOLATILE_PARENT_NAME
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        metadata = parent.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise RepairExecutorError("repair volatile root is not private")
        parent.chmod(0o700)
    return parent


def _write_volatile_owner(path: Path) -> None:
    (path / _VOLATILE_OWNER_FILE).write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "start_token": _process_start_token(os.getpid()),
            }
        )
    )


def _volatile_child_name_is_valid(name: str, prefix: str) -> bool:
    suffix = name.removeprefix(prefix)
    return (
        name.startswith(prefix)
        and len(suffix) >= 6
        and all(character.isalnum() or character in "_-" for character in suffix)
    )


def _read_volatile_owner(directory_fd: int) -> tuple[dict[str, object], bytes]:
    try:
        expected = os.stat(
            _VOLATILE_OWNER_FILE,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RepairExecutorError(
            "repair volatile root owner marker is not authoritative"
        ) from error
    if not stat.S_ISREG(expected.st_mode) or expected.st_uid != os.geteuid():
        raise RepairExecutorError(
            "repair volatile root owner marker is not authoritative"
        )
    os.chmod(
        _VOLATILE_OWNER_FILE,
        0o600,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        marker_fd = os.open(_VOLATILE_OWNER_FILE, flags, dir_fd=directory_fd)
    except OSError as error:
        raise RepairExecutorError(
            "repair volatile root owner marker is not authoritative"
        ) from error
    try:
        metadata = os.fstat(marker_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_dev != expected.st_dev
            or metadata.st_ino != expected.st_ino
        ):
            raise RepairExecutorError(
                "repair volatile root owner marker is not authoritative"
            )
        chunks: list[bytes] = []
        remaining = _VOLATILE_OWNER_LIMIT_BYTES + 1
        while remaining:
            chunk = os.read(marker_fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > _VOLATILE_OWNER_LIMIT_BYTES:
            raise RepairExecutorError("repair volatile root owner marker is malformed")
    finally:
        os.close(marker_fd)
    try:
        marker = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairExecutorError(
            "repair volatile root owner marker is malformed"
        ) from error
    if not isinstance(marker, dict):
        raise RepairExecutorError("repair volatile root owner marker is malformed")
    pid = marker.get("pid")
    start_token = marker.get("start_token")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or pid > 2_147_483_647
        or (
            start_token is not None
            and (not isinstance(start_token, str) or not start_token)
        )
    ):
        raise RepairExecutorError("repair volatile root owner marker is malformed")
    return marker, encoded


def _volatile_owner_is_live(marker: dict[str, object]) -> bool:
    pid = int(marker["pid"])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    expected = marker.get("start_token")
    return expected is None or _process_start_token(pid) == expected


def _open_owned_directory(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise RepairExecutorError("repair volatile root ownership is not authoritative")
    os.chmod(name, 0o700, dir_fd=parent_fd, follow_symlinks=False)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(name, flags, dir_fd=parent_fd)
    opened = os.fstat(directory_fd)
    if (
        opened.st_dev != metadata.st_dev
        or opened.st_ino != metadata.st_ino
        or opened.st_uid != os.geteuid()
    ):
        os.close(directory_fd)
        raise RepairExecutorError("repair volatile root changed during cleanup")
    return directory_fd, opened


def _remove_owned_directory_contents(
    directory_fd: int,
    device: int,
    *,
    preserve: frozenset[str] = frozenset(),
) -> None:
    for name in os.listdir(directory_fd):
        if name in preserve:
            continue
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if metadata.st_uid != os.geteuid():
            raise RepairExecutorError(
                "repair volatile root contains an unowned filesystem entry"
            )
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            if metadata.st_dev != device:
                raise RepairExecutorError(
                    "repair volatile root crosses a filesystem boundary"
                )
            child_fd, opened = _open_owned_directory(directory_fd, name)
            try:
                if opened.st_dev != device:
                    raise RepairExecutorError(
                        "repair volatile root crosses a filesystem boundary"
                    )
                _remove_owned_directory_contents(child_fd, device)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _restore_volatile_owner(directory_fd: int, encoded: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    marker_fd = os.open(
        _VOLATILE_OWNER_FILE,
        flags,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        written = 0
        while written < len(encoded):
            count = os.write(marker_fd, encoded[written:])
            if count == 0:
                raise RepairExecutorError(
                    "repair volatile root owner marker could not be restored"
                )
            written += count
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)
    os.fsync(directory_fd)


def remove_owned_volatile_root(
    path: Path,
    *,
    parent: Path,
    child_prefix: str,
    stale_only: bool,
) -> bool:
    """Remove one authority-checked root without following any contained symlink."""

    path = path.absolute()
    parent = parent.absolute()
    if path.parent != parent or not _volatile_child_name_is_valid(
        path.name,
        child_prefix,
    ):
        raise RepairExecutorError("repair volatile root path is not authoritative")
    try:
        parent_metadata = parent.lstat()
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
        ):
            raise RepairExecutorError("repair volatile parent is not authoritative")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        parent_fd = os.open(parent, flags)
        try:
            root_fd, root_metadata = _open_owned_directory(parent_fd, path.name)
            try:
                marker, encoded_marker = _read_volatile_owner(root_fd)
                if stale_only and _volatile_owner_is_live(marker):
                    return False
                _remove_owned_directory_contents(
                    root_fd,
                    root_metadata.st_dev,
                    preserve=frozenset({_VOLATILE_OWNER_FILE}),
                )
                os.unlink(_VOLATILE_OWNER_FILE, dir_fd=root_fd)
                try:
                    os.rmdir(path.name, dir_fd=parent_fd)
                except OSError:
                    _restore_volatile_owner(root_fd, encoded_marker)
                    raise
            finally:
                os.close(root_fd)
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return True
            raise RepairExecutorError(
                "repair volatile root removal was not authoritative"
            )
        finally:
            os.close(parent_fd)
    except RepairExecutorError:
        raise
    except OSError as error:
        raise RepairExecutorError("repair volatile root cleanup failed") from error


def _scavenge_stale_volatile_roots() -> None:
    parent = _volatile_parent()
    for entry in parent.iterdir():
        if not _volatile_child_name_is_valid(entry.name, _VOLATILE_CHILD_PREFIX):
            continue
        metadata = entry.lstat()
        if metadata.st_uid != os.geteuid():
            raise RepairExecutorError("repair volatile root is not owned")
        if stat.S_ISLNK(metadata.st_mode):
            entry.unlink()
        elif stat.S_ISDIR(metadata.st_mode):
            remove_owned_volatile_root(
                entry,
                parent=parent,
                child_prefix=_VOLATILE_CHILD_PREFIX,
                stale_only=True,
            )
        else:
            raise RepairExecutorError("repair volatile root candidate is malformed")


def _socket_address(socket_path: str | Path) -> str:
    value = str(socket_path)
    if not value.startswith("@"):
        return value
    if sys.platform != "linux":
        raise RepairExecutorError("abstract repair sockets require Linux")
    return f"\0{value[1:]}"


def _filesystem_socket_path(socket_path: str | Path) -> Path | None:
    return None if str(socket_path).startswith("@") else Path(socket_path)


def _remove_socket_path(
    socket_path: Path,
    *,
    replace_poisoned: bool = False,
) -> None:
    try:
        mode = socket_path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISSOCK(mode) or (replace_poisoned and not stat.S_ISDIR(mode)):
        socket_path.unlink()
        return
    if replace_poisoned:
        shutil.rmtree(socket_path)
        return
    if not stat.S_ISSOCK(mode):
        raise RepairExecutorError(f"refusing to replace non-socket path: {socket_path}")


def _is_managed_executor_socket_path(socket_path: Path) -> bool:
    return str(socket_path) in {
        DEFAULT_EXECUTOR_SOCKET,
        f"{DEFAULT_EXECUTOR_SOCKET}.health",
    }


def _health_socket_path(socket_path: str | Path) -> str:
    return f"{socket_path}.health"


def _heartbeat_status_is_healthy(
    status: object,
    *,
    expected_pid: int,
    now: float | None = None,
) -> bool:
    """Validate a heartbeat published by the executor's PID-1 main loop."""

    if not isinstance(status, dict):
        return False
    observed_at = time.monotonic() if now is None else now
    try:
        protocol = status["protocol"]
        server_pid = int(status["server_pid"])
        state = status["state"]
        heartbeat = float(status["heartbeat_monotonic"])
        deadline_value = status["operation_deadline_monotonic"]
    except (KeyError, TypeError, ValueError):
        return False
    if (
        protocol != REPAIR_EXECUTOR_PROTOCOL
        or server_pid != expected_pid
        or state not in {"idle", "active", "stopping"}
        or heartbeat > observed_at + _HEARTBEAT_STALE_SECONDS
        or observed_at - heartbeat > _HEARTBEAT_STALE_SECONDS
    ):
        return False
    if state != "active":
        return deadline_value is None
    try:
        deadline = float(deadline_value)
    except (TypeError, ValueError):
        return False
    return observed_at <= deadline + _OPERATION_DEADLINE_GRACE_SECONDS


class _HeartbeatPublisher:
    """Expose main-loop liveness on an immutable abstract health endpoint."""

    def __init__(self, socket_path: str | Path):
        self.socket_path = _health_socket_path(socket_path)
        self._lock = threading.Lock()
        self._state = "starting"
        self._updated_at = time.monotonic()
        self._operation_deadline: float | None = None
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="senpai-repair-health",
            daemon=True,
        )

    def start(self) -> None:
        filesystem_path = _filesystem_socket_path(self.socket_path)
        if filesystem_path is not None:
            filesystem_path.parent.mkdir(parents=True, exist_ok=True)
            _remove_socket_path(
                filesystem_path,
                replace_poisoned=_is_managed_executor_socket_path(filesystem_path),
            )
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(_socket_address(self.socket_path))
        if filesystem_path is not None:
            os.chmod(filesystem_path, 0o600)
        listener.listen(4)
        listener.settimeout(_HEARTBEAT_INTERVAL_SECONDS)
        self._listener = listener
        self._thread.start()

    def publish(
        self,
        state: str,
        operation_deadline: float | None = None,
    ) -> None:
        with self._lock:
            self._state = state
            self._operation_deadline = operation_deadline
            self._updated_at = time.monotonic()

    def close_in_child(self) -> None:
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    def close(self) -> None:
        self.publish("stopping")
        self._stop.set()
        listener = self._listener
        if listener is not None:
            listener.close()
        if self._thread.ident is not None:
            self._thread.join(timeout=1)
        filesystem_path = _filesystem_socket_path(self.socket_path)
        if filesystem_path is not None:
            _remove_socket_path(
                filesystem_path,
                replace_poisoned=_is_managed_executor_socket_path(filesystem_path),
            )

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
                    "protocol": REPAIR_EXECUTOR_PROTOCOL,
                    "server_pid": os.getpid(),
                    "state": self._state,
                    "heartbeat_monotonic": self._updated_at,
                    "operation_deadline_monotonic": self._operation_deadline,
                }
                try:
                    connection.sendall(_encode_frame(status, 16 * 1024))
                except OSError:
                    continue


def check_repair_executor_health(
    socket_path: str | Path,
    *,
    expected_pid: int = 1,
) -> None:
    """Fail unless the executor main loop is live or supervising an in-budget task."""

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(1)
        connection.connect(_socket_address(_health_socket_path(socket_path)))
        status = json.loads(_receive_frame(connection, 16 * 1024, 1))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairExecutorError("repair executor heartbeat is unavailable") from error
    finally:
        connection.close()
    if not _heartbeat_status_is_healthy(status, expected_pid=expected_pid):
        raise RepairExecutorError("repair executor heartbeat is stale or invalid")


@dataclass(frozen=True, slots=True)
class RepairExecutionRequest:
    command: str
    cwd: Path
    timeout_seconds: float

    @classmethod
    def parse(cls, payload: bytes) -> RepairExecutionRequest:
        try:
            value = json.loads(payload)
            protocol = value["protocol"]
            command = value["command"]
            cwd = Path(value["cwd"])
            timeout_seconds = float(value["timeout_seconds"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RepairExecutorError("invalid repair execution request") from error
        if protocol != REPAIR_EXECUTOR_PROTOCOL:
            raise RepairExecutorError("unsupported repair executor protocol")
        if not isinstance(command, str) or not command or len(command) > 65_536:
            raise RepairExecutorError("repair command must contain 1-65536 characters")
        if not cwd.is_absolute():
            raise RepairExecutorError("repair working directory must be absolute")
        if not 0 < timeout_seconds <= 3_600:
            raise RepairExecutorError("repair timeout must be between 0 and 3600 seconds")
        return cls(command=command, cwd=cwd, timeout_seconds=timeout_seconds)

    def payload(self) -> dict[str, str | float]:
        return {
            "protocol": REPAIR_EXECUTOR_PROTOCOL,
            "command": self.command,
            "cwd": str(self.cwd),
            "timeout_seconds": self.timeout_seconds,
        }


class _BoundedByteTail:
    def __init__(self, limit: int = REPAIR_STREAM_LIMIT_BYTES):
        if limit < 1:
            raise ValueError("repair output limit must be positive")
        self.limit = limit
        self._value = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        if len(chunk) >= self.limit:
            if self._value or len(chunk) > self.limit:
                self.truncated = True
            self._value[:] = chunk[-self.limit :]
            return
        self._value.extend(chunk)
        overflow = len(self._value) - self.limit
        if overflow > 0:
            del self._value[:overflow]
            self.truncated = True

    def text(self) -> str:
        value = bytes(self._value).decode("utf-8", errors="replace")
        encoded = value.encode("utf-8")
        if len(encoded) <= self.limit:
            return value
        self.truncated = True
        tail = encoded[-self.limit :]
        for offset in range(min(4, len(tail) + 1)):
            try:
                return tail[offset:].decode("utf-8")
            except UnicodeDecodeError:
                continue
        raise RuntimeError("could not bound repair output at a UTF-8 boundary")


def _receive_frame(
    connection: socket.socket,
    max_bytes: int,
    timeout_seconds: float,
) -> bytes:
    if timeout_seconds <= 0:
        raise RepairExecutorError("repair executor frame timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    previous_timeout = connection.gettimeout()
    payload = bytearray()
    try:
        while True:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise RepairExecutorError("repair executor frame deadline expired")
            connection.settimeout(remaining_time)
            remaining = max_bytes - len(payload)
            try:
                chunk = connection.recv(min(65_536, remaining + 1))
            except TimeoutError as error:
                raise RepairExecutorError(
                    "repair executor frame deadline expired"
                ) from error
            if not chunk:
                return bytes(payload)
            newline = chunk.find(b"\n")
            if newline >= 0:
                payload.extend(chunk[:newline])
                if len(payload) > max_bytes:
                    raise RepairExecutorError(
                        "repair executor frame exceeded its byte limit"
                    )
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise RepairExecutorError("repair executor frame exceeded its byte limit")
    finally:
        try:
            connection.settimeout(previous_timeout)
        except OSError:
            pass


def _encode_frame(value: Any, max_bytes: int = EXECUTOR_FRAME_LIMIT_BYTES) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError) as error:
        raise RepairExecutorError("repair executor produced invalid UTF-8 JSON") from error
    if len(payload) > max_bytes:
        raise RepairExecutorError("repair executor frame exceeded its byte limit")
    return payload + b"\n"


def become_child_subreaper() -> None:
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _process_parents() -> dict[int, int]:
    if sys.platform != "linux":
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            check=True,
            capture_output=True,
            text=True,
        )
        observed = {
            int(pid): int(ppid)
            for line in completed.stdout.splitlines()
            if len(parts := line.split()) == 2
            for pid, ppid in (parts,)
        }
        # The `ps` process can appear in its own snapshot but is gone by the
        # time the command returns. Do not manufacture an immortal descendant.
        for pid in tuple(observed):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                observed.pop(pid, None)
            except PermissionError:
                pass
        return observed
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(stat_fields[1])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
    return parents


def _descendants(parent_pid: int) -> set[int]:
    parents = _process_parents()
    descendants: set[int] = set()
    frontier = {parent_pid}
    while frontier:
        children = {
            pid
            for pid, ppid in parents.items()
            if ppid in frontier and pid not in descendants
        }
        descendants.update(children)
        frontier = children
    return descendants


def _signal_processes(processes: set[int], signum: int) -> None:
    for pid in processes:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            continue


def reap_children() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def clean_current_process_descendants() -> None:
    worker_pid = os.getpid()
    stopped: set[int] = set()
    while True:
        current = _descendants(worker_pid)
        new = current - stopped
        if not new:
            stopped = current
            break
        _signal_processes(new, signal.SIGSTOP)
        stopped.update(new)
    _signal_processes(stopped, signal.SIGKILL)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        reap_children()
        remaining = _descendants(worker_pid)
        if not remaining:
            return
        _signal_processes(remaining, signal.SIGKILL)
        time.sleep(0.005)
    remaining = _descendants(worker_pid)
    if remaining:
        raise RepairExecutorError(
            f"repair cleanup left {len(remaining)} descendant processes"
        )


def kill_process_tree(root_pid: int) -> None:
    """Atomically freeze and kill one external process tree."""

    stopped: set[int] = set()
    while True:
        current = {root_pid, *_descendants(root_pid)}
        new = current - stopped
        if not new:
            stopped = current
            break
        _signal_processes(new, signal.SIGSTOP)
        stopped.update(new)
    _signal_processes(stopped, signal.SIGKILL)


def direct_children(parent_pid: int) -> set[int]:
    return {
        pid
        for pid, process_parent in _process_parents().items()
        if process_parent == parent_pid
    }


def _client_is_connected(connection: socket.socket) -> bool:
    try:
        readable, _, _ = select_with_retry([connection], [], [], 0)
        if not readable:
            return True
        return bool(connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT))
    except OSError:
        return False


def select_with_retry(
    readable: list[socket.socket],
    writable: list[socket.socket],
    exceptional: list[socket.socket],
    timeout: float,
) -> tuple[list[socket.socket], list[socket.socket], list[socket.socket]]:
    import select

    while True:
        try:
            return select.select(readable, writable, exceptional, timeout)
        except InterruptedError:
            continue


def _drain_ready(
    selector: selectors.BaseSelector,
    tails: dict[int, _BoundedByteTail],
    timeout: float,
) -> None:
    for key, _ in selector.select(timeout):
        descriptor = key.fd
        try:
            chunk = os.read(descriptor, 65_536)
        except BlockingIOError:
            continue
        if chunk:
            tails[descriptor].append(chunk)
            continue
        selector.unregister(descriptor)
        os.close(descriptor)


def _fresh_repair_environment(cwd: Path) -> tuple[Path, dict[str, str]]:
    root = Path(
        tempfile.mkdtemp(
            prefix=_VOLATILE_CHILD_PREFIX,
            dir=_volatile_parent(),
        )
    )
    try:
        _write_volatile_owner(root)
        home = root / "home"
        temporary = root / "tmp"
        cache = root / "cache"
        config = root / "config"
        data = root / "data"
        for directory in (home, temporary, cache, config, data):
            directory.mkdir()
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in _REPAIR_ENVIRONMENT_KEYS
        }
        environment.update(
            {
                "PATH": _TRUSTED_COMMAND_PATH,
                "HOME": str(home),
                "TMPDIR": str(temporary),
                "TMP": str(temporary),
                "TEMP": str(temporary),
                "XDG_CACHE_HOME": str(cache),
                "XDG_CONFIG_HOME": str(config),
                "XDG_DATA_HOME": str(data),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": str(cwd),
            }
        )
    except BaseException:
        try:
            remove_owned_volatile_root(
                root,
                parent=_volatile_parent(),
                child_prefix=_VOLATILE_CHILD_PREFIX,
                stale_only=False,
            )
        except BaseException as cleanup_error:
            raise RepairExecutorError(
                "repair environment setup cleanup failed"
            ) from cleanup_error
        raise
    return root, environment


def _execute_in_worker(
    request: RepairExecutionRequest,
    connection: socket.socket,
) -> dict[str, bool | int | str]:
    become_child_subreaper()
    stdout_tail = _BoundedByteTail()
    stderr_tail = _BoundedByteTail()
    stop = False

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stop
        stop = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    volatile_root, environment = _fresh_repair_environment(request.cwd)
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    try:
        try:
            process = subprocess.Popen(
                ["/bin/bash", "-lc", request.command],
                cwd=request.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
        except OSError as error:
            return {
                "exit_code": 126,
                "stdout": "",
                "stderr": f"repair command could not start ({type(error).__name__})",
                "stdout_truncated": False,
                "stderr_truncated": False,
            }
        assert process.stdout is not None
        assert process.stderr is not None
        tails = {
            process.stdout.fileno(): stdout_tail,
            process.stderr.fileno(): stderr_tail,
        }
        for descriptor in tails:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)

        deadline = time.monotonic() + request.timeout_seconds
        outcome = "completed"
        root_return_code: int | None = None
        while True:
            _drain_ready(selector, tails, _POLL_SECONDS)
            root_return_code = process.poll()
            if root_return_code is not None:
                break
            if stop:
                outcome = "shutdown"
                break
            if not _client_is_connected(connection):
                outcome = "disconnected"
                break
            if time.monotonic() >= deadline:
                outcome = "timeout"
                break

        clean_current_process_descendants()
        process.wait(timeout=2)
        drain_deadline = time.monotonic() + 1
        while selector.get_map() and time.monotonic() < drain_deadline:
            _drain_ready(selector, tails, _POLL_SECONDS)
        for descriptor in list(selector.get_map()):
            selector.unregister(descriptor)
            os.close(descriptor)

        if outcome == "timeout":
            stderr_tail.append(b"\nrepair command timed out")
            exit_code = 124
        elif outcome == "disconnected":
            stderr_tail.append(b"\nrepair client disconnected")
            exit_code = 125
        elif outcome == "shutdown":
            stderr_tail.append(b"\nrepair executor shut down")
            exit_code = 143
        else:
            exit_code = root_return_code if root_return_code is not None else 126
        return {
            "exit_code": exit_code,
            "stdout": stdout_tail.text(),
            "stderr": stderr_tail.text(),
            "stdout_truncated": stdout_tail.truncated,
            "stderr_truncated": stderr_tail.truncated,
        }
    finally:
        try:
            if process is not None and process.poll() is None:
                clean_current_process_descendants()
        finally:
            try:
                selector.close()
            finally:
                try:
                    remove_owned_volatile_root(
                        volatile_root,
                        parent=_volatile_parent(),
                        child_prefix=_VOLATILE_CHILD_PREFIX,
                        stale_only=False,
                    )
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)


def _serve_worker(
    connection: socket.socket,
    request: RepairExecutionRequest,
    *,
    result_fd: int | None = None,
) -> None:
    try:
        try:
            frame = _encode_frame(_execute_in_worker(request, connection))
        except BaseException as error:  # noqa: BLE001
            frame = _encode_frame(
                {
                    "error_type": type(error).__name__,
                    "error": str(error)[:4_096],
                }
            )
        if result_fd is not None:
            with os.fdopen(result_fd, "wb", closefd=True) as result_file:
                result_file.write(frame)
                result_file.flush()
        elif _client_is_connected(connection):
            connection.sendall(frame)
    except OSError:
        pass
    finally:
        connection.close()


def _read_worker_result(result_fd: int) -> bytes:
    os.lseek(result_fd, 0, os.SEEK_SET)
    chunks = []
    size = 0
    while size <= EXECUTOR_FRAME_LIMIT_BYTES + 1:
        chunk = os.read(
            result_fd,
            min(65_536, EXECUTOR_FRAME_LIMIT_BYTES + 2 - size),
        )
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    frame = b"".join(chunks)
    if (
        not frame.endswith(b"\n")
        or len(frame) > EXECUTOR_FRAME_LIMIT_BYTES + 1
        or b"\n" in frame[:-1]
    ):
        raise RepairExecutorError("repair worker returned an invalid result frame")
    return frame


def _managed_result_fd() -> int:
    if sys.platform != "linux":
        raise RepairExecutorError("managed repair execution requires Linux")
    return os.memfd_create("senpai-repair-result", os.MFD_CLOEXEC)


def _spawn_worker(
    connection: socket.socket,
    request: RepairExecutionRequest,
    *,
    result_fd: int | None = None,
) -> subprocess.Popen[bytes]:
    """Exec a fresh interpreter so a threaded PID 1 never runs after fork."""

    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "worker",
        "--connection-fd",
        str(connection.fileno()),
    ]
    inherited_fds = [connection.fileno()]
    if result_fd is not None:
        command.extend(("--result-fd", str(result_fd)))
        inherited_fds.append(result_fd)
    worker = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        pass_fds=tuple(inherited_fds),
    )
    assert worker.stdin is not None
    try:
        worker.stdin.write(
            _encode_frame(request.payload(), EXECUTOR_REQUEST_LIMIT_BYTES)
        )
    finally:
        worker.stdin.close()
    return worker


class RepairExecutorServer:
    """Own repair workers and reap them from the container's PID-1 process."""

    def __init__(self, socket_path: str | Path):
        self.socket_path = socket_path
        self._stop = False
        self._active_worker: int | None = None

    def _request_stop(self, _signum: int, _frame: FrameType | None) -> None:
        self._stop = True
        if self._active_worker is not None:
            try:
                os.kill(self._active_worker, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _bind_listener(self) -> socket.socket:
        self._remove_stale_socket()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(_socket_address(self.socket_path))
            filesystem_path = _filesystem_socket_path(self.socket_path)
            if filesystem_path is not None:
                os.chmod(filesystem_path, 0o600)
            listener.listen(1)
            listener.settimeout(0.2)
            return listener
        except BaseException:
            listener.close()
            raise

    def serve_forever(self) -> None:
        become_child_subreaper()
        _scavenge_stale_volatile_roots()
        heartbeat = _HeartbeatPublisher(self.socket_path)
        heartbeat.start()
        filesystem_path = _filesystem_socket_path(self.socket_path)
        if filesystem_path is not None:
            filesystem_path.parent.mkdir(parents=True, exist_ok=True)
        previous_handlers = {
            signum: signal.signal(signum, self._request_stop)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
        listener: socket.socket | None = None
        try:
            while not self._stop:
                if listener is None:
                    listener = self._bind_listener()
                    heartbeat.publish("idle")
                try:
                    while not self._stop:
                        try:
                            connection, _ = listener.accept()
                            break
                        except TimeoutError:
                            reap_children()
                            heartbeat.publish("idle")
                    else:
                        listener.close()
                        break
                except OSError:
                    listener.close()
                    if self._stop:
                        break
                    raise
                try:
                    connection.settimeout(5)
                    request = RepairExecutionRequest.parse(
                        _receive_frame(
                            connection,
                            EXECUTOR_REQUEST_LIMIT_BYTES,
                            5,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    try:
                        connection.sendall(
                            _encode_frame(
                                {
                                    "error_type": type(error).__name__,
                                    "error": str(error)[:4_096],
                                }
                            )
                        )
                    except OSError:
                        pass
                    connection.close()
                    continue

                # The command shares this container. Remove every listening FD
                # and path before it starts so it cannot enqueue a nested,
                # unaudited repair for later execution.
                listener.close()
                listener = None
                self._remove_stale_socket()
                operation_deadline = time.monotonic() + request.timeout_seconds
                heartbeat.publish("active", operation_deadline)
                result_fd = _managed_result_fd()
                try:
                    worker = _spawn_worker(
                        connection,
                        request,
                        result_fd=result_fd,
                    )
                    self._active_worker = worker.pid
                    if self._stop:
                        self._request_stop(signal.SIGTERM, None)
                    while True:
                        try:
                            if worker.poll() is not None:
                                break
                            heartbeat.publish("active", operation_deadline)
                            time.sleep(_HEARTBEAT_INTERVAL_SECONDS)
                        except InterruptedError:
                            if self._stop:
                                self._request_stop(signal.SIGTERM, None)
                            continue
                        except ChildProcessError:
                            break
                    self._active_worker = None
                    # The request worker is normally its own subreaper, but a
                    # crash can reparent detached command descendants to PID 1.
                    # Reconcile before the executor becomes reachable again.
                    clean_current_process_descendants()
                    _scavenge_stale_volatile_roots()
                    try:
                        result_frame = _read_worker_result(result_fd)
                    except RepairExecutorError as error:
                        result_frame = None
                        print(
                            "SENPAI_REPAIR_WORKER_RESULT_ERROR "
                            f"{type(error).__name__}",
                            file=sys.stderr,
                        )

                    # A successful response is a readiness barrier: only
                    # release it after PID 1 has restored the listener.
                    if not self._stop:
                        listener = self._bind_listener()
                        heartbeat.publish("idle")
                    if result_frame is not None and _client_is_connected(connection):
                        try:
                            connection.sendall(result_frame)
                        except OSError:
                            pass
                finally:
                    self._active_worker = None
                    os.close(result_fd)
                    connection.close()
        finally:
            if listener is not None:
                listener.close()
            heartbeat.close()
            self._request_stop(signal.SIGTERM, None)
            self._remove_stale_socket()
            clean_current_process_descendants()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    def _remove_stale_socket(self) -> None:
        socket_path = _filesystem_socket_path(self.socket_path)
        if socket_path is None:
            return
        _remove_socket_path(
            socket_path,
            replace_poisoned=_is_managed_executor_socket_path(socket_path),
        )


class RepairExecutorClient:
    def __init__(self, socket_path: str | Path):
        self.socket_path = socket_path

    def execute(self, request: RepairExecutionRequest) -> dict[str, bool | int | str]:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(5)
            connection.connect(_socket_address(self.socket_path))
        except OSError as error:
            connection.close()
            raise RepairExecutorError("repair executor is unavailable") from error
        request_sent = False
        try:
            with connection:
                request_sent = True
                connection.sendall(
                    _encode_frame(request.payload(), EXECUTOR_REQUEST_LIMIT_BYTES)
                )
                connection.settimeout(request.timeout_seconds + 30)
                response = json.loads(
                    _receive_frame(
                        connection,
                        EXECUTOR_FRAME_LIMIT_BYTES,
                        request.timeout_seconds + 30,
                    )
                )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            if request_sent:
                raise RepairExecutorOutcomeUnknown(
                    "repair command may have executed; its outcome is unknown"
                ) from error
            raise RepairExecutorError("repair executor is unavailable") from error
        if not isinstance(response, dict):
            raise RepairExecutorOutcomeUnknown(
                "repair executor returned an invalid response after execution"
            )
        if "error" in response:
            raise RepairExecutorError(
                f"{response.get('error_type', 'RepairExecutorError')}: "
                f"{response['error']}"
            )
        return response


def execute_local_repair(
    command: str,
    cwd: Path,
    timeout_seconds: float,
) -> dict[str, bool | int | str]:
    """Execute through the same per-request subreaper used by the PID-1 daemon."""

    request = RepairExecutionRequest(command, cwd.resolve(), timeout_seconds)
    parent, worker_connection = socket.socketpair()
    worker = _spawn_worker(worker_connection, request)
    worker_connection.close()
    try:
        response = json.loads(
            _receive_frame(
                parent,
                EXECUTOR_FRAME_LIMIT_BYTES,
                timeout_seconds + 30,
            )
        )
    finally:
        parent.close()
        worker.wait()
    if "error" in response:
        raise RepairExecutorError(
            f"{response.get('error_type', 'RepairExecutorError')}: {response['error']}"
        )
    return response


def repair_executor_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute isolated repair commands.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--socket", default=DEFAULT_EXECUTOR_SOCKET)
    client = subparsers.add_parser("client")
    client.add_argument("--socket", default=DEFAULT_EXECUTOR_SOCKET)
    client.add_argument("--cwd", required=True, type=Path)
    client.add_argument("--timeout", required=True, type=float)
    health = subparsers.add_parser("health")
    health.add_argument("--socket", default=DEFAULT_EXECUTOR_SOCKET)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--connection-fd", required=True, type=int)
    worker.add_argument("--result-fd", type=int)
    args = parser.parse_args(argv)

    if args.operation == "serve":
        if sys.platform != "linux" or os.getpid() != 1:
            parser.error("repair executor serve must be Linux container PID 1")
        RepairExecutorServer(args.socket).serve_forever()
        return 0

    if args.operation == "health":
        if sys.platform != "linux":
            return 1
        try:
            check_repair_executor_health(args.socket)
        except RepairExecutorError:
            return 1
        return 0

    if args.operation == "worker":
        connection = socket.socket(fileno=args.connection_fd)
        payload = sys.stdin.buffer.readline(EXECUTOR_REQUEST_LIMIT_BYTES + 2)
        if not payload.endswith(b"\n") or len(payload) > EXECUTOR_REQUEST_LIMIT_BYTES + 1:
            raise RepairExecutorError("repair executor frame exceeded its byte limit")
        request = RepairExecutionRequest.parse(payload[:-1])
        _serve_worker(connection, request, result_fd=args.result_fd)
        return 0

    request = RepairExecutionRequest(
        command=sys.stdin.read(),
        cwd=args.cwd,
        timeout_seconds=args.timeout,
    )
    try:
        result = RepairExecutorClient(args.socket).execute(request)
    except RepairExecutorOutcomeUnknown as error:
        print(
            json.dumps(
                {
                    "outcome": "unknown",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                separators=(",", ":"),
            )
        )
        return 0
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(repair_executor_main())
