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
DEFAULT_EXECUTOR_SOCKET = "@senpai-repair-executor"
REPAIR_EXECUTOR_PROTOCOL = "senpai-repair-executor/v1"
_POLL_SECONDS = 0.02
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


def _socket_address(socket_path: str | Path) -> str:
    value = str(socket_path)
    if not value.startswith("@"):
        return value
    if sys.platform != "linux":
        raise RepairExecutorError("abstract repair sockets require Linux")
    return f"\0{value[1:]}"


def _filesystem_socket_path(socket_path: str | Path) -> Path | None:
    return None if str(socket_path).startswith("@") else Path(socket_path)


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
        self.limit = limit
        self._value = bytearray()

    def append(self, chunk: bytes) -> None:
        if len(chunk) >= self.limit:
            self._value[:] = chunk[-self.limit :]
            return
        self._value.extend(chunk)
        overflow = len(self._value) - self.limit
        if overflow > 0:
            del self._value[:overflow]

    def text(self) -> str:
        return bytes(self._value).decode("utf-8", errors="replace")


def _receive_frame(connection: socket.socket, max_bytes: int) -> bytes:
    payload = bytearray()
    while True:
        remaining = max_bytes - len(payload)
        chunk = connection.recv(min(65_536, remaining + 1))
        if not chunk:
            return bytes(payload)
        newline = chunk.find(b"\n")
        if newline >= 0:
            payload.extend(chunk[:newline])
            if len(payload) > max_bytes:
                raise RepairExecutorError("repair executor frame exceeded its byte limit")
            return bytes(payload)
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise RepairExecutorError("repair executor frame exceeded its byte limit")


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


def _fresh_repair_environment() -> tuple[Path, dict[str, str]]:
    root = Path(tempfile.mkdtemp(prefix="senpai-repair-operation-"))
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
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "XDG_DATA_HOME": str(data),
        }
    )
    return root, environment


def _execute_in_worker(
    request: RepairExecutionRequest,
    connection: socket.socket,
) -> dict[str, int | str]:
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
    volatile_root, environment = _fresh_repair_environment()
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
        }
    finally:
        try:
            if process is not None and process.poll() is None:
                clean_current_process_descendants()
        finally:
            selector.close()
            shutil.rmtree(volatile_root, ignore_errors=True)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def _serve_worker(
    connection: socket.socket,
    request: RepairExecutionRequest,
) -> None:
    try:
        result = _execute_in_worker(request, connection)
        if _client_is_connected(connection):
            connection.sendall(_encode_frame(result))
    except BaseException as error:  # noqa: BLE001
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
    finally:
        connection.close()


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

    def serve_forever(self) -> None:
        become_child_subreaper()
        filesystem_path = _filesystem_socket_path(self.socket_path)
        if filesystem_path is not None:
            filesystem_path.parent.mkdir(parents=True, exist_ok=True)
        previous_handlers = {
            signum: signal.signal(signum, self._request_stop)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            while not self._stop:
                self._remove_stale_socket()
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(_socket_address(self.socket_path))
                if filesystem_path is not None:
                    os.chmod(filesystem_path, 0o600)
                listener.listen(1)
                listener.settimeout(0.2)
                try:
                    while not self._stop:
                        try:
                            connection, _ = listener.accept()
                            break
                        except TimeoutError:
                            reap_children()
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
                        _receive_frame(connection, EXECUTOR_REQUEST_LIMIT_BYTES)
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
                    listener.close()
                    self._remove_stale_socket()
                    continue

                # The command shares this container. Remove every listening FD
                # and path before it starts so it cannot enqueue a nested,
                # unaudited repair for later execution.
                listener.close()
                self._remove_stale_socket()
                worker = os.fork()
                if worker == 0:
                    _serve_worker(connection, request)
                    os._exit(0)
                connection.close()
                self._active_worker = worker
                while True:
                    try:
                        waited, _ = os.waitpid(worker, 0)
                        if waited == worker:
                            break
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
        finally:
            self._request_stop(signal.SIGTERM, None)
            self._remove_stale_socket()
            clean_current_process_descendants()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    def _remove_stale_socket(self) -> None:
        socket_path = _filesystem_socket_path(self.socket_path)
        if socket_path is None:
            return
        try:
            mode = socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RepairExecutorError(
                f"refusing to replace non-socket path: {socket_path}"
            )
        socket_path.unlink()


class RepairExecutorClient:
    def __init__(self, socket_path: str | Path):
        self.socket_path = socket_path

    def execute(self, request: RepairExecutionRequest) -> dict[str, int | str]:
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
                    _receive_frame(connection, EXECUTOR_FRAME_LIMIT_BYTES)
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
) -> dict[str, int | str]:
    """Execute through the same per-request subreaper used by the PID-1 daemon."""

    request = RepairExecutionRequest(command, cwd.resolve(), timeout_seconds)
    parent, worker_connection = socket.socketpair()
    worker = os.fork()
    if worker == 0:
        parent.close()
        _serve_worker(worker_connection, request)
        os._exit(0)
    worker_connection.close()
    try:
        response = json.loads(_receive_frame(parent, EXECUTOR_FRAME_LIMIT_BYTES))
    finally:
        parent.close()
        os.waitpid(worker, 0)
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
            command = Path("/proc/1/cmdline").read_bytes()
        except OSError:
            return 1
        return int(
            b"senpai-repair-executor" not in command or b"serve" not in command
        )

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
