"""Wake-isolated OpenHands terminal transport for a secret-free sidecar."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, TracebackType
from typing import Literal, Self

from openhands.sdk.tool import ToolDefinition, ToolExecutor, register_tool
from openhands.tools.terminal import TerminalAction, TerminalObservation, TerminalTool

from senpai_agent.repair_executor import (
    become_child_subreaper,
    clean_current_process_descendants,
    direct_children,
    kill_process_tree,
    reap_children,
)
from senpai_agent.socket_framing import (
    SocketFrameError,
    SocketFrameTooLarge,
    encode_json_frame,
    receive_frame,
    unix_socket_address,
)


TERMINAL_PROTOCOL = "senpai-isolated-terminal/v1"
DEFAULT_TERMINAL_SOCKET = "@senpai-isolated-terminal"
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_CONNECT_ATTEMPTS = 20
_CONNECT_DELAY_SECONDS = 0.1
_ENVIRONMENT_LOCK = threading.Lock()
_REQUEST_READ_TIMEOUT_SECONDS = 5
_UNTIMED_RESPONSE_TIMEOUT_SECONDS = 24 * 60 * 60
_RESPONSE_GRACE_SECONDS = 30
_WORKER_START_SECONDS = 30
_WORKER_STOP_SECONDS = 0.25
_WORKER_FORCE_STOP_SECONDS = 0.5
_ADOPTEE_CLEANUP_SECONDS = 0.5
_TERMINAL_ENV_ALLOWLIST = {
    "ALL_PROXY",
    "CURL_CA_BUNDLE",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "NO_PROXY",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SENPAI_RESEARCH_TAG",
    "SENPAI_SUPERVISOR_REPAIR_SOCKET",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
}


class TerminalTransportError(RuntimeError):
    """The isolated terminal process could not execute an action."""


class TerminalOutcomeUnknown(TerminalTransportError):
    """The request was sent, but no authoritative response was received."""


class StaleTerminalWake(TerminalTransportError):
    """A terminal action named a retired or inactive wake."""


class ConcurrentTerminalWake(TerminalTransportError):
    """Two different wake transitions raced."""


class _WorkerShutdown(BaseException):
    pass


@contextmanager
def _process_environment(environment: Mapping[str, str] | None):
    if environment is None:
        yield
        return
    with _ENVIRONMENT_LOCK:
        inherited = dict(os.environ)
        os.environ.clear()
        os.environ.update(environment)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(inherited)


def _receive_line(connection: socket.socket) -> bytes:
    return receive_frame(connection, max_bytes=_MAX_MESSAGE_BYTES)


def _connect(socket_path: Path) -> socket.socket:
    last_error: OSError | None = None
    for attempt in range(_CONNECT_ATTEMPTS):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
            connection.connect(unix_socket_address(socket_path))
            return connection
        except OSError as error:
            connection.close()
            last_error = error
            if attempt + 1 < _CONNECT_ATTEMPTS:
                time.sleep(_CONNECT_DELAY_SECONDS)
    raise TerminalTransportError("isolated terminal sidecar is unavailable") from last_error


def _request(
    socket_path: Path,
    payload: dict[str, object],
    *,
    timeout_seconds: float,
    unknown_message: str,
) -> dict[str, object]:
    try:
        frame = encode_json_frame(payload, max_bytes=_MAX_MESSAGE_BYTES)
    except SocketFrameError as error:
        raise TerminalTransportError(
            "terminal request exceeded the encoded byte limit"
        ) from error
    connection = _connect(socket_path)
    try:
        with connection:
            connection.sendall(frame)
            connection.settimeout(timeout_seconds)
            envelope = json.loads(_receive_line(connection))
    except (OSError, SocketFrameError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TerminalOutcomeUnknown(unknown_message) from error
    if not isinstance(envelope, dict):
        raise TerminalOutcomeUnknown(unknown_message)
    if "error" in envelope:
        error_type = str(envelope.get("error_type", "TerminalError"))
        message = str(envelope["error"])
        if error_type == "StaleTerminalWake":
            raise StaleTerminalWake(message)
        if error_type == "ConcurrentTerminalWake":
            raise ConcurrentTerminalWake(message)
        if error_type == "TerminalOutcomeUnknown":
            raise TerminalOutcomeUnknown(message)
        raise TerminalTransportError(f"{error_type}: {message}")
    return envelope


def begin_isolated_terminal_wake(socket_path: str | Path, wake_id: str) -> None:
    """Atomically replace the prior terminal process tree before a fresh turn."""

    if not wake_id or len(wake_id) > 200:
        raise ValueError("terminal wake ID must contain 1-200 characters")
    response = _request(
        Path(socket_path),
        {
            "protocol": TERMINAL_PROTOCOL,
            "operation": "begin_wake",
            "wake_id": wake_id,
        },
        timeout_seconds=(
            _WORKER_START_SECONDS
            + _WORKER_STOP_SECONDS
            + _WORKER_FORCE_STOP_SECONDS
            + _ADOPTEE_CLEANUP_SECONDS
            + _RESPONSE_GRACE_SECONDS
        ),
        unknown_message=(
            f"terminal wake {wake_id!r} cleanup outcome is unknown; abort the turn"
        ),
    )
    if response.get("wake_id") != wake_id or response.get("status") != "ready":
        raise TerminalTransportError("isolated terminal returned an invalid wake receipt")


def end_isolated_terminal_wake(socket_path: str | Path, wake_id: str) -> None:
    """End a wake and synchronously remove every process that it started."""

    response = _request(
        Path(socket_path),
        {
            "protocol": TERMINAL_PROTOCOL,
            "operation": "end_wake",
            "wake_id": wake_id,
        },
        timeout_seconds=_WORKER_STOP_SECONDS + _RESPONSE_GRACE_SECONDS,
        unknown_message=(
            f"terminal wake {wake_id!r} end outcome is unknown; the next begin "
            "remains authoritative"
        ),
    )
    if response.get("wake_id") != wake_id or response.get("status") != "ended":
        raise TerminalTransportError("isolated terminal returned an invalid end receipt")


def check_isolated_terminal_health(socket_path: str | Path) -> None:
    response = _request(
        Path(socket_path),
        {"protocol": TERMINAL_PROTOCOL, "operation": "health"},
        timeout_seconds=_REQUEST_READ_TIMEOUT_SECONDS,
        unknown_message="isolated terminal health is unknown",
    )
    if response.get("status") != "clean":
        raise TerminalTransportError("isolated terminal is not clean")


class IsolatedTerminalClientExecutor(
    ToolExecutor[TerminalAction, TerminalObservation]
):
    """Forward terminal actions only to the active fresh-wake worker."""

    def __init__(self, socket_path: str | Path, wake_id: str):
        self.socket_path = Path(socket_path)
        self.wake_id = wake_id

    def __call__(
        self,
        action: TerminalAction,
        conversation: object | None = None,
    ) -> TerminalObservation:
        del conversation
        response = _request(
            self.socket_path,
            {
                "protocol": TERMINAL_PROTOCOL,
                "operation": "execute",
                "wake_id": self.wake_id,
                "action": action.model_dump(mode="json"),
            },
            timeout_seconds=(
                (
                    action.timeout
                    if action.timeout is not None
                    else _UNTIMED_RESPONSE_TIMEOUT_SECONDS
                )
                + _RESPONSE_GRACE_SECONDS
            ),
            unknown_message="terminal action may have executed; its outcome is unknown",
        )
        try:
            return TerminalObservation.model_validate(response["observation"])
        except (KeyError, ValueError, TypeError) as error:
            raise TerminalOutcomeUnknown(
                "terminal action may have executed; its observation was invalid"
            ) from error


class IsolatedTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    """Resolve to OpenHands' native terminal schema with a wake-bound executor."""

    name = "terminal"

    @classmethod
    def create(
        cls,
        conv_state: object,
        *,
        socket_path: str,
        wake_id: str,
    ) -> Sequence[ToolDefinition]:
        return TerminalTool.create(
            conv_state,
            executor=IsolatedTerminalClientExecutor(socket_path, wake_id),
        )


def register_isolated_terminal_tool() -> None:
    """Replace `terminal` only in the supervisor's dedicated process."""

    register_tool("terminal", IsolatedTerminalTool)


@dataclass(slots=True)
class _TerminalWorker:
    wake_id: str
    process: subprocess.Popen[bytes]
    connection: socket.socket
    volatile_root: Path
    request_lock: threading.Lock = field(default_factory=threading.Lock)


def _worker_response(response: dict[str, object]) -> bytes:
    try:
        return encode_json_frame(response, max_bytes=_MAX_MESSAGE_BYTES)
    except SocketFrameTooLarge:
        return encode_json_frame(
            {
                "error_type": "TerminalFrameTooLarge",
                "error": "isolated terminal result exceeded the encoded byte limit",
            },
            max_bytes=_MAX_MESSAGE_BYTES,
        )
    except SocketFrameError:
        return encode_json_frame(
            {
                "error_type": "TerminalEncodingError",
                "error": "isolated terminal result was not valid UTF-8",
            },
            max_bytes=_MAX_MESSAGE_BYTES,
        )


def _terminal_worker(
    control_fd: int,
    working_dir: Path,
    terminal_type: Literal["tmux", "subprocess", "powershell"] | None,
) -> int:
    from openhands.tools.terminal.impl import TerminalExecutor

    become_child_subreaper()
    connection = socket.socket(fileno=control_fd)
    connection.set_inheritable(False)

    def stop_worker(_signum: int, _frame: FrameType | None) -> None:
        # Make shutdown one-shot before unwinding arbitrary executor code.
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        raise _WorkerShutdown()

    previous_handlers = {
        signum: signal.signal(signum, stop_worker)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    executor = None
    try:
        environment = dict(os.environ)
        with _process_environment(environment):
            executor = TerminalExecutor(
                working_dir=str(working_dir),
                terminal_type=terminal_type,
                env=environment,
            )
        connection.sendall(
            encode_json_frame({"status": "ready"}, max_bytes=_MAX_MESSAGE_BYTES)
        )
        while True:
            payload = _receive_line(connection)
            if not payload:
                break
            try:
                request = json.loads(payload)
                action = TerminalAction.model_validate(request["action"])
                with _process_environment(environment):
                    observation = executor(action)
                response = {"observation": observation.model_dump(mode="json")}
            except _WorkerShutdown:
                raise
            except Exception as error:  # noqa: BLE001
                response = {
                    "error_type": type(error).__name__,
                    "error": str(error)[:4_096],
                }
            connection.sendall(_worker_response(response))
    except (_WorkerShutdown, BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        # Repeated shutdown signals must not interrupt cleanup. The server is
        # also a subreaper and independently cleans adopted descendants.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if executor is not None:
                try:
                    executor.close()
                except BaseException:  # noqa: BLE001
                    pass
        finally:
            try:
                clean_current_process_descendants()
            finally:
                connection.close()
    return 0


class IsolatedTerminalServer:
    """Own one replaceable terminal worker per fresh supervisor wake."""

    def __init__(
        self,
        *,
        socket_path: str | Path,
        working_dir: str | Path,
        environment: Mapping[str, str] | None = None,
        terminal_type: Literal["tmux", "subprocess", "powershell"] | None = None,
    ):
        self.socket_path = Path(socket_path)
        self.working_dir = Path(working_dir).resolve()
        self.environment = dict(environment or {})
        self.terminal_type = terminal_type
        if not self.working_dir.is_dir():
            raise ValueError(f"terminal workspace does not exist: {self.working_dir}")
        become_child_subreaper()
        reap_children()
        self._baseline_children = frozenset(direct_children(os.getpid()))
        self._abstract_socket = str(self.socket_path).startswith("@")
        if not self._abstract_socket:
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_socket()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(unix_socket_address(self.socket_path))
        if not self._abstract_socket:
            os.chmod(self.socket_path, 0o600)
        self._socket.listen(16)
        self._socket.settimeout(0.2)
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._transition_lock = threading.Lock()
        self._worker: _TerminalWorker | None = None
        self._retired_wakes: set[str] = set()
        self._dirty = False
        self._handlers: set[threading.Thread] = set()
        self._handlers_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._serve,
            name="senpai-isolated-terminal",
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
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._socket.close()
        if self._thread.ident is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        with self._transition_lock:
            with self._state_lock:
                worker = self._worker
                self._worker = None
            if worker is not None:
                self._terminate_worker(worker)
        with self._handlers_lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            if handler is not threading.current_thread():
                handler.join(timeout=2)
        self._remove_stale_socket()

    def serve_forever(self) -> None:
        previous_handlers = {
            signum: signal.signal(signum, lambda *_args: self._stop.set())
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
        self._thread.start()
        try:
            while self._thread.is_alive() and not self._stop.wait(1):
                pass
        finally:
            self.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            handler = threading.Thread(
                target=self._handle,
                args=(connection,),
                name="senpai-isolated-terminal-request",
                daemon=True,
            )
            with self._handlers_lock:
                self._handlers.add(handler)
            handler.start()

    def _handle(self, connection: socket.socket) -> None:
        try:
            with connection:
                connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
                try:
                    request = json.loads(_receive_line(connection))
                    if request.get("protocol") != TERMINAL_PROTOCOL:
                        raise TerminalTransportError(
                            "unsupported isolated terminal protocol"
                        )
                    operation = request.get("operation")
                    if operation == "health":
                        response = self._health()
                        connection.sendall(_worker_response(response))
                        return
                    wake_id = request.get("wake_id")
                    if not isinstance(wake_id, str) or not wake_id:
                        raise TerminalTransportError("terminal wake ID is required")
                    if operation == "begin_wake":
                        response = self._begin_wake(wake_id)
                    elif operation == "end_wake":
                        response = self._end_wake(wake_id)
                    elif operation == "execute":
                        response = self._execute(wake_id, request.get("action"))
                    else:
                        raise TerminalTransportError(
                            "unsupported isolated terminal operation"
                        )
                except Exception as error:  # noqa: BLE001
                    response = {
                        "error_type": type(error).__name__,
                        "error": str(error)[:4_096],
                    }
                try:
                    connection.sendall(_worker_response(response))
                except OSError:
                    pass
        finally:
            with self._handlers_lock:
                self._handlers.discard(threading.current_thread())

    def _begin_wake(self, wake_id: str) -> dict[str, object]:
        if not self._transition_lock.acquire(blocking=False):
            raise ConcurrentTerminalWake("another terminal wake transition is active")
        try:
            with self._state_lock:
                current = self._worker
                if wake_id in self._retired_wakes:
                    raise StaleTerminalWake(f"terminal wake {wake_id!r} is retired")
                if current is not None and current.wake_id == wake_id:
                    if current.process.poll() is None:
                        return {"wake_id": wake_id, "status": "ready"}
                    # A dead worker may have executed an action whose reply was
                    # lost. Retire its ID instead of replaying that wake.
                    self._worker = None
                    self._retired_wakes.add(wake_id)
                if current is not None:
                    self._worker = None
                    self._retired_wakes.add(current.wake_id)
            if current is not None:
                self._terminate_worker(current)
                if current.wake_id == wake_id:
                    raise StaleTerminalWake(
                        f"terminal wake {wake_id!r} crashed and is retired"
                    )
            worker = self._start_worker(wake_id)
            with self._state_lock:
                stopping = self._stop.is_set()
                if not stopping:
                    self._worker = worker
            if stopping:
                self._terminate_worker(worker)
                raise TerminalTransportError("terminal server is stopping")
            return {"wake_id": wake_id, "status": "ready"}
        finally:
            self._transition_lock.release()

    def _health(self) -> dict[str, object]:
        with self._state_lock:
            worker_crashed = (
                self._worker is not None
                and self._worker.process.poll() is not None
            )
            if worker_crashed:
                self._dirty = True
            if self._dirty:
                raise TerminalTransportError(
                    "terminal server requires authoritative process cleanup"
                )
        return {"status": "clean"}

    def _execute(self, wake_id: str, action: object) -> dict[str, object]:
        with self._state_lock:
            worker = self._worker
            if worker is None or worker.wake_id != wake_id:
                raise StaleTerminalWake(f"terminal wake {wake_id!r} is not active")
        with worker.request_lock:
            with self._state_lock:
                if self._worker is not worker:
                    raise StaleTerminalWake(f"terminal wake {wake_id!r} is retired")
            try:
                worker.connection.sendall(
                    encode_json_frame(
                        {"action": action},
                        max_bytes=_MAX_MESSAGE_BYTES,
                    )
                )
                response = json.loads(_receive_line(worker.connection))
            except (OSError, SocketFrameError, json.JSONDecodeError) as error:
                raise TerminalOutcomeUnknown(
                    f"terminal wake {wake_id!r} action outcome is unknown"
                ) from error
        if not isinstance(response, dict):
            raise TerminalOutcomeUnknown(
                f"terminal wake {wake_id!r} returned invalid data"
            )
        return response

    def _end_wake(self, wake_id: str) -> dict[str, object]:
        if not self._transition_lock.acquire(blocking=False):
            raise ConcurrentTerminalWake("another terminal wake transition is active")
        try:
            with self._state_lock:
                current = self._worker
                if current is None:
                    self._retired_wakes.add(wake_id)
                    reconcile_dirty = self._dirty
                else:
                    reconcile_dirty = False
                if current is not None and current.wake_id != wake_id:
                    if wake_id in self._retired_wakes:
                        return {"wake_id": wake_id, "status": "ended"}
                    raise StaleTerminalWake(
                        f"terminal wake {wake_id!r} is not active"
                    )
                if current is not None:
                    self._worker = None
                    self._retired_wakes.add(wake_id)
            if current is None:
                if reconcile_dirty:
                    self._reconcile_server_adoptees()
                return {"wake_id": wake_id, "status": "ended"}
            self._terminate_worker(current)
            return {"wake_id": wake_id, "status": "ended"}
        finally:
            self._transition_lock.release()

    def _start_worker(self, wake_id: str) -> _TerminalWorker:
        # Never bless a leaked child as part of a later wake. The server's
        # process baseline is fixed before its first terminal worker starts.
        self._reconcile_server_adoptees()
        parent, child = socket.socketpair()
        child.set_inheritable(True)
        volatile_root = Path(tempfile.mkdtemp(prefix="senpai-terminal-wake-"))
        home = volatile_root / "home"
        temporary = volatile_root / "tmp"
        xdg_cache = volatile_root / "cache"
        xdg_config = volatile_root / "config"
        xdg_data = volatile_root / "data"
        for directory in (home, temporary, xdg_cache, xdg_config, xdg_data):
            directory.mkdir()
        command = [
            sys.executable,
            "-m",
            "senpai_agent.isolated_terminal",
            "worker",
            "--control-fd",
            str(child.fileno()),
            "--workspace",
            str(self.working_dir),
        ]
        if self.terminal_type is not None:
            command.extend(("--terminal-type", self.terminal_type))
        environment = {
            **self.environment,
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "XDG_CACHE_HOME": str(xdg_cache),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_DATA_HOME": str(xdg_data),
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parent.parent,
                env=environment,
                pass_fds=(child.fileno(),),
                start_new_session=True,
            )
        except BaseException:
            parent.close()
            child.close()
            shutil.rmtree(volatile_root, ignore_errors=True)
            raise
        child.close()
        worker = _TerminalWorker(
            wake_id,
            process,
            parent,
            volatile_root,
        )
        try:
            parent.settimeout(_WORKER_START_SECONDS)
            response = json.loads(_receive_line(parent))
            if response.get("status") != "ready":
                raise TerminalTransportError("terminal worker did not become ready")
            parent.settimeout(None)
            return worker
        except BaseException:
            self._terminate_worker(worker)
            raise

    def _terminate_worker(self, worker: _TerminalWorker) -> None:
        termination_error: BaseException | None = None
        try:
            worker.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        worker.connection.close()
        try:
            if worker.process.poll() is None:
                worker.process.terminate()
                try:
                    worker.process.wait(timeout=_WORKER_STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    kill_process_tree(worker.process.pid)
                    worker.process.wait(timeout=_WORKER_FORCE_STOP_SECONDS)
            else:
                worker.process.wait()
        except BaseException as error:  # noqa: BLE001
            termination_error = error
        finally:
            try:
                self._reconcile_server_adoptees()
            except BaseException as error:  # noqa: BLE001
                if termination_error is None:
                    termination_error = error
            finally:
                threading.Thread(
                    target=shutil.rmtree,
                    args=(worker.volatile_root,),
                    kwargs={"ignore_errors": True},
                    name="senpai-terminal-volatile-cleanup",
                    daemon=True,
                ).start()
        if termination_error is not None:
            raise TerminalTransportError(
                f"terminal worker cleanup failed ({type(termination_error).__name__})"
            ) from termination_error

    def _clean_server_adoptees(self) -> None:
        if sys.platform != "linux":
            return
        deadline = time.monotonic() + _ADOPTEE_CLEANUP_SECONDS
        while True:
            reap_children()
            with self._state_lock:
                active = self._worker
            protected = (
                {active.process.pid}
                if active is not None
                else set()
            )
            adopted = (
                direct_children(os.getpid())
                - set(self._baseline_children)
                - protected
            )
            if not adopted:
                reap_children()
                return
            for pid in adopted:
                kill_process_tree(pid)
            reap_children()
            if time.monotonic() >= deadline:
                remaining = (
                    direct_children(os.getpid())
                    - set(self._baseline_children)
                    - protected
                )
                if remaining:
                    raise TerminalTransportError(
                        "terminal cleanup left adopted descendant processes"
                    )
                return
            time.sleep(0.01)

    def _reconcile_server_adoptees(self) -> None:
        try:
            self._clean_server_adoptees()
        except BaseException:  # noqa: BLE001
            with self._state_lock:
                self._dirty = True
            raise
        with self._state_lock:
            self._dirty = False

    def _remove_stale_socket(self) -> None:
        if self._abstract_socket:
            return
        try:
            mode = self.socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(
                f"refusing to replace non-socket path: {self.socket_path}"
            )
        self.socket_path.unlink()


def isolated_terminal_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Senpai's isolated terminal.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--socket", default=DEFAULT_TERMINAL_SOCKET)
    serve.add_argument("--workspace", required=True, type=Path)
    health = subparsers.add_parser("health")
    health.add_argument("--socket", default=DEFAULT_TERMINAL_SOCKET)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--control-fd", required=True, type=int)
    worker.add_argument("--workspace", required=True, type=Path)
    worker.add_argument(
        "--terminal-type",
        choices=("tmux", "subprocess", "powershell"),
    )
    args = parser.parse_args(argv)
    if args.operation == "health":
        try:
            check_isolated_terminal_health(args.socket)
        except TerminalTransportError:
            return 1
        return 0
    if args.operation == "worker":
        return _terminal_worker(
            args.control_fd,
            args.workspace,
            args.terminal_type,
        )

    environment = {
        name: value
        for name, value in os.environ.items()
        if name in _TERMINAL_ENV_ALLOWLIST
    }
    server = IsolatedTerminalServer(
        socket_path=args.socket,
        working_dir=args.workspace,
        environment=environment,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(isolated_terminal_main())
