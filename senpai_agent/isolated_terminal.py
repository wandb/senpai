"""OpenHands terminal transport for a secret-free sidecar process."""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from openhands.sdk.tool import ToolDefinition, ToolExecutor, register_tool
from openhands.tools.terminal import TerminalAction, TerminalObservation, TerminalTool


_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_CONNECT_ATTEMPTS = 20
_CONNECT_DELAY_SECONDS = 0.1
_ENVIRONMENT_LOCK = threading.Lock()
_REQUEST_READ_TIMEOUT_SECONDS = 5
_UNTIMED_RESPONSE_TIMEOUT_SECONDS = 24 * 60 * 60
_RESPONSE_GRACE_SECONDS = 30


class TerminalTransportError(RuntimeError):
    """The isolated terminal process could not execute an action."""


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
            raise TerminalTransportError("terminal message exceeded the size limit")
    payload = b"".join(chunks)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise TerminalTransportError("terminal message exceeded the size limit")
    return payload


class IsolatedTerminalClientExecutor(
    ToolExecutor[TerminalAction, TerminalObservation]
):
    """Forward TerminalAction values to the sidecar-owned executor."""

    def __init__(self, socket_path: str | Path):
        self.socket_path = Path(socket_path)

    def __call__(
        self,
        action: TerminalAction,
        conversation: object | None = None,
    ) -> TerminalObservation:
        del conversation
        payload = action.model_dump_json().encode() + b"\n"
        last_error: OSError | None = None
        for attempt in range(_CONNECT_ATTEMPTS):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
                    connection.connect(str(self.socket_path))
                    connection.sendall(payload)
                    connection.settimeout(
                        (
                            action.timeout
                            if action.timeout is not None
                            else _UNTIMED_RESPONSE_TIMEOUT_SECONDS
                        )
                        + _RESPONSE_GRACE_SECONDS
                    )
                    response = _receive_line(connection)
                break
            except OSError as error:
                last_error = error
                if attempt + 1 == _CONNECT_ATTEMPTS:
                    raise TerminalTransportError(
                        "isolated terminal sidecar is unavailable"
                    ) from error
                time.sleep(_CONNECT_DELAY_SECONDS)
        else:  # pragma: no cover - loop always raises on exhaustion
            raise TerminalTransportError(
                "isolated terminal sidecar is unavailable"
            ) from last_error

        try:
            envelope = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise TerminalTransportError(
                "isolated terminal returned an invalid response"
            ) from error
        if not isinstance(envelope, dict):
            raise TerminalTransportError(
                "isolated terminal returned an invalid response"
            )
        if "error" in envelope:
            error_type = str(envelope.get("error_type", "TerminalError"))
            raise TerminalTransportError(f"{error_type}: {envelope['error']}")
        try:
            return TerminalObservation.model_validate(envelope["observation"])
        except (KeyError, ValueError, TypeError) as error:
            raise TerminalTransportError(
                "isolated terminal returned an invalid observation"
            ) from error


class IsolatedTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    """Resolve to OpenHands' native terminal schema with a remote executor."""

    name = "terminal"

    @classmethod
    def create(
        cls,
        conv_state: object,
        *,
        socket_path: str,
    ) -> Sequence[ToolDefinition]:
        return TerminalTool.create(
            conv_state,
            executor=IsolatedTerminalClientExecutor(socket_path),
        )


def register_isolated_terminal_tool() -> None:
    """Replace `terminal` only in the supervisor's dedicated process."""

    register_tool("terminal", IsolatedTerminalTool)


class IsolatedTerminalServer:
    """Own one persistent native TerminalExecutor behind a private Unix socket."""

    def __init__(
        self,
        *,
        socket_path: str | Path,
        working_dir: str | Path,
        environment: Mapping[str, str] | None = None,
        terminal_type: Literal["tmux", "subprocess", "powershell"] | None = None,
    ):
        from openhands.tools.terminal.impl import TerminalExecutor

        self.socket_path = Path(socket_path)
        self.working_dir = Path(working_dir)
        self.environment = dict(environment) if environment is not None else None
        if not self.working_dir.is_dir():
            raise ValueError(f"terminal workspace does not exist: {self.working_dir}")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_socket()
        # OpenHands intentionally merges terminal env overrides with its
        # parent. The sidecar is a process boundary in production; make that
        # replacement semantic explicit as defense in depth and for in-process
        # regression tests. Reset actions receive the same treatment below.
        with _process_environment(self.environment):
            self._executor = TerminalExecutor(
                working_dir=str(self.working_dir),
                terminal_type=terminal_type,
                env=self.environment,
            )
        self._executor_lock = threading.Lock()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._socket.listen(8)
        self._socket.settimeout(0.2)
        self._stop = threading.Event()
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
        self._stop.set()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(self.socket_path))
        except OSError:
            pass
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
        self._socket.close()
        self._executor.close()
        self._remove_stale_socket()

    def serve_forever(self) -> None:
        self._thread.start()
        try:
            while self._thread.is_alive():
                self._thread.join(timeout=1)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

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
                try:
                    action = TerminalAction.model_validate_json(
                        _receive_line(connection)
                    )
                    with self._executor_lock:
                        with _process_environment(self.environment):
                            observation = self._executor(action)
                    response = {
                        "observation": observation.model_dump(mode="json")
                    }
                except Exception as error:  # noqa: BLE001
                    response = {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                try:
                    connection.sendall(json.dumps(response).encode() + b"\n")
                except OSError:
                    continue

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


def isolated_terminal_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Senpai's isolated terminal.")
    parser.add_argument("serve", nargs="?")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args(argv)
    environment = dict(os.environ)
    environment.pop("SENPAI_SUPERVISOR_TERMINAL_SOCKET", None)
    server = IsolatedTerminalServer(
        socket_path=args.socket,
        working_dir=args.workspace,
        environment=environment,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(isolated_terminal_main())
