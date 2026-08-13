"""Stdlib-only health client for the isolated terminal sidecar."""

from __future__ import annotations

import argparse
import json
import socket
from collections.abc import Sequence
from pathlib import Path

from senpai_agent.protocols import ISOLATED_TERMINAL_PROTOCOL_VERSION
from senpai_agent.socket_framing import (
    SocketFrameError,
    encode_json_frame,
    receive_frame,
    unix_socket_address,
)


DEFAULT_TERMINAL_SOCKET = "@senpai-isolated-terminal"
_MAX_HEALTH_FRAME_BYTES = 16 * 1024
_HEALTH_TIMEOUT_SECONDS = 1.0


class IsolatedTerminalHealthError(RuntimeError):
    pass


def check_isolated_terminal_health(socket_path: str | Path) -> None:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(_HEALTH_TIMEOUT_SECONDS)
        connection.connect(unix_socket_address(socket_path))
        connection.sendall(
            encode_json_frame(
                {
                    "protocol": ISOLATED_TERMINAL_PROTOCOL_VERSION,
                    "operation": "health",
                },
                max_bytes=_MAX_HEALTH_FRAME_BYTES,
            )
        )
        response = json.loads(
            receive_frame(
                connection,
                max_bytes=_MAX_HEALTH_FRAME_BYTES,
                timeout_seconds=_HEALTH_TIMEOUT_SECONDS,
            )
        )
    except (OSError, SocketFrameError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IsolatedTerminalHealthError(
            "isolated terminal health is unavailable"
        ) from error
    finally:
        connection.close()
    if not isinstance(response, dict) or response.get("status") != "clean":
        raise IsolatedTerminalHealthError("isolated terminal is not clean")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check isolated terminal health.")
    parser.add_argument("--socket", default=DEFAULT_TERMINAL_SOCKET)
    args = parser.parse_args(argv)
    try:
        check_isolated_terminal_health(args.socket)
    except IsolatedTerminalHealthError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
