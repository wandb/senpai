"""Stdlib-only authoritative health protocol for the repair broker."""

from __future__ import annotations

import argparse
import json
import socket
import time
from collections.abc import Sequence
from pathlib import Path

from senpai_agent.socket_framing import (
    SocketFrameError,
    receive_frame,
    unix_socket_address,
)


REPAIR_BROKER_HEALTH_PROTOCOL = "senpai-repair-broker-health/v1"
DEFAULT_REPAIR_BROKER_SOCKET = "/run/senpai-repair/repair.sock"
_HEARTBEAT_STALE_SECONDS = 2.0
_OPERATION_DEADLINE_GRACE_SECONDS = 5.0
_HEALTH_FRAME_LIMIT_BYTES = 16 * 1024


class RepairBrokerHealthError(RuntimeError):
    pass


def repair_broker_health_socket(socket_path: str | Path) -> str:
    return f"{socket_path}.health"


def broker_heartbeat_is_healthy(
    status: object,
    *,
    expected_pid: int,
    now: float | None = None,
) -> bool:
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
        protocol != REPAIR_BROKER_HEALTH_PROTOCOL
        or server_pid != expected_pid
        or state not in {"idle", "active"}
        or heartbeat > observed_at + _HEARTBEAT_STALE_SECONDS
    ):
        return False
    if state == "idle":
        return (
            deadline_value is None
            and observed_at - heartbeat <= _HEARTBEAT_STALE_SECONDS
        )
    try:
        deadline = float(deadline_value)
    except (TypeError, ValueError):
        return False
    return observed_at <= deadline + _OPERATION_DEADLINE_GRACE_SECONDS


def check_repair_broker_health(
    socket_path: str | Path = DEFAULT_REPAIR_BROKER_SOCKET,
    *,
    expected_pid: int = 1,
) -> None:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(1)
        connection.connect(
            unix_socket_address(repair_broker_health_socket(socket_path))
        )
        status = json.loads(
            receive_frame(
                connection,
                max_bytes=_HEALTH_FRAME_LIMIT_BYTES,
                timeout_seconds=1,
            )
        )
    except (OSError, SocketFrameError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairBrokerHealthError("repair broker heartbeat is unavailable") from error
    finally:
        connection.close()
    if not broker_heartbeat_is_healthy(status, expected_pid=expected_pid):
        raise RepairBrokerHealthError("repair broker heartbeat is stale or invalid")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check repair broker health.")
    parser.add_argument("--socket", default=DEFAULT_REPAIR_BROKER_SOCKET)
    parser.add_argument("--expected-pid", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        check_repair_broker_health(args.socket, expected_pid=args.expected_pid)
    except RepairBrokerHealthError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
