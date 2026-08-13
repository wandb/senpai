"""Byte-bounded newline-delimited JSON for local control sockets."""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any


class SocketFrameError(RuntimeError):
    """A local socket message was incomplete, invalid, or too large."""


class SocketFrameTooLarge(SocketFrameError):
    """A complete encoded frame exceeded its protocol byte limit."""


class SocketFrameTimeout(SocketFrameError):
    """A peer failed to complete one frame within its absolute deadline."""


def unix_socket_address(socket_path: str | Path) -> str:
    """Map an `@name` to Linux's filesystem-independent abstract namespace."""

    value = str(socket_path)
    if not value.startswith("@"):
        return value
    if sys.platform != "linux":
        raise SocketFrameError("abstract Unix sockets require Linux")
    return f"\0{value[1:]}"


def receive_frame(
    connection: socket.socket,
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> bytes:
    """Read one newline-delimited frame without buffering beyond its limit."""

    if timeout_seconds <= 0:
        raise ValueError("socket frame timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    previous_timeout = connection.gettimeout()
    payload = bytearray()
    try:
        while True:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise SocketFrameTimeout("socket frame deadline expired")
            connection.settimeout(remaining_time)
            remaining = max_bytes - len(payload)
            try:
                chunk = connection.recv(min(65_536, remaining + 1))
            except TimeoutError as error:
                raise SocketFrameTimeout("socket frame deadline expired") from error
            if not chunk:
                return bytes(payload)
            newline = chunk.find(b"\n")
            if newline >= 0:
                payload.extend(chunk[:newline])
                if len(payload) > max_bytes:
                    raise SocketFrameTooLarge(
                        f"socket frame exceeded {max_bytes} encoded bytes"
                    )
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise SocketFrameTooLarge(
                    f"socket frame exceeded {max_bytes} encoded bytes"
                )
    finally:
        try:
            connection.settimeout(previous_timeout)
        except OSError:
            pass


def encode_json_frame(value: Any, *, max_bytes: int) -> bytes:
    """Encode one valid UTF-8 JSON frame and enforce its wire-size limit."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError) as error:
        raise SocketFrameError("socket response was not valid UTF-8 JSON") from error
    if len(payload) > max_bytes:
        raise SocketFrameTooLarge(
            f"socket frame exceeded {max_bytes} encoded bytes"
        )
    return payload + b"\n"


def send_json_frame(
    connection: socket.socket,
    value: Any,
    *,
    max_bytes: int,
) -> None:
    """Encode completely before sending so oversize messages send no prefix."""

    connection.sendall(encode_json_frame(value, max_bytes=max_bytes))
