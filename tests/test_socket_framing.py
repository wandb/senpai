import json
import socket
import threading
import time

import pytest

from senpai_agent.socket_framing import (
    SocketFrameTimeout,
    SocketFrameTooLarge,
    encode_json_frame,
    receive_frame,
    send_json_frame,
)


def test_frame_timeout_is_one_absolute_deadline_not_a_per_recv_idle_timeout():
    sender, receiver = socket.socketpair()

    def drip() -> None:
        try:
            for byte in b'{"slow":true}\n':
                sender.send(bytes([byte]))
                time.sleep(0.04)
        except OSError:
            pass

    thread = threading.Thread(target=drip)
    started = time.monotonic()
    thread.start()
    try:
        with pytest.raises(SocketFrameTimeout):
            receive_frame(receiver, max_bytes=1_024, timeout_seconds=0.12)
    finally:
        receiver.close()
        sender.close()
        thread.join(timeout=1)

    assert time.monotonic() - started < 0.3


def test_json_frames_enforce_encoded_utf8_bytes_not_character_count():
    value = {"text": "\N{COLLISION SYMBOL}" * 8}
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()

    assert encode_json_frame(value, max_bytes=len(encoded)) == encoded + b"\n"
    with pytest.raises(SocketFrameTooLarge):
        encode_json_frame(value, max_bytes=len(encoded) - 1)


def test_oversize_json_frame_sends_no_partial_prefix():
    sender, receiver = socket.socketpair()
    receiver.setblocking(False)
    try:
        with pytest.raises(SocketFrameTooLarge):
            send_json_frame(sender, {"text": "x" * 100}, max_bytes=10)
        with pytest.raises(BlockingIOError):
            receiver.recv(1)
    finally:
        sender.close()
        receiver.close()
