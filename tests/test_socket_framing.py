import json
import socket

import pytest

from senpai_agent.socket_framing import (
    SocketFrameTooLarge,
    encode_json_frame,
    send_json_frame,
)


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

