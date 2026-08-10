import json
import hashlib
import multiprocessing
import os
import shlex
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

import senpai_agent.repair_executor as repair_executor

from senpai_agent.protocols import REPAIR_PROTOCOL_VERSION
from senpai_agent.repair_executor import EXECUTOR_REQUEST_LIMIT_BYTES
from senpai_agent.repair_executor import DEFAULT_EXECUTOR_SOCKET
from senpai_agent.repair_executor import REPAIR_EXECUTOR_PROTOCOL
from senpai_agent.repair_executor import RepairExecutorClient
from senpai_agent.repair_executor import RepairExecutorServer
from senpai_agent.repair_executor import RepairExecutionRequest
from senpai_agent.repair_executor import _heartbeat_status_is_healthy
from senpai_agent.repair_executor import _encode_frame
from senpai_agent.repair_executor import check_repair_executor_health
from senpai_agent.repair_executor import execute_local_repair


def _serve(socket_path: str) -> None:
    RepairExecutorServer(Path(socket_path)).serve_forever()


def _test_socket(tmp_path: Path, suffix: str) -> Path:
    digest = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:10]
    return Path("/private/tmp") / f"senpai-{os.getpid()}-{digest}-{suffix}.sock"


def test_repair_startup_scavenges_only_owned_stale_volatile_children(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(repair_executor.tempfile, "tempdir", str(tmp_path))
    parent = tmp_path / "senpai-repair-operations"
    parent.mkdir()
    stale = parent / "operation-stale123"
    stale.mkdir()
    (stale / ".senpai-owner.json").write_text(
        json.dumps({"pid": 999_999_999, "start_token": None})
    )
    target = tmp_path / "must-survive"
    target.mkdir()
    link = parent / "operation-symlink1"
    link.symlink_to(target)
    unrelated = parent / "operator-notes"
    unrelated.write_text("keep")

    repair_executor._scavenge_stale_volatile_roots()

    assert not stale.exists()
    assert not link.exists()
    assert target.exists()
    assert unrelated.read_text() == "keep"


def test_repair_output_bounds_invalid_utf8_without_quadratic_trimming():
    exact = repair_executor._BoundedByteTail(limit=4)
    exact.append(b"abcd")

    assert exact.text() == "abcd"
    assert exact.truncated is False

    tail = repair_executor._BoundedByteTail(limit=16)

    tail.append(b"\xff" * 8)
    value = tail.text()

    assert len(value.encode("utf-8")) <= 16
    assert tail.truncated is True


def test_repair_executor_health_rejects_a_wedged_heartbeat():
    now = time.monotonic()

    assert not _heartbeat_status_is_healthy(
        {
            "protocol": REPAIR_EXECUTOR_PROTOCOL,
            "server_pid": os.getpid(),
            "state": "idle",
            "heartbeat_monotonic": now - 10,
            "operation_deadline_monotonic": None,
        },
        expected_pid=os.getpid(),
        now=now,
    )


def test_repair_executor_health_accepts_an_in_deadline_long_command():
    now = time.monotonic()

    assert _heartbeat_status_is_healthy(
        {
            "protocol": REPAIR_EXECUTOR_PROTOCOL,
            "server_pid": os.getpid(),
            "state": "active",
            "heartbeat_monotonic": now,
            "operation_deadline_monotonic": now + 3_000,
        },
        expected_pid=os.getpid(),
        now=now,
    )


def test_repair_executor_health_rejects_an_expired_command():
    now = time.monotonic()

    assert not _heartbeat_status_is_healthy(
        {
            "protocol": REPAIR_EXECUTOR_PROTOCOL,
            "server_pid": os.getpid(),
            "state": "active",
            "heartbeat_monotonic": now,
            "operation_deadline_monotonic": now - 10,
        },
        expected_pid=os.getpid(),
        now=now,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process health")
def test_repair_executor_remains_healthy_during_a_long_command(tmp_path):
    socket_path = _test_socket(tmp_path, "active-executor")
    server = multiprocessing.get_context("fork").Process(
        target=_serve,
        args=(str(socket_path),),
    )
    server.start()
    deadline = time.monotonic() + 3
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    result: dict[str, object] = {}

    def execute() -> None:
        result.update(
            RepairExecutorClient(socket_path).execute(
                RepairExecutionRequest("sleep 2", tmp_path, 10)
            )
        )

    worker = threading.Thread(target=execute)
    worker.start()
    time.sleep(0.5)
    try:
        check_repair_executor_health(socket_path, expected_pid=server.pid)
    finally:
        worker.join(timeout=5)
        server.terminate()
        server.join(timeout=3)
        if server.is_alive():
            server.kill()
            server.join(timeout=3)

    assert result["exit_code"] == 0


def test_maximum_unicode_command_fits_the_executor_request_frame(tmp_path):
    request = RepairExecutionRequest(
        command="\N{COLLISION SYMBOL}" * 65_536,
        cwd=tmp_path,
        timeout_seconds=60,
    )

    frame = _encode_frame(request.payload(), EXECUTOR_REQUEST_LIMIT_BYTES)

    assert len(frame) <= EXECUTOR_REQUEST_LIMIT_BYTES + 1
    assert RepairExecutionRequest.parse(frame[:-1]).command == request.command


def test_repair_operations_get_fresh_home_and_temporary_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://credential@database/research")
    first = execute_local_repair(
        "printf poisoned > \"$HOME/.gitconfig\"; "
        "mkdir -p \"$HOME/.local/bin\"; "
        "printf '#!/bin/sh\\nexit 99\\n' > \"$HOME/.local/bin/git\"; "
        "chmod +x \"$HOME/.local/bin/git\"; "
        "printf temporary > \"$TMPDIR/poison\"; printf '%s' \"$HOME\"",
        tmp_path,
        5,
    )
    second = execute_local_repair(
        "test ! -e \"$HOME/.gitconfig\"; "
        "test ! -e \"$TMPDIR/poison\"; "
        "test -z \"$DATABASE_URL\"; "
        "test \"$PATH\" = '/opt/senpai-venv/bin:/usr/local/bin:/usr/bin:/bin'; "
        "test \"$(command -v git)\" != \"$HOME/.local/bin/git\"; "
        "printf '%s' \"$HOME\"",
        tmp_path,
        5,
    )

    assert first["exit_code"] == 0
    assert second["exit_code"] == 0
    assert first["stdout"] != second["stdout"]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_repair_executor_cleans_descendants_when_its_client_disconnects(tmp_path):
    socket_path = _test_socket(tmp_path, "executor")
    marker = tmp_path / "disconnected-command-survived"
    server = multiprocessing.get_context("fork").Process(
        target=_serve,
        args=(str(socket_path),),
    )
    server.start()
    deadline = time.monotonic() + 3
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()

    request = {
        "protocol": REPAIR_EXECUTOR_PROTOCOL,
        "command": f"sleep .4; printf survived > {marker}; sleep 5",
        "cwd": str(tmp_path),
        "timeout_seconds": 10,
    }
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(socket_path))
    connection.sendall(json.dumps(request).encode() + b"\n")
    connection.shutdown(socket.SHUT_RDWR)
    connection.close()
    time.sleep(0.7)

    server.terminate()
    server.join(timeout=3)
    if server.is_alive():
        server.kill()
        server.join(timeout=3)

    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
@pytest.mark.parametrize("timeout_seconds", [0.1, 5])
def test_repair_reaps_setsid_double_fork_descendants_on_timeout_and_success(
    tmp_path,
    timeout_seconds,
):
    pid_file = tmp_path / f"escaped-{timeout_seconds}.pid"
    daemonizer = (
        "import os,time,pathlib; "
        "pid=os.fork(); "
        "pid and os._exit(0); "
        "os.setsid(); "
        "pid=os.fork(); "
        "pid and os._exit(0); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(5)"
    )
    suffix = " & sleep 5" if timeout_seconds < 1 else " & sleep .2"

    result = execute_local_repair(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(daemonizer)}{suffix}",
        tmp_path,
        timeout_seconds,
    )
    time.sleep(0.1)

    assert result["exit_code"] in {0, 124}, result
    assert pid_file.exists()
    assert not Path(f"/proc/{pid_file.read_text()}").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_repair_command_cannot_enqueue_a_nested_executor_request(tmp_path):
    socket_path = _test_socket(tmp_path, "nested-executor")
    marker = tmp_path / "nested-connected"
    server = multiprocessing.get_context("fork").Process(
        target=_serve,
        args=(str(socket_path),),
    )
    server.start()
    deadline = time.monotonic() + 3
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()

    nested_probe = (
        "import pathlib,socket; "
        "client=socket.socket(socket.AF_UNIX); "
        f"client.connect({str(socket_path)!r}); "
        f"pathlib.Path({str(marker)!r}).write_text('connected')"
    )
    request = {
        "protocol": REPAIR_EXECUTOR_PROTOCOL,
        "command": (
            f"{shlex.quote(sys.executable)} -c {shlex.quote(nested_probe)}"
        ),
        "cwd": str(tmp_path),
        "timeout_seconds": 5,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        connection.sendall(json.dumps(request).encode() + b"\n")
        response = json.loads(connection.makefile("rb").readline())

    server.terminate()
    server.join(timeout=3)
    if server.is_alive():
        server.kill()
        server.join(timeout=3)

    assert response["exit_code"] != 0
    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_executor_reaps_command_tree_after_request_worker_is_killed(tmp_path):
    socket_path = _test_socket(tmp_path, "crashed-executor")
    marker = tmp_path / "crashed-worker-command-survived"
    server = multiprocessing.get_context("fork").Process(
        target=_serve,
        args=(str(socket_path),),
    )
    server.start()
    deadline = time.monotonic() + 3
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()

    request = {
        "protocol": REPAIR_EXECUTOR_PROTOCOL,
        "command": (
            f"(sleep .8; printf survived > {shlex.quote(str(marker))}) & "
            "kill -9 $PPID; sleep 5"
        ),
        "cwd": str(tmp_path),
        "timeout_seconds": 10,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        connection.sendall(json.dumps(request).encode() + b"\n")
        try:
            connection.recv(1)
        except ConnectionError:
            pass

    deadline = time.monotonic() + 3
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()
    healthy = RepairExecutorClient(socket_path).execute(
        RepairExecutionRequest("true", tmp_path, 5)
    )
    time.sleep(0.7)

    server.terminate()
    server.join(timeout=3)
    if server.is_alive():
        server.kill()
        server.join(timeout=3)

    assert healthy["exit_code"] == 0
    assert not marker.exists()


def test_default_executor_socket_is_scoped_to_the_repair_only_mount():
    assert DEFAULT_EXECUTOR_SOCKET == (
        "/run/senpai-repair-executor/executor.sock"
    )
    assert repair_executor._health_socket_path(DEFAULT_EXECUTOR_SOCKET) == (
        "@senpai-repair-executor-health-v2"
    )
    assert REPAIR_EXECUTOR_PROTOCOL == REPAIR_PROTOCOL_VERSION


def test_explicit_executor_socket_keeps_a_sibling_health_socket(tmp_path):
    socket_path = _test_socket(tmp_path, "custom")

    assert repair_executor._health_socket_path(socket_path) == f"{socket_path}.health"


def test_filesystem_executor_socket_refuses_a_poisoned_non_socket_path(tmp_path):
    socket_path = _test_socket(tmp_path, "poisoned")
    socket_path.mkdir()
    try:
        with pytest.raises(repair_executor.RepairExecutorError, match="non-socket"):
            RepairExecutorServer(socket_path).serve_forever()
        assert socket_path.is_dir()
    finally:
        socket_path.rmdir()


def test_managed_executor_socket_cleanup_removes_poison_without_following_links(
    tmp_path,
):
    target = tmp_path / "must-survive"
    target.mkdir()
    poisoned = tmp_path / "executor.sock"
    poisoned.symlink_to(target)

    repair_executor._remove_socket_path(poisoned, replace_poisoned=True)

    assert not poisoned.exists()
    assert target.is_dir()
    poisoned.mkdir()
    (poisoned / "junk").write_text("blocked")

    repair_executor._remove_socket_path(poisoned, replace_poisoned=True)

    assert not poisoned.exists()
    assert target.is_dir()
