import json
import multiprocessing
import shlex
import socket
import sys
import time
from pathlib import Path

import pytest

from senpai_agent.repair_executor import EXECUTOR_REQUEST_LIMIT_BYTES
from senpai_agent.repair_executor import DEFAULT_EXECUTOR_SOCKET
from senpai_agent.repair_executor import REPAIR_EXECUTOR_PROTOCOL
from senpai_agent.repair_executor import RepairExecutorClient
from senpai_agent.repair_executor import RepairExecutorServer
from senpai_agent.repair_executor import RepairExecutionRequest
from senpai_agent.repair_executor import _encode_frame
from senpai_agent.repair_executor import execute_local_repair


def _serve(socket_path: str) -> None:
    RepairExecutorServer(Path(socket_path)).serve_forever()


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
        "printf temporary > \"$TMPDIR/poison\"; printf '%s' \"$HOME\"",
        tmp_path,
        5,
    )
    second = execute_local_repair(
        "test ! -e \"$HOME/.gitconfig\"; "
        "test ! -e \"$TMPDIR/poison\"; "
        "test -z \"$DATABASE_URL\"; printf '%s' \"$HOME\"",
        tmp_path,
        5,
    )

    assert first["exit_code"] == 0
    assert second["exit_code"] == 0
    assert first["stdout"] != second["stdout"]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_repair_executor_cleans_descendants_when_its_client_disconnects(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-executor.sock"
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
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-nested-executor.sock"
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
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-crashed-executor.sock"
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


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux abstract sockets")
def test_default_executor_socket_cannot_be_poisoned_by_filesystem_state(tmp_path):
    assert DEFAULT_EXECUTOR_SOCKET.startswith("@")
    poisoned_legacy_path = tmp_path / "senpai-repair-executor.sock"
    poisoned_legacy_path.mkdir()
    (poisoned_legacy_path / "hostile").write_text("state")
    server = multiprocessing.get_context("fork").Process(
        target=_serve,
        args=(DEFAULT_EXECUTOR_SOCKET,),
    )
    server.start()
    time.sleep(0.1)
    try:
        result = RepairExecutorClient(DEFAULT_EXECUTOR_SOCKET).execute(
            RepairExecutionRequest("true", tmp_path, 5)
        )
    finally:
        server.terminate()
        server.join(timeout=3)
        if server.is_alive():
            server.kill()
            server.join(timeout=3)

    assert result["exit_code"] == 0
    assert poisoned_legacy_path.is_dir()
