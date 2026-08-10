import os
import signal
import shlex
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

from openhands.tools.terminal import TerminalAction

from senpai_agent.isolated_terminal import (
    IsolatedTerminalClientExecutor,
    IsolatedTerminalServer,
    StaleTerminalWake,
    TerminalOutcomeUnknown,
    TerminalTransportError,
    begin_isolated_terminal_wake,
    check_isolated_terminal_health,
    end_isolated_terminal_wake,
)


def test_terminal_socket_preserves_shell_fidelity_without_control_secrets(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal.sock"
    canaries = {
        "GITHUB_TOKEN": "control-github-canary",
        "WANDB_API_KEY": "control-wandb-canary",
        "OPENAI_API_KEY": "control-model-canary",
    }
    for name, value in canaries.items():
        monkeypatch.setenv(name, value)
    shell_environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": os.environ["PATH"],
        "SHELL_MARKER": "sidecar",
    }
    (tmp_path / "home").mkdir()

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment=shell_environment,
        terminal_type="subprocess",
    ):
        begin_isolated_terminal_wake(socket_path, "wake-fidelity")
        client = IsolatedTerminalClientExecutor(socket_path, "wake-fidelity")
        action = TerminalAction(
            command=(
                "printf '%s\\n' \"$SHELL_MARKER\"; "
                f"{shlex.quote(sys.executable)} -c "
                + shlex.quote(
                    "import os; print('|'.join(sorted(os.environ)))"
                )
            )
        )
        observation = client(action)

    assert observation.metadata.exit_code == 0
    assert "sidecar" in observation.text
    assert all(name not in observation.text for name in canaries)
    assert all(value not in observation.text for value in canaries.values())


def test_terminal_socket_preserves_timeout_poll_and_reset_semantics(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-reset.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment=environment,
        terminal_type="subprocess",
    ):
        begin_isolated_terminal_wake(socket_path, "wake-reset")
        client = IsolatedTerminalClientExecutor(socket_path, "wake-reset")
        timed_out = client(
            TerminalAction(command="sleep 1; printf done", timeout=0.1)
        )
        continued = client(TerminalAction(command="", timeout=2))
        reset = client(TerminalAction(command="", reset=True))

    assert timed_out.exit_code == -1
    assert continued.exit_code == 0
    assert "done" in continued.text
    assert reset.exit_code == 0


def test_terminal_does_not_replay_an_action_after_the_request_was_sent(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-drop.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(20)
    listener.settimeout(0.5)
    executions = []

    def drop_replies() -> None:
        try:
            while True:
                connection, _ = listener.accept()
                with connection:
                    payload = b""
                    while not payload.endswith(b"\n"):
                        chunk = connection.recv(65_536)
                        if not chunk:
                            break
                        payload += chunk
                    executions.append(payload)
                    connection.setsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_LINGER,
                        struct.pack("ii", 1, 0),
                    )
        except TimeoutError:
            return

    server = threading.Thread(target=drop_replies)
    server.start()
    try:
        with pytest.raises(TerminalOutcomeUnknown, match="may have executed"):
            IsolatedTerminalClientExecutor(socket_path, "wake-unknown")(
                TerminalAction(command="touch important-state")
            )
    finally:
        server.join(timeout=3)
        listener.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass

    time.sleep(0.1)
    assert len(executions) == 1


def test_begin_wake_recreates_pristine_shell_and_rejects_stale_actions(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-wakes.sock"
    workspace = tmp_path / "workspace"
    nested = workspace / "nested"
    nested.mkdir(parents=True)
    environment = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment=environment,
        terminal_type="subprocess",
    ):
        begin_isolated_terminal_wake(socket_path, "wake-one")
        first = IsolatedTerminalClientExecutor(socket_path, "wake-one")
        changed = first(
            TerminalAction(
                command=(
                    "cd nested; export LEAKED_WAKE=yes; "
                    "printf persisted > ../workspace-state; "
                    "printf '[user]\\nname = poisoned\\n' > \"$HOME/.gitconfig\"; pwd"
                )
            )
        )
        assert str(nested) in changed.text

        begin_isolated_terminal_wake(socket_path, "wake-two")
        second = IsolatedTerminalClientExecutor(socket_path, "wake-two")
        pristine = second(
            TerminalAction(
                command=(
                    "pwd; printf '|%s|' \"$LEAKED_WAKE\"; "
                    "test -f workspace-state && printf '|workspace-persisted|'; "
                    "test ! -e \"$HOME/.gitconfig\" && printf '|home-clean|'"
                )
            )
        )

        assert str(workspace) in pristine.text
        assert "|yes|" not in pristine.text
        assert "|workspace-persisted|" in pristine.text
        assert "|home-clean|" in pristine.text
        with pytest.raises(StaleTerminalWake):
            first(TerminalAction(command="touch stale-wake-ran"))

    assert not (workspace / "stale-wake-ran").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_end_wake_immediately_reaps_background_processes(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-end.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "ended-wake-survived"

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment={"PATH": os.environ["PATH"]},
        terminal_type="subprocess",
    ):
        begin_isolated_terminal_wake(socket_path, "wake-end")
        IsolatedTerminalClientExecutor(socket_path, "wake-end")(
            TerminalAction(
                command=(
                    f"(sleep .5; printf survived > {shlex.quote(str(marker))}) &"
                )
            )
        )
        end_isolated_terminal_wake(socket_path, "wake-end")
        end_isolated_terminal_wake(socket_path, "wake-end")
        time.sleep(0.7)

    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process trees")
def test_begin_wake_preempts_a_long_foreground_action_within_a_bound(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-preempt.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    old_outcome = []

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        terminal_type="subprocess",
    ):
        begin_isolated_terminal_wake(socket_path, "wake-long")
        old_client = IsolatedTerminalClientExecutor(socket_path, "wake-long")

        def run_old_action() -> None:
            try:
                old_client(TerminalAction(command="sleep 30"))
            except TerminalTransportError as error:
                old_outcome.append(error)

        old_turn = threading.Thread(target=run_old_action)
        old_turn.start()
        time.sleep(0.2)
        started = time.monotonic()
        next_wake = threading.Thread(
            target=begin_isolated_terminal_wake,
            args=(socket_path, "wake-next"),
        )
        next_wake.start()
        old_turn.join(timeout=3)
        preempt_elapsed = time.monotonic() - started

        assert not old_turn.is_alive()
        assert preempt_elapsed < 3
        assert old_outcome
        next_wake.join(timeout=10)
        assert not next_wake.is_alive()
        next_observation = IsolatedTerminalClientExecutor(
            socket_path,
            "wake-next",
        )(TerminalAction(command="printf ready"))
        assert "ready" in next_observation.text


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_begin_wake_reaps_background_setsid_double_fork(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-tree.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "old-wake-survived"
    daemonizer = (
        "import os,time,pathlib; pid=os.fork(); pid and os._exit(0); "
        "os.setsid(); "
        "pid=os.fork(); pid and os._exit(0); "
        f"time.sleep(.5); pathlib.Path({str(marker)!r}).write_text('survived')"
    )

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        terminal_type="subprocess",
    ):
        begin_isolated_terminal_wake(socket_path, "wake-one")
        first = IsolatedTerminalClientExecutor(socket_path, "wake-one")
        first(
            TerminalAction(
                command=(
                    f"{shlex.quote(sys.executable)} -c "
                    f"{shlex.quote(daemonizer)} >/dev/null 2>&1 &"
                )
            )
        )
        begin_isolated_terminal_wake(socket_path, "wake-two")
        time.sleep(0.7)

    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_next_wake_reaps_detached_child_after_worker_crash(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-crash.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child_ready = workspace / "detached-child.pid"
    marker = workspace / "crashed-worker-child-survived"
    daemonizer = (
        "import os,time,pathlib; pid=os.fork(); pid and os._exit(0); "
        "os.setsid(); "
        "pid=os.fork(); pid and os._exit(0); "
        f"pathlib.Path({str(child_ready)!r}).write_text(str(os.getpid())); "
        f"time.sleep(.7); pathlib.Path({str(marker)!r}).write_text('survived')"
    )

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment={"PATH": os.environ["PATH"]},
        terminal_type="subprocess",
    ) as server:
        begin_isolated_terminal_wake(socket_path, "wake-crash")
        IsolatedTerminalClientExecutor(socket_path, "wake-crash")(
            TerminalAction(
                command=(
                    f"{shlex.quote(sys.executable)} -c "
                    f"{shlex.quote(daemonizer)} >/dev/null 2>&1 &"
                )
            )
        )
        deadline = time.monotonic() + 2
        while not child_ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_ready.exists()
        with server._state_lock:
            assert server._worker is not None
            os.kill(server._worker.process.pid, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while server._worker.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        with pytest.raises(TerminalTransportError, match="requires authoritative"):
            check_isolated_terminal_health(socket_path)
        begin_isolated_terminal_wake(socket_path, "wake-after-crash")
        time.sleep(0.9)

    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_later_begin_reconciles_an_orphan_after_one_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-reconcile.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child_ready = workspace / "orphan-ready.pid"
    marker = workspace / "orphan-was-blessed"
    daemonizer = (
        "import os,time,pathlib; pid=os.fork(); pid and os._exit(0); "
        "os.setsid(); "
        "pid=os.fork(); pid and os._exit(0); "
        f"pathlib.Path({str(child_ready)!r}).write_text(str(os.getpid())); "
        f"time.sleep(.8); pathlib.Path({str(marker)!r}).write_text('survived')"
    )

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment={"PATH": os.environ["PATH"]},
        terminal_type="subprocess",
    ) as server:
        begin_isolated_terminal_wake(socket_path, "wake-poisoned")
        IsolatedTerminalClientExecutor(socket_path, "wake-poisoned")(
            TerminalAction(
                command=(
                    f"{shlex.quote(sys.executable)} -c "
                    f"{shlex.quote(daemonizer)} >/dev/null 2>&1 &"
                )
            )
        )
        deadline = time.monotonic() + 2
        while not child_ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_ready.exists()
        with server._state_lock:
            assert server._worker is not None
            os.kill(server._worker.process.pid, signal.SIGKILL)

        clean = server._clean_server_adoptees
        monkeypatch.setattr(
            server,
            "_clean_server_adoptees",
            lambda: (_ for _ in ()).throw(RuntimeError("injected cleanup failure")),
        )
        with pytest.raises(TerminalTransportError, match="cleanup failed"):
            begin_isolated_terminal_wake(socket_path, "wake-cleanup-failed")
        with pytest.raises(TerminalTransportError, match="requires authoritative"):
            check_isolated_terminal_health(socket_path)
        monkeypatch.setattr(server, "_clean_server_adoptees", clean)

        end_isolated_terminal_wake(socket_path, "wake-poisoned")
        check_isolated_terminal_health(socket_path)

        begin_isolated_terminal_wake(socket_path, "wake-reconciled")
        check_isolated_terminal_health(socket_path)
        time.sleep(1)

    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreapers")
def test_terminal_server_close_reaps_active_wake_background_processes(tmp_path):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-terminal-close.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "closed-server-survived"

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        terminal_type="subprocess",
    ):
        begin_isolated_terminal_wake(socket_path, "wake-close")
        IsolatedTerminalClientExecutor(socket_path, "wake-close")(
            TerminalAction(
                command=(
                    f"(sleep .5; printf survived > {shlex.quote(str(marker))}) &"
                )
            )
        )

    time.sleep(0.7)
    assert not marker.exists()
