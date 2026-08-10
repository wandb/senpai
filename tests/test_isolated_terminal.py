import os
import shlex
import sys
from pathlib import Path

from openhands.tools.terminal import TerminalAction

from senpai_agent.isolated_terminal import (
    IsolatedTerminalClientExecutor,
    IsolatedTerminalServer,
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
        client = IsolatedTerminalClientExecutor(socket_path)
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
        client = IsolatedTerminalClientExecutor(socket_path)
        timed_out = client(
            TerminalAction(command="sleep 1; printf done", timeout=0.1)
        )
        continued = client(TerminalAction(command="", timeout=2))
        reset = client(TerminalAction(command="", reset=True))

    assert timed_out.exit_code == -1
    assert continued.exit_code == 0
    assert "done" in continued.text
    assert reset.exit_code == 0
