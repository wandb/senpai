from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.tools.terminal import TerminalAction, TerminalObservation

from senpai_agent.tools import SenpaiTerminalExecutor, SenpaiTerminalTool


class FakeTerminal:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.interrupted = False

    def __call__(self, action, conversation=None):
        self.calls.append((action, conversation))
        return TerminalObservation.from_text(
            "allowed",
            command=action.command,
            exit_code=0,
        )

    def close(self) -> None:
        self.closed = True

    def interrupt(self) -> None:
        self.interrupted = True


def test_terminal_executor_delegates_only_after_policy_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from senpai_agent import hooks

    decisions = []

    def allow(command: str, role: str, workspace: Path):
        decisions.append((command, role, workspace))
        return SimpleNamespace(allowed=True, reason="")

    monkeypatch.setattr(hooks, "terminal_policy", allow)
    delegate = FakeTerminal()
    executor = SenpaiTerminalExecutor(
        delegate,
        role="student",
        workspace=tmp_path,
    )
    action = TerminalAction(command="git status --short")
    conversation = SimpleNamespace()

    observation = executor(action, conversation)
    executor.interrupt()
    executor.close()

    assert observation.text == "allowed"
    assert decisions == [("git status --short", "student", tmp_path)]
    assert delegate.calls == [(action, conversation)]
    assert delegate.interrupted is True
    assert delegate.closed is True


def test_terminal_executor_returns_long_foreground_calls_for_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from senpai_agent import hooks

    monkeypatch.setattr(
        hooks,
        "terminal_policy",
        lambda *_args: SimpleNamespace(allowed=True, reason=""),
    )
    delegate = FakeTerminal()
    executor = SenpaiTerminalExecutor(
        delegate,
        role="student",
        workspace=tmp_path,
        foreground_timeout_seconds=600,
    )
    action = TerminalAction(command="swift test", timeout=1800)

    executor(action)

    delegated = delegate.calls[0][0]
    assert action.timeout == 1800
    assert delegated.timeout == 600


@pytest.mark.parametrize("policy_error", [False, True])
def test_terminal_executor_fails_closed_without_invoking_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy_error: bool,
):
    from senpai_agent import hooks

    def policy(_command: str, _role: str, _workspace: Path):
        if policy_error:
            raise RuntimeError("parser unavailable")
        return SimpleNamespace(allowed=False, reason="Use the typed GitHub tool.")

    monkeypatch.setattr(hooks, "terminal_policy", policy)
    delegate = FakeTerminal()
    executor = SenpaiTerminalExecutor(
        delegate,
        role="student",
        workspace=tmp_path,
    )
    action = TerminalAction(command="git push origin experiment")

    observation = executor(action)

    assert observation.is_error is True
    assert observation.command == action.command
    assert observation.exit_code is None
    assert "denied" in observation.text.lower()
    assert delegate.calls == []


def test_terminal_tool_bounds_silent_commands(monkeypatch, tmp_path):
    captured = {}
    native = SimpleNamespace(
        executor=FakeTerminal(),
        set_executor=lambda executor: (captured.setdefault("executor", executor),)[0],
    )

    def create(_conv_state, **kwargs):
        captured.update(kwargs)
        return [native]

    monkeypatch.setattr("senpai_agent.tools.TerminalTool.create", create)
    monkeypatch.setenv("SENPAI_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("SENPAI_TARGET_PYTHON_ENV", "/home/senpai/.venvs/target")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    conv_state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(tmp_path))
    )

    SenpaiTerminalTool.create(conv_state, role="student")

    assert captured["no_change_timeout_seconds"] == 600
    assert captured["env"] == {
        "PATH": "/home/senpai/.venvs/target/bin:/usr/bin:/bin",
        "UV_PROJECT_ENVIRONMENT": "/home/senpai/.venvs/target",
        "UV_PYTHON": "/home/senpai/.venvs/target/bin/python",
        "VIRTUAL_ENV": "/home/senpai/.venvs/target",
    }
    assert isinstance(captured["executor"], SenpaiTerminalExecutor)
    assert captured["executor"].foreground_timeout_seconds == 600
