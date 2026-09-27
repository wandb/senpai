import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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

    def create(_conv_state, **_kwargs):
        return [native]

    def create_executor(**kwargs):
        captured.update(kwargs)
        return native.executor

    monkeypatch.setattr("senpai_agent.tools.TerminalTool.create", create)
    monkeypatch.setattr("senpai_agent.tools.TargetTerminalExecutor", create_executor)
    monkeypatch.setenv("SENPAI_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS", "600")
    conv_state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(tmp_path)),
        env_observation_persistence_dir=None,
    )

    SenpaiTerminalTool.create(conv_state, role="student")

    assert captured["no_change_timeout_seconds"] == 600
    assert isinstance(captured["executor"], SenpaiTerminalExecutor)
    assert captured["executor"].foreground_timeout_seconds == 600


@pytest.fixture
def native_target_terminal(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text(
        'export PATH="/usr/bin:/bin:$HOME/custom-bin"\n'
        'export SHELL_SENTINEL="from startup"\n'
    )
    target = tmp_path / "target env's venv"
    subprocess.run(
        [sys.executable, "-P", "-m", "venv", "--without-pip", str(target)],
        check=True,
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SENPAI_TARGET_PYTHON_ENV", str(target))
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    (tmp_path / "project_module.py").write_text("VALUE = 'project import'\n")
    (tmp_path / "check_target.py").write_text(
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "from project_module import VALUE\n"
        "Path(sys.argv[1]).write_text(json.dumps({"
        "'prefix': sys.prefix, 'value': VALUE, "
        "'uv': os.environ['UV_PROJECT_ENVIRONMENT'], "
        "'uv_python': os.environ['UV_PYTHON'], 'path': os.environ['PATH'].split(':'), "
        "'shell': os.environ['SHELL_SENTINEL'], 'secret': os.environ.get('PRIVATE_AUTH')}))\n"
        "if len(sys.argv) > 2:\n"
        "    deadline = time.monotonic() + 5\n"
        "    while len(list(Path('.').glob('parallel-*.json'))) < 2:\n"
        "        if time.monotonic() >= deadline:\n"
        "            raise RuntimeError('second terminal did not run concurrently')\n"
        "        time.sleep(0.05)\n"
    )
    state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(tmp_path)),
        env_observation_persistence_dir=None,
    )
    tool = SenpaiTerminalTool.create(state, role="advisor")[0]
    try:
        yield tool.executor, target
    finally:
        tool.executor.close()


def test_native_terminal_uses_target_python_and_project_imports(
    native_target_terminal, tmp_path,
):
    executor, target = native_target_terminal
    registry = SimpleNamespace(
        get_secrets_as_env_vars=lambda _command: {"PRIVATE_AUTH": "configured-value"},
        mask_secrets_in_output=lambda text: text,
    )
    conversation = SimpleNamespace(state=SimpleNamespace(secret_registry=registry))
    for reset in (False, True):
        result = executor(TerminalAction(
            command="python check_target.py result.json", reset=reset, timeout=30,
        ), conversation)
        assert not result.is_error, result.text
        assert result.exit_code == 0, result.text
        observed = json.loads((tmp_path / "result.json").read_text())
        assert observed.pop("path")[0] == str(target / "bin")
        assert observed == {
            "prefix": str(target), "value": "project import", "uv": str(target),
            "uv_python": str(target / "bin" / "python"),
            "shell": "from startup", "secret": "configured-value",
        }
        result = executor(TerminalAction(
            command='export PATH="$HOME/later-bin:$PATH" && export UV_PYTHON=custom-python',
            timeout=30,
        ))
        assert result.exit_code == 0, result.text
        result = executor(TerminalAction(
            command="python check_target.py customized.json", timeout=30,
        ))
        assert result.exit_code == 0, result.text
        customized = json.loads((tmp_path / "customized.json").read_text())
        assert customized["path"][0] == str(tmp_path / "home" / "later-bin")
        assert str(tmp_path / "home" / "custom-bin") in customized["path"]
        assert customized["uv_python"] == "custom-python"
        result = executor(TerminalAction(command="export PYTHONSAFEPATH=1", timeout=30))
        assert result.exit_code == 0, result.text
        result = executor(TerminalAction(
            command="python -c 'import sys; assert sys.flags.safe_path'", timeout=30,
        ))
        assert result.exit_code == 0, result.text

    with (tmp_path / "home" / ".bashrc").open("a") as startup:
        startup.write("readonly UV_PYTHON\n")
    with pytest.raises(ValueError, match="Target Python environment setup failed"):
        executor(TerminalAction(command="touch must-not-run", reset=True, timeout=30))
    assert not (tmp_path / "must-not-run").exists()


def test_native_tmux_initializes_each_parallel_pane(native_target_terminal, tmp_path):
    executor, target = native_target_terminal
    if shutil.which("tmux") is None:
        pytest.skip("tmux is unavailable; subprocess terminals are serial")
    assert executor.is_pooled, "tmux is installed but the native pool was not selected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda index: executor(TerminalAction(
                command=f"python check_target.py parallel-{index}.json wait", timeout=15,
            )),
            range(2),
        ))

    for index, result in enumerate(results):
        assert result.exit_code == 0, result.text
        observed = json.loads((tmp_path / f"parallel-{index}.json").read_text())
        assert observed["prefix"] == str(target)
        assert observed["path"][0] == str(target / "bin")
        assert observed["shell"] == "from startup"
