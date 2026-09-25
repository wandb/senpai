# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import json
from pathlib import Path

import pytest
from openhands.sdk.plugin import Plugin

from senpai_agent.hooks import hook_main, queued_feedback_marker

PLUGIN_DIR = Path(__file__).parents[1] / "plugins" / "senpai"
TRAINING_ID = "b81440b1-b803-471e-9fe0-6dcabd756b83"


def invoke_hook(
    command: str,
    event: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    monkeypatch.setattr("sys.stdin.read", lambda: json.dumps(event))
    exit_code = hook_main([command], env or {})
    return exit_code, json.loads(capsys.readouterr().out)


@pytest.fixture
def assignment_workspace(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def write_running_training(state_dir: Path, *, monitored: bool) -> None:
    training_dir = state_dir / "training"
    training_dir.mkdir(parents=True)
    (training_dir / f"{TRAINING_ID}.json").write_text('{"state":"running"}')
    if monitored:
        monitor_dir = training_dir / "monitors"
        monitor_dir.mkdir()
        (monitor_dir / f"{TRAINING_ID}.json").write_text("{}")


def test_pre_tool_hook_emits_a_native_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    exit_code, output = invoke_hook(
        "pre-tool-use",
        {
            "tool_input": {"command": "git push origin experiment"},
            "working_dir": str(tmp_path),
        },
        monkeypatch,
        capsys,
        {"SENPAI_ROLE": "student"},
    )

    assert (exit_code, output["decision"]) == (2, "deny")


def test_pre_tool_hook_fails_closed_on_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr("sys.stdin.read", lambda: "not-json")

    assert hook_main(["pre-tool-use"], {}) == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "deny"


def test_plugin_loads_terminal_safety_and_lifecycle_hooks():
    plugin = Plugin.load(PLUGIN_DIR)
    hooks = json.loads((PLUGIN_DIR / "hooks" / "hooks.json").read_text())

    assert plugin.hooks is not None
    assert {hook["matcher"] for hook in hooks["PreToolUse"]} == {
        "senpai_terminal",
        "terminal",
    }
    assert hooks["Stop"] and hooks["SessionEnd"]
    commands = [
        hook["command"]
        for group in hooks.values()
        for registration in group
        for hook in registration["hooks"]
        if hook["type"] == "command"
    ]
    assert commands
    assert all(
        command.startswith("/opt/senpai-venv/bin/python -P -m senpai_agent.hooks ")
        for command in commands
    )


@pytest.mark.parametrize("queued_feedback", (False, True))
def test_student_stop_denies_unmonitored_training(
    assignment_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    queued_feedback: bool,
):
    state_dir = tmp_path / "state"
    write_running_training(state_dir, monitored=False)
    if queued_feedback:
        queued_feedback_marker(state_dir).touch()

    exit_code, output = invoke_hook(
        "stop",
        {"working_dir": str(assignment_workspace)},
        monkeypatch,
        capsys,
        {
            "SENPAI_ROLE": "student",
            "SENPAI_OPENHANDS_STATE_DIR": str(state_dir),
        },
    )

    assert (exit_code, output["decision"]) == (2, "deny")
    assert TRAINING_ID in str(output["reason"])


def test_student_stop_allows_durable_monitored_training(
    assignment_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    state_dir = tmp_path / "state"
    write_running_training(state_dir, monitored=True)
    monkeypatch.setattr(
        "senpai_agent.hooks.run_git",
        lambda *_args, **_kwargs: "",
    )

    exit_code, output = invoke_hook(
        "stop",
        {"working_dir": str(assignment_workspace)},
        monkeypatch,
        capsys,
        {
            "SENPAI_ROLE": "student",
            "SENPAI_OPENHANDS_STATE_DIR": str(state_dir),
        },
    )

    assert (exit_code, output["decision"]) == (0, "allow")


def test_student_stop_ignores_non_training_json_sidecars(
    assignment_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    state_dir = tmp_path / "state"
    training_dir = state_dir / "training"
    training_dir.mkdir(parents=True)
    (training_dir / f"{TRAINING_ID}.score.json").write_text(
        '{"metrics": {}, "passed": true, "score": 1.0}'
    )
    monkeypatch.setattr(
        "senpai_agent.hooks.run_git",
        lambda *_args, **_kwargs: "",
    )

    exit_code, output = invoke_hook(
        "stop",
        {"working_dir": str(assignment_workspace)},
        monkeypatch,
        capsys,
        {
            "SENPAI_ROLE": "student",
            "SENPAI_OPENHANDS_STATE_DIR": str(state_dir),
        },
    )

    assert (exit_code, output["decision"]) == (0, "allow")


def test_student_stop_denies_a_dirty_assignment_workspace(
    assignment_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        "senpai_agent.hooks.run_git",
        lambda *_args, **_kwargs: " M model.py\n",
    )

    exit_code, output = invoke_hook(
        "stop",
        {"working_dir": str(assignment_workspace)},
        monkeypatch,
        capsys,
        {"SENPAI_ROLE": "student"},
    )

    assert (exit_code, output["decision"]) == (2, "deny")


def test_queued_feedback_temporarily_allows_a_clean_unwind(
    assignment_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    queued_feedback_marker(state_dir).touch()
    monkeypatch.setattr(
        "senpai_agent.hooks.run_git",
        lambda *_args, **_kwargs: " M model.py\n",
    )

    exit_code, output = invoke_hook(
        "stop",
        {"working_dir": str(assignment_workspace)},
        monkeypatch,
        capsys,
        {
            "SENPAI_ROLE": "student",
            "SENPAI_OPENHANDS_STATE_DIR": str(state_dir),
        },
    )

    assert (exit_code, output["decision"]) == (0, "allow")


def test_session_end_allows_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    exit_code, output = invoke_hook("session-end", {}, monkeypatch, capsys)

    assert (exit_code, output["decision"]) == (0, "allow")
