from pathlib import Path

import pytest

from senpai_agent.training import TrainingState, TrainingSupervisor
from training_test_support import (
    assert_process_stopped,
    make_supervisor,
    run_python,
    wait_for_path,
    wait_for_terminal,
)


def test_finished_training_persists_its_result_and_log(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    running = run_python(
        supervisor,
        workspace,
        "print('https://wandb.ai/acme/cfd/runs/run-123', flush=True)",
    )

    terminal = wait_for_terminal(supervisor, running.training_id)
    reopened = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
    ).get_training_status(running.training_id)

    assert running.state is TrainingState.RUNNING
    assert running.pid is not None
    assert terminal.state is TrainingState.FINISHED
    assert terminal.exit_code == 0
    assert terminal.wandb_run_ids == ("run-123",)
    assert Path(terminal.log_path).read_text().strip().endswith("/runs/run-123")
    assert reopened == terminal


def test_recovery_ignores_non_training_json_sidecars(tmp_path: Path):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    state_dir.mkdir()
    sidecar = state_dir / "b81440b1-b803-471e-9fe0-6dcabd756b83.score.json"
    contents = '{"metrics": {}, "passed": true, "score": 1.0}'
    sidecar.write_text(contents)

    supervisor = TrainingSupervisor(workspace=workspace, state_dir=state_dir)

    assert sidecar.read_text() == contents
    supervisor.close()


def test_recovery_rejects_a_corrupt_training_result(tmp_path: Path):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    state_dir.mkdir()
    result = state_dir / "b81440b1-b803-471e-9fe0-6dcabd756b83.json"
    result.write_text('{"state": "running"}')

    with pytest.raises(ValueError):
        TrainingSupervisor(workspace=workspace, state_dir=state_dir)


def test_training_passes_shell_metacharacters_as_a_literal_argument(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    literal = "result; $(echo not-a-shell)"
    running = run_python(
        supervisor,
        workspace,
        "import sys; print(sys.argv[1])",
        literal,
    )

    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FINISHED
    assert Path(terminal.log_path).read_text().strip() == literal


def test_training_rejects_a_working_directory_outside_the_workspace(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)

    with pytest.raises(ValueError, match="inside"):
        run_python(
            supervisor,
            workspace.parent,
            "print('never launched')",
        )


def test_training_rejects_timeout_above_the_launch_ceiling(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path, max_timeout_seconds=30)

    with pytest.raises(ValueError, match="configured maximum of 30 seconds"):
        run_python(
            supervisor,
            workspace,
            "print('never launched')",
            timeout_seconds=31,
        )


def test_supervisor_close_cancels_active_training(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    running = run_python(
        supervisor,
        workspace,
        "import time; time.sleep(60)",
        timeout_seconds=60,
    )

    supervisor.close()

    assert supervisor.get_training_status(running.training_id).state is (
        TrainingState.CANCELLED
    )


def test_cancel_training_stops_one_run_and_is_idempotent(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    running = run_python(
        supervisor,
        workspace,
        "import time; time.sleep(60)",
        timeout_seconds=60,
    )

    cancelled = supervisor.cancel_training(running.training_id)

    assert cancelled.state is TrainingState.CANCELLED
    assert supervisor.cancel_training(running.training_id) == cancelled


def test_cancel_training_stops_term_ignoring_descendants(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    descendant_path = workspace / "descendant.pid"
    running = run_python(
        supervisor,
        workspace,
        (
            "import pathlib, signal, subprocess, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal, time; signal.signal(signal.SIGTERM, "
            "signal.SIG_IGN); time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
            "time.sleep(60)"
        ),
        str(descendant_path),
        timeout_seconds=60,
    )
    wait_for_path(descendant_path)
    descendant_pid = int(descendant_path.read_text())

    cancelled = supervisor.cancel_training(running.training_id)

    assert cancelled.state is TrainingState.CANCELLED
    assert running.pid is not None
    assert_process_stopped(running.pid)
    assert_process_stopped(descendant_pid)


def test_supervisor_drain_waits_for_training_to_finish(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    running = run_python(
        supervisor,
        workspace,
        "import time; time.sleep(0.1)",
    )

    supervisor.drain()

    assert supervisor.get_training_status(running.training_id).state is (
        TrainingState.FINISHED
    )
