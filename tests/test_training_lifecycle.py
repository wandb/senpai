import sys
import time
from pathlib import Path

import pytest

from senpai_agent.worktree_snapshot import WorktreeSnapshot
from senpai_agent.training import TrainingSpec, TrainingState, TrainingSupervisor
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


def test_worktree_facts_survive_every_record_rewrite(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    worktree = WorktreeSnapshot(
        head_sha="a" * 40,
        untracked_paths=("lane/results.json",),
        snapshot_sha="b" * 40,
    )

    running = supervisor.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "import time; print('x', flush=True); time.sleep(0.3)"),
            cwd=workspace,
            timeout_seconds=20,
        ),
        worktree=worktree,
    )
    time.sleep(0.1)
    mid_run = supervisor.get_training_status(running.training_id)
    terminal = wait_for_terminal(supervisor, running.training_id)

    for record in (running, mid_run, terminal):
        assert record.source_commit == "a" * 40
        assert record.worktree_snapshot == "b" * 40
        assert record.untracked_paths == ("lane/results.json",)
    assert terminal.state is TrainingState.FINISHED

    plain = run_python(supervisor, workspace, "print('no worktree')")
    assert wait_for_terminal(supervisor, plain.training_id).source_commit is None


@pytest.mark.parametrize(
    ("expected", "code", "state"),
    [
        ((0,), 3, TrainingState.FAILED),
        ((0, 3), 3, TrainingState.FINISHED),
        ((0, 3), 0, TrainingState.FINISHED),
        ((0, 3), 1, TrainingState.FAILED),
    ],
)
def test_a_declared_exit_code_completes_the_run(tmp_path: Path, expected, code, state):
    workspace, supervisor = make_supervisor(tmp_path)

    running = supervisor.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", f"print('gate', flush=True); raise SystemExit({code})"),
            cwd=workspace,
            timeout_seconds=20,
            expected_exit_codes=expected,
        )
    )
    terminal = wait_for_terminal(supervisor, running.training_id)

    assert running.expected_exit_codes == expected
    assert (terminal.state, terminal.exit_code, terminal.expected_exit_codes) == (
        state, code, expected,
    )
    # A declared non-zero exit keeps the tail that says why the gate exited.
    assert (terminal.error_tail == "") is (code == 0)
    assert ("gate" in terminal.error_tail) is (code != 0)
    reopened = TrainingSupervisor(workspace=workspace, state_dir=tmp_path / "state")
    assert reopened.get_training_status(running.training_id).expected_exit_codes == expected


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


def test_terminal_records_set_the_wake_event_and_running_records_do_not(
    tmp_path: Path,
):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    try:
        running = run_python(
            supervisor,
            workspace,
            "import time; time.sleep(60)",
            timeout_seconds=60,
        )

        assert supervisor.wake.is_set() is False

        supervisor.cancel_training(running.training_id)

        assert supervisor.wake.is_set() is True

        supervisor.wake.clear()
        finished = run_python(supervisor, workspace, "print('done')")
        wait_for_terminal(supervisor, finished.training_id)

        assert supervisor.wake.is_set() is True
    finally:
        supervisor.close()
