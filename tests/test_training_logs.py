import os
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from senpai_agent.training import TrainingState
from training_test_support import make_supervisor, run_python, wait_for_terminal


def test_training_captures_output_with_many_open_files(tmp_path: Path):
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit < 2048:
        pytest.skip("requires room for a subprocess above file descriptor 1024")
    workspace, supervisor = make_supervisor(tmp_path, terminate_grace_seconds=0.1)
    with ExitStack() as files:
        # Force the training pipe above select()'s descriptor limit.
        while files.enter_context(open(os.devnull, "rb")).fileno() < 1024:
            pass
        try:
            running = run_python(
                supervisor,
                workspace,
                "import time; time.sleep(0.1); print('captured', flush=True)",
            )
            terminal = wait_for_terminal(supervisor, running.training_id)

            assert terminal.state is TrainingState.FINISHED
            assert Path(terminal.log_path).read_text() == "captured\n"
        finally:
            supervisor.close()


def test_running_training_publishes_wandb_id_before_exit(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    running = run_python(
        supervisor,
        workspace,
        (
            "import time; "
            "print('https://wandb.ai/acme/cfd/runs/live-run', flush=True); "
            "time.sleep(60)"
        ),
    )

    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            status = supervisor.get_training_status(running.training_id)
            if status.wandb_run_ids:
                break
            time.sleep(0.02)

        assert status.state is TrainingState.RUNNING
        assert status.wandb_run_ids == ("live-run",)
    finally:
        supervisor.close()


def test_large_failed_log_keeps_run_ids_and_only_the_bounded_tail(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    output_code = (
        "import sys;"
        "print('https://wandb.ai/acme/cfd/runs/first-run', flush=True);"
        "sys.stdout.write('x' * 2_000_000);"
        "print('\\nhttps://wandb.ai/acme/cfd/runs/last-run', flush=True);"
        "raise SystemExit(7)"
    )
    running = run_python(supervisor, workspace, output_code)

    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FAILED
    assert terminal.exit_code == 7
    assert terminal.wandb_run_ids == ("first-run", "last-run")
    assert len(terminal.error_tail.encode()) <= 8192
    assert "last-run" in terminal.error_tail
    assert "first-run" not in terminal.error_tail


def test_wandb_url_can_span_log_read_chunks(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    run_url = b"https://wandb.ai/acme/cfd/runs/split-run\n"
    output = b"x" * (64 * 1024 - len(run_url) // 2) + run_url
    running = run_python(
        supervisor,
        workspace,
        f"import os; os.write(1, {output!r})",
    )

    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FINISHED
    assert terminal.wandb_run_ids == ("split-run",)
