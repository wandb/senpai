import time
from pathlib import Path

from senpai_agent.jobs import JobState
from job_test_support import make_supervisor, run_python, wait_for_terminal


def test_running_job_publishes_wandb_id_before_exit(tmp_path: Path):
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
            status = supervisor.get_job_status(running.job_id)
            if status.wandb_run_ids:
                break
            time.sleep(0.02)

        assert status.state is JobState.RUNNING
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

    terminal = wait_for_terminal(supervisor, running.job_id)

    assert terminal.state is JobState.FAILED
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

    terminal = wait_for_terminal(supervisor, running.job_id)

    assert terminal.state is JobState.FINISHED
    assert terminal.wandb_run_ids == ("split-run",)
