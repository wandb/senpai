import sys
import time
from pathlib import Path

import psutil

from senpai_agent.jobs import (
    JobResult,
    JobSpec,
    JobState,
    JobSupervisor,
)


def make_supervisor(
    tmp_path: Path,
    **kwargs,
) -> tuple[Path, JobSupervisor]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace, JobSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        **kwargs,
    )


def run_python(
    supervisor: JobSupervisor,
    workspace: Path,
    code: str,
    *args: str,
    timeout_seconds: int = 20,
) -> JobResult:
    return supervisor.run_job(
        JobSpec(
            argv=(sys.executable, "-c", code, *args),
            cwd=workspace,
            timeout_seconds=timeout_seconds,
        )
    )


def wait_for_terminal(
    supervisor: JobSupervisor,
    job_id: str,
    *,
    timeout: float = 5,
) -> JobResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = supervisor.get_job_status(job_id)
        if result.state is not JobState.RUNNING:
            return result
        time.sleep(0.02)
    raise AssertionError("job did not reach a terminal state")


def wait_for_path(path: Path, *, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists()


def assert_process_stopped(pid: int, *, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.05)
    raise AssertionError(f"job descendant {pid} is still running")
