import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.processes import signal_process_group, terminate_process_group

_WANDB_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/[^/\s]+/[^/\s]+/runs/([^:;,#?/\'\"\s]+)"
)
_WANDB_COMPLETE_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/[^/\s]+/[^/\s]+/runs/"
    rb"([^:;,#?/\'\"\s]+)(?=[:;,#?/\'\"\s])"
)
_LOG_READ_BYTES = 64 * 1024
_WANDB_SCAN_OVERLAP_BYTES = 4096
_ERROR_TAIL_BYTES = 8192


class JobState(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[str, ...] = Field(min_length=1)
    cwd: Path
    timeout_seconds: int = Field(gt=0)
    workspace_access: Literal["read_only", "mutable"] = "mutable"


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    state: JobState
    pid: int | None = Field(default=None, gt=0)
    process_group_id: int | None = Field(default=None, gt=0)
    process_start_time: float | None = Field(default=None, gt=0)
    exit_code: int | None
    elapsed_seconds: float
    log_path: str
    wandb_run_ids: tuple[str, ...] = ()
    error_tail: str = ""
    workspace_access: Literal["read_only", "mutable"] = "mutable"


def job_result_paths(state_dir: Path) -> Iterator[Path]:
    """Yield only process records owned by the job supervisor."""

    for path in state_dir.glob("*.json"):
        try:
            job_id = uuid.UUID(path.stem)
        except ValueError:
            continue
        if path.name == f"{job_id}.json":
            yield path


@dataclass
class _ActiveJob:
    process: subprocess.Popen[bytes]
    process_group_id: int
    process_start_time: float
    started: float
    timeout_seconds: int
    log_path: Path
    redacted_values: tuple[bytes, ...] = ()
    workspace_access: Literal["read_only", "mutable"] = "mutable"
    cancelled: bool = False
    thread: threading.Thread | None = None


class JobSupervisor:
    def __init__(
        self,
        *,
        workspace: Path,
        state_dir: Path,
        terminate_grace_seconds: float = 10,
        max_timeout_seconds: int | None = None,
    ):
        self.workspace = workspace.resolve()
        self.state_dir = state_dir.resolve()
        self.terminate_grace_seconds = terminate_grace_seconds
        if max_timeout_seconds is not None and max_timeout_seconds <= 0:
            raise ValueError("max_timeout_seconds must be positive")
        self.max_timeout_seconds = max_timeout_seconds
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveJob] = {}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._cancel_orphaned_runs()

    def _cancel_orphaned_runs(self) -> None:
        for path in job_result_paths(self.state_dir):
            result = JobResult.model_validate_json(path.read_text())
            if result.state is JobState.RUNNING:
                process_was_terminated = self._terminate_recovered_process(result)
                recovered = result.model_copy(
                    update={
                        "state": JobState.CANCELLED,
                        "error_tail": (
                            "Job state was recovered after its "
                            "supervisor restarted."
                            if process_was_terminated
                            else (
                                "Job state was recovered after its "
                                "supervisor restarted; its persisted process "
                                "identity was no longer live, so no signal "
                                "was sent."
                            )
                        ),
                    }
                )
                self._write_result(recovered)

    def run_job(
        self,
        spec: JobSpec,
        *,
        env: Mapping[str, str] | None = None,
        redacted_values: Sequence[str] = (),
    ) -> JobResult:
        if isinstance(spec.argv, str) or not spec.argv:
            raise ValueError("argv must be a non-empty sequence, not a shell string")
        if (
            self.max_timeout_seconds is not None
            and spec.timeout_seconds > self.max_timeout_seconds
        ):
            raise ValueError(
                "job timeout exceeds the configured maximum of "
                f"{self.max_timeout_seconds} seconds"
            )

        cwd = Path(spec.cwd).resolve()
        if cwd != self.workspace and not cwd.is_relative_to(self.workspace):
            raise ValueError("job cwd must be inside the assignment workspace")

        job_id = str(uuid.uuid4())
        log_path = self.state_dir / f"{job_id}.log"
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        result_written = False
        active_registered = False
        with self._lock:
            if spec.workspace_access == "mutable":
                active_mutable = self._active_mutable_job_ids_locked()
                if active_mutable:
                    raise RuntimeError(
                        "a mutable workspace job is already running: "
                        + ", ".join(active_mutable)
                    )
            try:
                log_descriptor = os.open(
                    log_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(log_descriptor, "wb") as log:
                    process = subprocess.Popen(
                        list(spec.argv),
                        cwd=cwd,
                        env=dict(env) if env is not None else None,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        start_new_session=True,
                    )
                process_start_time = psutil.Process(process.pid).create_time()
                process_group_id = process.pid
                result = JobResult(
                    job_id=job_id,
                    state=JobState.RUNNING,
                    pid=process.pid,
                    process_group_id=process_group_id,
                    process_start_time=process_start_time,
                    exit_code=None,
                    elapsed_seconds=0,
                    log_path=str(log_path),
                    workspace_access=spec.workspace_access,
                )
                self._write_result(result)
                result_written = True
                active = _ActiveJob(
                    process=process,
                    process_group_id=process_group_id,
                    process_start_time=process_start_time,
                    started=started,
                    timeout_seconds=spec.timeout_seconds,
                    log_path=log_path,
                    redacted_values=tuple(
                        value.encode() for value in redacted_values if value
                    ),
                    workspace_access=spec.workspace_access,
                )
                thread = threading.Thread(
                    target=self._monitor,
                    args=(job_id,),
                    name=f"senpai-job-{job_id}",
                )
                active.thread = thread
                self._active[job_id] = active
                active_registered = True
                thread.start()
            except BaseException as error:
                if active_registered:
                    self._active.pop(job_id, None)
                if process is not None and process.poll() is None:
                    self._terminate_process_group(
                        process,
                        process.pid,
                        grace_seconds=self.terminate_grace_seconds,
                    )
                if result_written:
                    self._write_result(
                        result.model_copy(
                            update={
                                "state": JobState.CANCELLED,
                                "exit_code": process.returncode if process else None,
                                "elapsed_seconds": time.monotonic() - started,
                                "error_tail": (
                                    "Job launch rolled back after "
                                    f"{type(error).__name__}."
                                ),
                            }
                        )
                    )
                raise
        return result

    def active_mutable_job_ids(self) -> tuple[str, ...]:
        """Return durable workspace leases held by mutable running jobs."""

        with self._lock:
            return self._active_mutable_job_ids_locked()

    def _active_mutable_job_ids_locked(self) -> tuple[str, ...]:
        return tuple(
            job_id
            for job_id, active in self._active.items()
            if active.workspace_access == "mutable"
        )

    def get_job_status(self, job_id: str) -> JobResult:
        path = self.state_dir / f"{uuid.UUID(job_id)}.json"
        result = JobResult.model_validate_json(path.read_text())
        with self._lock:
            active = self._active.get(job_id)
            if active is not None and result.state is JobState.RUNNING:
                result = result.model_copy(
                    update={"elapsed_seconds": time.monotonic() - active.started}
                )
        return result

    def cancel_job(self, job_id: str) -> JobResult:
        """Cancel one supervised run and wait for its terminal state."""

        result = self.get_job_status(job_id)
        if result.state is not JobState.RUNNING:
            return result

        with self._lock:
            active = self._active.get(job_id)
            if active is not None:
                active.cancelled = True
                thread = active.thread
            else:
                thread = None
        if thread is not None:
            thread.join()
        return self.get_job_status(job_id)

    def _monitor(self, job_id: str) -> None:
        with self._lock:
            active = self._active[job_id]
        try:
            self._monitor_process(job_id, active)
        except BaseException as error:  # noqa: BLE001
            try:
                if active.process.poll() is None:
                    self._terminate_process_group(
                        active.process,
                        active.process_group_id,
                    )
            except BaseException as terminate_error:  # noqa: BLE001
                try:
                    signal_process_group(active.process_group_id, signal.SIGKILL)
                except BaseException as kill_error:  # noqa: BLE001
                    print(
                        "SENPAI_JOB_CLEANUP_ERROR "
                        f"job_id={job_id} "
                        f"terminate_error={type(terminate_error).__name__} "
                        f"kill_error={type(kill_error).__name__}",
                        file=sys.stderr,
                        flush=True,
                    )
            try:
                self._write_result(
                    JobResult(
                        job_id=job_id,
                        state=JobState.FAILED,
                        pid=active.process.pid,
                        process_group_id=active.process_group_id,
                        process_start_time=active.process_start_time,
                        exit_code=active.process.poll(),
                        elapsed_seconds=time.monotonic() - active.started,
                        log_path=str(active.log_path),
                        error_tail=(
                            "Job supervisor failed internally "
                            f"({type(error).__name__})."
                        ),
                        workspace_access=active.workspace_access,
                    )
                )
            except BaseException as persistence_error:  # noqa: BLE001
                print(
                    "SENPAI_JOB_STATE_WRITE_ERROR "
                    f"job_id={job_id} "
                    f"error={type(persistence_error).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            with self._lock:
                self._active.pop(job_id, None)

    def _monitor_process(
        self,
        job_id: str,
        active: _ActiveJob,
    ) -> None:
        run_ids: dict[str, None] = {}
        published_run_ids: tuple[str, ...] = ()
        error_tail = b""
        scan_overlap = b""
        tail_bytes = (
            _ERROR_TAIL_BYTES
            + max(
                (len(value) for value in active.redacted_values),
                default=1,
            )
            - 1
        )
        deadline = active.started + active.timeout_seconds
        terminate_at = max(
            active.started,
            deadline - self.terminate_grace_seconds,
        )

        def remaining_grace() -> float:
            return min(
                self.terminate_grace_seconds,
                max(0.0, deadline - time.monotonic()),
            )

        with active.log_path.open("rb") as log:
            while True:
                error_tail, scan_overlap = self._consume_log(
                    log,
                    error_tail,
                    scan_overlap,
                    run_ids,
                    tail_bytes=tail_bytes,
                )
                discovered_run_ids = tuple(run_ids)
                if discovered_run_ids != published_run_ids:
                    self._write_result(
                        JobResult(
                            job_id=job_id,
                            state=JobState.RUNNING,
                            pid=active.process.pid,
                            process_group_id=active.process_group_id,
                            process_start_time=active.process_start_time,
                            exit_code=None,
                            elapsed_seconds=time.monotonic() - active.started,
                            log_path=str(active.log_path),
                            wandb_run_ids=discovered_run_ids,
                            workspace_access=active.workspace_access,
                        )
                    )
                    published_run_ids = discovered_run_ids
                if active.cancelled:
                    state = JobState.CANCELLED
                    self._terminate_process_group(
                        active.process,
                        active.process_group_id,
                        grace_seconds=remaining_grace(),
                    )
                    exit_code = active.process.returncode
                    break
                exit_code = active.process.poll()
                if exit_code is not None:
                    state = (
                        JobState.CANCELLED
                        if active.cancelled
                        else (
                            JobState.FINISHED
                            if exit_code == 0
                            else JobState.FAILED
                        )
                    )
                    self._terminate_process_group(
                        active.process,
                        active.process_group_id,
                        grace_seconds=remaining_grace(),
                    )
                    break
                if time.monotonic() >= terminate_at:
                    state = JobState.TIMED_OUT
                    self._terminate_process_group(
                        active.process,
                        active.process_group_id,
                        grace_seconds=remaining_grace(),
                    )
                    exit_code = active.process.returncode
                    break
                time.sleep(0.05)
            error_tail, scan_overlap = self._consume_log(
                log,
                error_tail,
                scan_overlap,
                run_ids,
                tail_bytes=tail_bytes,
            )
            while log.peek(1):
                error_tail, scan_overlap = self._consume_log(
                    log,
                    error_tail,
                    scan_overlap,
                    run_ids,
                    tail_bytes=tail_bytes,
                )
            for match in _WANDB_RUN_URL_BYTES.findall(scan_overlap):
                run_ids.setdefault(match.decode(), None)

        result = JobResult(
            job_id=job_id,
            state=state,
            pid=active.process.pid,
            process_group_id=active.process_group_id,
            process_start_time=active.process_start_time,
            exit_code=exit_code,
            elapsed_seconds=time.monotonic() - active.started,
            log_path=str(active.log_path),
            wandb_run_ids=tuple(run_ids),
            error_tail=(
                _redact_bytes(error_tail, active.redacted_values)[
                    -_ERROR_TAIL_BYTES:
                ].decode(errors="ignore")
                if state is not JobState.FINISHED
                else ""
            ),
            workspace_access=active.workspace_access,
        )
        self._write_result(result)

    @staticmethod
    def _consume_log(
        log,
        error_tail: bytes,
        scan_overlap: bytes,
        run_ids: dict[str, None],
        *,
        tail_bytes: int = _ERROR_TAIL_BYTES,
    ) -> tuple[bytes, bytes]:
        chunk = log.read(_LOG_READ_BYTES)
        if not chunk:
            return error_tail, scan_overlap
        scan = scan_overlap + chunk
        for match in _WANDB_COMPLETE_RUN_URL_BYTES.findall(scan):
            run_ids.setdefault(match.decode(), None)
        scan_overlap = scan[-_WANDB_SCAN_OVERLAP_BYTES:]
        error_tail = (error_tail + chunk)[-tail_bytes:]
        return error_tail, scan_overlap

    def _terminate_process_group(
        self,
        process: subprocess.Popen[bytes],
        process_group_id: int,
        *,
        grace_seconds: float | None = None,
    ) -> None:
        terminate_process_group(
            process,
            process_group_id=process_group_id,
            grace_seconds=(
                self.terminate_grace_seconds if grace_seconds is None else grace_seconds
            ),
            wait_full_grace=True,
        )

    def _terminate_recovered_process(self, result: JobResult) -> bool:
        if (
            result.pid is None
            or result.process_group_id is None
            or result.process_start_time is None
        ):
            return False
        if not self._process_identity_matches(result):
            return False
        signal_process_group(result.process_group_id, signal.SIGKILL)
        return True

    @staticmethod
    def _process_identity_matches(result: JobResult) -> bool:
        assert result.pid is not None
        assert result.process_group_id is not None
        assert result.process_start_time is not None
        try:
            process = psutil.Process(result.pid)
            return (
                process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
                and os.getpgid(result.pid) == result.process_group_id
                and process.create_time() == result.process_start_time
            )
        except (ProcessLookupError, psutil.NoSuchProcess):
            return False

    def close(self) -> None:
        with self._lock:
            threads = []
            for job in self._active.values():
                job.cancelled = True
                if job.thread is not None:
                    threads.append(job.thread)
        for thread in threads:
            thread.join()

    def drain(self) -> None:
        with self._lock:
            threads = tuple(
                job.thread
                for job in self._active.values()
                if job.thread is not None
            )
        for thread in threads:
            thread.join()

    def _write_result(self, result: JobResult) -> None:
        path = self.state_dir / f"{result.job_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2))
        temporary.replace(path)


def _redact_bytes(content: bytes, values: Sequence[bytes]) -> bytes:
    for value in values:
        if value:
            content = content.replace(value, b"<secret-hidden>")
    return content
