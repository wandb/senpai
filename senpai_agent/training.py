import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.processes import signal_process_group, terminate_process_group

# A W&B run path is entity/project/run_id; the URL decides which project the
# monitor queries, not the launcher's defaults.
_WANDB_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/([^/\s]+/[^/\s]+)/runs/([A-Za-z0-9_-]+)"
)
_WANDB_COMPLETE_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/([^/\s]+/[^/\s]+)/runs/"
    rb"([A-Za-z0-9_-]+)(?=[^A-Za-z0-9_-])"
)
_LOG_READ_BYTES = 64 * 1024
_WANDB_SCAN_OVERLAP_BYTES = 4096
_ERROR_TAIL_BYTES = 8192


class TrainingState(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class TrainingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[str, ...] = Field(min_length=1)
    cwd: Path
    timeout_seconds: int = Field(gt=0)


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_id: str
    state: TrainingState
    pid: int | None = Field(default=None, gt=0)
    process_group_id: int | None = Field(default=None, gt=0)
    process_start_time: float | None = Field(default=None, gt=0)
    exit_code: int | None
    elapsed_seconds: float
    log_path: str
    wandb_run_paths: tuple[str, ...] = ()
    error_tail: str = ""


def training_result_paths(state_dir: Path) -> Iterator[Path]:
    """Yield only process records owned by the training supervisor."""

    for path in state_dir.glob("*.json"):
        try:
            training_id = uuid.UUID(path.stem)
        except ValueError:
            continue
        if path.name == f"{training_id}.json":
            yield path


@dataclass
class _ActiveTraining:
    process: subprocess.Popen[bytes]
    process_group_id: int
    process_start_time: float
    started: float
    timeout_seconds: int
    log_path: Path
    cancelled: bool = False
    thread: threading.Thread | None = None


class TrainingSupervisor:
    def __init__(
        self,
        *,
        workspace: Path,
        state_dir: Path,
        terminate_grace_seconds: float = 10,
    ):
        self.workspace = workspace.resolve()
        self.state_dir = state_dir.resolve()
        self.terminate_grace_seconds = terminate_grace_seconds
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveTraining] = {}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._cancel_orphaned_runs()

    def _cancel_orphaned_runs(self) -> None:
        for path in training_result_paths(self.state_dir):
            result = TrainingResult.model_validate_json(path.read_text())
            if result.state is TrainingState.RUNNING:
                process_was_terminated = self._terminate_recovered_process(result)
                recovered = result.model_copy(
                    update={
                        "state": TrainingState.CANCELLED,
                        "error_tail": (
                            "Training state was recovered after its "
                            "supervisor restarted."
                            if process_was_terminated
                            else (
                                "Training state was recovered after its "
                                "supervisor restarted; its persisted process "
                                "identity was no longer live, so no signal "
                                "was sent."
                            )
                        ),
                    }
                )
                self._write_result(recovered)

    def run_training(self, spec: TrainingSpec) -> TrainingResult:
        if isinstance(spec.argv, str) or not spec.argv:
            raise ValueError("argv must be a non-empty sequence, not a shell string")
        cwd = Path(spec.cwd).resolve()
        if cwd != self.workspace and not cwd.is_relative_to(self.workspace):
            raise ValueError("training cwd must be inside the assignment workspace")

        training_id = str(uuid.uuid4())
        log_path = self.state_dir / f"{training_id}.log"
        started = time.monotonic()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                list(spec.argv),
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        process_start_time = psutil.Process(process.pid).create_time()
        process_group_id = process.pid

        result = TrainingResult(
            training_id=training_id,
            state=TrainingState.RUNNING,
            pid=process.pid,
            process_group_id=process_group_id,
            process_start_time=process_start_time,
            exit_code=None,
            elapsed_seconds=0,
            log_path=str(log_path),
        )
        self._write_result(result)
        active = _ActiveTraining(
            process=process,
            process_group_id=process_group_id,
            process_start_time=process_start_time,
            started=started,
            timeout_seconds=spec.timeout_seconds,
            log_path=log_path,
        )
        thread = threading.Thread(
            target=self._monitor,
            args=(training_id,),
            name=f"senpai-training-{training_id}",
        )
        active.thread = thread
        with self._lock:
            self._active[training_id] = active
            thread.start()
        return result

    def get_training_status(self, training_id: str) -> TrainingResult:
        path = self.state_dir / f"{uuid.UUID(training_id)}.json"
        result = TrainingResult.model_validate_json(path.read_text())
        with self._lock:
            active = self._active.get(training_id)
            if active is not None and result.state is TrainingState.RUNNING:
                result = result.model_copy(
                    update={"elapsed_seconds": time.monotonic() - active.started}
                )
        return result

    def cancel_training(self, training_id: str) -> TrainingResult:
        """Cancel one supervised run and wait for its terminal state."""

        result = self.get_training_status(training_id)
        if result.state is not TrainingState.RUNNING:
            return result

        with self._lock:
            active = self._active.get(training_id)
            if active is not None:
                active.cancelled = True
                thread = active.thread
            else:
                thread = None
        if thread is not None:
            thread.join()
        return self.get_training_status(training_id)

    def _monitor(self, training_id: str) -> None:
        with self._lock:
            active = self._active[training_id]
        run_paths: dict[str, None] = {}
        published_run_paths: tuple[str, ...] = ()
        error_tail = b""
        scan_overlap = b""
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
                    run_paths,
                )
                discovered_run_paths = tuple(run_paths)
                if discovered_run_paths != published_run_paths:
                    self._write_result(
                        TrainingResult(
                            training_id=training_id,
                            state=TrainingState.RUNNING,
                            pid=active.process.pid,
                            process_group_id=active.process_group_id,
                            process_start_time=active.process_start_time,
                            exit_code=None,
                            elapsed_seconds=time.monotonic() - active.started,
                            log_path=str(active.log_path),
                            wandb_run_paths=discovered_run_paths,
                        )
                    )
                    published_run_paths = discovered_run_paths
                if active.cancelled:
                    state = TrainingState.CANCELLED
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
                        TrainingState.CANCELLED
                        if active.cancelled
                        else (
                            TrainingState.FINISHED
                            if exit_code == 0
                            else TrainingState.FAILED
                        )
                    )
                    self._terminate_process_group(
                        active.process,
                        active.process_group_id,
                        grace_seconds=remaining_grace(),
                    )
                    break
                if time.monotonic() >= terminate_at:
                    state = TrainingState.TIMED_OUT
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
                run_paths,
            )
            while log.peek(1):
                error_tail, scan_overlap = self._consume_log(
                    log,
                    error_tail,
                    scan_overlap,
                    run_paths,
                )
            for match in _WANDB_RUN_URL_BYTES.findall(scan_overlap):
                run_paths.setdefault(b"/".join(match).decode(), None)

        result = TrainingResult(
            training_id=training_id,
            state=state,
            pid=active.process.pid,
            process_group_id=active.process_group_id,
            process_start_time=active.process_start_time,
            exit_code=exit_code,
            elapsed_seconds=time.monotonic() - active.started,
            log_path=str(active.log_path),
            wandb_run_paths=tuple(run_paths),
            error_tail=(
                error_tail.decode(errors="ignore")
                if state is not TrainingState.FINISHED
                else ""
            ),
        )
        self._write_result(result)
        with self._lock:
            self._active.pop(training_id, None)

    @staticmethod
    def _consume_log(
        log,
        error_tail: bytes,
        scan_overlap: bytes,
        run_paths: dict[str, None],
    ) -> tuple[bytes, bytes]:
        chunk = log.read(_LOG_READ_BYTES)
        if not chunk:
            return error_tail, scan_overlap
        scan = scan_overlap + chunk
        for match in _WANDB_COMPLETE_RUN_URL_BYTES.findall(scan):
            run_paths.setdefault(b"/".join(match).decode(), None)
        scan_overlap = scan[-_WANDB_SCAN_OVERLAP_BYTES:]
        error_tail = (error_tail + chunk)[-_ERROR_TAIL_BYTES:]
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
                self.terminate_grace_seconds
                if grace_seconds is None
                else grace_seconds
            ),
            wait_full_grace=True,
        )

    def _terminate_recovered_process(self, result: TrainingResult) -> bool:
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
    def _process_identity_matches(result: TrainingResult) -> bool:
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
            for training in self._active.values():
                training.cancelled = True
                if training.thread is not None:
                    threads.append(training.thread)
        for thread in threads:
            thread.join()

    def drain(self) -> None:
        with self._lock:
            threads = tuple(
                training.thread
                for training in self._active.values()
                if training.thread is not None
            )
        for thread in threads:
            thread.join()

    def _write_result(self, result: TrainingResult) -> None:
        path = self.state_dir / f"{result.training_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2))
        temporary.replace(path)
