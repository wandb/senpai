import array
import fcntl
import os
import re
import select
import signal
import subprocess
import termios
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from senpai_agent.processes import signal_process_group, terminate_process_group
from senpai_agent.secrets import WANDB_TRAINING_API_KEY_ENV

# A redaction marker can truncate an ID, even when only its '<' has been read.
_WANDB_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/[^/\s]+/[^/\s]+/runs/"
    rb"([A-Za-z0-9_-]+)(?![A-Za-z0-9_<-])"
)
_WANDB_COMPLETE_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/[^/\s]+/[^/\s]+/runs/"
    rb"([A-Za-z0-9_-]+)(?=[^A-Za-z0-9_<-])"
)
_LOG_READ_BYTES = 64 * 1024
_WANDB_SCAN_OVERLAP_BYTES = 4096
_ERROR_TAIL_BYTES = 8192


TARGET_PYTHON_ENV = "SENPAI_TARGET_PYTHON_ENV"


def target_python_environment(
    environment: Mapping[str, str] = os.environ,
) -> dict[str, str]:
    """Point interpreter, PATH, and uv project commands at the target venv."""

    target_env = environment.get(TARGET_PYTHON_ENV, "").strip()
    if not target_env:
        return {}
    return {
        "PATH": f"{target_env}/bin:{environment['PATH']}",
        "UV_PROJECT_ENVIRONMENT": target_env,
        "UV_PYTHON": f"{target_env}/bin/python",
        "VIRTUAL_ENV": target_env,
    }


def _mask_output_chunk(data: bytes, secret: bytes) -> tuple[bytes, bytes]:
    """Emit complete bytes while retaining a possible split secret prefix."""
    if not secret:
        return data, b""
    pieces = []
    start = 0
    while (match := data.find(secret, start)) >= 0:
        pieces.extend((data[start:match], b"<secret-hidden>"))
        start = match + len(secret)
    tail = data[start:]
    overlap = next(
        (
            size
            for size in range(min(len(tail), len(secret) - 1), 0, -1)
            if tail.endswith(secret[:size])
        ),
        0,
    )
    pieces.append(tail[:-overlap] if overlap else tail)
    return b"".join(pieces), tail[-overlap:] if overlap else b""


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
    wandb_run_ids: tuple[str, ...] = ()
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
    output_thread: threading.Thread | None = None
    output_stop: threading.Event = field(default_factory=threading.Event)
    output_error: str | None = None


class TrainingSupervisor:
    def __init__(
        self,
        *,
        workspace: Path,
        state_dir: Path,
        wandb_api_key: SecretStr | None = None,
        terminate_grace_seconds: float = 10,
    ):
        self.workspace = workspace.resolve()
        self.state_dir = state_dir.resolve()
        self.terminate_grace_seconds = terminate_grace_seconds
        self.wandb_api_key = wandb_api_key
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
        log_path.touch()
        environment = dict(os.environ)
        environment.pop("PYTHONSAFEPATH", None)
        environment.update(target_python_environment(environment))
        environment.pop(WANDB_TRAINING_API_KEY_ENV, None)
        environment.pop("WANDB_INFERENCE_API_KEY", None)
        environment.pop("WANDB_SERVICE", None)
        if self.wandb_api_key is not None:
            environment.pop("WANDB_IDENTITY_TOKEN_FILE", None)
            environment["WANDB_API_KEY"] = self.wandb_api_key.get_secret_value()
        process = subprocess.Popen(
            list(spec.argv),
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
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
        active.output_thread = threading.Thread(
            target=self._record_output,
            args=(active,),
            name=f"senpai-training-output-{training_id}",
        )
        with self._lock:
            self._active[training_id] = active
            active.output_thread.start()
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

    def _record_output(self, active: _ActiveTraining) -> None:
        """Mask the writer before bytes reach the model-readable training log."""
        secret = (
            self.wandb_api_key.get_secret_value().encode()
            if self.wandb_api_key
            else b""
        )
        pending = b""
        remaining = None
        assert active.process.stdout is not None
        try:
            with active.process.stdout as stream, active.log_path.open("wb") as log:
                os.set_blocking(stream.fileno(), False)
                while True:
                    if remaining is None and active.output_stop.is_set():
                        # Drain the queued backlog, without allowing escaped
                        # descendants to extend the training lifecycle.
                        queued = array.array("i", [0])
                        fcntl.ioctl(stream.fileno(), termios.FIONREAD, queued, True)
                        remaining = queued[0]
                    if remaining == 0:
                        break
                    try:
                        chunk = os.read(
                            stream.fileno(),
                            _LOG_READ_BYTES
                            if remaining is None
                            else min(_LOG_READ_BYTES, remaining),
                        )
                    except BlockingIOError:
                        select.select([stream], [], [], 0.05)
                        continue
                    if not chunk:
                        break
                    if remaining is not None:
                        remaining -= len(chunk)
                    sanitized, pending = _mask_output_chunk(pending + chunk, secret)
                    log.write(sanitized)
                    log.flush()
                if pending:
                    log.write(b"<secret-hidden>")
        except Exception as error:
            active.output_error = (
                f"Training output capture failed ({type(error).__name__})."
            )

    def _monitor(self, training_id: str) -> None:
        with self._lock:
            active = self._active[training_id]
        run_ids: dict[str, None] = {}
        published_run_ids: tuple[str, ...] = ()
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
                    run_ids,
                )
                discovered_run_ids = tuple(run_ids)
                if discovered_run_ids != published_run_ids:
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
                            wandb_run_ids=discovered_run_ids,
                        )
                    )
                    published_run_ids = discovered_run_ids
                if active.cancelled or active.output_error is not None:
                    state = (
                        TrainingState.CANCELLED
                        if active.cancelled
                        else TrainingState.FAILED
                    )
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
            active.output_stop.set()
            assert active.output_thread is not None
            active.output_thread.join()
            error_tail, scan_overlap = self._consume_log(
                log,
                error_tail,
                scan_overlap,
                run_ids,
            )
            while log.peek(1):
                error_tail, scan_overlap = self._consume_log(
                    log,
                    error_tail,
                    scan_overlap,
                    run_ids,
                )
            for match in _WANDB_RUN_URL_BYTES.findall(scan_overlap):
                run_ids.setdefault(match.decode(), None)
        if active.output_error is not None:
            state = TrainingState.FAILED
            error_tail = active.output_error.encode()

        result = TrainingResult(
            training_id=training_id,
            state=state,
            pid=active.process.pid,
            process_group_id=active.process_group_id,
            process_start_time=active.process_start_time,
            exit_code=exit_code,
            elapsed_seconds=time.monotonic() - active.started,
            log_path=str(active.log_path),
            wandb_run_ids=tuple(run_ids),
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
        run_ids: dict[str, None],
    ) -> tuple[bytes, bytes]:
        chunk = log.read(_LOG_READ_BYTES)
        if not chunk:
            return error_tail, scan_overlap
        scan = scan_overlap + chunk
        for match in _WANDB_COMPLETE_RUN_URL_BYTES.findall(scan):
            run_ids.setdefault(match.decode(), None)
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
