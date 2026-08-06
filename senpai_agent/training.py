import hashlib
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import psutil
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from senpai_agent.processes import signal_process_group, terminate_process_group

_WANDB_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/[^/\s]+/[^/\s]+/runs/([A-Za-z0-9_-]+)"
)
_WANDB_COMPLETE_RUN_URL_BYTES = re.compile(
    rb"https?://wandb\.ai/[^/\s]+/[^/\s]+/runs/"
    rb"([A-Za-z0-9_-]+)(?=[^A-Za-z0-9_-])"
)
_LOG_READ_BYTES = 64 * 1024
_WANDB_SCAN_OVERLAP_BYTES = 4096
_ERROR_TAIL_BYTES = 8192
TRAINING_INVENTORY_FILENAME = "inventory.json"
_RECENT_TRAINING_LIMIT = 64
_RECENT_WANDB_RUN_LIMIT = 200
_RECENT_ERROR_LIMIT = 20


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


class TrainingInventoryEntry(BaseModel):
    """One current or recently terminal result in the bounded inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: TrainingResult
    updated_at: AwareDatetime


class TrainingInventoryError(BaseModel):
    """Sanitized error evidence retained independently of result history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_id: str
    state: TrainingState
    observed_at: AwareDatetime
    fingerprint: str = Field(pattern=r"^[0-9a-f]{16}$")


class TrainingInventory(BaseModel):
    """Bounded observation index; full per-training results remain on disk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    active: tuple[TrainingInventoryEntry, ...] = ()
    recent_terminal: tuple[TrainingInventoryEntry, ...] = Field(
        default=(),
        max_length=_RECENT_TRAINING_LIMIT,
    )
    recent_wandb_run_ids: tuple[str, ...] = Field(
        default=(),
        max_length=_RECENT_WANDB_RUN_LIMIT,
    )
    wandb_run_inventory_overflow: bool = False
    recent_errors: tuple[TrainingInventoryError, ...] = Field(
        default=(),
        max_length=_RECENT_ERROR_LIMIT,
    )


def training_inventory_path(state_dir: Path) -> Path:
    directory = state_dir.resolve()
    path = (directory / TRAINING_INVENTORY_FILENAME).resolve()
    if path.parent != directory:
        raise ValueError("training inventory must remain inside its state directory")
    return path


def training_result_path(state_dir: Path, training_id: str) -> Path:
    directory = state_dir.resolve()
    path = (directory / f"{training_id}.json").resolve()
    if (
        not training_id
        or path.parent != directory
        or path.name == TRAINING_INVENTORY_FILENAME
    ):
        raise ValueError("training id does not name a local result")
    return path


def read_training_inventory(state_dir: Path) -> TrainingInventory:
    return TrainingInventory.model_validate_json(
        training_inventory_path(state_dir).read_text(encoding="utf-8")
    )


def _inventory_with_result(
    inventory: TrainingInventory,
    result: TrainingResult,
    observed_at: datetime,
) -> TrainingInventory:
    entry = TrainingInventoryEntry(result=result, updated_at=observed_at)
    active = {
        item.result.training_id: item
        for item in inventory.active
        if item.result.training_id != result.training_id
    }
    terminal = [
        item
        for item in inventory.recent_terminal
        if item.result.training_id != result.training_id
    ]
    if result.state is TrainingState.RUNNING:
        active[result.training_id] = entry
    else:
        terminal.append(entry)

    run_ids = dict.fromkeys(inventory.recent_wandb_run_ids)
    overflow = inventory.wandb_run_inventory_overflow
    for run_id in result.wandb_run_ids:
        if run_id in run_ids:
            continue
        run_ids[run_id] = None
        if len(run_ids) > _RECENT_WANDB_RUN_LIMIT:
            run_ids.pop(next(iter(run_ids)))
            overflow = True

    errors = [
        item
        for item in inventory.recent_errors
        if item.training_id != result.training_id
    ]
    if result.error_tail:
        errors.append(
            TrainingInventoryError(
                training_id=result.training_id,
                state=result.state,
                observed_at=observed_at,
                fingerprint=hashlib.sha256(
                    result.error_tail.encode(errors="ignore")
                ).hexdigest()[:16],
            )
        )

    return TrainingInventory(
        active=tuple(active.values()),
        recent_terminal=tuple(terminal[-_RECENT_TRAINING_LIMIT:]),
        recent_wandb_run_ids=tuple(run_ids),
        wandb_run_inventory_overflow=overflow,
        recent_errors=tuple(errors[-_RECENT_ERROR_LIMIT:]),
    )


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
        max_timeout_seconds: int | None = None,
    ):
        self.workspace = workspace.resolve()
        self.state_dir = state_dir.resolve()
        self.terminate_grace_seconds = terminate_grace_seconds
        if max_timeout_seconds is not None and max_timeout_seconds <= 0:
            raise ValueError("max_timeout_seconds must be positive")
        self.max_timeout_seconds = max_timeout_seconds
        self._lock = threading.Lock()
        self._inventory_lock = threading.Lock()
        self._active: dict[str, _ActiveTraining] = {}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_inventory()
        self._cancel_orphaned_runs()

    def _cancel_orphaned_runs(self) -> None:
        inventory = read_training_inventory(self.state_dir)
        for entry in inventory.active:
            result = entry.result
            process_was_terminated = self._terminate_recovered_process(result)
            recovered = result.model_copy(
                update={
                    "state": TrainingState.CANCELLED,
                    "error_tail": (
                        "Training state was recovered after its supervisor restarted."
                        if process_was_terminated
                        else (
                            "Training state was recovered after its supervisor "
                            "restarted; its persisted process identity was no longer "
                            "live, so no signal was sent."
                        )
                    ),
                }
            )
            self._write_result(recovered)

    def _ensure_inventory(self) -> None:
        path = training_inventory_path(self.state_dir)
        if path.is_file():
            inventory = read_training_inventory(self.state_dir)
            self._repair_indexed_results(inventory)
            return

        inventory = TrainingInventory()
        paths = sorted(
            (
                candidate
                for candidate in self.state_dir.glob("*.json")
                if candidate.name != TRAINING_INVENTORY_FILENAME
            ),
            key=lambda candidate: (candidate.stat().st_mtime, candidate.name),
        )
        for candidate in paths:
            result = TrainingResult.model_validate_json(
                candidate.read_text(encoding="utf-8")
            )
            if candidate.stem != result.training_id:
                raise ValueError("training result path does not match its id")
            inventory = _inventory_with_result(
                inventory,
                result,
                datetime.fromtimestamp(candidate.stat().st_mtime, UTC),
            )
        self._write_inventory(inventory)

    def _repair_indexed_results(self, inventory: TrainingInventory) -> None:
        for entry in (*inventory.active, *inventory.recent_terminal):
            path = training_result_path(self.state_dir, entry.result.training_id)
            try:
                persisted = TrainingResult.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                persisted = None
            if persisted != entry.result:
                self._write_result_file(entry.result)

    def run_training(self, spec: TrainingSpec) -> TrainingResult:
        if isinstance(spec.argv, str) or not spec.argv:
            raise ValueError("argv must be a non-empty sequence, not a shell string")
        if (
            self.max_timeout_seconds is not None
            and spec.timeout_seconds > self.max_timeout_seconds
        ):
            raise ValueError(
                "training timeout exceeds the configured maximum of "
                f"{self.max_timeout_seconds} seconds"
            )

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
        return training_process_is_live(result)

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
        with self._inventory_lock:
            training_result_path(self.state_dir, result.training_id)
            inventory = _inventory_with_result(
                read_training_inventory(self.state_dir),
                result,
                datetime.now(UTC),
            )
            self._write_inventory(inventory)
            self._write_result_file(result)

    def _write_inventory(self, inventory: TrainingInventory) -> None:
        path = training_inventory_path(self.state_dir)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def _write_result_file(self, result: TrainingResult) -> None:
        path = training_result_path(self.state_dir, result.training_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)


def training_process_is_live(result: TrainingResult) -> bool:
    """Verify a persisted RUNNING result still names the same live process."""

    if (
        result.pid is None
        or result.process_group_id is None
        or result.process_start_time is None
    ):
        return False
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
