"""Small process boundary that restarts a crashed or wedged Senpai controller."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import psutil
from pydantic import SecretStr

from senpai_agent.operations import (
    RestartCompletion,
    RestartRequest,
    RestartRequestStore,
    RoleTarget,
)
from senpai_agent.processes import terminate_process_group
from senpai_agent.secrets import (
    GITHUB_TOKEN_FD_ENV,
    GITHUB_TOKEN_FILE_ENV,
    scrub_github_credentials,
)

LEASE_ENV = "SENPAI_CONTROLLER_LEASE_PATH"
GENERATION_ENV = "SENPAI_CONTROLLER_GENERATION"


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    startup_timeout_seconds: float = 300
    check_interval_seconds: float = 5
    terminate_grace_seconds: float = 60
    initial_backoff_seconds: float = 1
    max_backoff_seconds: float = 300

    def __post_init__(self) -> None:
        if (
            min(
                self.startup_timeout_seconds,
                self.check_interval_seconds,
                self.terminate_grace_seconds,
                self.initial_backoff_seconds,
                self.max_backoff_seconds,
            )
            <= 0
        ):
            raise ValueError("supervisor durations must be positive")


@dataclass(frozen=True, slots=True)
class WorkerLease:
    pid: int
    phase: str
    deadline: float
    completed_turns: int = 0
    conversation_id: str | None = None
    generation: int | None = None

    @classmethod
    def read(cls, path: Path) -> WorkerLease:
        value = json.loads(path.read_text(encoding="utf-8"))
        lease = cls(
            pid=int(value["pid"]),
            phase=str(value["phase"]),
            deadline=float(value["deadline"]),
            completed_turns=int(value.get("completed_turns", 0)),
            conversation_id=(
                str(value["conversation_id"])
                if value.get("conversation_id") is not None
                else None
            ),
            generation=(
                int(value["generation"])
                if value.get("generation") is not None
                else None
            ),
        )
        if (
            lease.pid <= 0
            or not lease.phase
            or lease.completed_turns < 0
            or (lease.generation is not None and lease.generation <= 0)
        ):
            raise ValueError("invalid controller lease")
        return lease


class ProgressLease:
    """Publish the worker's current phase and non-cooperative hard deadline."""

    def __init__(self, path: Path, *, generation: int | None = None):
        self.path = path.resolve()
        if generation is not None and generation <= 0:
            raise ValueError("worker generation must be positive")
        self.generation = generation
        self.completed_turns = 0
        self.conversation_id: str | None = None

    def update(
        self,
        phase: str,
        timeout_seconds: float,
        *,
        completed_turn: bool = False,
        conversation_id: str | None = None,
    ) -> None:
        if not phase or timeout_seconds <= 0:
            raise ValueError("progress phase and timeout must be positive")
        if completed_turn:
            self.completed_turns += 1
        if conversation_id is not None:
            self.conversation_id = conversation_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "phase": phase,
                    "deadline": time.monotonic() + timeout_seconds,
                    "completed_turns": self.completed_turns,
                    "conversation_id": self.conversation_id,
                    "generation": self.generation,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)


class WorkerSupervisor:
    """Keep one long-running controller worker alive behind a hard lease."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        lease_path: Path,
        config: SupervisorConfig | None = None,
        environment: Mapping[str, str] | None = None,
        github_token: SecretStr | None = None,
    ):
        if not command:
            raise ValueError("worker command must not be empty")
        self.command = tuple(command)
        self.lease_path = lease_path.resolve()
        self.config = config or SupervisorConfig()
        self.environment = dict(os.environ if environment is None else environment)
        scrub_github_credentials(self.environment)
        self.github_token = github_token
        self.restart_path = self.lease_path.parent / "controller-restarts.sqlite3"
        self.role_target = self._role_target(self.environment)

    def run(self, stop: threading.Event | None = None) -> int:
        stop = stop or threading.Event()
        failures = 0
        with RestartRequestStore(self.restart_path) as restarts:
            while not stop.is_set():
                generation = restarts.allocate_worker_generation()
                self.lease_path.unlink(missing_ok=True)
                environment = {
                    **self.environment,
                    LEASE_ENV: str(self.lease_path),
                    GENERATION_ENV: str(generation),
                }
                token_fd = self._open_github_token_pipe()
                if token_fd is not None:
                    environment[GITHUB_TOKEN_FD_ENV] = str(token_fd)
                try:
                    process = subprocess.Popen(
                        self.command,
                        env=environment,
                        start_new_session=True,
                        pass_fds=(token_fd,) if token_fd is not None else (),
                    )
                finally:
                    if token_fd is not None:
                        os.close(token_fd)
                started = time.monotonic()
                descendants: dict[int, float] = {}
                try:
                    reason, made_progress, planned = self._wait_for_worker(
                        process,
                        descendants,
                        stop,
                        started,
                        generation=generation,
                        restarts=restarts,
                    )
                finally:
                    self._terminate_worker(process, descendants)
                    self._reap_orphaned_children(None)
                if stop.is_set():
                    return 0
                if planned is not None:
                    print(
                        "SENPAI_CONTROLLER_PLANNED_RESTART "
                        f"request_id={planned.request_id} "
                        f"source_generation={planned.expected_worker_generation}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                runtime = time.monotonic() - started
                failures = 1 if made_progress else failures + 1
                exponential_delay = min(
                    self.config.max_backoff_seconds,
                    self.config.initial_backoff_seconds
                    * (2 ** min(failures - 1, 16)),
                )
                delay = min(
                    self.config.max_backoff_seconds,
                    exponential_delay * random.uniform(0.8, 1.2),
                )
                print(
                    "SENPAI_CONTROLLER_RESTART "
                    f"reason={reason} runtime_seconds={runtime:.1f} "
                    f"completed_turn={str(made_progress).lower()} "
                    f"restart_failures={failures} backoff_seconds={delay:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                stop.wait(delay)
        return 0

    def _open_github_token_pipe(self) -> int | None:
        if self.github_token is None:
            return None
        read_fd, write_fd = os.pipe()
        try:
            os.write(
                write_fd,
                self.github_token.get_secret_value().encode(),
            )
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        return read_fd

    def _wait_for_worker(
        self,
        process: subprocess.Popen[bytes],
        descendants: dict[int, float],
        stop: threading.Event,
        started: float,
        *,
        generation: int | None = None,
        restarts: RestartRequestStore | None = None,
    ) -> tuple[str, bool, RestartRequest | None]:
        made_progress = False
        while not stop.is_set():
            self._remember_descendants(process, descendants)
            self._reap_orphaned_children(process.pid)
            lease = self._read_lease()
            now = time.monotonic()
            if lease is not None and lease.pid == process.pid:
                made_progress = made_progress or lease.completed_turns > 0
                if (
                    generation is not None
                    and restarts is not None
                    and lease.generation == generation
                    and self.role_target is not None
                    and now <= lease.deadline
                ):
                    self._complete_replacements(restarts, lease)
                    planned = self._claim_planned_restart(restarts, lease)
                    if planned is not None:
                        return f"planned:{planned.request_id}", made_progress, planned
                if now > lease.deadline:
                    return f"overdue:{lease.phase}", made_progress, None
            elif now - started > self.config.startup_timeout_seconds:
                return "startup-timeout", made_progress, None

            exit_code = process.poll()
            if exit_code is not None:
                final_lease = self._read_lease()
                if final_lease is not None and final_lease.pid == process.pid:
                    made_progress = (
                        made_progress or final_lease.completed_turns > 0
                    )
                return f"exit:{exit_code}", made_progress, None
            stop.wait(self.config.check_interval_seconds)
        return "shutdown", made_progress, None

    def _claim_planned_restart(
        self,
        restarts: RestartRequestStore,
        lease: WorkerLease,
    ) -> RestartRequest | None:
        assert self.role_target is not None
        if (
            lease.generation is None
            or lease.phase != "sleep"
            or time.monotonic() > lease.deadline
        ):
            return None
        request = restarts.claim_next(
            self.role_target,
            worker_generation=lease.generation,
            replacement_generation=lease.generation + 1,
        )
        if request is None:
            return None
        if lease.conversation_id != str(request.expected_conversation_id):
            restarts.reject(request.request_id, "conversation-changed")
            return None
        if lease.completed_turns != request.expected_completed_turns:
            restarts.reject(request.request_id, "worker-progressed")
            return None
        if not self._compute_is_quiescent():
            restarts.reject(request.request_id, "compute-not-quiescent")
            return None
        return request

    def _complete_replacements(
        self,
        restarts: RestartRequestStore,
        lease: WorkerLease,
    ) -> None:
        assert self.role_target is not None and lease.generation is not None
        if lease.conversation_id is None:
            return
        for request in restarts.missed_sources(
            self.role_target,
            live_generation=lease.generation,
        ):
            restarts.reject(request.request_id, "source-generation-missed")
        for request in restarts.missed_replacements(
            self.role_target,
            replacement_generation=lease.generation,
        ):
            restarts.reject(request.request_id, "replacement-generation-missed")
        for request in restarts.awaiting_replacement(
            self.role_target,
            replacement_generation=lease.generation,
        ):
            if lease.conversation_id != str(request.expected_conversation_id):
                restarts.reject(request.request_id, "replacement-conversation-changed")
                continue
            restarts.complete(
                RestartCompletion(
                    request_id=request.request_id,
                    target=request.target,
                    conversation_id=request.expected_conversation_id,
                    source_generation=request.expected_worker_generation,
                    replacement_generation=lease.generation,
                    state_preserved=True,
                    compute_preserved=True,
                )
            )

    def _compute_is_quiescent(self) -> bool:
        if self.role_target is None:
            return False
        state_dir = self.lease_path.parent
        training_dir = state_dir / "training"
        if training_dir.exists():
            from senpai_agent.training import read_training_inventory

            try:
                if read_training_inventory(training_dir).active:
                    return False
            except (OSError, ValueError):
                return False
        delegation_db = state_dir / "delegation" / "tasks.sqlite3"
        if not delegation_db.is_file():
            return True
        try:
            database = sqlite3.connect(
                f"file:{delegation_db}?mode=ro",
                uri=True,
                timeout=2,
            )
            try:
                row = database.execute(
                    "SELECT COUNT(*) FROM tasks "
                    "WHERE status IN ('queued', 'starting', 'running')"
                ).fetchone()
            finally:
                database.close()
        except sqlite3.Error:
            return False
        return row is not None and row[0] == 0

    @staticmethod
    def _role_target(environment: Mapping[str, str]) -> RoleTarget | None:
        role = environment.get("SENPAI_ROLE")
        research_tag = environment.get("RESEARCH_TAG")
        if role not in {"advisor", "student"} or not research_tag:
            return None
        student = environment.get("STUDENT_NAME") if role == "student" else None
        if role == "student" and not student:
            return None
        return RoleTarget(research_tag=research_tag, role=role, student=student)

    def _read_lease(self) -> WorkerLease | None:
        try:
            return WorkerLease.read(self.lease_path)
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    @staticmethod
    def _remember_descendants(
        process: subprocess.Popen[bytes],
        descendants: dict[int, float],
    ) -> None:
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except (OSError, psutil.Error):
            return
        for child in children:
            try:
                descendants[child.pid] = child.create_time()
            except (OSError, psutil.Error):
                continue

    def _terminate_worker(
        self,
        process: subprocess.Popen[bytes],
        descendants: Mapping[int, float],
    ) -> None:
        self._signal_descendants(descendants, signal.SIGTERM)
        terminate_process_group(
            process,
            grace_seconds=self.config.terminate_grace_seconds,
        )
        self._signal_descendants(descendants, signal.SIGKILL)

    @staticmethod
    def _signal_descendants(
        descendants: Mapping[int, float],
        sig: signal.Signals,
    ) -> None:
        for pid, started_at in descendants.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == started_at:
                    process.send_signal(sig)
            except (OSError, psutil.Error):
                continue

    @staticmethod
    def _reap_orphaned_children(worker_pid: int | None) -> None:
        """Reap children adopted by the supervisor when it is container PID 1."""

        if os.getpid() != 1:
            return
        try:
            children = psutil.Process().children()
        except (OSError, psutil.Error):
            return
        for child in children:
            if child.pid == worker_pid:
                continue
            try:
                os.waitpid(child.pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                continue


def lease_is_healthy(path: Path) -> bool:
    try:
        lease = WorkerLease.read(path)
        process = psutil.Process(lease.pid)
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and time.monotonic() <= lease.deadline
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        psutil.Error,
    ):
        return False


def supervisor_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    parser = argparse.ArgumentParser(
        description="Supervise a durable Senpai controller worker."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for role in ("advisor", "student"):
        subparsers.add_parser(role)
    health = subparsers.add_parser("health")
    health.add_argument("lease_path", type=Path)
    args = parser.parse_args(argv)

    if args.command == "health":
        return 0 if lease_is_healthy(args.lease_path) else 1

    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"]).resolve()
    github_token = _consume_github_token(env)
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        return WorkerSupervisor(
            command=(
                sys.executable,
                "-m",
                "senpai_agent.controller",
                args.command,
            ),
            lease_path=state_dir / "controller-lease.json",
            environment=env,
            github_token=github_token,
        ).run(stop)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _consume_github_token(env: Mapping[str, str]) -> SecretStr:
    value = env.get(GITHUB_TOKEN_FILE_ENV)
    if not value:
        raise RuntimeError(f"{GITHUB_TOKEN_FILE_ENV} is required")
    path = Path(value).resolve()
    try:
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise RuntimeError("GitHub token handoff must be a private regular file")
        token = path.read_text(encoding="utf-8").strip()
    finally:
        path.unlink(missing_ok=True)
    if not token:
        raise RuntimeError("GitHub token handoff is empty")
    return SecretStr(token)


if __name__ == "__main__":
    raise SystemExit(supervisor_main())
