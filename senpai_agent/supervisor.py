"""Small process boundary that restarts a crashed or wedged Senpai controller."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import psutil
from pydantic import SecretStr

from senpai_agent.launch_context import render_role_prompt
from senpai_agent.processes import terminate_process_group
from senpai_agent.program_context import (
    PROGRAM_PATH_ENV,
    load_program_system_prompt,
)
from senpai_agent.secrets import (
    GITHUB_TOKEN_FD_ENV,
    GITHUB_TOKEN_FILE_ENV,
    scrub_github_credentials,
)

LEASE_ENV = "SENPAI_CONTROLLER_LEASE_PATH"


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
    llm_request_started_at: float | None = None
    llm_request_heartbeat_at: float | None = None

    @classmethod
    def read(cls, path: Path) -> WorkerLease:
        value = json.loads(path.read_text(encoding="utf-8"))
        lease = cls(
            pid=int(value["pid"]),
            phase=str(value["phase"]),
            deadline=float(value["deadline"]),
            completed_turns=int(value.get("completed_turns", 0)),
            llm_request_started_at=(
                float(value["llm_request_started_at"])
                if value.get("llm_request_started_at") is not None
                else None
            ),
            llm_request_heartbeat_at=(
                float(value["llm_request_heartbeat_at"])
                if value.get("llm_request_heartbeat_at") is not None
                else None
            ),
        )
        if (
            lease.pid <= 0
            or not lease.phase
            or lease.completed_turns < 0
            or (lease.llm_request_started_at is None)
            != (lease.llm_request_heartbeat_at is None)
        ):
            raise ValueError("invalid controller lease")
        return lease


class ProgressLease:
    """Publish the worker's current phase and non-cooperative hard deadline."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.completed_turns = 0
        self._lock = threading.Lock()
        self._phase: str | None = None
        self._deadline: float | None = None
        self._llm_request_started_at: float | None = None
        self._llm_request_heartbeat_at: float | None = None

    def update(
        self,
        phase: str,
        timeout_seconds: float,
        *,
        completed_turn: bool = False,
    ) -> None:
        if not phase or timeout_seconds <= 0:
            raise ValueError("progress phase and timeout must be positive")
        with self._lock:
            if completed_turn:
                self.completed_turns += 1
            self._phase = phase
            self._deadline = time.monotonic() + timeout_seconds
            self._write_locked()

    def update_llm_request(
        self,
        started_at: float | None,
        heartbeat_at: float | None,
    ) -> None:
        if (started_at is None) != (heartbeat_at is None):
            raise ValueError("LLM request timestamps must be set or cleared together")
        with self._lock:
            if self._phase is None or self._deadline is None:
                raise RuntimeError("progress lease must be initialized")
            self._llm_request_started_at = started_at
            self._llm_request_heartbeat_at = heartbeat_at
            self._write_locked()

    def _write_locked(self) -> None:
        assert self._phase is not None
        assert self._deadline is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "phase": self._phase,
                    "deadline": self._deadline,
                    "completed_turns": self.completed_turns,
                    "llm_request_started_at": self._llm_request_started_at,
                    "llm_request_heartbeat_at": self._llm_request_heartbeat_at,
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

    def run(self, stop: threading.Event | None = None) -> int:
        stop = stop or threading.Event()
        failures = 0
        while not stop.is_set():
            self.lease_path.unlink(missing_ok=True)
            environment = {
                **self.environment,
                LEASE_ENV: str(self.lease_path),
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
                reason, made_progress = self._wait_for_worker(
                    process,
                    descendants,
                    stop,
                    started,
                )
            finally:
                self._terminate_worker(process, descendants)
                self._reap_orphaned_children(None)
            if stop.is_set():
                return 0

            runtime = time.monotonic() - started
            failures = 1 if made_progress else failures + 1
            exponential_delay = min(
                self.config.max_backoff_seconds,
                self.config.initial_backoff_seconds * (2 ** min(failures - 1, 16)),
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
            # The dead worker's lease would fail the health probe for the whole
            # backoff; publish the supervisor's own deliberate wait instead.
            ProgressLease(self.lease_path).update(
                "restart-backoff",
                delay + self.config.startup_timeout_seconds,
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
    ) -> tuple[str, bool]:
        made_progress = False
        while not stop.is_set():
            self._remember_descendants(process, descendants)
            self._reap_orphaned_children(process.pid)
            lease = self._read_lease()
            now = time.monotonic()
            if lease is not None and lease.pid == process.pid:
                made_progress = made_progress or lease.completed_turns > 0
                if now > lease.deadline:
                    return f"overdue:{lease.phase}", made_progress
            elif now - started > self.config.startup_timeout_seconds:
                return "startup-timeout", made_progress

            exit_code = process.poll()
            if exit_code is not None:
                final_lease = self._read_lease()
                if final_lease is not None and final_lease.pid == process.pid:
                    made_progress = (
                        made_progress or final_lease.completed_turns > 0
                    )
                return f"exit:{exit_code}", made_progress
            stop.wait(self.config.check_interval_seconds)
        return "shutdown", made_progress

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
    worker_environment = prepare_system_context_environment(
        args.command,
        state_dir,
        env,
    )
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
            environment=worker_environment,
            github_token=github_token,
        ).run(stop)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def prepare_system_context_environment(
    role: Literal["advisor", "student"],
    state_dir: Path,
    env: Mapping[str, str],
) -> dict[str, str]:
    """Snapshot the stable system context before any model process starts."""

    environment = dict(env)
    program = load_program_system_prompt(
        Path(environment["SENPAI_OPENHANDS_WORKSPACE"]),
        environment.get(PROGRAM_PATH_ENV, ""),
    )
    environment[PROGRAM_PATH_ENV] = program.program_path
    role_prompt = state_dir / "system-instructions" / f"{role}.md"
    if role_prompt.exists():
        if not role_prompt.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"persisted role prompt is empty: {role_prompt}")
    else:
        source_value = environment.get("SENPAI_OPENHANDS_ROLE_FILE")
        if not source_value:
            raise RuntimeError(
                "OpenHands role instructions are required; set "
                "SENPAI_OPENHANDS_ROLE_FILE"
            )
        rendered = render_role_prompt(
            Path(source_value).resolve(),
            role,
            environment,
        )
        role_prompt.parent.mkdir(parents=True, exist_ok=True)
        temporary = role_prompt.with_suffix(".tmp")
        temporary.write_text(f"{rendered}\n", encoding="utf-8")
        temporary.replace(role_prompt)
    environment["SENPAI_OPENHANDS_ROLE_FILE"] = str(role_prompt)
    print(
        f"SENPAI_PROGRAM_CONTEXT path={program.program_path}",
        flush=True,
    )
    print(f"SENPAI_ROLE_PROMPT path={role_prompt}", flush=True)
    return environment


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
