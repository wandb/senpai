"""One-shot process boundary for a credential-bearing Senpai controller."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import psutil
from pydantic import SecretStr

from senpai_agent.agent_markdown import read_agent_markdown
from senpai_agent.launch_context import (
    LAUNCH_CONTEXT_ENV,
    decode_launch_context,
    render_role_prompt,
)
from senpai_agent.processes import terminate_process_group
from senpai_agent.program_context import (
    PROGRAM_CONTEXT_FILE_ENV,
    PROGRAM_PATH_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    decode_program_system_prompt,
)
from senpai_agent.secrets import (
    GITHUB_TOKEN_FD_ENV,
    GITHUB_TOKEN_FILE_ENV,
    PRIVATE_CREDENTIAL_FD_ENVS,
    PRIVATE_CREDENTIAL_FILE_ENVS,
    SERVICE_CREDENTIAL_ENV_NAMES,
    WANDB_TRAINING_API_KEY_ENV,
    scrub_github_credentials,
    scrub_service_credentials,
    set_process_nondumpable,
)
from senpai_agent.system_instructions import (
    SYSTEM_INSTRUCTIONS_FILE_ENV,
    SYSTEM_INSTRUCTIONS_SHA256_ENV,
    SenpaiSystemInstructions,
    decode_system_instructions,
    encode_system_instructions,
)

LEASE_ENV = "SENPAI_CONTROLLER_LEASE_PATH"
HEALTH_PORT_ENV = "SENPAI_HEALTH_PORT"
DEFAULT_HEALTH_PORT = 8080


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    startup_timeout_seconds: float = 300
    check_interval_seconds: float = 5
    terminate_grace_seconds: float = 60

    def __post_init__(self) -> None:
        if (
            min(
                self.startup_timeout_seconds,
                self.check_interval_seconds,
                self.terminate_grace_seconds,
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
    """Run one controller worker and let the container runtime restart it."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        lease_path: Path,
        config: SupervisorConfig | None = None,
        environment: Mapping[str, str] | None = None,
        github_token: SecretStr | None = None,
        private_credentials: Mapping[str, SecretStr] | None = None,
    ):
        if not command:
            raise ValueError("worker command must not be empty")
        self.command = tuple(command)
        self.lease_path = lease_path.resolve()
        self.config = config or SupervisorConfig()
        self.environment = dict(os.environ if environment is None else environment)
        scrub_github_credentials(self.environment)
        scrub_service_credentials(self.environment)
        self.github_token = github_token
        self.private_credentials = dict(private_credentials or {})

    def run(self, stop: threading.Event | None = None) -> int:
        stop = stop or threading.Event()
        if stop.is_set():
            self._forget_credentials()
            return 0
        self.lease_path.unlink(missing_ok=True)
        environment = {
            **self.environment,
            LEASE_ENV: str(self.lease_path),
        }
        credential_fds = self._open_credential_pipes()
        for env_name, descriptor in credential_fds.items():
            environment[env_name] = str(descriptor)
        try:
            process = subprocess.Popen(
                self.command,
                env=environment,
                start_new_session=True,
                pass_fds=tuple(credential_fds.values()),
            )
        finally:
            for descriptor in credential_fds.values():
                os.close(descriptor)
            self._forget_credentials()

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
            self._terminate_adopted_children()
            self._reap_orphaned_children(None)
        if stop.is_set():
            return 0

        print(
            "SENPAI_CONTROLLER_EXIT "
            f"reason={reason} runtime_seconds={time.monotonic() - started:.1f} "
            f"completed_turn={str(made_progress).lower()}",
            file=sys.stderr,
            flush=True,
        )
        if reason.startswith("exit:"):
            exit_code = int(reason.partition(":")[2])
            return exit_code or 1
        return 1

    def _forget_credentials(self) -> None:
        self.github_token = None
        self.private_credentials.clear()

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

    def _open_credential_pipes(self) -> dict[str, int]:
        descriptors: dict[str, int] = {}
        if (github_fd := self._open_github_token_pipe()) is not None:
            descriptors[GITHUB_TOKEN_FD_ENV] = github_fd
        for credential_name, credential in self.private_credentials.items():
            read_fd, write_fd = os.pipe()
            try:
                os.write(write_fd, credential.get_secret_value().encode())
            except BaseException:
                os.close(read_fd)
                raise
            finally:
                os.close(write_fd)
            descriptors[PRIVATE_CREDENTIAL_FD_ENVS[credential_name]] = read_fd
        return descriptors

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

    def _terminate_adopted_children(self) -> None:
        """Stop detached descendants before container PID 1 exits."""

        if os.getpid() != 1:
            return
        try:
            children = psutil.Process().children(recursive=True)
        except (OSError, psutil.Error):
            return
        for child in children:
            try:
                child.terminate()
            except (OSError, psutil.Error):
                continue
        _, alive = psutil.wait_procs(
            children,
            timeout=self.config.terminate_grace_seconds,
        )
        for child in alive:
            try:
                child.kill()
            except (OSError, psutil.Error):
                continue
        psutil.wait_procs(alive, timeout=self.config.terminate_grace_seconds)

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


@contextmanager
def serve_lease_health(
    lease_path: Path,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_HEALTH_PORT,
) -> Iterator[ThreadingHTTPServer]:
    """Serve the worker lease without spawning credential-bearing probes."""

    class LeaseHealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self.send_error(404)
                return
            healthy = lease_is_healthy(lease_path)
            body = b"ok\n" if healthy else b"unhealthy\n"
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), LeaseHealthHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="senpai-health",
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def supervisor_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run one Senpai controller worker per container start."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for role in ("advisor", "student"):
        subparsers.add_parser(role)
    health = subparsers.add_parser("health")
    health.add_argument("lease_path", type=Path)
    args = parser.parse_args(argv)

    if args.command == "health":
        return 0 if lease_is_healthy(args.lease_path) else 1

    set_process_nondumpable()
    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"]).resolve()
    worker_environment = prepare_system_context_environment(
        args.command,
        state_dir,
        env,
    )
    github_token = _consume_github_token(env)
    private_credentials = _consume_private_credential_files(env)
    _require_private_credentials(args.command, private_credentials)
    try:
        health_port = int(env.get(HEALTH_PORT_ENV, str(DEFAULT_HEALTH_PORT)))
    except ValueError as error:
        raise RuntimeError(f"{HEALTH_PORT_ENV} must be an integer") from error
    if not 1 <= health_port <= 65535:
        raise RuntimeError(f"{HEALTH_PORT_ENV} must be between 1 and 65535")
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        lease_path = state_dir / "controller-lease.json"
        with serve_lease_health(lease_path, port=health_port):
            worker = WorkerSupervisor(
                command=(
                    sys.executable,
                    "-P",
                    "-m",
                    "senpai_agent.controller",
                    args.command,
                ),
                lease_path=lease_path,
                environment=worker_environment,
                github_token=github_token,
                private_credentials=private_credentials,
            )
            github_token = None
            private_credentials.clear()
            return worker.run(stop)
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
    source_commit = environment.get(PROGRAM_SOURCE_COMMIT_ENV)
    if not source_commit:
        raise RuntimeError(f"{PROGRAM_SOURCE_COMMIT_ENV} is required")
    program_file = environment.get(PROGRAM_CONTEXT_FILE_ENV)
    if not program_file:
        raise RuntimeError(f"{PROGRAM_CONTEXT_FILE_ENV} is required")
    program = decode_program_system_prompt(
        Path(program_file).read_text(encoding="utf-8").strip()
    )
    if source_commit != program.source_commit:
        raise RuntimeError(
            f"{PROGRAM_SOURCE_COMMIT_ENV} does not match the launch snapshot"
        )
    configured_program_path = environment.get(PROGRAM_PATH_ENV, "")
    if configured_program_path != program.program_path:
        raise RuntimeError(f"{PROGRAM_PATH_ENV} does not match the launch snapshot")
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
    role_prompt = state_dir / "system-instructions" / f"{role}.md"
    if role_prompt.exists():
        if role_prompt.read_text(encoding="utf-8").strip() != rendered:
            raise RuntimeError(
                "persisted role prompt does not match the controller-rendered "
                f"snapshot: {role_prompt}"
            )
    else:
        role_prompt.parent.mkdir(parents=True, exist_ok=True)
        temporary = role_prompt.with_suffix(".tmp")
        temporary.write_text(f"{rendered}\n", encoding="utf-8")
        temporary.replace(role_prompt)
    environment["SENPAI_OPENHANDS_ROLE_FILE"] = str(role_prompt)
    harness_value = environment.get("SENPAI_OPENHANDS_HARNESS_FILE")
    if not harness_value:
        raise RuntimeError(
            "OpenHands harness instructions are required; set "
            "SENPAI_OPENHANDS_HARNESS_FILE"
        )
    harness = read_agent_markdown(Path(harness_value).resolve()).strip()
    if not harness:
        raise RuntimeError("OpenHands harness instructions must not be empty")
    instructions = SenpaiSystemInstructions(
        harness=harness,
        role=rendered,
        program=program,
        launch=decode_launch_context(environment.get(LAUNCH_CONTEXT_ENV, "")),
    )
    encoded_instructions = encode_system_instructions(instructions)
    system_context = role_prompt.with_suffix(".context.b64")
    if system_context.exists():
        persisted = decode_system_instructions(
            system_context.read_text(encoding="utf-8").strip(),
            instructions.content_sha256,
        )
        if persisted != instructions:
            raise RuntimeError(
                "persisted system context does not match the controller snapshot"
            )
    else:
        temporary = system_context.with_suffix(".tmp")
        temporary.write_text(f"{encoded_instructions}\n", encoding="utf-8")
        temporary.replace(system_context)
    environment[SYSTEM_INSTRUCTIONS_FILE_ENV] = str(system_context)
    environment[SYSTEM_INSTRUCTIONS_SHA256_ENV] = instructions.content_sha256
    print(
        "SENPAI_PROGRAM_CONTEXT "
        f"path={program.program_path} commit={program.source_commit} "
        f"sha256={program.content_sha256}",
        flush=True,
    )
    print(
        f"SENPAI_SYSTEM_CONTEXT path={system_context} "
        f"sha256={instructions.content_sha256}",
        flush=True,
    )
    return environment


def _consume_github_token(env: Mapping[str, str]) -> SecretStr:
    value = env.get(GITHUB_TOKEN_FILE_ENV)
    if not value:
        raise RuntimeError(f"{GITHUB_TOKEN_FILE_ENV} is required")
    token = _consume_private_file(
        Path(value),
        "GitHub token handoff",
    )
    if not token:
        raise RuntimeError("GitHub token handoff is empty")
    return SecretStr(token)


def _consume_private_credential_files(
    env: Mapping[str, str],
) -> dict[str, SecretStr]:
    credentials: dict[str, SecretStr] = {}
    for credential_name, file_env in PRIVATE_CREDENTIAL_FILE_ENVS.items():
        file_value = env.get(file_env)
        if not file_value:
            continue
        value = _consume_private_file(Path(file_value), file_env)
        if value:
            credentials[credential_name] = SecretStr(value)
    return credentials


def _consume_private_file(path: Path, label: str) -> str:
    """Read and unlink an owner-only handoff without following symlinks."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise RuntimeError(f"{label} must be an owner-only regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.geteuid()
        ):
            raise RuntimeError(f"{label} must be an owner-only regular file")
        with os.fdopen(descriptor, encoding="utf-8") as source:
            descriptor = -1
            return source.read().strip()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _require_private_credentials(
    role: Literal["advisor", "student"],
    credentials: Mapping[str, SecretStr],
) -> None:
    required = set(SERVICE_CREDENTIAL_ENV_NAMES)
    if role == "student":
        required.add(WANDB_TRAINING_API_KEY_ENV)
    if missing := sorted(required - credentials.keys()):
        file_names = [PRIVATE_CREDENTIAL_FILE_ENVS[name] for name in missing]
        raise RuntimeError(
            "missing private credential handoff files: " + ", ".join(file_names)
        )


if __name__ == "__main__":
    raise SystemExit(supervisor_main())
