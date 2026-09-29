"""Run an ordinary training command in a fresh Kubernetes worker environment."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import select
import signal
import shlex
import socket
import subprocess
import sys
import sysconfig
import time


def mask_output_chunk(data: bytes, secret: bytes) -> tuple[bytes, bytes]:
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


WORKER_LOG_MAX_BYTES = 64 * 1024 * 1024
_METADATA_RESERVE = 1024  # Current JSON and its atomic replacement, each at most 512 bytes.
_MIRROR_LIMIT = 64 * 1024
_TRUNCATED = b"[SENPAI_LOG_TRUNCATED: older worker output was discarded by bounded retention]\n"
_MIRROR_TRUNCATED = b"[SENPAI_LIVE_LOG_TRUNCATED: output consumer stalled; retained logs are on the shared volume]\n"
_MIRROR_BUDGET_REACHED = b"[SENPAI_LIVE_LOG_TRUNCATED: lifetime byte limit reached; recent output remains on the shared volume]\n"
_DRAIN_EXPIRED = b"[SENPAI_LOG_TRUNCATED: inherited output remained open after command exit]\n"


class _WorkerLog:
    def __init__(self, output: Path, environment: dict[str, str]):
        rank = int(environment["NODE_RANK"])
        nodes = int(environment.get("NNODES", "1"))
        budget = int(environment.get("SENPAI_TRAINING_LOG_MAX_BYTES", str(WORKER_LOG_MAX_BYTES)))
        if nodes < 1 or not 0 <= rank < nodes or budget // nodes < 4096:
            raise ValueError("worker logs require a valid node rank and at least 4096 bytes per node")
        directory = output / ".senpai-logs"
        directory.mkdir(exist_ok=True)
        self.path = directory / f"node-{rank}.log"
        self.previous = directory / f"node-{rank}.log.1"
        self.metadata = directory / f"node-{rank}.json"
        self.node_budget = budget // nodes
        self.segment_limit = (self.node_budget - _METADATA_RESERVE) // 2
        self.stream = self.path.open("xb", buffering=0)
        self.size = 0
        self.record = {"node_rank": rank, "nodes": nodes, "run_max_bytes": budget,
                       "segment_max_bytes": self.segment_limit, "truncated": False,
                       "live_output_truncated": False, "complete": False, "exit_code": None}
        self._publish()

    def _publish(self) -> None:
        contents = json.dumps(self.record, separators=(",", ":")).encode() + b"\n"
        assert len(contents) <= _METADATA_RESERVE // 2
        temporary = self.metadata.with_suffix(".tmp")
        temporary.write_bytes(contents)
        temporary.replace(self.metadata)

    def write(self, data: bytes) -> None:
        while data:
            if self.size == self.segment_limit:
                os.fsync(self.stream.fileno())
                self.stream.close()
                self.record["truncated"] |= self.previous.exists()
                self.path.replace(self.previous)
                self.stream = self.path.open("xb", buffering=0)
                self.size = 0
                if self.record["truncated"]:
                    self.stream.write(_TRUNCATED)
                    self.size = len(_TRUNCATED)
                self._publish()
            chunk = data[:self.segment_limit - self.size]
            self.stream.write(chunk)
            self.size += len(chunk)
            data = data[len(chunk):]

    def mirror_truncated(self) -> None:
        if not self.record["live_output_truncated"]:
            self.record["live_output_truncated"] = True
            self._publish()

    def close(self, returncode: int | None) -> None:
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.record.update(complete=returncode is not None, exit_code=returncode)
        self._publish()


def _signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _run_logged_command(argv: list[str], cwd: Path, environment: dict[str, str], output: Path) -> None:
    log = _WorkerLog(output, environment)
    child = None
    pending_signal = None

    def forward(signum, _frame):
        nonlocal pending_signal
        if child is None:
            pending_signal = signum
        else:
            _signal_group(child.pid, signum)

    handlers = {signum: signal.signal(signum, forward)
                for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    mirror_fd = sys.stdout.fileno()
    blocking = os.get_blocking(mirror_fd)
    os.set_blocking(mirror_fd, False)
    mirror = bytearray()
    mirror_open = True
    mirrored_bytes = 0
    mirror_budget_reached = False
    mirror_progress_at = time.monotonic()
    secret = environment.get("WANDB_API_KEY", "").encode()
    pending_secret = b""
    returncode = None

    def emit(data):
        nonlocal mirror_budget_reached, mirror_progress_at
        log.write(data)
        if mirror_open and not mirror_budget_reached:
            if not mirror:
                mirror_progress_at = time.monotonic()
            mirror.extend(data)
            if len(mirror) > _MIRROR_LIMIT:
                mirror[:] = _MIRROR_TRUNCATED + mirror[-(_MIRROR_LIMIT - len(_MIRROR_TRUNCATED)):]
                log.mirror_truncated()
            remaining = log.node_budget - mirrored_bytes
            if len(mirror) > remaining - len(_MIRROR_BUDGET_REACHED):
                mirror[:] = mirror[:remaining - len(_MIRROR_BUDGET_REACHED)] + _MIRROR_BUDGET_REACHED
                mirror_budget_reached = True
                log.mirror_truncated()

    try:
        child = subprocess.Popen(argv, cwd=cwd, env=environment, start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if pending_signal is not None:
            _signal_group(child.pid, pending_signal)
        pipe = child.stdout
        os.set_blocking(pipe.fileno(), False)
        pipe_open = True
        pipe_closed_at = None
        finished_at = None
        terminated_descendants = False
        killed_descendants = False
        while True:
            returncode = child.poll()
            now = time.monotonic()
            if returncode is not None:
                if finished_at is None:
                    finished_at = now
                # Let completed output drain before stopping descendants that
                # keep the pipe open after the command itself has exited.
                if pipe_open and now - finished_at >= .1 and not terminated_descendants:
                    _signal_group(child.pid, signal.SIGTERM)
                    terminated_descendants = True
                elif pipe_open and now - finished_at >= 1 and not killed_descendants:
                    _signal_group(child.pid, signal.SIGKILL)
                    killed_descendants = True
                if pipe_open and now - finished_at >= 1.1:
                    # A detached descendant can escape the process group while
                    # retaining stdout. Pod teardown owns it; do not await EOF.
                    pipe.close()
                    pipe_open = False
                    pipe_closed_at = now
                    log.record["truncated"] = True
                    emit((b"<secret-hidden>" if pending_secret else b"") + _DRAIN_EXPIRED)
                    pending_secret = b""
                if not pipe_open and (not mirror or now - pipe_closed_at >= .1):
                    if mirror:
                        log.mirror_truncated()
                    break
            # Give a draining consumer time to catch up, but keep reading and
            # rotating after a short stall instead of blocking the command.
            readable, writable, _ = select.select(
                [pipe] if pipe_open and (not mirror or now - mirror_progress_at >= .1) else [],
                [mirror_fd] if mirror_open and mirror else [], [], .1,
            )
            if writable:
                try:
                    count = os.write(mirror_fd, mirror)
                    mirrored_bytes += count
                    mirror_progress_at = time.monotonic()
                    del mirror[:count]
                except BlockingIOError:
                    pass
                except BrokenPipeError:
                    mirror_open = False
                    mirror.clear()
                    log.mirror_truncated()
            if readable:
                chunk = os.read(pipe.fileno(), 65536)
                if chunk:
                    masked, pending_secret = mask_output_chunk(pending_secret + chunk, secret)
                    emit(masked)
                else:
                    pipe_open = False
                    pipe_closed_at = time.monotonic()
                    if pending_secret:
                        emit(b"<secret-hidden>")
                        pending_secret = b""
    finally:
        if child is not None:
            if child.poll() is None:
                _signal_group(child.pid, signal.SIGKILL)
            child.wait()
            child.stdout.close()
        os.set_blocking(mirror_fd, blocking)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        log.close(returncode)
    if returncode < 0:
        signum = -returncode
        if signum != signal.SIGKILL:
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    raise SystemExit(returncode)


def run() -> None:
    environment = dict(os.environ)
    if "OMPI_COMM_WORLD_RANK" in environment and not environment.get("SENPAI_TARGET_PYTHON_ENV"):
        # SSH replaces the image environment before starting the MPI daemon.
        environment.update(json.loads((Path.home() / ".senpai-ssh/environment.json").read_text()))
    workspace = Path(environment.get("SENPAI_TRAINING_WORKSPACE", "/workspace")).resolve()
    command = json.loads(base64.b64decode(environment["SENPAI_TRAINING_COMMAND_B64"], validate=True))
    cwd = Path(command["cwd"]).resolve()
    if not cwd.is_relative_to(workspace):
        raise ValueError("training directory must be inside the source checkout")
    output = Path(environment["SENPAI_TRAINING_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    for key in ("WANDB_SERVICE", "WANDB_IDENTITY_TOKEN_FILE", "PYTHONSAFEPATH"):
        environment.pop(key, None)
    environment["NODE_RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
    if target_path := environment.get("SENPAI_TARGET_PYTHON_ENV"):
        # Only the standard Senpai image has an immutable base environment.
        from senpai_agent.target_environment import install_shared_console_scripts
        from senpai_agent.training import target_python_environment

        target_env = Path(target_path)
        environment.update({"UV_CACHE_DIR": str(output.parent / ".uv-cache"), "UV_LINK_MODE": "copy"})
        subprocess.run(
            ["uv", "venv", "--no-project", "--no-config", "--no-python-downloads",
             "--python", sys.executable, str(target_env)],
            check=True, env=environment,
        )
        site = Path(sysconfig.get_path("purelib", vars={"base": str(target_env)}))
        (site / "senpai-runtime.pth").write_text(sysconfig.get_path("purelib") + "\n")
        install_shared_console_scripts(target_env)
        environment.update(target_python_environment(environment))
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(workspace)],
            check=True, env=environment,
        )
    _run_logged_command(command["argv"], cwd, environment, output)


def _private_ssh_key() -> Path:
    # The operator's nonroot Secret mount is readable but not private enough for OpenSSH.
    directory = Path.home() / ".senpai-ssh"
    directory.mkdir(mode=0o700)
    key = directory / "id_rsa"
    with key.open("xb") as output:
        key.chmod(0o600)
        output.write((Path.home() / ".ssh/id_rsa").read_bytes())
    return key


def mpi() -> None:
    """Start one command per worker node using the operator's hostfile and key."""
    home = Path.home()
    private_key = _private_ssh_key()
    public_key = (home / ".ssh/id_rsa.pub").read_text().strip()
    hosts = [line.split()[0] for line in Path("/etc/mpi/hostfile").read_text().splitlines() if line]
    known_hosts = home / ".senpai-known-hosts"
    known_hosts.write_text("".join(f"[{host}]:2222 {public_key}\n" for host in hosts))
    # A suspended MPIJob can create its launcher before any workers exist.
    # The executor's workload deadline also bounds this readiness wait.
    for host in hosts:
        print(f"Waiting for worker SSH on {host}:2222", flush=True)
        while True:
            try:
                with socket.create_connection((host, 2222), timeout=2):
                    break
            except OSError:
                time.sleep(1)
    ssh_arguments = shlex.join([
        "-p", "2222", "-o", "StrictHostKeyChecking=yes", "-o",
        f"UserKnownHostsFile={known_hosts}", "-i", str(private_key),
    ])
    argv = [
        "/usr/bin/mpirun", "--hostfile", "/etc/mpi/hostfile",
        "-np", os.environ["NNODES"], "--map-by", "ppr:1:node", "--bind-to", "none",
        "--prefix", "/usr", "--prtemca", "plm_ssh_args", ssh_arguments,
        "--prtemca", "plm_ssh_no_tree_spawn", "1",
    ]
    for name in (
        "NNODES", "GPUS_PER_NODE", "MASTER_ADDR", "MASTER_PORT",
        "SENPAI_TRAINING_COMMAND_B64", "SENPAI_TRAINING_WORKSPACE",
        "SENPAI_TARGET_PYTHON_ENV", "SENPAI_TRAINING_OUTPUT_DIR",
        "SENPAI_TRAINING_LOG_MAX_BYTES",
        "WANDB_API_KEY", "WANDB_RUN_ID", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_TAGS",
    ):
        if name in os.environ:
            argv.extend(("-x", name))
    if os.environ.get("SENPAI_TARGET_PYTHON_ENV"):
        for name in ("PATH", "LD_LIBRARY_PATH"):
            if name in os.environ:
                argv.extend(("-x", name))
        argv.extend((sys.executable, "-P", "-m", "senpai_agent.training_worker", "run"))
    else:
        argv.append("/home/senpai/.senpai-run")
    os.execv(argv[0], argv)


def sshd() -> None:
    home = Path.home()
    private_key = _private_ssh_key()
    if not os.environ.get("SENPAI_TARGET_PYTHON_ENV"):
        environment_path = private_key.parent / "environment.json"
        with environment_path.open("x") as output:
            environment_path.chmod(0o600)
            json.dump(dict(os.environ), output)
        wrapper = home / ".senpai-run"
        bootstrap = "".join(
            f"export {name}={shlex.quote(os.environ[name])}\n"
            for name in ("PATH", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH")
            if name in os.environ
        )
        wrapper.write_text("#!/bin/sh\n" + bootstrap + "exec " + shlex.join([
            sys.executable, str(Path(__file__).resolve()), "run",
        ]) + "\n")
        wrapper.chmod(0o700)
    config = home / ".senpai-sshd-config"
    config.write_text(
        "Port 2222\n"
        f"PidFile {home}/.senpai-sshd.pid\n"
        f"HostKey {private_key}\n"
        f"AuthorizedKeysFile {home}/.ssh/authorized_keys\n"
        "StrictModes no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n"
        "UsePAM no\nAllowUsers senpai\n"
    )
    os.execv("/usr/sbin/sshd", ["/usr/sbin/sshd", "-D", "-e", "-f", str(config)])


if __name__ == "__main__":
    {"run": run, "mpi": mpi, "sshd": sshd}[sys.argv[1]]()
