"""Run an ordinary training command in a fresh Kubernetes worker environment."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import sysconfig
import time

from senpai_agent.target_environment import install_shared_console_scripts
from senpai_agent.training import target_python_environment


def run() -> None:
    workspace = Path(os.environ.get("SENPAI_TRAINING_WORKSPACE", "/workspace")).resolve()
    command = json.loads(base64.b64decode(os.environ["SENPAI_TRAINING_COMMAND_B64"], validate=True))
    cwd = Path(command["cwd"]).resolve()
    if not cwd.is_relative_to(workspace):
        raise ValueError("training directory must be inside the source checkout")
    target_env = Path(os.environ.get(
        "SENPAI_TARGET_PYTHON_ENV", "/home/senpai/.venvs/senpai-target",
    ))
    output = Path(os.environ["SENPAI_TRAINING_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for key in ("WANDB_SERVICE", "WANDB_IDENTITY_TOKEN_FILE", "PYTHONSAFEPATH"):
        environment.pop(key, None)
    environment.update({
        "SENPAI_TARGET_PYTHON_ENV": str(target_env),
        "UV_CACHE_DIR": str(output.parent / ".uv-cache"),
        "UV_LINK_MODE": "copy",
        "NODE_RANK": environment.get("OMPI_COMM_WORLD_RANK", "0"),
    })
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
    os.chdir(cwd)
    os.execvpe(command["argv"][0], command["argv"], environment)


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
        "--prefix", "/usr", "--mca", "plm_rsh_args", ssh_arguments,
        "--mca", "plm_rsh_no_tree_spawn", "1",
    ]
    for name in (
        "PATH", "LD_LIBRARY_PATH", "NNODES", "GPUS_PER_NODE", "MASTER_ADDR", "MASTER_PORT",
        "SENPAI_TRAINING_COMMAND_B64", "SENPAI_TRAINING_WORKSPACE",
        "SENPAI_TARGET_PYTHON_ENV", "SENPAI_TRAINING_OUTPUT_DIR",
        "WANDB_API_KEY", "WANDB_RUN_ID", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_TAGS",
    ):
        if name in os.environ:
            argv.extend(("-x", name))
    argv.extend((sys.executable, "-P", "-m", "senpai_agent.training_worker", "run"))
    os.execv(argv[0], argv)


def sshd() -> None:
    home = Path.home()
    private_key = _private_ssh_key()
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
