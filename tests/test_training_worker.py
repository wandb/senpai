from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def worker_run(tmp_path):
    workspace = tmp_path / "target workspace"
    cwd = workspace / "experiment"
    cwd.mkdir(parents=True)
    (cwd / "target_local.py").write_text("value = 'local target import'\n")
    home = tmp_path / "home"
    home.mkdir()
    target_environment = home / ".venvs" / "target"
    output = tmp_path / "runs" / "run-one"
    environment = {
        **os.environ,
        "HOME": str(home),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "PYTHONSAFEPATH": "1",
        "WANDB_SERVICE": "stale-controller-service",
        "WANDB_IDENTITY_TOKEN_FILE": "/controller-only/token",
        "SENPAI_TRAINING_WORKSPACE": str(workspace),
        "SENPAI_TARGET_PYTHON_ENV": str(target_environment),
        "SENPAI_TRAINING_OUTPUT_DIR": str(output),
        "NNODES": "2",
        "GPUS_PER_NODE": "1",
        "MASTER_ADDR": "worker-zero",
        "MASTER_PORT": "29500",
        "OMPI_COMM_WORLD_RANK": "1",
        "UV_PYTHON_DOWNLOADS": "never",
    }

    def run(argv):
        environment["SENPAI_TRAINING_COMMAND_B64"] = base64.b64encode(
            json.dumps({"argv": argv, "cwd": str(cwd)}).encode()
        ).decode()
        result = subprocess.run(
            [sys.executable, "-P", "-m", "senpai_agent.training_worker", "run"],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        return result, workspace, cwd, target_environment, output

    return run


def test_worker_executes_literal_command_in_writable_target_environment(worker_run):
    command = """
import json, os, pathlib, subprocess, sys, sysconfig
import target_local, yaml

base_yaml = yaml.__file__
site_packages = pathlib.Path(sysconfig.get_path('purelib'))
(site_packages / 'yaml.py').write_text('target_only = True\\n')
subprocess.run([sys.executable, '-c', 'import yaml; assert yaml.target_only'], check=True)
pathlib.Path('target_local.py').write_text("value = 'edited'\\n")
record = {
    'argv': sys.argv[1:], 'cwd': os.getcwd(), 'prefix': sys.prefix,
    'base_yaml': base_yaml, 'target_import': target_local.value,
    'safe_path': sys.flags.safe_path,
    'environment': {name: os.environ.get(name) for name in (
        'PATH', 'UV_PYTHON', 'UV_PROJECT_ENVIRONMENT', 'VIRTUAL_ENV',
        'WANDB_SERVICE', 'WANDB_IDENTITY_TOKEN_FILE', 'PYTHONSAFEPATH',
        'NODE_RANK', 'NNODES', 'GPUS_PER_NODE', 'MASTER_ADDR', 'MASTER_PORT',
        'UV_CACHE_DIR', 'UV_LINK_MODE',
    )},
    'git_trust': subprocess.check_output(
        ['git', 'config', '--global', '--get-all', 'safe.directory'], text=True,
    ).splitlines(),
}
pathlib.Path(os.environ['SENPAI_TRAINING_OUTPUT_DIR'], 'observed.json').write_text(json.dumps(record))
"""
    literal_arguments = ["spaces stay together", "$(touch must-not-exist)", "; exit 99", "*"]
    result, workspace, cwd, target_environment, output = worker_run(
        ["python", "-c", command, *literal_arguments]
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads((output / "observed.json").read_text())
    assert observed["argv"] == literal_arguments
    assert observed["cwd"] == str(cwd)
    assert observed["prefix"] == str(target_environment)
    assert observed["base_yaml"] == yaml.__file__
    assert observed["target_import"] == "local target import"
    assert observed["safe_path"] is False
    assert observed["git_trust"] == [str(workspace)]
    environment = observed["environment"]
    assert environment.pop("PATH").split(os.pathsep)[0] == str(target_environment / "bin")
    assert environment == {
        "UV_PYTHON": str(target_environment / "bin" / "python"),
        "UV_PROJECT_ENVIRONMENT": str(target_environment),
        "VIRTUAL_ENV": str(target_environment),
        "WANDB_SERVICE": None, "WANDB_IDENTITY_TOKEN_FILE": None, "PYTHONSAFEPATH": None,
        "NODE_RANK": "1", "NNODES": "2", "GPUS_PER_NODE": "1",
        "MASTER_ADDR": "worker-zero", "MASTER_PORT": "29500",
        "UV_CACHE_DIR": str(output.parent / ".uv-cache"), "UV_LINK_MODE": "copy",
    }
    assert (cwd / "target_local.py").read_text() == "value = 'edited'\n"
    assert not (cwd / "must-not-exist").exists()
    unchanged = subprocess.check_output(
        [sys.executable, "-P", "-c", "import yaml; print(yaml.__file__)"], text=True,
    ).strip()
    assert unchanged == yaml.__file__


def test_worker_propagates_the_training_command_exit_status(worker_run):
    result, *_ = worker_run(["python", "-c", "raise SystemExit(37)"])

    assert result.returncode == 37, result.stderr


def test_worker_ssh_server_starts_with_operator_key_permissions(tmp_path):
    import socket
    import time
    if not Path("/usr/sbin/sshd").exists():
        pytest.skip("OpenSSH server is unavailable")
    home = tmp_path / "home"
    operator_keys = home / ".ssh"
    operator_keys.mkdir(parents=True)
    key = operator_keys / "id_rsa"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ecdsa", "-b", "256", "-N", "", "-f", str(key)],
        check=True, timeout=10,
    )
    key.chmod(0o644)  # MPI operator's nonroot Secret volume permissions.
    (operator_keys / "authorized_keys").write_bytes(key.with_suffix(".pub").read_bytes())
    with socket.socket() as port_reservation:
        port_reservation.bind(("127.0.0.1", 0))
        port = port_reservation.getsockname()[1]
    # Only redirect the production SSH listener to a temporary loopback port.
    check_startup = """
import os, runpy, sys
execv = os.execv
os.execv = lambda executable, argv: execv(executable, [
    *argv, '-o', 'Port=' + os.environ['TEST_SSH_PORT'], '-o', 'ListenAddress=127.0.0.1',
])
sys.argv = ['senpai_agent.training_worker', 'sshd']
runpy.run_module('senpai_agent.training_worker', run_name='__main__')
"""
    server = subprocess.Popen(
        [sys.executable, "-P", "-c", check_startup],
        env={**os.environ, "HOME": str(home), "TEST_SSH_PORT": str(port),
             "SENPAI_TARGET_PYTHON_ENV": str(home / ".venvs/target"),
             "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server.poll() is None:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1) as connection:
                    connection.settimeout(5)
                    assert connection.recv(256).startswith(b"SSH-2.0-")
                    return
            except ConnectionRefusedError:
                time.sleep(0.02)
        server.terminate()
        _, stderr = server.communicate(timeout=3)
        pytest.fail(f"worker SSH server failed to start: {stderr}")
    finally:
        if server.poll() is None:
            server.terminate()
        server.communicate(timeout=3)


@pytest.mark.parametrize("custom", [False, True])
def test_mpi_launch_keeps_ssh_on_launcher_with_its_known_hosts(tmp_path, monkeypatch, custom):
    import contextlib
    import socket

    from senpai_agent import training_worker

    home = tmp_path / "home"
    keys = home / ".ssh"
    keys.mkdir(parents=True)
    (keys / "id_rsa").write_text("operator private key")
    (keys / "id_rsa.pub").write_text("ssh-rsa operator-public-key\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NNODES", "128")
    monkeypatch.setenv("SENPAI_TARGET_PYTHON_ENV", "" if custom else str(home / ".venvs/target"))
    read_text = Path.read_text

    def operator_hostfile(path, *args, **kwargs):
        if path == Path("/etc/mpi/hostfile"):
            return "".join(f"worker-{rank}.job slots=8\n" for rank in range(128))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", operator_hostfile)
    ready_hosts = set()
    attempts = {}

    @contextlib.contextmanager
    def connect(address, timeout):
        host, port = address
        assert port == 2222
        assert timeout > 0
        attempts[host] = attempts.get(host, 0) + 1
        if host == "worker-0.job" and attempts[host] == 1:
            raise socket.gaierror(socket.EAI_NONAME, "worker DNS is not published yet")
        if host == "worker-1.job" and attempts[host] == 1:
            raise ConnectionRefusedError("worker SSH has not started yet")
        ready_hosts.add(host)
        yield

    monkeypatch.setattr(socket, "create_connection", connect)
    invocations = []

    def launch(executable, argv):
        assert ready_hosts == {f"worker-{rank}.job" for rank in range(128)}
        invocations.append((executable, argv))

    monkeypatch.setattr(os, "execv", launch)
    training_worker.mpi()
    executable, argv = invocations.pop()
    assert executable == "/usr/bin/mpirun"
    parameters = {argv[index + 1]: argv[index + 2] for index, value in enumerate(argv) if value == "--prtemca"}
    # OpenMPI5's default tree launch requires worker-to-worker SSH. Only the
    # launcher owns the complete known_hosts file, so all SSH starts belong here.
    assert parameters.get("plm_ssh_no_tree_spawn") == "1"
    assert "2222" in parameters["plm_ssh_args"]
    assert argv[argv.index("-np") + 1] == "128"
    assert attempts["worker-0.job"] == attempts["worker-1.job"] == 2
    exports = [argv[index + 1] for index, value in enumerate(argv) if value == "-x"]
    assert ("PATH" in exports) is not custom
    assert argv[-1] == ("/home/senpai/.senpai-run" if custom else "run")


@pytest.mark.parametrize("through_ssh", [False, True])
def test_custom_image_runs_without_senpai_git_or_dependency_tools(tmp_path, through_ssh):
    import venv

    image = tmp_path / "custom-python"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(image)
    python = image / "bin" / "python3"
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    output = tmp_path / "shared-output"
    script = tmp_path / "runtime" / "worker.py"
    script.parent.mkdir()
    script.write_bytes((Path(__file__).resolve().parents[1] / "senpai_agent/training_worker.py").read_bytes())
    command = """
import json, os, pathlib, sys
record = {'prefix': sys.prefix, 'path': os.environ['PATH'], 'cwd': os.getcwd(),
          'library_path': os.environ['LD_LIBRARY_PATH'], 'rank': os.environ['NODE_RANK'],
          'argv': sys.argv[1:], 'safe_path': sys.flags.safe_path}
pathlib.Path(os.environ['SENPAI_TRAINING_OUTPUT_DIR'], 'custom.json').write_text(json.dumps(record))
"""
    environment = {
        "PATH": str(image / "bin"), "HOME": str(tmp_path), "LD_LIBRARY_PATH": "/custom/lib",
        "SENPAI_TRAINING_WORKSPACE": str(workspace),
        "SENPAI_TRAINING_OUTPUT_DIR": str(output),
        "SENPAI_TRAINING_COMMAND_B64": base64.b64encode(json.dumps({
            "argv": ["python3", "-c", command, "literal ; argument"], "cwd": str(workspace),
        }).encode()).decode(),
    }
    invocation = [str(python), "-I", str(script), "run"]
    if through_ssh:
        keys = tmp_path / ".ssh"
        keys.mkdir()
        (keys / "id_rsa").write_text("operator key")
        # Stop only at the SSH daemon exec boundary; exercise its real bootstrap.
        bootstrap = "import os,runpy,sys; os.execv=lambda *args: None; sys.argv=[sys.argv[1], 'sshd']; runpy.run_path(sys.argv[0], run_name='__main__')"
        subprocess.run([str(python), "-I", "-c", bootstrap, str(script)],
                       env=environment, check=True, capture_output=True, text=True, timeout=30)
        invocation = [str(tmp_path / ".senpai-run")]
        environment = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "OMPI_COMM_WORLD_RANK": "1"}
    result = subprocess.run(invocation, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "custom.json").read_text()) == {
        "prefix": str(image), "path": str(image / "bin"), "cwd": str(workspace),
        "library_path": "/custom/lib", "rank": "1" if through_ssh else "0", "argv": ["literal ; argument"],
        "safe_path": False,
    }
