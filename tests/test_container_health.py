import json
import os
import socket
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from launch_test_support import render_role

ROOT = Path(__file__).resolve().parents[1]
HEALTH_SCRIPT = ROOT / "scripts" / "senpai-container-health.sh"


def _fake_python(tmp_path: Path, exit_code: int) -> tuple[Path, dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    invocation = tmp_path / "python-invoked"
    python = fake_bin / "python"
    python.write_text(
        f"#!/bin/sh\nprintf invoked > {invocation}\nexit {exit_code}\n"
    )
    python.chmod(0o755)
    return invocation, {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SENPAI_PYTHON": str(python),
    }


def test_container_health_allows_slow_bootstrap_before_lease_exists(tmp_path: Path):
    started = tmp_path / "bootstrap-started"
    failures = tmp_path / "health-failures"
    started.write_text(str(int(time.time())))
    invocation, environment = _fake_python(tmp_path, 1)
    environment.update(
        {
            "SENPAI_BOOTSTRAP_STARTED_PATH": str(started),
            "SENPAI_BOOTSTRAP_GRACE_SECONDS": "600",
            "SENPAI_HEALTH_FAILURES_PATH": str(failures),
        }
    )

    result = subprocess.run(
        ["sh", str(HEALTH_SCRIPT), str(tmp_path / "missing-lease.json")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert not invocation.exists()
    assert not failures.exists()


def test_container_health_honors_retries_before_terminating(tmp_path: Path):
    started = tmp_path / "bootstrap-started"
    failures = tmp_path / "health-failures"
    started.write_text("1")
    _invocation, environment = _fake_python(tmp_path, 1)
    environment.update(
        {
            "SENPAI_BOOTSTRAP_STARTED_PATH": str(started),
            "SENPAI_BOOTSTRAP_GRACE_SECONDS": "1",
            "SENPAI_HEALTH_FAILURES_PATH": str(failures),
            "SENPAI_HEALTH_FAILURE_THRESHOLD": "5",
        }
    )

    for expected in range(1, 5):
        result = subprocess.run(
            ["sh", str(HEALTH_SCRIPT), str(tmp_path / "missing-lease.json")],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert failures.read_text().strip() == str(expected)
        assert f"health failure {expected}/5" in result.stderr


@pytest.mark.parametrize("surface", [
    "container", "student-startupProbe", "student-livenessProbe",
    "advisor-startupProbe", "advisor-livenessProbe",
])
def test_health_uses_trusted_python_from_writable_workdir(tmp_path: Path, surface):
    started = tmp_path / "bootstrap-started"
    failures = tmp_path / "health-failures"
    started.write_text("1")
    failures.write_text("4")
    invocation, environment = _fake_python(tmp_path, 99)
    hostile_package = tmp_path / "senpai_agent"
    hostile_package.mkdir()
    module_invoked = tmp_path / "module-invoked"
    (hostile_package / "__init__.py").write_text(
        f"from pathlib import Path; Path({str(module_invoked)!r}).touch(); raise SystemExit(99)\n"
    )
    role = surface.split("-", 1)[0]
    state_dir = tmp_path / "probe/advisor" if role == "advisor" else tmp_path
    lease = state_dir / "openhands_state/controller-lease.json"
    lease.parent.mkdir(parents=True)
    lease.write_text(json.dumps({
        "pid": os.getpid(), "phase": "test", "deadline": time.monotonic() + 60,
    }))
    environment.update(
        {
            "SENPAI_BOOTSTRAP_STARTED_PATH": str(started),
            "SENPAI_BOOTSTRAP_GRACE_SECONDS": "1",
            "SENPAI_HEALTH_FAILURES_PATH": str(failures),
            "SENPAI_HEALTH_FAILURE_THRESHOLD": "100",
            "SENPAI_PYTHON": sys.executable,
            "PYTHONPATH": str(ROOT),
            "RESEARCH_TAG": "probe",
        }
    )
    environment.pop("PYTHONSAFEPATH", None)
    command = ["sh", str(HEALTH_SCRIPT), str(lease)]
    if surface != "container":
        _, manifest, _ = render_role(role)
        container = yaml.safe_load(manifest)["spec"]["template"]["spec"]["containers"][0]
        command = container[surface.split("-", 1)[1]]["exec"]["command"]
        command = [item.replace("/var/lib/senpai", str(tmp_path)) for item in command]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not invocation.exists()
    assert not module_invoked.exists()
    if surface == "container":
        assert not failures.exists()


def test_role_images_use_the_bootstrap_aware_health_wrapper():
    for name in ("Dockerfile.advisor", "Dockerfile.student"):
        dockerfile = (ROOT / name).read_text()
        assert "CMD senpai-container-health" in dockerfile
        assert "|| kill -TERM 1" not in dockerfile
    for name in ("entrypoint-advisor.sh", "entrypoint-student.sh"):
        entrypoint = (ROOT / "k8s" / name).read_text()
        assert entrypoint.index(".bootstrap-started") < entrypoint.index("git clone")


def test_role_entrypoints_default_openhands_turns_to_two_hours_of_inactivity():
    for name in ("entrypoint-advisor.sh", "entrypoint-student.sh"):
        entrypoint = (ROOT / "k8s" / name).read_text()
        assert 'SENPAI_OPENHANDS_TIMEOUT_SECONDS:-7200' in entrypoint


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_role_startup_isolates_target_uv_commands_from_agent_environment(
    tmp_path: Path, monkeypatch, role: str
):
    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text()
    startup = entrypoint[entrypoint.index("export IS_SANDBOX=1"):]
    uv = shutil.which("uv")
    assert uv is not None, "the bootstrap contract requires uv"
    startup = startup.replace("/usr/local/bin/uv", shlex.quote(uv))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").write_text("#!/bin/sh\nexit 99\n")
    (fake_bin / "python").chmod(0o755)
    python = tmp_path / "runner-python"
    # Execute real bootstrap probes and setup, then record the supervisor handoff.
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:4] != ['-P', '-m', 'senpai_agent.supervisor']:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "print(json.dumps({'args': sys.argv[1:], 'environment': "
        "{key: os.environ.get(key) for key in "
        "('UV_PROJECT_ENVIRONMENT', 'UV_PYTHON', 'VIRTUAL_ENV', 'SENPAI_PYTHON')}}))\n"
    )
    python.chmod(0o755)

    executor_socket = tmp_path / "executor.sock"
    monkeypatch.chdir(tmp_path)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(executor_socket.name)
    completed = subprocess.run(
        ["bash", "-e", "-c", startup],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(tmp_path / "home"),
            "PYTHONPATH": str(ROOT),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "LOGDIR": str(tmp_path),
            "WORKDIR": str(tmp_path),
            "TARGET_WORKDIR": str(tmp_path / "target"),
            "GIT_ASKPASS_FILE": str(tmp_path / "askpass"),
            "SENPAI_GITHUB_TOKEN_FILE": str(tmp_path / "token"),
            "NODES_PER_STUDENT": "1",
            "SENPAI_KUBERNETES_EXECUTOR_SOCKET": str(executor_socket),
            "SENPAI_PYTHON": str(python),
            "UV_PROJECT_ENVIRONMENT": "/opt/senpai-venv",
            "UV_PYTHON": "/opt/senpai-venv/bin/python",
            "VIRTUAL_ENV": "/opt/senpai-venv",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    launched = json.loads(completed.stdout)
    assert launched["args"] == ["-P", "-m", "senpai_agent.supervisor", role]
    assert launched["environment"] == {
        "UV_PROJECT_ENVIRONMENT": None,
        "UV_PYTHON": None,
        "VIRTUAL_ENV": None,
        "SENPAI_PYTHON": str(python),
    }


def test_kubectl_proxy_uses_agent_python_inside_target_uv_environment(tmp_path: Path):
    entrypoint = (ROOT / "k8s" / "entrypoint-student.sh").read_text()
    proxy_setup = entrypoint[
        entrypoint.index('proxy_dir="$LOGDIR/bin"'):
        entrypoint.index('export PATH="$proxy_dir:$PATH"')
    ]
    runner_python = tmp_path / "runner-python"
    runner_python.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    runner_python.chmod(0o755)
    target_bin = tmp_path / "target-venv" / "bin"
    target_bin.mkdir(parents=True)
    (target_bin / "python").write_text("#!/bin/sh\nexit 99\n")
    (target_bin / "python").chmod(0o755)
    environment = {
        **os.environ,
        "LOGDIR": str(tmp_path),
        "SENPAI_PYTHON": str(runner_python),
        "PATH": f"{target_bin}:{os.environ['PATH']}",
    }
    subprocess.run(["bash", "-c", proxy_setup], env=environment, check=True)

    completed = subprocess.run(
        [str(tmp_path / "bin" / "kubectl"), "apply", "-f", "-"],
        env=environment, capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout) == [
        "-P", "-m", "senpai_agent.kubernetes_executor", "kubectl", "apply", "-f", "-",
    ]
