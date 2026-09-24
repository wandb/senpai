import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

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
    return invocation, {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}


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


def test_container_health_success_resets_consecutive_failures(tmp_path: Path):
    started = tmp_path / "bootstrap-started"
    failures = tmp_path / "health-failures"
    started.write_text("1")
    failures.write_text("4")
    _invocation, environment = _fake_python(tmp_path, 0)
    environment.update(
        {
            "SENPAI_BOOTSTRAP_STARTED_PATH": str(started),
            "SENPAI_BOOTSTRAP_GRACE_SECONDS": "1",
            "SENPAI_HEALTH_FAILURES_PATH": str(failures),
        }
    )

    result = subprocess.run(
        ["sh", str(HEALTH_SCRIPT), str(tmp_path / "lease.json")],
        env=environment,
        check=False,
    )

    assert result.returncode == 0
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
    tmp_path: Path, role: str
):
    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text()
    startup = entrypoint[entrypoint.index("export IS_SANDBOX=1"):]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'environment': "
        "{key: os.environ.get(key) for key in "
        "('UV_PROJECT_ENVIRONMENT', 'UV_PYTHON', 'VIRTUAL_ENV', 'SENPAI_PYTHON')}}))\n"
    )
    python.chmod(0o755)

    completed = subprocess.run(
        ["bash", "-c", startup],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LOGDIR": str(tmp_path),
            "WORKDIR": str(tmp_path),
            "TARGET_WORKDIR": str(tmp_path / "target"),
            "GIT_ASKPASS_FILE": str(tmp_path / "askpass"),
            "SENPAI_GITHUB_TOKEN_FILE": str(tmp_path / "token"),
            "NODES_PER_STUDENT": "1",
            "SENPAI_PYTHON": "/opt/senpai-venv/bin/python",
            "UV_PROJECT_ENVIRONMENT": "/opt/senpai-venv",
            "UV_PYTHON": "/opt/senpai-venv/bin/python",
            "VIRTUAL_ENV": "/opt/senpai-venv",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    launched = json.loads(completed.stdout)
    assert launched["args"] == ["-m", "senpai_agent.supervisor", role]
    assert launched["environment"] == {
        "UV_PROJECT_ENVIRONMENT": None,
        "UV_PYTHON": None,
        "VIRTUAL_ENV": None,
        "SENPAI_PYTHON": "/opt/senpai-venv/bin/python",
    }
