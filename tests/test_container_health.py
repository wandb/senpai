import os
import subprocess
import time
from pathlib import Path

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
        assert 'SENPAI_OPENHANDS_TIMEOUT_SECONDS:-57600' in entrypoint
