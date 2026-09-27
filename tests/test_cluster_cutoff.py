import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CUTOFF = REPO_ROOT / "scripts" / "arm_senpai_cluster_cutoff.sh"


def run_cutoff(*args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(CUTOFF), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        check=False,
    )


def render_cutoff(tmp_path: Path, *args: str):
    captured_script = tmp_path / "cutoff-job.sh"
    fake_kubectl = tmp_path / "kubectl"
    fake_kubectl.write_text(
        """#!/bin/sh
for arg in "$@"; do
  case "$arg" in
    --from-file=cutoff-job.sh=*)
      cp "${arg#--from-file=cutoff-job.sh=}" "$CAPTURED_CUTOFF_SCRIPT"
      ;;
  esac
done
printf '%s\n' 'apiVersion: v1' 'kind: ConfigMap'
""",
        encoding="utf-8",
    )
    fake_kubectl.chmod(0o755)
    result = run_cutoff(
        *args,
        "--dry-run",
        env={
            "KUBECTL": str(fake_kubectl),
            "CAPTURED_CUTOFF_SCRIPT": str(captured_script),
        },
    )
    return result, captured_script


def execute_job(script: Path, env: dict[str, str]):
    with subprocess.Popen(
        ["bash", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, **env},
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            pytest.fail(f"cutoff blocked beyond its deadline:\n{stdout}\n{stderr}")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


@pytest.fixture
def run_generated_cutoff(tmp_path):
    rendered, script = render_cutoff(
        tmp_path,
        "--run-slug",
        "restart",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
    )
    assert rendered.returncode == 0, rendered.stderr
    job = yaml.safe_load(rendered.stdout.split("--- Job ---\n")[1])
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "date": '''if [ "$2" = '+%s' ]; then cat "$CLOCK_FILE"; else exec "$REAL_DATE" "$@"; fi''',
        "sleep": '''printf '%s\\n' "$1" >> "$SLEEP_LOG"
printf '%s\\n' "$(( $(cat "$CLOCK_FILE") + $1 ))" > "$CLOCK_FILE"''',
        "kubectl": '''printf '%s %s\\n' "$(cat "$CLOCK_FILE")" "$*" >> "$KUBECTL_LOG"
case "$*" in
  *"get pods"*) printf '{"items":[{"status":{"containerStatuses":[{"ready":%s}]}}]}\\n' "$PODS_READY" ;;
  *"get deployments"*) printf '%s\\n' 'senpai-track-a' ;;
  *"delete deployments"*) ;;
  *) exit 2 ;;
esac''',
    }
    for name, contents in commands.items():
        command = bin_dir / name
        command.write_text(f"#!/bin/sh\n{contents}\n", encoding="utf-8")
        command.chmod(0o755)
    (tmp_path / "clock").write_text("1000\n", encoding="utf-8")
    env.update(
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "REAL_DATE": shutil.which("date"),
            "CLOCK_FILE": str(tmp_path / "clock"),
            "SLEEP_LOG": str(tmp_path / "sleeps"),
            "KUBECTL_LOG": str(tmp_path / "calls"),
            "PODS_READY": "true",
            "PVC_LOG_ROOT": str(tmp_path / "state"),
            "READINESS_TIMEOUT_SECONDS": "100",
            "ARMING_DEADLINE_EPOCH": "1100",
            "BUDGET_SECONDS": "120",
            "HARD_KILL_AT_EPOCH": "1220",
            "NAMESPACE": "test-ns",
        }
    )

    def run(**overrides):
        (tmp_path / "calls").write_text("", encoding="utf-8")
        (tmp_path / "sleeps").write_text("", encoding="utf-8")
        return execute_job(script, {**env, **overrides})

    return run


def test_cutoff_defaults_to_the_image_built_from_the_checked_out_commit(tmp_path):
    result, _ = render_cutoff(
        tmp_path,
        "--run-slug",
        "acceptance",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--budget-hours",
        "0",
    )

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert result.returncode == 0, result.stderr
    assert f"image: ghcr.io/wandb/senpai-cutoff:sha-{revision}" in result.stdout


def test_cutoff_rejects_a_mutable_image_reference():
    result = run_cutoff(
        "--run-slug",
        "acceptance",
        "--tags-csv",
        "track-a",
        "--image",
        "ghcr.io/wandb/senpai-cutoff:latest",
        "--dry-run",
    )

    assert result.returncode == 2
    assert "immutable" in result.stderr.lower()


def test_cutoff_rejects_a_start_gate_outside_its_shared_pvc():
    result = run_cutoff(
        "--run-slug",
        "acceptance",
        "--tags-csv",
        "track-a",
        "--pvc-mount-path",
        "/mnt/shared",
        "--start-gate-path",
        "/tmp/start-gate",
        "--dry-run",
    )

    assert result.returncode == 2
    assert "start-gate-path" in result.stderr
    assert "shared PVC" in result.stderr


def test_cutoff_has_a_minimal_commit_built_image():
    dockerfile = (REPO_ROOT / "Dockerfile.cutoff").read_text(encoding="utf-8")
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "build.yaml").read_text(encoding="utf-8")
    )

    assert "ARG SENPAI_SOURCE_REVISION=unknown" in dockerfile
    assert 'LABEL org.opencontainers.image.revision="${SENPAI_SOURCE_REVISION}"' in (
        dockerfile
    )
    assert "kubectl version --client" in dockerfile
    assert "python --version" in dockerfile
    assert "openhands" not in dockerfile.lower()
    assert "torch" not in dockerfile.lower()
    assert set(workflow["jobs"]["build"]["strategy"]["matrix"]["role"]) == {
        "advisor",
        "student",
        "executor",
        "cutoff",
    }


def test_cutoff_dry_run_keeps_readiness_and_delete_without_archive_rbac(tmp_path):
    started_at = int(time.time())
    result, captured_script = render_cutoff(
        tmp_path,
        "--run-slug",
        "acceptance",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--budget-hours",
        "0",
    )

    assert result.returncode == 0, result.stderr
    assert captured_script.is_file()
    rendered = result.stdout
    job_script = captured_script.read_text(encoding="utf-8")
    assert "Waiting for ready gate" in job_script
    assert 'sleep_until "$KILL_AT_EPOCH" "hard cutoff delete"' in job_script
    assert "delete deployments" in job_script
    assert "harvest" not in job_script.lower()
    assert "kubectl exec" not in job_script
    rbac = yaml.safe_load_all(
        rendered.split("--- RBAC ---\n")[1].split("--- ConfigMap ---\n")[0]
    )
    role = next(document for document in rbac if document["kind"] == "Role")
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]},
        {
            "apiGroups": ["apps"],
            "resources": ["deployments"],
            "verbs": ["get", "list", "watch", "delete"],
        },
    ]
    job = yaml.safe_load(rendered.split("--- Job ---\n")[1])
    pod = job["spec"]["template"]["spec"]
    assert pod["securityContext"] == {
        "runAsNonRoot": True,
        "runAsUser": 10001,
        "runAsGroup": 10001,
        "fsGroup": 10001,
        "fsGroupChangePolicy": "OnRootMismatch",
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    container = pod["containers"][0]
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    env = {item["name"]: item["value"] for item in container["env"]}
    assert started_at + 1800 <= int(env["ARMING_DEADLINE_EPOCH"]) <= int(time.time()) + 1800
    assert env["HARD_KILL_AT_EPOCH"] == env["ARMING_DEADLINE_EPOCH"]


@pytest.mark.parametrize("replaced_temporary", [None, "cutoff_state.json", "start-gate"])
def test_generated_cutoff_waits_for_readiness_then_deletes_selected_resources(
    tmp_path, replaced_temporary
):
    generated, captured_script = render_cutoff(
        tmp_path,
        "--run-slug",
        "acceptance",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--budget-hours",
        "0",
    )
    assert generated.returncode == 0, generated.stderr

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    delete_log = tmp_path / "delete.log"
    runtime_kubectl = bin_dir / "kubectl"
    runtime_kubectl.write_text(
        """#!/bin/sh
case "$*" in
  *"get pods"*)
    printf '%s\n' '{"items":[{"status":{"containerStatuses":[{"ready":true}]}}]}'
    ;;
  *"get deployments"*)
    printf '%s\n' 'senpai-track-a'
    ;;
  *"delete deployments"*)
    printf '%s\n' "$*" > "$DELETE_LOG"
    ;;
  *)
    printf 'unexpected kubectl call: %s\n' "$*" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    runtime_kubectl.chmod(0o755)
    replacement_log = tmp_path / "replacement.log"
    if replaced_temporary is not None:
        # Replace the newly created file before the cutoff reopens its path.
        # This schedules the shared-PVC race deterministically at the boundary.
        racing_mktemp = bin_dir / "mktemp"
        racing_mktemp.write_text(
            """#!/bin/sh
created=$("$REAL_MKTEMP" "$@") || exit
case "$created" in
  *"$REPLACED_TEMPORARY".tmp.*)
    rm "$created"
    mkfifo "$created"
    printf '%s\\n' "$created" > "$REPLACEMENT_LOG"
    ;;
esac
printf '%s\\n' "$created"
""",
            encoding="utf-8",
        )
        racing_mktemp.chmod(0o755)
    state_root = tmp_path / "state"
    gate = tmp_path / "start-gate"
    result = execute_job(
        captured_script,
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DELETE_LOG": str(delete_log),
            "RUN_SLUG": "acceptance",
            "TAGS_CSV": "track-a",
            "EXPECTED_PODS": "1",
            "EXPECTED_DEPLOYMENTS": "1",
            "READINESS_TIMEOUT_SECONDS": "1800",
            "BUDGET_SECONDS": "0",
            "ARMING_DEADLINE_EPOCH": "0",
            "HARD_KILL_AT_EPOCH": "0",
            "ARM_ID": "acceptance-arm",
            "STATE_AUTH_KEY": "a" * 64,
            "PVC_LOG_ROOT": str(state_root),
            "START_GATE_PATH": str(gate),
            "NAMESPACE": "test-ns",
            "REAL_MKTEMP": shutil.which("mktemp"),
            "REPLACED_TEMPORARY": replaced_temporary or "",
            "REPLACEMENT_LOG": str(replacement_log),
        },
    )

    assert result.returncode == 0, result.stderr
    if replaced_temporary != "start-gate":
        assert gate.is_file()
    if replaced_temporary is not None:
        assert replacement_log.is_file()
    deleted = delete_log.read_text(encoding="utf-8")
    assert "delete deployments" in deleted
    assert "research-tag in (track-a)" in deleted


def test_generated_cutoff_arms_after_readiness_deadline_when_a_pod_never_readies(
    tmp_path,
):
    generated, captured_script = render_cutoff(
        tmp_path,
        "--run-slug",
        "never-ready",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--readiness-timeout-minutes",
        "0",
        "--budget-hours",
        "0",
    )
    assert generated.returncode == 0, generated.stderr

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    delete_log = tmp_path / "delete.log"
    runtime_kubectl = bin_dir / "kubectl"
    runtime_kubectl.write_text(
        """#!/bin/sh
case "$*" in
  *"get pods"*)
    printf '%s\n' '{"items":[{"status":{"containerStatuses":[{"ready":false}]}}]}'
    ;;
  *"get deployments"*)
    printf '%s\n' 'senpai-track-a'
    ;;
  *"delete deployments"*)
    printf '%s\n' "$*" > "$DELETE_LOG"
    ;;
  *)
    printf 'unexpected kubectl call: %s\n' "$*" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    runtime_kubectl.chmod(0o755)
    state_root = tmp_path / "state"
    gate = tmp_path / "start-gate"

    result = subprocess.run(
        ["bash", str(captured_script)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DELETE_LOG": str(delete_log),
            "RUN_SLUG": "never-ready",
            "TAGS_CSV": "track-a",
            "EXPECTED_PODS": "1",
            "EXPECTED_DEPLOYMENTS": "1",
            "READINESS_TIMEOUT_SECONDS": "0",
            "BUDGET_SECONDS": "0",
            "ARMING_DEADLINE_EPOCH": "0",
            "HARD_KILL_AT_EPOCH": "0",
            "ARM_ID": "never-ready-arm",
            "STATE_AUTH_KEY": "a" * 64,
            "PVC_LOG_ROOT": str(state_root),
            "START_GATE_PATH": str(gate),
            "NAMESPACE": "test-ns",
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Readiness deadline reached; arming cutoff anyway" in result.stdout
    assert gate.is_file()
    assert delete_log.is_file()


def test_rearming_a_used_slug_replaces_its_expired_cutoff_state(tmp_path):
    generated, captured_script = render_cutoff(
        tmp_path,
        "--run-slug",
        "reused",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--budget-hours",
        "0",
    )
    assert generated.returncode == 0, generated.stderr

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runtime_kubectl = bin_dir / "kubectl"
    runtime_kubectl.write_text(
        """#!/bin/sh
case "$*" in
  *"get pods"*)
    printf '%s\n' '{"items":[{"status":{"containerStatuses":[{"ready":true}]}}]}'
    ;;
  *"get deployments"*) printf '%s\n' 'senpai-track-a' ;;
  *"delete deployments"*) exit 0 ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    runtime_kubectl.chmod(0o755)
    state_root = tmp_path / "state"
    run_dir = state_root / "reused"
    run_dir.mkdir(parents=True)
    state_file = run_dir / "cutoff_state.json"
    executed = tmp_path / "payload-executed"
    state_file.write_text(
        f"PERSISTED_ARM_ID=$(touch {executed})\n"
        "RUN_SLUG=reused\n"
        "TAGS_CSV=track-a\n"
        "ARMED_AT_UTC=2026-01-01T00:00:00Z\n"
        "ARM_REASON=all_ready\n"
        "KILL_AT_EPOCH=1\n"
        "KILL_AT_UTC=2026-01-01T00:00:01Z\n"
        "EXPECTED_PODS=1\n"
        "EXPECTED_DEPLOYMENTS=1\n"
        "SELECTOR='research-tag in (track-a)'\n"
        "START_GATE_PATH=''\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(captured_script)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RUN_SLUG": "reused",
            "TAGS_CSV": "track-a",
            "EXPECTED_PODS": "1",
            "EXPECTED_DEPLOYMENTS": "1",
            "READINESS_TIMEOUT_SECONDS": "0",
            "BUDGET_SECONDS": "0",
            "ARMING_DEADLINE_EPOCH": "0",
            "HARD_KILL_AT_EPOCH": "0",
            "ARM_ID": "fresh-arm",
            "STATE_AUTH_KEY": "a" * 64,
            "PVC_LOG_ROOT": str(state_root),
            "NAMESPACE": "test-ns",
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Discarding cutoff state from an earlier arm" in result.stdout
    assert not executed.exists()
    state = json.loads(state_file.read_text(encoding="utf-8"))["payload"]
    assert state["PERSISTED_ARM_ID"] == "fresh-arm"
    assert state["KILL_AT_EPOCH"] != 1


def test_generated_cutoff_deletes_when_shared_state_cannot_be_replaced(tmp_path):
    generated, captured_script = render_cutoff(
        tmp_path,
        "--run-slug",
        "state-race",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--budget-hours",
        "0",
    )
    assert generated.returncode == 0, generated.stderr

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    delete_log = tmp_path / "delete.log"
    runtime_kubectl = bin_dir / "kubectl"
    runtime_kubectl.write_text(
        """#!/bin/sh
case "$*" in
  *"get pods"*)
    printf '%s\n' '{"items":[{"status":{"containerStatuses":[{"ready":true}]}}]}'
    ;;
  *"get deployments"*) printf '%s\n' 'senpai-track-a' ;;
  *"delete deployments"*) printf '%s\n' "$*" > "$DELETE_LOG" ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    runtime_kubectl.chmod(0o755)
    state_root = tmp_path / "state"
    run_dir = state_root / "state-race"
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o555)

    result = subprocess.run(
        ["bash", str(captured_script)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DELETE_LOG": str(delete_log),
            "RUN_SLUG": "state-race",
            "TAGS_CSV": "track-a",
            "EXPECTED_PODS": "1",
            "EXPECTED_DEPLOYMENTS": "1",
            "READINESS_TIMEOUT_SECONDS": "0",
            "BUDGET_SECONDS": "0",
            "ARMING_DEADLINE_EPOCH": "0",
            "HARD_KILL_AT_EPOCH": "0",
            "ARM_ID": "state-race-arm",
            "STATE_AUTH_KEY": "a" * 64,
            "PVC_LOG_ROOT": str(state_root),
            "NAMESPACE": "test-ns",
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "using in-memory deadline" in result.stdout
    assert delete_log.is_file()


def test_operator_job_replacement_is_scoped_to_the_requested_namespace(tmp_path):
    kubectl_log = tmp_path / "kubectl.log"
    fake_kubectl = tmp_path / "kubectl"
    fake_kubectl.write_text(
        """#!/bin/sh
printf '%s\n' "$*" >> "$KUBECTL_LOG"
case "$*" in
  *"create configmap"*) printf '%s\n' 'apiVersion: v1' 'kind: ConfigMap' ;;
esac
""",
        encoding="utf-8",
    )
    fake_kubectl.chmod(0o755)

    result = run_cutoff(
        "--run-slug",
        "namespaced",
        "--tags-csv",
        "track-a",
        env={
            "KUBECTL": str(fake_kubectl),
            "KUBECTL_LOG": str(kubectl_log),
            "NAMESPACE": "review-ns",
        },
    )

    assert result.returncode == 0, result.stderr
    assert (
        "--context pai-2 -n review-ns delete job senpai-cutoff-namespaced "
        "--ignore-not-found=true"
    ) in kubectl_log.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("budget, deadline", [("0", 1000), ("120", 1120)])
@pytest.mark.parametrize("obstacle", ["parent_file", "gate_directory"])
def test_generated_cutoff_deletes_at_deadline_when_the_start_gate_cannot_open(
    tmp_path, run_generated_cutoff, budget, deadline, obstacle
):
    blocker = tmp_path / "blocker"
    if obstacle == "parent_file":
        blocker.write_text("not a directory", encoding="utf-8")
        gate = blocker / "start-gate"
    else:
        blocker.mkdir()
        gate = blocker
    result = run_generated_cutoff(
        BUDGET_SECONDS=budget, START_GATE_PATH=str(gate)
    )

    assert result.returncode == 0, result.stderr
    assert "Start gate remains unavailable at the cutoff deadline" in result.stdout
    assert (tmp_path / "sleeps").read_text().splitlines() == (
        ["60", "60"] if budget == "120" else []
    )
    assert f"{deadline} -n test-ns delete deployments -l research-tag in (track-a)" in (
        tmp_path / "calls"
    ).read_text()
    if obstacle == "gate_directory":
        assert list(gate.iterdir()) == []


@pytest.mark.parametrize(
    "state_change, deadline",
    [
        ("none", 1120),
        ("tamper", 1220),
        ("missing", 1220),
        ("symlink", 1220),
        ("fifo", 1220),
        ("directory", 1220),
        ("hard_cap", 1080),
    ],
)
def test_cutoff_restart_authenticates_state_and_preserves_deadlines(
    tmp_path, run_generated_cutoff, state_change, deadline
):
    first = run_generated_cutoff()
    assert first.returncode == 0, first.stderr
    state_file = tmp_path / "state" / "restart" / "cutoff_state.json"
    saved_state = state_file.read_text()
    assert json.loads(saved_state)["payload"]["KILL_AT_EPOCH"] == 1120
    if state_change == "tamper":
        state = json.loads(saved_state)
        state["payload"]["KILL_AT_EPOCH"] = 9999
        state_file.write_text(json.dumps(state))
    elif state_change in {"missing", "symlink", "fifo", "directory"}:
        state_file.unlink()
        if state_change == "symlink":
            target = tmp_path / "untrusted-state"
            target.write_text(saved_state)
            state_file.symlink_to(target)
        elif state_change == "fifo":
            os.mkfifo(state_file)
        elif state_change == "directory":
            state_file.mkdir()
    (tmp_path / "clock").write_text("1005\n")
    overrides = {"PODS_READY": "false"}
    if state_change == "hard_cap":
        overrides["HARD_KILL_AT_EPOCH"] = "1080"

    restarted = run_generated_cutoff(**overrides)

    assert restarted.returncode == 0, restarted.stderr
    reused = state_change in {"none", "hard_cap"}
    assert ("Loaded existing cutoff state" in restarted.stdout) == reused
    calls = (tmp_path / "calls").read_text()
    assert ("get pods" in calls) != reused
    assert f"{deadline} -n test-ns delete deployments -l research-tag in (track-a)" in calls
    if state_change == "symlink":
        assert target.read_text() == saved_state
    elif state_change == "directory":
        assert "using in-memory deadline" in restarted.stdout
        assert list(state_file.iterdir()) == []
