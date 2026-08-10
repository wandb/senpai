import json
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.supervisor as supervisor_module
from senpai_agent.operations import RestartRequest, RestartRequestStore, RoleTarget
from senpai_agent.supervisor import (
    CONTROL_DIR_ENV,
    GENERATION_ENV,
    ProgressLease,
    SupervisorConfig,
    WorkerLease,
    WorkerSupervisor,
)
from senpai_agent.supervisor_pause import (
    DEFAULT_REPAIR_PAUSE_SOCKET,
    REPAIR_PAUSE_PROTOCOL,
    RepairPauseAcknowledgement,
    RepairPauseClient,
    RepairPauseControlServer,
    RepairPauseGrant,
    RepairPauseStore,
)


def wait_for(path: Path, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"{path} was not created")


def pause_socket(tmp_path: Path) -> Path:
    fingerprint = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    return Path(f"/private/tmp/senpai-pause-test-{fingerprint}.sock")


def run_supervisor(
    supervisor: WorkerSupervisor,
    stop: threading.Event,
) -> tuple[threading.Thread, list[int]]:
    results: list[int] = []
    thread = threading.Thread(target=lambda: results.append(supervisor.run(stop)))
    thread.start()
    return thread, results


def restart_backoffs(stderr: str) -> list[float]:
    return [
        float(match.group(1))
        for match in re.finditer(r"backoff_seconds=([0-9.]+)", stderr)
    ]


def test_supervisor_default_termination_grace_is_sixty_seconds():
    assert SupervisorConfig().terminate_grace_seconds == 60


def test_supervisor_caps_repeated_restart_backoff_at_five_minutes():
    assert SupervisorConfig().max_backoff_seconds == 300


def test_progress_lease_carries_the_worker_generation_and_legacy_is_unowned(
    tmp_path: Path,
):
    """
    Requirement: the process owner can distinguish the worker it spawned from a
    stale or legacy process before honoring a planned restart.
    Interface: the controller lease exchanged by ProgressLease and WorkerSupervisor.
    """

    lease_path = tmp_path / "controller-lease.json"
    ProgressLease(lease_path, generation=7).update("sleep", 30)

    assert WorkerLease.read(lease_path).generation == 7

    lease_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "phase": "sleep",
                "deadline": time.monotonic() + 30,
            }
        )
    )
    assert WorkerLease.read(lease_path).generation is None


def test_pid_one_reaps_adopted_children_without_reaping_its_worker(monkeypatch):
    reaped = []
    monkeypatch.setattr(supervisor_module.os, "getpid", lambda: 1)
    monkeypatch.setattr(
        supervisor_module.psutil,
        "Process",
        lambda: type(
            "SupervisorProcess",
            (),
            {
                "children": lambda self: [
                    type("Child", (), {"pid": 41})(),
                    type("Child", (), {"pid": 42})(),
                ]
            },
        )(),
    )
    monkeypatch.setattr(
        supervisor_module.os,
        "waitpid",
        lambda pid, options: reaped.append((pid, options)),
    )

    WorkerSupervisor._reap_orphaned_children(worker_pid=41)

    assert reaped == [(42, os.WNOHANG)]


def test_restarted_workers_receive_github_token_without_environment_exposure(
    tmp_path: Path,
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
token_fd = int(os.environ["SENPAI_GITHUB_TOKEN_FD"])
with os.fdopen(token_fd) as token_stream:
    token = token_stream.read()
with (state / "observations").open("a") as output:
    output.write(json.dumps({
        "token": token,
        "github_env": os.environ.get("GITHUB_TOKEN"),
        "gh_env": os.environ.get("GH_TOKEN"),
        "token_file_env": os.environ.get("SENPAI_GITHUB_TOKEN_FILE"),
    }) + "\\n")

lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "ready",
    "deadline": time.monotonic() + 30,
}))
if count == 1:
    raise SystemExit(19)
(state / "ready").write_text("ready")
while True:
    time.sleep(1)
""".strip()
    )
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        github_token=SecretStr("write-token-sentinel"),
        environment={
            **os.environ,
            "GITHUB_TOKEN": "must-not-survive",
            "GH_TOKEN": "must-not-survive",
        },
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.1,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "ready")
    stop.set()
    thread.join(5)

    assert results == [0]
    observations = [
        json.loads(line)
        for line in (tmp_path / "observations").read_text().splitlines()
    ]
    assert len(observations) == 2
    assert all(item["token"] == "write-token-sentinel" for item in observations)
    assert all(item["github_env"] is None for item in observations)
    assert all(item["gh_env"] is None for item in observations)
    assert all(item["token_file_env"] is None for item in observations)
    assert not list(tmp_path.glob(".github-token-*"))


def test_overdue_worker_is_killed_and_restarted(tmp_path: Path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import signal
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
signal.signal(signal.SIGTERM, signal.SIG_IGN)

lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
temporary = lease.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "wedged-turn" if count == 1 else "healthy-turn",
    "deadline": time.monotonic() + (0.05 if count == 1 else 30),
}))
temporary.replace(lease)

if count > 1:
    (state / "restarted").write_text("restarted")
while True:
    time.sleep(1)
""".strip()
    )
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.05,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "restarted")
    stop.set()
    thread.join(5)

    assert not thread.is_alive()
    assert results == [0]
    assert int((tmp_path / "starts").read_text()) >= 2


def test_repair_pause_stops_worker_before_ack_and_blocks_restart(tmp_path: Path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import time
from pathlib import Path

state = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"]).parent
with (state / "starts").open("a") as output:
    output.write(f"{os.getpid()}\\n")
lease = state / "controller-lease.json"
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "sleep",
    "deadline": time.monotonic() + 30,
}))
while True:
    time.sleep(1)
""".strip()
    )
    control_dir = tmp_path / "control"
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker)),
        lease_path=tmp_path / "controller-lease.json",
        control_dir=control_dir,
        pause_socket=pause_socket(tmp_path),
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.1,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
    )
    supervisor._network_listeners_absent = lambda: True
    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "controller-lease.json")
    client = RepairPauseClient(
        pause_socket(tmp_path),
        expected_supervisor_pid=os.getpid(),
        peer_pid=lambda _connection: os.getpid(),
    )

    grant = client.pause(
        "repair-lease-1",
        duration_seconds=10,
        wait_seconds=5,
    )
    starts_while_paused = (tmp_path / "starts").read_text().splitlines()
    time.sleep(0.1)

    assert grant.acknowledgement.supervisor_pid == os.getpid()
    assert (tmp_path / "starts").read_text().splitlines() == starts_while_paused
    assert not (tmp_path / "controller-lease.json").exists()

    client.resume(
        grant.acknowledgement.lease_id,
        grant.resume_capability,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if len((tmp_path / "starts").read_text().splitlines()) >= 2:
            break
        time.sleep(0.01)
    stop.set()
    thread.join(5)

    assert results == [0]
    assert len((tmp_path / "starts").read_text().splitlines()) >= 2


def test_health_accepts_only_an_acknowledged_live_repair_pause(tmp_path: Path):
    control_dir = tmp_path / "control"
    store = RepairPauseStore(control_dir)
    socket_path = pause_socket(tmp_path)
    server = RepairPauseControlServer(store, socket_path)
    server.start()
    client = RepairPauseClient(
        socket_path,
        expected_supervisor_pid=os.getpid(),
        peer_pid=lambda _connection: os.getpid(),
    )

    try:
        assert client.is_paused() is False

        grants = []

        requester = threading.Thread(
            target=lambda: grants.append(
                client.pause(
                    "repair-lease-health",
                    duration_seconds=30,
                    wait_seconds=5,
                )
            )
        )
        requester.start()
        wait_for(store.request_path)
        pause = store.current()
        assert pause is not None
        assert client.is_paused() is False
        assert client.is_quiescing_or_paused() is True
        server.acknowledge(pause)
        requester.join(5)

        assert client.is_paused() is True

        client.resume(pause.lease_id, grants[0].resume_capability)
        assert client.is_paused() is False
    finally:
        server.close()


def test_concurrent_repair_pause_cannot_destroy_the_active_acknowledgement(
    tmp_path: Path,
):
    store = RepairPauseStore(tmp_path / "control")
    server = RepairPauseControlServer(store, pause_socket(tmp_path))
    pause = store.request("repair-active", duration_seconds=30)
    acknowledgement = server.acknowledge(pause)

    with pytest.raises(RuntimeError, match="already active"):
        store.request("repair-racing", duration_seconds=30)

    assert server.acknowledgement() == acknowledgement


def test_worker_written_ack_file_cannot_authorize_repair(tmp_path: Path):
    """A same-UID worker cannot forge PID 1's pause acknowledgement."""

    control_dir = tmp_path / "control"
    store = RepairPauseStore(control_dir)
    socket_path = pause_socket(tmp_path)
    server = RepairPauseControlServer(store, socket_path)
    server.start()
    client = RepairPauseClient(
        socket_path,
        expected_supervisor_pid=os.getpid(),
        peer_pid=lambda _connection: os.getpid(),
    )
    outcome: list[BaseException] = []

    def request_pause() -> None:
        try:
            client.pause(
                "repair-forgery",
                duration_seconds=10,
                wait_seconds=0.25,
            )
        except BaseException as error:  # noqa: BLE001
            outcome.append(error)

    requester = threading.Thread(target=request_pause)
    requester.start()
    wait_for(store.request_path)
    pause = store.current()
    assert pause is not None
    (control_dir / "repair-pause-ack.json").write_text(
        json.dumps(
            {
                "protocol": pause.protocol,
                "lease_id": pause.lease_id,
                "expires_at": pause.expires_at,
                "acknowledged_at": time.time(),
                "supervisor_pid": 1,
            }
        )
    )
    requester.join(2)
    server.close()

    assert outcome
    assert "deadline" in str(outcome[0])


def test_pause_client_rejects_ack_from_non_supervisor_peer(tmp_path: Path):
    client = RepairPauseClient(
        pause_socket(tmp_path),
        expected_supervisor_pid=1,
        peer_pid=lambda _connection: 4242,
    )

    with pytest.raises(RuntimeError, match="PID 1"):
        client.validate_peer(object())


def test_kubectl_exec_pause_uses_pid_one_socket_not_runtime_private_directory(
    monkeypatch,
    capsys,
):
    observed = {}
    capability = "kubectl-exec-one-time-capability"

    class RecordingClient:
        def __init__(self, socket_path):
            observed["socket_path"] = socket_path

        def pause(self, lease_id, *, duration_seconds, wait_seconds):
            observed["lease_id"] = lease_id
            now = time.monotonic()
            acknowledgement = RepairPauseAcknowledgement(
                protocol=REPAIR_PAUSE_PROTOCOL,
                lease_id=lease_id,
                expires_at=now + duration_seconds,
                acknowledged_at=now,
                supervisor_pid=1,
                resume_capability_sha256=hashlib.sha256(
                    capability.encode()
                ).hexdigest(),
            )
            return RepairPauseGrant(acknowledgement, capability)

    monkeypatch.setattr(supervisor_module, "RepairPauseClient", RecordingClient)

    assert supervisor_module.supervisor_main(
        [
            "repair-pause",
            "--lease-id",
            "repair-container-env",
            "--duration-seconds",
            "300",
            "--wait-seconds",
            "90",
        ],
        env={CONTROL_DIR_ENV: "/fs-group-visible-volume-root"},
    ) == 0

    assert observed == {
        "socket_path": DEFAULT_REPAIR_PAUSE_SOCKET,
        "lease_id": "repair-container-env",
    }
    output = json.loads(capsys.readouterr().out)
    assert output["resume_capability"] == capability
    assert 0 < output["remaining_seconds"] <= 300


def test_pre_ack_wrong_and_replayed_resume_capabilities_never_release_pause(
    tmp_path: Path,
):
    store = RepairPauseStore(tmp_path / "control")
    socket_path = pause_socket(tmp_path)
    server = RepairPauseControlServer(store, socket_path)
    server.start()
    client = RepairPauseClient(
        socket_path,
        expected_supervisor_pid=os.getpid(),
        peer_pid=lambda _connection: os.getpid(),
    )
    grants = []
    requester = threading.Thread(
        target=lambda: grants.append(
            client.pause(
                "repair-resume-race",
                duration_seconds=10,
                wait_seconds=5,
            )
        )
    )
    requester.start()
    wait_for(store.request_path)

    with pytest.raises(RuntimeError, match="not been acknowledged"):
        client.resume("repair-resume-race", "buffered-worker-capability")
    pause = store.current()
    assert pause is not None
    assert pause.acknowledged_at is None

    server.acknowledge(pause)
    requester.join(5)
    assert grants
    with pytest.raises(RuntimeError, match="capability was rejected"):
        client.resume("repair-resume-race", "wrong-capability")
    assert store.current() is not None

    client.resume("repair-resume-race", grants[0].resume_capability)
    with pytest.raises(RuntimeError, match="not active"):
        client.resume("repair-resume-race", grants[0].resume_capability)
    server.close()


def test_expired_resume_capability_cannot_release_a_new_or_missing_pause(
    tmp_path: Path,
):
    store = RepairPauseStore(tmp_path / "control")
    socket_path = pause_socket(tmp_path)
    server = RepairPauseControlServer(store, socket_path)
    server.start()
    client = RepairPauseClient(
        socket_path,
        expected_supervisor_pid=os.getpid(),
        peer_pid=lambda _connection: os.getpid(),
    )
    grants = []
    requester = threading.Thread(
        target=lambda: grants.append(
            client.pause(
                "repair-expiring",
                duration_seconds=0.2,
                wait_seconds=1,
            )
        )
    )
    requester.start()
    wait_for(store.request_path)
    pause = store.current()
    assert pause is not None
    server.acknowledge(pause)
    requester.join(2)
    time.sleep(0.25)

    with pytest.raises(RuntimeError, match="not active"):
        client.resume("repair-expiring", grants[0].resume_capability)
    server.close()


def test_pause_control_socket_is_never_inherited_by_workers(tmp_path: Path):
    server = RepairPauseControlServer(
        RepairPauseStore(tmp_path / "control"),
        pause_socket(tmp_path),
    )
    server.start()
    try:
        assert server._listener is not None
        assert server._listener.get_inheritable() is False
    finally:
        server.close()


def test_repair_pause_requires_the_role_network_listener_boundary_to_be_empty(
    tmp_path: Path,
    monkeypatch,
):
    supervisor = WorkerSupervisor(
        command=("worker",),
        lease_path=tmp_path / "controller-lease.json",
        control_dir=tmp_path / "control",
        config=SupervisorConfig(
            check_interval_seconds=0.001,
            terminate_grace_seconds=0.005,
        ),
    )
    monkeypatch.setattr(supervisor, "_network_listeners_absent", lambda: False)

    assert supervisor._repair_boundary_is_quiescent(
        {},
        worker_ownership_token="worker-token",
    ) is False

    monkeypatch.setattr(supervisor, "_network_listeners_absent", lambda: True)
    assert supervisor._repair_boundary_is_quiescent(
        {},
        worker_ownership_token="worker-token",
    ) is True


def test_listener_probe_unavailability_denies_repair(tmp_path: Path):
    assert WorkerSupervisor._network_listeners_absent(
        proc_network=tmp_path / "missing-proc-net"
    ) is False


def test_worker_ownership_scan_selects_only_the_exact_inherited_capability(
    tmp_path: Path,
):
    for pid, environment in (
        ("41", b"SENPAI_WORKER_OWNERSHIP_TOKEN=owned\0OTHER=x\0"),
        ("42", b"SENPAI_WORKER_OWNERSHIP_TOKEN=other\0"),
        ("self", b"SENPAI_WORKER_OWNERSHIP_TOKEN=owned\0"),
    ):
        process = tmp_path / pid
        process.mkdir()
        (process / "environ").write_bytes(environment)

    assert WorkerSupervisor._owned_worker_pids(
        "owned",
        proc_root=tmp_path,
        require_pid_one=False,
    ) == (41,)

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux peer credentials")
def test_repair_pause_cli_round_trip_uses_pid_one_control_socket(
    tmp_path: Path,
):
    # The subprocess CLI requires a real container PID 1 peer. Its production
    # round trip is exercised by the Kubernetes canary; unit tests cover the
    # same client/server wire contract with an injected peer-credential reader.
    assert sys.platform.startswith("linux")


def test_worker_uptime_without_a_completed_turn_does_not_reset_restart_backoff(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "sleep",
    "deadline": 1e100,
}))
time.sleep(0.03)
if count < 3:
    raise SystemExit(19)
(state / "ready").write_text("ready")
while True:
    time.sleep(1)
""".strip()
    )
    class AcceleratedClock:
        @staticmethod
        def monotonic():
            return time.monotonic() * 100_000

    monkeypatch.setattr(supervisor_module, "time", AcceleratedClock())
    monkeypatch.setattr(supervisor_module.random, "uniform", lambda _a, _b: 1.2)
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        config=SupervisorConfig(
            startup_timeout_seconds=1_000_000_000,
            check_interval_seconds=0.005,
            terminate_grace_seconds=0.05,
            initial_backoff_seconds=0.2,
            max_backoff_seconds=0.4,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "ready")
    stop.set()
    thread.join(5)

    assert results == [0]
    assert restart_backoffs(capsys.readouterr().err) == [0.2, 0.4]


def test_worker_exit_samples_its_final_completed_turn(tmp_path: Path, monkeypatch):
    supervisor = WorkerSupervisor(
        command=("worker",),
        lease_path=tmp_path / "controller-lease.json",
    )
    leases = iter(
        (
            WorkerLease(
                pid=123,
                phase="openhands-turn",
                deadline=time.monotonic() + 30,
            ),
            WorkerLease(
                pid=123,
                phase="turn-complete",
                deadline=time.monotonic() + 30,
                completed_turns=1,
            ),
        )
    )
    monkeypatch.setattr(supervisor, "_read_lease", lambda: next(leases))
    monkeypatch.setattr(supervisor, "_remember_descendants", lambda *_args: None)

    class ExitedProcess:
        pid = 123

        @staticmethod
        def poll():
            return 19

    reason, made_progress, planned = supervisor._wait_for_worker(
        ExitedProcess(),
        {},
        threading.Event(),
        time.monotonic(),
    )

    assert reason == "exit:19"
    assert made_progress is True
    assert planned is None


def test_completed_turn_resets_restart_backoff(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "poll",
    "deadline": time.monotonic() + 30,
    "completed_turns": 1 if count == 2 else 0,
}))
time.sleep(0.03)
if count < 4:
    raise SystemExit(19)
(state / "ready").write_text("ready")
while True:
    time.sleep(1)
""".strip()
    )
    monkeypatch.setattr(supervisor_module.random, "uniform", lambda _a, _b: 1.0)
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.005,
            terminate_grace_seconds=0.05,
            initial_backoff_seconds=0.1,
            max_backoff_seconds=0.4,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "ready")
    stop.set()
    thread.join(5)

    assert results == [0]
    assert restart_backoffs(capsys.readouterr().err) == [0.1, 0.1, 0.2]


def test_planned_restart_is_owner_consumed_completed_and_has_no_crash_backoff(
    tmp_path: Path,
    capsys,
):
    """
    Requirement: WorkerSupervisor alone replaces a quiescent controller, records
    the replacement generation, and does not classify the planned stop as a crash.
    Interface: durable restart status, worker lease, and supervisor restart log.
    """

    conversation_id = "00000000-0000-0000-0000-000000000211"
    worker = tmp_path / "planned_worker.py"
    worker.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
generation = int(os.environ["SENPAI_CONTROLLER_GENERATION"])
with (state / "starts").open("a") as output:
    output.write(f"{generation}\\n")
lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
temporary = lease.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "sleep",
    "deadline": time.monotonic() + 30,
    "completed_turns": 0,
    "conversation_id": "00000000-0000-0000-0000-000000000211",
    "generation": generation,
}))
temporary.replace(lease)
if generation > 1:
    (state / "replacement-ready").write_text(str(generation))
while True:
    time.sleep(1)
""".strip()
    )
    lease_path = tmp_path / "controller-lease.json"
    restart_path = tmp_path / "controller-restarts.sqlite3"
    environment = {
        **os.environ,
        "SENPAI_ROLE": "student",
        "RESEARCH_TAG": "maple",
        "STUDENT_NAME": "fern",
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path),
    }
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=lease_path,
        environment=environment,
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.1,
            initial_backoff_seconds=0.2,
            max_backoff_seconds=0.2,
        ),
    )
    stop = threading.Event()
    thread, results = run_supervisor(supervisor, stop)
    wait_for(lease_path)
    first = WorkerLease.read(lease_path)
    assert first.generation == 1
    request = RestartRequest(
        request_id="planned-restart-211",
        target=RoleTarget(research_tag="maple", role="student", student="fern"),
        expected_conversation_id=conversation_id,
        expected_restart_control_token="opaque-role-authorization",
        expected_worker_generation=1,
        expected_completed_turns=0,
    )
    with RestartRequestStore(restart_path) as store:
        store.enqueue(request)

    wait_for(tmp_path / "replacement-ready")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with RestartRequestStore(restart_path) as store:
            result = store.result(request.request_id)
        if result.status == "completed":
            break
        time.sleep(0.01)
    stop.set()
    thread.join(5)

    assert results == [0]
    assert result.status == "completed"
    assert result.completion is not None
    assert result.completion.source_generation == 1
    assert result.completion.replacement_generation == 2
    stderr = capsys.readouterr().err
    assert "SENPAI_CONTROLLER_PLANNED_RESTART" in stderr
    assert "backoff_seconds=" not in stderr


def test_worker_owner_rejects_a_restart_if_compute_became_active_after_queueing(
    tmp_path: Path,
):
    """
    Requirement: the process owner repeats safety checks immediately before stop,
    even when role control observed an earlier quiescent boundary.
    Interface: durable rejection status and unchanged worker generation.
    """

    conversation_id = "00000000-0000-0000-0000-000000000213"
    worker = tmp_path / "guarded_worker.py"
    worker.write_text(
        f"""
import json
import os
import time
from pathlib import Path

generation = int(os.environ["{GENERATION_ENV}"])
Path({str(tmp_path / "starts")!r}).write_text(str(generation))
lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({{
    "pid": os.getpid(),
    "phase": "sleep",
    "deadline": time.monotonic() + 30,
    "completed_turns": 0,
    "conversation_id": "{conversation_id}",
    "generation": generation,
}}))
while True:
    time.sleep(1)
""".strip()
    )
    delegation_dir = tmp_path / "delegation"
    delegation_dir.mkdir()
    with sqlite3.connect(delegation_dir / "tasks.sqlite3") as database:
        database.execute("CREATE TABLE tasks (status TEXT NOT NULL)")
        database.execute("INSERT INTO tasks VALUES ('running')")
    restart_path = tmp_path / "controller-restarts.sqlite3"
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker)),
        lease_path=tmp_path / "controller-lease.json",
        environment={
            **os.environ,
            "SENPAI_ROLE": "advisor",
            "RESEARCH_TAG": "maple",
            "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path),
        },
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.1,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
    )
    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "controller-lease.json")
    request = RestartRequest(
        request_id="guard-race-213",
        target=RoleTarget(research_tag="maple", role="advisor"),
        expected_conversation_id=conversation_id,
        expected_restart_control_token="previously-quiescent",
        expected_worker_generation=1,
        expected_completed_turns=0,
    )
    with RestartRequestStore(restart_path) as store:
        store.enqueue(request)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with RestartRequestStore(restart_path) as store:
            result = store.result(request.request_id)
        if result.status == "rejected":
            break
        time.sleep(0.01)
    stop.set()
    thread.join(5)

    assert results == [0]
    assert result.status == "rejected"
    assert result.rejection_code == "compute-not-quiescent"
    assert (tmp_path / "starts").read_text() == "1"


@pytest.mark.parametrize(
    ("claimed_before_crash", "expected_status", "rejection_code"),
    [
        (True, "completed", None),
        (False, "rejected", "source-generation-missed"),
    ],
)
def test_new_owner_reconciles_restart_requests_after_the_source_worker_is_gone(
    tmp_path: Path,
    claimed_before_crash: bool,
    expected_status: str,
    rejection_code: str | None,
):
    """
    Requirement: if WorkerSupervisor crashes after claiming a restart, its next
    process completes that same request after a newer worker publishes its lease.
    Interface: RestartRequestStore across supervisor process generations.
    """

    conversation_id = "00000000-0000-0000-0000-000000000212"
    restart_path = tmp_path / "controller-restarts.sqlite3"
    request = RestartRequest(
        request_id="crash-recovery-212",
        target=RoleTarget(research_tag="maple", role="advisor"),
        expected_conversation_id=conversation_id,
        expected_restart_control_token="opaque-role-authorization",
        expected_worker_generation=1,
        expected_completed_turns=0,
    )
    with RestartRequestStore(restart_path) as store:
        assert store.allocate_worker_generation() == 1
        store.enqueue(request)
        if claimed_before_crash:
            assert store.claim_next(
                request.target,
                worker_generation=1,
                replacement_generation=2,
            ) == request

    worker = tmp_path / "replacement_worker.py"
    worker.write_text(
        f"""
import json
import os
import time
from pathlib import Path

generation = int(os.environ["{GENERATION_ENV}"])
lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({{
    "pid": os.getpid(),
    "phase": "startup",
    "deadline": time.monotonic() + 30,
    "completed_turns": 0,
    "conversation_id": None,
    "generation": generation,
}}))
time.sleep(0.05)
lease.write_text(json.dumps({{
    "pid": os.getpid(),
    "phase": "sleep",
    "deadline": time.monotonic() + 30,
    "completed_turns": 0,
    "conversation_id": "{conversation_id}",
    "generation": generation,
}}))
Path({str(tmp_path / "ready")!r}).write_text(str(generation))
while True:
    time.sleep(1)
""".strip()
    )
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker)),
        lease_path=tmp_path / "controller-lease.json",
        environment={
            **os.environ,
            "SENPAI_ROLE": "advisor",
            "RESEARCH_TAG": "maple",
            "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path),
        },
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.1,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
    )
    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "ready")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with RestartRequestStore(restart_path) as store:
            result = store.result(request.request_id)
        if result.status == expected_status:
            break
        time.sleep(0.01)
    stop.set()
    thread.join(5)

    assert results == [0]
    assert result.status == expected_status
    assert result.rejection_code == rejection_code
    if claimed_before_crash:
        assert result.completion is not None
        assert result.completion.replacement_generation == 2
    else:
        assert result.completion is None


def test_health_command_reports_live_and_expired_worker_leases(tmp_path: Path):
    lease = tmp_path / "controller-lease.json"
    lease.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "phase": "sleep",
                "deadline": time.monotonic() + 30,
            }
        )
    )

    healthy = subprocess.run(
        [
            sys.executable,
            "-m",
            "senpai_agent.supervisor",
            "health",
            str(lease),
        ],
        check=False,
    )
    lease.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "phase": "openhands-turn",
                "deadline": time.monotonic() - 1,
            }
        )
    )
    expired = subprocess.run(
        [
            sys.executable,
            "-m",
            "senpai_agent.supervisor",
            "health",
            str(lease),
        ],
        check=False,
    )

    assert healthy.returncode == 0
    assert expired.returncode == 1


def test_openhands_reopens_durable_events_after_an_unclean_worker_exit(
    tmp_path: Path,
):
    conversation_id = "00000000-0000-0000-0000-000000000049"
    crash = tmp_path / "crash.py"
    crash.write_text(
        """
import os
import sys
from pathlib import Path
from uuid import UUID

from pydantic import SecretStr
from openhands.sdk import Agent, Conversation, LLM

state_dir, workspace, conversation_id = sys.argv[1:]
conversation = Conversation(
    agent=Agent(
        llm=LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key")),
        tools=[],
    ),
    workspace=Path(workspace),
    persistence_dir=Path(state_dir),
    conversation_id=UUID(conversation_id),
    visualizer=None,
    delete_on_close=False,
)
conversation.send_message("resume this durable event")
os._exit(17)
""".strip()
    )
    resume = tmp_path / "resume.py"
    resume.write_text(
        """
import sys
from pathlib import Path
from uuid import UUID

from pydantic import SecretStr
from openhands.sdk import Agent, Conversation, LLM

state_dir, workspace, conversation_id = sys.argv[1:]
conversation = Conversation(
    agent=Agent(
        llm=LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key")),
        tools=[],
    ),
    workspace=Path(workspace),
    persistence_dir=Path(state_dir),
    conversation_id=UUID(conversation_id),
    visualizer=None,
    delete_on_close=False,
)
assert any(
    "resume this durable event" in str(event)
    for event in conversation.state.view.events
)
conversation.close()
""".strip()
    )
    environment = {
        **os.environ,
        "OPENHANDS_SUPPRESS_BANNER": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    arguments = [
        str(tmp_path / "openhands-state"),
        str(tmp_path / "workspace"),
        conversation_id,
    ]

    crashed = subprocess.run(
        [sys.executable, str(crash), *arguments],
        env=environment,
        check=False,
    )
    resumed = subprocess.run(
        [sys.executable, str(resume), *arguments],
        env=environment,
        check=False,
    )

    assert crashed.returncode == 17
    assert resumed.returncode == 0
