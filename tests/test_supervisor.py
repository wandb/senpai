import errno
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from base64 import b64encode
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from pydantic import SecretStr

from openhands_support import launch_env
from senpai_agent.program_context import (
    PROGRAM_CONTENT_SHA256_ENV,
    PROGRAM_CONTEXT_FILE_ENV,
    PROGRAM_PATH_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    decode_program_system_prompt,
    encode_program_system_prompt,
)
from senpai_agent.system_instructions import (
    SYSTEM_INSTRUCTIONS_FILE_ENV,
    SYSTEM_INSTRUCTIONS_SHA256_ENV,
    decode_system_instructions,
)

import senpai_agent.supervisor as supervisor_module
from senpai_agent.supervisor import (
    ProgressLease,
    SupervisorConfig,
    WorkerLease,
    WorkerSupervisor,
    prepare_system_context_environment,
    serve_lease_health,
)


def wait_for(path: Path, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"{path} was not created")


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


def test_inference_heartbeat_is_observational_controller_state(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"
    progress = ProgressLease(lease_path)
    progress.update("openhands-turn", 300)
    initial = WorkerLease.read(lease_path)

    progress.update_llm_request(1_755_000_000.0, 1_755_000_001.0)
    active = WorkerLease.read(lease_path)

    assert active.phase == initial.phase
    assert active.deadline == initial.deadline
    assert active.completed_turns == initial.completed_turns
    assert active.llm_request_started_at == 1_755_000_000.0
    assert active.llm_request_heartbeat_at == 1_755_000_001.0

    progress.update_llm_request(1_755_000_000.0, 1_755_000_030.0)
    pulsed = WorkerLease.read(lease_path)
    assert pulsed.deadline == initial.deadline
    assert pulsed.llm_request_started_at == active.llm_request_started_at
    assert pulsed.llm_request_heartbeat_at == 1_755_000_030.0

    progress.update_llm_request(None, None)
    idle = WorkerLease.read(lease_path)
    assert idle.deadline == initial.deadline
    assert idle.llm_request_started_at is None
    assert idle.llm_request_heartbeat_at is None


def test_worker_lease_reads_legacy_state_without_inference_fields(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"
    lease_path.write_text(
        json.dumps(
            {
                "pid": 123,
                "phase": "poll",
                "deadline": 456.0,
            }
        )
    )

    lease = WorkerLease.read(lease_path)

    assert lease.llm_request_started_at is None
    assert lease.llm_request_heartbeat_at is None


def test_supervisor_preserves_launch_snapshot_across_restart(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    env = launch_env(tmp_path, program_path="senpai/program.md")
    role_template = Path(env["SENPAI_OPENHANDS_ROLE_FILE"])
    role_template.write_text(
        "Role={{ROLE}} Repo={{GH_REPO}} Project={{WANDB_PROJECT}}\n"
    )
    env.update({
        "WANDB_PROJECT": "cfd",
        "GITHUB_TOKEN": "github-secret-sentinel",
        "WANDB_API_KEY": "wandb-secret-sentinel",
    })
    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"])

    environment = prepare_system_context_environment("advisor", state_dir, env)
    role_prompt = Path(environment["SENPAI_OPENHANDS_ROLE_FILE"])
    system_context = Path(environment[SYSTEM_INSTRUCTIONS_FILE_ENV])
    instructions = decode_system_instructions(
        system_context.read_text().strip(), environment[SYSTEM_INSTRUCTIONS_SHA256_ENV]
    )

    assert role_prompt.read_text() == "Role=advisor Repo=acme/widgets Project=cfd\n"
    assert instructions.role == "Role=advisor Repo=acme/widgets Project=cfd"
    assert instructions.program.content == "# Test programme\n\nUse the target contract."
    assert instructions.program.source_commit == env[PROGRAM_SOURCE_COMMIT_ENV]
    assert role_template.read_text().startswith("Role={{ROLE}}")
    output = capsys.readouterr().out
    assert env[PROGRAM_SOURCE_COMMIT_ENV] in output
    assert environment[SYSTEM_INSTRUCTIONS_SHA256_ENV] in output
    assert "github-secret-sentinel" not in output + instructions.prompt
    assert "wandb-secret-sentinel" not in output + instructions.prompt

    (Path(env["SENPAI_OPENHANDS_WORKSPACE"]) / "senpai/program.md").write_text(
        "Workspace policy changed after launch."
    )
    restarted = prepare_system_context_environment("advisor", state_dir, env)

    assert restarted[SYSTEM_INSTRUCTIONS_SHA256_ENV] == environment[SYSTEM_INSTRUCTIONS_SHA256_ENV]
    assert decode_system_instructions(
        Path(restarted[SYSTEM_INSTRUCTIONS_FILE_ENV]).read_text().strip(),
        restarted[SYSTEM_INSTRUCTIONS_SHA256_ENV],
    ).prompt == instructions.prompt


@pytest.mark.parametrize("component", ["role", "system", "harness", "launch"])
def test_supervisor_rejects_changed_persisted_or_source_context(
    tmp_path: Path, component: str,
):
    env = launch_env(tmp_path)
    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"])
    prepared = prepare_system_context_environment("advisor", state_dir, env)
    if component == "role":
        Path(prepared["SENPAI_OPENHANDS_ROLE_FILE"]).write_text("Tampered role")
    elif component == "system":
        Path(prepared[SYSTEM_INSTRUCTIONS_FILE_ENV]).write_text("not-base64")
    elif component == "harness":
        Path(env["SENPAI_OPENHANDS_HARNESS_FILE"]).write_text("Changed harness")
    else:
        env["SENPAI_LAUNCH_CONTEXT_B64"] = b64encode(b"Changed launch").decode()

    with pytest.raises((RuntimeError, ValueError), match="snapshot|controller-held"):
        prepare_system_context_environment("advisor", state_dir, env)


def test_supervisor_checks_launch_digest_before_first_snapshot(tmp_path: Path):
    env = launch_env(tmp_path)
    program_file = Path(env[PROGRAM_CONTEXT_FILE_ENV])
    program = decode_program_system_prompt(program_file.read_text())
    program_file.write_text(
        encode_program_system_prompt(replace(program, content="Altered policy"))
    )

    with pytest.raises(RuntimeError, match=PROGRAM_CONTENT_SHA256_ENV):
        prepare_system_context_environment("advisor", tmp_path / "state", env)

    assert not (tmp_path / "state" / "system-instructions").exists()


@pytest.mark.parametrize(
    ("key", "replacement", "message"),
    [
        (PROGRAM_SOURCE_COMMIT_ENV, None, "is required"),
        (PROGRAM_CONTEXT_FILE_ENV, None, "is required"),
        (PROGRAM_CONTENT_SHA256_ENV, None, "launch snapshot"),
        (PROGRAM_SOURCE_COMMIT_ENV, "b" * 40, "launch snapshot"),
        (PROGRAM_PATH_ENV, "other/program.md", "launch snapshot"),
        (PROGRAM_CONTENT_SHA256_ENV, "0" * 64, "launch snapshot"),
    ],
)
def test_supervisor_does_not_start_worker_with_invalid_launch_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    replacement: str | None,
    message: str,
):
    env = launch_env(tmp_path)
    if replacement is None:
        env.pop(key)
    else:
        env[key] = replacement

    def unexpected_worker(*_args, **_kwargs):
        pytest.fail("worker must not start without a verified launch snapshot")

    monkeypatch.setattr(supervisor_module, "WorkerSupervisor", unexpected_worker)
    with pytest.raises(RuntimeError, match=message):
        supervisor_module.supervisor_main(["advisor"], env)


def test_supervisor_fails_before_snapshotting_a_role_with_missing_values(
    tmp_path: Path,
):
    env = launch_env(tmp_path, role="student")
    Path(env["SENPAI_OPENHANDS_ROLE_FILE"]).write_text(
        "Student={{STUDENT_NAME}} Repo={{GH_REPO}}\n"
    )

    with pytest.raises(ValueError, match="Missing SENPAI-STUDENT.md values: STUDENT_NAME"):
        prepare_system_context_environment("student", tmp_path / "state", env)

    assert not (tmp_path / "state" / "system-instructions" / "student.md").exists()


def test_supervisor_starts_the_controller_with_trusted_safe_path_python(
    tmp_path, monkeypatch
):
    environment = launch_env(tmp_path)
    token_file = tmp_path / "github-token"
    token_file.write_text("test-token")
    token_file.chmod(0o600)
    environment["SENPAI_GITHUB_TOKEN_FILE"] = str(token_file)
    commands = []

    def capture_worker(self, _stop):
        commands.append(self.command)
        return 0

    monkeypatch.setattr(WorkerSupervisor, "run", capture_worker)
    assert supervisor_module.supervisor_main(["advisor"], environment) == 0
    assert commands == [
        (sys.executable, "-P", "-m", "senpai_agent.controller", "advisor")
    ]


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

    reason, made_progress = supervisor._wait_for_worker(
        ExitedProcess(),
        {},
        threading.Event(),
        time.monotonic(),
    )

    assert reason == "exit:19"
    assert made_progress is True


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


def test_http_health_endpoint_tracks_the_worker_lease(tmp_path: Path):
    lease = tmp_path / "controller-lease.json"
    live = json.dumps({
        "pid": os.getpid(), "phase": "poll", "deadline": time.monotonic() + 30,
    })
    expired = json.dumps({
        "pid": os.getpid(), "phase": "poll", "deadline": time.monotonic() - 1,
    })
    with serve_lease_health(lease, host="127.0.0.1", port=0) as server:
        url = f"http://127.0.0.1:{server.server_port}/healthz"
        for content, status, body in (
            (None, 503, b"unhealthy\n"),
            (live, 200, b"ok\n"),
            (expired, 503, b"unhealthy\n"),
            (live, 200, b"ok\n"),
            ("incomplete lease", 503, b"unhealthy\n"),
        ):
            if content is not None:
                lease.write_text(content)
            try:
                response = urlopen(url, timeout=1)
            except HTTPError as error:
                response = error
            with response:
                assert response.status == status
                assert response.read() == body

        with pytest.raises(HTTPError) as unknown:
            urlopen(f"http://127.0.0.1:{server.server_port}/unknown", timeout=1)
        assert unknown.value.code == 404
        unknown.value.close()


def test_http_health_server_closes_when_the_supervisor_fails(tmp_path: Path):
    with pytest.raises(RuntimeError, match="worker failed"):
        with serve_lease_health(
            tmp_path / "lease.json", host="127.0.0.1", port=0
        ) as server:
            address = ("127.0.0.1", server.server_port)
            raise RuntimeError("worker failed")

    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(address, timeout=1)


def supervisor_environment(tmp_path: Path) -> dict[str, str]:
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "program.md").write_text("Research policy.")
    role = tmp_path / "ADVISOR.md"
    role.write_text("Advisor policy.")
    token = tmp_path / "github-token"
    token.write_text("test-token")
    token.chmod(0o600)
    return {
        "SENPAI_GITHUB_TOKEN_FILE": str(token),
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path / "state"),
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role),
    }


@pytest.mark.parametrize("port", ["abc", "0", "-1", "65536"])
def test_invalid_health_port_fails_before_consuming_the_token(
    tmp_path: Path, port: str
):
    environment = supervisor_environment(tmp_path)
    environment["SENPAI_HEALTH_PORT"] = port

    with pytest.raises(RuntimeError, match="SENPAI_HEALTH_PORT must be"):
        supervisor_module.supervisor_main(["advisor"], environment)

    assert Path(environment["SENPAI_GITHUB_TOKEN_FILE"]).read_text() == "test-token"


def test_unavailable_health_port_preserves_the_token_handoff(tmp_path: Path):
    environment = supervisor_environment(tmp_path)
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        environment["SENPAI_HEALTH_PORT"] = str(occupied.getsockname()[1])
        with pytest.raises(OSError) as error:
            supervisor_module.supervisor_main(["advisor"], environment)

    assert error.value.errno == errno.EADDRINUSE
    assert Path(environment["SENPAI_GITHUB_TOKEN_FILE"]).read_text() == "test-token"


def test_http_health_server_closes_idle_connections(tmp_path: Path):
    with serve_lease_health(
        tmp_path / "lease.json", host="127.0.0.1", port=0
    ) as server:
        with socket.create_connection(
            ("127.0.0.1", server.server_port), timeout=10
        ) as connection:
            assert connection.recv(1) == b""


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
