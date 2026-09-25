import errno
import json
import os
import socket
import subprocess
import sys
import threading
import time
from base64 import b64encode
from dataclasses import replace
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
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


def test_supervisor_default_termination_grace_is_sixty_seconds():
    assert SupervisorConfig().terminate_grace_seconds == 60


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


def test_one_worker_consumes_credentials_once_and_forces_a_container_restart(
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
private = {}
for name, fd_name in {
    "wandb": "SENPAI_WANDB_API_KEY_FD",
    "writer": "SENPAI_WANDB_TRAINING_API_KEY_FD",
    "exa": "SENPAI_EXA_API_KEY_FD",
}.items():
    with os.fdopen(int(os.environ[fd_name])) as stream:
        private[name] = stream.read()
with (state / "observations").open("a") as output:
    output.write(json.dumps({
        "token": token,
        "private": private,
        "github_env": os.environ.get("GITHUB_TOKEN"),
        "gh_env": os.environ.get("GH_TOKEN"),
        "wandb_env": os.environ.get("WANDB_API_KEY"),
        "exa_env": os.environ.get("EXA_API_KEY"),
        "writer_env": os.environ.get("SENPAI_WANDB_TRAINING_API_KEY"),
        "token_file_env": os.environ.get("SENPAI_GITHUB_TOKEN_FILE"),
    }) + "\\n")

lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "ready",
    "deadline": time.monotonic() + 30,
}))
while not (state / "release").exists():
    time.sleep(0.01)
raise SystemExit(19)
""".strip()
    )
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        github_token=SecretStr("write-token-sentinel"),
        private_credentials={
            "WANDB_API_KEY": SecretStr("wandb-controller-sentinel"),
            "SENPAI_WANDB_TRAINING_API_KEY": SecretStr("writer-sentinel"),
            "EXA_API_KEY": SecretStr("exa-sentinel"),
        },
        environment={
            **os.environ,
            "GITHUB_TOKEN": "must-not-survive",
            "GH_TOKEN": "must-not-survive",
            "WANDB_API_KEY": "must-not-survive",
            "SENPAI_WANDB_TRAINING_API_KEY": "must-not-survive",
            "EXA_API_KEY": "must-not-survive",
        },
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.1,
        ),
    )

    results = []
    thread = threading.Thread(target=lambda: results.append(supervisor.run(stop)))
    thread.start()
    deadline = time.monotonic() + 5
    while not (tmp_path / "observations").exists():
        if time.monotonic() > deadline:
            pytest.fail("worker did not consume credential handoffs")
        time.sleep(0.01)

    assert supervisor.github_token is None
    assert supervisor.private_credentials == {}
    assert supervisor.environment == {}
    (tmp_path / "release").write_text("release")
    thread.join(5)

    assert not thread.is_alive()
    assert results == [19]
    observations = [
        json.loads(line)
        for line in (tmp_path / "observations").read_text().splitlines()
    ]
    assert len(observations) == 1
    assert (tmp_path / "starts").read_text() == "1"
    assert all(item["token"] == "write-token-sentinel" for item in observations)
    assert all(
        item["private"]
        == {
            "wandb": "wandb-controller-sentinel",
            "writer": "writer-sentinel",
            "exa": "exa-sentinel",
        }
        for item in observations
    )
    assert all(item["github_env"] is None for item in observations)
    assert all(item["gh_env"] is None for item in observations)
    assert all(item["wandb_env"] is None for item in observations)
    assert all(item["exa_env"] is None for item in observations)
    assert all(item["writer_env"] is None for item in observations)
    assert all(item["token_file_env"] is None for item in observations)
    assert not list(tmp_path.glob(".github-token-*"))


def test_overdue_worker_is_killed_without_an_in_container_restart(tmp_path: Path):
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
    "phase": "wedged-turn",
    "deadline": time.monotonic() + 0.05,
}))
temporary.replace(lease)

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
        ),
    )

    assert supervisor.run(stop) == 1
    assert (tmp_path / "starts").read_text() == "1"


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
    environment = launch_env(tmp_path, program_content="Research policy.")
    Path(environment["SENPAI_OPENHANDS_ROLE_FILE"]).write_text("Advisor policy.")
    token = tmp_path / "github-token"
    token.write_text("test-token")
    token.chmod(0o600)
    environment["SENPAI_GITHUB_TOKEN_FILE"] = str(token)
    return environment


def test_supervisor_entrypoint_forwards_private_credentials_and_clears_parent_keys(
    tmp_path, monkeypatch
):
    environment = supervisor_environment(tmp_path)
    environment.update(
        {
            "SENPAI_HEALTH_PORT": "8765",
            "ANTHROPIC_API_KEY": "anthropic-fixture",
            "OPENAI_API_KEY": "openai-fixture",
            "WANDB_API_KEY": "wandb-current-fixture",
            "EXA_API_KEY": "exa-current-fixture",
            "GITHUB_TOKEN": "github-current-fixture",
            "GH_TOKEN": "gh-current-fixture",
        }
    )
    handoffs = [Path(environment["SENPAI_GITHUB_TOKEN_FILE"])]
    for name, file_env in (
        ("wandb", "SENPAI_WANDB_API_KEY_FILE"),
        ("exa", "SENPAI_EXA_API_KEY_FILE"),
    ):
        path = tmp_path / f"{name}-handoff"
        path.write_text(f"{name}-private-fixture")
        path.chmod(0o600)
        environment[file_env] = str(path)
        handoffs.append(path)
    model_names = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "WANDB_API_KEY"}
    for profile in ("", "SMART_", "FAST_", "FRONTIER_"):
        name = f"TEST_{profile}MODEL_CREDENTIAL"
        environment[f"SENPAI_OPENHANDS_{profile}API_KEY_ENV"] = f" {name} "
        environment[name] = f"{profile.lower()}model-fixture"
        model_names.add(name)
    cleared_names = model_names | {
        "EXA_API_KEY", "GITHUB_TOKEN", "GH_TOKEN",
        "SENPAI_GITHUB_TOKEN_FILE", "SENPAI_GITHUB_TOKEN_FD",
        "SENPAI_WANDB_API_KEY_FILE", "SENPAI_WANDB_API_KEY_FD",
        "SENPAI_EXA_API_KEY_FILE", "SENPAI_EXA_API_KEY_FD",
        "SENPAI_MODEL_CREDENTIALS_FD",
    }
    for name in cleared_names:
        # Register every key that supervisor_main removes so pytest restores
        # any pre-existing operator environment after this test.
        monkeypatch.setenv(name, environment.get(name, "stale-handoff"))
    observed = {}

    @contextmanager
    def health_listener(lease_path, *, port):
        assert port == 8765
        with serve_lease_health(lease_path, host="127.0.0.1", port=0) as server:
            observed["address"] = ("127.0.0.1", server.server_port)
            yield server

    def run_worker(worker, stop):
        observed["worker"] = worker
        assert not stop.is_set()
        assert all(name not in os.environ for name in cleared_names)
        assert all(not path.exists() for path in handoffs)
        address, port = observed["address"]
        with pytest.raises(HTTPError) as unhealthy:
            urlopen(f"http://{address}:{port}/healthz", timeout=1)
        assert unhealthy.value.code == 503
        unhealthy.value.close()
        return 23

    monkeypatch.setattr(supervisor_module, "serve_lease_health", health_listener)
    # Keep the real constructor: it must copy state before supervisor_main
    # clears its temporary dictionaries. Existing tests exercise worker exec.
    monkeypatch.setattr(WorkerSupervisor, "run", run_worker)

    assert supervisor_module.supervisor_main(["advisor"], environment) == 23

    worker = observed["worker"]
    assert worker.github_token.get_secret_value() == "test-token"
    assert {
        name: value.get_secret_value()
        for name, value in worker.private_credentials.items()
    } == {"WANDB_API_KEY": "wandb-private-fixture", "EXA_API_KEY": "exa-private-fixture"}
    # W&B is restored through its private service handoff. Other model keys
    # remain in the prepared worker environment even after parent cleanup.
    for name in model_names - {"WANDB_API_KEY"}:
        assert worker.environment[name] == environment[name]
    assert "WANDB_API_KEY" not in worker.environment
    assert "EXA_API_KEY" not in worker.environment
    assert worker.environment["SENPAI_PROGRAM_PATH"] == "program.md"
    assert Path(worker.environment["SENPAI_OPENHANDS_ROLE_FILE"]).read_text() == "Advisor policy.\n"
    assert worker.command == (sys.executable, "-P", "-m", "senpai_agent.controller", "advisor")
    assert worker.lease_path == tmp_path / "state" / "controller-lease.json"
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(observed["address"], timeout=1)


@pytest.mark.parametrize(
    ("credential", "file_env", "sibling_file_env"),
    [
        ("WANDB_API_KEY", "SENPAI_WANDB_API_KEY_FILE", "SENPAI_EXA_API_KEY_FILE"),
        ("EXA_API_KEY", "SENPAI_EXA_API_KEY_FILE", "SENPAI_WANDB_API_KEY_FILE"),
    ],
)
def test_configured_service_requires_private_handoff_and_cleans_siblings(
    tmp_path, monkeypatch, credential, file_env, sibling_file_env
):
    environment = supervisor_environment(tmp_path)
    environment[credential] = "raw-service-fixture"
    sibling = tmp_path / "sibling-handoff"
    sibling.write_text("private-sibling-fixture")
    sibling.chmod(0o600)
    environment[sibling_file_env] = str(sibling)
    monkeypatch.setattr(
        supervisor_module, "serve_lease_health", lambda *_args, **_kwargs: nullcontext()
    )

    def unexpected_worker(*_args, **_kwargs):
        pytest.fail("raw service credentials must fail before worker construction")

    monkeypatch.setattr(supervisor_module, "WorkerSupervisor", unexpected_worker)

    with pytest.raises(RuntimeError, match=file_env):
        supervisor_module.supervisor_main(["advisor"], environment)

    assert not sibling.exists()
    assert not Path(environment["SENPAI_GITHUB_TOKEN_FILE"]).exists()


@pytest.mark.parametrize("port", ["abc", "0", "-1", "65536"])
def test_invalid_health_port_discards_the_token_handoff(
    tmp_path: Path, port: str
):
    environment = supervisor_environment(tmp_path)
    environment["SENPAI_HEALTH_PORT"] = port

    with pytest.raises(RuntimeError, match="SENPAI_HEALTH_PORT must be"):
        supervisor_module.supervisor_main(["advisor"], environment)

    assert not Path(environment["SENPAI_GITHUB_TOKEN_FILE"]).exists()


def test_unavailable_health_port_discards_handoff_without_reading_it(
    tmp_path: Path, monkeypatch,
):
    environment = supervisor_environment(tmp_path)
    monkeypatch.setattr(
        supervisor_module, "_consume_github_token",
        lambda _env: pytest.fail("credentials read before health listener bound"),
    )
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        environment["SENPAI_HEALTH_PORT"] = str(occupied.getsockname()[1])
        with pytest.raises(OSError) as error:
            supervisor_module.supervisor_main(["advisor"], environment)

    assert error.value.errno == errno.EADDRINUSE
    assert not Path(environment["SENPAI_GITHUB_TOKEN_FILE"]).exists()


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


def test_private_service_handoff_files_are_consumed_once(tmp_path: Path):
    paths = {}
    environment = {}
    for credential, file_env in {
        "WANDB_API_KEY": "SENPAI_WANDB_API_KEY_FILE",
        "SENPAI_WANDB_TRAINING_API_KEY": "SENPAI_WANDB_TRAINING_API_KEY_FILE",
        "EXA_API_KEY": "SENPAI_EXA_API_KEY_FILE",
    }.items():
        path = tmp_path / credential.lower()
        path.write_text(f"{credential}-value")
        path.chmod(0o600)
        paths[credential] = path
        environment[file_env] = str(path)

    credentials = supervisor_module._consume_private_credential_files(environment)

    assert {
        name: value.get_secret_value() for name, value in credentials.items()
    } == {name: f"{name}-value" for name in paths}
    assert all(not path.exists() for path in paths.values())


@pytest.mark.parametrize("failure", ["github", "wandb", "context"])
def test_failed_supervisor_start_removes_all_credential_handoffs(tmp_path, failure):
    environment = launch_env(tmp_path, program_content="Research policy.")
    if failure == "context":
        environment[PROGRAM_CONTENT_SHA256_ENV] = "0" * 64
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir(mode=0o700)
    paths = []
    for name, file_env in (
        ("github", "SENPAI_GITHUB_TOKEN_FILE"),
        ("wandb", "SENPAI_WANDB_API_KEY_FILE"),
        ("exa", "SENPAI_EXA_API_KEY_FILE"),
    ):
        path = handoff_dir / name
        path.write_text("" if failure == name else f"{name}-fixture")
        path.chmod(0o600)
        environment[file_env] = str(path)
        paths.append(path)
    expected_error = {
        "github": "GitHub token handoff is empty",
        "wandb": "SENPAI_WANDB_API_KEY_FILE is empty",
        "context": "SENPAI_PROGRAM_CONTENT_SHA256 does not match the launch snapshot",
    }[failure]

    with pytest.raises(RuntimeError, match=expected_error):
        supervisor_module.supervisor_main(["advisor"], environment)

    assert all(not path.exists() for path in paths)
    assert handoff_dir.is_dir()


def test_private_service_handoffs_do_not_follow_symlinks(tmp_path: Path):
    target = tmp_path / "credential"
    target.write_text("secret")
    target.chmod(0o600)
    handoff = tmp_path / "handoff"
    handoff.symlink_to(target)

    with pytest.raises(RuntimeError, match="owner-only regular file"):
        supervisor_module._consume_private_credential_files(
            {"SENPAI_EXA_API_KEY_FILE": str(handoff)}
        )

    assert target.read_text() == "secret"


def test_pid_one_reserves_reap_time_after_worker_exhausts_term_budget(
    tmp_path, monkeypatch,
):
    class Child:
        pid = 41

        def __init__(self):
            self.signals = []

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

    child = Child()
    clock = [100.0]
    worker_waits = []
    adopted_waits = []
    monkeypatch.setattr(
        supervisor_module, "time", SimpleNamespace(monotonic=lambda: clock[0]),
    )
    monkeypatch.setattr(
        supervisor_module, "os", SimpleNamespace(**(vars(os) | {"getpid": lambda: 1})),
    )
    monkeypatch.setattr(
        supervisor_module.psutil, "Process",
        lambda: SimpleNamespace(children=lambda recursive: [child]),
    )
    monkeypatch.setattr(
        supervisor_module.subprocess, "Popen", lambda *args, **kwargs: object(),
    )

    def wait_procs(processes, timeout):
        adopted_waits.append(timeout)
        clock[0] += timeout
        return ([], processes) if len(adopted_waits) == 1 else (processes, [])

    def exhaust_worker_term_budget(_process, _descendants, deadline):
        worker_waits.append(deadline - clock[0])
        clock[0] = deadline

    monkeypatch.setattr(supervisor_module.psutil, "wait_procs", wait_procs)
    supervisor = WorkerSupervisor(
        command=("worker",),
        lease_path=tmp_path / "lease.json",
        config=SupervisorConfig(terminate_grace_seconds=7),
        environment={},
    )
    monkeypatch.setattr(
        supervisor, "_wait_for_worker", lambda *args: ("exit:17", False),
    )
    monkeypatch.setattr(supervisor, "_terminate_worker", exhaust_worker_term_budget)
    monkeypatch.setattr(supervisor, "_reap_orphaned_children", lambda _worker: None)

    assert supervisor.run() == 17

    assert child.signals == ["terminate", "kill"]
    assert len(adopted_waits) == 2
    assert adopted_waits[1] > 0, "SIGKILL must retain time to observe child exit"
    assert 0 <= sum(worker_waits + adopted_waits) <= 7
    assert clock[0] <= 107


def test_pid_one_cleanup_reaps_term_ignoring_child_before_return(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, signal, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import psutil
import senpai_agent.supervisor as supervisor

# Simulate only this module's PID-1 gate. psutil must use the real subprocess PID.
supervisor.os = SimpleNamespace(**(vars(os) | {"getpid": lambda: 1}))
child = subprocess.Popen(
    [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"],
    stdout=subprocess.PIPE, text=True, start_new_session=True,
)
try:
    assert child.stdout.readline().strip() == "ready"
    worker = supervisor.WorkerSupervisor(command=("unused",), lease_path=Path(sys.argv[1]))
    started = time.monotonic()
    worker._terminate_adopted_children(started + 0.02, started + 0.5)
    assert not psutil.pid_exists(child.pid), "child PID survived adopted cleanup return"
finally:
    if child.poll() is None:
        child.kill()
    child.wait(timeout=5)
    child.stdout.close()
""",
            str(tmp_path / "lease.json"),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failure", ["allocation", "spawn"])
def test_failed_worker_start_closes_handoffs_and_forgets_credentials(
    tmp_path, monkeypatch, failure,
):
    descriptors = []
    original_dup = os.dup

    def duplicate(fd):
        if failure == "allocation" and descriptors:
            raise OSError("handoff failed")
        result = original_dup(fd)
        descriptors.append(result)
        return result

    def spawn(*args, **kwargs):
        assert failure == "spawn"
        raise OSError("handoff failed")

    monkeypatch.setattr(supervisor_module.os, "dup", duplicate)
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", spawn)
    supervisor = WorkerSupervisor(
        command=("worker",),
        lease_path=tmp_path / "lease.json",
        github_token=SecretStr("github-secret"),
        private_credentials={"WANDB_API_KEY": SecretStr("wandb-secret")},
        environment={"ANTHROPIC_API_KEY": "model-secret"},
    )

    with pytest.raises(OSError, match="handoff failed"):
        supervisor.run()

    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert supervisor.github_token is None
    assert supervisor.private_credentials == {}
    assert supervisor.environment == {}


def test_requested_stop_forgets_credentials_without_starting_worker(tmp_path):
    stop = threading.Event()
    stop.set()
    supervisor = WorkerSupervisor(
        command=("must-not-run",),
        lease_path=tmp_path / "lease.json",
        github_token=SecretStr("github-secret"),
        private_credentials={"EXA_API_KEY": SecretStr("exa-secret")},
        environment={"OPENAI_API_KEY": "model-secret"},
    )
    assert supervisor.run(stop) == 0
    assert supervisor.github_token is None
    assert supervisor.private_credentials == {}
    assert supervisor.environment == {}


@pytest.mark.parametrize(
    ("command", "expected"),
    [("pass", 1), ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", 143)],
)
def test_worker_exit_reports_status_to_external_process_manager(tmp_path, command, expected):
    supervisor = WorkerSupervisor(
        command=(sys.executable, "-c", command),
        lease_path=tmp_path / "lease.json",
        config=SupervisorConfig(check_interval_seconds=0.01),
    )
    assert supervisor.run() == expected


@pytest.mark.parametrize("kind", ["fifo", "public", "empty"])
def test_invalid_private_handoff_fails_promptly_and_is_unlinked(tmp_path, kind):
    handoff = tmp_path / "credential"
    if kind == "fifo":
        os.mkfifo(handoff, 0o600)
    else:
        handoff.write_text("" if kind == "empty" else "secret")
        handoff.chmod(0o644 if kind == "public" else 0o600)
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
from senpai_agent.supervisor import _consume_private_credential_files
try:
    _consume_private_credential_files({"SENPAI_EXA_API_KEY_FILE": sys.argv[1]})
except RuntimeError:
    raise SystemExit(23)
""", str(handoff)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 23, result.stderr.decode()
    assert not handoff.exists()


@pytest.mark.parametrize("shutdown", ["stop", "overdue"])
def test_supervisor_cleans_up_a_detached_term_ignoring_child(tmp_path, shutdown):
    child_code = (
        "import os,signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    worker = tmp_path / "worker.py"
    worker.write_text(f'''
import json, os, subprocess, sys, time
from pathlib import Path
state = Path(sys.argv[1])
subprocess.Popen([sys.executable, "-c", {child_code!r}, str(state / "child")],
                 start_new_session=True)
while not (state / "child").exists():
    time.sleep(0.01)
Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"]).write_text(json.dumps({{
    "pid": os.getpid(), "phase": "ready",
    "deadline": time.monotonic() + {30 if shutdown == 'stop' else 0.2},
}}))
while True:
    time.sleep(1)
''')
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "lease.json",
        config=SupervisorConfig(check_interval_seconds=0.01, terminate_grace_seconds=0.1),
    )
    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "lease.json")
    child = supervisor_module.psutil.Process(int((tmp_path / "child").read_text()))
    try:
        if shutdown == "stop":
            stop.set()
        thread.join(5)
        assert not thread.is_alive()
        assert results == [0 if shutdown == "stop" else 1]
        deadline = time.monotonic() + 2
        while child.is_running() and child.status() != supervisor_module.psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "detached child survived cleanup"
            time.sleep(0.01)
    finally:
        stop.set()
        if child.is_running():
            child.kill()
        thread.join(5)
