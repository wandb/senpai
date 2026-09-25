import json
import os
import subprocess
import sys
import threading
import time
from base64 import b64encode
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from git_workflow_support import commit_workspace
from pydantic import SecretStr

import senpai_agent.supervisor as supervisor_module
from senpai_agent.program_context import (
    PROGRAM_CONTEXT_FILE_ENV,
    PROGRAM_PATH_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    encode_program_system_prompt,
    load_program_system_prompt,
)
from senpai_agent.supervisor import (
    ProgressLease,
    SupervisorConfig,
    WorkerLease,
    WorkerSupervisor,
    prepare_system_context_environment,
    serve_lease_health,
)
from senpai_agent.system_instructions import (
    SYSTEM_INSTRUCTIONS_FILE_ENV,
    SYSTEM_INSTRUCTIONS_SHA256_ENV,
    decode_system_instructions,
)


def test_supervisor_default_termination_grace_is_sixty_seconds():
    assert SupervisorConfig().terminate_grace_seconds == 60


def test_private_service_handoff_files_are_consumed_once(tmp_path: Path):
    paths = {}
    environment = {}
    for credential, file_env in {
        "WANDB_API_KEY": "SENPAI_WANDB_API_KEY_FILE",
        "EXA_API_KEY": "SENPAI_EXA_API_KEY_FILE",
        "SENPAI_WANDB_TRAINING_API_KEY": (
            "SENPAI_WANDB_TRAINING_API_KEY_FILE"
        ),
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


def test_supervisor_requires_file_handoffs_for_role_credentials():
    assert supervisor_module._consume_private_credential_files(
        {
            "WANDB_API_KEY": "raw-wandb",
            "EXA_API_KEY": "raw-exa",
            "SENPAI_WANDB_TRAINING_API_KEY": "raw-training",
        }
    ) == {}

    controller_credentials = {
        "WANDB_API_KEY": SecretStr("wandb"),
        "EXA_API_KEY": SecretStr("exa"),
    }
    supervisor_module._require_private_credentials(
        "advisor", controller_credentials
    )
    with pytest.raises(
        RuntimeError,
        match="SENPAI_WANDB_TRAINING_API_KEY_FILE",
    ):
        supervisor_module._require_private_credentials(
            "student", controller_credentials
        )


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


def test_supervisor_snapshots_program_and_rendered_role_before_starting_workers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    workspace = tmp_path / "target"
    program = workspace / "senpai" / "program.md"
    program.parent.mkdir(parents=True)
    program.write_text("Research policy.")
    source_commit = commit_workspace(workspace)
    role_template = tmp_path / "ADVISOR.md"
    role_template.write_text(
        "Role={{ROLE}} Repo={{GH_REPO}} Project={{WANDB_PROJECT}}\n"
    )
    state_dir = tmp_path / "state"
    harness = tmp_path / "SENPAI-HARNESS.md"
    harness.write_text("Harness policy.\n")
    program_snapshot = load_program_system_prompt(
        workspace, "senpai/program.md", source_commit
    )
    program_context = tmp_path / "program-context.b64"
    program_context.write_text(encode_program_system_prompt(program_snapshot))
    source_environment = {
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_template),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness),
        PROGRAM_CONTEXT_FILE_ENV: str(program_context),
        PROGRAM_PATH_ENV: program_snapshot.program_path,
        PROGRAM_SOURCE_COMMIT_ENV: source_commit,
        "SENPAI_LAUNCH_CONTEXT_B64": b64encode(b"Launch policy.").decode(),
        "GH_REPO": "acme/widgets",
        "WANDB_PROJECT": "cfd",
        "GITHUB_TOKEN": "github-secret-sentinel",
        "WANDB_API_KEY": "wandb-secret-sentinel",
    }

    environment = prepare_system_context_environment(
        "advisor",
        state_dir,
        source_environment,
    )

    assert environment["SENPAI_PROGRAM_PATH"] == "senpai/program.md"
    snapshot = decode_system_instructions(
        Path(environment[SYSTEM_INSTRUCTIONS_FILE_ENV]).read_text().strip(),
        environment[SYSTEM_INSTRUCTIONS_SHA256_ENV],
    ).program
    assert snapshot == program_snapshot
    role_prompt = Path(environment["SENPAI_OPENHANDS_ROLE_FILE"])
    assert role_prompt == state_dir / "system-instructions" / "advisor.md"
    assert role_prompt.read_text() == "Role=advisor Repo=acme/widgets Project=cfd\n"
    assert role_template.read_text().startswith("Role={{ROLE}}")
    assert capsys.readouterr().out == (
        "SENPAI_PROGRAM_CONTEXT "
        f"path=senpai/program.md commit={source_commit} "
        f"sha256={snapshot.content_sha256}\n"
        f"SENPAI_SYSTEM_CONTEXT path={role_prompt.with_suffix('.context.b64')} "
        f"sha256={environment[SYSTEM_INSTRUCTIONS_SHA256_ENV]}\n"
    )

    program.write_text("Unreviewed replacement.")
    commit_workspace(workspace, "unreviewed local policy")
    (workspace / ".git").rename(tmp_path / "unavailable-target-objects")

    restarted = prepare_system_context_environment(
        "advisor",
        state_dir,
        {
            **source_environment,
            "GH_REPO": "acme/widgets",
            "WANDB_PROJECT": "cfd",
        },
    )

    assert restarted["SENPAI_OPENHANDS_ROLE_FILE"] == str(role_prompt)
    assert restarted[SYSTEM_INSTRUCTIONS_SHA256_ENV] == environment[
        SYSTEM_INSTRUCTIONS_SHA256_ENV
    ]
    assert role_prompt.read_text() == "Role=advisor Repo=acme/widgets Project=cfd\n"


def test_supervisor_rejects_a_tampered_persisted_role_snapshot(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "program.md").write_text("Research policy.")
    source_commit = commit_workspace(workspace)
    program = load_program_system_prompt(workspace, "program.md", source_commit)
    program_file = tmp_path / "program-context.b64"
    program_file.write_text(encode_program_system_prompt(program))
    role = tmp_path / "ADVISOR.md"
    role.write_text("Advisor policy.")
    harness = tmp_path / "SENPAI-HARNESS.md"
    harness.write_text("Harness policy.")
    source = {
        "SENPAI_OPENHANDS_ROLE_FILE": str(role),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness),
        PROGRAM_CONTEXT_FILE_ENV: str(program_file),
        PROGRAM_PATH_ENV: "program.md",
        PROGRAM_SOURCE_COMMIT_ENV: source_commit,
        "SENPAI_LAUNCH_CONTEXT_B64": b64encode(b"Launch policy.").decode(),
    }
    prepared = prepare_system_context_environment("advisor", tmp_path / "state", source)
    Path(prepared["SENPAI_OPENHANDS_ROLE_FILE"]).write_text("Attacker policy.")

    with pytest.raises(RuntimeError, match="persisted role prompt"):
        prepare_system_context_environment("advisor", tmp_path / "state", source)


def test_supervisor_fails_before_snapshotting_a_role_with_missing_values(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "program.md").write_text("Research policy.")
    source_commit = commit_workspace(workspace)
    role_template = tmp_path / "STUDENT.md"
    role_template.write_text("Student={{STUDENT_NAME}} Repo={{GH_REPO}}\n")
    program = load_program_system_prompt(workspace, "program.md", source_commit)
    program_file = tmp_path / "program-context.b64"
    program_file.write_text(encode_program_system_prompt(program))
    harness = tmp_path / "SENPAI-HARNESS.md"
    harness.write_text("Harness policy.")

    with pytest.raises(ValueError, match="Missing STUDENT.md values: STUDENT_NAME"):
        prepare_system_context_environment(
            "student",
            tmp_path / "state",
            {
                "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
                "SENPAI_OPENHANDS_ROLE_FILE": str(role_template),
                "SENPAI_OPENHANDS_HARNESS_FILE": str(harness),
                PROGRAM_CONTEXT_FILE_ENV: str(program_file),
                PROGRAM_PATH_ENV: "program.md",
                PROGRAM_SOURCE_COMMIT_ENV: source_commit,
                "SENPAI_LAUNCH_CONTEXT_B64": b64encode(b"Launch policy.").decode(),
                "GH_REPO": "acme/widgets",
            },
        )

    assert not (tmp_path / "state" / "system-instructions" / "student.md").exists()


def test_supervisor_does_not_start_a_worker_without_a_launch_program_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    source_commit = "a" * 40

    def unexpected_worker(*_args, **_kwargs):
        pytest.fail("worker must not be constructed when program.md is missing")

    monkeypatch.setattr(supervisor_module, "WorkerSupervisor", unexpected_worker)

    with pytest.raises(RuntimeError) as error:
        supervisor_module.supervisor_main(
            ["advisor"],
            {
                "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path / "state"),
                "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
                PROGRAM_SOURCE_COMMIT_ENV: source_commit,
            },
        )

    assert PROGRAM_CONTEXT_FILE_ENV in str(error.value)


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


def test_pid_one_terminates_and_kills_detached_descendants(monkeypatch):
    class Child:
        def __init__(self, pid):
            self.pid = pid
            self.signals = []

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

    children = [Child(41), Child(42)]
    waits = []
    monkeypatch.setattr(supervisor_module.os, "getpid", lambda: 1)
    monkeypatch.setattr(
        supervisor_module.psutil,
        "Process",
        lambda: type(
            "SupervisorProcess",
            (),
            {"children": lambda self, recursive: children},
        )(),
    )

    def wait_procs(processes, timeout):
        waits.append((list(processes), timeout))
        return ([], [children[1]]) if len(waits) == 1 else (list(processes), [])

    monkeypatch.setattr(supervisor_module.psutil, "wait_procs", wait_procs)
    supervisor = WorkerSupervisor(
        command=("worker",),
        lease_path=Path("lease.json"),
        config=SupervisorConfig(terminate_grace_seconds=7),
    )

    supervisor._terminate_adopted_children()

    assert children[0].signals == ["terminate"]
    assert children[1].signals == ["terminate", "kill"]
    assert [timeout for _processes, timeout in waits] == [7, 7]


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
    "exa": "SENPAI_EXA_API_KEY_FD",
    "training": "SENPAI_WANDB_TRAINING_API_KEY_FD",
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
            "EXA_API_KEY": SecretStr("exa-sentinel"),
            "SENPAI_WANDB_TRAINING_API_KEY": SecretStr(
                "wandb-training-sentinel"
            ),
        },
        environment={
            **os.environ,
            "GITHUB_TOKEN": "must-not-survive",
            "GH_TOKEN": "must-not-survive",
            "WANDB_API_KEY": "must-not-survive",
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
            "exa": "exa-sentinel",
            "training": "wandb-training-sentinel",
        }
        for item in observations
    )
    assert all(item["github_env"] is None for item in observations)
    assert all(item["gh_env"] is None for item in observations)
    assert all(item["wandb_env"] is None for item in observations)
    assert all(item["exa_env"] is None for item in observations)
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


def test_http_health_endpoint_reports_the_in_process_worker_lease(tmp_path: Path):
    lease = tmp_path / "controller-lease.json"
    with serve_lease_health(lease, host="127.0.0.1", port=0) as server:
        url = f"http://127.0.0.1:{server.server_port}/healthz"
        with pytest.raises(HTTPError) as missing:
            urlopen(url, timeout=1)
        assert missing.value.code == 503

        lease.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "phase": "poll",
                    "deadline": time.monotonic() + 30,
                }
            )
        )
        with urlopen(url, timeout=1) as response:
            assert response.status == 200
            assert response.read() == b"ok\n"

        with pytest.raises(HTTPError) as unknown:
            urlopen(
                f"http://127.0.0.1:{server.server_port}/unknown",
                timeout=1,
            )
        assert unknown.value.code == 404


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
