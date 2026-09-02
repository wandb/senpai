import json
import os
import subprocess
import sys
import time
import uuid
from base64 import b64decode, b64encode
from pathlib import Path

import psutil
import pytest
from git_workflow_support import commit_workspace
from openhands.sdk.llm import Message, TextContent
from openhands_support import TEST_LAUNCH_CONTEXT, runtime_config

import senpai_agent.delegation as delegation_module

from senpai_agent.delegation import (
    AgentTask,
    DelegationConfig,
    DelegationRegistry,
    DelegationRequest,
    OpenHandsChildProcess,
    record_delegated_task_result,
    render_child_prompt,
    run_child_process,
)
from senpai_agent.launch_context import (
    INSTRUCTIONS_ROOT,
    LAUNCH_CONTEXT_ENV,
    PLACEHOLDER,
)
from senpai_agent.local_events import LocalEventStore
from senpai_agent.openhands_runner import delegation_config as runner_delegation_config
from senpai_agent.program_context import (
    PROGRAM_CONTEXT_FILE_ENV,
    PROGRAM_PATH_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    ProgramSystemPrompt,
    encode_program_system_prompt,
    load_program_system_prompt,
)
from senpai_agent.secrets import (
    CUSTOM_SECRET_ENV_NAMES_ENV,
    MODEL_CREDENTIALS_FD_ENV,
    consume_model_credential_fd,
)
from senpai_agent.supervisor import prepare_system_context_environment
from senpai_agent.system_instructions import (
    SYSTEM_INSTRUCTIONS_FILE_ENV,
    SYSTEM_INSTRUCTIONS_SHA256_ENV,
    decode_system_instructions,
)


def delegation_request(
    *,
    parent_context: tuple[Message, ...] | None = None,
    agent: str = "explore",
    model: str = "fast",
    search_mode: str | None = None,
) -> DelegationRequest:
    return DelegationRequest(
        task_id=str(uuid.uuid4()),
        parent_conversation_id=str(uuid.uuid4()),
        parent_context=parent_context
        if parent_context is not None
        else (
            Message(
                role="user",
                content=[
                    TextContent(text="Inspect PR #17."),
                    TextContent(text="Progressively disclosed skill body."),
                ],
            ),
            Message(
                role="assistant",
                content=[TextContent(text="I will compare the evidence.")],
            ),
        ),
        agent=agent,
        model=model,
        search_mode=search_mode,
    )


def delegation_config(tmp_path: Path, **updates) -> DelegationConfig:
    values = {
        "python_executable": Path(sys.executable),
        "workspace": tmp_path / "target",
        "state_dir": tmp_path / "state",
        "smart_model": "anthropic/claude-opus-4-8",
        "smart_reasoning_effort": "xhigh",
        "smart_api_key_env": "ANTHROPIC_API_KEY",
        "smart_api_key": "anthropic-secret",
        "fast_model": "anthropic/claude-haiku-4-5",
        "fast_reasoning_effort": "low",
        "fast_api_key_env": "ANTHROPIC_API_KEY",
        "fast_api_key": "anthropic-secret",
        "frontier_model": "openai/gpt-5.6",
        "frontier_reasoning_effort": "max",
        "frontier_api_key_env": "OPENAI_API_KEY",
        "frontier_api_key": "openai-secret",
        "compaction_trigger_tokens": 200_000,
        "github_repo": "acme/widgets",
        "github_trusted_actor": None,
        "role_file": tmp_path / "ADVISOR.md",
        "harness_file": tmp_path / "SENPAI-HARNESS.md",
        "plugin_dir": tmp_path / "plugin",
        "enable_browser": True,
        "conversation_secrets": {"PRIVATE_AUTH": "private-secret"},
        "role": "advisor",
        "harness_context": "harness instructions",
        "role_context": "advisor role",
        "program": ProgramSystemPrompt(
            program_path="program.md",
            source_commit="a" * 40,
            content="Research policy.",
        ),
        "launch_context": "# Authoritative launch context\n\nSystem policy.",
    }
    values.update(updates)
    return DelegationConfig(**values)


def test_child_prompt_contains_complete_snapshot_and_task():
    request = delegation_request()

    prompt = render_child_prompt(request, "Review the result and report next steps.")

    assert "Review the result and report next steps." in prompt
    assert "Inspect PR #17." in prompt
    assert "Progressively disclosed skill body." in prompt
    assert "I will compare the evidence." in prompt
    payload = prompt.split("<parent_context_json>\n", 1)[1].split(
        "\n</parent_context_json>", 1
    )[0]
    assert [message["role"] for message in json.loads(payload)] == [
        "user",
        "assistant",
    ]


def test_context_free_search_prompt_contains_mode_and_task():
    request = delegation_request(
        parent_context=(),
        agent="search",
        model="smart",
        search_mode="research-publications",
    )

    prompt = render_child_prompt(request, "Find neural operator papers.")

    assert "Search mode: research-publications" in prompt
    assert "Find neural operator papers." in prompt
    assert "parent_context_json" not in prompt


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group deadline")
def test_optional_process_deadline_kills_an_uncooperative_group(tmp_path: Path):
    pid_file = tmp_path / "pid"
    code = (
        "import os,pathlib,signal,time;"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="runtime"):
        run_child_process(
            (sys.executable, "-c", code),
            input_text="",
            env=dict(os.environ),
            timeout_seconds=1,
            terminate_grace_seconds=0.05,
        )

    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_child_command_selects_agent_model_effort_and_credential(tmp_path: Path):
    config = delegation_config(tmp_path)

    fast = OpenHandsChildProcess(
        config,
        delegation_request(agent="bash-runner"),
    )
    smart = OpenHandsChildProcess(
        config,
        delegation_request(agent="search", model="smart", search_mode="general-web"),
    )
    frontier = OpenHandsChildProcess(
        config,
        delegation_request(agent="general-purpose", model="frontier"),
    )

    assert "--agent" in fast.command
    assert fast.command[fast.command.index("--agent") + 1] == "bash-runner"
    assert "anthropic/claude-haiku-4-5" in fast.command
    assert fast.command[fast.command.index("--reasoning-effort") + 1] == "low"
    assert fast.command[fast.command.index("--api-key-env") + 1] == (
        "ANTHROPIC_API_KEY"
    )
    assert fast.command[fast.command.index("--role-file") + 1] == str(
        config.role_file
    )
    assert "anthropic/claude-opus-4-8" in smart.command
    assert smart.command[smart.command.index("--reasoning-effort") + 1] == "xhigh"
    assert "--child" in smart.command
    assert smart.command[smart.command.index("--plugin-dir") + 1] == str(
        config.plugin_dir
    )
    assert "openai/gpt-5.6" in frontier.command
    assert frontier.command[frontier.command.index("--reasoning-effort") + 1] == "max"
    assert frontier.command[frontier.command.index("--api-key-env") + 1] == (
        "OPENAI_API_KEY"
    )
    assert fast.state_dir.parent == config.state_dir / "children"
    assert "ANTHROPIC_API_KEY" not in fast.environment
    assert "OPENAI_API_KEY" not in fast.environment
    assert fast.environment["SENPAI_OPENHANDS_API_KEY_ENV"] == "ANTHROPIC_API_KEY"
    assert frontier.environment["SENPAI_OPENHANDS_API_KEY_ENV"] == "OPENAI_API_KEY"
    assert fast.environment["GH_REPO"] == "acme/widgets"
    assert "GITHUB_TOKEN" not in fast.environment
    assert "GH_TOKEN" not in fast.environment
    assert "EXA_API_KEY" not in fast.environment
    assert fast.environment["SENPAI_OPENHANDS_SMART_MODEL"] == config.smart_model
    assert fast.environment["SENPAI_OPENHANDS_SMART_API_KEY_ENV"] == (
        config.smart_api_key_env
    )
    assert fast.environment["SENPAI_OPENHANDS_FAST_MODEL"] == config.fast_model
    assert fast.environment["SENPAI_OPENHANDS_FAST_API_KEY_ENV"] == (
        config.fast_api_key_env
    )
    assert fast.environment["SENPAI_OPENHANDS_FRONTIER_MODEL"] == (
        config.frontier_model
    )
    assert fast.environment["SENPAI_OPENHANDS_FRONTIER_API_KEY_ENV"] == (
        config.frontier_api_key_env
    )
    assert fast.environment["SENPAI_OPENHANDS_SMART_REASONING_EFFORT"] == (
        config.smart_reasoning_effort
    )
    assert fast.environment["SENPAI_OPENHANDS_FAST_REASONING_EFFORT"] == (
        config.fast_reasoning_effort
    )
    assert fast.environment["SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT"] == (
        config.frontier_reasoning_effort
    )
    assert fast.environment["SENPAI_COMPACTION_TRIGGER_TOKENS"] == "200000"


def test_child_environment_carries_the_resolved_program_path(tmp_path: Path):
    child = OpenHandsChildProcess(
        delegation_config(
            tmp_path,
            program=ProgramSystemPrompt(
                program_path="senpai/program.md",
                source_commit="b" * 40,
                content="Nested research policy.",
            ),
        ),
        delegation_request(),
    )

    environment = child.environment
    assert environment[PROGRAM_PATH_ENV] == "senpai/program.md"
    assert environment[PROGRAM_SOURCE_COMMIT_ENV] == "b" * 40
    assert decode_system_instructions(
        Path(environment[SYSTEM_INSTRUCTIONS_FILE_ENV]).read_text().strip(),
        environment[SYSTEM_INSTRUCTIONS_SHA256_ENV],
    ).program == ProgramSystemPrompt(
        program_path="senpai/program.md",
        source_commit="b" * 40,
        content="Nested research policy.",
    )
    assert (
        b64decode(environment[LAUNCH_CONTEXT_ENV], validate=True).decode()
        == "# Authoritative launch context\n\nSystem policy."
    )


def test_child_restart_rejects_a_tampered_system_snapshot(tmp_path: Path):
    child = OpenHandsChildProcess(delegation_config(tmp_path), delegation_request())
    environment = child.environment
    Path(environment[SYSTEM_INSTRUCTIONS_FILE_ENV]).write_text("attacker policy")

    with pytest.raises(ValueError, match="controller-held"):
        _ = child.environment


def test_maximum_program_snapshot_is_transported_by_file_not_environment(
    tmp_path: Path,
):
    child = OpenHandsChildProcess(
        delegation_config(
            tmp_path,
            program=ProgramSystemPrompt(
                program_path="program.md",
                source_commit="a" * 40,
                content="\\" * (64 * 1024),
            ),
        ),
        delegation_request(),
    )

    environment = child.environment

    assert max(
        len(f"{name}={value}".encode()) for name, value in environment.items()
    ) < (128 * 1024)
    assert Path(environment[SYSTEM_INSTRUCTIONS_FILE_ENV]).stat().st_size > 128 * 1024


def test_child_reuses_the_supervisor_rendered_role_prompt(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "program.md").write_text("Research policy.\n")
    source_commit = commit_workspace(workspace)
    program = load_program_system_prompt(workspace, "program.md", source_commit)
    program_file = tmp_path / "program-context.b64"
    program_file.write_text(encode_program_system_prompt(program))
    prepared = prepare_system_context_environment(
        "advisor",
        tmp_path / "state",
        {
            "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
            "SENPAI_OPENHANDS_ROLE_FILE": str(INSTRUCTIONS_ROOT / "ADVISOR.md"),
            "SENPAI_OPENHANDS_HARNESS_FILE": str(
                INSTRUCTIONS_ROOT / "SENPAI-HARNESS.md"
            ),
            PROGRAM_CONTEXT_FILE_ENV: str(program_file),
            PROGRAM_PATH_ENV: "program.md",
            PROGRAM_SOURCE_COMMIT_ENV: source_commit,
            LAUNCH_CONTEXT_ENV: b64encode(TEST_LAUNCH_CONTEXT.encode()).decode(),
            "GH_REPO": "acme/widgets",
            "ADVISOR_BRANCH": "research",
            "WANDB_ENTITY": "acme",
            "WANDB_PROJECT": "cfd",
            "STUDENT_NAMES": "fern,frieren",
            "GPUS_PER_STUDENT": "2",
            "GITHUB_TOKEN": "github-secret-sentinel",
            "WANDB_API_KEY": "wandb-secret-sentinel",
        },
    )
    role_file = Path(prepared["SENPAI_OPENHANDS_ROLE_FILE"])
    prepared_instructions = decode_system_instructions(
        Path(prepared[SYSTEM_INSTRUCTIONS_FILE_ENV]).read_text().strip(),
        prepared[SYSTEM_INSTRUCTIONS_SHA256_ENV],
    )
    parent = runtime_config(
        tmp_path,
        role_file=role_file,
        harness_file=Path(prepared["SENPAI_OPENHANDS_HARNESS_FILE"]),
        instructions=prepared_instructions,
    )
    delegated = runner_delegation_config(parent)
    child = OpenHandsChildProcess(
        delegated,
        delegation_request(),
    )

    assert delegated.role_file == parent.role_file
    assert delegated.harness_file == parent.harness_file
    assert delegated.program == parent.instructions.program
    assert delegated.launch_context == parent.instructions.launch
    assert child.command[child.command.index("--role-file") + 1] == str(role_file)
    assert child.command[child.command.index("--harness-file") + 1] == str(
        parent.harness_file
    )
    assert (
        child.environment[PROGRAM_PATH_ENV]
        == parent.instructions.program.program_path
    )
    child_environment = child.environment
    assert decode_system_instructions(
        Path(child_environment[SYSTEM_INSTRUCTIONS_FILE_ENV]).read_text().strip(),
        child_environment[SYSTEM_INSTRUCTIONS_SHA256_ENV],
    ).program == parent.instructions.program
    role_prompt = role_file.read_text()
    assert "## Runtime identity" not in role_prompt
    assert "Use the `2` GPUs available to each student" in role_prompt
    assert PLACEHOLDER.search(role_prompt) is None
    assert "github-secret-sentinel" not in role_prompt
    assert "wandb-secret-sentinel" not in role_prompt


def test_child_environment_replaces_ambient_model_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "unconfigured-model-key")

    environment = OpenHandsChildProcess(
        delegation_config(tmp_path),
        delegation_request(model="frontier", agent="general-purpose"),
    ).environment

    assert "ANTHROPIC_API_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert "GEMINI_API_KEY" not in environment


def test_child_model_credentials_use_a_private_fd_bundle(tmp_path: Path):
    runner = OpenHandsChildProcess(delegation_config(tmp_path), delegation_request())

    descriptor = runner._open_model_credentials_fd()
    environment = {MODEL_CREDENTIALS_FD_ENV: str(descriptor)}

    assert consume_model_credential_fd(environment) == {
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "OPENAI_API_KEY": "openai-secret",
    }
    assert MODEL_CREDENTIALS_FD_ENV not in environment
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_wandb_children_receive_only_the_inference_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "controller-key")
    monkeypatch.setenv("SENPAI_WANDB_TRAINING_API_KEY", "training-key")
    config = delegation_config(
        tmp_path,
        smart_model="wandb/zai-org/GLM-5.2",
        smart_api_key_env="WANDB_INFERENCE_API_KEY",
        smart_api_key="inference-key",
        fast_model="wandb/zai-org/GLM-5.2",
        fast_api_key_env="WANDB_INFERENCE_API_KEY",
        fast_api_key="inference-key",
        frontier_model="wandb/zai-org/GLM-5.2",
        frontier_api_key_env="WANDB_INFERENCE_API_KEY",
        frontier_api_key="inference-key",
    )
    runner = OpenHandsChildProcess(config, delegation_request())

    assert "WANDB_API_KEY" not in runner.environment
    assert "SENPAI_WANDB_TRAINING_API_KEY" not in runner.environment
    assert "WANDB_INFERENCE_API_KEY" not in runner.environment
    descriptor = runner._open_model_credentials_fd()
    credentials = consume_model_credential_fd(
        {MODEL_CREDENTIALS_FD_ENV: str(descriptor)}
    )
    assert credentials == {"WANDB_INFERENCE_API_KEY": "inference-key"}


def test_child_rejects_conflicting_keys_for_one_provider_env(tmp_path: Path):
    runner = OpenHandsChildProcess(
        delegation_config(tmp_path, fast_api_key="different-anthropic-key"),
        delegation_request(),
    )

    with pytest.raises(RuntimeError, match="conflicting values.*ANTHROPIC_API_KEY"):
        runner._model_credentials()


def test_child_start_passes_model_credentials_only_through_the_fd(monkeypatch, tmp_path):
    captured = {}

    class Process:
        pass

    class Thread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()

    monkeypatch.setattr(delegation_module.subprocess, "Popen", popen)
    monkeypatch.setattr(delegation_module.threading, "Thread", Thread)
    runner = OpenHandsChildProcess(delegation_config(tmp_path), delegation_request())

    runner.start("Inspect the result.", 60, lambda _result, _error: None)

    environment = captured["env"]
    descriptor = int(environment[MODEL_CREDENTIALS_FD_ENV])
    assert captured["pass_fds"] == (descriptor,)
    assert "ANTHROPIC_API_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert "anthropic-secret" not in repr(environment)
    assert "openai-secret" not in repr(environment)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_child_environment_carries_only_configured_custom_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(CUSTOM_SECRET_ENV_NAMES_ENV, "STALE_AUTH")
    monkeypatch.setenv("STALE_AUTH", "stale-secret")
    config = delegation_config(
        tmp_path,
        conversation_secrets={
            "EXA_API_KEY": "exa-secret",
            "PRIVATE_AUTH": "private-secret",
            "REGISTRY_API_KEY": "registry-secret",
        },
    )

    environment = OpenHandsChildProcess(
        config,
        delegation_request(model="fast", agent="general-purpose"),
    ).environment

    assert environment[CUSTOM_SECRET_ENV_NAMES_ENV] == (
        "PRIVATE_AUTH,REGISTRY_API_KEY"
    )
    assert environment["PRIVATE_AUTH"] == "private-secret"
    assert environment["REGISTRY_API_KEY"] == "registry-secret"
    assert "EXA_API_KEY" not in environment
    assert "STALE_AUTH" not in environment


def test_child_process_never_receives_the_github_write_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = delegation_request()
    config = delegation_config(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")

    environment = OpenHandsChildProcess(config, request).environment

    assert "GITHUB_TOKEN" not in environment
    assert "GH_TOKEN" not in environment
    assert "SENPAI_GITHUB_TOKEN_FILE" not in environment
    assert "ambient-write-token" not in repr(environment)
    assert not list(config.state_dir.rglob(".github-token-*"))


def test_child_result_parser_uses_only_terminal_result_record():
    output = (
        'OPENHANDS_EVENT {"text":"intermediate"}\n'
        'OPENHANDS_RESULT {"status":"finished","result":"final report"}'
    )

    assert OpenHandsChildProcess.parse_result(output) == "final report"

    with pytest.raises(RuntimeError, match="terminal result"):
        OpenHandsChildProcess.parse_result('OPENHANDS_EVENT {"text":"not enough"}')


def test_child_monitor_invokes_a_failing_completion_callback_once(tmp_path: Path):
    class CompletedProcess:
        returncode = 0

        def wait(self, timeout):
            return 0

    runner = OpenHandsChildProcess(delegation_config(tmp_path), delegation_request())
    runner.output_path.parent.mkdir(parents=True)
    runner.output_path.write_text(
        'OPENHANDS_RESULT {"status":"finished","result":"done"}\n'
    )
    process = CompletedProcess()
    runner._process = process
    calls = []

    def fail(result, error):
        calls.append((result, error))
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        runner._monitor(process, 1, fail)

    assert calls == [("done", None)]
    assert runner._process is None


def test_late_result_event_is_acknowledged_when_await_collected_first(tmp_path: Path):
    registry_path = tmp_path / "state" / "delegation" / "tasks.sqlite3"
    event_path = tmp_path / "events.sqlite3"
    registry = DelegationRegistry(registry_path)
    rows, _created = registry.reserve(
        operation_key="conversation:batch",
        tree_id="tree",
        parent_conversation_id="conversation",
        parent_task_id=None,
        depth=1,
        specs=[AgentTask(key="task", task="Inspect", model="smart")],
        deadlines=[time.time() + 60],
    )
    reserved_id = rows[0]["task_id"]
    registry.mark_running(reserved_id, None)
    registry.finish(reserved_id, result="done")
    registry.mark_collected([reserved_id])

    record_delegated_task_result(
        reserved_id,
        result="done",
        env={
            "SENPAI_DELEGATION_REGISTRY_PATH": str(registry_path),
            "SENPAI_DELEGATION_EVENT_DB_PATH": str(event_path),
        },
    )

    with LocalEventStore(event_path) as events:
        assert events.pending() == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cancellation")
def test_child_interrupt_kills_stubborn_descendant_after_leader_exits(tmp_path: Path):
    child_pid_path = tmp_path / "child-pid"
    code = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)']);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        (sys.executable, "-c", code),
        start_new_session=True,
    )
    runner = OpenHandsChildProcess(delegation_config(tmp_path), delegation_request())
    runner._process = process
    deadline = time.monotonic() + 2
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_path.read_text())

    runner.interrupt()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            child = psutil.Process(child_pid)
            if child.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    else:
        pytest.fail("stubborn descendant survived process-group cancellation")
