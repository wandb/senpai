import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from base64 import b64decode

import pytest
import psutil
from openhands.sdk.llm import Message, TextContent

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
from senpai_agent.program_context import PROGRAM_PATH_ENV
from senpai_agent.secrets import CUSTOM_SECRET_ENV_NAMES_ENV
from senpai_agent.supervisor import prepare_system_context_environment
from openhands_support import runtime_config


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
        "conversation_secrets": {"EXA_API_KEY": "exa-secret"},
        "role": "advisor",
        "program_path": "program.md",
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
    assert fast.environment["ANTHROPIC_API_KEY"] == "anthropic-secret"
    assert fast.environment["OPENAI_API_KEY"] == "openai-secret"
    assert fast.environment["SENPAI_OPENHANDS_API_KEY_ENV"] == "ANTHROPIC_API_KEY"
    assert frontier.environment["SENPAI_OPENHANDS_API_KEY_ENV"] == "OPENAI_API_KEY"
    assert fast.environment["GH_REPO"] == "acme/widgets"
    assert "GITHUB_TOKEN" not in fast.environment
    assert "GH_TOKEN" not in fast.environment
    assert fast.environment["EXA_API_KEY"] == "exa-secret"
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
            program_path="senpai/program.md",
        ),
        delegation_request(),
    )

    assert child.environment[PROGRAM_PATH_ENV] == "senpai/program.md"
    assert (
        b64decode(child.environment[LAUNCH_CONTEXT_ENV], validate=True).decode()
        == "# Authoritative launch context\n\nSystem policy."
    )


def test_child_reuses_the_supervisor_rendered_role_prompt(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "program.md").write_text("Research policy.\n")
    prepared = prepare_system_context_environment(
        "advisor",
        tmp_path / "state",
        {
            "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
            "SENPAI_OPENHANDS_ROLE_FILE": str(INSTRUCTIONS_ROOT / "ADVISOR.md"),
            "GH_REPO": "acme/widgets",
            "ADVISOR_BRANCH": "research",
            "WANDB_ENTITY": "acme",
            "WANDB_PROJECT": "cfd",
            "STUDENT_NAMES": "fern,frieren",
            "NODES_PER_STUDENT": "2",
            "GPUS_PER_STUDENT_NODE": "2",
            "GITHUB_TOKEN": "github-secret-sentinel",
            "WANDB_API_KEY": "wandb-secret-sentinel",
        },
    )
    role_file = Path(prepared["SENPAI_OPENHANDS_ROLE_FILE"])
    parent = runtime_config(tmp_path, role_file=role_file)
    delegated = runner_delegation_config(parent)
    child = OpenHandsChildProcess(
        delegated,
        delegation_request(),
    )

    assert delegated.role_file == parent.role_file
    assert delegated.harness_file == parent.harness_file
    assert delegated.program_path == parent.instructions.program.program_path
    assert delegated.launch_context == parent.instructions.launch
    assert child.command[child.command.index("--role-file") + 1] == str(role_file)
    assert child.command[child.command.index("--harness-file") + 1] == str(
        parent.harness_file
    )
    assert (
        child.environment[PROGRAM_PATH_ENV]
        == parent.instructions.program.program_path
    )
    role_prompt = role_file.read_text()
    assert "## Runtime identity" not in role_prompt
    assert (
        "Use the `2` worker nodes x `2` GPUs per node available to each student"
        in role_prompt
    )
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

    assert environment["ANTHROPIC_API_KEY"] == "anthropic-secret"
    assert environment["OPENAI_API_KEY"] == "openai-secret"
    assert "GEMINI_API_KEY" not in environment


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
