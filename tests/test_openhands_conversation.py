import json
import signal
import threading
from io import StringIO
from types import SimpleNamespace

import pytest
from openhands.sdk import Agent, LLM
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.llm import Message, TextContent
from pydantic import SecretStr

import senpai_agent.openhands_runner as runner
from senpai_agent.openhands_runner import (
    graceful_interrupts,
    main,
    reject_recovered_actions,
    run_openhands,
)
from openhands_support import (
    PLUGIN_DIR,
    isolate_agent_discovery,
    runtime_config,
)


def test_run_initializes_role_plugin_and_secrets_before_the_first_message(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class FakeConversation:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.plugins = kwargs["plugins"]
            self.id = kwargs["conversation_id"]
            captured["secrets"] = kwargs["secrets"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            captured["llm_timeout"] = agent.llm.timeout
            captured["llm_num_retries"] = agent.llm.num_retries
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, prompt):
            captured["prompt"] = prompt
            captured["role"] = self.agent.agent_context.system_message_suffix
            captured["plugin"] = self.plugins[0].source
            captured["conversation_id_env"] = runner.os.environ[
                "SENPAI_CONVERSATION_ID"
            ]

        async def arun(self):
            pass

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(tmp_path)

    assert run_openhands("first task", config) == 0
    assert captured["prompt"] == "first task"
    assert captured["role"] == (
        "# Senpai harness\n\nharness instructions\n\n"
        "# Senpai role\n\nadvisor role\n"
    )
    assert captured["plugin"] == str(PLUGIN_DIR)
    assert captured["secrets"] == {"WANDB_API_KEY": "wandb-key"}
    assert captured["conversation_id_env"] == config.conversation_id.hex
    assert captured["delete_on_close"] is False
    assert captured["llm_timeout"] == 900
    assert captured["llm_num_retries"] == 1
    assert captured["closed"] is True


def test_context_reset_preserves_history_and_starts_a_fresh_active_branch(
    tmp_path,
    monkeypatch,
):
    history = ["old message", "old response"]
    active = list(history)
    calls = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: active,
                events=history,
                execution_status=ConversationExecutionStatus.FINISHED,
            )

        def reject_pending_actions(self, reason):
            calls.append(("reject", reason))

        def navigate_to(self, event_id):
            calls.append(("navigate", event_id))
            active.clear()

        def send_message(self, prompt):
            calls.append(("send", prompt))
            active.append(prompt)

        async def arun(self):
            calls.append(("run", None))

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [object()],
    )
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(tmp_path)

    assert run_openhands("fresh recovery prompt", config, reset_context=True) == 0

    assert history == ["old message", "old response"]
    assert active == ["fresh recovery prompt"]
    assert [name for name, _ in calls] == [
        "reject",
        "navigate",
        "send",
        "run",
        "close",
    ]
    assert calls[1] == ("navigate", None)


def test_child_requests_ephemeral_storage_and_emits_its_terminal_report(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = {}

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.IDLE,
                view=SimpleNamespace(events=[]),
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            self.state.view.events.append(
                runner.MessageEvent(
                    source="agent",
                    llm_message=Message(
                        role="assistant",
                        content=[TextContent(text="bounded child report")],
                    ),
                )
            )
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "WEAVE_PROJECT", "wandb-applied-ai-team/senpai-v1")
    config = runtime_config(tmp_path, child=True)

    assert run_openhands("child task", config) == 0

    records = capsys.readouterr().out.splitlines()
    result = json.loads(
        next(
            line.removeprefix("OPENHANDS_RESULT ")
            for line in records
            if line.startswith("OPENHANDS_RESULT ")
        )
    )
    run = json.loads(
        next(
            line.removeprefix("OPENHANDS_RUN ")
            for line in records
            if line.startswith("OPENHANDS_RUN ")
        )
    )
    assert captured == {"delete_on_close": True, "closed": True}
    assert result["result"] == "bounded child report"
    assert run["weave_url"] == (
        "https://wandb.ai/wandb-applied-ai-team/senpai-v1/"
        f"weave/agents/conversations/{config.conversation_id}"
    )


def test_student_requests_persistent_storage_for_monitor_wake(
    tmp_path,
    monkeypatch,
):
    captured = {}
    delegation = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            captured["tool_concurrency_limit"] = kwargs[
                "agent"
            ].tool_concurrency_limit
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "configure_delegation", delegation.append)

    assert run_openhands("student task", runtime_config(tmp_path, role="student")) == 0
    assert captured == {
        "delete_on_close": False,
        "tool_concurrency_limit": 8,
    }
    assert delegation[0].role == "student"
    assert delegation[-1] is None


def test_named_agent_compaction_uses_its_own_provider(tmp_path, monkeypatch):
    captured = {}
    named_llm = LLM(
        model="wandb/zai-org/GLM-5.2",
        api_key=SecretStr("test-key"),
        reasoning_effort="max",
        **runner.model_runtime_configuration(
            "wandb/zai-org/GLM-5.2",
            "max",
            wandb_entity="research-team",
            wandb_project="mlxfast",
        ),
    )

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["condenser"] = kwargs["agent"].condenser
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            pass

        def close(self):
            pass

    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "find_named_agent", lambda *_: object())
    monkeypatch.setattr(
        runner,
        "agent_definition_to_factory",
        lambda *_args, **_kwargs: lambda _parent_llm: Agent(
            llm=named_llm,
            tools=[],
        ),
    )
    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)

    config = runtime_config(
        tmp_path,
        agent_name="explore",
        local_condenser_max_events=180,
    )
    assert run_openhands("named task", config) == 0
    assert captured["condenser"].max_size == 180


def test_github_tokens_never_reach_the_agent_environment(
    tmp_path,
    monkeypatch,
):
    observed = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            observed.append(runner.os.environ.get("GITHUB_TOKEN"))

        async def arun(self):
            observed.append(runner.os.environ.get("GITHUB_TOKEN"))

        def close(self):
            pass

    monkeypatch.setenv("GITHUB_TOKEN", "stale-env-secret")
    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands("task", runtime_config(tmp_path, role="student")) == 0
    assert observed == [None, None]
    assert "GITHUB_TOKEN" not in runner.os.environ


@pytest.mark.parametrize(
    "status",
    [
        status
        for status in ConversationExecutionStatus
        if status is not ConversationExecutionStatus.FINISHED
    ],
    ids=lambda status: status.value,
)
def test_nonfinished_conversation_returns_failure(tmp_path, monkeypatch, status):
    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(execution_status=status)

        def send_message(self, _prompt):
            pass

        async def arun(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands("task", runtime_config(tmp_path)) == 1


@pytest.mark.parametrize("failure_stage", ["send_message", "run"])
def test_conversation_and_credentials_are_cleaned_up_after_failures(
    tmp_path,
    monkeypatch,
    failure_stage: str,
):
    closed = []
    cleared = []
    delegation = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]

        def send_message(self, _prompt):
            if failure_stage == "send_message":
                raise RuntimeError("initialization failed")

        async def arun(self):
            if failure_stage == "run":
                raise RuntimeError("execution failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "clear_github_credentials", lambda: cleared.append(True))
    monkeypatch.setattr(runner, "configure_delegation", delegation.append)

    with pytest.raises(RuntimeError, match="failed"):
        run_openhands("task", runtime_config(tmp_path))

    assert closed == [True]
    assert cleared
    assert delegation[-1] is None


def test_turn_deadline_requests_conversation_interrupt(tmp_path, monkeypatch):
    interrupted = threading.Event()
    cancelled = threading.Event()

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.RUNNING
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            try:
                await runner.asyncio.Event().wait()
            finally:
                cancelled.set()

        def interrupt(self):
            interrupted.set()

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    assert (
            run_openhands(
                "task",
                runtime_config(tmp_path, timeout_seconds=0.2),
            )
        == 1
    )
    assert interrupted.is_set()
    assert cancelled.is_set()


def test_signal_interrupts_the_conversation_and_restores_handlers(monkeypatch):
    calls = []
    installed = {}
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        installed[signum] = handler
        return previous[signum]

    conversation = SimpleNamespace(interrupt=lambda: calls.append("interrupt"))
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with pytest.raises(SystemExit) as raised:
        with graceful_interrupts(conversation):
            installed[signal.SIGTERM](signal.SIGTERM, None)
            calls.append("handler-returned")

    assert raised.value.code == 128 + signal.SIGTERM
    assert "interrupt" in calls
    assert "handler-returned" in calls
    assert calls[-2:] == [
        (signal.SIGTERM, previous[signal.SIGTERM]),
        (signal.SIGINT, previous[signal.SIGINT]),
    ]


def test_recovered_actions_are_rejected_before_the_conversation_resumes(
    monkeypatch,
):
    rejected = []
    conversation = SimpleNamespace(
        state=SimpleNamespace(active_branch=lambda: ["persisted action"]),
        reject_pending_actions=lambda reason: rejected.append(reason),
    )
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [object()],
    )

    assert reject_recovered_actions(conversation) == 1
    assert "rerun it explicitly" in rejected[0]


def test_main_removes_the_model_key_and_flushes_weave_after_failure(monkeypatch):
    flushed = []

    def fail_run(_prompt, _config):
        assert "ANTHROPIC_API_KEY" not in runner.os.environ
        raise RuntimeError("run failed")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr(runner.sys, "stdin", StringIO("first task"))
    monkeypatch.setattr(
        runner,
        "resolve_config",
        lambda _args: SimpleNamespace(api_key_env="ANTHROPIC_API_KEY"),
    )
    monkeypatch.setattr(runner, "run_openhands", fail_run)
    monkeypatch.setattr(runner, "finish_weave_monitoring", lambda: flushed.append(True))

    with pytest.raises(RuntimeError, match="run failed"):
        main(["--max-turns", "1"])

    assert flushed == [True]
