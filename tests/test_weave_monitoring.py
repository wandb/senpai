import json

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from weave_openhands import TracingConfig, instrument, is_instrumented, uninstrument

import senpai_agent.weave_monitoring as monitoring
from senpai_agent.secrets import CUSTOM_SECRET_ENV_NAMES_ENV


@pytest.fixture(scope="module")
def trace_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


def test_monitoring_uses_the_senpai_wandb_project_and_student_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(monitoring, "_initialized", False)
    monkeypatch.setattr(monitoring, "_project_name", None)
    monkeypatch.setattr(
        monitoring,
        "weave_init",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(monitoring, "weave_finish", lambda: calls.append("finish"))
    env = {
        "WANDB_ENTITY": "wandb-applied-ai-team",
        "WANDB_PROJECT": "senpai-v1",
        "WANDB_API_KEY": "wandb-secret",
        "SENPAI_ROLE": "student",
        "STUDENT_NAME": "charlie",
    }

    assert monitoring.initialize_weave_monitoring(env) == (
        "wandb-applied-ai-team/senpai-v1"
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("wandb-applied-ai-team/senpai-v1",)
    assert kwargs["agent_name"] == "student-charlie"
    assert kwargs["capture_content"] is True
    assert kwargs["content_transform"]("token=wandb-secret") == "token=<secret-hidden>"

    monitoring.finish_weave_monitoring()

    assert calls[-1] == "finish"


@pytest.mark.parametrize(
    "env",
    [
        {"WANDB_ENTITY": "wandb-applied-ai-team"},
        {"WANDB_PROJECT": "senpai-v1"},
    ],
)
def test_monitoring_requires_complete_wandb_project_configuration(env):
    with pytest.raises(RuntimeError, match="must be set together"):
        monitoring.weave_project_name(env)


def test_agent_observability_url_targets_the_durable_conversation():
    assert monitoring.weave_conversation_url(
        "wandb-applied-ai-team/senpai-v1",
        "conversation-17",
    ) == (
        "https://wandb.ai/wandb-applied-ai-team/senpai-v1/"
        "weave/agents/conversations/conversation-17"
    )
    assert monitoring.weave_conversation_url(None, "conversation-17") is None


def test_monitoring_redacts_a_secret_registered_after_initialization(monkeypatch):
    calls = []
    monkeypatch.setattr(monitoring, "_initialized", False)
    monkeypatch.setattr(monitoring, "_project_name", None)
    monkeypatch.setattr(
        monitoring,
        "weave_init",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(monitoring, "weave_finish", lambda: None)

    monitoring.initialize_weave_monitoring(
        {
            "WANDB_ENTITY": "wandb-applied-ai-team",
            "WANDB_PROJECT": "senpai-v1",
        }
    )
    transform = calls[0][1]["content_transform"]
    assert transform("late-write-token") == "late-write-token"

    monitoring.register_trace_secret("late-write-token")

    assert transform("late-write-token") == "<secret-hidden>"
    monitoring.finish_weave_monitoring()


def test_secret_redactor_replaces_overlapping_values_longest_first():
    redact = monitoring.secret_redactor(
        {
            "GITHUB_TOKEN": "token-prefix",
            "GH_TOKEN": "token",
            "WANDB_API_KEY": "wandb-secret",
            "EXA_API_KEY": "exa-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "OPENAI_API_KEY": "openai-secret",
            "SERVICE_PASSWORD": "service-secret",
            "DATABASE_PASSWORD": "database-secret",
            "CLIENT_SECRET": "client-secret",
            "SENPAI_OPENHANDS_API_KEY_ENV": "CUSTOM_MODEL_CREDENTIAL",
            "CUSTOM_MODEL_CREDENTIAL": "custom-model-secret",
        }
    )

    assert redact(
        "token-prefix token wandb-secret exa-secret anthropic-secret "
        "openai-secret service-secret database-secret client-secret "
        "custom-model-secret"
    ) == " ".join(["<secret-hidden>"] * 10)


def test_secret_redactor_includes_explicit_custom_secret_names():
    redact = monitoring.secret_redactor(
        {
            CUSTOM_SECRET_ENV_NAMES_ENV: "PRIVATE_AUTH",
            "PRIVATE_AUTH": "private-value",
        }
    )

    assert redact("credential=private-value") == "credential=<secret-hidden>"


def test_weave_openhands_traces_a_real_openhands_turn(
    tmp_path, monkeypatch, trace_exporter: InMemorySpanExporter
):
    uninstrument()
    trace_exporter.clear()
    try:
        instrument(
            TracingConfig(
                agent_name="student-charlie",
                content_transform=monitoring.secret_redactor(
                    {"ANTHROPIC_API_KEY": "anthropic-secret"}
                ),
            )
        )
        assert is_instrumented()
        from openhands.sdk import Agent, Conversation
        from openhands.sdk.llm import Message, MessageToolCall, TextContent
        from openhands.sdk.testing import TestLLM

        monkeypatch.setattr(
            "openhands.sdk.llm.llm_profile_store._DEFAULT_PROFILE_DIR",
            tmp_path / ".openhands" / "profiles",
        )
        response = Message(
            role="assistant",
            content=[TextContent(text="Finishing the task")],
            tool_calls=[
                MessageToolCall(
                    id="finish-call",
                    name="finish",
                    arguments=json.dumps({"message": "Task complete"}),
                    origin="completion",
                )
            ],
        )
        agent = Agent(
            llm=TestLLM.from_messages([response], model="test-model"),
            tools=[],
        )
        conversation = Conversation(
            agent=agent,
            workspace=tmp_path,
            visualizer=None,
        )
        conversation.send_message("Use anthropic-secret without exposing it")

        conversation.run()

        spans = [
            span
            for span in trace_exporter.get_finished_spans()
            if span.instrumentation_scope.name == "weave_openhands"
        ]
        root = next(
            span for span in spans if span.name == "invoke_agent student-charlie"
        )
        assert {span.attributes["gen_ai.operation.name"] for span in spans} == {
            "invoke_agent",
            "chat",
            "execute_tool",
        }
        assert {span.attributes["gen_ai.conversation.id"] for span in spans} == {
            str(conversation.id)
        }
        assert "anthropic-secret" not in str(
            [dict(span.attributes or {}) for span in spans]
        )
        assert "<secret-hidden>" in str(root.attributes)
    finally:
        uninstrument()
        trace_exporter.clear()


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_runner_attributes_real_model_spans_before_conversation_finishes(
    tmp_path, monkeypatch, trace_exporter, depth
):
    from litellm import ModelResponse
    from openhands.sdk import LLM
    from openhands_support import isolate_agent_discovery, runtime_config

    import senpai_agent.openhands_runner as runner

    for variable in tuple(monitoring.os.environ):
        if (
            variable.startswith("SENPAI_DELEGATION_")
            or variable == "SENPAI_PARENT_CONVERSATION_ID"
        ):
            monkeypatch.delenv(variable)
    monkeypatch.setenv("SENPAI_ROLE", "student")
    monkeypatch.setenv("STUDENT_NAME", "charlie")
    monkeypatch.setenv("GH_REPO", "acme/research")
    monkeypatch.setenv("ADVISOR_BRANCH", "campaign")
    if depth:
        monkeypatch.setenv("SENPAI_DELEGATION_DEPTH", str(depth))
        monkeypatch.setenv("SENPAI_DELEGATION_TASK_ID", "child-task")
        monkeypatch.setenv("SENPAI_DELEGATION_TREE_ID", "root-conversation")
        monkeypatch.setenv("SENPAI_PARENT_CONVERSATION_ID", "parent-conversation")
        monkeypatch.setenv("SENPAI_DELEGATION_MODEL_TIER", "frontier")
        monkeypatch.setenv("SENPAI_DELEGATION_INPUT_CHARACTERS", "12345")
        monkeypatch.setenv("SENPAI_DELEGATION_PARENT_MESSAGES", "12")
        if depth == 2:
            monkeypatch.setenv("SENPAI_DELEGATION_PARENT_TASK_ID", "parent-task")
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "build_main_tools", lambda _config: [])
    monkeypatch.setattr(
        "openhands.sdk.llm.llm_profile_store._DEFAULT_PROFILE_DIR",
        tmp_path / "profiles",
    )
    observed = []

    async def transport(_self, **_kwargs):
        observed.append(dict(trace.get_current_span().attributes))
        assert not trace_exporter.get_finished_spans()
        return ModelResponse(
            model="claude-sonnet-4-5",
            choices=[
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Done",
                        "tool_calls": [
                            {
                                "id": "finish-call",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps(
                                        {"message": "Task complete"}
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    monkeypatch.setattr(LLM, "_atransport_call", transport)
    config = runtime_config(
        tmp_path,
        model="anthropic/claude-sonnet-4-5",
        reasoning_effort="high",
        github_token=None,
    )
    uninstrument()
    trace_exporter.clear()
    instrument(TracingConfig(agent_name="student-charlie"))
    try:
        assert runner.run_openhands("Inspect the result.", config) == 0
        chats = [
            s for s in trace_exporter.get_finished_spans() if s.name.startswith("chat ")
        ]
        assert len(chats) == len(observed) == 1
        attributes = chats[0].attributes
        assert attributes["senpai.owner"] == "student-charlie"
        assert attributes["senpai.repository"] == "acme/research"
        assert attributes["senpai.advisor_branch"] == "campaign"
        assert attributes["senpai.delegation.depth"] == depth
        assert attributes["gen_ai.request.model"] == "anthropic/claude-sonnet-4-5"
        assert observed[0]["senpai.delegation.depth"] == depth
        if depth:
            assert attributes["senpai.delegation.task_id"] == "child-task"
            assert (
                attributes["senpai.delegation.parent_conversation_id"]
                == "parent-conversation"
            )
            assert attributes["senpai.delegation.model_tier"] == "frontier"
            assert attributes["senpai.delegation.input_characters"] == 12345
            assert attributes["senpai.delegation.parent_messages"] == 12
        else:
            assert "senpai.delegation.task_id" not in attributes
        if depth == 2:
            assert attributes["senpai.delegation.parent_task_id"] == "parent-task"
        else:
            assert "senpai.delegation.parent_task_id" not in attributes
    finally:
        uninstrument()
        trace_exporter.clear()
