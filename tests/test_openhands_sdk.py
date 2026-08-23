import re
import tomllib
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from openhands.sdk import Agent, LLM, LocalConversation
from openhands.sdk.llm import Message, TextContent
from pydantic import SecretStr

from senpai_agent.openhands_runner import (
    EVENT_TEXT_LIMIT,
    apply_reasoning_profile,
    compaction_configuration,
    conversation_prompt_cache_key,
    event_summary,
    model_runtime_configuration,
    openai_responses_configuration,
    openhands_reasoning_effort,
    parse_runner_args,
    prompt_cache_configuration,
)
from senpai_agent.inbox import DeliveryState, PersistentInbox, deliver_turn_messages
from openhands_support import REPO_ROOT, runtime_config

TEST_COMPACTION_TRIGGER_TOKENS = 180_000


def test_openhands_fork_revision_is_consistent_across_install_paths():
    package_names = {"openhands-sdk", "openhands-tools"}
    fork_url = "git+https://github.com/morganmcg1/software-agent-sdk.git"
    fork_revision = "f69134273ee3a31a233d6201786570eb9c4c141b"
    expected_requirements = {
        f"{name} @ {fork_url}@{fork_revision}#subdirectory={name}"
        for name in package_names
    }

    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project_requirements = {
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.partition(" @ ")[0] in package_names
    }
    assert project_requirements == expected_requirements

    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yaml").read_text(
        encoding="utf-8"
    )
    ci_requirements = {
        match.group(1)
        for line in workflow.splitlines()
        if (
            match := re.fullmatch(
                r'\s*"(openhands-(?:sdk|tools) @ git\+[^"]+)"(?:\s+\\)?\s*',
                line,
            )
        )
    }
    assert ci_requirements == expected_requirements

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_packages = {
        package["name"]: package
        for package in lock["package"]
        if package["name"] in package_names
    }
    assert locked_packages.keys() == package_names

    resolved_revisions = set()
    for name, package in locked_packages.items():
        source = urlsplit(package["source"]["git"])
        assert source.scheme == "https"
        assert source.netloc == "github.com"
        assert source.path == "/morganmcg1/software-agent-sdk.git"
        assert parse_qs(source.query) == {
            "subdirectory": [name],
            "rev": [fork_revision],
        }
        assert source.fragment == fork_revision
        resolved_revisions.add(source.fragment)

    assert len(resolved_revisions) == 1


@pytest.mark.parametrize(
    ("effort", "model", "expected"),
    [
        ("max", "openai/gpt-5.6-sol", "max"),
        ("high", "openai/gpt-5.6", "high"),
        ("xhigh", "anthropic/claude-opus-4-8", "xhigh"),
        ("max", "anthropic/claude-fable-5", "max"),
        ("max", "anthropic/claude-opus-5", "max"),
        ("max", "anthropic/claude-sonnet-5", "max"),
        ("high", "wandb/zai-org/GLM-5.2", "high"),
        ("max", "wandb/zai-org/GLM-5.2", "max"),
    ],
)
def test_supported_reasoning_effort_is_preserved(
    effort: str,
    model: str,
    expected: str,
):
    args = parse_runner_args(["--max-turns", "1", "--reasoning-effort", effort])

    assert args.reasoning_effort == effort
    assert openhands_reasoning_effort(effort, model) == expected


@pytest.mark.parametrize(
    ("effort", "model"),
    [
        ("max", "openai/gpt-5.4"),
        ("max", "openai/gpt-5.60"),
        ("medium", "wandb/zai-org/GLM-5.2"),
        ("extreme", "openai/gpt-5.6-sol"),
        ("ultra", "openai/gpt-5.6-sol"),
    ],
)
def test_unsupported_reasoning_effort_fails_instead_of_being_rewritten(
    effort: str,
    model: str,
):
    with pytest.raises(ValueError, match="unsupported reasoning effort|unsupported for"):
        openhands_reasoning_effort(effort, model)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "anthropic/claude-opus-4-8",
            {"prompt_cache_ttl": "1h"}
            if "prompt_cache_ttl" in LLM.model_fields
            else {},
        ),
        ("openai/gpt-5.4", {"prompt_cache_retention": "24h"}),
        (
            "openai/gpt-5.6",
            {
                "prompt_cache_retention": None,
                "responses_prompt_cache_breakpoint": True,
                "litellm_extra_body": {
                    "prompt_cache_options": {
                        "mode": "explicit",
                        "ttl": "30m",
                    }
                },
            },
        ),
        ("gemini/gemini-3-pro", {}),
    ],
)
def test_prompt_cache_configuration_is_provider_specific(model: str, expected):
    assert prompt_cache_configuration(model) == expected


def test_openai_response_configuration_is_accepted_by_the_pinned_sdk():
    configuration = openai_responses_configuration("openai/gpt-5.6-sol")
    llm = LLM(
        model="openai/gpt-5.6-sol",
        api_key=SecretStr("test-key"),
        reasoning_effort="max",
        **configuration,
    )

    assert configuration == {
        "api_mode": "responses",
        "reasoning_summary": "auto",
        "reasoning_context": "all_turns",
        "responses_store": True,
        "responses_use_previous_response_id": True,
    }
    assert llm.uses_responses_api() is True
    assert llm.reasoning_effort == "max"
    assert llm.responses_store is True
    assert llm.responses_use_previous_response_id is True
    assert openai_responses_configuration("anthropic/claude-opus-4-8") == {}


def test_openai_max_uses_pro_mode_on_the_wire():
    configuration = model_runtime_configuration(
        "openai/gpt-5.6-sol",
        "max",
        compaction_trigger_tokens=TEST_COMPACTION_TRIGGER_TOKENS,
    )
    llm = LLM(
        model="openai/gpt-5.6-sol",
        api_key=SecretStr("test-key"),
        reasoning_effort=openhands_reasoning_effort(
            "max",
            "openai/gpt-5.6-sol",
        ),
        **configuration,
    )
    _instructions, _inputs, _tools, call_kwargs, _telemetry = (
        llm._prepare_responses_params(
            [Message(role="user", content=[TextContent(text="Investigate")])],
            tools=None,
            include=None,
            store=None,
            add_security_risk_prediction=False,
            kwargs={},
        )
    )

    assert llm.reasoning_effort == "max"
    assert call_kwargs["reasoning"] == {
        "effort": "max",
        "summary": "auto",
        "context": "all_turns",
    }
    assert call_kwargs["context_management"] == [
        {
            "type": "compaction",
            "compact_threshold": TEST_COMPACTION_TRIGGER_TOKENS,
        }
    ]
    assert call_kwargs["extra_body"] == {
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "reasoning": {
            "effort": "max",
            "mode": "pro",
            "summary": "auto",
            "context": "all_turns",
        },
    }
    wire_request = {
        key: value for key, value in call_kwargs.items() if key != "extra_body"
    }
    wire_request.update(call_kwargs["extra_body"])
    assert wire_request["reasoning"] == {
        "effort": "max",
        "mode": "pro",
        "summary": "auto",
        "context": "all_turns",
    }


@pytest.mark.parametrize(
    ("parent_effort", "override", "expected_effort", "expects_pro"),
    [
        ("xhigh", "max", "max", True),
        ("max", "xhigh", "xhigh", False),
    ],
)
def test_file_agent_reasoning_override_replaces_the_parent_request_profile(
    parent_effort,
    override,
    expected_effort,
    expects_pro,
):
    model = "openai/gpt-5.6-sol"
    parent = LLM(
        model=model,
        api_key=SecretStr("test-key"),
        reasoning_effort=parent_effort,
        **model_runtime_configuration(
            model,
            parent_effort,
            compaction_trigger_tokens=TEST_COMPACTION_TRIGGER_TOKENS,
        ),
    )

    configured = apply_reasoning_profile(
        parent.model_copy(update={"reasoning_effort": override})
    )

    assert configured.reasoning_effort == expected_effort
    assert ("reasoning" in configured.litellm_extra_body) is expects_pro
    assert configured.litellm_extra_body["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
    ],
)
def test_anthropic_max_uses_provider_native_effort(model):
    llm = apply_reasoning_profile(
        LLM(
            model=model,
            api_key=SecretStr("test-key"),
            reasoning_effort="max",
            **model_runtime_configuration(
                model,
                "max",
                compaction_trigger_tokens=TEST_COMPACTION_TRIGGER_TOKENS,
            ),
        )
    )
    _messages, _tools, _mocked, call_kwargs, _telemetry = (
        llm._prepare_completion_params(
            [Message(role="user", content=[TextContent(text="Investigate")])],
            tools=None,
            add_security_risk_prediction=False,
            kwargs={},
        )
    )

    assert llm.reasoning_effort == "max"
    assert call_kwargs["reasoning_effort"] == "max"
    assert "reasoning" not in (llm.litellm_extra_body or {})
    assert "reasoning" not in call_kwargs.get("extra_body", {})


def test_wandb_gateway_uses_chat_thinking_and_project_routing():
    configuration = model_runtime_configuration(
        "wandb/zai-org/GLM-5.2",
        "max",
        compaction_trigger_tokens=TEST_COMPACTION_TRIGGER_TOKENS,
        wandb_entity="research-team",
        wandb_project="mlxfast",
    )
    llm = LLM(
        model="wandb/zai-org/GLM-5.2",
        api_key=SecretStr("test-key"),
        reasoning_effort="max",
        **configuration,
    )
    _messages, _tools, _mocked, call_kwargs, _telemetry = (
        llm._prepare_completion_params(
            [Message(role="user", content=[TextContent(text="Investigate")])],
            tools=None,
            add_security_risk_prediction=False,
            kwargs={},
        )
    )

    assert configuration == {
        "api_mode": "chat",
        "base_url": "https://api.inference.wandb.ai/v1",
        "extra_headers": {"OpenAI-Project": "research-team/mlxfast"},
        "capability_overrides": {
            "supports_reasoning_effort": False,
            "supports_responses_api": False,
        },
        "max_input_tokens": 262_144,
        "max_output_tokens": 16_384,
        "litellm_extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "max",
            },
        },
    }
    assert call_kwargs["extra_headers"]["OpenAI-Project"] == (
        "research-team/mlxfast"
    )
    assert call_kwargs["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "max",
        },
    }
    assert call_kwargs["max_completion_tokens"] == 16_384
    assert "reasoning_effort" not in call_kwargs
    assert llm._provider_info.name == "wandb"
    assert llm._provider_info.api_base == "https://api.inference.wandb.ai/v1"


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("openai/gpt-5.6-sol", "xhigh"),
        ("anthropic/claude-fable-5", "max"),
    ],
)
def test_pro_mode_is_only_enabled_by_openai_max(model, effort):
    extra_body = model_runtime_configuration(
        model,
        effort,
        compaction_trigger_tokens=TEST_COMPACTION_TRIGGER_TOKENS,
    ).get(
        "litellm_extra_body",
        {},
    )

    assert "reasoning" not in extra_body


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "openai/gpt-5.6",
            {"responses_compact_threshold": TEST_COMPACTION_TRIGGER_TOKENS},
        ),
        (
            "anthropic/claude-opus-4-8",
            {
                "anthropic_compact_threshold": TEST_COMPACTION_TRIGGER_TOKENS,
                "anthropic_compaction_instructions": None,
            },
        ),
        ("gemini/gemini-3-pro", {}),
    ],
)
def test_compaction_configuration_translates_the_universal_trigger(
    model,
    expected,
):
    assert (
        compaction_configuration(model, TEST_COMPACTION_TRIGGER_TOKENS)
        == expected
    )


def test_universal_compaction_configuration_reaches_anthropic_sdk():
    configuration = model_runtime_configuration(
        "anthropic/claude-opus-4-8",
        "max",
        compaction_trigger_tokens=TEST_COMPACTION_TRIGGER_TOKENS,
    )
    llm = LLM(
        model="anthropic/claude-opus-4-8",
        api_key=SecretStr("test-key"),
        **configuration,
    )
    _messages, _tools, _mocked, call_kwargs, _telemetry = (
        llm._prepare_completion_params(
            [Message(role="user", content=[TextContent(text="Investigate")])],
            tools=None,
            add_security_risk_prediction=False,
            kwargs={},
        )
    )

    assert (
        configuration["anthropic_compact_threshold"]
        == TEST_COMPACTION_TRIGGER_TOKENS
    )
    assert configuration["anthropic_compaction_instructions"] is None
    assert llm.uses_anthropic_compaction() is True
    assert call_kwargs["context_management"] == {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {
                    "type": "input_tokens",
                    "value": TEST_COMPACTION_TRIGGER_TOKENS,
                },
            }
        ]
    }


def test_gpt56_marks_only_the_stable_system_cache_boundary():
    llm = LLM(
        model="openai/gpt-5.6",
        api_key=SecretStr("test-key"),
        **prompt_cache_configuration("openai/gpt-5.6"),
        **openai_responses_configuration("openai/gpt-5.6"),
    )
    instructions, inputs = llm.format_messages_for_responses(
        [
            Message(
                role="system",
                content=[
                    TextContent(text="stable harness and role"),
                    TextContent(text="dynamic project context"),
                ],
            ),
            Message(role="user", content=[TextContent(text="Investigate")]),
        ]
    )

    assert instructions is None
    assert inputs[0]["content"] == [
        {
            "type": "input_text",
            "text": "stable harness and role",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {
            "type": "input_text",
            "text": "dynamic project context",
        },
    ]


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"model": "openai/gpt-5.6", "role": "student"}, "senpai:student:main"),
        (
            {"model": "openai/gpt-5.6", "role": "advisor", "child": True},
            "senpai:advisor:child",
        ),
        (
            {"model": "openai/gpt-5.6", "agent_name": "explore"},
            "senpai:advisor:explore",
        ),
        ({"model": "anthropic/claude-opus-4-8"}, None),
    ],
)
def test_prompt_cache_key_is_scoped_by_role_and_agent(tmp_path, updates, expected):
    assert conversation_prompt_cache_key(runtime_config(tmp_path, **updates)) == expected


def test_local_conversation_exposes_the_configured_prompt_cache_key(tmp_path):
    conversation = LocalConversation(
        agent=Agent(
            llm=LLM(
                model="openai/gpt-5.6",
                api_key=SecretStr("test-key"),
            ),
            tools=[],
        ),
        workspace=tmp_path,
        visualizer=None,
        prompt_cache_key="senpai:student:main",
    )
    try:
        assert (
            conversation.get_llm_call_context().prompt_cache_key
            == "senpai:student:main"
        )
    finally:
        conversation.close()


def test_local_conversation_persists_delivery_sender_and_payload(tmp_path):
    conversation_id = UUID("00000000-0000-0000-0000-000000000117")
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(conversation_id, "event:1", "durable event")
    turn = inbox.next_turn(conversation_id, "durable prompt")
    assert turn is not None

    def conversation():
        return LocalConversation(
            agent=Agent(
                llm=LLM(
                    model="openai/gpt-5.6",
                    api_key=SecretStr("test-key"),
                ),
                tools=[],
            ),
            workspace=tmp_path,
            persistence_dir=tmp_path / "openhands-state",
            conversation_id=conversation_id,
            visualizer=None,
        )

    first = conversation()
    try:
        for message in turn.messages:
            first.send_message(message.body, sender=message.sender)
    finally:
        first.close()

    reopened = conversation()
    try:
        event_count = len(reopened.state.events)
        recovered = deliver_turn_messages(reopened, inbox, turn.turn_id)
        assert len(reopened.state.events) == event_count
        assert recovered.state is DeliveryState.DELIVERED
    finally:
        reopened.close()


def test_event_summary_bounds_fields_and_keeps_the_latest_text():
    long_text = "discarded-prefix-" + "x" * EVENT_TEXT_LIMIT + "-latest"
    event = SimpleNamespace(
        source="agent",
        thought=long_text,
        action={"command": long_text},
        status="running",
    )

    summary = event_summary(event)

    assert len(summary["thought"].encode()) <= EVENT_TEXT_LIMIT
    assert summary["thought"].endswith("-latest")
    assert not summary["thought"].startswith("discarded-prefix-")
    assert len(summary["action"].encode()) <= EVENT_TEXT_LIMIT
    assert summary["status"] == "running"
