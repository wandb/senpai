import asyncio
import re
import tomllib
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx
import pytest
from litellm.exceptions import ContentPolicyViolationError, ServiceUnavailableError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from openhands.sdk import Agent, LLM, LocalConversation
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.fallback_strategy import FallbackStrategy
from pydantic import SecretStr

from senpai_agent.anthropic_safety import (
    AnthropicModelFallbackError,
    AnthropicSafetyLLM,
    AnthropicSafetyRefusalError,
    enforce_anthropic_safety,
)
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


def _completion_response(
    *,
    model: str,
    finish_reason: str = "stop",
    content: str = "accepted response",
    iterations: list[dict[str, object]] | None = None,
) -> ModelResponse:
    return ModelResponse(
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            iterations=iterations or [],
        ),
    )


def _anthropic_llm() -> AnthropicSafetyLLM:
    return AnthropicSafetyLLM(
        model="anthropic/claude-fable-5",
        api_key=SecretStr("test-key"),
        num_retries=5,
    )


def _normalized_anthropic_fallback() -> ModelResponse:
    payload = {
        "id": "msg_fallback_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [
            {
                "type": "fallback",
                "from": {"model": "claude-fable-5"},
                "to": {"model": "claude-opus-4-8"},
            },
            {"type": "text", "text": "substituted response"},
        ],
        "stop_reason": "end_turn",
        "stop_details": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "iterations": [
                {
                    "type": "message",
                    "model": "claude-fable-5",
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-opus-4-8",
                    "input_tokens": 12,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            ],
        },
    }
    return AnthropicConfig().transform_parsed_response(
        completion_response=payload,
        raw_response=httpx.Response(200),
        model_response=ModelResponse(),
    )


def _streamed_anthropic_fallback() -> list[ModelResponseStream]:
    return [
        ModelResponseStream(
            id="msg_fallback_test",
            model="claude-opus-4-8",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content="substituted response"),
                )
            ],
        ),
        ModelResponseStream(
            id="msg_fallback_test",
            model="claude-opus-4-8",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=12,
                completion_tokens=2,
                total_tokens=14,
                iterations=[
                    {"type": "message", "model": "claude-fable-5"},
                    {
                        "type": "fallback_message",
                        "model": "claude-opus-4-8",
                    },
                ],
            ),
        ),
    ]


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


def test_anthropic_fallback_fails_without_retry(monkeypatch):
    response = _normalized_anthropic_fallback()
    calls = 0

    def complete(**_kwargs):
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_completion", complete)

    with pytest.raises(
        AnthropicModelFallbackError,
        match="Anthropic safety fallback rejected.*claude-opus-4-8",
    ):
        _anthropic_llm().completion(
            [Message(role="user", content=[TextContent(text="Investigate")])]
        )

    assert calls == 1


def test_streamed_anthropic_fallback_fails_without_retry(monkeypatch):
    chunks = _streamed_anthropic_fallback()
    seen_chunks = []
    calls = 0

    def complete(**_kwargs):
        nonlocal calls
        calls += 1
        return iter(chunks)

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_completion", complete)
    llm = _anthropic_llm().model_copy(update={"stream": True})

    with pytest.raises(
        AnthropicModelFallbackError,
        match="Anthropic safety fallback rejected.*claude-opus-4-8",
    ):
        llm.completion(
            [Message(role="user", content=[TextContent(text="Investigate")])],
            on_token=seen_chunks.append,
        )

    assert calls == 1
    assert seen_chunks == chunks


def test_async_streamed_anthropic_fallback_fails_without_retry(monkeypatch):
    chunks = _streamed_anthropic_fallback()
    seen_chunks = []
    calls = 0

    async def stream():
        for chunk in chunks:
            yield chunk

    async def complete(**_kwargs):
        nonlocal calls
        calls += 1
        return stream()

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_acompletion", complete)
    llm = _anthropic_llm().model_copy(update={"stream": True})

    async def call():
        return await llm.acompletion(
            [Message(role="user", content=[TextContent(text="Investigate")])],
            on_token=seen_chunks.append,
        )

    with pytest.raises(AnthropicModelFallbackError):
        asyncio.run(call())

    assert calls == 1
    assert seen_chunks == chunks


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (
            _completion_response(
                model="claude-opus-4-8",
                iterations=[
                    {"type": "message", "model": "claude-fable-5"},
                    {
                        "type": "fallback_message",
                        "model": "claude-opus-4-8",
                    },
                ],
            ),
            AnthropicModelFallbackError,
        ),
        (
            _completion_response(
                model="claude-fable-5",
                finish_reason="content_filter",
                content="refused",
                iterations=[{"type": "message", "model": "claude-fable-5"}],
            ),
            AnthropicSafetyRefusalError,
        ),
    ],
)
def test_async_anthropic_safety_response_fails_without_retry(
    monkeypatch,
    response,
    error_type,
):
    calls = 0

    async def complete(**_kwargs):
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_acompletion", complete)

    async def call():
        return await _anthropic_llm().acompletion(
            [Message(role="user", content=[TextContent(text="Investigate")])]
        )

    with pytest.raises(error_type):
        asyncio.run(call())

    assert calls == 1


def test_anthropic_refusal_fails_without_retry(monkeypatch):
    response = _completion_response(
        model="claude-fable-5",
        finish_reason="content_filter",
        content="refused",
        iterations=[{"type": "message", "model": "claude-fable-5"}],
    )
    calls = 0

    def complete(**_kwargs):
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_completion", complete)

    with pytest.raises(
        AnthropicSafetyRefusalError,
        match="Anthropic safety refusal.*will not retry",
    ):
        _anthropic_llm().completion(
            [Message(role="user", content=[TextContent(text="Investigate")])]
        )

    assert calls == 1


def test_anthropic_provider_exception_becomes_a_loud_refusal(monkeypatch):
    calls = 0

    def refuse(**_kwargs):
        nonlocal calls
        calls += 1
        raise ContentPolicyViolationError(
            "blocked",
            model="claude-fable-5",
            llm_provider="anthropic",
        )

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_completion", refuse)

    with pytest.raises(AnthropicSafetyRefusalError, match="Anthropic safety refusal"):
        _anthropic_llm().completion(
            [Message(role="user", content=[TextContent(text="Investigate")])]
        )

    assert calls == 1


def test_async_anthropic_provider_exception_becomes_a_loud_refusal(monkeypatch):
    calls = 0

    async def refuse(**_kwargs):
        nonlocal calls
        calls += 1
        raise ContentPolicyViolationError(
            "blocked",
            model="claude-fable-5",
            llm_provider="anthropic",
        )

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_acompletion", refuse)

    async def call():
        return await _anthropic_llm().acompletion(
            [Message(role="user", content=[TextContent(text="Investigate")])]
        )

    with pytest.raises(AnthropicSafetyRefusalError, match="Anthropic safety refusal"):
        asyncio.run(call())

    assert calls == 1


def test_normal_anthropic_response_is_unchanged(monkeypatch):
    response = _completion_response(
        model="claude-fable-5",
        iterations=[
            {"type": "compaction", "model": "claude-fable-5"},
            {"type": "message", "model": "claude-fable-5"},
        ],
    )
    monkeypatch.setattr(
        "openhands.sdk.llm.llm.litellm_completion",
        lambda **_kwargs: response,
    )

    result = _anthropic_llm().completion(
        [Message(role="user", content=[TextContent(text="Investigate")])]
    )

    assert result.raw_response is response
    assert result.message.content == [TextContent(text="accepted response")]


def test_anthropic_request_omits_server_side_fallback():
    model = "anthropic/claude-fable-5"
    configuration = model_runtime_configuration(
        model,
        "max",
        compaction_trigger_tokens=TEST_COMPACTION_TRIGGER_TOKENS,
    )
    llm = AnthropicSafetyLLM(
        model=model,
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

    assert "fallbacks" not in call_kwargs
    assert "fallbacks" not in call_kwargs.get("extra_body", {})


def test_anthropic_server_fallback_configuration_fails_before_request():
    llm = AnthropicSafetyLLM(
        model="anthropic/claude-fable-5",
        api_key=SecretStr("test-key"),
        litellm_extra_body={"fallbacks": "default"},
    )

    with pytest.raises(
        AnthropicModelFallbackError,
        match="server-side fallback configuration rejected before request",
    ):
        llm._prepare_completion_params(
            [Message(role="user", content=[TextContent(text="Investigate")])],
            tools=None,
            add_security_risk_prediction=False,
            kwargs={},
        )


def test_async_anthropic_server_fallback_configuration_fails_before_request():
    llm = AnthropicSafetyLLM(
        model="anthropic/claude-fable-5",
        api_key=SecretStr("test-key"),
        litellm_extra_body={"fallbacks": "default"},
    )

    async def call():
        return await llm.acompletion(
            [Message(role="user", content=[TextContent(text="Investigate")])]
        )

    with pytest.raises(
        AnthropicModelFallbackError,
        match="server-side fallback configuration rejected before request",
    ):
        asyncio.run(call())


def test_anthropic_profile_preserves_streaming_and_fallback_configuration():
    def retry_listener(*_args):
        pass
    profile_llm = LLM(
        model="anthropic/claude-fable-5",
        api_key=SecretStr("profile-key"),
        stream=True,
        fallback_strategy=FallbackStrategy(fallback_llms=["backup-profile"]),
        retry_listener=retry_listener,
    )

    guarded_llm = enforce_anthropic_safety(profile_llm)

    assert type(guarded_llm) is AnthropicSafetyLLM
    assert guarded_llm.model == profile_llm.model
    assert guarded_llm.api_key == profile_llm.api_key
    assert guarded_llm.stream is True
    assert guarded_llm.fallback_strategy == profile_llm.fallback_strategy
    assert guarded_llm.retry_listener is retry_listener
    assert type(guarded_llm.model_copy()) is AnthropicSafetyLLM


def test_anthropic_transport_failure_keeps_openhands_fallback(monkeypatch):
    strategy = FallbackStrategy(fallback_llms=["unused"])
    strategy._resolved = [
        LLM(
            model="openai/gpt-5",
            api_key=SecretStr("test-key"),
            num_retries=0,
        )
    ]
    llm = AnthropicSafetyLLM(
        model="anthropic/claude-fable-5",
        api_key=SecretStr("test-key"),
        num_retries=0,
        fallback_strategy=strategy,
    )
    calls = []

    def complete(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise ServiceUnavailableError(
                "down",
                model="claude-fable-5",
                llm_provider="anthropic",
            )
        return _completion_response(model="gpt-5")

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_completion", complete)

    result = llm.completion(
        [Message(role="user", content=[TextContent(text="Investigate")])]
    )

    assert result.message.content == [TextContent(text="accepted response")]
    assert calls == ["claude-fable-5", "gpt-5"]


def test_non_anthropic_profile_is_unchanged():
    profile_llm = LLM(
        model="openai/gpt-5",
        api_key=SecretStr("profile-key"),
        stream=True,
        fallback_strategy=FallbackStrategy(fallback_llms=["backup-profile"]),
    )

    assert enforce_anthropic_safety(profile_llm) is profile_llm


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
