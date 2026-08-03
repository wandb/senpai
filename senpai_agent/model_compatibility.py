"""Model capability compatibility for pinned Senpai dependencies."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


OPENAI_EXTENDED_EFFORT_PREFIX = "openai/gpt-5.6"
ANTHROPIC_MAX_EFFORT_MODELS = frozenset(
    {
        "anthropic/claude-fable-5",
        "anthropic/claude-mythos-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4-6",
    }
)


CLAUDE_OPUS_5_MODEL_INFO: dict[str, Any] = {
    "cache_creation_input_token_cost": 6.25e-06,
    "cache_creation_input_token_cost_above_1hr": 1e-05,
    "cache_read_input_token_cost": 5e-07,
    "input_cost_per_token": 5e-06,
    "litellm_provider": "anthropic",
    "max_input_tokens": 1_000_000,
    "max_output_tokens": 128_000,
    "max_tokens": 128_000,
    "mode": "chat",
    "output_cost_per_token": 2.5e-05,
    "prompt_cache_min_tokens": 512,
    "provider_specific_entry": {"fast": 2.0, "us": 1.1},
    "search_context_cost_per_query": {
        "search_context_size_high": 0.01,
        "search_context_size_low": 0.01,
        "search_context_size_medium": 0.01,
    },
    "supports_adaptive_thinking": True,
    "supports_assistant_prefill": False,
    "supports_computer_use": True,
    "supports_function_calling": True,
    "supports_max_reasoning_effort": True,
    "supports_native_structured_output": True,
    "supports_output_config": True,
    "supports_pdf_input": True,
    "supports_prompt_caching": True,
    "supports_reasoning": True,
    "supports_response_schema": True,
    "supports_sampling_params": False,
    "supports_speed": True,
    "supports_tool_choice": True,
    "supports_vision": True,
    "supports_xhigh_reasoning_effort": True,
}


def is_openai_extended_effort_model(model: str) -> bool:
    normalized = model.lower()
    return normalized == OPENAI_EXTENDED_EFFORT_PREFIX or normalized.startswith(
        f"{OPENAI_EXTENDED_EFFORT_PREFIX}-"
    )


def supports_reasoning_effort(model: str, effort: str) -> bool:
    if effort == "ultra":
        return is_openai_extended_effort_model(model)
    if effort == "max":
        return is_openai_extended_effort_model(
            model
        ) or model.lower() in ANTHROPIC_MAX_EFFORT_MODELS
    return True


def register_litellm_model_compatibility(
    model_cost: MutableMapping[str, dict[str, Any]] | None = None,
) -> None:
    """Register Opus 5 when the pinned LiteLLM's remote catalog is unavailable."""
    if model_cost is None:
        import litellm

        model_cost = litellm.model_cost
        register_model = litellm.register_model
    else:
        register_model = None

    if "claude-opus-5" in model_cost:
        return
    if register_model is not None:
        register_model({"claude-opus-5": CLAUDE_OPUS_5_MODEL_INFO})
    else:
        model_cost["claude-opus-5"] = CLAUDE_OPUS_5_MODEL_INFO.copy()
