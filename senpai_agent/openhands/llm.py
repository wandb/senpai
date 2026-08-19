"""Model profiles and provider-specific OpenHands LLM configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from openhands.sdk import LLM
from pydantic import SecretStr

from senpai_agent.secrets import resolve_api_key

DEFAULT_MODEL = "openai/gpt-5.6-sol"
DEFAULT_FAST_MODEL = "openai/gpt-5.6-luna"
DEFAULT_FRONTIER_MODEL = "openai/gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_FAST_REASONING_EFFORT = "high"
DEFAULT_FRONTIER_REASONING_EFFORT = "max"
DEFAULT_COMPACTION_TRIGGER_TOKENS = 200_000
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max", "none")
PROVIDER_API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "wandb": "WANDB_API_KEY",
}


@dataclass(frozen=True)
class _ModelProfile:
    model: str
    api_key_env: str
    api_key: SecretStr
    reasoning_effort: str


@dataclass(frozen=True)
class _ModelProfiles:
    main: _ModelProfile
    smart: _ModelProfile
    fast: _ModelProfile
    frontier: _ModelProfile

    @property
    def uses_wandb(self) -> bool:
        return any(model_provider(p.model) == "wandb" for p in vars(self).values())


def openhands_reasoning_effort(reasoning_effort: str, model: str) -> str:
    provider, _, model_name = model.lower().partition("/")
    supports_openai_pro = provider == "openai" and (
        model_name == "gpt-5.6" or model_name.startswith("gpt-5.6-")
    )
    if reasoning_effort not in REASONING_EFFORTS:
        choices = ", ".join(REASONING_EFFORTS)
        raise ValueError(
            f"unsupported reasoning effort {reasoning_effort!r}; "
            f"choose one of: {choices}"
        )
    if provider == "wandb" and model_name == "zai-org/glm-5.2":
        if reasoning_effort not in {"high", "max"}:
            raise ValueError(
                f"reasoning effort {reasoning_effort!r} is unsupported for "
                f"model {model!r}; use 'high' or 'max'"
            )
        return reasoning_effort
    if (
        reasoning_effort == "max"
        and provider != "anthropic"
        and not supports_openai_pro
    ):
        raise ValueError(
            f"reasoning effort {reasoning_effort!r} is unsupported for model "
            f"{model!r}; "
            "use an anthropic model, an openai/gpt-5.6 model, or select a lower effort"
        )
    return reasoning_effort


def uses_openai_pro_mode(model: str, reasoning_effort: str | None) -> bool:
    return (
        reasoning_effort is not None
        and model.split("/", 1)[0].lower() == "openai"
        and openhands_reasoning_effort(reasoning_effort, model) == "max"
    )


def _openai_pro_reasoning(
    model: str,
    reasoning_effort: str | None,
) -> dict[str, str] | None:
    if not uses_openai_pro_mode(model, reasoning_effort):
        return None
    return {
        "effort": "max", "mode": "pro", "summary": "auto", "context": "all_turns"
    }


def apply_reasoning_profile(llm: LLM) -> LLM:
    """Validate effort and replace only Senpai's reasoning request body."""

    reasoning_effort = openhands_reasoning_effort(
        llm.reasoning_effort,
        llm.model,
    )
    extra_body = dict(llm.litellm_extra_body or {})
    if reasoning := _openai_pro_reasoning(llm.model, reasoning_effort):
        extra_body["reasoning"] = reasoning
    else:
        extra_body.pop("reasoning", None)
    return llm.model_copy(
        update={
            "reasoning_effort": reasoning_effort,
            "litellm_extra_body": extra_body,
        }
    )


def model_provider(model: str) -> str:
    provider, separator, model_name = model.lower().partition("/")
    if not separator or not provider or not model_name:
        raise ValueError(
            f"model {model!r} must use a provider/model identifier such as "
            "'anthropic/claude-opus-4-8' or 'openai/gpt-5.6-sol'"
        )
    return provider


def infer_api_key_env(model: str, *, override_env: str | None = None) -> str:
    provider = model_provider(model)
    try:
        return PROVIDER_API_KEY_ENVS[provider]
    except KeyError as error:
        hint = f"; set {override_env} explicitly" if override_env else ""
        raise ValueError(
            f"cannot infer an API key environment variable for model provider "
            f"{provider!r}{hint}"
        ) from error


def profile_api_key_env(
    model: str,
    env: Mapping[str, str],
    env_key: str,
    *,
    inherited_model: str,
    inherited_api_key_env: str,
) -> str:
    if explicit := env.get(env_key, "").strip():
        return explicit
    if model_provider(model) == model_provider(inherited_model):
        return inherited_api_key_env
    return infer_api_key_env(model, override_env=env_key)


def _select_profile(
    env: Mapping[str, str],
    *,
    model_override: str | None,
    effort_override: str | None,
    model_env: str,
    effort_env: str,
    api_key_env: str,
    default_model: str,
    default_effort: str,
    inherited: _ModelProfile | None = None,
    api_key_env_override: str | None = None,
) -> _ModelProfile:
    model = (model_override if model_override is not None else env.get(model_env)) or (
        default_model
    )
    effort = (effort_override if effort_override is not None else env.get(effort_env))
    effort = openhands_reasoning_effort(effort or default_effort, model)
    if inherited is None:
        selected_key_env = (
            api_key_env_override
            if api_key_env_override is not None
            else env.get(api_key_env)
        ) or infer_api_key_env(model, override_env=api_key_env)
    else:
        selected_key_env = profile_api_key_env(
            model,
            env,
            api_key_env,
            inherited_model=inherited.model,
            inherited_api_key_env=inherited.api_key_env,
        )
    return _ModelProfile(
        model,
        selected_key_env,
        resolve_api_key(env, selected_key_env),
        effort,
    )


def resolve_model_profiles(
    env: Mapping[str, str],
    *,
    child: bool,
    model: str | None = None,
    api_key_env: str | None = None,
    reasoning_effort: str | None = None,
    smart_model: str | None = None,
    smart_reasoning_effort: str | None = None,
    fast_model: str | None = None,
    fast_reasoning_effort: str | None = None,
    frontier_model: str | None = None,
    frontier_reasoning_effort: str | None = None,
) -> _ModelProfiles:
    main = _select_profile(
        env,
        model_override=model,
        effort_override=reasoning_effort,
        model_env="SENPAI_OPENHANDS_MODEL",
        effort_env="SENPAI_OPENHANDS_REASONING_EFFORT",
        api_key_env="SENPAI_OPENHANDS_API_KEY_ENV",
        default_model=DEFAULT_MODEL,
        default_effort=DEFAULT_REASONING_EFFORT,
        api_key_env_override=api_key_env,
    )
    smart_default_model = (
        env.get("SENPAI_OPENHANDS_MODEL") if child else main.model
    ) or DEFAULT_MODEL
    smart_default_effort = (
        env.get("SENPAI_OPENHANDS_REASONING_EFFORT")
        if child
        else main.reasoning_effort
    ) or DEFAULT_REASONING_EFFORT
    smart = _select_profile(
        env,
        model_override=smart_model,
        effort_override=smart_reasoning_effort,
        model_env="SENPAI_OPENHANDS_SMART_MODEL",
        effort_env="SENPAI_OPENHANDS_SMART_REASONING_EFFORT",
        api_key_env="SENPAI_OPENHANDS_SMART_API_KEY_ENV",
        default_model=smart_default_model,
        default_effort=smart_default_effort,
        inherited=main,
    )
    fast_default_model = (
        DEFAULT_FAST_MODEL
        if model_provider(smart.model) == model_provider(DEFAULT_FAST_MODEL)
        else smart.model
    )
    fast = _select_profile(
        env,
        model_override=fast_model,
        effort_override=fast_reasoning_effort,
        model_env="SENPAI_OPENHANDS_FAST_MODEL",
        effort_env="SENPAI_OPENHANDS_FAST_REASONING_EFFORT",
        api_key_env="SENPAI_OPENHANDS_FAST_API_KEY_ENV",
        default_model=fast_default_model,
        default_effort=DEFAULT_FAST_REASONING_EFFORT,
        inherited=smart,
    )
    frontier = _select_profile(
        env,
        model_override=frontier_model,
        effort_override=frontier_reasoning_effort,
        model_env="SENPAI_OPENHANDS_FRONTIER_MODEL",
        effort_env="SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT",
        api_key_env="SENPAI_OPENHANDS_FRONTIER_API_KEY_ENV",
        default_model=DEFAULT_FRONTIER_MODEL,
        default_effort=DEFAULT_FRONTIER_REASONING_EFFORT,
        inherited=main,
    )
    return _ModelProfiles(
        main=main,
        smart=smart,
        fast=fast,
        frontier=frontier,
    )


def resolve_compaction_trigger_tokens(
    parsed_value: int | None,
    env: Mapping[str, str],
) -> int:
    raw_value = (
        parsed_value
        if parsed_value is not None
        else env.get(
            "SENPAI_COMPACTION_TRIGGER_TOKENS",
            str(DEFAULT_COMPACTION_TRIGGER_TOKENS),
        )
    )
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            "SENPAI_COMPACTION_TRIGGER_TOKENS must be an integer"
        ) from error
    if value < 50_000:
        raise RuntimeError("SENPAI_COMPACTION_TRIGGER_TOKENS must be at least 50000")
    return value


def prompt_cache_configuration(model: str) -> dict[str, object]:
    provider, _, model_name = model.lower().partition("/")
    if provider == "anthropic" and "prompt_cache_ttl" in LLM.model_fields:
        return {"prompt_cache_ttl": "1h"}
    if provider == "openai":
        if model_name.startswith("gpt-5.6"):
            return {
                "prompt_cache_retention": None,
                "responses_prompt_cache_breakpoint": True,
                "litellm_extra_body": {
                    "prompt_cache_options": {
                        "mode": "explicit",
                        "ttl": "30m",
                    }
                },
            }
        return {"prompt_cache_retention": "24h"}
    return {}


def openai_responses_configuration(
    model: str,
    reasoning_effort: str | None = None,
) -> dict[str, object]:
    if model.split("/", 1)[0].lower() != "openai":
        return {}
    configuration: dict[str, object] = {
        "api_mode": "responses",
        # OpenAI defines "auto" as the most detailed summarizer available.
        "reasoning_summary": "auto",
        "reasoning_context": "all_turns",
        "responses_store": True,
        "responses_use_previous_response_id": True,
    }
    if reasoning := _openai_pro_reasoning(model, reasoning_effort):
        configuration["litellm_extra_body"] = {"reasoning": reasoning}
    return configuration


def compaction_configuration(
    model: str,
    trigger_tokens: int,
) -> dict[str, int]:
    """Translate the universal token trigger to the provider SDK field."""

    provider = model_provider(model)
    if provider == "openai":
        return {"responses_compact_threshold": trigger_tokens}
    if provider == "anthropic":
        return {"anthropic_compact_threshold": trigger_tokens}
    return {}


def model_runtime_configuration(
    model: str,
    reasoning_effort: str,
    *,
    compaction_trigger_tokens: int,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> dict[str, object]:
    """Merge provider options, including nested LiteLLM request fields."""

    if model_provider(model) == "wandb":
        if not (wandb_entity and wandb_project):
            raise ValueError(
                "wandb_entity and wandb_project are required for W&B Inference"
            )
        configuration: dict[str, object] = {
            "api_mode": "chat",
            "base_url": WANDB_INFERENCE_BASE_URL,
            "extra_headers": {"OpenAI-Project": f"{wandb_entity}/{wandb_project}"},
            "capability_overrides": {
                "supports_reasoning_effort": False,
                "supports_responses_api": False,
            },
            "max_input_tokens": 262_144,
            "max_output_tokens": 16_384,
        }
        if model.lower() == "wandb/zai-org/glm-5.2":
            configuration["litellm_extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": reasoning_effort,
                },
            }
        return configuration

    configuration = {}
    extra_body: dict[str, object] = {}
    for options in (
        prompt_cache_configuration(model),
        openai_responses_configuration(model, reasoning_effort),
        compaction_configuration(model, compaction_trigger_tokens),
    ):
        for key, value in options.items():
            if key == "litellm_extra_body":
                extra_body.update(value)
            else:
                configuration[key] = value
    if extra_body:
        configuration["litellm_extra_body"] = extra_body
    return configuration
