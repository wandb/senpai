# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Reasoning-effort choices and the per-model rules shared by the launcher and runtime."""

REASONING_EFFORTS = (
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "none",
)


def openhands_reasoning_effort(reasoning_effort: str, model: str) -> str:
    provider, _, model_name = model.lower().partition("/")
    supports_openai_pro = provider == "openai" and (
        model_name in {"gpt-5.6", "gpt-6-astra"}
        or model_name.startswith("gpt-5.6-")
    )
    if reasoning_effort not in REASONING_EFFORTS:
        choices = ", ".join(REASONING_EFFORTS)
        raise ValueError(
            f"unsupported reasoning effort {reasoning_effort!r}; "
            f"choose one of: {choices}"
        )
    if provider == "openai" and model_name == "gpt-6-astra" and reasoning_effort == "none":
        raise ValueError(
            f"reasoning effort {reasoning_effort!r} is unsupported for model {model!r}"
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
            "use an anthropic model, openai/gpt-5.6, openai/gpt-6-astra, "
            "or select a lower effort"
        )
    return reasoning_effort
