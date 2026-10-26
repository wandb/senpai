"""Settings for a self-hosted OpenAI-compatible chat-completions server."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

PROVIDER = "custom_openai"
DEFAULT_MAX_OUTPUT_TOKENS = 65_536
# OpenHands rejects a model whose input window is smaller than this.
MIN_INPUT_TOKENS = 16_384


@dataclass(frozen=True)
class CustomOpenAIEndpoint:
    """A self-hosted OpenAI-compatible chat-completions server."""

    base_url: str
    context_tokens: int
    max_output_tokens: int
    canonical_model: str | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("custom_openai_base_url must be an http(s) URL")
        if self.max_output_tokens <= 0:
            raise ValueError("custom_openai_max_output_tokens must be positive")
        if self.max_input_tokens < MIN_INPUT_TOKENS:
            raise ValueError(
                "custom_openai_context_tokens must exceed "
                f"custom_openai_max_output_tokens by at least {MIN_INPUT_TOKENS}"
            )

    @property
    def max_input_tokens(self) -> int:
        # vLLM-style servers reject input plus requested output beyond the window.
        return self.context_tokens - self.max_output_tokens


def served_model(models: Iterable[str]) -> str | None:
    """Return the one served model ID that every custom_openai profile shares."""
    served = {
        model_name
        for provider, _, model_name in (model.partition("/") for model in models)
        if provider.lower() == PROVIDER
    }
    # The endpoint's window and canonical model describe a single served model.
    if len(served) > 1:
        raise ValueError(
            "custom_openai profiles must share one served model; got "
            + ", ".join(sorted(served))
        )
    return served.pop() if served else None
