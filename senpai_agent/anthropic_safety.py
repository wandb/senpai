"""Reject Anthropic classifier refusals and server-side model substitutions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from litellm.types.utils import ModelResponse
from openhands.sdk import LLM
from openhands.sdk.llm.exceptions import (
    LLMContentPolicyViolationError,
    LLMError,
    is_content_policy_violation,
)
from openhands.sdk.llm.llm_response import LLMResponse


class AnthropicModelFallbackError(LLMError):
    """Anthropic server-side fallback was configured or served."""


class AnthropicSafetyRefusalError(LLMError):
    """Anthropic refused a request under its safety policy."""


def _is_anthropic(model: str) -> bool:
    return model.partition("/")[0].casefold() == "anthropic"


def _usage_iterations(response: ModelResponse) -> list[dict[str, Any]]:
    usage = response.get("usage")
    if not usage:
        return []
    return usage.get("iterations") or []


def _server_fallback_requested(call_kwargs: dict[str, Any]) -> bool:
    extra_body = call_kwargs.get("extra_body") or {}
    return "fallbacks" in call_kwargs or "fallbacks" in extra_body


def _restore_stream_iterations(
    response: ModelResponse,
    iterations: list[dict[str, Any]],
) -> ModelResponse:
    if iterations:
        response["usage"]["iterations"] = iterations
    return response


class AnthropicSafetyLLM(LLM):
    """OpenHands LLM that rejects Anthropic's safety fallback path."""

    def _finalize_completion_params(self, *args: Any, **kwargs: Any) -> Any:
        prepared = super()._finalize_completion_params(*args, **kwargs)
        if _is_anthropic(self.model) and _server_fallback_requested(prepared[3]):
            raise AnthropicModelFallbackError(
                "Anthropic server-side fallback configuration rejected before "
                f"request: requested={self.model!r}."
            )
        return prepared

    def _transport_call(self, *, on_token: Any = None, **kwargs: Any) -> ModelResponse:
        iterations: list[dict[str, Any]] = []

        def inspect_token(chunk: ModelResponse) -> Any:
            if chunk_iterations := _usage_iterations(chunk):
                iterations[:] = chunk_iterations
            return on_token(chunk)

        response = super()._transport_call(
            on_token=(
                inspect_token
                if _is_anthropic(self.model) and on_token is not None
                else on_token
            ),
            **kwargs,
        )
        return _restore_stream_iterations(response, iterations)

    async def _atransport_call(
        self,
        *,
        on_token: Any = None,
        **kwargs: Any,
    ) -> ModelResponse:
        iterations: list[dict[str, Any]] = []

        def inspect_token(chunk: ModelResponse) -> Any:
            if chunk_iterations := _usage_iterations(chunk):
                iterations[:] = chunk_iterations
            return on_token(chunk)

        response = await super()._atransport_call(
            on_token=(
                inspect_token
                if _is_anthropic(self.model) and on_token is not None
                else on_token
            ),
            **kwargs,
        )
        return _restore_stream_iterations(response, iterations)

    def _validate_chat_response(
        self,
        response: ModelResponse,
        **kwargs: Any,
    ) -> ModelResponse:
        response = super()._validate_chat_response(response, **kwargs)
        iterations = _usage_iterations(response)
        if _is_anthropic(self.model) and any(
            iteration.get("type") == "fallback_message"
            for iteration in iterations
        ):
            models = " -> ".join(
                iteration["model"]
                for iteration in iterations
                if iteration.get("type") in {"message", "fallback_message"}
                and iteration.get("model")
            )
            raise AnthropicModelFallbackError(
                "Anthropic safety fallback rejected: "
                f"requested={self.model!r}, served={response.get('model')!r}, "
                f"attempts={models or 'unknown'}. The response was discarded."
            )

        finish_reason = response["choices"][0].get("finish_reason")
        if _is_anthropic(self.model) and finish_reason in {
            "content_filter",
            "refusal",
        }:
            raise AnthropicSafetyRefusalError(
                "Anthropic safety refusal: "
                f"requested={self.model!r}, finish_reason={finish_reason!r}. "
                "Senpai will not retry the same request or route it to another model."
            )
        return response

    def _handle_error(
        self,
        error: Exception,
        fallback_call_fn: Callable[[LLM], LLMResponse],
    ) -> LLMResponse:
        self._raise_for_anthropic_safety(error)
        return super()._handle_error(error, fallback_call_fn)

    async def _ahandle_error(
        self,
        error: Exception,
        fallback_call_fn: Callable[[LLM], LLMResponse],
    ) -> LLMResponse:
        self._raise_for_anthropic_safety(error)
        return await super()._ahandle_error(error, fallback_call_fn)

    def _raise_for_anthropic_safety(self, error: Exception) -> None:
        if not _is_anthropic(self.model):
            return
        if not isinstance(
            error,
            (AnthropicModelFallbackError, AnthropicSafetyRefusalError),
        ) and not (
            isinstance(error, LLMContentPolicyViolationError)
            or is_content_policy_violation(error)
        ):
            return

        assert self._telemetry is not None
        self._telemetry.on_error(error)
        if isinstance(
            error,
            (AnthropicModelFallbackError, AnthropicSafetyRefusalError),
        ):
            raise error
        raise AnthropicSafetyRefusalError(
            "Anthropic safety refusal: "
            f"requested={self.model!r}. Senpai will not retry the same request "
            "or route it to another model."
        ) from error


def enforce_anthropic_safety(llm: LLM) -> LLM:
    """Add Anthropic safety-fallback detection without changing LLM behavior."""

    if not _is_anthropic(llm.model) or isinstance(llm, AnthropicSafetyLLM):
        return llm
    values = llm.model_dump()
    values["fallback_strategy"] = llm.fallback_strategy
    values["retry_listener"] = llm.retry_listener
    return AnthropicSafetyLLM.model_validate(values)
