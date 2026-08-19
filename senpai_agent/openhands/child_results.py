"""Extract and compact delegated OpenHands child results."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from openhands.sdk import LLM
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.llm import Message, TextContent

from senpai_agent.openhands.config import RunnerConfig
from senpai_agent.openhands.conversation import graceful_interrupts, run_conversation
from senpai_agent.PROMPTS import DELEGATED_RESULT_SUMMARY_PROMPT, render_prompt

MAX_INLINE_CHILD_RESULT_TOKENS = 15_000


def final_agent_result(
    conversation: object,
    *,
    exclude_event_ids: frozenset[str] = frozenset(),
) -> str:
    for event in reversed(conversation.state.view.events):
        if str(event.id) in exclude_event_ids:
            continue
        if isinstance(event, MessageEvent) and event.source == "agent":
            text = "".join(
                content.text
                for content in event.to_llm_message().content
                if isinstance(content, TextContent)
            ).strip()
            if text:
                return text
        if isinstance(event, ActionEvent):
            message = getattr(event.action, "message", None)
            if isinstance(message, str) and message.strip():
                return message.strip()
    raise RuntimeError("child finished without a model-visible result")


def _result_token_count(llm: LLM, result: str) -> int:
    return llm.get_token_count(
        [
            Message(
                role="assistant",
                content=[TextContent(text=result)],
            )
        ]
    )


def _store_oversized_child_result(config: RunnerConfig, result: str) -> Path:
    if config.delegation_root_state_dir is None or config.delegation_task_id is None:
        raise RuntimeError(
            "oversized child result requires delegated role-state storage"
        )
    directory = config.delegation_root_state_dir / "delegation" / "results"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(dir=directory)
    temporary_path = Path(temporary_name)
    path = directory / f"{config.delegation_task_id}.md"
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(result)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def compact_child_result(
    conversation: object,
    llm: LLM,
    config: RunnerConfig,
    result: str,
    run_deadline: float,
) -> str:
    token_count = _result_token_count(llm, result)
    if 0 < token_count <= MAX_INLINE_CHILD_RESULT_TOKENS:
        return result

    artifact = _store_oversized_child_result(config, result)
    existing_event_ids = frozenset(
        str(event.id) for event in conversation.state.view.events
    )
    try:
        conversation.send_message(
            render_prompt(
                DELEGATED_RESULT_SUMMARY_PROMPT,
                RESULT_PATH=str(artifact),
            )
        )
        with graceful_interrupts(conversation):
            run_conversation(conversation, run_deadline - time.time())
        if conversation.state.execution_status != ConversationExecutionStatus.FINISHED:
            raise RuntimeError("summary turn did not finish")
        summary = final_agent_result(
            conversation,
            exclude_event_ids=existing_event_ids,
        )
        summary_tokens = _result_token_count(llm, summary)
        if not 0 < summary_tokens <= MAX_INLINE_CHILD_RESULT_TOKENS:
            raise RuntimeError(
                "summary token count is unavailable or exceeds the child result limit"
            )
    except Exception as error:
        raise RuntimeError(
            f"oversized child report saved at {artifact}; summarization failed"
        ) from error
    return f"{summary}\n\nFull report: {artifact}"
