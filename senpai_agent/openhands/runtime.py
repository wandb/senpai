"""Coordinate one Senpai OpenHands runtime turn."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from openhands.sdk import LLM
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.tools.preset.default import register_default_tools

from senpai_agent.delegation import (
    cancel_pending_descendants,
    configure_delegation,
    record_delegated_task_result,
)
from senpai_agent.github.tools import (
    clear_github_credentials,
    configure_github_credentials,
)
from senpai_agent.inbox import DeliveryState, PersistentInbox
from senpai_agent.inference_heartbeat import InferenceHeartbeat
from senpai_agent.openhands import WEAVE_PROJECT
from senpai_agent.openhands import agents, child_results, conversation, llm
from senpai_agent.openhands.config import RunnerConfig, scrub_model_credentials
from senpai_agent.secrets import scrub_github_credentials
from senpai_agent.tools import register_senpai_tools
from senpai_agent.weave_monitoring import register_trace_secret, weave_conversation_url

EVENT_TEXT_LIMIT = 20_000


def event_summary(event: object) -> dict[str, object]:
    summary: dict[str, object] = {"event": event.__class__.__name__}
    for attr in ("source", "tool_name", "action", "status"):
        value = getattr(event, attr, None)
        if value is not None:
            summary[attr] = _bounded_event_text(value)
    thought = getattr(event, "thought", None)
    if thought:
        summary["thought"] = _bounded_event_text(thought)

    message = getattr(event, "llm_message", None)
    if getattr(event, "source", None) == "agent" and message is not None:
        text_parts = [
            getattr(part, "text", "")
            for part in getattr(message, "content", [])
            if getattr(part, "text", "")
        ]
        text = "\n".join(text_parts).strip()
        if text:
            summary["text"] = _bounded_event_text(text)
    return summary


def _bounded_event_text(value: object) -> str:
    text = str(value)
    encoded = text.encode()
    if len(encoded) <= EVENT_TEXT_LIMIT:
        return text
    return encoded[-EVENT_TEXT_LIMIT:].decode(errors="ignore")


def print_event(event: object) -> None:
    print(
        "OPENHANDS_EVENT " + json.dumps(event_summary(event), sort_keys=True),
        flush=True,
    )


def _run_record(
    config: RunnerConfig,
    available_agents: list[str],
    *,
    reset_context: bool,
) -> dict[str, object]:
    return {
        "workspace": str(config.workspace),
        "state_dir": str(config.state_dir),
        "conversation_id": str(config.conversation_id),
        "role": config.role,
        "model": config.model,
        "smart_model": config.smart_model,
        "smart_reasoning_effort": config.smart_reasoning_effort,
        "fast_model": config.fast_model,
        "fast_reasoning_effort": config.fast_reasoning_effort,
        "frontier_model": config.frontier_model,
        "frontier_reasoning_effort": config.frontier_reasoning_effort,
        "compaction_trigger_tokens": config.compaction_trigger_tokens,
        "prompt_cache": (
            llm.prompt_cache_configuration(config.model)
            or {"provider_default": True}
        ),
        "reasoning_effort": config.reasoning_effort,
        "openhands_reasoning_effort": llm.openhands_reasoning_effort(
            config.reasoning_effort,
            config.model,
        ),
        "reasoning_mode": (
            "pro"
            if llm.uses_openai_pro_mode(
                config.model,
                config.reasoning_effort,
            )
            else "standard"
        ),
        "agent": config.agent_name,
        "enable_browser": config.enable_browser,
        "role_file": str(config.role_file),
        "plugin_dir": str(config.plugin_dir),
        "available_agents": available_agents,
        "weave_project": WEAVE_PROJECT,
        "weave_url": weave_conversation_url(
            WEAVE_PROJECT,
            config.conversation_id,
        ),
        "child": config.child,
        "reset_context": reset_context,
    }


def run_openhands(
    prompt: str,
    config: RunnerConfig,
    *,
    reset_context: bool = False,
    inbox: PersistentInbox | None = None,
    inbox_turn_id: str | None = None,
    recovery_prompt: str | None = None,
    on_activity: Callable[[], None] | None = None,
    on_inference_state: (
        Callable[[float | None, float | None], None] | None
    ) = None,
) -> int:
    if (inbox is None) != (inbox_turn_id is None):
        raise ValueError("inbox and inbox_turn_id must be provided together")

    started_at = time.time()
    run_deadline = (
        min(
            started_at + config.timeout_seconds,
            config.delegation_deadline_epoch or float("inf"),
        )
        if config.child
        else None
    )
    if run_deadline is not None and run_deadline <= started_at:
        raise TimeoutError("the inherited OpenHands deadline has expired")

    scrub_model_credentials(os.environ, config)
    register_default_tools(enable_browser=False)
    register_senpai_tools()
    file_agents = agents.sanitized_agent_definitions(config.workspace)
    project_skills = agents.sanitized_project_skills(config.workspace)
    available_agents = [definition.name for definition in file_agents]
    os.environ["SENPAI_CONVERSATION_ID"] = config.conversation_id.hex
    print(
        "OPENHANDS_RUN "
        + json.dumps(
            _run_record(
                config,
                available_agents,
                reset_context=reset_context,
            ),
            sort_keys=True,
        ),
        flush=True,
    )

    if config.github_token is not None:
        register_trace_secret(config.github_token.get_secret_value())
        configure_github_credentials(
            config.github_repo,
            config.github_token,
            trusted_actor=config.github_trusted_actor,
        )
    configure_delegation(
        agents.delegation_config(config, deadline_epoch=run_deadline)
    )
    scrub_github_credentials(os.environ)

    active_conversation = None
    inference_heartbeat = None
    cleanup_error: BaseException | None = None
    active_inbox_turn_id = inbox_turn_id
    try:
        inference_heartbeat = (
            InferenceHeartbeat(on_inference_state)
            if on_inference_state is not None
            else None
        )
        runtime_llm = LLM(
            model=config.model,
            api_key=config.api_key,
            timeout=config.llm_timeout_seconds,
            num_retries=config.llm_num_retries,
            reasoning_effort=llm.openhands_reasoning_effort(
                config.reasoning_effort,
                config.model,
            ),
            usage_id="senpai",
            **llm.model_runtime_configuration(
                config.model,
                config.reasoning_effort,
                compaction_trigger_tokens=config.compaction_trigger_tokens,
                wandb_entity=config.wandb_entity,
                wandb_project=config.wandb_project,
            ),
        )
        if inference_heartbeat is not None:
            runtime_llm.set_request_scope(inference_heartbeat.request)
        agent = agents.build_agent(
            config,
            runtime_llm,
            file_agents,
            project_skills,
        )

        last_activity = time.monotonic()

        def observe_event(event: object) -> None:
            nonlocal last_activity
            last_activity = time.monotonic()
            if on_activity is not None:
                on_activity()
            print_event(event)

        active_conversation = conversation.create_conversation(
            agent,
            config,
            observe_event,
        )
        outcome = conversation.run_turn(
            active_conversation,
            prompt,
            config,
            run_deadline=run_deadline,
            reset_context=reset_context,
            inbox=inbox,
            inbox_turn_id=inbox_turn_id,
            recovery_prompt=recovery_prompt,
            activity=lambda: last_activity,
        )
        status = outcome.status
        active_inbox_turn_id = outcome.active_inbox_turn_id
        child_result = (
            child_results.final_agent_result(active_conversation)
            if config.child and status == ConversationExecutionStatus.FINISHED
            else None
        )
        if child_result is not None:
            assert run_deadline is not None
            child_result = child_results.compact_child_result(
                active_conversation,
                agent.llm,
                config,
                child_result,
                run_deadline,
            )
    finally:
        primary_exception = sys.exc_info()[1]
        if (
            config.child
            and config.delegation_task_id
            and active_conversation is not None
        ):
            registry_value = os.environ.get("SENPAI_DELEGATION_REGISTRY_PATH")
            if registry_value:
                try:
                    if primary_exception is not None:
                        record_delegated_task_result(
                            config.delegation_task_id,
                            error=(
                                f"{type(primary_exception).__name__}: "
                                f"{primary_exception}"
                            ),
                        )
                    detached = cancel_pending_descendants(
                        Path(registry_value),
                        str(config.conversation_id),
                    )
                    if detached:
                        cleanup_error = RuntimeError(
                            "child agent exited with uncollected descendants; it "
                            "must await or cancel every spawned task first: "
                            f"{', '.join(detached)}"
                        )
                        record_delegated_task_result(
                            config.delegation_task_id,
                            error=f"RuntimeError: {cleanup_error}",
                        )
                except BaseException as error:  # noqa: BLE001
                    if cleanup_error is None:
                        cleanup_error = error
        clear_github_credentials()
        configure_delegation(None)
        if inference_heartbeat is not None:
            inference_heartbeat.close()
        if active_conversation is not None:
            active_conversation.close()
        if cleanup_error is not None and primary_exception is None:
            raise cleanup_error

    if config.child and config.delegation_task_id:
        record_delegated_task_result(
            config.delegation_task_id,
            result=child_result,
            error=(
                None
                if child_result is not None
                else f"child execution ended with status {status.value}"
            ),
        )

    print(
        "OPENHANDS_RESULT "
        + json.dumps(
            {
                "conversation_id": str(active_conversation.id),
                "status": status.value,
                **({"result": child_result} if config.child else {}),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    durable_turn_processed = (
        inbox is not None
        and active_inbox_turn_id is not None
        and inbox.latest_turn(active_inbox_turn_id).state is DeliveryState.PROCESSED
    )
    return 0 if (
        status == ConversationExecutionStatus.FINISHED or durable_turn_processed
    ) else 1
