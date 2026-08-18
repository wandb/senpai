"""Stable CLI and import surface for the Senpai OpenHands runtime."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

# This import initializes Weave before any module imports the OpenHands SDK.
from senpai_agent.openhands import WEAVE_PROJECT
from senpai_agent.delegation import record_delegated_task_result
from senpai_agent.openhands.agents import (
    build_agent,
    build_main_agent_context,
    build_main_tools,
    delegation_config,
    depth_aware_child_definition,
    find_named_agent,
    is_exposed_skill,
    resolve_agent_skills,
    sanitized_agent_definitions,
    sanitized_project_skills,
    with_system_instructions,
    with_tool_concurrency,
    without_eager_skill_discovery,
)
from senpai_agent.openhands.child_results import (
    MAX_INLINE_CHILD_RESULT_TOKENS,
    compact_child_result,
    final_agent_result,
)
from senpai_agent.openhands.config import (
    RunnerArgs,
    RunnerConfig,
    env_value,
    find_harness_file,
    find_role_file,
    github_repo,
    local_event_db_path,
    parse_runner_args,
    read_instruction_file,
    resolve_config,
    resolve_plugin_dir,
    scrub_model_credentials,
)
from senpai_agent.openhands.conversation import (
    ConversationOutcome,
    arun_conversation,
    arun_steerable_conversation,
    conversation_prompt_cache_key,
    create_conversation,
    graceful_interrupts,
    reject_recovered_actions,
    run_conversation,
    run_steerable_conversation,
    run_turn,
)
from senpai_agent.openhands.llm import (
    apply_reasoning_profile,
    compaction_configuration,
    infer_api_key_env,
    model_provider,
    model_runtime_configuration,
    openai_responses_configuration,
    openhands_reasoning_effort,
    profile_api_key_env,
    prompt_cache_configuration,
    resolve_api_key,
)
from senpai_agent.openhands.runtime import (
    EVENT_TEXT_LIMIT,
    event_summary,
    print_event,
    run_openhands,
)
from senpai_agent.secrets import conversation_secrets, github_token
from senpai_agent.weave_monitoring import finish_weave_monitoring


def main(argv: Sequence[str] | None = None) -> int:
    try:
        try:
            args = parse_runner_args(argv)
            prompt = sys.stdin.read()
            if not prompt:
                raise RuntimeError("OpenHands runner requires a prompt on stdin")
            config = resolve_config(args)
            os.environ.pop(config.api_key_env, None)
            return run_openhands(prompt, config)
        except BaseException as error:  # noqa: BLE001
            if task_id := os.environ.get("SENPAI_DELEGATION_TASK_ID"):
                record_delegated_task_result(
                    task_id,
                    error=f"{type(error).__name__}: {error}",
                )
            raise
    finally:
        finish_weave_monitoring()


if __name__ == "__main__":
    raise SystemExit(main())
