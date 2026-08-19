"""CLI and environment configuration for the Senpai OpenHands runtime."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr
from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import flag

from senpai_agent.advisor import advisor_conversation_id
from senpai_agent.agent_markdown import read_agent_markdown
from senpai_agent.launch_context import LAUNCH_CONTEXT_ENV, decode_launch_context
from senpai_agent.openhands import REPOSITORY_ROOT
from senpai_agent.openhands.llm import (
    REASONING_EFFORTS,
    resolve_compaction_trigger_tokens,
    resolve_model_profiles,
)
from senpai_agent.program_context import PROGRAM_PATH_ENV, load_program_system_prompt
from senpai_agent.secrets import (
    conversation_secrets,
    github_repo,
    github_token,
    scrub_model_credentials,
)
from senpai_agent.system_instructions import SenpaiSystemInstructions


@dataclass(frozen=True)
class RunnerArgs:
    max_turns: int = field(alias="--max-turns")
    model: str | None = field(default=None, alias="--model")
    api_key_env: str | None = field(default=None, alias="--api-key-env")
    reasoning_effort: str | None = field(
        default=None,
        alias="--reasoning-effort",
        choices=REASONING_EFFORTS,
    )
    smart_model: str | None = field(default=None, alias="--smart-model")
    smart_reasoning_effort: str | None = field(
        default=None,
        alias="--smart-reasoning-effort",
        choices=REASONING_EFFORTS,
    )
    fast_model: str | None = field(default=None, alias="--fast-model")
    fast_reasoning_effort: str | None = field(
        default=None,
        alias="--fast-reasoning-effort",
        choices=REASONING_EFFORTS,
    )
    frontier_model: str | None = field(default=None, alias="--frontier-model")
    frontier_reasoning_effort: str | None = field(
        default=None,
        alias="--frontier-reasoning-effort",
        choices=REASONING_EFFORTS,
    )
    compaction_trigger_tokens: int | None = field(
        default=None,
        alias="--compaction-trigger-tokens",
    )
    workspace: str | None = field(default=None, alias="--workspace")
    state_dir: str | None = field(default=None, alias="--state-dir")
    conversation_id: str | None = field(default=None, alias="--conversation-id")
    role_file: str | None = field(default=None, alias="--role-file")
    harness_file: str | None = field(default=None, alias="--harness-file")
    plugin_dir: str | None = field(default=None, alias="--plugin-dir")
    agent: str | None = field(default=None, alias="--agent")
    enable_browser: bool = flag(
        default=True,
        alias="--browser",
        negative_option="--no-browser",
    )
    child: bool = flag(default=False, alias="--child")


@dataclass(frozen=True)
class RunnerConfig:
    max_turns: int
    model: str
    api_key_env: str
    api_key: SecretStr
    github_repo: str
    github_token: SecretStr | None
    github_trusted_actor: str | None
    conversation_secrets: Mapping[str, str]
    reasoning_effort: str
    smart_model: str
    smart_api_key_env: str
    smart_api_key: SecretStr
    smart_reasoning_effort: str
    fast_model: str
    fast_api_key_env: str
    fast_api_key: SecretStr
    fast_reasoning_effort: str
    frontier_model: str
    frontier_api_key_env: str
    frontier_api_key: SecretStr
    frontier_reasoning_effort: str
    compaction_trigger_tokens: int
    workspace: Path
    state_dir: Path
    conversation_id: uuid.UUID
    role: str
    enable_browser: bool
    agent_name: str | None
    harness_file: Path
    role_file: Path
    plugin_dir: Path
    instructions: SenpaiSystemInstructions
    advisor_branch: str | None = None
    student_names: tuple[str, ...] | None = None
    student_name: str | None = None
    wandb_entity: str | None = None
    wandb_project: str | None = None
    timeout_seconds: float = 7200
    llm_timeout_seconds: int = 5400
    llm_num_retries: int = 1
    inbox_max_stalled_attempts: int = 3
    inbox_max_recovery_generations: int = 1
    child: bool = False
    delegation_root_state_dir: Path | None = None
    delegation_tree_id: str | None = None
    delegation_depth: int = 0
    delegation_deadline_epoch: float | None = None
    delegation_task_id: str | None = None


def parse_runner_args(argv: Sequence[str] | None = None) -> RunnerArgs:
    parser = ArgumentParser(description="Run a Senpai OpenHands agent.")
    parser.add_arguments(RunnerArgs, dest="args")
    return parser.parse_args(argv).args


def env_value(
    parsed_value: str | None,
    env: Mapping[str, str],
    key: str,
    default: str | None = None,
) -> str | None:
    return parsed_value if parsed_value is not None else env.get(key, default)


def find_role_file(explicit: str | None) -> Path:
    if not explicit:
        raise RuntimeError(
            "OpenHands role instructions are required; set SENPAI_OPENHANDS_ROLE_FILE"
        )
    path = Path(explicit).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"OpenHands role file does not exist: {path}")
    return path


def find_harness_file(explicit: str | None = None) -> Path:
    path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else REPOSITORY_ROOT / "system_instructions" / "SENPAI-HARNESS.md"
    )
    if not path.is_file():
        raise RuntimeError(f"OpenHands harness file does not exist: {path}")
    return path


def read_instruction_file(path: Path) -> str:
    instructions = read_agent_markdown(path).strip()
    if not instructions:
        raise RuntimeError(f"OpenHands instruction file is empty: {path}")
    return instructions


def resolve_plugin_dir(explicit: str | None = None) -> Path:
    path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else REPOSITORY_ROOT / "plugins" / "senpai"
    )
    manifest = path / ".plugin" / "plugin.json"
    if not path.is_dir() or not manifest.is_file():
        raise RuntimeError(f"Senpai OpenHands plugin does not exist: {path}")
    return path


def _conversation_id(
    args: RunnerArgs,
    env: Mapping[str, str],
    role: str,
    state_dir: Path,
) -> uuid.UUID:
    explicit = env_value(
        args.conversation_id,
        env,
        "SENPAI_OPENHANDS_CONVERSATION_ID",
    )
    if role == "advisor":
        return advisor_conversation_id(state_dir, explicit)
    return uuid.UUID(explicit) if explicit else uuid.uuid4()


def resolve_config(
    args: RunnerArgs,
    env: Mapping[str, str] = os.environ,
) -> RunnerConfig:
    workspace_arg = (
        env_value(args.workspace, env, "SENPAI_OPENHANDS_WORKSPACE", os.getcwd())
        or os.getcwd()
    )
    workspace = Path(workspace_arg).expanduser().resolve()
    if not workspace.exists():
        raise RuntimeError(f"OpenHands workspace does not exist: {workspace}")

    state_dir_arg = env_value(args.state_dir, env, "SENPAI_OPENHANDS_STATE_DIR")
    if not state_dir_arg:
        raise RuntimeError(
            "OpenHands state directory is required; set SENPAI_OPENHANDS_STATE_DIR"
        )
    state_dir = Path(state_dir_arg).expanduser().resolve()
    if state_dir == workspace or state_dir.is_relative_to(workspace):
        raise RuntimeError(
            "OpenHands state directory must be outside the target workspace"
        )

    role = env.get("SENPAI_ROLE", "")
    if role not in {"advisor", "student"}:
        raise RuntimeError("SENPAI_ROLE must be advisor or student")
    harness_file = find_harness_file(
        env_value(args.harness_file, env, "SENPAI_OPENHANDS_HARNESS_FILE")
    )
    role_file = find_role_file(
        env_value(args.role_file, env, "SENPAI_OPENHANDS_ROLE_FILE")
    )
    instructions = SenpaiSystemInstructions(
        harness=read_instruction_file(harness_file),
        role=read_instruction_file(role_file),
        program=load_program_system_prompt(
            workspace,
            env.get(PROGRAM_PATH_ENV, ""),
        ),
        launch=decode_launch_context(env.get(LAUNCH_CONTEXT_ENV, "")),
    )

    try:
        timeout_seconds = float(env.get("SENPAI_OPENHANDS_TIMEOUT_SECONDS", "7200"))
    except ValueError as error:
        raise RuntimeError(
            "SENPAI_OPENHANDS_TIMEOUT_SECONDS must be numeric"
        ) from error
    if timeout_seconds <= 0:
        raise RuntimeError("SENPAI_OPENHANDS_TIMEOUT_SECONDS must be positive")
    try:
        llm_timeout_seconds = int(env.get("SENPAI_LLM_TIMEOUT_SECONDS", "5400"))
        llm_num_retries = int(env.get("SENPAI_LLM_NUM_RETRIES", "1"))
    except ValueError as error:
        raise RuntimeError("Senpai LLM timeout settings must be numeric") from error
    if llm_timeout_seconds <= 0 or llm_num_retries <= 0:
        raise RuntimeError("Senpai LLM timeout and attempts must be positive")

    compaction_trigger_tokens = resolve_compaction_trigger_tokens(
        args.compaction_trigger_tokens,
        env,
    )

    try:
        inbox_max_stalled_attempts = int(
            env.get(
                "SENPAI_INBOX_MAX_STALLED_ATTEMPTS",
                "3",
            )
        )
        inbox_max_recovery_generations = int(
            env.get(
                "SENPAI_INBOX_MAX_RECOVERY_GENERATIONS",
                "1",
            )
        )
    except ValueError as error:
        raise RuntimeError("inbox recovery budget must be numeric") from error
    if inbox_max_stalled_attempts <= 0 or inbox_max_recovery_generations < 0:
        raise RuntimeError(
            "inbox recovery budget requires a positive attempt limit and a "
            "non-negative recovery-generation limit"
        )

    delegation_root_value = env.get("SENPAI_DELEGATION_ROOT_STATE_DIR")
    delegation_root_state_dir = (
        Path(delegation_root_value).expanduser().resolve()
        if delegation_root_value
        else None
    )
    delegation_tree_id = env.get("SENPAI_DELEGATION_TREE_ID") or None
    try:
        delegation_depth = int(env.get("SENPAI_DELEGATION_DEPTH", "0"))
    except ValueError as error:
        raise RuntimeError("SENPAI_DELEGATION_DEPTH must be an integer") from error
    if not 0 <= delegation_depth <= 2:
        raise RuntimeError("SENPAI_DELEGATION_DEPTH must be between 0 and 2")
    deadline_value = env.get("SENPAI_DELEGATION_DEADLINE_EPOCH")
    try:
        delegation_deadline_epoch = float(deadline_value) if deadline_value else None
    except ValueError as error:
        raise RuntimeError(
            "SENPAI_DELEGATION_DEADLINE_EPOCH must be numeric"
        ) from error

    wandb_entity = env.get("WANDB_ENTITY", "").strip() or None
    wandb_project = env.get("WANDB_PROJECT", "").strip() or None
    profiles = resolve_model_profiles(
        env,
        child=args.child,
        model=args.model,
        api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort,
        smart_model=args.smart_model,
        smart_reasoning_effort=args.smart_reasoning_effort,
        fast_model=args.fast_model,
        fast_reasoning_effort=args.fast_reasoning_effort,
        frontier_model=args.frontier_model,
        frontier_reasoning_effort=args.frontier_reasoning_effort,
    )
    if profiles.uses_wandb and not (wandb_entity and wandb_project):
        raise RuntimeError(
            "WANDB_ENTITY and WANDB_PROJECT are required for W&B Inference"
        )
    resolved_conversation_secrets = conversation_secrets(
        env,
        model_api_key_env_names=(
            profiles.main.api_key_env,
            profiles.smart.api_key_env,
            profiles.fast.api_key_env,
            profiles.frontier.api_key_env,
        ),
    )

    return RunnerConfig(
        max_turns=args.max_turns,
        model=profiles.main.model,
        api_key_env=profiles.main.api_key_env,
        api_key=profiles.main.api_key,
        github_repo=github_repo(env),
        github_token=github_token(env, required=not args.child),
        github_trusted_actor=env.get("SENPAI_GITHUB_ACTOR"),
        conversation_secrets=resolved_conversation_secrets,
        reasoning_effort=profiles.main.reasoning_effort,
        smart_model=profiles.smart.model,
        smart_api_key_env=profiles.smart.api_key_env,
        smart_api_key=profiles.smart.api_key,
        smart_reasoning_effort=profiles.smart.reasoning_effort,
        fast_model=profiles.fast.model,
        fast_api_key_env=profiles.fast.api_key_env,
        fast_api_key=profiles.fast.api_key,
        fast_reasoning_effort=profiles.fast.reasoning_effort,
        frontier_model=profiles.frontier.model,
        frontier_api_key_env=profiles.frontier.api_key_env,
        frontier_api_key=profiles.frontier.api_key,
        frontier_reasoning_effort=profiles.frontier.reasoning_effort,
        compaction_trigger_tokens=compaction_trigger_tokens,
        workspace=workspace,
        state_dir=state_dir,
        conversation_id=_conversation_id(args, env, role, state_dir),
        role=role,
        enable_browser=args.enable_browser,
        agent_name=env_value(args.agent, env, "SENPAI_OPENHANDS_AGENT"),
        harness_file=harness_file,
        role_file=role_file,
        plugin_dir=resolve_plugin_dir(
            env_value(args.plugin_dir, env, "SENPAI_PLUGIN")
        ),
        instructions=instructions,
        advisor_branch=env.get("ADVISOR_BRANCH") or None,
        student_names=tuple(
            name.strip()
            for name in env.get("STUDENT_NAMES", "").split(",")
            if name.strip()
        ),
        student_name=env.get("STUDENT_NAME") or None,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        timeout_seconds=timeout_seconds,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_num_retries=llm_num_retries,
        inbox_max_stalled_attempts=inbox_max_stalled_attempts,
        inbox_max_recovery_generations=inbox_max_recovery_generations,
        child=args.child,
        delegation_root_state_dir=delegation_root_state_dir,
        delegation_tree_id=delegation_tree_id,
        delegation_depth=delegation_depth,
        delegation_deadline_epoch=delegation_deadline_epoch,
        delegation_task_id=env.get("SENPAI_DELEGATION_TASK_ID") or None,
    )


def local_event_db_path(config: RunnerConfig) -> Path:
    return config.state_dir / f"{config.role}-events.sqlite3"
