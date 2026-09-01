"""Run one bounded Senpai OpenHands turn for the Python controller."""

# ruff: noqa: E402
# OpenHands imports intentionally follow Weave initialization below.

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from senpai_agent.advisor import (
    AdvisorEventPump,
    advisor_conversation_id,
)
from senpai_agent.delegation import (
    MAX_DELEGATION_DEPTH,
    MAX_PARALLEL_AGENTS,
    DelegationConfig,
    cancel_pending_descendants,
    configure_delegation,
    record_delegated_task_result,
)
from senpai_agent.inbox import (
    DeliveryState,
    InboxTurn,
    PersistentInbox,
    deliver_turn_messages,
    events_after_turn_delivery,
    turn_has_finished_response,
)
from senpai_agent.secrets import (
    BUILTIN_CONVERSATION_SECRET_ENV_NAMES,
    GITHUB_TOKEN_ENV_NAMES,
    GITHUB_TOKEN_FD_ENV,
    GITHUB_TOKEN_FILE_ENV,
    configured_custom_secret_env_names,
    scrub_github_credentials,
)
from senpai_agent.weave_monitoring import (
    finish_weave_monitoring,
    initialize_weave_monitoring,
    register_trace_secret,
    weave_conversation_url,
)

WEAVE_PROJECT = initialize_weave_monitoring()

from openhands.sdk import LLM, Agent, AgentContext, LocalConversation, Tool
from openhands.sdk.agent.parallel_executor import ParallelToolExecutor
from openhands.sdk.conversation import ConversationExecutionStatus, ConversationState
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.plugin import PluginSource
from openhands.sdk.skills import (
    Skill,
    load_skills_from_dir,
    merge_skills_by_name,
)
from openhands.sdk.subagent import (
    AgentDefinition,
    agent_definition_to_factory,
    discover_agents,
)
from openhands.tools.preset.default import (
    get_default_condenser,
    get_default_tools,
    register_default_tools,
)
from pydantic import SecretStr
from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import flag

from senpai_agent.agent_markdown import read_agent_markdown, strip_spdx_header
from senpai_agent.github.tools import (
    clear_github_credentials,
    configure_github_credentials,
)
from senpai_agent.inference_heartbeat import InferenceHeartbeat
from senpai_agent.launch_context import LAUNCH_CONTEXT_ENV, decode_launch_context
from senpai_agent.local_events import LocalEventStore
from senpai_agent.openhands_security import disable_ambient_plugin_discovery
from senpai_agent.program_context import (
    PROGRAM_PATH_ENV,
    load_program_system_prompt,
)
from senpai_agent.PROMPTS import (
    DELEGATED_RESULT_SUMMARY_PROMPT,
    RECOVERED_ACTION_PROMPT,
    render_prompt,
)
from senpai_agent.system_instructions import SenpaiSystemInstructions
from senpai_agent.tools import register_senpai_tools

DEFAULT_MODEL = "openai/gpt-5.6-sol"
DEFAULT_FAST_MODEL = "openai/gpt-5.6-luna"
DEFAULT_FRONTIER_MODEL = "openai/gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_FAST_REASONING_EFFORT = "high"
DEFAULT_FRONTIER_REASONING_EFFORT = "max"
DEFAULT_COMPACTION_TRIGGER_TOKENS = 200_000
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
REASONING_EFFORTS = (
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "none",
)
SENPAI_AGENT_NAMES = ("bash-runner", "general-purpose", "explore", "search")
SENPAI_AGENT_DIR = Path(__file__).resolve().parents[1] / ".agents" / "agents"
REPOSITORY_INSTRUCTION_FILENAMES = frozenset(
    {"agents.md", "agent.md", "claude.md"}
)
PROVIDER_API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "wandb": "WANDB_API_KEY",
}
EVENT_TEXT_LIMIT = 20000
MAX_INLINE_CHILD_RESULT_TOKENS = 15_000
DEFAULT_INBOX_MAX_STALLED_ATTEMPTS = 3
DEFAULT_INBOX_MAX_RECOVERY_GENERATIONS = 1


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
    llm_num_retries: int = 5
    inbox_max_stalled_attempts: int = DEFAULT_INBOX_MAX_STALLED_ATTEMPTS
    inbox_max_recovery_generations: int = DEFAULT_INBOX_MAX_RECOVERY_GENERATIONS
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


def _uses_openai_pro_mode(model: str, reasoning_effort: str | None) -> bool:
    return (
        reasoning_effort is not None
        and model.split("/", 1)[0].lower() == "openai"
        and openhands_reasoning_effort(reasoning_effort, model) == "max"
    )


def _openai_pro_reasoning(
    model: str,
    reasoning_effort: str | None,
) -> dict[str, str] | None:
    if not _uses_openai_pro_mode(model, reasoning_effort):
        return None
    return {
        "effort": "max",
        "mode": "pro",
        "summary": "auto",
        "context": "all_turns",
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


def env_value(
    parsed_value: str | None,
    env: Mapping[str, str],
    key: str,
    default: str | None = None,
) -> str | None:
    return parsed_value if parsed_value is not None else env.get(key, default)


def resolve_api_key(env: Mapping[str, str], key_env: str) -> SecretStr:
    value = env.get(key_env)
    if not value:
        raise RuntimeError(f"{key_env} is required for the OpenHands runtime")
    return SecretStr(value)


def conversation_secrets(
    env: Mapping[str, str],
    *,
    model_api_key_env_names: Sequence[str],
) -> dict[str, str]:
    custom_secret_env_names = configured_custom_secret_env_names(env)
    model_credentials = set(model_api_key_env_names)
    overlap = tuple(
        name for name in custom_secret_env_names if name in model_credentials
    )
    if overlap:
        raise RuntimeError(
            "model credential environment variables cannot also be custom "
            f"secrets: {', '.join(overlap)}"
        )

    custom_secrets = {}
    for name in custom_secret_env_names:
        value = env.get(name)
        if value is None or not value.strip():
            raise RuntimeError(f"configured custom secret {name} is required")
        custom_secrets[name] = value

    return {
        name: value
        for name in BUILTIN_CONVERSATION_SECRET_ENV_NAMES
        if (value := env.get(name))
    } | custom_secrets


def github_token(
    env: Mapping[str, str],
    *,
    required: bool = True,
) -> SecretStr | None:
    token_fd = env.get(GITHUB_TOKEN_FD_ENV)
    if token_fd:
        try:
            descriptor = int(token_fd)
        except ValueError as error:
            raise RuntimeError(f"{GITHUB_TOKEN_FD_ENV} must be an integer") from error
        with os.fdopen(descriptor, encoding="utf-8") as token_stream:
            value = token_stream.read().strip()
        if not value:
            raise RuntimeError(f"{GITHUB_TOKEN_FD_ENV} is empty")
        return SecretStr(value)

    token_file = env.get(GITHUB_TOKEN_FILE_ENV)
    if token_file:
        path = Path(token_file)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise RuntimeError(
                f"{GITHUB_TOKEN_FILE_ENV} must be a private regular file"
            )
        try:
            value = path.read_text(encoding="utf-8")
        finally:
            path.unlink(missing_ok=True)
        value = value.strip()
        if not value:
            raise RuntimeError(f"{GITHUB_TOKEN_FILE_ENV} is empty")
        return SecretStr(value)

    value = next(
        (
            candidate
            for name in GITHUB_TOKEN_ENV_NAMES
            if (candidate := env.get(name, "").strip())
        ),
        None,
    )
    if value is None:
        if required:
            raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required")
        return None
    return SecretStr(value)


def github_repo(env: Mapping[str, str]) -> str:
    value = env.get("GH_REPO", "")
    if len(value.split("/")) != 2 or not all(value.split("/")):
        raise RuntimeError("GH_REPO must use owner/name form")
    return value


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
        else Path(__file__).resolve().parents[1]
        / "system_instructions"
        / "SENPAI-HARNESS.md"
    )
    if not path.is_file():
        raise RuntimeError(f"OpenHands harness file does not exist: {path}")
    return path


def read_instruction_file(path: Path) -> str:
    instructions = read_agent_markdown(path).strip()
    if not instructions:
        raise RuntimeError(f"OpenHands instruction file is empty: {path}")
    return instructions


def is_exposed_skill(skill: Skill, root: Path | None = None) -> bool:
    if not skill.source:
        return True
    source = Path(skill.source)
    source_path = source if source.is_absolute() else (root or Path()) / source
    resolved = source_path.resolve()
    return (
        source.name.casefold() not in REPOSITORY_INSTRUCTION_FILENAMES
        and resolved.name.casefold() not in REPOSITORY_INSTRUCTION_FILENAMES
        and (root is None or resolved.is_relative_to(root.resolve()))
        and not (source_path.parent / ".senpai-developer-only").is_file()
    )


def sanitized_project_skills(workspace: Path) -> list[Skill]:
    """Load explicit project skills without repository instruction files."""

    skills: list[Skill] = []
    for relative_path in (
        ".agents/skills",
        ".openhands/skills",
        ".openhands/microagents",
    ):
        groups = load_skills_from_dir(workspace / relative_path)
        candidates = (
            skill
            for group in groups
            for skill in group.values()
            if is_exposed_skill(skill, workspace)
        )
        skills = merge_skills_by_name(skills, candidates)

    return [
        skill.model_copy(update={"content": strip_spdx_header(skill.content)})
        for skill in skills
    ]


def sanitized_agent_definitions(workspace: Path) -> list[AgentDefinition]:
    """Load Senpai agents first, then unshadowed target and user agents."""

    reserved = {
        name: AgentDefinition.load(SENPAI_AGENT_DIR / f"{name}.md")
        for name in SENPAI_AGENT_NAMES
    }
    return [
        definition.model_copy(
            update={"system_prompt": strip_spdx_header(definition.system_prompt)}
        )
        for definition in (
            *reserved.values(),
            *(
                candidate
                for candidate in discover_agents(workspace)
                if candidate.name not in reserved
            ),
        )
    ]


def resolve_agent_skills(
    definition: AgentDefinition,
    project_skills: Sequence[Skill],
) -> list[Skill]:
    """Resolve declared skills without invoking SDK project-instruction loading."""

    if not definition.skills:
        return []
    available = {skill.name: skill for skill in project_skills}
    missing = [name for name in definition.skills if name not in available]
    if missing:
        raise ValueError(
            f"Skills {', '.join(missing)} were not found for agent "
            f"{definition.name}."
        )
    return [available[name] for name in definition.skills]


def without_eager_skill_discovery(
    definition: AgentDefinition,
) -> AgentDefinition:
    """Keep skill resolution on Senpai's explicit, filtered path."""

    return definition.model_copy(update={"skills": []})


def depth_aware_child_definition(
    definition: AgentDefinition,
    *,
    child: bool,
    depth: int,
) -> AgentDefinition:
    """Expose recursive spawn only to the one child role that can use it."""

    if not child or (
        definition.name == "general-purpose"
        and 0 < depth < MAX_DELEGATION_DEPTH
    ):
        return definition
    return definition.model_copy(
        update={"tools": [tool for tool in definition.tools if tool != "spawn_agents"]}
    )


def resolve_plugin_dir(explicit: str | None = None) -> Path:
    path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else Path(__file__).resolve().parents[1] / "plugins" / "senpai"
    )
    manifest = path / ".plugin" / "plugin.json"
    if not path.is_dir() or not manifest.is_file():
        raise RuntimeError(f"Senpai OpenHands plugin does not exist: {path}")
    return path


def fresh_conversation_id() -> uuid.UUID:
    return uuid.uuid4()


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
    program = load_program_system_prompt(
        workspace,
        env.get(PROGRAM_PATH_ENV, ""),
    )
    role = env.get("SENPAI_ROLE", "")
    if role not in {"advisor", "student"}:
        raise RuntimeError("SENPAI_ROLE must be advisor or student")
    harness_file = find_harness_file(
        env_value(
            args.harness_file,
            env,
            "SENPAI_OPENHANDS_HARNESS_FILE",
        )
    )
    role_file = find_role_file(
        env_value(args.role_file, env, "SENPAI_OPENHANDS_ROLE_FILE"),
    )
    instructions = SenpaiSystemInstructions(
        harness=read_instruction_file(harness_file),
        role=read_instruction_file(role_file),
        program=program,
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
        llm_num_retries = int(env.get("SENPAI_LLM_NUM_RETRIES", "5"))
    except ValueError as error:
        raise RuntimeError("Senpai LLM timeout settings must be numeric") from error
    if llm_timeout_seconds <= 0 or llm_num_retries <= 0:
        raise RuntimeError("Senpai LLM timeout and attempts must be positive")
    try:
        compaction_trigger_tokens = int(
            args.compaction_trigger_tokens
            if args.compaction_trigger_tokens is not None
            else env.get(
                "SENPAI_COMPACTION_TRIGGER_TOKENS",
                str(DEFAULT_COMPACTION_TRIGGER_TOKENS),
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "SENPAI_COMPACTION_TRIGGER_TOKENS must be an integer"
        ) from error
    if compaction_trigger_tokens < 50_000:
        raise RuntimeError(
            "SENPAI_COMPACTION_TRIGGER_TOKENS must be at least 50000"
        )
    try:
        inbox_max_stalled_attempts = int(
            env.get(
                "SENPAI_INBOX_MAX_STALLED_ATTEMPTS",
                str(DEFAULT_INBOX_MAX_STALLED_ATTEMPTS),
            )
        )
        inbox_max_recovery_generations = int(
            env.get(
                "SENPAI_INBOX_MAX_RECOVERY_GENERATIONS",
                str(DEFAULT_INBOX_MAX_RECOVERY_GENERATIONS),
            )
        )
    except ValueError as error:
        raise RuntimeError("inbox recovery budget must be numeric") from error
    if inbox_max_stalled_attempts <= 0 or inbox_max_recovery_generations < 0:
        raise RuntimeError(
            "inbox recovery budget requires a positive attempt limit and a "
            "non-negative recovery-generation limit"
        )
    wandb_entity = env.get("WANDB_ENTITY", "").strip() or None
    wandb_project = env.get("WANDB_PROJECT", "").strip() or None
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
        delegation_deadline_epoch = (
            float(deadline_value) if deadline_value else None
        )
    except ValueError as error:
        raise RuntimeError(
            "SENPAI_DELEGATION_DEADLINE_EPOCH must be numeric"
        ) from error

    model = env_value(args.model, env, "SENPAI_OPENHANDS_MODEL", DEFAULT_MODEL)
    if not model:
        model = DEFAULT_MODEL
    reasoning_effort = (
        env_value(
            args.reasoning_effort,
            env,
            "SENPAI_OPENHANDS_REASONING_EFFORT",
            DEFAULT_REASONING_EFFORT,
        )
        or DEFAULT_REASONING_EFFORT
    )
    reasoning_effort = openhands_reasoning_effort(reasoning_effort, model)
    api_key_env = (
        env_value(args.api_key_env, env, "SENPAI_OPENHANDS_API_KEY_ENV")
        or infer_api_key_env(
            model,
            override_env="SENPAI_OPENHANDS_API_KEY_ENV",
        )
    )

    smart_default_model = (
        env.get("SENPAI_OPENHANDS_MODEL") if args.child else model
    ) or DEFAULT_MODEL
    smart_model = (
        env_value(
            args.smart_model,
            env,
            "SENPAI_OPENHANDS_SMART_MODEL",
            smart_default_model,
        )
        or smart_default_model
    )
    smart_default_effort = (
        env.get("SENPAI_OPENHANDS_REASONING_EFFORT") if args.child else reasoning_effort
    ) or DEFAULT_REASONING_EFFORT
    smart_reasoning_effort = (
        env_value(
            args.smart_reasoning_effort,
            env,
            "SENPAI_OPENHANDS_SMART_REASONING_EFFORT",
            smart_default_effort,
        )
        or smart_default_effort
    )
    smart_reasoning_effort = openhands_reasoning_effort(
        smart_reasoning_effort,
        smart_model,
    )

    fast_default_model = (
        DEFAULT_FAST_MODEL
        if model_provider(smart_model) == model_provider(DEFAULT_FAST_MODEL)
        else smart_model
    )
    fast_model = (
        env_value(
            args.fast_model,
            env,
            "SENPAI_OPENHANDS_FAST_MODEL",
            fast_default_model,
        )
        or fast_default_model
    )
    fast_reasoning_effort = (
        env_value(
            args.fast_reasoning_effort,
            env,
            "SENPAI_OPENHANDS_FAST_REASONING_EFFORT",
            DEFAULT_FAST_REASONING_EFFORT,
        )
        or DEFAULT_FAST_REASONING_EFFORT
    )
    fast_reasoning_effort = openhands_reasoning_effort(
        fast_reasoning_effort,
        fast_model,
    )

    frontier_model = (
        env_value(
            args.frontier_model,
            env,
            "SENPAI_OPENHANDS_FRONTIER_MODEL",
            DEFAULT_FRONTIER_MODEL,
        )
        or DEFAULT_FRONTIER_MODEL
    )
    frontier_reasoning_effort = (
        env_value(
            args.frontier_reasoning_effort,
            env,
            "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT",
            DEFAULT_FRONTIER_REASONING_EFFORT,
        )
        or DEFAULT_FRONTIER_REASONING_EFFORT
    )
    frontier_reasoning_effort = openhands_reasoning_effort(
        frontier_reasoning_effort,
        frontier_model,
    )

    smart_api_key_env = profile_api_key_env(
        smart_model,
        env,
        "SENPAI_OPENHANDS_SMART_API_KEY_ENV",
        inherited_model=model,
        inherited_api_key_env=api_key_env,
    )
    fast_api_key_env = profile_api_key_env(
        fast_model,
        env,
        "SENPAI_OPENHANDS_FAST_API_KEY_ENV",
        inherited_model=smart_model,
        inherited_api_key_env=smart_api_key_env,
    )
    frontier_api_key_env = profile_api_key_env(
        frontier_model,
        env,
        "SENPAI_OPENHANDS_FRONTIER_API_KEY_ENV",
        inherited_model=model,
        inherited_api_key_env=api_key_env,
    )
    api_key = resolve_api_key(env, api_key_env)
    smart_api_key = resolve_api_key(env, smart_api_key_env)
    fast_api_key = resolve_api_key(env, fast_api_key_env)
    frontier_api_key = resolve_api_key(env, frontier_api_key_env)
    resolved_conversation_secrets = conversation_secrets(
        env,
        model_api_key_env_names=(
            api_key_env,
            smart_api_key_env,
            fast_api_key_env,
            frontier_api_key_env,
        ),
    )
    models = (model, smart_model, fast_model, frontier_model)
    if any(model_provider(profile) == "wandb" for profile in models) and not (
        wandb_entity and wandb_project
    ):
        raise RuntimeError(
            "WANDB_ENTITY and WANDB_PROJECT are required for W&B Inference"
        )

    return RunnerConfig(
        max_turns=args.max_turns,
        model=model,
        api_key_env=api_key_env,
        api_key=api_key,
        github_repo=github_repo(env),
        github_token=github_token(env, required=not args.child),
        github_trusted_actor=env.get("SENPAI_GITHUB_ACTOR"),
        conversation_secrets=resolved_conversation_secrets,
        reasoning_effort=reasoning_effort,
        smart_model=smart_model,
        smart_api_key_env=smart_api_key_env,
        smart_api_key=smart_api_key,
        smart_reasoning_effort=smart_reasoning_effort,
        fast_model=fast_model,
        fast_api_key_env=fast_api_key_env,
        fast_api_key=fast_api_key,
        fast_reasoning_effort=fast_reasoning_effort,
        frontier_model=frontier_model,
        frontier_api_key_env=frontier_api_key_env,
        frontier_api_key=frontier_api_key,
        frontier_reasoning_effort=frontier_reasoning_effort,
        compaction_trigger_tokens=compaction_trigger_tokens,
        workspace=workspace,
        state_dir=state_dir,
        conversation_id=(
            advisor_conversation_id(
                state_dir,
                env_value(
                    args.conversation_id,
                    env,
                    "SENPAI_OPENHANDS_CONVERSATION_ID",
                ),
            )
            if role == "advisor"
            else (
                uuid.UUID(explicit_id)
                if (
                    explicit_id := env_value(
                        args.conversation_id,
                        env,
                        "SENPAI_OPENHANDS_CONVERSATION_ID",
                    )
                )
                else fresh_conversation_id()
            )
        ),
        role=role,
        enable_browser=args.enable_browser,
        agent_name=env_value(args.agent, env, "SENPAI_OPENHANDS_AGENT"),
        harness_file=harness_file,
        role_file=role_file,
        plugin_dir=resolve_plugin_dir(
            env_value(args.plugin_dir, env, "SENPAI_PLUGIN"),
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


def find_named_agent(
    name: str,
    definitions: Sequence[AgentDefinition],
) -> AgentDefinition:
    for definition in definitions:
        if definition.name == name:
            return definition
    raise RuntimeError(f"OpenHands agent not found: {name}")


def with_system_instructions(
    agent: Agent,
    instructions: SenpaiSystemInstructions,
    project_skills: Sequence[Skill] = (),
) -> Agent:
    context = agent.agent_context or AgentContext()
    skills = {skill.name: skill for skill in context.skills}
    skills.update({skill.name: skill for skill in project_skills})
    system_suffix = (
        f"{context.system_message_suffix}\n\n{instructions.prompt}"
        if context.system_message_suffix
        else instructions.prompt
    )
    return agent.model_copy(
        update={
            "agent_context": context.model_copy(
                update={
                    "system_message_suffix": system_suffix,
                    "current_datetime": None,
                    "skills": list(skills.values()),
                    "load_user_skills": False,
                    "load_project_skills": False,
                }
            )
        }
    )


def build_main_agent_context(
    instructions: SenpaiSystemInstructions,
    project_skills: Sequence[Skill] = (),
) -> AgentContext:
    return AgentContext(
        skills=list(project_skills),
        system_message_suffix=instructions.prompt,
        current_datetime=None,
        load_public_skills=False,
        load_user_skills=False,
        load_project_skills=False,
    )


def with_tool_concurrency(agent: Agent, limit: int) -> Agent:
    """Keep the serialized field and runtime executor on the same limit."""

    updated = agent.model_copy(update={"tool_concurrency_limit": limit})
    updated._parallel_executor = ParallelToolExecutor(max_workers=limit)
    return updated


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


def conversation_prompt_cache_key(config: RunnerConfig) -> str | None:
    if config.model.split("/", 1)[0].lower() != "openai":
        return None
    agent_kind = config.agent_name or ("child" if config.child else "main")
    return f"senpai:{config.role}:{agent_kind}"


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
        configuration["litellm_extra_body"] = {
            "reasoning": reasoning,
        }
    return configuration


def compaction_configuration(
    model: str,
    trigger_tokens: int,
) -> dict[str, int | None]:
    """Translate the universal token trigger to the provider SDK field."""
    provider = model_provider(model)
    if provider == "openai":
        return {"responses_compact_threshold": trigger_tokens}
    if provider == "anthropic":
        return {
            "anthropic_compact_threshold": trigger_tokens,
            "anthropic_compaction_instructions": None,
        }
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
            "extra_headers": {
                "OpenAI-Project": f"{wandb_entity}/{wandb_project}"
            },
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

    configuration: dict[str, object] = {}
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


def local_event_db_path(config: RunnerConfig) -> Path:
    return config.state_dir / f"{config.role}-events.sqlite3"


def scrub_model_credentials(
    environment: MutableMapping[str, str],
    config: RunnerConfig,
) -> None:
    for key_env in {
        config.api_key_env,
        config.smart_api_key_env,
        config.fast_api_key_env,
        config.frontier_api_key_env,
    }:
        environment.pop(key_env, None)


def build_main_tools(config: RunnerConfig) -> list[Tool]:
    """Keep native reasoning tools while replacing unsafe control boundaries."""

    if config.github_token is None:
        raise RuntimeError("main agents require GitHub credentials")
    register_senpai_tools()
    tools = [
        tool
        for tool in get_default_tools(
            enable_browser=False,
            enable_sub_agents=False,
        )
        if tool.name != "terminal"
    ]
    tools.extend(
        (
            Tool(name="senpai_terminal", params={"role": config.role}),
            Tool(
                name="senpai_github",
                params={
                    "role": config.role,
                    "state_dir": str(config.state_dir / "github"),
                    "advisor_branch": config.advisor_branch,
                    "student_names": config.student_names,
                    "student_name": config.student_name,
                },
            ),
        )
    )
    if config.enable_browser:
        # Keep the persisted spec name stable while its resolver exposes only
        # the lightweight load_browser definition until the model opts in.
        tools.append(Tool(name="browser_tool_set"))
    delegation_params = {"event_db_path": str(local_event_db_path(config))}
    if not config.child:
        tools.append(Tool(name="delegate_agent", params=delegation_params))
    tools.extend(
        Tool(name=name, params=delegation_params)
        for name in (
            "spawn_agents",
            "await_agents",
            "agent_status",
            "cancel_agents",
        )
    )
    if config.role == "student" and not config.child:
        training_params = {"state_dir": str(config.state_dir / "training")}
        tools.append(Tool(name="senpai_training", params=training_params))
    return tools


def delegation_config(
    config: RunnerConfig,
    *,
    deadline_epoch: float | None = None,
) -> DelegationConfig:
    return DelegationConfig(
        python_executable=Path(sys.executable),
        workspace=config.workspace,
        state_dir=config.state_dir,
        smart_model=config.smart_model,
        smart_reasoning_effort=config.smart_reasoning_effort,
        smart_api_key_env=config.smart_api_key_env,
        smart_api_key=config.smart_api_key.get_secret_value(),
        fast_model=config.fast_model,
        fast_reasoning_effort=config.fast_reasoning_effort,
        fast_api_key_env=config.fast_api_key_env,
        fast_api_key=config.fast_api_key.get_secret_value(),
        frontier_model=config.frontier_model,
        frontier_reasoning_effort=config.frontier_reasoning_effort,
        frontier_api_key_env=config.frontier_api_key_env,
        frontier_api_key=config.frontier_api_key.get_secret_value(),
        compaction_trigger_tokens=config.compaction_trigger_tokens,
        github_repo=config.github_repo,
        github_trusted_actor=config.github_trusted_actor,
        role_file=config.role_file,
        harness_file=config.harness_file,
        plugin_dir=config.plugin_dir,
        enable_browser=config.enable_browser,
        conversation_secrets=config.conversation_secrets,
        role=config.role,
        program_path=config.instructions.program.program_path,
        launch_context=config.instructions.launch,
        root_state_dir=config.delegation_root_state_dir,
        tree_id=config.delegation_tree_id,
        depth=config.delegation_depth,
        deadline_epoch=deadline_epoch or config.delegation_deadline_epoch,
        agent_name=config.agent_name,
        current_task_id=config.delegation_task_id,
    )


@contextmanager
def graceful_interrupts(conversation: object) -> Iterator[Callable[[], bool]]:
    interrupted_by: list[int] = []

    def interrupt(signum: int, _frame: object) -> None:
        print(f"OPENHANDS_INTERRUPT signal={signum}", file=sys.stderr, flush=True)
        if not interrupted_by:
            interrupted_by.append(signum)
        conversation.interrupt()

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        yield lambda: bool(interrupted_by)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if interrupted_by:
        raise SystemExit(128 + interrupted_by[0])


async def arun_conversation(
    conversation: object,
    timeout_seconds: float,
    activity: Callable[[], float] | None = None,
    *,
    started: Callable[[], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Run until completion or one full timeout passes without activity."""

    task = asyncio.create_task(conversation.arun())

    async def cancel_run() -> None:
        conversation.interrupt()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    await asyncio.sleep(0)
    if started is not None:
        started()
    if stop_requested is not None and stop_requested():
        await cancel_run()
        return
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=max(0, deadline - time.monotonic()),
            )
            return
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
            return
        except TimeoutError:
            if task.done():
                await task
                return
            renewed = (
                activity() + timeout_seconds if activity is not None else deadline
            )
            if renewed > time.monotonic():
                deadline = renewed
                continue
            print(
                f"OPENHANDS_TIMEOUT seconds={timeout_seconds:g}",
                file=sys.stderr,
                flush=True,
            )
            await cancel_run()
            return


def run_conversation(
    conversation: object,
    timeout_seconds: float,
) -> None:
    if timeout_seconds <= 0:
        print(
            f"OPENHANDS_TIMEOUT seconds={max(timeout_seconds, 0):g}",
            file=sys.stderr,
            flush=True,
        )
        conversation.interrupt()
        return
    asyncio.run(arun_conversation(conversation, timeout_seconds))


async def arun_steerable_conversation(
    conversation: object,
    pump: AdvisorEventPump,
    timeout_seconds: float,
    activity: Callable[[], float] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    while stop_requested is None or not stop_requested():
        if not pump.prepare_run():
            continue
        if stop_requested is not None and stop_requested():
            return
        await arun_conversation(
            conversation,
            timeout_seconds,
            activity,
            started=pump.run_started,
            stop_requested=stop_requested,
        )
        if not pump.finish_run():
            return


def run_steerable_conversation(
    conversation: object,
    pump: AdvisorEventPump,
    timeout_seconds: float,
    activity: Callable[[], float] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    asyncio.run(
        arun_steerable_conversation(
            conversation,
            pump,
            timeout_seconds,
            activity,
            stop_requested,
        )
    )


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


def reject_recovered_actions(conversation: object) -> int:
    """Pair crash-orphaned actions so resuming cannot replay them implicitly."""

    state = getattr(conversation, "state", None)
    active_branch = getattr(state, "active_branch", None)
    if active_branch is None:
        return 0
    pending = ConversationState.get_unmatched_actions(active_branch())
    if not pending:
        return 0
    conversation.reject_pending_actions(reason=RECOVERED_ACTION_PROMPT)
    print(
        f"OPENHANDS_RECOVERED_ACTIONS rejected={len(pending)}",
        file=sys.stderr,
        flush=True,
    )
    return len(pending)


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
        if (
            conversation.state.execution_status
            != ConversationExecutionStatus.FINISHED
        ):
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


def _activate_inbox_turn(
    conversation: object,
    inbox: PersistentInbox,
    turn_id: str,
) -> InboxTurn:
    """Reset a recovery branch when needed, then deliver its canonical messages."""

    turn = inbox.turn(turn_id)
    if turn.context_reset_required:
        active_branch = tuple(conversation.state.active_branch())
        active_senders = {getattr(event, "sender", None) for event in active_branch}
        recovery_started = any(
            message.sender in active_senders for message in turn.messages
        )
        if active_branch and not recovery_started:
            preserved_events = len(conversation.state.events)
            conversation.navigate_to(None)
            print(
                "OPENHANDS_CONTEXT_RESET "
                f"conversation_id={turn.conversation_id} "
                f"preserved_events={preserved_events}",
                file=sys.stderr,
                flush=True,
            )
        inbox.record_context_reset(turn_id)
    return deliver_turn_messages(conversation, inbox, turn_id)


def _latest_completed_tool_event_id(
    conversation: object,
    turn: InboxTurn,
) -> str | None:
    for event in reversed(events_after_turn_delivery(conversation, turn)):
        if type(event) is ObservationEvent:
            event_id = getattr(event, "id", None)
            if isinstance(event_id, str) and event_id:
                return event_id
    return None


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
    disable_ambient_plugin_discovery()
    register_default_tools(enable_browser=False)
    register_senpai_tools()
    file_agents = sanitized_agent_definitions(config.workspace)
    available_agents = [definition.name for definition in file_agents]
    project_skills = sanitized_project_skills(config.workspace)
    os.environ["SENPAI_CONVERSATION_ID"] = config.conversation_id.hex

    print(
        "OPENHANDS_RUN "
        + json.dumps(
            {
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
                    prompt_cache_configuration(config.model)
                    or {"provider_default": True}
                ),
                "reasoning_effort": config.reasoning_effort,
                "openhands_reasoning_effort": openhands_reasoning_effort(
                    config.reasoning_effort, config.model
                ),
                "reasoning_mode": (
                    "pro"
                    if _uses_openai_pro_mode(
                        config.model,
                        config.reasoning_effort,
                    )
                    else "standard"
                ),
                "agent": config.agent_name,
                "enable_browser": config.enable_browser,
                "role_file": str(config.role_file) if config.role_file else None,
                "plugin_dir": str(config.plugin_dir),
                "available_agents": available_agents,
                "weave_project": WEAVE_PROJECT,
                "weave_url": weave_conversation_url(
                    WEAVE_PROJECT,
                    config.conversation_id,
                ),
                "child": config.child,
                "reset_context": reset_context,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if config.github_token is not None:
        # The one-shot handoff is read after trace initialization.
        register_trace_secret(config.github_token.get_secret_value())
        configure_github_credentials(
            config.github_repo,
            config.github_token,
            trusted_actor=config.github_trusted_actor,
        )
    configure_delegation(delegation_config(config, deadline_epoch=run_deadline))
    scrub_github_credentials(os.environ)
    conversation = None
    inference_heartbeat = None
    cleanup_error: BaseException | None = None
    active_inbox_turn_id = inbox_turn_id
    try:
        retried_provider_errors: ContextVar[tuple[BaseException, ...]] = ContextVar(
            "retried_provider_errors",
            default=(),
        )

        def record_retry(
            _attempt: int,
            _total: int,
            error: BaseException | None,
        ) -> None:
            if error is not None:
                retried_provider_errors.set((*retried_provider_errors.get(), error))

        inference_heartbeat = (
            InferenceHeartbeat(on_inference_state)
            if on_inference_state is not None
            else None
        )

        @contextmanager
        def model_request() -> Iterator[None]:
            token = retried_provider_errors.set(())
            try:
                with (
                    inference_heartbeat.request()
                    if inference_heartbeat is not None
                    else nullcontext()
                ):
                    yield
            except Exception as error:
                retried = (
                    *getattr(error, "_senpai_retried_provider_errors", ()),
                    *retried_provider_errors.get(),
                )
                if retried:
                    setattr(error, "_senpai_retried_provider_errors", retried)
                raise
            finally:
                retried_provider_errors.reset(token)

        llm = LLM(
            model=config.model,
            api_key=config.api_key,
            timeout=config.llm_timeout_seconds,
            num_retries=config.llm_num_retries,
            reasoning_effort=openhands_reasoning_effort(
                config.reasoning_effort, config.model
            ),
            usage_id="senpai",
            retry_listener=record_retry,
            **model_runtime_configuration(
                config.model,
                config.reasoning_effort,
                compaction_trigger_tokens=config.compaction_trigger_tokens,
                wandb_entity=config.wandb_entity,
                wandb_project=config.wandb_project,
            ),
        )
        llm.set_request_scope(model_request)
        if config.agent_name:
            definition = depth_aware_child_definition(
                find_named_agent(config.agent_name, file_agents),
                child=config.child,
                depth=config.delegation_depth,
            )
            resolved_skills = resolve_agent_skills(definition, project_skills)
            agent = agent_definition_to_factory(
                without_eager_skill_discovery(definition),
            )(llm)
            agent = agent.model_copy(
                update={
                    "llm": apply_reasoning_profile(agent.llm),
                    "agent_context": (
                        agent.agent_context or AgentContext()
                    ).model_copy(update={"skills": resolved_skills}),
                }
            )
            agent = with_system_instructions(
                agent,
                config.instructions,
                project_skills,
            )
            agent = with_tool_concurrency(agent, MAX_PARALLEL_AGENTS)
            if (
                agent.llm.responses_use_previous_response_id
                or agent.llm.uses_anthropic_compaction()
            ):
                agent = agent.model_copy(update={"condenser": None})
        else:
            condenser = (
                None
                if (
                    llm.responses_use_previous_response_id
                    or llm.uses_anthropic_compaction()
                )
                else get_default_condenser(
                    llm.model_copy(update={"usage_id": "senpai-condenser"})
                )
            )
            agent = Agent(
                llm=llm,
                tools=build_main_tools(config),
                agent_context=build_main_agent_context(
                    config.instructions,
                    project_skills,
                ),
                system_prompt_kwargs={"cli_mode": True},
                condenser=condenser,
                tool_concurrency_limit=MAX_PARALLEL_AGENTS,
            )
        last_activity = time.monotonic()

        def observe_event(event: object) -> None:
            nonlocal last_activity
            last_activity = time.monotonic()
            if on_activity is not None:
                on_activity()
            print_event(event)

        conversation = LocalConversation(
            agent=agent,
            workspace=config.workspace,
            plugins=[PluginSource(source=str(config.plugin_dir))],
            persistence_dir=config.state_dir,
            conversation_id=config.conversation_id,
            callbacks=[] if config.child else [observe_event],
            max_iteration_per_run=config.max_turns,
            visualizer=None,
            secrets=dict(config.conversation_secrets),
            tags={"runtime": "senpai-openhands"},
            delete_on_close=config.child,
            prompt_cache_key=conversation_prompt_cache_key(config),
        )
        reject_recovered_actions(conversation)
        if inbox is not None and active_inbox_turn_id is not None:
            active_inbox_turn_id = inbox.latest_turn(active_inbox_turn_id).turn_id
            if reset_context:
                recovery = inbox.recover_turn(
                    active_inbox_turn_id,
                    recovery_prompt or prompt,
                    max_generations=config.inbox_max_recovery_generations,
                )
                active_inbox_turn_id = recovery.turn_id
        elif reset_context:
            preserved_events = len(conversation.state.events)
            conversation.navigate_to(None)
            print(
                "OPENHANDS_CONTEXT_RESET "
                f"conversation_id={config.conversation_id} "
                f"preserved_events={preserved_events}",
                file=sys.stderr,
                flush=True,
            )
        inference_required = True
        if inbox is None or active_inbox_turn_id is None:
            conversation.send_message(prompt)
        else:
            turn = _activate_inbox_turn(
                conversation,
                inbox,
                active_inbox_turn_id,
            )
            if turn.state is DeliveryState.PROCESSED:
                inference_required = False
            elif turn_has_finished_response(conversation, turn):
                inbox.record_processed(active_inbox_turn_id)
                inference_required = False
        if inference_required:
            if inbox is not None and active_inbox_turn_id is not None:
                turn = inbox.turn(active_inbox_turn_id)
                inbox.record_progress(
                    active_inbox_turn_id,
                    _latest_completed_tool_event_id(conversation, turn),
                )
                if inbox.terminal_recovery_due(
                    active_inbox_turn_id,
                    max_attempts=config.inbox_max_stalled_attempts,
                ):
                    stalled_turn_id = active_inbox_turn_id
                    recovery = inbox.recover_turn(
                        stalled_turn_id,
                        recovery_prompt or prompt,
                        max_generations=config.inbox_max_recovery_generations,
                    )
                    active_inbox_turn_id = recovery.turn_id
                    print(
                        "SENPAI_TERMINAL_TURN_RECOVERY "
                        f"conversation_id={config.conversation_id} "
                        f"stalled_turn_id={stalled_turn_id} "
                        f"recovery_turn_id={active_inbox_turn_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    recovery_turn = _activate_inbox_turn(
                        conversation,
                        inbox,
                        active_inbox_turn_id,
                    )
                    inbox.record_progress(
                        active_inbox_turn_id,
                        _latest_completed_tool_event_id(
                            conversation,
                            recovery_turn,
                        ),
                    )
                inbox.record_inference_attempt(active_inbox_turn_id)
            with graceful_interrupts(conversation) as stop_requested:
                if not config.child:
                    with LocalEventStore(
                        local_event_db_path(config)
                    ) as event_store:
                        event_pump = AdvisorEventPump(
                            event_store,
                            conversation,
                            parent_conversation_id=(
                                str(config.conversation_id)
                                if config.role == "student"
                                else None
                            ),
                            inbox=inbox,
                            conversation_id=config.conversation_id,
                        )
                        with event_pump:
                            run_steerable_conversation(
                                conversation,
                                event_pump,
                                config.timeout_seconds,
                                lambda: last_activity,
                                stop_requested,
                            )
                else:
                    assert run_deadline is not None
                    run_conversation(conversation, run_deadline - time.time())
        status = conversation.state.execution_status
        if (
            inference_required
            and inbox is not None
            and active_inbox_turn_id is not None
            and status == ConversationExecutionStatus.FINISHED
        ):
            inbox.record_processed(active_inbox_turn_id)
        child_result = (
            final_agent_result(conversation)
            if config.child and status == ConversationExecutionStatus.FINISHED
            else None
        )
        if child_result is not None:
            child_result = compact_child_result(
                conversation,
                agent.llm,
                config,
                child_result,
                run_deadline,
            )
    finally:
        primary_exception = sys.exc_info()[1]
        primary_error = primary_exception is not None
        if config.child and config.delegation_task_id and conversation is not None:
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
                            "child agent exited with uncollected descendants; it must "
                            "await or cancel every spawned task first: "
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
        if conversation is not None:
            conversation.close()
        if cleanup_error is not None and not primary_error:
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
                "conversation_id": str(conversation.id),
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
