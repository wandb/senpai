"""Run one bounded Senpai OpenHands turn for the Python controller."""

# OpenHands imports intentionally follow Weave initialization below.

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from senpai_agent.advisor import (
    AdvisorEventPump,
    AdvisorEventStore,
    advisor_conversation_id,
    compose_system_instructions,
)
from senpai_agent.delegation import (
    MAX_DELEGATION_DEPTH,
    MAX_PARALLEL_AGENTS,
    DelegationConfig,
    cancel_pending_descendants,
    configure_delegation,
    record_delegated_task_result,
)
from senpai_agent.model_compatibility import (
    REASONING_EFFORTS,
    WANDB_GLM_52_MODEL,
    WANDB_GLM_52_TOKENIZER,
    register_litellm_model_compatibility,
    supports_reasoning_effort,
)
from senpai_agent.secrets import (
    GITHUB_TOKEN_ENV_NAMES,
    GITHUB_TOKEN_FD_ENV,
    GITHUB_TOKEN_FILE_ENV,
    scrub_github_credentials,
)
from senpai_agent.weave_monitoring import (
    finish_weave_monitoring,
    initialize_weave_monitoring,
    register_trace_secret,
    weave_conversation_url,
)

WEAVE_PROJECT = initialize_weave_monitoring()
register_litellm_model_compatibility()

from openhands.sdk import LLM, Agent, AgentContext, LocalConversation, Tool
from openhands.sdk.agent.parallel_executor import ParallelToolExecutor
from openhands.sdk.context.condenser import CondenserBase, LLMSummarizingCondenser
from openhands.sdk.conversation import ConversationExecutionStatus, ConversationState
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.llm import TextContent
from openhands.sdk.plugin import PluginSource
from openhands.sdk.skills import Skill, load_project_skills
from openhands.sdk.subagent import (
    AgentDefinition,
    agent_definition_to_factory,
    discover_agents,
    register_agent_if_absent,
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
from senpai_agent.tools import register_senpai_tools

DEFAULT_MODEL = "openai/gpt-5.6-sol"
DEFAULT_FAST_MODEL = "openai/gpt-5.6-luna"
DEFAULT_FRONTIER_MODEL = "openai/gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_FAST_REASONING_EFFORT = "high"
DEFAULT_FRONTIER_REASONING_EFFORT = "max"
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
SENPAI_AGENT_NAMES = ("bash-runner", "general-purpose", "explore", "search")
SENPAI_AGENT_DIR = Path(__file__).resolve().parents[1] / ".agents" / "agents"
PROVIDER_API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "wandb": "WANDB_API_KEY",
}
COMMAND_SECRET_ENV_NAMES = (
    "WANDB_API_KEY",
    "EXA_API_KEY",
)
EVENT_TEXT_LIMIT = 20000
DEFAULT_LOCAL_CONDENSER_MAX_EVENTS = 0
GENERIC_LOCAL_CONDENSER_MAX_EVENTS = 80
MIN_LOCAL_CONDENSER_MAX_EVENTS = 12
DEFAULT_LOCAL_CONDENSER_MAX_TOKENS = 0
DEFAULT_LOCAL_CONDENSER_TARGET_EVENTS = 0
WANDB_GLM_52_MAX_EVENTS = 600
WANDB_GLM_52_MAX_TOKENS = 180_000
WANDB_GLM_52_TARGET_EVENTS = 40


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
    command_secrets: Mapping[str, str]
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
    workspace: Path
    state_dir: Path
    conversation_id: uuid.UUID
    role: str
    enable_browser: bool
    agent_name: str | None
    harness_file: Path
    role_file: Path
    plugin_dir: Path
    advisor_branch: str | None = None
    student_names: tuple[str, ...] | None = None
    student_name: str | None = None
    wandb_entity: str | None = None
    wandb_project: str | None = None
    training_max_timeout_seconds: int = 1800
    timeout_seconds: float = 3600
    llm_timeout_seconds: int = 900
    llm_num_retries: int = 1
    local_condenser_max_events: int = DEFAULT_LOCAL_CONDENSER_MAX_EVENTS
    local_condenser_max_tokens: int = DEFAULT_LOCAL_CONDENSER_MAX_TOKENS
    local_condenser_target_events: int = DEFAULT_LOCAL_CONDENSER_TARGET_EVENTS
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
    if reasoning_effort not in REASONING_EFFORTS:
        choices = ", ".join(REASONING_EFFORTS)
        raise ValueError(
            f"unsupported reasoning effort {reasoning_effort!r}; "
            f"choose one of: {choices}"
        )
    if not supports_reasoning_effort(model, reasoning_effort):
        raise ValueError(
            f"reasoning effort {reasoning_effort!r} is unsupported for model "
            f"{model!r}; select a supported effort"
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


def command_secrets(env: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value for name in COMMAND_SECRET_ENV_NAMES if (value := env.get(name))
    }


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


def read_role_instructions(path: Path) -> str:
    instructions = read_agent_markdown(path).strip()
    if not instructions:
        raise RuntimeError(f"OpenHands role file is empty: {path}")
    return instructions


def sanitized_project_skills(workspace: Path) -> list[Skill]:
    """Load project instructions without exposing their SPDX boilerplate."""

    return [
        skill.model_copy(update={"content": strip_spdx_header(skill.content)})
        for skill in load_project_skills(workspace)
        if not (
            skill.source
            and (
                workspace / Path(skill.source).parent / ".senpai-developer-only"
            ).is_file()
        )
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


def depth_aware_child_definition(
    definition: AgentDefinition,
    *,
    child: bool,
    depth: int,
) -> AgentDefinition:
    """Expose recursive spawn only to the one child role that can use it."""

    if not child or (
        definition.name == "general-purpose" and 0 < depth < MAX_DELEGATION_DEPTH
    ):
        return definition
    return definition.model_copy(
        update={"tools": [tool for tool in definition.tools if tool != "spawn_agents"]}
    )


def register_agent_definitions(
    definitions: Sequence[AgentDefinition], workspace: Path
) -> None:
    for definition in definitions:
        register_agent_if_absent(
            name=definition.name,
            factory_func=agent_definition_to_factory(definition, work_dir=workspace),
            description=definition,
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
    role = env.get("SENPAI_ROLE", "")
    if role not in {"advisor", "student"}:
        raise RuntimeError("SENPAI_ROLE must be advisor or student")
    try:
        training_max_timeout_seconds = round(
            float(env.get("SENPAI_TIMEOUT_MINUTES", "30")) * 60
        )
    except ValueError as error:
        raise RuntimeError("SENPAI_TIMEOUT_MINUTES must be numeric") from error
    if training_max_timeout_seconds <= 0:
        raise RuntimeError("SENPAI_TIMEOUT_MINUTES must be positive")
    try:
        timeout_seconds = float(env.get("SENPAI_OPENHANDS_TIMEOUT_SECONDS", "3600"))
    except ValueError as error:
        raise RuntimeError(
            "SENPAI_OPENHANDS_TIMEOUT_SECONDS must be numeric"
        ) from error
    if timeout_seconds <= 0:
        raise RuntimeError("SENPAI_OPENHANDS_TIMEOUT_SECONDS must be positive")
    try:
        llm_timeout_seconds = int(env.get("SENPAI_LLM_TIMEOUT_SECONDS", "900"))
        llm_num_retries = int(env.get("SENPAI_LLM_NUM_RETRIES", "1"))
    except ValueError as error:
        raise RuntimeError("Senpai LLM timeout settings must be numeric") from error
    try:
        local_condenser_max_events = int(
            env.get(
                "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS",
                str(DEFAULT_LOCAL_CONDENSER_MAX_EVENTS),
            )
        )
        local_condenser_max_tokens = int(
            env.get(
                "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS",
                str(DEFAULT_LOCAL_CONDENSER_MAX_TOKENS),
            )
        )
        local_condenser_target_events = int(
            env.get(
                "SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS",
                str(DEFAULT_LOCAL_CONDENSER_TARGET_EVENTS),
            )
        )
    except ValueError as error:
        raise RuntimeError("local condenser limits must be integers") from error
    if llm_timeout_seconds <= 0 or llm_num_retries <= 0:
        raise RuntimeError("Senpai LLM timeout and attempts must be positive")
    if (
        local_condenser_max_events
        and local_condenser_max_events < MIN_LOCAL_CONDENSER_MAX_EVENTS
    ):
        raise RuntimeError(
            "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS must be 0 or at least "
            f"{MIN_LOCAL_CONDENSER_MAX_EVENTS}"
        )
    if local_condenser_max_tokens < 0:
        raise RuntimeError(
            "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS must be non-negative"
        )
    if local_condenser_target_events < 0:
        raise RuntimeError(
            "SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS must be non-negative"
        )
    if (
        local_condenser_max_events
        and local_condenser_target_events
        and local_condenser_target_events >= local_condenser_max_events
    ):
        raise RuntimeError(
            "SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS must be less than "
            "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS"
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
        delegation_deadline_epoch = float(deadline_value) if deadline_value else None
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
    api_key_env = env_value(
        args.api_key_env, env, "SENPAI_OPENHANDS_API_KEY_ENV"
    ) or infer_api_key_env(
        model,
        override_env="SENPAI_OPENHANDS_API_KEY_ENV",
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
        command_secrets=command_secrets(env),
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
        harness_file=find_harness_file(
            env_value(
                args.harness_file,
                env,
                "SENPAI_OPENHANDS_HARNESS_FILE",
            )
        ),
        role_file=find_role_file(
            env_value(args.role_file, env, "SENPAI_OPENHANDS_ROLE_FILE"),
        ),
        plugin_dir=resolve_plugin_dir(
            env_value(args.plugin_dir, env, "SENPAI_PLUGIN"),
        ),
        advisor_branch=env.get("ADVISOR_BRANCH") or None,
        student_names=tuple(
            name.strip()
            for name in env.get("STUDENT_NAMES", "").split(",")
            if name.strip()
        ),
        student_name=env.get("STUDENT_NAME") or None,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        training_max_timeout_seconds=training_max_timeout_seconds,
        timeout_seconds=timeout_seconds,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_num_retries=llm_num_retries,
        local_condenser_max_events=local_condenser_max_events,
        local_condenser_max_tokens=local_condenser_max_tokens,
        local_condenser_target_events=local_condenser_target_events,
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


def with_role_and_project_context(
    agent: Agent,
    harness_instructions: str,
    role_instructions: str,
    project_skills: Sequence[Skill] = (),
    runtime_invariant: str = "",
) -> Agent:
    context = agent.agent_context or AgentContext()
    skills = {skill.name: skill for skill in context.skills}
    skills.update({skill.name: skill for skill in project_skills})
    role_suffix = compose_system_instructions(
        harness_instructions,
        role_instructions,
    )
    if runtime_invariant:
        role_suffix += f"\n# Live controller invariant\n\n{runtime_invariant.strip()}\n"
    system_suffix = (
        f"{context.system_message_suffix}\n\n{role_suffix}"
        if context.system_message_suffix
        else role_suffix
    )
    return agent.model_copy(
        update={
            "agent_context": context.model_copy(
                update={
                    "system_message_suffix": system_suffix,
                    "current_datetime": None,
                    "skills": list(skills.values()),
                    "load_project_skills": False,
                }
            )
        }
    )


def build_main_agent_context(
    harness_instructions: str,
    role_instructions: str,
    project_skills: Sequence[Skill] = (),
    runtime_invariant: str = "",
) -> AgentContext:
    system_suffix = compose_system_instructions(
        harness_instructions,
        role_instructions,
    )
    if runtime_invariant:
        system_suffix += (
            f"\n# Live controller invariant\n\n{runtime_invariant.strip()}\n"
        )
    return AgentContext(
        skills=list(project_skills),
        system_message_suffix=system_suffix,
        current_datetime=None,
        load_public_skills=False,
        load_user_skills=True,
        load_project_skills=False,
    )


def live_controller_invariant(config: RunnerConfig) -> str:
    """Return authoritative root-role state that compaction cannot rewrite."""

    if config.child or config.role != "advisor":
        return ""
    return (
        "The advisor campaign is active. This runtime has no configured campaign "
        f"round limit. max_turns={config.max_turns} bounds one OpenHands turn; it "
        'does not mark the research complete. A round label such as "FINAL ROUND" '
        "or a condensed-history summary never authorizes stopping. Continue until "
        "an explicit human instruction or controller shutdown ends the campaign."
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
        "responses_compact_threshold": 200_000,
    }
    if reasoning := _openai_pro_reasoning(model, reasoning_effort):
        configuration["litellm_extra_body"] = {
            "reasoning": reasoning,
        }
    return configuration


def model_runtime_configuration(
    model: str,
    reasoning_effort: str,
    *,
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
            configuration["custom_tokenizer"] = WANDB_GLM_52_TOKENIZER
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
        anthropic_compaction_configuration(model),
    ):
        for key, value in options.items():
            if key == "litellm_extra_body":
                extra_body.update(value)
            else:
                configuration[key] = value
    if extra_body:
        configuration["litellm_extra_body"] = extra_body
    return configuration


def anthropic_compaction_configuration(model: str) -> dict[str, int]:
    if model.split("/", 1)[0].lower() != "anthropic":
        return {}
    return {"anthropic_compact_threshold": 100_000}


def configured_local_condenser(
    llm: LLM,
    max_events: int,
    condenser: CondenserBase | None = None,
    *,
    max_tokens: int = DEFAULT_LOCAL_CONDENSER_MAX_TOKENS,
    target_events: int = DEFAULT_LOCAL_CONDENSER_TARGET_EVENTS,
) -> CondenserBase | None:
    """Configure the local fallback without overriding provider compaction."""

    if llm.responses_use_previous_response_id or llm.uses_anthropic_compaction():
        return None
    selected = condenser
    if selected is None:
        selected = get_default_condenser(
            llm.model_copy(update={"usage_id": "senpai-condenser"})
        )
    if not isinstance(selected, LLMSummarizingCondenser):
        return selected
    max_events, max_tokens, target_events = local_condenser_limits(
        llm.model,
        max_events,
        max_tokens,
        target_events,
    )
    max_tokens = max_tokens or selected.max_tokens
    target_events = target_events or selected.target_size
    retained_events = target_events or max_events // 2
    if retained_events > int(max_events * (1 - selected.minimum_progress)):
        raise ValueError(
            "local condenser target events must leave the configured minimum progress"
        )
    if retained_events <= selected.keep_first + 1:
        raise ValueError(
            "local condenser target events must leave room after keep_first"
        )
    return selected.model_copy(
        update={
            "max_size": max_events,
            "max_tokens": max_tokens or None,
            "target_size": target_events or None,
        }
    )


def local_condenser_limits(
    model: str,
    max_events: int,
    max_tokens: int,
    target_events: int,
) -> tuple[int, int | None, int | None]:
    """Resolve zero-valued knobs against the actual model profile."""

    if model.lower() == WANDB_GLM_52_MODEL:
        return (
            max_events or WANDB_GLM_52_MAX_EVENTS,
            max_tokens or WANDB_GLM_52_MAX_TOKENS,
            target_events or WANDB_GLM_52_TARGET_EVENTS,
        )
    return (
        max_events or GENERIC_LOCAL_CONDENSER_MAX_EVENTS,
        max_tokens or None,
        target_events or None,
    )


def require_exact_tokenizer(llm: LLM) -> None:
    """Fail before a GLM turn if exact chat-template accounting is unavailable."""

    if (
        llm.model.lower() == WANDB_GLM_52_MODEL
        and not llm.has_chat_template_tokenizer()
    ):
        raise RuntimeError(
            f"{WANDB_GLM_52_MODEL} requires the exact "
            f"{WANDB_GLM_52_TOKENIZER} chat-template tokenizer"
        )


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
    """Build Senpai's role-safe root tool surface."""

    if config.github_token is None:
        raise RuntimeError("main agents require GitHub credentials")
    register_senpai_tools()
    tools = [
        tool
        for tool in get_default_tools(
            enable_browser=False,
            enable_sub_agents=False,
        )
        if tool.name not in {"terminal", "think"}
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
    tools.extend(
        Tool(name=name, params=delegation_params)
        for name in (
            "spawn_agents",
            "await_agents",
            "agent_status",
            "cancel_agents",
        )
    )
    if not config.child:
        job_params: dict[str, str | int] = {
            "state_dir": str(config.state_dir / "training"),
            "max_timeout_seconds": config.training_max_timeout_seconds,
        }
        tools.append(Tool(name="senpai_training", params=job_params))
    return tools


def without_legacy_think(agent: Agent) -> Agent:
    """Remove OpenHands' legacy scratchpad from the actual runtime surface."""

    return agent.model_copy(
        update={
            "tools": [tool for tool in agent.tools if tool.name != "think"],
            "include_default_tools": [
                name for name in agent.include_default_tools if name != "ThinkTool"
            ],
        }
    )


def migrate_persisted_disabled_tools(
    state_dir: Path,
    conversation_id: uuid.UUID,
) -> bool:
    """Atomically remove only Think declarations from one saved agent spec."""

    path = Path(state_dir) / conversation_id.hex / "base_state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"cannot migrate persisted agent state at {path}: {error}"
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("agent"), dict):
        raise TypeError(f"persisted agent state at {path} has an unknown shape")
    saved_agent = payload["agent"]
    tools = saved_agent.get("tools")
    defaults = saved_agent.get("include_default_tools")
    if not isinstance(tools, list) or not all(
        isinstance(tool, dict) and isinstance(tool.get("name"), str) for tool in tools
    ):
        raise RuntimeError(f"persisted agent tools at {path} have an unknown shape")
    if defaults is not None and (
        not isinstance(defaults, list)
        or not all(isinstance(name, str) for name in defaults)
    ):
        raise RuntimeError(f"persisted default tools at {path} have an unknown shape")

    migrated_tools = [tool for tool in tools if tool["name"] != "think"]
    migrated_defaults = [
        name
        for name in (defaults if defaults is not None else ["FinishTool", "ThinkTool"])
        if name != "ThinkTool"
    ]
    if migrated_tools == tools and migrated_defaults == defaults:
        return False
    saved_agent["tools"] = migrated_tools
    saved_agent["include_default_tools"] = migrated_defaults

    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return True


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
        github_repo=config.github_repo,
        github_trusted_actor=config.github_trusted_actor,
        role_file=config.role_file,
        harness_file=config.harness_file,
        plugin_dir=config.plugin_dir,
        enable_browser=config.enable_browser,
        command_secrets=config.command_secrets,
        role=config.role,
        local_condenser_max_events=config.local_condenser_max_events,
        local_condenser_max_tokens=config.local_condenser_max_tokens,
        local_condenser_target_events=config.local_condenser_target_events,
        root_state_dir=config.delegation_root_state_dir,
        tree_id=config.delegation_tree_id,
        depth=config.delegation_depth,
        deadline_epoch=deadline_epoch or config.delegation_deadline_epoch,
        agent_name=config.agent_name,
        current_task_id=config.delegation_task_id,
    )


@contextmanager
def graceful_interrupts(conversation: object) -> Iterator[None]:
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
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if interrupted_by:
        raise SystemExit(128 + interrupted_by[0])


async def arun_conversation(
    conversation: object,
    timeout_seconds: float,
) -> None:
    """Run the async OpenHands path so timeout cancellation reaches tools."""

    task = asyncio.create_task(conversation.arun())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
    except TimeoutError:
        print(
            f"OPENHANDS_TIMEOUT seconds={timeout_seconds:g}",
            file=sys.stderr,
            flush=True,
        )
        conversation.interrupt()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def run_conversation(conversation: object, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        print(
            f"OPENHANDS_TIMEOUT seconds={max(timeout_seconds, 0):g}",
            file=sys.stderr,
            flush=True,
        )
        conversation.interrupt()
        return
    asyncio.run(arun_conversation(conversation, timeout_seconds))


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
    conversation.reject_pending_actions(
        reason=(
            "Senpai restarted before this action completed. Inspect the preserved "
            "workspace and rerun it explicitly only if it is still needed."
        )
    )
    print(
        f"OPENHANDS_RECOVERED_ACTIONS rejected={len(pending)}",
        file=sys.stderr,
        flush=True,
    )
    return len(pending)


def final_agent_result(conversation: object) -> str:
    for event in reversed(conversation.state.view.events):
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


def run_openhands(
    prompt: str,
    config: RunnerConfig,
    *,
    reset_context: bool = False,
) -> int:
    started_at = time.time()
    run_deadline = min(
        started_at + config.timeout_seconds,
        config.delegation_deadline_epoch or float("inf"),
    )
    run_timeout = run_deadline - started_at
    if run_timeout <= 0:
        raise TimeoutError("the inherited OpenHands deadline has expired")
    scrub_model_credentials(os.environ, config)
    harness_instructions = read_role_instructions(config.harness_file)
    role_instructions = read_role_instructions(config.role_file)
    register_default_tools(enable_browser=False)
    register_senpai_tools()
    file_agents = sanitized_agent_definitions(config.workspace)
    register_agent_definitions(file_agents, config.workspace)
    available_agents = [definition.name for definition in file_agents]
    project_skills = sanitized_project_skills(config.workspace)
    os.environ["SENPAI_CONVERSATION_ID"] = config.conversation_id.hex
    local_condenser = (
        (None, None, None)
        if model_provider(config.model) in {"openai", "anthropic"}
        else local_condenser_limits(
            config.model,
            config.local_condenser_max_events,
            config.local_condenser_max_tokens,
            config.local_condenser_target_events,
        )
    )

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
                "local_condenser_max_events": local_condenser[0],
                "local_condenser_max_tokens": local_condenser[1],
                "local_condenser_target_events": local_condenser[2],
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
        register_trace_secret(config.github_token.get_secret_value())
        configure_github_credentials(
            config.github_repo,
            config.github_token,
            trusted_actor=config.github_trusted_actor,
        )
    configure_delegation(delegation_config(config, deadline_epoch=run_deadline))
    scrub_github_credentials(os.environ)
    conversation = None
    cleanup_error: BaseException | None = None
    try:
        llm = LLM(
            model=config.model,
            api_key=config.api_key,
            timeout=config.llm_timeout_seconds,
            num_retries=config.llm_num_retries,
            reasoning_effort=openhands_reasoning_effort(
                config.reasoning_effort, config.model
            ),
            usage_id="senpai",
            **model_runtime_configuration(
                config.model,
                config.reasoning_effort,
                wandb_entity=config.wandb_entity,
                wandb_project=config.wandb_project,
            ),
        )
        require_exact_tokenizer(llm)
        if config.agent_name:
            definition = depth_aware_child_definition(
                find_named_agent(config.agent_name, file_agents),
                child=config.child,
                depth=config.delegation_depth,
            )
            agent = agent_definition_to_factory(
                definition,
                work_dir=config.workspace,
            )(llm)
            agent = agent.model_copy(update={"llm": apply_reasoning_profile(agent.llm)})
            require_exact_tokenizer(agent.llm)
            agent = with_role_and_project_context(
                agent,
                harness_instructions,
                role_instructions,
                project_skills,
                live_controller_invariant(config),
            )
            agent = with_tool_concurrency(agent, MAX_PARALLEL_AGENTS)
            agent = agent.model_copy(
                update={
                    "condenser": configured_local_condenser(
                        agent.llm,
                        config.local_condenser_max_events,
                        agent.condenser,
                        max_tokens=config.local_condenser_max_tokens,
                        target_events=config.local_condenser_target_events,
                    )
                }
            )
        else:
            condenser = configured_local_condenser(
                llm,
                config.local_condenser_max_events,
                max_tokens=config.local_condenser_max_tokens,
                target_events=config.local_condenser_target_events,
            )
            agent = Agent(
                llm=llm,
                tools=build_main_tools(config),
                agent_context=build_main_agent_context(
                    harness_instructions,
                    role_instructions,
                    project_skills,
                    live_controller_invariant(config),
                ),
                system_prompt_kwargs={"cli_mode": True},
                condenser=condenser,
                tool_concurrency_limit=MAX_PARALLEL_AGENTS,
            )
        agent = without_legacy_think(agent)
        if migrate_persisted_disabled_tools(
            config.state_dir,
            config.conversation_id,
        ):
            print(
                "OPENHANDS_STATE_MIGRATION removed_tool=think "
                f"conversation_id={config.conversation_id}",
                file=sys.stderr,
                flush=True,
            )
        conversation = LocalConversation(
            agent=agent,
            workspace=config.workspace,
            plugins=[PluginSource(source=str(config.plugin_dir))],
            persistence_dir=config.state_dir,
            conversation_id=config.conversation_id,
            callbacks=[] if config.child else [print_event],
            max_iteration_per_run=config.max_turns,
            visualizer=None,
            secrets=dict(config.command_secrets),
            tags={"runtime": "senpai-openhands"},
            delete_on_close=config.child,
            prompt_cache_key=conversation_prompt_cache_key(config),
        )
        reject_recovered_actions(conversation)
        if reset_context:
            preserved_events = len(conversation.state.events)
            conversation.navigate_to(None)
            print(
                "OPENHANDS_CONTEXT_RESET "
                f"conversation_id={config.conversation_id} "
                f"preserved_events={preserved_events}",
                file=sys.stderr,
                flush=True,
            )
        try:
            # send_message performs OpenHands' lazy tool initialization.
            conversation.send_message(prompt)
        finally:
            clear_github_credentials()
            configure_delegation(None)
        with graceful_interrupts(conversation):
            if not config.child:
                with (
                    AdvisorEventStore(local_event_db_path(config)) as event_store,
                    AdvisorEventPump(
                        event_store,
                        conversation,
                        parent_conversation_id=(
                            str(config.conversation_id)
                            if config.role == "student"
                            else None
                        ),
                    ),
                ):
                    run_conversation(conversation, run_deadline - time.time())
            else:
                run_conversation(conversation, run_deadline - time.time())
        status = conversation.state.execution_status
        child_result = (
            final_agent_result(conversation)
            if config.child and status == ConversationExecutionStatus.FINISHED
            else None
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
    return 0 if status == ConversationExecutionStatus.FINISHED else 1


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
        except BaseException as error:
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
