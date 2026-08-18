"""Build Senpai OpenHands agents, tools, and delegation configuration."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from openhands.sdk import Agent, AgentContext, LLM, Tool
from openhands.sdk.agent.parallel_executor import ParallelToolExecutor
from openhands.sdk.skills import Skill, load_skills_from_dir, merge_skills_by_name
from openhands.sdk.subagent import (
    AgentDefinition,
    agent_definition_to_factory,
    discover_agents,
)
from openhands.tools.preset.default import get_default_condenser, get_default_tools

from senpai_agent.agent_markdown import strip_spdx_header
from senpai_agent.delegation import (
    MAX_DELEGATION_DEPTH,
    MAX_PARALLEL_AGENTS,
    DelegationConfig,
)
from senpai_agent.openhands import REPOSITORY_ROOT
from senpai_agent.openhands.config import RunnerConfig, local_event_db_path
from senpai_agent.openhands.llm import apply_reasoning_profile
from senpai_agent.system_instructions import SenpaiSystemInstructions
from senpai_agent.tools import register_senpai_tools

SENPAI_AGENT_NAMES = ("bash-runner", "general-purpose", "explore", "search")
SENPAI_AGENT_DIR = REPOSITORY_ROOT / ".agents" / "agents"
REPOSITORY_INSTRUCTION_FILENAMES = frozenset({"agents.md", "agent.md", "claude.md"})


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


def without_eager_skill_discovery(definition: AgentDefinition) -> AgentDefinition:
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
        definition.name == "general-purpose" and 0 < depth < MAX_DELEGATION_DEPTH
    ):
        return definition
    return definition.model_copy(
        update={"tools": [tool for tool in definition.tools if tool != "spawn_agents"]}
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


def build_agent(
    config: RunnerConfig,
    llm: LLM,
    file_agents: Sequence[AgentDefinition],
    project_skills: Sequence[Skill],
) -> Agent:
    """Build the configured main or named agent with Senpai's runtime policy."""

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
        return agent

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
    return Agent(
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
