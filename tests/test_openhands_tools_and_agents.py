import os
import shutil
import subprocess
import sys
import textwrap

import pytest
from openhands.sdk import Agent, LLM, Tool
from openhands.sdk.plugin import Plugin
from openhands.sdk.subagent import AgentDefinition, agent_definition_to_factory
from openhands.tools.preset.default import register_default_tools
from pydantic import SecretStr

from senpai_agent.openhands_runner import (
    build_main_tools,
    delegation_config,
    find_named_agent,
    resolve_plugin_dir,
    sanitized_agent_definitions,
)
from senpai_agent.tools import register_senpai_tools
from openhands_support import AGENT_DIR, PLUGIN_DIR, REPO_ROOT, runtime_config


def test_runtime_agent_keeps_the_persisted_delegate_tool_compatible():
    llm = LLM(
        model="anthropic/claude-opus-4-8",
        api_key=SecretStr("test-key"),
    )
    persisted = Agent(llm=llm, tools=[Tool(name="delegate_agent")])
    runtime = Agent(
        llm=llm,
        tools=[
            Tool(name="delegate_agent"),
            Tool(name="spawn_agents"),
            Tool(name="await_agents"),
        ],
    )

    assert runtime.verify(persisted) is runtime


def test_child_mode_keeps_bounded_delegation_lifecycle_tools(tmp_path):
    config = runtime_config(tmp_path, child=True)
    names = {tool.name for tool in build_main_tools(config)}

    assert "task_tool_set" not in names
    assert {
        "spawn_agents",
        "await_agents",
        "agent_status",
        "cancel_agents",
    } <= names
    assert "delegate_agent" not in names
    assert "senpai_training" not in names
    assert delegation_config(config).depth == 0


@pytest.mark.parametrize(
    ("role", "expected_custom"),
    [
        (
            "advisor",
            {
                "senpai_terminal",
                "get_prs",
                "github_transition",
                "delegate_agent",
                "spawn_agents",
                "await_agents",
                "agent_status",
                "cancel_agents",
            },
        ),
        (
            "student",
            {
                "senpai_terminal",
                "get_prs",
                "github_transition",
                "delegate_agent",
                "spawn_agents",
                "await_agents",
                "agent_status",
                "cancel_agents",
                "senpai_training",
            },
        ),
    ],
)
def test_main_tools_replace_unsafe_defaults_with_role_scoped_boundaries(
    tmp_path,
    role: str,
    expected_custom: set[str],
):
    config = runtime_config(tmp_path, role=role)
    by_name = {tool.name: tool for tool in build_main_tools(config)}

    assert "terminal" not in by_name
    assert "task_tool_set" not in by_name
    assert expected_custom <= set(by_name)
    assert by_name["senpai_terminal"].params == {"role": role}
    delegation_params = {
        "event_db_path": str(config.state_dir / f"{role}-events.sqlite3")
    }
    for name in (
        "delegate_agent",
        "spawn_agents",
        "await_agents",
        "agent_status",
        "cancel_agents",
    ):
        assert by_name[name].params == delegation_params
    assert by_name["get_prs"].params == {
        "state_dir": str(config.state_dir / "github")
    }
    if role == "student":
        assert by_name["senpai_training"].params == {
            "state_dir": str(config.state_dir / "training"),
            "max_timeout_seconds": 1800,
        }


def test_supervisor_receives_only_the_campaign_operations_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDENT_NAMES", "fern,frieren")
    monkeypatch.setenv("SENPAI_KUBECTL_NAMESPACE", "research")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setenv("ADVISOR_BRANCH", "maple-advisor")
    config = runtime_config(tmp_path, role="supervisor")

    tools = build_main_tools(config)

    assert [tool.name for tool in tools] == ["senpai_operations"]
    assert tools[0].params == {
        "state_dir": str(config.state_dir),
        "namespace": "research",
        "research_tag": "maple",
        "repo": "acme/widgets",
        "advisor_branch": "maple-advisor",
        "students": ("fern", "frieren"),
        "mutation_cooldown_seconds": 1800.0,
    }


def test_native_senpai_plugin_loads_its_runtime_skills():
    assert resolve_plugin_dir(str(PLUGIN_DIR)) == PLUGIN_DIR

    plugin = Plugin.load(PLUGIN_DIR)

    assert plugin.manifest.name == "senpai"
    skills = {skill.name: skill for skill in plugin.skills}
    assert set(skills) == {
        "assign-experiment",
        "bootstrap-target",
        "check-human-issues",
        "merge-winner",
        "submit-experiment-results",
    }
    assert "merge_experiment" in skills["merge-winner"].content
    assert "close_experiment" in skills["merge-winner"].content
    assert plugin.mcp_config == {}


def test_target_agents_cannot_shadow_senpai_delegation_agents(tmp_path):
    target_agents = tmp_path / ".agents" / "agents"
    target_agents.mkdir(parents=True)
    (target_agents / "general-purpose.md").write_text(
        "---\n"
        "name: general-purpose\n"
        "description: Shadow Senpai's generalist.\n"
        "reasoning_effort: low\n"
        "tools: [terminal]\n"
        "---\n\n"
        "Shadowed instructions.\n",
        encoding="utf-8",
    )

    definition = find_named_agent(
        "general-purpose",
        sanitized_agent_definitions(tmp_path),
    )

    assert definition.reasoning_effort is None
    assert set(definition.tools) == {
        "terminal",
        "file_editor",
        "task_tracker",
        "spawn_agents",
        "await_agents",
        "agent_status",
        "cancel_agents",
    }
    assert "Shadowed instructions" not in definition.system_prompt


def test_markdown_agents_register_and_construct_with_the_native_loader(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "target"
    workspace.mkdir()
    shutil.copytree(REPO_ROOT / ".agents", home / ".agents")
    program = textwrap.dedent(
        """
        import os
        from pathlib import Path

        from openhands.sdk import LLM
        from openhands.sdk.subagent import (
            agent_definition_to_factory,
            get_registered_agent_definitions,
            register_file_agents,
        )
        from pydantic import SecretStr

        import openhands.tools
        from senpai_agent.tools import register_senpai_tools

        register_senpai_tools()
        workspace = Path(os.environ["SENPAI_TEST_WORKSPACE"])
        registered = register_file_agents(workspace)
        assert set(registered) == {
            "bash-runner",
            "general-purpose",
            "explore",
            "search",
        }
        definitions = {
            definition.name: definition
            for definition in get_registered_agent_definitions()
        }
        llm = LLM(
            model="anthropic/claude-opus-4-8",
            api_key=SecretStr("test-key"),
            reasoning_effort="low",
        )
        agents = {
            name: agent_definition_to_factory(definition, work_dir=workspace)(llm)
            for name, definition in definitions.items()
        }
        assert {tool.name for tool in agents["search"].tools} == {
            "terminal",
            "file_editor",
        }
        assert {tool.name for tool in agents["bash-runner"].tools} == {"terminal"}
        assert agents["search"].llm.reasoning_effort == "low"
        assert agents["explore"].llm.reasoning_effort == "low"
        """
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "OPENHANDS_SUPPRESS_BANNER": "1",
        "SENPAI_TEST_WORKSPACE": str(workspace),
    }

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_search_agent_loads_its_progressive_skills_and_inherits_reasoning_effort(
    monkeypatch,
):
    import openhands.sdk.skills.skill as skill_module

    monkeypatch.setattr(
        skill_module,
        "USER_SKILLS_DIRS",
        [REPO_ROOT / ".agents" / "skills", PLUGIN_DIR / "skills"],
    )
    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    register_default_tools(enable_browser=False)
    register_senpai_tools()
    definition = AgentDefinition.load(AGENT_DIR / "search.md")
    agent = agent_definition_to_factory(definition, work_dir=REPO_ROOT)(
        LLM(
            model="anthropic/claude-opus-4-8",
            api_key=SecretStr("test-key"),
            reasoning_effort="low",
        )
    )

    assert agent.llm.reasoning_effort == "low"
    assert {skill.name for skill in agent.agent_context.skills} == {
        "exa-search",
        "alphaxiv-paper-lookup",
    }
    assert all(skill.is_agentskills_format for skill in agent.agent_context.skills)
    assert all(
        skill.content not in agent.agent_context.system_message_suffix
        for skill in agent.agent_context.skills
    )


@pytest.mark.parametrize(
    ("filename", "name", "effort", "tools", "skills"),
    [
        ("bash-runner.md", "bash-runner", None, {"terminal"}, set()),
        (
            "explore.md",
            "explore",
            None,
            {"terminal", "file_editor"},
            set(),
        ),
        (
            "general-purpose.md",
            "general-purpose",
            None,
            {
                "terminal",
                "file_editor",
                "task_tracker",
                "spawn_agents",
                "await_agents",
                "agent_status",
                "cancel_agents",
            },
            set(),
        ),
        (
            "search.md",
            "search",
            None,
            {"terminal", "file_editor"},
            {"exa-search", "alphaxiv-paper-lookup"},
        ),
    ],
)
def test_file_agent_definitions_keep_bounded_tools_and_no_github_mutations(
    filename: str,
    name: str,
    effort: str | None,
    tools: set[str],
    skills: set[str],
):
    definition = AgentDefinition.load(AGENT_DIR / filename)

    assert definition.name == name
    assert definition.model == "inherit"
    assert definition.reasoning_effort == effort
    assert definition.permission_mode == "never_confirm"
    assert set(definition.tools) == tools
    assert set(definition.skills) == skills
    assert {"get_prs", "github_transition"}.isdisjoint(definition.tools)


def test_advisor_research_precedes_assignment_without_idle_dispatch_priority():
    instructions = (
        REPO_ROOT / "system_instructions" / "ADVISOR.md"
    ).read_text(encoding="utf-8")

    assert "Assigning high-value work to idle students" not in instructions
    assert instructions.index("Research and synthesis needed") < instructions.index(
        "Well-founded experiment assignments"
    )
    assert "Idleness is not a reason to skip" in instructions

    template = (
        REPO_ROOT
        / "plugins"
        / "senpai"
        / "skills"
        / "bootstrap-target"
        / "references"
        / "role-overlay-template.md"
    ).read_text(encoding="utf-8")
    assert "Assign work to every idle student" not in template
    assert template.index("Research and synthesize") < template.index(
        "Assign the best well-founded experiments"
    )


def test_harness_states_bounded_delegation_tree_contract():
    instructions = (
        REPO_ROOT / "system_instructions" / "SENPAI-HARNESS.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(instructions.split())

    for required in (
        "`spawn_agents`",
        "`await_agents`",
        "`agent_status`",
        "`cancel_agents`",
        "timeout of at most five minutes",
        "at most eight children in total",
        "depth-one general-purpose child",
        "Explore, Search, Bash Runner, and every depth-two child are leaves",
        "ten minutes for `fast`",
        "thirty for `smart`",
        "one hour for `frontier`",
    ):
        assert required in normalized
