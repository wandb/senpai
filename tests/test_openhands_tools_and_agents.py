import os
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from openhands.sdk import Agent, LLM, Tool
from openhands.sdk.tool import resolve_tool
from openhands.sdk.plugin import Plugin
from openhands.sdk.subagent import AgentDefinition, agent_definition_to_factory
from openhands.tools.preset.default import register_default_tools
from pydantic import SecretStr

from senpai_agent.openhands_runner import (
    build_main_tools,
    depth_aware_child_definition,
    delegation_config,
    find_named_agent,
    resolve_plugin_dir,
    sanitized_agent_definitions,
)
from senpai_agent.tools import (
    LoadBrowserAction,
    LoadBrowserTool,
    SenpaiTaskTrackerTool,
    register_senpai_tools,
)
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


def test_browser_family_is_lazy_and_respects_disable_flag(tmp_path):
    enabled_tools = build_main_tools(runtime_config(tmp_path, enable_browser=True))
    enabled = {tool.name for tool in enabled_tools}
    disabled = {tool.name for tool in build_main_tools(runtime_config(tmp_path, enable_browser=False))}
    state = SimpleNamespace(agent_state={})
    resolved = resolve_tool(
        next(tool for tool in enabled_tools if tool.name == "browser_tool_set"),
        state,
    )

    assert "browser_tool_set" in enabled
    assert [tool.name for tool in resolved] == ["load_browser"]
    assert "load_browser" not in disabled
    assert "browser_tool_set" not in disabled


def test_lazy_browser_keeps_the_persisted_tool_spec_compatible():
    llm = LLM(
        model="anthropic/claude-opus-4-8",
        api_key=SecretStr("test-key"),
    )
    persisted = Agent(llm=llm, tools=[Tool(name="browser_tool_set")])
    runtime = Agent(llm=llm, tools=[Tool(name="browser_tool_set")])

    assert runtime.verify(persisted) is runtime


def test_browser_loader_uses_runtime_tools_and_persists_activation(monkeypatch):
    from openhands.tools.browser_use import BrowserToolSet

    browser_tool = SimpleNamespace(name="browser_navigate")
    monkeypatch.setattr(
        BrowserToolSet,
        "create",
        classmethod(lambda cls, state: [browser_tool]),
    )

    class RuntimeAgent:
        def __init__(self):
            self.tools_map = {"load_browser": object()}
            self.added = []

        def add_runtime_tools(self, tools):
            self.added.extend(tools)
            self.tools_map.update({tool.name: tool for tool in tools})

    state = SimpleNamespace(agent_state={})
    agent = RuntimeAgent()
    conversation = SimpleNamespace(state=state, agent=agent)
    loader = LoadBrowserTool.create(state)[0]

    observation = loader(LoadBrowserAction(), conversation)

    assert observation.tools == ("browser_navigate",)
    assert agent.added == [browser_tool]
    assert state.agent_state == {"senpai.browser_enabled": True}
    assert LoadBrowserTool.create(state) == [browser_tool]


@pytest.mark.parametrize(
    ("role", "expected_custom"),
    [
        (
            "advisor",
            {
                "senpai_terminal",
                "senpai_github",
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
                "senpai_github",
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
    config = runtime_config(
        tmp_path,
        role=role,
        advisor_branch="advisor-branch" if role == "advisor" else None,
        student_names=("student-one",) if role == "advisor" else None,
        student_name="student-one" if role == "student" else None,
    )
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
    assert by_name["senpai_github"].params == {
        "role": role,
        "state_dir": str(config.state_dir / "github"),
        "advisor_branch": "advisor-branch" if role == "advisor" else None,
        "student_names": ("student-one",) if role == "advisor" else None,
        "student_name": "student-one" if role == "student" else None,
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
        "review-experiment",
        "submit-experiment-results",
    }
    assert "merge_experiment" in skills["review-experiment"].content
    assert "close_experiment" in skills["review-experiment"].content
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


def test_child_definition_exposes_spawn_only_to_depth_one_generalist():
    general = AgentDefinition.load(AGENT_DIR / "general-purpose.md")
    explore = AgentDefinition.load(AGENT_DIR / "explore.md")

    assert "spawn_agents" in depth_aware_child_definition(
        general, child=False, depth=2
    ).tools
    assert "spawn_agents" in depth_aware_child_definition(
        general, child=True, depth=1
    ).tools
    assert "spawn_agents" not in depth_aware_child_definition(
        general, child=True, depth=2
    ).tools
    assert "spawn_agents" not in depth_aware_child_definition(
        explore, child=True, depth=1
    ).tools


def test_senpai_task_tracker_description_is_concise_and_parallel_safe(tmp_path):
    state = type("State", (), {"persistence_dir": str(tmp_path)})()
    tool = SenpaiTaskTrackerTool.create(state)[0]

    assert "genuinely parallel" in tool.description
    assert "Limit active work to ONE" not in tool.description
    assert len(tool.description) < 700


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
    assert {
        "senpai_github",
        "get_prs",
        "create_assignment",
        "publish_advisor_branch",
        "repair_assignment_routing",
        "send_assignment_feedback",
        "request_assignment_revision",
        "accept_result_on_current_base",
        "merge_experiment",
        "close_experiment",
        "respond_to_human_issue",
        "submit_experiment_result",
    }.isdisjoint(definition.tools)


def test_advisor_research_precedes_assignment_without_idle_dispatch_priority():
    instructions = (
        REPO_ROOT / "system_instructions" / "ADVISOR.md"
    ).read_text(encoding="utf-8")

    assert "Assigning high-value work to idle students" not in instructions
    assert instructions.index("Research and synthesis needed") < instructions.index(
        "Well-founded experiment assignments"
    )
    assert "Idleness is not a reason to skip" in instructions


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
