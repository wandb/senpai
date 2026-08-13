import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk import LLM, Agent, Tool
from openhands.sdk.plugin import Plugin
from openhands.sdk.subagent import AgentDefinition, agent_definition_to_factory
from openhands.sdk.tool import resolve_tool
from openhands.tools.preset.default import register_default_tools
from openhands.tools.terminal import TerminalAction
from openhands.tools.terminal.impl import TerminalExecutor
from openhands_support import AGENT_DIR, PLUGIN_DIR, REPO_ROOT, runtime_config
from pydantic import SecretStr

from socket_test_support import short_socket_path

from senpai_agent.openhands_runner import (
    build_main_tools,
    delegation_config,
    depth_aware_child_definition,
    find_named_agent,
    resolve_plugin_dir,
    sanitized_agent_definitions,
    without_legacy_think,
)
from senpai_agent.tools import (
    LoadBrowserAction,
    LoadBrowserTool,
    SenpaiTaskTrackerTool,
    register_senpai_tools,
)


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
    assert "senpai_training" not in names
    assert "think" not in names
    assert delegation_config(config).depth == 0


def test_browser_family_is_lazy_and_respects_disable_flag(tmp_path):
    enabled_tools = build_main_tools(runtime_config(tmp_path, enable_browser=True))
    enabled = {tool.name for tool in enabled_tools}
    disabled = {
        tool.name
        for tool in build_main_tools(runtime_config(tmp_path, enable_browser=False))
    }
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
                "spawn_agents",
                "await_agents",
                "agent_status",
                "cancel_agents",
                "senpai_training",
            },
        ),
        (
            "student",
            {
                "senpai_terminal",
                "senpai_github",
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
    assert "think" not in by_name
    assert "task_tool_set" not in by_name
    assert expected_custom <= set(by_name)
    assert by_name["senpai_terminal"].params == {"role": role}
    delegation_params = {
        "event_db_path": str(config.state_dir / f"{role}-events.sqlite3")
    }
    for name in (
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
    assert by_name["senpai_training"].params == {
        "state_dir": str(config.state_dir / "training"),
        "max_timeout_seconds": 1800,
    }


def test_supervisor_receives_isolated_terminal_and_campaign_operations(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("STUDENT_NAMES", "fern,frieren")
    monkeypatch.setenv("SENPAI_KUBECTL_NAMESPACE", "research")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setenv("ADVISOR_BRANCH", "maple-advisor")
    config = runtime_config(tmp_path, role="supervisor")

    tools = build_main_tools(config)

    assert [tool.name for tool in tools] == [
        "terminal",
        "senpai_operations",
    ]
    assert tools[0].params == {
        "socket_path": "@senpai-isolated-terminal",
        "wake_id": config.conversation_id.hex,
    }
    assert tools[1].params == {
        "state_dir": str(config.state_dir),
        "namespace": "research",
        "research_tag": "maple",
        "repo": "acme/widgets",
        "advisor_branch": "maple-advisor",
        "students": ("fern", "frieren"),
        "mutation_cooldown_seconds": 1800.0,
    }
    assert "senpai_terminal" not in {tool.name for tool in tools}


def test_supervisor_terminal_resolves_to_isolated_executor_and_allows_git_push(
    tmp_path,
    monkeypatch,
):
    remote = tmp_path / "remote.git"
    workspace = tmp_path / "workspace"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", str(workspace)], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.name", "Supervisor Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "user.email",
            "supervisor@example.com",
        ],
        check=True,
    )
    (workspace / "README.md").write_text("supervisor terminal\n")
    subprocess.run(["git", "-C", str(workspace), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-m", "initial"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "remote", "add", "origin", str(remote)],
        check=True,
    )
    monkeypatch.setenv("STUDENT_NAMES", "fern")
    monkeypatch.setenv("SENPAI_KUBECTL_NAMESPACE", "maple")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setenv("ADVISOR_BRANCH", "maple-advisor")
    monkeypatch.setattr(
        "senpai_agent.hooks.terminal_policy",
        lambda *_args, **_kwargs: pytest.fail("research terminal policy was called"),
    )
    from senpai_agent.isolated_terminal import (
        IsolatedTerminalClientExecutor,
        IsolatedTerminalServer,
        begin_isolated_terminal_wake,
    )

    register_default_tools(enable_browser=False)
    state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(workspace)),
        env_observation_persistence_dir=None,
    )
    socket_path = short_socket_path(tmp_path, "git-terminal")
    monkeypatch.setenv("SENPAI_SUPERVISOR_TERMINAL_SOCKET", str(socket_path))
    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
        environment={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        terminal_type="subprocess",
    ):
        config = runtime_config(tmp_path, role="supervisor")
        begin_isolated_terminal_wake(socket_path, config.conversation_id.hex)
        terminal = resolve_tool(
            build_main_tools(config)[0],
            state,
        )[0]
        assert isinstance(terminal.executor, IsolatedTerminalClientExecutor)
        observation = terminal.executor(
            TerminalAction(command="git push origin HEAD:main")
        )

    assert observation.metadata.exit_code == 0
    assert subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/main"],
        check=False,
    ).returncode == 0


@pytest.mark.skipif(
    not os.path.exists("/proc/self/environ"),
    reason="Linux procfs is required for the initial-environment regression",
)
def test_supervisor_secrets_are_absent_from_native_terminal_ancestor_environments(
    tmp_path,
):
    sentinels = {
        "GITHUB_TOKEN": "proc-github-sentinel",
        "GH_TOKEN": "proc-gh-sentinel",
        "WANDB_API_KEY": "proc-wandb-sentinel",
        "OPENAI_API_KEY": "proc-openai-sentinel",
        "ANTHROPIC_API_KEY": "proc-anthropic-sentinel",
    }
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    probe_path = tmp_path / "probe.py"
    probe_path.write_text(
        textwrap.dedent(
            """
            import os
            import shlex
            import sys
            import textwrap
            from pathlib import Path

            from openhands.tools.terminal import TerminalAction
            from openhands.tools.terminal.impl import TerminalExecutor
            from senpai_agent.secrets import consume_supervisor_secret_directory

            sentinels = (
                b"proc-github-sentinel",
                b"proc-gh-sentinel",
                b"proc-wandb-sentinel",
                b"proc-openai-sentinel",
                b"proc-anthropic-sentinel",
            )
            private = consume_supervisor_secret_directory(os.environ, required=True)
            assert private["GITHUB_TOKEN"] == "proc-github-sentinel"
            assert private["WANDB_API_KEY"] == "proc-wandb-sentinel"
            assert private["OPENAI_API_KEY"] == "proc-openai-sentinel"
            assert private["ANTHROPIC_API_KEY"] == "proc-anthropic-sentinel"
            if any(value in Path("/proc/self/environ").read_bytes() for value in sentinels):
                raise SystemExit(17)

            child_probe = textwrap.dedent('''
            import os
            from pathlib import Path

            sentinels = (
                b"proc-github-sentinel",
                b"proc-gh-sentinel",
                b"proc-wandb-sentinel",
                b"proc-openai-sentinel",
                b"proc-anthropic-sentinel",
            )
            pid = os.getppid()
            leaked = False
            while pid > 1:
                try:
                    environ = Path(f"/proc/{pid}/environ").read_bytes()
                    leaked = leaked or any(value in environ for value in sentinels)
                    status = Path(f"/proc/{pid}/status").read_text()
                except (OSError, PermissionError):
                    break
                parent = next(
                    line for line in status.splitlines() if line.startswith("PPid:")
                )
                pid = int(parent.split()[1])
            raise SystemExit(19 if leaked else 0)
            ''')
            terminal = TerminalExecutor(
                working_dir=os.environ["PROBE_WORKSPACE"],
                terminal_type="subprocess",
            )
            try:
                observation = terminal(
                    TerminalAction(
                        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(child_probe)}"
                    )
                )
            finally:
                terminal.close()
            raise SystemExit(observation.metadata.exit_code or 0)
            """
        )
    )
    home = tmp_path / "home"
    home.mkdir()
    environment = {
        "HOME": str(home),
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(REPO_ROOT),
        "PROBE_WORKSPACE": str(workspace),
        "OPENHANDS_SUPPRESS_BANNER": "1",
        **sentinels,
    }
    helper = REPO_ROOT / "k8s" / "handoff-operational-supervisor-secrets.sh"
    command = (
        f"source {shlex.quote(str(helper))}; "
        "handoff_operational_supervisor_secrets; "
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(probe_path))}"
    )

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert all(value not in result.stdout for value in sentinels.values())
    assert all(value not in result.stderr for value in sentinels.values())


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

    assert (
        "spawn_agents"
        in depth_aware_child_definition(general, child=False, depth=2).tools
    )
    assert (
        "spawn_agents"
        in depth_aware_child_definition(general, child=True, depth=1).tools
    )
    assert (
        "spawn_agents"
        not in depth_aware_child_definition(general, child=True, depth=2).tools
    )
    assert (
        "spawn_agents"
        not in depth_aware_child_definition(explore, child=True, depth=1).tools
    )


def test_senpai_task_tracker_description_is_concise_and_parallel_safe(tmp_path):
    state = type("State", (), {"persistence_dir": str(tmp_path)})()
    tool = SenpaiTaskTrackerTool.create(state)[0]
    description = " ".join(tool.description.split())

    assert "persisted task list as working memory across turns" in description
    assert "delegated agents" in description
    assert "long-running jobs" in description
    assert "multiple items" in description
    assert "For straightforward work, proceed directly" in description
    assert "Limit active work to ONE" not in description
    assert len(tool.description) < 700


def test_think_is_absent_from_every_root_and_child_tool_surface(tmp_path):
    for role in ("advisor", "student"):
        assert "think" not in {
            tool.name for tool in build_main_tools(runtime_config(tmp_path, role=role))
        }
    for filename in ("general-purpose.md", "explore.md", "search.md", "bash-runner.md"):
        assert "think" not in AgentDefinition.load(AGENT_DIR / filename).tools
    agent = without_legacy_think(
        Agent(
            llm=LLM(
                model="openai/gpt-4o-mini",
                api_key=SecretStr("test-key"),
            ),
            tools=[],
        )
    )
    assert "ThinkTool" not in agent.include_default_tools


def test_operational_guidance_uses_only_the_public_job_tool_names():
    readme = REPO_ROOT / "README.md"
    operational_files = [readme, REPO_ROOT / "SPEC.md"]
    operational_files.extend((REPO_ROOT / "system_instructions").glob("*.md"))
    operational_files.extend(
        (REPO_ROOT / "plugins" / "senpai" / "skills").glob("**/*.md")
    )
    operational_files.extend((REPO_ROOT / ".agents" / "agents").glob("*.md"))

    stale_names = {
        "run_training",
        "get_training_status",
        "monitor_training",
        "cancel_training",
    }
    findings = {
        str(path.relative_to(REPO_ROOT)): sorted(
            name for name in stale_names if name in path.read_text()
        )
        for path in operational_files
    }

    assert readme in operational_files
    assert not {path: names for path, names in findings.items() if names}


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
    instructions = (REPO_ROOT / "system_instructions" / "ADVISOR.md").read_text(
        encoding="utf-8"
    )

    assert "Assigning high-value work to idle students" not in instructions
    assert instructions.index("Research and synthesis needed") < instructions.index(
        "Well-founded experiment assignments"
    )
    assert "Idleness is not a reason to skip" in instructions


def test_harness_states_bounded_delegation_tree_contract():
    instructions = (REPO_ROOT / "system_instructions" / "SENPAI-HARNESS.md").read_text(
        encoding="utf-8"
    )
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
