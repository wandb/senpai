import uuid
from base64 import b64encode
from pathlib import Path

from pydantic import SecretStr

from senpai_agent.openhands_runner import RunnerConfig
from senpai_agent.launch_context import LAUNCH_CONTEXT_ENV
from senpai_agent.program_context import ProgramSystemPrompt
from senpai_agent.system_instructions import SenpaiSystemInstructions

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugins" / "senpai"
AGENT_DIR = REPO_ROOT / ".agents" / "agents"
TEST_LAUNCH_CONTEXT = "# Authoritative launch context\n\nTest launch policy."


def runtime_config(tmp_path: Path, **updates) -> RunnerConfig:
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    role_file = tmp_path / "ADVISOR.md"
    role_file.write_text("advisor role", encoding="utf-8")
    values = {
        "max_turns": 1,
        "model": "anthropic/claude-opus-4-8",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key": SecretStr("test-key"),
        "github_repo": "acme/widgets",
        "github_token": SecretStr("github-key"),
        "github_trusted_actor": None,
        "command_secrets": {"WANDB_API_KEY": "wandb-key"},
        "reasoning_effort": "xhigh",
        "smart_model": "anthropic/claude-opus-4-8",
        "smart_api_key_env": "ANTHROPIC_API_KEY",
        "smart_api_key": SecretStr("test-key"),
        "smart_reasoning_effort": "xhigh",
        "fast_model": "anthropic/claude-haiku-4-5",
        "fast_api_key_env": "ANTHROPIC_API_KEY",
        "fast_api_key": SecretStr("test-key"),
        "fast_reasoning_effort": "low",
        "frontier_model": "openai/gpt-5.6-sol",
        "frontier_api_key_env": "OPENAI_API_KEY",
        "frontier_api_key": SecretStr("frontier-key"),
        "frontier_reasoning_effort": "max",
        "workspace": tmp_path,
        "state_dir": tmp_path / "state",
        "conversation_id": uuid.uuid4(),
        "role": "advisor",
        "enable_browser": False,
        "agent_name": None,
        "harness_file": harness_file,
        "role_file": role_file,
        "plugin_dir": PLUGIN_DIR,
        "instructions": SenpaiSystemInstructions(
            harness="harness instructions",
            role="advisor role",
            program=ProgramSystemPrompt(
                program_path="program.md",
                prompt="# program.md - program.md\n\nTest programme.",
            ),
            launch=TEST_LAUNCH_CONTEXT,
        ),
    }
    values.update(updates)
    return RunnerConfig(**values)


def runtime_env(
    tmp_path: Path,
    *,
    role: str = "advisor",
    program_path: str = "program.md",
    program_content: str = "# Test programme\n\nUse the target contract.\n",
) -> dict[str, str]:
    workspace = tmp_path / "target"
    workspace.mkdir(exist_ok=True)
    program = workspace / program_path
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(program_content, encoding="utf-8")
    role_file = tmp_path / f"SENPAI-{role.upper()}.md"
    role_file.write_text(f"{role} role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    return {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "OPENAI_API_KEY": "openai-key",
        "GITHUB_TOKEN": "github-key",
        "GH_REPO": "acme/widgets",
        "SENPAI_ROLE": role,
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path / "state"),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
        LAUNCH_CONTEXT_ENV: b64encode(TEST_LAUNCH_CONTEXT.encode()).decode(),
    }


def isolate_agent_discovery(monkeypatch, runner) -> None:
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "sanitized_agent_definitions", lambda _: [])
    monkeypatch.setattr(runner, "sanitized_project_skills", lambda _: [])
