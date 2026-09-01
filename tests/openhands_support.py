import uuid
from base64 import b64encode
from pathlib import Path

from git_workflow_support import commit_workspace
from pydantic import SecretStr

from senpai_agent.launch_context import LAUNCH_CONTEXT_ENV
from senpai_agent.openhands_runner import RunnerConfig
from senpai_agent.program_context import (
    PROGRAM_PATH_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    ProgramSystemPrompt,
    load_program_system_prompt,
)
from senpai_agent.system_instructions import (
    SYSTEM_INSTRUCTIONS_FILE_ENV,
    SYSTEM_INSTRUCTIONS_SHA256_ENV,
    SenpaiSystemInstructions,
    encode_system_instructions,
)

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
        "conversation_secrets": {"PRIVATE_AUTH": "private-key"},
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
        "compaction_trigger_tokens": 200_000,
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
                source_commit="a" * 40,
                content="Test programme.",
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
    program_source_commit: str | None = None,
    launch_context: str = TEST_LAUNCH_CONTEXT,
) -> dict[str, str]:
    workspace = tmp_path / "target"
    workspace.mkdir(exist_ok=True)
    program = workspace / program_path
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(program_content, encoding="utf-8")
    commit_workspace(workspace)
    program_snapshot = load_program_system_prompt(workspace, program_path)
    if program_source_commit is not None:
        program_snapshot = ProgramSystemPrompt(
            program_path=program_snapshot.program_path,
            source_commit=program_source_commit,
            content=program_snapshot.content,
        )
    role_file = tmp_path / f"SENPAI-{role.upper()}.md"
    role_file.write_text(f"{role} role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    instructions = SenpaiSystemInstructions(
        harness="harness instructions",
        role=f"{role} role",
        program=program_snapshot,
        launch=launch_context,
    )
    system_context = tmp_path / f"{role}-system-instructions.b64"
    system_context.write_text(
        f"{encode_system_instructions(instructions)}\n",
        encoding="utf-8",
    )
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
        PROGRAM_PATH_ENV: program_snapshot.program_path,
        PROGRAM_SOURCE_COMMIT_ENV: program_snapshot.source_commit,
        SYSTEM_INSTRUCTIONS_FILE_ENV: str(system_context),
        SYSTEM_INSTRUCTIONS_SHA256_ENV: instructions.content_sha256,
        LAUNCH_CONTEXT_ENV: b64encode(launch_context.encode()).decode(),
    }


def isolate_agent_discovery(monkeypatch, runner) -> None:
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "sanitized_agent_definitions", lambda _: [])
    monkeypatch.setattr(runner, "sanitized_project_skills", lambda _: [])
