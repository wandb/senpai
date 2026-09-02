import os
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.openhands_runner as runner
from senpai_agent.openhands_runner import (
    build_main_agent_context,
    find_role_file,
    parse_runner_args,
    read_instruction_file,
    resolve_agent_skills,
    resolve_config,
    sanitized_agent_definitions,
    sanitized_project_skills,
    scrub_model_credentials,
    without_eager_skill_discovery,
)
from senpai_agent.program_context import ProgramSystemPrompt
from senpai_agent.secrets import CUSTOM_SECRET_ENV_NAMES_ENV
from senpai_agent.system_instructions import SenpaiSystemInstructions
from openhands_support import TEST_LAUNCH_CONTEXT, runtime_config, runtime_env
from test_agent_markdown import HTML_HEADER, PLAIN_HEADER

ROOT = Path(__file__).resolve().parents[1]


def test_browser_is_enabled_by_default_and_can_be_disabled():
    default_args = parse_runner_args(["--max-turns", "1"])
    disabled_args = parse_runner_args(["--max-turns", "1", "--no-browser"])

    assert default_args.enable_browser is True
    assert disabled_args.enable_browser is False


def test_explicit_role_file_is_loaded(tmp_path: Path):
    role_file = tmp_path / "STUDENT.md"
    role_file.write_text(HTML_HEADER + "student role", encoding="utf-8")

    selected = find_role_file(str(role_file))

    assert selected == role_file
    assert read_instruction_file(selected) == "student role"


@pytest.mark.parametrize("explicit", [None, "missing.md"])
def test_role_file_must_be_explicit_and_exist(tmp_path: Path, explicit: str | None):
    path = None if explicit is None else str(tmp_path / explicit)

    with pytest.raises(RuntimeError, match="role instructions|required|does not exist"):
        find_role_file(path)


def test_main_agent_context_appends_program_after_harness_and_role():
    context = build_main_agent_context(
        SenpaiSystemInstructions(
            harness="harness instructions",
            role="advisor role",
            program=ProgramSystemPrompt(
                program_path="senpai/program.md",
                prompt="# program.md - senpai/program.md\n\nResearch policy.",
            ),
            launch="# Authoritative launch context\n\nRuntime policy.",
        ),
    )

    assert context.system_message_suffix == (
        "# Senpai harness\n\nharness instructions\n\n"
        "# Senpai role\n\nadvisor role\n\n"
        "# program.md - senpai/program.md\n\nResearch policy.\n\n"
        "# Authoritative launch context\n\nRuntime policy.\n"
    )
    assert context.current_datetime is None
    assert context.load_user_skills is False
    assert context.load_project_skills is False


def test_student_charter_requires_typed_workflow_and_training_tools():
    instructions = (ROOT / "system_instructions" / "STUDENT.md").read_text()
    submission_skill = (
        ROOT
        / "plugins"
        / "senpai"
        / "skills"
        / "submit-experiment-results"
        / "SKILL.md"
    ).read_text()

    assert "Use `post_assignment_comment`" in instructions
    assert "When `post_assignment_comment` is present" not in instructions
    assert "ask the advisor a meaningful interim question" in instructions
    assert "fresh `comment_id`" not in instructions
    assert "fresh `comment_id`" in submission_skill
    assert "Keep the PR concise" in submission_skill
    assert "Use `submit_experiment_result` for the terminal result" in instructions
    assert "must use `run_training`" in instructions
    assert "Never launch training through the terminal" in instructions
    assert "`monitor_training`" in instructions
    assert "`get_training_status`" in instructions
    assert "`cancel_training`" in instructions


def test_advisor_charter_explains_student_feedback_without_tool_protocol():
    instructions = (ROOT / "system_instructions" / "ADVISOR.md").read_text()

    assert "Treat student questions and interim feedback as current evidence" in instructions
    assert "refresh the complete experiment context" in instructions
    assert "distinguish a clarification or hold from a revised experiment" in instructions
    for operational_detail in (
        "student_assignment_comment",
        "get_prs",
        "send_assignment_feedback",
    ):
        assert operational_detail not in instructions

    harness = (ROOT / "system_instructions" / "SENPAI-HARNESS.md").read_text()
    assert "A `student_assignment_comment` event is interim feedback" in harness
    assert "may refer to an earlier assignment revision" in harness
    assert "respond on the current revision" in harness


def test_project_instruction_files_are_not_loaded_but_explicit_skills_are(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    agents = workspace / ".agents" / "agents"
    skill_dir = workspace / ".agents" / "skills" / "agents"
    agents.mkdir(parents=True)
    skill_dir.mkdir(parents=True)
    instructions = (
        workspace / "AGENTS.md",
        workspace / "CLAUDE.md",
        workspace / "nested" / "AGENT.md",
        workspace / ".agents" / "skills" / "AGENTS.md",
        workspace / ".openhands" / "skills" / "CLAUDE.md",
    )
    definition = agents / "review.md"
    for path in instructions:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HTML_HEADER + "# Project rules\n", encoding="utf-8")
    (workspace / ".agents" / "skills" / "linked.md").symlink_to(
        workspace / "AGENTS.md"
    )
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: agents\ndescription: Review code.\n---\n\n"
        + PLAIN_HEADER
        + "Review carefully.\n",
        encoding="utf-8",
    )
    definition.write_text(
        "---\nname: review\ndescription: Review code.\nskills:\n  - agents\n---\n\n"
        + PLAIN_HEADER
        + "Review carefully.\n",
        encoding="utf-8",
    )

    skills = sanitized_project_skills(workspace)
    definitions = sanitized_agent_definitions(workspace)

    assert [skill.name for skill in skills] == ["agents"]
    assert skills[0].source == str(skill_file)
    assert "SPDX-" not in skills[0].content
    assert "Review carefully." in skills[0].content
    assert "Project rules" not in skills[0].content
    review = next(item for item in definitions if item.name == "review")
    assert "SPDX-" not in review.system_prompt
    assert review.skills == ["agents"]
    assert resolve_agent_skills(review, skills) == [skills[0]]
    assert without_eager_skill_discovery(review).skills == []
    assert all(
        path.read_text(encoding="utf-8").startswith("<!--\nSPDX-")
        for path in instructions
    )
    assert "# SPDX-" in skill_file.read_text(encoding="utf-8")
    assert "# SPDX-" in definition.read_text(encoding="utf-8")


def test_developer_only_project_skills_are_not_exposed_to_senpai(tmp_path: Path):
    workspace = tmp_path / "target"
    skill_dir = workspace / ".agents" / "skills" / "telemetry"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: telemetry\ndescription: Developer diagnostics.\n---\n",
        encoding="utf-8",
    )
    (skill_dir / ".senpai-developer-only").touch()

    assert "telemetry" not in {
        skill.name for skill in sanitized_project_skills(workspace)
    }


def test_resolved_config_separates_runtime_credentials_from_conversation_secrets(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env.update(
        {
            "GH_TOKEN": "secondary-github-key",
            "WANDB_API_KEY": "wandb-key",
            "EXA_API_KEY": "exa-key",
            CUSTOM_SECRET_ENV_NAMES_ENV: "PRIVATE_AUTH",
            "PRIVATE_AUTH": "private-key",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.api_key.get_secret_value() == "anthropic-key"
    assert config.smart_api_key.get_secret_value() == "anthropic-key"
    assert config.fast_api_key.get_secret_value() == "anthropic-key"
    assert config.frontier_api_key.get_secret_value() == "anthropic-key"
    assert config.github_token.get_secret_value() == "github-key"
    assert config.conversation_secrets == {
        "WANDB_API_KEY": "wandb-key",
        "EXA_API_KEY": "exa-key",
        "PRIVATE_AUTH": "private-key",
    }
    assert "ANTHROPIC_API_KEY" not in config.conversation_secrets
    assert "OPENAI_API_KEY" not in config.conversation_secrets
    assert config.timeout_seconds == 7200
    assert config.llm_timeout_seconds == 5400
    assert config.llm_num_retries == 5
    assert config.compaction_trigger_tokens == 200_000

    delegated = runner.delegation_config(config)
    assert delegated.smart_api_key == "anthropic-key"
    assert delegated.fast_api_key == "anthropic-key"
    assert delegated.frontier_api_key == "anthropic-key"


def test_training_limits_are_not_read_from_environment(tmp_path: Path):
    env = runtime_env(tmp_path)
    env["SENPAI_TIMEOUT_MINUTES"] = "not-a-number"
    env["SENPAI_MAX_EPOCHS"] = "not-an-integer"

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert not hasattr(config, "training_max_timeout_seconds")


def test_configured_custom_secret_requires_a_nonblank_value(tmp_path: Path):
    env = runtime_env(tmp_path)
    env[CUSTOM_SECRET_ENV_NAMES_ENV] = "PRIVATE_AUTH"
    env["PRIVATE_AUTH"] = "  "

    with pytest.raises(RuntimeError, match="custom secret PRIVATE_AUTH is required"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_resolved_config_discovers_one_level_program_from_target_workspace(
    tmp_path: Path,
):
    env = runtime_env(
        tmp_path,
        program_path="senpai/program.md",
        program_content="# Mission\n\nImprove the model.\n",
    )
    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.instructions.program.program_path == "senpai/program.md"
    assert config.instructions.program.prompt == (
        "# program.md - senpai/program.md\n\n"
        "# Mission\n\nImprove the model."
    )
    assert config.instructions.launch == TEST_LAUNCH_CONTEXT
    assert config.instructions.prompt == (
        "# Senpai harness\n\nharness instructions\n\n"
        "# Senpai role\n\nadvisor role\n\n"
        "# program.md - senpai/program.md\n\n"
        "# Mission\n\nImprove the model.\n\n"
        f"{TEST_LAUNCH_CONTEXT}\n"
    )
    delegated = runner.delegation_config(config)
    assert delegated.program_path == config.instructions.program.program_path


def test_resolved_system_instructions_do_not_change_with_source_files(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)
    prompt = config.instructions.prompt

    Path(env["SENPAI_OPENHANDS_HARNESS_FILE"]).write_text("changed harness")
    Path(env["SENPAI_OPENHANDS_ROLE_FILE"]).write_text("changed role")
    (Path(env["SENPAI_OPENHANDS_WORKSPACE"]) / "program.md").write_text(
        "changed program"
    )
    env["SENPAI_LAUNCH_CONTEXT_B64"] = "Y2hhbmdlZCBsYXVuY2g="

    assert config.instructions.prompt == prompt


@pytest.mark.parametrize("encoded", [None, "", "not base64", "8A==", "IA=="])
def test_launch_context_must_be_present_valid_utf8_and_nonempty(
    tmp_path: Path,
    encoded: str | None,
):
    env = runtime_env(tmp_path)
    if encoded is None:
        env.pop("SENPAI_LAUNCH_CONTEXT_B64")
    else:
        env["SENPAI_LAUNCH_CONTEXT_B64"] = encoded

    with pytest.raises(ValueError, match="SENPAI_LAUNCH_CONTEXT_B64"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"SENPAI_LLM_TIMEOUT_SECONDS": "zero"}, "must be numeric"),
        ({"SENPAI_LLM_NUM_RETRIES": "0"}, "must be positive"),
    ],
)
def test_runtime_stall_bounds_are_validated(tmp_path, updates, message):
    env = runtime_env(tmp_path)
    env.update(updates)

    with pytest.raises(RuntimeError, match=message):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_inbox_recovery_budget_is_explicit_and_configurable(tmp_path):
    default = resolve_config(
        parse_runner_args(["--max-turns", "1"]),
        runtime_env(tmp_path),
    )
    env = runtime_env(tmp_path)
    env.update(
        {
            "SENPAI_INBOX_MAX_STALLED_ATTEMPTS": "4",
            "SENPAI_INBOX_MAX_RECOVERY_GENERATIONS": "2",
        }
    )

    configured = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert default.inbox_max_stalled_attempts == 3
    assert default.inbox_max_recovery_generations == 1
    assert configured.inbox_max_stalled_attempts == 4
    assert configured.inbox_max_recovery_generations == 2


def test_compaction_trigger_tokens_are_explicit_and_configurable(tmp_path):
    default = resolve_config(
        parse_runner_args(["--max-turns", "1"]),
        runtime_env(tmp_path),
    )
    configured = resolve_config(
        parse_runner_args(
            ["--max-turns", "1", "--compaction-trigger-tokens", "180000"]
        ),
        runtime_env(tmp_path),
    )

    assert default.compaction_trigger_tokens == 200_000
    assert configured.compaction_trigger_tokens == 180_000


@pytest.mark.parametrize(
    ("value", "message"),
    [("not-a-number", "must be an integer"), ("49999", "at least 50000")],
)
def test_compaction_trigger_tokens_reject_invalid_environment_values(
    tmp_path,
    value,
    message,
):
    env = runtime_env(tmp_path)
    env["SENPAI_COMPACTION_TRIGGER_TOKENS"] = value

    with pytest.raises(RuntimeError, match=message):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SENPAI_INBOX_MAX_STALLED_ATTEMPTS", "0"),
        ("SENPAI_INBOX_MAX_RECOVERY_GENERATIONS", "-1"),
    ],
)
def test_inbox_recovery_budget_rejects_invalid_values(tmp_path, key, value):
    env = runtime_env(tmp_path)
    env[key] = value

    with pytest.raises(RuntimeError, match="inbox recovery budget"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


@pytest.mark.parametrize(
    ("role", "main_reasoning_effort"),
    [("advisor", "high"), ("student", "medium")],
)
def test_default_model_profiles_are_explicit_and_provider_credentials_are_inferred(
    tmp_path: Path,
    role: str,
    main_reasoning_effort: str,
):
    config = resolve_config(
        parse_runner_args(["--max-turns", "1"]),
        runtime_env(tmp_path, role=role),
    )

    assert (
        config.model,
        config.api_key_env,
        config.reasoning_effort,
    ) == (
        "anthropic/claude-fable-5-1",
        "ANTHROPIC_API_KEY",
        main_reasoning_effort,
    )
    assert (
        config.smart_model,
        config.smart_api_key_env,
        config.smart_reasoning_effort,
    ) == ("anthropic/claude-fable-5-1", "ANTHROPIC_API_KEY", "high")
    assert (
        config.fast_model,
        config.fast_api_key_env,
        config.fast_reasoning_effort,
    ) == ("anthropic/claude-sonnet-5", "ANTHROPIC_API_KEY", "medium")
    assert (
        config.frontier_model,
        config.frontier_api_key_env,
        config.frontier_reasoning_effort,
    ) == ("anthropic/claude-fable-5-1", "ANTHROPIC_API_KEY", "max")


def test_ultra_environment_value_is_rejected(tmp_path: Path):
    env = runtime_env(tmp_path)
    env["SENPAI_OPENHANDS_REASONING_EFFORT"] = "ultra"

    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_wandb_gateway_configuration_is_explicit_and_uses_max_glm_reasoning(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env.update(
        {
            "WANDB_API_KEY": "wandb-key",
            "WANDB_ENTITY": "research-team",
            "WANDB_PROJECT": "mlxfast",
            "SENPAI_OPENHANDS_MODEL": "wandb/zai-org/GLM-5.2",
            "SENPAI_OPENHANDS_REASONING_EFFORT": "max",
            "SENPAI_OPENHANDS_SMART_MODEL": "wandb/zai-org/GLM-5.2",
            "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": "max",
            "SENPAI_OPENHANDS_FAST_MODEL": "wandb/zai-org/GLM-5.2",
            "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": "max",
            "SENPAI_OPENHANDS_FRONTIER_MODEL": "wandb/zai-org/GLM-5.2",
            "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": "max",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.wandb_entity == "research-team"
    assert config.wandb_project == "mlxfast"
    assert config.model == config.smart_model == config.fast_model
    assert config.model == config.frontier_model == "wandb/zai-org/GLM-5.2"
    assert config.api_key_env == "WANDB_API_KEY"
    assert config.api_key.get_secret_value() == "wandb-key"
    assert config.reasoning_effort == "max"
    assert config.smart_reasoning_effort == "max"
    assert config.fast_reasoning_effort == "max"
    assert config.frontier_reasoning_effort == "max"


def test_fast_model_inherits_an_openai_main_profile(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "SENPAI_OPENHANDS_MODEL": "openai/gpt-5.6",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.smart_model == "openai/gpt-5.6"
    assert config.fast_model == "openai/gpt-5.6"
    assert config.fast_reasoning_effort == "high"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.smart_api_key_env == "OPENAI_API_KEY"
    assert config.fast_api_key_env == "OPENAI_API_KEY"


def test_fast_model_uses_sonnet_for_an_anthropic_main_profile(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "SENPAI_OPENHANDS_MODEL": "anthropic/claude-opus-4-8",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.smart_model == "anthropic/claude-opus-4-8"
    assert config.fast_model == "anthropic/claude-sonnet-5"
    assert config.fast_reasoning_effort == "medium"
    assert config.smart_api_key_env == "ANTHROPIC_API_KEY"
    assert config.fast_api_key_env == "ANTHROPIC_API_KEY"


def test_fast_profile_inherits_smart_effort_for_a_wandb_main_override(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env.update(
        {
            "WANDB_API_KEY": "wandb-key",
            "WANDB_ENTITY": "research-team",
            "WANDB_PROJECT": "mlxfast",
            "SENPAI_OPENHANDS_MODEL": "wandb/zai-org/GLM-5.2",
            "SENPAI_OPENHANDS_REASONING_EFFORT": "max",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert (config.smart_model, config.smart_reasoning_effort) == (
        "wandb/zai-org/GLM-5.2",
        "high",
    )
    assert (config.fast_model, config.fast_reasoning_effort) == (
        "wandb/zai-org/GLM-5.2",
        "high",
    )


def test_all_model_profiles_accept_independent_cli_model_and_effort_settings(
    tmp_path: Path,
):
    args = parse_runner_args(
        [
            "--max-turns",
            "1",
            "--model",
            "anthropic/main",
            "--reasoning-effort",
            "medium",
            "--smart-model",
            "anthropic/smart",
            "--smart-reasoning-effort",
            "high",
            "--fast-model",
            "anthropic/fast",
            "--fast-reasoning-effort",
            "none",
            "--frontier-model",
            "openai/gpt-5.6-sol",
            "--frontier-reasoning-effort",
            "max",
        ]
    )

    config = resolve_config(args, runtime_env(tmp_path))

    assert (config.model, config.reasoning_effort) == ("anthropic/main", "medium")
    assert (config.smart_model, config.smart_reasoning_effort) == (
        "anthropic/smart",
        "high",
    )
    assert (config.fast_model, config.fast_reasoning_effort) == (
        "anthropic/fast",
        "none",
    )
    assert (config.frontier_model, config.frontier_reasoning_effort) == (
        "openai/gpt-5.6-sol",
        "max",
    )


def test_anthropic_max_is_accepted_across_model_profiles(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "SENPAI_OPENHANDS_MODEL": "anthropic/claude-fable-5-1",
            "SENPAI_OPENHANDS_REASONING_EFFORT": "max",
            "SENPAI_OPENHANDS_SMART_MODEL": "anthropic/claude-opus-5",
            "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": "max",
            "SENPAI_OPENHANDS_FAST_MODEL": "anthropic/claude-sonnet-5",
            "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": "max",
            "SENPAI_OPENHANDS_FRONTIER_MODEL": "anthropic/claude-fable-5-1",
            "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": "max",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert (config.model, config.reasoning_effort) == (
        "anthropic/claude-fable-5-1",
        "max",
    )
    assert (config.smart_model, config.smart_reasoning_effort) == (
        "anthropic/claude-opus-5",
        "max",
    )
    assert (config.fast_model, config.fast_reasoning_effort) == (
        "anthropic/claude-sonnet-5",
        "max",
    )
    assert (config.frontier_model, config.frontier_reasoning_effort) == (
        "anthropic/claude-fable-5-1",
        "max",
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"SENPAI_OPENHANDS_REASONING_EFFORT": "extreme"}, "unsupported"),
        (
            {
                "SENPAI_OPENHANDS_MODEL": "anthropic/claude-opus-4-8",
                "SENPAI_OPENHANDS_REASONING_EFFORT": "ultra",
            },
            "unsupported",
        ),
    ],
)
def test_invalid_model_profile_effort_fails_configuration(
    tmp_path: Path,
    updates: dict[str, str],
    message: str,
):
    env = runtime_env(tmp_path)
    env.update(updates)

    with pytest.raises(ValueError, match=message):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_explicit_api_key_env_preserves_custom_provider_support(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "CUSTOM_API_KEY": "custom-key",
            "SENPAI_OPENHANDS_MODEL": "custom/main",
            "SENPAI_OPENHANDS_API_KEY_ENV": "CUSTOM_API_KEY",
            "SENPAI_OPENHANDS_SMART_MODEL": "custom/smart",
            "SENPAI_OPENHANDS_FAST_MODEL": "custom/fast",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.api_key_env == "CUSTOM_API_KEY"
    assert config.smart_api_key_env == "CUSTOM_API_KEY"
    assert config.fast_api_key_env == "CUSTOM_API_KEY"
    assert config.api_key.get_secret_value() == "custom-key"
    assert config.smart_api_key.get_secret_value() == "custom-key"
    assert config.fast_api_key.get_secret_value() == "custom-key"


def test_custom_model_credential_cannot_also_be_a_custom_secret(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "PRIVATE_AUTH": "custom-key",
            CUSTOM_SECRET_ENV_NAMES_ENV: "PRIVATE_AUTH",
            "SENPAI_OPENHANDS_MODEL": "custom/main",
            "SENPAI_OPENHANDS_API_KEY_ENV": "PRIVATE_AUTH",
            "SENPAI_OPENHANDS_SMART_MODEL": "custom/smart",
            "SENPAI_OPENHANDS_FAST_MODEL": "custom/fast",
        }
    )

    with pytest.raises(RuntimeError, match="cannot also be custom secrets"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_custom_main_provider_requires_the_existing_api_key_env_override(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env["SENPAI_OPENHANDS_MODEL"] = "custom/main"

    with pytest.raises(ValueError, match="cannot infer.*custom") as raised:
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert "SENPAI_OPENHANDS_API_KEY_ENV" in str(raised.value)


def test_cross_provider_profile_requires_an_inferable_or_explicit_api_key_env(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env["SENPAI_OPENHANDS_SMART_MODEL"] = "custom/smart"

    with pytest.raises(ValueError, match="cannot infer.*custom") as raised:
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)
    assert "SENPAI_OPENHANDS_SMART_API_KEY_ENV" in str(raised.value)

    env.update(
        {
            "CUSTOM_API_KEY": "custom-key",
            "SENPAI_OPENHANDS_SMART_API_KEY_ENV": "CUSTOM_API_KEY",
        }
    )
    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)
    assert config.smart_api_key.get_secret_value() == "custom-key"


def test_model_credentials_are_removed_from_the_agent_environment(tmp_path: Path):
    environment = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "OPENAI_API_KEY": "openai-key",
        "WANDB_API_KEY": "wandb-key",
    }

    scrub_model_credentials(environment, runtime_config(tmp_path))

    assert environment == {"WANDB_API_KEY": "wandb-key"}


def test_config_consumes_a_private_one_use_github_token_file(tmp_path: Path):
    env = runtime_env(tmp_path)
    token_file = tmp_path / "github-token"
    token_file.write_text("one-use-token", encoding="utf-8")
    token_file.chmod(0o600)
    env.pop("GITHUB_TOKEN")
    env["SENPAI_GITHUB_TOKEN_FILE"] = str(token_file)

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.github_token.get_secret_value() == "one-use-token"
    assert not token_file.exists()


def test_github_token_rejects_a_non_private_file(tmp_path: Path):
    token_file = tmp_path / "github-token"
    token_file.write_text("exposed-token", encoding="utf-8")
    token_file.chmod(0o644)

    with pytest.raises(RuntimeError, match="private regular file"):
        runner.github_token({"SENPAI_GITHUB_TOKEN_FILE": str(token_file)})

    assert token_file.exists()


def test_github_token_can_be_consumed_from_an_inherited_pipe():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"pipe-token")
    finally:
        os.close(write_fd)

    assert runner.github_token({"SENPAI_GITHUB_TOKEN_FD": str(read_fd)}) == SecretStr(
        "pipe-token"
    )


def test_github_token_ignores_blank_ambient_values():
    assert runner.github_token(
        {"GITHUB_TOKEN": " \n", "GH_TOKEN": "fallback-token"}
    ) == SecretStr("fallback-token")

    with pytest.raises(RuntimeError, match="GITHUB_TOKEN or GH_TOKEN is required"):
        runner.github_token({"GITHUB_TOKEN": " \n"})


def test_child_config_requires_no_github_credential(tmp_path: Path):
    env = runtime_env(tmp_path, role="student")
    env.pop("GITHUB_TOKEN")

    config = resolve_config(
        parse_runner_args(["--max-turns", "1", "--child", "--agent", "explore"]),
        env,
    )

    assert config.child is True
    assert config.github_token is None


def test_advisor_config_reuses_its_durable_conversation_id(tmp_path: Path):
    env = runtime_env(tmp_path)

    first = resolve_config(parse_runner_args(["--max-turns", "1"]), env)
    second = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert first.conversation_id == second.conversation_id


@pytest.mark.parametrize("state_location", [None, "inside-workspace"])
def test_state_directory_is_explicit_and_outside_the_target_checkout(
    tmp_path: Path,
    state_location: str | None,
):
    env = runtime_env(tmp_path)
    if state_location is None:
        env.pop("SENPAI_OPENHANDS_STATE_DIR")
        message = "state directory is required"
    else:
        workspace = Path(env["SENPAI_OPENHANDS_WORKSPACE"])
        env["SENPAI_OPENHANDS_STATE_DIR"] = str(workspace / ".senpai" / "state")
        message = "outside the target workspace"

    with pytest.raises(RuntimeError, match=message):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)
