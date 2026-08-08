import os
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.openhands_runner as runner
from senpai_agent.openhands_runner import (
    build_main_agent_context,
    find_role_file,
    parse_runner_args,
    read_role_instructions,
    resolve_config,
    sanitized_agent_definitions,
    sanitized_project_skills,
    scrub_model_credentials,
)
from openhands_support import runtime_config, runtime_env
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
    assert read_role_instructions(selected) == "student role"


@pytest.mark.parametrize("explicit", [None, "missing.md"])
def test_role_file_must_be_explicit_and_exist(tmp_path: Path, explicit: str | None):
    path = None if explicit is None else str(tmp_path / explicit)

    with pytest.raises(RuntimeError, match="role instructions|required|does not exist"):
        find_role_file(path)


def test_main_agent_context_places_harness_and_role_before_project_skills():
    context = build_main_agent_context("harness instructions", "advisor role")

    assert context.system_message_suffix == (
        "# Senpai harness\n\nharness instructions\n\n"
        "# Senpai role\n\nadvisor role\n"
    )
    assert context.current_datetime is None
    assert context.load_user_skills is True
    assert context.load_project_skills is False


def test_student_charter_requires_typed_tools_for_every_training_operation():
    instructions = (ROOT / "system_instructions" / "STUDENT.md").read_text()

    assert "must use `run_training`" in instructions
    assert "Never launch training through the terminal" in instructions
    assert "`monitor_training`" in instructions
    assert "`get_training_status`" in instructions
    assert "`cancel_training`" in instructions


def test_project_instructions_and_file_agents_are_sanitized_without_mutation(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    agents = workspace / ".agents" / "agents"
    agents.mkdir(parents=True)
    instructions = workspace / "AGENTS.md"
    definition = agents / "review.md"
    instructions.write_text(HTML_HEADER + "# Project rules\n", encoding="utf-8")
    definition.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\n"
        + PLAIN_HEADER
        + "Review carefully.\n",
        encoding="utf-8",
    )

    skills = sanitized_project_skills(workspace)
    definitions = sanitized_agent_definitions(workspace)

    assert "SPDX-" not in next(skill.content for skill in skills if skill.name == "agents")
    assert "SPDX-" not in next(item.system_prompt for item in definitions if item.name == "review")
    assert instructions.read_text(encoding="utf-8").startswith("<!--\nSPDX-")
    assert "# SPDX-" in definition.read_text(encoding="utf-8")


def test_resolved_config_separates_runtime_credentials_from_command_secrets(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env.update(
        {
            "GH_TOKEN": "secondary-github-key",
            "WANDB_API_KEY": "wandb-key",
            "EXA_API_KEY": "exa-key",
            "SENPAI_TIMEOUT_MINUTES": "0.5",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.api_key.get_secret_value() == "openai-key"
    assert config.smart_api_key.get_secret_value() == "openai-key"
    assert config.fast_api_key.get_secret_value() == "openai-key"
    assert config.frontier_api_key.get_secret_value() == "openai-key"
    assert config.github_token.get_secret_value() == "github-key"
    assert config.command_secrets == {
        "WANDB_API_KEY": "wandb-key",
        "EXA_API_KEY": "exa-key",
    }
    assert "ANTHROPIC_API_KEY" not in config.command_secrets
    assert "OPENAI_API_KEY" not in config.command_secrets
    assert config.training_max_timeout_seconds == 30
    assert config.llm_timeout_seconds == 900
    assert config.llm_num_retries == 1

    delegated = runner.delegation_config(config)
    assert delegated.smart_api_key == "openai-key"
    assert delegated.fast_api_key == "openai-key"
    assert delegated.frontier_api_key == "openai-key"


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


def test_local_condenser_limits_are_explicit_and_configurable(tmp_path):
    default = resolve_config(
        parse_runner_args(["--max-turns", "1"]),
        runtime_env(tmp_path),
    )
    env = runtime_env(tmp_path)
    env["SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS"] = "180"
    env["SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS"] = "190000"
    env["SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS"] = "40"

    configured = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert default.local_condenser_max_events == 0
    assert default.local_condenser_max_tokens == 0
    assert default.local_condenser_target_events == 0
    assert configured.local_condenser_max_events == 180
    assert configured.local_condenser_max_tokens == 190_000
    assert configured.local_condenser_target_events == 40


@pytest.mark.parametrize("value", ["eleven", "11"])
def test_local_condenser_event_cap_rejects_invalid_values(tmp_path, value):
    env = runtime_env(tmp_path)
    env["SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS"] = value

    with pytest.raises(RuntimeError, match="condenser|integers|at least 12"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS", "-1"),
        ("SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS", "-1"),
    ],
)
def test_local_condenser_limits_reject_negative_values(tmp_path, name, value):
    env = runtime_env(tmp_path)
    env[name] = value

    with pytest.raises(RuntimeError, match="non-negative"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_default_model_profiles_are_explicit_and_provider_credentials_are_inferred(
    tmp_path: Path,
):
    config = resolve_config(
        parse_runner_args(["--max-turns", "1"]),
        runtime_env(tmp_path),
    )

    assert (
        config.model,
        config.api_key_env,
        config.reasoning_effort,
    ) == ("openai/gpt-5.6-sol", "OPENAI_API_KEY", "xhigh")
    assert (
        config.smart_model,
        config.smart_api_key_env,
        config.smart_reasoning_effort,
    ) == ("openai/gpt-5.6-sol", "OPENAI_API_KEY", "xhigh")
    assert (
        config.fast_model,
        config.fast_api_key_env,
        config.fast_reasoning_effort,
    ) == ("openai/gpt-5.6-luna", "OPENAI_API_KEY", "high")
    assert (
        config.frontier_model,
        config.frontier_api_key_env,
        config.frontier_reasoning_effort,
    ) == ("openai/gpt-5.6-sol", "OPENAI_API_KEY", "max")


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


def test_fast_model_uses_luna_for_an_openai_main_profile(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "SENPAI_OPENHANDS_MODEL": "openai/gpt-5.6",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.smart_model == "openai/gpt-5.6"
    assert config.fast_model == "openai/gpt-5.6-luna"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.smart_api_key_env == "OPENAI_API_KEY"
    assert config.fast_api_key_env == "OPENAI_API_KEY"


def test_fast_model_inherits_a_non_openai_main_profile(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "SENPAI_OPENHANDS_MODEL": "anthropic/claude-opus-4-8",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.smart_model == "anthropic/claude-opus-4-8"
    assert config.fast_model == "anthropic/claude-opus-4-8"
    assert config.smart_api_key_env == "ANTHROPIC_API_KEY"
    assert config.fast_api_key_env == "ANTHROPIC_API_KEY"


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
        (
            {"SENPAI_OPENHANDS_FRONTIER_MODEL": "anthropic/claude-haiku-4-5"},
            "unsupported for",
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


def test_requested_anthropic_profiles_resolve_with_documented_efforts(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env.update(
        {
            "ANTHROPIC_API_KEY": "anthropic-key",
            "SENPAI_OPENHANDS_MODEL": "anthropic/claude-opus-5",
            "SENPAI_OPENHANDS_REASONING_EFFORT": "xhigh",
            "SENPAI_OPENHANDS_SMART_MODEL": "anthropic/claude-opus-5",
            "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": "xhigh",
            "SENPAI_OPENHANDS_FAST_MODEL": "anthropic/claude-sonnet-5",
            "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": "high",
            "SENPAI_OPENHANDS_FRONTIER_MODEL": "anthropic/claude-fable-5",
            "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": "max",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert (config.model, config.reasoning_effort) == (
        "anthropic/claude-opus-5",
        "xhigh",
    )
    assert (config.smart_model, config.smart_reasoning_effort) == (
        "anthropic/claude-opus-5",
        "xhigh",
    )
    assert (config.fast_model, config.fast_reasoning_effort) == (
        "anthropic/claude-sonnet-5",
        "high",
    )
    assert (config.frontier_model, config.frontier_reasoning_effort) == (
        "anthropic/claude-fable-5",
        "max",
    )


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
