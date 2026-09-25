from pathlib import Path

import pytest
import yaml
from openhands.sdk import LLM
from openhands.sdk.llm import Message, TextContent

from launch_test_support import launch, launch_args, render_role
from openhands_support import launch_env
from senpai_agent.openhands_runner import (
    model_runtime_configuration,
    parse_runner_args,
    resolve_config,
)
from senpai_agent.program_context import (
    PROGRAM_CONTEXT_FILE_ENV,
    decode_program_system_prompt,
)
from senpai_agent.supervisor import prepare_system_context_environment


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-5.6-sol", "anthropic/claude-opus-4-8"],
)
def test_main_yaml_token_trigger_reaches_the_provider_request(tmp_path, model):
    project = yaml.safe_load(launch.SENPAI_CONFIG.read_text())
    args = launch_args(
        student_model=model,
        compaction_trigger_tokens=project["compaction_trigger_tokens"],
    )
    environment = launch_env(tmp_path, role="student")
    snapshot_file = environment[PROGRAM_CONTEXT_FILE_ENV]
    program = decode_program_system_prompt(Path(snapshot_file).read_text())
    configmap, _deployment, _secret = render_role("student", args, program=program)
    environment.update(yaml.safe_load(configmap)["data"])
    environment[PROGRAM_CONTEXT_FILE_ENV] = snapshot_file
    environment = prepare_system_context_environment(
        "student", Path(environment["SENPAI_OPENHANDS_STATE_DIR"]), environment
    )
    config = resolve_config(
        parse_runner_args(["--max-turns", "1"]),
        environment,
    )
    llm = LLM(
        model=config.model,
        api_key=config.api_key,
        **model_runtime_configuration(
            config.model,
            config.reasoning_effort,
            compaction_trigger_tokens=config.compaction_trigger_tokens,
        ),
    )
    message = Message(role="user", content=[TextContent(text="Investigate")])
    if model.startswith("openai/"):
        _instructions, _inputs, _tools, call_kwargs, _telemetry = (
            llm._prepare_responses_params(
                [message],
                tools=None,
                include=None,
                store=None,
                add_security_risk_prediction=False,
                kwargs={},
            )
        )
        trigger = call_kwargs["context_management"][0]["compact_threshold"]
    else:
        _messages, _tools, _mocked, call_kwargs, _telemetry = (
            llm._prepare_completion_params(
                [message],
                tools=None,
                add_security_risk_prediction=False,
                kwargs={},
            )
        )
        trigger = call_kwargs["context_management"]["edits"][0]["trigger"][
            "value"
        ]
        assert "instructions" not in call_kwargs["context_management"][
            "edits"
        ][0]

    assert config.compaction_trigger_tokens == 200_000
    assert trigger == 200_000
