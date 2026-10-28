from pathlib import Path

import pytest
import yaml
from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.base import CondensationRequirement
from openhands.sdk.context.condenser.utils import get_total_token_count
from openhands.sdk.context.view import View
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent
from pydantic import SecretStr

from launch_test_support import launch, launch_args, render_role
from openhands_support import launch_env
from senpai_agent.custom_openai import CustomOpenAIEndpoint
from senpai_agent.openhands_runner import (
    local_condenser,
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


def test_main_yaml_event_trigger_reaches_the_runner_config(tmp_path):
    project = yaml.safe_load(launch.SENPAI_CONFIG.read_text())
    args = launch_args(
        compaction_trigger_events=project["compaction_trigger_events"],
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

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), environment)

    assert project["compaction_trigger_events"] == 400
    assert config.compaction_trigger_events == 400


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-5.6-sol", "anthropic/claude-opus-4-8"],
)
def test_native_compaction_models_run_without_a_local_condenser(model):
    llm = LLM(
        model=model,
        api_key=SecretStr("test-key"),
        **model_runtime_configuration(
            model, "high", compaction_trigger_tokens=200_000
        ),
    )

    assert local_condenser(llm, trigger_tokens=200_000, trigger_events=400) is None


def user_message(text: str) -> MessageEvent:
    return MessageEvent(
        source="user",
        llm_message=Message(role="user", content=[TextContent(text=text)]),
    )


SERVED_MODEL = "custom_openai/served-model"
# A 262,144-token window less 65,536 output tokens leaves a 196,608-token input limit.
ENDPOINT = CustomOpenAIEndpoint(
    base_url="https://inference.example.test/v1",
    context_tokens=262_144,
    max_output_tokens=65_536,
)


def served_model_llm(compaction_trigger_tokens: int) -> LLM:
    return LLM(
        model=SERVED_MODEL,
        api_key=SecretStr("test-key"),
        usage_id="senpai",
        **model_runtime_configuration(
            SERVED_MODEL,
            "high",
            compaction_trigger_tokens=compaction_trigger_tokens,
            custom_openai=ENDPOINT,
        ),
    )


def test_custom_openai_condenser_compacts_on_the_launch_triggers():
    llm = served_model_llm(compaction_trigger_tokens=50_000)

    condenser = local_condenser(llm, trigger_tokens=50_000, trigger_events=400)

    assert isinstance(condenser, LLMSummarizingCondenser)
    assert condenser.max_size == 400
    assert condenser.max_tokens == 50_000
    assert condenser.keep_first == 4
    assert condenser.llm.model == SERVED_MODEL
    assert condenser.llm.usage_id == "senpai-condenser"

    # OpenHands' own default compacted at 80 events; 400 short events stay put.
    full_window = View(events=[user_message(f"step {i}") for i in range(400)])
    assert condenser.condensation_requirement(full_window, agent_llm=llm) is None

    one_over = View(events=[user_message(f"step {i}") for i in range(401)])
    assert (
        condenser.condensation_requirement(one_over, agent_llm=llm)
        is CondensationRequirement.SOFT
    )

    # Token pressure fires first on long events and must be handled at once.
    heavy = View(events=[user_message("token " * 1_000) for _ in range(120)])
    assert (
        condenser.condensation_requirement(heavy, agent_llm=llm)
        is CondensationRequirement.HARD
    )


def test_custom_openai_condenser_caps_the_token_trigger_at_the_input_limit():
    llm = served_model_llm(compaction_trigger_tokens=200_000)
    condenser = local_condenser(llm, trigger_tokens=200_000, trigger_events=400)
    assert condenser is not None

    # 197 events of 1,000 words count to about 198,600 tokens: under the launch
    # trigger but over the endpoint's input limit, which the SDK applies instead.
    between = View(events=[user_message("token " * 1_000) for _ in range(197)])
    counted = get_total_token_count(between.events, llm)
    assert ENDPOINT.max_input_tokens < counted < 200_000
    assert (
        condenser.condensation_requirement(between, agent_llm=llm)
        is CondensationRequirement.HARD
    )
