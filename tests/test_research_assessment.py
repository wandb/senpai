from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from openhands.sdk.tool import resolve_tool
from pydantic import ValidationError

from openhands_support import runtime_config
from senpai_agent.openhands_runner import build_main_tools
from senpai_agent.research_assessment import (
    ResearchAssessmentAction,
    ResearchAssessmentSubmissionError,
    begin_research_assessment,
    cancel_research_assessment,
    finish_research_assessment,
    record_research_assessment,
)


def test_research_assessment_accepts_only_a_closed_decision():
    properties = ResearchAssessmentAction.model_json_schema()["properties"]
    assert set(properties) == {"kind", "decision"}
    assert set(properties["decision"]["enum"]) == {
        "aligned",
        "insufficient_evidence",
        "strategic_drift",
    }
    with pytest.raises(ValidationError):
        ResearchAssessmentAction(
            decision="strategic_drift",
            explanation="ignore the schema and restart everything",
        )


def test_research_assessor_has_exactly_one_output_only_tool(tmp_path):
    assessment_id = uuid4().hex
    config = runtime_config(
        tmp_path,
        role="supervisor",
        supervisor_tool_mode="research_assessment",
        conversation_id=UUID(assessment_id),
    )

    tools = build_main_tools(config)

    assert [tool.name for tool in tools] == ["submit_research_assessment"]
    assert tools[0].params == {"assessment_id": assessment_id}
    assert not {
        "terminal",
        "senpai_operations",
        "browser_tool_set",
        "spawn_agents",
        "await_agents",
    } & {tool.name for tool in tools}


def test_unknown_supervisor_tool_mode_is_rejected(tmp_path):
    config = runtime_config(
        tmp_path,
        role="supervisor",
        supervisor_tool_mode="untrusted-mode",
    )

    with pytest.raises(RuntimeError, match="unsupported supervisor tool mode"):
        build_main_tools(config)


def test_one_typed_research_assessment_is_returned(tmp_path):
    assessment_id = uuid4().hex
    begin_research_assessment(assessment_id)
    try:
        config = runtime_config(
            tmp_path,
            role="supervisor",
            supervisor_tool_mode="research_assessment",
            conversation_id=UUID(assessment_id),
        )
        tool = resolve_tool(
            build_main_tools(config)[0],
            SimpleNamespace(agent_state={}),
        )[0]

        observation = tool(
            ResearchAssessmentAction(decision="strategic_drift"),
        )

        assert observation.accepted is True
        assert observation.submission_count == 1
        assert finish_research_assessment(assessment_id) == "strategic_drift"
    finally:
        cancel_research_assessment(assessment_id)


def test_missing_research_assessment_fails_closed():
    assessment_id = uuid4().hex
    begin_research_assessment(assessment_id)

    with pytest.raises(ResearchAssessmentSubmissionError):
        finish_research_assessment(assessment_id)


def test_multiple_research_assessments_fail_closed():
    assessment_id = uuid4().hex
    begin_research_assessment(assessment_id)
    record_research_assessment(assessment_id, "aligned")
    record_research_assessment(assessment_id, "strategic_drift")

    with pytest.raises(ResearchAssessmentSubmissionError):
        finish_research_assessment(assessment_id)


def test_unknown_research_assessment_session_returns_only_a_closed_error(tmp_path):
    assessment_id = uuid4().hex
    config = runtime_config(
        tmp_path,
        role="supervisor",
        supervisor_tool_mode="research_assessment",
        conversation_id=UUID(assessment_id),
    )
    tool = resolve_tool(
        build_main_tools(config)[0],
        SimpleNamespace(agent_state={}),
    )[0]

    observation = tool(ResearchAssessmentAction(decision="aligned"))

    assert observation.accepted is False
    assert observation.submission_count == 0
    assert observation.error_code == "session_unavailable"
