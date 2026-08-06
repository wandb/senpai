from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from senpai_agent.operational_tools import SupervisorOperationAction


CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000201")


def test_supervisor_tool_exposes_only_campaign_operations():
    schema = SupervisorOperationAction.model_json_schema()
    operations = set(schema["properties"]["operation"]["enum"])

    assert operations == {
        "inspect",
        "nudge",
        "restart_controller",
        "reset_context",
    }
    assert not {
        "host",
        "pod",
        "namespace",
        "cwd",
        "environment",
        "argv",
        "command",
    } & schema["properties"].keys()


def test_supervisor_tool_requires_exact_student_identity_and_mutation_context():
    with pytest.raises(ValidationError, match="configured student name"):
        SupervisorOperationAction(
            operation="inspect",
            operation_key="inspect-student",
            role="student",
        )

    with pytest.raises(ValidationError, match="incident_key, anomaly_category"):
        SupervisorOperationAction(
            operation="restart_controller",
            operation_key="restart-advisor",
            role="advisor",
        )


def test_supervisor_tool_uses_a_typed_anomaly_category_for_cooldowns():
    action = SupervisorOperationAction(
        operation="restart_controller",
        operation_key="restart-advisor",
        incident_key="advisor-loop-17",
        anomaly_category="restart_churn",
        reason="Repeated controller restarts made no progress.",
        role="advisor",
        expected_conversation_id=CONVERSATION_ID,
    )

    assert action.anomaly_category == "restart_churn"
    with pytest.raises(ValidationError, match="Input should be"):
        SupervisorOperationAction(
            operation="restart_controller",
            operation_key="restart-advisor-again",
            incident_key="advisor-loop-18",
            anomaly_category="model-invented-category",
            reason="The model renamed the category.",
            role="advisor",
            expected_conversation_id=CONVERSATION_ID,
        )


def test_supervisor_tool_rejects_restart_without_expected_conversation():
    with pytest.raises(ValidationError, match="expected conversation UUID"):
        SupervisorOperationAction(
            operation="restart_controller",
            operation_key="restart-without-conversation",
            incident_key="advisor-loop-19",
            anomaly_category="restart_churn",
            reason="The controller appears stuck.",
            role="advisor",
        )
