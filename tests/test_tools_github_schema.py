from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from senpai_agent.github.tools import (
    AcceptResultOnCurrentBaseTool,
    CloseExperimentTool,
    CreateAssignmentTool,
    GitHubToolRuntime,
    MergeExperimentTool,
    PublishAdvisorBranchTool,
    RepairAssignmentRoutingTool,
    RequestAssignmentRevisionTool,
    RespondToHumanIssueTool,
    SendAssignmentFeedbackAction,
    SendAssignmentFeedbackTool,
    SubmitExperimentResultAction,
    SubmitExperimentResultTool,
)


EXPECTED_FIELDS = {
    "create_assignment": {
        "assignment_id",
        "revision_id",
        "student",
        "expected_base_sha",
        "head_branch",
        "title",
        "body",
    },
    "publish_advisor_branch": {
        "remote_branch_sha_before_push",
        "local_commit_sha",
    },
    "repair_assignment_routing": {"assignment", "working_state", "blockers"},
    "send_assignment_feedback": {"assignment", "feedback_id", "comment"},
    "request_assignment_revision": {
        "assignment",
        "new_revision_id",
        "required_base_sha",
        "comment",
    },
    "accept_result_on_current_base": {
        "assignment",
        "expected_current_base_sha",
        "reason",
    },
    "merge_experiment": {
        "assignment",
        "expected_current_base_sha",
        "merge_method",
    },
    "close_experiment": {"assignment", "reason"},
    "respond_to_human_issue": {"issue_number", "human_message_id", "response"},
    "submit_experiment_result": {
        "branch",
        "remote_branch_sha_before_push",
        "result",
    },
}

OPTIONAL_FIELDS = {
    "repair_assignment_routing": {"blockers"},
    "merge_experiment": {"merge_method"},
}


def github_tools(tmp_path: Path):
    runtime = GitHubToolRuntime(
        workflow=SimpleNamespace(repo="acme/widgets"),
        workspace=tmp_path,
        git_token=None,
        role="advisor",
        advisor_branch="advisor-branch",
        student_names=frozenset({"student-one"}),
        student_name=None,
    )
    tool_types = (
        CreateAssignmentTool,
        PublishAdvisorBranchTool,
        RepairAssignmentRoutingTool,
        SendAssignmentFeedbackTool,
        RequestAssignmentRevisionTool,
        AcceptResultOnCurrentBaseTool,
        MergeExperimentTool,
        CloseExperimentTool,
        RespondToHumanIssueTool,
        SubmitExperimentResultTool,
    )
    return [tool_type.create(runtime)[0] for tool_type in tool_types]


@pytest.mark.parametrize("provider", ["to_openai_tool", "to_responses_tool"])
def test_provider_facing_github_schemas_are_single_intent_and_unambiguous(
    tmp_path: Path,
    provider: str,
):
    for tool in github_tools(tmp_path):
        advertised = getattr(tool, provider)()
        function = advertised["function"] if provider == "to_openai_tool" else advertised
        schema = function["parameters"]
        properties = schema["properties"]
        fields = set(properties) - {"summary"}

        assert fields == EXPECTED_FIELDS[tool.name]
        assert set(schema["required"]) == fields - OPTIONAL_FIELDS.get(tool.name, set())
        assert "transition" not in properties
        assert "operation" not in properties
        assert all(properties[field].get("description") for field in fields)
        assert all(" / " not in properties[field]["description"] for field in fields)

        if "assignment" in fields:
            assignment = properties["assignment"]
            assert set(assignment["properties"]) == {
                "pr_number",
                "assignment_id",
                "revision_id",
                "expected_pr_head_sha",
            }
            assert set(assignment["required"]) == set(assignment["properties"])
            assert all(
                field.get("description")
                for field in assignment["properties"].values()
            )


def test_operation_specific_actions_reject_fields_from_other_tools():
    assignment = {
        "pr_number": 17,
        "assignment_id": "assignment-17",
        "revision_id": "revision-1",
        "expected_pr_head_sha": "a" * 40,
    }
    with pytest.raises(ValidationError, match="accepted_base_sha"):
        SendAssignmentFeedbackAction.model_validate(
            {
                "assignment": assignment,
                "feedback_id": "inspect-seed",
                "comment": "Inspect the failed seed.",
                "accepted_base_sha": "b" * 40,
            }
        )
    with pytest.raises(ValidationError, match="accepted_base_sha"):
        SubmitExperimentResultAction.model_validate(
            {
                "branch": "student-one/candidate",
                "remote_branch_sha_before_push": "a" * 40,
                "result": {},
                "accepted_base_sha": "b" * 40,
            }
        )


@pytest.mark.parametrize("provider", ["to_openai_tool", "to_responses_tool"])
def test_submit_result_provider_schema_describes_every_nested_property(
    tmp_path: Path,
    provider: str,
):
    tool = next(
        tool for tool in github_tools(tmp_path) if tool.name == "submit_experiment_result"
    )
    advertised = getattr(tool, provider)()
    function = advertised["function"] if provider == "to_openai_tool" else advertised

    def assert_described(schema: dict, path: str) -> None:
        for name, field in schema.get("properties", {}).items():
            current = f"{path}.{name}"
            assert field.get("description"), current
            assert_described(field, current)
            items = field.get("items")
            if isinstance(items, dict):
                assert_described(items, f"{current}[]")

    assert_described(function["parameters"], "submit_experiment_result")
