"""A tool call that fails validation is answered with field, schema and example."""

import json
from types import SimpleNamespace

import pytest
from openhands.sdk.tool import Tool, resolve_tool
from openhands.tools.file_editor import FileEditorAction
from openhands.tools.terminal import TerminalAction
from pydantic import ValidationError

from senpai_agent.delegation import AwaitAgentsAction
from senpai_agent.github.tools import (
    BroadcastMessageAction,
    PostAssignmentCommentAction,
    PostPeerCommentAction,
    PushExperimentCommitAction,
    SubmitExperimentResultAction,
)
from senpai_agent.schema_help import (
    EXAMPLES,
    SchemaHelp,
    describe_validation_error,
    explaining_validation_errors,
    with_schema_help,
)
from senpai_agent.tools import (
    CancelTrainingAction,
    GetTrainingStatusAction,
    GetTrainingStatusTool,
    RunTrainingAction,
    register_senpai_tools,
)

ACTIONS = {
    cls.__name__: cls
    for cls in (
        AwaitAgentsAction,
        BroadcastMessageAction,
        CancelTrainingAction,
        FileEditorAction,
        GetTrainingStatusAction,
        PostAssignmentCommentAction,
        PostPeerCommentAction,
        PushExperimentCommitAction,
        RunTrainingAction,
        SubmitExperimentResultAction,
        TerminalAction,
    )
}
# The action types with the most validation errors in the 2026-11-03 review.
MOST_REJECTED = {
    "SubmitExperimentResultAction",
    "FileEditorAction",
    "TerminalAction",
    "PostAssignmentCommentAction",
    "AwaitAgentsAction",
    "RunTrainingAction",
}


def editor_state(tmp_path):
    return SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(tmp_path)),
        agent=SimpleNamespace(llm=SimpleNamespace(vision_is_active=lambda: False)),
    )


def test_every_example_is_a_valid_payload_for_its_action():
    assert MOST_REJECTED <= set(EXAMPLES) <= set(ACTIONS)
    for name, example in EXAMPLES.items():
        ACTIONS[name].model_validate(example)


def test_description_names_each_field_its_schema_and_a_valid_example():
    with pytest.raises(ValidationError) as error:
        SubmitExperimentResultAction.model_validate(
            {
                "branch": "student-one/x",
                "result": {
                    "assignment": {"pr_number": "seventeen"},
                    "runs": [
                        {
                            "run_id": "ab12cd34",
                            "url": "https://wandb.ai/a/b/runs/ab12cd34",
                            "state": "done",
                        }
                    ],
                },
            }
        )

    text = describe_validation_error(error.value, SubmitExperimentResultAction)

    assert text.splitlines()[0].endswith(" validation error(s) for SubmitExperimentResultAction.")
    assert "- remote_branch_sha_before_push: Field required (missing)." in text
    assert "- result.assignment.pr_number: Input should be a valid integer" in text
    assert '  schema at result.assignment.pr_number: {"description":' in text
    assert "- result.runs[0].state: Input should be 'finished', 'failed', 'crashed' or 'killed' (got str)." in text
    assert '"enum":["finished","failed","crashed","killed"]' in text
    assert "Required: branch, remote_branch_sha_before_push, result; all fields: branch, remote_branch_sha_before_push, result." in text
    assert "Valid example: " + json.dumps(
        EXAMPLES["SubmitExperimentResultAction"], separators=(",", ":")
    ) in text


def test_a_missing_object_is_summarised_by_field_type():
    with pytest.raises(ValidationError) as error:
        PostAssignmentCommentAction.model_validate({"comment_id": "x", "comment": "y"})

    text = describe_validation_error(error.value, PostAssignmentCommentAction)

    assert "- assignment: Field required (missing)." in text
    assert (
        '"properties":{"pr_number":"integer","assignment_id":"string",'
        '"revision_id":"string","expected_pr_head_sha":"string"}'
    ) in text
    assert '"required":["pr_number","assignment_id","revision_id","expected_pr_head_sha"]' in text


def test_an_unknown_field_and_a_bad_timeout_are_explained_for_the_terminal():
    with pytest.raises(ValidationError) as error:
        TerminalAction.model_validate({"cmd": "ls", "timeout": -1})

    text = describe_validation_error(error.value, TerminalAction)

    assert "- timeout: Input should be greater than or equal to 0 (got int)." in text
    assert '  schema at timeout: {' in text and '"minimum":0' in text
    assert "all fields: command, is_input, timeout, reset." in text
    assert '"command":"python -m pytest -q tests/test_model.py"' in text


def test_file_editor_from_the_registry_explains_a_bad_call_and_runs_a_good_one(tmp_path):
    register_senpai_tools()
    editor = resolve_tool(Tool(name="file_editor"), editor_state(tmp_path))[0]

    assert isinstance(editor, SchemaHelp)
    assert (editor.name, editor.kind) == ("file_editor", "FileEditorTool")
    with pytest.raises(ValueError) as error:
        editor.action_from_arguments({"command": "str_replace", "path": 5})
    assert "- path: Input should be a valid string (got int)." in str(error.value)
    assert '"old_str":"lr = 1e-3"' in str(error.value)

    (tmp_path / "note.txt").write_text("hello\n")
    action = editor.action_from_arguments({"command": "view", "path": str(tmp_path / "note.txt")})
    assert editor(action).is_error is False


def test_wrapping_keeps_the_executor_and_schema_and_is_idempotent():
    training = SimpleNamespace()
    tool = GetTrainingStatusTool.create(training)[0]

    helped = with_schema_help(tool)

    assert with_schema_help(helped) is helped
    assert helped.executor is tool.executor
    assert helped.to_openai_tool() == tool.to_openai_tool()
    assert helped.kind == tool.kind == "GetTrainingStatusTool"
    with pytest.raises(ValueError, match="training_id: Field required"):
        helped.action_from_arguments({})
    assert helped.action_from_arguments({"training_id": "t-1"}).training_id == "t-1"


def test_explaining_class_creates_helped_tools_under_the_original_name():
    from senpai_agent.github.tools import GetPRsTool

    explained = explaining_validation_errors(GetTrainingStatusTool)

    assert explained is not GetTrainingStatusTool
    assert explained.__name__ == "GetTrainingStatusTool"
    # A tool whose name is not the snake case of its class keeps its name.
    assert explaining_validation_errors(GetPRsTool).name == GetPRsTool.name == "get_prs"
    assert explaining_validation_errors(explained) is explained
    [tool] = explained.create(SimpleNamespace())
    assert isinstance(tool, SchemaHelp)
    assert tool.kind == "GetTrainingStatusTool"
