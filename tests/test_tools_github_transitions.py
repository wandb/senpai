from contextlib import contextmanager
from pathlib import Path

import pytest

from senpai_agent.git_workflow import PushResult
from senpai_agent.github.tools import (
    AcceptResultOnCurrentBaseAction,
    AcceptResultOnCurrentBaseTool,
    AssignmentVersion,
    CloseExperimentAction,
    CloseExperimentTool,
    CreateAssignmentAction,
    CreateAssignmentTool,
    GitHubToolRuntime,
    MergeExperimentAction,
    MergeExperimentTool,
    PublishAdvisorBranchAction,
    PublishAdvisorBranchTool,
    RepairAssignmentRoutingAction,
    RepairAssignmentRoutingTool,
    RequestAssignmentRevisionAction,
    RequestAssignmentRevisionTool,
    SendAssignmentFeedbackAction,
    SendAssignmentFeedbackTool,
)
from senpai_agent.github.workflow import MutationResult


class RecordingWorkflow:
    repo = "acme/widgets"

    def __init__(self):
        self.calls = []

    @contextmanager
    def serialized_assignment_mutation(self):
        self.calls.append(("lock_enter", None, {}))
        try:
            yield
        finally:
            self.calls.append(("lock_exit", None, {}))

    def __getattr__(self, name):
        def call(number, **kwargs):
            self.calls.append((name, number, kwargs))
            return MutationResult(
                changed=True,
                resource_url=f"https://github.test/pull/{number}",
                state=name,
                version=kwargs.get("expected_head_sha"),
            )

        return call


def runtime(workflow: RecordingWorkflow, workspace: Path) -> GitHubToolRuntime:
    return GitHubToolRuntime(
        workflow=workflow,
        workspace=workspace,
        git_token=None,
        role="advisor",
        advisor_branch="advisor-branch",
        student_names=frozenset({"student-one"}),
        student_name=None,
    )


def assignment() -> AssignmentVersion:
    return AssignmentVersion(
        pr_number=17,
        assignment_id="assignment-17",
        revision_id="revision-1",
        expected_pr_head_sha="a" * 40,
    )


def test_create_assignment_uses_the_created_branch_head_for_the_pr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workflow = RecordingWorkflow()
    branch_calls = []

    def create_branch(workspace, **kwargs):
        workflow.calls.append(("create_branch", None, {}))
        branch_calls.append((workspace, kwargs))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha="c" * 40,
        )

    monkeypatch.setattr(
        "senpai_agent.github.tools.advisor.git_workflow.create_assignment_branch",
        create_branch,
    )
    tool = CreateAssignmentTool.create(runtime(workflow, tmp_path))[0]
    action = CreateAssignmentAction(
        assignment_id="assignment-18",
        revision_id="revision-1",
        student="student-one",
        expected_base_sha="b" * 40,
        head_branch="student-one/lower-lr",
        title="Try a lower learning rate",
        body="Run one bounded comparison.",
    )

    observation = tool(action)

    assert observation.state == "create_assignment"
    assert branch_calls[0][1] == {
        "branch": "student-one/lower-lr",
        "base_branch": "advisor-branch",
        "expected_base_sha": "b" * 40,
        "assignment_id": "assignment-18",
        "token": None,
    }
    assert [call[0] for call in workflow.calls] == [
        "lock_enter",
        "create_branch",
        "create_assignment",
        "lock_exit",
    ]
    _, _, fields = workflow.calls[2]
    created = workflow.calls[2][1]
    assert created.repo == "acme/widgets"
    assert created.head_sha == "c" * 40
    assert fields == {
        "title": "Try a lower learning rate",
        "body": "Run one bounded comparison.",
    }


def test_create_assignment_rejects_students_outside_the_launch_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workflow = RecordingWorkflow()
    monkeypatch.setattr(
        "senpai_agent.github.tools.advisor.git_workflow.create_assignment_branch",
        lambda *_args, **_kwargs: pytest.fail("git mutation reached"),
    )
    tool = CreateAssignmentTool.create(runtime(workflow, tmp_path))[0]
    action = CreateAssignmentAction(
        assignment_id="assignment-18",
        revision_id="revision-1",
        student="student-outside-launch",
        expected_base_sha="b" * 40,
        head_branch="student-outside-launch/lower-lr",
        title="Try a lower learning rate",
        body="Run one bounded comparison.",
    )

    with pytest.raises(PermissionError, match="outside this launch"):
        tool(action)

    assert workflow.calls == []


def test_publish_advisor_branch_uses_configured_branch_and_distinct_shas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    pushes = []

    workflow = RecordingWorkflow()

    def push(workspace, **kwargs):
        workflow.calls.append(("push", None, {}))
        pushes.append((workspace, kwargs))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha=kwargs["expected_local_sha"],
        )

    monkeypatch.setattr(
        "senpai_agent.github.tools.advisor.git_workflow.push_assignment_branch",
        push,
    )
    tool = PublishAdvisorBranchTool.create(
        runtime(workflow, tmp_path)
    )[0]
    observation = tool(
        PublishAdvisorBranchAction(
            remote_branch_sha_before_push="a" * 40,
            local_commit_sha="b" * 40,
        )
    )

    assert observation.state == "branch_pushed"
    assert [call[0] for call in workflow.calls] == ["lock_enter", "push", "lock_exit"]
    assert pushes == [
        (
            tmp_path,
            {
                "branch": "advisor-branch",
                "expected_remote_sha": "a" * 40,
                "expected_local_sha": "b" * 40,
                "token": None,
            },
        )
    ]


@pytest.mark.parametrize(
    ("tool_type", "action", "method", "expected"),
    [
        (
            RepairAssignmentRoutingTool,
            RepairAssignmentRoutingAction(
                assignment=assignment(),
                working_state="review",
                blockers={"hold"},
            ),
            "repair_assignment_routing",
            {"working_state": "review", "blockers": {"hold"}},
        ),
        (
            SendAssignmentFeedbackTool,
            SendAssignmentFeedbackAction(
                assignment=assignment(),
                feedback_id="inspect-seed",
                comment="Inspect the failed seed.",
            ),
            "send_assignment_feedback",
            {"feedback_id": "inspect-seed", "comment": "Inspect the failed seed."},
        ),
        (
            RequestAssignmentRevisionTool,
            RequestAssignmentRevisionAction(
                assignment=assignment(),
                new_revision_id="revision-2",
                required_base_sha="b" * 40,
                comment="Rerun on the current research base.",
            ),
            "request_revision",
            {
                "new_revision_id": "revision-2",
                "required_base_sha": "b" * 40,
            },
        ),
        (
            AcceptResultOnCurrentBaseTool,
            AcceptResultOnCurrentBaseAction(
                assignment=assignment(),
                expected_current_base_sha="b" * 40,
                reason="The changed files do not intersect this mechanism.",
            ),
            "accept_result_on_current_base",
            {"expected_current_base_sha": "b" * 40},
        ),
        (
            MergeExperimentTool,
            MergeExperimentAction(
                assignment=assignment(),
                expected_current_base_sha="b" * 40,
            ),
            "merge_experiment",
            {"expected_current_base_sha": "b" * 40, "merge_method": "squash"},
        ),
        (
            CloseExperimentTool,
            CloseExperimentAction(
                assignment=assignment(),
                reason="The hypothesis was falsified.",
            ),
            "close_experiment",
            {"reason": "The hypothesis was falsified."},
        ),
    ],
)
def test_assignment_tools_forward_one_exact_assignment_version(
    tmp_path: Path,
    tool_type,
    action,
    method: str,
    expected: dict,
):
    workflow = RecordingWorkflow()
    observation = tool_type.create(runtime(workflow, tmp_path))[0](action)

    assert observation.state == method
    name, number, fields = workflow.calls[0]
    assert name == method
    assert number == 17
    assert fields["assignment_id"] == "assignment-17"
    assert fields["expected_head_sha"] == "a" * 40
    revision_field = (
        "revision_id"
        if method == "send_assignment_feedback"
        else "current_revision_id"
    )
    if method != "send_assignment_feedback":
        assert fields[revision_field] == "revision-1"
    else:
        assert fields["revision_id"] == "revision-1"
    for key, value in expected.items():
        assert fields[key] == value
