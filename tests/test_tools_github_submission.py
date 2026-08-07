from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import ConversationExecutionStatus

from senpai_agent.git_workflow import PushResult
from senpai_agent.github.tools import (
    GitHubToolRuntime,
    SubmitExperimentResultAction,
    SubmitExperimentResultTool,
)
from senpai_agent.github.workflow import (
    MutationResult,
    PullHeadMismatchError,
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.models import (
    AssignmentKey,
    ExperimentResult,
    ResultStatus,
)


def experiment_result(head_sha: str = "c" * 40) -> ExperimentResult:
    return ExperimentResult(
        assignment=AssignmentKey(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            expected_head_sha=head_sha,
            student="student-one",
        ),
        status=ResultStatus.INCONCLUSIVE,
        hypothesis="Check the bounded candidate.",
        summary="The bounded comparison completed.",
        runs=(),
        primary_metric=None,
        commit_sha=head_sha,
    )


def submit_action() -> SubmitExperimentResultAction:
    return SubmitExperimentResultAction(
        branch="student-one/candidate",
        remote_branch_sha_before_push="a" * 40,
        result=experiment_result(),
    )


class RecordingWorkflow:
    def __init__(self):
        self.repo = "acme/widgets"
        self.events: list[tuple[str, int, dict]] = []
        self.lock_depth = 0

    @contextmanager
    def serialized_assignment_mutation(self):
        self.lock_depth += 1
        try:
            yield
        finally:
            self.lock_depth -= 1

    def preflight_submit_result(self, number, **kwargs):
        self.events.append(("preflight", number, kwargs))
        return SimpleNamespace(assignment=SimpleNamespace(base_sha="b" * 40))

    def submit_result(self, number, **kwargs):
        self.events.append(("submit", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}",
            state="result_submitted",
            version=kwargs["expected_head_sha"],
        )


def student_tool(workflow, workspace: Path):
    runtime = GitHubToolRuntime(
        workflow=workflow,
        workspace=workspace,
        git_token=None,
        role="student",
        advisor_branch=None,
        student_names=frozenset(),
        student_name="student-one",
    )
    return SubmitExperimentResultTool.create(runtime)[0]


def test_submit_result_preflights_before_any_git_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class RejectingWorkflow(RecordingWorkflow):
        def preflight_submit_result(self, number, **kwargs):
            super().preflight_submit_result(number, **kwargs)
            raise WorkflowPreconditionError("invalid assignment")

    workflow = RejectingWorkflow()
    pushes = []
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.push_assignment_branch",
        lambda *args, **kwargs: pushes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.require_commit_contains_base",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(WorkflowPreconditionError, match="invalid assignment"):
        student_tool(workflow, tmp_path)(submit_action())

    assert [event[0] for event in workflow.events] == ["preflight"]
    assert pushes == []


def test_stale_submission_finishes_the_obsolete_conversation_without_pushing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class StaleWorkflow(RecordingWorkflow):
        def preflight_submit_result(self, number, **kwargs):
            super().preflight_submit_result(number, **kwargs)
            raise StaleAssignmentRevisionError(
                "revision='revision-2'; result revision='revision-1'. Refresh PR #17."
            )

    workflow = StaleWorkflow()
    pushes = []
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.push_assignment_branch",
        lambda *args, **kwargs: pushes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.require_commit_contains_base",
        lambda *args, **kwargs: None,
    )
    conversation = SimpleNamespace(
        state=SimpleNamespace(execution_status=ConversationExecutionStatus.RUNNING)
    )

    with pytest.raises(ValueError, match="controller can resume"):
        student_tool(workflow, tmp_path)(submit_action(), conversation)

    assert conversation.state.execution_status is ConversationExecutionStatus.FINISHED
    assert [event[0] for event in workflow.events] == ["preflight"]
    assert pushes == []


def test_submit_result_pushes_the_validated_local_head_before_github_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    order = []

    class Workflow(RecordingWorkflow):
        def preflight_submit_result(self, number, **kwargs):
            order.append("preflight")
            return super().preflight_submit_result(number, **kwargs)

        def submit_result(self, number, **kwargs):
            order.append("submit")
            return super().submit_result(number, **kwargs)

    def push(workspace, **kwargs):
        order.append("push")
        assert workspace == tmp_path
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha=kwargs["expected_local_sha"],
        )

    workflow = Workflow()
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.push_assignment_branch",
        push,
    )
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.require_commit_contains_base",
        lambda *_args, **_kwargs: order.append("ancestry"),
    )

    observation = student_tool(workflow, tmp_path)(submit_action())

    assert observation.state == "result_submitted"
    assert order == ["preflight", "ancestry", "push", "submit"]
    _, _, preflight = workflow.events[0]
    assert preflight["current_head_sha"] == "a" * 40
    assert preflight["expected_result_head_sha"] == "c" * 40
    assert workflow.events[1][2]["expected_head_sha"] == "c" * 40


def test_submit_result_holds_one_workflow_lock_across_the_full_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    phases = []

    class Workflow(RecordingWorkflow):
        def preflight_submit_result(self, number, **kwargs):
            phases.append(("preflight", self.lock_depth))
            return super().preflight_submit_result(number, **kwargs)

        def submit_result(self, number, **kwargs):
            phases.append(("submit", self.lock_depth))
            return super().submit_result(number, **kwargs)

    workflow = Workflow()

    def ancestry(*_args, **_kwargs):
        phases.append(("ancestry", workflow.lock_depth))

    def push(_workspace, **kwargs):
        phases.append(("push", workflow.lock_depth))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha=kwargs["expected_local_sha"],
        )

    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.require_commit_contains_base",
        ancestry,
    )
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.push_assignment_branch",
        push,
    )

    student_tool(workflow, tmp_path)(submit_action())

    assert phases == [
        ("preflight", 1),
        ("ancestry", 1),
        ("push", 1),
        ("submit", 1),
    ]
    assert workflow.lock_depth == 0


@pytest.mark.parametrize(
    ("error_type", "failures", "raises", "expected_attempts", "expected_sleeps"),
    [
        (PullHeadMismatchError, 2, False, 3, [0.5, 1.0]),
        (WorkflowPreconditionError, 1, True, 1, []),
        (PullHeadMismatchError, 6, True, 6, [0.5, 1.0, 2.0, 4.0, 8.0]),
    ],
    ids=["eventual-convergence", "unrelated-precondition", "retry-budget"],
)
def test_submit_result_retries_only_bounded_post_push_head_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type,
    failures: int,
    raises: bool,
    expected_attempts: int,
    expected_sleeps: list[float],
):
    class RetryingWorkflow(RecordingWorkflow):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def submit_result(self, number, **kwargs):
            self.attempts += 1
            if self.attempts <= failures:
                raise error_type("GitHub has not reached the requested state")
            return super().submit_result(number, **kwargs)

    workflow = RetryingWorkflow()
    pushes = []
    sleeps = []

    def push(_workspace, **kwargs):
        pushes.append(kwargs)
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha=kwargs["expected_local_sha"],
        )

    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.push_assignment_branch",
        push,
    )
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.require_commit_contains_base",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("senpai_agent.github.tools.runtime.time.sleep", sleeps.append)
    call = lambda: student_tool(workflow, tmp_path)(submit_action())

    if raises:
        with pytest.raises(error_type):
            call()
    else:
        assert call().state == "result_submitted"

    assert workflow.attempts == expected_attempts
    assert len(pushes) == 1
    assert sleeps == expected_sleeps
