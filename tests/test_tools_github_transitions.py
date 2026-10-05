from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import ConversationExecutionStatus

from senpai_agent.git_workflow import PushResult
from senpai_agent.github.tools import (
    AcceptResultOnCurrentBaseAction,
    AcceptResultOnCurrentBaseTool,
    AssignmentVersion,
    BroadcastMessageAction,
    BroadcastMessageTool,
    CloseExperimentAction,
    CloseExperimentTool,
    CreateAssignmentAction,
    CreateAssignmentTool,
    GitHubToolRuntime,
    MergeExperimentAction,
    MergeExperimentTool,
    PostAssignmentCommentAction,
    PostAssignmentCommentTool,
    PostPeerCommentAction,
    PostPeerCommentTool,
    PublishAdvisorBranchAction,
    PublishAdvisorBranchTool,
    RepairAssignmentRoutingAction,
    RepairAssignmentRoutingTool,
    RequestAssignmentRevisionAction,
    RequestAssignmentRevisionTool,
    SendAssignmentFeedbackAction,
    SendAssignmentFeedbackTool,
)
from senpai_agent.github.workflow import (
    MutationResult,
    ReconciliationError,
    StaleAssignmentRevisionError,
)
from senpai_agent.github.workflow.responses import BroadcastResult
from senpai_agent.models import DispositionRecord, render_disposition_marker


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


def student_runtime(
    workflow: RecordingWorkflow,
    workspace: Path,
    *,
    student_name: str | None = "student-one",
    advisor_branch: str | None = "advisor-branch",
) -> GitHubToolRuntime:
    return GitHubToolRuntime(
        workflow=workflow,
        workspace=workspace,
        git_token=None,
        role="student",
        advisor_branch=advisor_branch,
        student_names=frozenset(),
        student_name=student_name,
    )


def assignment() -> AssignmentVersion:
    return AssignmentVersion(
        pr_number=17,
        assignment_id="assignment-17",
        revision_id="revision-1",
        expected_pr_head_sha="a" * 40,
    )


@pytest.mark.parametrize(
    ("tool_type", "action_type", "method", "target"),
    [
        (
            PostAssignmentCommentTool,
            PostAssignmentCommentAction,
            "post_assignment_comment",
            {},
        ),
        (
            PostPeerCommentTool,
            PostPeerCommentAction,
            "post_peer_comment",
            {"target_pr_number": 18},
        ),
    ],
)
def test_student_comment_binds_runtime_student_and_exact_assignment(
    tmp_path: Path, tool_type, action_type, method, target
):
    workflow = RecordingWorkflow()
    tool = tool_type.create(student_runtime(workflow, tmp_path))[0]

    observation = tool(
        action_type(
            assignment=assignment(),
            comment_id="paired-run-started",
            comment="The paired run has started.",
            **target,
        )
    )

    assert observation.state == method
    assert workflow.calls == [
        (
            method,
            17,
            {
                "assignment_id": "assignment-17",
                "revision_id": "revision-1",
                "expected_head_sha": "a" * 40,
                "student": "student-one",
                "comment_id": "paired-run-started",
                "comment": "The paired run has started.",
                **target,
                **({"advisor_branch": "advisor-branch"} if target else {}),
            },
        )
    ]


@pytest.mark.parametrize(
    ("tool_type", "action_type", "target", "runtime_options", "error"),
    [
        (
            PostAssignmentCommentTool,
            PostAssignmentCommentAction,
            {},
            {"student_name": None},
            "student name",
        ),
        (
            PostPeerCommentTool,
            PostPeerCommentAction,
            {"target_pr_number": 18},
            {"student_name": None},
            "student name",
        ),
        (
            PostPeerCommentTool,
            PostPeerCommentAction,
            {"target_pr_number": 18},
            {"advisor_branch": None},
            "advisor branch",
        ),
    ],
)
def test_student_comment_requires_configured_identity_before_mutation(
    tmp_path: Path, tool_type, action_type, target, runtime_options, error
):
    workflow = RecordingWorkflow()
    tool = tool_type.create(
        student_runtime(workflow, tmp_path, **runtime_options)
    )[0]

    with pytest.raises(RuntimeError, match=error):
        tool(
            action_type(
                assignment=assignment(),
                comment_id="blocked",
                comment="The experiment is blocked.",
                **target,
            )
        )

    assert workflow.calls == []


def test_stale_student_comment_finishes_the_obsolete_conversation(tmp_path: Path):
    class StaleWorkflow(RecordingWorkflow):
        def post_assignment_comment(self, number, **kwargs):
            self.calls.append(("post_assignment_comment", number, kwargs))
            raise StaleAssignmentRevisionError(
                "assignment revision is 'revision-2', expected 'revision-1'"
            )

    workflow = StaleWorkflow()
    tool = PostAssignmentCommentTool.create(student_runtime(workflow, tmp_path))[0]
    conversation = SimpleNamespace(
        state=SimpleNamespace(execution_status=ConversationExecutionStatus.RUNNING)
    )

    with pytest.raises(ValueError, match="controller can resume"):
        tool(
            PostAssignmentCommentAction(
                assignment=assignment(),
                comment_id="stale-progress",
                comment="This turn is obsolete.",
            ),
            conversation,
        )

    assert conversation.state.execution_status is ConversationExecutionStatus.FINISHED
    assert [call[0] for call in workflow.calls] == ["post_assignment_comment"]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (StaleAssignmentRevisionError, ConversationExecutionStatus.FINISHED),
        (ReconciliationError, ConversationExecutionStatus.RUNNING),
    ],
    ids=("sender-reassigned", "recipient-reassigned"),
)
@pytest.mark.parametrize(
    ("tool_type", "action_type", "fields"),
    [
        (
            PostPeerCommentTool,
            PostPeerCommentAction,
            {
                "target_pr_number": 18,
                "comment_id": "stale-peer-message",
                "comment": "This question requires the current assignment.",
            },
        ),
        (
            BroadcastMessageTool,
            BroadcastMessageAction,
            {"broadcast_id": "stale-discovery", "message": "A discovery."},
        ),
    ],
)
def test_student_peer_tools_only_finish_a_stale_sender_turn(
    tmp_path, error, expected_status, tool_type, action_type, fields
):
    class ChangedWorkflow(RecordingWorkflow):
        def post_peer_comment(self, number, **kwargs):
            raise error("assignment changed while posting comment")

        def broadcast_message(self, number, **kwargs):
            raise error("assignment changed while broadcasting discovery")

    tool = tool_type.create(student_runtime(ChangedWorkflow(), tmp_path))[0]
    conversation = SimpleNamespace(
        state=SimpleNamespace(execution_status=ConversationExecutionStatus.RUNNING)
    )

    with pytest.raises(ValueError if error is StaleAssignmentRevisionError else error):
        tool(action_type(assignment=assignment(), **fields), conversation)

    assert conversation.state.execution_status is expected_status


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
        "authenticated_remote": "https://github.com/acme/widgets.git",
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


@pytest.mark.parametrize(
    ("student", "head_branch", "error"),
    [
        ("student-outside-launch", "student-outside-launch/run", "outside this launch"),
        ("student-one", "unowned/run", "must belong to student"),
        ("student-one", "student-one-other/run", "must belong to student"),
        ("student-one", "student-one", "must belong to student"),
    ],
)
def test_create_assignment_rejects_unowned_students_or_branches_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    student: str,
    head_branch: str,
    error: str,
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
        student=student,
        expected_base_sha="b" * 40,
        head_branch=head_branch,
        title="Try a lower learning rate",
        body="Run one bounded comparison.",
    )

    with pytest.raises(PermissionError, match=error):
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
                "authenticated_remote": "https://github.com/acme/widgets.git",
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
            {
                "marker": render_disposition_marker(
                    DispositionRecord(
                        repo="acme/widgets",
                        pr_number=17,
                        assignment_id="assignment-17",
                        head_sha="a" * 40,
                    )
                ),
                "reason": "The hypothesis was falsified.",
            },
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


class RecordingBroadcastWorkflow(RecordingWorkflow):
    def broadcast_message(self, number, **kwargs):
        self.calls.append(("broadcast_message", number, kwargs))
        return BroadcastResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}#issuecomment-1",
            state="broadcast_posted",
            version=kwargs["expected_head_sha"],
            delivered_pr_numbers=(18, 19),
        )


def test_broadcast_tool_binds_runtime_identity_and_returns_delivery_receipt(tmp_path):
    workflow = RecordingBroadcastWorkflow()
    tool = BroadcastMessageTool.create(student_runtime(workflow, tmp_path))[0]

    observation = tool(
        BroadcastMessageAction(
            assignment=assignment(),
            broadcast_id="shared-discovery",
            message="Checkpoint resumes omit optimizer state. See the working PR.",
        )
    )

    assert observation.delivered_pr_numbers == (18, 19)
    assert observation.changed is True
    assert observation.resource_url == "https://github.test/pull/17#issuecomment-1"
    assert workflow.calls == [
        (
            "broadcast_message",
            17,
            {
                "assignment_id": "assignment-17",
                "revision_id": "revision-1",
                "expected_head_sha": "a" * 40,
                "student": "student-one",
                "advisor_branch": "advisor-branch",
                "broadcast_id": "shared-discovery",
                "message": "Checkpoint resumes omit optimizer state. See the working PR.",
            },
        )
    ]


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"student_name": None}, "student name"),
        ({"advisor_branch": None}, "advisor branch"),
    ],
)
def test_broadcast_requires_runtime_identity_before_publication(tmp_path, options, error):
    workflow = RecordingBroadcastWorkflow()
    tool = BroadcastMessageTool.create(student_runtime(workflow, tmp_path, **options))[0]

    with pytest.raises(RuntimeError, match=error):
        tool(
            BroadcastMessageAction(
                assignment=assignment(), broadcast_id="discovery", message="A finding."
            )
        )

    assert workflow.calls == []
