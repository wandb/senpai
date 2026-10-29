"""Student GitHub tools refuse identifiers that no runtime record accounts for."""

from pathlib import Path

import pytest
from git_workflow_support import repository
from test_provenance import FakeLookup
from test_tools_github_submission import RecordingWorkflow, experiment_result

from senpai_agent.github.tools import (
    AssignmentVersion,
    GitHubToolRuntime,
    PostAssignmentCommentAction,
    PostAssignmentCommentTool,
    SubmitExperimentResultAction,
    SubmitExperimentResultTool,
)
from senpai_agent.github.workflow import MutationResult
from senpai_agent.models import WandbRunRef
from senpai_agent.provenance import IdentifierProvenance, ProvenanceError


class CommentWorkflow(RecordingWorkflow):
    def post_assignment_comment(self, number, **kwargs):
        self.events.append(("comment", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}#issuecomment-1",
            state="assignment_comment_posted",
            version=kwargs["expected_head_sha"],
        )


def student_runtime(workflow, workspace: Path, lookup: FakeLookup) -> GitHubToolRuntime:
    return GitHubToolRuntime(
        workflow=workflow,
        workspace=workspace,
        git_token=None,
        role="student",
        advisor_branch=None,
        student_names=frozenset(),
        student_name="student-one",
        provenance=IdentifierProvenance(
            workspace=workspace,
            wandb_entity="acme",
            wandb_project="cfd",
            lookup=lookup,
        ),
    )


def comment_action(comment: str) -> PostAssignmentCommentAction:
    return PostAssignmentCommentAction(
        assignment=AssignmentVersion(
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            expected_pr_head_sha="a" * 40,
        ),
        comment_id="progress-1",
        comment=comment,
    )


def test_post_assignment_comment_refuses_an_unknown_run_id_before_posting(tmp_path):
    workspace, _remote, _head = repository(tmp_path)
    workflow = CommentWorkflow()
    lookup = FakeLookup({"ab12cd34": "finished"})
    tool = PostAssignmentCommentTool.create(student_runtime(workflow, workspace, lookup))[0]

    with pytest.raises(ProvenanceError, match="'58vh6sck'"):
        tool(comment_action("Runs `ab12cd34` and `58vh6sck` finished."))
    assert workflow.events == []

    posted = tool(comment_action("Run `ab12cd34` finished at 118.2 µs."))

    assert posted.state == "assignment_comment_posted"
    assert [event[0] for event in workflow.events] == ["comment"]


def test_submit_result_resolves_short_shas_and_refuses_unknown_runs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    workspace, _remote, head = repository(tmp_path)
    workflow = RecordingWorkflow()
    lookup = FakeLookup({"ab12cd34": "finished"})
    pushes = []
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.push_assignment_branch",
        lambda *args, **kwargs: pushes.append(kwargs),
    )
    monkeypatch.setattr(
        "senpai_agent.github.tools.runtime.git_workflow.require_commit_contains_base",
        lambda *args, **kwargs: None,
    )
    tool = SubmitExperimentResultTool.create(student_runtime(workflow, workspace, lookup))[0]
    result = experiment_result(head[:8]).model_copy(
        update={
            "runs": (
                WandbRunRef(
                    run_id="ab12cd34",
                    url="https://wandb.ai/acme/cfd/runs/ab12cd34",
                    state="finished",
                ),
            )
        }
    )

    submitted = tool(
        SubmitExperimentResultAction(
            branch="student-one/candidate",
            remote_branch_sha_before_push="a" * 40,
            result=result,
        )
    )

    assert submitted.version == head
    preflight = workflow.events[0]
    assert preflight[0] == "preflight"
    assert preflight[2]["expected_result_head_sha"] == head
    assert preflight[2]["result"].commit_sha == head
    assert preflight[2]["result"].assignment.expected_head_sha == head
    assert pushes[0]["expected_local_sha"] == head

    invented = result.model_copy(
        update={
            "runs": (
                WandbRunRef(
                    run_id="58vh6sck",
                    url="https://wandb.ai/acme/cfd/runs/58vh6sck",
                    state="finished",
                ),
            )
        }
    )
    workflow.events.clear()
    with pytest.raises(ProvenanceError, match="'58vh6sck'"):
        tool(
            SubmitExperimentResultAction(
                branch="student-one/candidate",
                remote_branch_sha_before_push="a" * 40,
                result=invented,
            )
        )
    assert workflow.events == []
