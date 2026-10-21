from uuid import uuid4

import pytest
from git_workflow_support import commit_file, detached_commit, git, repository
from github_retrieval_support import (
    FakeGitHubReader,
    inline_comment,
    install_fake_github,
    issue_comment,
    review,
)
from github_retrieval_support import (
    pull_request as retrieved_pull,
)
from github_workflow_support import FakeGitHub, assignment_record, pull_request
from openhands_support import runtime_config

from senpai_agent.git_transport import GitWorkflowPreconditionError
from senpai_agent.github.supervision import (
    SupervisorEnvelope,
    SupervisorGateway,
    SupervisorRequest,
)
from senpai_agent.github.tools.contracts import AssignmentVersion
from senpai_agent.github.workflow import WorkflowPreconditionError
from senpai_agent.models import render_assignment_marker


@pytest.mark.parametrize(
    ("role", "case", "error", "message"),
    [
        ("student", "aligned", None, None),
        ("student", "oversized-request", ValueError, "too large"),
        ("advisor", "aligned", None, None),
        ("advisor", "student-report", None, None),
        ("advisor", "wrong-branch", WorkflowPreconditionError, "current checkout"),
        ("advisor", "wrong-pod", PermissionError, "different pod"),
        ("student", "wrong-pod", PermissionError, "different pod"),
        ("student", "wrong-branch", WorkflowPreconditionError, "current checkout"),
        (
            "student",
            "diverged",
            GitWorkflowPreconditionError,
            "merge-base --is-ancestor failed",
        ),
    ],
)
def test_gateway_requires_the_target_checkout_and_preserves_unpublished_work(
    tmp_path,
    monkeypatch,
    role,
    case,
    error,
    message,
):
    workspace, _remote, published_head = repository(tmp_path)
    local_head = commit_file(
        workspace, "model.py", "baseline = 2\n", "unpublished repair"
    )
    (workspace / "model.py").write_text("baseline = 3\n")
    (workspace / "scratch.py").write_text("preserved = True\n")
    if role == "advisor":
        git(workspace, "branch", "-m", "schmidhuber")
    if case == "wrong-branch":
        git(workspace, "branch", "-m", "other-assignment")
    if case == "diverged":
        published_head = detached_commit(workspace, published_head, "published sibling")
    assignment = assignment_record(head_ref="experiment-7", head_sha=published_head)
    body = (
        render_assignment_marker(assignment)
        + "\n\nAssignment PR: preserve the data loader."
    )
    github = FakeGitHub(
        pull_request(
            labels={"status:wip", "student:student-one"},
            body=body,
            head_ref="experiment-7",
            head_sha=published_head,
        )
    )
    monkeypatch.setattr(
        "senpai_agent.github.workflow.core.UrllibTransport", lambda: github
    )
    discussions = {
        number: {
            "body": body
            if number == 7
            else "Related student-two PR: same loader failure.",
            "comment": f"PR {number}: full diagnostic comment.",
            "review": f"PR {number}: full review conclusion.",
            "inline": f"PR {number}: loader.py inline finding.",
        }
        for number in (7, 8)
    }
    install_fake_github(
        monkeypatch,
        FakeGitHubReader(
            {
                n: retrieved_pull(n, head=published_head, body=text["body"])
                for n, text in discussions.items()
            },
            comments={
                n: [issue_comment(n, text["comment"])]
                for n, text in discussions.items()
            },
            reviews={n: [review(n, text["review"])] for n, text in discussions.items()},
            inline_comments={
                n: [inline_comment(n, text["inline"])]
                for n, text in discussions.items()
            },
        ),
    )
    gateway = SupervisorGateway(
        runtime_config(
            tmp_path,
            workspace=workspace,
            role=role,
            student_name="student-one",
            advisor_branch="schmidhuber",
        )
    )
    request = SupervisorRequest(
        request_id="repair-loader",
        target="student-two"
        if case == "wrong-pod"
        else "advisor"
        if role == "advisor"
        else "student-one",
        assignment=AssignmentVersion(
            pr_number=7,
            assignment_id=assignment.assignment_id,
            revision_id=assignment.revision_id,
            expected_pr_head_sha=published_head,
        )
        if role == "student" or case in {"student-report", "wrong-pod"}
        else None,
        task=">" * 5_000
        if case == "oversized-request"
        else "Repair the local data loader while preserving ongoing work.",
        context_prs=[7, 8],
    )
    envelope = SupervisorEnvelope(
        repo=gateway.config.github_repo,
        advisor_branch="schmidhuber",
        requester="student-one" if case == "student-report" else "advisor",
        request=request,
        parent_conversation_id=uuid4(),
    )
    branch = git(workspace, "branch", "--show-current")
    status = git(workspace, "status", "--short")

    if error is not None:
        with pytest.raises(error, match=message):
            gateway.prepare(envelope)
    else:
        prompt = gateway.prepare(envelope)
        assert (
            f"Requested by: {envelope.requester} ({'student' if case == 'student-report' else 'advisor'})"
            in prompt
        )
        if role == "advisor" and case != "student-report":
            assert "Assignment: (none; advisor repair)" in prompt
        for text in (request.task, local_head, "model.py", "scratch.py"):
            assert text in prompt
        for discussion in discussions.values():
            for text in discussion.values():
                assert text in prompt
    assert git(workspace, "branch", "--show-current") == branch
    assert git(workspace, "rev-parse", "HEAD") == local_head
    assert git(workspace, "status", "--short") == status
    assert (workspace / "model.py").read_text() == "baseline = 3\n"
    assert (workspace / "scratch.py").read_text() == "preserved = True\n"
    assert not github.mutations
