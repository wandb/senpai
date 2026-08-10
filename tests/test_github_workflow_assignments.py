from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from senpai_agent.github.workflow import (
    MutationResult,
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker, render_result_comment
from github_workflow_support import (
    ASSIGNMENT_ID,
    AmbiguousMutationGitHub,
    BASE_SHA,
    HEAD_SHA,
    FakeGitHub,
    assignment_record,
    comment,
    experiment_result,
    pull_request,
    workflow,
)


def test_create_assignment_converges_one_draft_pull_request_and_replays():
    assignment = assignment_record()
    fake = FakeGitHub(pull_request(labels=set(), title="", body=""))
    fake.pr["number"] = 0
    client = workflow(fake)

    first = client.create_assignment(
        assignment,
        title="Try lower learning rate",
        body="Run the bounded learning-rate experiment.",
    )
    mutations_after_first = list(fake.mutations)
    second = client.create_assignment(
        assignment,
        title="Try lower learning rate",
        body="Run the bounded learning-rate experiment.",
    )

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["body"] == (
        f"{render_assignment_marker(assignment)}\n\n"
        "Run the bounded learning-rate experiment."
    )
    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {
        "schmidhuber",
        "student:student-one",
        "status:wip",
    }
    assert fake.mutations == mutations_after_first


def test_create_assignment_rejects_an_unapplied_draft_mutation():
    assignment = assignment_record()
    visible_body = "Run the bounded learning-rate experiment."
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=False,
            body=f"{render_assignment_marker(assignment)}\n\n{visible_body}",
        ),
        ignore_draft_mutations=True,
    )

    with pytest.raises(ReconciliationError, match="not draft"):
        workflow(fake).create_assignment(
            assignment,
            title="Try lower learning rate",
            body=visible_body,
        )


def test_create_assignment_does_not_repurpose_a_foreign_pull_request():
    foreign_body = render_assignment_marker(
        assignment_record(assignment_id="someone-elses-assignment")
    )
    fake = FakeGitHub(pull_request(body=foreign_body))

    with pytest.raises(WorkflowPreconditionError, match="assignment marker"):
        workflow(fake).create_assignment(
            assignment_record(),
            title="Try lower learning rate",
            body="Run the bounded learning-rate experiment.",
        )

    assert fake.mutations == []


def test_create_assignment_rejects_another_wip_pull_for_the_student():
    assignment = assignment_record(
        assignment_id="assignment-8",
        head_ref="student-one/new-candidate",
    )
    fake = FakeGitHub(
        pull_request(
            labels={"other-base", "student:student-one", "status:wip"},
            base_ref="other-base",
            head_ref="student-one/other-candidate",
        )
    )

    with pytest.raises(WorkflowPreconditionError, match="already has active"):
        workflow(fake).create_assignment(
            assignment,
            title="Try another candidate",
            body="Run one bounded comparison.",
        )

    assert fake.mutations == []


def test_create_assignment_allows_a_review_ready_pull_for_the_student():
    assignment = assignment_record(
        assignment_id="assignment-8",
        head_ref="student-one/new-candidate",
    )
    fake = FakeGitHub(
        pull_request(
            labels={"other-base", "student:student-one", "status:review"},
            base_ref="other-base",
            head_ref="student-one/other-candidate",
        )
    )

    result = workflow(fake).create_assignment(
        assignment,
        title="Try another candidate",
        body="Run one bounded comparison.",
    )

    assert result.changed is True
    assert {"student:student-one", "status:wip"}.issubset(fake.pr["labels"])
    assert "status:review" not in fake.pr["labels"]


def test_adopt_assignment_adds_one_marker_and_replays_without_rewriting():
    assignment = assignment_record()
    visible_body = "Run the bounded learning-rate experiment."
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip", "keep"},
            draft=True,
            body=visible_body,
        )
    )
    client = workflow(fake)

    first = client.adopt_assignment(7, assignment)
    mutations_after_first = list(fake.mutations)
    second = client.adopt_assignment(7, assignment)

    assert first.changed is True
    assert second.changed is False
    assert first.state == "assignment_adopted"
    assert fake.pr["body"] == (
        f"{render_assignment_marker(assignment)}\n\n{visible_body}"
    )
    assert fake.pr["title"] == "Try lower learning rate"
    assert fake.pr["labels"] == {
        "schmidhuber",
        "student:student-one",
        "status:wip",
        "keep",
    }
    assert fake.pr["draft"] is True
    assert fake.mutations == mutations_after_first


def test_adopt_assignment_recovers_an_ambiguous_successful_patch():
    fake = AmbiguousMutationGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=True,
            body="Run the bounded learning-rate experiment.",
        ),
        fail_method="PATCH",
        fail_path="/repos/acme/widgets/pulls/7",
    )

    result = workflow(fake).adopt_assignment(7, assignment_record())

    assert result.changed is True
    assert fake.pr["body"].startswith(render_assignment_marker(assignment_record()))


def test_adopt_assignment_replay_survives_student_head_progress():
    assignment = assignment_record()
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=True,
            body=f"{render_assignment_marker(assignment)}\n\nRun the experiment.",
        )
    )
    fake.pr["head_sha"] = "d" * 40

    result = workflow(fake).adopt_assignment(7, assignment)

    assert result.changed is False
    assert result.version == "d" * 40
    assert fake.mutations == []


def test_adopt_assignment_accepts_head_progress_after_marker_patch():
    class AdvancingPatchGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if method == "PATCH" and url.endswith("/pulls/7"):
                self.pr["head_sha"] = "d" * 40
            return response

    fake = AdvancingPatchGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=True,
            body="Run the experiment.",
        )
    )

    result = workflow(fake).adopt_assignment(7, assignment_record())

    assert result.changed is True
    assert result.version == "d" * 40


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"draft": False}, "draft"),
        ({"labels": {"schmidhuber", "student:student-one", "status:review"}}, "status:wip"),
        ({"labels": {"schmidhuber", "student:student-one", "student:other", "status:wip"}}, "student label"),
        ({"labels": {"student:student-one", "status:wip"}}, "base label"),
        ({"author": "other-user"}, "authenticated actor"),
    ],
)
def test_adopt_assignment_rejects_ambiguous_pr_state_without_writing(
    changes,
    error,
):
    pr = pull_request(
        labels={"schmidhuber", "student:student-one", "status:wip"},
        draft=True,
        body="Run the bounded learning-rate experiment.",
    )
    pr.update(changes)
    fake = FakeGitHub(pr)

    with pytest.raises(WorkflowPreconditionError, match=error):
        workflow(fake).adopt_assignment(7, assignment_record())

    assert fake.mutations == []


@pytest.mark.parametrize(
    "body",
    [
        render_assignment_marker(
            assignment_record(assignment_id="someone-elses-assignment")
        ),
        "<!-- senpai-assignment:v1 not-json -->",
    ],
)
def test_adopt_assignment_rejects_existing_assignment_identity_without_writing(body):
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=True,
            body=body,
        )
    )

    with pytest.raises(WorkflowPreconditionError, match="assignment marker"):
        workflow(fake).adopt_assignment(7, assignment_record())

    assert fake.mutations == []


@pytest.mark.parametrize(
    ("body", "comments", "error"),
    [
        (
            "<!-- senpai-feedback:v1 {} -->\n\nRun the experiment.",
            [],
            "other Senpai markers",
        ),
        (
            "Run the experiment.",
            [comment(1, render_result_comment(experiment_result()))],
            "terminal Senpai evidence",
        ),
    ],
)
def test_adopt_assignment_rejects_existing_protocol_state_without_writing(
    body,
    comments,
    error,
):
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=True,
            body=body,
        ),
        comments=comments,
    )

    with pytest.raises(WorkflowPreconditionError, match=error):
        workflow(fake).adopt_assignment(7, assignment_record())

    assert fake.mutations == []


def test_untrusted_comment_cannot_block_assignment_adoption():
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=True,
            body="Run the experiment.",
        ),
        comments=[
            comment(
                1,
                render_result_comment(experiment_result()),
                author="outside-user",
                association="NONE",
            )
        ],
    )

    result = workflow(fake).adopt_assignment(7, assignment_record())

    assert result.changed is True


def test_student_workflow_cannot_adopt_assignment_identity():
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=True,
            body="Run the experiment.",
        )
    )

    with pytest.raises(WorkflowPreconditionError, match="only an advisor"):
        workflow(fake, role="student").adopt_assignment(7, assignment_record())

    assert fake.mutations == []


def test_create_and_revision_transitions_cannot_overlap(monkeypatch):
    client = workflow(FakeGitHub(pull_request()))
    create_entered = Event()
    release_create = Event()
    revision_started = Event()
    revision_entered = Event()
    result = MutationResult(False, "https://github.test/pr/7", "test")

    def hold_create(*_args, **_kwargs):
        create_entered.set()
        assert release_create.wait(1)
        return result

    def observe_revision(*_args, **_kwargs):
        revision_entered.set()
        return result

    monkeypatch.setattr(type(client), "_create_assignment", hold_create)
    monkeypatch.setattr(type(client), "_request_revision", observe_revision)

    def request_revision():
        revision_started.set()
        return client.request_revision(
            7,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-1",
            new_revision_id="revision-2",
            expected_head_sha=HEAD_SHA,
            required_base_sha=BASE_SHA,
            comment="Run the requested ablation.",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        create = executor.submit(
            client.create_assignment,
            assignment_record(),
            title="Try another candidate",
            body="Run one bounded comparison.",
        )
        assert create_entered.wait(1)
        revise = executor.submit(request_revision)
        assert revision_started.wait(1)
        overlapped = revision_entered.wait(0.1)
        release_create.set()
        assert create.result(timeout=1) is result
        assert revise.result(timeout=1) is result

    assert not overlapped
    assert revision_entered.is_set()


def test_submit_and_revision_transitions_cannot_overlap(monkeypatch):
    client = workflow(FakeGitHub(pull_request()))
    submit_entered = Event()
    release_submit = Event()
    revision_started = Event()
    revision_entered = Event()
    result = MutationResult(False, "https://github.test/pr/7", "test")

    def hold_submit(*_args, **_kwargs):
        submit_entered.set()
        assert release_submit.wait(1)
        return result

    def observe_revision(*_args, **_kwargs):
        revision_entered.set()
        return result

    monkeypatch.setattr(type(client), "_submit_result", hold_submit)
    monkeypatch.setattr(type(client), "_request_revision", observe_revision)

    def request_revision():
        revision_started.set()
        return client.request_revision(
            7,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-1",
            new_revision_id="revision-2",
            expected_head_sha=HEAD_SHA,
            required_base_sha=BASE_SHA,
            comment="Run the requested ablation.",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        submit = executor.submit(
            client.submit_result,
            7,
            expected_head_sha=HEAD_SHA,
            result=experiment_result(),
        )
        assert submit_entered.wait(1)
        revise = executor.submit(request_revision)
        assert revision_started.wait(1)
        overlapped = revision_entered.wait(0.1)
        release_submit.set()
        assert submit.result(timeout=1) is result
        assert revise.result(timeout=1) is result

    assert not overlapped
    assert revision_entered.is_set()


def test_repair_routing_sets_exact_protocol_labels_and_preserves_unrelated_labels():
    fake = FakeGitHub(
        pull_request(
            labels={"student:one", "status:wip", "status:hold", "keep"},
            draft=True,
        ),
        comments=[comment(1, render_result_comment(experiment_result()))],
    )
    client = workflow(fake)

    first = client.repair_assignment_routing(
        7,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id="revision-1",
        expected_head_sha=HEAD_SHA,
        working_state="review",
        blockers={"blocked"},
    )
    mutations_after_first = list(fake.mutations)
    second = client.repair_assignment_routing(
        7,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id="revision-1",
        expected_head_sha=HEAD_SHA,
        working_state="review",
        blockers={"blocked"},
    )

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["labels"] == {
        "keep",
        "schmidhuber",
        "student:student-one",
        "status:blocked",
        "status:review",
    }
    assert fake.pr["draft"] is False
    assert fake.mutations == mutations_after_first


def test_repair_routing_restores_wip_draft_state_without_label_changes():
    fake = FakeGitHub(
        pull_request(
            labels={
                "schmidhuber",
                "student:student-one",
                "status:wip",
            },
            draft=False,
        )
    )

    result = workflow(fake).repair_assignment_routing(
        7,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id="revision-1",
        expected_head_sha=HEAD_SHA,
        working_state="wip",
        blockers=set(),
    )

    assert result.changed is True
    assert fake.pr["draft"] is True


def test_repair_routing_rejects_an_unapplied_draft_mutation():
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"},
            draft=False,
        ),
        ignore_draft_mutations=True,
    )

    with pytest.raises(ReconciliationError, match="draft state"):
        workflow(fake).repair_assignment_routing(
            7,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-1",
            expected_head_sha=HEAD_SHA,
            working_state="wip",
            blockers=set(),
        )


def test_repair_routing_rejects_a_stale_head_before_writing():
    fake = FakeGitHub(pull_request())

    with pytest.raises(WorkflowPreconditionError, match="head SHA"):
        workflow(fake).repair_assignment_routing(
            7,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-1",
            expected_head_sha="b" * 40,
            working_state="review",
            blockers=set(),
        )

    assert fake.mutations == []


def test_repair_routing_rejects_a_stale_revision_before_writing():
    fake = FakeGitHub(pull_request())

    with pytest.raises(WorkflowPreconditionError, match="revision"):
        workflow(fake).repair_assignment_routing(
            7,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-0",
            expected_head_sha=HEAD_SHA,
            working_state="review",
            blockers=set(),
        )

    assert fake.mutations == []


def test_repair_routing_fails_if_github_does_not_apply_the_write():
    fake = FakeGitHub(
        pull_request(
            labels={"schmidhuber", "student:student-one", "status:wip"}
        ),
        comments=[comment(1, render_result_comment(experiment_result()))],
        ignore_label_mutations=True,
    )

    with pytest.raises(ReconciliationError, match="label state"):
        workflow(fake).repair_assignment_routing(
            7,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-1",
            expected_head_sha=HEAD_SHA,
            working_state="review",
            blockers=set(),
        )

    assert len(fake.mutations) == 2


def test_repair_routing_rejects_review_without_exact_terminal_result():
    fake = FakeGitHub(pull_request())

    with pytest.raises(WorkflowPreconditionError, match="terminal result"):
        workflow(fake).repair_assignment_routing(
            7,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-1",
            expected_head_sha=HEAD_SHA,
            working_state="review",
            blockers=set(),
        )

    assert fake.mutations == []


def test_repair_routing_rejects_a_foreign_assignment_before_writing():
    fake = FakeGitHub(pull_request())

    with pytest.raises(WorkflowPreconditionError, match="assignment"):
        workflow(fake).repair_assignment_routing(
            7,
            assignment_id="other-assignment",
            current_revision_id="revision-1",
            expected_head_sha=HEAD_SHA,
            working_state="review",
            blockers=set(),
        )

    assert fake.mutations == []
