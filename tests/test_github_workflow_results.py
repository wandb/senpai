from urllib.parse import urlsplit

import pytest

from senpai_agent.github.workflow import (
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker, render_result_comment
from github_workflow_support import (
    ASSIGNMENT_ID,
    BASE_SHA,
    HEAD_SHA,
    REPO,
    AmbiguousMutationGitHub,
    FakeGitHub,
    assignment_record,
    comment,
    experiment_result,
    pull_request,
    workflow,
)


def submit_result(client, result=None):
    result = experiment_result() if result is None else result
    return client.submit_result(
        7,
        expected_head_sha=HEAD_SHA,
        result=result,
    )


def close_experiment(
    client,
    *,
    assignment_id: str = ASSIGNMENT_ID,
    current_revision_id: str = "revision-1",
):
    return client.close_experiment(
        7,
        assignment_id=assignment_id,
        current_revision_id=current_revision_id,
        expected_head_sha=HEAD_SHA,
        marker="<!-- senpai-disposition:v1 dead-end-7 -->",
        reason="The hypothesis was falsified.",
    )


def test_submit_result_converges_review_state_and_replays_without_writes():
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )
    result = experiment_result()
    client = workflow(fake, role="student")

    first = submit_result(client, result)
    mutations_after_first = list(fake.mutations)
    second = submit_result(client, result)

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}
    assert len(fake.comments) == 1
    assert "\n\nSTUDENT: Status: succeeded" in str(fake.comments[0]["body"])
    assert fake.mutations == mutations_after_first


def test_published_result_is_immutable_at_the_same_revision_and_head():
    original = experiment_result()
    changed = original.model_copy(update={"summary": "Changed evidence."})
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False),
        comments=[comment(1, render_result_comment(original))],
    )

    with pytest.raises(WorkflowPreconditionError, match="immutable"):
        submit_result(workflow(fake), changed)

    assert fake.comments == [comment(1, render_result_comment(original))]
    assert fake.mutations == []


@pytest.mark.parametrize("advance", ["revision", "head"])
def test_changed_result_is_allowed_on_a_new_revision_or_head(advance):
    original = experiment_result()
    changed = original.model_copy(update={"summary": "Updated evidence."})
    pr_head = HEAD_SHA
    assignment = assignment_record()
    if advance == "revision":
        assignment = assignment_record(revision_id="revision-2")
        changed = changed.model_copy(
            update={
                "assignment": changed.assignment.model_copy(
                    update={"revision_id": "revision-2"}
                )
            }
        )
    else:
        pr_head = "c" * 40
        changed = changed.model_copy(
            update={
                "assignment": changed.assignment.model_copy(
                    update={"expected_head_sha": pr_head}
                ),
                "commit_sha": pr_head,
            }
        )
    fake = FakeGitHub(
        pull_request(
            labels={"student:one", "status:wip"},
            draft=True,
            head_sha=pr_head,
            body=render_assignment_marker(assignment),
        ),
        comments=[comment(1, render_result_comment(original))],
    )

    result = workflow(fake, role="student").submit_result(
        7,
        expected_head_sha=pr_head,
        result=changed,
    )

    assert result.state == "result_submitted"
    expected = render_result_comment(changed).replace(
        "\n\n", "\n\nSTUDENT: ", 1
    )
    assert fake.comments == [comment(1, expected)]


def test_identical_duplicate_result_replay_upgrades_legacy_role_prefix_once():
    terminal = experiment_result()
    body = render_result_comment(terminal)
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False),
        comments=[comment(1, body), comment(2, body)],
    )
    client = workflow(fake, role="student")

    submitted = submit_result(client, terminal)
    mutations_after_upgrade = list(fake.mutations)
    replayed = submit_result(client, terminal)

    assert submitted.state == "result_submitted"
    assert submitted.changed is True
    assert replayed.changed is False
    expected = body.replace("\n\n", "\n\nSTUDENT: ", 1)
    assert [item["body"] for item in fake.comments] == [expected, expected]
    assert sum(
        method == "PATCH" and "/issues/comments/" in path
        for method, path, _body in fake.mutations
    ) == 2
    assert fake.mutations == mutations_after_upgrade


def test_submit_preflight_accepts_the_result_head_before_it_is_pushed():
    final_head = "c" * 40
    fake = FakeGitHub(pull_request(head_sha=HEAD_SHA, draft=True))

    preflight = workflow(fake).preflight_submit_result(
        7,
        branch="student-one/lower-lr",
        current_head_sha=HEAD_SHA,
        expected_result_head_sha=final_head,
        result=experiment_result(
            commit_sha=final_head,
            expected_head_sha=final_head,
        ),
    )

    assert preflight.snapshot.head_sha == HEAD_SHA
    assert preflight.assignment.base_sha == BASE_SHA
    assert fake.mutations == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("assignment_id", "other-assignment"), ("student", "student-two")],
)
def test_submit_preflight_rejects_a_result_from_another_assignment(field, value):
    fake = FakeGitHub(pull_request(draft=True))
    result = experiment_result()
    result = result.model_copy(
        update={"assignment": result.assignment.model_copy(update={field: value})}
    )

    with pytest.raises(WorkflowPreconditionError, match="assignment mismatch"):
        workflow(fake).preflight_submit_result(
            7,
            branch="student-one/lower-lr",
            current_head_sha=HEAD_SHA,
            expected_result_head_sha=HEAD_SHA,
            result=result,
        )

    assert fake.mutations == []


def test_submit_preflight_reports_a_stale_assignment_revision_with_its_own_type():
    current = assignment_record(revision_id="revision-2")
    fake = FakeGitHub(
        pull_request(body=render_assignment_marker(current), draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        workflow(fake).preflight_submit_result(
            7,
            branch="student-one/lower-lr",
            current_head_sha=HEAD_SHA,
            expected_result_head_sha=HEAD_SHA,
            result=experiment_result(),
        )

    assert fake.mutations == []


def test_submit_preflight_rejects_a_foreign_branch_before_writing():
    fake = FakeGitHub(pull_request(draft=True))

    with pytest.raises(WorkflowPreconditionError, match="branch"):
        workflow(fake).preflight_submit_result(
            7,
            branch="student-one/unrelated",
            current_head_sha=HEAD_SHA,
            expected_result_head_sha=HEAD_SHA,
            result=experiment_result(),
        )

    assert fake.mutations == []


def test_submit_result_recovers_when_the_comment_response_is_lost():
    fake = AmbiguousMutationGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True),
        fail_method="POST",
        fail_path=f"/repos/{REPO}/issues/7/comments",
    )

    result = submit_result(workflow(fake, role="student"))

    assert result.state == "result_submitted"
    assert fake.failed is True
    assert len(fake.comments) == 1
    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}


def test_submit_result_stops_if_assignment_revision_changes_after_comment():
    class RevisingGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if (
                method == "POST"
                and urlsplit(url).path == f"/repos/{REPO}/issues/7/comments"
            ):
                self.pr["body"] = render_assignment_marker(
                    assignment_record(revision_id="revision-2")
                )
            return response

    fake = RevisingGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        submit_result(workflow(fake))

    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {"student:one", "status:wip"}
    assert len(fake.comments) == 1


def test_submit_result_restores_new_revision_to_wip_after_ready_race():
    class RevisionDuringReadyGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if method == "POST" and urlsplit(url).path == "/graphql":
                self.pr["body"] = render_assignment_marker(
                    assignment_record(revision_id="revision-2")
                )
            return response

    fake = RevisionDuringReadyGitHub(
        pull_request(
            labels={"student:one", "status:wip", "status:hold", "keep"},
            draft=True,
        )
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        submit_result(workflow(fake))

    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {
        "student:one",
        "status:wip",
        "status:hold",
        "keep",
    }


def test_stale_submit_does_not_overwrite_current_revision_review_result():
    current = experiment_result().model_copy(
        update={
            "assignment": experiment_result().assignment.model_copy(
                update={"revision_id": "revision-2"}
            ),
            "summary": "The current revision has valid evidence.",
        }
    )

    class CurrentResultDuringLabelsGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if (
                method == "PUT"
                and urlsplit(url).path == f"/repos/{REPO}/issues/7/labels"
            ):
                self.pr["body"] = render_assignment_marker(
                    assignment_record(revision_id="revision-2")
                )
                self.comments = [comment(1, render_result_comment(current))]
            return response

    fake = CurrentResultDuringLabelsGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        submit_result(workflow(fake))

    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}
    assert fake.comments == [comment(1, render_result_comment(current))]


def test_submit_result_does_not_treat_the_initial_assignment_head_as_a_lease():
    assignment = assignment_record(head_sha="c" * 40)
    fake = FakeGitHub(
        pull_request(
            labels={"student:one", "status:wip"},
            draft=True,
            body=render_assignment_marker(assignment),
        )
    )

    submitted = submit_result(workflow(fake, role="student"))

    assert submitted.state == "result_submitted"


@pytest.mark.parametrize(
    "result",
    [
        experiment_result(commit_sha="b" * 40),
        experiment_result(repo="other/widgets"),
        experiment_result(pr_number=8),
        experiment_result(expected_head_sha="b" * 40),
    ],
    ids=("commit", "repository", "pull-request", "expected-head"),
)
def test_submit_result_rejects_mismatched_result_location(result):
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(WorkflowPreconditionError, match="result"):
        submit_result(workflow(fake, role="student"), result)

    assert fake.mutations == []


def test_close_experiment_writes_one_reason_and_replays_without_writes():
    fake = FakeGitHub(pull_request(labels={"status:review"}))
    client = workflow(fake)

    first = close_experiment(client)
    mutations_after_first = list(fake.mutations)
    second = close_experiment(client)

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["state"] == "closed"
    assert fake.comments == [
        comment(
            1,
            "<!-- senpai-disposition:v1 dead-end-7 -->\n\n"
            "ADVISOR: The hypothesis was falsified.",
        )
    ]
    assert fake.mutations == mutations_after_first


def test_close_experiment_does_not_overwrite_a_merged_pull_request():
    fake = FakeGitHub(
        pull_request(labels={"status:review"}, state="closed", merged=True)
    )

    with pytest.raises(WorkflowPreconditionError, match="unmerged"):
        close_experiment(workflow(fake))

    assert fake.mutations == []


@pytest.mark.parametrize(
    ("merge_after", "expected_comments"),
    [("PATCH", 0), ("POST", 1)],
    ids=["before-disposition", "final-reconciliation"],
)
def test_close_experiment_rejects_a_pull_merged_during_the_transition(
    merge_after: str,
    expected_comments: int,
):
    class ConcurrentMergeGitHub(FakeGitHub):
        merge_on_next_pull = False

        def request(self, method, url, *, headers, json_body=None):
            path = urlsplit(url).path
            if (
                self.merge_on_next_pull
                and method == "GET"
                and path == f"/repos/{REPO}/pulls/7"
            ):
                self.pr.update(
                    state="closed",
                    merged=True,
                    merge_commit_sha="concurrent-merge",
                )
                self.merge_on_next_pull = False
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            trigger_path = (
                f"/repos/{REPO}/pulls/7"
                if merge_after == "PATCH"
                else f"/repos/{REPO}/issues/7/comments"
            )
            if method == merge_after and path == trigger_path:
                self.merge_on_next_pull = True
            return response

    fake = ConcurrentMergeGitHub(pull_request(labels={"status:review"}))

    with pytest.raises(WorkflowPreconditionError, match="unmerged"):
        close_experiment(workflow(fake))

    assert fake.pr["merged"] is True
    assert len(fake.comments) == expected_comments


def test_close_experiment_rejects_a_foreign_assignment_before_writing():
    fake = FakeGitHub(pull_request(labels={"status:review"}))

    with pytest.raises(WorkflowPreconditionError, match="assignment"):
        close_experiment(workflow(fake), assignment_id="other-assignment")

    assert fake.mutations == []


def test_close_experiment_rejects_a_stale_revision_before_writing():
    fake = FakeGitHub(pull_request())

    with pytest.raises(WorkflowPreconditionError, match="revision"):
        close_experiment(
            workflow(fake),
            current_revision_id="revision-0",
        )

    assert fake.mutations == []
