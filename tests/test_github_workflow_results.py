from urllib.parse import urlsplit

import pytest
from pydantic import SecretStr

from senpai_agent.github.workflow import (
    GitHubAPIError,
    GitHubTransportError,
    HttpResponse,
    GitHubWorkflow,
    ReconciliationError,
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.models import (
    parse_result_markers,
    render_assignment_marker,
    render_result_comment,
    render_result_marker,
)
from github_workflow_support import (
    ASSIGNMENT_ID,
    API_URL,
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
    assert "\n\nSTUDENT:\n\n## Experiment result" in str(
        fake.comments[0]["body"]
    )
    assert "\n### Hypothesis\n" in str(fake.comments[0]["body"])
    assert "\n### Summary\n" in str(fake.comments[0]["body"])
    assert fake.mutations == mutations_after_first


def test_result_replay_resolves_and_caches_the_authenticated_actor():
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True),
        actor_login="SENPAI-BOT",
    )
    client = GitHubWorkflow(
        REPO,
        SecretStr("github-secret"),
        role="student",
        transport=fake,
        api_url=API_URL,
    )

    submit_result(client)
    submit_result(client)

    assert sum(
        method == "GET" and path == "/user"
        for method, path, _body, _headers in fake.requests
    ) == 1


@pytest.mark.parametrize("separator", ["\n", "\r", "\x85", "\u2028"])
def test_result_marker_below_visible_prose_is_not_protocol_evidence(separator):
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True),
        comments=[
            comment(
                1,
                f"Visible prose{separator}{render_result_marker(experiment_result())}",
            )
        ],
    )

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
        "\n\n", "\n\nSTUDENT:\n\n", 1
    )
    assert fake.comments == [
        comment(1, render_result_comment(original)),
        comment(2, expected),
    ]
    assert workflow(fake).merge_experiment(
        7,
        expected_head_sha=pr_head,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id=changed.assignment.revision_id,
        expected_current_base_sha=BASE_SHA,
    ).state == "experiment_merged"


def test_duplicate_result_replay_upgrades_legacy_comment_formats_once():
    terminal = experiment_result()
    legacy_body = "\n".join(
        [
            render_result_marker(terminal),
            "",
            "STUDENT: Status: succeeded",
            f"Commit: `{HEAD_SHA}`",
            "",
            terminal.summary,
            "",
            "W&B runs:",
            "- https://wandb.ai/acme/project/runs/run-123",
        ]
    )
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False),
        comments=[
            comment(1, legacy_body),
            comment(2, legacy_body.replace("STUDENT: ", "", 1)),
        ],
    )
    client = workflow(fake, role="student")

    submitted = submit_result(client, terminal)
    mutations_after_upgrade = list(fake.mutations)
    replayed = submit_result(client, terminal)

    assert submitted.state == "result_submitted"
    assert submitted.changed is True
    assert replayed.changed is False
    expected = render_result_comment(terminal).replace(
        "\n\n", "\n\nSTUDENT:\n\n", 1
    )
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


def test_submit_result_rejects_an_ambiguous_unapplied_draft_mutation():
    class DroppedDraftMutation(FakeGitHub):
        dropped = False

        def request(self, method, url, *, headers, json_body=None):
            if not self.dropped and method == "POST" and urlsplit(url).path == "/graphql":
                self.dropped = True
                raise GitHubTransportError(method, url)
            return super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )

    fake = DroppedDraftMutation(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(ReconciliationError, match="ready for review"):
        submit_result(workflow(fake, role="student"))

    assert fake.pr["draft"] is True


@pytest.mark.parametrize("failure_mode", ["api", "transport"])
def test_submit_result_retry_repairs_a_partial_multi_comment_update(failure_mode):
    current = experiment_result()

    class SecondPatchFailsOnce(FakeGitHub):
        patch_count = 0

        def request(self, method, url, *, headers, json_body=None):
            if method == "PATCH" and "/issues/comments/" in urlsplit(url).path:
                self.patch_count += 1
                if self.patch_count == 2:
                    if failure_mode == "api":
                        return HttpResponse(500, {"message": "temporary failure"})
                    raise GitHubTransportError(method, url)
            return super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )

    fake = SecondPatchFailsOnce(
        pull_request(labels={"student:one", "status:wip"}, draft=True),
        comments=[
            comment(1, render_result_comment(current)),
            comment(2, render_result_comment(current)),
        ],
    )
    client = workflow(fake, role="student")

    error_type = GitHubAPIError if failure_mode == "api" else ReconciliationError
    with pytest.raises(error_type):
        submit_result(client, current)

    assert all(
        parse_result_markers(str(item["body"])) == (current,)
        for item in fake.comments
    )
    assert len({str(item["body"]) for item in fake.comments}) == 2

    submitted = submit_result(client, current)

    assert submitted.state == "result_submitted"
    assert all(
        parse_result_markers(str(item["body"])) == (current,)
        for item in fake.comments
    )


def test_stale_submit_cannot_overwrite_a_newer_result_before_upsert():
    current = experiment_result().model_copy(
        update={
            "assignment": experiment_result().assignment.model_copy(
                update={"revision_id": "revision-2"}
            ),
            "summary": "The current revision has durable evidence.",
        }
    )

    class CurrentResultBeforeStaleUpsert(FakeGitHub):
        advanced = False

        def request(self, method, url, *, headers, json_body=None):
            if (
                not self.advanced
                and method == "GET"
                and urlsplit(url).path == f"/repos/{REPO}/issues/7/comments"
            ):
                self.pr["body"] = render_assignment_marker(
                    assignment_record(revision_id="revision-2")
                )
                self.pr["draft"] = False
                self.pr["labels"] = {"student:one", "status:review"}
                self.comments = [comment(1, render_result_comment(current))]
                self.advanced = True
            return super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )

    fake = CurrentResultBeforeStaleUpsert(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        submit_result(workflow(fake, role="student"))

    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}
    assert parse_result_markers(str(fake.comments[0]["body"])) == (current,)
    assert fake.mutations == []


def test_submit_result_fails_closed_on_unexplained_distinct_results():
    first = experiment_result().model_copy(
        update={"summary": "First conflicting result."}
    )
    second = first.model_copy(
        update={"summary": "Second conflicting result."}
    )
    requested = second.model_copy(
        update={"summary": "Requested current result."}
    )
    fake = FakeGitHub(
        pull_request(),
        comments=[
            comment(1, render_result_comment(first)),
            comment(2, render_result_comment(second)),
        ],
    )

    with pytest.raises(ReconciliationError, match="multiple distinct"):
        submit_result(workflow(fake, role="student"), requested)

    assert fake.mutations == []


def test_submit_result_quotes_a_valid_prior_marker_in_its_summary():
    prior = experiment_result().model_copy(
        update={"summary": "Prior evidence quoted by the student."}
    )
    current = experiment_result().model_copy(
        update={
            "summary": (
                "The new evidence supersedes this prior marker:\n"
                f"{render_result_marker(prior)}"
            )
        }
    )
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )
    client = workflow(fake, role="student")

    first = submit_result(client, current)
    mutations_after_first = list(fake.mutations)
    second = submit_result(client, current)

    assert first.changed is True
    assert second.changed is False
    assert parse_result_markers(str(fake.comments[0]["body"])) == (current,)
    assert f"> {render_result_marker(prior)}" in str(fake.comments[0]["body"])
    assert fake.mutations == mutations_after_first


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


def test_submit_result_restores_wip_if_revision_changes_after_comment():
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
        pull_request(
            labels={"student:one", "status:review", "status:hold", "keep"},
            draft=False,
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
                method == "POST"
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


def test_submit_result_replays_and_merges_with_marker_text_in_visible_fields():
    injected_marker = "<!-- senpai-result:v2 {} -->"
    original = experiment_result()
    hostile = original.model_copy(
        update={
            "summary": f"Training completed.\n{injected_marker}\nMetrics follow.",
            "runs": (
                original.runs[0].model_copy(
                    update={"url": f"{original.runs[0].url}\n{injected_marker}"}
                ),
            ),
        }
    )
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )
    student = workflow(fake, role="student")

    first = submit_result(student, hostile)
    mutations_after_first = list(fake.mutations)
    second = submit_result(student, hostile)

    assert first.changed is True
    assert second.changed is False
    assert fake.mutations == mutations_after_first
    assert str(fake.comments[0]["body"]).splitlines().count(
        f"> {injected_marker}"
    ) == 2

    merged = workflow(fake).merge_experiment(
        7,
        expected_head_sha=HEAD_SHA,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id="revision-1",
        expected_current_base_sha=BASE_SHA,
    )

    assert merged.state == "experiment_merged"


def test_other_trusted_comments_cannot_smuggle_protocol_markers():
    injected_markers = (
        "<!-- senpai-result:v2 {} -->",
        "<!-- senpai-human-response:student:fern:700 -->",
    )
    fake = FakeGitHub(
        pull_request(
            labels={"student:student-one", "status:wip"},
            draft=True,
        )
    )
    workflow(fake).send_assignment_feedback(
        7,
        assignment_id=ASSIGNMENT_ID,
        revision_id="revision-1",
        expected_head_sha=HEAD_SHA,
        feedback_id="marker-looking-guidance",
        comment="Inspect the failed seed.\n" + "\n".join(injected_markers),
    )

    submitted = submit_result(workflow(fake, role="student"))

    assert submitted.state == "result_submitted"
    assert all(
        f"> {marker}" in str(fake.comments[0]["body"])
        for marker in injected_markers
    )
    assert workflow(fake).merge_experiment(
        7,
        expected_head_sha=HEAD_SHA,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id="revision-1",
        expected_current_base_sha=BASE_SHA,
    ).state == "experiment_merged"


def test_submit_result_preserves_a_hold_added_during_label_transition():
    class ConcurrentHoldGitHub(FakeGitHub):
        hold_added = False

        def request(self, method, url, *, headers, json_body=None):
            path = urlsplit(url).path
            if (
                not self.hold_added
                and method == "POST"
                and path == f"/repos/{REPO}/issues/7/labels"
            ):
                labels = self.pr["labels"]
                assert isinstance(labels, set)
                labels.add("status:hold")
                self.hold_added = True
            return super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )

    fake = ConcurrentHoldGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    submitted = submit_result(workflow(fake, role="student"))

    assert submitted.state == "result_submitted"
    assert fake.pr["labels"] == {
        "student:one",
        "status:hold",
        "status:review",
    }
    assert all(
        not (method == "PUT" and path.endswith("/labels"))
        for method, path, _body in fake.mutations
    )
    mutations_before_merge = list(fake.mutations)

    with pytest.raises(WorkflowPreconditionError, match="blocking label"):
        workflow(fake).merge_experiment(
            7,
            expected_head_sha=HEAD_SHA,
            assignment_id=ASSIGNMENT_ID,
            current_revision_id="revision-1",
            expected_current_base_sha=BASE_SHA,
        )

    assert fake.mutations == mutations_before_merge


def test_stale_submit_restores_a_current_result_arriving_after_draft_recovery():
    current = experiment_result()
    current = current.model_copy(
        update={
            "assignment": current.assignment.model_copy(
                update={"revision_id": "revision-2"}
            ),
            "summary": "Current evidence arrived during recovery.",
        }
    )

    class CurrentResultAfterDraftGitHub(FakeGitHub):
        revision_changed = False
        result_published = False

        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if method != "POST" or urlsplit(url).path != "/graphql":
                return response
            query = str(json_body)
            if "markPullRequestReadyForReview" in query and not self.revision_changed:
                self.pr["body"] = render_assignment_marker(
                    assignment_record(revision_id="revision-2")
                )
                self.revision_changed = True
            elif "convertPullRequestToDraft" in query and not self.result_published:
                self.comments = [comment(1, render_result_comment(current))]
                self.result_published = True
            return response

    fake = CurrentResultAfterDraftGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        submit_result(workflow(fake, role="student"))

    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}
    assert fake.comments == [comment(1, render_result_comment(current))]


def test_stale_submit_restores_a_current_result_arriving_after_wip_recovery():
    current = experiment_result()
    current = current.model_copy(
        update={
            "assignment": current.assignment.model_copy(
                update={"revision_id": "revision-2"}
            ),
            "summary": "Current evidence arrived after label recovery.",
        }
    )

    class CurrentResultAfterLabelsGitHub(FakeGitHub):
        revision_changed = False
        result_published = False

        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            path = urlsplit(url).path
            if method == "POST" and path == "/graphql" and not self.revision_changed:
                if "markPullRequestReadyForReview" in str(json_body):
                    self.pr["body"] = render_assignment_marker(
                        assignment_record(revision_id="revision-2")
                    )
                    self.revision_changed = True
            elif (
                method == "DELETE"
                and path.endswith("/labels/status%3Areview")
                and not self.result_published
            ):
                self.comments = [comment(1, render_result_comment(current))]
                self.result_published = True
            return response

    fake = CurrentResultAfterLabelsGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        submit_result(workflow(fake, role="student"))

    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}
    assert fake.comments == [comment(1, render_result_comment(current))]


def test_stale_submit_recovery_stops_after_three_assignment_changes():
    class RepeatedRevisionGitHub(FakeGitHub):
        revision = 1

        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if method != "POST" or urlsplit(url).path != "/graphql":
                return response
            query = str(json_body)
            if "markPullRequestReadyForReview" in query and self.revision == 1:
                self.revision = 2
            elif "convertPullRequestToDraft" in query:
                self.revision += 1
                self.pr["draft"] = False
            else:
                return response
            self.pr["body"] = render_assignment_marker(
                assignment_record(revision_id=f"revision-{self.revision}")
            )
            return response

    fake = RepeatedRevisionGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(ReconciliationError, match="kept changing"):
        submit_result(workflow(fake, role="student"))


def test_stale_submit_retries_when_assignment_changes_after_label_recovery():
    class RevisionAfterLabelsGitHub(FakeGitHub):
        initial_revision_changed = False
        recovery_revision_changed = False

        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            path = urlsplit(url).path
            if (
                method == "POST"
                and path == "/graphql"
                and "markPullRequestReadyForReview" in str(json_body)
                and not self.initial_revision_changed
            ):
                self.pr["body"] = render_assignment_marker(
                    assignment_record(revision_id="revision-2")
                )
                self.initial_revision_changed = True
            elif (
                method == "DELETE"
                and path.endswith("/labels/status%3Areview")
                and not self.recovery_revision_changed
            ):
                self.pr["body"] = render_assignment_marker(
                    assignment_record(revision_id="revision-3")
                )
                self.recovery_revision_changed = True
            return response

    fake = RevisionAfterLabelsGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        submit_result(workflow(fake, role="student"))

    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {"student:one", "status:wip"}
    assert "revision-3" in str(fake.pr["body"])


@pytest.mark.parametrize(
    "ignored_mutation",
    ["draft", "labels"],
)
def test_current_result_restore_rejects_unapplied_routing_mutations(
    ignored_mutation,
):
    current = experiment_result()
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True),
        comments=[comment(1, render_result_comment(current))],
        ignore_draft_mutations=ignored_mutation == "draft",
        ignore_label_mutations=ignored_mutation == "labels",
    )
    client = workflow(fake, role="student")

    with pytest.raises(ReconciliationError, match="did not remain reviewable"):
        client._restore_current_result_review(
            client.pull_request(7),
            assignment_id=ASSIGNMENT_ID,
            expected_head_sha=HEAD_SHA,
        )


def test_stale_recovery_restores_a_result_when_the_revision_reverts():
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True),
        comments=[comment(1, render_result_comment(experiment_result()))],
    )
    client = workflow(fake, role="student")

    client._reconcile_current_result_routing(
        7,
        assignment_id=ASSIGNMENT_ID,
        expected_head_sha=HEAD_SHA,
    )

    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}


def test_stale_recovery_rejects_an_unapplied_wip_label_rollback():
    fake = FakeGitHub(
        pull_request(
            labels={"student:one", "status:review"},
            draft=False,
            body=render_assignment_marker(
                assignment_record(revision_id="revision-2")
            ),
        ),
        comments=[comment(1, render_result_comment(experiment_result()))],
        ignore_label_mutations=True,
    )
    client = workflow(fake, role="student")

    with pytest.raises(ReconciliationError, match="restore.*to WIP"):
        client._reconcile_current_result_routing(
            7,
            assignment_id=ASSIGNMENT_ID,
            expected_head_sha=HEAD_SHA,
        )


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
            "ADVISOR:\n\nThe hypothesis was falsified.",
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
