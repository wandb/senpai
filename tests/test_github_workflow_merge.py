from urllib.parse import urlsplit

import pytest
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

from senpai_agent.github.workflow import (
    ReconciliationError,
    StaleAssignmentRevisionError,
    StaleResearchBaseError,
    WorkflowPreconditionError,
)
from senpai_agent.models import (
    render_assignment_marker,
    render_result_comment,
    render_result_marker,
)


def mergeable_pull(**overrides):
    pr = pull_request(labels={"status:review"}, draft=False, mergeable=True)
    pr.update(overrides)
    return pr


def result_comment():
    return comment(1, render_result_comment(experiment_result()))


def merge_experiment(
    client,
    *,
    expected_head_sha: str = HEAD_SHA,
    revision_id: str = "revision-1",
    expected_current_base_sha: str = BASE_SHA,
):
    return client.merge_experiment(
        7,
        expected_head_sha=expected_head_sha,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id=revision_id,
        expected_current_base_sha=expected_current_base_sha,
    )


def accept_result(
    client,
    *,
    expected_current_base_sha: str,
    revision_id: str = "revision-1",
    reason: str = "The result remains valid against the current research base.",
):
    return client.accept_result_on_current_base(
        7,
        assignment_id=ASSIGNMENT_ID,
        current_revision_id=revision_id,
        expected_head_sha=HEAD_SHA,
        expected_current_base_sha=expected_current_base_sha,
        reason=reason,
    )


def test_merge_sends_the_expected_head_and_replays_without_baseline_reads():
    fake = FakeGitHub(mergeable_pull(), comments=[result_comment()])
    client = workflow(fake)

    first = merge_experiment(client)
    mutations_after_first = list(fake.mutations)
    base_reads_after_first = sum(
        method == "GET" and "/git/ref/heads/" in path
        for method, path, _body, _headers in fake.requests
    )
    second = merge_experiment(client)

    assert first.changed is True
    assert first.version == "merge-sha"
    assert second.changed is False
    assert fake.pr["state"] == "closed"
    assert fake.pr["merged"] is True
    assert mutations_after_first == [
        (
            "PUT",
            f"/repos/{REPO}/pulls/7/merge",
            {"sha": HEAD_SHA, "merge_method": "squash"},
        )
    ]
    assert fake.mutations == mutations_after_first
    assert sum(
        method == "GET" and "/git/ref/heads/" in path
        for method, path, _body, _headers in fake.requests
    ) == base_reads_after_first


@pytest.mark.parametrize(
    ("filename", "status"),
    [
        ("program.md", "added"),
        ("policy/program.md", "modified"),
        ("nested/program.md", "removed"),
    ],
)
def test_merge_rejects_every_program_policy_change(filename, status):
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        files=[{"filename": filename, "status": status}],
    )

    with pytest.raises(WorkflowPreconditionError, match="operator publication"):
        merge_experiment(workflow(fake))

    assert fake.pr["merged"] is False
    assert fake.mutations == []


@pytest.mark.parametrize(
    "changed_file",
    [
        {
            "filename": "policy.txt",
            "previous_filename": "program.md",
            "status": "renamed",
        },
        {
            "filename": "nested/program.md",
            "previous_filename": "policy.txt",
            "status": "renamed",
        },
    ],
    ids=("rename-away", "rename-into"),
)
def test_merge_rejects_program_policy_renames(changed_file):
    fake = FakeGitHub(
        mergeable_pull(), comments=[result_comment()], files=[changed_file]
    )

    with pytest.raises(WorkflowPreconditionError, match="operator publication"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_program_policy_check_matches_case_sensitive_typed_push_semantics():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        files=[{"filename": "policy/PROGRAM.md", "status": "modified"}],
    )

    assert merge_experiment(workflow(fake)).state == "experiment_merged"


def test_merge_checks_every_changed_file_page_for_program_policy():
    files = [
        {"filename": f"results/{index}.json", "status": "added"}
        for index in range(100)
    ]
    files.append({"filename": "nested/program.md", "status": "removed"})
    fake = FakeGitHub(mergeable_pull(), comments=[result_comment()], files=files)

    with pytest.raises(WorkflowPreconditionError, match="operator publication"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_recovers_when_the_success_response_is_lost():
    fake = AmbiguousMutationGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        fail_method="PUT",
        fail_path=f"/repos/{REPO}/pulls/7/merge",
    )

    merged = merge_experiment(workflow(fake))

    assert fake.failed is True
    assert merged.changed is True
    assert merged.version == "merge-sha"
    assert fake.pr["merged"] is True


def test_merge_rejects_changed_research_base_without_durable_acceptance():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )

    with pytest.raises(StaleResearchBaseError, match="no durable acceptance"):
        merge_experiment(
            workflow(fake),
            expected_current_base_sha=current_base_sha,
        )

    assert fake.mutations == []


def test_merge_rejects_a_stale_expected_current_research_base():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": "c" * 40},
    )

    with pytest.raises(StaleResearchBaseError, match="but live base is"):
        merge_experiment(
            workflow(fake),
            expected_current_base_sha="d" * 40,
        )

    assert fake.mutations == []


def test_accept_result_is_durable_idempotent_and_bound_to_the_live_base():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )

    client = workflow(fake)

    first = accept_result(client, expected_current_base_sha=current_base_sha)
    mutations_after_first = list(fake.mutations)
    second = accept_result(client, expected_current_base_sha=current_base_sha)

    assert first.changed is True
    assert first.state == "research_base_accepted"
    assert second.changed is False
    assert "<!-- senpai-research-base-acceptance:v1 " in str(
        fake.comments[-1]["body"]
    )
    assert fake.mutations == mutations_after_first


def test_accept_result_treats_identical_duplicate_results_as_one():
    current_base_sha = "c" * 40
    result = result_comment()
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result, comment(2, str(result["body"]))],
        branch_heads={"schmidhuber": current_base_sha},
    )

    accepted = accept_result(
        workflow(fake),
        expected_current_base_sha=current_base_sha,
    )

    assert accepted.state == "research_base_accepted"


def test_accept_result_rejects_stale_revision_and_base_before_writing():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    client = workflow(fake)

    with pytest.raises(StaleAssignmentRevisionError, match="revision"):
        accept_result(
            client,
            revision_id="revision-0",
            expected_current_base_sha=current_base_sha,
        )
    with pytest.raises(StaleResearchBaseError, match="but live base is"):
        accept_result(
            client,
            expected_current_base_sha="d" * 40,
        )

    assert fake.mutations == []


def test_acceptance_becomes_harmless_when_base_moves_during_reconciliation():
    class MovingBaseGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if method == "POST" and url.endswith("/issues/7/comments"):
                self.branch_heads["schmidhuber"] = "d" * 40
            return response

    fake = MovingBaseGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": "c" * 40},
    )

    with pytest.raises(StaleResearchBaseError, match="but live base is"):
        accept_result(
            workflow(fake),
            expected_current_base_sha="c" * 40,
        )

    assert "<!-- senpai-research-base-acceptance:v1 " in str(
        fake.comments[-1]["body"]
    )
    assert fake.pr["merged"] is False


def test_merge_consumes_exact_durable_research_base_acceptance():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    client = workflow(fake)

    accept_result(client, expected_current_base_sha=current_base_sha)
    result = merge_experiment(
        client,
        expected_current_base_sha=current_base_sha,
    )

    assert result.state == "experiment_merged"
    assert fake.pr["merged"] is True


@pytest.mark.parametrize("separator", ["\n\n", "\r", "\x85", "\u2028"])
def test_merge_rejects_an_acceptance_below_another_protocol_marker(separator):
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    client = workflow(fake)
    accept_result(client, expected_current_base_sha=current_base_sha)
    acceptance = fake.comments[-1]
    acceptance["body"] = (
        f"<!-- senpai-assignment-feedback:v1 {{}} -->{separator}"
        + str(acceptance["body"])
    )

    with pytest.raises(StaleResearchBaseError, match="no durable acceptance"):
        merge_experiment(client, expected_current_base_sha=current_base_sha)

    assert fake.pr["merged"] is False


def test_result_replay_preserves_a_durable_acceptance_with_legacy_marker_prose():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    advisor = workflow(fake)
    accept_result(advisor, expected_current_base_sha=current_base_sha)
    acceptance = fake.comments[-1]
    acceptance["body"] = (
        str(acceptance["body"])
        + "\n\nLegacy quoted evidence:\n"
        + render_result_marker(experiment_result())
    )
    acceptance_body = acceptance["body"]

    workflow(fake, role="student").submit_result(
        7,
        expected_head_sha=HEAD_SHA,
        result=experiment_result(),
    )

    assert acceptance["body"] == acceptance_body
    assert merge_experiment(
        advisor,
        expected_current_base_sha=current_base_sha,
    ).state == "experiment_merged"


def test_merge_treats_identical_duplicate_results_as_one():
    result = result_comment()
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result, comment(2, str(result["body"]))],
    )

    merged = merge_experiment(workflow(fake))

    assert merged.state == "experiment_merged"
    assert fake.pr["merged"] is True


def test_trusted_result_actor_login_is_case_insensitive():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[
            comment(
                1,
                render_result_comment(experiment_result()),
                author="SENPAI-BOT",
            )
        ],
    )

    merged = merge_experiment(workflow(fake))

    assert merged.state == "experiment_merged"


def test_acceptance_for_result_a_does_not_authorize_result_b_at_same_head():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    client = workflow(fake)
    accept_result(client, expected_current_base_sha=current_base_sha)
    result_b = experiment_result().model_copy(
        update={"summary": "A different result at the same commit."}
    )
    fake.comments[0]["body"] = render_result_comment(result_b)

    with pytest.raises(StaleResearchBaseError, match="no durable acceptance"):
        merge_experiment(client, expected_current_base_sha=current_base_sha)

    assert fake.pr["merged"] is False


def test_merge_ignores_malformed_acceptance_marker_when_exact_one_exists():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    client = workflow(fake)
    accept_result(client, expected_current_base_sha=current_base_sha)
    fake.comments.append(
        comment(99, "<!-- senpai-research-base-acceptance:v1 malformed -->")
    )

    merged = merge_experiment(client, expected_current_base_sha=current_base_sha)

    assert merged.state == "experiment_merged"
    assert fake.pr["merged"] is True


def test_merge_treats_duplicate_exact_acceptances_as_idempotent():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    client = workflow(fake)
    accept_result(client, expected_current_base_sha=current_base_sha)
    fake.comments.append(comment(99, str(fake.comments[-1]["body"])))

    merged = merge_experiment(client, expected_current_base_sha=current_base_sha)

    assert merged.state == "experiment_merged"
    assert fake.pr["merged"] is True


def test_merge_ignores_an_acceptance_marker_from_an_untrusted_author():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )
    client = workflow(fake)
    accept_result(client, expected_current_base_sha=current_base_sha)
    fake.comments[-1]["user"] = {"login": "untrusted-user", "type": "User"}

    with pytest.raises(StaleResearchBaseError, match="no durable acceptance"):
        merge_experiment(client, expected_current_base_sha=current_base_sha)

    assert fake.pr["merged"] is False


def test_merge_rechecks_the_research_base_immediately_before_mutation():
    class MovingBaseGitHub(FakeGitHub):
        base_reads = 0

        def request(self, method, url, *, headers, json_body=None):
            if method == "GET" and "/git/ref/heads/" in url:
                self.base_reads += 1
                if self.base_reads == 2:
                    self.branch_heads["schmidhuber"] = "c" * 40
            return super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )

    fake = MovingBaseGitHub(
        mergeable_pull(),
        comments=[result_comment()],
    )

    with pytest.raises(StaleResearchBaseError, match="but live base is"):
        merge_experiment(workflow(fake))

    assert fake.pr["merged"] is False
    assert fake.mutations == []


@pytest.mark.parametrize(
    ("change_phase", "expected_merged", "message"),
    [
        ("before", False, "immediately before merge"),
        ("after", True, "immediately after merge"),
    ],
)
def test_merge_detects_terminal_result_change_at_the_mutation_boundary(
    change_phase,
    expected_merged,
    message,
):
    changed = experiment_result().model_copy(
        update={"summary": "Evidence changed during merge."}
    )

    class ChangingResultGitHub(FakeGitHub):
        base_reads = 0

        def request(self, method, url, *, headers, json_body=None):
            path = urlsplit(url).path
            if method == "GET" and "/git/ref/heads/" in path:
                self.base_reads += 1
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if (
                change_phase == "before"
                and method == "GET"
                and "/git/ref/heads/" in path
                and self.base_reads == 2
            ) or (
                change_phase == "after"
                and method == "PUT"
                and path == f"/repos/{REPO}/pulls/7/merge"
            ):
                self.comments[0]["body"] = render_result_comment(changed)
            return response

    fake = ChangingResultGitHub(
        mergeable_pull(),
        comments=[result_comment()],
    )

    with pytest.raises(ReconciliationError, match=message):
        merge_experiment(workflow(fake))

    assert fake.pr["merged"] is expected_merged


@pytest.mark.parametrize(
    ("pr_overrides", "has_result", "expected_head_sha", "message"),
    [
        ({}, True, "b" * 40, "head SHA"),
        ({"draft": True}, True, HEAD_SHA, "draft"),
        (
            {"labels": {"status:review", "status:hold"}},
            True,
            HEAD_SHA,
            "blocking label",
        ),
        ({"labels": {"student:one"}}, True, HEAD_SHA, "status:review"),
        ({"mergeable": False}, True, HEAD_SHA, "merge conflict"),
        ({"mergeable": None}, True, HEAD_SHA, "unknown"),
        ({"state": "closed"}, True, HEAD_SHA, "open"),
        ({}, False, HEAD_SHA, "terminal result"),
    ],
    ids=(
        "stale-head",
        "draft",
        "blocking-label",
        "missing-review-label",
        "merge-conflict",
        "unknown-mergeability",
        "closed-unmerged",
        "missing-result",
    ),
)
def test_merge_rejects_unsafe_state_or_missing_evidence_before_writing(
    pr_overrides,
    has_result,
    expected_head_sha,
    message,
):
    fake = FakeGitHub(
        mergeable_pull(**pr_overrides),
        comments=[result_comment()] if has_result else [],
    )

    with pytest.raises(WorkflowPreconditionError, match=message):
        merge_experiment(
            workflow(fake),
            expected_head_sha=expected_head_sha,
        )

    assert fake.mutations == []


def test_merge_rejects_a_result_for_an_older_assignment_revision():
    current_assignment = assignment_record(revision_id="revision-2")
    fake = FakeGitHub(
        mergeable_pull(body=render_assignment_marker(current_assignment)),
        comments=[result_comment()],
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_rejects_a_result_for_an_older_head():
    stale = experiment_result(commit_sha="b" * 40)
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[comment(1, render_result_comment(stale))],
    )

    with pytest.raises(WorkflowPreconditionError, match="result commit"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_does_not_treat_assignment_prose_as_terminal_evidence():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[comment(1, f"prose mentions {ASSIGNMENT_ID}")],
    )

    with pytest.raises(WorkflowPreconditionError, match="terminal result"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_malformed_result_does_not_poison_unique_valid_result():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[
            result_comment(),
            comment(
                2,
                '<!-- senpai-result:v2 {"assignment_id":"assignment-7"} -->',
            ),
        ],
    )

    merged = merge_experiment(workflow(fake))

    assert merged.state == "experiment_merged"
    assert fake.pr["merged"] is True


def test_merge_fails_closed_on_conflicting_distinct_valid_results():
    conflicting = experiment_result().model_copy(
        update={"summary": "A conflicting terminal conclusion."}
    )
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[
            result_comment(),
            comment(2, render_result_comment(conflicting)),
        ],
    )

    with pytest.raises(ReconciliationError, match="multiple"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_ignores_a_result_marker_from_an_untrusted_author():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[
            comment(
                1,
                render_result_comment(experiment_result()),
                author="untrusted-user",
            )
        ],
    )

    with pytest.raises(WorkflowPreconditionError, match="terminal result"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []
