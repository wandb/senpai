# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from eval.adjudication import adjudicate_trial
from senpai_agent.models import (
    AssignmentKey,
    AssignmentRecord,
    ExperimentResult,
    MetricComparison,
    ResultStatus,
    WandbRunRef,
    render_assignment_marker,
    render_result_marker,
)

REPO = "acme/research-target"
BRANCH = "senpai-eval/eval-1/nano-t01"
GROUP = "eval-1/nanogpt/trial-01"
METRIC = "speedrun/final_first_step_to_target"
STUDENT = "eval-student"
ACTOR = "senpai-bot"
FROZEN_HEAD = "f" * 40


def target(**overrides):
    return {
        "repo_url": f"https://github.com/{REPO}.git",
        "primary_metric": METRIC,
        "direction": "minimize",
        **overrides,
    }


def trial(**overrides):
    return {
        "advisor_branch": BRANCH,
        "wandb_group": GROUP,
        "student_name": STUDENT,
        **overrides,
    }


def candidate(
    run_id: str,
    score: float,
    commit_sha: str,
    **overrides,
):
    return {
        "run_id": run_id,
        "score": score,
        "metric": METRIC,
        "group": GROUP,
        "commit_sha": commit_sha,
        "state": "finished",
        "url": f"https://wandb.ai/acme/project/runs/{run_id}",
        **overrides,
    }


def pull_request(
    number: int,
    *,
    head_sha: str,
    merge_sha: str | None,
    base_ref: str = BRANCH,
    merged: bool = True,
):
    head_ref = f"{STUDENT}/experiment-{number}"
    assignment = AssignmentRecord(
        repo=REPO,
        assignment_id=f"assignment-{number}",
        revision_id=f"revision-{number}",
        student=STUDENT,
        base_ref=base_ref,
        base_sha=f"base-{number}",
        head_ref=head_ref,
        head_sha=f"assignment-head-{number}",
    )
    return {
        "number": number,
        "node_id": f"PR_{number}",
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "title": f"Experiment {number}",
        "draft": False,
        "state": "closed",
        "merged": merged,
        "mergeable": None,
        "merge_commit_sha": merge_sha,
        "base": {"ref": base_ref},
        "head": {"ref": head_ref, "sha": head_sha},
        "labels": [],
        "body": render_assignment_marker(assignment),
    }


def experiment_result(
    number: int,
    *,
    head_sha: str,
    run_id: str,
    baseline: float = 4000,
    score: float = 3500,
    metric_name: str = METRIC,
    direction: str = "minimize",
    status: ResultStatus = ResultStatus.SUCCEEDED,
    primary: bool = True,
    repo: str = REPO,
):
    return ExperimentResult(
        assignment=AssignmentKey(
            repo=repo,
            pr_number=number,
            assignment_id=f"assignment-{number}",
            revision_id=f"revision-{number}",
            expected_head_sha=head_sha,
            student=STUDENT,
        ),
        status=status,
        hypothesis="The candidate improves the primary metric.",
        summary="Complete terminal evidence.",
        runs=(
            WandbRunRef(
                run_id=run_id,
                url=f"https://wandb.ai/acme/project/runs/{run_id}",
                state="finished",
            ),
        ),
        primary_metric=(
            MetricComparison(
                name=metric_name,
                direction=direction,
                baseline=baseline,
                candidate=score,
                delta=score - baseline,
            )
            if primary
            else None
        ),
        commit_sha=head_sha,
    )


def result_comment(
    result: ExperimentResult,
    *,
    actor: str = ACTOR,
    comment_id: int = 1,
):
    return {
        "id": comment_id,
        "html_url": f"https://github.com/{REPO}/pull/1#issuecomment-{comment_id}",
        "user": {"login": actor, "type": "User"},
        "author_association": "MEMBER",
        "body": render_result_marker(result),
    }


class FakeGitHubReader:
    def __init__(
        self,
        pulls=(),
        comments=None,
        *,
        ancestors=(),
        branch_head=FROZEN_HEAD,
    ):
        self.pulls = {pull["number"]: pull for pull in pulls}
        self.comments = comments or {}
        self.ancestors = set(ancestors)
        self.branch_head = branch_head
        self.calls = []

    def actor(self):
        self.calls.append(("actor", ""))
        return ACTOR

    def get(self, endpoint):
        self.calls.append(("get", endpoint))
        path = urlsplit(endpoint).path
        ref_prefix = f"/repos/{REPO}/git/ref/heads/"
        if path.startswith(ref_prefix):
            assert unquote(path.removeprefix(ref_prefix)) == BRANCH
            return {
                "ref": f"refs/heads/{BRANCH}",
                "object": {"sha": self.branch_head},
            }
        pull_prefix = f"/repos/{REPO}/pulls/"
        if path.startswith(pull_prefix):
            return self.pulls[int(path.removeprefix(pull_prefix))]
        compare_prefix = f"/repos/{REPO}/compare/"
        if path.startswith(compare_prefix):
            ancestor, descendant = path.removeprefix(compare_prefix).split("...")
            return {
                "status": (
                    "ahead"
                    if (ancestor, descendant) in self.ancestors
                    else "diverged"
                ),
                "merge_base_commit": {
                    "sha": (
                        ancestor
                        if (ancestor, descendant) in self.ancestors
                        else "0" * 40
                    )
                },
            }
        raise AssertionError(f"unexpected GitHub GET: {endpoint}")

    def objects(self, endpoint):
        self.calls.append(("objects", endpoint))
        parsed = urlsplit(endpoint)
        if parsed.path == f"/repos/{REPO}/pulls":
            query = parse_qs(parsed.query)
            assert query["state"] == ["closed"]
            assert query["base"] == [BRANCH]
            return [{"number": number} for number in reversed(self.pulls)]
        issue_prefix = f"/repos/{REPO}/issues/"
        if parsed.path.startswith(issue_prefix) and parsed.path.endswith("/comments"):
            number = int(parsed.path.removeprefix(issue_prefix).split("/", 1)[0])
            return self.comments.get(number, [])
        raise AssertionError(f"unexpected GitHub list: {endpoint}")


def adjudicate(reader, candidates, *, frozen_head_sha=None):
    return adjudicate_trial(
        target(),
        trial(),
        reader,
        candidates,
        frozen_head_sha=frozen_head_sha,
    )


def test_only_a_merged_structured_result_can_select_a_raw_candidate():
    head = "a" * 40
    merge = "1" * 40
    pull = pull_request(
        1,
        head_sha=head,
        merge_sha=merge,
    )
    result = experiment_result(1, head_sha=head, run_id="winner", score=3500)
    reader = FakeGitHubReader(
        [pull],
        {1: [result_comment(result)]},
        ancestors={(merge, FROZEN_HEAD)},
    )
    candidates = [
        candidate("debug-minimum", 1, "d" * 40, debug=True),
        candidate("test-minimum", 2, "e" * 40, test_run=True),
        candidate("winner", 3500, head),
    ]

    decision = adjudicate(reader, candidates)

    assert decision["status"] == "accepted"
    assert decision["selected_run_id"] == "winner"
    assert decision["score"] == 3500


def test_spoofed_result_comment_actor_cannot_nominate_a_run():
    head = "a" * 40
    merge = "1" * 40
    pull = pull_request(
        1,
        head_sha=head,
        merge_sha=merge,
    )
    result = experiment_result(1, head_sha=head, run_id="spoofed")
    reader = FakeGitHubReader(
        [pull],
        {1: [result_comment(result, actor="mallory")]},
        ancestors={(merge, FROZEN_HEAD)},
    )

    decision = adjudicate(reader, [candidate("spoofed", 3500, head)])

    assert decision["status"] == "rejected"
    pull_evidence = decision["evidence"]["pulls"][0]
    assert pull_evidence["ignored_comment_actors"] == ["mallory"]


def test_unmerged_result_is_not_adjudicated():
    head = "a" * 40
    pull = pull_request(1, head_sha=head, merge_sha=None, merged=False)
    result = experiment_result(1, head_sha=head, run_id="unmerged")
    reader = FakeGitHubReader([pull], {1: [result_comment(result)]})

    decision = adjudicate(reader, [candidate("unmerged", 3500, head)])

    assert decision["status"] == "rejected"
    assert "pull request is not merged" in decision["evidence"]["pulls"][0][
        "rejections"
    ]


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("branch", "wrong advisor branch"),
        ("commit", "result head or commit mismatch"),
        ("wandb_commit", "W&B source commit mismatch"),
        ("metric", "primary metric name or direction mismatch"),
        ("direction", "primary metric name or direction mismatch"),
        ("repo", "result assignment repository or PR mismatch"),
        ("run", "no referenced scored W&B candidate"),
        ("group", "W&B group mismatch"),
        ("url", "W&B URL mismatch"),
        ("score", "W&B score disagrees with structured candidate"),
    ],
)
def test_rejects_wrong_branch_commit_metric_run_group_or_score(case, reason):
    head = "a" * 40
    merge = "1" * 40
    base_ref = "another/advisor" if case == "branch" else BRANCH
    pull = pull_request(
        1,
        head_sha=head,
        merge_sha=merge,
        base_ref=base_ref,
    )
    result = experiment_result(
        1,
        head_sha="b" * 40 if case == "commit" else head,
        run_id="another-run" if case == "run" else "candidate",
        metric_name="another/metric" if case == "metric" else METRIC,
        direction="maximize" if case == "direction" else "minimize",
        repo="another/repo" if case == "repo" else REPO,
    )
    scored = candidate(
        "candidate",
        3499 if case == "score" else 3500,
        "b" * 40 if case == "wandb_commit" else head,
        group="another/group" if case == "group" else GROUP,
        url=(
            "https://wandb.ai/other/project/runs/candidate"
            if case == "url"
            else "https://wandb.ai/acme/project/runs/candidate"
        ),
    )
    reader = FakeGitHubReader(
        [pull],
        {1: [result_comment(result)]},
        ancestors={(merge, FROZEN_HEAD)},
    )

    decision = adjudicate(reader, [scored])

    assert decision["status"] == "rejected"
    serialized = repr(decision["evidence"])
    assert reason in serialized


def test_non_succeeded_result_is_rejected():
    head = "a" * 40
    merge = "1" * 40
    pull = pull_request(1, head_sha=head, merge_sha=merge)
    result = experiment_result(
        1,
        head_sha=head,
        run_id="failed",
        status=ResultStatus.FAILED,
    )
    reader = FakeGitHubReader(
        [pull],
        {1: [result_comment(result)]},
        ancestors={(merge, FROZEN_HEAD)},
    )

    decision = adjudicate(reader, [candidate("failed", 3500, head)])

    assert decision["status"] == "rejected"
    assert "result status is not succeeded" in repr(decision["evidence"])


def test_malformed_authenticated_marker_rejects_the_pr_fail_closed():
    head = "a" * 40
    merge = "1" * 40
    pull = pull_request(1, head_sha=head, merge_sha=merge)
    result = experiment_result(1, head_sha=head, run_id="winner")
    malformed = result_comment(result, comment_id=2)
    malformed["body"] = "<!-- senpai-result:v1 {} -->"
    reader = FakeGitHubReader(
        [pull],
        {1: [result_comment(result), malformed]},
        ancestors={(merge, FROZEN_HEAD)},
    )

    decision = adjudicate(reader, [candidate("winner", 3500, head)])

    assert decision["status"] == "rejected"
    pull_evidence = decision["evidence"]["pulls"][0]
    assert pull_evidence["malformed_authenticated_comment_ids"] == [2]
    assert "malformed authenticated structured result" in pull_evidence["rejections"]


def test_latest_sequential_merged_winner_is_selected():
    first_head, second_head = "a" * 40, "b" * 40
    first_merge, second_merge = "1" * 40, "2" * 40
    first = pull_request(
        2,
        head_sha=first_head,
        merge_sha=first_merge,
    )
    second = pull_request(
        1,
        head_sha=second_head,
        merge_sha=second_merge,
    )
    results = {
        2: [
            result_comment(
                experiment_result(
                    2, head_sha=first_head, run_id="first", baseline=4000, score=3600
                )
            )
        ],
        1: [
            result_comment(
                experiment_result(
                    1,
                    head_sha=second_head,
                    run_id="second",
                    baseline=3600,
                    score=3400,
                )
            )
        ],
    }
    reader = FakeGitHubReader(
        [first, second],
        results,
        ancestors={
            (first_merge, second_merge),
            (first_merge, FROZEN_HEAD),
            (second_merge, FROZEN_HEAD),
        },
    )

    decision = adjudicate(
        reader,
        [candidate("first", 3600, first_head), candidate("second", 3400, second_head)],
    )

    assert decision["status"] == "accepted"
    assert decision["pr_number"] == 1
    assert decision["selected_run_id"] == "second"


def test_incomparable_accepted_merges_are_rejected_as_ambiguous():
    first_head, second_head = "a" * 40, "b" * 40
    first_merge, second_merge = "1" * 40, "2" * 40
    pulls = [
        pull_request(1, head_sha=first_head, merge_sha=first_merge),
        pull_request(2, head_sha=second_head, merge_sha=second_merge),
    ]
    comments = {
        1: [
            result_comment(
                experiment_result(1, head_sha=first_head, run_id="first")
            )
        ],
        2: [
            result_comment(
                experiment_result(2, head_sha=second_head, run_id="second")
            )
        ],
    }
    reader = FakeGitHubReader(
        pulls,
        comments,
        ancestors={(first_merge, FROZEN_HEAD), (second_merge, FROZEN_HEAD)},
    )

    decision = adjudicate(
        reader,
        [candidate("first", 3500, first_head), candidate("second", 3500, second_head)],
    )

    assert decision["status"] == "rejected"
    assert decision["selected_run_id"] is None
    assert "no unique latest merge" in decision["evidence"]["selection_rejection"]


def test_later_cleanup_pr_without_a_metric_does_not_replace_the_winner():
    winner_head, cleanup_head = "a" * 40, "b" * 40
    winner_merge, cleanup_merge = "1" * 40, "2" * 40
    winner = pull_request(
        1,
        head_sha=winner_head,
        merge_sha=winner_merge,
    )
    cleanup = pull_request(
        2,
        head_sha=cleanup_head,
        merge_sha=cleanup_merge,
    )
    comments = {
        1: [
            result_comment(
                experiment_result(1, head_sha=winner_head, run_id="winner")
            )
        ],
        2: [
            result_comment(
                experiment_result(
                    2,
                    head_sha=cleanup_head,
                    run_id="cleanup",
                    primary=False,
                )
            )
        ],
    }
    reader = FakeGitHubReader(
        [winner, cleanup],
        comments,
        ancestors={(winner_merge, FROZEN_HEAD), (cleanup_merge, FROZEN_HEAD)},
    )

    decision = adjudicate(
        reader,
        [
            candidate("winner", 3500, winner_head),
            candidate("cleanup", 1, cleanup_head),
        ],
    )

    assert decision["status"] == "accepted"
    assert decision["selected_run_id"] == "winner"
    assert "missing primary metric" in repr(decision["evidence"])


def test_merge_must_be_reachable_from_the_frozen_advisor_head():
    head = "a" * 40
    merge = "1" * 40
    pull = pull_request(
        1,
        head_sha=head,
        merge_sha=merge,
    )
    result = experiment_result(1, head_sha=head, run_id="orphan")
    reader = FakeGitHubReader([pull], {1: [result_comment(result)]})

    decision = adjudicate(reader, [candidate("orphan", 3500, head)])

    assert decision["status"] == "rejected"
    assert "not in the frozen advisor branch" in repr(decision["evidence"])


def test_no_merged_winner_is_rejected_and_unscored_without_network_access():
    reader = FakeGitHubReader()

    decision = adjudicate(reader, [])

    assert decision["status"] == "rejected"
    assert decision["selected_run_id"] is None
    assert decision["score"] is None
    assert decision["evidence"]["frozen_advisor_head"] == FROZEN_HEAD
    assert reader.calls[0][0] == "get"
    assert "/git/ref/heads/" in reader.calls[0][1]
    assert all(type(call).__name__ == "tuple" for call in reader.calls)


def test_provided_frozen_head_makes_replay_independent_of_the_live_ref():
    head = "a" * 40
    merge = "1" * 40
    pull = pull_request(1, head_sha=head, merge_sha=merge)
    comments = {1: [result_comment(experiment_result(1, head_sha=head, run_id="run"))]}
    ancestors = {(merge, FROZEN_HEAD)}
    first = FakeGitHubReader(
        [pull],
        comments,
        ancestors=ancestors,
        branch_head="c" * 40,
    )
    second = FakeGitHubReader(
        [pull],
        comments,
        ancestors=ancestors,
        branch_head="d" * 40,
    )

    first_decision = adjudicate(
        first,
        [candidate("run", 3500, head)],
        frozen_head_sha=FROZEN_HEAD,
    )
    second_decision = adjudicate(
        second,
        [candidate("run", 3500, head)],
        frozen_head_sha=FROZEN_HEAD,
    )

    assert first_decision == second_decision
    assert first_decision["evidence"]["frozen_advisor_head_source"] == "provided"
    assert not any("/git/ref/heads/" in endpoint for _, endpoint in first.calls)
