"""Select the final eval result from Senpai's authenticated GitHub ledger."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import quote, urlencode, urlsplit

from senpai_agent.github.workflow.responses import (
    GitRefResponse,
    IssueCommentResponse,
    NumberedResponse,
    PullRequestResponse,
    validated_response,
)
from senpai_agent.models import (
    AssignmentRecord,
    ExperimentResult,
    ResultMarkerError,
    ResultStatus,
    authoritative_marker_line,
    experiment_result_digest,
    parse_assignment_markers,
    parse_result_markers,
)

AncestryCheck = Callable[[str, str], bool]


class GitHubReads(Protocol):
    """The read-only GitHub surface needed for adjudication."""

    def get(self, path: str) -> object: ...

    def objects(self, path: str) -> list[dict[str, object]]: ...

    def actor(self) -> str: ...


def freeze_advisor_head(
    target: Mapping[str, Any],
    trial: Mapping[str, Any],
    github: GitHubReads,
) -> str:
    """Read the exact advisor-branch commit that bounds adjudication."""

    return _branch_head(
        github,
        _target_repo(target),
        _required_text(trial, "advisor_branch"),
    )


def adjudicate_trial(
    target: Mapping[str, Any],
    trial: Mapping[str, Any],
    github: GitHubReads,
    scored_candidates: Sequence[Mapping[str, Any]],
    *,
    frozen_head_sha: str | None = None,
    is_ancestor: AncestryCheck | None = None,
) -> dict[str, Any]:
    """Return the latest merged, authenticated, baseline-changing result.

    The advisor branch is frozen before any PR reads. W&B candidates provide
    metric evidence, but only a merged Senpai structured result can nominate
    one of them for the final score.
    """

    repo = _target_repo(target)
    branch = _required_text(trial, "advisor_branch")
    group = _required_text(trial, "wandb_group")
    student = _required_text(trial, "student_name")
    metric_name = _required_text(target, "primary_metric")
    direction = str(target.get("direction", "minimize"))
    if direction not in {"minimize", "maximize"}:
        raise ValueError("target direction must be minimize or maximize")

    if frozen_head_sha is None:
        frozen_head = freeze_advisor_head(target, trial, github)
        frozen_head_source = "github_ref"
    else:
        frozen_head = _nonempty_text(frozen_head_sha, "frozen_head_sha")
        frozen_head_source = "provided"
    trusted_actor = _nonempty_text(github.actor(), "authenticated GitHub actor")
    candidates = _candidate_index(scored_candidates)
    evidence: dict[str, Any] = {
        "repo": repo,
        "advisor_branch": branch,
        "frozen_advisor_head": frozen_head,
        "frozen_advisor_head_source": frozen_head_source,
        "trusted_actor": trusted_actor,
        "student": student,
        "wandb_group": group,
        "primary_metric": metric_name,
        "direction": direction,
        "ancestry_checks": [],
        "pulls": [],
    }
    ancestry_cache: dict[tuple[str, str], bool] = {}

    def reachable(ancestor: str, descendant: str) -> bool:
        key = (ancestor, descendant)
        if key not in ancestry_cache:
            result = (
                is_ancestor(ancestor, descendant)
                if is_ancestor is not None
                else _github_is_ancestor(github, repo, ancestor, descendant)
            )
            if not isinstance(result, bool):
                raise TypeError("is_ancestor must return bool")
            ancestry_cache[key] = result
            evidence["ancestry_checks"].append(
                {
                    "ancestor": ancestor,
                    "descendant": descendant,
                    "reachable": result,
                }
            )
        return ancestry_cache[key]

    query = urlencode(
        {
            "state": "closed",
            "base": branch,
            "per_page": 100,
        }
    )
    pull_numbers = sorted(
        {
            validated_response(NumberedResponse, item, "pull request list item").number
            for item in github.objects(f"/repos/{repo}/pulls?{query}")
        }
    )
    accepted: list[dict[str, Any]] = []
    for number in pull_numbers:
        pull = validated_response(
            PullRequestResponse,
            github.get(f"/repos/{repo}/pulls/{number}"),
            "pull request",
        ).snapshot()
        pull_evidence: dict[str, Any] = {"pr_number": number, "rejections": []}
        evidence["pulls"].append(pull_evidence)

        base_ref = pull.base_ref
        head_ref = pull.head_ref
        head_sha = pull.head_sha
        merge_sha = pull.merge_commit_sha
        pull_evidence.update(
            {
                "base_ref": base_ref,
                "head_ref": head_ref,
                "head_sha": head_sha,
                "merge_commit_sha": merge_sha,
            }
        )
        if base_ref != branch:
            pull_evidence["rejections"].append("wrong advisor branch")
        if pull.state != "closed" or not pull.merged:
            pull_evidence["rejections"].append("pull request is not merged")
        if not isinstance(merge_sha, str) or not merge_sha:
            pull_evidence["rejections"].append("missing merge commit")
        elif not reachable(merge_sha, frozen_head):
            pull_evidence["rejections"].append(
                "merge commit is not in the frozen advisor branch"
            )
        if pull_evidence["rejections"]:
            continue

        assignment = _assignment_for_pull(pull.body, pull_evidence)
        if assignment is None:
            continue
        if assignment.repo != repo:
            pull_evidence["rejections"].append("assignment repository mismatch")
        if assignment.base_ref != branch:
            pull_evidence["rejections"].append("assignment base ref mismatch")
        if assignment.head_ref != head_ref:
            pull_evidence["rejections"].append("assignment head ref mismatch")
        if assignment.student != student:
            pull_evidence["rejections"].append("assignment student mismatch")
        if pull_evidence["rejections"]:
            continue

        results, malformed = _authenticated_results(
            github, repo, number, trusted_actor, pull_evidence
        )
        if malformed:
            pull_evidence["rejections"].append(
                "malformed authenticated structured result"
            )
            continue
        valid_results: list[tuple[ExperimentResult, Mapping[str, Any]]] = []
        for result in sorted(results, key=experiment_result_digest):
            result_rejections, candidate = _validate_result(
                result,
                repo=repo,
                pr_number=number,
                assignment=assignment,
                head_sha=head_sha,
                metric_name=metric_name,
                direction=direction,
                group=group,
                candidates=candidates,
            )
            if result_rejections:
                pull_evidence.setdefault("result_rejections", []).append(
                    {
                        "result_digest": experiment_result_digest(result),
                        "reasons": result_rejections,
                    }
                )
            elif candidate is not None:
                valid_results.append((result, candidate))

        distinct = {
            experiment_result_digest(result): (result, candidate)
            for result, candidate in valid_results
        }
        if not distinct:
            pull_evidence["rejections"].append(
                "no authenticated eligible structured result"
            )
            continue
        if len(distinct) != 1:
            pull_evidence["rejections"].append(
                "multiple distinct eligible structured results"
            )
            continue

        result, candidate = next(iter(distinct.values()))
        accepted_entry = {
            "pr_number": number,
            "merge_commit_sha": merge_sha,
            "result_commit_sha": result.commit_sha,
            "result_digest": experiment_result_digest(result),
            "run_id": candidate["run_id"],
            "score": float(candidate["score"]),
        }
        pull_evidence["accepted_candidate"] = accepted_entry
        accepted.append(accepted_entry)

    if not accepted:
        return {
            "status": "rejected",
            "reason": "no accepted merged structured result",
            "selected_run_id": None,
            "score": None,
            "evidence": evidence,
        }

    latest = [
        candidate
        for candidate in accepted
        if all(
            other is candidate
            or reachable(
                str(other["merge_commit_sha"]),
                str(candidate["merge_commit_sha"]),
            )
            for other in accepted
        )
    ]
    if len(latest) != 1:
        evidence["selection_rejection"] = (
            "accepted results have no unique latest merge in the frozen advisor "
            "branch"
        )
        return {
            "status": "rejected",
            "reason": "accepted structured results are not uniquely ordered",
            "selected_run_id": None,
            "score": None,
            "evidence": evidence,
        }

    selected = latest[0]
    evidence["selected"] = selected
    return {
        "status": "accepted",
        "reason": "latest baseline-changing merged structured result",
        "selected_run_id": selected["run_id"],
        "score": selected["score"],
        "pr_number": selected["pr_number"],
        "result_commit_sha": selected["result_commit_sha"],
        "merge_commit_sha": selected["merge_commit_sha"],
        "result_digest": selected["result_digest"],
        "evidence": evidence,
    }


def _validate_result(
    result: ExperimentResult,
    *,
    repo: str,
    pr_number: int,
    assignment: AssignmentRecord,
    head_sha: str,
    metric_name: str,
    direction: str,
    group: str,
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], Mapping[str, Any] | None]:
    reasons: list[str] = []
    key = result.assignment
    if result.status != ResultStatus.SUCCEEDED:
        reasons.append("result status is not succeeded")
    if key.repo != repo or key.pr_number != pr_number:
        reasons.append("result assignment repository or PR mismatch")
    if key.assignment_id != assignment.assignment_id:
        reasons.append("result assignment ID mismatch")
    if key.revision_id != assignment.revision_id:
        reasons.append("result assignment revision mismatch")
    if key.student != assignment.student:
        reasons.append("result assignment student mismatch")
    if key.expected_head_sha != head_sha or result.commit_sha != head_sha:
        reasons.append("result head or commit mismatch")

    metric = result.primary_metric
    if metric is None:
        reasons.append("missing primary metric")
        return reasons, None
    if metric.name != metric_name or metric.direction != direction:
        reasons.append("primary metric name or direction mismatch")
    baseline = _finite_number(metric.baseline)
    candidate_value = _finite_number(metric.candidate)
    if not _baseline_improved(baseline, candidate_value, direction):
        reasons.append("primary metric did not improve its baseline")
    if (
        metric.delta is not None
        and baseline is not None
        and candidate_value is not None
        and not math.isclose(
            metric.delta,
            candidate_value - baseline,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        reasons.append("primary metric delta is inconsistent")

    matching: list[Mapping[str, Any]] = []
    for run in result.runs:
        candidate = candidates.get(run.run_id)
        if candidate is None:
            continue
        run_reasons = []
        if run.state != "finished" or candidate.get("state") != "finished":
            run_reasons.append("run is not finished")
        if candidate.get("group") != group:
            run_reasons.append("W&B group mismatch")
        if candidate.get("url") != run.url:
            run_reasons.append("W&B URL mismatch")
        if candidate.get("commit_sha") != result.commit_sha:
            run_reasons.append("W&B source commit mismatch")
        if candidate.get("metric") != metric_name:
            run_reasons.append("W&B metric mismatch")
        score = _finite_number(candidate.get("score"))
        if score is None or not math.isclose(
            score, metric.candidate, rel_tol=1e-9, abs_tol=1e-9
        ):
            run_reasons.append("W&B score disagrees with structured candidate")
        if run_reasons:
            reasons.extend(f"run {run.run_id}: {reason}" for reason in run_reasons)
        else:
            matching.append(candidate)

    if not matching:
        reasons.append("no referenced scored W&B candidate")
        return reasons, None
    if len(matching) != 1:
        reasons.append("multiple scored W&B candidates match the primary metric")
        return reasons, None
    return reasons, matching[0]


def _authenticated_results(
    github: GitHubReads,
    repo: str,
    number: int,
    trusted_actor: str,
    evidence: dict[str, Any],
) -> tuple[tuple[ExperimentResult, ...], bool]:
    results: list[ExperimentResult] = []
    ignored_actors: set[str] = set()
    malformed_comment_ids: list[int] = []
    for value in github.objects(
        f"/repos/{repo}/issues/{number}/comments?per_page=100"
    ):
        comment = validated_response(
            IssueCommentResponse, value, "issue comment"
        ).comment()
        if comment.author.casefold() != trusted_actor.casefold():
            ignored_actors.add(comment.author)
            continue
        try:
            results.extend(
                parse_result_markers(
                    authoritative_marker_line(comment.body)
                )
            )
        except ResultMarkerError:
            malformed_comment_ids.append(comment.id)
    evidence["authenticated_result_markers"] = len(results)
    if ignored_actors:
        evidence["ignored_comment_actors"] = sorted(ignored_actors)
    if malformed_comment_ids:
        evidence["malformed_authenticated_comment_ids"] = sorted(
            malformed_comment_ids
        )
    return tuple(results), bool(malformed_comment_ids)


def _assignment_for_pull(
    body: str, evidence: dict[str, Any]
) -> AssignmentRecord | None:
    try:
        assignments = parse_assignment_markers(body)
    except ValueError:
        evidence["rejections"].append("malformed assignment marker")
        return None
    if len(assignments) != 1:
        evidence["rejections"].append("expected exactly one assignment marker")
        return None
    return assignments[0]


def _candidate_index(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        run_id = _required_text(candidate, "run_id")
        if run_id in indexed:
            raise ValueError(f"duplicate scored W&B candidate: {run_id}")
        indexed[run_id] = candidate
    return indexed


def _branch_head(github: GitHubReads, repo: str, branch: str) -> str:
    ref = validated_response(
        GitRefResponse,
        github.get(f"/repos/{repo}/git/ref/heads/{quote(branch, safe='')}"),
        "advisor branch ref",
    )
    return ref.object.sha


def _github_is_ancestor(
    github: GitHubReads, repo: str, ancestor: str, descendant: str
) -> bool:
    comparison = _object(
        github.get(
            f"/repos/{repo}/compare/{quote(ancestor, safe='')}..."
            f"{quote(descendant, safe='')}"
        ),
        "commit comparison",
    )
    status = comparison.get("status")
    if status not in {"ahead", "behind", "diverged", "identical"}:
        raise ValueError("commit comparison has an invalid status")
    merge_base_sha = _nested_text(comparison, "merge_base_commit", "sha")
    return status in {"ahead", "identical"} and merge_base_sha == ancestor


def _target_repo(target: Mapping[str, Any]) -> str:
    if repo := target.get("repo"):
        return _nonempty_text(repo, "repo")
    path = urlsplit(_required_text(target, "repo_url")).path.strip("/")
    path = path.removesuffix(".git")
    if len(path.split("/")) != 2:
        raise ValueError("target repo_url must identify one owner/repository")
    return path


def _baseline_improved(
    baseline: object, candidate: object, direction: str
) -> bool:
    baseline_number = _finite_number(baseline)
    candidate_number = _finite_number(candidate)
    if baseline_number is None or candidate_number is None:
        return False
    if direction == "minimize":
        return candidate_number < baseline_number
    return candidate_number > baseline_number


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _required_text(value: Mapping[str, Any], key: str) -> str:
    return _nonempty_text(value.get(key), key)


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _nested_text(value: Mapping[str, Any], owner: str, key: str) -> str:
    nested = value.get(owner)
    if not isinstance(nested, Mapping):
        raise ValueError(f"{owner} must be an object")
    return _required_text(nested, key)


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
