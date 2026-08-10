"""Advisor events derived from open pull requests."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import quote

from senpai_agent.github.http import GitHubReadError
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import (
    AssignmentRecord,
    ResearchBaseAcceptanceRecord,
    ResultMarkerError,
    authoritative_marker_line,
    experiment_result_digest,
    parse_assignment_markers,
    parse_research_base_acceptance_markers,
    parse_result_markers,
)

from .values import (
    github_datetime,
    label_names,
    object_value,
    pull_reference,
    result_matches_assignment,
    versioned_event,
)
from .issues import human_issue_events

if TYPE_CHECKING:
    from .core import GitHubMailbox


def advisor_events(
    mailbox: GitHubMailbox,
    pulls: Sequence[dict[str, object]],
    issues: Sequence[dict[str, object]],
) -> tuple[ControllerEvent, ...]:
    events: list[ControllerEvent] = []
    active_assignments: list[tuple[dict[str, object], AssignmentRecord]] = []
    active_by_student: dict[str, list[int]] = {student: [] for student in mailbox.students}
    now = datetime.now(UTC)
    for pull in pulls:
        labels = label_names(pull)
        number = int(pull["number"])
        head_sha = str(object_value(pull["head"])["sha"])
        students = sorted(
            label.removeprefix("student:")
            for label in labels
            if label.startswith("student:")
        )
        if "status:wip" in labels:
            for student in students:
                active_by_student.setdefault(student, []).append(number)
        reference = pull_reference(pull)
        assignment = None
        if {"status:wip", "status:review"} & labels:
            try:
                assignments = parse_assignment_markers(str(pull.get("body") or ""))
            except ValueError:
                assignments = []
            if len(assignments) == 1:
                assignment = assignments[0]
                active_assignments.append((pull, assignment))
        if "status:review" in labels:
            review_payload = reference
            review_identity: tuple[object, ...] = (number, head_sha)
            if assignment is not None:
                review_identity = (
                    number,
                    assignment.assignment_id,
                    assignment.revision_id,
                    head_sha,
                )
                review_payload = {
                    **reference,
                    "assignment_id": assignment.assignment_id,
                    "revision_id": assignment.revision_id,
                }
            events.append(
                versioned_event("review_ready", *review_identity, payload=review_payload)
            )
        reasons: list[str] = []
        if "status:blocked" in labels:
            reasons.append("blocked")
        if "status:hold" in labels:
            reasons.append("hold")
        if "status:needs-rebase" in labels:
            reasons.append("needs_rebase")
        if not students:
            reasons.append("missing_student")
        if len(students) > 1:
            reasons.append("multiple_students")
        if "status:wip" in labels:
            updated = github_datetime(str(pull["updated_at"]))
            if (now - updated).total_seconds() >= mailbox.stale_wip_seconds:
                reasons.append("stale_wip")
        if reasons:
            action_payload = {**reference, "reasons": reasons}
            events.append(
                versioned_event(
                    "advisor_action",
                    number,
                    head_sha,
                    ",".join(reasons),
                    payload=action_payload,
                )
            )

    for student, numbers in active_by_student.items():
        if not numbers:
            events.append(
                ControllerEvent(
                    kind="idle_student",
                    dedupe_key=f"idle_student:{student}",
                    payload={"student": student},
                )
            )
        elif len(numbers) > 1:
            events.append(
                ControllerEvent(
                    kind="duplicate_assignment",
                    dedupe_key=(
                        f"duplicate_assignment:{student}:"
                        f"{','.join(map(str, sorted(numbers)))}"
                    ),
                    payload={
                        "student": student,
                        "pull_requests": sorted(numbers),
                    },
                )
            )

    events.extend(_research_base_events(mailbox, active_assignments))
    events.extend(human_issue_events(mailbox, issues))
    return tuple(events)


def _research_base_events(
    mailbox: GitHubMailbox,
    assignments: Sequence[tuple[dict[str, object], AssignmentRecord]],
) -> list[ControllerEvent]:
    events: list[ControllerEvent] = []
    current_bases: dict[str, str] = {}
    failed_bases: set[str] = set()
    for pull, assignment in assignments:
        if assignment.base_ref in failed_bases:
            continue
        if assignment.base_ref not in current_bases:
            try:
                current_bases[assignment.base_ref] = branch_head_sha(
                    mailbox, assignment.base_ref
                )
            except (GitHubReadError, TypeError) as error:
                failed_bases.add(assignment.base_ref)
                print(
                    "SENPAI_RESEARCH_BASE_WATCH_ERROR "
                    f"base_ref={assignment.base_ref!r} "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
        current_base_sha = current_bases[assignment.base_ref]
        if assignment.base_sha == current_base_sha:
            continue
        number = int(pull["number"])
        head_sha = str(object_value(pull["head"])["sha"])
        if (
            "status:review" in label_names(pull)
            and has_research_base_acceptance(
                mailbox,
                pull,
                assignment=assignment,
                head_sha=head_sha,
                current_base_sha=current_base_sha,
            )
        ):
            continue
        payload = {
            **pull_reference(pull),
            "assignment_id": assignment.assignment_id,
            "revision_id": assignment.revision_id,
            "student": assignment.student,
            "base_ref": assignment.base_ref,
            "required_base_sha": assignment.base_sha,
            "current_base_sha": current_base_sha,
            "compare_url": (
                f"{str(pull['html_url']).rsplit('/pull/', 1)[0]}"
                f"/compare/{assignment.base_sha}...{current_base_sha}"
            ),
        }
        events.append(
            versioned_event(
                "research_base_changed",
                number,
                assignment.assignment_id,
                assignment.revision_id,
                head_sha,
                assignment.base_ref,
                assignment.base_sha,
                current_base_sha,
                payload=payload,
            )
        )
    return events


def branch_head_sha(mailbox: GitHubMailbox, branch: str) -> str:
    ref = mailbox._github.get(
        f"/repos/{mailbox.repo}/git/ref/heads/{quote(branch, safe='')}"
    )
    if not isinstance(ref, dict):
        raise TypeError("GitHub research-base ref is not an object")
    target = ref.get("object")
    if not isinstance(target, dict) or not isinstance(target.get("sha"), str):
        raise TypeError("GitHub research-base ref has no target SHA")
    return target["sha"]


def has_research_base_acceptance(
    mailbox: GitHubMailbox,
    pull: Mapping[str, object],
    *,
    assignment: AssignmentRecord,
    head_sha: str,
    current_base_sha: str,
) -> bool:
    comments_url = pull.get("comments_url")
    if not comments_url:
        return False
    try:
        actor = mailbox._github.actor()
        comments = mailbox._github.objects(f"{comments_url}?per_page=100")
    except (GitHubReadError, TypeError) as error:
        print(
            "SENPAI_RESEARCH_BASE_ACCEPTANCE_READ_ERROR "
            f"pr={pull['number']} {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return False
    result_digests: list[str] = []
    for comment in comments:
        try:
            author = str(object_value(comment["user"])["login"])
        except (KeyError, TypeError):
            continue
        if author.casefold() != actor.casefold():
            continue
        first_line = authoritative_marker_line(str(comment.get("body") or ""))
        try:
            results = parse_result_markers(first_line)
        except ResultMarkerError:
            continue
        result_digests.extend(
            experiment_result_digest(result)
            for result in results
            if result_matches_assignment(
                result,
                repo=mailbox.repo,
                pr_number=int(pull["number"]),
                assignment=assignment,
                head_sha=head_sha,
            )
        )
    distinct_result_digests = set(result_digests)
    if len(distinct_result_digests) != 1:
        return False
    expected = ResearchBaseAcceptanceRecord(
        repo=mailbox.repo,
        pr_number=int(pull["number"]),
        assignment_id=assignment.assignment_id,
        revision_id=assignment.revision_id,
        result_head_sha=head_sha,
        result_digest=next(iter(distinct_result_digests)),
        evaluated_base_sha=assignment.base_sha,
        base_ref=assignment.base_ref,
        accepted_base_sha=current_base_sha,
    )
    for comment in comments:
        try:
            author = str(object_value(comment["user"])["login"])
        except (KeyError, TypeError):
            continue
        if author.casefold() != actor.casefold():
            continue
        try:
            acceptances = parse_research_base_acceptance_markers(
                authoritative_marker_line(str(comment.get("body") or ""))
            )
        except ValueError:
            continue
        if expected in acceptances:
            return True
    return False
