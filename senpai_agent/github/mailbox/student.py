"""Student assignment events derived from open pull requests."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import AssignmentRecord, parse_assignment_markers

from .feedback import student_pr_feedback_events
from .issues import human_issue_events
from .values import label_names, object_value, pull_reference, versioned_event

if TYPE_CHECKING:
    from .core import GitHubMailbox


_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def _validate_assignment_route(
    pull: Mapping[str, object],
    assignment: AssignmentRecord,
    *,
    repo: str,
    student: str,
) -> None:
    head = pull.get("head")
    base = pull.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise ValueError("assigned PR has invalid head or base metadata")
    expected = {
        "repo": (assignment.repo, repo),
        "student": (assignment.student, student),
        "head_ref": (assignment.head_ref, str(head.get("ref") or "")),
        "base_ref": (assignment.base_ref, str(base.get("ref") or "")),
    }
    mismatches = [
        name for name, (recorded, live) in expected.items() if recorded != live
    ]
    if mismatches:
        raise ValueError(
            "assignment marker does not match PR routing: " + ", ".join(mismatches)
        )
    for name, value in (
        ("assignment head SHA", assignment.head_sha),
        ("assignment base SHA", assignment.base_sha),
        ("live head SHA", str(head.get("sha") or "")),
    ):
        if _GIT_OBJECT_ID.fullmatch(value) is None:
            raise ValueError(f"{name} is not a full Git object ID")


def student_events(
    mailbox: GitHubMailbox,
    pulls: Sequence[dict[str, object]],
    issues: Sequence[dict[str, object]],
) -> tuple[ControllerEvent, ...]:
    assert mailbox.student_name is not None
    assignment_label = f"student:{mailbox.student_name}"
    relevant = [
        pull
        for pull in pulls
        if assignment_label in label_names(pull)
        and {"status:wip", "status:review"} & label_names(pull)
    ]
    wip = [pull for pull in relevant if "status:wip" in label_names(pull)]

    events: list[ControllerEvent] = []
    duplicate_wip = len(wip) > 1
    if duplicate_wip:
        numbers = sorted(int(pull["number"]) for pull in wip)
        events.append(
            ControllerEvent(
                kind="duplicate_assignment",
                dedupe_key=(
                    f"duplicate_assignment:{mailbox.student_name}:"
                    f"{','.join(map(str, numbers))}"
                ),
                payload={
                    "student": mailbox.student_name,
                    "pull_requests": numbers,
                },
            )
        )

    for pull in relevant:
        try:
            student_labels = {
                label
                for label in label_names(pull)
                if label.startswith("student:")
            }
            if student_labels != {assignment_label}:
                raise ValueError(
                    "assigned PR must contain exactly one student label"
                )
            markers = parse_assignment_markers(str(pull.get("body") or ""))
            if len(markers) != 1:
                raise ValueError(
                    "assigned PR must contain exactly one Senpai assignment marker"
                )
            assignment = markers[0]
            if assignment.student != mailbox.student_name:
                raise ValueError(
                    "assignment marker student does not match the student label"
                )
            _validate_assignment_route(
                pull,
                assignment,
                repo=mailbox.repo,
                student=mailbox.student_name,
            )
        except ValueError as error:
            number = int(pull["number"])
            head_sha = str(object_value(pull["head"])["sha"])
            payload = {
                **pull_reference(pull),
                "error": f"Assigned PR #{number}: {error}",
            }
            events.append(
                versioned_event(
                    "malformed_assignment", number, head_sha, payload=payload
                )
            )
            continue

        feedback = student_pr_feedback_events(mailbox, pull, assignment)
        prior_revision_pending = any(
            event.payload["assignment_id"] != assignment.assignment_id
            or event.payload["revision_id"] != assignment.revision_id
            for event in feedback
        )
        if (
            "status:wip" in label_names(pull)
            and not duplicate_wip
            and not prior_revision_pending
        ):
            number = int(pull["number"])
            head_sha = str(object_value(pull["head"])["sha"])
            blockers = sorted(
                label.removeprefix("status:")
                for label in label_names(pull)
                if label
                in {
                    "status:blocked",
                    "status:hold",
                    "status:needs-rebase",
                }
            )
            payload = {
                **pull_reference(pull),
                "assignment_id": assignment.assignment_id,
                "revision_id": assignment.revision_id,
                "base_ref": assignment.base_ref,
                "base_sha": assignment.base_sha,
                "blockers": blockers,
            }
            events.append(
                versioned_event(
                    "student_assignment",
                    number,
                    assignment.assignment_id,
                    assignment.revision_id,
                    assignment.base_ref,
                    assignment.head_ref,
                    assignment.base_sha,
                    head_sha,
                    payload=payload,
                )
            )
        events.extend(feedback)
    events.extend(human_issue_events(mailbox, issues))
    return tuple(events)
