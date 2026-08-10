"""Student assignment events derived from open pull requests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import AssignmentRecord

from .feedback import student_pr_feedback_events
from .issues import human_issue_events
from .values import (
    assignment_from_pull,
    label_names,
    malformed_assignment_event,
    object_value,
    pull_reference,
    versioned_event,
)

if TYPE_CHECKING:
    from .core import GitHubMailbox


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
    events: list[ControllerEvent] = []
    assignments: list[tuple[dict[str, object], AssignmentRecord]] = []
    for pull in relevant:
        try:
            assignment = assignment_from_pull(
                pull,
                repo=mailbox.repo,
                expected_student=mailbox.student_name,
            )
        except ValueError as error:
            events.append(malformed_assignment_event(pull, error))
            continue
        assignments.append((pull, assignment))

    wip = [
        (pull, assignment)
        for pull, assignment in assignments
        if "status:wip" in label_names(pull)
    ]
    duplicate_wip = len(wip) > 1
    if duplicate_wip:
        numbers = sorted(int(pull["number"]) for pull, _assignment in wip)
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

    for pull, assignment in assignments:
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
