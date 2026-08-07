"""Student assignment events derived from open pull requests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import parse_assignment_markers

from .feedback import student_pr_feedback_events
from .issues import human_issue_events
from .values import label_names, object_value, pull_payload

if TYPE_CHECKING:
    from .core import GitHubMailbox


def student_events(
    mailbox: GitHubMailbox,
    pulls: Sequence[dict[str, object]],
    issues: Sequence[dict[str, object]],
) -> tuple[ControllerEvent, ...]:
    assert mailbox.student_name is not None
    assignment_label = f"student:{mailbox.student_name}"
    assigned = [
        pull
        for pull in pulls
        if assignment_label in label_names(pull)
        and "status:wip" in label_names(pull)
    ]
    if len(assigned) > 1:
        numbers = sorted(int(pull["number"]) for pull in assigned)
        return (
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
            ),
            *human_issue_events(mailbox, issues),
        )

    events: list[ControllerEvent] = []
    if assigned:
        pull = assigned[0]
        try:
            markers = parse_assignment_markers(str(pull.get("body") or ""))
            if len(markers) != 1:
                raise ValueError(
                    "assigned PR must contain exactly one Senpai assignment marker"
                )
        except ValueError as error:
            number = int(pull["number"])
            head_sha = str(object_value(pull["head"])["sha"])
            events.append(
                ControllerEvent(
                    kind="malformed_assignment",
                    dedupe_key=f"malformed_assignment:{number}:{head_sha}",
                    payload={
                        **pull_payload(pull),
                        "error": f"Assigned PR #{number}: {error}",
                    },
                )
            )
        else:
            assignment = markers[0]
            feedback = student_pr_feedback_events(mailbox, pull, assignment)
            prior_revision_pending = any(
                event.payload["assignment_id"] != assignment.assignment_id
                or event.payload["revision_id"] != assignment.revision_id
                for event in feedback
            )
            if "status:wip" in label_names(pull) and not prior_revision_pending:
                events.append(
                    ControllerEvent(
                        kind="student_assignment",
                        dedupe_key=(
                            f"student_assignment:{assignment.assignment_id}:"
                            f"{assignment.revision_id}"
                        ),
                        payload={
                            **pull_payload(pull),
                            "assignment_id": assignment.assignment_id,
                            "revision_id": assignment.revision_id,
                            "base_ref": assignment.base_ref,
                        },
                    )
                )
            events.extend(feedback)
    events.extend(human_issue_events(mailbox, issues))
    return tuple(events)
