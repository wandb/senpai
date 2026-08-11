"""Trusted student assignment comments delivered to advisor controllers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from senpai_agent.github.http import GitHubReadError
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import (
    AssignmentCommentRecord,
    AssignmentRecord,
    authoritative_marker_line,
    parse_assignment_comment_markers,
    render_assignment_comment_marker,
)

from .values import (
    FEEDBACK_EXCERPT_BYTES,
    bounded_text,
    github_datetime,
    object_value,
    payload_digest,
)

if TYPE_CHECKING:
    from .core import GitHubMailbox


def student_assignment_comment_events(
    mailbox: GitHubMailbox,
    assignments: Sequence[tuple[dict[str, object], AssignmentRecord]],
) -> list[ControllerEvent]:
    """Wake the advisor for trusted typed messages from assigned students."""

    with_comments = [
        (pull, assignment)
        for pull, assignment in assignments
        if pull.get("comments_url")
    ]
    if not with_comments:
        return []
    try:
        actor = mailbox._github.actor()
    except (GitHubReadError, TypeError) as error:
        _report_read_error(f"actor {type(error).__name__}: {error}")
        return []

    events_by_key: dict[str, ControllerEvent] = {}
    for pull, assignment in with_comments:
        comments_url = str(pull["comments_url"])
        number = int(pull["number"])
        try:
            comments = mailbox._github.objects(f"{comments_url}?per_page=100")
        except (GitHubReadError, TypeError) as error:
            _report_read_error(f"pr={number} {type(error).__name__}: {error}")
            continue
        for item in comments:
            event = _comment_event(
                mailbox,
                pull,
                assignment,
                item,
                actor=actor,
            )
            if event is not None:
                previous = events_by_key.get(event.dedupe_key)
                if previous is None or int(event.payload["github_comment_id"]) < int(
                    previous.payload["github_comment_id"]
                ):
                    events_by_key[event.dedupe_key] = event
    events = list(events_by_key.values())
    events.sort(
        key=lambda event: (
            github_datetime(str(event.payload["created_at"])),
            int(event.payload["github_comment_id"]),
        )
    )
    return events


def _comment_event(
    mailbox: GitHubMailbox,
    pull: dict[str, object],
    assignment: AssignmentRecord,
    item: dict[str, object],
    *,
    actor: str,
) -> ControllerEvent | None:
    try:
        author = str(object_value(item["user"])["login"])
        body = str(item.get("body") or "")
        records = parse_assignment_comment_markers(body)
        github_comment_id = int(item["id"])
        comment_url = str(item["html_url"])
        created_at = str(item["created_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if author.casefold() != actor.casefold() or len(records) != 1:
        return None
    record: AssignmentCommentRecord = records[0]
    number = int(pull["number"])
    if (
        record.repo != mailbox.repo
        or record.pr_number != number
        or record.assignment_id != assignment.assignment_id
        or record.revision_id != assignment.revision_id
        or record.student != assignment.student
    ):
        return None
    marker = render_assignment_comment_marker(record)
    if authoritative_marker_line(body) != marker:
        return None
    message = "\n".join(body.splitlines()[1:]).strip()
    if not message:
        return None
    payload = {
        "number": number,
        "pr_url": str(pull["html_url"]),
        "comment_url": comment_url,
        "github_comment_id": github_comment_id,
        "comment_id": record.comment_id,
        "assignment_id": record.assignment_id,
        "revision_id": record.revision_id,
        "student": record.student,
        "message": bounded_text(message, limit=FEEDBACK_EXCERPT_BYTES),
        "created_at": created_at,
    }
    semantic_payload = {
        "number": number,
        "assignment_id": record.assignment_id,
        "revision_id": record.revision_id,
        "student": record.student,
        "comment_id": record.comment_id,
        "message": message,
    }
    return ControllerEvent(
        kind="student_assignment_comment",
        dedupe_key=(
            "student_assignment_comment:v2:" + payload_digest(semantic_payload)
        ),
        payload=payload,
    )


def _report_read_error(message: str) -> None:
    print(
        f"SENPAI_STUDENT_COMMENT_READ_ERROR {message}",
        file=sys.stderr,
        flush=True,
    )
