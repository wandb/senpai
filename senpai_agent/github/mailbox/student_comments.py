"""Trusted student assignment comments delivered to advisor controllers."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from senpai_agent.event_kinds import EventKind
from senpai_agent.github.http import GitHubReadError
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import (
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


@dataclass(frozen=True, slots=True)
class _CommentCandidate:
    event: ControllerEvent
    content_digest: str
    created_at: str
    github_comment_id: int


def student_assignment_comment_events(
    mailbox: GitHubMailbox,
    assignments: Sequence[tuple[Mapping[str, object], AssignmentRecord]],
) -> list[ControllerEvent]:
    """Wake the advisor for trusted typed messages from assigned students."""

    if not assignments:
        return []
    try:
        actor = mailbox._github.actor()
    except GitHubReadError as error:
        _report_read_error(f"actor {type(error).__name__}: {error}")
        return []

    candidates_by_key: dict[str, _CommentCandidate] = {}
    conflicted_keys: set[str] = set()
    for pull, assignment in assignments:
        number = int(pull["number"])
        try:
            comments = mailbox._pull_comments(number)
        except GitHubReadError as error:
            _report_read_error(f"pr={number} {type(error).__name__}: {error}")
            continue
        for comment in comments:
            candidate = _comment_candidate(
                mailbox,
                pull,
                assignment,
                comment,
                actor=actor,
            )
            if candidate is None:
                continue
            event = candidate.event
            if event.dedupe_key in conflicted_keys:
                continue
            previous = candidates_by_key.get(event.dedupe_key)
            if (
                previous is not None
                and previous.content_digest != candidate.content_digest
            ):
                candidates_by_key.pop(event.dedupe_key)
                conflicted_keys.add(event.dedupe_key)
                _report_read_error(
                    "conflicting bodies for assignment comment identity "
                    f"pr={number} comment_id={event.payload['comment_id']!r}"
                )
                continue
            if (
                previous is None
                or candidate.github_comment_id < previous.github_comment_id
            ):
                candidates_by_key[event.dedupe_key] = candidate

    return [
        candidate.event
        for candidate in sorted(
            candidates_by_key.values(),
            key=lambda candidate: (
                github_datetime(candidate.created_at),
                candidate.github_comment_id,
            ),
        )
    ]


def _comment_candidate(
    mailbox: GitHubMailbox,
    pull: Mapping[str, object],
    assignment: AssignmentRecord,
    comment: Mapping[str, object],
    *,
    actor: str,
) -> _CommentCandidate | None:
    try:
        author = str(object_value(comment["user"])["login"])
        body = str(comment.get("body") or "")
        records = parse_assignment_comment_markers(body)
        github_comment_id = int(comment["id"])
        created_at = str(comment["created_at"])
        updated_at = str(comment["updated_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if author.casefold() != actor.casefold() or len(records) != 1:
        return None
    record = records[0]
    number = int(pull["number"])
    if (
        record.repo != mailbox.repo
        or record.pr_number != number
        or record.assignment_id != assignment.assignment_id
        or record.student != assignment.student
        or authoritative_marker_line(body)
        != render_assignment_comment_marker(record)
    ):
        return None
    message = "\n".join(body.splitlines()[1:]).strip()
    if not message:
        return None
    if updated_at != created_at:
        _report_read_error(
            "edited assignment comment rejected "
            f"pr={number} comment_id={record.comment_id!r} "
            f"github_comment_id={github_comment_id}"
        )
        return None
    content_digest = payload_digest({"body": body})
    payload = {
        "number": number,
        "pr_url": str(pull["html_url"]),
        "comment_id": record.comment_id,
        "assignment_id": record.assignment_id,
        "revision_id": record.revision_id,
        "student": record.student,
        "message": bounded_text(message, limit=FEEDBACK_EXCERPT_BYTES),
        "content_digest": content_digest,
    }
    return _CommentCandidate(
        event=ControllerEvent(
            kind=EventKind.STUDENT_ASSIGNMENT_COMMENT,
            dedupe_key=(
                f"{EventKind.STUDENT_ASSIGNMENT_COMMENT}:v2:"
                + payload_digest(record.model_dump(mode="json"))
            ),
            payload=payload,
        ),
        content_digest=content_digest,
        created_at=created_at,
        github_comment_id=github_comment_id,
    )


def _report_read_error(message: str) -> None:
    print(
        f"SENPAI_STUDENT_COMMENT_READ_ERROR {message}",
        file=sys.stderr,
        flush=True,
    )
