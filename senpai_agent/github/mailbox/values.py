"""Shared values and parsing helpers for GitHub mailbox events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from senpai_agent.github.human_messages import is_trusted_human_author
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import (
    AssignmentRecord,
    ExperimentResult,
    parse_assignment_feedback_markers,
)

FEEDBACK_KEY_PREFIX = "student_pr_feedback:"
FEEDBACK_EXCERPT_BYTES = 4_000
DEFAULT_FEEDBACK_BATCH_EVENTS = 8
DEFAULT_FEEDBACK_BATCH_BYTES = 32_000


@dataclass(frozen=True, slots=True)
class FeedbackBinding:
    assignment_id: str
    revision_id: str
    acknowledged: bool = False
    source_key: str | None = None
    content_digest: str | None = None
    payload: dict[str, object] | None = None


def pull_reference(pull: Mapping[str, object]) -> dict[str, object]:
    head = object_value(pull["head"])
    return {
        "number": int(pull["number"]),
        "url": str(pull["html_url"]),
        "head_ref": str(head["ref"]),
        "head_sha": str(head["sha"]),
    }


def payload_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def versioned_event(
    kind: str,
    *identity: object,
    payload: dict[str, object],
) -> ControllerEvent:
    prefix = ":".join((kind, "v2", *(str(value) for value in identity)))
    return ControllerEvent(
        kind=kind,
        dedupe_key=f"{prefix}:{payload_digest(payload)}",
        payload=payload,
    )


def result_matches_assignment(
    result: ExperimentResult,
    *,
    repo: str,
    pr_number: int,
    assignment: AssignmentRecord,
    head_sha: str,
) -> bool:
    key = result.assignment
    return (
        key.repo == repo
        and key.pr_number == pr_number
        and key.assignment_id == assignment.assignment_id
        and key.revision_id == assignment.revision_id
        and key.student == assignment.student
        and key.expected_head_sha == head_sha
        and result.commit_sha == head_sha
    )


def label_names(value: Mapping[str, object]) -> set[str]:
    labels = value.get("labels")
    if not isinstance(labels, list):
        raise TypeError("GitHub mailbox item has invalid labels")
    return {str(object_value(label)["name"]) for label in labels}


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("GitHub mailbox returned an invalid object")
    return value


def github_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def bounded_text(value: str, *, limit: int) -> str:
    return _middle_excerpt(
        value,
        limit=limit,
        marker="\n\n[... middle omitted; open the event URL for full text ...]\n\n",
    )[0]


def feedback_excerpt(value: str, *, limit: int) -> tuple[str, bool]:
    return _middle_excerpt(
        value,
        limit=limit,
        marker=(
            "\n\n[... middle omitted; open feedback_url for full text ...]\n\n"
        ),
    )


def _middle_excerpt(
    value: str,
    *,
    limit: int,
    marker: str,
) -> tuple[str, bool]:
    encoded = value.encode()
    if len(encoded) <= limit:
        return value, False
    marker_bytes = marker.encode()
    content_bytes = limit - len(marker_bytes)
    head_bytes = 3 * content_bytes // 4
    excerpt = (
        encoded[:head_bytes].decode(errors="ignore")
        + marker
        + encoded[-(content_bytes - head_bytes) :].decode(errors="ignore")
    )
    return excerpt, True


def trusted_feedback(
    item: Mapping[str, object],
    *,
    actor: str,
    repo: str,
    pr_number: int,
    assignment: AssignmentRecord,
) -> tuple[FeedbackBinding, str] | None:
    user = item.get("user")
    if not isinstance(user, dict):
        return None
    body = str(item.get("body") or "")
    current = FeedbackBinding(
        assignment_id=assignment.assignment_id,
        revision_id=assignment.revision_id,
    )
    same_actor = str(user.get("login") or "").casefold() == actor.casefold()
    trusted_human = is_trusted_human_author(
        author_type=str(user.get("type") or ""),
        association=str(item.get("author_association") or ""),
    )
    if not same_actor:
        return (current, body) if trusted_human else None

    protocol_markers = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith("<!-- senpai-")
        and line.strip().endswith(" -->")
    ]
    if not protocol_markers:
        return (current, body) if trusted_human else None
    try:
        records = parse_assignment_feedback_markers(body)
    except ValueError:
        return None
    if len(protocol_markers) != 1 or len(records) != 1:
        return None
    record = records[0]
    if (
        record.repo != repo
        or record.pr_number != pr_number
        or record.assignment_id != assignment.assignment_id
    ):
        return None
    content = "\n".join(
        line for line in body.splitlines() if line.strip() != protocol_markers[0]
    ).strip()
    if not content:
        return None
    return (
        FeedbackBinding(
            assignment_id=record.assignment_id,
            revision_id=record.revision_id,
        ),
        content,
    )
