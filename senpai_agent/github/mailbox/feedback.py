"""Trusted pull-request feedback events for student controllers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import AssignmentRecord
from senpai_agent.PROMPTS import TRUNCATED_FEEDBACK_PROMPT

from .ledger import read_feedback_ledger, write_feedback_ledger
from .values import (
    FEEDBACK_EXCERPT_BYTES,
    FEEDBACK_KEY_PREFIX,
    FeedbackBinding,
    feedback_excerpt,
    github_datetime,
    object_value,
    payload_digest,
    trusted_feedback,
)

if TYPE_CHECKING:
    from .core import GitHubMailbox


@dataclass(frozen=True, slots=True)
class _FeedbackCandidate:
    source_key: str
    content_digest: str
    payload: dict[str, object]


def student_pr_feedback_events(
    mailbox: GitHubMailbox,
    pull: Mapping[str, object],
    assignment: AssignmentRecord,
) -> list[ControllerEvent]:
    number = int(pull["number"])
    sources: list[tuple[str, str]] = []
    if comments_url := pull.get("comments_url"):
        sources.append(("issue_comment", f"{comments_url}?per_page=100"))
    if pull_url := pull.get("url"):
        sources.append(
            ("review", f"{str(pull_url).rstrip('/')}/reviews?per_page=100")
        )
    if comments_url := pull.get("review_comments_url"):
        sources.append(("inline_comment", f"{comments_url}?per_page=100"))
    if not sources:
        return []

    actor = mailbox._github.actor()
    feedback_by_surface = {
        surface: mailbox._github.objects(url) for surface, url in sources
    }
    submitted_review_ids = {
        int(item["id"])
        for item in feedback_by_surface.get("review", ())
        if item.get("submitted_at") is not None
    }
    candidates: list[_FeedbackCandidate] = []
    for surface, _url in sources:
        for item in feedback_by_surface[surface]:
            if surface == "review" and item.get("submitted_at") is None:
                continue
            if (
                surface == "inline_comment"
                and (
                    item.get("pull_request_review_id") is None
                    or int(item["pull_request_review_id"])
                    not in submitted_review_ids
                )
            ):
                continue
            trusted = trusted_feedback(
                item,
                actor=actor,
                repo=mailbox.repo,
                pr_number=number,
                assignment=assignment,
            )
            if trusted is None:
                continue
            binding, feedback_body = trusted
            feedback_id = int(item["id"])
            created_at = str(
                item["submitted_at"] if surface == "review" else item["created_at"]
            )
            message, message_truncated = feedback_excerpt(
                feedback_body,
                limit=FEEDBACK_EXCERPT_BYTES,
            )
            payload: dict[str, object] = {
                "number": number,
                "pr_url": str(pull["html_url"]),
                "feedback_url": str(item["html_url"]),
                "feedback_id": feedback_id,
                "feedback_type": surface,
                "assignment_id": binding.assignment_id,
                "revision_id": binding.revision_id,
                "base_ref": assignment.base_ref,
                "base_sha": assignment.base_sha,
                "head_ref": str(object_value(pull["head"])["ref"]),
                "head_sha": str(object_value(pull["head"])["sha"]),
                "author": str(object_value(item["user"])["login"]),
                "author_association": str(item["author_association"]),
                "message": message,
                "created_at": created_at,
            }
            if message_truncated:
                payload.update(
                    message_truncated=True,
                    full_message_instruction=TRUNCATED_FEEDBACK_PROMPT,
                )
            if surface == "review":
                payload["state"] = str(item["state"])
            elif surface == "inline_comment":
                payload["path"] = str(item["path"])
                line = item.get("line") or item.get("original_line")
                if line is not None:
                    payload["line"] = int(line)
            content = {
                "feedback_type": surface,
                "feedback_id": feedback_id,
                "author": payload["author"],
                "author_association": payload["author_association"],
                "message": str(item.get("body") or ""),
                "created_at": created_at,
                "updated_at": str(
                    item.get("updated_at")
                    or item.get("submitted_at")
                    or item.get("created_at")
                    or ""
                ),
            }
            content.update(
                {
                    key: payload[key]
                    for key in ("state", "path", "line")
                    if key in payload
                }
            )
            candidates.append(
                _FeedbackCandidate(
                    source_key=(
                        f"student_pr_feedback:{surface}:{number}:{feedback_id}"
                    ),
                    content_digest=payload_digest(content),
                    payload=payload,
                )
            )
    candidates.sort(
        key=lambda candidate: (
            github_datetime(str(candidate.payload["created_at"])),
            str(candidate.payload["feedback_type"]),
            int(candidate.payload["feedback_id"]),
        )
    )
    return _pending_feedback(mailbox, candidates, assignment)


def _pending_feedback(
    mailbox: GitHubMailbox,
    candidates: Iterable[_FeedbackCandidate],
    assignment: AssignmentRecord,
) -> list[ControllerEvent]:
    ledger = read_feedback_ledger(mailbox)
    ledger_changed = False
    bound_events: list[ControllerEvent] = []
    for candidate in candidates:
        event_key = (
            f"student_pr_feedback:v2:{candidate.source_key.removeprefix(FEEDBACK_KEY_PREFIX)}:"
            f"{candidate.content_digest}"
        )
        binding = ledger.get(event_key)
        if binding is None:
            prior = [
                value
                for key, value in ledger.items()
                if key == candidate.source_key
                or value.source_key == candidate.source_key
            ]
            identities = {
                (value.assignment_id, value.revision_id) for value in prior
            }
            if len(identities) > 1:
                raise RuntimeError(
                    f"feedback source {candidate.source_key!r} has conflicting bindings"
                )
            if identities:
                assignment_id, revision_id = next(iter(identities))
            else:
                assignment_id = str(candidate.payload["assignment_id"])
                revision_id = str(candidate.payload["revision_id"])
            payload = {
                **candidate.payload,
                "assignment_id": assignment_id,
                "revision_id": revision_id,
            }
            versioned_prior = any(
                value.source_key == candidate.source_key for value in prior
            )
            acknowledged = (
                bool(prior)
                and not versioned_prior
                and all(value.acknowledged for value in prior)
            )
            binding = FeedbackBinding(
                assignment_id=assignment_id,
                revision_id=revision_id,
                acknowledged=acknowledged,
                source_key=candidate.source_key,
                content_digest=candidate.content_digest,
                payload=payload,
            )
            ledger[event_key] = binding
            ledger_changed = True
        if (
            binding.source_key != candidate.source_key
            or binding.content_digest != candidate.content_digest
            or binding.payload is None
            or binding.payload.get("assignment_id") != binding.assignment_id
            or binding.payload.get("revision_id") != binding.revision_id
        ):
            raise RuntimeError(
                f"feedback event {event_key!r} has incomplete durable content"
            )
        bound_events.append(
            ControllerEvent(
                kind="student_pr_feedback",
                dedupe_key=event_key,
                payload=binding.payload,
            )
        )
    if ledger_changed:
        write_feedback_ledger(mailbox, ledger)
    pending = [
        event
        for event in bound_events
        if not ledger[event.dedupe_key].acknowledged
    ]
    prior_revision = [
        event
        for event in pending
        if event.payload["assignment_id"] != assignment.assignment_id
        or event.payload["revision_id"] != assignment.revision_id
    ]
    return feedback_batch(mailbox, prior_revision or pending)


def feedback_batch(
    mailbox: GitHubMailbox,
    events: Iterable[ControllerEvent],
) -> list[ControllerEvent]:
    selected: list[ControllerEvent] = []
    prompt_bytes = 0
    for event in events:
        if len(selected) >= mailbox.feedback_batch_events:
            break
        event_bytes = len(event.to_prompt().encode())
        separator_bytes = 2 if selected else 0
        if event_bytes > mailbox.feedback_batch_bytes:
            if selected:
                break
            raise RuntimeError(
                "student PR feedback event exceeds "
                f"{mailbox.feedback_batch_bytes} prompt bytes"
            )
        if (
            prompt_bytes + separator_bytes + event_bytes
            > mailbox.feedback_batch_bytes
        ):
            break
        selected.append(event)
        prompt_bytes += separator_bytes + event_bytes
    return selected
