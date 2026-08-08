"""Trusted pull-request feedback events for student controllers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import AssignmentRecord

from .ledger import read_feedback_ledger, write_feedback_ledger
from .values import (
    FEEDBACK_EXCERPT_BYTES,
    FeedbackBinding,
    feedback_excerpt,
    github_datetime,
    object_value,
    trusted_feedback,
)

if TYPE_CHECKING:
    from .core import GitHubMailbox


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
    events: list[ControllerEvent] = []
    for surface, _url in sources:
        for item in feedback_by_surface[surface]:
            if surface == "review" and item.get("submitted_at") is None:
                continue
            if (
                surface == "inline_comment"
                and int(item["pull_request_review_id"])
                not in submitted_review_ids
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
                    full_message_instruction=(
                        "Open feedback_url to read the omitted text."
                    ),
                )
            if surface == "review":
                payload["state"] = str(item["state"])
            elif surface == "inline_comment":
                payload["path"] = str(item["path"])
                line = item.get("line") or item.get("original_line")
                if line is not None:
                    payload["line"] = int(line)
            events.append(
                ControllerEvent(
                    kind="student_pr_feedback",
                    dedupe_key=(
                        f"student_pr_feedback:{surface}:{number}:{feedback_id}"
                    ),
                    payload=payload,
                )
            )
    events.sort(
        key=lambda event: (
            github_datetime(str(event.payload["created_at"])),
            str(event.payload["feedback_type"]),
            int(event.payload["feedback_id"]),
        )
    )
    return _pending_feedback(mailbox, events, assignment)


def _pending_feedback(
    mailbox: GitHubMailbox,
    events: Iterable[ControllerEvent],
    assignment: AssignmentRecord,
) -> list[ControllerEvent]:
    ledger = read_feedback_ledger(mailbox)
    ledger_changed = False
    bound_events: list[ControllerEvent] = []
    for event in events:
        binding = ledger.get(event.dedupe_key)
        if binding is None:
            binding = FeedbackBinding(
                assignment_id=str(event.payload["assignment_id"]),
                revision_id=str(event.payload["revision_id"]),
            )
            ledger[event.dedupe_key] = binding
            ledger_changed = True
        bound_events.append(
            ControllerEvent(
                kind=event.kind,
                dedupe_key=event.dedupe_key,
                payload={
                    **event.payload,
                    "assignment_id": binding.assignment_id,
                    "revision_id": binding.revision_id,
                },
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
