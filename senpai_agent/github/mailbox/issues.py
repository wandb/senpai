"""Trusted human issue events for advisor and student controllers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from senpai_agent.event_kinds import EventKind
from senpai_agent.github.human_messages import is_trusted_human_message
from senpai_agent.mailbox import ControllerEvent

from .values import (
    bounded_text,
    github_datetime,
    label_names,
    object_value,
    payload_digest,
    versioned_event,
)

if TYPE_CHECKING:
    from .core import GitHubMailbox


def human_issue_events(
    mailbox: GitHubMailbox,
    issues: Sequence[dict[str, object]],
) -> list[ControllerEvent]:
    role_labels = {"team"}
    if mailbox.role == "advisor":
        role_labels.add(mailbox.advisor_branch)
    else:
        assert mailbox.student_name is not None
        role_labels.add(f"student:{mailbox.student_name}")
    events = []
    for issue in issues:
        labels = label_names(issue)
        if "human" not in labels or not role_labels & labels:
            continue
        actor = mailbox._github.actor()
        human_messages = []
        for item in (issue, *mailbox._issue_comments(issue)):
            user = object_value(item["user"])
            author = str(user["login"])
            body = str(item.get("body") or "")
            if not is_trusted_human_message(
                author=author,
                author_type=str(user.get("type") or ""),
                association=str(item.get("author_association") or ""),
                body=body,
                actor=actor,
            ):
                continue
            human_messages.append(
                {
                    "id": int(item["id"]),
                    "author": author,
                    "body": body,
                    "created_at": str(item["created_at"]),
                }
            )
        if not human_messages:
            continue
        latest = max(
            human_messages,
            key=lambda message: (
                github_datetime(str(message["created_at"])),
                int(message["id"]),
            ),
        )
        number = int(issue["number"])
        full_message = str(latest["body"])
        payload = {
            "number": number,
            "title": str(issue["title"]),
            "url": str(issue["html_url"]),
            "human_message_id": int(latest["id"]),
            "author": str(latest["author"]),
            "message": bounded_text(
                full_message,
                limit=12_000,
            ),
            "created_at": str(latest["created_at"]),
        }
        events.append(
            versioned_event(
                EventKind.HUMAN_ISSUE,
                number,
                latest["id"],
                payload_digest({"message": full_message}),
                payload=payload,
            )
        )
    return events
