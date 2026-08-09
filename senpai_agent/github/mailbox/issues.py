"""Trusted human issue events for advisor and student controllers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from senpai_agent.mailbox import ControllerEvent

from .values import (
    TRUSTED_HUMAN_ASSOCIATIONS,
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
        messages = []
        for item in (issue, *mailbox._issue_comments(issue)):
            user = object_value(item["user"])
            if (
                user.get("type") != "User"
                or item.get("author_association")
                not in TRUSTED_HUMAN_ASSOCIATIONS
            ):
                continue
            messages.append(
                {
                    "id": int(item["id"]),
                    "author": str(user["login"]),
                    "body": str(item.get("body") or ""),
                    "created_at": str(item["created_at"]),
                }
            )
        human_messages = [
            message
            for message in messages
            if str(message["author"]).casefold() != actor.casefold()
        ]
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
                "human_issue",
                number,
                latest["id"],
                payload_digest({"message": full_message}),
                payload=payload,
            )
        )
    return events
