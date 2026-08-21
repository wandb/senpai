"""Trusted human pull-request comments delivered to advisor controllers."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from senpai_agent.github.human_messages import is_trusted_human_message
from senpai_agent.github.http import GitHubReadError
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.PROMPTS import TRUNCATED_FEEDBACK_PROMPT

from .values import (
    FEEDBACK_EXCERPT_BYTES,
    feedback_excerpt,
    github_datetime,
    object_value,
    payload_digest,
    versioned_event,
)

if TYPE_CHECKING:
    from .core import GitHubMailbox


@dataclass(frozen=True, slots=True)
class _HumanComment:
    event: ControllerEvent
    created_at: datetime
    comment_id: int


def human_pr_comment_events(
    mailbox: GitHubMailbox,
    pulls: Sequence[Mapping[str, object]],
) -> list[ControllerEvent]:
    """Return trusted human comments from every open advisor-branch PR."""

    try:
        actor = mailbox._github.actor()
    except GitHubReadError as error:
        _report_read_error(f"actor {type(error).__name__}: {error}")
        return []

    candidates: list[_HumanComment] = []
    for pull in pulls:
        try:
            number = int(pull["number"])
            pr_url = str(pull["html_url"])
            comments = mailbox._pull_comments(number)
        except (GitHubReadError, KeyError, TypeError, ValueError) as error:
            _report_read_error(
                f"pr={pull.get('number')!r} {type(error).__name__}: {error}"
            )
            continue
        for comment in comments:
            try:
                candidate = _human_comment(
                    number=number,
                    pr_url=pr_url,
                    comment=comment,
                    actor=actor,
                )
            except (KeyError, TypeError, ValueError) as error:
                _report_read_error(
                    f"pr={number} comment_id={comment.get('id')!r} "
                    f"{type(error).__name__}: {error}"
                )
                continue
            if candidate is not None:
                candidates.append(candidate)

    return [
        candidate.event
        for candidate in sorted(
            candidates,
            key=lambda candidate: (candidate.created_at, candidate.comment_id),
        )
    ]


def _human_comment(
    *,
    number: int,
    pr_url: str,
    comment: Mapping[str, object],
    actor: str,
) -> _HumanComment | None:
    user = object_value(comment["user"])
    author = user["login"]
    author_type = user["type"]
    association = comment["author_association"]
    body = comment.get("body")
    if not isinstance(author, str) or not isinstance(author_type, str):
        raise TypeError("GitHub PR comment has invalid author fields")
    if not isinstance(association, str) or not isinstance(body, str):
        raise TypeError("GitHub PR comment has invalid author or body fields")
    comment_id = int(comment["id"])
    feedback_url = comment["html_url"]
    created_at_value = comment["created_at"]
    updated_at = comment["updated_at"]
    if (
        not isinstance(feedback_url, str)
        or not isinstance(created_at_value, str)
        or not isinstance(updated_at, str)
    ):
        raise TypeError("GitHub PR comment has invalid URL or timestamp fields")
    created_at = github_datetime(created_at_value)
    github_datetime(updated_at)
    if not body.strip() or not is_trusted_human_message(
        author=author,
        author_type=author_type,
        association=association,
        body=body,
        actor=actor,
    ):
        return None

    message, message_truncated = feedback_excerpt(
        body.strip(),
        limit=FEEDBACK_EXCERPT_BYTES,
    )
    content_digest = payload_digest({"body": body, "updated_at": updated_at})
    payload: dict[str, object] = {
        "number": number,
        "pr_url": pr_url,
        "feedback_url": feedback_url,
        "feedback_id": comment_id,
        "feedback_type": "issue_comment",
        "author": author,
        "message": message,
        "created_at": created_at_value,
        "updated_at": updated_at,
        "content_digest": content_digest,
    }
    if message_truncated:
        payload.update(
            message_truncated=True,
            full_message_instruction=TRUNCATED_FEEDBACK_PROMPT,
        )
    return _HumanComment(
        event=versioned_event(
            "human_pr_comment",
            "issue_comment",
            number,
            comment_id,
            payload=payload,
        ),
        created_at=created_at,
        comment_id=comment_id,
    )


def _report_read_error(message: str) -> None:
    print(
        f"SENPAI_HUMAN_PR_COMMENT_READ_ERROR {message}",
        file=sys.stderr,
        flush=True,
    )
