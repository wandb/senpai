"""Advisor events derived from PR and Issue communication."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import AssignmentRecord

from .issues import human_issue_events
from .student_comments import student_assignment_comment_events

if TYPE_CHECKING:
    from .core import GitHubMailbox


def advisor_communication_events(
    mailbox: GitHubMailbox,
    assignments: Sequence[tuple[dict[str, object], AssignmentRecord]],
    issues: Sequence[dict[str, object]],
) -> list[ControllerEvent]:
    return [
        *student_assignment_comment_events(mailbox, assignments),
        *human_issue_events(mailbox, issues),
    ]
