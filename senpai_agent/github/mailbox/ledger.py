"""Durable acknowledgement state for student pull-request feedback."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .values import FEEDBACK_KEY_PREFIX, FeedbackBinding

if TYPE_CHECKING:
    from .core import GitHubMailbox


def acknowledge_feedback(
    mailbox: GitHubMailbox,
    dedupe_keys: Sequence[str],
) -> None:
    feedback_keys = {
        key for key in dedupe_keys if key.startswith(FEEDBACK_KEY_PREFIX)
    }
    if not feedback_keys:
        return
    ledger = read_feedback_ledger(mailbox)
    missing = feedback_keys - ledger.keys()
    if missing:
        raise RuntimeError(
            "cannot acknowledge unseen student PR feedback: "
            f"{', '.join(sorted(missing))}"
        )
    changed = False
    for key in feedback_keys:
        binding = ledger[key]
        if binding.acknowledged:
            continue
        ledger[key] = FeedbackBinding(
            assignment_id=binding.assignment_id,
            revision_id=binding.revision_id,
            acknowledged=True,
        )
        changed = True
    if changed:
        write_feedback_ledger(mailbox, ledger)


def read_feedback_ledger(
    mailbox: GitHubMailbox,
) -> dict[str, FeedbackBinding]:
    if mailbox.feedback_path is None:
        return dict(mailbox._memory_feedback)
    if not mailbox.feedback_path.exists():
        return {}
    value = json.loads(mailbox.feedback_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(
            f"invalid student PR feedback ledger: {mailbox.feedback_path}"
        )
    ledger: dict[str, FeedbackBinding] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key.startswith(FEEDBACK_KEY_PREFIX)
            or not isinstance(item, dict)
            or not isinstance(item.get("assignment_id"), str)
            or not isinstance(item.get("revision_id"), str)
            or not isinstance(item.get("acknowledged"), bool)
        ):
            raise RuntimeError(
                f"invalid student PR feedback ledger: {mailbox.feedback_path}"
            )
        ledger[key] = FeedbackBinding(
            assignment_id=item["assignment_id"],
            revision_id=item["revision_id"],
            acknowledged=item["acknowledged"],
        )
    return ledger


def write_feedback_ledger(
    mailbox: GitHubMailbox,
    ledger: Mapping[str, FeedbackBinding],
) -> None:
    if mailbox.feedback_path is None:
        mailbox._memory_feedback = dict(ledger)
        return
    mailbox.feedback_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mailbox.feedback_path.with_suffix(
        f"{mailbox.feedback_path.suffix}.tmp"
    )
    temporary.write_text(
        json.dumps(
            {
                key: {
                    "assignment_id": binding.assignment_id,
                    "revision_id": binding.revision_id,
                    "acknowledged": binding.acknowledged,
                }
                for key, binding in sorted(ledger.items())
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(mailbox.feedback_path)
