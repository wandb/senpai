"""Durable delivery and acknowledgement of student and Supervisor feedback."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from senpai_agent.github.supervision import (
    SupervisorEnvelope,
    _identity,
    _trusted_envelope,
)
from senpai_agent.github.workflow import ReconciliationError
from senpai_agent.mailbox import ControllerEvent

from .values import FEEDBACK_KEY_PREFIX, FeedbackBinding, versioned_event

if TYPE_CHECKING:
    from .core import GitHubMailbox

_COMPLETION_KEY_PREFIX = "supervisor_completed:v2:github:"


def acknowledge_feedback(
    mailbox: GitHubMailbox,
    dedupe_keys: Sequence[str],
) -> None:
    feedback_keys = {key for key in dedupe_keys if key.startswith(FEEDBACK_KEY_PREFIX)}
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
        ledger[key] = replace(binding, acknowledged=True)
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
            or (
                item.get("source_key") is not None
                and not isinstance(item.get("source_key"), str)
            )
            or (
                item.get("content_digest") is not None
                and not isinstance(item.get("content_digest"), str)
            )
            or (
                item.get("payload") is not None
                and not isinstance(item.get("payload"), dict)
            )
        ):
            raise RuntimeError(
                f"invalid student PR feedback ledger: {mailbox.feedback_path}"
            )
        ledger[key] = FeedbackBinding(
            assignment_id=item["assignment_id"],
            revision_id=item["revision_id"],
            acknowledged=item["acknowledged"],
            source_key=item.get("source_key"),
            content_digest=item.get("content_digest"),
            payload=item.get("payload"),
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
    temporary = mailbox.feedback_path.with_suffix(f"{mailbox.feedback_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {
                key: {
                    "assignment_id": binding.assignment_id,
                    "revision_id": binding.revision_id,
                    "acknowledged": binding.acknowledged,
                    **(
                        {
                            "source_key": binding.source_key,
                            "content_digest": binding.content_digest,
                            "payload": binding.payload,
                        }
                        if binding.source_key is not None
                        else {}
                    ),
                }
                for key, binding in sorted(ledger.items())
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(mailbox.feedback_path)


def _completion_event(
    issue: Mapping[str, object], envelope: SupervisorEnvelope
) -> ControllerEvent:
    assert envelope.result is not None
    payload = {
        "number": int(issue["number"]),
        "url": str(issue["html_url"]),
        "request": envelope.request.model_dump(mode="json"),
        "result": envelope.result.model_dump(mode="json"),
    }
    if envelope.parent_conversation_id is not None:
        payload["parent_conversation_id"] = str(envelope.parent_conversation_id)
    return versioned_event(
        "supervisor_completed",
        "github",
        payload["number"],
        _identity(envelope),
        payload=payload,
    )


def supervisor_events(mailbox: GitHubMailbox) -> tuple[ControllerEvent, ...]:
    target = "advisor" if mailbox.role == "advisor" else mailbox.student_name
    query = urlencode(
        {
            "state": "open",
            "labels": f"supervisor,{mailbox.advisor_branch}",
            "per_page": 100,
        }
    )
    issues = mailbox._github.objects(f"/repos/{mailbox.repo}/issues?{query}")
    if not issues:
        return ()
    actor = mailbox._github.actor().casefold()
    events = []
    for issue in issues:
        envelope = _trusted_envelope(
            issue,
            repo=mailbox.repo,
            advisor_branch=mailbox.advisor_branch,
            actor=actor,
        )
        if envelope is None or issue["state"] != "open":
            continue
        if envelope.result is None and envelope.request.target == target:
            events.append(
                ControllerEvent(
                    "supervisor_requested",
                    f"supervisor_requested:{_identity(envelope)}",
                    {
                        "number": int(issue["number"]),
                        "url": str(issue["html_url"]),
                        "request": envelope.request.model_dump(mode="json"),
                    },
                )
            )
        elif envelope.result is not None and envelope.requester == target:
            events.append(_completion_event(issue, envelope))
    return tuple(events)


def acknowledge_supervisor_results(
    mailbox: GitHubMailbox, dedupe_keys: Sequence[str]
) -> None:
    workflow = mailbox._workflow
    requester = "advisor" if mailbox.role == "advisor" else mailbox.student_name
    for key in dedupe_keys:
        if not key.startswith(_COMPLETION_KEY_PREFIX):
            continue
        number, _ = key.removeprefix(_COMPLETION_KEY_PREFIX).split(":", 1)
        path = f"/repos/{mailbox.repo}/issues/{int(number)}"
        issue = workflow._request("GET", path, expected_statuses={200}).json_body
        envelope = _trusted_envelope(
            issue,
            repo=mailbox.repo,
            advisor_branch=mailbox.advisor_branch,
            actor=workflow._actor(),
        )
        if (
            envelope is None
            or envelope.requester != requester
            or envelope.result is None
            or _completion_event(issue, envelope).dedupe_key != key
        ):
            continue  # A changed Issue must not block acknowledgement of an old receipt.
        if issue["state"] == "closed":
            continue
        workflow._mutate(
            "PATCH",
            path,
            json_body={"state": "closed"},
            expected_statuses={200},
        )
        saved = workflow._request("GET", path, expected_statuses={200}).json_body
        if saved.get("state") != "closed":
            raise ReconciliationError(
                "GitHub did not acknowledge the Supervisor result"
            )
