"""Durable conversation identity and delivery state."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.PROMPTS import TRAINING_HANDOVER_PROMPT

if TYPE_CHECKING:
    from senpai_agent.monitor import MonitorStore


def _replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


class AssignmentConversationRegistry:
    """Persist one OpenHands conversation UUID per assignment revision."""

    def __init__(self, path: Path):
        self.path = path

    def for_assignment(self, assignment_id: str, revision_id: str) -> UUID:
        key = f"{assignment_id}:{revision_id}"
        values = self._read()
        if key not in values:
            values[key] = str(uuid.uuid4())
            _replace_json(self.path, values)
        return UUID(values[key])

    def has_assignment(self, assignment_id: str, revision_id: str) -> bool:
        return f"{assignment_id}:{revision_id}" in self._read()

    def revision_of(self, conversation_id: UUID, assignment_id: str) -> str | None:
        """Return the revision of this assignment that owns the conversation."""

        prefix = f"{assignment_id}:"
        return next(
            (
                key.removeprefix(prefix)
                for key, value in self._read().items()
                if value == str(conversation_id) and key.startswith(prefix)
            ),
            None,
        )

    def require_assignment(
        self, conversation_id: UUID, assignment_id: str, revision_id: str
    ) -> None:
        """Reject a conversation that is not bound to this assignment revision."""

        values = self._read()
        current = f"{assignment_id}:{revision_id}"
        if values.get(current) == str(conversation_id):
            return
        bound = next(
            (key for key, value in values.items() if value == str(conversation_id)),
            None,
        )
        raise PermissionError(
            f"training conversation is bound to assignment revision {bound!r}; "
            f"current assignment revision is {current!r}. "
            "Only the current revision's conversation may launch training. If "
            "this conversation belongs to an earlier revision, do not continue "
            "the assignment here: finish this turn. The controller resumes the "
            "current revision in its own conversation and hands it any active "
            "training run."
        )

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise RuntimeError(f"invalid conversation registry: {self.path}")
        return value


class StartedConversationLedger:
    """Record conversations that received their initial controller context."""

    def __init__(self, path: Path):
        self.path = path

    def has_started(self, conversation_id: UUID) -> bool:
        return str(conversation_id) in self._read()

    def mark_started(self, conversation_id: UUID) -> None:
        values = self._read()
        value = str(conversation_id)
        if value in values:
            return
        values.add(value)
        _replace_json(self.path, sorted(values))

    def _read(self) -> set[str]:
        if not self.path.exists():
            return set()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise RuntimeError(f"invalid conversation ledger: {self.path}")
        return set(value)


class WorkspaceDivergenceLedger:
    """Persist the last handled workspace blocker for each conversation."""

    def __init__(self, path: Path):
        self.path = path

    def current(self, conversation_id: UUID) -> str | None:
        return self._read().get(str(conversation_id))

    def record(self, conversation_id: UUID, event_key: str) -> None:
        values = self._read()
        key = str(conversation_id)
        if values.get(key) == event_key:
            return
        values[key] = event_key
        _replace_json(self.path, values)

    def clear(self, conversation_id: UUID) -> None:
        values = self._read()
        if values.pop(str(conversation_id), None) is not None:
            _replace_json(self.path, values)

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise RuntimeError(f"invalid workspace divergence ledger: {self.path}")
        return value


@dataclass(frozen=True, slots=True)
class ConversationBatch:
    conversation_id: UUID
    events: tuple[ControllerEvent, ...]


class StudentConversationSelector:
    """Route student events, moving active training to the current revision."""

    def __init__(
        self,
        registry: AssignmentConversationRegistry,
        monitor_store: MonitorStore | None = None,
    ):
        self.registry = registry
        self.monitor_store = monitor_store

    def __call__(
        self,
        events: Sequence[ControllerEvent],
    ) -> tuple[ConversationBatch, ...]:
        handovers = [
            handover
            for event in events
            if event.kind == "student_assignment"
            for handover in self._hand_over_training(event)
        ]
        grouped: dict[UUID, list[ControllerEvent]] = {}
        for event in (*events, *handovers):
            event = self._with_current_monitor_owner(event)
            conversation_id = self._conversation_for(event)
            grouped.setdefault(conversation_id, []).append(event)
        return tuple(
            ConversationBatch(conversation_id, tuple(batch_events))
            for conversation_id, batch_events in grouped.items()
        )

    def _hand_over_training(self, event: ControllerEvent) -> list[ControllerEvent]:
        """Give runs from superseded revisions to the current revision.

        Only the current revision may launch, and only a run's owner may
        cancel it or receive its monitor signals, so a run left with an old
        revision would keep resuming that revision and block the new one.
        """

        if self.monitor_store is None:
            return []
        assignment_id = str(event.payload["assignment_id"])
        revision_id = str(event.payload["revision_id"])
        current = self.registry.for_assignment(assignment_id, revision_id)
        handovers: list[ControllerEvent] = []
        for spec in self.monitor_store.active():
            previous_revision = self.registry.revision_of(
                spec.conversation_id, assignment_id
            )
            if previous_revision in (None, revision_id):
                continue
            self.monitor_store.reassign(spec.training_id, current)
            handovers.append(
                ControllerEvent(
                    kind="training_handover",
                    dedupe_key=f"training_handover:{spec.training_id}:{current}",
                    payload={
                        "training_id": spec.training_id,
                        "assignment_id": assignment_id,
                        "previous_revision_id": previous_revision,
                        "revision_id": revision_id,
                        "instruction": TRAINING_HANDOVER_PROMPT,
                        "parent_conversation_id": str(current),
                    },
                )
            )
        return handovers

    def _with_current_monitor_owner(self, event: ControllerEvent) -> ControllerEvent:
        if event.kind != "training_monitor" or self.monitor_store is None:
            return event
        owner = str(
            self.monitor_store.spec(str(event.payload["training_id"])).conversation_id
        )
        if event.payload["conversation_id"] == owner:
            return event
        return replace(event, payload={**event.payload, "conversation_id": owner})

    def _conversation_for(self, event: ControllerEvent) -> UUID:
        if event.kind == "training_monitor":
            return UUID(str(event.payload["conversation_id"]))
        parent_id = event.payload.get("parent_conversation_id")
        if isinstance(parent_id, str):
            return UUID(parent_id)
        if event.kind in {"student_assignment", "student_pr_feedback"}:
            return self.registry.for_assignment(
                str(event.payload["assignment_id"]),
                str(event.payload["revision_id"]),
            )
        if event.kind == "human_issue":
            return self.registry.for_assignment(
                f"human-issue-{event.payload['number']}",
                "thread",
            )
        return self.registry.for_assignment("student-control", "current")
