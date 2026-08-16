"""Durable conversation identity and delivery state."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from senpai_agent.mailbox import ControllerEvent


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
    def __init__(self, registry: AssignmentConversationRegistry):
        self.registry = registry

    def __call__(
        self,
        events: Sequence[ControllerEvent],
    ) -> tuple[ConversationBatch, ...]:
        grouped: dict[UUID, list[ControllerEvent]] = {}
        for event in events:
            conversation_id = self._conversation_for(event)
            grouped.setdefault(conversation_id, []).append(event)
        return tuple(
            ConversationBatch(conversation_id, tuple(batch_events))
            for conversation_id, batch_events in grouped.items()
        )

    def _conversation_for(self, event: ControllerEvent) -> UUID:
        if event.kind in {"job_monitor", "training_monitor"}:
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
