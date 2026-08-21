"""Controller event values and local mailbox composition."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from senpai_agent.event_kinds import EventKind
from senpai_agent.inbox import PersistentInbox
from senpai_agent.local_events import LocalEventStore
from senpai_agent.model_markdown import (
    canonical_event_identity,
    render_event_prompt,
)


_AVAILABILITY_EVENT_PREFIX = f"{EventKind.STUDENT_AVAILABLE_FOR_ASSIGNMENT}:"


@dataclass(frozen=True, slots=True)
class ControllerEvent:
    kind: str
    dedupe_key: str
    payload: dict[str, object]

    def to_prompt(self) -> str:
        return render_event_prompt(self.kind, self.payload)

    def event_identity(self) -> str:
        return canonical_event_identity(self.kind, self.payload)


def report_event_render_error(
    kind: str,
    event_key: str,
    error: Exception,
    *,
    disposition: str,
) -> None:
    print(
        "SENPAI_EVENT_RENDER_ERROR "
        f"kind={ascii(kind)} event_key={ascii(event_key)} "
        f"error={type(error).__name__}:{ascii(error)} "
        f"disposition={disposition}",
        file=sys.stderr,
        flush=True,
    )


class Mailbox(Protocol):
    def poll(self) -> Sequence[ControllerEvent]: ...

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None: ...


class CompositeMailbox:
    """Merge independent mailboxes without letting one failure suppress another."""

    def __init__(self, *mailboxes: Mailbox):
        self.mailboxes = mailboxes

    def poll(self) -> tuple[ControllerEvent, ...]:
        by_key: dict[str, ControllerEvent] = {}
        for mailbox in self.mailboxes:
            try:
                events = mailbox.poll()
            except Exception as error:  # noqa: BLE001
                print(
                    f"SENPAI_MAILBOX_ERROR {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            for event in events:
                by_key.setdefault(event.dedupe_key, event)
        return tuple(by_key.values())

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        for mailbox in self.mailboxes:
            mailbox.acknowledge(dedupe_keys)


class StudentAssignmentAvailabilityMailbox:
    """Reconcile student availability after each successful GitHub snapshot."""

    def __init__(
        self,
        mailbox: Mailbox,
        *,
        inbox: PersistentInbox,
        conversation_id: UUID | str,
        event_store_path: Path,
    ):
        self.mailbox = mailbox
        self.inbox = inbox
        self.conversation_id = conversation_id
        self.event_store_path = event_store_path

    def poll(self) -> tuple[ControllerEvent, ...]:
        events = tuple(self.mailbox.poll())
        current = {
            event.dedupe_key
            for event in events
            if event.kind == EventKind.STUDENT_AVAILABLE_FOR_ASSIGNMENT
        }
        with LocalEventStore(self.event_store_path) as store:
            store.discard_prefix(
                _AVAILABILITY_EVENT_PREFIX,
                retained_keys=tuple(current),
            )
        self.inbox.retract_pending_prefix(
            self.conversation_id,
            _AVAILABILITY_EVENT_PREFIX,
            retained_keys=tuple(current),
        )
        return events

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        self.mailbox.acknowledge(dedupe_keys)


class LocalAdvisorMailbox:
    """Deliver durable local child results directly to the advisor inbox."""

    def __init__(self, store_path: Path):
        self.store_path = store_path

    def poll(self) -> tuple[ControllerEvent, ...]:
        with LocalEventStore(self.store_path) as store:
            pending = store.pending()
        return tuple(
            ControllerEvent(
                kind=event.kind,
                dedupe_key=event.dedupe_key,
                payload=event.payload,
            )
            for event in pending
        )

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        with LocalEventStore(self.store_path) as store:
            for key in dedupe_keys:
                store.acknowledge(key)


class LocalStudentMailbox:
    """Deliver local child results directly to their parent conversations."""

    def __init__(self, store_path: Path):
        self.store_path = store_path

    def poll(self) -> tuple[ControllerEvent, ...]:
        with LocalEventStore(self.store_path) as store:
            pending = store.pending()
        for event in pending:
            if not isinstance(event.payload.get("parent_conversation_id"), str):
                raise RuntimeError("student child event has no parent conversation")
        return tuple(
            ControllerEvent(
                kind=event.kind,
                dedupe_key=event.dedupe_key,
                payload=event.payload,
            )
            for event in pending
        )

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        with LocalEventStore(self.store_path) as store:
            for key in dedupe_keys:
                store.acknowledge(key)
