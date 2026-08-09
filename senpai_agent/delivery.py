"""Durable, model-invisible delivery identities for OpenHands messages."""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import UUID

_SENDER_PREFIX = "senpai-delivery:"


@dataclass(frozen=True, slots=True)
class MessageDelivery:
    delivery_id: str
    message: str

    def __post_init__(self) -> None:
        if not self.delivery_id:
            raise ValueError("delivery ID must not be empty")
        if not self.message:
            raise ValueError("delivered message must not be empty")


class PendingDeliveryLedger:
    """Keep one delivery attempt per unacknowledged conversation event."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.RLock()
        self._values = self._read()

    def claim(
        self,
        conversation_id: UUID | str,
        event_keys: list[str] | tuple[str, ...] | frozenset[str],
    ) -> dict[str, str]:
        conversation_key = str(conversation_id)
        with self._lock:
            attempts = self._values.setdefault(conversation_key, {})
            changed = False
            for event_key in event_keys:
                if event_key not in attempts:
                    attempts[event_key] = str(uuid.uuid4())
                    changed = True
            if changed:
                self._write()
            return {event_key: attempts[event_key] for event_key in event_keys}

    def complete(
        self,
        conversation_id: UUID | str,
        event_keys: list[str] | tuple[str, ...] | frozenset[str],
    ) -> None:
        conversation_key = str(conversation_id)
        with self._lock:
            attempts = self._values.get(conversation_key)
            if attempts is None:
                return
            changed = False
            for event_key in event_keys:
                changed = attempts.pop(event_key, None) is not None or changed
            if not attempts:
                self._values.pop(conversation_key, None)
            if changed:
                self._write()

    def _read(self) -> dict[str, dict[str, str]]:
        if self.path is None or not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid pending delivery ledger: {self.path}")
        for conversation_id, attempts in value.items():
            if not isinstance(conversation_id, str) or not isinstance(attempts, dict):
                raise RuntimeError(f"invalid pending delivery ledger: {self.path}")
            for event_key, attempt_id in attempts.items():
                if not isinstance(event_key, str) or not isinstance(attempt_id, str):
                    raise RuntimeError(
                        f"invalid pending delivery ledger: {self.path}"
                    )
                try:
                    uuid.UUID(attempt_id)
                except ValueError as error:
                    raise RuntimeError(
                        f"invalid pending delivery ledger: {self.path}"
                    ) from error
        return value

    def _write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(self._values, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


class MessageConversation(Protocol):
    state: object

    def send_message(self, message: str, sender: str | None = None) -> None: ...


def send_message_once(
    conversation: MessageConversation,
    delivery: MessageDelivery,
) -> bool:
    """Append a message once on the active durable conversation branch."""

    sender = _delivery_sender(delivery.delivery_id)
    state = getattr(conversation, "state", None)
    active_branch = getattr(state, "active_branch", None)
    guard = state if hasattr(state, "__enter__") else nullcontext()
    with guard:
        if active_branch is not None and any(
            getattr(event, "sender", None) == sender for event in active_branch()
        ):
            return False
        conversation.send_message(delivery.message, sender=sender)
    return True


def _delivery_sender(delivery_id: str) -> str:
    digest = sha256(delivery_id.encode()).hexdigest()
    return f"{_SENDER_PREFIX}{digest}"
