"""Durable exactly-once delivery between Senpai and OpenHands."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import UUID


MAX_EVENTS_PER_TURN = 16
MAX_EVENT_BYTES_PER_TURN = 64 * 1024
_SENDER_PREFIX = "senpai-delivery:"


class DeliveryState(StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSED = "processed"


@dataclass(frozen=True, slots=True)
class InboxMessage:
    delivery_id: str
    sender: str
    body: str
    state: DeliveryState
    event_key: str | None
    requires_ack: bool


@dataclass(frozen=True, slots=True)
class InboxTurn:
    turn_id: str
    conversation_id: str
    state: DeliveryState
    messages: tuple[InboxMessage, ...]
    superseded_by: str | None = None
    recovery_of: str | None = None
    legacy_prompt_delivery_id: str | None = None
    context_reset_completed: bool = True
    acknowledged: bool = False
    recovery_generation: int = 0
    quarantine_reason: str | None = None

    @property
    def context_reset_required(self) -> bool:
        return self.recovery_of is not None and not self.context_reset_completed

    @property
    def prompt(self) -> InboxMessage:
        return self.messages[0]

    @property
    def events(self) -> tuple[InboxMessage, ...]:
        return self.messages[1:]

    @property
    def event_keys(self) -> tuple[str, ...]:
        return tuple(
            message.event_key
            for message in self.events
            if message.event_key is not None
        )

    @property
    def acknowledgement_keys(self) -> tuple[str, ...]:
        return tuple(
            message.event_key
            for message in self.events
            if message.event_key is not None and message.requires_ack
        )


class InboxTurnQuarantined(RuntimeError):
    def __init__(self, turn_id: str, reason: str):
        self.turn_id = turn_id
        self.reason = reason
        super().__init__(f"inbox turn {turn_id} quarantined: {reason}")


class PersistentInbox:
    """Persist event order, turn membership, and monotonic delivery receipts."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        legacy_path: Path | None = None,
    ):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path) if path is not None else ":memory:",
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS inbox_turns (
                turn_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN ('pending', 'delivered', 'processed')
                ),
                recovery_of TEXT,
                superseded_by TEXT,
                legacy_prompt_delivery_id TEXT,
                context_reset_completed INTEGER NOT NULL DEFAULT 1,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                stalled_attempts INTEGER NOT NULL DEFAULT 0,
                progress_event_id TEXT,
                recovery_generation INTEGER NOT NULL DEFAULT 0,
                quarantine_reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                processed_at TEXT,
                acknowledged_at TEXT
            );

            CREATE TABLE IF NOT EXISTS inbox_messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                event_key TEXT,
                body TEXT NOT NULL,
                body_sha256 TEXT NOT NULL,
                delivery_id TEXT NOT NULL UNIQUE,
                sender TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL CHECK (
                    state IN ('pending', 'delivered', 'processed')
                ),
                requires_ack INTEGER NOT NULL,
                legacy INTEGER NOT NULL DEFAULT 0,
                turn_id TEXT,
                position INTEGER,
                FOREIGN KEY (turn_id) REFERENCES inbox_turns(turn_id)
            );

            CREATE INDEX IF NOT EXISTS inbox_pending_by_conversation
            ON inbox_messages(conversation_id, state, turn_id, sequence);

            CREATE INDEX IF NOT EXISTS inbox_event_identity
            ON inbox_messages(conversation_id, event_key);

            CREATE INDEX IF NOT EXISTS inbox_turns_by_conversation
            ON inbox_turns(conversation_id, acknowledged, superseded_by, created_at);

            CREATE TABLE IF NOT EXISTS legacy_deliveries (
                conversation_id TEXT NOT NULL,
                event_key TEXT NOT NULL,
                delivery_id TEXT NOT NULL UNIQUE,
                claimed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (conversation_id, event_key)
            );
            """
        )
        turn_columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(inbox_turns)")
        }
        if "legacy_prompt_delivery_id" not in turn_columns:
            self._connection.execute(
                "ALTER TABLE inbox_turns ADD COLUMN legacy_prompt_delivery_id TEXT"
            )
        if "context_reset_completed" not in turn_columns:
            self._connection.execute(
                """
                ALTER TABLE inbox_turns
                ADD COLUMN context_reset_completed INTEGER NOT NULL DEFAULT 1
                """
            )
            self._connection.execute(
                """
                UPDATE inbox_turns
                SET context_reset_completed = 0
                WHERE recovery_of IS NOT NULL AND state != 'processed'
                """
            )
        if "stalled_attempts" not in turn_columns:
            self._connection.execute(
                """
                ALTER TABLE inbox_turns
                ADD COLUMN stalled_attempts INTEGER NOT NULL DEFAULT 0
                """
            )
        if "progress_event_id" not in turn_columns:
            self._connection.execute(
                "ALTER TABLE inbox_turns ADD COLUMN progress_event_id TEXT"
            )
        if "recovery_generation" not in turn_columns:
            self._connection.execute(
                """
                ALTER TABLE inbox_turns
                ADD COLUMN recovery_generation INTEGER NOT NULL DEFAULT 0
                """
            )
            self._connection.execute(
                """
                UPDATE inbox_turns
                SET recovery_generation = 1
                WHERE recovery_of IS NOT NULL
                """
            )
        if "quarantine_reason" not in turn_columns:
            self._connection.execute(
                "ALTER TABLE inbox_turns ADD COLUMN quarantine_reason TEXT"
            )
        message_columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(inbox_messages)")
        }
        if "legacy" not in message_columns:
            self._connection.execute(
                """
                ALTER TABLE inbox_messages
                ADD COLUMN legacy INTEGER NOT NULL DEFAULT 0
                """
            )
        if legacy_path is not None and legacy_path.is_file():
            self._import_legacy_deliveries(legacy_path)
        self._connection.commit()

    def enqueue(
        self,
        conversation_id: UUID | str,
        event_key: str,
        body: str,
        *,
        requires_ack: bool = True,
    ) -> bool:
        if not event_key:
            raise ValueError("event key must not be empty")
        if not body:
            raise ValueError("event body must not be empty")
        conversation = str(conversation_id)
        with self._transaction() as database:
            payload_conflict = database.execute(
                """
                SELECT 1
                FROM inbox_messages
                WHERE conversation_id = ?
                  AND event_key = ?
                  AND legacy = 0
                  AND body != ?
                LIMIT 1
                """,
                (conversation, event_key, body),
            ).fetchone()
            if payload_conflict is not None:
                raise RuntimeError(
                    f"event {event_key!r} was reused with a different payload"
                )
            existing = database.execute(
                """
                SELECT message.sequence, message.requires_ack
                FROM inbox_messages AS message
                LEFT JOIN inbox_turns AS turn ON turn.turn_id = message.turn_id
                WHERE message.conversation_id = ?
                  AND message.event_key = ?
                  AND (
                      message.turn_id IS NULL
                      OR (
                          turn.acknowledged = 0
                          AND turn.superseded_by IS NULL
                      )
                  )
                ORDER BY message.sequence DESC
                LIMIT 1
                """,
                (conversation, event_key),
            ).fetchone()
            if existing is not None:
                if requires_ack and not existing["requires_ack"]:
                    database.execute(
                        "UPDATE inbox_messages SET requires_ack = 1 WHERE sequence = ?",
                        (existing["sequence"],),
                    )
                return False
            legacy = database.execute(
                """
                SELECT delivery_id
                FROM legacy_deliveries
                WHERE conversation_id = ? AND event_key = ? AND claimed = 0
                """,
                (conversation, event_key),
            ).fetchone()
            delivery_id = (
                str(legacy["delivery_id"])
                if legacy is not None
                else str(uuid.uuid4())
            )
            database.execute(
                """
                INSERT INTO inbox_messages (
                    conversation_id,
                    event_key,
                    body,
                    body_sha256,
                    delivery_id,
                    sender,
                    state,
                    requires_ack,
                    legacy
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    conversation,
                    event_key,
                    body,
                    _digest(body),
                    delivery_id,
                    _sender(delivery_id),
                    int(requires_ack),
                    int(legacy is not None),
                ),
            )
            if legacy is not None:
                database.execute(
                    """
                    UPDATE legacy_deliveries
                    SET claimed = 1
                    WHERE conversation_id = ? AND event_key = ?
                    """,
                    (conversation, event_key),
                )
            return True

    def next_turn(
        self,
        conversation_id: UUID | str,
        prompt: str,
        *,
        max_events: int = MAX_EVENTS_PER_TURN,
        max_bytes: int = MAX_EVENT_BYTES_PER_TURN,
        legacy_prompt_identity: str | None = None,
    ) -> InboxTurn | None:
        if not prompt:
            raise ValueError("turn prompt must not be empty")
        if max_events <= 0 or max_bytes <= 0:
            raise ValueError("turn limits must be positive")
        conversation = str(conversation_id)
        with self._transaction() as database:
            active = database.execute(
                """
                SELECT turn_id
                FROM inbox_turns
                WHERE conversation_id = ?
                  AND acknowledged = 0
                  AND superseded_by IS NULL
                  AND state != 'processed'
                  AND quarantine_reason IS NULL
                ORDER BY rowid
                LIMIT 1
                """,
                (conversation,),
            ).fetchone()
            if active is not None:
                return self._turn(database, active["turn_id"])

            quarantined = database.execute(
                """
                SELECT 1
                FROM inbox_turns
                WHERE conversation_id = ?
                  AND acknowledged = 0
                  AND superseded_by IS NULL
                  AND quarantine_reason IS NOT NULL
                LIMIT 1
                """,
                (conversation,),
            ).fetchone()
            if quarantined is not None:
                return None

            waiting_for_ack = database.execute(
                """
                SELECT 1
                FROM inbox_turns
                WHERE conversation_id = ?
                  AND acknowledged = 0
                  AND superseded_by IS NULL
                  AND state = 'processed'
                LIMIT 1
                """,
                (conversation,),
            ).fetchone()
            if waiting_for_ack is not None:
                return None

            pending = database.execute(
                """
                SELECT sequence, body, delivery_id, legacy
                FROM inbox_messages
                WHERE conversation_id = ?
                  AND state = 'pending'
                  AND turn_id IS NULL
                ORDER BY legacy DESC, sequence
                """,
                (conversation,),
            ).fetchall()
            selected: list[sqlite3.Row] = []
            selected_bytes = 0
            legacy_batch = bool(pending and pending[0]["legacy"])
            for row in pending:
                if bool(row["legacy"]) != legacy_batch:
                    break
                size = len(row["body"].encode("utf-8"))
                if selected and (
                    len(selected) >= max_events or selected_bytes + size > max_bytes
                ):
                    break
                selected.append(row)
                selected_bytes += size
                if len(selected) >= max_events:
                    break
            if not selected:
                return None

            turn_id = str(uuid.uuid4())
            legacy_prompt_delivery_id = (
                _legacy_prompt_delivery_id(
                    tuple(
                        str(row["delivery_id"])
                        for row in pending
                        if bool(row["legacy"])
                    ),
                    identity=legacy_prompt_identity,
                )
                if legacy_batch
                else None
            )
            database.execute(
                """
                INSERT INTO inbox_turns (
                    turn_id,
                    conversation_id,
                    state,
                    legacy_prompt_delivery_id
                ) VALUES (?, ?, 'pending', ?)
                """,
                (turn_id, conversation, legacy_prompt_delivery_id),
            )
            prompt_id = str(uuid.uuid4())
            database.execute(
                """
                INSERT INTO inbox_messages (
                    conversation_id,
                    event_key,
                    body,
                    body_sha256,
                    delivery_id,
                    sender,
                    state,
                    requires_ack,
                    turn_id,
                    position
                ) VALUES (?, NULL, ?, ?, ?, ?, 'pending', 0, ?, 0)
                """,
                (
                    conversation,
                    prompt,
                    _digest(prompt),
                    prompt_id,
                    _sender(prompt_id),
                    turn_id,
                ),
            )
            for position, row in enumerate(selected, start=1):
                database.execute(
                    """
                    UPDATE inbox_messages
                    SET turn_id = ?, position = ?
                    WHERE sequence = ?
                    """,
                    (turn_id, position, row["sequence"]),
                )
            return self._turn(database, turn_id)

    def adopt_legacy_prompt(
        self,
        turn_id: str,
        body: str,
        *,
        sender: str | None = None,
    ) -> InboxTurn:
        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            delivery_id = turn.legacy_prompt_delivery_id
            if delivery_id is None and sender is None:
                return turn
            prompt = turn.prompt
            if prompt.state is not DeliveryState.PENDING:
                if prompt.body != body or (sender is not None and prompt.sender != sender):
                    raise RuntimeError(
                        f"legacy prompt for turn {turn_id} conflicts with its receipt"
                    )
                return turn
            adopted_sender = sender or _sender(str(delivery_id))
            adopted_id = (
                str(delivery_id)
                if sender is None or adopted_sender == _sender(str(delivery_id))
                else f"legacy-prompt:{adopted_sender.removeprefix(_SENDER_PREFIX)}"
            )
            self._adopt_legacy_message_row(
                database,
                prompt.delivery_id,
                body,
                delivery_id=adopted_id,
                sender=adopted_sender,
            )
            return self._turn(database, turn_id)

    def prepare_legacy_turn(
        self,
        turn_id: str,
        branch: Sequence[object],
    ) -> InboxTurn:
        """Adopt one already-visible #3472 delivery batch without replaying it."""

        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            if not turn.events or not all(
                self._is_legacy_message(database, message.delivery_id)
                for message in turn.events
            ):
                return turn

            rows = database.execute(
                """
                SELECT message.*
                FROM inbox_messages AS message
                LEFT JOIN inbox_turns AS owner ON owner.turn_id = message.turn_id
                WHERE message.conversation_id = ?
                  AND message.legacy = 1
                  AND (
                      message.turn_id IS NULL
                      OR (
                          owner.acknowledged = 0
                          AND owner.superseded_by IS NULL
                      )
                  )
                ORDER BY message.sequence
                """,
                (turn.conversation_id,),
            ).fetchall()
            legacy_by_sender = {str(row["sender"]): row for row in rows}
            branch_senders = [getattr(event, "sender", None) for event in branch]
            positions: dict[str, list[int]] = {}
            for index, sender_value in enumerate(branch_senders):
                if isinstance(sender_value, str):
                    positions.setdefault(sender_value, []).append(index)
            for sender_value in legacy_by_sender:
                if len(positions.get(sender_value, ())) > 1:
                    raise RuntimeError(
                        f"duplicate sender {sender_value!r} on active branch"
                    )

            selected_positions = [
                positions[message.sender][0]
                for message in turn.events
                if message.sender in positions
            ]
            persisted_prompt_positions = positions.get(turn.prompt.sender, [])
            if len(persisted_prompt_positions) > 1:
                raise RuntimeError(
                    f"duplicate sender {turn.prompt.sender!r} on active branch"
                )
            prompt_index = (
                persisted_prompt_positions[0]
                if persisted_prompt_positions
                else self._legacy_prompt_index(
                    turn,
                    branch_senders,
                    frozenset(legacy_by_sender),
                    selected_positions,
                )
            )
            if prompt_index is None:
                return turn

            prompt_event = branch[prompt_index]
            prompt_sender = str(branch_senders[prompt_index])
            prompt = turn.prompt
            if persisted_prompt_positions:
                prompt_row = self._message_row(database, prompt.delivery_id)
                _verify_body(prompt_row, _event_body(prompt_event))
                if DeliveryState(prompt_row["state"]) is DeliveryState.PENDING:
                    database.execute(
                        "UPDATE inbox_messages SET state = 'delivered' "
                        "WHERE delivery_id = ?",
                        (prompt.delivery_id,),
                    )
                database.execute(
                    "UPDATE inbox_turns SET legacy_prompt_delivery_id = ? "
                    "WHERE turn_id = ?",
                    (prompt.delivery_id, turn_id),
                )
            else:
                prompt_id = turn.legacy_prompt_delivery_id
                if prompt_id is None or _sender(prompt_id) != prompt_sender:
                    prompt_id = (
                        "legacy-prompt:"
                        f"{prompt_sender.removeprefix(_SENDER_PREFIX)}"
                    )
                self._adopt_legacy_message_row(
                    database,
                    prompt.delivery_id,
                    _event_body(prompt_event),
                    delivery_id=prompt_id,
                    sender=prompt_sender,
                )

            next_prompt = next(
                (
                    index
                    for index in range(prompt_index + 1, len(branch))
                    if isinstance(branch_senders[index], str)
                    and str(branch_senders[index]).startswith(_SENDER_PREFIX)
                    and branch_senders[index] not in legacy_by_sender
                ),
                len(branch),
            )
            for index in range(prompt_index + 1, next_prompt):
                row = legacy_by_sender.get(str(branch_senders[index]))
                if row is None:
                    continue
                if row["turn_id"] not in (None, turn_id):
                    raise RuntimeError(
                        f"legacy delivery {row['delivery_id']} belongs to another turn"
                    )
                self._adopt_legacy_message_row(
                    database,
                    str(row["delivery_id"]),
                    _event_body(branch[index]),
                )
                database.execute(
                    """
                    UPDATE inbox_messages
                    SET turn_id = ?
                    WHERE delivery_id = ?
                    """,
                    (turn_id, row["delivery_id"]),
                )

            turn_rows = database.execute(
                """
                SELECT sequence, delivery_id
                FROM inbox_messages
                WHERE turn_id = ? AND event_key IS NOT NULL
                ORDER BY sequence
                """,
                (turn_id,),
            ).fetchall()
            for position, row in enumerate(turn_rows, start=1):
                database.execute(
                    "UPDATE inbox_messages SET position = ? WHERE delivery_id = ?",
                    (position, row["delivery_id"]),
                )
            self._refresh_turn_state(database, turn_id)
            return self._turn(database, turn_id)

    def turn(self, turn_id: str) -> InboxTurn:
        with self._lock:
            return self._turn(self._connection, turn_id)

    def active_turn(self, conversation_id: UUID | str) -> InboxTurn | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT turn_id
                FROM inbox_turns
                WHERE conversation_id = ?
                  AND acknowledged = 0
                  AND superseded_by IS NULL
                  AND state != 'processed'
                  AND quarantine_reason IS NULL
                ORDER BY rowid
                LIMIT 1
                """,
                (str(conversation_id),),
            ).fetchone()
            return None if row is None else self._turn(self._connection, row[0])

    def record_pending(self, delivery_id: str) -> InboxMessage:
        with self._lock:
            message = self._message_by_delivery(self._connection, delivery_id)
            if message.state is not DeliveryState.PENDING:
                raise ValueError(
                    f"cannot move backwards from {message.state.value} to pending"
                )
            return message

    def record_delivered(self, delivery_id: str, body: str) -> InboxMessage:
        with self._transaction() as database:
            row = self._message_row(database, delivery_id)
            _verify_body(row, body)
            state = DeliveryState(row["state"])
            if state is DeliveryState.PENDING:
                database.execute(
                    "UPDATE inbox_messages SET state = 'delivered' WHERE delivery_id = ?",
                    (delivery_id,),
                )
                if row["turn_id"] is not None:
                    self._refresh_turn_state(database, row["turn_id"])
            return self._message_by_delivery(database, delivery_id)

    def record_processed(self, turn_id: str) -> InboxTurn:
        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            if turn.state is DeliveryState.PROCESSED:
                return turn
            if any(message.state is DeliveryState.PENDING for message in turn.messages):
                raise ValueError("cannot process a turn before every message is delivered")
            database.execute(
                """
                UPDATE inbox_messages
                SET state = 'processed'
                WHERE turn_id = ? AND state = 'delivered'
                """,
                (turn_id,),
            )
            database.execute(
                """
                UPDATE inbox_turns
                SET state = 'processed', processed_at = CURRENT_TIMESTAMP
                WHERE turn_id = ?
                """,
                (turn_id,),
            )
            return self._turn(database, turn_id)

    def record_inference_attempt(self, turn_id: str) -> InboxTurn:
        """Persist one attempt immediately before model inference starts."""

        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            if turn.state is not DeliveryState.DELIVERED:
                raise ValueError(
                    "inference can start only after the complete turn is delivered"
                )
            database.execute(
                """
                UPDATE inbox_turns
                SET stalled_attempts = stalled_attempts + 1
                WHERE turn_id = ?
                """,
                (turn_id,),
            )
            return self._turn(database, turn_id)

    def record_progress(self, turn_id: str, progress_event_id: str | None) -> InboxTurn:
        """Renew the attempt budget after a new completed tool observation."""

        if progress_event_id is not None and not progress_event_id:
            raise ValueError("progress event ID must be non-empty")
        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            if turn.state is DeliveryState.PROCESSED:
                return turn
            row = database.execute(
                "SELECT progress_event_id FROM inbox_turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if (
                progress_event_id is not None
                and progress_event_id != row["progress_event_id"]
            ):
                database.execute(
                    """
                    UPDATE inbox_turns
                    SET progress_event_id = ?, stalled_attempts = 0
                    WHERE turn_id = ?
                    """,
                    (progress_event_id, turn_id),
                )
            return self._turn(database, turn_id)

    def terminal_recovery_due(
        self,
        turn_id: str,
        *,
        max_attempts: int,
        max_age_seconds: float,
        now: float | None = None,
    ) -> bool:
        """Return whether one unresolved delivered turn has exhausted its budget."""

        if max_attempts <= 0 or max_age_seconds <= 0:
            raise ValueError("terminal recovery limits must be positive")
        current_time = time.time() if now is None else now
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    state,
                    superseded_by,
                    stalled_attempts,
                    quarantine_reason,
                    unixepoch(created_at) AS created_epoch
                FROM inbox_turns
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown inbox turn {turn_id}")
        if (
            DeliveryState(row["state"]) is not DeliveryState.DELIVERED
            or row["superseded_by"] is not None
            or row["quarantine_reason"] is not None
        ):
            return False
        age_seconds = max(0.0, current_time - float(row["created_epoch"]))
        return (
            int(row["stalled_attempts"]) >= max_attempts
            or age_seconds >= max_age_seconds
        )

    def latest_turn(self, turn_id: str) -> InboxTurn:
        with self._lock:
            return self._latest_turn(self._connection, turn_id)

    def recover_turn(
        self,
        turn_id: str,
        prompt: str,
        *,
        max_generations: int,
    ) -> InboxTurn:
        if max_generations < 0:
            raise ValueError("maximum recovery generations must be non-negative")
        current = self.latest_turn(turn_id)
        if current.recovery_generation >= max_generations:
            reason = "recovery budget exhausted"
            self.quarantine(current.turn_id, reason)
            raise InboxTurnQuarantined(current.turn_id, reason)
        return self.reset_turn(current.turn_id, prompt)

    def quarantine(self, turn_id: str, reason: str) -> InboxTurn:
        if not reason:
            raise ValueError("quarantine reason must not be empty")
        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            if turn.state is DeliveryState.PROCESSED:
                raise ValueError("cannot quarantine a processed turn")
            row = database.execute(
                "SELECT quarantine_reason FROM inbox_turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            existing = row["quarantine_reason"]
            if existing is not None and existing != reason:
                raise RuntimeError(
                    f"inbox turn {turn_id} already quarantined for {existing}"
                )
            database.execute(
                """
                UPDATE inbox_turns
                SET quarantine_reason = ?
                WHERE turn_id = ?
                """,
                (reason, turn_id),
            )
            return self._turn(database, turn_id)

    def quarantined_turns(self) -> tuple[InboxTurn, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT turn_id
                FROM inbox_turns
                WHERE quarantine_reason IS NOT NULL
                  AND superseded_by IS NULL
                ORDER BY rowid
                """
            ).fetchall()
            return tuple(self._turn(self._connection, row[0]) for row in rows)

    def processed_turns(self) -> tuple[InboxTurn, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT turn_id
                FROM inbox_turns
                WHERE state = 'processed'
                  AND acknowledged = 0
                  AND superseded_by IS NULL
                ORDER BY rowid
                """
            ).fetchall()
            return tuple(self._turn(self._connection, row[0]) for row in rows)

    def acknowledge(self, turn_id: str) -> None:
        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            if turn.state is not DeliveryState.PROCESSED:
                raise ValueError("cannot acknowledge a turn before it is processed")
            database.execute(
                """
                UPDATE inbox_turns
                SET acknowledged = 1, acknowledged_at = CURRENT_TIMESTAMP
                WHERE turn_id = ?
                """,
                (turn_id,),
            )

    def reset_turn(self, turn_id: str, prompt: str) -> InboxTurn:
        if not prompt:
            raise ValueError("recovery prompt must not be empty")
        with self._transaction() as database:
            original = self._turn(database, turn_id)
            if original.superseded_by is not None:
                return self._latest_turn(database, turn_id)
            if original.state is DeliveryState.PROCESSED:
                return original
            if original.quarantine_reason is not None:
                raise InboxTurnQuarantined(
                    original.turn_id,
                    original.quarantine_reason,
                )

            recovery_id = str(uuid.uuid4())
            database.execute(
                """
                INSERT INTO inbox_turns (
                    turn_id,
                    conversation_id,
                    state,
                    recovery_of,
                    context_reset_completed,
                    recovery_generation
                ) VALUES (?, ?, 'pending', ?, 0, ?)
                """,
                (
                    recovery_id,
                    original.conversation_id,
                    turn_id,
                    original.recovery_generation + 1,
                ),
            )
            prompt_id = str(uuid.uuid4())
            database.execute(
                """
                INSERT INTO inbox_messages (
                    conversation_id,
                    event_key,
                    body,
                    body_sha256,
                    delivery_id,
                    sender,
                    state,
                    requires_ack,
                    turn_id,
                    position
                ) VALUES (?, NULL, ?, ?, ?, ?, 'pending', 0, ?, 0)
                """,
                (
                    original.conversation_id,
                    prompt,
                    _digest(prompt),
                    prompt_id,
                    _sender(prompt_id),
                    recovery_id,
                ),
            )
            for position, message in enumerate(original.events, start=1):
                delivery_id = str(uuid.uuid4())
                database.execute(
                    """
                    INSERT INTO inbox_messages (
                        conversation_id,
                        event_key,
                        body,
                        body_sha256,
                        delivery_id,
                        sender,
                        state,
                        requires_ack,
                        legacy,
                        turn_id,
                        position
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                    """,
                    (
                        original.conversation_id,
                        message.event_key,
                        message.body,
                        _digest(message.body),
                        delivery_id,
                        _sender(delivery_id),
                        int(message.requires_ack),
                        int(self._is_legacy_message(database, message.delivery_id)),
                        recovery_id,
                        position,
                    ),
                )
            database.execute(
                "UPDATE inbox_turns SET superseded_by = ? WHERE turn_id = ?",
                (recovery_id, turn_id),
            )
            return self._turn(database, recovery_id)

    def record_context_reset(self, turn_id: str) -> InboxTurn:
        with self._transaction() as database:
            turn = self._turn(database, turn_id)
            if turn.recovery_of is None:
                raise ValueError("only a recovery turn can record a context reset")
            database.execute(
                """
                UPDATE inbox_turns
                SET context_reset_completed = 1
                WHERE turn_id = ?
                """,
                (turn_id,),
            )
            return self._turn(database, turn_id)

    def pending_count(self, conversation_id: UUID | str | None = None) -> int:
        where = "state = 'pending' AND turn_id IS NULL"
        parameters: tuple[object, ...] = ()
        if conversation_id is not None:
            where += " AND conversation_id = ?"
            parameters = (str(conversation_id),)
        with self._lock:
            row = self._connection.execute(
                f"SELECT COUNT(*) FROM inbox_messages WHERE {where}",
                parameters,
            ).fetchone()
            return int(row[0])

    def ready_conversation_ids(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT conversation_id, MIN(sequence) AS first_sequence
                FROM inbox_messages
                WHERE conversation_id NOT IN (
                        SELECT conversation_id
                        FROM inbox_turns
                        WHERE acknowledged = 0
                          AND superseded_by IS NULL
                          AND quarantine_reason IS NOT NULL
                    )
                  AND (
                      turn_id IS NULL OR turn_id IN (
                          SELECT turn_id
                          FROM inbox_turns
                          WHERE acknowledged = 0
                            AND superseded_by IS NULL
                            AND quarantine_reason IS NULL
                      )
                  )
                GROUP BY conversation_id
                ORDER BY first_sequence
                """
            ).fetchall()
            return tuple(str(row[0]) for row in rows)

    def acknowledged_event_keys(self) -> frozenset[str]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT message.event_key
                FROM inbox_messages AS message
                JOIN inbox_turns AS turn ON turn.turn_id = message.turn_id
                WHERE turn.acknowledged = 1
                  AND message.event_key IS NOT NULL
                """
            ).fetchall()
            return frozenset(str(row[0]) for row in rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _turn(self, database: sqlite3.Connection, turn_id: str) -> InboxTurn:
        row = database.execute(
            "SELECT * FROM inbox_turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown inbox turn {turn_id}")
        messages = database.execute(
            """
            SELECT *
            FROM inbox_messages
            WHERE turn_id = ?
            ORDER BY position
            """,
            (turn_id,),
        ).fetchall()
        return InboxTurn(
            turn_id=turn_id,
            conversation_id=str(row["conversation_id"]),
            state=DeliveryState(row["state"]),
            messages=tuple(_message(message) for message in messages),
            superseded_by=row["superseded_by"],
            recovery_of=row["recovery_of"],
            legacy_prompt_delivery_id=row["legacy_prompt_delivery_id"],
            context_reset_completed=bool(row["context_reset_completed"]),
            acknowledged=bool(row["acknowledged"]),
            recovery_generation=int(row["recovery_generation"]),
            quarantine_reason=row["quarantine_reason"],
        )

    def _latest_turn(
        self,
        database: sqlite3.Connection,
        turn_id: str,
    ) -> InboxTurn:
        turn = self._turn(database, turn_id)
        seen = {turn.turn_id}
        while turn.superseded_by is not None:
            if turn.superseded_by in seen:
                raise RuntimeError("inbox turn supersession cycle")
            seen.add(turn.superseded_by)
            turn = self._turn(database, turn.superseded_by)
        return turn

    def _message_row(
        self,
        database: sqlite3.Connection,
        delivery_id: str,
    ) -> sqlite3.Row:
        row = database.execute(
            "SELECT * FROM inbox_messages WHERE delivery_id = ?",
            (delivery_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown delivery {delivery_id}")
        return row

    def _message_by_delivery(
        self,
        database: sqlite3.Connection,
        delivery_id: str,
    ) -> InboxMessage:
        return _message(self._message_row(database, delivery_id))

    def _adopt_legacy_message_row(
        self,
        database: sqlite3.Connection,
        current_delivery_id: str,
        body: str,
        *,
        delivery_id: str | None = None,
        sender: str | None = None,
    ) -> None:
        row = self._message_row(database, current_delivery_id)
        if row["event_key"] is not None and not row["legacy"]:
            raise RuntimeError(
                f"native delivery {current_delivery_id} cannot adopt legacy payload"
            )
        already_adopted = (
            row["event_key"] is None
            and bool(row["legacy"])
        )
        if (
            DeliveryState(row["state"]) is not DeliveryState.PENDING
            or already_adopted
        ):
            _verify_body(row, body)
            if delivery_id is not None and row["delivery_id"] != delivery_id:
                raise RuntimeError(
                    f"delivery {current_delivery_id} identity mismatch"
                )
            if sender is not None and row["sender"] != sender:
                raise RuntimeError(
                    f"delivery {current_delivery_id} sender mismatch"
                )
            if DeliveryState(row["state"]) is DeliveryState.PENDING:
                database.execute(
                    "UPDATE inbox_messages SET state = 'delivered' WHERE delivery_id = ?",
                    (current_delivery_id,),
                )
                if row["turn_id"] is not None:
                    self._refresh_turn_state(database, row["turn_id"])
            return
        adopted_id = delivery_id or str(row["delivery_id"])
        adopted_sender = sender or str(row["sender"])
        database.execute(
            """
            UPDATE inbox_messages
            SET body = ?,
                body_sha256 = ?,
                delivery_id = ?,
                sender = ?,
                legacy = 1,
                state = 'delivered'
            WHERE delivery_id = ?
            """,
            (
                body,
                _digest(body),
                adopted_id,
                adopted_sender,
                current_delivery_id,
            ),
        )
        if row["turn_id"] is not None:
            self._refresh_turn_state(database, row["turn_id"])

    def _is_legacy_message(
        self,
        database: sqlite3.Connection,
        delivery_id: str,
    ) -> bool:
        return bool(self._message_row(database, delivery_id)["legacy"])

    @staticmethod
    def _legacy_prompt_index(
        turn: InboxTurn,
        branch_senders: Sequence[object],
        legacy_senders: frozenset[str],
        selected_positions: Sequence[int],
    ) -> int | None:
        if turn.legacy_prompt_delivery_id is not None:
            exact_sender = _sender(turn.legacy_prompt_delivery_id)
            exact = [
                index
                for index, value in enumerate(branch_senders)
                if value == exact_sender
            ]
            if len(exact) > 1:
                raise RuntimeError(
                    f"duplicate sender {exact_sender!r} on active branch"
                )
            if exact:
                return exact[0]
        if not selected_positions:
            return None
        first_event = min(selected_positions)
        return next(
            (
                index
                for index in range(first_event - 1, -1, -1)
                if isinstance(branch_senders[index], str)
                and str(branch_senders[index]).startswith(_SENDER_PREFIX)
                and branch_senders[index] not in legacy_senders
            ),
            None,
        )

    def _refresh_turn_state(
        self,
        database: sqlite3.Connection,
        turn_id: str,
    ) -> None:
        pending = database.execute(
            """
            SELECT 1 FROM inbox_messages
            WHERE turn_id = ? AND state = 'pending'
            LIMIT 1
            """,
            (turn_id,),
        ).fetchone()
        if pending is None:
            database.execute(
                """
                UPDATE inbox_turns
                SET state = 'delivered'
                WHERE turn_id = ? AND state = 'pending'
                """,
                (turn_id,),
            )

    def _transaction(self):
        return _Transaction(self._connection, self._lock)

    def _import_legacy_deliveries(self, path: Path) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid pending delivery ledger: {path}")
        for conversation_id, deliveries in value.items():
            if not isinstance(conversation_id, str) or not isinstance(deliveries, dict):
                raise RuntimeError(f"invalid pending delivery ledger: {path}")
            UUID(conversation_id)
            for event_key, delivery_id in deliveries.items():
                if not isinstance(event_key, str) or not isinstance(delivery_id, str):
                    raise RuntimeError(f"invalid pending delivery ledger: {path}")
                UUID(delivery_id)
                existing = self._connection.execute(
                    """
                    SELECT delivery_id
                    FROM legacy_deliveries
                    WHERE conversation_id = ? AND event_key = ?
                    """,
                    (conversation_id, event_key),
                ).fetchone()
                if existing is not None and existing["delivery_id"] != delivery_id:
                    raise RuntimeError(
                        f"legacy delivery {event_key!r} changed identity"
                    )
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO legacy_deliveries (
                        conversation_id,
                        event_key,
                        delivery_id
                    ) VALUES (?, ?, ?)
                    """,
                    (conversation_id, event_key, delivery_id),
                )


class _Transaction:
    def __init__(self, database: sqlite3.Connection, lock: threading.RLock):
        self.database = database
        self.lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self.lock.acquire()
        self.database.execute("BEGIN IMMEDIATE")
        return self.database

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        try:
            if exc_type is None:
                self.database.commit()
            else:
                self.database.rollback()
        finally:
            self.lock.release()


def deliver_turn_messages(
    conversation: object,
    inbox: PersistentInbox,
    turn_id: str,
) -> InboxTurn:
    """Append one turn's messages exactly once on the active branch."""

    turn = inbox.turn(turn_id)
    if turn.state is DeliveryState.PROCESSED:
        return turn
    state = getattr(conversation, "state", None)
    guard = state if hasattr(state, "__enter__") else nullcontext()
    with guard:
        branch = _active_branch(conversation)
        turn = inbox.prepare_legacy_turn(turn_id, branch)
        if turn.legacy_prompt_delivery_id is not None:
            legacy_sender = _sender(turn.legacy_prompt_delivery_id)
            legacy_prompts = [
                event
                for event in branch
                if getattr(event, "sender", None) == legacy_sender
            ]
            if len(legacy_prompts) > 1:
                raise RuntimeError(
                    f"duplicate sender {legacy_sender!r} on active branch"
                )
            if legacy_prompts:
                turn = inbox.adopt_legacy_prompt(
                    turn_id,
                    _event_body(legacy_prompts[0]),
                )
        by_sender: dict[str, list[object]] = {}
        for event in branch:
            sender = getattr(event, "sender", None)
            if isinstance(sender, str):
                by_sender.setdefault(sender, []).append(event)

        for message in turn.messages:
            existing = by_sender.get(message.sender, [])
            if len(existing) > 1:
                raise RuntimeError(
                    f"duplicate sender {message.sender!r} on active branch"
                )
            if existing:
                body = _event_body(existing[0])
                if body != message.body:
                    raise RuntimeError(
                        f"delivery {message.delivery_id} payload mismatch on active branch"
                    )
            else:
                conversation.send_message(message.body, sender=message.sender)
            inbox.record_delivered(message.delivery_id, message.body)
    return inbox.turn(turn_id)


def turn_has_finished_response(conversation: object, turn: InboxTurn) -> bool:
    terminal = False
    finish_calls: set[str] = set()
    for event in events_after_turn_delivery(conversation, turn):
        kind = type(event).__name__
        if _is_agent_message(event):
            terminal = True
            continue
        if kind == "ActionEvent":
            terminal = False
            if getattr(event, "tool_name", None) == "finish":
                finish_calls.update(_event_call_ids(event))
            continue
        if kind == "ObservationEvent":
            terminal = bool(finish_calls.intersection(_event_call_ids(event)))
            continue
        if kind in {"AgentErrorEvent", "UserRejectObservation"}:
            terminal = False
            continue
        if kind == "MessageEvent" and getattr(event, "source", None) != "agent":
            terminal = False
            continue
        if (
            kind == "SimpleNamespace"
            and getattr(event, "source", None) in {"user", "environment"}
            and (
                hasattr(event, "message")
                or hasattr(event, "llm_message")
            )
        ):
            terminal = False
    return terminal


def events_after_turn_delivery(
    conversation: object,
    turn: InboxTurn,
) -> tuple[object, ...]:
    branch = _active_branch(conversation)
    positions = {
        getattr(event, "sender", None): index for index, event in enumerate(branch)
    }
    try:
        last_delivery = max(positions[message.sender] for message in turn.messages)
    except KeyError:
        return ()
    return tuple(branch[last_delivery + 1 :])


def _active_branch(conversation: object) -> Sequence[object]:
    state = getattr(conversation, "state", None)
    active_branch = getattr(state, "active_branch", None)
    return () if active_branch is None else tuple(active_branch())


def _event_body(event: object) -> str:
    for attribute in ("message", "prompt"):
        value = getattr(event, attribute, None)
        if isinstance(value, str):
            return value
    llm_message = getattr(event, "llm_message", None)
    if llm_message is not None:
        parts = [
            getattr(content, "text", "")
            for content in getattr(llm_message, "content", ())
            if getattr(content, "text", "")
        ]
        if parts:
            return "\n".join(parts)
    raise RuntimeError("delivery sender exists without a verifiable message payload")


def _is_agent_message(event: object) -> bool:
    if type(event).__name__ == "MessageEvent":
        return (
            getattr(event, "source", None) == "agent"
            and bool(getattr(event, "llm_response_id", None))
            and getattr(getattr(event, "llm_message", None), "role", None)
            == "assistant"
        )
    if getattr(event, "sender", None) == "agent":
        return isinstance(getattr(event, "message", None), str)
    return False


def _event_call_ids(event: object) -> set[str]:
    return {
        value
        for attribute in ("tool_call_id", "action_id", "id")
        if isinstance((value := getattr(event, attribute, None)), str) and value
    }


def _message(row: sqlite3.Row) -> InboxMessage:
    return InboxMessage(
        delivery_id=str(row["delivery_id"]),
        sender=str(row["sender"]),
        body=str(row["body"]),
        state=DeliveryState(row["state"]),
        event_key=row["event_key"],
        requires_ack=bool(row["requires_ack"]),
    )


def _verify_body(row: sqlite3.Row, body: str) -> None:
    if row["body"] != body or row["body_sha256"] != _digest(body):
        raise RuntimeError(f"delivery {row['delivery_id']} payload mismatch")


def _digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sender(delivery_id: str) -> str:
    digest = hashlib.sha256(delivery_id.encode("utf-8")).hexdigest()
    return f"{_SENDER_PREFIX}{digest}"


def _legacy_prompt_delivery_id(
    delivery_ids: Sequence[str],
    *,
    identity: str | None = None,
) -> str:
    prompt_identity = identity or "turn:" + "|".join(sorted(delivery_ids))
    return (
        "controller-prompt:"
        f"{hashlib.sha256(prompt_identity.encode()).hexdigest()}"
    )
