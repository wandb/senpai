"""Durable local event values and storage."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.PROMPTS import (
    LOCAL_EVENT_PROMPT,
    render_event_prompt,
    render_prompt,
)

_LOCAL_EVENT_STORE_SETUP_LOCK = threading.Lock()


class LocalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    dedupe_key: str
    payload: dict[str, object]
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_user_message(self) -> str:
        payload = json.dumps(self.payload, indent=2, sort_keys=True)
        observed_at = self.observed_at.astimezone(UTC).isoformat()
        return render_prompt(
            LOCAL_EVENT_PROMPT,
            KIND=self.kind,
            OBSERVED_AT=observed_at,
            PAYLOAD=payload,
        )

    def to_inbox_message(self) -> str:
        payload = {
            key: value
            for key, value in self.payload.items()
            if key != "parent_conversation_id"
        }
        return render_event_prompt(self.kind, payload)


class LocalEventStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with _LOCAL_EVENT_STORE_SETUP_LOCK:
            self._connection = sqlite3.connect(path, check_same_thread=False)
            self._connection.execute("PRAGMA busy_timeout=5000")
            journal_mode = self._connection.execute("PRAGMA journal_mode").fetchone()
            if journal_mode != ("wal",):
                self._connection.execute("PRAGMA journal_mode=WAL")
            table_exists = self._connection.execute(
                """
                SELECT 1
                FROM sqlite_schema
                WHERE type = 'table' AND name = 'advisor_events'
                """
            ).fetchone()
            if table_exists is None:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS advisor_events (
                        dedupe_key TEXT PRIMARY KEY,
                        event_json TEXT NOT NULL,
                        acknowledged INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS advisor_event_acknowledgements (
                    dedupe_key TEXT PRIMARY KEY
                )
                """
            )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO advisor_event_acknowledgements (dedupe_key)
                SELECT dedupe_key FROM advisor_events WHERE acknowledged = 1
                """
            )
            self._connection.commit()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def enqueue(self, event: LocalEvent) -> bool:
        with self._transaction():
            acknowledged = (
                self._connection.execute(
                    """
                SELECT 1 FROM advisor_event_acknowledgements
                WHERE dedupe_key = ?
                """,
                    (event.dedupe_key,),
                ).fetchone()
                is not None
            )
            cursor = self._connection.execute(
                """
                INSERT INTO advisor_events (dedupe_key, event_json, acknowledged)
                VALUES (?, ?, ?)
                ON CONFLICT(dedupe_key) DO NOTHING
                """,
                (event.dedupe_key, event.model_dump_json(), int(acknowledged)),
            )
            row = self._connection.execute(
                """
                SELECT event_json, acknowledged
                FROM advisor_events
                WHERE dedupe_key = ?
                """,
                (event.dedupe_key,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"event {event.dedupe_key!r} disappeared")
            existing = LocalEvent.model_validate_json(row[0])
            if existing.kind != event.kind or existing.payload != event.payload:
                raise RuntimeError(
                    f"event {event.dedupe_key!r} was reused with a different payload"
                )
            if acknowledged and not row[1]:
                self._connection.execute(
                    """
                    UPDATE advisor_events SET acknowledged = 1
                    WHERE dedupe_key = ?
                    """,
                    (event.dedupe_key,),
                )
            return cursor.rowcount == 1 and not acknowledged

    def pending(self) -> list[LocalEvent]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT event_json
                FROM advisor_events
                WHERE acknowledged = 0
                ORDER BY rowid
                """
            ).fetchall()
        return [LocalEvent.model_validate_json(row[0]) for row in rows]

    def pending_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*)
                FROM advisor_events
                WHERE acknowledged = 0
                """
            ).fetchone()
        return int(row[0])

    def acknowledge(self, dedupe_key: str) -> None:
        with self._transaction():
            self._connection.execute(
                """
                INSERT OR IGNORE INTO advisor_event_acknowledgements (dedupe_key)
                VALUES (?)
                """,
                (dedupe_key,),
            )
            self._connection.execute(
                "UPDATE advisor_events SET acknowledged = 1 WHERE dedupe_key = ?",
                (dedupe_key,),
            )

    def discard_prefix(
        self,
        dedupe_key_prefix: str,
        *,
        retained_keys: Sequence[str] = (),
    ) -> int:
        """Remove staged level-triggers absent from the current snapshot."""

        if not dedupe_key_prefix:
            raise ValueError("dedupe key prefix must not be empty")
        retained = tuple(dict.fromkeys(retained_keys))
        retention_clause = ""
        if retained:
            placeholders = ",".join("?" for _ in retained)
            retention_clause = f"AND dedupe_key NOT IN ({placeholders})"
        with self._transaction():
            cursor = self._connection.execute(
                f"""
                DELETE FROM advisor_events
                WHERE substr(dedupe_key, 1, ?) = ?
                  {retention_clause}
                """,
                (len(dedupe_key_prefix), dedupe_key_prefix, *retained),
            )
            self._connection.execute(
                f"""
                DELETE FROM advisor_event_acknowledgements
                WHERE substr(dedupe_key, 1, ?) = ?
                  {retention_clause}
                """,
                (len(dedupe_key_prefix), dedupe_key_prefix, *retained),
            )
            return cursor.rowcount

    def acknowledged(self, dedupe_keys: Sequence[str]) -> set[str]:
        if not dedupe_keys:
            return set()
        placeholders = ",".join("?" for _ in dedupe_keys)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT dedupe_key
                FROM advisor_event_acknowledgements
                WHERE dedupe_key IN ({placeholders})
                """,
                tuple(dedupe_keys),
            ).fetchall()
        return {str(row[0]) for row in rows}

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
