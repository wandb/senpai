"""Preserve contradictory event evidence without replacing accepted messages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass


@dataclass
class EventIdentityConflict(RuntimeError):
    conversation_id: str
    event_key: str
    original_body: str
    candidate_body: str

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(
                [
                    self.conversation_id,
                    self.event_key,
                    self.original_body,
                    self.candidate_body,
                ],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    def __str__(self) -> str:
        return f"event {self.event_key!r} was reused with a different payload"


class DeliveryBlocked(RuntimeError):
    """An unresolved event conflict prevents decisions in this conversation."""


def create_conflict_table(database: sqlite3.Connection) -> None:
    database.execute("""
        CREATE TABLE IF NOT EXISTS event_conflicts (
            identity TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            event_key TEXT NOT NULL,
            original_body TEXT NOT NULL,
            candidate_body TEXT NOT NULL,
            observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            resolution TEXT,
            resolved_at TEXT
        )
    """)


def record_conflict(
    database: sqlite3.Connection, conflict: EventIdentityConflict
) -> None:
    inserted = database.execute(
        """INSERT OR IGNORE INTO event_conflicts
           (identity, conversation_id, event_key, original_body, candidate_body)
           VALUES (?, ?, ?, ?, ?)""",
        (
            conflict.identity,
            conflict.conversation_id,
            conflict.event_key,
            conflict.original_body,
            conflict.candidate_body,
        ),
    ).rowcount
    if inserted:
        print(
            f"SENPAI_EVENT_CONFLICT identity={conflict.identity} "
            f"conversation_id={conflict.conversation_id} event_key={conflict.event_key}",
            file=sys.stderr,
            flush=True,
        )


def unresolved_conflicts(
    database: sqlite3.Connection,
) -> tuple[EventIdentityConflict, ...]:
    return tuple(
        EventIdentityConflict(*row)
        for row in database.execute(
            """SELECT conversation_id, event_key, original_body, candidate_body
           FROM event_conflicts WHERE resolution IS NULL ORDER BY rowid"""
        ).fetchall()
    )


def resolve_conflict(database: sqlite3.Connection, identity: str, reason: str) -> None:
    """Reject the conflicting candidate explicitly; retain both bodies and their history."""
    if not reason.strip():
        raise ValueError("conflict resolution requires an evidence-backed reason")
    updated = database.execute(
        """UPDATE event_conflicts SET resolution = ?, resolved_at = CURRENT_TIMESTAMP
           WHERE identity = ? AND resolution IS NULL""",
        (reason, identity),
    ).rowcount
    if updated:
        return
    existing = database.execute(
        "SELECT resolution FROM event_conflicts WHERE identity = ?",
        (identity,),
    ).fetchone()
    if existing is None:
        raise KeyError(identity)
    if existing[0] != reason:
        raise ValueError("conflict already has a different resolution")


def withdraw_monitor_signal(
    database: sqlite3.Connection, event_key: str, reason: str
) -> None:
    """Withdraw a monitor candidate without changing its payload or delivery receipt."""
    if not reason.strip():
        raise ValueError("withdrawal requires an evidence-backed reason")
    updated = database.execute(
        """UPDATE monitor_signals SET superseded_reason = ?
           WHERE dedupe_key = ? AND handled = 0 AND superseded_reason IS NULL""",
        (reason, event_key),
    ).rowcount
    if updated:
        return
    existing = database.execute(
        "SELECT handled, superseded_reason FROM monitor_signals WHERE dedupe_key = ?",
        (event_key,),
    ).fetchone()
    if existing is None:
        raise KeyError(event_key)
    if existing[0] or existing[1] != reason:
        raise ValueError(
            "notification was already handled or withdrawn with a different reason"
        )


def main() -> None:
    """Inspect conflicts or record an operator's explicit rejection in an existing store.

    Repair or withdraw the producer's conflicting candidate before rejecting it.
    This command never rewrites messages, reissues events, or acknowledges delivery.
    If a local staging conflict has reached the inbox, resolve it in both stores.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("store", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    reject = commands.add_parser("reject")
    reject.add_argument("identity")
    reject.add_argument("--reason", required=True)
    withdraw = commands.add_parser("withdraw-monitor")
    withdraw.add_argument("event_key")
    withdraw.add_argument("--reason", required=True)
    args = parser.parse_args()
    mode = "ro" if args.command == "list" else "rw"
    with sqlite3.connect(
        f"{args.store.resolve().as_uri()}?mode={mode}", uri=True, timeout=30
    ) as database:
        table = (
            "monitor_signals"
            if args.command == "withdraw-monitor"
            else "event_conflicts"
        )
        if (
            database.execute(
                "SELECT 1 FROM sqlite_schema WHERE name=? AND type='table'", (table,)
            ).fetchone()
            is None
        ):
            parser.error(
                f"{table} is absent; select the correct store after upgrading the controller"
            )
        if args.command == "reject":
            resolve_conflict(database, args.identity, args.reason)
        elif args.command == "withdraw-monitor":
            withdraw_monitor_signal(database, args.event_key, args.reason)
        else:
            database.row_factory = sqlite3.Row
            rows = database.execute(
                "SELECT * FROM event_conflicts ORDER BY rowid"
            ).fetchall()
            print(json.dumps([dict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
