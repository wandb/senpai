from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlite_test_support import assert_concurrent_first_open

from senpai_agent.sqlite_store import initialize_sqlite_store


def test_initialized_store_uses_rollback_journal(tmp_path: Path) -> None:
    database = initialize_sqlite_store(
        tmp_path / "state.sqlite3",
        lambda connection: connection.execute(
            "CREATE TABLE state (value TEXT NOT NULL)"
        ),
    )
    try:
        mode = database.execute("PRAGMA journal_mode").fetchone()
    finally:
        database.close()

    assert mode is not None
    assert mode[0] == "delete"


def test_concurrent_first_open_migrates_legacy_wal_store(tmp_path: Path) -> None:
    path = tmp_path / "legacy-wal.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.execute("PRAGMA journal_mode=WAL")
    legacy.execute("CREATE TABLE state (value TEXT NOT NULL)")
    legacy.commit()
    legacy.close()

    def open_store():
        return initialize_sqlite_store(
            path,
            lambda connection: connection.execute(
                "CREATE TABLE IF NOT EXISTS state (value TEXT NOT NULL)"
            ),
        )

    assert_concurrent_first_open(open_store, workers=8)

    migrated = sqlite3.connect(path)
    try:
        mode = migrated.execute("PRAGMA journal_mode").fetchone()
        table = migrated.execute(
            "SELECT name FROM sqlite_master WHERE name = 'state'"
        ).fetchone()
    finally:
        migrated.close()

    assert mode == ("delete",)
    assert table == ("state",)
