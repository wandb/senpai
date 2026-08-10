"""Shared, race-safe SQLite setup for Senpai's process-visible stores.

Stores use SQLite's rollback journal because Senpai may place state on a
filesystem where WAL's shared-memory contract is unavailable. The filesystem
must still provide correct POSIX locking and atomic file operations.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

_INITIALIZATION_TIMEOUT_SECONDS = 30
_RUNTIME_BUSY_TIMEOUT_SECONDS = 5
_RETRY_INTERVAL_SECONDS = 0.01


def connect_sqlite_store(
    path: Path,
    *,
    busy_timeout_seconds: float = _RUNTIME_BUSY_TIMEOUT_SECONDS,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """Open an initialized store with consistent row and lock semantics."""

    connection = sqlite3.connect(
        path,
        timeout=busy_timeout_seconds,
        check_same_thread=check_same_thread,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={round(busy_timeout_seconds * 1000)}")
    return connection


def initialize_sqlite_store(
    path: Path | None,
    initialize: Callable[[sqlite3.Connection], None],
    *,
    runtime_busy_timeout_seconds: float = _RUNTIME_BUSY_TIMEOUT_SECONDS,
) -> sqlite3.Connection:
    """Open a store and serialize journal negotiation plus schema migration."""

    database_path = path.expanduser().resolve() if path is not None else None
    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
    lock = (
        _initialization_lock(database_path)
        if database_path is not None
        else nullcontext()
    )
    with lock:
        connection = connect_sqlite_store(
            database_path or Path(":memory:"),
            busy_timeout_seconds=_INITIALIZATION_TIMEOUT_SECONDS,
        )
        try:
            if database_path is not None:
                _enable_rollback_journal(connection)
            connection.execute("BEGIN IMMEDIATE")
            initialize(connection)
            connection.commit()
            connection.execute(
                f"PRAGMA busy_timeout={round(runtime_busy_timeout_seconds * 1000)}"
            )
            if database_path is not None:
                database_path.chmod(0o600)
            return connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
            raise


@contextmanager
def _initialization_lock(database_path: Path) -> Iterator[None]:
    lock_path = database_path.with_name(f"{database_path.name}.init.lock")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        deadline = time.monotonic() + _INITIALIZATION_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out initializing SQLite store {database_path}"
                    ) from None
                time.sleep(_RETRY_INTERVAL_SECONDS)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _enable_rollback_journal(connection: sqlite3.Connection) -> None:
    deadline = time.monotonic() + _INITIALIZATION_TIMEOUT_SECONDS
    while True:
        try:
            row = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            if row is None or str(row[0]).lower() != "delete":
                raise RuntimeError("Senpai SQLite stores require rollback-journal mode")
            return
        except sqlite3.OperationalError as error:
            detail = str(error).lower()
            if (
                not any(word in detail for word in ("busy", "locked"))
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(_RETRY_INTERVAL_SECONDS)
