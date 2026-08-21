"""Background delivery of newly observed GitHub mailbox events."""

from __future__ import annotations

import sqlite3
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Self

from senpai_agent.event_kinds import EventKind
from senpai_agent.github.http import GitHubReadError
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import (
    ControllerEvent,
    Mailbox,
    report_event_render_error,
)


class ActiveGitHubWatcher:
    """Feed new GitHub state into a running agent at SDK-safe boundaries."""

    def __init__(
        self,
        mailbox: Mailbox,
        store_path: Path,
        *,
        known_keys: frozenset[str],
        poll_interval_seconds: float = 75,
        map_event: Callable[[ControllerEvent], LocalEvent | None] | None = None,
    ):
        self.mailbox = mailbox
        self.store_path = store_path
        self.known_keys = set(known_keys)
        self.poll_interval_seconds = poll_interval_seconds
        self.map_event = map_event or _local_event
        self.enqueued_keys: set[str] = set()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="senpai-github-watcher",
        )

    def _run(self) -> None:
        try:
            with LocalEventStore(self.store_path) as store:
                while not self.stop.wait(self.poll_interval_seconds):
                    try:
                        events = self.mailbox.poll()
                    except GitHubReadError as error:
                        print(
                            "SENPAI_GITHUB_WATCHER_POLL_ERROR "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    current = {event.dedupe_key for event in events}
                    for event in events:
                        # The foreground poll reconciles availability before the
                        # next turn; staging it here would preserve stale state.
                        if event.kind == EventKind.STUDENT_AVAILABLE_FOR_ASSIGNMENT:
                            continue
                        if event.dedupe_key in self.known_keys:
                            continue
                        try:
                            local_event = self.map_event(event)
                            if local_event is None:
                                continue
                            if store.enqueue(local_event):
                                self.enqueued_keys.add(local_event.dedupe_key)
                        except sqlite3.Error:
                            raise
                        except Exception as error:  # noqa: BLE001
                            report_event_render_error(
                                event.kind,
                                event.dedupe_key,
                                error,
                                disposition="deferred",
                            )
                    self.known_keys = current
        except BaseException as error:  # noqa: BLE001
            self.error = error
            self.stop.set()

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.stop.set()
        self.thread.join()
        if exc_type is None and self.error is not None:
            print(
                "SENPAI_GITHUB_WATCHER_ERROR "
                f"{type(self.error).__name__}: {self.error}",
                file=sys.stderr,
                flush=True,
            )


def _local_event(event: ControllerEvent) -> LocalEvent:
    return LocalEvent(
        kind=event.kind,
        dedupe_key=event.dedupe_key,
        payload=event.payload,
    )
