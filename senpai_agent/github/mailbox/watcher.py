"""Background delivery of newly observed GitHub mailbox events."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Self

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.github.http import GitHubReadError
from senpai_agent.mailbox import ControllerEvent

from .core import GitHubMailbox


class ActiveGitHubWatcher:
    """Feed new GitHub state into a running agent at SDK-safe boundaries."""

    def __init__(
        self,
        mailbox: GitHubMailbox,
        store_path: Path,
        *,
        known_keys: frozenset[str],
        poll_interval_seconds: float = 30,
        map_event: Callable[[ControllerEvent], AdvisorEvent | None] | None = None,
    ):
        self.mailbox = mailbox
        self.store_path = store_path
        self.known_keys = set(known_keys)
        self.poll_interval_seconds = poll_interval_seconds
        self.map_event = map_event or _advisor_event
        self.enqueued_keys: set[str] = set()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="senpai-github-watcher",
        )

    def _run(self) -> None:
        try:
            with AdvisorEventStore(self.store_path) as store:
                delay = self.poll_interval_seconds
                while not self.stop.wait(delay):
                    try:
                        events = self.mailbox.poll()
                    except GitHubReadError as error:
                        delay = max(
                            self.poll_interval_seconds,
                            error.retry_after_seconds
                            if error.retry_after_seconds is not None
                            else min(delay * 2, 600),
                        )
                        print(
                            "SENPAI_GITHUB_WATCHER_POLL_ERROR "
                            f"{type(error).__name__}: {error}; "
                            f"retry_after_seconds={delay:g}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    delay = self.poll_interval_seconds
                    current = {event.dedupe_key for event in events}
                    for event in events:
                        if event.dedupe_key in self.known_keys:
                            continue
                        local_event = self.map_event(event)
                        if local_event is None:
                            continue
                        if store.enqueue(local_event):
                            self.enqueued_keys.add(local_event.dedupe_key)
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


def _advisor_event(event: ControllerEvent) -> AdvisorEvent:
    return AdvisorEvent(
        kind=event.kind,
        dedupe_key=event.dedupe_key,
        payload=event.payload,
    )
