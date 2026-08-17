"""Persist mailbox events observed while an agent turn is running."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import TracebackType
from typing import Self

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.mailbox import ControllerEvent, Mailbox


MailboxFactory = Callable[[], AbstractContextManager[Mailbox]]


class ActiveMailboxWatcher:
    """Poll a mailbox quietly without injecting events into the active turn."""

    def __init__(
        self,
        mailbox: Mailbox | MailboxFactory,
        store_path: Path,
        *,
        known_keys: frozenset[str] = frozenset(),
        poll_interval_seconds: float = 30,
        map_event: Callable[[ControllerEvent], AdvisorEvent | None] | None = None,
        shutdown_timeout_seconds: float = 1,
        thread_name: str = "senpai-mailbox-watcher",
    ):
        if poll_interval_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("watcher intervals must be positive")
        self.mailbox = mailbox
        self.store_path = store_path
        self.known_keys = set(known_keys)
        self.poll_interval_seconds = poll_interval_seconds
        self.map_event = map_event or _advisor_event
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )

    def _run(self) -> None:
        try:
            mailbox_context = (
                self.mailbox()
                if callable(self.mailbox)
                else nullcontext(self.mailbox)
            )
            with mailbox_context as mailbox, AdvisorEventStore(self.store_path) as store:
                delay = self.poll_interval_seconds
                while not self.stop.wait(delay):
                    try:
                        events = mailbox.poll()
                    except Exception as error:  # noqa: BLE001
                        retry_after = getattr(error, "retry_after_seconds", None)
                        delay = (
                            max(self.poll_interval_seconds, retry_after)
                            if retry_after is not None
                            else min(max(delay * 2, self.poll_interval_seconds), 600)
                        )
                        print(
                            "SENPAI_MAILBOX_WATCHER_POLL_ERROR "
                            f"{type(error).__name__}: {error}; "
                            f"retry_after_seconds={delay:g}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    current = {event.dedupe_key for event in events}
                    for event in events:
                        if event.dedupe_key in self.known_keys:
                            continue
                        local_event = self.map_event(event)
                        if local_event is None:
                            continue
                        store.enqueue(local_event)
                    self.known_keys = current
                    delay = self.poll_interval_seconds
        except BaseException as error:  # noqa: BLE001
            self.error = error
            self.stop.set()
            print(
                "SENPAI_MAILBOX_WATCHER_ERROR "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.stop.set()
        self.thread.join(self.shutdown_timeout_seconds)
        if self.thread.is_alive():
            print(
                "SENPAI_MAILBOX_WATCHER_SHUTDOWN_TIMEOUT "
                f"thread={self.thread.name}",
                file=sys.stderr,
                flush=True,
            )


def _advisor_event(event: ControllerEvent) -> AdvisorEvent:
    return AdvisorEvent(
        kind=event.kind,
        dedupe_key=event.dedupe_key,
        payload=event.payload,
    )
