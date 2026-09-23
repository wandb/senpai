"""One controller-lifetime GitHub snapshot and active-turn event source."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from senpai_agent.github.http import GitHubRateLimitError, GitHubReadError
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import ControllerEvent, Mailbox
from senpai_agent.wake import WakeCoordinator


@dataclass(slots=True)
class _ActiveBinding:
    conversation_id: object
    known_keys: set[str]
    enqueued_keys: set[str] = field(default_factory=set)


class GitHubMailboxWatcher:
    """Poll GitHub once per controller and expose its latest complete snapshot."""

    _SHUTDOWN_TIMEOUT_SECONDS = 35.0

    def __init__(
        self,
        mailbox: Mailbox,
        store_path: Path,
        *,
        coordinator: WakeCoordinator,
        poll_interval_seconds: float = 300,
        map_event: (
            Callable[[ControllerEvent, object], LocalEvent | None] | None
        ) = None,
    ):
        if poll_interval_seconds <= 0:
            raise ValueError("GitHub poll interval must be positive")
        self.mailbox = mailbox
        self.store_path = store_path
        self.coordinator = coordinator
        self.poll_interval_seconds = poll_interval_seconds
        self.map_event = map_event or _local_event
        self._snapshot: tuple[ControllerEvent, ...] = ()
        self._has_snapshot = False
        self._binding: _ActiveBinding | None = None
        self._state_lock = threading.Lock()
        self._mailbox_lock = threading.Lock()
        self._attempted = threading.Condition()
        self._attempts = 0
        self._retry_not_before = 0.0
        self._refresh = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._last_read_error: GitHubReadError | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="senpai-github-watcher",
            daemon=True,
        )

    def _run(self) -> None:
        try:
            wait_seconds = 0.0
            while not self._stop.is_set():
                if wait_seconds > 0:
                    self._refresh.wait(wait_seconds)
                    self._refresh.clear()
                    if self._stop.is_set():
                        break
                    with self._attempted:
                        wait_seconds = max(
                            0.0,
                            self._retry_not_before - time.monotonic(),
                        )
                    if wait_seconds > 0:
                        continue
                retry_after = self._poll_once()
                cooldown = (
                    max(self.poll_interval_seconds, retry_after)
                    if retry_after is not None
                    else None
                )
                with self._attempted:
                    self._attempts += 1
                    self._retry_not_before = (
                        time.monotonic() + cooldown
                        if cooldown is not None
                        else 0.0
                    )
                    self._attempted.notify_all()
                wait_seconds = cooldown or self.poll_interval_seconds
        except BaseException as error:  # noqa: BLE001
            with self._attempted:
                self._error = error
                self._stop.set()
                self._attempted.notify_all()
            self.coordinator.fail(error)

    def _poll_once(self) -> float | None:
        try:
            with self._mailbox_lock:
                events = tuple(self.mailbox.poll())
        except GitHubRateLimitError as error:
            with self._state_lock:
                self._last_read_error = error
            print(
                "SENPAI_GITHUB_WATCHER_RATE_LIMIT "
                f"retry_after_seconds={error.retry_after_seconds:g} "
                f"error={error}",
                file=sys.stderr,
                flush=True,
            )
            return error.retry_after_seconds
        except GitHubReadError as error:
            with self._state_lock:
                self._last_read_error = error
            print(
                "SENPAI_GITHUB_WATCHER_POLL_ERROR "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            return None
        with self._state_lock:
            changed = not self._has_snapshot or events != self._snapshot
            self._snapshot = events
            self._has_snapshot = True
            self._last_read_error = None
            self._stage_active(events)
        if changed:
            self.coordinator.wake()
        return None

    def _stage_active(
        self,
        events: Sequence[ControllerEvent],
        *,
        initial: bool = False,
    ) -> None:
        binding = self._binding
        if binding is None:
            return
        current = {event.dedupe_key for event in events}
        with LocalEventStore(self.store_path) as store:
            for event in events:
                if event.kind == "student_available_for_assignment":
                    continue
                if event.dedupe_key in binding.known_keys:
                    continue
                local_event = self.map_event(event, binding.conversation_id)
                if local_event is not None and store.enqueue(local_event):
                    binding.enqueued_keys.add(local_event.dedupe_key)
        if initial:
            binding.known_keys.update(current)
        else:
            binding.known_keys = current

    def poll(self) -> tuple[ControllerEvent, ...]:
        self._raise_if_failed()
        with self._state_lock:
            if not self._has_snapshot:
                raise self._last_read_error or GitHubReadError(
                    "GitHub watcher has no complete snapshot"
                )
            return self._snapshot

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        with self._mailbox_lock:
            self.mailbox.acknowledge(dedupe_keys)

    def request_refresh(self) -> None:
        with self._attempted:
            if self._error is not None:
                self._raise_if_failed()
            if time.monotonic() < self._retry_not_before:
                return
            self._refresh.set()

    def error(self) -> BaseException | None:
        with self._attempted:
            return self._error

    @contextmanager
    def bind_active(
        self,
        conversation_id: object,
        *,
        visible_event_keys: frozenset[str],
    ) -> Iterator[_ActiveBinding]:
        self._raise_if_failed()
        binding = _ActiveBinding(conversation_id, set(visible_event_keys))
        with self._state_lock:
            if self._binding is not None:
                raise RuntimeError("GitHub watcher already has an active turn")
            self._binding = binding
            try:
                self._stage_active(self._snapshot, initial=True)
            except BaseException:
                self._binding = None
                raise
        primary_error = True
        try:
            yield binding
            primary_error = False
        finally:
            with self._state_lock:
                if self._binding is binding:
                    self._binding = None
            if not primary_error:
                self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("GitHub watcher failed") from self._error

    def __enter__(self) -> Self:
        try:
            self._thread.start()
            with self._attempted:
                self._attempted.wait_for(
                    lambda: (
                        self._attempts > 0
                        or self._error is not None
                        or self._stop.is_set()
                    )
                )
            self._raise_if_failed()
            if self._stop.is_set():
                raise RuntimeError("GitHub watcher closed during startup")
            return self
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._stop.set()
        self._refresh.set()
        cancel_poll = getattr(self.mailbox, "cancel_poll", None)
        if callable(cancel_poll):
            cancel_poll()
        with self._attempted:
            self._attempted.notify_all()
        if self._thread.ident is None or self._thread is threading.current_thread():
            return
        self._thread.join(self._SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            print(
                "SENPAI_GITHUB_WATCHER_SHUTDOWN_TIMEOUT "
                f"seconds={self._SHUTDOWN_TIMEOUT_SECONDS:g}",
                file=sys.stderr,
                flush=True,
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
        if exc_type is None:
            self._raise_if_failed()


def _local_event(event: ControllerEvent, _conversation_id: object) -> LocalEvent:
    return LocalEvent(
        kind=event.kind,
        dedupe_key=event.dedupe_key,
        payload=event.payload,
    )
