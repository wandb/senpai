"""Publish model-request activity without adding conversation events."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import isfinite
from queue import Empty, SimpleQueue


InferenceState = tuple[float | None, float | None, float | None]
InferenceStateCallback = Callable[[float | None, float | None, float | None], None]
_STOP = None
INFERENCE_CANCELLATION_GRACE_SECONDS = 60


class ModelRequestDeadlineExceeded(TimeoutError):
    """One logical model request exhausted its total wall-clock budget."""


class InferenceHeartbeat:
    """Publish local request activity with a fixed monotonic deadline."""

    def __init__(
        self,
        publish_state: InferenceStateCallback,
        *,
        timeout_seconds: float,
        interval_seconds: float = 30,
    ):
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("inference request timeout must be finite and positive")
        if interval_seconds <= 0:
            raise ValueError("inference heartbeat interval must be positive")
        self.publish_state = publish_state
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._active: dict[object, tuple[float, float]] = {}
        self._expired: tuple[float, float] | None = None
        self._states: SimpleQueue[InferenceState | None] = SimpleQueue()
        self._closed = False
        self._worker_error: BaseException | None = None
        self._worker = threading.Thread(
            target=self._publish,
            name="senpai-inference-heartbeat",
            daemon=True,
        )
        self._worker.start()

    @contextmanager
    def request(self) -> Iterator[None]:
        token = object()
        with self._lock:
            if self._closed:
                raise RuntimeError("inference heartbeat is closed")
            self._active[token] = (
                time.time(),
                time.monotonic() + self.timeout_seconds,
            )
            self._enqueue_state_locked()
        try:
            yield
        finally:
            with self._lock:
                self._active.pop(token, None)
                self._enqueue_state_locked()

    def deadline(self) -> float | None:
        """Observe the earliest budget, retaining observed expiry through cleanup.

        The watchdog calls this before cancelling. A request that completes
        before expiry is observed clears normally, even if it finished late.
        """
        with self._lock:
            if self._expired is None and self._active:
                state = min(self._active.values(), key=lambda item: item[1])
                if time.monotonic() >= state[1]:
                    self._expired = state
                return state[1]
            return self._expired[1] if self._expired is not None else None

    def close(self) -> None:
        """Flush the final state and stop the publisher."""
        with self._lock:
            if self._active:
                raise RuntimeError("cannot close an active inference heartbeat")
            if not self._closed:
                self._expired = None
                self._enqueue_state_locked()
                self._closed = True
                self._states.put(_STOP)
        self._worker.join()
        if self._worker_error is not None:
            raise self._worker_error

    def _publish(self) -> None:
        started_at = None
        deadline = None
        try:
            while True:
                try:
                    state = self._states.get(
                        timeout=(
                            self.interval_seconds if started_at is not None else None
                        )
                    )
                except Empty:
                    self.publish_state(started_at, time.time(), deadline)
                    continue
                if state is _STOP:
                    return
                started_at, heartbeat_at, deadline = state
                self.publish_state(started_at, heartbeat_at, deadline)
        except BaseException as error:  # noqa: BLE001
            self._worker_error = error

    def _enqueue_state_locked(self) -> None:
        if self._expired is not None:
            started_at, deadline = self._expired
            self._states.put((started_at, time.time(), deadline))
            return
        if not self._active:
            self._states.put((None, None, None))
            return
        started_at, deadline = min(self._active.values(), key=lambda item: item[1])
        self._states.put((started_at, time.time(), deadline))
