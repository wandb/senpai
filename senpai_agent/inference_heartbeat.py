"""Publish model-request activity without adding conversation events."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from queue import Empty, SimpleQueue


InferenceState = tuple[float | None, float | None]
InferenceStateCallback = Callable[[float | None, float | None], None]
_STOP = None


class InferenceHeartbeat:
    """Track active requests and publish their earliest start plus a live pulse."""

    def __init__(
        self,
        publish_state: InferenceStateCallback,
        *,
        interval_seconds: float = 30,
    ):
        if interval_seconds <= 0:
            raise ValueError("inference heartbeat interval must be positive")
        self.publish_state = publish_state
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._active: dict[object, float] = {}
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
            self._active[token] = time.time()
            self._enqueue_state_locked()
        try:
            yield
        finally:
            with self._lock:
                self._active.pop(token, None)
                self._enqueue_state_locked()

    def close(self) -> None:
        """Flush the final state and stop the publisher."""
        with self._lock:
            if self._active:
                raise RuntimeError("cannot close an active inference heartbeat")
            if not self._closed:
                self._closed = True
                self._states.put(_STOP)
        self._worker.join()
        if self._worker_error is not None:
            raise self._worker_error

    def _publish(self) -> None:
        started_at = None
        try:
            while True:
                try:
                    state = self._states.get(
                        timeout=(
                            self.interval_seconds if started_at is not None else None
                        )
                    )
                except Empty:
                    self.publish_state(started_at, time.time())
                    continue
                if state is _STOP:
                    return
                started_at, heartbeat_at = state
                self.publish_state(started_at, heartbeat_at)
        except BaseException as error:  # noqa: BLE001
            self._worker_error = error

    def _enqueue_state_locked(self) -> None:
        if not self._active:
            self._states.put((None, None))
            return
        self._states.put((min(self._active.values()), time.time()))
