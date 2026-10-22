"""Publish model-request activity without adding conversation events."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite
from queue import Empty, SimpleQueue

from senpai_agent.provider_retry import retry_after_seconds


InferenceState = tuple[float | None, float | None, float | None]
InferenceStateCallback = Callable[[float | None, float | None, float | None], None]
_STOP = None
INFERENCE_CANCELLATION_GRACE_SECONDS = 60


class ModelRequestDeadlineExceeded(TimeoutError):
    """One logical model request exhausted its total wall-clock budget."""

    retry_at: float = 0


@dataclass
class _Request:
    started_at: float
    deadline: float
    retry_at: float = 0


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
        self._active: dict[object, _Request] = {}
        self._expired: _Request | None = None
        self._requests: ContextVar[tuple[_Request, ...]] = ContextVar(
            "inference_requests", default=(),
        )
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
            state = self._active[token] = _Request(
                time.time(),
                time.monotonic() + self.timeout_seconds,
            )
            self._enqueue_state_locked()
        context_token = self._requests.set((*self._requests.get(), state))
        try:
            yield
        finally:
            self._requests.reset(context_token)
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
                state = min(self._active.values(), key=lambda item: item.deadline)
                if time.monotonic() >= state.deadline:
                    self._expired = state
                return state.deadline
            return self._expired.deadline if self._expired is not None else None

    def record_retry(self, error: BaseException) -> None:
        """Retain provider guidance for this request and its enclosing scopes."""
        now = time.time()
        retry_at = now + retry_after_seconds(error, now)
        with self._lock:
            for request in self._requests.get():
                request.retry_at = max(request.retry_at, retry_at)

    def expired_retry_at(self) -> float:
        with self._lock:
            return self._expired.retry_at if self._expired is not None else 0

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
            state = self._expired
            self._states.put((state.started_at, time.time(), state.deadline))
            return
        if not self._active:
            self._states.put((None, None, None))
            return
        state = min(self._active.values(), key=lambda item: item.deadline)
        self._states.put((state.started_at, time.time(), state.deadline))
