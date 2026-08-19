"""Lost-wake-safe coordination for controller-local event sources."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingHint:
    training_id: str
    version: int


class WakeCoordinator:
    """Wake one controller and coalesce terminal training hints."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._generation = 0
        self._training_versions: dict[str, int] = {}
        self._error: BaseException | None = None

    def checkpoint(self) -> int:
        with self._condition:
            self._raise_if_failed()
            return self._generation

    def wake(self) -> None:
        with self._condition:
            self._generation += 1
            self._condition.notify_all()

    def training_finished(self, training_id: str) -> None:
        with self._condition:
            self._generation += 1
            self._training_versions[training_id] = self._generation
            self._condition.notify_all()

    def training_hints(self) -> tuple[TrainingHint, ...]:
        with self._condition:
            return tuple(
                TrainingHint(training_id, version)
                for training_id, version in sorted(self._training_versions.items())
            )

    def acknowledge_training(self, hints: tuple[TrainingHint, ...]) -> None:
        with self._condition:
            for hint in hints:
                if self._training_versions.get(hint.training_id) == hint.version:
                    del self._training_versions[hint.training_id]

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._generation += 1
            self._condition.notify_all()

    def wait(self, checkpoint: int, *, timeout_seconds: float) -> bool:
        with self._condition:
            changed = self._condition.wait_for(
                lambda: self._generation != checkpoint or self._error is not None,
                timeout=max(0, timeout_seconds),
            )
            self._raise_if_failed()
            return changed

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("event source failed") from self._error
