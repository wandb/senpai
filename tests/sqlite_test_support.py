from __future__ import annotations

import threading
from collections.abc import Callable


def assert_concurrent_first_open(
    factory: Callable[[], object], *, workers: int = 6
) -> None:
    """Require every simultaneous public store constructor to finish cleanly."""

    start = threading.Barrier(workers + 1)
    opened = threading.Barrier(workers)
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def open_store() -> None:
        store = None
        try:
            start.wait(timeout=5)
            store = factory()
            opened.wait(timeout=5)
        except Exception as error:
            with failure_lock:
                failures.append(error)
            opened.abort()
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()

    threads = [threading.Thread(target=open_store, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert not [thread for thread in threads if thread.is_alive()]
    assert not failures, [f"{type(error).__name__}: {error}" for error in failures]


def assert_repeated_concurrent_first_open(
    factory: Callable[[int], object],
    *,
    attempts: int,
) -> None:
    for attempt in range(attempts):
        assert_concurrent_first_open(
            lambda attempt=attempt: factory(attempt),
            workers=8,
        )
