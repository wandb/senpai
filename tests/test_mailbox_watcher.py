import threading
import time
from contextlib import contextmanager
from pathlib import Path

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.mailbox_watcher import ActiveMailboxWatcher


class EventuallyReadyMailbox:
    def __init__(self):
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self.polls == 1:
            raise RuntimeError("temporary read failure")
        return (
            ControllerEvent(
                kind="job_monitor",
                dedupe_key="job-1:status:failed",
                payload={"job_id": "job-1", "state": "failed"},
            ),
        )


def test_active_mailbox_watcher_recovers_after_one_poll_error(tmp_path: Path):
    store_path = tmp_path / "events.sqlite3"
    mailbox = EventuallyReadyMailbox()

    with ActiveMailboxWatcher(
        mailbox,
        store_path,
        known_keys=frozenset(),
        poll_interval_seconds=0.01,
    ):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with AdvisorEventStore(store_path) as store:
                if store.pending_count() == 1:
                    break
            time.sleep(0.01)

    with AdvisorEventStore(store_path) as store:
        assert [event.dedupe_key for event in store.pending()] == [
            "job-1:status:failed"
        ]
    assert mailbox.polls >= 2


def test_advisor_event_enqueue_is_atomic_across_connections(tmp_path: Path):
    path = tmp_path / "events.sqlite3"
    event = AdvisorEvent(
        kind="job_monitor",
        dedupe_key="job-1:status:failed",
        payload={"job_id": "job-1", "state": "failed"},
    )
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []
    errors: list[BaseException] = []

    def enqueue() -> None:
        try:
            with AdvisorEventStore(path) as store:
                barrier.wait()
                outcomes.append(store.enqueue(event))
        except BaseException as error:  # assertion captures the worker failure
            errors.append(error)

    threads = [threading.Thread(target=enqueue) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(outcomes) == [False, True]


def test_acknowledgement_before_late_watcher_enqueue_stays_acknowledged(
    tmp_path: Path,
):
    path = tmp_path / "events.sqlite3"
    event = AdvisorEvent(
        kind="job_monitor",
        dedupe_key="job-1:status:failed",
        payload={"job_id": "job-1", "state": "failed"},
    )

    with AdvisorEventStore(path) as store:
        store.acknowledge(event.dedupe_key)
    with AdvisorEventStore(path) as store:
        assert not store.enqueue(event)
        assert store.pending() == []
        assert store.acknowledged((event.dedupe_key,)) == {event.dedupe_key}


def test_late_watcher_enqueue_cannot_shadow_a_synchronous_acknowledgement(
    tmp_path: Path,
):
    path = tmp_path / "events.sqlite3"
    mapping = threading.Event()
    release = threading.Event()
    event = ControllerEvent(
        kind="job_monitor",
        dedupe_key="job-1:status:failed",
        payload={"job_id": "job-1", "state": "failed"},
    )

    class Mailbox:
        def poll(self):
            return (event,)

    def map_event(controller_event: ControllerEvent) -> AdvisorEvent:
        mapping.set()
        release.wait(5)
        return AdvisorEvent(
            kind=controller_event.kind,
            dedupe_key=controller_event.dedupe_key,
            payload=controller_event.payload,
        )

    try:
        with ActiveMailboxWatcher(
            Mailbox(),
            path,
            poll_interval_seconds=0.01,
            shutdown_timeout_seconds=0.05,
            map_event=map_event,
        ):
            assert mapping.wait(1)
            with AdvisorEventStore(path) as store:
                store.acknowledge(event.dedupe_key)
    finally:
        release.set()

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with AdvisorEventStore(path) as store:
            if store.acknowledged((event.dedupe_key,)):
                assert store.pending() == []
                break
        time.sleep(0.01)
    else:
        raise AssertionError("late watcher event was not persisted as acknowledged")


def test_active_mailbox_watcher_shutdown_does_not_wait_for_blocked_poll(
    tmp_path: Path,
):
    polling = threading.Event()
    release = threading.Event()

    class BlockedMailbox:
        def poll(self):
            polling.set()
            release.wait(5)
            return ()

    started = time.monotonic()
    try:
        with ActiveMailboxWatcher(
            BlockedMailbox(),
            tmp_path / "events.sqlite3",
            known_keys=frozenset(),
            poll_interval_seconds=0.01,
            shutdown_timeout_seconds=0.1,
        ):
            assert polling.wait(1)
    finally:
        release.set()

    assert time.monotonic() - started < 1


def test_watcher_factory_owns_mailbox_until_a_late_poll_finishes(tmp_path: Path):
    polling = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class BlockedMailbox:
        def poll(self):
            polling.set()
            release.wait(5)
            return ()

    @contextmanager
    def mailbox_factory():
        try:
            yield BlockedMailbox()
        finally:
            closed.set()

    with ActiveMailboxWatcher(
        mailbox_factory,
        tmp_path / "events.sqlite3",
        poll_interval_seconds=0.01,
        shutdown_timeout_seconds=0.05,
    ):
        assert polling.wait(1)

    assert not closed.is_set()
    release.set()
    assert closed.wait(1)
