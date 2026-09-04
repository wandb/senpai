import threading
import time
from pathlib import Path
from uuid import UUID

from senpai_agent.controller import Controller, TurnResult
import pytest

from senpai_agent.github.http import GitHubRateLimitError, GitHubReadError
from senpai_agent.github.mailbox.watcher import GitHubMailboxWatcher
from senpai_agent.inbox import PersistentInbox
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import (
    CompositeMailbox,
    ControllerEvent,
    StudentAssignmentAvailabilityMailbox,
)
from senpai_agent.monitor import (
    MonitorMailbox,
    MonitorStore,
    TerminalPollResult,
    TrainingMonitorEngine,
    TrainingMonitorSpec,
)
from senpai_agent.training import TrainingResult, TrainingState
from senpai_agent.wake import WakeCoordinator


CONVERSATION_ONE = UUID("00000000-0000-0000-0000-000000000001")
CONVERSATION_TWO = UUID("00000000-0000-0000-0000-000000000002")


class SnapshotMailbox:
    def __init__(self, events=()):
        self._events = tuple(events)
        self._condition = threading.Condition()
        self.calls = 0
        self.acknowledged = []

    def poll(self):
        with self._condition:
            self.calls += 1
            self._condition.notify_all()
            return self._events

    def acknowledge(self, dedupe_keys):
        self.acknowledged.append(tuple(dedupe_keys))

    def replace(self, events):
        with self._condition:
            self._events = tuple(events)

    def replace_and_wait(self, events, timeout=1):
        deadline = time.monotonic() + timeout
        with self._condition:
            previous_calls = self.calls
            self._events = tuple(events)
            while self.calls == previous_calls:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            assert self.calls > previous_calls

    def wait_for_calls(self, minimum, timeout=1):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self.calls < minimum:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            assert self.calls >= minimum


class ObservedWake(WakeCoordinator):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, checkpoint, *, timeout_seconds):
        self.waiting.set()
        return super().wait(checkpoint, timeout_seconds=timeout_seconds)


class CompletingTurns:
    def __init__(self, before_run=None):
        self.calls = []
        self.before_run = before_run

    def run(
        self,
        _prompt,
        *,
        conversation_id,
        event_keys,
        visible_event_keys,
        inbox,
        inbox_turn_id,
    ):
        if self.before_run is not None:
            self.before_run()
        self.calls.append((conversation_id, event_keys, visible_event_keys))
        turn = inbox.turn(inbox_turn_id)
        for message in turn.messages:
            inbox.record_delivered(message.delivery_id, message.body)
        inbox.record_processed(inbox_turn_id)
        return TurnResult(exit_code=0)


class TerminalMailbox:
    def __init__(self):
        self.store = type("Store", (), {"active": lambda _self: []})()
        self.forced = []
        self.acknowledged = []

    def poll(self):
        return ()

    def poll_terminal(self, training_ids):
        self.forced.append(tuple(training_ids))
        return TerminalPollResult(
            items=tuple(
                ControllerEvent(
                    kind="training_monitor",
                    dedupe_key=f"{training_id}:status:finished",
                    payload={"training_id": training_id},
                )
                for training_id in training_ids
            ),
            resolved_training_ids=frozenset(training_ids),
        )

    def acknowledge(self, dedupe_keys):
        self.acknowledged.append(tuple(dedupe_keys))

    def seconds_until_next_poll(self):
        return None


def event(kind, key):
    return ControllerEvent(kind=kind, dedupe_key=key, payload={"key": key})


def pending_events(path: Path):
    with LocalEventStore(path) as store:
        return store.pending()


def wait_for_pending(path: Path, count: int, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = pending_events(path)
        if len(pending) == count:
            return pending
        time.sleep(0.005)
    assert len(pending_events(path)) == count


def test_wake_before_wait_is_not_lost():
    """
    Requirement: an event observed after a controller checkpoint wakes the next wait.
    Interface: WakeCoordinator checkpoint, wake, and bounded wait.
    """
    wakes = WakeCoordinator()
    checkpoint = wakes.checkpoint()

    wakes.wake()

    assert wakes.wait(checkpoint, timeout_seconds=0) is True


def test_wake_during_wait_returns_without_waiting_for_the_deadline():
    """
    Requirement: a sleeping controller resumes as soon as its watcher observes work.
    Interface: WakeCoordinator wait from a controller thread and wake from a watcher.
    """
    wakes = WakeCoordinator()
    checkpoint = wakes.checkpoint()
    started = threading.Event()
    result = []

    def wait_for_work():
        started.set()
        result.append(wakes.wait(checkpoint, timeout_seconds=10))

    waiter = threading.Thread(target=wait_for_work)
    waiter.start()
    assert started.wait(1)

    wakes.wake()
    waiter.join(1)

    assert not waiter.is_alive()
    assert result == [True]


def test_training_hint_acknowledgement_preserves_a_newer_same_id_wake():
    wakes = WakeCoordinator()
    wakes.training_finished("training-1")
    observed = wakes.training_hints()

    wakes.training_finished("training-1")
    wakes.acknowledge_training(observed)

    remaining = wakes.training_hints()
    assert [hint.training_id for hint in remaining] == ["training-1"]
    assert remaining[0].version > observed[0].version


def test_lifetime_watcher_polls_immediately_and_exposes_a_cached_mailbox(
    tmp_path: Path,
):
    """
    Requirement: the controller and active delivery path share one GitHub source.
    Interface: GitHubMailboxWatcher context, cached poll, and acknowledgement.
    """
    incoming = event("review_ready", "review_ready:17:abc")
    source = SnapshotMailbox((incoming,))
    wakes = WakeCoordinator()

    with GitHubMailboxWatcher(
        source,
        tmp_path / "advisor-events.sqlite3",
        coordinator=wakes,
        poll_interval_seconds=60,
    ) as watcher:
        assert watcher.poll() == (incoming,)
        assert source.calls == 1

        assert watcher.poll() == (incoming,)
        assert source.calls == 1

        watcher.acknowledge((incoming.dedupe_key,))

    assert source.acknowledged == [(incoming.dedupe_key,)]


def test_active_binding_stages_steer_and_queue_inputs_but_idle_events_only_wake(
    tmp_path: Path,
):
    """
    Requirement: active GitHub input reaches the existing AdvisorEventPump unchanged.
    Interface: watcher active binding, LocalEventStore, cache, and wake generation.

    AdvisorEventPump tests separately prove that human_issue is STEER while
    student_pr_feedback is QUEUE; this boundary must preserve both event kinds.
    """
    steer = event("human_issue", "human_issue:v2:23:702:abc")
    queue = event(
        "student_pr_feedback",
        "student_pr_feedback:issue_comment:17:101",
    )
    availability = event(
        "student_available_for_assignment",
        "student_available_for_assignment:Fern",
    )
    idle = event("review_ready", "review_ready:18:def")
    source = SnapshotMailbox()
    store_path = tmp_path / "advisor-events.sqlite3"
    wakes = WakeCoordinator()

    with GitHubMailboxWatcher(
        source,
        store_path,
        coordinator=wakes,
        poll_interval_seconds=0.005,
    ) as watcher:
        source.wait_for_calls(1)
        with watcher.bind_active(
            CONVERSATION_ONE,
            visible_event_keys=frozenset(),
        ):
            checkpoint = wakes.checkpoint()
            source.replace_and_wait((steer, queue, availability))
            assert wakes.wait(checkpoint, timeout_seconds=1)
            staged = wait_for_pending(store_path, 2)

        assert [(item.kind, item.dedupe_key) for item in staged] == [
            (steer.kind, steer.dedupe_key),
            (queue.kind, queue.dedupe_key),
        ]

        checkpoint = wakes.checkpoint()
        source.replace_and_wait((idle,))

        assert wakes.wait(checkpoint, timeout_seconds=1)
        assert watcher.poll() == (idle,)
        assert [item.dedupe_key for item in pending_events(store_path)] == [
            steer.dedupe_key,
            queue.dedupe_key,
        ]


def test_one_lifetime_watcher_routes_events_across_two_active_turns(tmp_path: Path):
    """
    Requirement: changing active conversations does not create a new source watcher.
    Interface: two watcher active bindings and one durable LocalEventStore.
    """
    first = event("human_issue", "human_issue:v2:23:701:first")
    second = event("human_issue", "human_issue:v2:23:702:second")
    source = SnapshotMailbox()
    store_path = tmp_path / "student-events.sqlite3"
    wakes = WakeCoordinator()

    def route(incoming, conversation_id):
        return LocalEvent(
            kind=incoming.kind,
            dedupe_key=incoming.dedupe_key,
            payload={
                **incoming.payload,
                "parent_conversation_id": str(conversation_id),
            },
        )

    with GitHubMailboxWatcher(
        source,
        store_path,
        coordinator=wakes,
        poll_interval_seconds=0.005,
        map_event=route,
    ) as watcher:
        source.wait_for_calls(1)
        with watcher.bind_active(
            CONVERSATION_ONE,
            visible_event_keys=frozenset(),
        ):
            source.replace_and_wait((first,))
            wait_for_pending(store_path, 1)

        with watcher.bind_active(
            CONVERSATION_TWO,
            visible_event_keys=frozenset({first.dedupe_key}),
        ):
            source.replace_and_wait((first, second))
            staged = wait_for_pending(store_path, 2)

    assert [item.payload["parent_conversation_id"] for item in staged] == [
        str(CONVERSATION_ONE),
        str(CONVERSATION_TWO),
    ]


def test_visible_key_is_suppressed_until_a_later_snapshot_reintroduces_it(
    tmp_path: Path,
):
    """
    Requirement: prompt-visible state is not injected twice, but reappearance wakes.
    Interface: active binding visible keys and successive complete GitHub snapshots.
    """
    incoming = event("review_ready", "review_ready:17:abc")
    source = SnapshotMailbox()
    store_path = tmp_path / "advisor-events.sqlite3"

    with GitHubMailboxWatcher(
        source,
        store_path,
        coordinator=WakeCoordinator(),
        poll_interval_seconds=0.005,
    ) as watcher:
        source.wait_for_calls(1)
        with watcher.bind_active(
            CONVERSATION_ONE,
            visible_event_keys=frozenset({incoming.dedupe_key}),
        ):
            source.replace_and_wait((incoming,))
            assert pending_events(store_path) == []

            source.replace_and_wait(())

            source.replace_and_wait((incoming,))
            staged = wait_for_pending(store_path, 1)

    assert [item.dedupe_key for item in staged] == [incoming.dedupe_key]


def test_github_change_wakes_an_idle_controller_from_the_cached_source(
    tmp_path: Path,
):
    incoming = event("review_ready", "review_ready:17:abc")
    source = SnapshotMailbox()
    wakes = ObservedWake()
    turns = CompletingTurns()
    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "advisor-events.sqlite3",
        coordinator=wakes,
        poll_interval_seconds=60,
    )
    controller = Controller(
        role="advisor",
        mailbox=watcher,
        turns=turns,
        conversation_id=CONVERSATION_ONE,
        full_prompt="programme",
        inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
        poll_interval_seconds=600,
        jitter_seconds=0,
        wake=wakes,
        request_mailbox_refresh=watcher.request_refresh,
    )

    with watcher:
        worker = threading.Thread(target=controller.run, kwargs={"max_cycles": 2})
        worker.start()
        assert wakes.waiting.wait(1)
        assert source.calls == 1

        source.replace((incoming,))
        watcher.request_refresh()
        worker.join(2)

    assert not worker.is_alive()
    assert [call[1] for call in turns.calls] == [frozenset({incoming.dedupe_key})]
    assert source.acknowledged == [(incoming.dedupe_key,)]


def test_training_completion_waits_for_the_active_turn_and_skips_github_poll(
    tmp_path: Path,
):
    initial = event("review_ready", "review_ready:17:abc")
    source = SnapshotMailbox((initial,))
    wakes = WakeCoordinator()
    terminal = TerminalMailbox()
    first_turn_started = threading.Event()
    release_first_turn = threading.Event()

    def block_first_turn():
        if not first_turn_started.is_set():
            first_turn_started.set()
            assert release_first_turn.wait(2)

    turns = CompletingTurns(before_run=block_first_turn)
    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "advisor-events.sqlite3",
        coordinator=wakes,
        poll_interval_seconds=60,
    )
    controller = Controller(
        role="advisor",
        mailbox=CompositeMailbox(watcher, terminal),
        turns=turns,
        conversation_id=CONVERSATION_ONE,
        full_prompt="programme",
        inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
        poll_interval_seconds=600,
        jitter_seconds=0,
        wake=wakes,
        monitor_mailbox=terminal,
        request_mailbox_refresh=watcher.request_refresh,
    )

    with watcher:
        worker = threading.Thread(target=controller.run, kwargs={"max_cycles": 1})
        worker.start()
        assert first_turn_started.wait(1)

        wakes.training_finished("training-1")
        time.sleep(0.02)
        assert terminal.forced == []
        assert source.calls == 1
        assert len(turns.calls) == 0

        release_first_turn.set()
        worker.join(2)

    assert not worker.is_alive()
    assert terminal.forced == [("training-1",)]
    assert [call[1] for call in turns.calls] == [
        frozenset({initial.dedupe_key}),
        frozenset({"training-1:status:finished"}),
    ]


def test_rate_limited_watcher_keeps_its_snapshot_without_spinning(tmp_path: Path):
    class RateLimitedMailbox(SnapshotMailbox):
        def poll(self):
            with self._condition:
                self.calls += 1
                self._condition.notify_all()
            raise GitHubRateLimitError("GitHub throttled", retry_after_seconds=0.2)

    source = RateLimitedMailbox()
    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "advisor-events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=0.001,
    )

    with watcher:
        started = time.monotonic()
        watcher.request_refresh()
        elapsed = time.monotonic() - started
        time.sleep(0.03)

    assert elapsed < 0.05
    assert source.calls == 1
    with pytest.raises(GitHubRateLimitError):
        watcher.poll()


def test_failed_initial_snapshot_cannot_retract_durable_availability(tmp_path: Path):
    availability = ControllerEvent(
        kind="student_available_for_assignment",
        dedupe_key="student_available_for_assignment:fern",
        payload={"student": "fern"},
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(
        CONVERSATION_ONE,
        availability.dedupe_key,
        availability.to_prompt(),
    )
    store_path = tmp_path / "advisor-events.sqlite3"
    with LocalEventStore(store_path) as store:
        store.enqueue(
            LocalEvent(
                kind=availability.kind,
                dedupe_key=availability.dedupe_key,
                payload=availability.payload,
            )
        )

    class FailingMailbox(SnapshotMailbox):
        def poll(self):
            raise GitHubReadError("GitHub is temporarily unavailable")

    watcher = GitHubMailboxWatcher(
        FailingMailbox(),
        store_path,
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
    )
    mailbox = StudentAssignmentAvailabilityMailbox(
        watcher,
        inbox=inbox,
        conversation_id=CONVERSATION_ONE,
        event_store_path=store_path,
    )

    with watcher, pytest.raises(GitHubReadError):
        mailbox.poll()

    assert inbox.pending_count(CONVERSATION_ONE) == 1
    assert [item.dedupe_key for item in pending_events(store_path)] == [
        availability.dedupe_key
    ]


def test_read_failure_retains_the_last_complete_snapshot(tmp_path: Path):
    incoming = event("review_ready", "review_ready:17:abc")

    class FlakyMailbox(SnapshotMailbox):
        def poll(self):
            if self.calls:
                with self._condition:
                    self.calls += 1
                    self._condition.notify_all()
                raise GitHubReadError("GitHub comments endpoint unavailable")
            return super().poll()

    source = FlakyMailbox((incoming,))
    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
    )

    with watcher:
        watcher.request_refresh()
        source.wait_for_calls(2)
        assert watcher.poll() == (incoming,)


def test_watcher_shutdown_does_not_make_a_final_github_request(tmp_path: Path):
    source = SnapshotMailbox()

    with GitHubMailboxWatcher(
        source,
        tmp_path / "events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
    ):
        assert source.calls == 1

    assert source.calls == 1


def test_failed_initial_active_staging_does_not_poison_the_next_binding(
    tmp_path: Path,
):
    incoming = event("human_issue", "human_issue:v2:23:702:abc")
    source = SnapshotMailbox((incoming,))
    fail = True

    def route(value, conversation_id):
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("cannot persist active input")
        return LocalEvent(
            kind=value.kind,
            dedupe_key=value.dedupe_key,
            payload={"parent_conversation_id": str(conversation_id)},
        )

    with GitHubMailboxWatcher(
        source,
        tmp_path / "events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
        map_event=route,
    ) as watcher:
        with pytest.raises(RuntimeError, match="cannot persist active input"):
            with watcher.bind_active(
                CONVERSATION_ONE,
                visible_event_keys=frozenset(),
            ):
                pass

        with watcher.bind_active(
            CONVERSATION_TWO,
            visible_event_keys=frozenset(),
        ):
            pass


def test_fatal_watcher_error_wakes_and_fails_the_controller_wait(tmp_path: Path):
    class FatalMailbox(SnapshotMailbox):
        def poll(self):
            if self.calls:
                raise RuntimeError("watcher invariant failed")
            return super().poll()

    source = FatalMailbox()
    wakes = WakeCoordinator()

    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "events.sqlite3",
        coordinator=wakes,
        poll_interval_seconds=60,
    )
    with pytest.raises(RuntimeError, match="GitHub watcher failed"):
        with watcher:
            checkpoint = wakes.checkpoint()
            watcher.request_refresh()
            with pytest.raises(RuntimeError, match="event source failed"):
                wakes.wait(checkpoint, timeout_seconds=1)


def test_terminal_hint_is_retried_after_a_transient_monitor_error(tmp_path: Path):
    initial = event("review_ready", "review_ready:17:abc")
    source = SnapshotMailbox((initial,))
    wakes = WakeCoordinator()

    class FlakyTerminalMailbox(TerminalMailbox):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def poll_terminal(self, training_ids):
            self.attempts += 1
            if self.attempts == 1:
                self.forced.append(tuple(training_ids))
                raise RuntimeError("monitor store busy")
            return super().poll_terminal(training_ids)

    terminal = FlakyTerminalMailbox()

    def finish_during_turn():
        if not terminal.forced:
            wakes.training_finished("training-1")

    turns = CompletingTurns(before_run=finish_during_turn)
    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "events.sqlite3",
        coordinator=wakes,
        poll_interval_seconds=60,
    )
    controller = Controller(
        role="advisor",
        mailbox=CompositeMailbox(watcher, terminal),
        turns=turns,
        conversation_id=CONVERSATION_ONE,
        full_prompt="programme",
        inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
        poll_interval_seconds=600,
        jitter_seconds=0,
        wake=wakes,
        monitor_mailbox=terminal,
        request_mailbox_refresh=watcher.request_refresh,
    )

    with watcher:
        controller.run(max_cycles=2)

    assert terminal.forced == [
        ("training-1",),
        ("training-1",),
    ]
    assert [call[1] for call in turns.calls] == [
        frozenset({initial.dedupe_key}),
        frozenset({"training-1:status:finished"}),
    ]


def test_monitor_deadline_clamps_the_controller_reconciliation_sleep(tmp_path: Path):
    terminal = TerminalMailbox()
    terminal.seconds_until_next_poll = lambda: 60
    sleeps = []

    Controller(
        role="student",
        mailbox=terminal,
        turns=CompletingTurns(),
        conversation_id=CONVERSATION_ONE,
        full_prompt="programme",
        inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
        poll_interval_seconds=300,
        jitter_seconds=120,
        sleep=sleeps.append,
        monitor_mailbox=terminal,
    ).run(max_cycles=2)

    assert sleeps == [60]


def test_due_monitor_deadline_runs_the_engine_before_github_reconciliation(
    tmp_path: Path,
):
    conversation_id = CONVERSATION_ONE
    spec = TrainingMonitorSpec(
        training_id="training-deadline",
        conversation_id=conversation_id,
        poll_interval_seconds=0.02,
    )

    class Training:
        def __init__(self):
            self.calls = 0

        def get_training_status(self, training_id):
            self.calls += 1
            return TrainingResult(
                training_id=training_id,
                state=TrainingState.RUNNING,
                elapsed_seconds=1,
                log_path=str(tmp_path / "training.log"),
            )

    training = Training()
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        monitor_mailbox = MonitorMailbox(
            TrainingMonitorEngine(
                training=training,
                store=store,
                metrics=object(),
            ),
            store,
        )
        started = time.monotonic()
        Controller(
            role="student",
            mailbox=monitor_mailbox,
            turns=CompletingTurns(),
            conversation_id=conversation_id,
            full_prompt="programme",
            inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
            poll_interval_seconds=300,
            jitter_seconds=0,
            monitor_mailbox=monitor_mailbox,
        ).run(max_cycles=3)

    assert training.calls >= 3
    assert time.monotonic() - started < 1


def test_github_wake_does_not_bypass_turn_failure_backoff(tmp_path: Path):
    wakes = WakeCoordinator()
    sleeps = []

    class FailingTurns(CompletingTurns):
        def run(self, *args, **kwargs):
            wakes.wake()
            return TurnResult(exit_code=1)

    with pytest.raises(RuntimeError, match="consecutive turn-failure limit"):
        Controller(
            role="advisor",
            mailbox=SnapshotMailbox((event("review_ready", "review:1"),)),
            turns=FailingTurns(),
            conversation_id=CONVERSATION_ONE,
            full_prompt="programme",
            inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
            poll_interval_seconds=300,
            jitter_seconds=0,
            sleep=sleeps.append,
            wake=wakes,
        ).run(max_cycles=2)

    assert sleeps == [2]


def test_watcher_entry_interruption_stops_its_worker(tmp_path: Path, monkeypatch):
    watcher = GitHubMailboxWatcher(
        SnapshotMailbox(),
        tmp_path / "events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
    )
    monkeypatch.setattr(
        watcher,
        "_raise_if_failed",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    try:
        with pytest.raises(KeyboardInterrupt):
            watcher.__enter__()
        assert not watcher._thread.is_alive()
    finally:
        watcher._stop.set()
        watcher._refresh.set()
        watcher._thread.join(1)


def test_watcher_close_during_entry_unblocks_the_startup_wait(tmp_path: Path):
    release_worker = threading.Event()
    worker_started = threading.Event()

    class CancellableMailbox(SnapshotMailbox):
        def cancel_poll(self):
            release_worker.set()

    watcher = GitHubMailboxWatcher(
        CancellableMailbox(),
        tmp_path / "events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
    )
    original_run = watcher._run

    def delayed_run():
        worker_started.set()
        release_worker.wait()
        original_run()

    watcher._thread = threading.Thread(target=delayed_run, daemon=True)
    errors = []

    def enter():
        try:
            watcher.__enter__()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    entry = threading.Thread(target=enter, daemon=True)
    entry.start()
    assert worker_started.wait(1)

    try:
        watcher.close()
        entry.join(1)
        assert not entry.is_alive()
        assert len(errors) == 1
        assert "closed during startup" in str(errors[0])
    finally:
        if entry.is_alive():
            with watcher._attempted:
                watcher._error = RuntimeError("test cleanup")
                watcher._attempted.notify_all()
            entry.join(1)


def test_watcher_close_cancels_an_inflight_snapshot(tmp_path: Path):
    class BlockingMailbox(SnapshotMailbox):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.cancelled = threading.Event()

        def poll(self):
            self.started.set()
            assert self.cancelled.wait(1)
            return ()

        def cancel_poll(self):
            self.cancelled.set()

    source = BlockingMailbox()
    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
    )
    watcher._thread.start()
    assert source.started.wait(1)

    watcher.close()

    assert source.cancelled.is_set()
    assert not watcher._thread.is_alive()


def test_watcher_close_is_bounded_when_a_source_cannot_cancel(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    class BlockingMailbox(SnapshotMailbox):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def poll(self):
            self.started.set()
            self.release.wait()
            return ()

    source = BlockingMailbox()
    watcher = GitHubMailboxWatcher(
        source,
        tmp_path / "events.sqlite3",
        coordinator=WakeCoordinator(),
        poll_interval_seconds=60,
    )
    monkeypatch.setattr(watcher, "_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    watcher._thread.start()
    assert source.started.wait(1)

    try:
        started = time.monotonic()
        watcher.close()
        assert time.monotonic() - started < 0.2
        assert watcher._thread.is_alive()
        assert "SENPAI_GITHUB_WATCHER_SHUTDOWN_TIMEOUT" in capsys.readouterr().err
    finally:
        source.release.set()
        watcher._thread.join(1)
