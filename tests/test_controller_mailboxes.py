from pathlib import Path
import time
import threading
from types import SimpleNamespace
from uuid import UUID

import pytest

from senpai_agent.inbox import PersistentInbox
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import (
    CompositeMailbox,
    ControllerEvent,
    StudentAssignmentAvailabilityMailbox,
)
from senpai_agent.monitor import (
    MonitorEvaluation,
    MetricGate,
    MetricSample,
    MonitorMailbox,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorSpec,
    TrainingMonitorEngine,
    TrainingMonitorWatcher,
)
from senpai_agent.training import TrainingResult, TrainingState


class StaticMailbox:
    def __init__(self, events):
        self.events = tuple(events)

    def poll(self):
        return self.events

    def acknowledge(self, _dedupe_keys):
        return


def availability_event(student: str = "Fern") -> ControllerEvent:
    return ControllerEvent(
        kind="student_available_for_assignment",
        dedupe_key=f"student_available_for_assignment:{student}",
        payload={"student": student},
    )


def test_composite_mailbox_preserves_healthy_events_when_a_peer_fails(capsys):
    event = ControllerEvent(
        kind="student_assignment",
        dedupe_key="assignment:healthy",
        payload={"assignment_id": "healthy"},
    )

    class BrokenMailbox:
        def poll(self):
            raise RuntimeError("monitor backend unavailable")

        def acknowledge(self, _dedupe_keys):
            return

    mailbox = CompositeMailbox(BrokenMailbox(), StaticMailbox((event,)))

    assert mailbox.poll() == (event,)
    assert "SENPAI_MAILBOX_ERROR RuntimeError" in capsys.readouterr().err


def test_reserved_assignment_retracts_unseen_availability_events(tmp_path: Path):
    conversation_id = UUID(int=123)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    store_path = tmp_path / "advisor-events.sqlite3"
    event = availability_event()
    inbox.enqueue(conversation_id, event.dedupe_key, event.to_prompt())
    with LocalEventStore(store_path) as store:
        store.enqueue(
            LocalEvent(
                kind=event.kind,
                dedupe_key=event.dedupe_key,
                payload=event.payload,
            )
        )
        store.acknowledge(event.dedupe_key)

    mailbox = StudentAssignmentAvailabilityMailbox(
        StaticMailbox(()),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=store_path,
    )

    assert mailbox.poll() == ()
    assert inbox.pending_count(conversation_id) == 0
    with LocalEventStore(store_path) as store:
        assert (
            store.enqueue(
                LocalEvent(
                    kind=event.kind,
                    dedupe_key=event.dedupe_key,
                    payload=event.payload,
                )
            )
            is True
        )


def test_available_student_preserves_its_queued_event(tmp_path: Path):
    conversation_id = UUID(int=124)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    event = availability_event()
    inbox.enqueue(conversation_id, event.dedupe_key, event.to_prompt())
    mailbox = StudentAssignmentAvailabilityMailbox(
        StaticMailbox((event,)),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=tmp_path / "advisor-events.sqlite3",
    )

    assert mailbox.poll() == (event,)
    assert inbox.pending_count(conversation_id) == 1


def test_snapshot_retracts_removed_student_availability(tmp_path: Path):
    conversation_id = UUID(int=125)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    store_path = tmp_path / "advisor-events.sqlite3"
    stale = availability_event("Old-name")
    current = availability_event("New-name")
    inbox.enqueue(conversation_id, stale.dedupe_key, stale.to_prompt())
    with LocalEventStore(store_path) as store:
        store.enqueue(
            LocalEvent(
                kind=stale.kind,
                dedupe_key=stale.dedupe_key,
                payload=stale.payload,
            )
        )
        store.acknowledge(stale.dedupe_key)

    mailbox = StudentAssignmentAvailabilityMailbox(
        StaticMailbox((current,)),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=store_path,
    )

    assert mailbox.poll() == (current,)
    assert inbox.pending_count(conversation_id) == 0
    with LocalEventStore(store_path) as store:
        assert (
            store.enqueue(
                LocalEvent(
                    kind=stale.kind,
                    dedupe_key=stale.dedupe_key,
                    payload=stale.payload,
                )
            )
            is True
        )


def test_failed_github_poll_does_not_retract_queued_availability(tmp_path: Path):
    class BrokenMailbox:
        def poll(self):
            raise RuntimeError("GitHub unavailable")

        def acknowledge(self, _dedupe_keys):
            return

    conversation_id = UUID(int=125)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    event = availability_event()
    inbox.enqueue(conversation_id, event.dedupe_key, event.to_prompt())
    mailbox = StudentAssignmentAvailabilityMailbox(
        BrokenMailbox(),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=tmp_path / "advisor-events.sqlite3",
    )

    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        mailbox.poll()
    assert inbox.pending_count(conversation_id) == 1


def test_monitor_mailbox_routes_and_acknowledges_each_signal_independently(
    tmp_path: Path,
):
    first_id = UUID("00000000-0000-0000-0000-000000000086")
    second_id = UUID("00000000-0000-0000-0000-000000000087")
    first = MonitorSignal(
        kind="training_status",
        dedupe_key="training:first:failed",
        training_id="training-first",
        state=TrainingState.FAILED,
        detail="first training failed",
    )
    second = MonitorSignal(
        kind="training_status",
        dedupe_key="training:second:finished",
        training_id="training-second",
        state=TrainingState.FINISHED,
        detail="second training finished",
    )

    class Engine:
        def poll(self):
            return ()

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        for signal, conversation_id in ((first, first_id), (second, second_id)):
            spec = TrainingMonitorSpec(
                training_id=signal.training_id,
                conversation_id=conversation_id,
            )
            store.register(spec)
            store.record_poll(spec, MonitorEvaluation(signals=(signal,)), None)
        mailbox = MonitorMailbox(Engine(), store)

        events = mailbox.poll()

        assert {
            event.dedupe_key: event.payload["conversation_id"] for event in events
        } == {
            first.dedupe_key: str(first_id),
            second.dedupe_key: str(second_id),
        }
        mailbox.acknowledge((first.dedupe_key,))
        assert [event.dedupe_key for event in mailbox.poll()] == [second.dedupe_key]


def test_background_monitor_delivers_durable_policy_events_from_another_connection(
    tmp_path: Path,
):
    monitor_path = tmp_path / "monitors.sqlite3"
    event_path = tmp_path / "events.sqlite3"
    spec = TrainingMonitorSpec(training_id="train-1", conversation_id=UUID(int=87))
    training = SimpleNamespace(
        get_training_status=lambda _id: TrainingResult(
            training_id="train-1",
            state=TrainingState.FINISHED,
            exit_code=0,
            elapsed_seconds=1,
            log_path=str(tmp_path / "run.log"),
            kubernetes_released=True,
        )
    )
    with MonitorStore(monitor_path) as store:
        with TrainingMonitorWatcher(
            monitor_path, event_path, training, SimpleNamespace()
        ):
            store.register(spec)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with LocalEventStore(event_path) as events:
                    pending = events.pending()
                if pending and not store.active():
                    break
                time.sleep(0.01)
        assert len(pending) == 1
        event = pending[0]
        assert event.kind == "training_monitor"
        assert event.payload["parent_conversation_id"] == str(spec.conversation_id)
        assert "released" in event.payload["summary"]
        assert store.pending_signals() == []
        assert not store.register(spec)
    with LocalEventStore(event_path) as events:
        events.acknowledge(event.dedupe_key)
    with TrainingMonitorWatcher(monitor_path, event_path, training, SimpleNamespace()):
        with LocalEventStore(event_path) as events:
            assert events.pending() == []


@pytest.mark.parametrize("metric_failure", [False, True])
def test_slow_wandb_does_not_delay_terminal_delivery_or_publish_a_late_metric(
    tmp_path: Path,
    metric_failure: bool,
):
    monitor_path = tmp_path / "monitors.sqlite3"
    event_path = tmp_path / "events.sqlite3"
    reading = threading.Event()
    release_metric = threading.Event()
    current = {
        name: TrainingResult(
            training_id=name,
            state=TrainingState.RUNNING,
            elapsed_seconds=1,
            exit_code=None,
            log_path=str(tmp_path / f"{name}.log"),
            wandb_run_ids=(name,),
        )
        for name in ("slow-metric", "other-run")
    }

    class Metrics:
        def latest(self, *_args):
            reading.set()
            assert release_metric.wait(10)
            if metric_failure:
                raise RuntimeError("late W&B failure")
            return MetricSample(value=0.1, observed_at=spec.registered_at)

    with MonitorStore(monitor_path) as monitors:
        spec = TrainingMonitorSpec(
            training_id="slow-metric",
            conversation_id=UUID(int=87),
            metric="loss",
            gates=(MetricGate(operator="lte", threshold=0.5),),
        )
        monitors.register(spec)
        monitors.register(
            TrainingMonitorSpec(
                training_id="other-run",
                conversation_id=UUID(int=88),
            )
        )
    with TrainingMonitorWatcher(
        monitor_path,
        event_path,
        SimpleNamespace(get_training_status=lambda name: current[name]),
        Metrics(),
    ):
        try:
            assert reading.wait(3)
            current = {
                name: result.model_copy(
                    update={
                        "state": TrainingState.FINISHED,
                        "kubernetes_released": True,
                        "exit_code": 0,
                    }
                )
                for name, result in current.items()
            }
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with LocalEventStore(event_path) as events:
                    pending = events.pending()
                if len(pending) == 2:
                    break
                time.sleep(0.01)
            assert {event.payload["training_id"] for event in pending} == set(current)
        finally:
            release_metric.set()
    with LocalEventStore(event_path) as events:
        assert len(events.pending()) == 2
        assert all(
            event.payload["signal"]["kind"] == "training_status"
            for event in events.pending()
        )


def test_conflicting_local_receipt_does_not_block_other_monitor_signals(tmp_path, capsys):
    database, event_path = tmp_path / "monitors.sqlite3", tmp_path / "events.sqlite3"
    training = SimpleNamespace(get_training_status=lambda name: TrainingResult(
        training_id=name, state=TrainingState.FINISHED, exit_code=0, elapsed_seconds=1,
        log_path=str(tmp_path / "run.log"), kubernetes_released=True,
    ))
    with MonitorStore(database) as monitors:
        for index, name in enumerate(("conflict", "healthy")):
            monitors.register(TrainingMonitorSpec(training_id=name, conversation_id=UUID(int=index+1)))
        first, second = TrainingMonitorEngine(monitors, training, SimpleNamespace()).poll()
    with LocalEventStore(event_path) as events:
        events.enqueue(LocalEvent(kind="training_monitor", dedupe_key=first.dedupe_key, payload={"summary": "conflicting persisted content"}))
    with TrainingMonitorWatcher(database, event_path, training, SimpleNamespace()):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with LocalEventStore(event_path) as events:
                keys = {event.dedupe_key for event in events.pending()}
            if second.dedupe_key in keys:
                break
            time.sleep(0.01)
        assert second.dedupe_key in keys
    with MonitorStore(database) as monitors:
        assert [signal.dedupe_key for signal in monitors.pending_signals()] == [first.dedupe_key]
    assert "reused with a different payload" in capsys.readouterr().err
