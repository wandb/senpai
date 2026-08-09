from pathlib import Path
from uuid import UUID

from senpai_agent.mailbox import CompositeMailbox, ControllerEvent
from senpai_agent.monitor import (
    MonitorEvaluation,
    MonitorMailbox,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorSpec,
)
from senpai_agent.training import TrainingState


class StaticMailbox:
    def __init__(self, events):
        self.events = tuple(events)

    def poll(self):
        return self.events

    def acknowledge(self, _dedupe_keys):
        return


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
        assert {event.kind for event in events} == {"job_monitor"}
        assert {event.payload["job_id"] for event in events} == {
            "training-first",
            "training-second",
        }
        assert {event.payload["signal"]["kind"] for event in events} == {"job_status"}
        assert all("training_id" not in event.payload["signal"] for event in events)
        assert all("metric" not in event.payload["signal"] for event in events)
        mailbox.acknowledge((first.dedupe_key,))
        assert [event.dedupe_key for event in mailbox.poll()] == [second.dedupe_key]
