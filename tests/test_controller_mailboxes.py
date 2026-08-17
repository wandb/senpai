from pathlib import Path
from uuid import UUID

from senpai_agent.mailbox import CompositeMailbox, ControllerEvent
from senpai_agent.monitor import (
    JobMonitorMailbox,
    JobMonitorSpec,
    JobMonitorStore,
    MonitorEvaluation,
    MonitorSignal,
)
from senpai_agent.jobs import JobState


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


def test_job_monitor_mailbox_routes_and_acknowledges_each_signal_independently(
    tmp_path: Path,
):
    first_id = UUID("00000000-0000-0000-0000-000000000086")
    second_id = UUID("00000000-0000-0000-0000-000000000087")
    first = MonitorSignal(
        kind="job_status",
        dedupe_key="job-first:status:failed",
        job_id="job-first",
        state=JobState.FAILED,
        detail="first job failed",
    )
    second = MonitorSignal(
        kind="job_status",
        dedupe_key="job-second:status:finished",
        job_id="job-second",
        state=JobState.FINISHED,
        detail="second job finished",
    )

    class Engine:
        def poll(self):
            return ()

    with JobMonitorStore(tmp_path / "monitors.sqlite3") as store:
        for signal, conversation_id in ((first, first_id), (second, second_id)):
            spec = JobMonitorSpec(
                job_id=signal.job_id,
                conversation_id=conversation_id,
            )
            store.register(spec)
            store.record_poll(spec, MonitorEvaluation(signals=(signal,)), {})
        mailbox = JobMonitorMailbox(Engine(), store)

        events = mailbox.poll()

        assert {
            event.dedupe_key: event.payload["conversation_id"] for event in events
        } == {
            first.dedupe_key: str(first_id),
            second.dedupe_key: str(second_id),
        }
        assert {event.kind for event in events} == {"job_monitor"}
        assert {event.payload["job_id"] for event in events} == {
            "job-first",
            "job-second",
        }
        assert {event.payload["signal"]["kind"] for event in events} == {"job_status"}
        assert all("metric" not in event.payload["signal"] for event in events)
        mailbox.acknowledge((first.dedupe_key,))
        assert [event.dedupe_key for event in mailbox.poll()] == [second.dedupe_key]
