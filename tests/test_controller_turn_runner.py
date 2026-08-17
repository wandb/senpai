import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.llm.exceptions import (
    LLMContextWindowExceedError,
    LLMMalformedConversationHistoryError,
)

from senpai_agent.advisor import AdvisorEvent, AdvisorEventPump, AdvisorEventStore
from senpai_agent.controller import (
    ConversationRecoveryExhausted,
    OpenHandsTurnRunner,
    _context_recovery_prompt,
    _open_job_monitor_mailbox,
)
from senpai_agent.github.http import GitHubReadError
from senpai_agent.mailbox_watcher import ActiveMailboxWatcher
from senpai_agent.inbox import DeliveryState, PersistentInbox, deliver_turn_messages
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.monitor import JobMonitorSpec, JobMonitorStore
from senpai_agent.jobs import JobState
from senpai_agent.state import AssignmentConversationRegistry


@dataclass(frozen=True)
class Config:
    role: str
    state_dir: Path
    conversation_id: UUID
    timeout_seconds: float = 3600
    wandb_entity: str | None = None
    wandb_project: str | None = None


class Mailbox:
    def __init__(self, events):
        self.events = tuple(events)

    def poll(self):
        return self.events


def feedback_event(revision_id="revision-2"):
    return ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key=f"student_pr_feedback:issue_comment:17:{revision_id}",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": revision_id,
            "message": f"Feedback for {revision_id}.",
        },
    )


def human_issue_event():
    return ControllerEvent(
        kind="human_issue",
        dedupe_key="human_issue:v2:23:702:abc",
        payload={
            "number": 23,
            "human_message_id": 702,
            "message": "Stop and inspect the active experiment.",
        },
    )


def advisor_event(number=17):
    return ControllerEvent(
        kind="review_ready",
        dedupe_key=f"review_ready:{number}:abc",
        payload={"number": number},
    )


def test_advisor_job_monitor_fires_in_background_without_preempting_active_turn(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000081")
    job_id = "wandb.run.81"

    class StatusSource:
        def get_job_status(self, _job_id):
            return SimpleNamespace(
                state=JobState.FINISHED,
                exit_code=None,
                wandb_run_ids=(job_id,),
            )

    def monitor_factory():
        return _open_job_monitor_mailbox(
            state_dir / "advisor-job-monitors" / "monitors.sqlite3",
            StatusSource(),
            SimpleNamespace(latest=lambda *_values: None),
        )

    inbox = PersistentInbox(state_dir / "delivery-inbox.sqlite3")
    inbox.enqueue(conversation_id, "active-event", "active event")
    active = inbox.next_turn(conversation_id, "active prompt", max_events=1)
    assert active is not None
    inbox.enqueue(conversation_id, "ordinary-backlog", "ordinary backlog")

    def run_openhands(_prompt, config, **options):
        with JobMonitorStore(
            state_dir / "advisor-job-monitors" / "monitors.sqlite3"
        ) as tool_store:
            tool_store.register(
                JobMonitorSpec(
                        job_id=job_id,
                        conversation_id=conversation_id,
                        poll_interval_seconds=5,
                )
            )

        class Conversation:
            def send_message(self, _message):
                raise AssertionError("job event entered the active turn")

        with AdvisorEventStore(
            state_dir / "advisor-events.sqlite3"
        ) as event_store, AdvisorEventPump(
            event_store,
            Conversation(),
            poll_interval=0.001,
            inbox=options["inbox"],
            conversation_id=config.conversation_id,
        ):
            deadline = time.monotonic() + 1
            while inbox.pending_count(conversation_id) < 2:
                assert time.monotonic() < deadline
                time.sleep(0.001)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config(
            "advisor",
            state_dir,
            conversation_id,
            wandb_entity="milieu",
            wandb_project="nn_cfd",
        ),
        full_prompt="advisor initial controller context",
        active_mailbox_factories=(monitor_factory,),
        active_monitor_poll_interval_seconds=0.001,
    ).run(
        "active prompt",
        conversation_id=conversation_id,
        event_keys=frozenset({"active-event"}),
        inbox=inbox,
        inbox_turn_id=active.turn_id,
    )

    assert result.exit_code == 0
    for message in active.messages:
        inbox.record_delivered(message.delivery_id, message.body)
    inbox.record_processed(active.turn_id)
    inbox.acknowledge(active.turn_id)
    next_turn = inbox.next_turn(conversation_id, "next prompt", max_events=1)
    assert next_turn is not None
    assert next_turn.event_keys == (f"{job_id}:status:finished",)


def test_advisor_job_monitor_quiet_background_polls_create_no_events(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000082")
    with JobMonitorStore(
        state_dir / "advisor-job-monitors" / "monitors.sqlite3"
    ) as store:
        store.register(
            JobMonitorSpec(
                    job_id="wandb-run-82",
                    conversation_id=conversation_id,
                    poll_interval_seconds=5,
            )
        )

    polls = []

    class StatusSource:
        def get_job_status(self, job_id):
            polls.append(job_id)
            return SimpleNamespace(
                state=JobState.RUNNING,
                exit_code=None,
                wandb_run_ids=(job_id,),
            )

    def monitor_factory():
        return _open_job_monitor_mailbox(
            state_dir / "advisor-job-monitors" / "monitors.sqlite3",
            StatusSource(),
            SimpleNamespace(latest=lambda *_values: None),
        )

    def run_openhands(_prompt, _config):
        deadline = time.monotonic() + 1
        while not polls:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    OpenHandsTurnRunner(
        Config(
            "advisor",
            state_dir,
            conversation_id,
            wandb_entity="milieu",
            wandb_project="nn_cfd",
        ),
        full_prompt="advisor initial controller context",
        active_mailbox_factories=(monitor_factory,),
        active_monitor_poll_interval_seconds=0.001,
    ).run(
        "continue unrelated work",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert polls == ["wandb-run-82"]
    with AdvisorEventStore(state_dir / "advisor-events.sqlite3") as store:
        assert store.pending() == []


def test_context_recovery_prompt_does_not_repeat_an_embedded_research_brief():
    prompt = _context_recovery_prompt(
        "complete initial controller context",
        "updated operating context\n\ncomplete initial controller context\n\nactionable event",
    )

    assert prompt.count("complete initial controller context") == 1


def test_running_student_receives_only_feedback_bound_to_its_conversation(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    current = feedback_event("revision-2")
    other_revision = feedback_event("revision-3")
    messages = []

    def run_openhands(_prompt, config):
        class Conversation:
            def send_message(self, message, sender=None):
                messages.append(message)

        with AdvisorEventStore(
            state_dir / "student-events.sqlite3"
        ) as store, AdvisorEventPump(
            store,
            Conversation(),
            poll_interval=0.001,
            parent_conversation_id=str(config.conversation_id),
        ):
            deadline = time.monotonic() + 1
            while not messages and time.monotonic() < deadline:
                time.sleep(0.001)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("student", state_dir, conversation_id),
        full_prompt="student initial controller context",
        github_mailbox=Mailbox((current, other_revision)),
        active_poll_interval_seconds=0.001,
    ).run(
        "current student turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert len(messages) == 1
    assert "Feedback for revision-2." in messages[0]
    assert "Feedback for revision-3." not in messages[0]
    assert str(conversation_id) in messages[0]
    with AdvisorEventStore(state_dir / "student-events.sqlite3") as store:
        assert store.pending() == []


@pytest.mark.parametrize("incoming", [feedback_event(), human_issue_event()])
def test_observed_student_input_routes_to_the_active_pump_until_it_is_delivered(
    tmp_path: Path,
    monkeypatch,
    incoming: ControllerEvent,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    store_path = state_dir / "student-events.sqlite3"

    def run_openhands(_prompt, _config):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with AdvisorEventStore(store_path) as store:
                if store.pending_count():
                    break
            time.sleep(0.001)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("student", state_dir, conversation_id),
        full_prompt="student initial controller context",
        github_mailbox=Mailbox((incoming,)),
        active_poll_interval_seconds=0.001,
    ).run(
        "student turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    with AdvisorEventStore(store_path) as store:
        pending = store.pending()
        assert [event.dedupe_key for event in pending] == [incoming.dedupe_key]
        assert pending[0].kind == incoming.kind
        assert pending[0].payload["parent_conversation_id"] == str(
            conversation_id
        )


def test_prompt_delivery_suppresses_a_late_duplicate_watcher_event(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    feedback = feedback_event()
    store_path = state_dir / "student-events.sqlite3"
    with AdvisorEventStore(store_path) as store:
        store.enqueue(
            AdvisorEvent(
                kind=feedback.kind,
                dedupe_key=feedback.dedupe_key,
                payload={
                    **feedback.payload,
                    "parent_conversation_id": str(conversation_id),
                },
            )
        )
    messages = []

    def run_openhands(_prompt, config):
        class Conversation:
            def send_message(self, message, sender=None):
                messages.append(message)

        with AdvisorEventStore(store_path) as store, AdvisorEventPump(
            store,
            Conversation(),
            poll_interval=0.001,
            parent_conversation_id=str(config.conversation_id),
        ):
            time.sleep(0.01)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("student", state_dir, conversation_id),
        full_prompt="student initial controller context",
        github_mailbox=Mailbox((feedback,)),
        active_poll_interval_seconds=0.001,
    ).run(
        feedback.to_prompt(),
        conversation_id=conversation_id,
        event_keys=frozenset({feedback.dedupe_key}),
    )

    assert messages == []
    with AdvisorEventStore(store_path) as store:
        assert store.pending() == []


def test_full_visible_set_suppresses_events_handled_in_an_earlier_turn(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000017")
    handled = advisor_event(17)
    current = advisor_event(18)
    store_path = state_dir / "advisor-events.sqlite3"
    messages = []

    def run_openhands(_prompt, _config):
        class Conversation:
            def send_message(self, message, sender=None):
                messages.append(message)

        with AdvisorEventStore(store_path) as store, AdvisorEventPump(
            store,
            Conversation(),
            poll_interval=0.001,
        ):
            time.sleep(0.01)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("advisor", state_dir, conversation_id),
        full_prompt="advisor initial controller context",
        github_mailbox=Mailbox((handled, current)),
        active_poll_interval_seconds=0.001,
    ).run(
        current.to_prompt(),
        conversation_id=conversation_id,
        event_keys=frozenset({current.dedupe_key}),
        visible_event_keys=frozenset(
            {handled.dedupe_key, current.dedupe_key}
        ),
    )

    assert messages == []
    with AdvisorEventStore(store_path) as store:
        assert store.pending() == []


def test_acknowledged_store_rows_are_not_reported_as_this_turn_deliveries(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000018")
    handled = advisor_event()
    store_path = state_dir / "advisor-events.sqlite3"
    with AdvisorEventStore(store_path) as store:
        store.enqueue(
            AdvisorEvent(
                kind=handled.kind,
                dedupe_key=handled.dedupe_key,
                payload=handled.payload,
            )
        )
        store.acknowledge(handled.dedupe_key)

    def run_openhands(_prompt, _config):
        time.sleep(0.01)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("advisor", state_dir, conversation_id),
        full_prompt="advisor initial controller context",
        github_mailbox=Mailbox((handled,)),
        active_poll_interval_seconds=0.001,
    ).run(
        "current advisor turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )



def test_active_watcher_retries_after_a_transient_github_read_error(
    tmp_path: Path,
    capsys,
):
    event = advisor_event()

    class FlakyMailbox:
        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            if self.calls == 1:
                raise GitHubReadError("temporary outage")
            return (event,)

    mailbox = FlakyMailbox()
    store_path = tmp_path / "advisor-events.sqlite3"
    with ActiveMailboxWatcher(
        mailbox,
        store_path,
        known_keys=frozenset(),
        poll_interval_seconds=0.001,
    ) as watcher:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with AdvisorEventStore(store_path) as store:
                if store.pending_count():
                    break
            time.sleep(0.001)
        assert watcher.thread.is_alive()

    assert watcher.error is None
    assert mailbox.calls >= 2
    with AdvisorEventStore(store_path) as store:
        assert [pending.dedupe_key for pending in store.pending()] == [
            event.dedupe_key
        ]
    assert "SENPAI_MAILBOX_WATCHER_POLL_ERROR" in capsys.readouterr().err


def test_active_watcher_honors_github_retry_delay(tmp_path: Path):
    event = advisor_event()

    class RateLimitedMailbox:
        def __init__(self):
            self.calls: list[float] = []

        def poll(self):
            self.calls.append(time.monotonic())
            if len(self.calls) == 1:
                raise GitHubReadError("rate limited", retry_after_seconds=0.03)
            return (event,)

    mailbox = RateLimitedMailbox()
    store_path = tmp_path / "advisor-events.sqlite3"
    with ActiveMailboxWatcher(
        mailbox,
        store_path,
        known_keys=frozenset(),
        poll_interval_seconds=0.001,
    ):
        deadline = time.monotonic() + 1
        while len(mailbox.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)

    assert len(mailbox.calls) >= 2
    assert mailbox.calls[1] - mailbox.calls[0] >= 0.025


def test_context_exhaustion_retries_once_on_a_fresh_branch_with_the_same_id(
    tmp_path: Path,
    monkeypatch,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000091")
    clock = [100.0]
    calls = []

    def run_openhands(prompt, config, *, reset_context=False):
        calls.append(
            (
                prompt,
                config.conversation_id,
                reset_context,
                config.timeout_seconds,
            )
        )
        if len(calls) == 1:
            clock[0] += 25
            raise ConversationRunError(
                conversation_id,
                LLMContextWindowExceedError("context length exceeded"),
            )
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)
    monkeypatch.setattr("senpai_agent.controller.time.monotonic", lambda: clock[0])

    result = OpenHandsTurnRunner(
        Config("advisor", tmp_path / "state", conversation_id, timeout_seconds=100),
        full_prompt="complete current controller context",
    ).run(
        "current actionable event",
        conversation_id=conversation_id,
        event_keys=frozenset({"event:91"}),
    )

    assert result.exit_code == 0
    assert calls[0] == ("current actionable event", conversation_id, False, 100)
    assert calls[1][1:] == (conversation_id, True, 100)
    assert "complete current controller context" in calls[1][0]
    assert "current actionable event" in calls[1][0]
    assert "raw trace and workspace are preserved" in calls[1][0]


def test_terminal_turn_recovery_preserves_history_and_excludes_new_events(
    tmp_path: Path,
    monkeypatch,
):
    """
    Requirement: a terminally stalled turn gets one fresh canonical branch.
    Interface: OpenHandsTurnRunner and the persistent inbox.
    """
    conversation_id = UUID("00000000-0000-0000-0000-000000000095")
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(conversation_id, "event:old", "old canonical event")
    original = inbox.next_turn(conversation_id, "unchanged controller prompt")
    assert original is not None

    class Conversation:
        def __init__(self):
            self.events = []
            self.state = SimpleNamespace(active_branch=lambda: list(self.events))

        def send_message(self, message, sender=None):
            self.events.append(SimpleNamespace(message=message, sender=sender))

    # The model-visible append happened once on the preserved old branch.
    conversation = Conversation()
    deliver_turn_messages(conversation, inbox, original.turn_id)
    for _attempt in range(3):
        inbox.record_inference_attempt(original.turn_id)

    inbox.enqueue(conversation_id, "event:new", "new event stays queued")
    calls = []

    def run_openhands(prompt, _config, **kwargs):
        calls.append((prompt, kwargs))
        recovery = inbox.recover_turn(
            kwargs["inbox_turn_id"],
            kwargs["recovery_prompt"],
            max_generations=1,
        )
        assert recovery.recovery_of == original.turn_id
        assert [event.event_key for event in recovery.events] == ["event:old"]
        assert inbox.pending_count(conversation_id) == 1
        for message in recovery.messages:
            inbox.record_delivered(message.delivery_id, message.body)
        inbox.record_processed(recovery.turn_id)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("advisor", tmp_path / "state", conversation_id),
        full_prompt="complete current controller context",
    ).run(
        original.prompt.body,
        conversation_id=conversation_id,
        event_keys=frozenset(original.event_keys),
        inbox=inbox,
        inbox_turn_id=original.turn_id,
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == "unchanged controller prompt"
    assert "complete current controller context" in calls[0][1]["recovery_prompt"]
    recovery = inbox.turn(inbox.turn(original.turn_id).superseded_by)
    assert recovery.state is DeliveryState.PROCESSED


def test_context_recovery_attempt_is_not_retried(
    tmp_path: Path,
    monkeypatch,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000092")
    calls = []

    def run_openhands(prompt, config, *, reset_context=False):
        calls.append((prompt, config.conversation_id, reset_context))
        error = (
            LLMContextWindowExceedError("context length exceeded")
            if len(calls) == 1
            else LLMMalformedConversationHistoryError("invalid tool history")
        )
        raise ConversationRunError(
            conversation_id,
            error,
        )

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    with pytest.raises(ConversationRecoveryExhausted) as raised:
        OpenHandsTurnRunner(
            Config("advisor", tmp_path / "state", conversation_id),
            full_prompt="complete current controller context",
        ).run(
            "current actionable event",
            conversation_id=conversation_id,
            event_keys=frozenset(),
        )

    assert [reset_context for _, _, reset_context in calls] == [False, True]
    assert raised.value.conversation_id == conversation_id
    assert isinstance(raised.value.__cause__, ConversationRunError)


def test_transient_failure_during_context_recovery_uses_normal_retry_semantics(
    tmp_path: Path,
    monkeypatch,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000093")
    calls = []

    def run_openhands(prompt, config, *, reset_context=False):
        calls.append((prompt, config.conversation_id, reset_context))
        if len(calls) == 1:
            raise ConversationRunError(
                conversation_id,
                LLMContextWindowExceedError("context length exceeded"),
            )
        raise RuntimeError("temporary provider outage")

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    with pytest.raises(RuntimeError, match="temporary provider outage"):
        OpenHandsTurnRunner(
            Config("advisor", tmp_path / "state", conversation_id),
            full_prompt="complete current controller context",
        ).run(
            "current actionable event",
            conversation_id=conversation_id,
            event_keys=frozenset(),
        )

    assert [reset_context for _, _, reset_context in calls] == [False, True]
