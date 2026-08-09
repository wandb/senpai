import time
import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.llm.exceptions import (
    LLMContextWindowExceedError,
    LLMMalformedConversationHistoryError,
)

from senpai_agent.advisor import AdvisorEvent, AdvisorEventPump, AdvisorEventStore
from senpai_agent.controller import (
    Controller,
    ConversationRecoveryExhausted,
    OpenHandsTurnRunner,
    _claim_context_reset,
    _context_recovery_prompt,
)
from senpai_agent.mailbox import (
    CompositeMailbox,
    ContextResetMailbox,
    ControllerEvent,
    LocalAdvisorMailbox,
)
from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.mailbox import ActiveGitHubWatcher
from senpai_agent.state import AssignmentConversationRegistry
from senpai_agent.operations import (
    ContextResetRequest,
    ContextResetRequestStore,
    RoleTarget,
)
from senpai_agent.role_control import _control_token, _raw_history_checkpoint
from senpai_agent.supervisor import WorkerLease


@dataclass(frozen=True)
class Config:
    role: str
    state_dir: Path
    conversation_id: UUID
    timeout_seconds: float = 3600


class Mailbox:
    def __init__(self, events):
        self.events = tuple(events)

    def poll(self):
        return self.events

    def acknowledge(self, _dedupe_keys):
        return


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


def advisor_event(number=17):
    return ControllerEvent(
        kind="review_ready",
        dedupe_key=f"review_ready:{number}:abc",
        payload={"number": number},
    )


def test_context_recovery_prompt_does_not_repeat_an_embedded_research_brief():
    prompt = _context_recovery_prompt(
        "complete research brief",
        "updated operating context\n\ncomplete research brief\n\nactionable event",
    )

    assert prompt.count("complete research brief") == 1


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
            def send_message(self, message):
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
        full_prompt="student research brief",
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
    assert result.delivered_event_keys == frozenset({current.dedupe_key})
    with AdvisorEventStore(state_dir / "student-events.sqlite3") as store:
        assert store.pending() == []


def test_observed_feedback_is_not_reported_delivered_until_the_event_pump_sends_it(
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
        full_prompt="student research brief",
        github_mailbox=Mailbox((feedback,)),
        active_poll_interval_seconds=0.001,
    ).run(
        "student turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert result.delivered_event_keys == frozenset()
    with AdvisorEventStore(store_path) as store:
        assert [event.dedupe_key for event in store.pending()] == [
            feedback.dedupe_key
        ]


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
            def send_message(self, message):
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
        full_prompt="student research brief",
        github_mailbox=Mailbox((feedback,)),
        active_poll_interval_seconds=0.001,
    ).run(
        feedback.to_prompt(),
        conversation_id=conversation_id,
        event_keys=frozenset({feedback.dedupe_key}),
    )

    assert messages == []
    assert result.delivered_event_keys == frozenset()
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
            def send_message(self, message):
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
        full_prompt="advisor research brief",
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
    assert result.delivered_event_keys == frozenset()
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
        full_prompt="advisor research brief",
        github_mailbox=Mailbox((handled,)),
        active_poll_interval_seconds=0.001,
    ).run(
        "current advisor turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert result.delivered_event_keys == frozenset()


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
    with ActiveGitHubWatcher(
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
    assert "SENPAI_GITHUB_WATCHER_POLL_ERROR" in capsys.readouterr().err


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
        full_prompt="complete current research brief",
    ).run(
        "current actionable event",
        conversation_id=conversation_id,
        event_keys=frozenset({"event:91"}),
    )

    assert result.exit_code == 0
    assert calls[0] == ("current actionable event", conversation_id, False, 100)
    assert calls[1][1:] == (conversation_id, True, 75)
    assert "complete current research brief" in calls[1][0]
    assert "current actionable event" in calls[1][0]
    assert "raw trace and workspace are preserved" in calls[1][0]


def test_queued_context_reset_is_consumed_by_the_owner_with_the_same_uuid(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000095")
    (state_dir / conversation_id.hex).mkdir(parents=True)
    count, digest = _raw_history_checkpoint(state_dir, conversation_id)
    assert count == 0 and digest is not None
    lease = WorkerLease(
        pid=os.getpid(),
        phase="sleep",
        deadline=10_000,
        conversation_id=str(conversation_id),
    )
    lease_path = state_dir / "controller-lease.json"
    lease_path.write_text(
        json.dumps(
            {
                "pid": lease.pid,
                "phase": lease.phase,
                "deadline": lease.deadline,
                "conversation_id": lease.conversation_id,
            }
        )
    )
    target = RoleTarget(research_tag="maple", role="advisor")
    request = ContextResetRequest(
        request_id="reset-95",
        target=target,
        expected_conversation_id=conversation_id,
        expected_control_token=_control_token(
            lease,
            conversation_id,
            digest,
            (),
        ),
        expected_raw_history_event_count=count,
        expected_raw_history_digest=digest,
        expected_pending_event_keys=(),
        recovery_prompt="Discard the noisy active branch and resume the current work.",
    )
    with ContextResetRequestStore(state_dir / "context-resets.sqlite3") as store:
        store.enqueue(request)

    calls = []

    def run_openhands(
        prompt,
        config,
        *,
        reset_context=False,
        context_reset_applied=None,
    ):
        calls.append((prompt, config.conversation_id, reset_context))
        assert context_reset_applied is not None
        context_reset_applied()
        return 0

    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("advisor", state_dir, conversation_id),
        full_prompt="complete research brief",
    ).run(
        "current actionable event",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert result.exit_code == 0
    assert calls[0][1:] == (conversation_id, True)
    assert request.recovery_prompt in calls[0][0]
    with ContextResetRequestStore(state_dir / "context-resets.sqlite3") as store:
        status = store.result(request.request_id)
    assert status.status == "completed"
    assert status.completion is not None
    assert status.completion.conversation_id == conversation_id


def test_queued_context_reset_wakes_an_otherwise_idle_controller(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000096")
    (state_dir / conversation_id.hex).mkdir(parents=True)
    count, digest = _raw_history_checkpoint(state_dir, conversation_id)
    assert count == 0 and digest is not None
    lease = WorkerLease(
        pid=os.getpid(),
        phase="sleep",
        deadline=10_000,
        conversation_id=str(conversation_id),
    )
    (state_dir / "controller-lease.json").write_text(
        json.dumps(
            {
                "pid": lease.pid,
                "phase": lease.phase,
                "deadline": lease.deadline,
                "conversation_id": lease.conversation_id,
            }
        )
    )
    target = RoleTarget(research_tag="maple", role="advisor")
    request = ContextResetRequest(
        request_id="reset-idle-96",
        target=target,
        expected_conversation_id=conversation_id,
        expected_control_token=_control_token(
            lease,
            conversation_id,
            digest,
            (),
        ),
        expected_raw_history_event_count=count,
        expected_raw_history_digest=digest,
        expected_pending_event_keys=(),
        recovery_prompt="Resume useful work from a clean active branch.",
    )
    queue_path = state_dir / "context-resets.sqlite3"
    with ContextResetRequestStore(queue_path) as store:
        store.enqueue(request)

    calls = []

    def run_openhands(
        prompt,
        config,
        *,
        reset_context=False,
        context_reset_applied=None,
    ):
        calls.append((prompt, config.conversation_id, reset_context))
        assert context_reset_applied is not None
        context_reset_applied()
        return 0

    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)
    mailbox = ContextResetMailbox(queue_path, target)
    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=OpenHandsTurnRunner(
            Config("advisor", state_dir, conversation_id),
            full_prompt="complete research brief",
        ),
        conversation_id=conversation_id,
        full_prompt="complete research brief",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)

    assert len(calls) == 1
    assert calls[0][1:] == (conversation_id, True)
    assert request.recovery_prompt in calls[0][0]
    with ContextResetRequestStore(queue_path) as store:
        assert store.result(request.request_id).status == "completed"


def test_context_reset_preserves_a_local_nudge_batched_into_the_same_turn(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000097")
    (state_dir / conversation_id.hex).mkdir(parents=True)
    count, digest = _raw_history_checkpoint(state_dir, conversation_id)
    assert count == 0 and digest is not None
    target = RoleTarget(research_tag="maple", role="advisor")
    event_store_path = state_dir / "advisor-events.sqlite3"
    nudge = AdvisorEvent(
        kind="supervisor_nudge",
        dedupe_key="supervisor-nudge:nudge-97",
        payload={"message": "Check the preserved experiment state."},
    )
    with AdvisorEventStore(event_store_path) as store:
        store.enqueue(nudge)
    lease = WorkerLease(
        pid=os.getpid(),
        phase="sleep",
        deadline=10_000,
        conversation_id=str(conversation_id),
    )
    (state_dir / "controller-lease.json").write_text(
        json.dumps(
            {
                "pid": lease.pid,
                "phase": lease.phase,
                "deadline": lease.deadline,
                "conversation_id": lease.conversation_id,
            }
        )
    )
    request = ContextResetRequest(
        request_id="reset-with-nudge-97",
        target=target,
        expected_conversation_id=conversation_id,
        expected_control_token=_control_token(
            lease,
            conversation_id,
            digest,
            (nudge.dedupe_key,),
        ),
        expected_raw_history_event_count=count,
        expected_raw_history_digest=digest,
        expected_pending_event_keys=(nudge.dedupe_key,),
        recovery_prompt="Recover without losing the already queued nudge.",
    )
    queue_path = state_dir / "context-resets.sqlite3"
    with ContextResetRequestStore(queue_path) as store:
        store.enqueue(request)

    delivered = []

    def run_openhands(
        prompt,
        _config,
        *,
        reset_context=False,
        context_reset_applied=None,
    ):
        delivered.append((prompt, reset_context))
        assert context_reset_applied is not None
        context_reset_applied()
        return 0

    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)
    mailbox = CompositeMailbox(
        LocalAdvisorMailbox(event_store_path),
        ContextResetMailbox(queue_path, target),
    )
    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=OpenHandsTurnRunner(
            Config("advisor", state_dir, conversation_id),
            full_prompt="complete research brief",
            github_mailbox=Mailbox(()),
            active_poll_interval_seconds=0.001,
        ),
        conversation_id=conversation_id,
        full_prompt="complete research brief",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)

    assert delivered and delivered[0][1] is True
    with ContextResetRequestStore(queue_path) as store:
        completion = store.result(request.request_id).completion
    assert completion is not None
    assert completion.pending_event_keys == (nudge.dedupe_key,)
    with AdvisorEventStore(event_store_path) as store:
        assert [event.dedupe_key for event in store.pending()] == [nudge.dedupe_key]


def test_context_reset_completion_allows_a_new_role_event_racing_after_claim(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000098")
    (state_dir / conversation_id.hex).mkdir(parents=True)
    count, digest = _raw_history_checkpoint(state_dir, conversation_id)
    assert count == 0 and digest is not None
    lease = WorkerLease(
        pid=os.getpid(),
        phase="sleep",
        deadline=10_000,
        conversation_id=str(conversation_id),
    )
    (state_dir / "controller-lease.json").write_text(
        json.dumps(
            {
                "pid": lease.pid,
                "phase": lease.phase,
                "deadline": lease.deadline,
                "conversation_id": lease.conversation_id,
            }
        )
    )
    target = RoleTarget(research_tag="maple", role="advisor")
    request = ContextResetRequest(
        request_id="reset-race-98",
        target=target,
        expected_conversation_id=conversation_id,
        expected_control_token=_control_token(lease, conversation_id, digest, ()),
        expected_raw_history_event_count=count,
        expected_raw_history_digest=digest,
        expected_pending_event_keys=(),
        recovery_prompt="Recover while preserving any newly arrived work.",
    )
    queue_path = state_dir / "context-resets.sqlite3"
    event_store_path = state_dir / "advisor-events.sqlite3"
    with ContextResetRequestStore(queue_path) as store:
        store.enqueue(request)

    def run_openhands(
        _prompt,
        _config,
        *,
        reset_context=False,
        context_reset_applied=None,
    ):
        assert reset_context is True
        with AdvisorEventStore(event_store_path) as store:
            store.enqueue(
                AdvisorEvent(
                    kind="supervisor_nudge",
                    dedupe_key="supervisor-nudge:raced-98",
                    payload={"message": "New work arrived during reset."},
                )
            )
        assert context_reset_applied is not None
        context_reset_applied()
        return 0

    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("advisor", state_dir, conversation_id),
        full_prompt="complete research brief",
    ).run(
        "current actionable event",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert result.exit_code == 0
    with ContextResetRequestStore(queue_path) as store:
        completion = store.result(request.request_id).completion
    assert completion is not None
    assert completion.pending_event_keys == ("supervisor-nudge:raced-98",)
    with AdvisorEventStore(event_store_path) as store:
        assert [event.dedupe_key for event in store.pending()] == [
            "supervisor-nudge:raced-98"
        ]


def test_context_reset_records_a_late_github_event_delivered_in_the_reset_prompt(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000099")
    (state_dir / conversation_id.hex).mkdir(parents=True)
    count, digest = _raw_history_checkpoint(state_dir, conversation_id)
    assert count == 0 and digest is not None
    lease = WorkerLease(
        pid=os.getpid(),
        phase="sleep",
        deadline=10_000,
        conversation_id=str(conversation_id),
    )
    (state_dir / "controller-lease.json").write_text(
        json.dumps(
            {
                "pid": lease.pid,
                "phase": lease.phase,
                "deadline": lease.deadline,
                "conversation_id": lease.conversation_id,
            }
        )
    )
    event_key = "github:issue-comment:99"
    event_store_path = state_dir / "advisor-events.sqlite3"
    with AdvisorEventStore(event_store_path) as store:
        store.enqueue(
            AdvisorEvent(
                kind="github_comment",
                dedupe_key=event_key,
                payload={"message": "Review feedback delivered in this prompt."},
            )
        )
    target = RoleTarget(research_tag="maple", role="advisor")
    request = ContextResetRequest(
        request_id="reset-github-99",
        target=target,
        expected_conversation_id=conversation_id,
        expected_control_token=_control_token(
            lease,
            conversation_id,
            digest,
            (event_key,),
        ),
        expected_raw_history_event_count=count,
        expected_raw_history_digest=digest,
        expected_pending_event_keys=(event_key,),
        recovery_prompt="Recover and act on the feedback in this prompt.",
    )
    queue_path = state_dir / "context-resets.sqlite3"
    with ContextResetRequestStore(queue_path) as store:
        store.enqueue(request)
    github_event = ControllerEvent(
        kind="github_comment",
        dedupe_key=event_key,
        payload={"message": "Review feedback delivered in this prompt."},
    )

    def run_openhands(
        _prompt,
        _config,
        *,
        reset_context=False,
        context_reset_applied=None,
    ):
        assert reset_context is True
        with AdvisorEventStore(event_store_path) as store:
            assert [event.dedupe_key for event in store.pending()] == [event_key]
        assert context_reset_applied is not None
        context_reset_applied()
        return 0

    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    monkeypatch.setenv("RESEARCH_TAG", "maple")
    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)
    mailbox = CompositeMailbox(
        Mailbox((github_event,)),
        ContextResetMailbox(queue_path, target),
    )
    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=OpenHandsTurnRunner(
            Config("advisor", state_dir, conversation_id),
            full_prompt="complete research brief",
            github_mailbox=Mailbox(()),
            active_poll_interval_seconds=0.001,
        ),
        conversation_id=conversation_id,
        full_prompt="complete research brief",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)

    with AdvisorEventStore(event_store_path) as store:
        assert store.pending() == []
    with ContextResetRequestStore(queue_path) as store:
        completion = store.result(request.request_id).completion
    assert completion is not None
    assert completion.pending_event_keys == ()
    assert completion.delivered_event_keys == (event_key,)


def test_context_reset_claim_preserves_a_new_event_arriving_after_enqueue(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000099")
    (state_dir / conversation_id.hex).mkdir(parents=True)
    count, digest = _raw_history_checkpoint(state_dir, conversation_id)
    assert count == 0 and digest is not None
    lease = WorkerLease(
        pid=os.getpid(),
        phase="sleep",
        deadline=10_000,
        conversation_id=str(conversation_id),
    )
    (state_dir / "controller-lease.json").write_text(
        json.dumps(
            {
                "pid": lease.pid,
                "phase": lease.phase,
                "deadline": lease.deadline,
                "conversation_id": lease.conversation_id,
            }
        )
    )
    target = RoleTarget(research_tag="maple", role="advisor")
    event_store_path = state_dir / "advisor-events.sqlite3"
    with AdvisorEventStore(event_store_path) as events:
        events.enqueue(
            AdvisorEvent(kind="nudge", dedupe_key="existing", payload={})
        )
    request = ContextResetRequest(
        request_id="reset-before-claim-99",
        target=target,
        expected_conversation_id=conversation_id,
        expected_control_token=_control_token(
            lease,
            conversation_id,
            digest,
            ("existing",),
        ),
        expected_raw_history_event_count=count,
        expected_raw_history_digest=digest,
        expected_pending_event_keys=("existing",),
        recovery_prompt="Recover while preserving all pending work.",
    )
    queue_path = state_dir / "context-resets.sqlite3"
    with ContextResetRequestStore(queue_path) as resets:
        resets.enqueue(request)
    with AdvisorEventStore(event_store_path) as events:
        events.enqueue(AdvisorEvent(kind="nudge", dedupe_key="new", payload={}))

    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    monkeypatch.setenv("RESEARCH_TAG", "maple")

    assert _claim_context_reset(state_dir, conversation_id) == request
    with ContextResetRequestStore(queue_path) as resets:
        assert resets.result(request.request_id).status == "processing"
    with AdvisorEventStore(event_store_path) as events:
        assert [event.dedupe_key for event in events.pending()] == ["existing", "new"]


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
            full_prompt="complete current research brief",
        ).run(
            "current actionable event",
            conversation_id=conversation_id,
            event_keys=frozenset(),
        )

    assert [reset_context for _, _, reset_context in calls] == [False, True]
    assert raised.value.conversation_id == conversation_id
    assert isinstance(raised.value.__cause__, ConversationRunError)


def test_exhausted_turn_budget_defers_without_starting_a_doomed_recovery(
    tmp_path: Path,
    monkeypatch,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000094")
    clock = [100.0]
    calls = []

    def run_openhands(prompt, config, *, reset_context=False):
        calls.append((prompt, config.conversation_id, reset_context))
        clock[0] += 100
        raise ConversationRunError(
            conversation_id,
            LLMContextWindowExceedError("context length exceeded"),
        )

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)
    monkeypatch.setattr("senpai_agent.controller.time.monotonic", lambda: clock[0])

    with pytest.raises(ConversationRecoveryExhausted) as raised:
        OpenHandsTurnRunner(
            Config("advisor", tmp_path / "state", conversation_id, timeout_seconds=100),
            full_prompt="complete current research brief",
        ).run(
            "current actionable event",
            conversation_id=conversation_id,
            event_keys=frozenset(),
        )

    assert len(calls) == 1
    assert isinstance(raised.value.error, TimeoutError)
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
            full_prompt="complete current research brief",
        ).run(
            "current actionable event",
            conversation_id=conversation_id,
            event_keys=frozenset(),
        )

    assert [reset_context for _, _, reset_context in calls] == [False, True]
