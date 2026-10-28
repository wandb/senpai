import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from litellm.exceptions import PermissionDeniedError
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMContextWindowExceedError,
    LLMMalformedConversationHistoryError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)

from senpai_agent.advisor import AdvisorEventPump
from senpai_agent.controller import (
    ConversationRecoveryExhausted,
    OpenHandsTurnRunner,
    _context_recovery_prompt,
    _provider_failure,
)
from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.mailbox import ActiveGitHubWatcher
from senpai_agent.inbox import (
    MAX_EVENTS_PER_TURN,
    DeliveryState,
    PersistentInbox,
    deliver_turn_messages,
)
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.state import AssignmentConversationRegistry


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


def assignment_event(revision_id, assignment_id="assignment-17", number=17):
    return ControllerEvent(
        kind="student_assignment",
        dedupe_key=(
            f"student_assignment:v2:{number}:{assignment_id}:{revision_id}:abc"
        ),
        payload={
            "number": number,
            "url": f"https://github.test/acme/widgets/pull/{number}",
            "assignment_id": assignment_id,
            "revision_id": revision_id,
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


def test_context_recovery_prompt_does_not_repeat_an_embedded_research_brief():
    prompt = _context_recovery_prompt(
        "complete initial controller context",
        "updated operating context\n\ncomplete initial controller context\n\nactionable event",
    )

    assert prompt.count("complete initial controller context") == 1


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (LLMRateLimitError("rate limit"), True),
        (LLMTimeoutError("provider timeout"), True),
        (LLMServiceUnavailableError("provider unavailable"), True),
        (LLMAuthenticationError("invalid key"), False),
        (LLMBadRequestError("malformed request"), False),
    ],
)
def test_provider_failure_classification(error, retryable):
    wrapped = ConversationRunError(
        UUID("00000000-0000-0000-0000-000000000090"), error
    )

    assert _provider_failure(wrapped, 1_700_000_000).retryable is retryable


@pytest.mark.parametrize(
    "error",
    [
        LLMContextWindowExceedError("context length exceeded"),
        LLMMalformedConversationHistoryError("invalid tool history"),
        TimeoutError("local mailbox timeout"),
        ConnectionError("connection error talking to GitHub"),
    ],
)
def test_non_provider_failures_bypass_provider_cooldown(error):
    assert _provider_failure(error, 1_700_000_000) is None


def test_provider_permission_failure_is_permanent():
    error = PermissionDeniedError(
        "permission denied",
        "anthropic",
        "claude",
        httpx.Response(403, request=httpx.Request("POST", "https://anthropic.com")),
    )

    assert _provider_failure(error, 1_700_000_000).retryable is False


def test_provider_failure_honors_retry_after_from_the_exhausted_response():
    raw = RuntimeError("provider overloaded")
    raw.litellm_response_headers = {"retry-after": "90"}
    mapped = LLMServiceUnavailableError("provider overloaded")
    mapped.__cause__ = raw

    failure = _provider_failure(mapped, 1_700_000_000)

    assert failure is not None
    assert failure.retry_after == 90


def test_turn_runner_passes_the_inference_state_out_of_band(
    tmp_path: Path,
    monkeypatch,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000094")
    observed = []

    def run_openhands(_prompt, _config, *, on_inference_state):
        on_inference_state(1_755_000_000.0, 1_755_000_001.0, 123.0)
        return 0

    def publish_state(started_at, heartbeat_at, deadline):
        observed.append((started_at, heartbeat_at, deadline))

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("advisor", tmp_path / "state", conversation_id),
        full_prompt="complete current controller context",
        on_inference_state=publish_state,
    ).run(
        "current actionable event",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert result.exit_code == 0
    assert observed == [(1_755_000_000.0, 1_755_000_001.0, 123.0)]


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

        with LocalEventStore(
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
    assert result.delivered_event_keys == frozenset({current.dedupe_key})
    with LocalEventStore(state_dir / "student-events.sqlite3") as store:
        assert store.pending() == []


@pytest.mark.parametrize(
    "incoming",
    [
        assignment_event("revision-3"),
        assignment_event("revision-1", assignment_id="assignment-21", number=21),
    ],
    ids=["new-revision", "new-pull-request"],
)
def test_running_student_is_told_when_a_new_assignment_supersedes_it(
    tmp_path: Path,
    monkeypatch,
    incoming,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.assign("assignment-17", "revision-2")
    current = f"{incoming.payload['assignment_id']}:{incoming.payload['revision_id']}"
    messages = []

    def run_openhands(_prompt, config):
        class Conversation:
            def send_message(self, message):
                messages.append(message)

        with LocalEventStore(
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
        github_mailbox=Mailbox((assignment_event("revision-2"), incoming)),
        active_poll_interval_seconds=0.001,
    ).run(
        "current student turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert len(messages) == 1
    assert "assignment_superseded" in messages[0]
    assert (
        '"superseded_assignment_revision": "assignment-17:revision-2"'
        in messages[0]
    )
    assert f'"current_assignment_revision": "{current}"' in messages[0]
    assert "finish this turn" in messages[0]
    assert result.delivered_event_keys == frozenset(
        {f"assignment_superseded:{conversation_id}:{current}"}
    )


def test_resumed_superseded_conversation_is_told_to_finish(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    superseded = registry.assign("assignment-17", "revision-2")
    current = registry.assign("assignment-21", "revision-1")
    human_thread = registry.for_assignment("human-issue-4", "thread")
    store_path = state_dir / "student-events.sqlite3"
    routed = assignment_event("revision-1", assignment_id="assignment-21", number=21)
    pending_at_start = {}

    def run_openhands(_prompt, config):
        with LocalEventStore(store_path) as store:
            pending_at_start[config.conversation_id] = store.pending()
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)
    runner = OpenHandsTurnRunner(
        Config("student", state_dir, superseded),
        full_prompt="student initial controller context",
        github_mailbox=Mailbox((routed,)),
        active_poll_interval_seconds=0.001,
    )

    # A stale monitor signal resumes the superseded conversation after the
    # current assignment was already routed and is no longer a new event.
    for conversation_id in (superseded, current, human_thread):
        runner.run(
            "training monitor signal",
            conversation_id=conversation_id,
            event_keys=frozenset(),
            visible_event_keys=frozenset({routed.dedupe_key}),
        )

    (notice,) = pending_at_start[superseded]
    assert notice.kind == "assignment_superseded"
    assert notice.payload["superseded_assignment_revision"] == (
        "assignment-17:revision-2"
    )
    assert notice.payload["current_assignment_revision"] == "assignment-21:revision-1"
    assert notice.payload["parent_conversation_id"] == str(superseded)
    assert pending_at_start[current] == []
    assert pending_at_start[human_thread] == []
    with LocalEventStore(store_path) as store:
        assert store.pending() == []


def test_superseded_notice_does_not_outlive_the_running_turn(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.assign("assignment-17", "revision-2")
    store_path = state_dir / "student-events.sqlite3"

    def run_openhands(_prompt, _config):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with LocalEventStore(store_path) as store:
                if store.pending_count():
                    break
            time.sleep(0.001)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("student", state_dir, conversation_id),
        full_prompt="student initial controller context",
        github_mailbox=Mailbox((assignment_event("revision-3"),)),
        active_poll_interval_seconds=0.001,
    ).run(
        "student turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert result.delivered_event_keys == frozenset()
    with LocalEventStore(store_path) as store:
        assert store.pending() == []


def superseded_inbox_turn(tmp_path: Path, *, queued_events: int):
    """A superseded conversation's turn, built the way the controller builds it."""

    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    superseded = registry.assign("assignment-17", "revision-2")
    registry.assign("assignment-21", "revision-1")
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    for index in range(queued_events):
        inbox.enqueue(superseded, f"event:{index}", f"queued event {index}")
    turn = inbox.next_turn(superseded, "student controller prompt")
    assert turn is not None
    return state_dir, superseded, inbox, turn


def run_turn_through_inbox_pump(monkeypatch, state_dir, inbox, *, until):
    """Stand in for run_openhands with the production inbox-backed event pump."""

    seen = {}

    def run_openhands(_prompt, config, **kwargs):
        turn_id = kwargs["inbox_turn_id"]
        with LocalEventStore(
            state_dir / "student-events.sqlite3"
        ) as store, AdvisorEventPump(
            store,
            SimpleNamespace(),
            poll_interval=0.001,
            parent_conversation_id=str(config.conversation_id),
            inbox=inbox,
            conversation_id=config.conversation_id,
        ):
            deadline = time.monotonic() + 1
            while not until(turn_id) and time.monotonic() < deadline:
                time.sleep(0.001)
        seen["attached"] = inbox.turn(turn_id).event_keys
        seen["unattached"] = inbox.pending_count(config.conversation_id)
        for message in inbox.turn(turn_id).messages:
            if message.state is DeliveryState.PENDING:
                inbox.record_delivered(message.delivery_id, message.body)
        inbox.record_processed(turn_id)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)
    return seen


def test_superseded_notice_steers_the_running_turn_through_the_inbox(
    tmp_path: Path,
    monkeypatch,
):
    state_dir, superseded, inbox, turn = superseded_inbox_turn(
        tmp_path, queued_events=1
    )
    notice_key = f"assignment_superseded:{superseded}:assignment-21:revision-1"
    seen = run_turn_through_inbox_pump(
        monkeypatch,
        state_dir,
        inbox,
        until=lambda turn_id: notice_key in inbox.turn(turn_id).event_keys,
    )

    result = OpenHandsTurnRunner(
        Config("student", state_dir, superseded),
        full_prompt="student initial controller context",
        github_mailbox=Mailbox(()),
        active_poll_interval_seconds=0.001,
    ).run(
        turn.prompt.body,
        conversation_id=superseded,
        event_keys=frozenset(turn.event_keys),
        inbox=inbox,
        inbox_turn_id=turn.turn_id,
    )

    assert result.exit_code == 0
    assert seen["attached"] == ("event:0", notice_key)
    assert seen["unattached"] == 0
    processed = inbox.turn(turn.turn_id)
    assert processed.state is DeliveryState.PROCESSED
    notice = next(
        message for message in processed.events if message.event_key == notice_key
    )
    assert "finish this turn" in notice.body
    assert inbox.pending_count(superseded) == 0
    with LocalEventStore(state_dir / "student-events.sqlite3") as store:
        assert store.pending() == []


def test_superseded_notice_a_full_turn_cannot_take_is_retracted_with_it(
    tmp_path: Path,
    monkeypatch,
):
    state_dir, superseded, inbox, turn = superseded_inbox_turn(
        tmp_path, queued_events=MAX_EVENTS_PER_TURN
    )
    notice_key = f"assignment_superseded:{superseded}:assignment-21:revision-1"
    seen = run_turn_through_inbox_pump(
        monkeypatch,
        state_dir,
        inbox,
        until=lambda _turn_id: inbox.pending_count(superseded) == 1,
    )

    result = OpenHandsTurnRunner(
        Config("student", state_dir, superseded),
        full_prompt="student initial controller context",
        github_mailbox=Mailbox(()),
        active_poll_interval_seconds=0.001,
    ).run(
        turn.prompt.body,
        conversation_id=superseded,
        event_keys=frozenset(turn.event_keys),
        inbox=inbox,
        inbox_turn_id=turn.turn_id,
    )

    assert result.exit_code == 0
    # The pump moved the notice into the inbox, but the full turn could not take it.
    assert notice_key not in seen["attached"]
    assert seen["unattached"] == 1
    # It must not become this superseded conversation's next turn.
    assert inbox.pending_count(superseded) == 0
    inbox.acknowledge(turn.turn_id)
    assert inbox.next_turn(superseded, "student continuation prompt") is None


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
    conversation_id = registry.assign("assignment-17", "revision-2")
    store_path = state_dir / "student-events.sqlite3"

    def run_openhands(_prompt, _config):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with LocalEventStore(store_path) as store:
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

    assert result.delivered_event_keys == frozenset()
    with LocalEventStore(store_path) as store:
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
    with LocalEventStore(store_path) as store:
        store.enqueue(
            LocalEvent(
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

        with LocalEventStore(store_path) as store, AdvisorEventPump(
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
    assert result.delivered_event_keys == frozenset()
    with LocalEventStore(store_path) as store:
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

        with LocalEventStore(store_path) as store, AdvisorEventPump(
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
    assert result.delivered_event_keys == frozenset()
    with LocalEventStore(store_path) as store:
        assert store.pending() == []


def test_acknowledged_store_rows_are_not_reported_as_this_turn_deliveries(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    conversation_id = UUID("00000000-0000-0000-0000-000000000018")
    handled = advisor_event()
    store_path = state_dir / "advisor-events.sqlite3"
    with LocalEventStore(store_path) as store:
        store.enqueue(
            LocalEvent(
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
            with LocalEventStore(store_path) as store:
                if store.pending_count():
                    break
            time.sleep(0.001)
        assert watcher.thread.is_alive()

    assert watcher.error is None
    assert mailbox.calls >= 2
    with LocalEventStore(store_path) as store:
        assert [pending.dedupe_key for pending in store.pending()] == [
            event.dedupe_key
        ]
    assert "SENPAI_GITHUB_WATCHER_POLL_ERROR" in capsys.readouterr().err


def test_active_watcher_does_not_queue_student_availability(tmp_path: Path):
    event = ControllerEvent(
        kind="student_available_for_assignment",
        dedupe_key="student_available_for_assignment:Fern",
        payload={"student": "Fern"},
    )

    class Mailbox:
        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return (event,)

    mailbox = Mailbox()
    store_path = tmp_path / "advisor-events.sqlite3"
    with ActiveGitHubWatcher(
        mailbox,
        store_path,
        known_keys=frozenset(),
        poll_interval_seconds=0.001,
    ) as watcher:
        deadline = time.monotonic() + 1
        while mailbox.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.001)

    assert mailbox.calls >= 2
    assert watcher.enqueued_keys == set()
    with LocalEventStore(store_path) as store:
        assert store.pending() == []


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
