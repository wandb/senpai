import time
from base64 import b64encode
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMServiceUnavailableError,
)

import senpai_agent.controller as controller_module
from senpai_agent.controller import (
    ConversationRecoveryExhausted,
    Controller,
    TurnResult,
    _activity_lease,
    _full_prompt,
    _inference_state_lease,
    _provider_retry_delay,
)
from senpai_agent.inbox import (
    InboxTurnQuarantined,
    PersistentInbox,
    deliver_turn_messages,
)
from senpai_agent.mailbox import (
    ControllerEvent,
    StudentAssignmentAvailabilityMailbox,
)
from senpai_agent.program_context import ProgramSystemPrompt
from senpai_agent.state import StartedConversationLedger, WorkspaceDivergenceLedger
from senpai_agent.supervisor import ProgressLease, WorkerLease
from senpai_agent.system_instructions import SenpaiSystemInstructions
from senpai_agent.workspace import WorkspaceDivergence
from test_agent_markdown import HTML_HEADER


CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000001")


class Mailbox:
    def __init__(self, polls):
        self.polls = list(polls)
        self.calls = 0
        self.acknowledged = []

    def poll(self):
        self.calls += 1
        return self.polls.pop(0) if self.polls else ()

    def acknowledge(self, dedupe_keys):
        self.acknowledged.append(tuple(dedupe_keys))


class Turns:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []

    def run(
        self,
        prompt,
        *,
        conversation_id,
        event_keys,
        visible_event_keys=frozenset(),
        inbox,
        inbox_turn_id,
    ):
        self.calls.append(
            (prompt, conversation_id, event_keys, visible_event_keys)
        )
        outcome = self.outcomes.pop(0) if self.outcomes else TurnResult(exit_code=0)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome.exit_code == 0:
            turn = inbox.turn(inbox_turn_id)
            for message in turn.messages:
                inbox.record_delivered(message.delivery_id, message.body)
            inbox.record_processed(inbox_turn_id)
        return outcome


class ProviderTurns(Turns):
    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.turn_ids = []

    def run(self, *args, inbox, inbox_turn_id, **kwargs):
        self.turn_ids.append(inbox_turn_id)
        if isinstance(self.outcomes[0], Exception):
            mark_turn_delivered(inbox, inbox_turn_id)
            inbox.record_inference_attempt(inbox_turn_id)
        return super().run(
            *args,
            inbox=inbox,
            inbox_turn_id=inbox_turn_id,
            **kwargs,
        )


def provider_error() -> ConversationRunError:
    return ConversationRunError(
        CONVERSATION_ID,
        LLMServiceUnavailableError("anthropic overloaded"),
    )


def mark_turn_delivered(inbox: PersistentInbox, turn_id: str) -> None:
    for message in inbox.turn(turn_id).messages:
        inbox.record_delivered(message.delivery_id, message.body)


def controller(mailbox, turns, **overrides):
    return Controller(
        role=overrides.pop("role", "advisor"),
        mailbox=mailbox,
        turns=turns,
        conversation_id=overrides.pop("conversation_id", CONVERSATION_ID),
        full_prompt=overrides.pop("full_prompt", "programme"),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
        **overrides,
    )


def review_event(number=17):
    return ControllerEvent(
        kind="review_ready",
        dedupe_key=f"review_ready:{number}:abc",
        payload={"number": number},
    )


def human_event(message_id=101):
    return ControllerEvent(
        kind="human_issue",
        dedupe_key=f"human_issue:1:{message_id}",
        payload={"number": 1, "human_message_id": message_id},
    )


def human_pr_comment_event(comment_id=601):
    return ControllerEvent(
        kind="human_pr_comment",
        dedupe_key=(
            f"human_pr_comment:v2:issue_comment:17:{comment_id}:abc"
        ),
        payload={"number": 17, "feedback_id": comment_id},
    )


class MultiGenerationRecoveryTurns:
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
        first = inbox.reset_turn(inbox_turn_id, "first recovery")
        second = inbox.reset_turn(first.turn_id, "second recovery")
        assert first.turn_id != second.turn_id
        deliver_turn_messages(SimpleNamespace(
            events=[],
            state=SimpleNamespace(active_branch=lambda: []),
            send_message=lambda *_args, **_kwargs: None,
        ), inbox, second.turn_id)
        inbox.record_processed(second.turn_id)
        return TurnResult(exit_code=0)


class QuarantiningTurns:
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
        inbox.quarantine(inbox_turn_id, "recovery budget exhausted")
        raise InboxTurnQuarantined(inbox_turn_id, "recovery budget exhausted")

def research_base_event(current_sha="def"):
    return ControllerEvent(
        kind="research_base_changed",
        dedupe_key=f"research_base_changed:17:abc:{current_sha}",
        payload={
            "number": 17,
            "required_base_sha": "abc",
            "current_base_sha": current_sha,
        },
    )


def test_first_turn_contains_operator_instructions_without_runtime_identity():
    prompt = _full_prompt(
        {
            "GH_REPO": "acme/widgets",
            "ADVISOR_BRANCH": "research",
            "WANDB_ENTITY": "acme",
            "WANDB_PROJECT": "cfd",
            "WANDB_API_KEY": "live-secret",
            "STUDENT_NAME": "fern",
            "EXTRA_INSTRUCTIONS_B64": b64encode(
                (HTML_HEADER + "Use typed tools.").encode()
            ).decode(),
            "SENPAI_LAUNCH_CONTEXT_B64": b64encode(
                b"# Authoritative launch context\n\nSystem policy."
            ).decode(),
        },
    )

    assert "# Research programme" not in prompt
    assert "# Student task" not in prompt
    assert "live-secret" not in prompt
    assert "# Additional operator instructions\n\nUse typed tools." in prompt
    assert "# Runtime identity" not in prompt
    assert "acme/widgets" not in prompt
    assert "research" not in prompt
    assert "acme/cfd" not in prompt
    assert "fern" not in prompt
    assert "Authoritative launch context" not in prompt
    assert "SPDX-" not in prompt


def test_first_turn_without_launch_instructions_has_no_user_level_identity():
    prompt = _full_prompt(
        {
            "GH_REPO": "acme/widgets",
            "ADVISOR_BRANCH": "research",
            "WANDB_ENTITY": "acme",
            "WANDB_PROJECT": "cfd",
            "STUDENT_NAMES": "fern,frieren",
        },
    )

    assert prompt == ""


def test_controller_accepts_an_empty_optional_launch_prompt():
    prompt = controller(
        Mailbox([]),
        Turns(),
        full_prompt="",
    )._prompt((), continuing=False)

    assert prompt.startswith("Current time (UTC):")
    assert "Runtime identity" not in prompt


def test_empty_mailbox_does_not_start_a_model_turn():
    turns = Turns()

    controller(Mailbox([(), ()]), turns, role="student").run(max_cycles=2)

    assert turns.calls == []


def test_successful_turn_repolls_immediately_and_continues_without_full_brief():
    first = ControllerEvent(
        kind="student_available_for_assignment",
        dedupe_key="student_available_for_assignment:student-1",
        payload={"student": "student-1"},
    )
    second = review_event()
    mailbox = Mailbox([(first,), (second,), ()])
    turns = Turns()

    controller(mailbox, turns).run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({first.dedupe_key}),
        frozenset({second.dedupe_key}),
    ]
    assert "programme" in turns.calls[0][0]
    assert "programme" not in turns.calls[1][0]
    assert mailbox.calls == 3


def test_post_turn_snapshot_retracts_availability_queued_during_active_turn(
    tmp_path: Path,
):
    availability = ControllerEvent(
        kind="student_available_for_assignment",
        dedupe_key="student_available_for_assignment:student-1",
        payload={"student": "student-1"},
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    source = Mailbox([(review_event(),), ()])
    mailbox = StudentAssignmentAvailabilityMailbox(
        source,
        inbox=inbox,
        conversation_id=CONVERSATION_ID,
        event_store_path=tmp_path / "advisor-events.sqlite3",
    )

    class QueuingTurns(Turns):
        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            inbox.enqueue(
                CONVERSATION_ID,
                availability.dedupe_key,
                availability.to_prompt(),
            )
            return result

    turns = QueuingTurns()
    controller(mailbox, turns, inbox=inbox).run(max_cycles=1)

    assert len(turns.calls) == 1
    assert inbox.pending_count(CONVERSATION_ID) == 0
    assert source.calls == 2


def test_post_turn_poll_at_reminder_boundary_only_delivers_new_state(
    monkeypatch,
):
    original = review_event(17)
    changed = review_event(18)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PostTurnMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            if self.calls == 1:
                return (original,)
            if self.calls == 2:
                clock[0] = 30
            return (original, changed)

    mailbox = PostTurnMailbox([])
    turns = Turns()

    controller_module.Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=30,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=2)

    assert [call[2] for call in turns.calls] == [
        frozenset({original.dedupe_key}),
        frozenset({changed.dedupe_key}),
        frozenset({original.dedupe_key}),
    ]
    assert [call[3] for call in turns.calls] == [
        frozenset({original.dedupe_key}),
        frozenset({original.dedupe_key, changed.dedupe_key}),
        frozenset({original.dedupe_key, changed.dedupe_key}),
    ]


def test_failed_turn_resumes_without_admitting_new_events(tmp_path: Path):
    """
    Requirement: an unresolved turn is resumed before newly observed work.
    Interface: Controller turns and mailbox acknowledgements across retries.
    """
    first = review_event(17)
    later = review_event(18)
    mailbox = Mailbox([(first,), (first, later), ()])
    turns = Turns([TurnResult(exit_code=19), TurnResult(exit_code=0)])
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")

    controller(mailbox, turns, inbox=inbox).run(max_cycles=2)

    assert [call[2] for call in turns.calls[:2]] == [
        frozenset({first.dedupe_key}),
        frozenset({first.dedupe_key}),
    ]
    assert later.dedupe_key not in turns.calls[1][2]
    assert [call[2] for call in turns.calls] == [
        frozenset({first.dedupe_key}),
        frozenset({first.dedupe_key}),
        frozenset({later.dedupe_key}),
    ]
    assert mailbox.acknowledged == [
        (first.dedupe_key,),
        (later.dedupe_key,),
    ]


def test_108_events_survive_four_failed_inferences_without_visible_replay(
    tmp_path: Path,
):
    events = tuple(review_event(index) for index in range(108))

    class Conversation:
        def __init__(self):
            self.events = []
            self.sent = []
            self.state = SimpleNamespace(active_branch=lambda: list(self.events))

        def send_message(self, message, sender=None):
            self.sent.append(message)
            self.events.append(SimpleNamespace(message=message, sender=sender))

    class RetryingTurns:
        def __init__(self):
            self.calls = 0
            self.conversation = Conversation()

        def run(
            self,
            _prompt,
            *,
            conversation_id,
            event_keys,
            visible_event_keys=frozenset(),
            inbox,
            inbox_turn_id,
        ):
            del conversation_id, event_keys, visible_event_keys
            self.calls += 1
            deliver_turn_messages(self.conversation, inbox, inbox_turn_id)
            if self.calls <= 4:
                return TurnResult(exit_code=1)
            inbox.record_processed(inbox_turn_id)
            return TurnResult(exit_code=0)

    turns = RetryingTurns()
    Controller(
        role="advisor",
        mailbox=Mailbox([events]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
        max_consecutive_turn_failures=5,
    ).run(max_cycles=5)

    visible = Counter(turns.conversation.sent)
    assert turns.calls == 11
    assert all(visible[event.to_prompt()] == 1 for event in events)


def test_processed_turn_retries_mailbox_acknowledgement_without_inference(
    tmp_path: Path,
):
    """
    Requirement: post-inference recovery performs acknowledgement only.
    Interface: persistent inbox, Controller, mailbox, and TurnRunner calls.
    """
    event = review_event()
    path = tmp_path / "inbox.sqlite3"

    class FailingAckMailbox(Mailbox):
        def acknowledge(self, _dedupe_keys):
            raise RuntimeError("mailbox acknowledgement failed")

    first_turns = Turns()
    with pytest.raises(RuntimeError, match="mailbox acknowledgement failed"):
        controller(
            FailingAckMailbox([(event,)]),
            first_turns,
            inbox=PersistentInbox(path),
        ).run(max_cycles=1)

    assert len(first_turns.calls) == 1
    assert len(PersistentInbox(path).processed_turns()) == 1

    mailbox = Mailbox([(), ()])
    turns = Turns()
    controller(
        mailbox,
        turns,
        inbox=PersistentInbox(path),
    ).run(max_cycles=1)

    assert turns.calls == []
    assert mailbox.acknowledged == [(event.dedupe_key,)]
    assert PersistentInbox(path).processed_turns() == ()


def test_level_trigger_is_quiet_while_visible_and_wakes_after_reappearing():
    event = review_event()
    mailbox = Mailbox([(event,), (event,), (), (event,), ()])
    turns = Turns()

    controller(mailbox, turns).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]
    assert mailbox.acknowledged == [
        (event.dedupe_key,),
        (event.dedupe_key,),
    ]


def test_still_actionable_event_is_redelivered_after_one_poll_interval(
    monkeypatch,
):
    event = review_event()
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def __init__(self):
            super().__init__([])

        def poll(self):
            self.calls += 1
            return (event,)

    mailbox = PersistentMailbox()
    turns = Turns()

    controller_module.Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=30,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]


def test_restart_waits_one_reminder_interval_before_redelivery(
    tmp_path: Path,
    monkeypatch,
):
    event = review_event()
    path = tmp_path / "inbox.sqlite3"
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (event,)

    Controller(
        role="advisor",
        mailbox=PersistentMailbox([]),
        turns=Turns(),
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        inbox=PersistentInbox(path),
        sleep=lambda _seconds: None,
        poll_interval_seconds=30,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=1)

    restarted_turns = Turns()
    Controller(
        role="advisor",
        mailbox=PersistentMailbox([]),
        turns=restarted_turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        inbox=PersistentInbox(path),
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=30,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=2)

    assert [call[2] for call in restarted_turns.calls] == [
        frozenset({event.dedupe_key})
    ]


def test_older_visible_event_does_not_lose_its_reminder_to_another_turn(
    monkeypatch,
):
    older = review_event(17)
    newer = review_event(18)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (older,) if self.calls <= 2 else (older, newer)

    class WatcherAwareTurns(Turns):
        def run(
            self,
            prompt,
            *,
            conversation_id,
            event_keys,
            visible_event_keys=frozenset(),
            inbox,
            inbox_turn_id,
        ):
            result = super().run(
                prompt,
                conversation_id=conversation_id,
                event_keys=event_keys,
                visible_event_keys=visible_event_keys,
                inbox=inbox,
                inbox_turn_id=inbox_turn_id,
            )
            if (
                event_keys == frozenset({newer.dedupe_key})
                and older.dedupe_key not in visible_event_keys
            ):
                return TurnResult(
                    exit_code=result.exit_code,
                    delivered_event_keys=frozenset({older.dedupe_key}),
                )
            return result

    turns = WatcherAwareTurns()
    Controller(
        role="advisor",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=20,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({older.dedupe_key}),
        frozenset({newer.dedupe_key}),
        frozenset({older.dedupe_key}),
    ]
    assert turns.calls[1][3] == frozenset(
        {older.dedupe_key, newer.dedupe_key}
    )


def test_fast_poll_defaults_to_ten_minute_level_trigger_reminders(monkeypatch):
    event = review_event()
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def __init__(self):
            super().__init__([])

        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="advisor",
        mailbox=PersistentMailbox(),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=30,
        jitter_seconds=0,
    ).run(max_cycles=21)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(research_base_event(), id="research-base"),
        pytest.param(human_event(), id="human-issue"),
        pytest.param(
            ControllerEvent(
                kind="student_assignment_comment",
                dedupe_key="student_assignment_comment:v2:message-1",
                payload={"comment_id": "message-1"},
            ),
            id="student-comment",
        ),
        pytest.param(human_pr_comment_event(), id="human-pr-comment"),
    ],
)
def test_edge_triggered_event_does_not_repeat_on_reminder_cadence(
    monkeypatch,
    event,
):
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="advisor",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=600,
        jitter_seconds=0,
        event_reminder_seconds=600,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
    ]


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(human_event(), id="human-issue"),
        pytest.param(
            ControllerEvent(
                kind="student_assignment_comment",
                dedupe_key="student_assignment_comment:v2:message-1",
                payload={
                    "comment_id": "message-1",
                    "message": "STUDENT: Running.",
                },
            ),
            id="student-comment",
        ),
        pytest.param(human_pr_comment_event(), id="human-pr-comment"),
    ],
)
def test_edge_triggered_event_is_not_replayed_after_restart(
    tmp_path: Path,
    monkeypatch,
    event: ControllerEvent,
):
    inbox_path = tmp_path / "inbox.sqlite3"
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    controller(
        Mailbox([(event,), (event,)]),
        Turns(),
        inbox=PersistentInbox(inbox_path),
    ).run(max_cycles=1)
    restarted_turns = Turns()

    Controller(
        role="advisor",
        mailbox=Mailbox([(event,), (event,)]),
        turns=restarted_turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        inbox=PersistentInbox(inbox_path),
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=1,
        jitter_seconds=0,
        event_reminder_seconds=1,
    ).run(max_cycles=2)

    assert restarted_turns.calls == []


def test_new_human_message_wakes_after_previous_message_is_acknowledged():
    first = human_event(101)
    follow_up = human_event(102)

    class HumanMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (first,) if self.calls <= 2 else (follow_up,)

    turns = Turns()
    controller(HumanMailbox([]), turns).run(max_cycles=2)

    assert [call[2] for call in turns.calls] == [
        frozenset({first.dedupe_key}),
        frozenset({follow_up.dedupe_key}),
    ]


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(human_event(), id="human-issue"),
        pytest.param(human_pr_comment_event(), id="human-pr-comment"),
    ],
)
def test_acknowledged_human_instruction_does_not_replay_after_a_poll_gap(
    tmp_path: Path,
    event: ControllerEvent,
):
    inbox_path = tmp_path / "inbox.sqlite3"

    controller(
        Mailbox([(event,), (event,)]),
        Turns(),
        inbox=PersistentInbox(inbox_path),
    ).run(max_cycles=1)
    restarted_turns = Turns()

    controller(
        Mailbox([(), (event,), (event,)]),
        restarted_turns,
        inbox=PersistentInbox(inbox_path),
    ).run(max_cycles=2)

    assert restarted_turns.calls == []


def test_visible_event_identity_cannot_change_payload_between_polls(tmp_path: Path):
    first = ControllerEvent(
        kind="student_assignment_comment",
        dedupe_key="student_assignment_comment:v2:message-1",
        payload={"comment_id": "message-1", "message": "STUDENT: Running."},
    )
    edited = ControllerEvent(
        kind=first.kind,
        dedupe_key=first.dedupe_key,
        payload={"comment_id": "message-1", "message": "STUDENT: Changed."},
    )

    with pytest.raises(RuntimeError, match="reused with a different payload"):
        controller(
            Mailbox([(first,), (edited,)]),
            Turns(),
            inbox=PersistentInbox(tmp_path / "inbox.sqlite3"),
        ).run(max_cycles=1)


def test_changed_research_base_sha_wakes_immediately():
    first = research_base_event("def")
    changed = research_base_event("fed")
    turns = Turns()

    controller(Mailbox([(first,), (changed,), ()]), turns).run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({first.dedupe_key}),
        frozenset({changed.dedupe_key}),
    ]


def test_research_base_change_wakes_again_after_disappearing():
    event = research_base_event()
    turns = Turns()

    controller(
        Mailbox([(event,), (event,), (), (event,), ()]),
        turns,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]


def test_controller_main_does_not_derive_reminders_from_fast_polling(
    monkeypatch,
    tmp_path: Path,
):
    import senpai_agent.openhands_runner as runner_module
    import senpai_agent.tools as tools_module
    import senpai_agent.weave_monitoring as weave_module

    config = SimpleNamespace(
        model="anthropic/claude-opus-4-8",
        github_token="token",
        github_repo="acme/widgets",
        github_trusted_actor=None,
        state_dir=tmp_path,
        workspace=tmp_path,
        conversation_id=CONVERSATION_ID,
        timeout_seconds=7200,
        training_wandb_api_key=None,
        harness_file=tmp_path / "harness.md",
        role_file=tmp_path / "role.md",
        instructions=SenpaiSystemInstructions(
            harness="harness instructions",
            role="advisor role",
            program=ProgramSystemPrompt(
                program_path="program.md",
                source_commit="a" * 40,
                content="Test programme.",
            ),
            launch="# Authoritative launch context\n\nSystem policy.",
        ),
    )
    monkeypatch.setattr(runner_module, "parse_runner_args", lambda _argv: object())
    monkeypatch.setattr(
        runner_module,
        "resolve_config",
        lambda _args, _env: config,
    )
    monkeypatch.setattr(runner_module, "scrub_model_credentials", lambda *_: None)
    monkeypatch.setattr(tools_module, "close_training_runtimes", lambda: None)
    monkeypatch.setattr(weave_module, "finish_weave_monitoring", lambda: None)
    monkeypatch.setattr(
        controller_module,
        "GitHubMailbox",
        lambda **_kwargs: Mailbox([]),
    )
    monkeypatch.setattr(controller_module, "_full_prompt", lambda *_: "programme")

    created = []

    class CapturingController(Controller):
        def run(self, *, max_cycles=None):
            created.append(self)

    monkeypatch.setattr(controller_module, "Controller", CapturingController)

    assert (
        controller_module.controller_main(
            ["advisor"],
            {
                "SENPAI_ROLE": "advisor",
                "ADVISOR_BRANCH": "main",
                "SENPAI_POLL_INTERVAL_S": "30",
            },
        )
        == 0
    )
    assert created[0].poll_interval_seconds == 30
    assert created[0].event_reminder_seconds == 600
    assert created[0].full_prompt == "programme"
    assert created[0].turns.full_prompt == "programme"
    assert created[0].turns.active_poll_interval_seconds == 75
    assert isinstance(
        created[0].mailbox.mailboxes[0],
        StudentAssignmentAvailabilityMailbox,
    )
    assert created[0].turns.github_mailbox is created[0].mailbox.mailboxes[0]
    assert created[0].turn_timeout_seconds == 7260


def test_repeated_turn_failures_exit_to_the_supervisor_for_a_clean_restart():
    event = review_event()
    mailbox = Mailbox([(event,), (event,)])
    turns = Turns([RuntimeError("transport failed"), RuntimeError("still failed")])

    with pytest.raises(RuntimeError, match="turn-failure limit"):
        controller(
            mailbox,
            turns,
            max_consecutive_turn_failures=2,
        ).run(max_cycles=2)

    assert len(turns.calls) == 2


def test_exhausted_context_recovery_defers_then_retries_without_failure_streak(
    monkeypatch,
    capsys,
):
    event = review_event(17)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])
    mailbox = Mailbox([(event,), (event,), (event,), ()])
    turns = Turns(
        [
            ConversationRecoveryExhausted(
                CONVERSATION_ID,
                RuntimeError("clean branch also exceeded context"),
            ),
            TurnResult(exit_code=0),
        ]
    )

    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=600,
        jitter_seconds=0,
        max_consecutive_turn_failures=1,
    ).run(max_cycles=2)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]
    assert mailbox.acknowledged == [(event.dedupe_key,)]
    assert clock[0] == 600
    log = capsys.readouterr().err
    assert "SENPAI_TURN_DEFERRED" in log
    assert "retry_after_seconds=600" in log


def test_transient_provider_failure_defers_and_retries_the_same_turn(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "inbox.sqlite3"
    event = review_event(17)
    clock = [1_700_000_000.0]
    monkeypatch.setattr(controller_module.random, "uniform", lambda _low, _high: 0)
    monkeypatch.setattr(controller_module.time, "time", lambda: clock[0])
    failed_turn = ProviderTurns([provider_error()])
    inbox = PersistentInbox(path)
    Controller(
        role="advisor",
        mailbox=Mailbox([(event,)]),
        turns=failed_turn,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        inbox=inbox,
        max_consecutive_turn_failures=1,
    ).run(max_cycles=1)

    turn_id = failed_turn.turn_ids[0]
    preserved = inbox.latest_turn(turn_id)
    assert preserved.recovery_generation == 0
    assert preserved.quarantine_reason is None
    assert not inbox.terminal_recovery_due(turn_id, max_attempts=1)
    inbox.close()

    waiting_turn = ProviderTurns([TurnResult(exit_code=0)])
    Controller(
        role="advisor",
        mailbox=Mailbox([()]),
        turns=waiting_turn,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        inbox=PersistentInbox(path),
    ).run(max_cycles=1)
    assert waiting_turn.calls == []

    clock[0] += 30
    resumed_turn = ProviderTurns([TurnResult(exit_code=0)])
    mailbox = Mailbox([()])
    restarted_inbox = PersistentInbox(path)
    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=resumed_turn,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        inbox=restarted_inbox,
    ).run(max_cycles=1)

    assert resumed_turn.turn_ids == [turn_id]
    assert restarted_inbox.provider_cooldown() is None
    assert mailbox.acknowledged == [(event.dedupe_key,)]
    assert "SENPAI_PROVIDER_COOLDOWN" in capsys.readouterr().err


def test_permanent_provider_failure_exits_without_a_controller_retry():
    error = ConversationRunError(
        CONVERSATION_ID,
        LLMAuthenticationError("invalid provider key"),
    )
    turns = ProviderTurns([error])

    with pytest.raises(ConversationRunError):
        controller(
            Mailbox([(review_event(),)]),
            turns,
            max_consecutive_turn_failures=2,
        ).run(max_cycles=2)

    assert len(turns.calls) == 1


def test_provider_cooldown_schedule_honors_retry_after_and_adds_jitter(monkeypatch):
    monkeypatch.setattr(controller_module.random, "uniform", lambda _low, high: high)

    assert [_provider_retry_delay(failure, 0) for failure in range(6)] == [
        36,
        72,
        144,
        288,
        360,
        360,
    ]
    assert _provider_retry_delay(0, 90) == 96


def test_context_retry_deadline_survives_a_transiently_absent_event(
    monkeypatch,
):
    event = review_event(17)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])
    mailbox = Mailbox([(event,), (), (event,), ()])
    turns = Turns(
        [
            ConversationRecoveryExhausted(
                CONVERSATION_ID,
                RuntimeError("clean branch also exceeded context"),
            ),
        ]
    )

    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=30,
        jitter_seconds=0,
        max_consecutive_turn_failures=1,
    ).run(max_cycles=2)

    assert len(turns.calls) == 1
    assert mailbox.acknowledged == []
    assert clock[0] == 30


def test_preserved_workspace_divergence_is_delivered_to_the_existing_turn():
    event = review_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    turns = Turns()

    def reconcile(_events):
        raise conflict

    controller(
        Mailbox([(event,), ()]),
        turns,
        role="student",
        reconcile=reconcile,
    ).run(max_cycles=1)

    assert turns.calls[0][2] == frozenset(
        {event.dedupe_key, conflict.event.dedupe_key}
    )


def student_assignment_event():
    return ControllerEvent(
        kind="student_assignment",
        dedupe_key="student_assignment:assignment-17:revision-2:base:head",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
        },
    )


def test_identical_workspace_divergence_does_not_rewake_on_reminders(
    monkeypatch,
):
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="student",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=1,
        jitter_seconds=0,
        event_reminder_seconds=1,
    ).run(max_cycles=3)

    assert len(turns.calls) == 1


def test_workspace_divergence_suppression_survives_a_controller_restart(tmp_path):
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    ledger = WorkspaceDivergenceLedger(tmp_path / "workspace-divergence.json")
    first_turns = Turns()
    second_turns = Turns()

    for turns in (first_turns, second_turns):
        controller(
            Mailbox([(event,), ()]),
            turns,
            role="student",
            reconcile=lambda _events: (_ for _ in ()).throw(conflict),
            workspace_divergence_state=ledger,
        ).run(max_cycles=1)

    assert len(first_turns.calls) == 1
    assert second_turns.calls == []


def test_unacknowledged_workspace_divergence_is_not_suppressed_after_restart(
    tmp_path,
):
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    ledger = WorkspaceDivergenceLedger(tmp_path / "workspace-divergence.json")

    class FailingAcknowledgementMailbox(Mailbox):
        def acknowledge(self, _dedupe_keys):
            raise RuntimeError("mailbox acknowledgement failed")

    with pytest.raises(RuntimeError, match="mailbox acknowledgement failed"):
        controller(
            FailingAcknowledgementMailbox([(event,)]),
            Turns(),
            role="student",
            reconcile=lambda _events: (_ for _ in ()).throw(conflict),
            workspace_divergence_state=ledger,
        ).run(max_cycles=1)

    assert ledger.current(CONVERSATION_ID) is None

    restarted_turns = Turns()
    controller(
        Mailbox([(event,), ()]),
        restarted_turns,
        role="student",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
        workspace_divergence_state=ledger,
    ).run(max_cycles=1)

    assert len(restarted_turns.calls) == 1


def test_new_feedback_passes_through_an_unchanged_workspace_divergence():
    assignment = student_assignment_event()
    feedback = ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:issue_comment:17:101",
        payload={"message": "Use the accepted base."},
    )
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    turns = Turns()

    controller(
        Mailbox([(assignment,), (assignment, feedback), ()]),
        turns,
        role="student",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
    ).run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({assignment.dedupe_key, conflict.event.dedupe_key}),
        frozenset({feedback.dedupe_key, conflict.event.dedupe_key}),
    ]


def test_changed_workspace_divergence_wakes_again(monkeypatch):
    event = student_assignment_event()
    conflicts = iter(
        [
            WorkspaceDivergence(
                head_ref="student/candidate",
                expected_head="a" * 40,
                local_head="b" * 40,
            ),
            WorkspaceDivergence(
                head_ref="student/candidate",
                expected_head="a" * 40,
                local_head="c" * 40,
            ),
        ]
    )
    last = [None]
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    def reconcile(_events):
        last[0] = next(conflicts, last[0])
        raise last[0]

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="student",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        reconcile=reconcile,
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=1,
        jitter_seconds=0,
        event_reminder_seconds=1,
    ).run(max_cycles=2)

    assert len(turns.calls) == 2


def test_failed_workspace_divergence_turn_is_retried():
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    turns = Turns([TurnResult(exit_code=19), TurnResult(exit_code=0)])

    controller(
        Mailbox([(event,), (event,), ()]),
        turns,
        role="student",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
    ).run(max_cycles=2)

    assert len(turns.calls) == 2


def test_feedback_polled_after_a_turn_is_processed_in_the_next_turn():
    assignment = ControllerEvent(
        kind="student_assignment",
        dedupe_key="student_assignment:assignment-17:revision-2",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
        },
    )
    feedback = ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:issue_comment:17:101",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
        },
    )
    mailbox = Mailbox([(assignment,), (feedback,)])
    turns = Turns(
        [
            TurnResult(
                exit_code=0,
                delivered_event_keys=frozenset({feedback.dedupe_key}),
            )
        ]
    )

    controller(mailbox, turns, role="student").run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({assignment.dedupe_key}),
        frozenset({feedback.dedupe_key}),
    ]
    assert mailbox.acknowledged == [
        (assignment.dedupe_key,),
        (feedback.dedupe_key,),
    ]


def test_initial_assignment_precedes_feedback_when_the_batch_splits():
    base_assignment = student_assignment_event()
    assignment = ControllerEvent(
        kind=base_assignment.kind,
        dedupe_key=base_assignment.dedupe_key,
        payload={**base_assignment.payload, "brief": "x" * (70 * 1024)},
    )
    feedback = ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:issue_comment:17:101",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
            "message": "Apply this after reading the assignment.",
        },
    )
    turns = Turns()

    controller(
        Mailbox(((assignment, feedback), ())),
        turns,
        role="student",
    ).run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({assignment.dedupe_key}),
        frozenset({feedback.dedupe_key}),
    ]


def test_controller_follows_the_complete_recovery_chain_before_acknowledging():
    """
    Requirement: any bounded number of recovery generations completes one logical turn.
    Interface: controller acknowledgement of the original mailbox event.
    """
    event = review_event()
    mailbox = Mailbox(((event,), ()))

    controller(mailbox, MultiGenerationRecoveryTurns()).run(max_cycles=1)

    assert mailbox.acknowledged == [(event.dedupe_key,)]


def test_controller_quarantines_an_exhausted_turn_without_restarting(capsys):
    """
    Requirement: exhausting recovery is a durable visible stop, not a restart loop.
    Interface: controller lifetime, stderr, mailbox acknowledgement, and inbox readiness.
    """
    event = review_event()
    mailbox = Mailbox(((event,), ()))
    runtime = controller(
        mailbox,
        QuarantiningTurns(),
        max_consecutive_turn_failures=1,
    )

    runtime.run(max_cycles=1)

    assert mailbox.acknowledged == []
    assert runtime.inbox.ready_conversation_ids() == ()
    assert "SENPAI_TURN_QUARANTINED" in capsys.readouterr().err


@pytest.mark.parametrize(
    "instruction",
    [
        pytest.param(human_event(), id="human-issue"),
        pytest.param(human_pr_comment_event(), id="human-pr-comment"),
    ],
)
def test_new_human_instruction_reopens_the_same_quarantined_advisor(
    instruction,
):
    runtime = controller(
        Mailbox(((review_event(),), ())),
        QuarantiningTurns(),
        max_consecutive_turn_failures=1,
    )
    runtime.run(max_cycles=1)
    quarantined = runtime.inbox.quarantined_turns()[0]
    mailbox = Mailbox(((instruction,), ()))
    turns = Turns()

    controller(
        mailbox,
        turns,
        inbox=runtime.inbox,
    ).run(max_cycles=1)

    assert len(turns.calls) == 1
    assert turns.calls[0][1] == CONVERSATION_ID
    assert turns.calls[0][2] == frozenset(
        {review_event().dedupe_key, instruction.dedupe_key}
    )
    assert "programme" in turns.calls[0][0]
    assert runtime.inbox.turn(quarantined.turn_id).quarantine_reason is None
    assert mailbox.acknowledged == [
        (review_event().dedupe_key, instruction.dedupe_key)
    ]


def test_new_feedback_waits_behind_a_quarantined_student():
    assignment = student_assignment_event()
    runtime = controller(
        Mailbox(((assignment,), ())),
        QuarantiningTurns(),
        role="student",
        max_consecutive_turn_failures=1,
    )
    runtime.run(max_cycles=1)
    quarantined = runtime.inbox.quarantined_turns()[0]
    feedback = ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:issue_comment:17:101",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
            "message": "Try the narrower experiment next.",
        },
    )
    mailbox = Mailbox(((feedback,), ()))
    turns = Turns()

    controller(
        mailbox,
        turns,
        role="student",
        inbox=runtime.inbox,
    ).run(max_cycles=1)

    assert turns.calls == []
    assert runtime.inbox.turn(quarantined.turn_id).quarantine_reason == (
        "recovery budget exhausted"
    )
    assert runtime.inbox.pending_count(CONVERSATION_ID) == 1
    assert mailbox.acknowledged == []


def test_start_gate_wait_publishes_a_live_lease_before_polling(tmp_path: Path):
    gate = tmp_path / "start-gate"
    lease_path = tmp_path / "controller-lease.json"
    mailbox = Mailbox([()])
    observed = []

    def open_gate(seconds):
        observed.append((seconds, WorkerLease.read(lease_path)))
        gate.write_text("open")

    before = time.monotonic()
    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=Turns(),
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        progress=ProgressLease(lease_path),
        start_gate_path=gate,
        start_gate_poll_seconds=7,
        sleep=open_gate,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)
    after = time.monotonic()

    seconds, lease = observed[0]
    assert seconds == 7
    assert lease.phase == "start-gate"
    assert before + 307 <= lease.deadline <= after + 307
    assert mailbox.calls == 1
    assert WorkerLease.read(lease_path).phase == "poll"


def test_controller_requires_start_and_launch_gates_before_polling(
    tmp_path: Path,
):
    start_gate = tmp_path / "start-gate"
    launch_gate = tmp_path / "launch-gate"
    lease_path = tmp_path / "controller-lease.json"
    mailbox = Mailbox([()])
    sleeps = []
    lease_phases = []

    def open_next_gate(seconds):
        sleeps.append(seconds)
        lease_phases.append(WorkerLease.read(lease_path).phase)
        gate = start_gate if not start_gate.is_file() else launch_gate
        gate.write_text("open")

    controller = Controller(
        role="advisor",
        mailbox=mailbox,
        turns=Turns(),
        conversation_id=UUID("00000000-0000-0000-0000-000000000087"),
        full_prompt="programme",
        progress=ProgressLease(lease_path),
        start_gate_path=start_gate,
        launch_gate_path=launch_gate,
        start_gate_poll_seconds=7,
        sleep=open_next_gate,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert sleeps == [7, 7]
    assert lease_phases == ["start-gate", "start-gate"]
    assert mailbox.calls == 1


def test_restart_continues_a_conversation_after_its_first_success(tmp_path: Path):
    conversation_id = UUID("00000000-0000-0000-0000-000000000004")
    state_path = tmp_path / "started-conversations.json"
    StartedConversationLedger(state_path).mark_started(conversation_id)
    turns = Turns()

    controller(
        Mailbox(
            [
                (
                    ControllerEvent(
                        kind="training_monitor",
                        dedupe_key="training:finished",
                        payload={"conversation_id": str(conversation_id)},
                    ),
                ),
                (),
            ]
        ),
        turns,
        role="student",
        conversation_id=conversation_id,
        started_conversations=StartedConversationLedger(state_path),
    ).run(max_cycles=1)

    assert turns.calls[0][1] == conversation_id
    assert "programme" not in turns.calls[0][0]


def test_turn_lease_uses_the_configured_hard_deadline(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"
    observed = []

    class LeaseReadingTurns(Turns):
        def run(self, *args, **kwargs):
            observed.append(WorkerLease.read(lease_path))
            return super().run(*args, **kwargs)

    before = time.monotonic()
    controller(
        Mailbox([(review_event(),)]),
        LeaseReadingTurns(),
        progress=ProgressLease(lease_path),
        operation_timeout_seconds=123,
        turn_timeout_seconds=456,
    ).run(max_cycles=1)
    after = time.monotonic()

    assert observed[0].phase == "openhands-turn"
    assert before + 456 <= observed[0].deadline <= after + 456


def test_activity_lease_is_renewed_at_most_every_thirty_seconds(monkeypatch):
    clock = [100.0]
    updates = []
    progress = SimpleNamespace(
        update=lambda phase, timeout: updates.append((phase, timeout, clock[0]))
    )
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])
    renew = _activity_lease(progress, 3660)

    renew()
    clock[0] = 129.9
    renew()
    clock[0] = 130.0
    renew()

    assert updates == [
        ("openhands-turn", 3660, 100.0),
        ("openhands-turn", 3660, 130.0),
    ]


def test_activity_lease_write_failure_does_not_escape_the_event_path(
    monkeypatch,
    capsys,
):
    clock = [100.0]
    attempts = []

    def update(_phase, _timeout):
        attempts.append(clock[0])
        if len(attempts) == 1:
            raise OSError("lease volume unavailable")

    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])
    renew = _activity_lease(SimpleNamespace(update=update), 3660)

    renew()
    clock[0] = 101.0
    renew()
    clock[0] = 130.0
    renew()

    assert attempts == [100.0, 130.0]
    assert "SENPAI_LEASE_UPDATE_ERROR OSError: lease volume unavailable" in (
        capsys.readouterr().err
    )


def test_inference_state_lease_write_failure_does_not_mask_the_model_result(capsys):
    def update_llm_request(_started_at, _heartbeat_at):
        raise OSError("lease volume unavailable")

    publish = _inference_state_lease(
        SimpleNamespace(update_llm_request=update_llm_request)
    )

    publish(1_755_000_000.0, 1_755_000_001.0)

    assert "SENPAI_LEASE_UPDATE_ERROR OSError: lease volume unavailable" in (
        capsys.readouterr().err
    )


def test_only_an_acknowledged_turn_advances_supervisor_progress(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"
    event = review_event()
    mailbox = Mailbox([(event,), (event,), ()])

    controller(
        mailbox,
        Turns([TurnResult(exit_code=19), TurnResult(exit_code=0)]),
        progress=ProgressLease(lease_path),
    ).run(max_cycles=2)

    assert mailbox.acknowledged == [(event.dedupe_key,)]
    assert WorkerLease.read(lease_path).completed_turns == 1


def test_failed_acknowledgement_does_not_advance_supervisor_progress(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"

    class FailingAcknowledgeMailbox(Mailbox):
        def acknowledge(self, _dedupe_keys):
            raise RuntimeError("durable acknowledgement failed")

    with pytest.raises(RuntimeError, match="durable acknowledgement failed"):
        controller(
            FailingAcknowledgeMailbox([(review_event(),)]),
            Turns(),
            progress=ProgressLease(lease_path),
        ).run(max_cycles=1)

    assert WorkerLease.read(lease_path).completed_turns == 0
