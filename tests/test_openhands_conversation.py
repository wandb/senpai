import json
import signal
import threading
import time
from io import StringIO
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.event import (
    AgentErrorEvent,
    ConversationStateUpdateEvent,
    InterruptEvent,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import resolve_tool

import senpai_agent.openhands_runner as runner
from senpai_agent.controller import OpenHandsTurnRunner
from senpai_agent.inbox import (
    DeliveryState,
    InboxTurnQuarantined,
    PersistentInbox,
)
from senpai_agent.openhands_runner import (
    graceful_interrupts,
    main,
    reject_recovered_actions,
    run_openhands,
)
from openhands_support import (
    PLUGIN_DIR,
    isolate_agent_discovery,
    runtime_config,
)


def test_run_initializes_role_plugin_and_secrets_before_the_first_message(
    tmp_path,
    monkeypatch,
):
    captured = {}
    registered_secrets = []

    class FakeConversation:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.plugins = kwargs["plugins"]
            self.id = kwargs["conversation_id"]
            captured["secrets"] = kwargs["secrets"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            captured["llm_timeout"] = agent.llm.timeout
            captured["llm_num_retries"] = agent.llm.num_retries
            captured["llm_type"] = type(agent.llm)
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, prompt):
            captured["prompt"] = prompt
            captured["role"] = self.agent.agent_context.system_message_suffix
            captured["plugin"] = self.plugins[0].source
            captured["conversation_id_env"] = runner.os.environ[
                "SENPAI_CONVERSATION_ID"
            ]

        async def arun(self):
            pass

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "register_trace_secret", registered_secrets.append)
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(
        tmp_path,
        conversation_secrets={
            "WANDB_API_KEY": "wandb-key",
            "PRIVATE_AUTH": "private-key",
        },
    )

    assert run_openhands("first task", config) == 0
    assert captured["prompt"] == "first task"
    assert captured["role"] == (
        "# Senpai harness\n\nharness instructions\n\n"
        "# Senpai role\n\nadvisor role\n\n"
        "# program.md - program.md\n\nTest programme.\n\n"
        "# Authoritative launch context\n\nTest launch policy.\n"
    )
    assert captured["plugin"] == str(PLUGIN_DIR)
    assert captured["secrets"] == {
        "WANDB_API_KEY": "wandb-key",
        "PRIVATE_AUTH": "private-key",
    }
    assert registered_secrets == ["github-key"]
    assert captured["conversation_id_env"] == config.conversation_id.hex
    assert captured["delete_on_close"] is False
    assert captured["llm_timeout"] == 5400
    assert captured["llm_num_retries"] == 1
    assert captured["llm_type"] is runner.LLM
    assert captured["closed"] is True


def test_context_reset_preserves_history_and_starts_a_fresh_active_branch(
    tmp_path,
    monkeypatch,
):
    history = ["old message", "old response"]
    active = list(history)
    calls = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: active,
                events=history,
                execution_status=ConversationExecutionStatus.FINISHED,
            )

        def reject_pending_actions(self, reason):
            calls.append(("reject", reason))

        def navigate_to(self, event_id):
            calls.append(("navigate", event_id))
            active.clear()

        def send_message(self, prompt):
            calls.append(("send", prompt))
            active.append(prompt)

        async def arun(self):
            calls.append(("run", None))

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [object()],
    )
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(tmp_path)

    assert run_openhands("fresh recovery prompt", config, reset_context=True) == 0

    assert history == ["old message", "old response"]
    assert active == ["fresh recovery prompt"]
    assert [name for name, _ in calls] == [
        "reject",
        "navigate",
        "send",
        "run",
        "close",
    ]
    assert calls[1] == ("navigate", None)


def test_provider_timeout_resumes_inference_without_resending_the_turn(
    tmp_path,
    monkeypatch,
):
    """
    Requirement: a timed-out provider turn resumes with arun on its existing branch.
    Interface: run_openhands, the persistent inbox, and model-visible messages.
    """
    history = []
    sent = []
    arun_calls = []
    statuses = [
        ConversationExecutionStatus.PAUSED,
        ConversationExecutionStatus.FINISHED,
    ]

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(history),
                events=history,
                execution_status=ConversationExecutionStatus.IDLE,
            )

        def send_message(self, message, sender=None):
            sent.append((message, sender))
            history.append(SimpleNamespace(message=message, sender=sender))

        async def arun(self):
            arun_calls.append(1)
            self.state.execution_status = statuses.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(tmp_path)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "first event")
    turn = inbox.next_turn(config.conversation_id, "controller prompt")
    assert turn is not None

    assert run_openhands(
        "unused retry prompt",
        config,
        inbox=inbox,
        inbox_turn_id=turn.turn_id,
    ) == 1
    assert run_openhands(
        "another unused retry prompt",
        config,
        inbox=PersistentInbox(inbox.path),
        inbox_turn_id=turn.turn_id,
    ) == 0

    assert [message for message, _sender in sent] == [
        "controller prompt",
        "first event",
    ]
    assert len(arun_calls) == 2
    assert PersistentInbox(inbox.path).turn(turn.turn_id).state is (
        DeliveryState.PROCESSED
    )


def test_processed_turn_recovers_without_append_or_inference(tmp_path, monkeypatch):
    """
    Requirement: a processed turn is acknowledgement-only after restart.
    Interface: run_openhands against a reopened persistent inbox.
    """
    calls = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: [],
                events=[],
                execution_status=ConversationExecutionStatus.FINISHED,
            )

        def send_message(self, *_args, **_kwargs):
            calls.append("send")

        async def arun(self):
            calls.append("arun")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(tmp_path)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "first event")
    turn = inbox.next_turn(config.conversation_id, "controller prompt")
    assert turn is not None
    for message in turn.messages:
        inbox.record_delivered(message.delivery_id, message.body)
    inbox.record_processed(turn.turn_id)

    assert run_openhands(
        "unused",
        config,
        inbox=PersistentInbox(inbox.path),
        inbox_turn_id=turn.turn_id,
    ) == 0
    assert calls == ["close"]


def test_finished_branch_repairs_processing_receipt_without_inference(
    tmp_path,
    monkeypatch,
):
    """
    Requirement: a crash after inference is recovered from durable conversation state.
    Interface: active branch, persistent inbox, send count, and arun count.
    """
    calls = []
    config = runtime_config(tmp_path)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "first event")
    turn = inbox.next_turn(config.conversation_id, "controller prompt")
    assert turn is not None
    for message in turn.messages:
        inbox.record_delivered(message.delivery_id, message.body)
    branch = [
        SimpleNamespace(message=message.body, sender=message.sender)
        for message in turn.messages
    ]
    branch.append(SimpleNamespace(message="completed answer", sender="agent"))

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(branch),
                events=branch,
                execution_status=ConversationExecutionStatus.FINISHED,
            )

        def send_message(self, *_args, **_kwargs):
            calls.append("send")

        async def arun(self):
            calls.append("arun")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands(
        "unused",
        config,
        inbox=PersistentInbox(inbox.path),
        inbox_turn_id=turn.turn_id,
    ) == 0
    assert calls == ["close"]
    assert PersistentInbox(inbox.path).turn(turn.turn_id).state is (
        DeliveryState.PROCESSED
    )


def test_terminal_budget_reconciles_a_finished_response_before_recovery(
    tmp_path,
    monkeypatch,
):
    """
    Requirement: a crash after inference never causes duplicate inference.
    Interface: OpenHandsTurnRunner over the durable inbox and conversation branch.
    """
    calls = []
    config = runtime_config(tmp_path)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "first event")
    turn = inbox.next_turn(config.conversation_id, "controller prompt")
    assert turn is not None
    for message in turn.messages:
        inbox.record_delivered(message.delivery_id, message.body)
    for _attempt in range(3):
        inbox.record_inference_attempt(turn.turn_id)
    branch = [
        SimpleNamespace(message=message.body, sender=message.sender)
        for message in turn.messages
    ]
    branch.append(SimpleNamespace(message="completed answer", sender="agent"))

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(branch),
                events=branch,
                execution_status=ConversationExecutionStatus.FINISHED,
            )

        def send_message(self, *_args, **_kwargs):
            calls.append("send")

        def navigate_to(self, _event_id):
            calls.append("navigate")

        async def arun(self):
            calls.append("arun")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)

    result = OpenHandsTurnRunner(
        config,
        full_prompt="complete initial controller context",
    ).run(
        turn.prompt.body,
        conversation_id=config.conversation_id,
        event_keys=frozenset(turn.event_keys),
        inbox=inbox,
        inbox_turn_id=turn.turn_id,
    )

    assert result.exit_code == 0
    assert calls == ["close"]
    assert inbox.turn(turn.turn_id).state is DeliveryState.PROCESSED
    assert inbox.turn(turn.turn_id).superseded_by is None


def test_paused_persisted_final_response_reconciles_without_inference(
    tmp_path,
    monkeypatch,
):
    """
    Requirement: cancellation after a final response never reruns inference.
    Interface: durable response evidence with a restored PAUSED SDK status.
    """
    calls = []
    config = runtime_config(tmp_path)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "first event")
    turn = inbox.next_turn(config.conversation_id, "controller prompt")
    assert turn is not None
    branch = [
        SimpleNamespace(message=message.body, sender=message.sender)
        for message in turn.messages
    ]
    for message in turn.messages:
        inbox.record_delivered(message.delivery_id, message.body)
    branch.append(
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant",
                content=[TextContent(text="completed answer")],
            ),
            llm_response_id="response-1",
        )
    )

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(branch),
                events=branch,
                execution_status=ConversationExecutionStatus.PAUSED,
            )

        async def arun(self):
            calls.append("arun")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands(
        turn.prompt.body,
        config,
        inbox=inbox,
        inbox_turn_id=turn.turn_id,
    ) == 0
    assert calls == ["close"]
    assert inbox.turn(turn.turn_id).state is DeliveryState.PROCESSED


def test_stalled_recovery_is_bounded_and_quarantined_with_the_full_brief(
    tmp_path,
    monkeypatch,
):
    """
    Requirement: failed clean recovery cannot become a silent restart loop.
    Interface: repeated OpenHands turns, active model branch, and durable inbox.
    """
    active = []
    archived = []
    calls = []
    config = runtime_config(
        tmp_path,
        inbox_max_stalled_attempts=1,
        inbox_max_recovery_generations=1,
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "canonical event")
    turn = inbox.next_turn(config.conversation_id, "ordinary continuation")
    assert turn is not None

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(active),
                events=archived,
                execution_status=ConversationExecutionStatus.PAUSED,
            )

        def navigate_to(self, event_id):
            calls.append(("navigate", event_id))
            active.clear()

        def send_message(self, message, sender=None):
            calls.append(("send", message))
            event = SimpleNamespace(message=message, sender=sender)
            active.append(event)
            archived.append(event)

        async def arun(self):
            calls.append(("arun", None))
            self.state.execution_status = ConversationExecutionStatus.PAUSED

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)
    turns = OpenHandsTurnRunner(
        config, full_prompt="complete initial controller context"
    )

    first = turns.run(
        turn.prompt.body,
        conversation_id=config.conversation_id,
        event_keys=frozenset(turn.event_keys),
        inbox=inbox,
        inbox_turn_id=turn.turn_id,
    )
    second = turns.run(
        turn.prompt.body,
        conversation_id=config.conversation_id,
        event_keys=frozenset(turn.event_keys),
        inbox=inbox,
        inbox_turn_id=turn.turn_id,
    )

    assert first.exit_code == second.exit_code == 1
    assert [name for name, _value in calls].count("navigate") == 1
    assert "complete initial controller context" in active[0].message
    assert "Conversation context recovery" in active[0].message
    assert [event.message for event in active[1:]] == ["canonical event"]

    with pytest.raises(InboxTurnQuarantined, match="recovery budget exhausted"):
        turns.run(
            turn.prompt.body,
            conversation_id=config.conversation_id,
            event_keys=frozenset(turn.event_keys),
            inbox=inbox,
            inbox_turn_id=turn.turn_id,
        )

    quarantined = PersistentInbox(inbox.path).quarantined_turns()
    assert len(quarantined) == 1
    assert quarantined[0].quarantine_reason == "recovery budget exhausted"
    assert inbox.next_turn(config.conversation_id, "must remain blocked") is None


def test_model_visible_progress_renews_the_stalled_attempt_budget(
    tmp_path,
    monkeypatch,
):
    """
    Requirement: useful persisted activity is not mistaken for a stalled turn.
    Interface: repeated OpenHands turns and active model branch navigation.
    """
    active = []
    archived = []
    navigations = []
    config = runtime_config(
        tmp_path,
        inbox_max_stalled_attempts=1,
        inbox_max_recovery_generations=1,
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "canonical event")
    turn = inbox.next_turn(config.conversation_id, "ordinary continuation")
    assert turn is not None

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(active),
                events=archived,
                execution_status=ConversationExecutionStatus.PAUSED,
            )

        def navigate_to(self, event_id):
            navigations.append(event_id)
            active.clear()

        def send_message(self, message, sender=None):
            event = SimpleNamespace(message=message, sender=sender)
            active.append(event)
            archived.append(event)

        async def arun(self):
            event = ObservationEvent.model_construct(
                id=f"tool-progress-{len(archived)}",
                source="environment",
            )
            active.append(event)
            archived.append(event)
            self.state.execution_status = ConversationExecutionStatus.PAUSED

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)
    turns = OpenHandsTurnRunner(
        config, full_prompt="complete initial controller context"
    )

    for _attempt in range(2):
        assert turns.run(
            turn.prompt.body,
            conversation_id=config.conversation_id,
            event_keys=frozenset(turn.event_keys),
            inbox=inbox,
            inbox_turn_id=turn.turn_id,
        ).exit_code == 1

    assert navigations == []
    assert inbox.turn(turn.turn_id).superseded_by is None


def test_timeout_and_error_artifacts_do_not_renew_the_stalled_attempt_budget(
    tmp_path,
    monkeypatch,
):
    """
    Requirement: failed retries cannot manufacture their own liveness progress.
    Interface: persisted SDK timeout/error/state events across controller wakes.
    """
    active = []
    archived = []
    navigations = []
    artifacts = [
        InterruptEvent.model_construct(id="interrupt-1"),
        AgentErrorEvent.model_construct(id="error-1"),
        ConversationStateUpdateEvent.model_construct(id="state-1"),
    ]
    config = runtime_config(
        tmp_path,
        inbox_max_stalled_attempts=2,
        inbox_max_recovery_generations=1,
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "canonical event")
    turn = inbox.next_turn(config.conversation_id, "ordinary continuation")
    assert turn is not None

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(active),
                events=archived,
                execution_status=ConversationExecutionStatus.PAUSED,
            )

        def navigate_to(self, event_id):
            navigations.append(event_id)
            active.clear()

        def send_message(self, message, sender=None):
            event = SimpleNamespace(message=message, sender=sender)
            active.append(event)
            archived.append(event)

        async def arun(self):
            artifact = artifacts.pop(0)
            active.append(artifact)
            archived.append(artifact)
            self.state.execution_status = ConversationExecutionStatus.PAUSED

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)
    turns = OpenHandsTurnRunner(
        config, full_prompt="complete initial controller context"
    )

    for _attempt in range(3):
        assert turns.run(
            turn.prompt.body,
            conversation_id=config.conversation_id,
            event_keys=frozenset(turn.event_keys),
            inbox=inbox,
            inbox_turn_id=turn.turn_id,
        ).exit_code == 1

    assert navigations == [None]
    assert inbox.turn(turn.turn_id).superseded_by is not None


def test_context_reset_preserves_old_branch_and_requeues_once(tmp_path, monkeypatch):
    """
    Requirement: reset keeps the old trace and creates one canonical recovery copy.
    Interface: OpenHands branch navigation, sends, arun calls, and persistent inbox.
    """
    config = runtime_config(tmp_path)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "first event")
    old_turn = inbox.next_turn(config.conversation_id, "old controller prompt")
    assert old_turn is not None
    old_branch = [
        SimpleNamespace(message=message.body, sender=message.sender)
        for message in old_turn.messages
    ]
    for message in old_turn.messages:
        inbox.record_delivered(message.delivery_id, message.body)

    active = list(old_branch)
    archived = list(old_branch)
    calls = []
    statuses = [
        ConversationExecutionStatus.PAUSED,
        ConversationExecutionStatus.FINISHED,
    ]

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(active),
                events=archived,
                execution_status=ConversationExecutionStatus.ERROR,
            )

        def navigate_to(self, event_id):
            calls.append(("navigate", event_id))
            active.clear()

        def send_message(self, message, sender=None):
            calls.append(("send", message))
            event = SimpleNamespace(message=message, sender=sender)
            active.append(event)
            archived.append(event)

        async def arun(self):
            calls.append(("arun", None))
            self.state.execution_status = statuses.pop(0)

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)

    run_openhands(
        "fresh recovery prompt",
        config,
        reset_context=True,
        inbox=PersistentInbox(inbox.path),
        inbox_turn_id=old_turn.turn_id,
    )
    run_openhands(
        "unused retry prompt",
        config,
        inbox=PersistentInbox(inbox.path),
        inbox_turn_id=old_turn.turn_id,
    )

    assert archived[: len(old_branch)] == old_branch
    assert [name for name, _value in calls].count("navigate") == 1
    assert [name for name, _value in calls].count("send") == 2
    assert [name for name, _value in calls].count("arun") == 2
    assert len(active) == 2


def test_restart_after_recovery_commit_resets_before_delivering(tmp_path, monkeypatch):
    """A crash after reset persistence cannot deliver onto the polluted branch."""
    config = runtime_config(tmp_path)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(config.conversation_id, "event:1", "canonical event")
    old = inbox.next_turn(config.conversation_id, "old prompt")
    assert old is not None
    old_branch = [
        SimpleNamespace(message=message.body, sender=message.sender)
        for message in old.messages
    ]
    for message in old.messages:
        inbox.record_delivered(message.delivery_id, message.body)
    recovery = inbox.reset_turn(old.turn_id, "recovery prompt")

    active = list(old_branch)
    archived = list(old_branch)
    calls = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                active_branch=lambda: list(active),
                events=archived,
                execution_status=ConversationExecutionStatus.ERROR,
            )

        def navigate_to(self, event_id):
            calls.append(("navigate", event_id))
            active.clear()

        def send_message(self, message, sender=None):
            calls.append(("send", message))
            event = SimpleNamespace(message=message, sender=sender)
            active.append(event)
            archived.append(event)

        async def arun(self):
            calls.append(("arun", None))
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [],
    )
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands(
        "unused normal-restart prompt",
        config,
        inbox=PersistentInbox(inbox.path),
        inbox_turn_id=recovery.turn_id,
    ) == 0

    assert [name for name, _value in calls[:3]] == ["navigate", "send", "send"]
    assert [event.message for event in active] == [
        "recovery prompt",
        "canonical event",
    ]


def _child_result_conversation(reports, captured, *, rebuild_before_second=False):
    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.IDLE,
                view=SimpleNamespace(events=[]),
            )

        def send_message(self, prompt):
            captured.setdefault("prompts", []).append(prompt)
            if self.state.execution_status == ConversationExecutionStatus.FINISHED:
                self.state.execution_status = ConversationExecutionStatus.IDLE

        async def arun(self):
            report = reports[len(captured.setdefault("runs", []))]
            captured["runs"].append(report)
            if rebuild_before_second and len(captured["runs"]) == 2:
                self.state.view.events = []
            if report is not None:
                self.state.view.events.append(
                    runner.MessageEvent(
                        source="agent",
                        llm_message=Message(
                            role="assistant",
                            content=[TextContent(text=report)],
                        ),
                    )
                )
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            captured["closed"] = True

    return FakeConversation


def _output_record(output, prefix):
    return json.loads(
        next(
            line.removeprefix(prefix)
            for line in output.splitlines()
            if line.startswith(prefix)
        )
    )


def test_child_result_at_emergency_limit_is_returned_unchanged(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = {}
    monkeypatch.setattr(
        runner,
        "LocalConversation",
        _child_result_conversation(["bounded child report"], captured),
    )
    monkeypatch.setattr(
        runner.LLM,
        "get_token_count",
        lambda *_args, **_kwargs: runner.MAX_INLINE_CHILD_RESULT_TOKENS,
    )
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "WEAVE_PROJECT", "wandb-applied-ai-team/senpai-v1")
    config = runtime_config(tmp_path, child=True)

    assert run_openhands("child task", config) == 0

    output = capsys.readouterr().out
    result = _output_record(output, "OPENHANDS_RESULT ")
    run = _output_record(output, "OPENHANDS_RUN ")
    assert captured["delete_on_close"] is True
    assert captured["runs"] == ["bounded child report"]
    assert captured["closed"] is True
    assert result["result"] == "bounded child report"
    assert run["weave_url"] == (
        "https://wandb.ai/wandb-applied-ai-team/senpai-v1/"
        f"weave/agents/conversations/{config.conversation_id}"
    )


@pytest.mark.parametrize(
    ("initial_token_count", "rebuild_view"),
    [(0, False), (15_001, True)],
)
def test_oversized_child_result_is_spilled_then_replaced_by_one_summary(
    tmp_path,
    monkeypatch,
    capsys,
    initial_token_count,
    rebuild_view,
):
    raw_report = "RAW_REPORT_MUST_NOT_REACH_THE_PARENT"
    summary = "The strongest evidence points to the scheduler boundary."
    task_id = "task-1"
    target_checkout = tmp_path / "target"
    target_checkout.mkdir()
    role_state = tmp_path / "role-state"
    captured = {}
    token_counts = iter(
        [initial_token_count, runner.MAX_INLINE_CHILD_RESULT_TOKENS]
    )
    monkeypatch.setattr(
        runner,
        "LocalConversation",
        _child_result_conversation(
            [raw_report, summary],
            captured,
            rebuild_before_second=rebuild_view,
        ),
    )
    monkeypatch.setattr(
        runner.LLM,
        "get_token_count",
        lambda *_args, **_kwargs: next(token_counts),
    )
    isolate_agent_discovery(monkeypatch, runner)

    config = runtime_config(
        tmp_path,
        child=True,
        workspace=target_checkout,
        delegation_root_state_dir=role_state,
        delegation_task_id=task_id,
    )

    assert run_openhands("child task", config) == 0

    artifact = role_state / "delegation" / "results" / f"{task_id}.md"
    output = capsys.readouterr().out
    result = _output_record(output, "OPENHANDS_RESULT ")
    assert captured["runs"] == [raw_report, summary]
    assert len(captured["prompts"]) == 2
    assert "approxinately 1,500 tokens" in captured["prompts"][1]
    assert str(artifact) in captured["prompts"][1]
    assert artifact.read_text(encoding="utf-8") == raw_report
    assert not artifact.is_relative_to(target_checkout)
    assert artifact.parent.stat().st_mode & 0o777 == 0o700
    assert artifact.stat().st_mode & 0o777 == 0o600
    parent_state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(target_checkout)),
        agent=SimpleNamespace(
            llm=SimpleNamespace(vision_is_active=lambda: False),
        ),
    )
    parent_config = runtime_config(tmp_path, workspace=target_checkout)
    file_editor_spec = next(
        tool
        for tool in runner.build_main_tools(parent_config)
        if tool.name == "file_editor"
    )
    file_editor = resolve_tool(file_editor_spec, parent_state)[0]
    parent_read = file_editor(
        file_editor.action_type(command="view", path=str(artifact))
    )
    assert parent_read.is_error is False
    assert raw_report in parent_read.text
    assert result["result"] == f"{summary}\n\nFull report: {artifact}"
    assert raw_report not in result["result"]
    assert raw_report not in output


@pytest.mark.parametrize(
    ("summary", "summary_token_count"),
    [
        (None, 200),
        ("UNKNOWN_COUNT", 0),
        ("STILL_TOO_LARGE", runner.MAX_INLINE_CHILD_RESULT_TOKENS + 1),
    ],
)
def test_oversized_child_result_fails_closed_when_compression_fails(
    tmp_path,
    monkeypatch,
    capsys,
    summary,
    summary_token_count,
):
    raw_report = "RAW_REPORT_MUST_NOT_REACH_THE_PARENT"
    task_id = "task-1"
    role_state = tmp_path / "role-state"
    captured = {}
    token_counts = iter([15_001, summary_token_count])
    monkeypatch.setattr(
        runner,
        "LocalConversation",
        _child_result_conversation([raw_report, summary], captured),
    )
    monkeypatch.setattr(
        runner.LLM,
        "get_token_count",
        lambda *_args, **_kwargs: next(token_counts),
    )
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(
        tmp_path,
        child=True,
        delegation_root_state_dir=role_state,
        delegation_task_id=task_id,
    )

    with pytest.raises(RuntimeError) as raised:
        run_openhands("child task", config)

    artifact = role_state / "delegation" / "results" / f"{task_id}.md"
    output = capsys.readouterr().out
    assert str(artifact) in str(raised.value)
    assert raw_report not in str(raised.value)
    assert artifact.read_text(encoding="utf-8") == raw_report
    assert raw_report not in output


def test_student_requests_persistent_storage_for_monitor_wake(
    tmp_path,
    monkeypatch,
):
    captured = {}
    delegation = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            captured["tool_concurrency_limit"] = kwargs[
                "agent"
            ].tool_concurrency_limit
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "configure_delegation", delegation.append)

    assert run_openhands("student task", runtime_config(tmp_path, role="student")) == 0
    assert captured == {
        "delete_on_close": False,
        "tool_concurrency_limit": 8,
    }
    assert delegation[0].role == "student"
    assert delegation[-1] is None


def test_disabled_credentials_never_reach_the_agent_environment(
    tmp_path,
    monkeypatch,
):
    observed = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            observed.append(
                (
                    runner.os.environ.get("GITHUB_TOKEN"),
                    runner.os.environ.get("EXA_API_KEY"),
                )
            )

        async def arun(self):
            observed.append(
                (
                    runner.os.environ.get("GITHUB_TOKEN"),
                    runner.os.environ.get("EXA_API_KEY"),
                )
            )

        def close(self):
            pass

    monkeypatch.setenv("GITHUB_TOKEN", "stale-env-secret")
    monkeypatch.setenv("EXA_API_KEY", "ambient-exa-key")
    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands(
        "task", runtime_config(tmp_path, role="student", web_search=False)
    ) == 0
    assert observed == [(None, None), (None, None)]
    assert "GITHUB_TOKEN" not in runner.os.environ
    assert "EXA_API_KEY" not in runner.os.environ


@pytest.mark.parametrize(
    "status",
    [
        status
        for status in ConversationExecutionStatus
        if status is not ConversationExecutionStatus.FINISHED
    ],
    ids=lambda status: status.value,
)
def test_nonfinished_conversation_returns_failure(tmp_path, monkeypatch, status):
    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(execution_status=status)

        def send_message(self, _prompt):
            pass

        async def arun(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands("task", runtime_config(tmp_path)) == 1


@pytest.mark.parametrize("failure_stage", ["send_message", "run"])
def test_conversation_and_credentials_are_cleaned_up_after_failures(
    tmp_path,
    monkeypatch,
    failure_stage: str,
):
    closed = []
    cleared = []
    delegation = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]

        def send_message(self, _prompt):
            if failure_stage == "send_message":
                raise RuntimeError("initialization failed")

        async def arun(self):
            if failure_stage == "run":
                raise RuntimeError("execution failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "clear_github_credentials", lambda: cleared.append(True))
    monkeypatch.setattr(runner, "configure_delegation", delegation.append)

    with pytest.raises(RuntimeError, match="failed"):
        run_openhands("task", runtime_config(tmp_path))

    assert closed == [True]
    assert cleared
    assert delegation[-1] is None


def test_runtime_credentials_remain_configured_through_lazy_tool_initialization(
    tmp_path,
    monkeypatch,
):
    captured = {}
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class FakeConversation:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED,
                workspace=SimpleNamespace(working_dir=workspace),
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            specs = {
                tool.name: tool
                for tool in self.agent.tools
                if tool.name in {"senpai_github", "spawn_agents"}
            }
            captured["specs"] = specs
            captured["state"] = self.state
            captured["resolved"] = {
                name: {tool.name for tool in resolve_tool(spec, self.state)}
                for name, spec in specs.items()
            }

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    config = runtime_config(
        tmp_path,
        workspace=workspace,
        role="student",
        student_name="Fern",
    )
    assert run_openhands("task", config) == 0
    assert captured["resolved"] == {
        "senpai_github": {
            "get_prs",
            "post_assignment_comment",
            "respond_to_human_issue",
            "submit_experiment_result",
        },
        "spawn_agents": {"spawn_agents"},
    }
    with pytest.raises(
        RuntimeError,
        match="configure GitHub credentials before initializing workflows",
    ):
        resolve_tool(captured["specs"]["senpai_github"], captured["state"])
    with pytest.raises(RuntimeError, match="subagent runtime is not configured"):
        resolve_tool(captured["specs"]["spawn_agents"], captured["state"])


def test_turn_deadline_requests_conversation_interrupt(tmp_path, monkeypatch):
    interrupted = threading.Event()
    cancelled = threading.Event()

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.RUNNING
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            try:
                await runner.asyncio.Event().wait()
            finally:
                cancelled.set()

        def interrupt(self):
            interrupted.set()

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    assert (
            run_openhands(
                "task",
                runtime_config(tmp_path, timeout_seconds=0.2),
            )
        == 1
    )
    assert interrupted.is_set()
    assert cancelled.is_set()


def test_turn_activity_renews_the_inactivity_deadline():
    interrupted = False
    activity = [time.monotonic()]

    class Conversation:
        async def arun(self):
            for _ in range(3):
                await runner.asyncio.sleep(0.02)
                activity[0] = time.monotonic()

        def interrupt(self):
            nonlocal interrupted
            interrupted = True

    runner.asyncio.run(
        runner.arun_conversation(Conversation(), 0.03, lambda: activity[0])
    )

    assert not interrupted


def test_startup_interrupt_is_a_normal_conversation_end():
    class Conversation:
        task = None

        async def arun(self):
            self.task = runner.asyncio.current_task()
            await runner.asyncio.Event().wait()

        def interrupt(self):
            assert self.task is not None
            self.task.cancel()

    conversation = Conversation()

    runner.asyncio.run(
        runner.arun_conversation(
            conversation,
            1,
            started=conversation.interrupt,
        )
    )

    assert conversation.task.cancelled()


def test_steering_resumes_the_same_conversation_object():
    class Conversation:
        def __init__(self):
            self.id = "stable-advisor-id"
            self.runs = 0

        async def arun(self):
            self.runs += 1

        def interrupt(self):
            raise AssertionError("a completed run does not need interruption")

    resumes = iter((True, False))
    pump = SimpleNamespace(
        prepare_run=lambda: True,
        run_started=lambda: None,
        finish_run=lambda: next(resumes),
    )
    conversation = Conversation()

    runner.run_steerable_conversation(conversation, pump, 1)

    assert conversation.runs == 2
    assert conversation.id == "stable-advisor-id"


def test_signal_interrupts_the_conversation_and_restores_handlers(monkeypatch):
    calls = []
    installed = {}
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        installed[signum] = handler
        return previous[signum]

    conversation = SimpleNamespace(interrupt=lambda: calls.append("interrupt"))
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with pytest.raises(SystemExit) as raised:
        with graceful_interrupts(conversation):
            installed[signal.SIGTERM](signal.SIGTERM, None)
            calls.append("handler-returned")

    assert raised.value.code == 128 + signal.SIGTERM
    assert "interrupt" in calls
    assert "handler-returned" in calls
    assert calls[-2:] == [
        (signal.SIGTERM, previous[signal.SIGTERM]),
        (signal.SIGINT, previous[signal.SIGINT]),
    ]


def test_signal_does_not_resume_a_steering_interruption(monkeypatch):
    installed = {}
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}

    def fake_signal(signum, handler):
        installed[signum] = handler
        return previous[signum]

    class Conversation:
        def __init__(self):
            self.runs = 0

        async def arun(self):
            self.runs += 1

        def interrupt(self):
            pass

    class Pump:
        prepare_run = staticmethod(lambda: True)
        run_started = staticmethod(lambda: None)

        @staticmethod
        def finish_run():
            installed[signal.SIGTERM](signal.SIGTERM, None)
            return True

    conversation = Conversation()
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with pytest.raises(SystemExit):
        with graceful_interrupts(conversation) as stop_requested:
            runner.run_steerable_conversation(
                conversation,
                Pump(),
                1,
                stop_requested=stop_requested,
            )

    assert conversation.runs == 1


def test_signal_at_run_start_cancels_before_the_run_continues(monkeypatch):
    installed = {}
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}

    def fake_signal(signum, handler):
        installed[signum] = handler
        return previous[signum]

    class Conversation:
        def __init__(self):
            self.continued = False

        async def arun(self):
            await runner.asyncio.sleep(0)
            self.continued = True

        def interrupt(self):
            pass

    class Pump:
        prepare_run = staticmethod(lambda: True)
        finish_run = staticmethod(lambda: False)

        @staticmethod
        def run_started():
            installed[signal.SIGTERM](signal.SIGTERM, None)

    conversation = Conversation()
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with pytest.raises(SystemExit):
        with graceful_interrupts(conversation) as stop_requested:
            runner.run_steerable_conversation(
                conversation,
                Pump(),
                1,
                stop_requested=stop_requested,
            )

    assert not conversation.continued


def test_recovered_actions_are_rejected_before_the_conversation_resumes(
    monkeypatch,
):
    rejected = []
    conversation = SimpleNamespace(
        state=SimpleNamespace(active_branch=lambda: ["persisted action"]),
        reject_pending_actions=lambda reason: rejected.append(reason),
    )
    monkeypatch.setattr(
        runner.ConversationState,
        "get_unmatched_actions",
        lambda _events: [object()],
    )

    assert reject_recovered_actions(conversation) == 1
    assert "rerun it explicitly" in rejected[0]


def test_main_removes_the_model_key_and_flushes_weave_after_failure(monkeypatch):
    flushed = []

    def fail_run(_prompt, _config):
        assert "ANTHROPIC_API_KEY" not in runner.os.environ
        raise RuntimeError("run failed")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr(runner.sys, "stdin", StringIO("first task"))
    monkeypatch.setattr(
        runner,
        "resolve_config",
        lambda _args: SimpleNamespace(api_key_env="ANTHROPIC_API_KEY"),
    )
    monkeypatch.setattr(runner, "run_openhands", fail_run)
    monkeypatch.setattr(runner, "finish_weave_monitoring", lambda: flushed.append(True))

    with pytest.raises(RuntimeError, match="run failed"):
        main(["--max-turns", "1"])

    assert flushed == [True]
