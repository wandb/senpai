import json
import signal
import threading
from io import StringIO
from types import SimpleNamespace

import pytest
from openhands.sdk import LLM, Agent, Tool
from openhands.sdk.conversation import ConversationExecutionStatus, ConversationState
from openhands.sdk.event import (
    AgentErrorEvent,
    ConversationStateUpdateEvent,
    InterruptEvent,
    MessageEvent,
    ObservationEvent,
    SystemPromptEvent,
)
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import FinishTool, resolve_tool
from openhands.sdk.workspace import LocalWorkspace

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
    migrate_persisted_disabled_tools,
    migrate_persisted_retired_tool_definitions,
    reject_recovered_actions,
    run_openhands,
    without_legacy_think,
)
from openhands_support import (
    PLUGIN_DIR,
    isolate_agent_discovery,
    runtime_config,
)
from pydantic import SecretStr

_RETIRED_TOOL_TYPES = (
    (
        "DelegateAgentTool",
        "delegate_agent",
        "DelegateAgentAction",
        "DelegateAgentObservation",
    ),
    (
        "RunTrainingTool",
        "run_training",
        "RunTrainingAction",
        "TrainingResultObservation",
    ),
    (
        "GetTrainingStatusTool",
        "get_training_status",
        "GetTrainingStatusAction",
        "TrainingResultObservation",
    ),
    (
        "CancelTrainingTool",
        "cancel_training",
        "CancelTrainingAction",
        "TrainingResultObservation",
    ),
    (
        "MonitorTrainingTool",
        "monitor_training",
        "MonitorTrainingAction",
        "MonitorTrainingObservation",
    ),
)


def _persisted_conversation_with_retired_tool_snapshots(tmp_path):
    conversation_id = runner.uuid.uuid4()
    persistence_root = tmp_path / "state"
    persistence_dir = persistence_root / conversation_id.hex
    workspace = LocalWorkspace(working_dir=tmp_path / "workspace")
    llm = LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key"))
    persisted_agent = Agent(llm=llm, tools=[Tool(name="delegate_agent")])
    agent = without_legacy_think(Agent(llm=llm, tools=[]))
    state = ConversationState.create(
        id=conversation_id,
        agent=persisted_agent,
        workspace=workspace,
        persistence_dir=str(persistence_dir),
    )
    system_event = state.append_event(
        SystemPromptEvent(
            system_prompt=TextContent(text="preserve the historical system prompt"),
            tools=[FinishTool.create()[0]],
        )
    )
    message_event = state.append_event(
        runner.MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="preserve this historical message")],
            ),
        )
    )
    system_path = next(
        path
        for path in (persistence_dir / "events").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["kind"] == "SystemPromptEvent"
    )
    payload = json.loads(system_path.read_text(encoding="utf-8"))
    payload["tools"] = [
        {
            "description": f"Historical {title} description.",
            "action_type": action,
            "observation_type": observation,
            "annotations": None,
            "kind": kind,
            "title": title,
        }
        for kind, title, action, observation in _RETIRED_TOOL_TYPES
    ] + payload["tools"]
    system_path.write_text(json.dumps(payload), encoding="utf-8")
    message_path = next(
        path
        for path in (persistence_dir / "events").glob("*.json")
        if path != system_path
    )
    return SimpleNamespace(
        conversation_id=conversation_id,
        persistence_root=persistence_root,
        workspace=workspace,
        agent=agent,
        event_ids=[system_event.id, message_event.id],
        system_path=system_path,
        message_path=message_path,
    )


def test_run_initializes_role_plugin_and_secrets_before_the_first_message(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class FakeConversation:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.plugins = kwargs["plugins"]
            self.id = kwargs["conversation_id"]
            captured["secrets"] = kwargs["secrets"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            captured["llm_timeout"] = agent.llm.timeout
            captured["llm_num_retries"] = agent.llm.num_retries
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
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(
        tmp_path,
        command_secrets={
            "WANDB_API_KEY": "wandb-key",
            "MLXFAST_API_TOKEN": "mlxfast-key",
            "YUKON_API_TOKEN": "yukon-key",
        },
    )

    assert run_openhands("first task", config) == 0
    assert captured["prompt"] == "first task"
    assert captured["role"] == (
        "# Senpai harness\n\nharness instructions\n\n"
        "# Senpai role\n\nadvisor role\n\n"
        "# program.md - program.md\n\nTest programme.\n\n"
        "# Authoritative launch context\n\nTest launch policy.\n\n"
        "# Live controller invariant\n\n"
        f"{runner.live_controller_invariant(config)}\n"
    )
    assert captured["plugin"] == str(PLUGIN_DIR)
    assert captured["secrets"] == {
        "WANDB_API_KEY": "wandb-key",
        "MLXFAST_API_TOKEN": "mlxfast-key",
        "YUKON_API_TOKEN": "yukon-key",
    }
    assert captured["conversation_id_env"] == config.conversation_id.hex
    assert captured["delete_on_close"] is False
    assert captured["llm_timeout"] == 900
    assert captured["llm_num_retries"] == 1
    assert captured["closed"] is True


def test_disabled_tool_migration_preserves_history_and_job_factory_resume(tmp_path):
    conversation_id = runner.uuid.uuid4()
    persistence_root = tmp_path / "state"
    persistence_dir = persistence_root / conversation_id.hex
    workspace = LocalWorkspace(working_dir=tmp_path / "workspace")
    llm = LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key"))
    job_tool = Tool(
        name="senpai_training",
        params={"state_dir": str(tmp_path / "jobs")},
    )
    old_agent = Agent(
        llm=llm,
        tools=[job_tool, Tool(name="delegate_agent"), Tool(name="think")],
    )
    state = ConversationState.create(
        id=conversation_id,
        agent=old_agent,
        workspace=workspace,
        persistence_dir=str(persistence_dir),
    )
    message = state.append_event(
        runner.MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="preserve this research history")],
            ),
        )
    )
    event_files = tuple(
        (p, p.read_bytes()) for p in (persistence_dir / "events").iterdir()
    )
    before = json.loads((persistence_dir / "base_state.json").read_text())

    assert migrate_persisted_disabled_tools(persistence_root, conversation_id) == {
        "delegate_agent",
        "think",
    }
    assert not migrate_persisted_disabled_tools(persistence_root, conversation_id)

    persisted = json.loads((persistence_dir / "base_state.json").read_text())
    assert [tool["name"] for tool in persisted["agent"]["tools"]] == ["senpai_training"]
    assert persisted["agent"]["tools"][0] == job_tool.model_dump(mode="json")
    assert "ThinkTool" not in persisted["agent"]["include_default_tools"]
    assert {key: value for key, value in persisted.items() if key != "agent"} == {
        key: value for key, value in before.items() if key != "agent"
    }
    assert {
        key: value
        for key, value in persisted["agent"].items()
        if key not in {"tools", "include_default_tools"}
    } == {
        key: value
        for key, value in before["agent"].items()
        if key not in {"tools", "include_default_tools"}
    }
    assert all(path.read_bytes() == content for path, content in event_files)

    current_agent = without_legacy_think(Agent(llm=llm, tools=[job_tool]))
    conversation = runner.LocalConversation(
        agent=current_agent,
        workspace=workspace,
        persistence_dir=persistence_root,
        conversation_id=conversation_id,
        visualizer=None,
    )
    try:
        assert [event.id for event in conversation.state.events] == [message.id]
        assert [tool.name for tool in conversation.state.agent.tools] == [
            "senpai_training"
        ]
        assert conversation.state.agent.include_default_tools == ["FinishTool"]
    finally:
        conversation.close()


def test_retired_tool_snapshots_are_removed_before_history_deserialization(tmp_path):
    persisted = _persisted_conversation_with_retired_tool_snapshots(tmp_path)
    original_system = json.loads(persisted.system_path.read_text(encoding="utf-8"))
    original_message = persisted.message_path.read_bytes()

    assert migrate_persisted_retired_tool_definitions(
        persisted.persistence_root, persisted.conversation_id
    ) == len(_RETIRED_TOOL_TYPES)
    assert migrate_persisted_disabled_tools(
        persisted.persistence_root, persisted.conversation_id
    ) == {"delegate_agent", "think"}
    assert (
        migrate_persisted_retired_tool_definitions(
            persisted.persistence_root, persisted.conversation_id
        )
        == 0
    )

    migrated_system = json.loads(persisted.system_path.read_text(encoding="utf-8"))
    expected_system = original_system
    expected_system["tools"] = [
        tool
        for tool in expected_system["tools"]
        if tool["kind"] not in {item[0] for item in _RETIRED_TOOL_TYPES}
    ]
    assert migrated_system == expected_system
    assert persisted.message_path.read_bytes() == original_message

    conversation = runner.LocalConversation(
        agent=persisted.agent,
        workspace=persisted.workspace,
        persistence_dir=persisted.persistence_root,
        conversation_id=persisted.conversation_id,
        visualizer=None,
    )
    try:
        assert [event.id for event in conversation.state.events] == persisted.event_ids
        assert conversation.state.events[0].system_prompt.text == (
            "preserve the historical system prompt"
        )
        assert conversation.state.events[1].llm_message.content[0].text == (
            "preserve this historical message"
        )
    finally:
        conversation.close()


def test_retired_tool_snapshot_migration_is_narrow_and_atomic(tmp_path, monkeypatch):
    persisted = _persisted_conversation_with_retired_tool_snapshots(tmp_path)
    payload = json.loads(persisted.system_path.read_text(encoding="utf-8"))
    payload["tools"] = [
        {
            **payload["tools"][0],
            "kind": "UnknownRetiredTool",
            "meta": {"historical": "must remain visible"},
        },
        payload["tools"][1],
    ]
    persisted.system_path.write_text(json.dumps(payload), encoding="utf-8")
    original = persisted.system_path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic replacement failure")

    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated atomic replacement failure"):
            migrate_persisted_retired_tool_definitions(
                persisted.persistence_root, persisted.conversation_id
            )
    assert persisted.system_path.read_bytes() == original
    assert not tuple(
        persisted.system_path.parent.glob(f".{persisted.system_path.name}.*.tmp")
    )

    assert (
        migrate_persisted_retired_tool_definitions(
            persisted.persistence_root, persisted.conversation_id
        )
        == 1
    )
    migrated = json.loads(persisted.system_path.read_text(encoding="utf-8"))
    assert migrated["tools"] == [payload["tools"][0]]


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
        inbox_max_turn_age_seconds=10_000,
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
        inbox_max_turn_age_seconds=10_000,
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
        inbox_max_turn_age_seconds=10_000,
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


def test_child_requests_ephemeral_storage_and_emits_its_terminal_report(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = {}

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.IDLE,
                view=SimpleNamespace(events=[]),
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            self.state.view.events.append(
                runner.MessageEvent(
                    source="agent",
                    llm_message=Message(
                        role="assistant",
                        content=[TextContent(text="bounded child report")],
                    ),
                )
            )
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "WEAVE_PROJECT", "wandb-applied-ai-team/senpai-v1")
    config = runtime_config(tmp_path, child=True)

    assert run_openhands("child task", config) == 0

    records = capsys.readouterr().out.splitlines()
    result = json.loads(
        next(
            line.removeprefix("OPENHANDS_RESULT ")
            for line in records
            if line.startswith("OPENHANDS_RESULT ")
        )
    )
    run = json.loads(
        next(
            line.removeprefix("OPENHANDS_RUN ")
            for line in records
            if line.startswith("OPENHANDS_RUN ")
        )
    )
    assert captured == {"delete_on_close": True, "closed": True}
    assert result["result"] == "bounded child report"
    assert run["weave_url"] == (
        "https://wandb.ai/wandb-applied-ai-team/senpai-v1/"
        f"weave/agents/conversations/{config.conversation_id}"
    )


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
            captured["tool_concurrency_limit"] = kwargs["agent"].tool_concurrency_limit
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


def test_named_agent_compaction_uses_its_own_provider(tmp_path, monkeypatch):
    captured = {}
    named_llm = LLM(
        model="wandb/zai-org/GLM-5.2",
        api_key=SecretStr("test-key"),
        reasoning_effort="max",
        **runner.model_runtime_configuration(
            "wandb/zai-org/GLM-5.2",
            "max",
            wandb_entity="research-team",
            wandb_project="mlxfast",
        ),
    )

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["condenser"] = kwargs["agent"].condenser
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            pass

        def close(self):
            pass

    isolate_agent_discovery(monkeypatch, runner)
    monkeypatch.setattr(runner, "find_named_agent", lambda *_: object())
    monkeypatch.setattr(runner, "resolve_agent_skills", lambda *_: [])
    monkeypatch.setattr(
        runner,
        "without_eager_skill_discovery",
        lambda definition: definition,
    )
    monkeypatch.setattr(
        runner,
        "agent_definition_to_factory",
        lambda *_args, **_kwargs: (
            lambda _parent_llm: Agent(
                llm=named_llm,
                tools=[],
            )
        ),
    )
    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)

    config = runtime_config(
        tmp_path,
        agent_name="explore",
        local_condenser_max_events=180,
    )
    assert run_openhands("named task", config) == 0
    assert captured["condenser"].max_size == 180


def test_github_tokens_never_reach_the_agent_environment(
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
            observed.append(runner.os.environ.get("GITHUB_TOKEN"))

        async def arun(self):
            observed.append(runner.os.environ.get("GITHUB_TOKEN"))

        def close(self):
            pass

    monkeypatch.setenv("GITHUB_TOKEN", "stale-env-secret")
    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    isolate_agent_discovery(monkeypatch, runner)

    assert run_openhands("task", runtime_config(tmp_path, role="student")) == 0
    assert observed == [None, None]
    assert "GITHUB_TOKEN" not in runner.os.environ


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
    monkeypatch.setattr(
        runner, "clear_github_credentials", lambda: cleared.append(True)
    )
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

    with pytest.raises(SystemExit) as raised, graceful_interrupts(conversation):
        installed[signal.SIGTERM](signal.SIGTERM, None)
        calls.append("handler-returned")

    assert raised.value.code == 128 + signal.SIGTERM
    assert "interrupt" in calls
    assert "handler-returned" in calls
    assert calls[-2:] == [
        (signal.SIGTERM, previous[signal.SIGTERM]),
        (signal.SIGINT, previous[signal.SIGINT]),
    ]


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
