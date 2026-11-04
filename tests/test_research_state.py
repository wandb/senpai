"""The compact research state survives compaction and reaches the advisor."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import MessageToolCall, TextContent
from openhands.sdk.llm.message import AnthropicCompactionBlock
from openhands.sdk.tool.builtins import FinishAction
from openhands_support import isolate_agent_discovery, runtime_config

import senpai_agent.openhands_runner as runner
from senpai_agent.inbox import QUEUE_PRIORITY, STEERING_PRIORITIES
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.PROMPTS import COMPACTION_SUMMARY_PROMPT
from senpai_agent.research_state import (
    MAX_RESEARCH_STATE_BYTES,
    CurrentBest,
    LaneInFlight,
    ResearchStateContent,
    UpdateResearchStateAction,
    UpdateResearchStateTool,
    compaction_event,
    load_research_state,
    render_research_state,
    research_state_path,
    save_research_state,
)

STAMP = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"


def content(**updates) -> ResearchStateContent:
    values = {
        "current_best": [
            CurrentBest(
                what="advisor branch after PR #185",
                metric="primary score 2.4472 on the held-out benchmark",
                identifiers={
                    "pr": "#185",
                    "commit": "f" * 40,
                    "wandb_run": "run-8b7803d1",
                },
            )
        ],
        "lanes_in_flight": [
            LaneInFlight(
                lane="fused-finalize",
                owner="fern",
                pr="#201",
                deadline="2026-11-01T20:00:00Z",
                kill_criterion="step cost above +1.0 us after two builds",
                status="gating",
            )
        ],
        "bars": {"merge": "beats 2.4472 by more than two standard errors"},
        "budget": {"daily evaluation checks": "3 used, 17 remaining"},
        "operator_rules": ["Judge structural lanes on prototypes, not on twins."],
        "open_questions": ["Does the clock gain survive the judge's replays?"],
        "recent_decisions": [
            "2026-11-01T02:03Z: keep fern on fused-finalize until the gate resolves."
        ],
    }
    values.update(updates)
    return ResearchStateContent(**values)


def compaction_action(event_id: str = "evt-1") -> ActionEvent:
    action = FinishAction(message="Continuing.")
    return ActionEvent(
        thought=[TextContent(text="Context was compacted by the server.")],
        action=action,
        tool_name="finish",
        tool_call_id=f"call-{event_id}",
        tool_call=MessageToolCall(
            id=f"call-{event_id}",
            name="finish",
            arguments=json.dumps(action.model_dump(mode="json")),
            origin="completion",
        ),
        llm_response_id=f"response-{event_id}",
        anthropic_compaction_blocks=[AnthropicCompactionBlock(content="summary")],
    )


def test_update_research_state_tool_round_trips_through_a_bounded_private_file(
    tmp_path: Path,
):
    tool = UpdateResearchStateTool.create(state_dir=tmp_path / "state")[0]
    path = research_state_path(tmp_path / "state")

    observation = tool(UpdateResearchStateAction(state=content()))

    assert observation.path == str(path)
    assert re.fullmatch(STAMP, observation.updated_at)
    assert path.stat().st_mode & 0o777 == 0o600
    loaded = load_research_state(path)
    assert loaded is not None
    assert ResearchStateContent(**loaded.model_dump(exclude={"updated_at"})) == content()
    assert loaded.updated_at == observation.updated_at
    assert observation.bytes == len(path.read_bytes())
    assert json.loads(path.read_text()) == json.loads(render_research_state(loaded))

    tool(UpdateResearchStateAction(state=content(lanes_in_flight=[])))

    replaced = load_research_state(path)
    assert replaced is not None and replaced.lanes_in_flight == []
    assert tool.name == "update_research_state"
    assert "re-sends this file verbatim after every context compaction" in tool.description


def test_oversized_research_state_is_rejected_and_leaves_the_file_untouched(
    tmp_path: Path,
):
    path = research_state_path(tmp_path)
    saved = save_research_state(path, content())
    bloated = content(recent_decisions=["x" * (MAX_RESEARCH_STATE_BYTES + 1)])

    with pytest.raises(ValueError, match=str(MAX_RESEARCH_STATE_BYTES)):
        save_research_state(path, bloated)

    assert load_research_state(path) == saved
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


def test_compaction_event_resends_the_saved_state_verbatim_on_both_delivery_paths(
    tmp_path: Path,
):
    saved = save_research_state(
        research_state_path(tmp_path),
        content(),
        now=datetime(2026, 11, 1, 2, 3, tzinfo=UTC),
    )

    event = compaction_event(
        tmp_path,
        trigger="anthropic",
        event_id="evt-7",
        now=datetime(2026, 11, 1, 2, 17, 5, tzinfo=UTC),
    )

    assert event.kind == "context_compacted"
    assert event.dedupe_key == "context_compacted:evt-7"
    controller_prompt = ControllerEvent(
        kind=event.kind, dedupe_key=event.dedupe_key, payload=event.payload,
    ).to_prompt()
    assert event.to_inbox_message() == controller_prompt
    assert "compacted at 2026-11-01T02:17:05Z (anthropic)" in controller_prompt
    assert "saved with `update_research_state` at 2026-11-01T02:03:00Z" in controller_prompt
    assert render_research_state(saved).strip() in controller_prompt
    assert "#185" in controller_prompt and "run-8b7803d1" in controller_prompt


def test_compaction_without_a_saved_state_asks_for_one(tmp_path: Path):
    event = compaction_event(tmp_path, trigger="condenser", event_id="evt-8")

    prompt = event.to_inbox_message()

    assert event.payload["research_state"] is None
    assert "No research state has been saved yet" in prompt
    assert "`update_research_state`" in prompt
    assert "(condenser)" in prompt


def test_compacted_state_is_delivered_at_the_next_safe_boundary():
    assert STEERING_PRIORITIES["context_compacted"] == QUEUE_PRIORITY


@pytest.mark.parametrize(
    ("event", "trigger"),
    [
        (compaction_action(), "anthropic"),
        (
            Condensation(forgotten_event_ids=set(), llm_response_id="response-2"),
            "condenser",
        ),
    ],
    ids=["anthropic-block", "condenser"],
)
def test_runner_queues_the_state_for_the_advisor_after_a_compaction(
    tmp_path: Path, event, trigger,
):
    config = runtime_config(tmp_path, role="advisor")
    save_research_state(research_state_path(config.state_dir), content())

    assert runner.announce_compaction(config, event) is True

    with LocalEventStore(runner.local_event_db_path(config)) as store:
        pending = store.pending()
    assert [item.kind for item in pending] == ["context_compacted"]
    assert pending[0].payload["trigger"] == trigger
    assert pending[0].payload["research_state"]["current_best"][0]["identifiers"]["pr"] == "#185"


@pytest.mark.parametrize(
    ("updates", "event"),
    [
        ({}, None),
        ({"role": "student"}, "compaction"),
        ({"child": True}, "compaction"),
    ],
    ids=["ordinary-action", "student", "child"],
)
def test_runner_ignores_ordinary_events_and_other_roles(tmp_path: Path, updates, event):
    config = runtime_config(tmp_path, **updates)
    ordinary = compaction_action().model_copy(update={"anthropic_compaction_blocks": []})

    assert runner.announce_compaction(
        config, compaction_action() if event else ordinary,
    ) is False
    assert not runner.local_event_db_path(config).exists()


def test_conversation_callback_announces_compactions(tmp_path, monkeypatch):
    """The observer the runner installs is the hook, not a separate thread."""

    captured = {}

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["callbacks"] = kwargs["callbacks"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.IDLE,
                view=SimpleNamespace(events=[]),
                active_branch=lambda: [],
            )

        def send_message(self, prompt, sender=None):
            pass

        async def arun(self):
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner.ConversationState, "get_unmatched_actions", lambda _events: [])
    isolate_agent_discovery(monkeypatch, runner)
    config = runtime_config(tmp_path, role="advisor")
    save_research_state(research_state_path(config.state_dir), content())

    assert runner.run_openhands("advise", config) == 0

    callback = captured["callbacks"][0]
    compaction = compaction_action("evt-9")
    callback(compaction)
    with LocalEventStore(runner.local_event_db_path(config)) as store:
        pending = store.pending()
    assert [item.dedupe_key for item in pending] == [
        f"context_compacted:{compaction.id}"
    ]
    assert pending[0].payload["research_state"]["updated_at"]


def test_compaction_summary_keeps_identifiers_and_drops_procedural_reminders():
    assert "exact identifier" in COMPACTION_SUMMARY_PROMPT
    assert "every decision with its reason" in COMPACTION_SUMMARY_PROMPT
    assert "open questions" in COMPACTION_SUMMARY_PROMPT
    assert "Drop procedural reminders" in COMPACTION_SUMMARY_PROMPT
    assert "when to check the clock" in COMPACTION_SUMMARY_PROMPT
    assert "Do not call tools while writing this summary" in COMPACTION_SUMMARY_PROMPT


def test_only_the_advisor_root_gets_the_state_tool(tmp_path: Path):
    advisor = {tool.name: tool for tool in runner.build_main_tools(
        runtime_config(tmp_path, role="advisor", student_names=("fern",)),
    )}
    student = {tool.name for tool in runner.build_main_tools(
        runtime_config(tmp_path, role="student", student_name="fern"),
    )}
    child = {tool.name for tool in runner.build_main_tools(
        runtime_config(tmp_path, role="advisor", child=True),
    )}

    assert advisor["update_research_state"].params == {
        "state_dir": str(tmp_path / "state"),
    }
    assert "update_research_state" not in student
    assert "update_research_state" not in child
