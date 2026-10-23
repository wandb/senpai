import json
from datetime import UTC, datetime, timedelta

import pytest
from openhands.sdk.conversation.event_store import EventLog
from openhands.sdk.event import ActionEvent, Event, ObservationEvent, SystemPromptEvent
from openhands.sdk.io import InMemoryFileStore
from openhands.sdk.llm import MessageToolCall

from senpai_agent.capacity_tool import (
    ClusterCapacityAction,
    ClusterCapacityObservation,
    ClusterCapacityTool,
)
from senpai_agent.cluster_capacity import (
    SNAPSHOT_ENV,
    CapacityConfig,
    summarize_capacity,
)
from senpai_agent.models import (
    AssignmentKey,
    ExperimentResult,
    MetricComparison,
    ResultStatus,
)
from senpai_agent.delegation import (
    DelegateAgentAction,
    DelegateAgentObservation,
    DelegateAgentTool,
)
from senpai_agent.github.tools import SubmitExperimentResultAction
from senpai_agent.tools import (
    MonitorTrainingAction,
    TrainingResultObservation,
)
from senpai_agent.training import TrainingState


def round_trip(event: Event) -> Event:
    store = InMemoryFileStore()
    EventLog(store).append(event)
    return EventLog(store)[0]


def experiment_result(
    primary_metric: MetricComparison | None = None,
) -> ExperimentResult:
    head_sha = "c" * 40
    return ExperimentResult(
        assignment=AssignmentKey(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            expected_head_sha=head_sha,
            student="student-one",
        ),
        status=ResultStatus.INCONCLUSIVE,
        hypothesis="Check the bounded candidate.",
        summary="The bounded comparison completed.",
        runs=(),
        primary_metric=primary_metric,
        commit_sha=head_sha,
    )


def test_legacy_monitor_actions_restore_without_the_removed_status_filter():
    action = MonitorTrainingAction(
        training_id="training-17",
        metric="validation/loss",
        direction="min",
    )
    event = ActionEvent(
        thought=[],
        action=action,
        tool_name="monitor_training",
        tool_call_id="legacy-monitor",
        tool_call=MessageToolCall(
            id="legacy-monitor",
            name="monitor_training",
            arguments=json.dumps(action.model_dump(mode="json")),
            origin="completion",
        ),
        llm_response_id="legacy-response",
    )
    persisted = event.model_dump(mode="json")
    persisted["action"]["notify_on_status"] = ["finished"]

    restored = Event.model_validate_json(json.dumps(persisted))

    assert isinstance(restored, ActionEvent)
    assert isinstance(restored.action, MonitorTrainingAction)
    assert restored.action.training_id == "training-17"
    assert "notify_on_status" not in restored.action.model_dump()


def test_legacy_delegate_agent_events_and_tool_definition_still_restore():
    action = DelegateAgentAction(
        task="Inspect the prior benchmark",
        agent="explore",
        model="fast",
        background=True,
        include_context=True,
    )
    restored_action = round_trip(
        ActionEvent(
            thought=[],
            action=action,
            tool_name="delegate_agent",
            tool_call_id="legacy-delegate",
            tool_call=MessageToolCall(
                id="legacy-delegate",
                name="delegate_agent",
                arguments=json.dumps(action.model_dump(mode="json")),
                origin="completion",
            ),
            llm_response_id="legacy-response",
        )
    )
    restored_observation = round_trip(
        ObservationEvent(
            tool_name="delegate_agent",
            tool_call_id="legacy-delegate",
            action_id=restored_action.id,
            observation=DelegateAgentObservation(
                task_id="legacy-task",
                status="dispatched",
            ),
        )
    )
    restored_system = round_trip(
        SystemPromptEvent(
            system_prompt={"text": "Legacy system prompt"},
            tools=DelegateAgentTool.create()[0:1],
        )
    )

    assert isinstance(restored_action.action, DelegateAgentAction)
    assert restored_action.action.background is True
    assert isinstance(restored_observation.observation, DelegateAgentObservation)
    assert restored_observation.observation.status == "dispatched"
    assert restored_observation.observation.to_llm_content[0].text == (
        "Subagent task legacy-task is running in the background. Its result or "
        "error will arrive as a durable local event."
    )
    assert isinstance(restored_system.tools[0], DelegateAgentTool)


def test_running_training_observation_survives_event_log_restore():
    restored = round_trip(
        ObservationEvent(
            tool_name="run_training",
            tool_call_id="call-17",
            action_id="action-17",
            observation=TrainingResultObservation(
                training_id="training-17",
                state=TrainingState.RUNNING,
                exit_code=None,
                elapsed_seconds=12.5,
                log_path="/state/training-17.log",
            ),
        )
    )

    assert isinstance(restored, ObservationEvent)
    assert isinstance(restored.observation, TrainingResultObservation)
    assert restored.observation.state is TrainingState.RUNNING
    assert restored.observation.exit_code is None


@pytest.mark.parametrize("configured", [False, True], ids=["unknown-age", "known-age"])
def test_capacity_observation_survives_event_log_restore(
    tmp_path, monkeypatch, configured
):
    monkeypatch.delenv(SNAPSHOT_ENV, raising=False)
    if configured:
        path = tmp_path / "capacity.json"
        snapshot = summarize_capacity(
            [],
            [],
            CapacityConfig(
                nodes=1, gpus_per_node=1, cpu_per_node=1, memory_gib_per_node=1
            ),
            observed_at=datetime.now(UTC) - timedelta(seconds=40),
        )
        path.write_text(snapshot.model_dump_json())
        monkeypatch.setenv(SNAPSHOT_ENV, str(path))
    observation = ClusterCapacityTool.create(None)[0](ClusterCapacityAction())
    if configured:
        assert observation.snapshot.status == "available"
        assert observation.age_seconds > 0
    else:
        assert observation.snapshot.reason == "not_configured"
        assert observation.age_seconds is None

    restored = round_trip(
        ObservationEvent(
            tool_name="get_cluster_capacity",
            tool_call_id="capacity-call",
            action_id="capacity-action",
            observation=observation,
        )
    )

    assert isinstance(restored.observation, ClusterCapacityObservation)
    assert restored.observation.snapshot == observation.snapshot
    assert restored.observation.age_seconds == observation.age_seconds
    assert restored.observation.to_llm_content == observation.to_llm_content


@pytest.mark.parametrize(
    "primary_metric",
    [
        None,
        MetricComparison(
            name="validation_loss",
            direction="minimize",
            baseline=None,
            candidate=0.4,
            delta=None,
        ),
    ],
    ids=["no-primary-metric", "nullable-comparison-values"],
)
def test_submit_result_action_survives_event_log_restore(primary_metric):
    result = experiment_result(primary_metric)
    action = SubmitExperimentResultAction(
        branch="candidate",
        remote_branch_sha_before_push="a" * 40,
        result=result,
    )
    restored = round_trip(
        ActionEvent(
            thought=[],
            action=action,
            tool_name="submit_experiment_result",
            tool_call_id="call-17",
            tool_call=MessageToolCall(
                id="call-17",
                name="submit_experiment_result",
                arguments=json.dumps(action.model_dump(mode="json")),
                origin="completion",
            ),
            llm_response_id="response-17",
        )
    )

    assert isinstance(restored, ActionEvent)
    assert isinstance(restored.action, SubmitExperimentResultAction)
    assert restored.action.result.primary_metric == primary_metric
