import json

import pytest
from openhands.sdk.conversation.event_store import EventLog
from openhands.sdk.event import ActionEvent, Event, ObservationEvent
from openhands.sdk.io import InMemoryFileStore
from openhands.sdk.llm import MessageToolCall

from senpai_agent.github.tools import SubmitExperimentResultAction
from senpai_agent.models import (
    AssignmentKey,
    ExperimentResult,
    MetricComparison,
    ResultStatus,
)
from senpai_agent.tools import (
    JobResultObservation,
    JobSpec,
    MonitorTrainingAction,
    RunJobAction,
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


def test_job_actions_and_observations_survive_event_log_restore():
    action = RunJobAction(
        spec=JobSpec(
            argv=("python", "evaluate.py"),
            cwd="/workspace",
            timeout_seconds=600,
        )
    )
    restored_action = round_trip(
        ActionEvent(
            thought=[],
            action=action,
            tool_name="run_job",
            tool_call_id="job-action",
            tool_call=MessageToolCall(
                id="job-action",
                name="run_job",
                arguments=json.dumps(action.model_dump(mode="json")),
                origin="completion",
            ),
            llm_response_id="job-response",
        )
    )
    restored_observation = round_trip(
        ObservationEvent(
            tool_name="run_job",
            tool_call_id="job-observation",
            action_id="job-action",
            observation=JobResultObservation(
                job_id="job-17",
                state=TrainingState.RUNNING,
                elapsed_seconds=12.5,
                log_path="/state/job-17.log",
            ),
        )
    )

    assert isinstance(restored_action, ActionEvent)
    assert isinstance(restored_action.action, RunJobAction)
    assert restored_action.action.spec.argv == ("python", "evaluate.py")
    assert isinstance(restored_observation, ObservationEvent)
    assert isinstance(restored_observation.observation, JobResultObservation)
    assert restored_observation.observation.job_id == "job-17"


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
