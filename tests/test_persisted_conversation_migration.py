import json
import uuid
from pathlib import Path

import pytest
from openhands.sdk import LLM, Agent, Tool
from openhands.sdk.conversation import ConversationState
from openhands.sdk.event import ActionEvent, ObservationEvent, SystemPromptEvent
from openhands.sdk.llm import MessageToolCall, TextContent
from openhands.sdk.tool import FinishTool
from openhands.sdk.workspace import LocalWorkspace
from pydantic import SecretStr

import senpai_agent.persisted_conversation_migration as migration_module
from senpai_agent.jobs import JobState, JobSupervisor
from senpai_agent.monitor import MetricGate, MetricMonitorSpec
from senpai_agent.openhands_runner import LocalConversation, without_legacy_think
from senpai_agent.persisted_conversation_migration import (
    migrate_persisted_conversation,
    migrate_persisted_job_state,
)
from senpai_agent.state import job_state_dir
from senpai_agent.tools import (
    CancelJobAction,
    GetJobStatusAction,
    JobResultObservation,
    JobSpec,
    MonitorJobAction,
    MonitorJobObservation,
    RunJobAction,
)


_RETIRED_TOOLS = {
    "run_job": ("run_training", "RunTrainingAction"),
    "get_job_status": ("get_training_status", "GetTrainingStatusAction"),
    "cancel_job": ("cancel_training", "CancelTrainingAction"),
    "monitor_job": ("monitor_training", "MonitorTrainingAction"),
}
_RETIRED_DEFINITIONS = (
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


def _append_tool_pair(state, tool_name, action, observation) -> tuple[str, str]:
    call_id = f"call-{uuid.uuid4()}"
    action_event = state.append_event(
        ActionEvent(
            thought=[],
            action=action,
            tool_name=tool_name,
            tool_call_id=call_id,
            tool_call=MessageToolCall(
                id=call_id,
                name=tool_name,
                arguments=json.dumps(action.model_dump(mode="json")),
                origin="completion",
            ),
            llm_response_id=f"response-{call_id}",
        )
    )
    observation_event = state.append_event(
        ObservationEvent(
            tool_name=tool_name,
            tool_call_id=call_id,
            action_id=action_event.id,
            observation=observation,
        )
    )
    return action_event.id, observation_event.id


def _rewrite_as_retired_event(payload: dict[str, object]) -> None:
    current_name = payload.get("tool_name")
    if current_name not in _RETIRED_TOOLS:
        return
    retired_name, retired_kind = _RETIRED_TOOLS[current_name]
    payload["tool_name"] = retired_name
    if payload["kind"] == "ActionEvent":
        action = payload["action"]
        tool_call = payload["tool_call"]
        assert isinstance(action, dict) and isinstance(tool_call, dict)
        action["kind"] = retired_kind
        _rewrite_action_fields_as_retired(action)
        tool_call["name"] = retired_name
        arguments = json.loads(str(tool_call["arguments"]))
        arguments["kind"] = retired_kind
        _rewrite_action_fields_as_retired(arguments)
        tool_call["arguments"] = json.dumps(arguments)
        return

    observation = payload["observation"]
    assert isinstance(observation, dict)
    if current_name == "monitor_job":
        observation["kind"] = "MonitorTrainingObservation"
    else:
        observation["kind"] = "TrainingResultObservation"
    observation["training_id"] = observation.pop("job_id")


def _rewrite_action_fields_as_retired(payload: dict[str, object]) -> None:
    kind = payload.get("kind")
    if kind == "RunTrainingAction":
        spec = payload.get("spec")
        if isinstance(spec, dict):
            spec.pop("secret_env", None)
        return
    payload["training_id"] = payload.pop("job_id")
    if kind != "MonitorTrainingAction":
        return
    policies = payload.pop("metrics")
    assert isinstance(policies, list) and len(policies) == 1
    policy = policies[0]
    assert isinstance(policy, dict)
    payload.update(policy)


def _event_paths(events_dir: Path) -> dict[str, Path]:
    return {
        str(payload["id"]): path
        for path in events_dir.glob("*.json")
        if isinstance(
            payload := json.loads(path.read_text(encoding="utf-8")),
            dict,
        )
    }


def test_retired_tool_events_migrate_to_current_schemas_and_resume(tmp_path: Path):
    conversation_id = uuid.uuid4()
    persistence_root = tmp_path / "state"
    conversation_dir = persistence_root / conversation_id.hex
    workspace = LocalWorkspace(working_dir=tmp_path / "workspace")
    llm = LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key"))
    state = ConversationState.create(
        id=conversation_id,
        agent=Agent(llm=llm, tools=[Tool(name="senpai_training")]),
        workspace=workspace,
        persistence_dir=str(conversation_dir),
    )
    system_event = state.append_event(
        SystemPromptEvent(
            system_prompt=TextContent(text="preserve the historical system prompt"),
            tools=[FinishTool.create()[0]],
        )
    )

    job_id = "job-17"
    metric = MetricMonitorSpec(
        metric="validation_loss",
        direction="min",
        gates=(MetricGate(operator="lte", threshold=0.4),),
        stale_after_seconds=300,
    )
    pairs = [
        _append_tool_pair(
            state,
            "run_job",
            RunJobAction(
                spec=JobSpec(
                    argv=("python", "train.py"),
                    cwd=tmp_path,
                    timeout_seconds=600,
                )
            ),
            JobResultObservation(
                job_id=job_id,
                state=JobState.RUNNING,
                elapsed_seconds=4,
                log_path="/state/job-17.log",
            ),
        ),
        _append_tool_pair(
            state,
            "get_job_status",
            GetJobStatusAction(job_id=job_id),
            JobResultObservation(
                job_id=job_id,
                state=JobState.FINISHED,
                exit_code=0,
                elapsed_seconds=12,
                log_path="/state/job-17.log",
            ),
        ),
        _append_tool_pair(
            state,
            "cancel_job",
            CancelJobAction(job_id=job_id),
            JobResultObservation(
                job_id=job_id,
                state=JobState.CANCELLED,
                elapsed_seconds=12,
                log_path="/state/job-17.log",
            ),
        ),
        _append_tool_pair(
            state,
            "monitor_job",
            MonitorJobAction(job_id=job_id, metrics=(metric,)),
            MonitorJobObservation(
                job_id=job_id,
                conversation_id=str(conversation_id),
            ),
        ),
    ]
    old_schema_pair = _append_tool_pair(
        state,
        "monitor_job",
        MonitorJobAction(job_id=job_id, metrics=(metric,)),
        MonitorJobObservation(
            job_id=job_id,
            conversation_id=str(conversation_id),
        ),
    )

    paths = _event_paths(conversation_dir / "events")
    system_path = paths[system_event.id]
    system_payload = json.loads(system_path.read_text(encoding="utf-8"))
    system_payload["tools"] = [
        {
            "description": f"Historical {title} description.",
            "action_type": action,
            "observation_type": observation,
            "annotations": None,
            "kind": kind,
            "title": title,
        }
        for kind, title, action, observation in _RETIRED_DEFINITIONS
    ] + system_payload["tools"]
    system_path.write_text(json.dumps(system_payload), encoding="utf-8")

    for action_id, observation_id in pairs:
        for event_id in (action_id, observation_id):
            path = paths[event_id]
            payload = json.loads(path.read_text(encoding="utf-8"))
            _rewrite_as_retired_event(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")

    old_monitor_path = paths[old_schema_pair[0]]
    old_monitor_payload = json.loads(old_monitor_path.read_text(encoding="utf-8"))
    action = old_monitor_payload["action"]
    tool_call = old_monitor_payload["tool_call"]
    assert isinstance(action, dict) and isinstance(tool_call, dict)
    policy = action.pop("metrics")[0]
    action["wandb_metric"] = policy["metric"]
    action["direction"] = policy["direction"]
    action["gates"] = policy["gates"]
    action["stale_after_seconds"] = policy["stale_after_seconds"]
    arguments = json.loads(str(tool_call["arguments"]))
    policy = arguments.pop("metrics")[0]
    arguments["wandb_metric"] = policy["metric"]
    arguments["direction"] = policy["direction"]
    arguments["gates"] = policy["gates"]
    arguments["stale_after_seconds"] = policy["stale_after_seconds"]
    tool_call["arguments"] = json.dumps(arguments)
    old_monitor_path.write_text(json.dumps(old_monitor_payload), encoding="utf-8")

    originals = {
        path.relative_to(conversation_dir): path.read_bytes()
        for path in [
            conversation_dir / "base_state.json",
            *(conversation_dir / "events").glob("*.json"),
        ]
    }
    result = migrate_persisted_conversation(persistence_root, conversation_id)

    assert result.base_state_rewritten
    assert result.events_rewritten == 10
    assert result.tool_definitions_removed == len(_RETIRED_DEFINITIONS)
    assert not migrate_persisted_conversation(
        persistence_root,
        conversation_id,
    ).changed
    for relative, content in originals.items():
        backup = conversation_dir / ".pre-job-api" / relative
        if (conversation_dir / relative).read_bytes() != content:
            assert backup.read_bytes() == content

    current_payloads = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            conversation_dir / "base_state.json",
            *(conversation_dir / "events").glob("*.json"),
        ]
    )
    retired_literals = {
        "senpai_training",
        "TrainingResultObservation",
        "MonitorTrainingObservation",
        *(definition[0] for definition in _RETIRED_DEFINITIONS),
    }
    retired_literals.update(
        literal
        for names in _RETIRED_TOOLS.values()
        for literal in names
    )
    assert not any(name in current_payloads for name in retired_literals)

    current_agent = without_legacy_think(
        Agent(llm=llm, tools=[Tool(name="senpai_jobs")])
    )
    conversation = LocalConversation(
        agent=current_agent,
        workspace=workspace,
        persistence_dir=persistence_root,
        conversation_id=conversation_id,
        visualizer=None,
    )
    try:
        actions = [
            event
            for event in conversation.state.events
            if isinstance(event, ActionEvent)
        ]
        observations = [
            event
            for event in conversation.state.events
            if isinstance(event, ObservationEvent)
        ]
        assert [type(event.action) for event in actions] == [
            RunJobAction,
            GetJobStatusAction,
            CancelJobAction,
            MonitorJobAction,
            MonitorJobAction,
        ]
        assert all(
            event.tool_name in _RETIRED_TOOLS for event in actions + observations
        )
        assert actions[3].action.metrics == (metric,)
        assert actions[4].action.metrics == (metric,)
        assert isinstance(observations[0].observation, JobResultObservation)
        assert observations[0].observation.state is JobState.RUNNING
        assert isinstance(observations[3].observation, MonitorJobObservation)
    finally:
        conversation.close()


def test_conversation_migration_keeps_original_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    conversation_id = uuid.uuid4()
    conversation_dir = tmp_path / conversation_id.hex
    workspace = LocalWorkspace(working_dir=tmp_path / "workspace")
    state = ConversationState.create(
        id=conversation_id,
        agent=Agent(
            llm=LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key")),
            tools=[Tool(name="senpai_training")],
        ),
        workspace=workspace,
        persistence_dir=str(conversation_dir),
    )
    del state
    base_state = conversation_dir / "base_state.json"
    original = base_state.read_bytes()
    replace = migration_module.os.replace

    def fail_active_replace(source, destination):
        if Path(destination) == base_state:
            raise OSError("simulated atomic replacement failure")
        replace(source, destination)

    monkeypatch.setattr(migration_module.os, "replace", fail_active_replace)
    with pytest.raises(OSError, match="simulated atomic replacement failure"):
        migrate_persisted_conversation(tmp_path, conversation_id)

    assert base_state.read_bytes() == original
    assert (
        conversation_dir / ".pre-job-api" / "base_state.json"
    ).read_bytes() == original
    assert not tuple(conversation_dir.glob(".*.tmp"))


def test_previous_process_records_are_copied_and_reconciled_in_job_state(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    previous_dir = tmp_path / "training"
    previous_dir.mkdir()
    job_id = uuid.uuid4()
    source_log = previous_dir / f"{job_id}.log"
    source_log.write_text("preserve this process log", encoding="utf-8")
    source_record = previous_dir / f"{job_id}.json"
    source_record.write_text(
        json.dumps(
            {
                "training_id": str(job_id),
                "state": "running",
                "pid": None,
                "process_group_id": None,
                "process_start_time": None,
                "exit_code": None,
                "elapsed_seconds": 3,
                "log_path": str(source_log),
                "wandb_run_ids": [],
                "error_tail": "",
                "workspace_access": "mutable",
            }
        ),
        encoding="utf-8",
    )
    previous_monitor = previous_dir / "monitors.sqlite3"
    previous_monitor.write_bytes(b"retired monitor schema")
    original_record = source_record.read_bytes()

    result = migrate_persisted_job_state(tmp_path)

    assert result.records == 1
    assert result.previous_state_preserved
    assert source_record.read_bytes() == original_record
    assert source_log.read_text(encoding="utf-8") == "preserve this process log"
    assert previous_monitor.read_bytes() == b"retired monitor schema"
    jobs_dir = job_state_dir(tmp_path)
    migrated = json.loads((jobs_dir / source_record.name).read_text())
    assert migrated["job_id"] == str(job_id)
    assert "training_id" not in migrated
    assert migrated["log_path"] == str((jobs_dir / source_log.name).resolve())
    assert (jobs_dir / source_log.name).read_bytes() == source_log.read_bytes()
    assert json.loads(
        (jobs_dir / ".previous-job-state-imported.json").read_text()
    )["monitor_state_imported"] is False
    assert not migrate_persisted_job_state(tmp_path).changed

    supervisor = JobSupervisor(workspace=workspace, state_dir=jobs_dir)
    try:
        assert supervisor.get_job_status(str(job_id)).state is JobState.CANCELLED
    finally:
        supervisor.close()
