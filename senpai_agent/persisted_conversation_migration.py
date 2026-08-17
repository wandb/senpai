"""One-shot migration of persisted state from the retired supervision API."""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from senpai_agent.state import job_state_dir


_PREVIOUS_JOB_STATE_DIRECTORY = "training"
_PREVIOUS_JOB_FACTORY = "senpai_training"
_CURRENT_JOB_FACTORY = "senpai_jobs"
_PREVIOUS_JOB_ID = "training_id"
_DISABLED_AGENT_TOOLS = frozenset({"delegate_agent", "think"})
_RETIRED_TOOL_DEFINITION_KINDS = frozenset(
    {
        "DelegateAgentTool",
        "RunTrainingTool",
        "GetTrainingStatusTool",
        "CancelTrainingTool",
        "MonitorTrainingTool",
    }
)
_TOOL_NAME_MIGRATIONS = {
    "run_training": "run_job",
    "get_training_status": "get_job_status",
    "cancel_training": "cancel_job",
    "monitor_training": "monitor_job",
}
_ACTION_KIND_MIGRATIONS = {
    "RunTrainingAction": ("RunJobAction", "run_job"),
    "GetTrainingStatusAction": ("GetJobStatusAction", "get_job_status"),
    "CancelTrainingAction": ("CancelJobAction", "cancel_job"),
    "MonitorTrainingAction": ("MonitorJobAction", "monitor_job"),
}
_OBSERVATION_KIND_MIGRATIONS = {
    "TrainingResultObservation": "JobResultObservation",
    "MonitorTrainingObservation": "MonitorJobObservation",
}
_PREVIOUS_MONITOR_ARGUMENTS = frozenset(
    {
        "metric",
        "wandb_metric",
        "direction",
        "gates",
        "stale_after_seconds",
        "notify_on_status",
    }
)
_CONVERSATION_BACKUP_DIRECTORY = ".pre-job-api"
_JOB_STATE_MIGRATION_MARKER = ".previous-job-state-imported.json"


@dataclass(frozen=True, slots=True)
class ConversationMigration:
    base_state_rewritten: bool = False
    events_rewritten: int = 0
    tool_definitions_removed: int = 0

    @property
    def changed(self) -> bool:
        return self.base_state_rewritten or bool(self.events_rewritten)


@dataclass(frozen=True, slots=True)
class JobStateMigration:
    records: int = 0
    previous_state_preserved: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.records or self.previous_state_preserved)


def migrate_persisted_conversation(
    state_dir: Path,
    conversation_id: UUID,
) -> ConversationMigration:
    """Atomically rewrite one retired conversation to canonical job schemas."""

    conversation_dir = Path(state_dir) / conversation_id.hex
    base_state = conversation_dir / "base_state.json"
    base_state_rewritten = False
    if base_state.exists():
        payload = _read_json_object(base_state, "agent state")
        if _migrate_saved_agent(payload, base_state):
            _backup_and_write(conversation_dir, base_state, payload)
            base_state_rewritten = True

    events_rewritten = 0
    tool_definitions_removed = 0
    for path in sorted((conversation_dir / "events").glob("*.json")):
        payload = _read_json_object(path, "conversation event")
        changed, removed = _migrate_event(payload)
        if not changed:
            continue
        _backup_and_write(conversation_dir, path, payload)
        events_rewritten += 1
        tool_definitions_removed += removed
    return ConversationMigration(
        base_state_rewritten=base_state_rewritten,
        events_rewritten=events_rewritten,
        tool_definitions_removed=tool_definitions_removed,
    )


def migrate_persisted_job_state(state_dir: Path) -> JobStateMigration:
    """Copy previous process records into the canonical job-state directory."""

    state_dir = Path(state_dir)
    previous_dir = state_dir / _PREVIOUS_JOB_STATE_DIRECTORY
    if not previous_dir.exists():
        return JobStateMigration()
    if not previous_dir.is_dir():
        raise RuntimeError(f"persisted job state is not a directory: {previous_dir}")

    current_dir = job_state_dir(state_dir)
    current_dir.mkdir(parents=True, exist_ok=True)
    marker = current_dir / _JOB_STATE_MIGRATION_MARKER
    if marker.exists():
        return JobStateMigration()

    records = 0
    for source in sorted(previous_dir.glob("*.json")):
        try:
            job_id = UUID(source.stem)
        except ValueError:
            continue
        payload = _read_json_object(source, "job record")
        _migrate_job_record(payload, job_id, previous_dir, current_dir)
        destination = current_dir / source.name
        if destination.exists():
            if _read_json_object(destination, "job record") != payload:
                raise RuntimeError(
                    f"conflicting persisted job record at {destination}"
                )
        else:
            _write_json_atomically(
                destination,
                payload,
                mode=stat.S_IMODE(source.stat().st_mode),
            )

        source_log = previous_dir / f"{job_id}.log"
        destination_log = current_dir / source_log.name
        if source_log.exists():
            if destination_log.exists():
                if not filecmp.cmp(source_log, destination_log, shallow=False):
                    raise RuntimeError(
                        f"conflicting persisted job log at {destination_log}"
                    )
            else:
                _copy_file_atomically(source_log, destination_log)
        records += 1

    _write_json_atomically(
        marker,
        {
            "source": str(previous_dir),
            "records": records,
            "monitor_state_imported": False,
        },
        mode=0o600,
    )
    return JobStateMigration(records=records, previous_state_preserved=True)


def _migrate_saved_agent(payload: dict[str, object], path: Path) -> bool:
    saved_agent = payload.get("agent")
    if not isinstance(saved_agent, dict):
        raise TypeError(f"persisted agent state at {path} has an unknown shape")
    tools = saved_agent.get("tools")
    defaults = saved_agent.get("include_default_tools")
    if not isinstance(tools, list) or not all(
        isinstance(tool, dict) and isinstance(tool.get("name"), str) for tool in tools
    ):
        raise RuntimeError(f"persisted agent tools at {path} have an unknown shape")
    if defaults is not None and (
        not isinstance(defaults, list)
        or not all(isinstance(name, str) for name in defaults)
    ):
        raise RuntimeError(f"persisted default tools at {path} have an unknown shape")

    migrated_tools = []
    for tool in tools:
        if tool["name"] in _DISABLED_AGENT_TOOLS:
            continue
        if tool["name"] == _PREVIOUS_JOB_FACTORY:
            tool = {**tool, "name": _CURRENT_JOB_FACTORY}
            params = tool.get("params")
            if isinstance(params, dict) and "state_dir" in params:
                tool["params"] = {
                    **params,
                    "state_dir": str(job_state_dir(path.parent.parent)),
                }
        migrated_tools.append(tool)

    migrated_defaults = [
        name
        for name in (defaults if defaults is not None else ["FinishTool", "ThinkTool"])
        if name != "ThinkTool"
    ]
    if migrated_tools == tools and migrated_defaults == defaults:
        return False
    saved_agent["tools"] = migrated_tools
    saved_agent["include_default_tools"] = migrated_defaults
    return True


def _migrate_event(payload: dict[str, object]) -> tuple[bool, int]:
    kind = payload.get("kind")
    if kind == "SystemPromptEvent":
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return False, 0
        migrated = [
            tool
            for tool in tools
            if not (
                isinstance(tool, dict)
                and tool.get("kind") in _RETIRED_TOOL_DEFINITION_KINDS
            )
        ]
        removed = len(tools) - len(migrated)
        if removed:
            payload["tools"] = migrated
        return bool(removed), removed
    if kind == "ActionEvent":
        return _migrate_action_event(payload), 0
    if kind == "ObservationEvent":
        return _migrate_observation_event(payload), 0
    return False, 0


def _migrate_action_event(payload: dict[str, object]) -> bool:
    changed = False
    action = payload.get("action")
    action_kind = action.get("kind") if isinstance(action, dict) else None
    kind_migration = _ACTION_KIND_MIGRATIONS.get(action_kind)
    previous_tool_name = payload.get("tool_name")
    if not isinstance(previous_tool_name, str):
        previous_tool_name = None
    current_tool_name = _TOOL_NAME_MIGRATIONS.get(previous_tool_name or "")
    if kind_migration is not None:
        current_kind, kind_tool_name = kind_migration
        assert isinstance(action, dict)
        action["kind"] = current_kind
        current_tool_name = kind_tool_name
        changed = True

    if isinstance(action, dict):
        changed |= _migrate_action_fields(action, previous_tool_name)
    if current_tool_name is not None and payload.get("tool_name") != current_tool_name:
        payload["tool_name"] = current_tool_name
        changed = True

    tool_call = payload.get("tool_call")
    if isinstance(tool_call, dict):
        call_name = tool_call.get("name")
        call_migration = (
            _TOOL_NAME_MIGRATIONS.get(call_name) if isinstance(call_name, str) else None
        )
        if current_tool_name is None:
            current_tool_name = call_migration
        if current_tool_name is not None and call_name != current_tool_name:
            tool_call["name"] = current_tool_name
            changed = True
        arguments = tool_call.get("arguments")
        migrated_arguments = _migrate_tool_call_arguments(
            arguments,
            previous_tool_name or (call_name if isinstance(call_name, str) else None),
        )
        if migrated_arguments != arguments:
            tool_call["arguments"] = migrated_arguments
            changed = True
    return changed


def _migrate_action_fields(
    payload: dict[str, object],
    previous_tool_name: str | None,
) -> bool:
    changed = False
    kind = payload.get("kind")
    if kind in {"GetJobStatusAction", "CancelJobAction", "MonitorJobAction"}:
        changed |= _rename_job_id(payload)
    if previous_tool_name in {
        "get_training_status",
        "cancel_training",
        "monitor_training",
    }:
        changed |= _rename_job_id(payload)
    if kind == "MonitorJobAction" or previous_tool_name in {
        "monitor_training",
        "monitor_job",
    }:
        changed |= _migrate_monitor_arguments(payload)
    return changed


def _migrate_tool_call_arguments(
    arguments: object,
    previous_tool_name: str | None,
) -> object:
    if not isinstance(arguments, str):
        return arguments
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not isinstance(payload, dict):
        return arguments
    changed = _migrate_action_fields(payload, previous_tool_name)
    kind_migration = _ACTION_KIND_MIGRATIONS.get(payload.get("kind"))
    if kind_migration is not None:
        payload["kind"] = kind_migration[0]
        changed = True
    return (
        json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if changed
        else arguments
    )


def _rename_job_id(payload: dict[str, object]) -> bool:
    if _PREVIOUS_JOB_ID not in payload:
        return False
    previous = payload.pop(_PREVIOUS_JOB_ID)
    current = payload.get("job_id")
    if current is not None and current != previous:
        raise ValueError("persisted action contains conflicting job identifiers")
    payload["job_id"] = previous
    return True


def _migrate_monitor_arguments(payload: dict[str, object]) -> bool:
    if "metrics" in payload or not (_PREVIOUS_MONITOR_ARGUMENTS & payload.keys()):
        return False
    metric = payload.pop("wandb_metric", payload.pop("metric", None))
    direction = payload.pop("direction", None)
    gates = payload.pop("gates", [])
    stale_after_seconds = payload.pop("stale_after_seconds", 600)
    payload.pop("notify_on_status", None)
    payload["metrics"] = (
        [
            {
                "metric": metric,
                "direction": direction,
                "gates": gates,
                "stale_after_seconds": stale_after_seconds,
            }
        ]
        if metric is not None
        else []
    )
    return True


def _migrate_observation_event(payload: dict[str, object]) -> bool:
    changed = False
    previous_tool_name = payload.get("tool_name")
    if isinstance(previous_tool_name, str) and (
        current_tool_name := _TOOL_NAME_MIGRATIONS.get(previous_tool_name)
    ):
        payload["tool_name"] = current_tool_name
        changed = True

    observation = payload.get("observation")
    if not isinstance(observation, dict):
        return changed
    current_kind = _OBSERVATION_KIND_MIGRATIONS.get(observation.get("kind"))
    if current_kind is None:
        return changed
    observation["kind"] = current_kind
    changed = True
    if current_kind in {"JobResultObservation", "MonitorJobObservation"}:
        changed |= _rename_job_id(observation)
    return changed


def _migrate_job_record(
    payload: dict[str, object],
    job_id: UUID,
    previous_dir: Path,
    current_dir: Path,
) -> None:
    _rename_job_id(payload)
    stored_id = payload.get("job_id")
    if not isinstance(stored_id, str) or UUID(stored_id) != job_id:
        raise RuntimeError(f"job record ID does not match {job_id}")
    log_path = payload.get("log_path")
    previous_log = (previous_dir / f"{job_id}.log").resolve()
    if isinstance(log_path, str) and Path(log_path).resolve() == previous_log:
        payload["log_path"] = str((current_dir / f"{job_id}.log").resolve())


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"cannot migrate persisted {label} at {path}: {error}"
        ) from error
    return _decode_json_object(contents, f"persisted {label} at {path}")


def _decode_json_object(contents: str, label: str) -> dict[str, object]:
    try:
        payload = json.loads(contents)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"cannot decode {label}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} has an unknown shape")
    return payload


def _backup_and_write(
    conversation_dir: Path,
    path: Path,
    payload: object,
) -> None:
    backup = (
        conversation_dir
        / _CONVERSATION_BACKUP_DIRECTORY
        / path.relative_to(conversation_dir)
    )
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        _copy_file_atomically(path, backup)
    _write_json_atomically(path, payload)


def _copy_file_atomically(source: Path, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomically(
    path: Path,
    payload: object,
    *,
    mode: int | None = None,
) -> None:
    mode = mode if mode is not None else stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
