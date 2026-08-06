"""Role-local control surface used by the campaign supervisor.

This module runs inside an advisor or student container.  It never accepts a
host, namespace, pod, workspace, or state path from its caller; those values
come from the role's own environment.  Cross-pod code therefore cannot widen
the campaign scope by changing an argument.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import psutil
from openhands.sdk.conversation import ConversationState
from openhands.sdk.event import ActionEvent, Event, MessageEvent
from openhands.sdk.event.types import ROOT_PARENT_ID
from openhands.sdk.llm import TextContent
from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.operations import (
    ContextResetReceipt,
    ContextResetRequest,
    ContextResetRequestStore,
    ContextResetStatus,
    NudgeReceipt,
    RestartReceipt,
    RoleObservation,
    RoleTarget,
)
from senpai_agent.supervisor import WorkerLease
from senpai_agent.training import (
    TrainingInventory,
    TrainingInventoryEntry,
    TrainingResult,
    TrainingState,
    read_training_inventory,
    training_process_is_live,
    training_result_path,
)


class RoleRuntimeState(BaseModel):
    """Bounded diagnostic payload returned to the central collector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: RoleTarget
    observation: RoleObservation
    lease_deadline_seconds: float | None
    completed_turns: int | None
    running_training_count: int | None
    active_delegation_count: int | None = Field(default=None, ge=0, le=100)
    wandb_run_inventory_complete: bool | None = None
    running_wandb_run_ids: tuple[str, ...] = Field(default=(), max_length=50)
    recent_wandb_run_ids: tuple[str, ...] = Field(default=(), max_length=200)
    context_resets: tuple[ContextResetStatus, ...] = Field(default=(), max_length=20)
    cpu_percent: float | None
    memory_percent: float | None
    disk_percent: float | None
    gpu_percent: float | None
    recent_errors: tuple[str, ...] = Field(default=(), max_length=20)


class RoleResearchTailItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    kind: str = Field(min_length=1, max_length=200)
    source: str | None = Field(default=None, max_length=200)
    summary: str = Field(max_length=4_000)


class RoleResearchTail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: UUID
    observed_at: datetime
    messages: tuple[RoleResearchTailItem, ...] = Field(default=(), max_length=3)


class RoleControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: Literal[
        "observe",
        "research_tail",
        "nudge",
        "restart",
        "context_reset",
    ]
    expected_conversation_id: UUID | None = None
    control_token: str | None = None
    restart_control_token: str | None = None
    message: str | None = None
    operation_key: str | None = None
    recovery_prompt: str | None = None
    context_reset_request: ContextResetRequest | None = None


def role_target(env: Mapping[str, str] = os.environ) -> RoleTarget:
    role = env.get("SENPAI_ROLE")
    if role not in {"advisor", "student"}:
        raise RuntimeError("role control is available only inside advisor/student pods")
    return RoleTarget(
        research_tag=env["RESEARCH_TAG"],
        role=role,
        student=env.get("STUDENT_NAME") if role == "student" else None,
    )


def role_state_dir(env: Mapping[str, str] = os.environ) -> Path:
    value = env.get("SENPAI_OPENHANDS_STATE_DIR")
    if not value:
        raise RuntimeError("SENPAI_OPENHANDS_STATE_DIR is required")
    return Path(value).resolve()


def observe_role(
    env: Mapping[str, str] = os.environ,
    *,
    now: datetime | None = None,
) -> RoleRuntimeState:
    target = role_target(env)
    state_dir = role_state_dir(env)
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    lease = _read_lease(state_dir / "controller-lease.json")
    controller_alive = _controller_alive(lease, target.role)
    conversation_id = _conversation_id(state_dir, target, lease)
    active_turn = (
        lease.phase == "openhands-turn"
        if lease is not None and controller_alive
        else None
    )
    raw_history_event_count = None
    raw_history_digest = None
    unmatched_actions = None
    if conversation_id is not None and active_turn is False:
        raw_history_event_count, raw_history_digest = _raw_history_checkpoint(
            state_dir,
            conversation_id,
        )
        unmatched_actions = _unmatched_action_count(state_dir, conversation_id)
    pending_event_keys = _pending_event_keys(state_dir, target, conversation_id)
    active_delegation_count = _active_delegation_count(state_dir)
    (
        running_count,
        training_errors,
        running_run_ids,
        recent_run_ids,
        run_inventory_complete,
    ) = _training_state(state_dir, target.role)
    context_resets = _context_reset_statuses(state_dir, target)
    token = _control_token(
        lease,
        conversation_id,
        raw_history_digest,
        pending_event_keys,
        active_delegation_count,
    )
    restart_token = _restart_control_token(token, lease)
    return RoleRuntimeState(
        target=target,
        observation=RoleObservation(
            target=target,
            observed_at=observed_at,
            control_token=token,
            restart_control_token=restart_token,
            controller_alive=controller_alive,
            controller_phase=lease.phase if lease is not None else None,
            conversation_id=conversation_id,
            active_turn=active_turn,
            unmatched_actions=unmatched_actions,
            raw_history_event_count=raw_history_event_count,
            raw_history_digest=raw_history_digest,
            pending_event_keys=pending_event_keys,
            active_delegation_count=active_delegation_count,
        ),
        lease_deadline_seconds=(
            max(lease.deadline - time.monotonic(), 0.0) if lease is not None else None
        ),
        completed_turns=lease.completed_turns if lease is not None else None,
        running_training_count=running_count,
        active_delegation_count=active_delegation_count,
        wandb_run_inventory_complete=run_inventory_complete,
        running_wandb_run_ids=running_run_ids,
        recent_wandb_run_ids=recent_run_ids,
        context_resets=context_resets,
        cpu_percent=_bounded_percent(psutil.cpu_percent(interval=0.05)),
        memory_percent=_bounded_percent(psutil.virtual_memory().percent),
        disk_percent=_disk_percent(state_dir),
        gpu_percent=_gpu_percent(),
        recent_errors=training_errors,
    )


def _context_reset_statuses(
    state_dir: Path,
    target: RoleTarget,
) -> tuple[ContextResetStatus, ...]:
    queue_path = state_dir / "context-resets.sqlite3"
    if not queue_path.is_file():
        return ()
    with ContextResetRequestStore(queue_path) as store:
        return store.statuses(target, limit=20)


def advisor_research_tail(
    env: Mapping[str, str] = os.environ,
    *,
    now: datetime | None = None,
) -> RoleResearchTail:
    """Return a stable, bounded tail of advisor-authored active-branch text."""

    target = role_target(env)
    if target.role != "advisor":
        raise PermissionError("research-tail inspection is advisor-only")
    state_dir = role_state_dir(env)
    lease = _read_lease(state_dir / "controller-lease.json")
    conversation_id = _conversation_id(state_dir, target, lease)
    if conversation_id is None:
        raise RuntimeError("the advisor conversation is not initialized")
    before = _raw_history_checkpoint(state_dir, conversation_id)
    branch = _active_branch(state_dir, conversation_id)
    after = _raw_history_checkpoint(state_dir, conversation_id)
    if None in before or before != after or branch is None:
        raise RuntimeError("the advisor active branch changed during inspection")

    messages: list[RoleResearchTailItem] = []
    for index, event in enumerate(branch):
        summary = _agent_authored_summary(event)
        if summary is None:
            continue
        messages.append(
            RoleResearchTailItem(
                index=index,
                kind=type(event).__name__,
                source=getattr(event, "source", None),
                summary=_redact_role_secrets(summary, env)[-4_000:],
            )
        )
    return RoleResearchTail(
        conversation_id=conversation_id,
        observed_at=(now or datetime.now(UTC)).astimezone(UTC),
        messages=tuple(messages[-3:]),
    )


def nudge(
    expected_conversation_id: UUID,
    control_token: str,
    message: str,
    operation_key: str,
    env: Mapping[str, str] = os.environ,
) -> NudgeReceipt:
    state = _require_control_token(control_token, env)
    observation = state.observation
    if observation.conversation_id != expected_conversation_id:
        raise RuntimeError("the role conversation changed before nudge delivery")
    target = state.target
    payload: dict[str, object] = {
        "message": message,
        "request_id": operation_key,
        "source": "operational-supervisor",
    }
    if target.role == "student":
        payload["parent_conversation_id"] = str(expected_conversation_id)
    delivery_key = f"supervisor-nudge:{operation_key}"
    database = role_state_dir(env) / f"{target.role}-events.sqlite3"
    with AdvisorEventStore(database) as store:
        store.enqueue(
            AdvisorEvent(
                kind="operator_nudge",
                dedupe_key=delivery_key,
                payload=payload,
            )
        )
    return NudgeReceipt(
        target=target,
        conversation_id=expected_conversation_id,
        delivery_key=delivery_key,
    )


def restart_controller(
    expected_conversation_id: UUID,
    restart_control_token: str,
    env: Mapping[str, str] = os.environ,
) -> RestartReceipt:
    if expected_conversation_id is None:
        raise ValueError("restart requires the expected conversation UUID")
    state = _require_restart_control_token(restart_control_token, env)
    observation = state.observation
    if observation.conversation_id != expected_conversation_id:
        raise RuntimeError("the role conversation changed before restart")
    if observation.controller_phase != "sleep":
        raise RuntimeError("the controller is not in its quiescent sleep phase")
    if state.running_training_count is None:
        raise RuntimeError("training activity is unknown; refusing controller restart")
    if state.running_training_count:
        raise RuntimeError("a student experiment is running; refusing controller restart")
    if state.active_delegation_count != observation.active_delegation_count:
        raise RuntimeError(
            "delegation activity observation is inconsistent; refusing controller restart"
        )
    if state.active_delegation_count is None:
        raise RuntimeError("delegation activity is unknown; refusing controller restart")
    if state.active_delegation_count:
        raise RuntimeError("a delegated task is active; refusing controller restart")
    if observation.active_turn is not False:
        raise RuntimeError("the role is not at a safe controller restart boundary")
    lease = _read_lease(role_state_dir(env) / "controller-lease.json")
    if lease is None or not _controller_alive(lease, state.target.role):
        raise RuntimeError("the observed controller is no longer live")
    final_control_token = _control_token(
        lease,
        observation.conversation_id,
        observation.raw_history_digest,
        observation.pending_event_keys,
        observation.active_delegation_count,
    )
    if _restart_control_token(final_control_token, lease) != restart_control_token:
        raise RuntimeError("the controller left its quiescent restart boundary")
    os.kill(lease.pid, signal.SIGTERM)
    return RestartReceipt(
        target=state.target,
        conversation_id=expected_conversation_id,
        state_preserved=True,
        compute_preserved=True,
    )


def request_context_reset(
    request: ContextResetRequest,
    env: Mapping[str, str] = os.environ,
) -> ContextResetReceipt:
    state = _require_control_token(request.expected_control_token, env)
    observation = state.observation
    if request.target != state.target:
        raise PermissionError("context reset target does not match this role")
    if observation.conversation_id != request.expected_conversation_id:
        raise RuntimeError("the role conversation changed before reset was queued")
    if observation.active_turn is not False or observation.unmatched_actions != 0:
        raise RuntimeError("the role is not at a safe context-reset boundary")
    if (
        observation.raw_history_event_count
        != request.expected_raw_history_event_count
        or observation.raw_history_digest != request.expected_raw_history_digest
        or observation.pending_event_keys != request.expected_pending_event_keys
    ):
        raise RuntimeError("conversation state changed before reset was queued")
    with ContextResetRequestStore(
        role_state_dir(env) / "context-resets.sqlite3"
    ) as store:
        store.enqueue(request)
    return ContextResetReceipt(
        target=state.target,
        request_id=request.request_id,
        expected_conversation_id=request.expected_conversation_id,
        expected_raw_history_event_count=request.expected_raw_history_event_count,
        expected_raw_history_digest=request.expected_raw_history_digest,
        expected_pending_event_keys=request.expected_pending_event_keys,
    )


def _require_control_token(
    expected: str,
    env: Mapping[str, str],
) -> RoleRuntimeState:
    current = observe_role(env)
    if current.observation.control_token != expected:
        raise RuntimeError("role state changed after the supervisor observation")
    return current


def _require_restart_control_token(
    expected: str,
    env: Mapping[str, str],
) -> RoleRuntimeState:
    current = observe_role(env)
    actual = current.observation.restart_control_token
    if actual is None or actual != expected:
        raise RuntimeError("role restart state changed after the supervisor observation")
    return current


def _read_lease(path: Path) -> WorkerLease | None:
    try:
        return WorkerLease.read(path)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _controller_alive(lease: WorkerLease | None, role: str) -> bool | None:
    if lease is None:
        return None
    try:
        process = psutil.Process(lease.pid)
        command = process.cmdline()
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and "senpai_agent.controller" in command
            and role in command
        )
    except (OSError, psutil.Error):
        return False


def _conversation_id(
    state_dir: Path,
    target: RoleTarget,
    lease: WorkerLease | None,
) -> UUID | None:
    if lease is not None and lease.conversation_id:
        try:
            return UUID(lease.conversation_id)
        except ValueError:
            return None
    if target.role == "advisor":
        path = state_dir / "advisor-conversation-id"
        try:
            return UUID(path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return None
    path = state_dir / "student-conversations.json"
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        candidates = [UUID(value) for value in values.values()]
    except (
        FileNotFoundError,
        AttributeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: _conversation_mtime(state_dir, candidate),
    )


def _conversation_mtime(state_dir: Path, conversation_id: UUID) -> float:
    try:
        return (state_dir / conversation_id.hex / "base_state.json").stat().st_mtime
    except OSError:
        return 0.0


def _event_files(state_dir: Path, conversation_id: UUID) -> tuple[Path, ...]:
    directory = state_dir / conversation_id.hex / "events"
    return tuple(sorted(directory.glob("event-*.json")))


def _raw_history_checkpoint(
    state_dir: Path,
    conversation_id: UUID,
    *,
    event_count: int | None = None,
) -> tuple[int | None, str | None]:
    directory = state_dir / conversation_id.hex
    if not directory.is_dir():
        return None, None
    digest = hashlib.sha256()
    try:
        paths = _event_files(state_dir, conversation_id)
        if event_count is not None:
            if len(paths) < event_count:
                return None, None
            paths = paths[:event_count]
        for path in paths:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    except OSError:
        return None, None
    return len(paths), digest.hexdigest()


def _unmatched_action_count(state_dir: Path, conversation_id: UUID) -> int | None:
    branch = _active_branch(state_dir, conversation_id)
    if branch is None:
        return None
    return len(ConversationState.get_unmatched_actions(branch))


def _active_branch(state_dir: Path, conversation_id: UUID) -> list[Event] | None:
    base_path = state_dir / conversation_id.hex / "base_state.json"
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
        indexed = [
            Event.model_validate_json(path.read_text(encoding="utf-8"))
            for path in _event_files(state_dir, conversation_id)
        ]
        by_id = {event.id: (index, event) for index, event in enumerate(indexed)}
        leaf = base.get("leaf_event_id")
        if leaf is None and not base.get("head_is_empty", False) and indexed:
            leaf = indexed[-1].id
        branch: list[Event] = []
        seen: set[str] = set()
        while leaf is not None:
            if leaf in seen:
                return None
            seen.add(leaf)
            index, event = by_id[leaf]
            branch.append(event)
            if event.parent_id == ROOT_PARENT_ID:
                leaf = None
            elif event.parent_id is not None:
                leaf = event.parent_id
            else:
                leaf = indexed[index - 1].id if index else None
        branch.reverse()
        return branch
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _agent_authored_summary(event: Event) -> str | None:
    if getattr(event, "source", None) != "agent":
        return None
    if isinstance(event, MessageEvent):
        text = "".join(
            content.text
            for content in event.to_llm_message().content
            if isinstance(content, TextContent)
        ).strip()
        return text or None
    if isinstance(event, ActionEvent):
        message = getattr(event.action, "message", None)
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def _redact_role_secrets(value: str, env: Mapping[str, str]) -> str:
    secrets = (
        secret
        for name, secret in env.items()
        if secret
        and len(secret) >= 7
        and name.endswith(
            ("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET", "_CREDENTIAL")
        )
    )
    for secret in sorted(set(secrets), key=len, reverse=True):
        value = value.replace(secret, "<redacted>")
    return value


def _pending_event_keys(
    state_dir: Path,
    target: RoleTarget,
    conversation_id: UUID | None,
) -> tuple[str, ...]:
    path = state_dir / f"{target.role}-events.sqlite3"
    if not path.is_file():
        return ()
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        try:
            rows = connection.execute(
                "SELECT event_json FROM advisor_events "
                "WHERE acknowledged = 0 ORDER BY rowid"
            ).fetchall()
        finally:
            connection.close()
        events = [AdvisorEvent.model_validate_json(row[0]) for row in rows]
    except (sqlite3.Error, ValueError) as error:
        raise RuntimeError("pending role events are unreadable") from error
    if target.role == "student" and conversation_id is not None:
        events = [
            event
            for event in events
            if event.payload.get("parent_conversation_id") == str(conversation_id)
        ]
    return tuple(event.dedupe_key for event in events)


def _training_state(
    state_dir: Path,
    role: str,
) -> tuple[
    int | None,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    bool | None,
]:
    if role == "advisor":
        return 0, (), (), (), True
    directory = state_dir / "training"
    if not directory.exists():
        return 0, (), (), (), True
    running = 0
    error_events: list[tuple[datetime, str]] = []
    running_run_ids: dict[str, None] = {}
    try:
        inventory = read_training_inventory(directory)
        active = _validated_active_training_results(directory, inventory)
        inventory_complete = not inventory.wandb_run_inventory_overflow
        for error in inventory.recent_errors:
            error_events.append(
                (
                    error.observed_at,
                    "SENPAI_TRAINING_ERROR "
                    f"state={error.state.value} "
                    f"updated={error.observed_at.isoformat()} "
                    f"fingerprint={error.fingerprint}",
                )
            )
        for entry, result in active:
            if training_process_is_live(result):
                running += 1
                inventory_complete = inventory_complete and bool(result.wandb_run_ids)
                for run_id in result.wandb_run_ids:
                    running_run_ids[run_id] = None
            else:
                error_events.append(
                    (
                        entry.updated_at,
                        "SENPAI_TRAINING_STALE "
                        f"updated={entry.updated_at.isoformat()} "
                        "fingerprint="
                        f"{hashlib.sha256(result.training_id.encode()).hexdigest()[:16]}",
                    )
                )
    except (OSError, ValueError):
        return (
            None,
            ("SENPAI_TRAINING_INVENTORY_UNREADABLE",),
            (),
            (),
            None,
        )
    error_events.sort(key=lambda item: item[0])
    return (
        running,
        tuple(message for _, message in error_events[-20:]),
        tuple(running_run_ids)[-50:],
        inventory.recent_wandb_run_ids,
        inventory_complete
        and len(running_run_ids) <= 50
        and len(inventory.recent_wandb_run_ids) <= 200,
    )


def _validated_active_training_results(
    directory: Path,
    inventory: TrainingInventory,
) -> tuple[tuple[TrainingInventoryEntry, TrainingResult], ...]:
    active: list[tuple[TrainingInventoryEntry, TrainingResult]] = []
    for entry in inventory.active:
        if entry.result.state is not TrainingState.RUNNING:
            raise ValueError("training inventory has a non-running active result")
        path = training_result_path(directory, entry.result.training_id)
        result = TrainingResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result != entry.result:
            raise ValueError("training inventory disagrees with persisted result")
        active.append((entry, result))
    return tuple(active)


def _active_delegation_count(state_dir: Path) -> int | None:
    database_path = state_dir / "delegation" / "tasks.sqlite3"
    if not database_path.is_file():
        return 0
    try:
        database = sqlite3.connect(
            f"file:{database_path}?mode=ro",
            uri=True,
            timeout=2,
        )
        try:
            row = database.execute(
                "SELECT COUNT(*) FROM tasks "
                "WHERE status IN ('queued', 'starting', 'running')"
            ).fetchone()
        finally:
            database.close()
    except sqlite3.Error:
        return None
    if row is None or not isinstance(row[0], int) or not 0 <= row[0] <= 100:
        return None
    return row[0]


def _control_token(
    lease: WorkerLease | None,
    conversation_id: UUID | None,
    history_digest: str | None,
    pending_event_keys: Sequence[str],
    active_delegation_count: int | None = 0,
) -> str:
    value = {
        "pid": lease.pid if lease is not None else None,
        "conversation_id": str(conversation_id) if conversation_id else None,
        "history": history_digest,
        "pending": list(pending_event_keys),
        "active_delegations": active_delegation_count,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _restart_control_token(
    control_token: str,
    lease: WorkerLease | None,
) -> str:
    value = {
        "control": control_token,
        "phase": lease.phase if lease is not None else None,
        "completed_turns": lease.completed_turns if lease is not None else None,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bounded_percent(value: float) -> float:
    return min(max(float(value), 0.0), 100.0)


def _disk_percent(path: Path) -> float | None:
    try:
        return _bounded_percent(psutil.disk_usage(path).percent)
    except (OSError, psutil.Error):
        return None


def _gpu_percent() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    values = [
        float(value.strip())
        for value in result.stdout.splitlines()
        if value.strip()
    ]
    return _bounded_percent(sum(values) / len(values)) if values else None


def role_control_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    if argv:
        raise RuntimeError("role control accepts JSON on stdin, not command arguments")
    request = RoleControlRequest.model_validate_json(sys.stdin.read())
    if request.command == "observe":
        result: BaseModel = observe_role(env)
    elif request.command == "research_tail":
        result = advisor_research_tail(env)
    elif request.command == "nudge":
        if None in (
            request.expected_conversation_id,
            request.control_token,
            request.message,
            request.operation_key,
        ):
            raise ValueError("nudge requires conversation, token, message, and key")
        result = nudge(
            request.expected_conversation_id,
            request.control_token,
            request.message,
            request.operation_key,
            env,
        )
    elif request.command == "restart":
        if (
            request.restart_control_token is None
            or request.expected_conversation_id is None
        ):
            raise ValueError(
                "restart requires a conversation UUID and restart control token"
            )
        result = restart_controller(
            request.expected_conversation_id,
            request.restart_control_token,
            env,
        )
    elif request.command == "context_reset":
        if request.context_reset_request is None:
            raise ValueError("context_reset requires an owner-consumed request")
        result = request_context_reset(request.context_reset_request, env)
    else:
        raise RuntimeError(f"unsupported role control command: {request.command}")
    print(result.model_dump_json(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(role_control_main())
