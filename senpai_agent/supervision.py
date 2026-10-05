"""Run an explicit supervisor request while the local controller is idle."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from senpai_agent.delegation import (
    TERMINAL_TASK_STATUSES,
    AgentTask,
    DelegationManager,
    OpenHandsChildProcess,
    SpawnAgentsAction,
)
from senpai_agent.github.mailbox.values import versioned_event
from senpai_agent.github.supervision import (
    SupervisorGateway,
    SupervisorRequest,
    SupervisorResult,
)
from senpai_agent.inbox import STEER_PRIORITY, PersistentInbox
from senpai_agent.local_events import LocalEventStore
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.state import AssignmentConversationRegistry, _replace_json
from senpai_agent.supervisor import ProgressLease
from senpai_agent.training import TrainingState

if TYPE_CHECKING:
    from senpai_agent.monitor import MonitorStore
    from senpai_agent.openhands_runner import RunnerConfig
    from senpai_agent.tools import TrainingRuntime


class _SupervisorOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: SupervisorRequest
    parent_conversation_id: UUID
    task_id: str | None
    result: SupervisorResult


class SupervisorHandler:
    def __init__(
        self,
        config: RunnerConfig,
        inbox: PersistentInbox,
        *,
        progress: ProgressLease | None = None,
        training: TrainingRuntime | None = None,
        monitor_store: MonitorStore | None = None,
    ):
        self.config = config
        self.inbox = inbox
        self.progress = progress
        self.training = training
        self.monitor_store = monitor_store

    def __call__(self, event: ControllerEvent) -> UUID | None:
        from senpai_agent.openhands_runner import delegation_config

        request = SupervisorRequest.model_validate(event.payload["request"])
        issue_number = int(event.payload["number"])
        gateway = SupervisorGateway(self.config)
        child_config = delegation_config(self.config)
        manager = DelegationManager(
            child_config,
            lambda request: OpenHandsChildProcess(child_config, request),
        )
        parent_id = self.config.conversation_id
        if self.config.role == "student":
            assert request.assignment is not None
            parent_id = AssignmentConversationRegistry(
                self.config.state_dir / "student-conversations.json"
            ).for_assignment(
                request.assignment.assignment_id,
                request.assignment.revision_id,
            )
        outcome_path = self.config.state_dir / "supervision" / f"{issue_number}.json"
        saved = (
            _SupervisorOutcome.model_validate_json(outcome_path.read_text())
            if outcome_path.exists()
            else None
        )
        key = f"supervisor:{issue_number}"
        prior = next(
            (
                row
                for row in manager.registry.rows()
                if row["parent_conversation_id"] == str(parent_id)
                and row["task_key"] == key
            ),
            None,
        )
        task_id = prior["task_id"] if prior is not None else None
        if saved is not None:
            task_id = saved.task_id
        result = saved.result if saved is not None else None
        if saved is not None and (
            saved.request != request or saved.parent_conversation_id != parent_id
        ):
            result = SupervisorResult(
                resolved=False,
                repair_summary=(
                    "This Issue changed after its repair attempt. Create a new "
                    "Supervisor request for the changed task."
                ),
            )
        if saved is None:
            try:
                if self.progress is not None:
                    self.progress.update("supervisor-prepare", 300)
                if any(
                    row["task_id"] != task_id for row in manager.registry.active_rows()
                ):
                    raise RuntimeError(
                        "Supervisor requires all existing subagents to finish first."
                    )
                if self.training is not None and self.monitor_store is not None:
                    for monitor in self.monitor_store.active():
                        status = self.training.get_training_status(monitor.training_id)
                        if status.state == TrainingState.RUNNING:
                            raise RuntimeError(
                                "Supervisor requires active training to stop first."
                            )
                # Reuse the original snapshot if the controller restarts mid-repair.
                if prior is not None:
                    gateway.validate(request)
                    prompt = prior["task"]
                else:
                    prompt = gateway.prepare(request)
                    for turn in self.inbox.quarantined_turns():
                        if turn.conversation_id == str(parent_id):
                            prompt += (
                                f"\n\nQuarantined turn: {turn.turn_id}\n"
                                f"Reason: {turn.quarantine_reason}\n"
                            )
                task = manager.spawn_for_owner(
                    SpawnAgentsAction(
                        batch_key=key,
                        tasks=[
                            AgentTask(
                                key=key,
                                task=prompt,
                                agent="supervisor",
                                model="smart",
                            )
                        ],
                    ),
                    str(parent_id),
                )[0]
                task_id = task.task_id
                while task.status not in TERMINAL_TASK_STATUSES:
                    if self.progress is not None:
                        self.progress.update("supervisor", 60)
                    time.sleep(1)
                    task = manager.states_for_owner([task_id], str(parent_id))[0]
                if task.status != "finished":
                    raise RuntimeError(task.error or f"Supervisor task {task.status}.")
                result = SupervisorResult.model_validate_json(task.result or "")
                if result.resolved:
                    if self.progress is not None:
                        self.progress.update("supervisor-validate", 300)
                    gateway.validate(request)
            except BaseException as error:
                if task_id is not None:
                    manager.cancel_for_owner([task_id], str(parent_id))
                if not isinstance(error, Exception):
                    raise
                result = SupervisorResult(
                    resolved=False,
                    repair_summary=f"{type(error).__name__}: {error}",
                )

        _replace_json(
            outcome_path,
            _SupervisorOutcome(
                request=request,
                parent_conversation_id=parent_id,
                task_id=task_id,
                result=result,
            ).model_dump(mode="json"),
        )
        receipt = ControllerEvent(
            kind="supervisor_completed",
            dedupe_key=f"supervisor_recovered:{issue_number}",
            payload={
                "request_id": request.request_id,
                "resolved": result.resolved,
                "summary": result.repair_summary,
            },
        )
        if result.resolved:
            # Persist the resume before closing its GitHub request. Exact retries cannot
            # reset a later quarantine because this receipt is handled only once.
            self.inbox.steer(
                parent_id,
                receipt.dedupe_key,
                receipt.to_prompt(),
                priority=STEER_PRIORITY,
                once=True,
            )
        else:
            receipt = versioned_event(
                "supervisor_completed",
                "blocked",
                issue_number,
                payload=receipt.payload,
            )
            self.inbox.enqueue(
                parent_id,
                receipt.dedupe_key,
                receipt.to_prompt(),
                once=True,
            )
        if task_id is not None:
            manager.registry.mark_collected([task_id])
            # Startup reconciliation may have emitted an orphaned-task receipt.
            with LocalEventStore(
                self.config.state_dir / f"{self.config.role}-events.sqlite3"
            ) as store:
                store.acknowledge(f"agent_result:{task_id}")
        if self.progress is not None:
            self.progress.update("supervisor-complete", 300)
        gateway.complete(
            issue_number,
            request,
            result.repair_summary,
            result.resolved,
            delivered_to=parent_id,
        )
        return parent_id if result.resolved else None
