"""Run an explicit supervisor request while the local controller is idle."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING
from uuid import UUID

from senpai_agent.delegation import (
    TERMINAL_TASK_STATUSES,
    DelegationManager,
    OpenHandsChildProcess,
    SupervisorTask,
)
from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.mailbox.values import payload_digest
from senpai_agent.github.supervision import (
    SupervisorEnvelope,
    SupervisorGateway,
    SupervisorResult,
)
from senpai_agent.github.workflow import GitHubAPIError, GitHubTransportError
from senpai_agent.inbox import PersistentInbox
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.openhands_runner import delegation_config, local_event_db_path
from senpai_agent.state import AssignmentConversationRegistry
from senpai_agent.supervisor import ProgressLease
from senpai_agent.training import TrainingState

if TYPE_CHECKING:
    from senpai_agent.monitor import MonitorStore
    from senpai_agent.openhands_runner import RunnerConfig
    from senpai_agent.tools import TrainingRuntime


@dataclass
class SupervisorHandler:
    config: RunnerConfig
    inbox: PersistentInbox
    progress: ProgressLease | None = None
    training: TrainingRuntime | None = None
    monitor_store: MonitorStore | None = None

    @cached_property
    def gateway(self) -> SupervisorGateway:
        return SupervisorGateway(self.config)

    def __call__(self) -> None:
        gateway = self.gateway
        # Closing Issues changes pagination; finish the snapshot before handling it.
        for issue, envelope in list(gateway.pending()):
            try:
                self._handle(gateway, issue, envelope)
            except Exception as error:  # noqa: BLE001 - isolate each Issue failure
                print(
                    f"SENPAI_SUPERVISOR_ERROR issue={issue['number']} "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )

    def _handle(self, gateway, issue, envelope: SupervisorEnvelope) -> None:
        issue_number = int(issue["number"])
        if envelope.result is not None:
            payload = {
                "number": issue_number,
                "url": str(issue["html_url"]),
                **envelope.model_dump(mode="json"),
            }
            with LocalEventStore(local_event_db_path(self.config)) as store:
                store.enqueue(
                    LocalEvent(
                        kind="supervisor_completed",
                        dedupe_key=f"supervisor_reply:{issue_number}:{payload_digest(payload)}",
                        payload=payload,
                    )
                )
            gateway.close_reply(issue_number, envelope)
            return
        request = envelope.request
        source = envelope.model_dump(mode="json", exclude={"result"})
        parent_id = self.config.conversation_id
        if self.config.role == "student":
            assert request.assignment is not None
            parent_id = AssignmentConversationRegistry(
                self.config.state_dir / "student-conversations.json"
            ).for_assignment(
                request.assignment.assignment_id,
                request.assignment.revision_id,
            )
        key = f"supervisor:{issue_number}"
        with LocalEventStore(local_event_db_path(self.config)) as store:
            saved = store.get(key)
            if saved is None:
                result = self._repair(gateway, envelope, parent_id, key)
                saved = LocalEvent(
                    kind="supervisor_recovered"
                    if result.resolved
                    else "supervisor_completed",
                    dedupe_key=key,
                    payload={
                        "source": source,
                        "parent_conversation_id": str(parent_id),
                        "result": result.model_dump(mode="json"),
                    },
                )
                # This event is both the accepted outcome and the local receipt.
                # Mailbox delivery can recover the parent even if GitHub is unavailable.
                store.enqueue(saved)
        result = SupervisorResult.model_validate(saved.payload["result"])
        if saved.payload["source"] != source or saved.payload[
            "parent_conversation_id"
        ] != str(parent_id):
            result = SupervisorResult(
                resolved=False,
                repair_summary="This Issue changed after its repair attempt. Create a new Supervisor request for the changed task.",
            )
        if self.progress is not None:
            self.progress.update("supervisor-complete", 300)
        gateway.complete(
            issue_number,
            envelope,
            result.repair_summary,
            result.resolved,
            delivered_to=parent_id,
        )

    def _repair(
        self,
        gateway: SupervisorGateway,
        envelope: SupervisorEnvelope,
        parent_id: UUID,
        key: str,
    ) -> SupervisorResult:
        request = envelope.request
        child_config = delegation_config(self.config)
        manager = DelegationManager(
            child_config,
            lambda request: OpenHandsChildProcess(child_config, request),
            event_db_path=local_event_db_path(self.config),
        )
        source = envelope.model_dump(mode="json", exclude={"result"})
        attempt_key = f"{key}:{payload_digest(source)}"
        prior = next(
            (
                row
                for row in manager.registry.rows()
                if row["agent"] == "supervisor"
                and (row["task_key"] or "").startswith(f"{key}:")
            ),
            None,
        )
        task_id = prior["task_id"] if prior is not None else None
        owner = prior["parent_conversation_id"] if prior is not None else str(parent_id)
        try:
            if self.progress is not None:
                self.progress.update("supervisor-prepare", 300)
            if prior is not None and (
                prior["task_key"] != attempt_key or owner != str(parent_id)
            ):
                raise RuntimeError(
                    "This Issue changed during its repair attempt. "
                    "Create a new Supervisor request for the changed task."
                )
            if any(row["task_id"] != task_id for row in manager.registry.active_rows()):
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
                prompt = gateway.prepare(envelope)
                for turn in self.inbox.quarantined_turns():
                    if turn.conversation_id == str(parent_id):
                        prompt += (
                            f"\n\nQuarantined turn: {turn.turn_id}\n"
                            f"Reason: {turn.quarantine_reason}\n"
                        )
            task = manager.spawn(
                attempt_key,
                [SupervisorTask(key=attempt_key, task=prompt)],
                str(parent_id),
            )[0]
            task_id = task.task_id
            while task.status not in TERMINAL_TASK_STATUSES:
                if self.progress is not None:
                    self.progress.update("supervisor", 60)
                time.sleep(1)
                task = manager.states([task_id], str(parent_id))[0]
            if task.status != "finished":
                raise RuntimeError(task.error or f"Supervisor task {task.status}.")
            result = SupervisorResult.model_validate_json(task.result or "")
            if result.resolved:
                if self.progress is not None:
                    self.progress.update("supervisor-validate", 300)
                gateway.validate(request)
        except BaseException as error:
            if isinstance(
                error, (GitHubTransportError, GitHubAPIError, GitHubReadError)
            ):
                status = getattr(error, "status_code", None)
                if status is None or status in {408, 429} or status >= 500:
                    raise
            if task_id is not None:
                manager.cancel(
                    [task_id],
                    owner,
                    reason="Supervisor repair aborted."
                    if isinstance(error, Exception)
                    else "Controller shutdown interrupted the repair. Create a new Supervisor request with a new request_id.",
                )
            if not isinstance(error, Exception):
                raise
            result = SupervisorResult(
                resolved=False,
                repair_summary=f"{type(error).__name__}: {str(error)[:2000]}\nCreate a new Supervisor request with a new request_id to retry.",
            )
        if task_id is not None:
            manager.registry.mark_collected([task_id])
        return result
