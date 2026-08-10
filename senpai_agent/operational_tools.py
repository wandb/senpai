"""The supervisor's single campaign-scoped OpenHands operation tool."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from pydantic import Field, model_validator

from senpai_agent.kubernetes_operations import KubectlCampaignBackend
from senpai_agent.operations import (
    AnomalyCategory,
    CampaignInventory,
    CollectRole,
    CollectRoleReceipt,
    ContextReset,
    Nudge,
    OperationLedger,
    OperationPolicy,
    OperationService,
    Restart,
    RoleTarget,
)


class SupervisorOperationAction(Action):
    operation: Literal[
        "inspect",
        "nudge",
        "restart_controller",
        "reset_context",
    ]
    operation_key: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Trace key. Mutations replay an exact reused key; inspections always "
            "execute fresh."
        ),
    )
    incident_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Human-readable stable identity for the observed anomaly.",
    )
    anomaly_category: AnomalyCategory | None = Field(
        default=None,
        description=(
            "Typed operational anomaly class. Cooldown identity is derived from "
            "this category, the action, and the configured target."
        ),
    )
    role: Literal["advisor", "student"]
    student: str | None = Field(default=None, min_length=1, max_length=200)
    expected_conversation_id: UUID | None = None
    message: str | None = Field(default=None, min_length=1, max_length=8_000)
    recovery_prompt: str | None = Field(default=None, min_length=1, max_length=16_000)
    reason: str | None = Field(default=None, min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_operation_fields(self) -> SupervisorOperationAction:
        if self.role == "advisor" and self.student is not None:
            raise ValueError("advisor operations cannot name a student")
        if self.role == "student" and self.student is None:
            raise ValueError("student operations require a configured student name")
        if self.operation == "inspect":
            return self
        if not self.incident_key or not self.anomaly_category or not self.reason:
            raise ValueError(
                "mutations require incident_key, anomaly_category, and reason"
            )
        if (
            self.operation in {"nudge", "restart_controller", "reset_context"}
            and not self.expected_conversation_id
        ):
            raise ValueError("mutations require the expected conversation UUID")
        if self.operation == "nudge" and not self.message:
            raise ValueError("nudge requires a message")
        if self.operation == "reset_context" and not self.recovery_prompt:
            raise ValueError("reset_context requires a recovery prompt")
        return self


class SupervisorOperationObservation(Observation):
    result: dict[str, object]

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=json.dumps(
                    self.result,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ]


class _SupervisorOperationExecutor(
    ToolExecutor[SupervisorOperationAction, SupervisorOperationObservation]
):
    def __init__(self, service: OperationService):
        self.service = service

    def __call__(
        self,
        action: SupervisorOperationAction,
        conversation: object | None = None,
    ) -> SupervisorOperationObservation:
        target = RoleTarget(
            research_tag=self.service.inventory.research_tag,
            role=action.role,
            student=action.student,
        )
        if action.operation == "inspect":
            request = CollectRole(
                operation_key=action.operation_key,
                target=target,
            )
        elif action.operation == "nudge":
            request = Nudge(
                operation_key=action.operation_key,
                incident_key=action.incident_key,
                anomaly_category=action.anomaly_category,
                target=target,
                expected_conversation_id=action.expected_conversation_id,
                message=action.message,
                reason=action.reason,
            )
        elif action.operation == "restart_controller":
            request = Restart(
                operation_key=action.operation_key,
                incident_key=action.incident_key,
                anomaly_category=action.anomaly_category,
                target=target,
                expected_conversation_id=action.expected_conversation_id,
                reason=action.reason,
            )
        elif action.operation == "reset_context":
            request = ContextReset(
                operation_key=action.operation_key,
                incident_key=action.incident_key,
                anomaly_category=action.anomaly_category,
                target=target,
                expected_conversation_id=action.expected_conversation_id,
                recovery_prompt=action.recovery_prompt,
                reason=action.reason,
            )
        else:
            raise ValueError(f"unsupported supervisor operation: {action.operation}")
        outcome = self.service.execute(request)
        if action.operation == "inspect":
            receipt = outcome.receipt
            if not isinstance(receipt, CollectRoleReceipt):
                raise RuntimeError("role inspection returned an invalid receipt")
            observed = receipt.observation
            return SupervisorOperationObservation(
                result={
                    "operation_key": outcome.operation_key,
                    "disposition": outcome.disposition,
                    "observation": {
                        "target": observed.target.model_dump(mode="json"),
                        "observed_at": observed.observed_at.isoformat(),
                        "controller_alive": observed.controller_alive,
                        "phase": _controller_phase_category(
                            observed.controller_phase
                        ),
                        "worker_generation": observed.worker_generation,
                        "conversation_id": (
                            str(observed.conversation_id)
                            if observed.conversation_id is not None
                            else None
                        ),
                        "active_turn": observed.active_turn,
                        "unmatched_action_count": observed.unmatched_actions,
                        "history_event_count": observed.raw_history_event_count,
                        "history_fingerprint": (
                            hashlib.sha256(
                                observed.raw_history_digest.encode()
                            ).hexdigest()[:16]
                            if observed.raw_history_digest
                            else None
                        ),
                        "pending_event_count": len(observed.pending_event_keys),
                        "active_delegation_count": (
                            observed.active_delegation_count
                        ),
                        "restart_authorized": (
                            observed.restart_control_token is not None
                        ),
                    },
                }
            )
        return SupervisorOperationObservation(
            result=outcome.model_dump(mode="json")
        )

    def close(self) -> None:
        self.service.close()


class SupervisorOperationTool(
    ToolDefinition[SupervisorOperationAction, SupervisorOperationObservation]
):
    name = "senpai_operations"

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        *,
        state_dir: str | Path,
        namespace: str,
        research_tag: str,
        repo: str,
        advisor_branch: str,
        students: Sequence[str],
        mutation_cooldown_seconds: float = 1800,
    ) -> Sequence[Self]:
        inventory = CampaignInventory(
            research_tag=research_tag,
            repo=repo,
            advisor_branch=advisor_branch,
            students=tuple(students),
        )
        service = OperationService(
            inventory,
            KubectlCampaignBackend(inventory, namespace=namespace),
            OperationLedger(Path(state_dir) / "operations.sqlite3"),
            policy=OperationPolicy(
                mutation_cooldown_seconds=mutation_cooldown_seconds,
            ),
        )
        return [
            cls(
                description=(
                    "Inspect or repair only this supervisor's configured advisor "
                    "and students. Every inspect executes fresh; inspect first to "
                    "obtain the exact conversation UUID and typed role state. "
                    "Mutations are durably deduplicated and "
                    "cooldown-limited by typed anomaly category, action, and target. "
                    "Context reset preserves the complete raw "
                    "trace and is consumed only by the owning controller. Controller "
                    "restart refuses to interrupt running student experiments or "
                    "delegated agents."
                ),
                action_type=SupervisorOperationAction,
                observation_type=SupervisorOperationObservation,
                annotations=ToolAnnotations(
                    title="Senpai campaign operations",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_SupervisorOperationExecutor(service),
            )
        ]


_CONTROLLER_PHASES = frozenset(
    {
        "acknowledge",
        "monitor-backoff",
        "monitor-sleep",
        "openhands-turn",
        "poll",
        "reconcile",
        "sleep",
        "start-gate",
        "startup",
        "turn-backoff",
        "turn-complete",
    }
)


def _controller_phase_category(phase: str | None) -> str | None:
    if phase is None:
        return None
    return phase if phase in _CONTROLLER_PHASES else "other"
