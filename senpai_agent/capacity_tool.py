"""Read the observer's sanitized snapshot without cluster credentials or actions."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)

from senpai_agent.cluster_capacity import (
    SNAPSHOT_ENV,
    CapacitySnapshot,
    read_capacity_snapshot,
)


class ClusterCapacityAction(Action):
    """Read the configured worker shape's latest advisory resource-fit snapshot."""


class ClusterCapacityObservation(Observation):
    snapshot: CapacitySnapshot
    age_seconds: float | None

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=json.dumps(
                    {
                        "snapshot": self.snapshot.model_dump(mode="json"),
                        "age_seconds": self.age_seconds,
                    }
                )
            )
        ]


class _ClusterCapacityExecutor(
    ToolExecutor[ClusterCapacityAction, ClusterCapacityObservation]
):
    def __call__(
        self, action: ClusterCapacityAction, conversation=None
    ) -> ClusterCapacityObservation:
        configured = os.environ.get(SNAPSHOT_ENV)
        snapshot, age = read_capacity_snapshot(Path(configured) if configured else None)
        return ClusterCapacityObservation(snapshot=snapshot, age_seconds=age)


class ClusterCapacityTool(
    ToolDefinition[ClusterCapacityAction, ClusterCapacityObservation]
):
    name = "get_cluster_capacity"

    @classmethod
    def create(cls, conv_state) -> Sequence[ToolDefinition]:
        return [
            cls(
                description=(
                    "Read the latest sanitized cluster capacity snapshot for the configured worker shape. "
                    "Reports the observation time, age, GPU/CPU/memory resource-fit counts, placement exclusions "
                    "and verified preemptible capacity. Unknown or stale data cannot establish availability. "
                    "This is advisory: it never reserves resources, authorizes a launch or guarantees scheduling. "
                    "Use get_training_status for diagnostics of an existing owned run."
                ),
                action_type=ClusterCapacityAction,
                observation_type=ClusterCapacityObservation,
                annotations=ToolAnnotations(
                    title="Read cluster capacity",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_ClusterCapacityExecutor(),
            )
        ]
