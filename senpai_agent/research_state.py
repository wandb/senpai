# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Compact research state the harness keeps and re-sends after each compaction.

The advisor's prose notes live in the target repository and run to many
kilobytes; after a context compaction the advisor re-read them with greps.
This file is small, typed and machine-readable: the advisor replaces it with
`update_research_state`, and the runner queues its content as a
`context_compacted` event whenever the model's context is compacted.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.local_events import LocalEvent

RESEARCH_STATE_FILENAME = "research-state.json"
MAX_RESEARCH_STATE_BYTES = 16_000
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def research_state_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / RESEARCH_STATE_FILENAME


class CurrentBest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    what: str = Field(
        min_length=1,
        description="What holds the result: the branch, PR, or submission.",
    )
    metric: str = Field(
        min_length=1,
        description="Primary metric value with the split or benchmark it was measured on.",
    )
    identifiers: dict[str, str] = Field(
        default_factory=dict,
        description="Exact identifiers: commit, PR, W&B run, job or submission ids.",
    )


class LaneInFlight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lane: str = Field(min_length=1, description="Lane or hypothesis name.")
    owner: str = Field(min_length=1, description="Student that owns the lane.")
    pr: str = Field(default="", description="PR number or URL once opened.")
    deadline: str = Field(default="", description="UTC time by which a decision is due.")
    kill_criterion: str = Field(
        default="",
        description="Pre-registered condition that closes the lane.",
    )
    status: str = Field(default="", description="One short phrase on where it stands.")


class ResearchStateContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_best: list[CurrentBest] = Field(default_factory=list)
    lanes_in_flight: list[LaneInFlight] = Field(default_factory=list)
    bars: dict[str, str] = Field(
        default_factory=dict,
        description="Each named threshold and the decision it gates.",
    )
    budget: dict[str, str] = Field(
        default_factory=dict,
        description="Each budget as used and remaining, such as daily evaluation checks.",
    )
    operator_rules: list[str] = Field(
        default_factory=list,
        description="Standing operator instructions, close to verbatim.",
    )
    open_questions: list[str] = Field(default_factory=list)
    recent_decisions: list[str] = Field(
        default_factory=list,
        description="Latest decisions with their reason and UTC time; prune old ones.",
    )


class ResearchState(ResearchStateContent):
    updated_at: str


def render_research_state(state: ResearchState) -> str:
    return (
        json.dumps(
            state.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


def save_research_state(
    path: Path,
    content: ResearchStateContent,
    now: datetime | None = None,
) -> ResearchState:
    """Replace the state file atomically with a complete, bounded snapshot."""

    moment = (now or datetime.now(UTC)).astimezone(UTC)
    state = ResearchState(
        updated_at=moment.strftime(_TIME_FORMAT),
        **content.model_dump(),
    )
    text = render_research_state(state)
    size = len(text.encode())
    if size > MAX_RESEARCH_STATE_BYTES:
        raise ValueError(
            f"research state is {size} bytes; keep it under "
            f"{MAX_RESEARCH_STATE_BYTES} bytes by pruning old decisions and "
            "closed lanes"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".research-state-")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(temporary, 0o600)
    Path(temporary).replace(path)
    return state


def load_research_state(path: Path) -> ResearchState | None:
    if not path.is_file():
        return None
    return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))


def compaction_event(
    state_dir: str | Path,
    *,
    trigger: str,
    event_id: str,
    now: datetime | None = None,
) -> LocalEvent:
    """The durable notice that re-sends the saved state after a compaction."""

    moment = (now or datetime.now(UTC)).astimezone(UTC)
    state = load_research_state(research_state_path(state_dir))
    return LocalEvent(
        kind="context_compacted",
        dedupe_key=f"context_compacted:{event_id}",
        payload={
            "compacted_at": moment.strftime(_TIME_FORMAT),
            "trigger": trigger,
            "research_state": (
                state.model_dump(mode="json") if state is not None else None
            ),
        },
    )


class UpdateResearchStateAction(Action):
    """Replace the harness-kept research state with a complete current snapshot."""

    state: ResearchStateContent = Field(
        description=(
            "The complete current state. This call replaces the previous file, "
            "so include everything that is still true."
        ),
    )


class ResearchStateObservation(Observation):
    path: str
    updated_at: str
    bytes: int

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [TextContent(text=json.dumps(self.model_dump(mode="json"), sort_keys=True))]


class _UpdateResearchStateExecutor(
    ToolExecutor[UpdateResearchStateAction, ResearchStateObservation]
):
    def __init__(self, path: Path):
        self.path = path

    def __call__(
        self, action: UpdateResearchStateAction, conversation=None
    ) -> ResearchStateObservation:
        state = save_research_state(self.path, action.state)
        return ResearchStateObservation(
            path=str(self.path),
            updated_at=state.updated_at,
            bytes=len(render_research_state(state).encode()),
        )


class UpdateResearchStateTool(
    ToolDefinition[UpdateResearchStateAction, ResearchStateObservation]
):
    name = "update_research_state"

    @classmethod
    def create(
        cls, conv_state=None, *, state_dir: str | Path
    ) -> Sequence[ToolDefinition]:
        return [
            cls(
                description=(
                    "Replace the compact research state the harness keeps for you: "
                    "current best with identifiers, lanes in flight with owners, "
                    "deadlines and kill criteria, bars, budget used and remaining, "
                    "standing operator rules, open questions and recent decisions. "
                    "The harness re-sends this file verbatim after every context "
                    "compaction, so it is what you will remember. Update it after "
                    "every merge, kill, assignment, operator instruction or budget "
                    f"change; keep it under {MAX_RESEARCH_STATE_BYTES} bytes."
                ),
                action_type=UpdateResearchStateAction,
                observation_type=ResearchStateObservation,
                annotations=ToolAnnotations(
                    title="Update research state",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_UpdateResearchStateExecutor(
                    research_state_path(Path(state_dir))
                ),
            )
        ]
