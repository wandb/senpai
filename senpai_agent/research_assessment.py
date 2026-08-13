"""Capability-free, enum-only research-review result collection."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Literal, Self, TypeAlias
from uuid import UUID

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from pydantic import ConfigDict


ResearchAssessmentDecision: TypeAlias = Literal[
    "aligned",
    "insufficient_evidence",
    "strategic_drift",
]
_DECISIONS = frozenset(
    {"aligned", "insufficient_evidence", "strategic_drift"}
)


class ResearchAssessmentSubmissionError(RuntimeError):
    """The isolated assessor did not submit exactly one valid decision."""


class ResearchAssessmentAction(Action):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ResearchAssessmentDecision


class ResearchAssessmentObservation(Observation):
    accepted: bool
    submission_count: int
    error_code: Literal["session_unavailable", "duplicate_submission"] | None = None

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=json.dumps(
                    self.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ]


_LOCK = threading.Lock()
_SUBMISSIONS: dict[str, list[ResearchAssessmentDecision]] = {}


def _canonical_assessment_id(assessment_id: str) -> str:
    try:
        parsed = UUID(hex=assessment_id)
    except (AttributeError, ValueError) as error:
        raise ValueError("assessment id must be a UUID hex value") from error
    if parsed.hex != assessment_id.lower():
        raise ValueError("assessment id must use canonical UUID hex form")
    return parsed.hex


def begin_research_assessment(assessment_id: str) -> None:
    key = _canonical_assessment_id(assessment_id)
    with _LOCK:
        if key in _SUBMISSIONS:
            raise ResearchAssessmentSubmissionError(
                "research assessment session already exists"
            )
        _SUBMISSIONS[key] = []


def record_research_assessment(
    assessment_id: str,
    decision: ResearchAssessmentDecision,
) -> ResearchAssessmentObservation:
    key = _canonical_assessment_id(assessment_id)
    if decision not in _DECISIONS:
        raise ValueError("unsupported research assessment decision")
    with _LOCK:
        submissions = _SUBMISSIONS.get(key)
        if submissions is None:
            return ResearchAssessmentObservation(
                accepted=False,
                submission_count=0,
                error_code="session_unavailable",
            )
        submissions.append(decision)
        count = len(submissions)
    return ResearchAssessmentObservation(
        accepted=count == 1,
        submission_count=count,
        error_code="duplicate_submission" if count > 1 else None,
    )


def finish_research_assessment(
    assessment_id: str,
) -> ResearchAssessmentDecision:
    key = _canonical_assessment_id(assessment_id)
    with _LOCK:
        submissions = _SUBMISSIONS.pop(key, None)
    if submissions is None or len(submissions) != 1:
        raise ResearchAssessmentSubmissionError(
            "research assessment requires exactly one typed submission"
        )
    return submissions[0]


def cancel_research_assessment(assessment_id: str) -> None:
    key = _canonical_assessment_id(assessment_id)
    with _LOCK:
        _SUBMISSIONS.pop(key, None)


class _ResearchAssessmentExecutor(
    ToolExecutor[ResearchAssessmentAction, ResearchAssessmentObservation]
):
    def __init__(self, assessment_id: str):
        self.assessment_id = _canonical_assessment_id(assessment_id)

    def __call__(
        self,
        action: ResearchAssessmentAction,
        conversation: object | None = None,
    ) -> ResearchAssessmentObservation:
        return record_research_assessment(self.assessment_id, action.decision)


class ResearchAssessmentTool(
    ToolDefinition[ResearchAssessmentAction, ResearchAssessmentObservation]
):
    name = "submit_research_assessment"

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        *,
        assessment_id: str,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Submit exactly one closed research-alignment decision. "
                    "This records no explanation and performs no campaign action."
                ),
                action_type=ResearchAssessmentAction,
                observation_type=ResearchAssessmentObservation,
                annotations=ToolAnnotations(
                    title="Submit research assessment",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=_ResearchAssessmentExecutor(assessment_id),
            )
        ]
