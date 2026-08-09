"""Typed terminal experiment results and their GitHub marker format."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


_NonEmptyString = Annotated[str, Field(min_length=1)]


class AssignmentKey(Contract):
    """Exact assignment revision and result commit represented by a result."""

    repo: Annotated[
        str,
        Field(
            min_length=1,
            description="Repository containing the assignment, in owner/name form.",
        ),
    ]
    pr_number: int = Field(
        gt=0,
        description="GitHub pull request number for the assignment.",
    )
    assignment_id: Annotated[
        str,
        Field(
            min_length=1,
            description="Stable identifier shared by every revision of the assignment.",
        ),
    ]
    revision_id: Annotated[
        str,
        Field(
            min_length=1,
            description="Exact assignment revision that produced this result.",
        ),
    ]
    expected_head_sha: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Exact local result commit to publish. Must equal "
                "ExperimentResult.commit_sha; this is not the current remote "
                "branch SHA."
            ),
        ),
    ]
    student: Annotated[
        str,
        Field(min_length=1, description="Student assigned to the experiment."),
    ]


class AssignmentRecord(Contract):
    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    assignment_id: _NonEmptyString
    revision_id: _NonEmptyString
    student: _NonEmptyString
    base_ref: _NonEmptyString
    base_sha: _NonEmptyString
    head_ref: _NonEmptyString
    head_sha: _NonEmptyString


class RevisionRecord(Contract):
    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    pr_number: int = Field(gt=0)
    assignment_id: _NonEmptyString
    revision_id: _NonEmptyString
    requested_head_sha: _NonEmptyString


class AssignmentFeedbackRecord(Contract):
    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    pr_number: int = Field(gt=0)
    assignment_id: _NonEmptyString
    revision_id: _NonEmptyString
    feedback_id: _NonEmptyString


class ResearchBaseAcceptanceRecord(Contract):
    """Durable approval of one exact result against a changed research base."""

    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    pr_number: int = Field(gt=0)
    assignment_id: _NonEmptyString
    revision_id: _NonEmptyString
    result_head_sha: _NonEmptyString
    result_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluated_base_sha: _NonEmptyString
    base_ref: _NonEmptyString
    accepted_base_sha: _NonEmptyString


class DispositionRecord(Contract):
    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    pr_number: int = Field(gt=0)
    assignment_id: _NonEmptyString
    head_sha: _NonEmptyString


class ResultStatus(StrEnum):
    """Terminal outcome reported for an experiment."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    CANCELLED = "cancelled"


class WandbRunRef(Contract):
    run_id: Annotated[
        str,
        Field(min_length=1, description="W&B run identifier."),
    ]
    url: Annotated[
        str,
        Field(min_length=1, description="Direct URL to the W&B run."),
    ]
    state: Literal["finished", "failed", "crashed", "killed"] = Field(
        description="Terminal state reported by W&B for the run."
    )


class MetricComparison(Contract):
    name: Annotated[
        str,
        Field(min_length=1, description="Name of the measured metric."),
    ]
    direction: Literal["minimize", "maximize"] = Field(
        description="Whether lower or higher values are better."
    )
    baseline: float | None = Field(
        default=None,
        description="Baseline value, or null when no baseline was measured.",
    )
    candidate: float = Field(description="Candidate value produced by the experiment.")
    delta: float | None = Field(
        default=None,
        description="Candidate-minus-baseline change, or null when unavailable.",
    )


class ExperimentResult(Contract):
    """Complete terminal evidence published by submit_experiment_result."""

    schema_version: Literal[1] = Field(
        default=1,
        description="Schema version for the typed terminal result.",
    )
    assignment: AssignmentKey = Field(
        description=(
            "Assignment identity for this result. assignment.expected_head_sha "
            "must equal commit_sha and the submit_experiment_result commit."
        )
    )
    status: ResultStatus = Field(
        description="Terminal outcome of the experiment."
    )
    hypothesis: Annotated[
        str,
        Field(min_length=1, description="Hypothesis tested by the experiment."),
    ]
    summary: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4_000,
            description="Concise evidence-backed conclusion from the experiment.",
        ),
    ]
    runs: tuple[WandbRunRef, ...] = Field(
        description="W&B runs that provide evidence for the result."
    )
    primary_metric: MetricComparison | None = Field(
        default=None,
        description="Primary metric comparison, or null when none was measured.",
    )
    commit_sha: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Local result commit to publish. Must equal "
                "assignment.expected_head_sha."
            ),
        ),
    ]


class ResultMarkerError(ValueError):
    """A result marker line is malformed, unsupported, or schema-invalid."""


_RESULT_PREFIX = "<!-- senpai-result:"
_RESULT_MARKER = re.compile(
    r"<!-- senpai-result:v(?P<version>[0-9]+) "
    r"(?P<payload>\{.*\}) -->"
)
_ASSIGNMENT_PREFIX = "<!-- senpai-assignment:"
_ASSIGNMENT_MARKER = re.compile(
    r"<!-- senpai-assignment:v(?P<version>[0-9]+) "
    r"(?P<payload>\{.*\}) -->"
)
_ASSIGNMENT_FEEDBACK_PREFIX = "<!-- senpai-assignment-feedback:"
_ASSIGNMENT_FEEDBACK_MARKER = re.compile(
    r"<!-- senpai-assignment-feedback:v(?P<version>[0-9]+) "
    r"(?P<payload>\{.*\}) -->"
)
_RESEARCH_BASE_ACCEPTANCE_PREFIX = "<!-- senpai-research-base-acceptance:"
_RESEARCH_BASE_ACCEPTANCE_MARKER = re.compile(
    r"<!-- senpai-research-base-acceptance:v(?P<version>[0-9]+) "
    r"(?P<payload>\{.*\}) -->"
)


def _marker_payload(value: Contract) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).replace(">", r"\u003e")


def experiment_result_digest(result: ExperimentResult) -> str:
    """Return a canonical digest of every typed terminal-result field."""

    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be an ExperimentResult")
    payload = json.dumps(
        result.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(payload).hexdigest()


def render_assignment_marker(assignment: AssignmentRecord) -> str:
    return f"<!-- senpai-assignment:v1 {_marker_payload(assignment)} -->"


def authoritative_marker_line(comment_body: str) -> str:
    """Return the only logical line eligible to carry a trusted marker."""

    return next(iter(comment_body.splitlines()), "")


def render_revision_marker(revision: RevisionRecord) -> str:
    return f"<!-- senpai-revision:v1 {_marker_payload(revision)} -->"


def render_assignment_feedback_marker(feedback: AssignmentFeedbackRecord) -> str:
    return f"<!-- senpai-assignment-feedback:v1 {_marker_payload(feedback)} -->"


def parse_assignment_feedback_markers(
    body: str,
) -> tuple[AssignmentFeedbackRecord, ...]:
    feedback: list[AssignmentFeedbackRecord] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.startswith(_ASSIGNMENT_FEEDBACK_PREFIX):
            continue
        marker = _ASSIGNMENT_FEEDBACK_MARKER.fullmatch(line)
        if marker is None or marker.group("version") != "1":
            raise ValueError(
                "malformed or unsupported Senpai assignment feedback marker "
                f"on line {line_number}"
            )
        try:
            feedback.append(
                AssignmentFeedbackRecord.model_validate_json(marker.group("payload"))
            )
        except (ValidationError, ValueError) as error:
            raise ValueError(
                f"invalid Senpai assignment feedback marker on line {line_number}"
            ) from error
    return tuple(feedback)


def render_research_base_acceptance_marker(
    acceptance: ResearchBaseAcceptanceRecord,
) -> str:
    return (
        "<!-- senpai-research-base-acceptance:v1 "
        f"{_marker_payload(acceptance)} -->"
    )


def parse_research_base_acceptance_markers(
    body: str,
) -> tuple[ResearchBaseAcceptanceRecord, ...]:
    acceptances: list[ResearchBaseAcceptanceRecord] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.startswith(_RESEARCH_BASE_ACCEPTANCE_PREFIX):
            continue
        marker = _RESEARCH_BASE_ACCEPTANCE_MARKER.fullmatch(line)
        if marker is None or marker.group("version") != "1":
            raise ValueError(
                "malformed or unsupported Senpai research-base acceptance marker "
                f"on line {line_number}"
            )
        try:
            acceptances.append(
                ResearchBaseAcceptanceRecord.model_validate_json(
                    marker.group("payload")
                )
            )
        except (ValidationError, ValueError) as error:
            raise ValueError(
                "invalid Senpai research-base acceptance marker on line "
                f"{line_number}"
            ) from error
    return tuple(acceptances)


def render_disposition_marker(disposition: DispositionRecord) -> str:
    return f"<!-- senpai-disposition:v1 {_marker_payload(disposition)} -->"


def parse_assignment_markers(body: str) -> tuple[AssignmentRecord, ...]:
    assignments: list[AssignmentRecord] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.startswith(_ASSIGNMENT_PREFIX):
            continue
        marker = _ASSIGNMENT_MARKER.fullmatch(line)
        if marker is None or marker.group("version") != "1":
            raise ValueError(
                f"malformed or unsupported Senpai assignment marker on line "
                f"{line_number}"
            )
        try:
            assignments.append(
                AssignmentRecord.model_validate_json(marker.group("payload"))
            )
        except (ValidationError, ValueError) as error:
            raise ValueError(
                f"invalid Senpai assignment marker on line {line_number}"
            ) from error
    return tuple(assignments)


def render_result_marker(result: ExperimentResult) -> str:
    return f"<!-- senpai-result:v1 {_marker_payload(result)} -->"


def _quote_senpai_marker_lines(value: str) -> str:
    return "\n".join(
        f"> {line}" if line.lstrip().startswith("<!-- senpai-") else line
        for line in value.splitlines()
    )


def render_result_comment(result: ExperimentResult) -> str:
    lines = [
        render_result_marker(result),
        "",
        f"Status: {result.status.value}",
        f"Commit: `{result.commit_sha}`",
        "",
        _quote_senpai_marker_lines(result.summary),
    ]
    if result.runs:
        lines.extend(
            [
                "",
                "W&B runs:",
                *(f"- {_quote_senpai_marker_lines(run.url)}" for run in result.runs),
            ]
        )
    return "\n".join(lines)


def parse_result_markers(comment_body: str) -> tuple[ExperimentResult, ...]:
    results: list[ExperimentResult] = []
    for line_number, line in enumerate(comment_body.splitlines(), start=1):
        if not line.startswith(_RESULT_PREFIX):
            continue

        marker = _RESULT_MARKER.fullmatch(line)
        if marker is None:
            raise ResultMarkerError(
                f"malformed senpai result marker on line {line_number}"
            )

        version = marker.group("version")
        if version != "1":
            raise ResultMarkerError(
                f"unknown senpai result marker version v{version} on line {line_number}"
            )

        try:
            payload = json.loads(marker.group("payload"))
        except json.JSONDecodeError as error:
            raise ResultMarkerError(
                f"invalid JSON in senpai result marker on line {line_number}"
            ) from error

        try:
            results.append(ExperimentResult.model_validate(payload))
        except ValidationError as error:
            raise ResultMarkerError(
                f"invalid senpai result payload on line {line_number}: {error}"
            ) from error

    return tuple(results)
