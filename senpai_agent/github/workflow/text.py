"""Render and update human-readable workflow comments."""

import re
from typing import Literal

from senpai_agent.github.workflow.errors import WorkflowPreconditionError
from senpai_agent.models import AssignmentRecord, render_assignment_marker


_ROLE_COMMENT_PREFIX = re.compile(
    r"^[ \t]*(?:ADVISOR|STUDENT(?: [^:\s]+)?):[ \t]*",
    re.MULTILINE,
)


def marker_body(marker: str, content: str) -> str:
    content = _quote_senpai_marker_lines(content.strip())
    if not content:
        raise ValueError("marker comment content must not be empty")
    body = f"{marker}\n\n{content}"
    _validate_marker(marker, body)
    return body


def _quote_senpai_marker_lines(content: str) -> str:
    return "\n".join(
        f"> {line}" if line.lstrip().startswith("<!-- senpai-") else line
        for line in content.splitlines()
    )


def role_prefixed_comment(
    body: str,
    role: Literal["advisor", "student"],
) -> str:
    marker, separator, content = body.partition("\n\n")
    if not (separator and marker.startswith("<!-- senpai-")):
        marker, separator, content = "", "", body
    content = _quote_senpai_marker_lines(_ROLE_COMMENT_PREFIX.sub("", content))
    return f"{marker}{separator}{role.upper()}: {content}"


def replace_assignment_marker(
    body: str,
    assignment: AssignmentRecord,
) -> str:
    replacement = render_assignment_marker(assignment)
    lines = body.splitlines()
    indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("<!-- senpai-assignment:")
    ]
    if len(indexes) != 1:
        raise WorkflowPreconditionError(
            "pull request must contain exactly one assignment marker"
        )
    lines[indexes[0]] = replacement
    return "\n".join(lines)


def _validate_marker(marker: str, body: str) -> None:
    if (
        "\n" in marker
        or "\r" in marker
        or not marker.startswith("<!-- senpai-")
        or not marker.endswith("-->")
    ):
        raise ValueError("marker must be one hidden Senpai marker")
    protocol_lines = [
        line
        for line in body.splitlines()
        if line.lstrip().startswith("<!-- senpai-")
    ]
    if protocol_lines != [marker]:
        raise ValueError("comment body must contain exactly its intended marker")
