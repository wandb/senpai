# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Deterministic Markdown views of structured model-facing values."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import quote


EventRenderer = Callable[[Mapping[str, object]], str]

_ROUTING_FIELDS = frozenset({"parent_conversation_id"})
_BOUNDARY_TAGS = frozenset(
    {
        "agent-error",
        "agent-response",
        "assignment-error",
        "delegated-task",
        "human-message",
        "monitor-signal",
        "pr-feedback",
        "student-message",
        "training-error",
    }
)
_BOUNDARY_PATTERN = re.compile(
    rf"</?(?:{'|'.join(sorted(_BOUNDARY_TAGS))})(?:\s[^>]*)?\s*/?>",
    re.IGNORECASE,
)
_MARKDOWN_ESCAPES = str.maketrans(
    {
        "\\": "\\\\",
        "*": "\\*",
        "_": "\\_",
        "[": "\\[",
        "]": "\\]",
        "<": "\\<",
        ">": "\\>",
        "#": "\\#",
        "|": "\\|",
    }
)


def inline_code(value: object) -> str:
    """Render an opaque value without letting backticks break its boundary."""

    text = str(value)
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{padding}{text}{padding}{fence}"


def markdown_text(value: object) -> str:
    """Render one untrusted value as plain, single-line Markdown text."""

    return " ".join(str(value).split()).translate(_MARKDOWN_ESCAPES)


def markdown_url(value: object) -> str:
    """Percent-encode control characters while retaining normal URL syntax."""

    return quote(
        str(value).strip(),
        safe="/:?#@$&'+,;=%",
    )


def markdown_link(label: str, url: object) -> str:
    safe_label = (
        " ".join(label.split())
        .replace("\\", "\\\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )
    return f"[{safe_label}](<{markdown_url(url)}>)"


def tagged_block(tag: str, value: object) -> str:
    """Preserve Markdown inside one provenance boundary, not an authority boundary."""

    if tag not in _BOUNDARY_TAGS:
        raise ValueError(f"invalid model-facing tag {tag!r}")
    content = _BOUNDARY_PATTERN.sub(
        lambda match: match.group().replace("<", "&lt;").replace(">", "&gt;"),
        str(value),
    )
    return f"<{tag}>\n\n{content}\n\n</{tag}>"


def json_block(value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    longest_run = max((len(run) for run in re.findall(r"`+", payload)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}json\n{payload}\n{fence}"


def yes_no(value: object) -> str:
    return "Yes" if bool(value) else "No"


def _section(lines: list[str], title: str, body: Sequence[str] | str) -> None:
    content = [body] if isinstance(body, str) else list(body)
    lines.extend(["", f"### {title}", "", *content])


def _field(label: str, value: object, *, code: bool = True) -> str:
    rendered = inline_code(value) if code else markdown_text(value)
    return f"- {label}: {rendered}"


def _pull_request(payload: Mapping[str, object]) -> str:
    url = payload["pr_url"] if "pr_url" in payload else payload["url"]
    number = payload["number"]
    return f"- Pull request: {markdown_link(f'#{number}', url)}"


def _student_assignment(payload: Mapping[str, object]) -> str:
    lines = ["## Student Assignment", "", "This assignment revision is active."]
    _section(
        lines,
        "Identity",
        [
            _field("Assignment", payload["assignment_id"]),
            _field("Revision", payload["revision_id"]),
            _pull_request(payload),
        ],
    )
    _section(
        lines,
        "Repository State",
        [
            _field("Base", payload["base_ref"]),
            _field("Base commit", payload["base_sha"]),
            _field("Assignment branch", payload["head_ref"]),
            _field("Remote head", payload["head_sha"]),
        ],
    )
    blockers = list(payload["blockers"])  # type: ignore[arg-type]
    _section(
        lines,
        "Blockers",
        [f"- {inline_code(blocker)}" for blocker in blockers] or ["None."],
    )
    return "\n".join(lines)


def _malformed_assignment(payload: Mapping[str, object]) -> str:
    lines = ["## Malformed Student Assignment"]
    _section(
        lines,
        "Repository State",
        [
            _pull_request(payload),
            _field("Branch", payload["head_ref"]),
            _field("Head", payload["head_sha"]),
        ],
    )
    _section(
        lines,
        "Assignment Error:",
        tagged_block("assignment-error", payload["error"]),
    )
    return "\n".join(lines)


def _student_pr_feedback(payload: Mapping[str, object]) -> str:
    lines = [
        "## PR Feedback",
        "",
        tagged_block("pr-feedback", payload["message"]),
    ]

    feedback_type = payload["feedback_type"]
    if feedback_type == "inline_comment":
        location = _field("File", payload["path"])
        if "line" in payload:
            location += f" at line {inline_code(payload['line'])}"
        _section(lines, "Location", location)
    elif feedback_type == "review":
        _section(lines, "Review State", inline_code(payload["state"]))

    _section(
        lines,
        "Assignment",
        [
            _field("Assignment", payload["assignment_id"]),
            _field("Revision", payload["revision_id"]),
            _pull_request(payload),
        ],
    )
    _section(
        lines,
        "Repository State",
        [
            _field("Base", payload["base_ref"]),
            _field("Base commit", payload["base_sha"]),
            _field("Assignment branch", payload["head_ref"]),
            _field("Remote head", payload["head_sha"]),
        ],
    )
    _section(
        lines,
        "Source",
        [
            _field("Type", feedback_type),
            _field("Feedback ID", payload["feedback_id"]),
            _field("Author", payload["author"]),
            _field("Author association", payload["author_association"]),
            _field("Created", payload["created_at"]),
            "- Feedback URL (`feedback_url`): "
            f"{markdown_link('Open feedback', payload['feedback_url'])}",
        ],
    )
    if payload.get("message_truncated"):
        _section(
            lines,
            "Full Feedback",
            markdown_text(payload["full_message_instruction"]),
        )
    return "\n".join(lines)


def _human_issue(payload: Mapping[str, object]) -> str:
    lines = ["## Human Issue"]
    _section(
        lines,
        "Human Message:",
        tagged_block("human-message", payload["message"]),
    )
    number = payload["number"]
    _section(
        lines,
        "Source",
        [
            f"- Issue: {markdown_link(f'#{number}', payload['url'])}",
            _field("Title", payload["title"], code=False),
            _field("Message ID", payload["human_message_id"]),
            _field("Author", payload["author"]),
            _field("Created", payload["created_at"]),
        ],
    )
    return "\n".join(lines)


def _student_assignment_comment(payload: Mapping[str, object]) -> str:
    lines = ["## Student Progress Comment"]
    _section(
        lines,
        "Student Message:",
        tagged_block("student-message", payload["message"]),
    )
    _section(
        lines,
        "Assignment",
        [
            _field("Student", payload["student"]),
            _field("Assignment", payload["assignment_id"]),
            _field("Revision", payload["revision_id"]),
            _field("Comment ID", payload["comment_id"]),
            _pull_request(payload),
        ],
    )
    return "\n".join(lines)


def _review_ready(payload: Mapping[str, object]) -> str:
    title = (
        "## Experiment Ready for Review"
        if "assignment_id" in payload
        else "## Pull Request Ready for Review"
    )
    lines = [title]
    details = [_pull_request(payload)]
    if "assignment_id" in payload:
        details.extend(
            [
                _field("Assignment", payload["assignment_id"]),
                _field("Revision", payload["revision_id"]),
            ]
        )
    details.extend(
        [
            _field("Branch", payload["head_ref"]),
            _field("Result head", payload["head_sha"]),
        ]
    )
    _section(lines, "Review Target", details)
    return "\n".join(lines)


def _advisor_action(payload: Mapping[str, object]) -> str:
    lines = ["## Advisor Action Required"]
    _section(
        lines,
        "Pull Request",
        [
            _pull_request(payload),
            _field("Branch", payload["head_ref"]),
            _field("Head", payload["head_sha"]),
        ],
    )
    reasons = list(payload["reasons"])  # type: ignore[arg-type]
    _section(
        lines,
        "Reasons",
        [f"- {inline_code(reason)}" for reason in reasons],
    )
    return "\n".join(lines)


def _student_available(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            f"## Student available for assignment: {inline_code(payload['student'])}",
            "",
            f"{inline_code(payload['student'])} has no open "
            f"{inline_code('status:wip')} or "
            f"{inline_code('status:review')} assignment.",
        ]
    )


def _duplicate_assignment(payload: Mapping[str, object]) -> str:
    lines = [
        "## Student Has Multiple Active Assignments",
        "",
        f"Student: {inline_code(payload['student'])}",
    ]
    pull_requests = list(payload["pull_requests"])  # type: ignore[arg-type]
    _section(
        lines,
        "Pull Requests",
        [f"- PR #{number}" for number in pull_requests],
    )
    return "\n".join(lines)


def _research_base_changed(payload: Mapping[str, object]) -> str:
    lines = ["## Research Base Changed"]
    _section(
        lines,
        "Assignment",
        [
            _field("Assignment", payload["assignment_id"]),
            _field("Revision", payload["revision_id"]),
            _field("Student", payload["student"]),
            _field("Base branch", payload["base_ref"]),
            _field("Assignment branch", payload["head_ref"]),
            _field("Pull request head", payload["head_sha"]),
            _pull_request(payload),
        ],
    )
    _section(
        lines,
        "Base State",
        [
            _field("Assignment-required base", payload["required_base_sha"]),
            _field("Current base", payload["current_base_sha"]),
        ],
    )
    lines.extend(
        ["", markdown_link("Compare the base commits", payload["compare_url"])]
    )
    return "\n".join(lines)


def _training_monitor(payload: Mapping[str, object]) -> str:
    signal_value = payload["signal"]
    if not isinstance(signal_value, Mapping):
        raise TypeError("training monitor signal must be an object")
    signal_kind = signal_value["kind"]
    title = {
        "metric_gate": "## Training Metric Gate",
        "metric_stale": "## Training Metric Is Stale",
        "monitor_error": "## Training Monitor Error",
        "training_status": "## Training Status",
    }[signal_kind]
    lines = [title]
    status = [_field("Training ID", payload["training_id"])]
    for key, label in (
        ("metric", "Metric"),
        ("value", "Value"),
        ("state", "State"),
        ("hard_failure", "Hard failure"),
    ):
        value = signal_value[key]
        if key == "hard_failure":
            status.append(_field(label, yes_no(value), code=False))
        elif value is None:
            status.append(_field(label, "Unavailable", code=False))
        else:
            status.append(_field(label, value))
    _section(lines, "Status", status)
    _section(
        lines,
        "Signal",
        tagged_block("monitor-signal", signal_value["detail"]),
    )
    _section(lines, "Reason", markdown_text(payload["reason"]))
    return "\n".join(lines)


def _workspace_diverged(payload: Mapping[str, object]) -> str:
    lines = [
        "## Workspace Requires Manual Reconciliation",
        "",
        (
            "Senpai preserved the local workspace without resetting or "
            "discarding anything."
        ),
    ]
    state = [_field("Expected branch", payload["head_ref"])]
    current_branch = payload["current_branch"]
    if current_branch is not None:
        state.append(_field("Current branch", current_branch))
    state.extend(
        [
            _field("Expected remote head", payload["expected_remote_head"]),
            _field("Preserved local head", payload["preserved_local_head"]),
        ]
    )
    for key, label in (("base_ref", "Base"), ("base_sha", "Base commit")):
        value = payload[key]
        if value is not None:
            state.append(_field(label, value))
    _section(lines, "Preserved State", state)
    _section(lines, "Required Action", markdown_text(payload["instructions"]))
    return "\n".join(lines)


def _agent_response(payload: Mapping[str, object], *, failed: bool) -> str:
    lines = ["## Delegated Task Failed" if failed else "## Delegated Task Completed"]
    lines.extend(["", _field("Task ID", payload["task_id"])])
    if failed:
        _section(lines, "Agent Error:", tagged_block("agent-error", payload["error"]))
    else:
        _section(
            lines,
            "Agent Response:",
            tagged_block("agent-response", payload["result"]),
        )
    if "task" in payload:
        _section(
            lines,
            "Assigned Task (Provenance Only):",
            tagged_block("delegated-task", payload["task"]),
        )
    return "\n".join(lines)


def _agent_result(payload: Mapping[str, object]) -> str:
    return _agent_response(payload, failed=False)


def _agent_error(payload: Mapping[str, object]) -> str:
    return _agent_response(payload, failed=True)


_EVENT_RENDERERS: dict[str, EventRenderer] = {
    "advisor_action": _advisor_action,
    "agent_error": _agent_error,
    "agent_result": _agent_result,
    "duplicate_assignment": _duplicate_assignment,
    "human_issue": _human_issue,
    "malformed_assignment": _malformed_assignment,
    "research_base_changed": _research_base_changed,
    "review_ready": _review_ready,
    "student_assignment": _student_assignment,
    "student_assignment_comment": _student_assignment_comment,
    "student_available_for_assignment": _student_available,
    "student_pr_feedback": _student_pr_feedback,
    "training_monitor": _training_monitor,
    "workspace_diverged": _workspace_diverged,
}


def visible_event_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in payload.items() if key not in _ROUTING_FIELDS
    }


def render_event_prompt(kind: str, payload: Mapping[str, object]) -> str:
    """Render one structured event as stable, agent-readable Markdown."""

    visible = visible_event_payload(payload)
    renderer = _EVENT_RENDERERS.get(kind)
    if renderer is None:
        return f"## Senpai event: {inline_code(kind)}\n\n{json_block(visible)}"
    return renderer(visible)


def render_pre_markdown_event_prompt(
    kind: str,
    payload: Mapping[str, object],
) -> str:
    """Reproduce the pre-Markdown-v2 body for durable inbox compatibility."""

    visible = visible_event_payload(payload)
    if kind == "student_available_for_assignment":
        student = str(visible["student"])
        return (
            f"## Student available for assignment: `{student}`\n\n"
            f"`{student}` has no open `status:wip` or `status:review` assignment."
        )
    encoded = json.dumps(visible, sort_keys=True, separators=(",", ":"))
    return f"## {kind}\n\n{encoded}"


def canonical_event_identity(kind: str, payload: Mapping[str, object]) -> str:
    """Encode the complete model-relevant event independently of its display."""

    return json.dumps(
        {"kind": kind, "payload": visible_event_payload(payload)},
        sort_keys=True,
        separators=(",", ":"),
    )
