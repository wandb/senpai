# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Deterministic Markdown views of structured model-facing values."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from html import escape
from urllib.parse import quote

from senpai_agent.event_kinds import EVENT_KINDS, EventKind
from senpai_agent.json_values import canonical_json
from senpai_agent.text_values import utf8_text


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
    rf"<(?:(?:\s*/\s*)|\s*)(?:{'|'.join(sorted(_BOUNDARY_TAGS))})"
    r"(?=[\s/>]|$)(?:[^<>]*>)?",
    re.IGNORECASE,
)
_BACKTICK_RUN = re.compile(r"`+")
_TILDE_RUN = re.compile(r"~+")
_MAX_MARKDOWN_FENCE = 32
_MARKDOWN_ESCAPES = str.maketrans(
    {
        "\\": "\\\\",
        "`": "\\`",
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


def _single_line(value: object) -> str:
    return " ".join(utf8_text(value).split())


def _escape_boundary_tags(value: str) -> str:
    return _BOUNDARY_PATTERN.sub(
        lambda match: match.group().replace("<", "&lt;").replace(">", "&gt;"),
        value,
    )


def inline_code(value: object) -> str:
    """Render one opaque value inside a single-line code boundary."""

    text = _escape_boundary_tags(_single_line(value))
    if not text:
        return "—"
    longest_run = max(
        (len(match.group()) for match in _BACKTICK_RUN.finditer(text)),
        default=0,
    )
    if longest_run >= _MAX_MARKDOWN_FENCE:
        return f"<code>{escape(text, quote=False)}</code>"
    fence = "`" * (longest_run + 1)
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{padding}{text}{padding}{fence}"


def markdown_text(value: object) -> str:
    """Render one untrusted value as plain, single-line Markdown text."""

    return _escape_boundary_tags(_single_line(value)).translate(_MARKDOWN_ESCAPES)


def markdown_url(value: object) -> str:
    """Percent-encode control characters while retaining normal URL syntax."""

    return quote(
        utf8_text(value).strip(),
        safe="/:?#@$&'+,;=%",
    )


def markdown_link(label: str, url: object) -> str:
    safe_label = (
        _single_line(label)
        .replace("\\", "\\\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )
    return f"[{safe_label}](<{markdown_url(url)}>)"


def tagged_block(tag: str, value: object) -> str:
    """Preserve Markdown inside one provenance boundary, not an authority boundary."""

    if tag not in _BOUNDARY_TAGS:
        raise ValueError(f"invalid model-facing tag {tag!r}")
    content = _escape_boundary_tags(utf8_text(value))
    return f"<{tag}>\n\n{content}\n\n</{tag}>"


def json_block(value: object) -> str:
    payload = utf8_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    backtick_run = max(
        (len(match.group()) for match in _BACKTICK_RUN.finditer(payload)),
        default=0,
    )
    tilde_run = max(
        (len(match.group()) for match in _TILDE_RUN.finditer(payload)),
        default=0,
    )
    if backtick_run < _MAX_MARKDOWN_FENCE:
        fence = "`" * max(3, backtick_run + 1)
    elif tilde_run < _MAX_MARKDOWN_FENCE:
        fence = "~" * max(3, tilde_run + 1)
    else:
        return (
            '<pre><code class="language-json">\n'
            f"{escape(payload, quote=False)}\n"
            "</code></pre>"
        )
    return f"{fence}json\n{payload}\n{fence}"


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


def _repository_state(payload: Mapping[str, object]) -> list[str]:
    return [
        _field("Base", payload["base_ref"]),
        _field("Base commit", payload["base_sha"]),
        _field("Assignment branch", payload["head_ref"]),
        _field("Remote head", payload["head_sha"]),
    ]


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
        _repository_state(payload),
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
        _repository_state(payload),
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
            _field(
                "Live base to run against (`current_base_sha`)",
                payload["current_base_sha"],
            ),
            _field(
                "Assignment's original base (`required_base_sha`)",
                payload["required_base_sha"],
            ),
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
    }.get(signal_kind)
    if title is None:
        return (
            f"## Training Monitor Signal: {inline_code(signal_kind)}\n\n"
            f"{json_block(signal_value)}"
        )
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
            status.append(_field(label, str(value).lower()))
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
    EventKind.ADVISOR_ACTION: _advisor_action,
    EventKind.AGENT_ERROR: _agent_error,
    EventKind.AGENT_RESULT: _agent_result,
    EventKind.DUPLICATE_ASSIGNMENT: _duplicate_assignment,
    EventKind.HUMAN_ISSUE: _human_issue,
    EventKind.MALFORMED_ASSIGNMENT: _malformed_assignment,
    EventKind.RESEARCH_BASE_CHANGED: _research_base_changed,
    EventKind.REVIEW_READY: _review_ready,
    EventKind.STUDENT_ASSIGNMENT: _student_assignment,
    EventKind.STUDENT_ASSIGNMENT_COMMENT: _student_assignment_comment,
    EventKind.STUDENT_AVAILABLE_FOR_ASSIGNMENT: _student_available,
    EventKind.STUDENT_PR_FEEDBACK: _student_pr_feedback,
    EventKind.TRAINING_MONITOR: _training_monitor,
    EventKind.WORKSPACE_DIVERGED: _workspace_diverged,
}
if frozenset(_EVENT_RENDERERS) != EVENT_KINDS:
    raise RuntimeError("specialized event renderers do not match EventKind")


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


def canonical_event_identity(kind: str, payload: Mapping[str, object]) -> str:
    """Encode the complete model-relevant event independently of its display."""

    return canonical_json(
        {"kind": kind, "payload": visible_event_payload(payload)}
    )
