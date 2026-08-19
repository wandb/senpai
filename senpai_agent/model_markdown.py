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


def task_table(tasks: Sequence[Mapping[str, object]]) -> str:
    """Render ordered agent task state without allowing table injection."""

    if not tasks:
        return "No delegated tasks."

    def cell(value: object) -> str:
        return inline_code(" ".join(str(value).split())).replace("|", r"\|")

    rows = [
        "| Task ID | Key | Agent | Model | Status |",
        "|---|---|---|---|---|",
    ]
    for task in tasks:
        key = task.get("key")
        rows.append(
            "| "
            + " | ".join(
                (
                    cell(task["task_id"]),
                    cell(key) if key is not None else "—",
                    cell(task["agent"]),
                    cell(task["model"]),
                    cell(task["status"]),
                )
            )
            + " |"
        )
    return "\n".join(rows)


def task_details(tasks: Sequence[Mapping[str, object]]) -> str:
    sections: list[str] = []
    for task in tasks:
        task_id = inline_code(task["task_id"])
        if task.get("result") is not None:
            sections.append(
                f"### Agent Response: {task_id}\n\n"
                f"{tagged_block('agent-response', task['result'])}"
            )
        if task.get("error") is not None:
            sections.append(
                f"### Agent Error: {task_id}\n\n"
                f"{tagged_block('agent-error', task['error'])}"
            )
    return "\n\n".join(sections)


def _section(lines: list[str], title: str, body: Sequence[str] | str) -> None:
    content = [body] if isinstance(body, str) else list(body)
    if not content:
        return
    lines.extend(["", f"### {title}", "", *content])


def _field(label: str, value: object, *, code: bool = True) -> str:
    rendered = inline_code(value) if code else markdown_text(value)
    return f"- {label}: {rendered}"


def _pull_request(payload: Mapping[str, object]) -> str | None:
    number = payload.get("number")
    url = payload.get("pr_url", payload.get("url"))
    if number is None:
        return None
    if url is not None:
        return f"- Pull request: {markdown_link(f'#{number}', url)}"
    return f"- Pull request: #{number}"


def _additional_data(
    lines: list[str],
    payload: Mapping[str, object],
    consumed: set[str],
) -> None:
    additional = {
        key: value
        for key, value in payload.items()
        if key not in consumed and key not in _ROUTING_FIELDS
    }
    if additional:
        _section(lines, "Additional Data", json_block(additional))


def _student_assignment(payload: Mapping[str, object]) -> str:
    consumed = {
        "assignment_id",
        "base_ref",
        "base_sha",
        "blockers",
        "head_ref",
        "head_sha",
        "number",
        "revision_id",
        "url",
    }
    lines = ["## Student Assignment", "", "This assignment revision is active."]
    identity = []
    for key, label in (("assignment_id", "Assignment"), ("revision_id", "Revision")):
        if key in payload:
            identity.append(_field(label, payload[key]))
    if pull := _pull_request(payload):
        identity.append(pull)
    _section(lines, "Identity", identity)

    repository = []
    for key, label in (
        ("base_ref", "Base"),
        ("base_sha", "Base commit"),
        ("head_ref", "Assignment branch"),
        ("head_sha", "Remote head"),
    ):
        if key in payload:
            repository.append(_field(label, payload[key]))
    _section(lines, "Repository State", repository)

    if "blockers" in payload:
        blockers = list(payload["blockers"])  # type: ignore[arg-type]
        _section(
            lines,
            "Blockers",
            [f"- {inline_code(blocker)}" for blocker in blockers] or ["None."],
        )
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _malformed_assignment(payload: Mapping[str, object]) -> str:
    consumed = {"error", "head_ref", "head_sha", "number", "url"}
    lines = ["## Malformed Student Assignment"]
    details = []
    if pull := _pull_request(payload):
        details.append(pull)
    for key, label in (("head_ref", "Branch"), ("head_sha", "Head")):
        if key in payload:
            details.append(_field(label, payload[key]))
    _section(lines, "Repository State", details)
    if "error" in payload:
        _section(
            lines,
            "Assignment Error:",
            tagged_block("assignment-error", payload["error"]),
        )
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _student_pr_feedback(payload: Mapping[str, object]) -> str:
    consumed = {
        "assignment_id",
        "author",
        "author_association",
        "base_ref",
        "base_sha",
        "created_at",
        "feedback_id",
        "feedback_type",
        "feedback_url",
        "full_message_instruction",
        "head_ref",
        "head_sha",
        "line",
        "message",
        "message_truncated",
        "number",
        "path",
        "pr_url",
        "revision_id",
        "state",
    }
    lines = ["## PR Feedback"]
    if "message" in payload:
        lines.extend(["", tagged_block("pr-feedback", payload["message"])])

    if "path" in payload:
        location = _field("File", payload["path"])
        if "line" in payload:
            location += f" at line {inline_code(payload['line'])}"
        _section(lines, "Location", location)
    if "state" in payload:
        _section(lines, "Review State", inline_code(payload["state"]))

    assignment = []
    for key, label in (("assignment_id", "Assignment"), ("revision_id", "Revision")):
        if key in payload:
            assignment.append(_field(label, payload[key]))
    if pull := _pull_request(payload):
        assignment.append(pull)
    _section(lines, "Assignment", assignment)

    repository = []
    for key, label in (
        ("base_ref", "Base"),
        ("base_sha", "Base commit"),
        ("head_ref", "Assignment branch"),
        ("head_sha", "Remote head"),
    ):
        if key in payload:
            repository.append(_field(label, payload[key]))
    _section(lines, "Repository State", repository)

    source = []
    for key, label in (
        ("feedback_type", "Type"),
        ("feedback_id", "Feedback ID"),
        ("author", "Author"),
        ("author_association", "Author association"),
        ("created_at", "Created"),
    ):
        if key in payload:
            source.append(_field(label, payload[key]))
    if "feedback_url" in payload:
        source.append(
            "- Feedback URL (`feedback_url`): "
            f"{markdown_link('Open feedback', payload['feedback_url'])}"
        )
    _section(lines, "Source", source)
    if payload.get("message_truncated") and "full_message_instruction" in payload:
        _section(
            lines,
            "Full Feedback",
            markdown_text(payload["full_message_instruction"]),
        )
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _human_issue(payload: Mapping[str, object]) -> str:
    consumed = {
        "author",
        "created_at",
        "human_message_id",
        "message",
        "number",
        "title",
        "url",
    }
    lines = ["## Human Issue"]
    if "message" in payload:
        _section(
            lines,
            "Human Message:",
            tagged_block("human-message", payload["message"]),
        )
    source = []
    if "number" in payload:
        number = payload["number"]
        if "url" in payload:
            source.append(
                f"- Issue: {markdown_link(f'#{number}', payload['url'])}"
            )
        else:
            source.append(f"- Issue: #{number}")
    for key, label in (
        ("title", "Title"),
        ("human_message_id", "Message ID"),
        ("author", "Author"),
        ("created_at", "Created"),
    ):
        if key in payload:
            source.append(_field(label, payload[key], code=key != "title"))
    _section(lines, "Source", source)
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _student_assignment_comment(payload: Mapping[str, object]) -> str:
    consumed = {
        "assignment_id",
        "comment_id",
        "content_digest",
        "message",
        "number",
        "pr_url",
        "revision_id",
        "student",
    }
    lines = ["## Student Progress Comment"]
    if "message" in payload:
        _section(
            lines,
            "Student Message:",
            tagged_block("student-message", payload["message"]),
        )
    assignment = []
    for key, label in (
        ("student", "Student"),
        ("assignment_id", "Assignment"),
        ("revision_id", "Revision"),
        ("comment_id", "Comment ID"),
    ):
        if key in payload:
            assignment.append(_field(label, payload[key]))
    if pull := _pull_request(payload):
        assignment.append(pull)
    _section(lines, "Assignment", assignment)
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _review_ready(payload: Mapping[str, object]) -> str:
    consumed = {"assignment_id", "head_ref", "head_sha", "number", "revision_id", "url"}
    title = (
        "## Experiment Ready for Review"
        if "assignment_id" in payload
        else "## Pull Request Ready for Review"
    )
    lines = [title]
    details = []
    if pull := _pull_request(payload):
        details.append(pull)
    for key, label in (
        ("assignment_id", "Assignment"),
        ("revision_id", "Revision"),
        ("head_ref", "Branch"),
        ("head_sha", "Result head"),
    ):
        if key in payload:
            details.append(_field(label, payload[key]))
    _section(lines, "Review Target", details)
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _advisor_action(payload: Mapping[str, object]) -> str:
    consumed = {"head_ref", "head_sha", "number", "reasons", "url"}
    lines = ["## Advisor Action Required"]
    details = []
    if pull := _pull_request(payload):
        details.append(pull)
    for key, label in (("head_ref", "Branch"), ("head_sha", "Head")):
        if key in payload:
            details.append(_field(label, payload[key]))
    _section(lines, "Pull Request", details)
    if "reasons" in payload:
        reasons = list(payload["reasons"])  # type: ignore[arg-type]
        _section(
            lines,
            "Reasons",
            [f"- {inline_code(reason)}" for reason in reasons] or ["None."],
        )
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _student_available(payload: Mapping[str, object]) -> str:
    lines = [
        f"## Student available for assignment: {inline_code(payload['student'])}",
        "",
        f"{inline_code(payload['student'])} has no open "
        f"{inline_code('status:wip')} or {inline_code('status:review')} assignment.",
    ]
    _additional_data(lines, payload, {"student"})
    return "\n".join(lines)


def _duplicate_assignment(payload: Mapping[str, object]) -> str:
    lines = ["## Student Has Multiple Active Assignments"]
    if "student" in payload:
        lines.extend(["", f"Student: {inline_code(payload['student'])}"])
    if "pull_requests" in payload:
        pull_requests = list(payload["pull_requests"])  # type: ignore[arg-type]
        _section(
            lines,
            "Pull Requests",
            [f"- PR #{number}" for number in pull_requests] or ["None."],
        )
    _additional_data(lines, payload, {"student", "pull_requests"})
    return "\n".join(lines)


def _research_base_changed(payload: Mapping[str, object]) -> str:
    consumed = {
        "assignment_id",
        "base_ref",
        "compare_url",
        "current_base_sha",
        "head_ref",
        "head_sha",
        "number",
        "required_base_sha",
        "revision_id",
        "student",
        "url",
    }
    lines = ["## Research Base Changed"]
    assignment = []
    for key, label in (
        ("assignment_id", "Assignment"),
        ("revision_id", "Revision"),
        ("student", "Student"),
        ("base_ref", "Base branch"),
        ("head_ref", "Assignment branch"),
        ("head_sha", "Pull request head"),
    ):
        if key in payload:
            assignment.append(_field(label, payload[key]))
    if pull := _pull_request(payload):
        assignment.append(pull)
    _section(lines, "Assignment", assignment)

    base_state = []
    for key, label in (
        ("required_base_sha", "Assignment-required base"),
        ("current_base_sha", "Current base"),
    ):
        if key in payload:
            base_state.append(_field(label, payload[key]))
    _section(lines, "Base State", base_state)
    if "compare_url" in payload:
        lines.extend(
            ["", markdown_link("Compare the base commits", payload["compare_url"])]
        )
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _training_monitor(payload: Mapping[str, object]) -> str:
    consumed = {"conversation_id", "reason", "signal", "summary", "training_id"}
    signal = payload.get("signal")
    signal_value = signal if isinstance(signal, Mapping) else {}
    signal_kind = signal_value.get("kind")
    title = {
        "metric_gate": "## Training Metric Gate",
        "metric_stale": "## Training Metric Is Stale",
        "monitor_error": "## Training Monitor Error",
        "training_status": "## Training Status",
    }.get(signal_kind, "## Training Monitor Signal")
    lines = [title]
    status = []
    training_id = payload.get("training_id", signal_value.get("training_id"))
    if training_id is not None:
        status.append(_field("Training ID", training_id))
    for key, label in (
        ("metric", "Metric"),
        ("value", "Value"),
        ("state", "State"),
        ("hard_failure", "Hard failure"),
    ):
        if key not in signal_value:
            continue
        value = signal_value[key]
        if key == "hard_failure":
            status.append(_field(label, yes_no(value), code=False))
        elif value is None:
            status.append(_field(label, "Unavailable", code=False))
        else:
            status.append(_field(label, value))
    _section(lines, "Status", status)
    detail = signal_value.get("detail", payload.get("summary"))
    if detail is not None:
        _section(lines, "Signal", tagged_block("monitor-signal", detail))
    if "reason" in payload:
        _section(lines, "Reason", markdown_text(payload["reason"]))
    signal_additional = {
        key: value
        for key, value in signal_value.items()
        if key
        not in {
            "dedupe_key",
            "detail",
            "hard_failure",
            "kind",
            "metric",
            "state",
            "training_id",
            "value",
        }
    }
    additional_payload = {
        key: value
        for key, value in payload.items()
        if key not in consumed and key not in _ROUTING_FIELDS
    }
    if signal_additional or additional_payload:
        _section(
            lines,
            "Additional Data",
            json_block({**additional_payload, **signal_additional}),
        )
    return "\n".join(lines)


def _workspace_diverged(payload: Mapping[str, object]) -> str:
    consumed = {
        "base_ref",
        "base_sha",
        "current_branch",
        "expected_remote_head",
        "head_ref",
        "instructions",
        "preserved_local_head",
        "worktree_fingerprint",
    }
    lines = [
        "## Workspace Requires Manual Reconciliation",
        "",
        (
            "Senpai preserved the local workspace without resetting or "
            "discarding anything."
        ),
    ]
    state = []
    for key, label in (
        ("head_ref", "Expected branch"),
        ("current_branch", "Current branch"),
        ("expected_remote_head", "Expected remote head"),
        ("preserved_local_head", "Preserved local head"),
        ("base_ref", "Base"),
        ("base_sha", "Base commit"),
    ):
        value = payload.get(key)
        if value is not None:
            state.append(_field(label, value))
    _section(lines, "Preserved State", state)
    if "instructions" in payload:
        _section(lines, "Required Action", markdown_text(payload["instructions"]))
    _additional_data(lines, payload, consumed)
    return "\n".join(lines)


def _agent_response(payload: Mapping[str, object], *, failed: bool) -> str:
    consumed = {"error", "result", "task", "task_id"}
    lines = ["## Delegated Task Failed" if failed else "## Delegated Task Completed"]
    if "task_id" in payload:
        lines.extend(["", _field("Task ID", payload["task_id"])])
    if failed:
        _section(lines, "Agent Error:", tagged_block("agent-error", payload["error"]))
    elif "result" in payload:
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
    _additional_data(lines, payload, consumed)
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


def render_legacy_event_prompt(kind: str, payload: Mapping[str, object]) -> str:
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
