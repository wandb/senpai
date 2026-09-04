# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Model-facing instruction templates assembled by Python."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping


CONTEXT_RECOVERY_PROMPT = """# Conversation context recovery

The previous model-visible conversation branch exhausted or corrupted its context. Its complete raw trace and workspace are preserved, but the active model context was reset. Inspect preserved state as needed, and verify any interrupted action before relying on it.

# Current actionable state

{{CURRENT_PROMPT}}"""

INITIAL_CONTROLLER_PROMPT = """{{FULL_PROMPT}}

Current time (UTC): {{CURRENT_TIME}}

# Current GitHub state

Actionable events follow as separately tracked messages."""

CONTINUATION_CONTROLLER_PROMPT = """Continue the {{ROLE}} loop. Current time (UTC): {{CURRENT_TIME}}. Actionable GitHub events follow as separately tracked messages."""

OPERATOR_INSTRUCTIONS_PROMPT = """# Additional operator instructions

{{INSTRUCTIONS}}"""

SENPAI_SYSTEM_INSTRUCTIONS_PROMPT = """<SENPAI_HARNESS>
# Senpai harness

{{HARNESS}}
</SENPAI_HARNESS>

<SENPAI_ROLE>
# Senpai role

{{ROLE}}
</SENPAI_ROLE>

<SENPAI_PROGRAM>
{{PROGRAM}}
</SENPAI_PROGRAM>

<SENPAI_LAUNCH_CONTEXT>
{{LAUNCH}}
</SENPAI_LAUNCH_CONTEXT>"""

PROGRAM_SYSTEM_PROMPT = """# program.md - {{PROGRAM_PATH}}

{{PROGRAM_CONTENT}}"""

DELEGATED_SEARCH_MODE_PROMPT = """Search mode: {{SEARCH_MODE}}

{{ASSIGNMENT}}"""

DELEGATED_TASK_PROMPT = """# Delegated task

You are a fresh Senpai subagent. Perform only the assigned task and return a concise, evidence-linked report to the parent.

{{ASSIGNMENT}}"""

DELEGATED_TASK_WITH_CONTEXT_PROMPT = """# Delegated task with parent context

The JSON below is the complete model-visible parent context at delegation time. Use it as evidence, perform only the assigned task, and return a concise, evidence-linked report.

<parent_context_json>
{{PARENT_CONTEXT_JSON}}
</parent_context_json>

{{ASSIGNMENT}}"""

DELEGATED_RESULT_SUMMARY_PROMPT = """Your response is too large to send directly to your parent that requested this and risks blowing up its context window. Instead SENPAI stored your complete response at:

{{RESULT_PATH}}

This file is visible to your parent and it can read some or all of it as needed. Your task now is to generate a fresh, shorter, summary response of your findings / suggestions etc to the parent. Use only the response you just produced to generate your summary. Do not perform more research, edit files, or call tools. Return approximately 1,500 tokens of plain-language, high-signal, actionable text.

Response guidelines:

- Lead with the conclusion or recommendation.
- Include only the strongest evidence and precise paths or identifiers.
- State any suggested next actions.
- State any material risks and unresolved questions (if any).
- Refer to sections of the main response file to support your statements, as needed.
- Use clear, structured markdown formatting
- Include a link to the full response file.

Do not reproduce long excerpts or the full-response path."""

RECOVERED_ACTION_PROMPT = """Senpai restarted before this action completed. Inspect the preserved workspace and rerun it explicitly only if it is still needed."""

LOCAL_EVENT_PROMPT = """# Senpai event: {{KIND}}

Observed at (UTC): {{OBSERVED_AT}}

```json
{{PAYLOAD}}
```"""

EVENT_PROMPT = """## {{KIND}}

{{PAYLOAD}}"""

STUDENT_AVAILABLE_FOR_ASSIGNMENT_PROMPT = """## Student available for assignment: `{{STUDENT}}`

`{{STUDENT}}` has no open `status:wip` or `status:review` assignment."""

WORKSPACE_DIVERGENCE_PROMPT = """The workspace cannot be reconciled automatically because local assignment history diverged or dirty work belongs to another checkout. Senpai preserved every local commit and dirty file without changing the checkout. Inspect and reconcile it explicitly; do not reset or discard local work."""

TRUNCATED_FEEDBACK_PROMPT = """Open feedback_url to read the omitted text."""

MONITOR_TRAINING_STARTED_PROMPT = """Training {{TRAINING_ID}} is durably monitored. You may finish this turn; the controller will resume this same conversation ({{CONVERSATION_ID}}) when action is needed."""

AWAIT_AGENTS_SATISFIED_PROMPT = """Use the returned state now; unfinished sibling tasks keep running unless you cancel them explicitly."""

AWAIT_AGENTS_TIMEOUT_PROMPT = """The tasks keep running. Continue useful parent work, inspect later with agent_status, or use join='change' for the next bounded wait; repeating the same long all-results wait will block on the same unfinished tasks."""

DELEGATED_TASK_FINISHED_PROMPT = """Subagent task {{TASK_ID}} finished.

{{RESULT}}"""

DELEGATED_TASK_BACKGROUND_PROMPT = """Subagent task {{TASK_ID}} is running in the background. Its result or error will arrive as a durable local event."""

DELEGATE_AGENT_DEPRECATION_PROMPT = """delegate_agent is deprecated and cannot launch an agent. Use spawn_agents with a stable batch_key, then pass its task IDs to await_agents."""


_PLACEHOLDER = re.compile(r"{{([A-Z][A-Z0-9_]*)}}")


def render_prompt(template: str, /, **values: str) -> str:
    """Render one prompt without interpreting placeholders in inserted values."""

    placeholders = set(_PLACEHOLDER.findall(template))
    missing = sorted(placeholders - values.keys())
    unexpected = sorted(values.keys() - placeholders)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(f"invalid prompt values: {'; '.join(details)}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def render_event_prompt(kind: str, payload: Mapping[str, object]) -> str:
    """Render a controller event for the model."""

    if kind == "student_available_for_assignment":
        return render_prompt(
            STUDENT_AVAILABLE_FOR_ASSIGNMENT_PROMPT,
            STUDENT=str(payload["student"]),
        )
    return render_prompt(
        EVENT_PROMPT,
        KIND=kind,
        PAYLOAD=json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )
