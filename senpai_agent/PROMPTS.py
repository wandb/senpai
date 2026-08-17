# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Model-facing instruction templates assembled by Python."""

from __future__ import annotations

import re


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

SENPAI_SYSTEM_INSTRUCTIONS_PROMPT = """# Senpai harness

{{HARNESS}}

# Senpai role

{{ROLE}}

{{PROGRAM}}

{{LAUNCH}}"""

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

RECOVERED_ACTION_PROMPT = """Senpai restarted before this action completed. Inspect the preserved workspace and rerun it explicitly only if it is still needed."""

ADVISOR_EVENT_PROMPT = """# Senpai event: {{KIND}}

Observed at (UTC): {{OBSERVED_AT}}

```json
{{PAYLOAD}}
```"""

EVENT_PROMPT = """## {{KIND}}

{{PAYLOAD}}"""

WORKSPACE_DIVERGENCE_PROMPT = """The workspace cannot be reconciled automatically because local assignment history diverged or dirty work belongs to another checkout. Senpai preserved every local commit and dirty file without changing the checkout. Inspect and reconcile it explicitly; do not reset or discard local work."""

TRUNCATED_FEEDBACK_PROMPT = """Open feedback_url to read the omitted text."""

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
