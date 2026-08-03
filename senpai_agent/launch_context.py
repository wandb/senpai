# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Render the launch-time role context stored in system_instructions/."""

import re
from pathlib import Path

from senpai_agent.agent_markdown import read_agent_markdown, strip_spdx_header

INSTRUCTIONS_ROOT = Path(__file__).resolve().parent.parent / "system_instructions"
RUNTIME_TEMPLATE = INSTRUCTIONS_ROOT / "SENPAI-LAUNCH-RUNTIME.md"
ISOLATION_TEMPLATE = INSTRUCTIONS_ROOT / "SENPAI-LAUNCH-ISOLATION.md"
OPERATOR_TEMPLATE = INSTRUCTIONS_ROOT / "SENPAI-OPERATOR-INSTRUCTIONS.md"
PLACEHOLDER = re.compile(r"{{([A-Z_]+)}}")


def _render(path: Path, values: dict[str, str]) -> str:
    template = read_agent_markdown(path)
    missing = sorted(set(PLACEHOLDER.findall(template)) - values.keys())
    if missing:
        raise ValueError(f"Missing {path.name} values: {', '.join(missing)}")
    for key, value in values.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template.strip()


def _read_extra_instructions(value: str) -> str:
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    return read_agent_markdown(path) if is_file else strip_spdx_header(value)


def render_launch_context(
    *,
    backend: str,
    gpus_per_student: int,
    timeout_minutes: float,
    max_epochs: int,
    tag: str,
    advisor_branch: str,
    target_base: str,
    students: list[str],
    extra_instructions: str = "",
) -> str:
    """Render backend facts, isolation, and optional operator instructions."""

    sections = [
        _render(
            RUNTIME_TEMPLATE,
            {
                "BACKEND": backend,
                "GPUS_PER_STUDENT": str(gpus_per_student),
                "TIMEOUT_MINUTES": f"{timeout_minutes:g}",
                "MAX_EPOCHS": str(max_epochs),
            },
        ),
        _render(
            ISOLATION_TEMPLATE,
            {
                "TAG": tag,
                "ADVISOR_BRANCH": advisor_branch,
                "TARGET_BASE": target_base or "<default>",
                "STUDENTS": ", ".join(students),
            },
        ),
    ]
    if extra_instructions:
        text = _read_extra_instructions(extra_instructions)
        sections.append(_render(OPERATOR_TEMPLATE, {"EXTRA_INSTRUCTIONS": text}))
    return "\n\n".join(sections)
