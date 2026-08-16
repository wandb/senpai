# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Load a target repository's program.md for the system prompt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from senpai_agent.agent_markdown import read_agent_markdown
from senpai_agent.PROMPTS import PROGRAM_SYSTEM_PROMPT, render_prompt

PROGRAM_PATH_ENV = "SENPAI_PROGRAM_PATH"
PROGRAM_PATH_GUIDANCE = (
    "Set --program_path (or program_path in senpai.yaml) to a "
    "target-repository-relative path ending in program.md."
)


@dataclass(frozen=True, slots=True)
class ProgramSystemPrompt:
    program_path: str
    prompt: str


def normalize_program_path(value: str) -> str:
    """Return a normalized target-repository-relative program.md path."""

    if not value:
        return ""
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.name != "program.md"
        or path.as_posix() != value
        or ".." in path.parts
    ):
        raise ValueError(
            "must be a normalized target-repository-relative "
            "path ending in program.md"
        )
    return value


def load_program_system_prompt(
    workspace: Path,
    value: str,
) -> ProgramSystemPrompt:
    """Resolve, read, and format one program.md."""

    workspace = workspace.resolve()
    program_path = normalize_program_path(value) or _discover_program_path(workspace)
    source = _program_file(workspace, program_path)
    prompt = render_prompt(
        PROGRAM_SYSTEM_PROMPT,
        PROGRAM_PATH=program_path,
        PROGRAM_CONTENT=read_agent_markdown(source).strip(),
    )
    return ProgramSystemPrompt(program_path=program_path, prompt=prompt)


def _discover_program_path(workspace: Path) -> str:
    candidates = sorted(
        path.relative_to(workspace).as_posix()
        for path in (workspace / "program.md", *workspace.glob("*/program.md"))
        if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        matches = ", ".join(candidates)
        raise RuntimeError(
            f"found multiple program.md files: {matches}. Only one may exist "
            f"when program_path is blank. {PROGRAM_PATH_GUIDANCE}"
        )
    raise RuntimeError(
        "could not find program.md; searched program.md and */program.md "
        f"(exactly one directory below the repository root). {PROGRAM_PATH_GUIDANCE}"
    )


def _program_file(workspace: Path, program_path: str) -> Path:
    try:
        source = (workspace / program_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"program.md does not exist: {program_path}. {PROGRAM_PATH_GUIDANCE}"
        ) from error
    if not source.is_relative_to(workspace) or not source.is_file():
        raise RuntimeError(
            "program.md must be a file beneath the target workspace: "
            f"{program_path}. {PROGRAM_PATH_GUIDANCE}"
        )
    return source
