# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""One immutable system-instruction value for a Senpai agent process."""

from dataclasses import dataclass

from senpai_agent.program_context import ProgramSystemPrompt
from senpai_agent.PROMPTS import SENPAI_SYSTEM_INSTRUCTIONS_PROMPT, render_prompt


@dataclass(frozen=True, slots=True)
class SenpaiSystemInstructions:
    harness: str
    role: str
    program: ProgramSystemPrompt
    launch: str

    @property
    def prompt(self) -> str:
        return (
            render_prompt(
                SENPAI_SYSTEM_INSTRUCTIONS_PROMPT,
                HARNESS=self.harness.strip(),
                ROLE=self.role.strip(),
                PROGRAM=self.program.prompt.strip(),
                LAUNCH=self.launch.strip(),
            )
            + "\n"
        )
