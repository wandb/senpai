# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""One immutable system-instruction value for a Senpai agent process."""

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass

from senpai_agent.program_context import ProgramSystemPrompt
from senpai_agent.PROMPTS import SENPAI_SYSTEM_INSTRUCTIONS_PROMPT, render_prompt

SYSTEM_INSTRUCTIONS_FILE_ENV = "SENPAI_SYSTEM_INSTRUCTIONS_FILE"
SYSTEM_INSTRUCTIONS_SHA256_ENV = "SENPAI_SYSTEM_INSTRUCTIONS_SHA256"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SenpaiSystemInstructions:
    harness: str
    role: str
    program: ProgramSystemPrompt
    launch: str

    def _payload(self) -> dict[str, object]:
        harness = self.harness.strip()
        role = self.role.strip()
        launch = self.launch.strip()
        return {
            "harness": harness,
            "launch": launch,
            "program": {
                "content": self.program.content,
                "content_sha256": self.program.content_sha256,
                "program_path": self.program.program_path,
                "source_commit": self.program.source_commit,
            },
            "role": role,
            "sha256": {
                "harness": _sha256(harness),
                "launch": _sha256(launch),
                "program": self.program.content_sha256,
                "role": _sha256(role),
            },
        }

    @property
    def content_sha256(self) -> str:
        encoded = json.dumps(
            self._payload(), separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

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


def encode_system_instructions(instructions: SenpaiSystemInstructions) -> str:
    """Encode every trusted prompt component for inherited workers."""

    return base64.b64encode(
        json.dumps(
            instructions._payload(), separators=(",", ":"), sort_keys=True
        ).encode()
    ).decode()


def decode_system_instructions(
    encoded: str,
    expected_sha256: str,
) -> SenpaiSystemInstructions:
    """Decode a system snapshot and verify its controller-held digest."""

    try:
        payload = json.loads(base64.b64decode(encoded, validate=True))
        if not isinstance(payload, dict) or set(payload) != {
            "harness",
            "launch",
            "program",
            "role",
            "sha256",
        }:
            raise ValueError
        program_payload = payload["program"]
        if not isinstance(program_payload, dict) or set(program_payload) != {
            "content",
            "content_sha256",
            "program_path",
            "source_commit",
        }:
            raise ValueError
        if not all(
            isinstance(value, str)
            for value in (
                payload["harness"],
                payload["launch"],
                payload["role"],
                *program_payload.values(),
            )
        ):
            raise ValueError
        instructions = SenpaiSystemInstructions(
            harness=payload["harness"],
            role=payload["role"],
            program=ProgramSystemPrompt(
                program_path=program_payload["program_path"],
                source_commit=program_payload["source_commit"],
                content=program_payload["content"],
            ),
            launch=payload["launch"],
        )
        if payload != instructions._payload():
            raise ValueError
        if expected_sha256 != instructions.content_sha256:
            raise ValueError
        return instructions
    except (
        binascii.Error,
        json.JSONDecodeError,
        UnicodeDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{SYSTEM_INSTRUCTIONS_FILE_ENV} does not match the "
            f"controller-held {SYSTEM_INSTRUCTIONS_SHA256_ENV}"
        ) from error
