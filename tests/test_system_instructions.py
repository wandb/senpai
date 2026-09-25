from dataclasses import replace

import pytest

import senpai_agent.program_context as program_module
import senpai_agent.system_instructions as system_module
from senpai_agent.program_context import ProgramSystemPrompt
from senpai_agent.system_instructions import (
    SenpaiSystemInstructions,
    decode_system_instructions,
    encode_system_instructions,
)


@pytest.mark.parametrize(
    ("module", "template"),
    [
        (program_module, "PROGRAM_SYSTEM_PROMPT"),
        (system_module, "SENPAI_SYSTEM_INSTRUCTIONS_PROMPT"),
    ],
)
def test_persisted_snapshot_binds_the_rendered_suffix_templates(
    monkeypatch, module, template
):
    instructions = SenpaiSystemInstructions(
        harness="Harness.",
        role="Advisor.",
        program=ProgramSystemPrompt("program.md", "a" * 40, "Research policy."),
        launch="Launch.",
    )
    encoded = encode_system_instructions(instructions)
    expected = instructions.content_sha256
    assert decode_system_instructions(encoded, expected).prompt == instructions.prompt

    monkeypatch.setattr(module, template, getattr(module, template) + "\nNew wrapper.")

    with pytest.raises(ValueError, match="controller-held"):
        decode_system_instructions(encoded, expected)


@pytest.mark.parametrize("component", ["harness", "role", "launch", "program"])
def test_self_consistent_replacement_cannot_replace_the_parent_digest(component):
    instructions = SenpaiSystemInstructions(
        harness="Harness.",
        role="Advisor.",
        program=ProgramSystemPrompt("program.md", "a" * 40, "Research policy."),
        launch="Launch.",
    )
    replacement = (
        replace(instructions.program, content="Changed policy.")
        if component == "program"
        else "Changed instructions."
    )
    changed = replace(instructions, **{component: replacement})
    encoded = encode_system_instructions(changed)
    assert decode_system_instructions(encoded, changed.content_sha256) == changed

    with pytest.raises(ValueError, match="controller-held"):
        decode_system_instructions(encoded, instructions.content_sha256)
