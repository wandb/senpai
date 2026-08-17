from pathlib import Path

import pytest

import senpai_agent.PROMPTS as prompts


def test_prompt_module_defines_nonempty_uppercase_strings():
    prompt_values = {
        name: value
        for name, value in vars(prompts).items()
        if name.endswith("_PROMPT")
    }

    assert prompt_values
    assert all(name.isupper() for name in prompt_values)
    assert all(isinstance(value, str) for value in prompt_values.values())
    assert all(value == value.strip() for value in prompt_values.values())


def test_render_prompt_requires_exact_values_and_does_not_reprocess_insertions():
    template = "Payload: {{PAYLOAD}}"

    assert prompts.render_prompt(
        template,
        PAYLOAD='{"literal": "{{UNCHANGED}}"}',
    ) == 'Payload: {"literal": "{{UNCHANGED}}"}'
    with pytest.raises(ValueError, match="missing: PAYLOAD"):
        prompts.render_prompt(template)
    with pytest.raises(ValueError, match="unexpected: EXTRA"):
        prompts.render_prompt(template, PAYLOAD="value", EXTRA="value")


def test_render_prompt_preserves_placeholder_boundary_whitespace():
    assert prompts.render_prompt("Before\n\n{{VALUE}}", VALUE="value\n\n") == (
        "Before\n\nvalue\n\n"
    )


def test_python_sources_do_not_embed_centralized_prompt_text():
    source_root = Path(prompts.__file__).parent
    prompt_module = Path(prompts.__file__).resolve()
    fragments = (
        "# Conversation context recovery",
        "Actionable events follow as separately tracked messages.",
        "You are a fresh Senpai subagent.",
        "Senpai restarted before this action completed.",
        "Open feedback_url to read the omitted text.",
        "You may finish this turn; the controller will resume",
        "unfinished sibling tasks keep running unless you cancel",
        "repeating the same long all-results wait will block",
        "delegate_agent is deprecated and cannot launch an agent.",
    )

    for source in source_root.rglob("*.py"):
        if source.resolve() == prompt_module:
            continue
        text = source.read_text()
        for fragment in fragments:
            assert fragment not in text, f"{fragment!r} remains in {source}"
