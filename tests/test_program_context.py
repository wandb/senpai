from pathlib import Path

import pytest

from senpai_agent.program_context import (
    load_program_system_prompt,
    normalize_program_path,
)
from test_agent_markdown import HTML_HEADER


@pytest.mark.parametrize(
    "path",
    [
        "/program.md",
        "../program.md",
        "senpai/../program.md",
        "./senpai/program.md",
        "senpai//program.md",
        "senpai/PROGRAM.md",
    ],
)
def test_program_path_must_be_normalized_and_repo_relative(path: str):
    with pytest.raises(ValueError, match="target-repository-relative"):
        normalize_program_path(path)


def write_program(workspace: Path, path: str, content: str) -> None:
    program = workspace / path
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(content)


def test_blank_program_path_discovers_the_only_root_program(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Root policy.")

    program = load_program_system_prompt(workspace, "")

    assert program.program_path == "program.md"
    assert program.prompt == "# program.md - program.md\n\nRoot policy."


def test_blank_program_path_discovers_the_only_one_level_program(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "senpai/program.md", "Nested policy.")

    program = load_program_system_prompt(workspace, "")

    assert program.program_path == "senpai/program.md"
    assert program.prompt == (
        "# program.md - senpai/program.md\n\nNested policy."
    )


def test_blank_program_path_lists_every_match(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Root policy.")
    write_program(workspace, "alpha/program.md", "Alpha policy.")
    write_program(workspace, "beta/program.md", "Beta policy.")

    with pytest.raises(
        RuntimeError,
        match=r"alpha/program\.md, beta/program\.md, program\.md",
    ) as error:
        load_program_system_prompt(workspace, "")

    assert "Only one may exist when program_path is blank" in str(error.value)
    assert "--program_path" in str(error.value)


def test_blank_program_path_does_not_search_deeper_than_one_level(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "configs/senpai/program.md", "Too deep.")

    with pytest.raises(
        RuntimeError,
        match=r"searched program\.md and \*/program\.md",
    ) as error:
        load_program_system_prompt(workspace, "")

    assert "--program_path" in str(error.value)


def test_explicit_program_path_selects_one_of_multiple_matches(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Root policy.")
    write_program(workspace, "senpai/program.md", "Nested policy.")

    program = load_program_system_prompt(workspace, "senpai/program.md")

    assert program.program_path == "senpai/program.md"
    assert program.prompt.endswith("Nested policy.")


def test_program_prompt_strips_the_spdx_header(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(
        workspace,
        "program.md",
        HTML_HEADER + "# Research policy\n\nWin safely.\n",
    )

    program = load_program_system_prompt(workspace, "program.md")

    assert program.prompt == (
        "# program.md - program.md\n\n# Research policy\n\nWin safely."
    )


def test_explicit_program_path_must_exist(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()

    with pytest.raises(RuntimeError, match="does not exist: senpai/program.md"):
        load_program_system_prompt(workspace, "senpai/program.md")


def test_program_path_cannot_escape_through_a_symlink(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    outside = tmp_path / "program.md"
    outside.write_text("Outside policy.")
    (workspace / "program.md").symlink_to(outside)

    with pytest.raises(RuntimeError, match="beneath the target workspace"):
        load_program_system_prompt(workspace, "program.md")
