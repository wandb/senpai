import base64
import json
from pathlib import Path

import pytest
from git_workflow_support import commit_workspace, git
from test_agent_markdown import HTML_HEADER

import senpai_agent.program_context as program_context_module
from senpai_agent.program_context import (
    MAX_PROGRAM_BYTES,
    PROGRAM_SOURCE_COMMIT_ENV,
    decode_program_system_prompt,
    encode_program_system_prompt,
    load_program_system_prompt,
    normalize_program_path,
)


@pytest.mark.parametrize(
    "path",
    [
        "/program.md",
        "../program.md",
        "senpai/../program.md",
        "./senpai/program.md",
        "senpai//program.md",
        "senpai/PROGRAM.md",
        "unsafe path/program.md",
        "unsafe\npath/program.md",
    ],
)
def test_program_path_must_be_normalized_and_repo_relative(path: str):
    with pytest.raises(ValueError, match="target-repository-relative"):
        normalize_program_path(path)


def write_program(workspace: Path, path: str, content: str) -> None:
    program = workspace / path
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(content)


def loose_object(workspace: Path, object_id: str) -> Path:
    return workspace / ".git" / "objects" / object_id[:2] / object_id[2:]


def test_blank_program_path_discovers_the_only_root_program(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Root policy.")
    commit_workspace(workspace)

    program = load_program_system_prompt(workspace, "")

    assert program.program_path == "program.md"
    assert program.content == "Root policy."
    assert f"commit `{program.source_commit}`" in program.prompt
    assert f"Content SHA-256: `{program.content_sha256}`" in program.prompt
    assert "cannot override the Senpai harness" in program.prompt


def test_blank_program_path_discovers_the_only_one_level_program(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "senpai/program.md", "Nested policy.")
    commit_workspace(workspace)

    program = load_program_system_prompt(workspace, "")

    assert program.program_path == "senpai/program.md"
    assert program.content == "Nested policy."


def test_blank_program_path_lists_every_match(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Root policy.")
    write_program(workspace, "alpha/program.md", "Alpha policy.")
    write_program(workspace, "beta/program.md", "Beta policy.")
    commit_workspace(workspace)

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
    commit_workspace(workspace)

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
    commit_workspace(workspace)

    program = load_program_system_prompt(workspace, "senpai/program.md")

    assert program.program_path == "senpai/program.md"
    assert program.content == "Nested policy."


def test_program_prompt_strips_the_spdx_header(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(
        workspace,
        "program.md",
        HTML_HEADER + "# Research policy\n\nWin safely.\n",
    )
    commit_workspace(workspace)

    program = load_program_system_prompt(workspace, "program.md")

    assert program.content == "# Research policy\n\nWin safely."


def test_explicit_program_path_must_exist(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "README.md").write_text("Target repository.")
    commit_workspace(workspace)

    with pytest.raises(
        RuntimeError,
        match=r"does not exist at commit .*: senpai/program\.md",
    ):
        load_program_system_prompt(workspace, "senpai/program.md")


def test_program_path_cannot_escape_through_a_symlink(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    outside = tmp_path / "program.md"
    outside.write_text("Outside policy.")
    (workspace / "program.md").symlink_to(outside)
    commit_workspace(workspace)

    with pytest.raises(RuntimeError, match="regular file in the target commit"):
        load_program_system_prompt(workspace, "program.md")


def test_program_uses_the_launch_pinned_commit_not_current_head(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Reviewed policy.")
    source_commit = commit_workspace(workspace)
    (workspace / "program.md").write_text("Unreviewed replacement.")
    commit_workspace(workspace, "unreviewed local policy")

    program = load_program_system_prompt(workspace, "program.md", source_commit)

    assert program.source_commit == source_commit
    assert program.content == "Reviewed policy."
    assert "Unreviewed replacement" not in program.prompt


@pytest.mark.parametrize("source_commit", ["not-a-commit", "b" * 40])
def test_program_requires_an_available_full_launch_commit(
    tmp_path: Path,
    source_commit: str,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Reviewed policy.")
    commit_workspace(workspace)

    with pytest.raises(RuntimeError, match=PROGRAM_SOURCE_COMMIT_ENV):
        load_program_system_prompt(workspace, "program.md", source_commit)


@pytest.mark.parametrize("object_type", ["commit", "tree", "blob"])
def test_program_rejects_a_substituted_object_in_the_agent_writable_store(
    tmp_path: Path,
    object_type: str,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Reviewed policy.")
    source_commit = commit_workspace(workspace)
    reviewed = {
        "commit": source_commit,
        "tree": git(workspace, "rev-parse", f"{source_commit}^{{tree}}"),
        "blob": git(workspace, "rev-parse", f"{source_commit}:program.md"),
    }
    write_program(workspace, "program.md", "Substituted policy.")
    substituted_commit = commit_workspace(workspace, "substituted policy")
    substituted = {
        "commit": substituted_commit,
        "tree": git(workspace, "rev-parse", f"{substituted_commit}^{{tree}}"),
        "blob": git(workspace, "rev-parse", f"{substituted_commit}:program.md"),
    }

    reviewed_object = loose_object(workspace, reviewed[object_type])
    substituted_object = loose_object(workspace, substituted[object_type])
    reviewed_object.chmod(0o600)
    reviewed_object.write_bytes(substituted_object.read_bytes())

    with pytest.raises(RuntimeError, match="object integrity verification"):
        load_program_system_prompt(workspace, "program.md", source_commit)


def test_program_ignores_a_corrupt_unrelated_object(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Reviewed policy.")
    source_commit = commit_workspace(workspace)
    (workspace / "unrelated").write_text("Not reachable from the pinned commit.")
    unrelated = git(workspace, "hash-object", "-w", "unrelated")
    unrelated_object = loose_object(workspace, unrelated)
    unrelated_object.chmod(0o600)
    unrelated_object.write_bytes(b"not a Git object")

    program = load_program_system_prompt(workspace, "program.md", source_commit)

    assert program.content == "Reviewed policy."


def test_object_verification_uses_the_stable_git_cat_file_interface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Reviewed policy.")
    source_commit = commit_workspace(workspace)
    commands: list[list[str]] = []
    run = program_context_module.subprocess.run

    def capture(argv, **kwargs):
        commands.append(argv)
        return run(argv, **kwargs)

    monkeypatch.setattr(program_context_module.subprocess, "run", capture)

    load_program_system_prompt(workspace, "program.md", source_commit)

    assert commands
    assert all(
        command[0] == "/usr/bin/git"
        and command[1] == "cat-file"
        and command[2] in {"commit", "tree", "blob"}
        for command in commands
    )


def test_program_rejects_a_blob_larger_than_the_system_context_limit(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "x" * (MAX_PROGRAM_BYTES + 1))
    commit_workspace(workspace)

    with pytest.raises(RuntimeError, match="65536-byte system-context limit"):
        load_program_system_prompt(workspace, "program.md")


def test_inherited_program_snapshot_is_content_addressed(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    write_program(workspace, "program.md", "Reviewed policy.")
    commit_workspace(workspace)
    program = load_program_system_prompt(workspace, "program.md")

    encoded = encode_program_system_prompt(program)

    assert decode_program_system_prompt(encoded) == program

    payload = json.loads(base64.b64decode(encoded))
    payload["content"] = "Tampered policy."
    tampered = base64.b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(ValueError, match="content-addressed snapshot"):
        decode_program_system_prompt(tampered)
