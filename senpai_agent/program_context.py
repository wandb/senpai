# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Load one committed target program as immutable system context."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from senpai_agent.agent_markdown import strip_spdx_header
from senpai_agent.git_transport import GIT_EXECUTABLE, git_process_env, run_git
from senpai_agent.PROMPTS import PROGRAM_SYSTEM_PROMPT, render_prompt

MAX_PROGRAM_BYTES = 64 * 1024
PROGRAM_CONTEXT_FILE_ENV = "SENPAI_PROGRAM_CONTEXT_FILE"
PROGRAM_PATH_ENV = "SENPAI_PROGRAM_PATH"
PROGRAM_SOURCE_COMMIT_ENV = "SENPAI_PROGRAM_SOURCE_COMMIT"
PROGRAM_PATH_GUIDANCE = (
    "Set --program_path (or program_path in senpai.yaml) to a "
    "target-repository-relative path ending in program.md."
)
_COMMIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SAFE_PROGRAM_PATH = re.compile(r"[A-Za-z0-9._/-]+")
_GIT_READ_ENV = {"GIT_NO_REPLACE_OBJECTS": "1"}
_GIT_TREE_MODES = {"40000", "100644", "100755", "120000", "160000"}
_OBJECT_INTEGRITY_ERROR = (
    f"{PROGRAM_SOURCE_COMMIT_ENV} launch-pinned program commit failed "
    "Git object integrity verification"
)


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    mode: str
    name: bytes
    object_id: str


@dataclass(frozen=True, slots=True)
class ProgramSystemPrompt:
    program_path: str
    source_commit: str
    content: str

    def __post_init__(self) -> None:
        if (
            not self.program_path
            or normalize_program_path(self.program_path) != self.program_path
        ):
            raise ValueError("program_path must not be empty")
        if not _COMMIT_SHA.fullmatch(self.source_commit):
            raise ValueError("source_commit must be a full Git commit SHA")
        if not self.content or self.content != self.content.strip():
            raise ValueError("program content must be nonempty and stripped")
        if len(self.content.encode()) > MAX_PROGRAM_BYTES:
            raise ValueError(
                f"program.md exceeds the {MAX_PROGRAM_BYTES}-byte system-context limit"
            )

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()

    @property
    def prompt(self) -> str:
        return render_prompt(
            PROGRAM_SYSTEM_PROMPT,
            PROGRAM_PATH=self.program_path,
            PROGRAM_COMMIT=self.source_commit,
            PROGRAM_SHA256=self.content_sha256,
            PROGRAM_CONTENT=self.content,
        )


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
        or not _SAFE_PROGRAM_PATH.fullmatch(value)
    ):
        raise ValueError(
            "must be a normalized target-repository-relative "
            "path ending in program.md"
        )
    return value


def load_program_system_prompt(
    workspace: Path,
    value: str,
    source_commit: str | None = None,
) -> ProgramSystemPrompt:
    """Load one program.md blob from an exact commit."""

    workspace = workspace.resolve()
    if source_commit is not None and not _COMMIT_SHA.fullmatch(source_commit):
        raise RuntimeError(
            f"{PROGRAM_SOURCE_COMMIT_ENV} must name one available full commit SHA"
        )
    if source_commit is None:
        source_commit = run_git(
            workspace,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            extra_env=_GIT_READ_ENV,
        )
    commit = _read_verified_object(workspace, source_commit, "commit")
    root_tree = _commit_tree(commit, source_commit)
    program_path = normalize_program_path(value) or _discover_program_path(
        workspace, root_tree
    )
    blob_sha = _program_blob(workspace, root_tree, program_path, source_commit)
    blob = _read_verified_object(workspace, blob_sha, "blob")
    if len(blob) > MAX_PROGRAM_BYTES:
        raise RuntimeError(
            f"program.md exceeds the {MAX_PROGRAM_BYTES}-byte system-context limit: "
            f"{program_path} is {len(blob)} bytes"
        )
    try:
        decoded = blob.decode()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"program.md must be UTF-8: {program_path}") from error
    content = strip_spdx_header(decoded).strip()
    if not content:
        raise RuntimeError(f"program.md is empty at {source_commit}: {program_path}")
    return ProgramSystemPrompt(
        program_path=program_path,
        source_commit=source_commit,
        content=content,
    )


def _read_verified_object(
    workspace: Path,
    object_id: str,
    object_type: str,
) -> bytes:
    environment = git_process_env(None)
    environment.update(_GIT_READ_ENV)
    completed = subprocess.run(
        [GIT_EXECUTABLE, "cat-file", object_type, object_id],
        cwd=workspace,
        capture_output=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(_OBJECT_INTEGRITY_ERROR)
    algorithm = "sha1" if len(object_id) == 40 else "sha256"
    header = f"{object_type} {len(completed.stdout)}\0".encode()
    actual_id = hashlib.new(algorithm, header + completed.stdout).hexdigest()
    if actual_id != object_id:
        raise RuntimeError(_OBJECT_INTEGRITY_ERROR)
    return completed.stdout


def _commit_tree(commit: bytes, source_commit: str) -> str:
    first_line = commit.partition(b"\n")[0]
    if not first_line.startswith(b"tree "):
        raise RuntimeError(_OBJECT_INTEGRITY_ERROR)
    try:
        tree_id = first_line.removeprefix(b"tree ").decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError(_OBJECT_INTEGRITY_ERROR) from error
    if len(tree_id) != len(source_commit) or not _COMMIT_SHA.fullmatch(tree_id):
        raise RuntimeError(_OBJECT_INTEGRITY_ERROR)
    return tree_id


def encode_program_system_prompt(program: ProgramSystemPrompt) -> str:
    """Encode one validated snapshot for launch-owned file transport."""

    payload = {
        "content": program.content,
        "content_sha256": program.content_sha256,
        "program_path": program.program_path,
        "source_commit": program.source_commit,
    }
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).decode()


def decode_program_system_prompt(encoded: str) -> ProgramSystemPrompt:
    """Decode and verify one launch-owned program snapshot."""

    try:
        payload = json.loads(base64.b64decode(encoded, validate=True))
        if not isinstance(payload, dict) or set(payload) != {
            "content",
            "content_sha256",
            "program_path",
            "source_commit",
        }:
            raise ValueError
        if not all(isinstance(value, str) for value in payload.values()):
            raise ValueError
        program = ProgramSystemPrompt(
            program_path=payload["program_path"],
            source_commit=payload["source_commit"],
            content=payload["content"],
        )
        if payload["content_sha256"] != program.content_sha256:
            raise ValueError
        return program
    except (
        binascii.Error,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{PROGRAM_CONTEXT_FILE_ENV} must contain a valid "
            "content-addressed snapshot"
        ) from error


def _discover_program_path(workspace: Path, root_tree: str) -> str:
    root_entries = _tree_entries(workspace, root_tree)
    candidates = [
        "program.md" for entry in root_entries if entry.name == b"program.md"
    ]
    for directory in root_entries:
        if directory.mode != "40000" or not any(
            entry.name == b"program.md"
            for entry in _tree_entries(workspace, directory.object_id)
        ):
            continue
        try:
            name = directory.name.decode("ascii")
            candidates.append(normalize_program_path(f"{name}/program.md"))
        except (UnicodeDecodeError, ValueError):
            raise RuntimeError(
                f"program.md lives under directory {directory.name!r}, whose "
                "name is not a safe program path; rename the directory or "
                f"{PROGRAM_PATH_GUIDANCE}"
            ) from None
    candidates.sort()
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


def _program_blob(
    workspace: Path,
    root_tree: str,
    program_path: str,
    source_commit: str,
) -> str:
    tree_id = root_tree
    parts = PurePosixPath(program_path).parts
    for index, part in enumerate(parts):
        entry = next(
            (
                candidate
                for candidate in _tree_entries(workspace, tree_id)
                if candidate.name == part.encode()
            ),
            None,
        )
        if entry is None:
            raise RuntimeError(
                f"program.md does not exist at commit {source_commit}: "
                f"{program_path}. {PROGRAM_PATH_GUIDANCE}"
            )
        if index < len(parts) - 1:
            if entry.mode != "40000":
                raise RuntimeError(
                    f"program.md does not exist at commit {source_commit}: "
                    f"{program_path}. {PROGRAM_PATH_GUIDANCE}"
                )
            tree_id = entry.object_id
            continue
        if entry.mode not in {"100644", "100755"}:
            break
        return entry.object_id
    raise RuntimeError(
        "program.md must be a regular file in the target commit: "
        f"{program_path}. {PROGRAM_PATH_GUIDANCE}"
    )


def _tree_entries(workspace: Path, tree_id: str) -> tuple[_TreeEntry, ...]:
    tree = _read_verified_object(workspace, tree_id, "tree")
    digest_size = len(tree_id) // 2
    entries: list[_TreeEntry] = []
    names: set[bytes] = set()
    offset = 0
    while offset < len(tree):
        separator = tree.find(b" ", offset)
        terminator = tree.find(b"\0", separator + 1)
        digest_end = terminator + 1 + digest_size
        if separator < 0 or terminator < 0 or digest_end > len(tree):
            raise RuntimeError(_OBJECT_INTEGRITY_ERROR)
        try:
            mode = tree[offset:separator].decode("ascii")
        except UnicodeDecodeError as error:
            raise RuntimeError(_OBJECT_INTEGRITY_ERROR) from error
        name = tree[separator + 1 : terminator]
        if mode not in _GIT_TREE_MODES or not name or b"/" in name or name in names:
            raise RuntimeError(_OBJECT_INTEGRITY_ERROR)
        names.add(name)
        entries.append(
            _TreeEntry(
                mode=mode,
                name=name,
                object_id=tree[terminator + 1 : digest_end].hex(),
            )
        )
        offset = digest_end
    if offset != len(tree):
        raise RuntimeError(_OBJECT_INTEGRITY_ERROR)
    return tuple(entries)
