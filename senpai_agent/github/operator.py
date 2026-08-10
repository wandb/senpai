"""Operator access to Senpai's typed assignment transitions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from pydantic import SecretStr

from senpai_agent.github.tools.advisor import (
    AdoptAssignmentExecutor,
    CreateAssignmentExecutor,
)
from senpai_agent.github.tools.contracts import (
    AdoptAssignmentAction,
    CreateAssignmentAction,
)
from senpai_agent.github.tools.runtime import (
    GitHubToolRuntime,
    configured_student_names,
)
from senpai_agent.github.workflow import GitHubWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Senpai's guarded assignment operations as an operator."
    )
    parser.add_argument(
        "--workspace",
        required=True,
        type=Path,
        help="Target-repository checkout used for guarded Git operations.",
    )
    parser.add_argument(
        "--advisor-branch",
        required=True,
        help="Configured advisor branch that assignments target.",
    )
    parser.add_argument(
        "--student-names",
        required=True,
        help="Comma-separated allowlist of students in this launch.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        (
            "create-assignment",
            "Create or exactly replay one typed assignment branch and draft PR.",
        ),
        (
            "adopt-assignment",
            "Attach typed identity to one exact pre-existing assignment PR.",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "action",
            help="JSON action file, or - to read the action from standard input.",
        )
    return parser


def _github_repo(environment: Mapping[str, str]) -> str:
    repo = environment.get("GH_REPO", "").strip()
    if len(repo.split("/")) != 2 or not all(repo.split("/")):
        raise ValueError("GH_REPO must use owner/name form")
    return repo


def _github_token(environment: Mapping[str, str]) -> SecretStr:
    token = next(
        (
            value.strip()
            for name in ("GITHUB_TOKEN", "GH_TOKEN")
            if (value := environment.get(name, "")).strip()
        ),
        None,
    )
    if token is None:
        raise ValueError("GITHUB_TOKEN or GH_TOKEN is required")
    return SecretStr(token)


def _read_action(source: str, stdin: TextIO) -> str:
    if source == "-":
        return stdin.read()
    return Path(source).expanduser().read_text(encoding="utf-8")


def _runtime(args: argparse.Namespace, environment: Mapping[str, str]) -> GitHubToolRuntime:
    advisor_branch = args.advisor_branch.strip()
    if not advisor_branch:
        raise ValueError("--advisor-branch must not be empty")
    students = configured_student_names(args.student_names)
    if not students:
        raise ValueError("--student-names must name at least one student")
    token = _github_token(environment)
    trusted_actor = environment.get("SENPAI_GITHUB_ACTOR", "").strip() or None
    workflow = GitHubWorkflow(
        _github_repo(environment),
        token,
        role="advisor",
        trusted_actor=trusted_actor,
    )
    return GitHubToolRuntime(
        workflow=workflow,
        workspace=args.workspace.expanduser().resolve(),
        git_token=token,
        role="advisor",
        advisor_branch=advisor_branch,
        student_names=students,
        student_name=None,
    )


def operator_main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] = os.environ,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        runtime = _runtime(args, environment)
        payload = _read_action(args.action, stdin)
        if args.command == "create-assignment":
            action = CreateAssignmentAction.model_validate_json(payload)
            observation = CreateAssignmentExecutor(runtime)(action)
        else:
            action = AdoptAssignmentAction.model_validate_json(payload)
            observation = AdoptAssignmentExecutor(runtime)(action)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            observation.model_dump(
                mode="json",
                include={"changed", "resource_url", "state", "version"},
            ),
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(operator_main())
