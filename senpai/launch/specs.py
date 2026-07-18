# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Backend-neutral environment specifications for Senpai roles."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoleSpec:
    role: str
    name: str
    env: dict[str, str]
    secrets: dict[str, str]

    @property
    def key(self) -> str:
        return "advisor" if self.role == "advisor" else f"student-{self.name}"


def target_repo_slug(url: str) -> str:
    """Extract owner/repo from an HTTPS or SSH GitHub URL."""
    return url.split("github.com", 1)[-1].lstrip(":/").removesuffix(".git")


def load_extra_instructions(extra_instructions: str) -> str:
    if not extra_instructions:
        return ""
    path = Path(extra_instructions)
    return path.read_text() if path.exists() else extra_instructions


def build_extra_instructions(args, tag: str, student_list: list[str]) -> str:
    students = ", ".join(student_list)
    target_base = args.target_repo_branch or "<default>"
    isolation = f"""# Launch isolation and run-limit rules

- This launch is scoped to research tag `{tag}`, advisor branch `{args.advisor_branch}`, and target base branch `{target_base}`.
- Only inspect, modify, or reason from `{args.advisor_branch}` plus PR branches assigned to these students in this launch: {students}.
- Do not inspect, compare, summarize, cherry-pick, borrow from, or base decisions on any PR or branch outside `{args.advisor_branch}` and the assigned student PR branches for this launch.
- Do not use unrelated experiment runs or historical results unless the human explicitly names them during this launch.
- Students branch from `{args.advisor_branch}`. Do not rebase or retarget work onto unrelated branches.
- Treat `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` as hard per-training-run bounds. Do not override them or continue a run past them.
"""
    user_extra = load_extra_instructions(args.extra_instructions)
    return (
        isolation
        if not user_extra
        else isolation + "\n# Additional operator instructions\n\n" + user_extra
    )


def encoded_extra_instructions(args, tag: str, student_list: list[str]) -> str:
    return base64.b64encode(
        build_extra_instructions(args, tag, student_list).encode()
    ).decode()


def build_student_env(args, tag: str, student_name: str) -> dict[str, str]:
    return {
        "REPO_URL": args.repo_url,
        "REPO_BRANCH": args.repo_branch,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        "GH_REPO": target_repo_slug(args.target_repo_url),
        "STUDENT_NAME": student_name,
        "RESEARCH_TAG": tag,
        "GPUS_PER_STUDENT": str(args.gpus_per_student),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": "online",
        "ADVISOR_BRANCH": args.advisor_branch,
        "GH_HISTORY_SCOPE": args.gh_history_scope,
        "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
        "SENPAI_TIMEOUT_MINUTES": str(args.timeout_minutes),
        "SENPAI_MAX_EPOCHS": str(args.max_epochs),
        "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
        "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
        "STUDENT_CLAUDE_WATCHDOG_INTERVAL_S": str(
            args.student_claude_watchdog_interval_s
        ),
        "STUDENT_CLAUDE_WATCHDOG_JITTER_S": str(args.student_claude_watchdog_jitter_s),
        "STUDENT_CLAUDE_MIN_RUNTIME_S": str(args.student_claude_min_runtime_s),
        "STUDENT_CLAUDE_STALE_LOG_S": str(args.student_claude_stale_log_s),
        "STUDENT_ASSIGNMENT_DRIFT_GRACE_S": str(args.student_assignment_drift_grace_s),
        "EXTRA_INSTRUCTIONS_B64": encoded_extra_instructions(args, tag, [student_name]),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }


def build_advisor_env(args, tag: str, student_list: list[str]) -> dict[str, str]:
    return {
        "REPO_URL": args.repo_url,
        "REPO_BRANCH": args.repo_branch,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        "GH_REPO": target_repo_slug(args.target_repo_url),
        "RESEARCH_TAG": tag,
        "STUDENT_NAMES": ",".join(student_list),
        "GPUS_PER_STUDENT": str(args.gpus_per_student),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": "online",
        "ADVISOR_BRANCH": args.advisor_branch,
        "GH_HISTORY_SCOPE": args.gh_history_scope,
        "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
        "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
        "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
        "SENPAI_STALE_WIP_SECONDS": str(args.stale_wip_seconds),
        "ADVISOR_CLAUDE_WATCHDOG_INTERVAL_S": str(
            args.advisor_claude_watchdog_interval_s
        ),
        "ADVISOR_CLAUDE_MIN_RUNTIME_S": str(args.advisor_claude_min_runtime_s),
        "ADVISOR_CLAUDE_STALE_LOG_S": str(args.advisor_claude_stale_log_s),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
        "EXTRA_INSTRUCTIONS_B64": encoded_extra_instructions(args, tag, student_list),
    }


def build_student_spec(
    args, tag: str, student_name: str, secrets: dict[str, str]
) -> RoleSpec:
    return RoleSpec(
        "student", student_name, build_student_env(args, tag, student_name), secrets
    )


def build_advisor_spec(
    args, tag: str, student_list: list[str], secrets: dict[str, str]
) -> RoleSpec:
    return RoleSpec(
        "advisor", "advisor", build_advisor_env(args, tag, student_list), secrets
    )
