# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Backend-neutral environment specifications for Senpai roles."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

STUDENT_NAMES = (
    "frieren", "fern", "tanjiro", "nezuko", "alphonse", "edward",
    "thorfinn", "askeladd", "violet", "gilbert", "senku", "kohaku",
    "emma", "norman", "chihiro", "haku", "shoya", "shouko",
    "mitsuha", "taki", "shinji", "rei", "kaneda", "tetsuo",
    "naruto", "sasuke", "sakura", "kakashi", "hinata", "itachi",
    "roy", "winry", "eren", "mikasa", "armin", "levi",
    "historia", "ymir", "zenitsu", "inosuke", "giyu", "shinobu",
    "chrome", "gen", "ray", "asuka", "kaworu", "luffy",
    "zoro", "nami", "sanji", "robin", "chopper", "usopp",
    "franky", "brook", "yuji", "megumi", "nobara", "gojo",
    "sukuna", "spike", "jet", "faye", "vash", "wolfwood",
    "guts", "casca", "griffith", "einar", "canute", "stark",
    "himmel", "mugen", "jin",
)

LABEL_COLOR_ADVISOR_BRANCH = "0075ca"
LABEL_COLOR_STATUS_WIP = "fbca04"
LABEL_COLOR_STATUS_REVIEW = "0e8a16"
LABEL_COLOR_STUDENT = "f9d0c4"


@dataclass(frozen=True)
class RoleSpec:
    role: str
    name: str
    env: dict[str, str]

    @property
    def key(self) -> str:
        return "advisor" if self.role == "advisor" else f"student-{self.name}"


def expand_student_names(
    n: int, names: tuple[str, ...] = STUDENT_NAMES
) -> list[str]:
    """Return names, adding numeric suffixes after the base list is exhausted."""
    return [
        (
            names[index % len(names)]
            if index < len(names)
            else f"{names[index % len(names)]}{index // len(names) + 1}"
        )
        for index in range(n)
    ]


def routing_labels(
    advisor_branch: str, student_names: list[str]
) -> dict[str, tuple[str, str]]:
    """Labels required for advisor/student PR routing."""
    return {
        advisor_branch: (
            LABEL_COLOR_ADVISOR_BRANCH,
            f"Advisor branch: {advisor_branch}",
        ),
        "status:wip": (LABEL_COLOR_STATUS_WIP, "Work in progress"),
        "status:review": (LABEL_COLOR_STATUS_REVIEW, "Ready for advisor review"),
        **{
            f"student:{name}": (
                LABEL_COLOR_STUDENT,
                f"Assigned to student {name}",
            )
            for name in student_names
        },
    }


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
        **({"WANDB_ENTITY": args.wandb_entity} if args.wandb_entity else {}),
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
        **({"WANDB_ENTITY": args.wandb_entity} if args.wandb_entity else {}),
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


def build_student_spec(args, tag: str, student_name: str) -> RoleSpec:
    return RoleSpec("student", student_name, build_student_env(args, tag, student_name))


def build_advisor_spec(args, tag: str, student_list: list[str]) -> RoleSpec:
    return RoleSpec("advisor", "advisor", build_advisor_env(args, tag, student_list))


def validate_secret_config_separation(
    role_specs: list[RoleSpec], secrets: dict[str, str]
) -> None:
    """Reject .env keys owned by senpai.yaml or the launcher."""
    runtime_names = {"SENPAI_ROLE"}
    for spec in role_specs:
        runtime_names.update(spec.env)
    overlap = sorted(runtime_names.intersection(secrets))
    if overlap:
        raise SystemExit(
            "ERROR: .env overlaps Senpai runtime settings: "
            f"{', '.join(overlap)}. Keep only credentials in .env; "
            "set these values in senpai.yaml or launch arguments."
        )
