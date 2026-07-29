# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""User-facing Senpai launcher shared by every compute backend."""

import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

from .credentials import load_workload_secrets, workload_secret_names
from .docker_backend import launch_docker
from .kubernetes_backend import existing_student_names, launch_kubernetes
from .preflight import (
    ensure_advisor_branch,
    ensure_target_repo_labels,
    preflight_check_anthropic_api_key,
    preflight_check_exa_api_key,
    preflight_check_target_repo_access,
    preflight_check_target_repo_branch,
    preflight_check_wandb_api_key,
)
from .specs import (
    build_advisor_spec,
    build_student_spec,
    expand_student_names,
    routing_labels,
    target_repo_slug,
    validate_secret_config_separation,
)

ROOT = Path(__file__).resolve().parents[2]
SENPAI_CONFIG = ROOT / "senpai.yaml"
DOTENV_PATH = ROOT / ".env"


@dataclass
class Args:
    """Launch senpai advisor and/or student agents."""

    tag: str  # research tag (e.g. mar13)
    target_repo_url: str  # problem-package repo cloned into $PROBLEM_DIR — REQUIRED
    backend: str = "kubernetes"  # compute backend: kubernetes or docker
    target_repo_branch: str = ""  # target base branch; empty = repo default
    problem_dir: str = "target/"  # active problem directory
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # number of students (ignored when names is provided)
    student_prefix: str = ""  # unique prefix for parallel launch assignments
    gpus_per_student: int = 8  # GPUs requested by each student
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    repo_url: str = "https://github.com/wandb/senpai.git"  # runner repo
    repo_branch: str = "main"  # runner branch
    image: str = "ghcr.io/wandb/senpai:latest"  # worker container image
    wandb_entity: str | None = None  # W&B team; defaults to the API key's entity
    wandb_project: str = "senpai-v1"  # W&B project
    human_issues: bool = True  # allow human GitHub issue triage
    advisor_branch: str = "schmidhuber"  # target integration branch
    gh_history_scope: str = "branch"  # branch, fresh, or repo
    pvc_claim_name: str = "new-pvc"  # Kubernetes dataset PVC
    pvc_mount_path: str = "/mnt/new-pvc"  # dataset path inside workers
    advisor: bool = False  # also launch the advisor
    extra_instructions: str = ""  # literal instructions or a .md path
    timeout_minutes: float = 30.0  # per-training-run wall-clock limit
    max_epochs: int = 50  # per-training-run epoch limit
    poll_interval_s: int = 600  # outer-loop sleep between GitHub polls
    poll_jitter_s: int = 120  # max random jitter added to poll sleeps
    stale_wip_seconds: int = 7200  # advisor threshold for stale WIP PRs
    advisor_claude_watchdog_interval_s: int = 60
    advisor_claude_min_runtime_s: int = 600
    advisor_claude_stale_log_s: int = 1200
    student_claude_watchdog_interval_s: int = 300
    student_claude_watchdog_jitter_s: int = 60
    student_claude_min_runtime_s: int = 600
    student_claude_stale_log_s: int = 1200
    student_assignment_drift_grace_s: int = 1800
    start_gate_path: str = ""  # optional shared launch gate file
    docker_run_root: str = "~/.senpai/runs"  # saved Compose definitions
    docker_dataset_path: str = ""  # optional host dataset path
    docker_gpu_ids: str = ""  # comma-separated IDs; default assigns 0..N
    dry_run: bool = False  # render only; do not validate or change infrastructure
    preflight_only: bool = False  # validate only; do not launch


def _validate_args(args: Args) -> None:
    if args.backend not in {"kubernetes", "docker"}:
        sys.exit("ERROR: --backend must be one of: kubernetes, docker")
    if min(args.cpu_per_gpu, args.memory_gi_per_gpu) < 1:
        sys.exit("ERROR: --cpu_per_gpu and --memory_gi_per_gpu must both be at least 1")
    if args.gpus_per_student < 0:
        sys.exit("ERROR: --gpus_per_student must be non-negative")
    if args.backend == "kubernetes" and args.gpus_per_student < 1:
        sys.exit("ERROR: Kubernetes launches require --gpus_per_student at least 1")
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(args.repo_url):
        sys.exit("ERROR: --target_repo_url must be a different repo from --repo_url")

    positive = (
        "poll_interval_s",
        "advisor_claude_watchdog_interval_s",
        "advisor_claude_min_runtime_s",
        "advisor_claude_stale_log_s",
        "student_claude_watchdog_interval_s",
        "student_claude_min_runtime_s",
        "student_claude_stale_log_s",
    )
    non_negative = (
        "poll_jitter_s",
        "stale_wip_seconds",
        "student_claude_watchdog_jitter_s",
        "student_assignment_drift_grace_s",
    )
    for name in positive:
        if getattr(args, name) < 1:
            sys.exit(f"ERROR: --{name} must be at least 1")
    for name in non_negative:
        if getattr(args, name) < 0:
            sys.exit(f"ERROR: --{name} must be non-negative")


def _student_names(args: Args) -> list[str]:
    names = (
        [name.strip() for name in args.names.split(",")]
        if args.names
        else expand_student_names(args.n_students)
    )
    if args.student_prefix:
        names = [f"{args.student_prefix}-{name}" for name in names]
    return names


def _preflight(args: Args, secrets: dict[str, str]) -> str:
    github_token = secrets["GITHUB_TOKEN"]
    preflight_check_target_repo_access(args.target_repo_url, github_token)
    args.target_repo_branch = preflight_check_target_repo_branch(
        args.target_repo_url,
        github_token,
        args.target_repo_branch,
    )
    preflight_check_anthropic_api_key(secrets["ANTHROPIC_API_KEY"])
    preflight_check_exa_api_key(secrets["EXA_API_KEY"])
    args.wandb_entity = preflight_check_wandb_api_key(
        secrets["WANDB_API_KEY"],
        args.wandb_entity,
    )
    return github_token


def main() -> None:
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    _validate_args(args)

    if args.dry_run and not args.preflight_only:
        secrets = {name: "<REDACTED>" for name in workload_secret_names(DOTENV_PATH)}
        github_token = ""
    else:
        secrets = load_workload_secrets(DOTENV_PATH)
        github_token = _preflight(args, secrets)

    student_names = _student_names(args)
    advisor_student_names = student_names
    if (
        args.backend == "kubernetes"
        and args.advisor
        and not args.dry_run
        and not args.preflight_only
    ):
        advisor_student_names = list(
            dict.fromkeys(existing_student_names(args.tag) + student_names)
        )

    role_specs = [build_student_spec(args, args.tag, name) for name in student_names]
    if args.advisor:
        role_specs.append(build_advisor_spec(args, args.tag, advisor_student_names))
    validate_secret_config_separation(role_specs, secrets)

    if args.preflight_only:
        print("Preflight OK — credentials, configuration, and target repo verified.")
        return

    if not args.dry_run:
        ensure_advisor_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
            args.advisor_branch,
        )
        ensure_target_repo_labels(
            args.target_repo_url,
            github_token,
            routing_labels(args.advisor_branch, student_names),
        )

    if args.backend == "docker":
        launch_docker(args, role_specs, secrets)
    else:
        launch_kubernetes(args, role_specs, secrets)
