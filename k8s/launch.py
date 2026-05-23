#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

from launch_helpers import (
    ensure_advisor_branch,
    existing_student_names,
    expand_student_names,
    ensure_target_repo_labels,
    kubectl_apply,
    preflight_check_anthropic_api_key,
    preflight_check_exa_api_key,
    preflight_check_target_repo_branch,
    preflight_check_target_repo_access,
    render_configmap,
    render_launch_secret,
    render_template,
    resolve_anthropic_api_key,
    resolve_exa_api_key,
    resolve_github_token,
    routing_labels,
    target_repo_slug,
)

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SENPAI_CONFIG = Path(__file__).parent.parent / "senpai.yaml"
DOTENV_PATH = Path(__file__).parent.parent / ".env"


@dataclass
class Args:
    """Launch senpai advisor and/or student agents on Kubernetes."""
    tag: str  # research tag (e.g. mar13)
    target_repo_url: str  # problem-package repo (entrypoint clones this into $PROBLEM_DIR; agent commits/PRs land here) — REQUIRED, no default
    target_repo_branch: str = ""  # target repo branch used as the base when creating advisor_branch; empty = target repo default branch
    problem_dir: str = "target/"  # active problem directory — entrypoint clones target_repo_url here (from senpai.yaml)
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # number of students to launch (ignored if --names is provided)
    student_prefix: str = ""  # make assignment labels unique across parallel launches using the same base names
    gpus_per_student: int = 8  # GPUs requested by each singleton student pod or packed student group
    students_per_gpu_pod: int = 1  # logical student loops to pack into each GPU pod
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    repo_url: str = "https://github.com/wandb/senpai.git"  # git repo URL (senpai runner)
    repo_branch: str = "main"  # git branch to clone (senpai runner)
    image: str = "ghcr.io/wandb/senpai:latest"  # container image for students
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity (team or username)
    wandb_project: str = "senpai-v1"  # W&B project name
    human_issues: bool = True  # allow human GitHub issue triage; disable for isolated launches
    advisor_branch: str = "schmidhuber"  # branch the advisor works on inside the problem-package repo (students PR into it; created from target_repo_branch if missing)
    gh_history_scope: str = "branch"  # branch=normal track memory, fresh=clean ablation, repo=whole-repo memory
    pvc_claim_name: str = "new-pvc"  # PVC name mounted into pods
    pvc_mount_path: str = "/mnt/new-pvc"  # mount path for the dataset PVC inside the containers
    advisor: bool = False  # also deploy the advisor pod (default: students only)
    extra_instructions: str = ""  # extra prompt text for the advisor: a .md file path or a literal string
    timeout_minutes: float = 30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    poll_interval_s: int = 600  # default advisor/student outer-loop sleep between GitHub polls
    poll_jitter_s: int = 120  # max random jitter added to outer-loop sleeps
    stale_wip_seconds: int = 7200  # advisor-action threshold for stale WIP PRs
    advisor_claude_watchdog_interval_s: int = 60  # how often advisor watchdog checks active Claude
    advisor_claude_min_runtime_s: int = 600  # minimum advisor runtime before stale-log intervention
    advisor_claude_stale_log_s: int = 1200  # advisor stale-log intervention threshold
    student_claude_watchdog_interval_s: int = 300  # how often student watchdog checks active Claude
    student_claude_watchdog_jitter_s: int = 60  # max jitter added to student watchdog checks
    student_claude_min_runtime_s: int = 600  # minimum student runtime before stale-log intervention
    student_claude_stale_log_s: int = 1200  # student stale-log intervention threshold
    student_assignment_drift_grace_s: int = 1800  # grace before stopping active work after assignment changes
    start_gate_path: str = ""  # optional shared file path that must exist before advisor/student loops begin
    dry_run: bool = False  # render manifests only: do not apply them or validate credentials
    preflight_only: bool = False  # validate credentials/access only: do not render or apply manifests


def validate_timing_args(args: Args) -> None:
    positive = [
        "poll_interval_s",
        "advisor_claude_watchdog_interval_s",
        "advisor_claude_min_runtime_s",
        "advisor_claude_stale_log_s",
        "student_claude_watchdog_interval_s",
        "student_claude_min_runtime_s",
        "student_claude_stale_log_s",
    ]
    non_negative = [
        "poll_jitter_s",
        "stale_wip_seconds",
        "student_claude_watchdog_jitter_s",
        "student_assignment_drift_grace_s",
    ]
    for name in positive:
        if getattr(args, name) < 1:
            sys.exit(f"ERROR: --{name} must be at least 1")
    for name in non_negative:
        if getattr(args, name) < 0:
            sys.exit(f"ERROR: --{name} must be non-negative")


def load_extra_instructions(extra_instructions: str) -> str:
    if not extra_instructions:
        return ""
    p = Path(extra_instructions)
    return p.read_text() if p.exists() else extra_instructions


def build_extra_instructions(args: Args, tag: str, student_list: list[str]) -> str:
    students = ", ".join(student_list)
    target_base = args.target_repo_branch or "<default>"
    pod_packing = ""
    if args.students_per_gpu_pod > 1:
        pod_packing = (
            f"- This launch may pack up to `{args.students_per_gpu_pod}` logical students into one GPU pod. "
            f"The pod requests `{args.gpus_per_student}` GPU(s) total, so coordinate heavy GPU work with "
            "the advisor and other assigned students instead of assuming exclusive device access.\n"
        )
    isolation = f"""# Launch isolation and run-limit rules

- This launch is scoped to research tag `{tag}`, advisor branch `{args.advisor_branch}`, and target base branch `{target_base}`.
- Only inspect, modify, or reason from `{args.advisor_branch}` plus PR branches assigned to these students in this launch: {students}.
- Do not inspect, compare, summarize, cherry-pick, borrow from, or base decisions on any PR or branch outside `{args.advisor_branch}` and the assigned student PR branches for this launch.
- Do not use unrelated experiment runs or historical results unless the human explicitly names them during this launch.
- Students branch from `{args.advisor_branch}`. Do not rebase or retarget work onto unrelated branches.
{pod_packing}- Treat `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` as hard per-training-run bounds. Do not override them or continue a run past them.
"""
    user_extra = load_extra_instructions(args.extra_instructions)
    return isolation if not user_extra else isolation + "\n# Additional operator instructions\n\n" + user_extra


def encoded_extra_instructions(args: Args, tag: str, student_list: list[str]) -> str:
    return base64.b64encode(build_extra_instructions(args, tag, student_list).encode()).decode()


def student_chunks(student_list: list[str], chunk_size: int) -> list[list[str]]:
    return [student_list[i:i + chunk_size] for i in range(0, len(student_list), chunk_size)]


def student_common_config_data(args: Args, tag: str) -> dict[str, str]:
    return {
        "REPO_URL": args.repo_url,
        "REPO_BRANCH": args.repo_branch,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        "GH_REPO": target_repo_slug(args.target_repo_url),
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
        "STUDENT_CLAUDE_WATCHDOG_INTERVAL_S": str(args.student_claude_watchdog_interval_s),
        "STUDENT_CLAUDE_WATCHDOG_JITTER_S": str(args.student_claude_watchdog_jitter_s),
        "STUDENT_CLAUDE_MIN_RUNTIME_S": str(args.student_claude_min_runtime_s),
        "STUDENT_CLAUDE_STALE_LOG_S": str(args.student_claude_stale_log_s),
        "STUDENT_ASSIGNMENT_DRIFT_GRACE_S": str(args.student_assignment_drift_grace_s),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }


def render_student_deployment(
    template: str,
    *,
    deployment_name: str,
    configmap_name: str,
    id_label_key: str,
    id_label_value: str,
    student_names_csv: str,
    container_name: str,
    runner_workdir: str,
    entrypoint: str,
    tag: str,
    secret_name: str,
    args: Args,
) -> str:
    student_cpu = args.cpu_per_gpu * args.gpus_per_student
    student_memory_gi = args.memory_gi_per_gpu * args.gpus_per_student
    return render_template(template, {
        "STUDENT_DEPLOYMENT_NAME": deployment_name,
        "STUDENT_CONFIGMAP_NAME": configmap_name,
        "STUDENT_METADATA_ID_LABEL": f"    {id_label_key}: {id_label_value}",
        "STUDENT_SELECTOR_ID_LABEL": f"      {id_label_key}: {id_label_value}",
        "STUDENT_TEMPLATE_ID_LABEL": f"        {id_label_key}: {id_label_value}",
        "STUDENT_NAMES_CSV": student_names_csv,
        "STUDENT_CONTAINER_NAME": container_name,
        "STUDENT_RUNNER_WORKDIR": runner_workdir,
        "STUDENT_ENTRYPOINT": entrypoint,
        "RESEARCH_TAG": tag,
        "IMAGE": args.image,
        "PVC_CLAIM_NAME": args.pvc_claim_name,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "LAUNCH_SECRET_NAME": secret_name,
        "STUDENT_CPU": str(student_cpu),
        "STUDENT_MEMORY": f"{student_memory_gi}Gi",
        "GPUS_PER_STUDENT": str(args.gpus_per_student),
    })


def render_student(template: str, student_name: str, tag: str, secret_name: str, args: Args) -> str:
    student_configmap_name = f"senpai-config-student-{tag}-{student_name}"
    student_deployment_name = f"senpai-{tag}-{student_name}"
    data = student_common_config_data(args, tag)
    data.update({
        "STUDENT_NAME": student_name,
        "EXTRA_INSTRUCTIONS_B64": encoded_extra_instructions(args, tag, [student_name]),
    })
    configmap = render_configmap(
        name=student_configmap_name,
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data=data,
    )
    deployment = render_student_deployment(
        template,
        deployment_name=student_deployment_name,
        configmap_name=student_configmap_name,
        id_label_key="student",
        id_label_value=student_name,
        student_names_csv=student_name,
        container_name="student",
        runner_workdir="/workspace/senpai",
        entrypoint="k8s/entrypoint-student.sh",
        tag=tag,
        secret_name=secret_name,
        args=args,
    )
    return configmap + "\n---\n" + deployment


def render_student_group(template: str, student_names: list[str], group_index: int, tag: str, secret_name: str, args: Args) -> str:
    group_name = f"group-{group_index + 1}"
    student_names_csv = ",".join(student_names)
    configmap_name = f"senpai-config-student-{tag}-{group_name}"
    deployment_name = f"senpai-{tag}-{group_name}"
    extra_by_student = {
        student: encoded_extra_instructions(args, tag, [student])
        for student in student_names
    }
    data = student_common_config_data(args, tag)
    data.update({
        "STUDENT_GROUP_NAME": group_name,
        "STUDENT_NAMES": student_names_csv,
        "STUDENTS_PER_GPU_POD": str(args.students_per_gpu_pod),
        "STUDENT_EXTRA_INSTRUCTIONS_B64_JSON_B64": base64.b64encode(json.dumps(extra_by_student).encode()).decode(),
    })
    configmap = render_configmap(
        name=configmap_name,
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data=data,
    )
    deployment = render_student_deployment(
        template,
        deployment_name=deployment_name,
        configmap_name=configmap_name,
        id_label_key="student-group",
        id_label_value=group_name,
        student_names_csv=student_names_csv,
        container_name="student-group",
        runner_workdir="/workspace/senpai-group",
        entrypoint="k8s/entrypoint-student-group.sh",
        tag=tag,
        secret_name=secret_name,
        args=args,
    )
    return configmap + "\n---\n" + deployment


def render_advisor(template: str, tag: str, student_list: list[str], secret_name: str, args: Args) -> str:
    advisor_configmap_name = f"senpai-config-advisor-{tag}"
    advisor_deployment_name = f"senpai-advisor-{tag}"
    data = {
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
        "ADVISOR_CLAUDE_WATCHDOG_INTERVAL_S": str(args.advisor_claude_watchdog_interval_s),
        "ADVISOR_CLAUDE_MIN_RUNTIME_S": str(args.advisor_claude_min_runtime_s),
        "ADVISOR_CLAUDE_STALE_LOG_S": str(args.advisor_claude_stale_log_s),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }
    data["EXTRA_INSTRUCTIONS_B64"] = encoded_extra_instructions(args, tag, student_list)
    configmap = render_configmap(
        name=advisor_configmap_name,
        labels={"app": "senpai", "role": "advisor", "research-tag": tag},
        data=data,
    )
    deployment = render_template(template, {
        "ADVISOR_DEPLOYMENT_NAME": advisor_deployment_name,
        "ADVISOR_CONFIGMAP_NAME": advisor_configmap_name,
        "RESEARCH_TAG": tag,
        "IMAGE": args.image,
        "PVC_CLAIM_NAME": args.pvc_claim_name,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "LAUNCH_SECRET_NAME": secret_name,
    })
    return configmap + "\n---\n" + deployment


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    if min(args.gpus_per_student, args.cpu_per_gpu, args.memory_gi_per_gpu) < 1:
        sys.exit("ERROR: --gpus_per_student, --cpu_per_gpu, and --memory_gi_per_gpu must all be at least 1")
    if args.students_per_gpu_pod < 1:
        sys.exit("ERROR: --students_per_gpu_pod must be at least 1")
    validate_timing_args(args)
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(args.repo_url):
        sys.exit("ERROR: --target_repo_url must be a different repo from --repo_url")

    github_token = anthropic_api_key = exa_api_key = ""
    if not args.dry_run or args.preflight_only:
        github_token = resolve_github_token(DOTENV_PATH)
        anthropic_api_key = resolve_anthropic_api_key(DOTENV_PATH)
        exa_api_key = resolve_exa_api_key(DOTENV_PATH)
        preflight_check_target_repo_access(args.target_repo_url, github_token)
        args.target_repo_branch = preflight_check_target_repo_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
        )
        preflight_check_anthropic_api_key(anthropic_api_key)
        preflight_check_exa_api_key(exa_api_key)
        if args.preflight_only:
            print("Preflight OK — credentials and target repo access verified.")
            return

    # Resolve student list
    if args.names:
        student_list = [n.strip() for n in args.names.split(",")]
    else:
        student_list = expand_student_names(args.n_students)
    if args.student_prefix:
        student_list = [f"{args.student_prefix}-{name}" for name in student_list]

    if not args.dry_run:
        ensure_advisor_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
            args.advisor_branch,
        )
        ensure_target_repo_labels(args.target_repo_url, github_token, routing_labels(args.advisor_branch, student_list))

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()
    secret_name = f"senpai-launch-secrets-{args.tag}"

    # --- Apply per-launch token secret first (pods reference it on startup) ---
    if args.dry_run:
        print(f"--- Secret: {secret_name} ---")
        print(
            render_launch_secret(
                args.tag,
                "<REDACTED_GITHUB_TOKEN>",
                "<REDACTED_ANTHROPIC_API_KEY>",
                "<REDACTED_EXA_API_KEY>",
            )
        )
        print()
    else:
        kubectl_apply(
            render_launch_secret(args.tag, github_token, anthropic_api_key, exa_api_key),
            f"secret {secret_name}",
        )

    # --- Deploy students ---
    if args.students_per_gpu_pod == 1:
        for name in student_list:
            manifest = render_student(student_template, name, args.tag, secret_name, args)
            if args.dry_run:
                print(f"--- Student: {name} ---")
                print(manifest)
                print()
            else:
                kubectl_apply(manifest, f"student {name}")
    else:
        for group_index, names in enumerate(student_chunks(student_list, args.students_per_gpu_pod)):
            manifest = render_student_group(student_template, names, group_index, args.tag, secret_name, args)
            group_name = f"group-{group_index + 1}"
            if args.dry_run:
                print(f"--- Student group: {group_name} ({', '.join(names)}) ---")
                print(manifest)
                print()
            else:
                kubectl_apply(manifest, f"student group {group_name}")

    advisor_student_list = student_list
    if args.advisor and not args.dry_run:
        advisor_student_list = list(dict.fromkeys(existing_student_names(args.tag) + student_list))

    # --- Deploy advisor ---
    if args.advisor:
        manifest = render_advisor(advisor_template, args.tag, advisor_student_list, secret_name, args)
        if args.dry_run:
            print("--- Advisor ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, "advisor")

    if not args.dry_run:
        print(f"\nLaunched {len(student_list)} students: {', '.join(student_list)}")
        if args.students_per_gpu_pod > 1:
            group_count = len(student_chunks(student_list, args.students_per_gpu_pod))
            print(f"Packed into {group_count} GPU pod(s), up to {args.students_per_gpu_pod} students per pod")
        if args.advisor:
            print("Launched advisor pod")
        print(f"\nMonitor:")
        print(f"  kubectl get deployments -l research-tag={args.tag}")
        if args.advisor:
            print(f"  kubectl get deployment senpai-advisor-{args.tag}")
        if student_list and args.students_per_gpu_pod == 1:
            print(f"  kubectl logs -f deployment/senpai-{args.tag}-{student_list[0]}")
        elif student_list:
            print(f"  kubectl logs -f deployment/senpai-{args.tag}-group-1")
        print(f"\nStop:")
        print(f"  kubectl delete deployments,configmaps,secrets -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
