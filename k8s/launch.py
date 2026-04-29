#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import base64
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
    gpus_per_student: int = 8  # GPUs requested by each student pod
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
    dry_run: bool = False  # render manifests only: do not apply them or validate credentials
    preflight_only: bool = False  # validate credentials/access only: do not render or apply manifests


def render_student(template: str, student_name: str, tag: str, secret_name: str, args: Args) -> str:
    student_configmap_name = f"senpai-config-student-{tag}-{student_name}"
    student_deployment_name = f"senpai-{tag}-{student_name}"
    student_cpu = args.cpu_per_gpu * args.gpus_per_student
    student_memory_gi = args.memory_gi_per_gpu * args.gpus_per_student
    configmap = render_configmap(
        name=student_configmap_name,
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data={
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
            "PROBLEM_DIR": args.problem_dir,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
        },
    )
    deployment = render_template(template, {
        "STUDENT_DEPLOYMENT_NAME": student_deployment_name,
        "STUDENT_CONFIGMAP_NAME": student_configmap_name,
        "STUDENT_NAME": student_name,
        "RESEARCH_TAG": tag,
        "IMAGE": args.image,
        "ADVISOR_BRANCH": args.advisor_branch,
        "PVC_CLAIM_NAME": args.pvc_claim_name,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "LAUNCH_SECRET_NAME": secret_name,
        "STUDENT_CPU": str(student_cpu),
        "STUDENT_MEMORY": f"{student_memory_gi}Gi",
        "GPUS_PER_STUDENT": str(args.gpus_per_student),
    })
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
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
    }
    if args.extra_instructions:
        p = Path(args.extra_instructions)
        content = p.read_text() if p.exists() else args.extra_instructions
        data["EXTRA_INSTRUCTIONS_B64"] = base64.b64encode(content.encode()).decode()
    configmap = render_configmap(
        name=advisor_configmap_name,
        labels={"app": "senpai", "role": "advisor", "research-tag": tag},
        data=data,
    )
    deployment = render_template(template, {
        "ADVISOR_DEPLOYMENT_NAME": advisor_deployment_name,
        "ADVISOR_CONFIGMAP_NAME": advisor_configmap_name,
        "RESEARCH_TAG": tag,
        "PVC_CLAIM_NAME": args.pvc_claim_name,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "LAUNCH_SECRET_NAME": secret_name,
    })
    return configmap + "\n---\n" + deployment


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    if min(args.gpus_per_student, args.cpu_per_gpu, args.memory_gi_per_gpu) < 1:
        sys.exit("ERROR: --gpus_per_student, --cpu_per_gpu, and --memory_gi_per_gpu must all be at least 1")
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
    for name in student_list:
        manifest = render_student(student_template, name, args.tag, secret_name, args)
        if args.dry_run:
            print(f"--- Student: {name} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, f"student {name}")

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
        if args.advisor:
            print("Launched advisor pod")
        print(f"\nMonitor:")
        print(f"  kubectl get deployments -l research-tag={args.tag}")
        if args.advisor:
            print(f"  kubectl get deployment senpai-advisor-{args.tag}")
        if student_list:
            print(f"  kubectl logs -f deployment/senpai-{args.tag}-{student_list[0]}")
        print(f"\nStop:")
        print(f"  kubectl delete deployments,configmaps,secrets -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
