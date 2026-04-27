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
    existing_student_names,
    expand_student_names,
    kubectl_apply,
    preflight_check_anthropic_api_key,
    preflight_check_target_repo_access,
    render_configmap,
    render_launch_secret,
    render_template,
    resolve_anthropic_api_key,
    resolve_github_token,
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
    problem_dir: str = "target/"  # active problem directory — entrypoint clones target_repo_url here (from senpai.yaml)
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # number of students to launch (ignored if --names is provided)
    repo_url: str = "https://github.com/wandb/senpai.git"  # git repo URL (senpai runner)
    repo_branch: str = "main"  # git branch to clone (senpai runner)
    image: str = "ghcr.io/wandb/senpai:latest"  # container image for students
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity (team or username)
    wandb_project: str = "senpai-v1"  # W&B project name
    advisor_branch: str = "schmidhuber"  # branch the advisor works on inside the problem-package repo (students PR into it; created from the problem-package default branch if missing)
    pvc_claim_name: str = "new-pvc"  # PVC name mounted into pods
    pvc_mount_path: str = "/mnt/new-pvc"  # mount path for the dataset PVC inside the containers
    advisor: bool = False  # also deploy the advisor pod (default: students only)
    extra_instructions: str = ""  # extra prompt text for the advisor: a .md file path or a literal string
    timeout_minutes: float = 30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    dry_run: bool = False  # print manifests without applying


def render_student(template: str, student_name: str, tag: str, secret_name: str, args: Args) -> str:
    configmap = render_configmap(
        name=f"senpai-config-student-{student_name}",
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data={
            "REPO_URL": args.repo_url,
            "REPO_BRANCH": args.repo_branch,
            "TARGET_REPO_URL": args.target_repo_url,
            "GH_REPO": target_repo_slug(args.target_repo_url),
            "STUDENT_NAME": student_name,
            "RESEARCH_TAG": tag,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "ADVISOR_BRANCH": args.advisor_branch,
            "WANDB_MODE": "online",
            "SENPAI_TIMEOUT_MINUTES": str(args.timeout_minutes),
            "SENPAI_MAX_EPOCHS": str(args.max_epochs),
            "PROBLEM_DIR": args.problem_dir,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
        },
    )
    deployment = render_template(template, {
        "STUDENT_NAME": student_name,
        "RESEARCH_TAG": tag,
        "IMAGE": args.image,
        "ADVISOR_BRANCH": args.advisor_branch,
        "PVC_CLAIM_NAME": args.pvc_claim_name,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "LAUNCH_SECRET_NAME": secret_name,
    })
    return configmap + "\n---\n" + deployment


def render_advisor(template: str, tag: str, student_list: list[str], secret_name: str, args: Args) -> str:
    data = {
        "REPO_URL": args.repo_url,
        "REPO_BRANCH": args.repo_branch,
        "TARGET_REPO_URL": args.target_repo_url,
        "GH_REPO": target_repo_slug(args.target_repo_url),
        "RESEARCH_TAG": tag,
        "STUDENT_NAMES": ",".join(student_list),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "ADVISOR_BRANCH": args.advisor_branch,
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
    }
    if args.extra_instructions:
        p = Path(args.extra_instructions)
        content = p.read_text() if p.exists() else args.extra_instructions
        data["EXTRA_INSTRUCTIONS_B64"] = base64.b64encode(content.encode()).decode()
    configmap = render_configmap(
        name="senpai-config-advisor",
        labels={"app": "senpai", "role": "advisor", "research-tag": tag},
        data=data,
    )
    deployment = render_template(template, {
        "RESEARCH_TAG": tag,
        "PVC_CLAIM_NAME": args.pvc_claim_name,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "LAUNCH_SECRET_NAME": secret_name,
    })
    return configmap + "\n---\n" + deployment


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    github_token = resolve_github_token(DOTENV_PATH)
    anthropic_api_key = resolve_anthropic_api_key(DOTENV_PATH)

    preflight_check_target_repo_access(args.target_repo_url, github_token)
    preflight_check_anthropic_api_key(anthropic_api_key)

    # Resolve student list
    if args.names:
        student_list = [n.strip() for n in args.names.split(",")]
    else:
        student_list = expand_student_names(args.n_students)

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()
    secret_name = f"senpai-launch-secrets-{args.tag}"

    # --- Apply per-launch token secret first (pods reference it on startup) ---
    if args.dry_run:
        print(f"--- Secret: {secret_name} ---")
        print(render_launch_secret(args.tag, "<REDACTED_GITHUB_TOKEN>", "<REDACTED_ANTHROPIC_API_KEY>"))
        print()
    else:
        kubectl_apply(render_launch_secret(args.tag, github_token, anthropic_api_key), f"secret {secret_name}")

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
        print(f"  kubectl get deployment senpai-advisor")
        if student_list:
            print(f"  kubectl logs -f deployment/senpai-{student_list[0]}")
        print(f"\nStop:")
        print(f"  kubectl delete deployments,configmaps,secrets -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
