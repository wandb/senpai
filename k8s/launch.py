#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SENPAI_CONFIG = Path(__file__).parent.parent / "senpai.yaml"

STUDENT_NAMES = [
    "frieren", "fern", "tanjiro", "nezuko", "alphonse", "edward",
    "thorfinn", "askeladd", "violet", "gilbert", "senku", "kohaku",
    "emma", "norman", "chihiro", "haku", "shoya", "shouko",
    "mitsuha", "taki", "shinji", "rei", "kaneda", "tetsuo",
]


@dataclass
class Args:
    """Launch senpai advisor and/or student agents on Kubernetes."""
    tag: str  # research tag (e.g. mar13)
    problem: str = "cfd_tandemfoil"  # active problem directory (from senpai.yaml)
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # number of students to launch (ignored if --names is provided)
    repo_url: str = "https://github.com/wandb/senpai.git"  # git repo URL
    repo_branch: str = "main"  # git branch to clone
    image: str = "ghcr.io/wandb/senpai:latest"  # container image for students
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity (team or username)
    wandb_project: str = "senpai-v1"  # W&B project name
    advisor_branch: str = "noam"  # branch the advisor works on (PRs target this, not main)
    advisor: bool = False  # also deploy the advisor pod (default: students only)
    extra_instructions: str = ""  # extra prompt text for the advisor: a .md file path or a literal string
    timeout_minutes: float = 30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    dry_run: bool = False  # print manifests without applying


def render_template(template: str, replacements: dict[str, str]) -> str:
    """Replace {{PLACEHOLDER}} tokens in a K8s manifest template."""
    out = template
    for key, value in replacements.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def render_configmap(name: str, labels: dict[str, str], data: dict[str, str]) -> str:
    """Generate a ConfigMap YAML document."""
    lines = ["apiVersion: v1", "kind: ConfigMap", "metadata:", f"  name: {name}", "  labels:"]
    for k, v in labels.items():
        lines.append(f"    {k}: {v}")
    lines.append("data:")
    for k, v in data.items():
        lines.append(f"  {k}: \"{v}\"")
    return "\n".join(lines)


def render_student(template: str, student_name: str, tag: str, args: Args) -> str:
    configmap = render_configmap(
        name=f"senpai-config-student-{student_name}",
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data={
            "REPO_URL": args.repo_url,
            "REPO_BRANCH": args.repo_branch,
            "STUDENT_NAME": student_name,
            "RESEARCH_TAG": tag,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "ADVISOR_BRANCH": args.advisor_branch,
            "WANDB_MODE": "online",
            "SENPAI_TIMEOUT_MINUTES": str(args.timeout_minutes),
            "SENPAI_MAX_EPOCHS": str(args.max_epochs),
            "PROBLEM_DIR": args.problem,
        },
    )
    deployment = render_template(template, {
        "STUDENT_NAME": student_name,
        "RESEARCH_TAG": tag,
        "IMAGE": args.image,
        "ADVISOR_BRANCH": args.advisor_branch,
    })
    return configmap + "\n---\n" + deployment


def render_advisor(template: str, tag: str, student_list: list[str], args: Args) -> str:
    import base64
    data = {
        "REPO_URL": args.repo_url,
        "REPO_BRANCH": args.repo_branch,
        "RESEARCH_TAG": tag,
        "STUDENT_NAMES": ",".join(student_list),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "ADVISOR_BRANCH": args.advisor_branch,
        "PROBLEM_DIR": args.problem,
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
    deployment = render_template(template, {"RESEARCH_TAG": tag})
    return configmap + "\n---\n" + deployment


def kubectl_apply(manifest: str, name: str):
    """Apply a manifest via kubectl."""
    print(f"Launching: {name}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  {result.stdout.strip()}")


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))

    # Resolve student list
    if args.names:
        student_list = [n.strip() for n in args.names.split(",")]
    else:
        if args.n_students > len(STUDENT_NAMES):
            print(f"ERROR: max {len(STUDENT_NAMES)} students (got {args.n_students})", file=sys.stderr)
            sys.exit(1)
        student_list = STUDENT_NAMES[:args.n_students]

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()

    # --- Deploy students ---
    for name in student_list:
        manifest = render_student(student_template, name, args.tag, args)
        if args.dry_run:
            print(f"--- Student: {name} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, f"student {name}")

    # --- Deploy advisor ---
    if args.advisor:
        manifest = render_advisor(advisor_template, args.tag, student_list, args)
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
        print(f"  kubectl delete deployments,configmaps -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
