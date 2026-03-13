#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch N autonomous senpai research agents as K8s Jobs."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

TEMPLATE = Path(__file__).parent / "agent-job.yaml"


@dataclass
class Args:
    """Launch autonomous research agents on Kubernetes."""
    tag: str  # research tag (e.g. mar13)
    name: str = ""  # researcher name (e.g. pepe); defaults to tag
    n_agents: int = 4  # number of agents to launch
    repo_url: str = "https://github.com/wandb/senpai.git"  # git repo URL
    repo_branch: str = "main"  # git branch to clone
    image: str = "ghcr.io/tcapelle/senpai-agent:latest"  # container image
    wandb_entity: str = ""  # W&B entity (team or username)
    dry_run: bool = False  # print manifests without applying


def render_job(template: str, agent_name: str, tag: str, args: Args) -> str:
    """Replace {{PLACEHOLDER}} tokens in the job template."""
    replacements = {
        "AGENT_NAME": agent_name,
        "RESEARCH_TAG": tag,
        "WANDB_ENTITY": args.wandb_entity,
    }
    out = template
    for key, value in replacements.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    # These aren't wrapped in {{ }} — match on exact values
    out = out.replace(
        'value: "https://github.com/wandb/senpai.git"',
        f'value: "{args.repo_url}"',
    )
    out = out.replace(
        'value: "main"',
        f'value: "{args.repo_branch}"',
    )
    out = out.replace(
        "image: ghcr.io/tcapelle/senpai-agent:latest",
        f"image: {args.image}",
    )
    return out


def main():
    args = sp.parse(Args)
    template = TEMPLATE.read_text()

    base_name = args.name or args.tag
    for i in range(args.n_agents):
        agent_name = f"{base_name}-{i}"
        manifest = render_job(template, agent_name, args.tag, args)

        if args.dry_run:
            print(f"--- Agent: {agent_name} ---")
            print(manifest)
            print()
        else:
            print(f"Launching agent: {agent_name}")
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

    if not args.dry_run:
        print(f"\nLaunched {args.n_agents} agents with tag '{args.tag}'")
        print(f"Monitor: kubectl get jobs -l research-tag={args.tag}")
        print(f"Logs:    kubectl logs -f job/senpai-{base_name}-0")
        print(f"Stop:    kubectl delete jobs -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
