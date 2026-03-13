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

AGENT_NAMES = [
    "frieren", "fern", "tanjiro", "nezuko", "alphonse", "edward",
    "thorfinn", "askeladd", "violet", "gilbert", "senku", "kohaku",
    "emma", "norman", "chihiro", "haku", "shoya", "shouko",
    "mitsuha", "taki", "shinji", "rei", "kaneda", "tetsuo",
]


@dataclass
class Args:
    """Launch autonomous research agents on Kubernetes."""
    tag: str  # research tag (e.g. mar13)
    names: str = ""  # comma-separated agent names to launch (e.g. "agata,claudio")
    n_agents: int = 4  # number of agents to launch (ignored if --names is provided)
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

    if args.names:
        agent_list = [n.strip() for n in args.names.split(",")]
    else:
        if args.n_agents > len(AGENT_NAMES):
            print(f"ERROR: max {len(AGENT_NAMES)} agents (got {args.n_agents})", file=sys.stderr)
            sys.exit(1)
        agent_list = AGENT_NAMES[:args.n_agents]
    for agent_name in agent_list:
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
        print(f"\nLaunched {len(agent_list)} agents: {', '.join(agent_list)}")
        print(f"Monitor: kubectl get jobs -l research-tag={args.tag}")
        print(f"Logs:    kubectl logs -f job/senpai-{agent_list[0]}")
        print(f"Stop:    kubectl delete jobs -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
