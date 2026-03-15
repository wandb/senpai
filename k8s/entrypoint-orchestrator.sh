#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

RESEARCH_TAG="${SENPAI_RESEARCH_TAG:?SENPAI_RESEARCH_TAG is required}"
WORKDIR="/workspace/senpai"
cd "$WORKDIR"

# --- Install deps ---
uv pip install --system -e .

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install W&B skill ---
git clone --depth 1 https://github.com/wandb/skills.git /tmp/wandb-skills
cd /tmp/wandb-skills && bash install.sh --global --yes
cd "$WORKDIR"

# --- Launch Claude Code as orchestrator ---
export IS_SANDBOX=1

exec claude -p "$(cat <<EOF
You are the senpai orchestrator. You manage a fleet of autonomous research agents on Kubernetes.

Your job:
1. Monitor the research fleet: kubectl get jobs -l app=senpai
2. Read agent research journals: ls /mnt/new-pvc/senpai/journals/ — each agent maintains a markdown journal with what they've tried, their hypotheses, results, and plans
3. Query W&B (wandb-applied-ai-team/senpai) for metrics and run history
4. Launch new agents when needed: python k8s/launch.py --tag $RESEARCH_TAG --names "name1,name2" --wandb_entity wandb-applied-ai-team --repo_branch k8s-service
5. Stop underperforming agents: kubectl delete job senpai-<name>

Available tools:
- kubectl: manage pods and jobs in the cluster
- python k8s/launch.py: launch new research agents
- W&B skill: query wandb for run metrics, compare experiments
- Research journals: /mnt/new-pvc/senpai/journals/<agent>.md (read)

Read program.md for the full research protocol context.

The research tag is: $RESEARCH_TAG
The W&B project is: wandb-applied-ai-team/senpai

Start by surveying the current state: read all agent journals, check what agents are running, query W&B for the current best metrics. Then wait for instructions.
EOF
)" --dangerously-skip-permissions
