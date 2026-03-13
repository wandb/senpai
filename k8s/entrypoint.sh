#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

# --- Config from environment ---
REPO_URL="${SENPAI_REPO_URL:?SENPAI_REPO_URL is required}"
REPO_BRANCH="${SENPAI_REPO_BRANCH:-main}"
AGENT_ID="${SENPAI_AGENT_ID:-agent-$(hostname)}"
RESEARCH_TAG="${SENPAI_RESEARCH_TAG:?SENPAI_RESEARCH_TAG is required}"

WORKDIR="/workspace/senpai"

echo "=== Senpai Agent: $AGENT_ID ==="
echo "Repo:   $REPO_URL (branch: $REPO_BRANCH)"
echo "Tag:    $RESEARCH_TAG"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# --- Clone repo and install deps ---
if [ ! -d "$WORKDIR/.git" ]; then
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
uv pip install --system -e .

# --- Git identity for commits ---
git config user.name "senpai-$AGENT_ID"
git config user.email "senpai-$AGENT_ID@autoresearch"

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install W&B skill for Claude Code ---
git clone --depth 1 https://github.com/wandb/skills.git /tmp/wandb-skills
cd /tmp/wandb-skills && bash install.sh --global --yes
cd "$WORKDIR"

# --- Launch Claude Code ---
# IS_SANDBOX=1 allows --dangerously-skip-permissions to work.
export IS_SANDBOX=1

exec claude -p "$(cat <<EOF
You are an autonomous research agent (ID: $AGENT_ID).

Read program.md for the full protocol. Follow the setup and experiment loop.

Key context for this run:
- Research tag: $RESEARCH_TAG
- Your agent ID: $AGENT_ID — use this in your branch names (e.g. autoresearch/$RESEARCH_TAG/$AGENT_ID)
- You have 8 GPUs on this node. Use the worktree-based parallel workflow from program.md.
- You are one of several parallel agents. Use --wandb_group to group your runs (e.g. --wandb_group $AGENT_ID).
- Prefix all W&B run names with "$AGENT_ID/" (e.g. --wandb_name "$AGENT_ID/baseline", --wandb_name "$AGENT_ID/wider-model-v2").
- W&B project "senpai" is shared across all agents. Check existing runs there to avoid duplicating work.
- The dataset is at /mnt/new-pvc/datasets/tandemfoil/

Begin now.
EOF
)" --dangerously-skip-permissions
