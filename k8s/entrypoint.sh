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
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# --- Clone repo ---
git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"

# Install deps from the actual repo (in case pyproject.toml changed)
uv pip install --system -e .

# --- Git identity for commits ---
git config user.name "senpai-$AGENT_ID"
git config user.email "senpai-$AGENT_ID@autoresearch"

# --- Install W&B skill for Claude Code ---
git clone --depth 1 https://github.com/wandb/skills.git /tmp/wandb-skills
cd /tmp/wandb-skills && bash install.sh --global --yes
cd "$WORKDIR"

# --- Launch Claude Code ---
# IS_SANDBOX=1 allows --dangerously-skip-permissions to work.
# Each agent gets its own branch namespace under the shared research tag.
# Claude sees 1 GPU (the pod only has 1), so no CUDA_VISIBLE_DEVICES needed.
export IS_SANDBOX=1

exec claude -p "$(cat <<EOF
You are an autonomous research agent (ID: $AGENT_ID).

Read program.md for the full protocol. Follow the setup and experiment loop.

Key context for this run:
- Research tag: $RESEARCH_TAG
- Your agent ID: $AGENT_ID — use this in your branch names (e.g. autoresearch/$RESEARCH_TAG/$AGENT_ID)
- You have exactly 1 GPU (this pod has 1 GPU allocated). No need for CUDA_VISIBLE_DEVICES or worktrees.
- You are one of several parallel agents. Use --wandb_group to group your runs (e.g. --wandb_group $AGENT_ID).
- W&B project "senpai" is shared across all agents. Check existing runs there to avoid duplicating work.
- The dataset is at /mnt/new-pvc/datasets/tandemfoil/

Since you have a single GPU and no need for worktrees, your loop is simpler:
1. Create your branch: git checkout -b autoresearch/$RESEARCH_TAG/$AGENT_ID
2. Read the in-scope files (prepare.py, train.py, transolver.py, DATASET_REPORT.md)
3. Run baseline, record results
4. Loop: hypothesize, modify, commit, run, evaluate, advance or reset
5. Never stop.

Begin now.
EOF
)" --dangerously-skip-permissions
