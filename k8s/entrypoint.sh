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
git config user.email "senpai-$AGENT_ID@senpai"

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install W&B skill for Claude Code ---
git clone --depth 1 https://github.com/wandb/skills.git /tmp/wandb-skills
cd /tmp/wandb-skills && bash install.sh --global --yes
cd "$WORKDIR"

# --- Launch Claude Code in Ralph Loop ---
# IS_SANDBOX=1 allows --dangerously-skip-permissions to work.
export IS_SANDBOX=1

PROMPT="$(cat <<EOF
You are an autonomous research agent (ID: $AGENT_ID).

Read program.md for the full protocol. Follow the setup and experiment loop.

Key context for this run:
- Research tag: $RESEARCH_TAG
- Your agent ID: $AGENT_ID — use this in your branch names (e.g. senpai/$RESEARCH_TAG/$AGENT_ID)
- You have 8 GPUs on this node. Use the worktree-based parallel workflow from program.md.
- You are one of several parallel agents. Always pass these flags to train.py:
  --agent $AGENT_ID --wandb_name "$AGENT_ID/<experiment-description>"
  Use --wandb_group only to group iterations on the same idea (e.g. --wandb_group "multi-scale-attn").
  For example: --agent $AGENT_ID --wandb_name "$AGENT_ID/baseline"
  Or: --agent $AGENT_ID --wandb_group "local-attention" --wandb_name "$AGENT_ID/local-attention-v2"
- W&B project "senpai" is shared across all agents. Check existing runs there to avoid duplicating work.
- The dataset is at /mnt/new-pvc/datasets/tandemfoil/
- Keep a research journal at /mnt/new-pvc/senpai/journals/$AGENT_ID.md — update it after each experiment with: what you tried, your hypothesis, whether it worked, and what you'll try next. This is how you communicate with the orchestrator and other agents.
- Before starting a new experiment, read other agents' journals at /mnt/new-pvc/senpai/journals/ to see what's been tried and what's working. Avoid duplicating their work.

Continue the experiment loop. Check your journal and results.tsv to see where you left off.
EOF
)"

# Ralph loop: same prompt fed repeatedly. Claude sees its own previous
# work in files, git, journal. If it exits (context limit, error, etc.),
# we restart with --continue so it picks up the conversation, then fall
# back to a fresh prompt if --continue fails.
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Ralph Loop iteration $ITERATION ($(date)) ==="

    if [ "$ITERATION" -eq 1 ]; then
        # First run: fresh prompt
        echo "$PROMPT" | claude -p --dangerously-skip-permissions || true
    else
        # Subsequent runs: try to continue the conversation, fall back to fresh
        echo "$PROMPT" | claude -p --continue --dangerously-skip-permissions || \
        echo "$PROMPT" | claude -p --dangerously-skip-permissions || true
    fi

    echo "=== Claude exited at $(date), restarting in 5s ==="
    sleep 5
done
