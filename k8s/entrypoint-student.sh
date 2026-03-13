#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

# --- Config from environment ---
REPO_URL="${SENPAI_REPO_URL:?SENPAI_REPO_URL is required}"
REPO_BRANCH="${SENPAI_REPO_BRANCH:-main}"
STUDENT_NAME="${SENPAI_STUDENT_NAME:?SENPAI_STUDENT_NAME is required}"

WORKDIR="/workspace/senpai"

echo "=== Senpai Student: $STUDENT_NAME ==="
echo "Repo:   $REPO_URL (branch: $REPO_BRANCH)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# --- Clone repo and install deps ---
if [ ! -d "$WORKDIR/.git" ]; then
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
rm -rf .claude/skills/list-experiments
uv pip install --system -e .

# --- Git identity for commits ---
git config user.name "senpai-$STUDENT_NAME"
git config user.email "senpai-$STUDENT_NAME@senpai"

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install gh CLI ---
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
apt-get update && apt-get install -y gh
# gh uses GITHUB_TOKEN env var automatically, no explicit login needed
echo "=== gh auth ready (using GITHUB_TOKEN env var) ==="

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

PROMPT="$(cat <<EOF
You are a senpai research student (name: $STUDENT_NAME).

Read student.md for your full workflow, and program.md for the research context and constraints.

Your name is: $STUDENT_NAME
The dataset is at: /mnt/new-pvc/datasets/tandemfoil/
You have 8 GPUs on this node.

Always pass these flags to train.py:
  --agent $STUDENT_NAME --wandb_name "$STUDENT_NAME/<description>"

Start by checking for assigned PRs.
EOF
)"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Ralph Loop iteration $ITERATION ($(date)) ==="

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    else
        claude -c -p "$PROMPT" --dangerously-skip-permissions || \
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    fi

    echo "=== Claude exited at $(date), restarting in 5s ==="
    sleep 5
done
