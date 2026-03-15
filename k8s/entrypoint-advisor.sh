#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

REPO_URL="${SENPAI_REPO_URL:?SENPAI_REPO_URL is required}"
REPO_BRANCH="${SENPAI_REPO_BRANCH:-main}"
RESEARCH_TAG="${SENPAI_RESEARCH_TAG:?SENPAI_RESEARCH_TAG is required}"
STUDENT_NAMES="${SENPAI_STUDENT_NAMES:?SENPAI_STUDENT_NAMES is required}"

WORKDIR="/workspace/senpai"

echo "=== Senpai Advisor ==="
echo "Repo:     $REPO_URL (branch: $REPO_BRANCH)"
echo "Tag:      $RESEARCH_TAG"
echo "Students: $STUDENT_NAMES"

# --- Clone repo and install deps ---
if [ ! -d "$WORKDIR/.git" ]; then
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
uv pip install --system -e .

# --- Git identity ---
git config user.name "senpai-advisor"
git config user.email "senpai-advisor@senpai"

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install kubectl ---
curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /usr/local/bin/kubectl
chmod +x /usr/local/bin/kubectl

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
You are the senpai advisor.

Read advisor.md for your full workflow, and program.md for the research context and constraints.

Your students are: $STUDENT_NAMES
Research tag: $RESEARCH_TAG
W&B project: wandb-applied-ai-team/senpai

You can also monitor student pods: kubectl get deployments -l app=senpai

Start by surveying the current state: check W&B metrics, list existing PRs, and identify what needs attention.
EOF
)"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Advisor Loop iteration $ITERATION ($(date)) ==="

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    else
        claude -c -p "$PROMPT" --dangerously-skip-permissions || \
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    fi

    echo "=== Advisor exited at $(date), next check in 5 minutes ==="
    sleep 300
done
