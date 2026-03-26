#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="/workspace/senpai"

echo "=== Senpai Student: $STUDENT_NAME ==="
echo "Repo:   $REPO_URL (branch: $REPO_BRANCH)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Repo already cloned by the deployment args block
cd "$WORKDIR"
uv pip install --system -e .

# --- Git identity for commits ---
git config user.name "senpai-$STUDENT_NAME"
git config user.email "senpai-$STUDENT_NAME@senpai"

# --- Register Weave Claude Code Plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# --- Start Hivemind (streams CC session logs to hivemind.wandb.tools) ---
mkdir -p ~/.claude/projects
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Install role instructions ---
cp instructions/CLAUDE-STUDENT.md "$WORKDIR/CLAUDE.md"

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

PROMPT="$(envsubst < "$WORKDIR/instructions/prompt-student.md" | sed '/^<!--$/,/^-->$/d')"


ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Ralph Loop iteration $ITERATION ($(date)) ==="

    # Return to latest advisor branch so student starts from the current baseline
    git checkout "$ADVISOR_BRANCH" 2>/dev/null || true
    git pull origin "$ADVISOR_BRANCH" 2>/dev/null || true

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/instructions/CLAUDE-STUDENT.md" "$WORKDIR/CLAUDE.md"

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    else
        claude -c -p "$PROMPT" --dangerously-skip-permissions || \
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    fi

    echo "=== Claude exited at $(date), restarting in 5s ==="
    sleep 5
done
