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

# --- Install role instructions ---
cp instructions/CLAUDE-STUDENT.md "$WORKDIR/CLAUDE.md"

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

PROMPT="$(envsubst < "$WORKDIR/instructions/prompt-student.md" | sed '/^<!--$/,/^-->$/d')"

LOGDIR="/workspace/senpai/student_logs"
mkdir -p "$LOGDIR"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Ralph Loop iteration $ITERATION ($(date)) ==="
    echo "=== Log: $LOGFILE ==="

    # Return to latest advisor branch so student starts from the current baseline
    git checkout "$ADVISOR_BRANCH" 2>/dev/null || true
    git pull origin "$ADVISOR_BRANCH" 2>/dev/null || true

    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) ==="
    echo "=== GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null) ==="

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/instructions/CLAUDE-STUDENT.md" "$WORKDIR/CLAUDE.md"

    START_TS=$(date +%s)
    EXIT_CODE=0
    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || EXIT_CODE=$?
    else
        claude -c -p "$PROMPT" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || \
        claude -p "$PROMPT" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    echo "=== Claude exited code=$EXIT_CODE after ${DURATION}s at $(date), restarting in 5s ==="
    sleep 5
done
