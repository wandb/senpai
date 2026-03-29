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
cp "$WORKDIR/system_instructions/CLAUDE-STUDENT.md" "$WORKDIR/CLAUDE.md"

# --- Senpai plugin ---
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
source "$SENPAI_PLUGIN/scripts/senpai-gh.sh"

# --- Build prompts ---
# PROBLEM_DIR comes from ConfigMap (set by launch.py from senpai.yaml)
FULL_PROMPT="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-student.md" | sed '/^<!--$/,/^-->$/d')"
HEARTBEAT="Continue your student loop. Check for assigned PRs, check for human messages, and resume any in-progress work."

# --- System-level vars (survive compaction via --append-system-prompt) ---
SYSTEM_VARS="Student: $STUDENT_NAME | Branch: $ADVISOR_BRANCH | W&B: $WANDB_ENTITY/$WANDB_PROJECT"

# --- Launch Claude Code in Heartbeat Loop ---
export IS_SANDBOX=1

LOGDIR="/workspace/senpai/student_logs"
mkdir -p "$LOGDIR"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Student Heartbeat iteration $ITERATION ($(date)) ==="

    # Return to latest advisor branch so student starts from the current baseline
    git checkout "$ADVISOR_BRANCH" 2>/dev/null || true
    git pull origin "$ADVISOR_BRANCH" 2>/dev/null || true

    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) ==="
    echo "=== GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null) ==="

    # --- Pre-triage: check if there's work before invoking Claude ---
    ASSIGNED=$(senpai_poll_work "$STUDENT_NAME" 2>/dev/null || echo "[]")
    ASSIGNED_COUNT=$(echo "$ASSIGNED" | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))" 2>/dev/null || echo "0")
    ISSUE_COUNT=$(senpai_check_issues "student:$STUDENT_NAME" 2>/dev/null | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))" 2>/dev/null || echo "0")

    echo "=== Pre-triage: assigned=$ASSIGNED_COUNT issues=$ISSUE_COUNT ==="

    # --- Skip Claude if nothing to do ---
    if [ "$ASSIGNED_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ]; then
        echo "=== No work assigned, sleeping 60s ==="
        sleep 60
        continue
    fi

    # --- Build triaged state summary ---
    TRIAGE="Pre-triaged state: $ASSIGNED_COUNT assigned PRs | $ISSUE_COUNT human issues"

    # --- Choose prompt weight ---
    if [ "$ITERATION" -eq 1 ] || [ -f /tmp/student_compacted ]; then
        PROMPT="$FULL_PROMPT"
        rm -f /tmp/student_compacted
        echo "=== Using FULL prompt ==="
    else
        PROMPT="$HEARTBEAT"
        echo "=== Using HEARTBEAT prompt ==="
    fi

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/system_instructions/CLAUDE-STUDENT.md" "$WORKDIR/CLAUDE.md"

    echo "=== Log: $LOGFILE ==="

    START_TS=$(date +%s)
    EXIT_CODE=0
    if [ "$ITERATION" -eq 1 ]; then
        claude -p "${PROMPT}"$'\n\n'"${TRIAGE}" \
            --append-system-prompt "$SYSTEM_VARS" \
            --plugin-dir "$SENPAI_PLUGIN" \
            --max-turns 50 \
            --model "claude-opus-4-6[1m]" \
            --output-format stream-json --verbose \
            --dangerously-skip-permissions > "$LOGFILE" 2>&1 || EXIT_CODE=$?
    else
        claude -c -p "${PROMPT}"$'\n\n'"${TRIAGE}" \
            --append-system-prompt "$SYSTEM_VARS" \
            --plugin-dir "$SENPAI_PLUGIN" \
            --max-turns 50 \
            --model "claude-opus-4-6[1m]" \
            --output-format stream-json --verbose \
            --dangerously-skip-permissions > "$LOGFILE" 2>&1 || \
        claude -p "${FULL_PROMPT}"$'\n\n'"${TRIAGE}" \
            --append-system-prompt "$SYSTEM_VARS" \
            --plugin-dir "$SENPAI_PLUGIN" \
            --max-turns 50 \
            --model "claude-opus-4-6[1m]" \
            --output-format stream-json --verbose \
            --dangerously-skip-permissions > "$LOGFILE" 2>&1 || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    echo "=== Claude exited code=$EXIT_CODE after ${DURATION}s at $(date), next check in 5s ==="
    sleep 5
done
