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

# --- Start Hivemind (streams CC session logs to hivemind.wandb.tools) ---
mkdir -p ~/.claude/projects
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Load CC run command helper function ---
source "$WORKDIR/k8s/run-senpai-claude.sh"

# --- Register Weave Claude Code Plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# --- Register Senpai CC plugin ---
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
source "$SENPAI_PLUGIN/scripts/senpai-gh.sh"

# --- Build prompts (CC auto-discovers CLAUDE.md for role instructions) ---
TASK_INSTRUCTIONS="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-student.md" | sed '/^<!--$/,/^-->$/d')"
PROMPT="${TASK_INSTRUCTIONS}"

KEY_INFO=$'\n\nKey information:\n\nStudent: '"$STUDENT_NAME"' | Advisor Branch: '"$ADVISOR_BRANCH"' | W&B entity/project: '"$WANDB_ENTITY"'/'"$WANDB_PROJECT"$'\n'
FULL_PROMPT="${PROMPT}"$'\n\n'"${KEY_INFO}"

HEARTBEAT_PROMPT="Continue your student loop. Check for assigned PRs, check for GitHub issues, and resume any in-progress work."

# --- Launch Claude Code Loop ---
export IS_SANDBOX=1

LOGDIR="$WORKDIR/student_logs"
mkdir -p "$LOGDIR"
SLEEP_TIME_S=60
MAX_TURNS=10000

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Student Heartbeat iteration $ITERATION ($(date)) ==="

    # Return to latest advisor branch so student starts from the current baseline
    git checkout "$ADVISOR_BRANCH" 2>/dev/null || true
    git pull origin "$ADVISOR_BRANCH" 2>/dev/null || true

    # Overwrite CLAUDE.md with the student role instructions — git checkout/pull clobbers it with the developer copy
    sed '/^<!--$/,/^-->$/d' "$WORKDIR/system_instructions/CLAUDE-STUDENT.md" > "$WORKDIR/CLAUDE.md"

    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) ==="
    echo "=== GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null) ==="

    # --- Harvest backgrounded runs whose launching session already died ---
    if ! python3 "$SENPAI_PLUGIN/scripts/in_flight.py" harvest; then
        echo "=== HARNESS: in-flight harvest failed; continuing student heartbeat ==="
    fi

    # --- Check for work before invoking CC ---
    ASSIGNED_JSON=$(student_poll_for_work "$STUDENT_NAME")
    ASSIGNED_COUNT=$(printf '%s' "$ASSIGNED_JSON" | json_len)
    ISSUE_JSON=$(check_gh_issues "student:$STUDENT_NAME")
    ISSUE_COUNT=$(printf '%s' "$ISSUE_JSON" | json_len)

    # --- Build triage info ---
    TRIAGE_INFO="## Student research state"
    if [ "$ASSIGNED_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ]; then
        TRIAGE_INFO+=$'\n'"- No assigned PRs or issues."
    else
        [ "$ASSIGNED_COUNT" -gt 0 ] && TRIAGE_INFO+=$'\n'"- **Assigned PRs ($ASSIGNED_COUNT):** $(printf '%s' "$ASSIGNED_JSON" | json_numbers)"
        [ "$ISSUE_COUNT" -gt 0 ]    && TRIAGE_INFO+=$'\n'"- **GitHub issues ($ISSUE_COUNT):** $(printf '%s' "$ISSUE_JSON" | json_numbers)"
    fi
    echo "$TRIAGE_INFO"

    # --- Log triage and invoke CC ---
    echo "=== Log: $LOGFILE ==="
    echo "$TRIAGE_INFO" > "$LOGFILE"

    START_TS=$(date +%s)
    EXIT_CODE=0
    if [ "$ITERATION" -eq 1 ]; then
        echo "=== Iteration $ITERATION: Using FULL prompt + triage ==="
        echo "$FULL_PROMPT"
        echo "$TRIAGE_INFO"
        run_senpai_claude $MAX_TURNS "${FULL_PROMPT}"$'\n\n'"${TRIAGE_INFO}" || EXIT_CODE=$?
    else
        # --- Skip if nothing to do ---
        if [ "$ASSIGNED_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ]; then
            echo "=== No work assigned, sleeping $SLEEP_TIME_S seconds ==="
            sleep "$SLEEP_TIME_S"
            continue
        fi
        
        echo "=== Iteration $ITERATION: Using heartbeat prompt ==="
        echo "$HEARTBEAT_PROMPT"
        echo "$TRIAGE_INFO"
        # Student should start fresh each iteration (no -c) — experiments are self-contained
        run_senpai_claude $MAX_TURNS "${FULL_PROMPT}"$'\n\n'"${HEARTBEAT_PROMPT}"$'\n\n'"${TRIAGE_INFO}" || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    echo "=== Claude exited code=$EXIT_CODE after ${DURATION}s at $(date), next check in $SLEEP_TIME_S seconds ==="
    sleep "$SLEEP_TIME_S"
done
