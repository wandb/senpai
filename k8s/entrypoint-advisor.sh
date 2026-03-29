#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="/workspace/senpai"

echo "=== Senpai Advisor ==="
echo "Repo:     $REPO_URL (branch: $REPO_BRANCH)"
echo "Tag:      $RESEARCH_TAG"
echo "Students: $STUDENT_NAMES"

# Repo already cloned by the deployment args block
cd "$WORKDIR"

uv pip install --system -e .

# --- Git identity ---
git config user.name "senpai-advisor"
git config user.email "senpai-advisor@senpai"

# --- Create or checkout advisor branch ---
git fetch origin
if git rev-parse --verify "origin/$ADVISOR_BRANCH" >/dev/null 2>&1; then
    git checkout "$ADVISOR_BRANCH"
    git pull origin "$ADVISOR_BRANCH"
else
    git checkout -b "$ADVISOR_BRANCH"
    git push -u origin "$ADVISOR_BRANCH"
fi

# --- Install role instructions ---
cp "$WORKDIR/system_instructions/CLAUDE-ADVISOR.md" "$WORKDIR/CLAUDE.md"

# --- Register Weave Claude Plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# --- Start Hivemind (streams CC session logs to hivemind.wandb.tools) ---
mkdir -p ~/.claude/projects
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Senpai plugin ---
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
source "$SENPAI_PLUGIN/scripts/senpai-gh.sh"

# --- Build prompts ---
# PROBLEM_DIR comes from ConfigMap (set by launch.py from senpai.yaml)
FULL_PROMPT="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-advisor.md" | sed '/^<!--$/,/^-->$/d')"
HEARTBEAT="Continue your advisor loop. Survey state, review any completed experiments, assign work to idle students, and check for human messages."

# --- Append extra startup instructions if provided ---
if [ -n "${EXTRA_INSTRUCTIONS_B64:-}" ]; then
    FULL_PROMPT="${FULL_PROMPT}"$'\n\n# Finally, some additional instructions\n\n'"$(printf '%s' "$EXTRA_INSTRUCTIONS_B64" | base64 -d)"
fi

# --- System-level vars (survive compaction via --append-system-prompt) ---
SYSTEM_VARS="Students: $STUDENT_NAMES | Tag: $RESEARCH_TAG | Branch: $ADVISOR_BRANCH | W&B: $WANDB_ENTITY/$WANDB_PROJECT"

# --- Launch Claude Code in Heartbeat Loop ---
export IS_SANDBOX=1

LOGDIR="/workspace/senpai/advisor_logs"
mkdir -p "$LOGDIR"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Advisor Heartbeat iteration $ITERATION ($(date)) ==="
    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) ==="

    # --- Pre-triage: check GH state before invoking Claude ---
    REVIEW_COUNT=$(senpai_list_review_prs "$ADVISOR_BRANCH" 2>/dev/null | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))" 2>/dev/null || echo "0")
    ISSUE_COUNT=$(senpai_check_issues "$ADVISOR_BRANCH" 2>/dev/null | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))" 2>/dev/null || echo "0")
    IDLE=$(senpai_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH" 2>/dev/null || echo "")
    IDLE_COUNT=$(echo "$IDLE" | grep -c . 2>/dev/null || echo "0")

    echo "=== Pre-triage: reviews=$REVIEW_COUNT issues=$ISSUE_COUNT idle=$IDLE_COUNT ==="

    # --- Skip Claude if nothing to do ---
    if [ "$REVIEW_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ] && [ "$IDLE_COUNT" -eq 0 ]; then
        echo "=== Nothing actionable, sleeping 10 minutes ==="
        sleep 600
        continue
    fi

    # --- Build triaged state summary ---
    TRIAGE="Pre-triaged state: $REVIEW_COUNT PRs ready for review | $ISSUE_COUNT human issues | $IDLE_COUNT idle students"
    if [ -n "$IDLE" ]; then
        TRIAGE="$TRIAGE (idle: $(echo "$IDLE" | tr '\n' ',' | sed 's/,$//'))"
    fi

    # --- Choose prompt weight ---
    # Full prompt on first iteration or after compaction; heartbeat otherwise
    if [ "$ITERATION" -eq 1 ] || [ -f /tmp/advisor_compacted ]; then
        PROMPT="$FULL_PROMPT"
        rm -f /tmp/advisor_compacted
        echo "=== Using FULL prompt ==="
    else
        PROMPT="$HEARTBEAT"
        echo "=== Using HEARTBEAT prompt ==="
    fi

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/system_instructions/CLAUDE-ADVISOR.md" "$WORKDIR/CLAUDE.md"

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

    echo "=== Advisor exited code=$EXIT_CODE after ${DURATION}s at $(date), next check in 5 minutes ==="
    sleep 300
done
