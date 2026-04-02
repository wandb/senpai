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

# --- Create logs directory ---
LOGDIR="$WORKDIR/advisor_logs"
mkdir -p "$LOGDIR"

# --- Start Hivemind logging service (streams CC session logs to hivemind.wandb.tools) ---
mkdir -p ~/.claude/projects
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Load CC run command helper function ---
source "$WORKDIR/k8s/run-senpai-claude.sh"

# --- Register Weave CC plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# --- Register Senpai CC plugin (skills + tools for common git tasks; CC uses --plugin-dir SENPAI_PLUGIN for tools) ---
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
source "$SENPAI_PLUGIN/scripts/senpai-gh.sh"

# --- Build prompts (CC auto-discovers CLAUDE.md for role instructions) ---
TASK_INSTRUCTIONS="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-advisor.md" | sed '/^<!--$/,/^-->$/d')"
PROMPT="${TASK_INSTRUCTIONS}"

# Append extra instructions from launch.py if provided
if [ -n "${EXTRA_INSTRUCTIONS_B64:-}" ]; then
    PROMPT="${PROMPT}"$'\n\n# Finally, some additional instructions\n\n'"$(printf '%s' "$EXTRA_INSTRUCTIONS_B64" | base64 -d)"
fi

# Add "$KEY_INFO" (reminder of student names etc) to PROMPT
KEY_INFO=$'\n\n Key information:\n\n Students: '"$STUDENT_NAMES"' | Tag: '"$RESEARCH_TAG"' | Advisor Branch: '"$ADVISOR_BRANCH"' | W&B entity/project: '"$WANDB_ENTITY"'/'"$WANDB_PROJECT"$'\n'
FULL_PROMPT="${PROMPT}"$'\n\n'"${KEY_INFO}"

# Heartbeat prompt for polling
HEARTBEAT_PROMPT="Continue your advisor loop. Attached is the current research state. Review any completed experiment PRs, assign work to all idle students, and check for human gh issues and comments."

# --- Last-check timestamp state for filtering PRs and issues ---
LAST_CHECK_FILE="$LOGDIR/.last_check_ts"

# --- Launch Claude Code Loop ---
export IS_SANDBOX=1

SLEEP_TIME_S=600
MAX_TURNS=999

ITERATION=0
while true; do
    # Set log dir, print loop and git info
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Advisor Heartbeat iteration $ITERATION ($(date)) ==="
    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) ==="

    # Overwrite CLAUDE.md with advisor-specific one — CC's git operations may clobber it with the developer copy
    cp "$WORKDIR/system_instructions/CLAUDE-ADVISOR.md" "$WORKDIR/CLAUDE.md"

    # --- Read last-check timestamp for filtering PRs and issues (empty on first run = no filtering) ---
    SINCE=""
    [ -f "$LAST_CHECK_FILE" ] && SINCE=$(cat "$LAST_CHECK_FILE")

    # --- Check research state before invoking CC ---
    REVIEW_JSON=$(list_ready_for_review_prs "$ADVISOR_BRANCH" "$SINCE")
    REVIEW_COUNT=$(printf '%s' "$REVIEW_JSON" | json_len)
    ISSUE_JSON=$(check_gh_issues "$ADVISOR_BRANCH" "$SINCE")
    ISSUE_COUNT=$(printf '%s' "$ISSUE_JSON" | json_len)
    IDLE_JSON=$(list_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH")
    IDLE_COUNT=$(printf '%s' "$IDLE_JSON" | json_len)

    # --- Derive watermark from the data we actually fetched (no gap, no overlap) ---
    WATERMARK=$(max_updated_at "$REVIEW_JSON" "$ISSUE_JSON")

    # --- Build triage info (used in logs, CC prompt, and skip check) ---
    TRIAGE_INFO="## Research state (since ${SINCE:-boot})"
    [ "$REVIEW_COUNT" -gt 0 ] && TRIAGE_INFO+=$'\n'"- **GitHub PRs to review ($REVIEW_COUNT):** $(printf '%s' "$REVIEW_JSON" | json_numbers)"
    [ "$ISSUE_COUNT" -gt 0 ]  && TRIAGE_INFO+=$'\n'"- **GitHub issues ($ISSUE_COUNT):** $(printf '%s' "$ISSUE_JSON" | json_numbers)"
    [ "$IDLE_COUNT" -gt 0 ]   && TRIAGE_INFO+=$'\n'"- **Idle students ($IDLE_COUNT):** $(printf '%s' "$IDLE_JSON" | json_join)"
    echo "$TRIAGE_INFO"

    # --- Log triage state and select prompt ---
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
        # --- Programmatic skip: skip rest of CC loop if nothing actionable ---
        if [ "$REVIEW_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ] && [ "$IDLE_COUNT" -eq 0 ]; then
            echo "=== Iteration $ITERATION: Nothing actionable, sleeping $SLEEP_TIME_S seconds ==="
            sleep "$SLEEP_TIME_S"
            continue
        fi

        echo "=== Iteration $ITERATION: Using heartbeat (HEARTBEAT_PROMPT) prompt ==="
        echo "$HEARTBEAT_PROMPT"
        echo "$TRIAGE_INFO"
        
        CONTINUE_PROMPT="${HEARTBEAT_PROMPT}"$'\n\n'"${TRIAGE_INFO}"
        run_senpai_claude 50 "$CONTINUE_PROMPT" -c || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    # --- Advance watermark to max updatedAt of items we fetched (only on success so failed runs retry) ---
    if [ "$EXIT_CODE" -eq 0 ] && [ -n "$WATERMARK" ]; then
        echo "$WATERMARK" > "$LAST_CHECK_FILE"
    fi

    echo "=== Advisor exited code=$EXIT_CODE after ${DURATION}s at $(date), next check in $SLEEP_TIME_S seconds ==="
    sleep "$SLEEP_TIME_S"
done
