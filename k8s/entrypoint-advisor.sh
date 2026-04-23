#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="/workspace/senpai"

echo "=== Senpai Advisor ==="
echo "Runner repo:  $REPO_URL (branch: $REPO_BRANCH)"
echo "Target repo:  $TARGET_REPO_URL (branch: $TARGET_WORKING_BRANCH)"
echo "Problem dir:  $PROBLEM_DIR"
echo "Tag:          $RESEARCH_TAG"
echo "Students:     $STUDENT_NAMES"

# Senpai runner repo already cloned by the deployment args block.
cd "$WORKDIR"

# Clone the problem-package repo into $PROBLEM_DIR (bring-your-own-repo model —
# agent commits/PRs live in $TARGET_REPO_URL, not wandb/senpai).
if [ ! -d "$PROBLEM_DIR/.git" ]; then
    git clone --branch "$TARGET_WORKING_BRANCH" "$TARGET_REPO_URL" "$PROBLEM_DIR"
fi

uv pip install --system -e .

# --- Start Hivemind logging service ---
mkdir -p ~/.claude/projects
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Load CC run command helper function ---
source "$WORKDIR/k8s/run-senpai-claude.sh"

# --- Register Weave CC plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# --- Register Senpai CC plugin ---
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
source "$SENPAI_PLUGIN/scripts/senpai-gh.sh"

# Gh helpers target the problem-package repo, not senpai. Pre-seed the slug cache.
export _SENPAI_REPO="$TARGET_REPO"

# From here on, all git/gh ops happen inside the problem-package working tree.
cd "$WORKDIR/$PROBLEM_DIR"

git config user.name "senpai-advisor"
git config user.email "senpai-advisor@senpai"
git remote set-url origin "$TARGET_REPO_URL"

# --- Ensure the advisor integration branch exists in the problem-package repo ---
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

# --- Build prompts (CC auto-discovers CLAUDE.md for role instructions) ---
TASK_INSTRUCTIONS="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-advisor.md" | sed '/^<!--$/,/^-->$/d')"
PROMPT="${TASK_INSTRUCTIONS}"

# Append extra instructions from launch.py if provided
if [ -n "${EXTRA_INSTRUCTIONS_B64:-}" ]; then
    PROMPT="${PROMPT}"$'\n\n# Finally, some additional instructions\n\n'"$(printf '%s' "$EXTRA_INSTRUCTIONS_B64" | base64 -d)"
fi

KEY_INFO=$'\n\n Key information:\n\n Students: '"$STUDENT_NAMES"' | Tag: '"$RESEARCH_TAG"' | Target repo: '"$TARGET_REPO"' | Advisor branch: '"$ADVISOR_BRANCH"' | W&B entity/project: '"$WANDB_ENTITY"'/'"$WANDB_PROJECT"$'\n'
FULL_PROMPT="${PROMPT}"$'\n\n'"${KEY_INFO}"

HEARTBEAT_PROMPT="Continue your advisor loop. Attached is the current research state. Review any completed experiment PRs, assign work to all idle students, and check for human gh issues and comments."

# --- Last-check timestamp state for filtering PRs and issues ---
LAST_CHECK_FILE="$LOGDIR/.last_check_ts"

# --- Launch Claude Code Loop ---
export IS_SANDBOX=1

SLEEP_TIME_S=600
MAX_TURNS=10000

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Advisor Heartbeat iteration $ITERATION ($(date)) ==="

    cd "$WORKDIR/$PROBLEM_DIR"
    echo "=== Problem-package HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) in $PROBLEM_DIR ==="

    envsubst '$PROBLEM_DIR' < "$WORKDIR/system_instructions/CLAUDE-ADVISOR.md" | sed '/^<!--$/,/^-->$/d' > "$WORKDIR/CLAUDE.md"

    SINCE=""
    [ -f "$LAST_CHECK_FILE" ] && SINCE=$(cat "$LAST_CHECK_FILE")

    REVIEW_JSON=$(list_ready_for_review_prs "$ADVISOR_BRANCH" "$SINCE")
    REVIEW_COUNT=$(printf '%s' "$REVIEW_JSON" | json_len)
    ISSUE_JSON=$(check_gh_issues "$ADVISOR_BRANCH" "$SINCE")
    ISSUE_COUNT=$(printf '%s' "$ISSUE_JSON" | json_len)
    IDLE_JSON=$(list_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH")
    IDLE_COUNT=$(printf '%s' "$IDLE_JSON" | json_len)

    WATERMARK=$(max_updated_at "$REVIEW_JSON" "$ISSUE_JSON")

    TRIAGE_INFO="## Research state (since ${SINCE:-boot})"
    [ "$REVIEW_COUNT" -gt 0 ] && TRIAGE_INFO+=$'\n'"- **GitHub PRs to review ($REVIEW_COUNT):** $(printf '%s' "$REVIEW_JSON" | json_numbers)"
    [ "$ISSUE_COUNT" -gt 0 ]  && TRIAGE_INFO+=$'\n'"- **GitHub issues ($ISSUE_COUNT):** $(printf '%s' "$ISSUE_JSON" | json_numbers)"
    [ "$IDLE_COUNT" -gt 0 ]   && TRIAGE_INFO+=$'\n'"- **Idle students ($IDLE_COUNT):** $(printf '%s' "$IDLE_JSON" | json_join)"
    echo "$TRIAGE_INFO"

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
        if [ "$REVIEW_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ] && [ "$IDLE_COUNT" -eq 0 ]; then
            echo "=== Iteration $ITERATION: Nothing actionable, sleeping $SLEEP_TIME_S seconds ==="
            sleep "$SLEEP_TIME_S"
            continue
        fi

        echo "=== Iteration $ITERATION: Using heartbeat (HEARTBEAT_PROMPT) prompt ==="
        echo "$HEARTBEAT_PROMPT"
        echo "$TRIAGE_INFO"

        CONTINUE_PROMPT="${HEARTBEAT_PROMPT}"$'\n\n'"${TRIAGE_INFO}"
        run_senpai_claude 1000 "$CONTINUE_PROMPT" -c || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    if [ "$EXIT_CODE" -eq 0 ] && [ -n "$WATERMARK" ]; then
        echo "$WATERMARK" > "$LAST_CHECK_FILE"
    fi

    echo "=== Advisor exited code=$EXIT_CODE after ${DURATION}s at $(date), next check in $SLEEP_TIME_S seconds ==="
    sleep "$SLEEP_TIME_S"
done
