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
LOGDIR="/workspace/senpai/advisor_logs"
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

# --- Build prompts ---
# Advisor prompt (PROBLEM_DIR comes from ConfigMap (set by launch.py from senpai.yaml)
CLAUDE_DOT_MD="$(cat "$WORKDIR/system_instructions/CLAUDE-ADVISOR.md")"
TASK_INSTRUCTIONS="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-advisor.md" | sed '/^<!--$/,/^-->$/d')"
PROMPT="${CLAUDE_DOT_MD}"$'\n\n'"${TASK_INSTRUCTIONS}"

# Append extra instructions from launch.py if provided
if [ -n "${EXTRA_INSTRUCTIONS_B64:-}" ]; then
    PROMPT="${PROMPT}"$'\n\n# Finally, some additional instructions\n\n'"$(printf '%s' "$EXTRA_INSTRUCTIONS_B64" | base64 -d)"
fi

# Add "$KEY_INFO" (reminder of student names etc) to PROMPT
KEY_INFO=$'\n\n Key information:\n\n Students: '"$STUDENT_NAMES"' | Tag: '"$RESEARCH_TAG"' | Advisor Branch: '"$ADVISOR_BRANCH"' | W&B entity/project: '"$WANDB_ENTITY"'/'"$WANDB_PROJECT"$'\n'
FULL_PROMPT="${PROMPT}"$'\n\n'"${KEY_INFO}"

# Heartbeat prompt for polling
HEARTBEAT_PROMPT="Continue your advisor loop. Survey state, review any completed experiment PRs, assign work to all idle students, and check for human gh issues."

# --- Launch Claude Code Loop ---
export IS_SANDBOX=1

SLEEP_TIME_S=60  
MAX_TURNS=999

ITERATION=0
while true; do
    # Set log dir, print loop and git info
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Advisor Heartbeat iteration $ITERATION ($(date)) ==="
    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) ==="

    # --- Check research state before invoking CC ---
    REVIEW_COUNT=$(list_ready_for_review_prs "$ADVISOR_BRANCH" | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))")
    ISSUE_COUNT=$(check_gh_issues "$ADVISOR_BRANCH" | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))")
    IDLE_JSON=$(list_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH")
    IDLE_STUDENTS_COUNT=$(echo "$IDLE_JSON" | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))")

    TRIAGE_INFO="=== Research state: PR's ready for review count=$REVIEW_COUNT | Human issues count=$ISSUE_COUNT | Idle students count=$IDLE_STUDENTS_COUNT ==="
    echo "$TRIAGE_INFO"

    # --- Programmatic skip: skip rest of CC loop if nothing actionable ---
    if [ "$REVIEW_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ] && [ "$IDLE_STUDENTS_COUNT" -eq 0 ]; then
        echo "=== Nothing actionable, sleeping $SLEEP_TIME_S seconds ==="
        sleep "$SLEEP_TIME_S"
        continue
    fi

    # --- Continuing CC loop ---
    #  accumulate triage info
    if [ "$IDLE_STUDENTS_COUNT" -gt 0 ]; then
        IDLE_INFO="Idle student names: $(echo "$IDLE_JSON" | python3 -c "import sys,json; print(','.join(json.loads(sys.stdin.read())))")"
        echo "$IDLE_INFO"
        TRIAGE_INFO="${TRIAGE_INFO} | ${IDLE_INFO}"
    fi

    # --- Select prompt ---
    echo "=== Log: $LOGFILE ==="

    START_TS=$(date +%s)
    EXIT_CODE=0
    if [ "$ITERATION" -eq 1 ]; then
        echo "=== Using FULL prompt ($FULL_PROMPT) ==="
        run_senpai_claude $MAX_TURNS "$FULL_PROMPT" || EXIT_CODE=$?
    else
        echo "=== Using heartbeat (HEARTBEAT_PROMPT) prompt ==="
        CONTINUE_PROMPT="${HEARTBEAT_PROMPT}"$'\n\n'"${TRIAGE_INFO}"
        run_senpai_claude 50 "$CONTINUE_PROMPT" -c || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    echo "=== Advisor exited code=$EXIT_CODE after ${DURATION}s at $(date), next check in $SLEEP_TIME_S seconds ==="
    sleep "$SLEEP_TIME_S"
done
