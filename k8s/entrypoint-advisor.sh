#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="/workspace/senpai"
GH_HISTORY_SCOPE="${GH_HISTORY_SCOPE:-branch}"
TARGET_REPO_BRANCH="${TARGET_REPO_BRANCH:-}"
export SENPAI_ROLE="advisor"
export TARGET_WORKDIR="$WORKDIR/$PROBLEM_DIR"
GIT_CREDENTIAL_FILE="$WORKDIR/.git-credentials"
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"

echo "=== Senpai Advisor ==="
echo "Runner repo:  $REPO_URL (branch: $REPO_BRANCH)"
echo "Target repo:  $TARGET_REPO_URL (base branch: ${TARGET_REPO_BRANCH:-<default>}; advisor branch: $ADVISOR_BRANCH)"
echo "Problem dir:  $PROBLEM_DIR"
echo "Tag:          $RESEARCH_TAG"
echo "Students:     $STUDENT_NAMES"
echo "GitHub history: $GH_HISTORY_SCOPE"

source "$WORKDIR/k8s/wait-senpai-start-gate.sh"
wait_for_senpai_start_gate

# Senpai runner repo already cloned by the deployment args block
cd "$WORKDIR"
source "$SENPAI_PLUGIN/scripts/senpai-gh.sh"
install_senpai_git_guard "$WORKDIR" "$TARGET_WORKDIR" "$GIT_CREDENTIAL_FILE"

clone_single_target_branch() {
    local branch="$1"
    local depth=()
    [ "$GH_HISTORY_SCOPE" = "fresh" ] && depth=(--depth 1)
    git clone --branch "$branch" --single-branch "${depth[@]}" --no-tags "$TARGET_REPO_URL" "$PROBLEM_DIR"
}

clone_target_repo() {
    case "$GH_HISTORY_SCOPE" in
        branch|fresh)
            if clone_single_target_branch "$ADVISOR_BRANCH"; then
                return 0
            fi
            if [ -z "$TARGET_REPO_BRANCH" ]; then
                return 1
            fi
            rm -rf "$PROBLEM_DIR"
            if ! clone_single_target_branch "$TARGET_REPO_BRANCH"; then
                return 1
            fi
            cd "$WORKDIR/$PROBLEM_DIR"
            git checkout -b "$ADVISOR_BRANCH"
            git push -u origin "$ADVISOR_BRANCH"
            cd "$WORKDIR"
            ;;
        repo) git clone "$TARGET_REPO_URL" "$PROBLEM_DIR" ;;
        *) echo "ERROR: GH_HISTORY_SCOPE must be one of: branch, repo, fresh" >&2; exit 2 ;;
    esac
}

# Clone the problem-package repo into $PROBLEM_DIR (bring-your-own-repo —
# agent commits/PRs live in $TARGET_REPO_URL, not wandb/senpai).
if [ ! -d "$PROBLEM_DIR/.git" ] && ! clone_target_repo; then
    if [ -n "$TARGET_REPO_BRANCH" ]; then
        echo "ERROR: could not clone advisor branch '$ADVISOR_BRANCH' or target base branch '$TARGET_REPO_BRANCH'" >&2
        exit 1
    fi
    [ "$GH_HISTORY_SCOPE" = "repo" ] && exit 1
    depth=()
    [ "$GH_HISTORY_SCOPE" = "fresh" ] && depth=(--depth 1)
    git clone --single-branch "${depth[@]}" --no-tags "$TARGET_REPO_URL" "$PROBLEM_DIR"
    cd "$WORKDIR/$PROBLEM_DIR"
    git checkout -b "$ADVISOR_BRANCH"
    git push -u origin "$ADVISOR_BRANCH"
    cd "$WORKDIR"
fi
git config --global --unset-all credential.helper 2>/dev/null || true

uv pip install --system -e .

# --- Git identity (inside the problem-package repo) ---
cd "$WORKDIR/$PROBLEM_DIR"
git config user.name "senpai-advisor"
git config user.email "senpai-advisor@senpai"
gh repo set-default "$GH_REPO"
git config credential.helper "store --file=$GIT_CREDENTIAL_FILE"
install_senpai_target_git_guard "$TARGET_WORKDIR"

# --- Create or checkout advisor branch ---
if [ "$GH_HISTORY_SCOPE" != "repo" ]; then
    git remote set-branches origin "$ADVISOR_BRANCH"
    git config remote.origin.tagOpt --no-tags
fi
if git rev-parse --verify "origin/$ADVISOR_BRANCH" >/dev/null 2>&1; then
    git checkout "$ADVISOR_BRANCH"
    git pull --ff-only origin "$ADVISOR_BRANCH"
else
    if [ -n "$TARGET_REPO_BRANCH" ]; then
        git fetch origin "$TARGET_REPO_BRANCH"
        git checkout -B "$ADVISOR_BRANCH" "origin/$TARGET_REPO_BRANCH"
    else
        git checkout -b "$ADVISOR_BRANCH"
    fi
    git push -u origin "$ADVISOR_BRANCH"
fi

# --- Create logs directory ---
LOGDIR="$WORKDIR/advisor_logs"
mkdir -p "$LOGDIR"

# --- Install checked-in Claude Code config into user scope ---
mkdir -p "$HOME/.claude"
cp -a "$WORKDIR/.claude/." "$HOME/.claude/"

echo "=== Claude config installed ==="
ls "$HOME/.claude/skills/wandb-primary/SKILL.md" "$HOME/.claude/agents/researcher-agent.md"

# --- Start Hivemind logging service (streams CC session logs to hivemind.wandb.tools) ---
source "$WORKDIR/k8s/start-hivemind.sh"
start_hivemind

# --- Load CC run command helper function ---
source "$WORKDIR/k8s/run-senpai-claude.sh"
source "$WORKDIR/k8s/advisor-claude-watchdog.sh"

# --- Register Weave CC plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# $GH_REPO comes from the ConfigMap (set by launch.py = owner/repo of the
# problem-package repo). The gh CLI honours it natively, so every `gh`
# command and `gh api` call targets the problem-package repo, not senpai,
# regardless of the agent's cwd.

# --- Build prompts (CC auto-discovers CLAUDE.md for role instructions) ---
TASK_INSTRUCTIONS="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-advisor.md" | sed '/^<!--$/,/^-->$/d')"
PROMPT="${TASK_INSTRUCTIONS}"
EXTRA_LAUNCH_INSTRUCTIONS=""

# Append extra instructions from launch.py if provided
if [ -n "${EXTRA_INSTRUCTIONS_B64:-}" ]; then
    EXTRA_LAUNCH_INSTRUCTIONS="$(printf '%s' "$EXTRA_INSTRUCTIONS_B64" | base64 -d)"
    PROMPT="${PROMPT}"$'\n\n# Finally, some additional instructions\n\n'"${EXTRA_LAUNCH_INSTRUCTIONS}"
fi

# Add "$KEY_INFO" (reminder of student names etc) to PROMPT
KEY_INFO=$'\n\n Key information:\n\n Students: '"$STUDENT_NAMES"' | GPUs per Student: '"$GPUS_PER_STUDENT"' | Tag: '"$RESEARCH_TAG"' | Target repo: '"$GH_REPO"' | Target base branch: '"${TARGET_REPO_BRANCH:-<default>}"' | Advisor Branch: '"$ADVISOR_BRANCH"' | W&B entity/project: '"$WANDB_ENTITY"'/'"$WANDB_PROJECT"$'\n'
FULL_PROMPT="${PROMPT}"$'\n\n'"${KEY_INFO}"

# Heartbeat prompt for polling
HEARTBEAT_PROMPT="Continue your advisor loop. Attached is the current research state. Review any completed experiment PRs, assign work to all idle students, and check for human gh issues and comments. Protect the advisor context: use sub-agents for bulky PR, W&B, log, issue, literature, or repo scans; bring back compact sourced summaries; and write durable research state to PR comments or research docs."

# --- Last-check timestamp state for filtering GitHub issues ---
LAST_CHECK_FILE="$LOGDIR/.last_check_ts"

# --- Launch Claude Code Loop ---
export IS_SANDBOX=1

SLEEP_TIME_S="${SENPAI_ADVISOR_POLL_INTERVAL_S:-${SENPAI_POLL_INTERVAL_S:-600}}"
POLL_JITTER_S="${SENPAI_ADVISOR_POLL_JITTER_S:-${SENPAI_POLL_JITTER_S:-120}}"
MAX_TURNS=100000
export SENPAI_CLAUDE_TIMEOUT_SECONDS="${SENPAI_CLAUDE_TIMEOUT_SECONDS:-3600}"

ITERATION=0
while true; do
    # Set log dir, print loop and git info
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Advisor Heartbeat iteration $ITERATION ($(date)) ==="
    cd "$WORKDIR/$PROBLEM_DIR"
    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) in $PROBLEM_DIR ==="

    # Overwrite CLAUDE.md with advisor-specific one — CC's git operations may clobber it with the developer copy.
    # Whitelist vars so envsubst substitutes pod env vars but leaves Claude Code runtime vars
    # like ${CLAUDE_PLUGIN_ROOT} alone (those get expanded by the agent's shell at call time).
    envsubst '$PROBLEM_DIR $TARGET_REPO_URL $GH_REPO $ADVISOR_BRANCH $RESEARCH_TAG $GPUS_PER_STUDENT $WANDB_ENTITY $WANDB_PROJECT' \
        < "$WORKDIR/system_instructions/CLAUDE-ADVISOR.md" \
        | sed '/^<!--$/,/^-->$/d' \
        > "$WORKDIR/CLAUDE.md"

    # --- Read last-check timestamp for filtering PRs and issues (empty on first run = no filtering) ---
    SINCE=""
    [ -f "$LAST_CHECK_FILE" ] && SINCE=$(cat "$LAST_CHECK_FILE")

    # --- Check research state before invoking CC ---
    POLL_OK=1
    REVIEW_JSON=$(poll_or_empty "review-ready PR poll" list_ready_for_review_prs "$ADVISOR_BRANCH") || POLL_OK=0
    REVIEW_COUNT=$(printf '%s' "$REVIEW_JSON" | json_len)
    ADVISOR_ACTION_JSON=$(poll_or_empty "advisor-action PR poll" list_prs_requiring_advisor_action "$ADVISOR_BRANCH" "${SENPAI_STALE_WIP_SECONDS:-7200}" "$STUDENT_NAMES") || POLL_OK=0
    ADVISOR_ACTION_COUNT=$(printf '%s' "$ADVISOR_ACTION_JSON" | json_len)
    ISSUE_JSON=$(poll_or_empty "GitHub issue poll" check_gh_issues "$ADVISOR_BRANCH" "$SINCE") || POLL_OK=0
    ISSUE_COUNT=$(printf '%s' "$ISSUE_JSON" | json_len)
    IDLE_JSON=$(poll_or_empty "idle-student poll" list_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH") || POLL_OK=0
    IDLE_COUNT=$(printf '%s' "$IDLE_JSON" | json_len)
    POD_ANOMALY_JSON=$(poll_or_empty "student-pod anomaly poll" list_student_pod_anomalies "$STUDENT_NAMES" "$ADVISOR_BRANCH") || POLL_OK=0
    POD_ANOMALY_COUNT=$(printf '%s' "$POD_ANOMALY_JSON" | json_len)

    # --- Derive watermark from the data we actually fetched (no gap, no overlap) ---
    WATERMARK=$(max_updated_at "$ISSUE_JSON")

    # --- Build triage info (used in logs, CC prompt, and skip check) ---
    TRIAGE_INFO="## Research state (since ${SINCE:-boot})"
    [ "$REVIEW_COUNT" -gt 0 ] && TRIAGE_INFO+=$'\n'"- **GitHub PRs to review ($REVIEW_COUNT):** $(printf '%s' "$REVIEW_JSON" | json_numbers)"
    [ "$ADVISOR_ACTION_COUNT" -gt 0 ] && TRIAGE_INFO+=$'\n'"- **GitHub PRs requiring advisor action ($ADVISOR_ACTION_COUNT):** $(printf '%s' "$ADVISOR_ACTION_JSON" | json_advisor_action_summary)"
    [ "$ISSUE_COUNT" -gt 0 ]  && TRIAGE_INFO+=$'\n'"- **GitHub issues ($ISSUE_COUNT):** $(printf '%s' "$ISSUE_JSON" | json_numbers)"
    [ "$IDLE_COUNT" -gt 0 ]   && TRIAGE_INFO+=$'\n'"- **Idle students ($IDLE_COUNT):** $(printf '%s' "$IDLE_JSON" | json_join)"
    [ "$POD_ANOMALY_COUNT" -gt 0 ] && TRIAGE_INFO+=$'\n'"- **Student pod anomalies ($POD_ANOMALY_COUNT):** investigate these before assigning new work: $(printf '%s' "$POD_ANOMALY_JSON" | json_join)"
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
        run_advisor_claude_with_watchdog $MAX_TURNS "${FULL_PROMPT}"$'\n\n'"${TRIAGE_INFO}" || EXIT_CODE=$?
    else
        # --- Programmatic skip: skip rest of CC loop if nothing actionable ---
        if [ "$REVIEW_COUNT" -eq 0 ] && [ "$ADVISOR_ACTION_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ] && [ "$IDLE_COUNT" -eq 0 ] && [ "$POD_ANOMALY_COUNT" -eq 0 ]; then
            echo "=== Iteration $ITERATION: Nothing actionable, sleeping $SLEEP_TIME_S seconds + up to ${POLL_JITTER_S}s jitter ==="
            senpai_sleep_with_jitter "$SLEEP_TIME_S" "$POLL_JITTER_S"
            continue
        fi

        echo "=== Iteration $ITERATION: Using heartbeat (HEARTBEAT_PROMPT) prompt ==="
        echo "$HEARTBEAT_PROMPT"
        echo "$TRIAGE_INFO"

        CONTINUE_PROMPT="${HEARTBEAT_PROMPT}"
        if [ -n "$EXTRA_LAUNCH_INSTRUCTIONS" ]; then
            CONTINUE_PROMPT="${CONTINUE_PROMPT}"$'\n\n# Launch isolation and run-limit reminder\n\n'"${EXTRA_LAUNCH_INSTRUCTIONS}"
        fi
        CONTINUE_PROMPT="${CONTINUE_PROMPT}"$'\n\n'"${TRIAGE_INFO}"
        run_advisor_claude_with_watchdog 1000 "$CONTINUE_PROMPT" -c || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    # --- Advance watermark only after a complete poll and successful CC run. ---
    if [ "$EXIT_CODE" -eq 0 ] && [ "$POLL_OK" -eq 1 ] && [ -n "$WATERMARK" ]; then
        echo "$WATERMARK" > "$LAST_CHECK_FILE"
    elif [ "$EXIT_CODE" -eq 0 ] && [ "$POLL_OK" -ne 1 ]; then
        echo "=== Poll incomplete; not advancing last-check timestamp ==="
    fi

    echo "=== Advisor exited code=$EXIT_CODE after ${DURATION}s at $(date), next check in $SLEEP_TIME_S seconds + up to ${POLL_JITTER_S}s jitter ==="
    if [ "$EXIT_CODE" -eq 124 ]; then
        echo "=== Advisor Claude watchdog fired; re-polling immediately ==="
        continue
    fi
    senpai_sleep_with_jitter "$SLEEP_TIME_S" "$POLL_JITTER_S"
done
