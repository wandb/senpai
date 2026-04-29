#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="/workspace/senpai"
GH_HISTORY_SCOPE="${GH_HISTORY_SCOPE:-branch}"
export TARGET_WORKDIR="$WORKDIR/$PROBLEM_DIR"
GIT_CREDENTIAL_FILE="$WORKDIR/.git-credentials"
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"

echo "=== Senpai Student: $STUDENT_NAME ==="
echo "Runner repo:  $REPO_URL (branch: $REPO_BRANCH)"
echo "Target repo:  $TARGET_REPO_URL (branch: $ADVISOR_BRANCH)"
echo "Problem dir:  $PROBLEM_DIR"
echo "GitHub history: $GH_HISTORY_SCOPE"
echo "GPUs:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Senpai runner repo already cloned by the deployment args block
cd "$WORKDIR"
source "$SENPAI_PLUGIN/scripts/senpai-gh.sh"
install_senpai_git_guard "$WORKDIR" "$TARGET_WORKDIR" "$GIT_CREDENTIAL_FILE"

clone_target_repo() {
    local depth=()
    [ "$GH_HISTORY_SCOPE" = "fresh" ] && depth=(--depth 1)
    case "$GH_HISTORY_SCOPE" in
        branch|fresh) git clone --branch "$ADVISOR_BRANCH" --single-branch "${depth[@]}" --no-tags "$TARGET_REPO_URL" "$PROBLEM_DIR" ;;
        repo) git clone "$TARGET_REPO_URL" "$PROBLEM_DIR" ;;
        *) echo "ERROR: GH_HISTORY_SCOPE must be one of: branch, repo, fresh" >&2; exit 2 ;;
    esac
}

# Clone the problem-package repo into $PROBLEM_DIR (bring-your-own-repo —
# agent commits/PRs live in $TARGET_REPO_URL, not wandb/senpai).
[ -d "$PROBLEM_DIR/.git" ] || clone_target_repo
git config --global --unset-all credential.helper 2>/dev/null || true

uv pip install --system -e .

# --- Git identity for commits (inside the problem-package repo) ---
cd "$WORKDIR/$PROBLEM_DIR"
git config user.name "senpai-$STUDENT_NAME"
git config user.email "senpai-$STUDENT_NAME@senpai"
gh repo set-default "$GH_REPO"
git config credential.helper "store --file=$GIT_CREDENTIAL_FILE"
if [ "$GH_HISTORY_SCOPE" != "repo" ]; then
    git remote set-branches origin "$ADVISOR_BRANCH"
    git config remote.origin.tagOpt --no-tags
fi

# --- Install checked-in Claude Code config into user scope ---
mkdir -p "$HOME/.claude"
cp -a "$WORKDIR/.claude/." "$HOME/.claude/"

echo "=== Claude config installed ==="
ls "$HOME/.claude/skills/wandb-primary/SKILL.md" "$HOME/.claude/agents/researcher-agent.md"

# --- Start Hivemind (streams CC session logs to hivemind.wandb.tools) ---
mkdir -p "$HOME/.claude/projects"
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Load CC run command helper function ---
source "$WORKDIR/k8s/run-senpai-claude.sh"

# --- Register Weave Claude Code Plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# $GH_REPO comes from the ConfigMap (set by launch.py = owner/repo of the
# problem-package repo). The gh CLI honours it natively, so every `gh`
# command and `gh api` call targets the problem-package repo, not senpai,
# regardless of the agent's cwd.

# --- Build prompts (CC auto-discovers CLAUDE.md for role instructions) ---
TASK_INSTRUCTIONS="$(envsubst < "$WORKDIR/$PROBLEM_DIR/instructions/prompt-student.md" | sed '/^<!--$/,/^-->$/d')"
PROMPT="${TASK_INSTRUCTIONS}"

KEY_INFO=$'\n\nKey information:\n\nStudent: '"$STUDENT_NAME"' | GPUs per Student: '"$GPUS_PER_STUDENT"' | Target repo: '"$GH_REPO"' | Advisor Branch: '"$ADVISOR_BRANCH"' | W&B entity/project: '"$WANDB_ENTITY"'/'"$WANDB_PROJECT"$'\n'
FULL_PROMPT="${PROMPT}"$'\n\n'"${KEY_INFO}"

HEARTBEAT_PROMPT="Continue your student loop using the assigned PRs and GitHub issues listed in the Student research state below. The entrypoint owns assignment polling; do not start persistent GitHub polling monitors. For active training, use sparse wakeups plus training_log_status; do not stream per-epoch logs into Monitor."

# --- Launch Claude Code Loop ---
export IS_SANDBOX=1

LOGDIR="$WORKDIR/student_logs"
mkdir -p "$LOGDIR"
SLEEP_TIME_S=300
MAX_TURNS=100000

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Student Heartbeat iteration $ITERATION ($(date)) ==="

    # Return to latest advisor branch so student starts from the current baseline
    cd "$WORKDIR/$PROBLEM_DIR"
    git checkout "$ADVISOR_BRANCH" 2>/dev/null || true
    git pull --ff-only origin "$ADVISOR_BRANCH" 2>/dev/null || true

    # Overwrite CLAUDE.md with the student role instructions — git checkout/pull clobbers it with the developer copy.
    # Whitelist vars so envsubst substitutes pod env vars but leaves Claude Code runtime vars
    # like ${CLAUDE_PLUGIN_ROOT} alone (those get expanded by the agent's shell at call time).
    envsubst '$PROBLEM_DIR $TARGET_REPO_URL $GH_REPO $ADVISOR_BRANCH $RESEARCH_TAG $STUDENT_NAME $GPUS_PER_STUDENT $WANDB_ENTITY $WANDB_PROJECT' \
        < "$WORKDIR/system_instructions/CLAUDE-STUDENT.md" \
        | sed '/^<!--$/,/^-->$/d' \
        > "$WORKDIR/CLAUDE.md"

    echo "=== Git HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current) in $PROBLEM_DIR ==="
    echo "=== GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null) ==="

    # --- Check for work before invoking CC ---
    ASSIGNED_JSON=$(student_poll_for_work "$STUDENT_NAME" || printf '[]')
    ASSIGNED_COUNT=$(printf '%s' "$ASSIGNED_JSON" | json_len)
    ISSUE_JSON=$(check_gh_issues "student:$STUDENT_NAME" || printf '[]')
    ISSUE_COUNT=$(printf '%s' "$ISSUE_JSON" | json_len)

    if [ "$ASSIGNED_COUNT" -eq 1 ]; then
        ASSIGNED_HEAD=$(printf '%s' "$ASSIGNED_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["headRefName"])')
        git fetch origin "$ASSIGNED_HEAD"
        git checkout -B "$ASSIGNED_HEAD" FETCH_HEAD
    fi

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

    if [ "$ASSIGNED_COUNT" -gt 1 ]; then
        DUPLICATE_NUMBERS=$(printf '%s' "$ASSIGNED_JSON" | json_numbers)
        DUPLICATE_MARKER="STUDENT-DUPLICATE-ASSIGNMENT"
        DUPLICATE_BODY="${DUPLICATE_MARKER}: student:${STUDENT_NAME} has ${ASSIGNED_COUNT} active status:wip PRs (${DUPLICATE_NUMBERS}) on ${ADVISOR_BRANCH}. The student loop is skipping work until the advisor leaves exactly one active assignment."
        echo "ERROR: ${DUPLICATE_BODY}"
        printf '%s' "$ASSIGNED_JSON" | python3 -c 'import json,sys; print("\n".join(str(pr["number"]) for pr in json.loads(sys.stdin.read())))' |
        while IFS= read -r num; do
            [ -z "$num" ] && continue
            if ! pr_issue_comments "$num" | grep -q "$DUPLICATE_MARKER"; then
                gh_retry gh pr comment "$num" --repo "$GH_REPO" --body "$DUPLICATE_BODY"
            fi
        done
        sleep "$SLEEP_TIME_S"
        continue
    fi

    # The shell loop is the source of truth for assignment polling. If there
    # is no work, do not enter Claude Code; idle model sessions tend to invent
    # their own polling loops and can miss the label-based assignment contract.
    if [ "$ASSIGNED_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ]; then
        echo "=== No work assigned, sleeping $SLEEP_TIME_S seconds without invoking Claude ==="
        sleep "$SLEEP_TIME_S"
        continue
    fi

    START_TS=$(date +%s)
    EXIT_CODE=0
    if [ "$ITERATION" -eq 1 ]; then
        echo "=== Iteration $ITERATION: Using FULL prompt + triage ==="
        echo "$FULL_PROMPT"
        echo "$TRIAGE_INFO"
        run_senpai_claude $MAX_TURNS "${FULL_PROMPT}"$'\n\n'"${TRIAGE_INFO}" || EXIT_CODE=$?
    else
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
