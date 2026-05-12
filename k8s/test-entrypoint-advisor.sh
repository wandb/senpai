#!/bin/bash
# Test harness for entrypoint-advisor.sh loop logic.
# Mocks CC and gh calls, runs 3 iterations with short timers, then exits.
#
# Usage:
#   bash k8s/test-entrypoint-advisor.sh              # default: mixed scenario
#   SCENARIO=idle bash k8s/test-entrypoint-advisor.sh # all students busy, nothing to do

set -e
set -o pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

# --- Source real code ---
source "$WORKDIR/k8s/run-senpai-claude.sh"
source "$WORKDIR/plugins/senpai/scripts/senpai-gh.sh"

# --- Env vars the entrypoint expects ---
export ADVISOR_BRANCH="test-advisor"
export STUDENT_NAMES="alice,bob,charlie"
export RESEARCH_TAG="test"

# --- Mock: CC just logs and returns ---
run_senpai_claude() {
    local max_turns=$1 user_prompt=$2
    shift 2
    echo "{\"type\":\"mock\",\"iteration\":$ITERATION,\"turns\":$max_turns,\"extra\":\"$*\"}" >> "$LOGFILE"
    echo "  (mock CC: turns=$max_turns, prompt=${#user_prompt} chars, extra_args=[$*])"
    sleep 0.5
}

# --- Mock: gh triage functions (accept optional since arg, ignored) ---
SCENARIO="${SCENARIO:-mixed}"

case "$SCENARIO" in
    mixed)
        # 1 PR to review, 0 issues, alice is idle
        list_ready_for_review_prs() { echo '[{"number":42,"title":"test PR","updatedAt":"2026-04-01T20:00:00Z"}]'; }
        list_prs_requiring_advisor_action() { echo '[]'; }
        check_gh_issues() { echo '[]'; }
        list_idle_students() { echo '["alice"]'; }
        ;;
    idle)
        # nothing actionable — should hit the skip branch every time
        list_ready_for_review_prs() { echo '[]'; }
        list_prs_requiring_advisor_action() { echo '[]'; }
        check_gh_issues() { echo '[]'; }
        list_idle_students() { echo '[]'; }
        ;;
    busy)
        # PRs to review, issues open, multiple idle students
        list_ready_for_review_prs() { echo '[{"number":1,"updatedAt":"2026-04-01T20:00:00Z"},{"number":2,"updatedAt":"2026-04-01T21:00:00Z"}]'; }
        list_prs_requiring_advisor_action() { echo '[{"number":3,"reasons":["stale_wip","duplicate_student_wip"],"updatedAt":"2026-04-01T18:00:00Z"}]'; }
        check_gh_issues() { echo '[{"number":10,"title":"help","updatedAt":"2026-04-01T19:00:00Z"}]'; }
        list_idle_students() { echo '["bob","charlie"]'; }
        ;;
    since)
        # Review PRs are level-triggered; issues still use the timestamp filter.
        CALL_COUNT=0
        list_ready_for_review_prs() {
            CALL_COUNT=$((CALL_COUNT + 1))
            echo '[{"number":42,"updatedAt":"2026-04-01T20:00:00Z"}]'
        }
        list_prs_requiring_advisor_action() { echo '[]'; }
        check_gh_issues() { echo '[]'; }
        list_idle_students() { echo '[]'; }
        ;;
    pollfail)
        # A partial poll with work should run CC but should not advance the watermark.
        list_ready_for_review_prs() { echo '[{"number":42,"updatedAt":"2026-04-01T20:00:00Z"}]'; }
        list_prs_requiring_advisor_action() { return 1; }
        check_gh_issues() { return 1; }
        list_idle_students() { echo '[]'; }
        ;;
esac

# --- JSON helpers (copied from entrypoint) ---
json_len() { python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))"; }
json_join() { python3 -c "import sys,json; print(','.join(json.loads(sys.stdin.read())))"; }
json_numbers() { python3 -c "import sys,json; print(','.join(f'#{i[\"number\"]}' for i in json.loads(sys.stdin.read())))"; }
json_advisor_action_summary() { python3 -c 'import sys,json; print(",".join("#{}[{}]".format(i["number"], ",".join(i.get("reasons", []))) for i in json.loads(sys.stdin.read())))'; }

# --- Test config ---
LOGDIR="$WORKDIR/advisor_logs"
mkdir -p "$LOGDIR"
SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
LAST_CHECK_FILE="$LOGDIR/.last_check_ts"
rm -f "$LAST_CHECK_FILE"

SLEEP_TIME_S=1
MAX_TURNS=5
MAX_ITERATIONS=3

FULL_PROMPT="Test full prompt (would be CLAUDE-ADVISOR.md + task instructions + key info)"
HEARTBEAT_PROMPT="Test heartbeat prompt"

echo "=== Test entrypoint-advisor (scenario=$SCENARIO, ${MAX_ITERATIONS} iterations) ==="
echo ""

# --- Loop (mirrors entrypoint-advisor.sh) ---
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Advisor Heartbeat iteration $ITERATION ($(date)) ==="

    # --- Read last-check timestamp ---
    SINCE=""
    [ -f "$LAST_CHECK_FILE" ] && SINCE=$(cat "$LAST_CHECK_FILE")

    # --- Check research state ---
    POLL_OK=1
    REVIEW_JSON=$(poll_or_empty "review-ready PR poll" list_ready_for_review_prs "$ADVISOR_BRANCH") || POLL_OK=0
    REVIEW_COUNT=$(printf '%s' "$REVIEW_JSON" | json_len)
    ADVISOR_ACTION_JSON=$(poll_or_empty "advisor-action PR poll" list_prs_requiring_advisor_action "$ADVISOR_BRANCH") || POLL_OK=0
    ADVISOR_ACTION_COUNT=$(printf '%s' "$ADVISOR_ACTION_JSON" | json_len)
    ISSUE_JSON=$(poll_or_empty "GitHub issue poll" check_gh_issues "$ADVISOR_BRANCH" "$SINCE") || POLL_OK=0
    ISSUE_COUNT=$(printf '%s' "$ISSUE_JSON" | json_len)
    IDLE_JSON=$(poll_or_empty "idle-student poll" list_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH") || POLL_OK=0
    IDLE_COUNT=$(printf '%s' "$IDLE_JSON" | json_len)
    WATERMARK=$(max_updated_at "$ISSUE_JSON")

    TRIAGE_INFO="=== Research state (since ${SINCE:-boot}): reviews=$REVIEW_COUNT | advisor_action=$ADVISOR_ACTION_COUNT | issues=$ISSUE_COUNT | idle=$IDLE_COUNT ==="
    echo "$TRIAGE_INFO"

    # --- Skip if nothing actionable ---
    if [ "$REVIEW_COUNT" -eq 0 ] && [ "$ADVISOR_ACTION_COUNT" -eq 0 ] && [ "$ISSUE_COUNT" -eq 0 ] && [ "$IDLE_COUNT" -eq 0 ]; then
        echo "=== Nothing actionable, sleeping $SLEEP_TIME_S seconds ==="
        sleep "$SLEEP_TIME_S"
        [ "$ITERATION" -ge "$MAX_ITERATIONS" ] && break
        continue
    fi

    # --- Accumulate triage details ---
    if [ "$REVIEW_COUNT" -gt 0 ]; then
        REVIEW_NUMS=$(printf '%s' "$REVIEW_JSON" | json_numbers)
        TRIAGE_INFO="${TRIAGE_INFO} | Review PRs: ${REVIEW_NUMS}"
    fi
    if [ "$ADVISOR_ACTION_COUNT" -gt 0 ]; then
        ADVISOR_ACTION_SUMMARY=$(printf '%s' "$ADVISOR_ACTION_JSON" | json_advisor_action_summary)
        TRIAGE_INFO="${TRIAGE_INFO} | Advisor-action PRs: ${ADVISOR_ACTION_SUMMARY}"
    fi
    if [ "$ISSUE_COUNT" -gt 0 ]; then
        ISSUE_NUMS=$(printf '%s' "$ISSUE_JSON" | json_numbers)
        TRIAGE_INFO="${TRIAGE_INFO} | Issues: ${ISSUE_NUMS}"
    fi
    if [ "$IDLE_COUNT" -gt 0 ]; then
        IDLE_NAMES=$(printf '%s' "$IDLE_JSON" | json_join)
        TRIAGE_INFO="${TRIAGE_INFO} | Idle students: ${IDLE_NAMES}"
    fi
    echo "$TRIAGE_INFO"

    # --- Log triage state and invoke CC ---
    echo "=== Log: $LOGFILE ==="
    echo "$TRIAGE_INFO" > "$LOGFILE"

    START_TS=$(date +%s)
    EXIT_CODE=0
    if [ "$ITERATION" -eq 1 ]; then
        echo "=== Using FULL prompt (${#FULL_PROMPT} chars) ==="
        run_senpai_claude $MAX_TURNS "$FULL_PROMPT" || EXIT_CODE=$?
    else
        echo "=== Using heartbeat prompt ==="
        CONTINUE_PROMPT="${HEARTBEAT_PROMPT}"$'\n\n'"${TRIAGE_INFO}"
        run_senpai_claude 1000 "$CONTINUE_PROMPT" -c || EXIT_CODE=$?
    fi
    DURATION=$(( $(date +%s) - START_TS ))

    # --- Update last-check timestamp (only after complete poll and success) ---
    if [ "$EXIT_CODE" -eq 0 ] && [ "$POLL_OK" -eq 1 ] && [ -n "$WATERMARK" ]; then
        echo "$WATERMARK" > "$LAST_CHECK_FILE"
    elif [ "$EXIT_CODE" -eq 0 ] && [ "$POLL_OK" -ne 1 ]; then
        echo "=== Poll incomplete; not advancing last-check timestamp ==="
    fi

    echo "=== Exited code=$EXIT_CODE after ${DURATION}s, next check in $SLEEP_TIME_S seconds ==="
    [ "$ITERATION" -ge "$MAX_ITERATIONS" ] && break
    sleep "$SLEEP_TIME_S"
done

echo ""
echo "=== Done ($ITERATION iterations). Logfiles: ==="
shopt -s nullglob
LOGFILES=("$LOGDIR"/iteration_*.jsonl)
if [ ${#LOGFILES[@]} -eq 0 ]; then
    echo "(no logfiles — skip branch fired every iteration)"
else
    for f in "${LOGFILES[@]}"; do
        echo "--- $f ---"
        cat "$f"
    done
    rm -f "${LOGFILES[@]}"
fi

# Show final timestamp state
if [ -f "$LAST_CHECK_FILE" ]; then
    echo ""
    echo "=== Last check timestamp: $(cat "$LAST_CHECK_FILE") ==="
    rm -f "$LAST_CHECK_FILE"
fi
