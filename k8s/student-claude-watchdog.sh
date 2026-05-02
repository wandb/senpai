#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Watchdog for student Claude Code invocations.
#
# The student entrypoint owns GitHub assignment polling. If Claude blocks
# forever inside a child shell, the outer loop never polls again and can miss a
# closed/reassigned PR. This wrapper keeps the polling loop in charge.

STUDENT_CLAUDE_WATCHDOG_INTERVAL_S="${STUDENT_CLAUDE_WATCHDOG_INTERVAL_S:-60}"
STUDENT_CLAUDE_MIN_RUNTIME_S="${STUDENT_CLAUDE_MIN_RUNTIME_S:-600}"
STUDENT_CLAUDE_STALE_LOG_S="${STUDENT_CLAUDE_STALE_LOG_S:-1200}"
STUDENT_CLAUDE_KILL_GRACE_S="${STUDENT_CLAUDE_KILL_GRACE_S:-15}"
STUDENT_ASSIGNMENT_DRIFT_GRACE_S="${STUDENT_ASSIGNMENT_DRIFT_GRACE_S:-1800}"

student_file_mtime_s() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"
}

student_assignment_numbers() {
    python3 -c '
import json
import sys

items = json.loads(sys.stdin.read() or "[]")
numbers = sorted(int(item["number"]) for item in items)
print(",".join(f"#{number}" for number in numbers))
'
}

student_has_active_training() {
    ps -eo pid,ppid,comm,args |
        awk '$3 ~ /^(python[0-9.]*|torchrun)$/ && $0 ~ /train[.]py/ { found = 1 } END { exit !found }'
}

student_training_pids() {
    ps -eo pid,ppid,comm,args |
        awk '$3 ~ /^(python[0-9.]*|torchrun|pt_elastic)$/ && $0 ~ /train[.]py/ { print $1 }'
}

student_training_snapshot() {
    ps -eo pid,ppid,etimes,pcpu,pmem,comm,args |
        awk '$6 ~ /^(python[0-9.]*|torchrun|pt_elastic)$/ && $0 ~ /train[.]py/ { print }' |
        head -20
}

student_log_watchdog_trigger() {
    local message="$1"
    echo "$message"
    printf '%s\n' "$message" >> "$LOGFILE"
}

student_kill_process_tree() {
    local signal="$1" pid="$2" child
    [ -n "$pid" ] || return 0

    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        student_kill_process_tree "$signal" "$child"
    done

    kill "-$signal" "$pid" 2>/dev/null || true
}

student_stop_claude_tree() {
    local pid="$1"
    student_kill_process_tree TERM "$pid"
    sleep "$STUDENT_CLAUDE_KILL_GRACE_S"
    kill -0 "$pid" 2>/dev/null && student_kill_process_tree KILL "$pid"
}

student_stop_training_trees() {
    local pids pid
    pids=$(student_training_pids | sort -nr)
    [ -n "$pids" ] || return 0

    echo "=== Claude watchdog: stopping active train.py processes ==="
    printf '%s\n' "$pids" | while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        student_kill_process_tree TERM "$pid"
    done

    sleep "$STUDENT_CLAUDE_KILL_GRACE_S"

    printf '%s\n' "$pids" | while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        kill -0 "$pid" 2>/dev/null && student_kill_process_tree KILL "$pid"
    done
}

student_comment_watchdog_stop() {
    local reason="$1" start_numbers="$2" current_numbers="$3" snapshot body num
    [ -n "$start_numbers" ] || return 0

    snapshot=$(student_training_snapshot || true)
    [ -n "$snapshot" ] || snapshot="(no active train.py process found at comment time)"
    body=$(cat <<EOF
STUDENT-WATCHDOG: stopping this student invocation on $(hostname).

Reason: ${reason}
Original assignment: ${start_numbers:-none}
Current assignment: ${current_numbers:-none}

Active training processes before termination:

\`\`\`
${snapshot}
\`\`\`

The watchdog only takes this action after the assignment drift remains stable for ${STUDENT_ASSIGNMENT_DRIFT_GRACE_S}s, so transient GitHub/API wobbles should not trigger it.
EOF
)

    printf '%s\n' "$start_numbers" | tr ',' '\n' | sed 's/^#//' |
    while IFS= read -r num; do
        [ -n "$num" ] || continue
        gh_retry gh pr comment "$num" --repo "$GH_REPO" --body "$body" || true
    done
}

run_student_claude_with_watchdog() {
    local assigned_json="$1"
    shift

    local start_numbers
    start_numbers=$(printf '%s' "$assigned_json" | student_assignment_numbers)

    run_senpai_claude "$@" &
    local claude_pid=$!
    local start_ts now_ts runtime log_mtime log_age
    local current_json current_numbers poll_status reason rc active_training
    local assignment_drift_numbers="" assignment_drift_first_ts=0 assignment_drift_age=0
    local watchdog_fired=0
    start_ts=$(date +%s)

    while kill -0 "$claude_pid" 2>/dev/null; do
        sleep "$STUDENT_CLAUDE_WATCHDOG_INTERVAL_S"
        kill -0 "$claude_pid" 2>/dev/null || break

        now_ts=$(date +%s)
        runtime=$((now_ts - start_ts))

        reason=""
        active_training=0
        student_has_active_training && active_training=1

        if [ -n "$start_numbers" ]; then
            poll_status=0
            current_json=$(student_poll_for_work "$STUDENT_NAME" 2>/dev/null) || poll_status=$?
            if [ "$poll_status" -eq 0 ]; then
                current_numbers=$(printf '%s' "$current_json" | student_assignment_numbers)
                if [ "$current_numbers" != "$start_numbers" ]; then
                    if [ "$active_training" -eq 0 ]; then
                        if [ "$runtime" -ge "$STUDENT_CLAUDE_MIN_RUNTIME_S" ]; then
                            reason="assignment changed from ${start_numbers:-none} to ${current_numbers:-none}"
                        else
                            echo "=== Claude watchdog: assignment changed but no train.py is active; waiting until minimum runtime before stopping Claude ==="
                        fi
                    else
                        if [ "$assignment_drift_numbers" != "$current_numbers" ]; then
                            assignment_drift_numbers="$current_numbers"
                            assignment_drift_first_ts="$now_ts"
                        fi
                        assignment_drift_age=$((now_ts - assignment_drift_first_ts))
                        if [ "$assignment_drift_age" -ge "$STUDENT_ASSIGNMENT_DRIFT_GRACE_S" ]; then
                            reason="assignment changed from ${start_numbers:-none} to ${current_numbers:-none} for ${assignment_drift_age}s while train.py is active"
                        else
                            echo "=== Claude watchdog: assignment changed but train.py is active; waiting ${assignment_drift_age}/${STUDENT_ASSIGNMENT_DRIFT_GRACE_S}s before intervention ==="
                        fi
                    fi
                else
                    assignment_drift_numbers=""
                    assignment_drift_first_ts=0
                fi
            else
                echo "=== Claude watchdog: assignment poll failed; leaving Claude running ==="
                assignment_drift_numbers=""
                assignment_drift_first_ts=0
            fi
        fi

        [ "$runtime" -lt "$STUDENT_CLAUDE_MIN_RUNTIME_S" ] && [ -z "$reason" ] && continue

        if [ -z "$reason" ] && [ "$active_training" -eq 0 ]; then
            log_mtime=$(student_file_mtime_s "$LOGFILE" 2>/dev/null || printf '%s' "$now_ts")
            log_age=$((now_ts - log_mtime))
            if [ "$log_age" -ge "$STUDENT_CLAUDE_STALE_LOG_S" ]; then
                reason="no train.py process and Claude log stale for ${log_age}s"
            fi
        fi

        if [ -n "$reason" ]; then
            student_log_watchdog_trigger "=== Claude watchdog: stopping stale student invocation (${reason}) ==="
            if [ "$active_training" -eq 1 ]; then
                student_comment_watchdog_stop "$reason" "$start_numbers" "${current_numbers:-}" || true
            fi
            student_stop_training_trees
            student_stop_claude_tree "$claude_pid"
            watchdog_fired=1
            break
        fi
    done

    rc=0
    wait "$claude_pid" || rc=$?
    if [ "$watchdog_fired" -eq 1 ]; then
        return 124
    fi
    return "$rc"
}
