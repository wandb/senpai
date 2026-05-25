#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Watchdog for advisor Claude Code invocations.
#
# The advisor entrypoint owns PR/issue/student polling. If Claude blocks
# forever inside a child shell, the outer loop never polls again. This wrapper
# keeps the polling loop in charge and handles known self-referential wait
# failures, such as `pgrep -f wandb_sparse_val.py` matching the waiter itself.

ADVISOR_CLAUDE_WATCHDOG_INTERVAL_S="${ADVISOR_CLAUDE_WATCHDOG_INTERVAL_S:-60}"
ADVISOR_CLAUDE_MIN_RUNTIME_S="${ADVISOR_CLAUDE_MIN_RUNTIME_S:-600}"
ADVISOR_CLAUDE_STALE_LOG_S="${ADVISOR_CLAUDE_STALE_LOG_S:-1200}"
ADVISOR_CLAUDE_KILL_GRACE_S="${ADVISOR_CLAUDE_KILL_GRACE_S:-15}"
ADVISOR_CLAUDE_SELF_PGREP_STALE_S="${ADVISOR_CLAUDE_SELF_PGREP_STALE_S:-300}"
ADVISOR_CLAUDE_SELF_PGREP_PATTERNS="${ADVISOR_CLAUDE_SELF_PGREP_PATTERNS:-wandb_sparse_val.py}"
ADVISOR_IDLE_STUDENT_ACTION_S="${ADVISOR_IDLE_STUDENT_ACTION_S:-0}"

advisor_uint_or_zero() {
    case "${1:-}" in
        ''|*[!0-9]*) printf '0\n' ;;
        *) printf '%s\n' "$1" ;;
    esac
}

# Read log mtimes portably so stale-output checks work on Linux and macOS.
advisor_file_mtime_s() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"
}

# Emit intervention messages to both pod logs and the current Claude log.
advisor_log_watchdog_trigger() {
    local message="$1"
    echo "$message"
    [ -n "${LOGFILE:-}" ] && printf '%s\n' "$message" >> "$LOGFILE"
}

# Walk child processes so the watchdog can see shells below timeout wrappers.
advisor_descendant_pids() {
    local pid="$1" child

    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        printf '%s\n' "$child"
        advisor_descendant_pids "$child"
    done
}

# Include the root process because the stuck waiter can be the watched shell.
advisor_tree_pids() {
    local pid="$1"
    [ -n "$pid" ] || return 0

    printf '%s\n' "$pid"
    advisor_descendant_pids "$pid"
}

# Signal a whole process tree so child wait shells cannot survive their parent.
advisor_kill_process_tree() {
    local signal="$1" pid="$2" child
    [ -n "$pid" ] || return 0

    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        advisor_kill_process_tree "$signal" "$child"
    done

    kill "-$signal" "$pid" 2>/dev/null || true
}

# Stop selected process trees gently, then force-kill anything still alive.
advisor_stop_process_trees() {
    local pids="$1" pid
    [ -n "$pids" ] || return 0

    printf '%s\n' "$pids" | while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        advisor_kill_process_tree TERM "$pid"
    done

    sleep "$ADVISOR_CLAUDE_KILL_GRACE_S"

    printf '%s\n' "$pids" | while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        kill -0 "$pid" 2>/dev/null && advisor_kill_process_tree KILL "$pid"
    done
}

# Check whether a pattern still belongs to real work, not just waiter commands.
advisor_real_pattern_process_exists() {
    local pattern="$1"

    ps -eo pid=,comm=,args= |
        awk -v pattern="$pattern" '
            index($0, pattern) == 0 { next }
            $2 ~ /^(sh|bash|zsh|dash|awk|grep|pgrep|ps|sleep|sed|sort|head|tail|cut|timeout)$/ { next }
            $0 ~ /pgrep[[:space:]]+-f/ { next }
            { found = 1 }
            END { exit !found }
        '
}

# Find waiters whose broad pgrep pattern only matches the waiter itself.
advisor_self_pgrep_wait_pids() {
    local root_pid="$1" pattern pid line

    for pattern in $ADVISOR_CLAUDE_SELF_PGREP_PATTERNS; do
        [ -n "$pattern" ] || continue
        advisor_real_pattern_process_exists "$pattern" && continue

        for pid in $(advisor_tree_pids "$root_pid"); do
            line=$(ps -o comm= -o args= -p "$pid" 2>/dev/null || true)
            [ -n "$line" ] || continue
            case "$line" in
                *"pgrep -f $pattern"*|*"pgrep -f '$pattern'"*|*"pgrep -f \"$pattern\""*)
                    printf '%s\n' "$pid"
                    ;;
            esac
        done
    done | sort -nu
}

advisor_started_with_only_idle_students() {
    [ "$(advisor_uint_or_zero "$ADVISOR_IDLE_STUDENT_ACTION_S")" -gt 0 ] || return 1
    [ "$(advisor_uint_or_zero "${ADVISOR_INVOCATION_IDLE_COUNT:-0}")" -gt 0 ] || return 1
    [ "$(advisor_uint_or_zero "${ADVISOR_INVOCATION_REVIEW_COUNT:-0}")" -eq 0 ] || return 1
    [ "$(advisor_uint_or_zero "${ADVISOR_INVOCATION_ADVISOR_ACTION_COUNT:-0}")" -eq 0 ] || return 1
    [ "$(advisor_uint_or_zero "${ADVISOR_INVOCATION_ISSUE_COUNT:-0}")" -eq 0 ] || return 1
    [ "$(advisor_uint_or_zero "${ADVISOR_INVOCATION_POD_ANOMALY_COUNT:-0}")" -eq 0 ] || return 1
}

advisor_current_idle_students() {
    local idle_json

    idle_json=$(list_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH") || return 1
    printf '%s' "$idle_json" | json_join
}

# Run Claude under supervision and return 124 when the outer loop should re-poll.
run_advisor_claude_with_watchdog() {
    run_senpai_claude "$@" &
    local claude_pid=$!
    local start_ts now_ts runtime log_mtime log_age reason rc self_pgrep_wait_pids
    local idle_action_s current_idle_students
    local watchdog_fired=0
    start_ts=$(date +%s)

    while kill -0 "$claude_pid" 2>/dev/null; do
        sleep "$ADVISOR_CLAUDE_WATCHDOG_INTERVAL_S"
        kill -0 "$claude_pid" 2>/dev/null || break

        now_ts=$(date +%s)
        runtime=$((now_ts - start_ts))
        reason=""

        self_pgrep_wait_pids=$(advisor_self_pgrep_wait_pids "$claude_pid")
        if [ -n "$self_pgrep_wait_pids" ]; then
            if [ "$runtime" -ge "$ADVISOR_CLAUDE_SELF_PGREP_STALE_S" ]; then
                advisor_log_watchdog_trigger "=== Advisor Claude watchdog: stopping self-matching pgrep wait children after ${runtime}s: $(printf '%s' "$self_pgrep_wait_pids" | tr '\n' ' ')==="
                advisor_stop_process_trees "$self_pgrep_wait_pids"
                watchdog_fired=1
            else
                echo "=== Advisor Claude watchdog: saw self-matching pgrep wait; waiting ${runtime}/${ADVISOR_CLAUDE_SELF_PGREP_STALE_S}s before intervention ==="
            fi
        fi

        if advisor_started_with_only_idle_students; then
            idle_action_s=$(advisor_uint_or_zero "$ADVISOR_IDLE_STUDENT_ACTION_S")
            if [ "$runtime" -ge "$idle_action_s" ]; then
                current_idle_students=$(advisor_current_idle_students 2>/dev/null || true)
                if [ -n "$current_idle_students" ]; then
                    advisor_log_watchdog_trigger "=== Advisor Claude watchdog: stopping advisor invocation after ${runtime}s because students are still idle: ${current_idle_students} ==="
                    advisor_stop_process_trees "$claude_pid"
                    watchdog_fired=1
                    break
                fi
            fi
        fi

        [ "$runtime" -lt "$ADVISOR_CLAUDE_MIN_RUNTIME_S" ] && continue

        log_mtime=$(advisor_file_mtime_s "$LOGFILE" 2>/dev/null || printf '%s' "$now_ts")
        log_age=$((now_ts - log_mtime))
        if [ "$log_age" -ge "$ADVISOR_CLAUDE_STALE_LOG_S" ]; then
            reason="Claude log stale for ${log_age}s"
        fi

        if [ -n "$reason" ]; then
            advisor_log_watchdog_trigger "=== Advisor Claude watchdog: stopping stale advisor invocation (${reason}) ==="
            advisor_stop_process_trees "$claude_pid"
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
