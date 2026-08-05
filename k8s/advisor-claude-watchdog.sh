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
ADVISOR_CLAUDE_TASK_OUTPUT_WAIT_STALE_S="${ADVISOR_CLAUDE_TASK_OUTPUT_WAIT_STALE_S:-300}"

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

# Utility-only descendants, such as sleep/grep/cat, do not mean real work is
# still running below a Claude task-output waiter.
advisor_real_descendant_process_exists() {
    local root_pid="$1" pid comm

    for pid in $(advisor_descendant_pids "$root_pid"); do
        comm=$(ps -o comm= -p "$pid" 2>/dev/null | awk '{print $1}' || true)
        [ -n "$comm" ] || continue
        case "$comm" in
            awk|basename|cat|cut|dirname|grep|head|pgrep|ps|sed|sleep|sort|stat|tail|tee|test|tr|wc)
                ;;
            *)
                return 0
                ;;
        esac
    done

    return 1
}

advisor_is_task_output_wait_line() {
    local line="$1"

    case "$line" in
        *"/tmp/claude-"*"/tasks/"*".output"*) ;;
        *) return 1 ;;
    esac

    case "$line" in
        *"until "*"grep -q "*"sleep "*"done"*) return 0 ;;
        *) return 1 ;;
    esac
}

advisor_extract_task_output_path() {
    local line="$1"

    printf '%s\n' "$line" |
        awk '
            {
                for (i = 1; i <= NF; i++) {
                    token = $i
                    sub(/[;)]*$/, "", token)
                    if (token ~ /^\/tmp\/claude-[^[:space:]]*\/tasks\/[^[:space:]]*[.]output$/) {
                        print token
                        exit
                    }
                }
            }
        '
}

# Find Claude tool-output waiters whose output file stopped changing and whose
# real worker process is gone. This catches stale TaskOutput waits without
# binding the watchdog to one specific command or one specific error string.
advisor_stale_task_output_wait_pids() {
    local root_pid="$1" now_ts="$2"
    local pid line output_path output_mtime output_age

    for pid in $(advisor_tree_pids "$root_pid"); do
        line=$(ps -o comm= -o args= -p "$pid" 2>/dev/null || true)
        [ -n "$line" ] || continue
        advisor_is_task_output_wait_line "$line" || continue

        output_path=$(advisor_extract_task_output_path "$line")
        [ -n "$output_path" ] && [ -f "$output_path" ] || continue

        output_mtime=$(advisor_file_mtime_s "$output_path" 2>/dev/null || printf '%s' "$now_ts")
        output_age=$((now_ts - output_mtime))
        [ "$output_age" -ge "$ADVISOR_CLAUDE_TASK_OUTPUT_WAIT_STALE_S" ] || continue

        advisor_real_descendant_process_exists "$pid" && continue
        printf '%s\n' "$pid"
    done | sort -nu
}

# Run Claude under supervision and return 124 when the outer loop should re-poll.
run_advisor_claude_with_watchdog() {
    run_senpai_claude "$@" &
    local claude_pid=$!
    local start_ts now_ts runtime log_mtime log_age reason rc self_pgrep_wait_pids task_output_wait_pids
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

        task_output_wait_pids=$(advisor_stale_task_output_wait_pids "$claude_pid" "$now_ts")
        if [ -n "$task_output_wait_pids" ]; then
            advisor_log_watchdog_trigger "=== Advisor Claude watchdog: stopping stale task-output wait children after ${runtime}s: $(printf '%s' "$task_output_wait_pids" | tr '\n' ' ')==="
            advisor_stop_process_trees "$task_output_wait_pids"
            watchdog_fired=1
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
