#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$WORKDIR/k8s/student-claude-watchdog.sh"

export STUDENT_NAME="tanjiro"
export STUDENT_CLAUDE_WATCHDOG_INTERVAL_S=1
export STUDENT_CLAUDE_MIN_RUNTIME_S=1
export STUDENT_CLAUDE_STALE_LOG_S=2
export STUDENT_CLAUDE_KILL_GRACE_S=1

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

ASSIGNED_36='[{"number":36,"headRefName":"tanjiro/old"}]'
ASSIGNED_39='[{"number":39,"headRefName":"tanjiro/new"}]'

assert_rc() {
    local expected="$1" actual="$2" name="$3"
    if [ "$actual" -ne "$expected" ]; then
        echo "FAIL: $name expected rc=$expected got rc=$actual" >&2
        exit 1
    fi
    echo "PASS: $name"
}

assert_log_contains() {
    local logfile="$1" pattern="$2" name="$3"
    if ! grep -q "$pattern" "$logfile"; then
        echo "FAIL: $name missing '$pattern' in $logfile" >&2
        exit 1
    fi
    echo "PASS: $name"
}

run_assignment_changed_case() {
    LOGFILE="$TMPDIR/assignment_changed.log"
    printf 'start\n' > "$LOGFILE"

    student_poll_for_work() {
        printf '%s\n' "$ASSIGNED_39"
    }
    student_has_active_training() {
        return 1
    }
    run_senpai_claude() {
        sleep 60
    }

    local rc=0
    run_student_claude_with_watchdog "$ASSIGNED_36" 5 "prompt" || rc=$?
    assert_rc 124 "$rc" "assignment change stops Claude"
    assert_log_contains "$LOGFILE" "Claude watchdog: stopping stale student invocation" "assignment trigger is written to logfile"
}

run_assignment_changed_with_training_case() {
    LOGFILE="$TMPDIR/assignment_changed_with_training.log"
    printf 'start\n' > "$LOGFILE"

    student_poll_for_work() {
        printf '%s\n' "$ASSIGNED_39"
    }
    student_has_active_training() {
        return 0
    }
    run_senpai_claude() {
        sleep 2
        printf 'done\n' >> "$LOGFILE"
    }

    local rc=0
    run_student_claude_with_watchdog "$ASSIGNED_36" 5 "prompt" || rc=$?
    assert_rc 0 "$rc" "assignment change waits while training is active"
}

run_stale_log_case() {
    LOGFILE="$TMPDIR/stale_log.log"
    printf 'start\n' > "$LOGFILE"

    student_poll_for_work() {
        printf '%s\n' "$ASSIGNED_36"
    }
    student_has_active_training() {
        return 1
    }
    run_senpai_claude() {
        sleep 60
    }

    local rc=0
    run_student_claude_with_watchdog "$ASSIGNED_36" 5 "prompt" || rc=$?
    assert_rc 124 "$rc" "stale log without training stops Claude"
    assert_log_contains "$LOGFILE" "Claude watchdog: stopping stale student invocation" "stale-log trigger is written to logfile"
}

run_clean_exit_case() {
    LOGFILE="$TMPDIR/clean_exit.log"
    printf 'start\n' > "$LOGFILE"

    student_poll_for_work() {
        printf '%s\n' "$ASSIGNED_36"
    }
    student_has_active_training() {
        return 1
    }
    run_senpai_claude() {
        sleep 1
        printf 'done\n' >> "$LOGFILE"
    }

    local rc=0
    run_student_claude_with_watchdog "$ASSIGNED_36" 5 "prompt" || rc=$?
    assert_rc 0 "$rc" "clean Claude exit passes through"
}

run_assignment_changed_case
run_assignment_changed_with_training_case
run_stale_log_case
run_clean_exit_case
