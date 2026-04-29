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

run_active_training_detector_case() {
    local py
    for py in python3.10 python3.11 python3.12 python3.13 python3.14; do
        ps() {
            printf '123 1 %s %s train.py --epochs 1\n' "$py" "$py"
        }
        student_has_active_training || {
            echo "FAIL: active training detector missed $py" >&2
            exit 1
        }
    done
    unset -f ps

    echo "PASS: active training detector covers python3.10 through python3.14"
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

run_active_training_detector_case
run_assignment_changed_case
run_assignment_changed_with_training_case
run_stale_log_case
run_clean_exit_case
