#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/k8s/advisor-claude-watchdog.sh"

ADVISOR_CLAUDE_TASK_OUTPUT_WAIT_STALE_S=0

ROOT_PID=""
REAL_WORK_PID=""
TMPDIR=""

cleanup() {
    set +e
    [ -n "${ROOT_PID:-}" ] && kill -TERM "$ROOT_PID" 2>/dev/null
    [ -n "${REAL_WORK_PID:-}" ] && kill -TERM "$REAL_WORK_PID" 2>/dev/null
    sleep 0.2
    [ -n "${ROOT_PID:-}" ] && kill -KILL "$ROOT_PID" 2>/dev/null
    [ -n "${REAL_WORK_PID:-}" ] && kill -KILL "$REAL_WORK_PID" 2>/dev/null
    [ -n "${TMPDIR:-}" ] && rm -rf "$TMPDIR"
}
trap cleanup EXIT

make_task_output() {
    TMPDIR=$(mktemp -d /tmp/claude-watchdog-test.XXXXXX)
    mkdir -p "$TMPDIR/-workspace-target/session/tasks"
    TASK_OUTPUT="$TMPDIR/-workspace-target/session/tasks/bn7xfp7ad.output"
    printf 'wandb: [wandb.Api()] Loaded credentials\nTerminated\n' > "$TASK_OUTPUT"
}

start_waiter_without_real_worker() {
    make_task_output
    ROOT_PID=""
    bash -c '
        task_output="$1"
        bash -c '"'"'
            task_output="$1"
            until [ -s "$task_output" ] && grep -q "ERROR\|===" "$task_output" 2>/dev/null; do
                sleep 3
            done
            wait $(pgrep -P $$)
            cat "$task_output"
        '"'"' _ "$task_output" &
        wait
    ' _ "$TASK_OUTPUT" &
    ROOT_PID=$!
    sleep 0.5
}

start_waiter_with_real_worker() {
    make_task_output
    ROOT_PID=""
    bash -c '
        task_output="$1"
        bash -c '"'"'
            task_output="$1"
            python3 -c "import time; time.sleep(30)" &
            until [ -s "$task_output" ] && grep -q "ERROR\|===" "$task_output" 2>/dev/null; do
                sleep 3
            done
            wait $(pgrep -P $$)
            cat "$task_output"
        '"'"' _ "$task_output" &
        wait
    ' _ "$TASK_OUTPUT" &
    ROOT_PID=$!
    sleep 0.5
}

start_custom_grep_waiter_without_real_worker() {
    make_task_output
    ROOT_PID=""
    bash -c '
        task_output="$1"
        bash -c '"'"'
            task_output="$1"
            until grep -q "no val yet\|metric .*: state\|metric .*: error" "$task_output" 2>/dev/null; do
                sleep 3
            done
            tail -40 "$task_output"
        '"'"' _ "$task_output" &
        wait
    ' _ "$TASK_OUTPUT" &
    ROOT_PID=$!
    sleep 0.5
}

test_detects_stale_waiter_without_worker() {
    local pids
    start_waiter_without_real_worker
    pids=$(advisor_stale_task_output_wait_pids "$ROOT_PID" "$(date +%s)")
    if [ -z "$pids" ]; then
        echo "expected stale task-output waiter to be detected" >&2
        exit 1
    fi
    cleanup
}

test_detects_custom_grep_waiter_without_worker() {
    local pids
    start_custom_grep_waiter_without_real_worker
    pids=$(advisor_stale_task_output_wait_pids "$ROOT_PID" "$(date +%s)")
    if [ -z "$pids" ]; then
        echo "expected custom grep task-output waiter to be detected" >&2
        exit 1
    fi
    cleanup
}

test_ignores_waiter_with_real_worker() {
    local pids
    start_waiter_with_real_worker
    pids=$(advisor_stale_task_output_wait_pids "$ROOT_PID" "$(date +%s)")
    if [ -n "$pids" ]; then
        echo "expected waiter with real worker descendant to be ignored, got: $pids" >&2
        exit 1
    fi
    cleanup
}

test_detects_stale_waiter_without_worker
test_detects_custom_grep_waiter_without_worker
test_ignores_waiter_with_real_worker

echo "advisor Claude watchdog tests passed"
