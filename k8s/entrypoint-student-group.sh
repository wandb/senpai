#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

echo "=== Senpai Student Group: ${STUDENT_GROUP_NAME:-group} ==="
echo "Students: ${STUDENT_NAMES}"

repo_auth_url="$(printf '%s' "$REPO_URL" | sed "s#https://github.com/#https://${GITHUB_TOKEN}@github.com/#")"

student_extra_instructions() {
    local student="$1"
    [ -n "${STUDENT_EXTRA_INSTRUCTIONS_B64_JSON_B64:-}" ] || return 0
    printf '%s' "$STUDENT_EXTRA_INSTRUCTIONS_B64_JSON_B64" | base64 -d |
        STUDENT_FOR_EXTRA="$student" python3 -c '
import json
import os
import sys

mapping = json.load(sys.stdin)
print(mapping.get(os.environ["STUDENT_FOR_EXTRA"], ""))
'
}

terminate_group() {
    local pid
    echo "=== Terminating grouped student processes ==="
    for pid in "${student_pids[@]:-}"; do
        [ -n "$pid" ] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}

IFS=',' read -r -a raw_students <<< "$STUDENT_NAMES"
students=()
for raw_student in "${raw_students[@]}"; do
    student="$(printf '%s' "$raw_student" | xargs)"
    [ -n "$student" ] && students+=("$student")
done

if [ "${#students[@]}" -eq 0 ]; then
    echo "ERROR: STUDENT_NAMES is empty" >&2
    exit 2
fi

student_pids=()
trap terminate_group INT TERM

for student in "${students[@]}"; do
    (
        set -e
        export STUDENT_NAME="$student"
        export WORKDIR="/workspace/senpai-${student}"
        export HOME="/workspace/home-${student}"
        export SENPAI_GROUPED_STUDENT_POD=1
        export EXTRA_INSTRUCTIONS_B64

        EXTRA_INSTRUCTIONS_B64="$(student_extra_instructions "$student")"

        mkdir -p "$HOME"
        git clone --branch "$REPO_BRANCH" --single-branch --depth 1 --no-tags "$repo_auth_url" "$WORKDIR"
        git -C "$WORKDIR" remote set-url origin "$REPO_URL"
        cd "$WORKDIR"
        bash k8s/entrypoint-student.sh
    ) &
    pid="$!"
    student_pids+=("$pid")
    echo "Started student ${student} as pid ${pid}"
done

set +e
wait -n "${student_pids[@]}"
status=$?
set -e

echo "=== A grouped student process exited with status ${status}; restarting the group pod ==="
terminate_group
exit "$status"
