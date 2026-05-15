#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

wait_for_senpai_start_gate() {
    local gate_path="${SENPAI_START_GATE_PATH:-}"
    local poll_seconds="${SENPAI_START_GATE_POLL_SECONDS:-30}"

    [ -n "$gate_path" ] || return 0

    mkdir -p "$(dirname "$gate_path")"
    echo "=== Waiting for Senpai start gate: ${gate_path} ==="
    until [ -f "$gate_path" ]; do
        sleep "$poll_seconds"
    done
    echo "=== Senpai start gate opened at $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
    cat "$gate_path"
}
