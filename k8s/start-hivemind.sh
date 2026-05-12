#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

start_hivemind() {
    mkdir -p "$HOME/.claude/projects"
    export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"

    (
        while true; do
            if ! uvx --from wandb-hivemind hivemind whoami 2>/dev/null | grep -q "Status:      Valid"; then
                if [ -n "${GH_TOKEN:-}" ]; then
                    echo "=== Hivemind login refresh ($(date)) ==="
                    uvx --from wandb-hivemind hivemind login --method gh \
                        || echo "=== Hivemind login failed; run will retry auth ==="
                else
                    echo "=== Hivemind login skipped: GH_TOKEN/GITHUB_TOKEN not set ==="
                fi
            fi

            echo "=== Hivemind run starting ($(date)) ==="
            if uvx --from wandb-hivemind hivemind run; then
                code=0
            else
                code=$?
            fi
            echo "=== Hivemind exited code=$code at $(date), restarting in ${HIVEMIND_RESTART_SLEEP_S:-60}s ==="
            sleep "${HIVEMIND_RESTART_SLEEP_S:-60}"
        done
    ) &

    echo "=== Hivemind supervisor started (PID=$!) ==="
}
