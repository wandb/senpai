#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

handoff_operational_supervisor_secrets() {
    local name value
    local secret_dir

    secret_dir="$(mktemp -d /tmp/senpai-supervisor-secrets.XXXXXX)"
    chmod 700 "$secret_dir"
    for name in GITHUB_TOKEN WANDB_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY; do
        value="${!name:-}"
        if [ -n "$value" ]; then
            printf '%s' "$value" > "$secret_dir/$name"
            chmod 600 "$secret_dir/$name"
        fi
    done
    export SENPAI_SUPERVISOR_SECRET_DIR="$secret_dir"
    unset GITHUB_TOKEN GH_TOKEN WANDB_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
}
