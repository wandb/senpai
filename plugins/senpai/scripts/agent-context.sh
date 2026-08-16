#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

install_senpai_agent_context() {
    local workdir="$1"
    local source_plugin="$2"
    local runtime_root="$3"
    local runtime_plugin="$runtime_root/plugin"

    mkdir -p "$HOME/.agents/agents"
    cp -a "$workdir/.agents/agents/." "$HOME/.agents/agents/"
    cp -a "$source_plugin" "$runtime_plugin"
    "$SENPAI_PYTHON" -m senpai_agent.agent_markdown \
        "$HOME/.agents/agents" "$runtime_plugin"
    printf '%s\n' "$runtime_plugin"
}
