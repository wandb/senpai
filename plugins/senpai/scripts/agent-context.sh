#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

install_senpai_agent_context() {
    local workdir="$1"
    local source_plugin="$2"
    local runtime_root="$3"
    local runtime_plugin="$runtime_root/plugin"
    local web_search_args=()

    mkdir -p "$HOME/.agents/agents"
    cp -a "$workdir/.agents/agents/." "$HOME/.agents/agents/"
    cp -a "$source_plugin" "$runtime_plugin"
    case "${SENPAI_WEB_SEARCH:-true}" in
        true) ;;
        false)
            rm -rf \
                "$runtime_plugin/skills/exa-search" \
                "$runtime_plugin/skills/alphaxiv-paper-lookup"
            rm -f "$HOME/.agents/agents/search.md"
            web_search_args=(--without-web-search "$runtime_plugin")
            ;;
        *)
            echo "ERROR: SENPAI_WEB_SEARCH must be true or false" >&2
            return 2
            ;;
    esac
    "$SENPAI_PYTHON" -m senpai_agent.agent_markdown \
        "${web_search_args[@]}" \
        "$HOME/.agents/agents" "$runtime_plugin"
    printf '%s\n' "$runtime_plugin"
}
