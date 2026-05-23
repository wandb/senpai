# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
#
# Shared Claude Agent SDK invocation for Senpai advisor/student entrypoints.
# Each loop iteration must set LOGFILE before calling.
#
# Usage: run_senpai_claude <max_turns> <user_prompt> [extra argv, currently -c]

run_senpai_claude() {
    local max_turns=$1 user_prompt=$2
    shift 2
    export CLAUDE_PLUGIN_ROOT="$SENPAI_PLUGIN"
    local claude_cmd=(python3 "$WORKDIR/k8s/run_senpai_claude_sdk.py"
        --max-turns "$max_turns"
        --plugin-dir "$SENPAI_PLUGIN"
        "$@")
    if [ -n "${SENPAI_CLAUDE_TIMEOUT_SECONDS:-}" ] && command -v timeout >/dev/null 2>&1; then
        local kill_after="${SENPAI_CLAUDE_TIMEOUT_KILL_AFTER_SECONDS:-30}"
        claude_cmd=(timeout -k "$kill_after" "$SENPAI_CLAUDE_TIMEOUT_SECONDS" "${claude_cmd[@]}")
    fi

    # Pipe prompt via stdin instead of -p to keep prompt text out of the process
    # cmdline. Agents use `pkill -f "train.py"` to kill training runs, and -p
    # embeds the prompt (which mentions train.py) in the cmdline, causing the
    # agent to accidentally kill its own Claude Code process.
    printf '%s' "$user_prompt" | "${claude_cmd[@]}" >> "$LOGFILE" 2>&1
}
