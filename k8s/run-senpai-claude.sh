# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
#
# Shared Claude Code invocation for Senpai advisor/student entrypoints.
# Each loop iteration must set LOGFILE before calling.
#
# Usage: run_senpai_claude <max_turns> <user_prompt> [extra claude argv before -p, e.g. -c]

run_senpai_claude() {
    local max_turns=$1 user_prompt=$2
    shift 2
    export CLAUDE_PLUGIN_ROOT="$SENPAI_PLUGIN"
    local claude_cmd=(claude "$@" -p -
        --model "claude-opus-4-8[1m]"
        --effort "max"
        --max-turns "$max_turns"
        --output-format stream-json --verbose
        --plugin-dir "$SENPAI_PLUGIN"
        --dangerously-skip-permissions)
    if [ -n "${SENPAI_CLAUDE_TIMEOUT_SECONDS:-}" ] && command -v timeout >/dev/null 2>&1; then
        local kill_after="${SENPAI_CLAUDE_TIMEOUT_KILL_AFTER_SECONDS:-30}"
        claude_cmd=(timeout -k "$kill_after" "$SENPAI_CLAUDE_TIMEOUT_SECONDS" "${claude_cmd[@]}")
    fi

    # Pipe prompt via stdin instead of -p to keep prompt text out of the process
    # cmdline. Agents use `pkill -f "train.py"` to kill training runs, and -p
    # embeds the prompt (which mentions train.py) in the cmdline, causing the
    # agent to accidentally kill its own Claude Code process.
    #
    # SENPAI Console (Phase 2, live event ingestion): when $SENPAI_EVENT_DIR is set,
    # also mirror the stream-json to a stable per-iteration event file the console
    # tails, instead of scraping the iteration logs. Inert when unset.
    if [ -n "${SENPAI_EVENT_DIR:-}" ]; then
        mkdir -p "$SENPAI_EVENT_DIR"
        local event_file="$SENPAI_EVENT_DIR/event-$(date +%Y%m%d_%H%M%S)_$$.jsonl"
        printf '%s' "$user_prompt" | "${claude_cmd[@]}" 2>&1 | tee -a "$event_file" >> "$LOGFILE"
    else
        printf '%s' "$user_prompt" | "${claude_cmd[@]}" >> "$LOGFILE" 2>&1
    fi
}
