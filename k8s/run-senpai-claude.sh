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
    # Pipe prompt via stdin instead of -p to keep prompt text out of the process
    # cmdline. Agents use `pkill -f "train.py"` to kill training runs, and -p
    # embeds the prompt (which mentions train.py) in the cmdline, causing the
    # agent to accidentally kill its own Claude Code process.
    printf '%s' "$user_prompt" | claude "$@" -p - \
        --max-turns "$max_turns" \
        --model "claude-opus-4-6[1m]" \
        --output-format stream-json --verbose \
        --plugin-dir "$SENPAI_PLUGIN" \
        --dangerously-skip-permissions \
        >> "$LOGFILE" 2>&1
}
