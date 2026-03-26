#!/usr/bin/env python3
"""
Claude Code Stop hook: monitors session token usage from transcript JSONL.

Autocompact handles the primary compaction (via CLAUDE_AUTOCOMPACT_PCT_OVERRIDE).
This hook is a safety net — if the context is STILL above the hard limit after
autocompact, it flags a fresh restart for the Ralph loop.

No external dependencies — reads token data already present in the JSONL.

Configurable env vars:
  SENPAI_TOKEN_HARD_LIMIT  — tokens above which to flag fresh restart (default: 180000)
"""

import json
import os
import sys


def get_last_input_tokens(transcript_path: str) -> int | None:
    """Get total input tokens from the last completed API response in the JSONL.

    Each assistant entry with a non-null stop_reason contains usage data from
    the Anthropic API. The sum of input_tokens + cache tokens = full context size.
    """
    last_total = None
    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            message = entry.get("message", {})
            if not message.get("stop_reason"):
                continue
            usage = message.get("usage", {})
            total = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )
            if total > 0:
                last_total = total
    return last_total


def main() -> None:
    data = json.load(sys.stdin)
    transcript = data.get("transcript_path", "")
    if not transcript or not os.path.exists(transcript):
        return

    tokens = get_last_input_tokens(transcript)
    if tokens is None:
        return

    hard_limit = int(os.environ.get("SENPAI_TOKEN_HARD_LIMIT", "180000"))

    if tokens >= hard_limit:
        print(
            f"[context-monitor] {tokens:,} tokens >= hard limit {hard_limit:,}"
            " — flagging fresh restart",
            flush=True,
        )
        open("/tmp/senpai_needs_fresh_start", "w").close()
    else:
        print(
            f"[context-monitor] {tokens:,} tokens (hard limit {hard_limit:,})",
            flush=True,
        )


if __name__ == "__main__":
    main()
