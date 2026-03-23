#!/usr/bin/env python3
"""
Claude Code Stop hook: write /tmp/senpai_needs_fresh_start when the session
transcript exceeds SENPAI_TOKEN_LIMIT (default 150000 tokens).

Uses the Anthropic token-counting API for an exact count.
Requires ANTHROPIC_API_KEY in the environment.

Configurable env vars:
  SENPAI_TOKEN_LIMIT  — token threshold (default: 150000)
  SENPAI_MODEL        — model name for counting (default: claude-sonnet-4-6)
"""

import json
import os
import sys

import anthropic


def load_messages(transcript_path: str) -> list:
    messages = []
    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content")
            if not content:
                continue
            messages.append({"role": role, "content": content})
    return messages


def main() -> None:
    data = json.load(sys.stdin)
    transcript = data.get("transcript_path", "")
    if not transcript or not os.path.exists(transcript):
        return

    messages = load_messages(transcript)
    if not messages:
        return

    limit = int(os.environ.get("SENPAI_TOKEN_LIMIT", "150000"))
    model = os.environ.get("SENPAI_MODEL", "claude-sonnet-4-6")

    client = anthropic.Anthropic()
    response = client.messages.count_tokens(model=model, messages=messages)
    tokens = response.input_tokens

    if tokens >= limit:
        print(f"[check-context-limit] {tokens:,} tokens >= limit {limit:,} — flagging fresh restart", flush=True)
        open("/tmp/senpai_needs_fresh_start", "w").close()
    else:
        print(f"[check-context-limit] {tokens:,} tokens (limit {limit:,})", flush=True)


if __name__ == "__main__":
    main()
