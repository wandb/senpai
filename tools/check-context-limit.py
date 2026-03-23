#!/usr/bin/env python3
"""
Claude Code Stop hook: write /tmp/senpai_needs_fresh_start when the session
transcript exceeds SENPAI_TOKEN_LIMIT (default 150000 tokens).

Token count is approximated from content character lengths (÷4).
"""

import json
import os
import sys


def estimate_tokens(transcript_path: str) -> int:
    total_chars = 0
    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(block.get("text", ""))
    return total_chars // 4


def main() -> None:
    data = json.load(sys.stdin)
    transcript = data.get("transcript_path", "")
    if not transcript or not os.path.exists(transcript):
        return

    limit = int(os.environ.get("SENPAI_TOKEN_LIMIT", "150000"))
    tokens = estimate_tokens(transcript)

    if tokens >= limit:
        print(f"[check-context-limit] ~{tokens:,} tokens >= limit {limit:,} — flagging fresh restart", flush=True)
        open("/tmp/senpai_needs_fresh_start", "w").close()
    else:
        print(f"[check-context-limit] ~{tokens:,} tokens (limit {limit:,})", flush=True)


if __name__ == "__main__":
    main()
