# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Weave trace logger daemon — polls Claude Code session JSONL files and logs turns as Weave traces."""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import weave


POLL_INTERVAL = 60  # seconds
STATE_FILE = Path.home() / ".claude" / "weave_logger_state.json"


@dataclass
class _CallStub:
    """Minimal stand-in for a weave Call; create_call() only reads .id, .trace_id, .thread_id, ._children."""
    id: str
    trace_id: str
    thread_id: str | None = None
    _children: list = field(default_factory=list)


def extract_text(content) -> str:
    """Extract plain text from a message content field (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return ""


def find_human_text(start_uuid: str, all_messages: dict) -> str:
    """Walk parentUuid chain upward to find the nearest human-authored user message."""
    visited: set[str] = set()
    current = start_uuid
    while current and current not in visited:
        visited.add(current)
        msg = all_messages.get(current)
        if msg is None:
            break
        if msg.get("type") == "user":
            content = msg.get("message", {}).get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
                if joined := "\n".join(t for t in texts if t):
                    return joined
        current = msg.get("parentUuid")
    return ""


def process_session_file(session_file: Path, state: dict, agent_name: str, role: str, client) -> None:
    """Read new lines from a session JSONL, find completed turns, log to Weave."""
    session_id = session_file.stem
    file_state = state.setdefault(session_id, {"offset": 0, "logged": []})
    logged_set = set(file_state["logged"])

    file_size = session_file.stat().st_size
    if file_size <= file_state["offset"]:
        return

    # Read whole file to build parentUuid chain, then re-seek to only walk new lines
    all_messages: dict[str, dict] = {}
    with open(session_file, encoding="utf-8", errors="replace") as f:
        for raw in f:
            if raw := raw.strip():
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") in ("user", "assistant") and "uuid" in obj:
                    all_messages[obj["uuid"]] = obj
        f.seek(file_state["offset"])
        new_lines = f.readlines()

    for raw in new_lines:
        if not (raw := raw.strip()):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if obj.get("type") != "assistant":
            continue
        msg = obj["message"]
        if msg.get("stop_reason") != "end_turn":
            continue

        uuid = obj["uuid"]
        if uuid in logged_set:
            continue

        assistant_text = extract_text(msg.get("content", []))
        if not assistant_text:
            logged_set.add(uuid)
            continue

        user_text = find_human_text(obj.get("parentUuid"), all_messages)
        messages = [{"role": "user", "content": user_text}] if user_text else []
        messages.append({"role": "assistant", "content": assistant_text})

        usage_raw = msg.get("usage", {})
        usage = {
            "input_tokens": (usage_raw.get("input_tokens", 0)
                             + usage_raw.get("cache_read_input_tokens", 0)
                             + usage_raw.get("cache_creation_input_tokens", 0)),
            "output_tokens": usage_raw.get("output_tokens", 0),
            "cache_read_tokens": usage_raw.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": usage_raw.get("cache_creation_input_tokens", 0),
        }
        model = msg.get("model", "unknown")
        attrs = {"agent_name": agent_name, "role": role,
                 "git_branch": obj.get("gitBranch", "unknown"), "session_id": session_id}

        # Create or restore the session-level parent trace
        if not file_state.get("parent_call_id"):
            parent = client.create_call("claude_session", inputs={"session_id": session_id, **attrs}, use_stack=False)
            file_state["parent_call_id"] = parent.id
            file_state["trace_id"] = parent.trace_id
        else:
            parent = _CallStub(id=file_state["parent_call_id"], trace_id=file_state["trace_id"])

        turn_call = client.create_call(
            "agent_turn",
            inputs={"messages": messages, "model": model, "usage": usage, **attrs},
            parent=parent, use_stack=False,
        )
        turn_call.summary = {"usage": {model: {"input_tokens": usage["input_tokens"],
                                               "output_tokens": usage["output_tokens"], "requests": 1}}}
        client.finish_call(turn_call, output={"role": "assistant", "content": messages[-1]["content"]})

        logged_set.add(uuid)
        print(f"[weave_logger] turn logged  session={session_id[:8]}  uuid={uuid[:8]}"
              f"  tokens={usage['input_tokens']}+{usage['output_tokens']}", flush=True)

    file_state["offset"] = file_size
    file_state["logged"] = list(logged_set)


def main() -> None:
    p = argparse.ArgumentParser(description="Weave trace logger daemon for senpai Claude Code agents")
    p.add_argument("--role", required=True, choices=["advisor", "student"])
    p.add_argument("--agent-name", required=True)
    p.add_argument("--workdir", default="/workspace/senpai")
    p.add_argument("--project-dir", default=None, help="Override ~/.claude/projects/<hash>/ path")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-project", default="")
    args = p.parse_args()

    project_dir = Path(args.project_dir) if args.project_dir else (
        Path.home() / ".claude" / "projects" / args.workdir.replace("/", "-")
    )
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = args.wandb_project or os.environ.get("WANDB_PROJECT", "senpai-v1")

    client = weave.init(f"{entity}/{project}")
    print(f"[weave_logger] started  watching={project_dir}  role={args.role}  agent={args.agent_name}", flush=True)

    while True:
        state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        for session_file in sorted(project_dir.glob("*.jsonl")):
            process_session_file(session_file, state, args.agent_name, args.role, client)
        STATE_FILE.write_text(json.dumps(state, indent=2))
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
