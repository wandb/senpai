#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Run one Senpai Claude Code turn through the Python Agent SDK."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query

MODEL = "claude-opus-4-7"
EFFORT = "max"
SETTING_SOURCES = ["project"]
REQUIRED_PLUGIN = "senpai"
REQUIRED_SLASH_COMMANDS = {
    "alphaxiv-paper-lookup",
    "list-experiments",
    "senpai-status-check",
    "wandb-primary",
    "web-search-advanced-research-paper",
    "senpai:assign-experiment",
    "senpai:bootstrap-target",
    "senpai:check-human-issues",
    "senpai:merge-winner",
    "senpai:poll-for-work",
    "senpai:submit-experiment-results",
    "senpai:survey-prs",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-turns", required=True, type=int)
    parser.add_argument("--plugin-dir", required=True, type=Path)
    parser.add_argument("-c", "--continue", dest="continue_conversation", action="store_true")
    return parser.parse_args(argv)


def repo_root(plugin_dir: Path) -> Path:
    return plugin_dir.resolve().parents[1]


def mcp_config_path(plugin_dir: Path) -> Path:
    return repo_root(plugin_dir) / ".mcp.json"


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), default=str), flush=True)


def message_payload(message: Any) -> dict[str, Any]:
    name = type(message).__name__
    data = jsonable(message)

    if name == "SystemMessage":
        return data["data"]
    if name == "AssistantMessage":
        return {"type": "assistant", **data}
    if name == "UserMessage":
        return {"type": "user", **data}
    if name == "ResultMessage":
        data["modelUsage"] = data.pop("model_usage")
        return {"type": "result", **data}
    if name == "RateLimitEvent":
        return {"type": "system", "subtype": "rate_limit", **data}
    return {"type": name, **data}


def validate_init_event(payload: dict[str, Any]) -> None:
    plugins = {item.get("name") for item in payload.get("plugins", [])}
    if REQUIRED_PLUGIN not in plugins:
        raise RuntimeError(f"Claude SDK did not load required plugin: {REQUIRED_PLUGIN}")

    commands = set(payload.get("slash_commands", []))
    missing = sorted(REQUIRED_SLASH_COMMANDS - commands)
    if missing:
        raise RuntimeError(f"Claude SDK missing required skills: {', '.join(missing)}")


async def user_prompt_hook(
    input_data: dict[str, Any],
    _tool_use_id: str | None,
    _context: Any,
) -> dict[str, Any]:
    emit({
        "type": "senpai_hook",
        "hook_event_name": input_data["hook_event_name"],
        "prompt_chars": len(input_data.get("prompt", "")),
    })
    return {}


async def tool_hook(input_data: dict[str, Any], _tool_use_id: str | None, _context: Any) -> dict[str, Any]:
    payload = {
        "type": "senpai_hook",
        "hook_event_name": input_data["hook_event_name"],
        "tool_name": input_data.get("tool_name"),
    }
    command = input_data.get("tool_input", {}).get("command")
    if command:
        payload["command"] = command
    emit(payload)
    return {}


def build_options(args: argparse.Namespace) -> ClaudeAgentOptions:
    plugin_dir = args.plugin_dir.resolve()
    root = repo_root(plugin_dir)
    mcp_config = mcp_config_path(plugin_dir)

    env = dict(os.environ)
    env["CLAUDE_PLUGIN_ROOT"] = str(plugin_dir)
    env["CLAUDE_CODE_ALLOW_ROOT"] = "1"

    return ClaudeAgentOptions(
        cli_path=Path.home() / ".local/bin/claude",
        cwd=Path.cwd(),
        continue_conversation=args.continue_conversation,
        max_turns=args.max_turns,
        model=MODEL,
        effort=EFFORT,
        permission_mode="bypassPermissions",
        system_prompt={"type": "preset", "preset": "claude_code"},
        tools={"type": "preset", "preset": "claude_code"},
        add_dirs=[root],
        plugins=[{"type": "local", "path": str(plugin_dir)}],
        mcp_servers=mcp_config,
        strict_mcp_config=True,
        setting_sources=SETTING_SOURCES,
        skills="all",
        include_hook_events=True,
        env=env,
        hooks={
            "UserPromptSubmit": [HookMatcher(hooks=[user_prompt_hook])],
            "PreToolUse": [HookMatcher(hooks=[tool_hook])],
            "PostToolUse": [HookMatcher(hooks=[tool_hook])],
            "PostToolUseFailure": [HookMatcher(hooks=[tool_hook])],
        },
    )


async def run(prompt: str, args: argparse.Namespace) -> int:
    saw_init = False
    async for message in query(prompt=prompt, options=build_options(args)):
        payload = message_payload(message)
        if payload.get("type") == "system" and payload.get("subtype") == "init":
            validate_init_event(payload)
            saw_init = True
        emit(payload)

    if not saw_init:
        raise RuntimeError("Claude SDK session ended before init")
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    prompt = sys.stdin.read()
    return asyncio.run(run(prompt, args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
