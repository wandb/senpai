"""Minimal Anthropic helpers for teaching LLM calls and tool loops."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import WorkshopConfig


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[[dict[str, Any]], Any]


def anthropic_message(
    config: WorkshopConfig,
    prompt: str,
    *,
    system: str = "You are a concise staff-level ML research engineering instructor.",
    max_tokens: int = 800,
) -> str:
    payload = {
        "model": config.anthropic_model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "x-api-key": config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read())
    chunks = result.get("content", [])
    return "\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text").strip()


def ask_for_json(config: WorkshopConfig, prompt: str, *, max_tokens: int = 1000) -> dict[str, Any]:
    wrapped = (
        f"{prompt}\n\n"
        "Return only valid JSON. Do not include markdown fences or commentary."
    )
    text = anthropic_message(config, wrapped, max_tokens=max_tokens)
    return json.loads(text)


def choose_tool(config: WorkshopConfig, task: str, tools: list[Tool]) -> dict[str, Any]:
    tool_descriptions = [
        {"name": tool.name, "description": tool.description}
        for tool in tools
    ]
    return ask_for_json(
        config,
        "Choose exactly one tool for this task and provide JSON arguments.\n"
        f"Task: {task}\n"
        f"Tools: {json.dumps(tool_descriptions, indent=2)}\n"
        'Schema: {"tool": "<tool name>", "arguments": {}}',
    )


def run_chosen_tool(choice: dict[str, Any], tools: list[Tool]) -> Any:
    by_name = {tool.name: tool for tool in tools}
    name = choice.get("tool")
    if name not in by_name:
        raise ValueError(f"Unknown tool chosen: {name!r}")
    arguments = choice.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return by_name[name].fn(arguments)


def summarize_with_llm(config: WorkshopConfig, title: str, payload: object) -> str:
    return anthropic_message(
        config,
        f"Summarize this {title} for a staff-level autoresearch workshop.\n\n"
        f"{json.dumps(payload, indent=2, default=str)[:12000]}",
        max_tokens=600,
    )
