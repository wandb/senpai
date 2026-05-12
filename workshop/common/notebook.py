"""Small display helpers for notebook-style Python files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def bootstrap() -> Path:
    """Ensure `workshop/` is importable when running a notebook file directly."""
    workshop_root = Path(__file__).resolve().parents[1]
    if str(workshop_root) not in sys.path:
        sys.path.insert(0, str(workshop_root))
    return workshop_root


def h1(title: str) -> None:
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")


def h2(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def show_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def checkpoint(message: str) -> None:
    print(f"\nCHECKPOINT: {message}")


def repo_note(*paths: str) -> None:
    print("\nSENPAI source anchors:")
    for path in paths:
        print(f"- {path}")


def write_markdown(path: Path, title: str, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n")
    return path
