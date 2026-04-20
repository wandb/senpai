# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Helpers shared by dataset split manifest scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def first_existing(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def stable_tail_val_split(items: list[str], val_fraction: float = 0.10) -> tuple[list[str], list[str]]:
    """Match the official AirfRANS / Transolver convention: last 10% becomes val."""
    n_val = max(1, int(len(items) * val_fraction))
    return items[:-n_val], items[-n_val:]


def ensure_disjoint(split_map: dict[str, list[str] | list[int]]) -> None:
    seen: set[str | int] = set()
    overlap: set[str | int] = set()
    for values in split_map.values():
        for item in values:
            if item in seen:
                overlap.add(item)
            seen.add(item)
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(f"Split overlap detected, sample={sample}")


def write_json(path: str | Path, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
