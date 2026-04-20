# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Helpers shared by dataset split manifest scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

PVC_MOUNT_ENV = "PVC_MOUNT_PATH"
KNOWN_PVC_PREFIXES = ("/mnt/new-pvc", "/mnt/pvc")


def pvc_mount_path() -> str | None:
    value = os.getenv(PVC_MOUNT_ENV, "").strip()
    return value or None


def rewrite_under_pvc_mount(path: str | Path) -> Path:
    text = str(path)
    mount = pvc_mount_path()
    if mount:
        for prefix in KNOWN_PVC_PREFIXES:
            if text == prefix or text.startswith(prefix + "/"):
                return Path(mount + text[len(prefix):])
    return Path(text)


def expand_pvc_candidates(candidates: Iterable[str | Path]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        options = [str(candidate), str(rewrite_under_pvc_mount(candidate))]
        for option in options:
            if option not in seen:
                ordered.append(option)
                seen.add(option)
    return ordered


def first_existing(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in expand_pvc_candidates(candidates):
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
