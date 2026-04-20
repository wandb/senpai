# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Build a repo-local AirfRANS split manifest from the official manifest.json.

The official AirfRANS benchmark defines four train/test tasks:
  - full
  - scarce
  - reynolds
  - aoa

To match the official training code and the Transolver/SpiderSolver comparison
pipeline, validation is carved deterministically from the tail 10% of each
official training list, while official test lists are left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from data.split_utils import ensure_disjoint, stable_tail_val_split, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.split_utils import ensure_disjoint, stable_tail_val_split, write_json

DEFAULT_MANIFEST_PATHS = [
    "/mnt/pvc/datasets/airfrans/Dataset/manifest.json",
    "/mnt/new-pvc/datasets/airfrans/Dataset/manifest.json",
]
DEFAULT_ROOT_CANDIDATES = [
    "/mnt/pvc/datasets/airfrans/Dataset",
    "/mnt/new-pvc/datasets/airfrans/Dataset",
]
DEFAULT_OUTPUT = Path(__file__).with_name("split_manifest_airfrans.json")
TASKS = ("full", "scarce", "reynolds", "aoa")


def _resolve_manifest_path(candidate: str | None) -> Path:
    if candidate:
        return Path(candidate)
    for path in DEFAULT_MANIFEST_PATHS:
        p = Path(path)
        if p.exists():
            return p
    raise FileNotFoundError("Could not find AirfRANS manifest.json in default locations")


def build_manifest(source_manifest: dict, root_candidates: list[str], source_path: str) -> dict:
    splits: dict[str, list[str]] = {}

    for task in TASKS:
        official_train = list(source_manifest[f"{task}_train"])
        train, val = stable_tail_val_split(official_train, val_fraction=0.10)
        test = list(source_manifest["full_test"] if task == "scarce" else source_manifest[f"{task}_test"])

        splits[f"{task}_train_official"] = official_train
        splits[f"{task}_train"] = train
        splits[f"{task}_val"] = val
        splits[f"{task}_test"] = test

    return {
        "dataset": "AirfRANS",
        "manifest_version": 1,
        "source_manifest_path": source_path,
        "source_manifest_format": "official_airfrans_manifest_json",
        "license": "ODbL-1.0",
        "data_root_candidates": root_candidates,
        "tasks": list(TASKS),
        "splits": splits,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "notes": [
            "Official AirfRANS train/test tasks are preserved exactly.",
            "Validation is the last 10% of each official training list, matching the official AirfRANS and Transolver loaders.",
            "scarce_test is identical to full_test by benchmark design.",
        ],
    }


def verify_manifest(manifest: dict) -> None:
    splits = manifest["splits"]
    counts = manifest["split_counts"]

    expected = {
        "full_train_official": 800,
        "full_train": 720,
        "full_val": 80,
        "full_test": 200,
        "scarce_train_official": 200,
        "scarce_train": 180,
        "scarce_val": 20,
        "scarce_test": 200,
        "reynolds_train_official": 504,
        "reynolds_train": 454,
        "reynolds_val": 50,
        "reynolds_test": 496,
        "aoa_train_official": 804,
        "aoa_train": 724,
        "aoa_val": 80,
        "aoa_test": 196,
    }
    if counts != expected:
        raise ValueError(f"Unexpected AirfRANS split counts: {counts}")

    for task in TASKS:
        official = splits[f"{task}_train_official"]
        train = splits[f"{task}_train"]
        val = splits[f"{task}_val"]
        if train + val != official:
            raise ValueError(f"{task}: train/val no longer reconstructs official training order")
        ensure_disjoint({f"{task}_train": train, f"{task}_val": val})

    if splits["scarce_train_official"] != splits["full_train_official"][: len(splits["scarce_train_official"])]:
        raise ValueError("scarce_train_official is not the official full_train prefix")
    if splits["scarce_test"] != splits["full_test"]:
        raise ValueError("scarce_test must equal full_test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--data-root-candidate",
        action="append",
        default=[],
        help="Override/add AirfRANS root candidates.",
    )
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.source_manifest)
    with manifest_path.open() as f:
        source_manifest = json.load(f)

    root_candidates = args.data_root_candidate or DEFAULT_ROOT_CANDIDATES
    manifest = build_manifest(source_manifest, root_candidates, str(manifest_path))
    verify_manifest(manifest)
    write_json(args.out, manifest)

    print(f"Wrote {args.out}")
    for name, count in manifest["split_counts"].items():
        print(f"  {name:22s} {count:4d}")


if __name__ == "__main__":
    main()
