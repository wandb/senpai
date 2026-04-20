# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Normalize the official TandemFoilSet competition split manifest.

Source of truth:
  - https://github.com/tcapelle/kagent/blob/main/cfd-competition/organizer/SPLITS.md
  - https://github.com/tcapelle/kagent/blob/main/cfd-competition/organizer/split_manifest.json

This script consumes the checked-in competition manifest and rewrites it into the
schema used by this repo, while preserving the official global-index assignments.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

try:
    from data.split_utils import ensure_disjoint, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data.split_utils import ensure_disjoint, write_json

DEFAULT_SOURCE_MANIFEST = (
    "https://raw.githubusercontent.com/tcapelle/kagent/main/"
    "cfd-competition/organizer/split_manifest.json"
)
DEFAULT_OUTPUT = Path(__file__).with_name("split_manifest_tandemfoilset_v2.json")
DEFAULT_ROOT_CANDIDATES = [
    "/mnt/pvc/datasets/tandemfoil",
    "/mnt/new-pvc/datasets/tandemfoil",
]


def _load_json(source: str) -> dict:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=30) as resp:
            return json.load(resp)
    with open(source) as f:
        return json.load(f)


def build_manifest(source_manifest: dict, root_candidates: list[str], source_path: str) -> dict:
    split_map = source_manifest["splits"]
    ensure_disjoint(split_map)

    total = sum(source_manifest["file_sizes"])
    assigned = sum(len(v) for v in split_map.values())
    if assigned != total:
        raise ValueError(f"Expected {total} assigned indices, got {assigned}")

    return {
        "dataset": "TandemFoilSet",
        "manifest_version": 2,
        "source_manifest_path": source_path,
        "split_design_url": (
            "https://github.com/tcapelle/kagent/blob/main/cfd-competition/organizer/SPLITS.md"
        ),
        "seed": source_manifest["seed"],
        "n_per_val": source_manifest["n_per_val"],
        "n_per_test": source_manifest["n_per_test"],
        "data_root_candidates": root_candidates,
        "pickle_files": source_manifest["pickle_files"],
        "file_sizes": source_manifest["file_sizes"],
        "val_splits": source_manifest["val_splits"],
        "test_splits": source_manifest["test_splits"],
        "split_counts": source_manifest["split_counts"],
        "splits": {k: sorted(v) for k, v in split_map.items()},
        "domain_groups": source_manifest["domain_groups"],
        "notes": [
            "Exact competition split imported from kagent split_manifest.json.",
            "Train/val/test semantics follow TandemFoilSet Split Design (v2).",
            "Domain groups are train-local indices for balanced sampling.",
        ],
    }


def verify_manifest(manifest: dict) -> None:
    required = {"train", *manifest["val_splits"], *manifest["test_splits"]}
    missing = sorted(required - set(manifest["splits"]))
    if missing:
        raise ValueError(f"Missing required TandemFoilSet splits: {missing}")

    expected_counts = {
        "train": 1499,
        "val_single_in_dist": 100,
        "val_geom_camber_rc": 100,
        "val_geom_camber_cruise": 100,
        "val_re_rand": 100,
        "test_single_in_dist": 200,
        "test_geom_camber_rc": 200,
        "test_geom_camber_cruise": 200,
        "test_re_rand": 200,
    }
    actual = {k: len(v) for k, v in manifest["splits"].items()}
    if actual != expected_counts:
        raise ValueError(f"Unexpected TandemFoilSet split counts: {actual}")

    ensure_disjoint(manifest["splits"])

    total = sum(manifest["file_sizes"])
    assigned = sum(len(v) for v in manifest["splits"].values())
    if assigned != total:
        raise ValueError(f"TandemFoilSet split total mismatch: {assigned} vs {total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--data-root-candidate",
        action="append",
        default=[],
        help="Override/add TandemFoilSet root candidates.",
    )
    args = parser.parse_args()

    source_manifest = _load_json(args.source_manifest)
    root_candidates = args.data_root_candidate or DEFAULT_ROOT_CANDIDATES
    manifest = build_manifest(source_manifest, root_candidates, args.source_manifest)
    verify_manifest(manifest)
    write_json(args.out, manifest)

    print(f"Wrote {args.out}")
    for name, count in manifest["split_counts"].items():
        print(f"  {name:24s} {count:4d}")


if __name__ == "__main__":
    main()
