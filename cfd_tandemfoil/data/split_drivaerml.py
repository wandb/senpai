# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Build a DrivAerML split manifest from the preprocessed PVC manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from data.split_utils import ensure_disjoint, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.split_utils import ensure_disjoint, write_json

DEFAULT_SURFACE_MANIFEST = "/mnt/pvc/Processed/drivaerml_processed/manifest.csv"
DEFAULT_SURFACE_MANIFEST_FULL = "/mnt/pvc/Processed/drivaerml_processed/manifest_full_failed10_included.csv"
DEFAULT_VOLUME_MANIFEST = "/mnt/pvc/Processed/drivaerml_processed/volume_manifest.csv"
DEFAULT_CASE_ROOT = "/mnt/pvc/Processed/drivaerml_processed"
DEFAULT_CASE_ROOT_CANDIDATES = [
    "/mnt/pvc/Processed/drivaerml_processed",
    "/mnt/new-pvc/Processed/drivaerml_processed",
]
DEFAULT_OUTPUT = Path(__file__).with_name("split_manifest_drivaerml.json")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_manifest(
    surface_manifest_path: str,
    surface_full_manifest_path: str,
    volume_manifest_path: str,
    case_root: str,
    case_root_candidates: list[str],
) -> dict:
    surface_rows = _read_csv_rows(surface_manifest_path)
    surface_full_rows = _read_csv_rows(surface_full_manifest_path)
    volume_rows = _read_csv_rows(volume_manifest_path)

    surface_splits = {
        split: [row["case_id"] for row in surface_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    volume_splits = {
        split: [row["case_id"] for row in volume_rows if row["split"] == split]
        for split in sorted({row["split"] for row in volume_rows})
    }

    current_ids = {row["case_id"] for row in surface_rows}
    full_ids = {row["case_id"] for row in surface_full_rows}
    excluded_case_ids = sorted(full_ids - current_ids)

    return {
        "dataset": "DrivAerML",
        "manifest_version": 1,
        "source_surface_manifest": surface_manifest_path,
        "source_surface_manifest_full": surface_full_manifest_path,
        "source_volume_manifest": volume_manifest_path,
        "case_root": case_root,
        "case_root_candidates": case_root_candidates,
        "surface_splits": surface_splits,
        "surface_split_counts": {k: len(v) for k, v in surface_splits.items()},
        "volume_splits": volume_splits,
        "volume_split_counts": {k: len(v) for k, v in volume_splits.items()},
        "excluded_case_ids": excluded_case_ids,
        "excluded_case_count": len(excluded_case_ids),
        "notes": [
            "Surface splits come directly from the current preprocessed manifest.csv on the PVC.",
            "Volume splits come from volume_manifest.csv and are a strict subset of the surface cases.",
            "manifest_full_failed10_included.csv is used to identify cases excluded from the current processed manifest.",
        ],
    }


def verify_manifest(manifest: dict) -> None:
    surface_counts = manifest["surface_split_counts"]
    if surface_counts != {"train": 394, "val": 34, "test": 46}:
        raise ValueError(f"Unexpected DrivAerML surface split counts: {surface_counts}")

    ensure_disjoint(manifest["surface_splits"])

    volume_counts = manifest["volume_split_counts"]
    if volume_counts != {"train": 15, "val": 1}:
        raise ValueError(f"Unexpected DrivAerML volume split counts: {volume_counts}")

    ensure_disjoint(manifest["volume_splits"])

    surface_by_case = {}
    for split, case_ids in manifest["surface_splits"].items():
        for case_id in case_ids:
            surface_by_case[case_id] = split

    for split, case_ids in manifest["volume_splits"].items():
        for case_id in case_ids:
            if surface_by_case.get(case_id) != split:
                raise ValueError(
                    f"Volume case {case_id} split {split} mismatches surface split {surface_by_case.get(case_id)}"
                )

    if manifest["excluded_case_count"] != 10:
        raise ValueError(f"Expected 10 excluded DrivAerML cases, got {manifest['excluded_case_count']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-manifest", default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--surface-manifest-full", default=DEFAULT_SURFACE_MANIFEST_FULL)
    parser.add_argument("--volume-manifest", default=DEFAULT_VOLUME_MANIFEST)
    parser.add_argument("--case-root", default=DEFAULT_CASE_ROOT)
    parser.add_argument(
        "--case-root-candidate",
        action="append",
        default=[],
        help="Override/add DrivAerML processed root candidates.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    case_root_candidates = args.case_root_candidate or DEFAULT_CASE_ROOT_CANDIDATES
    manifest = build_manifest(
        surface_manifest_path=args.surface_manifest,
        surface_full_manifest_path=args.surface_manifest_full,
        volume_manifest_path=args.volume_manifest,
        case_root=args.case_root,
        case_root_candidates=case_root_candidates,
    )
    verify_manifest(manifest)
    write_json(args.out, manifest)

    print(f"Wrote {args.out}")
    print("Surface splits:", manifest["surface_split_counts"])
    print("Volume splits :", manifest["volume_split_counts"])
    print("Excluded cases:", manifest["excluded_case_count"], manifest["excluded_case_ids"])


if __name__ == "__main__":
    main()
