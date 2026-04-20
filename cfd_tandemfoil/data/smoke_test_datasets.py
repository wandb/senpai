# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Smoke-test repo-local dataset loaders against the mounted PVC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent

if PARENT_DIR.as_posix() not in sys.path:
    sys.path.append(PARENT_DIR.as_posix())

from data.prepare_airfrans import DEFAULT_MANIFEST as AIRFRANS_MANIFEST
from data.prepare_airfrans import load_data as load_airfrans_data
from data.prepare_drivaerml import DEFAULT_MANIFEST as DRIVAERML_MANIFEST
from data.prepare_drivaerml import DrivAerMLCaseStore, load_surface_data
from data.prepare_multi import load_data as load_tandem_data


def _shape_triplet(sample) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    x, y, surf = sample
    return tuple(x.shape), tuple(y.shape), tuple(surf.shape)


def smoke_tandem() -> dict:
    train_ds, val_splits, stats, _ = load_tandem_data(
        manifest_path=THIS_DIR / "split_manifest_tandemfoilset_v2.json",
        stats_file=THIS_DIR / "split_stats.json",
        debug=True,
    )
    sample = train_ds[0]
    return {
        "train_len": len(train_ds),
        "val_splits": {k: len(v) for k, v in val_splits.items()},
        "sample_shapes": _shape_triplet(sample),
        "x_stats_dim": int(stats["x_mean"].shape[0]),
        "y_stats_dim": int(stats["y_mean"].shape[0]),
    }


def smoke_airfrans() -> dict:
    train_ds, val_splits, stats, _ = load_airfrans_data(
        manifest_path=AIRFRANS_MANIFEST,
        debug=True,
    )
    sample = train_ds[0]
    return {
        "manifest": str(AIRFRANS_MANIFEST),
        "train_len": len(train_ds),
        "val_splits": {k: len(v) for k, v in val_splits.items()},
        "sample_shapes": _shape_triplet(sample),
        "x_stats_dim": int(stats["x_mean"].shape[0]),
        "y_stats_dim": int(stats["y_mean"].shape[0]),
    }


def smoke_drivaerml() -> dict:
    train_ds, val_splits, stats, _ = load_surface_data(
        manifest_path=DRIVAERML_MANIFEST,
        debug=True,
    )
    sample = train_ds[0]
    store = DrivAerMLCaseStore(DRIVAERML_MANIFEST)
    volume_cases = store.case_ids("train", domain="volume")
    volume_case = store.load_case(volume_cases[0]) if volume_cases else None
    return {
        "manifest": str(DRIVAERML_MANIFEST),
        "train_len": len(train_ds),
        "val_splits": {k: len(v) for k, v in val_splits.items()},
        "sample_shapes": _shape_triplet(sample),
        "x_stats_dim": int(stats["x_mean"].shape[0]),
        "y_stats_dim": int(stats["y_mean"].shape[0]),
        "volume_case": None if volume_case is None else {
            "case_id": volume_case.case_id,
            "volume_x_shape": None if volume_case.volume_x is None else tuple(volume_case.volume_x.shape),
            "volume_y_shape": None if volume_case.volume_y is None else tuple(volume_case.volume_y.shape),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["tandem", "airfrans", "drivaerml"],
        help="Subset of dataset smoke tests to run. Defaults to all.",
    )
    args = parser.parse_args()

    requested = args.dataset or ["tandem", "airfrans", "drivaerml"]
    results = {}
    if "tandem" in requested:
        results["tandem"] = smoke_tandem()
    if "airfrans" in requested:
        results["airfrans"] = smoke_airfrans()
    if "drivaerml" in requested:
        results["drivaerml"] = smoke_drivaerml()

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
