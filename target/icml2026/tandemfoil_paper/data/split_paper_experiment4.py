# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Materialize paper-faithful TandemFoilSet Experiment 4 tasks.

This generator targets the unambiguous high-Re tasks from the paper:

- Cruise Random, uniform 8:1:1
- Cruise Random extrapolation on AOA / Re / Stagger / Gap
- Race Car, uniform 8:1:1

The paper does not publish an exact split RNG seed, so this repo pins seed 42
for the uniform train/val sampling and uses stable rank order for tail
selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from data.split_utils import first_existing, write_json
    from tandemfoil.data.prepare import load_pickle
    from tandemfoil.data.prepare_multi import MultiFieldDataset
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data.split_utils import first_existing, write_json
    from tandemfoil.data.prepare import load_pickle
    from tandemfoil.data.prepare_multi import MultiFieldDataset


SEED = 42
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TAIL_FRACTION = 0.05
DEFAULT_ROOT_CANDIDATES = [
    "/mnt/pvc/datasets/tandemfoil",
    "/mnt/new-pvc/datasets/tandemfoil",
]
DEFAULT_MANIFEST = Path(__file__).with_name("split_manifest_tandemfoil_paper_experiment4.json")
DEFAULT_STATS = Path(__file__).with_name("split_stats_tandemfoil_paper_experiment4.json")


@dataclass(frozen=True)
class TaskSpec:
    dataset_group: str
    split_style: str
    variable: str | None
    pickle_files: tuple[str, ...]
    paper_table6_baseline: float
    paper_table6_best: float
    paper_table6_best_std: float
    notes: tuple[str, ...]


TASK_SPECS: dict[str, TaskSpec] = {
    "cruise_random_uniform": TaskSpec(
        dataset_group="cruise_random",
        split_style="uniform",
        variable=None,
        pickle_files=(
            "cruise_randomFields_mgn_Part1.pickle",
            "cruise_randomFields_mgn_Part2.pickle",
            "cruise_randomFields_mgn_Part3.pickle",
        ),
        paper_table6_baseline=1.79,
        paper_table6_best=0.10,
        paper_table6_best_std=0.13,
        notes=(
            "Experiment 4 Cruise Random uniform split.",
            "Paper states 8:1:1 train/val/test unless otherwise stated.",
        ),
    ),
    "cruise_random_aoa_extrap": TaskSpec(
        dataset_group="cruise_random",
        split_style="tail_extrapolation",
        variable="aoa0",
        pickle_files=(
            "cruise_randomFields_mgn_Part1.pickle",
            "cruise_randomFields_mgn_Part2.pickle",
            "cruise_randomFields_mgn_Part3.pickle",
        ),
        paper_table6_baseline=2.03,
        paper_table6_best=0.18,
        paper_table6_best_std=0.24,
        notes=(
            "Experiment 4 Cruise Random AOA extrapolation split.",
            "Test set is the highest and lowest 5% of AOA.",
        ),
    ),
    "cruise_random_re_extrap": TaskSpec(
        dataset_group="cruise_random",
        split_style="tail_extrapolation",
        variable="re",
        pickle_files=(
            "cruise_randomFields_mgn_Part1.pickle",
            "cruise_randomFields_mgn_Part2.pickle",
            "cruise_randomFields_mgn_Part3.pickle",
        ),
        paper_table6_baseline=4.85,
        paper_table6_best=0.36,
        paper_table6_best_std=0.53,
        notes=(
            "Experiment 4 Cruise Random Reynolds extrapolation split.",
            "Test set is the highest and lowest 5% of Reynolds number.",
        ),
    ),
    "cruise_random_stagger_extrap": TaskSpec(
        dataset_group="cruise_random",
        split_style="tail_extrapolation",
        variable="stagger",
        pickle_files=(
            "cruise_randomFields_mgn_Part1.pickle",
            "cruise_randomFields_mgn_Part2.pickle",
            "cruise_randomFields_mgn_Part3.pickle",
        ),
        paper_table6_baseline=1.74,
        paper_table6_best=0.13,
        paper_table6_best_std=0.17,
        notes=(
            "Experiment 4 Cruise Random stagger extrapolation split.",
            "Test set is the highest and lowest 5% of stagger.",
        ),
    ),
    "cruise_random_gap_extrap": TaskSpec(
        dataset_group="cruise_random",
        split_style="tail_extrapolation",
        variable="gap",
        pickle_files=(
            "cruise_randomFields_mgn_Part1.pickle",
            "cruise_randomFields_mgn_Part2.pickle",
            "cruise_randomFields_mgn_Part3.pickle",
        ),
        paper_table6_baseline=1.95,
        paper_table6_best=0.14,
        paper_table6_best_std=0.20,
        notes=(
            "Experiment 4 Cruise Random gap extrapolation split.",
            "Test set is the highest and lowest 5% of gap.",
        ),
    ),
    "racecar_uniform": TaskSpec(
        dataset_group="racecar",
        split_style="uniform",
        variable=None,
        pickle_files=(
            "raceCar_randomFields_mgn_Part1.pickle",
            "raceCar_randomFields_mgn_Part2.pickle",
            "raceCar_randomFields_mgn_Part3.pickle",
        ),
        paper_table6_baseline=0.61,
        paper_table6_best=0.21,
        paper_table6_best_std=0.29,
        notes=(
            "Experiment 4 Race Car uniform split.",
            "Paper uses uniform sampling on Race Car rather than an extrapolation split.",
        ),
    ),
}


def _resolve_task_pickle_paths(task: TaskSpec, root_candidates: list[str]) -> list[Path]:
    root = first_existing(root_candidates)
    if root is None:
        raise FileNotFoundError(
            "Could not find TandemFoilSet root. Checked: "
            + ", ".join(root_candidates)
        )
    return [root / name for name in task.pickle_files]


def _extract_records(pickle_paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    task_local_idx = 0
    for file_idx, path in enumerate(pickle_paths):
        raw = load_pickle(path)
        for local_idx, sample in enumerate(raw):
            aoa = sample.AoA
            if isinstance(aoa, list):
                aoa0 = float(aoa[0])
                aoa1 = float(aoa[1])
            else:
                aoa0 = float(aoa)
                aoa1 = None

            gap = getattr(sample, "gap", None)
            stagger = getattr(sample, "stagger", None)
            if gap is None or stagger is None:
                raise ValueError(f"Expected tandem metadata gap/stagger in {path.name}:{local_idx}")

            records.append(
                {
                    "task_local_idx": task_local_idx,
                    "file_idx": file_idx,
                    "local_idx": local_idx,
                    "re": float(sample.flowState["Re"]),
                    "aoa0": aoa0,
                    "aoa1": aoa1,
                    "gap": float(gap),
                    "stagger": float(stagger),
                }
            )
            task_local_idx += 1
        del raw
    return records


def split_uniform_indices(total: int, *, seed: int = SEED) -> dict[str, list[int]]:
    if total <= 0:
        raise ValueError("Uniform split requires at least one sample")
    rng = np.random.default_rng(seed)
    order = np.arange(total)
    rng.shuffle(order)
    n_test = max(1, round(total * VAL_FRACTION))
    n_val = max(1, round(total * VAL_FRACTION))
    n_train = total - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Invalid split sizes for total={total}")
    train = sorted(order[:n_train].tolist())
    val = sorted(order[n_train : n_train + n_val].tolist())
    test = sorted(order[n_train + n_val :].tolist())
    return {"train": train, "val": val, "test": test}


def split_tail_extrapolation_indices(values: list[float], *, seed: int = SEED) -> dict[str, list[int]]:
    total = len(values)
    if total <= 0:
        raise ValueError("Tail extrapolation requires at least one sample")
    n_tail = max(1, round(total * TAIL_FRACTION))
    order = sorted(range(total), key=lambda idx: (values[idx], idx))
    test_set = set(order[:n_tail]) | set(order[-n_tail:])
    if len(test_set) != n_tail * 2:
        raise ValueError("Tail extrapolation test set unexpectedly overlapped")

    middle = [idx for idx in range(total) if idx not in test_set]
    rng = np.random.default_rng(seed)
    rng.shuffle(middle)
    n_val = max(1, round(total * VAL_FRACTION))
    n_train = len(middle) - n_val
    if n_train <= 0:
        raise ValueError(f"Invalid tail split sizes for total={total}")
    train = sorted(middle[:n_train])
    val = sorted(middle[n_train:])
    test = sorted(test_set)
    return {"train": train, "val": val, "test": test}


def _compute_y_stats(pickle_paths: list[Path], train_indices: list[int]) -> dict[str, float | list[float]]:
    dataset = MultiFieldDataset(pickle_paths, cache_size=-1)
    train_sorted = sorted(train_indices, key=lambda idx: dataset.index[idx])

    n_channels = 3
    sum_y = torch.zeros(n_channels, dtype=torch.float64)
    finite_nodes = torch.zeros(n_channels, dtype=torch.int64)
    for idx in train_sorted:
        _, y, _ = dataset[idx]
        yd = y.double()
        mask = yd.isfinite()
        sum_y += torch.where(mask, yd, torch.zeros_like(yd)).sum(dim=0)
        finite_nodes += mask.sum(dim=0)
    if finite_nodes.min().item() <= 1:
        raise ValueError("Need at least two finite nodes per channel to compute y statistics")
    mean_y = sum_y / finite_nodes.double()

    sq_y = torch.zeros(n_channels, dtype=torch.float64)
    for idx in train_sorted:
        _, y, _ = dataset[idx]
        yd = y.double()
        mask = yd.isfinite()
        diff = torch.where(mask, yd - mean_y, torch.zeros_like(yd))
        sq_y += (diff ** 2).sum(dim=0)
    std_y = (sq_y / (finite_nodes.double() - 1)).sqrt().clamp(min=1e-6)
    return {
        "n_train_samples": len(train_indices),
        "n_train_nodes": int(finite_nodes.min().item()),
        "y_mean": mean_y.float().tolist(),
        "y_std": std_y.float().tolist(),
    }


def build_artifacts(root_candidates: list[str]) -> tuple[dict, dict]:
    manifest = {
        "dataset": "TandemFoilSet",
        "benchmark_variant": "paper_experiment4",
        "paper_source": "https://openreview.net/pdf?id=4Z0P4Nbosn",
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "tail_fraction": TAIL_FRACTION,
        "data_root_candidates": list(root_candidates),
        "tasks": {},
    }
    stats = {
        "dataset": "TandemFoilSet",
        "benchmark_variant": "paper_experiment4",
        "seed": SEED,
        "tasks": {},
    }

    for task_name, task in TASK_SPECS.items():
        pickle_paths = _resolve_task_pickle_paths(task, root_candidates)
        records = _extract_records(pickle_paths)
        if task.split_style == "uniform":
            splits = split_uniform_indices(len(records), seed=SEED)
        elif task.split_style == "tail_extrapolation":
            if task.variable is None:
                raise ValueError(f"Task {task_name} is missing its extrapolation variable")
            splits = split_tail_extrapolation_indices(
                [float(record[task.variable]) for record in records],
                seed=SEED,
            )
        else:
            raise ValueError(f"Unknown split style: {task.split_style}")

        manifest["tasks"][task_name] = {
            "dataset_group": task.dataset_group,
            "split_style": task.split_style,
            "variable": task.variable,
            "pickle_files": [path.name for path in pickle_paths],
            "n_cases": len(records),
            "split_counts": {name: len(indices) for name, indices in splits.items()},
            "splits": {name: sorted(indices) for name, indices in splits.items()},
            "paper_targets": {
                "table": "Table 6",
                "mgn_baseline_field_mse": task.paper_table6_baseline,
                "mgn_pre_res_free_res_comb_field_mse": task.paper_table6_best,
                "mgn_pre_res_free_res_comb_field_mse_std": task.paper_table6_best_std,
            },
            "notes": list(task.notes),
        }
        stats["tasks"][task_name] = _compute_y_stats(pickle_paths, splits["train"])
    return manifest, stats


def materialize_default_artifacts(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    stats_path: str | Path = DEFAULT_STATS,
    root_candidates: list[str] | None = None,
) -> tuple[Path, Path]:
    manifest_path = Path(manifest_path)
    stats_path = Path(stats_path)
    if manifest_path.exists() and stats_path.exists():
        return manifest_path, stats_path
    resolved_candidates = list(root_candidates or DEFAULT_ROOT_CANDIDATES)
    manifest, stats = build_artifacts(resolved_candidates)
    write_json(manifest_path, manifest)
    write_json(stats_path, stats)
    return manifest_path, stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize TandemFoilSet paper Experiment 4 artifacts")
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--stats-out", default=str(DEFAULT_STATS))
    parser.add_argument(
        "--data-root-candidate",
        action="append",
        default=[],
        help="Override/add TandemFoilSet root candidates.",
    )
    args = parser.parse_args()

    root_candidates = args.data_root_candidate or DEFAULT_ROOT_CANDIDATES
    manifest, stats = build_artifacts(root_candidates)
    write_json(args.manifest_out, manifest)
    write_json(args.stats_out, stats)

    print(f"Wrote {args.manifest_out}")
    print(f"Wrote {args.stats_out}")
    for task_name, task_manifest in manifest["tasks"].items():
        counts = task_manifest["split_counts"]
        print(
            f"  {task_name:30s} train={counts['train']:4d} "
            f"val={counts['val']:4d} test={counts['test']:4d}"
        )


if __name__ == "__main__":
    main()
