# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Extended preprocessing for multi-foil datasets.

Addresses three limitations of the read-only prepare.py:
  1. NACA[1] and AoA[1] are now encoded in x (foil 2 explicit features).
  2. SURFACE_IDS includes boundary ID 7 (foil 2 surface nodes).
  3. gap/stagger are added as global conditioning features.

For single-foil samples: AoA1=0, NACA1=[0,0,0], gap=0, stagger=0.
The 0-sentinel is distinguishable from real tandem values (cruise gap min ≈ -0.8,
racecar gap min ≈ 0.4), so the model can learn to attend to it appropriately.

x layout (N, 24):
  [pos(2), saf(2), dsdf(8), is_surface(1), log_Re(1),
   AoA0_rad(1), NACA0(3),
   AoA1_rad(1), NACA1(3), gap(1), stagger(1)]
   = 2+2+8+1+1+1+3+1+3+1+1 = 24 dimensions

Model config: space_dim=2, fun_dim=22  →  preprocess MLP input = 24.
"""

import json
import math
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path

# Reach repo root for load_pickle and DATA_ROOT
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare import load_pickle, DATA_ROOT, parse_naca, pad_collate  # noqa: F401 (re-export pad_collate)

# Includes foil 2 surface (ID 7) — fixes the SURFACE_IDS=(5,6) gap in prepare.py
SURFACE_IDS_MULTI = (5, 6, 7)

X_DIM = 24  # total x feature dimension


def preprocess_sample_multi(sample):
    """Convert raw PyG sample to (x, y, is_surface) with full dual-foil features.

    Returns:
        x:          (N, 24) float32
        y:          (N, 3)  float32  — [Ux, Uy, p]
        is_surface: (N,)    bool     — True for both foil 1 and foil 2 surface nodes
    """
    n = sample.pos.shape[0]

    # Surface mask — includes boundary ID 7 for foil 2
    is_surface = torch.zeros(n, dtype=torch.bool)
    for sid in SURFACE_IDS_MULTI:
        is_surface |= sample.boundary == sid

    # Foil 1 AoA and NACA
    aoa = sample.AoA
    if isinstance(aoa, list):
        aoa0 = float(aoa[0])
        aoa1 = float(aoa[1])
    else:
        aoa0 = float(aoa)
        aoa1 = 0.0  # single-foil sentinel

    naca0 = parse_naca(sample.NACA[0])
    naca1 = parse_naca(sample.NACA[1]) if len(sample.NACA) > 1 else (0.0, 0.0, 0.0)

    log_re = math.log(float(sample.flowState["Re"]))
    aoa0_rad = aoa0 * math.pi / 180.0
    aoa1_rad = aoa1 * math.pi / 180.0

    # Tandem geometry — 0 for single-foil samples
    gap_val = getattr(sample, "gap", None)
    stagger_val = getattr(sample, "stagger", None)
    gap = float(gap_val) if gap_val is not None else 0.0
    stagger = float(stagger_val) if stagger_val is not None else 0.0

    x = torch.cat([
        sample.pos.float(),                                        # 2
        sample.saf.float(),                                        # 2
        sample.dsdf.float(),                                       # 8
        is_surface.float().unsqueeze(1),                           # 1
        torch.full((n, 1), log_re),                                # 1
        torch.full((n, 1), aoa0_rad),                              # 1
        torch.tensor(naca0, dtype=torch.float32).expand(n, 3),     # 3
        torch.full((n, 1), aoa1_rad),                              # 1
        torch.tensor(naca1, dtype=torch.float32).expand(n, 3),     # 3
        torch.full((n, 1), gap),                                   # 1
        torch.full((n, 1), stagger),                               # 1
    ], dim=1)  # 2+2+8+1+1+1+3+1+3+1+1 = 24

    assert x.shape[1] == X_DIM, f"Expected {X_DIM} features, got {x.shape[1]}"

    y = sample.y.float()
    return x, y, is_surface


class MultiFieldDataset(Dataset):
    """Full-field flow dataset with extended dual-foil features.

    Drop-in replacement for FullFieldDataset using preprocess_sample_multi.
    Supports multiple pickle files and the same cache_size semantics:
        0  → eager load everything into RAM
        -1 → pure lazy loading (no cache)
        >0 → capacity-limited cache up to N samples (no eviction)
    """

    def __init__(self, pickle_paths: list, cache_size: int = 0):
        self.paths = [Path(p) for p in pickle_paths]
        self.cache_size = cache_size

        # Build flat index: [(file_idx, local_idx), ...]
        self.index = []
        for fi, p in enumerate(self.paths):
            raw = load_pickle(p)
            for si in range(len(raw)):
                self.index.append((fi, si))
            del raw

        self._cache: dict = {}
        self._file_cache_idx = -1
        self._file_cache_data = None

        if cache_size == 0:
            print(f"Preprocessing {len(self.index)} samples into RAM (multi-foil)...")
            for fi, p in enumerate(self.paths):
                raw = load_pickle(p)
                offset = next(
                    i for i, (f, _) in enumerate(self.index) if f == fi
                )
                for si, sample in enumerate(raw):
                    self._cache[offset + si] = preprocess_sample_multi(sample)
                del raw
            print(f"  Cached {len(self._cache)} samples")

    def _load_file(self, file_idx):
        if self._file_cache_idx != file_idx:
            self._file_cache_data = load_pickle(self.paths[file_idx])
            self._file_cache_idx = file_idx
        return self._file_cache_data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        fi, si = self.index[idx]
        raw_list = self._load_file(fi)
        result = preprocess_sample_multi(raw_list[si])

        if self.cache_size == -1:
            pass
        elif self.cache_size > 0 and len(self._cache) < self.cache_size:
            self._cache[idx] = result

        return result


VAL_SPLIT_NAMES = ["val_in_dist", "val_tandem_transfer", "val_ood_cond", "val_ood_re"]


def _stratified_sample(indices: list, n: int) -> list:
    """Pick n samples spread evenly across the index list."""
    if len(indices) <= n:
        return indices
    step = max(1, len(indices) // n)
    return indices[::step][:n]


def load_structured_split(
    manifest_path: str,
    stats_file: str,
    debug: bool = False,
) -> tuple[Subset, dict[str, Subset], dict[str, torch.Tensor], torch.Tensor]:
    """Load structured benchmark split from manifest.

    Returns:
        train_ds:       Subset for training
        val_splits:     {split_name: Subset} for each validation track
        stats:          {y_mean, y_std, x_mean, x_std} as CPU float32 tensors
        sample_weights: per-sample weights for balanced domain sampling
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(stats_file) as f:
        stats_data = json.load(f)

    cache_size = -1 if debug else 0
    ds = MultiFieldDataset(
        [Path(p) for p in manifest["pickle_paths"]],
        cache_size=cache_size,
    )

    # Validate manifest indices are in bounds
    all_idx = [i for v in manifest["splits"].values() for i in v]
    max_idx = max(all_idx) if all_idx else 0
    assert max_idx < len(ds), (
        f"Manifest references index {max_idx} but dataset only has {len(ds)} samples. "
        "Pickle files may have changed — re-run split.py."
    )

    train_ds = Subset(ds, manifest["splits"]["train"])
    val_splits = {name: Subset(ds, manifest["splits"][name]) for name in VAL_SPLIT_NAMES}

    if debug:
        dg = manifest["domain_groups"]
        debug_train = (
            _stratified_sample(dg["racecar_single"], 2) +
            _stratified_sample(dg["racecar_tandem"], 2) +
            _stratified_sample(dg["cruise"], 2)
        )
        train_ds = Subset(ds, debug_train)
        val_splits = {k: Subset(ds, _stratified_sample(manifest["splits"][k], 2))
                      for k in VAL_SPLIT_NAMES}

    # Normalization stats (CPU tensors — caller moves to device)
    stats = {
        "y_mean": torch.tensor(stats_data["y_mean"], dtype=torch.float32),
        "y_std":  torch.tensor(stats_data["y_std"],  dtype=torch.float32),
        "x_mean": torch.tensor(stats_data["x_mean"], dtype=torch.float32),
        "x_std":  torch.tensor(stats_data["x_std"],  dtype=torch.float32),
    }

    # Balanced domain sampler weights
    group_sizes = {name: len(idxs) for name, idxs in manifest["domain_groups"].items()}
    idx_to_group: dict[int, str] = {}
    for name, idxs in manifest["domain_groups"].items():
        for i in idxs:
            idx_to_group[i] = name

    sample_weights = torch.tensor(
        [1.0 / group_sizes[idx_to_group[i]] for i in train_ds.indices],
        dtype=torch.float64,
    )

    print(f"Train: {len(train_ds)}, " +
          ", ".join(f"{k}: {len(v)}" for k, v in val_splits.items()))

    return train_ds, val_splits, stats, sample_weights
