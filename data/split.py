# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""One-time offline script: generate split manifest and normalization stats.

Produces two committed JSON files used by train.py:
  split_manifest.json  — train/val indices with domain tags
  split_stats.json     — x/y normalization stats over training set only

Run: python data/split.py

KNOWN LIMITATIONS (inherited from read-only prepare.py):
  - Only NACA[0] and AoA[0] are encoded in x features. Foil 2 identity
    is implicit in dsdf/saf geometry only.
  - SURFACE_IDS=(5,6) misses boundary ID 7 (foil 2 surface in tandem data).
    Tandem surface nodes are underweighted in surf_loss.
  - Whether NACA 6416 (tandem Part2 front foil) appears in raceCar_single
    training is unverified; val_tandem_transfer may test "unseen shape in
    tandem" rather than true single→tandem transfer.
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from data.prepare import DATA_ROOT, load_pickle
from data.prepare_multi import MultiFieldDataset

SEED = 42
# Retain this fraction of each source (evenly spaced to preserve condition coverage).
# 0.70 → 30% reduction while keeping balanced representation across all data sources.
SAMPLE_FRACTION = 0.70
OUT_DIR = Path(__file__).parent
OUT_MANIFEST = OUT_DIR / "split_manifest.json"
OUT_STATS = OUT_DIR / "split_stats.json"

# Files in the exact order FullFieldDataset will see them.
# global_idx = sum(file_sample_counts[:file_idx]) + local_idx
# This ordering must match what train.py passes to FullFieldDataset.
PICKLE_FILES = [
    DATA_ROOT / "raceCar_single_randomFields.pickle",        # file_idx 0
    DATA_ROOT / "raceCar_randomFields_mgn_Part1.pickle",     # file_idx 1 (front=2412)
    DATA_ROOT / "raceCar_randomFields_mgn_Part2.pickle",     # file_idx 2 (front=6416) → val
    DATA_ROOT / "raceCar_randomFields_mgn_Part3.pickle",     # file_idx 3 (front=9412)
    DATA_ROOT / "cruise_randomFields_mgn_Part1.pickle",      # file_idx 4 (Re=1.475M)
    DATA_ROOT / "cruise_randomFields_mgn_Part2.pickle",      # file_idx 5 (Re=4.445M) → val
    DATA_ROOT / "cruise_randomFields_mgn_Part3.pickle",      # file_idx 6 (Re=802K)
]

# Expected total across all 7 files
FILE_SIZES_TOTAL = 2699  # total samples across all 7 pickle files


def extract_metadata(pickle_files):
    """Scan all pickle files and extract lightweight metadata only.

    Reads: flowState, AoA, NACA, gap, stagger per sample.
    Does NOT call preprocess_sample — avoids loading pos/y/saf/dsdf tensors.

    Returns list of dicts in global_idx order (same ordering FullFieldDataset uses).
    """
    records = []
    global_offset = 0

    for file_idx, path in enumerate(pickle_files):
        print(f"  Scanning [{file_idx}] {path.name} ...")
        raw = load_pickle(path)
        n = len(raw)

        for local_idx, sample in enumerate(raw):
            re = float(sample.flowState["Re"])

            aoa = sample.AoA
            if isinstance(aoa, list):
                aoa0, aoa1 = float(aoa[0]), float(aoa[1])
            else:
                aoa0, aoa1 = float(aoa), None

            # gap/stagger: present on tandem samples, None on single-foil
            # PyG Data.__getattr__ returns None for missing keys without raising
            gap = getattr(sample, "gap", None)
            stagger = getattr(sample, "stagger", None)
            if gap is not None:
                gap = float(gap)
            if stagger is not None:
                stagger = float(stagger)

            records.append({
                "global_idx": global_offset + local_idx,
                "file_idx": file_idx,
                "local_idx": local_idx,
                "re": re,
                "aoa0": aoa0,
                "aoa1": aoa1,
                "naca": list(sample.NACA),
                "gap": gap,
                "stagger": stagger,
            })

        del raw  # release immediately; each file is 3–6 GB
        print(f"    {n} samples, global range [{global_offset}, {global_offset + n - 1}]")
        global_offset += n

    return records


def _subsample(idxs: list, fraction: float, rng=None) -> list:
    """Return an evenly-spaced subset preserving condition-space coverage.

    Uses fixed stride (not random) so the subset spans the full index range.
    For the racecar_single case, pass rng to shuffle before subsampling so
    the val holdout is random rather than always the first 10%.
    """
    n = max(1, round(len(idxs) * fraction))
    if n >= len(idxs):
        return idxs
    if rng is not None:
        arr = np.array(idxs)
        rng.shuffle(arr)
        return arr[:n].tolist()
    step = len(idxs) / n
    return [idxs[round(i * step)] for i in range(n)]


def assign_splits(records):
    """Assign every sample to exactly one split.

    Applies SAMPLE_FRACTION evenly to each data source so the 30% reduction
    is balanced: racecar_single, racecar_tandem (train+val), cruise (train+val)
    are all reduced proportionally.

    Returns:
        splits       — {split_name: [global_idx, ...]}
        domain_groups — {domain_name: [global_idx, ...]}  (train indices only)
    """
    rng = np.random.default_rng(SEED)

    splits = {k: [] for k in
              ["train", "val_in_dist", "val_tandem_transfer", "val_ood_cond", "val_ood_re"]}
    domain_groups = {"racecar_single": [], "racecar_tandem": [], "cruise": []}

    # Group records by file_idx
    by_file: dict[int, list] = {}
    for rec in records:
        by_file.setdefault(rec["file_idx"], []).append(rec)

    # --- Rule 1: raceCar_single (file_idx=0) → subsample, then 90/10 train/val ---
    # Shuffle first so val holdout is a random draw, not always the last N samples.
    single_all = [r["global_idx"] for r in by_file[0]]
    single_keep = _subsample(single_all, SAMPLE_FRACTION, rng=rng)
    n_val = max(1, round(len(single_keep) * 0.10))
    splits["val_in_dist"].extend(single_keep[:n_val])
    train_single = single_keep[n_val:]
    splits["train"].extend(train_single)
    domain_groups["racecar_single"].extend(train_single)

    # --- Rule 2: raceCar tandem Part1+3 (file_idx=1,3) → subsample → train ---
    # Evenly spaced so both low and high stagger/gap values are retained.
    for fi in (1, 3):
        idxs = [r["global_idx"] for r in by_file[fi]]
        kept = _subsample(idxs, SAMPLE_FRACTION)
        splits["train"].extend(kept)
        domain_groups["racecar_tandem"].extend(kept)

    # --- Rule 3: raceCar tandem Part2 (file_idx=2) → subsample → val_tandem_transfer ---
    idxs_p2 = [r["global_idx"] for r in by_file[2]]
    splits["val_tandem_transfer"].extend(_subsample(idxs_p2, SAMPLE_FRACTION))

    # --- Rule 4: cruise Part1+3 (file_idx=4,6) → subsample first, then frontier split ---
    # Subsample before the frontier computation so the frontier 20% is drawn from
    # the kept subset (preserving the OOD character of the frontier samples).
    cruise_p1p3_all = by_file[4] + by_file[6]
    cruise_keep_idxs = _subsample(list(range(len(cruise_p1p3_all))), SAMPLE_FRACTION)
    cruise_p1p3 = [cruise_p1p3_all[i] for i in cruise_keep_idxs]

    feats = np.array(
        [[r["aoa0"], r["gap"], r["stagger"]] for r in cruise_p1p3],
        dtype=np.float64,
    )
    feat_min = feats.min(axis=0)
    feat_max = feats.max(axis=0)
    feat_range = np.where(feat_max - feat_min > 0, feat_max - feat_min, 1.0)
    feats_norm = (feats - feat_min) / feat_range
    centroid = feats_norm.mean(axis=0)
    dists = np.linalg.norm(feats_norm - centroid, axis=1)

    n_frontier = max(1, round(len(dists) * 0.20))
    frontier_set = set(np.argsort(-dists)[:n_frontier].tolist())

    cruise_train_idxs = []
    for i, rec in enumerate(cruise_p1p3):
        if i in frontier_set:
            splits["val_ood_cond"].append(rec["global_idx"])
        else:
            cruise_train_idxs.append(rec["global_idx"])
    splits["train"].extend(cruise_train_idxs)
    domain_groups["cruise"].extend(cruise_train_idxs)

    # --- Rule 5: cruise Part2 (file_idx=5) → subsample → val_ood_re ---
    idxs_p5 = [r["global_idx"] for r in by_file[5]]
    splits["val_ood_re"].extend(_subsample(idxs_p5, SAMPLE_FRACTION))

    # --- Limitation 3 check: NACA overlap for val_tandem_transfer validity ---
    # val_tandem_transfer is meaningful as a "transfer" test only if the tandem
    # Part2 front foil (NACA 6416) appears in raceCar_single training.
    single_train_nacas: set[str] = set()
    single_train_ids = set(train_single)
    for rec in by_file[0]:
        if rec["global_idx"] in single_train_ids:
            single_train_nacas.update(rec["naca"])

    tandem_p2_nacas: set[str] = set()
    for rec in by_file[2]:
        tandem_p2_nacas.update(rec["naca"])

    overlap = single_train_nacas & tandem_p2_nacas
    print(f"\n  [Transfer check] NACA codes in raceCar_single train: {sorted(single_train_nacas)}")
    print(f"  [Transfer check] NACA codes in tandem Part2 (val_tandem_transfer): {sorted(tandem_p2_nacas)}")
    if overlap:
        print(f"  [Transfer check] OVERLAP (true transfer test): {sorted(overlap)}")
    else:
        print("  [Transfer check] NO OVERLAP — val_tandem_transfer tests 'unseen shape in tandem',"
              " not single→tandem transfer. Consider this in result interpretation.")

    return splits, domain_groups


def compute_stats(pickle_files, train_indices):
    """Compute x/y normalization stats over training samples (two-pass).

    Uses MultiFieldDataset (24-dim x) so stats match train.py.
    Sorts train_indices by (file_idx, local_idx) for sequential file I/O.
    """
    import torch

    ds = MultiFieldDataset(pickle_files, cache_size=-1)  # lazy — no RAM accumulation

    # Sort for sequential file access
    train_sorted = sorted(train_indices, key=lambda i: ds.index[i])
    n = len(train_sorted)

    from data.prepare_multi import X_DIM
    print(f"  Pass 1/2 (mean) over {n} training samples ...")
    sum_x = torch.zeros(X_DIM)
    sum_y = torch.zeros(3)
    total_nodes = 0

    for i, idx in enumerate(train_sorted):
        if i % 200 == 0:
            print(f"    {i}/{n}")
        x, y, _ = ds[idx]
        sum_x += x.sum(dim=0)
        sum_y += y.sum(dim=0)
        total_nodes += x.shape[0]

    mean_x = sum_x / total_nodes
    mean_y = sum_y / total_nodes

    print(f"  Pass 2/2 (std) over {n} training samples ...")
    sq_x = torch.zeros(X_DIM)
    sq_y = torch.zeros(3)

    for i, idx in enumerate(train_sorted):
        if i % 200 == 0:
            print(f"    {i}/{n}")
        x, y, _ = ds[idx]
        sq_x += ((x - mean_x) ** 2).sum(dim=0)
        sq_y += ((y - mean_y) ** 2).sum(dim=0)

    std_x = (sq_x / (total_nodes - 1)).sqrt().clamp(min=1e-6)
    std_y = (sq_y / (total_nodes - 1)).sqrt().clamp(min=1e-6)

    return {
        "version": 1,
        "n_train_samples": n,
        "n_train_nodes": total_nodes,
        "y_mean": mean_y.tolist(),
        "y_std": std_y.tolist(),
        "x_mean": mean_x.tolist(),
        "x_std": std_x.tolist(),
    }


def make_quick_manifest():
    """Generate a debug manifest and placeholder stats instantly — no pickle loading.

    Uses known file sizes to build global index offsets directly.
    Stats use identity normalization (mean=0, std=1) — fine for debug smoke-tests
    since we only care about the training loop running, not absolute loss values.

    Run:  python data/split.py --quick
    """
    from data.prepare_multi import X_DIM

    # Known sample counts per file (see README.md dataset inventory)
    # Order must match PICKLE_FILES exactly.
    FILE_SIZES = [899, 300, 300, 300, 300, 300, 300]
    offsets = []
    acc = 0
    for n in FILE_SIZES:
        offsets.append(acc)
        acc += n

    def _pick(file_idx, local_indices):
        return [offsets[file_idx] + i for i in local_indices]

    # 2 samples per domain in train (spread across file)
    train = (
        _pick(0, [0, 450]) +           # racecar_single
        _pick(1, [0, 150]) +           # racecar_tandem P1
        _pick(3, [0, 150]) +           # racecar_tandem P3
        _pick(4, [0, 150]) +           # cruise P1
        _pick(6, [0, 150])             # cruise P3
    )
    domain_groups = {
        "racecar_single": _pick(0, [0, 450]),
        "racecar_tandem": _pick(1, [0, 150]) + _pick(3, [0, 150]),
        "cruise":         _pick(4, [0, 150]) + _pick(6, [0, 150]),
    }

    splits = {
        "train":               sorted(train),
        "val_in_dist":         _pick(0, [200, 600]),
        "val_tandem_transfer": _pick(2, [0, 150]),
        "val_ood_cond":        _pick(4, [290]) + _pick(6, [290]),
        "val_ood_re":          _pick(5, [0, 150]),
    }

    manifest = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat() + " [QUICK/DEBUG]",
        "pickle_paths": [str(p) for p in PICKLE_FILES],
        "splits": splits,
        "domain_groups": domain_groups,
        "split_counts": {k: len(v) for k, v in splits.items()},
    }

    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUT_MANIFEST} (quick debug manifest)")
    for k, v in manifest["split_counts"].items():
        print(f"  {k:30s}: {v:4d} samples")

    # Identity normalization — debug only
    stats = {
        "version": 1,
        "n_train_samples": len(train),
        "n_train_nodes": 0,
        "y_mean": [0.0, 0.0, 0.0],
        "y_std":  [1.0, 1.0, 1.0],
        "x_mean": [0.0] * X_DIM,
        "x_std":  [1.0] * X_DIM,
    }
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {OUT_STATS} (identity normalization — debug only)")


def main():
    import sys as _sys
    if "--quick" in _sys.argv:
        make_quick_manifest()
        return

    print("=== Phase 1: Metadata extraction ===")
    records = extract_metadata(PICKLE_FILES)

    print(f"\n=== Phase 2: Split assignment ===")
    splits, domain_groups = assign_splits(records)

    # Validate no leakage and correct total
    all_idx = [i for v in splits.values() for i in v]
    assert len(all_idx) == len(set(all_idx)), "BUG: duplicate indices across splits!"
    expected_approx = round(FILE_SIZES_TOTAL * SAMPLE_FRACTION)
    print(f"\n  Total assigned: {len(all_idx)} samples "
          f"(target ≈{expected_approx}, fraction={SAMPLE_FRACTION:.0%})")

    manifest = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "pickle_paths": [str(p) for p in PICKLE_FILES],
        "splits": {k: sorted(v) for k, v in splits.items()},
        "domain_groups": {k: sorted(v) for k, v in domain_groups.items()},
        "split_counts": {k: len(v) for k, v in splits.items()},
    }

    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {OUT_MANIFEST}")

    print("\nSplit summary:")
    for k, v in manifest["split_counts"].items():
        print(f"  {k:30s}: {v:4d} samples")

    print(f"\n=== Phase 3: Normalization stats ===")
    stats = compute_stats(PICKLE_FILES, splits["train"])

    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote {OUT_STATS}")
    print(f"  n_train_samples : {stats['n_train_samples']}")
    print(f"  n_train_nodes   : {stats['n_train_nodes']}")
    print(f"  y_mean          : {[f'{v:.3f}' for v in stats['y_mean']]}")
    print(f"  y_std           : {[f'{v:.3f}' for v in stats['y_std']]}")


if __name__ == "__main__":
    main()
