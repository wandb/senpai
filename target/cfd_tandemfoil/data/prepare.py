# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Prepare TandemFoilSet data for CFD surrogate training.

Full-field extraction with lazy loading and aggressive caching for fast training.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path

DATA_ROOT = Path("/mnt/new-pvc/datasets/tandemfoil")

SURFACE_IDS = (5, 6)


def load_pickle(path: str | Path) -> list:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_naca(naca_str: str) -> tuple[float, float, float]:
    """Parse NACA 4-digit code into (max_camber, camber_pos, thickness) in [0,1]."""
    if len(naca_str) == 4 and naca_str.isdigit():
        return int(naca_str[0]) / 9.0, int(naca_str[1]) / 9.0, int(naca_str[2:]) / 24.0
    return 0.0, 0.0, 0.0


def preprocess_sample(sample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw PyG sample into tensors ready for __getitem__.

    Returns (x, y, is_surface) with all feature engineering done upfront.
    """
    n = sample.pos.shape[0]

    is_surface = torch.zeros(n, dtype=torch.bool)
    for sid in SURFACE_IDS:
        is_surface |= sample.boundary == sid

    aoa = sample.AoA
    if isinstance(aoa, list):
        aoa = aoa[0]

    naca = parse_naca(sample.NACA[0])
    log_re = torch.log(torch.tensor(float(sample.flowState["Re"]))).item()
    aoa_rad = float(aoa) * torch.pi / 180.0

    # Build x: (N, 18) — all feature engineering done once
    x = torch.cat([
        sample.pos.float(),                                     # 2
        sample.saf.float(),                                     # 2
        sample.dsdf.float(),                                    # 8
        is_surface.float().unsqueeze(1),                        # 1
        torch.full((n, 1), log_re),                             # 1
        torch.full((n, 1), aoa_rad),                            # 1
        torch.tensor(naca, dtype=torch.float32).expand(n, 3),   # 3
    ], dim=1)

    y = sample.y.float()
    return x, y, is_surface


class FullFieldDataset(Dataset):
    """Full-field flow dataset with lazy loading and LRU file cache.

    Each sample returns (x, y, is_surface) where:
        x:          (N, 18) float32  — [pos(2), saf(2), dsdf(8), is_surface(1), log_Re(1), AoA(1), naca(3)]
        y:          (N, 3)  float32  — [Ux, Uy, p]
        is_surface: (N,)    bool     — True for airfoil surface nodes
    """

    def __init__(self, pickle_paths: list[str | Path], cache_size: int = 0):
        """
        Args:
            pickle_paths: list of pickle file paths to load
            cache_size: max samples to keep in RAM. 0 = cache everything (eager).
                        Use -1 for no cache (pure lazy loading).
        """
        self.paths = [Path(p) for p in pickle_paths]
        self.cache_size = cache_size

        # Build index: (file_idx, sample_idx_within_file)
        self.index = []
        for fi, p in enumerate(self.paths):
            raw = load_pickle(p)
            for si in range(len(raw)):
                self.index.append((fi, si))
            del raw

        # Sample cache: idx -> (x, y, is_surface) preprocessed tensors
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # File cache: file_idx -> raw list (for lazy loading)
        self._file_cache_idx = -1
        self._file_cache_data = None

        # If cache_size == 0, eagerly preprocess everything
        if cache_size == 0:
            print(f"Preprocessing {len(self.index)} samples into RAM...")
            for fi, p in enumerate(self.paths):
                raw = load_pickle(p)
                for si, sample in enumerate(raw):
                    global_idx = next(i for i, (f, s) in enumerate(self.index) if f == fi and s == si)
                    self._cache[global_idx] = preprocess_sample(sample)
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
        # Check preprocessed cache first
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        # Lazy load and preprocess
        fi, si = self.index[idx]
        raw_list = self._load_file(fi)
        result = preprocess_sample(raw_list[si])

        # Cache if within budget
        if self.cache_size == -1:
            pass  # no caching
        elif self.cache_size > 0 and len(self._cache) < self.cache_size:
            self._cache[idx] = result

        return result


def pad_collate(batch):
    """Collate variable-length samples into padded batches.

    Returns:
        x:          (B, N_max, 18) float32
        y:          (B, N_max, 3)  float32
        is_surface: (B, N_max)     bool
        mask:       (B, N_max)     bool
    """
    xs, ys, surfs = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    B = len(xs)
    x_pad = torch.zeros(B, max_n, xs[0].shape[1])
    y_pad = torch.zeros(B, max_n, ys[0].shape[1])
    surf_pad = torch.zeros(B, max_n, dtype=torch.bool)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (x, y, sf) in enumerate(zip(xs, ys, surfs)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        surf_pad[i, :n] = sf
        mask[i, :n] = True
    return x_pad, y_pad, surf_pad, mask


if __name__ == "__main__":
    paths = [DATA_ROOT / "raceCar_single_randomFields.pickle"]

    print("=== Eager mode (cache_size=0) ===")
    ds = FullFieldDataset(paths, cache_size=0)
    print(f"Loaded {len(ds)} samples")

    x, y, is_surf = ds[0]
    print(f"x: {x.shape}, y: {y.shape}, surface: {is_surf.sum()}/{x.shape[0]}")

    # Benchmark iteration speed
    import time
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, collate_fn=pad_collate,
                        num_workers=4, persistent_workers=True, pin_memory=True)
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 50:
            break
    t1 = time.time()
    print(f"50 batches in {t1-t0:.2f}s ({(t1-t0)/50*1000:.0f}ms/batch)")
