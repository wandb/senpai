# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Repo-local DrivAerML loaders backed only by the packaged PVC artifacts.

The processed dataset already provides `.npy` arrays plus `manifest.csv` and
`normalizers.json`. This module keeps that contract local so the training repo
does not depend on `milieu_cfd`, AB-UPT, or PhysicsNeMo at runtime.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    from data.split_utils import expand_pvc_candidates, first_existing
    from tandemfoil.data.prepare import pad_collate  # noqa: F401 re-export
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data.split_utils import expand_pvc_candidates, first_existing
    from tandemfoil.data.prepare import pad_collate  # noqa: F401 re-export

DEFAULT_MANIFEST = Path(__file__).with_name("split_manifest_drivaerml.json")
SURFACE_X_DIM = 7  # xyz(3) + normals(3) + area(1)
SURFACE_Y_DIM = 1  # cp
VOLUME_X_DIM = 4   # xyz(3) + sdf(1)
VOLUME_Y_DIM = 4   # velocity(3) + pressure(1)


@dataclass(frozen=True)
class DrivAerMLCase:
    case_id: str
    surface_x: torch.Tensor
    surface_y: torch.Tensor
    surface_is_surface: torch.Tensor
    volume_x: torch.Tensor | None
    volume_y: torch.Tensor | None
    metadata: dict[str, str | int]


def _load_manifest(manifest_path: str | Path) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


def _resolve_case_root(manifest: dict, override_root: str | Path | None = None) -> Path:
    if override_root is not None:
        root = Path(override_root)
        if not root.exists():
            raise FileNotFoundError(f"DrivAerML root does not exist: {root}")
        return root

    candidates = list(manifest.get("case_root_candidates", []))
    case_root = manifest.get("case_root")
    if case_root:
        candidates.append(case_root)
        if isinstance(case_root, str) and case_root.startswith("/mnt/pvc/"):
            candidates.append(case_root.replace("/mnt/pvc/", "/mnt/new-pvc/", 1))
        if isinstance(case_root, str) and case_root.startswith("/mnt/new-pvc/"):
            candidates.append(case_root.replace("/mnt/new-pvc/", "/mnt/pvc/", 1))
    candidates = expand_pvc_candidates(candidates)

    if candidates:
        root = first_existing(candidates)
        if root is not None:
            return root

    fallback = Path(case_root)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not resolve DrivAerML root from candidates: {manifest.get('case_root_candidates', [])}"
    )


def _case_dir(root: Path, case_id: str) -> Path:
    path = root / case_id
    if not path.exists():
        raise FileNotFoundError(f"DrivAerML case directory not found: {path}")
    return path


def _candidate_artifact_paths(path: Path) -> list[Path]:
    texts: list[str] = []
    seen: set[str] = set()

    def add(text: str | Path | None) -> None:
        if not text:
            return
        raw = str(text)
        if raw in seen:
            return
        seen.add(raw)
        texts.append(raw)

    add(path)
    add(path.resolve(strict=False))
    if str(path).startswith("/mnt/pvc/"):
        add(str(path).replace("/mnt/pvc/", "/mnt/new-pvc/", 1))
    if str(path).startswith("/mnt/new-pvc/"):
        add(str(path).replace("/mnt/new-pvc/", "/mnt/pvc/", 1))

    if path.is_symlink():
        raw_target = os.readlink(path)
        add(raw_target)
        target_path = Path(raw_target)
        if not target_path.is_absolute():
            target_path = (path.parent / target_path).resolve(strict=False)
        add(target_path)
        target_text = str(target_path)
        if target_text.startswith("/rsyncd-munged/"):
            stripped = target_text.removeprefix("/rsyncd-munged/").lstrip("/")
            add("/" + stripped)
            if stripped.startswith("mnt/pvc/"):
                add("/" + stripped.replace("mnt/pvc/", "mnt/new-pvc/", 1))
            if stripped.startswith("mnt/new-pvc/"):
                add("/" + stripped.replace("mnt/new-pvc/", "mnt/pvc/", 1))
        if target_text.startswith("/mnt/pvc/"):
            add(target_text.replace("/mnt/pvc/", "/mnt/new-pvc/", 1))
        if target_text.startswith("/mnt/new-pvc/"):
            add(target_text.replace("/mnt/new-pvc/", "/mnt/pvc/", 1))

    ordered: list[Path] = []
    for candidate in expand_pvc_candidates(texts):
        resolved = Path(candidate)
        if resolved not in ordered:
            ordered.append(resolved)
    return ordered


@functools.lru_cache(maxsize=16384)
def _resolve_artifact_path(path: Path) -> Path:
    for candidate in _candidate_artifact_paths(path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve DrivAerML artifact "
        f"{path}; checked {[str(candidate) for candidate in _candidate_artifact_paths(path)]}"
    )


def _artifact_exists(path: Path) -> bool:
    return any(candidate.exists() for candidate in _candidate_artifact_paths(path))


def _load_npy(path: Path) -> np.ndarray:
    return np.asarray(np.load(_resolve_artifact_path(path)), dtype=np.float32)


def _load_npy_rows(path: Path, rows: np.ndarray | None = None) -> np.ndarray:
    if rows is None:
        return _load_npy(path)
    mapped = np.load(_resolve_artifact_path(path), mmap_mode="r")
    return np.asarray(mapped[rows], dtype=np.float32)


def _npy_row_count(path: Path) -> int:
    arr = np.load(_resolve_artifact_path(path), mmap_mode="r")
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def _column(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    return arr[:, None] if arr.ndim == 1 else arr


def _surface_arrays(case_dir: Path, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    xyz = _load_npy_rows(case_dir / "surface_xyz.npy", rows)
    normals = _load_npy_rows(case_dir / "surface_normals.npy", rows)
    area = _column(_load_npy_rows(case_dir / "surface_area.npy", rows))
    cp = _column(_load_npy_rows(case_dir / "surface_cp.npy", rows))
    x = np.concatenate([xyz, normals, area], axis=1)
    return x, cp


def _volume_arrays(
    case_dir: Path,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    xyz_path = case_dir / "volume_xyz.npy"
    sdf_path = case_dir / "volume_sdf.npy"
    velocity_path = case_dir / "volume_velocity.npy"
    pressure_path = case_dir / "volume_pressure.npy"
    if not all(_artifact_exists(path) for path in [xyz_path, sdf_path, velocity_path, pressure_path]):
        return None, None

    x = np.concatenate(
        [
            _load_npy_rows(xyz_path, rows),
            _column(_load_npy_rows(sdf_path, rows)),
        ],
        axis=1,
    )
    y = np.concatenate(
        [
            _load_npy_rows(velocity_path, rows),
            _column(_load_npy_rows(pressure_path, rows)),
        ],
        axis=1,
    )
    return x, y


def load_drivaerml_case(
    root: str | Path,
    case_id: str,
    *,
    surface_rows: np.ndarray | None = None,
    volume_rows: np.ndarray | None = None,
) -> DrivAerMLCase:
    case_dir = _case_dir(Path(root), case_id)
    surface_x, surface_y = _surface_arrays(case_dir, rows=surface_rows)
    volume_x, volume_y = _volume_arrays(case_dir, rows=volume_rows)
    return DrivAerMLCase(
        case_id=case_id,
        surface_x=torch.from_numpy(surface_x),
        surface_y=torch.from_numpy(surface_y),
        surface_is_surface=torch.ones(surface_x.shape[0], dtype=torch.bool),
        volume_x=None if volume_x is None else torch.from_numpy(volume_x),
        volume_y=None if volume_y is None else torch.from_numpy(volume_y),
        metadata={
            "case_id": case_id,
            "n_surface": int(surface_x.shape[0]),
            "n_volume": int(0 if volume_x is None else volume_x.shape[0]),
        },
    )


def load_drivaerml_case_point_counts(root: str | Path, case_id: str) -> dict[str, int]:
    case_dir = _case_dir(Path(root), case_id)
    surface_count = _npy_row_count(case_dir / "surface_xyz.npy")

    volume_xyz_path = case_dir / "volume_xyz.npy"
    volume_count = 0
    if _artifact_exists(volume_xyz_path):
        volume_count = _npy_row_count(volume_xyz_path)

    return {
        "case_id": case_id,
        "n_surface": surface_count,
        "n_volume": volume_count,
    }


class DrivAerMLCaseStore:
    """Thin manifest-backed store for the processed DrivAerML PVC layout."""

    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST, root: str | Path | None = None):
        self.manifest_path = Path(manifest_path)
        self.manifest = _load_manifest(self.manifest_path)
        self.root = _resolve_case_root(self.manifest, override_root=root)
        self.normalizers_path = self.root / "normalizers.json"
        self._point_count_cache: dict[str, dict[str, int]] = {}

    def case_ids(self, split: str, domain: str = "surface") -> list[str]:
        if domain == "surface":
            return list(self.manifest["surface_splits"][split])
        if domain == "volume":
            return list(self.manifest["volume_splits"][split])
        raise ValueError(f"Unknown DrivAerML domain: {domain}")

    def load_case(
        self,
        case_id: str,
        *,
        surface_rows: np.ndarray | None = None,
        volume_rows: np.ndarray | None = None,
    ) -> DrivAerMLCase:
        return load_drivaerml_case(
            self.root,
            case_id,
            surface_rows=surface_rows,
            volume_rows=volume_rows,
        )

    def case_point_counts(self, case_id: str) -> dict[str, int]:
        cached = self._point_count_cache.get(case_id)
        if cached is None:
            cached = load_drivaerml_case_point_counts(self.root, case_id)
            self._point_count_cache[case_id] = cached
        return dict(cached)

    def load_normalizers(self) -> dict:
        with self.normalizers_path.open() as f:
            return json.load(f)


class DrivAerMLSurfaceDataset(Dataset):
    """Surface-pressure benchmark dataset built from the packaged PVC arrays."""

    def __init__(self, store: DrivAerMLCaseStore, case_ids: list[str], cache_size: int = 0):
        self.store = store
        self.case_ids = list(case_ids)
        self.cache_size = cache_size
        self._cache: dict[int, DrivAerMLCase] = {}
        if cache_size == 0:
            for idx in range(len(self.case_ids)):
                self._cache[idx] = self.store.load_case(self.case_ids[idx])

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case = self._cache.get(idx)
        if case is None:
            case = self.store.load_case(self.case_ids[idx])
            if self.cache_size > 0 and len(self._cache) < self.cache_size:
                self._cache[idx] = case
        return case.surface_x, case.surface_y, case.surface_is_surface

    def metadata(self, idx: int) -> dict[str, str | int]:
        case = self._cache.get(idx)
        if case is None:
            case = self.store.load_case(self.case_ids[idx])
        return dict(case.metadata)


class DrivAerMLVolumeDataset(Dataset):
    """Volume CFD benchmark dataset for the processed DrivAerML subset."""

    def __init__(self, store: DrivAerMLCaseStore, case_ids: list[str], cache_size: int = 0):
        self.store = store
        self.case_ids = list(case_ids)
        self.cache_size = cache_size
        self._cache: dict[int, DrivAerMLCase] = {}
        if cache_size == 0:
            for idx in range(len(self.case_ids)):
                self._cache[idx] = self.store.load_case(self.case_ids[idx])

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case = self._cache.get(idx)
        if case is None:
            case = self.store.load_case(self.case_ids[idx])
            if self.cache_size > 0 and len(self._cache) < self.cache_size:
                self._cache[idx] = case
        if case.volume_x is None or case.volume_y is None:
            raise ValueError(f"DrivAerML case {case.case_id} does not have processed volume arrays")
        return case.volume_x, case.volume_y, torch.zeros(case.volume_x.shape[0], dtype=torch.bool)


class DrivAerMLCaseDataset(Dataset):
    """Case-level dataset exposing both surface and volume tensors together."""

    def __init__(self, store: DrivAerMLCaseStore, case_ids: list[str]):
        self.store = store
        self.case_ids = list(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> DrivAerMLCase:
        return self.store.load_case(self.case_ids[idx])


def _stream_stats(ds: Dataset) -> dict[str, torch.Tensor]:
    x_sum = x_sq_sum = y_sum = y_sq_sum = None
    total_points = 0
    for x, y, _ in ds:
        x_np = np.asarray(x, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)
        if x_sum is None:
            x_sum = x_np.sum(axis=0)
            x_sq_sum = np.square(x_np).sum(axis=0)
            y_sum = y_np.sum(axis=0)
            y_sq_sum = np.square(y_np).sum(axis=0)
        else:
            x_sum += x_np.sum(axis=0)
            x_sq_sum += np.square(x_np).sum(axis=0)
            y_sum += y_np.sum(axis=0)
            y_sq_sum += np.square(y_np).sum(axis=0)
        total_points += x_np.shape[0]

    if total_points == 0:
        raise ValueError("Cannot compute stats on an empty dataset")

    x_mean = x_sum / total_points
    y_mean = y_sum / total_points
    x_var = np.maximum(x_sq_sum / total_points - np.square(x_mean), 1e-12)
    y_var = np.maximum(y_sq_sum / total_points - np.square(y_mean), 1e-12)
    return {
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_std": torch.tensor(np.sqrt(x_var), dtype=torch.float32),
        "y_mean": torch.tensor(y_mean, dtype=torch.float32),
        "y_std": torch.tensor(np.sqrt(y_var), dtype=torch.float32),
    }


def surface_stats_from_normalizers(store: DrivAerMLCaseStore) -> dict[str, torch.Tensor] | None:
    """Read target stats from the packaged normalizers file when available."""

    if not store.normalizers_path.exists():
        return None
    raw = store.load_normalizers()
    cp_stats = raw.get("surface_cp")
    geometry = raw.get("geometry")
    if not isinstance(cp_stats, dict) or not isinstance(geometry, dict):
        return None

    # The trainer expects x/y mean+std vectors. Only y can be recovered exactly
    # from normalizers.json; x still needs a train-split scan.
    return {
        "y_mean": torch.tensor([cp_stats["mean"]], dtype=torch.float32),
        "y_std": torch.tensor([cp_stats["std"]], dtype=torch.float32),
        "geometry_center": torch.tensor(geometry.get("center", [0.0, 0.0, 0.0]), dtype=torch.float32),
        "geometry_scale": torch.tensor([geometry.get("scale", 1.0)], dtype=torch.float32),
    }


def load_surface_data(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    root: str | Path | None = None,
    debug: bool = False,
    cache_size: int = -1,
) -> tuple[Subset, dict[str, Subset], dict[str, torch.Tensor], torch.Tensor]:
    """Load the public DrivAerML surface benchmark into the repo train/val shape."""

    store = DrivAerMLCaseStore(manifest_path=manifest_path, root=root)
    train_ids = store.case_ids("train", domain="surface")
    val_ids = store.case_ids("val", domain="surface")
    all_ids = train_ids + val_ids
    ds = DrivAerMLSurfaceDataset(store, all_ids, cache_size=-1 if debug else cache_size)

    train_indices = list(range(len(train_ids)))
    val_indices = list(range(len(train_ids), len(all_ids)))
    train_ds = Subset(ds, train_indices)
    val_splits = {"val_surface": Subset(ds, val_indices)}
    if debug:
        train_ds = Subset(ds, train_indices[: min(4, len(train_indices))])
        val_splits = {"val_surface": Subset(ds, val_indices[: min(2, len(val_indices))])}

    stats = _stream_stats(train_ds)
    normalizer_stats = surface_stats_from_normalizers(store)
    if normalizer_stats is not None:
        stats["y_mean"] = normalizer_stats["y_mean"]
        stats["y_std"] = normalizer_stats["y_std"]
        stats["geometry_center"] = normalizer_stats["geometry_center"]
        stats["geometry_scale"] = normalizer_stats["geometry_scale"]

    sample_weights = torch.ones(len(train_ds), dtype=torch.float32)
    return train_ds, val_splits, stats, sample_weights
