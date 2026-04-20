# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Repo-local AirfRANS loader.

This keeps the repo independent from the official AirfRANS training code by
reading the benchmark manifest plus the raw `.vtu` / `.vtp` files directly.

The returned dataset follows the same broad contract as TandemFoilSet:

- `x`: per-point engineered inputs
- `y`: per-point regression targets
- `is_surface`: surface mask for surface-aware losses
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    from data.prepare import pad_collate  # noqa: F401 re-export
    from data.split_utils import first_existing
    from data.vtk_xml import read_vtk_xml
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.prepare import pad_collate  # noqa: F401 re-export
    from data.split_utils import first_existing
    from data.vtk_xml import read_vtk_xml

DEFAULT_MANIFEST = Path(__file__).with_name("split_manifest_airfrans.json")
DEFAULT_TASK = "full"
X_DIM = 8

_INTERNAL_ARRAYS = ["U", "p", "implicit_distance", "nut"]
_SURFACE_ARRAYS = ["Normals", "U", "p", "nut"]
_FREESTREAM_ARRAYS = ["U"]


@dataclass(frozen=True)
class AirfRANSCase:
    case_id: str
    x: torch.Tensor
    y: torch.Tensor
    is_surface: torch.Tensor
    metadata: dict[str, float | str]


def _resolve_root(manifest: dict, override_root: str | Path | None = None) -> Path:
    if override_root is not None:
        root = Path(override_root)
        if not root.exists():
            raise FileNotFoundError(f"AirfRANS root does not exist: {root}")
        return root

    candidates = manifest.get("data_root_candidates", [])
    root = first_existing(candidates)
    if root is None:
        raise FileNotFoundError(
            f"Could not resolve AirfRANS dataset root from candidates: {candidates}"
        )
    return root


def _resolve_case_dir(root: Path, case_id: str) -> Path:
    case_dir = root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"AirfRANS case directory not found: {case_dir}")
    return case_dir


def _column(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    return value[:, None] if value.ndim == 1 else value


def _nearest_neighbor_values(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    """Assign each query point the value of its nearest reference point."""

    if len(query_points) == 0:
        return np.zeros((0, reference_values.shape[1]), dtype=np.float32)
    if len(reference_points) == 0:
        return np.zeros((len(query_points), reference_values.shape[1]), dtype=np.float32)

    diff = query_points[:, None, :] - reference_points[None, :, :]
    nearest = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    return np.asarray(reference_values[nearest], dtype=np.float32)


def _parse_case_metadata(case_id: str, freestream_velocity: np.ndarray) -> dict[str, float | str]:
    speed = float(np.linalg.norm(freestream_velocity))
    aoa_rad = float(math.atan2(float(freestream_velocity[1]), float(freestream_velocity[0])))
    metadata: dict[str, float | str] = {
        "case_id": case_id,
        "u_inf": speed,
        "alpha_deg": aoa_rad * 180.0 / math.pi,
    }
    parts = case_id.split("_")
    if len(parts) >= 4:
        try:
            metadata["case_u_inf_token"] = float(parts[2])
            metadata["case_alpha_token"] = float(parts[3])
        except ValueError:
            pass
    return metadata


def load_airfrans_case(
    root: str | Path,
    case_id: str,
    include_nut: bool = False,
    surface_sdf_tol: float = 1e-7,
    surface_u_tol: float = 1e-9,
) -> AirfRANSCase:
    """Load one AirfRANS case into repo-local tensors."""

    case_dir = _resolve_case_dir(Path(root), case_id)
    internal = read_vtk_xml(
        case_dir / f"{case_id}_internal.vtu",
        point_arrays=_INTERNAL_ARRAYS,
    )
    aerofoil = read_vtk_xml(
        case_dir / f"{case_id}_aerofoil.vtp",
        point_arrays=_SURFACE_ARRAYS,
    )
    freestream = read_vtk_xml(
        case_dir / f"{case_id}_freestream.vtp",
        point_arrays=_FREESTREAM_ARRAYS,
    )

    internal_pos = np.asarray(internal.points[:, :2], dtype=np.float32)
    velocity = np.asarray(internal.point_data["U"][:, :2], dtype=np.float32)
    pressure = _column(internal.point_data["p"]).astype(np.float32)
    implicit_distance = _column(internal.point_data["implicit_distance"]).astype(np.float32)
    sdf = -implicit_distance

    wall_from_distance = np.isclose(np.abs(implicit_distance[:, 0]), 0.0, atol=surface_sdf_tol)
    wall_from_velocity = np.linalg.norm(velocity, axis=1) <= surface_u_tol
    is_surface = wall_from_distance | wall_from_velocity

    surface_pos = np.asarray(aerofoil.points[:, :2], dtype=np.float32)
    surface_normals = np.asarray(aerofoil.point_data["Normals"][:, :2], dtype=np.float32)
    normals = np.zeros_like(internal_pos, dtype=np.float32)
    if np.any(is_surface):
        normals[is_surface] = _nearest_neighbor_values(
            internal_pos[is_surface],
            surface_pos,
            -surface_normals,
        )

    freestream_velocity = np.asarray(freestream.point_data["U"], dtype=np.float32)
    if freestream_velocity.ndim == 1:
        freestream_velocity = freestream_velocity[None, :]
    freestream_xy = np.asarray(freestream_velocity[:, :2].mean(axis=0), dtype=np.float32)

    x = np.concatenate(
        [
            internal_pos,
            np.broadcast_to(freestream_xy, (len(internal_pos), 2)).astype(np.float32),
            sdf.astype(np.float32),
            normals.astype(np.float32),
            is_surface.astype(np.float32)[:, None],
        ],
        axis=1,
    )
    y_parts = [velocity, pressure]
    if include_nut:
        nut = internal.point_data.get("nut")
        if nut is None:
            raise KeyError(f"AirfRANS case {case_id} does not expose point-data field 'nut'")
        y_parts.append(_column(nut).astype(np.float32))
    y = np.concatenate(y_parts, axis=1)

    return AirfRANSCase(
        case_id=case_id,
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        is_surface=torch.from_numpy(is_surface.astype(bool)),
        metadata=_parse_case_metadata(case_id, freestream_xy),
    )


class AirfRANSDataset(Dataset):
    """Manifest-backed lazy AirfRANS dataset."""

    def __init__(
        self,
        root: str | Path,
        case_ids: list[str],
        include_nut: bool = False,
        cache_size: int = 0,
    ):
        self.root = Path(root)
        self.case_ids = list(case_ids)
        self.include_nut = include_nut
        self.cache_size = cache_size
        self._cache: dict[int, AirfRANSCase] = {}

        if cache_size == 0:
            for idx in range(len(self.case_ids)):
                self._cache[idx] = self._load_case(idx)

    def _load_case(self, idx: int) -> AirfRANSCase:
        return load_airfrans_case(self.root, self.case_ids[idx], include_nut=self.include_nut)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cache.get(idx)
        if cached is None:
            cached = self._load_case(idx)
            if self.cache_size > 0 and len(self._cache) < self.cache_size:
                self._cache[idx] = cached
        return cached.x, cached.y, cached.is_surface

    def metadata(self, idx: int) -> dict[str, float | str]:
        cached = self._cache.get(idx)
        if cached is None:
            cached = self._load_case(idx)
        return dict(cached.metadata)


def _load_manifest(manifest_path: str | Path) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


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


def load_data(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    task: str = DEFAULT_TASK,
    root: str | Path | None = None,
    debug: bool = False,
    include_nut: bool = False,
    cache_size: int = -1,
) -> tuple[Subset, dict[str, Subset], dict[str, torch.Tensor], torch.Tensor]:
    """Load one AirfRANS task into the train/val contract used by the repo."""

    manifest = _load_manifest(manifest_path)
    dataset_root = _resolve_root(manifest, override_root=root)

    train_name = f"{task}_train"
    val_name = f"{task}_val"
    if train_name not in manifest["splits"] or val_name not in manifest["splits"]:
        raise KeyError(f"Task '{task}' not found in AirfRANS manifest")

    all_case_ids = manifest["splits"][train_name] + manifest["splits"][val_name]
    ds = AirfRANSDataset(
        root=dataset_root,
        case_ids=all_case_ids,
        include_nut=include_nut,
        cache_size=-1 if debug else cache_size,
    )
    train_indices = list(range(len(manifest["splits"][train_name])))
    val_indices = list(range(len(train_indices), len(all_case_ids)))
    train_ds = Subset(ds, train_indices)
    val_splits = {val_name: Subset(ds, val_indices)}

    if debug:
        train_ds = Subset(ds, train_indices[: min(4, len(train_indices))])
        val_splits = {val_name: Subset(ds, val_indices[: min(2, len(val_indices))])}

    stats = _stream_stats(train_ds)
    sample_weights = torch.ones(len(train_ds), dtype=torch.float32)
    return train_ds, val_splits, stats, sample_weights
