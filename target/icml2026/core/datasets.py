# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, Subset

from airfrans.data import prepare_airfrans
from core.contracts import (
    ABUPTBatch,
    CaseSample,
    DatasetBundle,
    DatasetSpec,
    GroupedBatch,
    TargetTransformStats,
)
from core.features import augment_case_sample, stable_hash32
from drivaerml.data import prepare_drivaerml
from tandemfoil.data import prepare_multi
from tandemfoil_paper.data import prepare_multi as paper_prepare_multi
from tandemfoil_paper.data import split_paper_experiment4


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TANDEM_MANIFEST = ROOT_DIR / "tandemfoil/data/split_manifest_tandemfoilset_v2.json"
DEFAULT_TANDEM_STATS = ROOT_DIR / "tandemfoil/data/split_stats.json"
DEFAULT_TANDEM_PAPER_MANIFEST = ROOT_DIR / "tandemfoil_paper/data/split_manifest_tandemfoil_paper_experiment4.json"
DEFAULT_TANDEM_PAPER_STATS = ROOT_DIR / "tandemfoil_paper/data/split_stats_tandemfoil_paper_experiment4.json"
DEFAULT_AIRFRANS_MANIFEST = ROOT_DIR / "airfrans/data/split_manifest_airfrans.json"
DEFAULT_DRIVAERML_MANIFEST = ROOT_DIR / "drivaerml/data/split_manifest_drivaerml.json"
EXPECTED_DRIVAERML_SURFACE_SPLIT_COUNTS = {"train": 400, "val": 34, "test": 50}
EXPECTED_DRIVAERML_EXCLUDED_CASE_COUNT = 0
REQUIRED_DRIVAERML_CASE_IDS = frozenset(
    {
        "run_44",
        "run_133",
        "run_158",
        "run_184",
        "run_203",
        "run_226",
        "run_249",
        "run_310",
        "run_416",
        "run_484",
    }
)
RUNTIME_CACHE_CASES = 16


@dataclass(frozen=True)
class DrivAerMLPointView:
    case_id: str
    view_index: int
    view_count: int
    sampling_mode: str


def _read_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _stats_from_json(stats_path: str | Path) -> TargetTransformStats:
    raw = _read_json(stats_path)
    return TargetTransformStats(
        x_mean=torch.tensor(raw["x_mean"], dtype=torch.float32) if "x_mean" in raw else None,
        x_std=torch.tensor(raw["x_std"], dtype=torch.float32) if "x_std" in raw else None,
        y_mean=torch.tensor(raw["y_mean"], dtype=torch.float32),
        y_std=torch.tensor(raw["y_std"], dtype=torch.float32),
    )


def _validate_drivaerml_manifest(manifest: dict, manifest_path: str | Path) -> None:
    surface_splits = manifest.get("surface_splits")
    if not isinstance(surface_splits, dict):
        raise ValueError(f"DrivAerML manifest {manifest_path} is missing surface_splits")

    missing_splits = sorted(set(EXPECTED_DRIVAERML_SURFACE_SPLIT_COUNTS) - set(surface_splits))
    if missing_splits:
        raise ValueError(f"DrivAerML manifest {manifest_path} is missing surface splits: {missing_splits}")

    actual_counts = {
        split: len(surface_splits[split])
        for split in EXPECTED_DRIVAERML_SURFACE_SPLIT_COUNTS
    }
    if actual_counts != EXPECTED_DRIVAERML_SURFACE_SPLIT_COUNTS:
        raise ValueError(
            "DrivAerML manifest does not match the repaired public benchmark split: "
            f"{actual_counts} vs {EXPECTED_DRIVAERML_SURFACE_SPLIT_COUNTS} ({manifest_path})"
        )

    split_sets = {
        split: set(surface_splits[split])
        for split in EXPECTED_DRIVAERML_SURFACE_SPLIT_COUNTS
    }
    surface_case_ids = set().union(*split_sets.values())
    if len(surface_case_ids) != sum(actual_counts.values()):
        raise ValueError(f"DrivAerML manifest {manifest_path} has overlapping surface splits")

    excluded_case_count = int(
        manifest.get("excluded_case_count", len(manifest.get("excluded_case_ids", [])))
    )
    if excluded_case_count != EXPECTED_DRIVAERML_EXCLUDED_CASE_COUNT:
        raise ValueError(
            "DrivAerML manifest still excludes repaired public cases: "
            f"{excluded_case_count} excluded in {manifest_path}"
        )

    missing_required = sorted(REQUIRED_DRIVAERML_CASE_IDS - surface_case_ids)
    if missing_required:
        raise ValueError(
            f"DrivAerML manifest {manifest_path} is missing restored public cases: {missing_required}"
        )

    volume_splits = manifest.get("volume_splits", {})
    if isinstance(volume_splits, dict):
        for split_name, case_ids in volume_splits.items():
            extra_case_ids = sorted(set(case_ids) - split_sets.get(split_name, set()))
            if extra_case_ids:
                raise ValueError(
                    "DrivAerML volume split must stay aligned with the same surface split: "
                    f"{split_name} has extras {extra_case_ids} in {manifest_path}"
                )


class TandemFoilCaseDataset(Dataset):
    def __init__(
        self,
        split_indices: list[int],
        manifest_path: str | Path = DEFAULT_TANDEM_MANIFEST,
        *,
        debug: bool = False,
    ):
        manifest = _read_json(manifest_path)
        cache_size = -1 if debug else RUNTIME_CACHE_CASES
        self.base = prepare_multi.MultiFieldDataset(
            prepare_multi._resolve_pickle_paths(manifest),
            cache_size=cache_size,
        )
        self.indices = list(split_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> CaseSample:
        base_idx = self.indices[idx]
        x, y, is_surface = self.base[base_idx]
        surface_x = x[is_surface]
        volume_x = x[~is_surface]
        surface_y = y[is_surface]
        volume_y = y[~is_surface]
        sample = CaseSample(
            case_id=f"tandem_{base_idx}",
            dataset_name="tandemfoilset",
            space_dim=2,
            surface_x=surface_x,
            surface_y=surface_y,
            volume_x=volume_x,
            volume_y=volume_y,
            metadata={"base_idx": base_idx},
        )
        return sample


class PaperTandemFoilCaseDataset(Dataset):
    def __init__(
        self,
        split_indices: list[int],
        pickle_files: list[str],
        *,
        root_candidates: list[str] | None = None,
        debug: bool = False,
        task_name: str,
        enable_fourier: bool = False,
        enable_wake_deficit: bool = False,
        enable_wake_angle: bool = False,
    ):
        task_spec = split_paper_experiment4.TASK_SPECS[task_name]
        candidates = list(root_candidates or split_paper_experiment4.DEFAULT_ROOT_CANDIDATES)
        self.pickle_paths = split_paper_experiment4._resolve_task_pickle_paths(task_spec, candidates)
        cache_size = -1 if debug else RUNTIME_CACHE_CASES
        self.base = paper_prepare_multi.MultiFieldDataset(self.pickle_paths, cache_size=cache_size)
        self.indices = list(split_indices)
        self.task_name = task_name
        self.augment = dict(
            enable_fourier=enable_fourier,
            enable_cp_panel=False,
            enable_wake_deficit=enable_wake_deficit,
            enable_wake_angle=enable_wake_angle,
        )
        expected_files = [path.name for path in self.pickle_paths]
        if list(pickle_files) != expected_files:
            raise ValueError(
                f"Manifest pickle files for {task_name} do not match resolved files: "
                f"{pickle_files} vs {expected_files}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> CaseSample:
        base_idx = self.indices[idx]
        x, y, is_surface = self.base[base_idx]
        surface_x = x[is_surface]
        volume_x = x[~is_surface]
        surface_y = y[is_surface]
        volume_y = y[~is_surface]
        sample = CaseSample(
            case_id=f"{self.task_name}_{base_idx}",
            dataset_name="tandemfoilset_paper",
            space_dim=2,
            surface_x=surface_x,
            surface_y=surface_y,
            volume_x=volume_x,
            volume_y=volume_y,
            metadata={"base_idx": base_idx, "task": self.task_name},
        )
        return augment_case_sample(sample, **self.augment)


class AirfRANSCaseDataset(Dataset):
    def __init__(
        self,
        case_ids: list[str],
        *,
        manifest_path: str | Path = DEFAULT_AIRFRANS_MANIFEST,
        root: str | Path | None = None,
        debug: bool = False,
        enable_fourier: bool = False,
        enable_cp_panel: bool = False,
        include_nut: bool = True,
    ):
        manifest = _read_json(manifest_path)
        dataset_root = prepare_airfrans._resolve_root(manifest, override_root=root)
        cache_size = -1 if debug else RUNTIME_CACHE_CASES
        self.base = prepare_airfrans.AirfRANSDataset(
            root=dataset_root,
            case_ids=case_ids,
            include_nut=include_nut,
            cache_size=cache_size,
        )
        self.augment = dict(
            enable_fourier=enable_fourier,
            enable_cp_panel=enable_cp_panel,
            enable_wake_deficit=False,
            enable_wake_angle=False,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> CaseSample:
        x, y, is_surface = self.base[idx]
        meta = self.base.metadata(idx)
        sample = CaseSample(
            case_id=str(meta["case_id"]),
            dataset_name="airfrans",
            space_dim=2,
            surface_x=x[is_surface],
            surface_y=y[is_surface],
            volume_x=x[~is_surface],
            volume_y=y[~is_surface],
            metadata=meta,
        )
        return augment_case_sample(sample, **self.augment)


class DrivAerMLCaseDataset(Dataset):
    def __init__(
        self,
        case_ids: list[str],
        *,
        manifest_path: str | Path = DEFAULT_DRIVAERML_MANIFEST,
        root: str | Path | None = None,
        enable_fourier: bool = False,
        surface_only: bool = True,
        max_surface_points: int = 0,
        max_volume_points: int = 0,
        sampling_mode: str = "full",
    ):
        self.store = prepare_drivaerml.DrivAerMLCaseStore(manifest_path=manifest_path, root=root)
        self.case_ids = list(case_ids)
        self.surface_only = surface_only
        self.max_surface_points = max_surface_points
        self.max_volume_points = max_volume_points
        self.sampling_mode = sampling_mode
        self.augment = dict(
            enable_fourier=enable_fourier,
            enable_cp_panel=False,
            enable_wake_deficit=False,
            enable_wake_angle=False,
        )
        self.views = self._build_views()

    def __len__(self) -> int:
        return len(self.views)

    @staticmethod
    def _view_count(total: int, points_per_view: int) -> int:
        if points_per_view <= 0 or total <= points_per_view:
            return 1
        return max(1, math.ceil(total / points_per_view))

    def _build_views(self) -> list[DrivAerMLPointView]:
        views: list[DrivAerMLPointView] = []
        for case_id in self.case_ids:
            counts = self.store.case_point_counts(case_id)
            surface_views = self._view_count(counts["n_surface"], self.max_surface_points)
            view_count = surface_views
            if not self.surface_only:
                volume_views = self._view_count(counts["n_volume"], self.max_volume_points)
                view_count = max(view_count, volume_views)
            for view_index in range(view_count):
                views.append(
                    DrivAerMLPointView(
                        case_id=case_id,
                        view_index=view_index,
                        view_count=view_count,
                        sampling_mode=self.sampling_mode,
                    )
                )
        return views

    def _random_subset_indices(self, total: int, count: int) -> torch.Tensor | None:
        if count <= 0 or count >= total:
            return None
        return torch.randint(total, (count,), dtype=torch.long).sort().values

    def _strided_partition_indices(self, total: int, count: int, view: DrivAerMLPointView) -> torch.Tensor | None:
        if count <= 0 or count >= total:
            return None
        return torch.arange(view.view_index, total, view.view_count, dtype=torch.long)

    def _surface_indices(self, total: int, view: DrivAerMLPointView) -> torch.Tensor | None:
        if self.max_surface_points <= 0 or total <= self.max_surface_points:
            return None
        if view.sampling_mode == "train_random":
            return self._random_subset_indices(total, self.max_surface_points)
        if view.sampling_mode == "eval_chunk":
            return self._strided_partition_indices(total, self.max_surface_points, view)
        return None

    def _volume_indices(self, total: int, view: DrivAerMLPointView) -> torch.Tensor | None:
        if self.max_volume_points <= 0 or total <= self.max_volume_points:
            return None
        if view.sampling_mode == "train_random":
            return self._random_subset_indices(total, self.max_volume_points)
        if view.sampling_mode == "eval_chunk":
            return self._strided_partition_indices(total, self.max_volume_points, view)
        return None

    def __getitem__(self, idx: int) -> CaseSample:
        view = self.views[idx]
        counts = self.store.case_point_counts(view.case_id)
        surface_idx = self._surface_indices(counts["n_surface"], view)
        volume_idx = None if self.surface_only else self._volume_indices(counts["n_volume"], view)
        case = self.store.load_case(
            view.case_id,
            surface_rows=None if surface_idx is None else surface_idx.numpy(),
            volume_rows=None if volume_idx is None else volume_idx.numpy(),
        )
        metadata = dict(case.metadata)

        surface_x = case.surface_x
        surface_y = case.surface_y
        metadata["n_surface_full"] = int(counts["n_surface"])
        metadata["n_surface_loaded"] = int(surface_x.shape[0])
        metadata["surface_view_index"] = int(view.view_index)
        metadata["surface_view_count"] = int(view.view_count)
        metadata["surface_sampling_mode"] = view.sampling_mode

        volume_x = None if self.surface_only else case.volume_x
        volume_y = None if self.surface_only else case.volume_y
        if volume_x is not None and volume_y is not None:
            metadata["n_volume_full"] = int(counts["n_volume"])
            metadata["n_volume_loaded"] = int(volume_x.shape[0])
            metadata["volume_view_index"] = int(view.view_index)
            metadata["volume_view_count"] = int(view.view_count)
            metadata["volume_sampling_mode"] = view.sampling_mode

        sample = CaseSample(
            case_id=case.case_id,
            dataset_name="drivaerml",
            space_dim=3,
            surface_x=surface_x,
            surface_y=surface_y,
            volume_x=volume_x,
            volume_y=volume_y,
            metadata=metadata,
        )
        return augment_case_sample(sample, **self.augment)


def _pad_group(
    values: list[torch.Tensor | None],
    *,
    feature_dim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    present = [value for value in values if value is not None]
    if not present:
        return None, None
    max_n = max(value.shape[0] for value in present)
    feat_dim = feature_dim if feature_dim is not None else present[0].shape[1]
    padded = torch.zeros(len(values), max_n, feat_dim, dtype=present[0].dtype)
    mask = torch.zeros(len(values), max_n, dtype=torch.bool)
    for i, value in enumerate(values):
        if value is None:
            continue
        padded[i, : value.shape[0], : value.shape[1]] = value
        mask[i, : value.shape[0]] = True
    return padded, mask


def collate_grouped(samples: list[CaseSample]) -> GroupedBatch:
    if not samples:
        raise ValueError("collate_grouped received an empty batch")
    surface_x, surface_mask = _pad_group([sample.surface_x for sample in samples])
    surface_y, _ = _pad_group([sample.surface_y for sample in samples]) if samples[0].surface_y is not None else (None, None)
    volume_x, volume_mask = _pad_group([sample.volume_x for sample in samples])
    volume_y, _ = _pad_group([sample.volume_y for sample in samples]) if any(sample.volume_y is not None for sample in samples) else (None, None)
    assert surface_x is not None and surface_mask is not None
    return GroupedBatch(
        case_ids=[sample.case_id for sample in samples],
        dataset_name=samples[0].dataset_name,
        space_dim=samples[0].space_dim,
        surface_x=surface_x,
        surface_y=surface_y,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_y=volume_y,
        volume_mask=volume_mask,
        metadata=[dict(sample.metadata) for sample in samples],
    )


@dataclass
class ABUPTCollate:
    geometry_points: int | None
    geometry_supernodes: int | None
    surface_anchor_points: int | None
    volume_anchor_points: int | None
    fixed_per_case: bool = True
    seed: int = 0

    OFFSETS = {
        "geometry": 1,
        "geometry_supernodes": 2,
        "surface_anchor": 3,
        "volume_anchor": 4,
    }

    def _choice(self, total: int, count: int | None, key: str) -> torch.Tensor:
        if count is None or count >= total:
            return torch.arange(total, dtype=torch.long)
        if self.fixed_per_case:
            stream_name = key.split(":", 1)[-1]
            offset = self.OFFSETS.get(stream_name, 0)
            case_id = key.split(":", 1)[0]
            generator = torch.Generator().manual_seed(self.seed + offset + stable_hash32(case_id))
        else:
            generator = torch.Generator().manual_seed(torch.seed())
        perm = torch.randperm(total, generator=generator)
        return perm[:count].sort().values

    def __call__(self, samples: list[CaseSample]) -> ABUPTBatch:
        if not samples:
            raise ValueError("ABUPTCollate received an empty batch")

        geometry_positions: list[torch.Tensor] = []
        geometry_supernodes: list[torch.Tensor] = []
        surface_positions: list[torch.Tensor] = []
        surface_targets: list[torch.Tensor] = []
        volume_positions: list[torch.Tensor] = []
        volume_targets: list[torch.Tensor] = []

        geometry_count = None
        geometry_super_count = None
        surface_anchor_count = None
        volume_anchor_count = None

        for batch_idx, sample in enumerate(samples):
            surf_pos = sample.surface_pos
            geom_idx = self._choice(len(surf_pos), self.geometry_points, f"{sample.case_id}:geometry")
            geom_pos = surf_pos[geom_idx]
            geometry_positions.append(geom_pos)
            geometry_count = geom_pos.shape[0] if geometry_count is None else geometry_count
            if geom_pos.shape[0] != geometry_count:
                raise ValueError("AB-UPT geometry_points must resolve to a fixed shape across the batch")

            local_super = self._choice(len(geom_pos), self.geometry_supernodes, f"{sample.case_id}:geometry_supernodes")
            geometry_supernodes.append(local_super)
            geometry_super_count = local_super.shape[0] if geometry_super_count is None else geometry_super_count
            if local_super.shape[0] != geometry_super_count:
                raise ValueError("AB-UPT geometry_supernodes must resolve to a fixed shape across the batch")

            surface_idx = self._choice(len(sample.surface_x), self.surface_anchor_points, f"{sample.case_id}:surface_anchor")
            surface_positions.append(sample.surface_pos[surface_idx])
            if sample.surface_y is None:
                raise ValueError("AB-UPT surface targets are required for the comparison contract")
            surface_targets.append(sample.surface_y[surface_idx])
            surface_anchor_count = surface_idx.shape[0] if surface_anchor_count is None else surface_anchor_count
            if surface_idx.shape[0] != surface_anchor_count:
                raise ValueError("AB-UPT surface_anchor_points must resolve to a fixed shape across the batch")

            if sample.volume_x is not None and sample.volume_y is not None and self.volume_anchor_points is not None:
                volume_idx = self._choice(len(sample.volume_x), self.volume_anchor_points, f"{sample.case_id}:volume_anchor")
                volume_positions.append(sample.volume_pos[volume_idx])  # type: ignore[index]
                volume_targets.append(sample.volume_y[volume_idx])
                volume_anchor_count = volume_idx.shape[0] if volume_anchor_count is None else volume_anchor_count
                if volume_idx.shape[0] != volume_anchor_count:
                    raise ValueError("AB-UPT volume_anchor_points must resolve to a fixed shape across the batch")

        geometry_position = torch.stack(geometry_positions, dim=0)
        geometry_supernode_idx = torch.stack(geometry_supernodes, dim=0)
        batch_size, num_geometry_points, _ = geometry_position.shape
        offsets = torch.arange(batch_size, dtype=torch.long) * num_geometry_points
        flat_geometry_position = geometry_position.reshape(-1, samples[0].space_dim)
        flat_geometry_supernode_idx = (geometry_supernode_idx + offsets[:, None]).reshape(-1)
        geometry_batch_idx = torch.arange(batch_size, dtype=torch.long).repeat_interleave(num_geometry_points)

        volume_anchor_position = torch.stack(volume_positions, dim=0) if volume_positions else None
        volume_anchor_target = torch.stack(volume_targets, dim=0) if volume_targets else None
        return ABUPTBatch(
            case_ids=[sample.case_id for sample in samples],
            dataset_name=samples[0].dataset_name,
            space_dim=samples[0].space_dim,
            geometry_position=flat_geometry_position,
            geometry_supernode_idx=flat_geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            surface_anchor_position=torch.stack(surface_positions, dim=0),
            surface_anchor_target=torch.stack(surface_targets, dim=0),
            volume_anchor_position=volume_anchor_position,
            volume_anchor_target=volume_anchor_target,
            metadata=[dict(sample.metadata) for sample in samples],
        )


def build_tandem_bundle(
    *,
    manifest_path: str | Path = DEFAULT_TANDEM_MANIFEST,
    stats_path: str | Path = DEFAULT_TANDEM_STATS,
    debug: bool = False,
    enable_fourier: bool = False,
    enable_te_coord_frame: bool = False,
    enable_cp_panel: bool = False,
    enable_wake_deficit: bool = False,
    enable_wake_angle: bool = False,
    enable_vortex_panel_velocity: bool = False,
) -> DatasetBundle:
    manifest = _read_json(manifest_path)
    train_indices = list(manifest["splits"]["train"])
    val_names = list(manifest.get("val_splits", []))
    train_dataset = TandemFoilCaseDataset(
        train_indices,
        manifest_path=manifest_path,
        debug=debug,
    )
    val_datasets = {
        name: TandemFoilCaseDataset(
            list(manifest["splits"][name]),
            manifest_path=manifest_path,
            debug=debug,
        )
        for name in val_names
    }
    test_datasets = {
        name: TandemFoilCaseDataset(
            list(manifest["splits"][name]),
            manifest_path=manifest_path,
            debug=debug,
        )
        for name in manifest.get("test_splits", [])
    }
    group_sizes = {name: len(indices) for name, indices in manifest["domain_groups"].items()}
    idx_to_group = {}
    for group_name, indices in manifest["domain_groups"].items():
        for idx in indices:
            idx_to_group[idx] = group_name
    sample_weights = torch.tensor(
        [1.0 / group_sizes[idx_to_group[idx]] for idx in range(len(train_indices))],
        dtype=torch.float32,
    )
    base_dim = prepare_multi.X_DIM + 2
    augmented_dim = (
        base_dim
        + (6 if enable_te_coord_frame else 0)
        + (2 if enable_wake_deficit else 0)
        + (1 if enable_wake_angle else 0)
        + (16 if enable_fourier else 0)
        + (1 if enable_cp_panel else 0)
        + (4 if enable_vortex_panel_velocity else 0)
    )
    return DatasetBundle(
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        spec=DatasetSpec(
            name="tandemfoilset",
            space_dim=2,
            surface_input_dim=augmented_dim,
            surface_output_dim=3,
            volume_input_dim=augmented_dim,
            volume_output_dim=3,
            pressure_output_index=2,
            default_metric="surface_pressure_mae",
            notes=[
                "TandemFoilSet grouped into surface and volume tokens from the overset point cloud.",
                "Noam-parity TandemFoil preprocessing is applied in the shared trainer, not in the dataset loader.",
            ],
        ),
        target_stats=_stats_from_json(stats_path),
        sample_weights=sample_weights,
        test_datasets=test_datasets,
    )


def build_airfrans_bundle(
    *,
    manifest_path: str | Path = DEFAULT_AIRFRANS_MANIFEST,
    root: str | Path | None = None,
    task: str = "full",
    debug: bool = False,
    enable_fourier: bool = False,
    enable_cp_panel: bool = False,
) -> DatasetBundle:
    manifest = _read_json(manifest_path)
    train_name = f"{task}_train"
    val_name = f"{task}_val"
    test_name = f"{task}_test"
    train_dataset = AirfRANSCaseDataset(
        list(manifest["splits"][train_name]),
        manifest_path=manifest_path,
        root=root,
        debug=debug,
        enable_fourier=enable_fourier,
        enable_cp_panel=enable_cp_panel,
        include_nut=True,
    )
    val_dataset = AirfRANSCaseDataset(
        list(manifest["splits"][val_name]),
        manifest_path=manifest_path,
        root=root,
        debug=debug,
        enable_fourier=enable_fourier,
        enable_cp_panel=enable_cp_panel,
        include_nut=True,
    )
    test_dataset = AirfRANSCaseDataset(
        list(manifest["splits"][test_name]),
        manifest_path=manifest_path,
        root=root,
        debug=debug,
        enable_fourier=enable_fourier,
        enable_cp_panel=enable_cp_panel,
        include_nut=True,
    )
    _, _, stats, _ = prepare_airfrans.load_data(
        manifest_path=manifest_path,
        task=task,
        root=root,
        debug=debug,
        include_nut=True,
        cache_size=-1 if debug else RUNTIME_CACHE_CASES,
    )
    base_dim = prepare_airfrans.X_DIM
    augmented_dim = base_dim + (16 if enable_fourier else 0) + (1 if enable_cp_panel else 0)
    return DatasetBundle(
        train_dataset=train_dataset,
        val_datasets={val_name: val_dataset},
        test_datasets={test_name: test_dataset},
        spec=DatasetSpec(
            name="airfrans",
            space_dim=2,
            surface_input_dim=augmented_dim,
            surface_output_dim=4,
            volume_input_dim=augmented_dim,
            volume_output_dim=4,
            pressure_output_index=2,
            default_metric="surface_mse",
            notes=[
                "AirfRANS uses repo-local VTK parsing and the official task split lists.",
                "Paper-facing metrics follow the official benchmark's normalized-space Surf/Vol MSE contract.",
                "Shared training uses the train/val split for tuning and reports final literature-facing metrics on the official task test split.",
            ],
        ),
        target_stats=TargetTransformStats(y_mean=stats["y_mean"], y_std=stats["y_std"]),
        sample_weights=torch.ones(len(train_dataset), dtype=torch.float32),
    )


def build_tandem_paper_bundle(
    *,
    task: str,
    manifest_path: str | Path = DEFAULT_TANDEM_PAPER_MANIFEST,
    stats_path: str | Path = DEFAULT_TANDEM_PAPER_STATS,
    debug: bool = False,
    enable_fourier: bool = False,
    enable_wake_deficit: bool = False,
    enable_wake_angle: bool = False,
) -> DatasetBundle:
    manifest_path = Path(manifest_path)
    stats_path = Path(stats_path)
    if not manifest_path.exists() or not stats_path.exists():
        split_paper_experiment4.materialize_default_artifacts(
            manifest_path=manifest_path,
            stats_path=stats_path,
        )
    manifest = _read_json(manifest_path)
    task_manifest = manifest["tasks"].get(task)
    if task_manifest is None:
        available = ", ".join(sorted(manifest["tasks"]))
        raise ValueError(f"Unknown tandemfoil paper task {task!r}; available tasks: {available}")
    task_stats = _read_json(stats_path)["tasks"].get(task)
    if task_stats is None:
        raise ValueError(f"Missing stats for tandemfoil paper task {task!r}")

    train_dataset = PaperTandemFoilCaseDataset(
        list(task_manifest["splits"]["train"]),
        task_manifest["pickle_files"],
        root_candidates=manifest.get("data_root_candidates"),
        debug=debug,
        task_name=task,
        enable_fourier=enable_fourier,
        enable_wake_deficit=enable_wake_deficit,
        enable_wake_angle=enable_wake_angle,
    )
    val_dataset = PaperTandemFoilCaseDataset(
        list(task_manifest["splits"]["val"]),
        task_manifest["pickle_files"],
        root_candidates=manifest.get("data_root_candidates"),
        debug=debug,
        task_name=task,
        enable_fourier=enable_fourier,
        enable_wake_deficit=enable_wake_deficit,
        enable_wake_angle=enable_wake_angle,
    )
    test_dataset = PaperTandemFoilCaseDataset(
        list(task_manifest["splits"]["test"]),
        task_manifest["pickle_files"],
        root_candidates=manifest.get("data_root_candidates"),
        debug=debug,
        task_name=task,
        enable_fourier=enable_fourier,
        enable_wake_deficit=enable_wake_deficit,
        enable_wake_angle=enable_wake_angle,
    )
    augmented_dim = (
        paper_prepare_multi.X_DIM
        + (16 if enable_fourier else 0)
        + (2 if enable_wake_deficit else 0)
        + (1 if enable_wake_angle else 0)
    )
    return DatasetBundle(
        train_dataset=train_dataset,
        val_datasets={"val": val_dataset},
        test_datasets={"test": test_dataset},
        spec=DatasetSpec(
            name="tandemfoilset_paper",
            space_dim=2,
            surface_input_dim=augmented_dim,
            surface_output_dim=3,
            volume_input_dim=augmented_dim,
            volume_output_dim=3,
            pressure_output_index=2,
            default_metric="field_mse",
            notes=[
                "Paper-faithful high-Re TandemFoilSet benchmark using Experiment 4 split semantics.",
                "Primary comparison metric is normalized full-field MSE, matching the paper contract.",
            ],
        ),
        target_stats=TargetTransformStats(
            y_mean=torch.tensor(task_stats["y_mean"], dtype=torch.float32),
            y_std=torch.tensor(task_stats["y_std"], dtype=torch.float32),
        ),
        sample_weights=torch.ones(len(train_dataset), dtype=torch.float32),
    )


def build_drivaerml_bundle(
    *,
    manifest_path: str | Path = DEFAULT_DRIVAERML_MANIFEST,
    root: str | Path | None = None,
    surface_only: bool = True,
    enable_fourier: bool = False,
    train_surface_points: int = 0,
    eval_surface_points: int = 0,
    train_volume_points: int = 0,
    eval_volume_points: int = 0,
) -> DatasetBundle:
    manifest = _read_json(manifest_path)
    _validate_drivaerml_manifest(manifest, manifest_path)
    train_sampling_mode = "train_random" if train_surface_points > 0 or train_volume_points > 0 else "full"
    eval_sampling_mode = "eval_chunk" if eval_surface_points > 0 or eval_volume_points > 0 else "full"
    train_dataset = DrivAerMLCaseDataset(
        list(manifest["surface_splits"]["train"]),
        manifest_path=manifest_path,
        root=root,
        enable_fourier=enable_fourier,
        surface_only=surface_only,
        max_surface_points=train_surface_points,
        max_volume_points=train_volume_points,
        sampling_mode=train_sampling_mode,
    )
    val_dataset = DrivAerMLCaseDataset(
        list(manifest["surface_splits"]["val"]),
        manifest_path=manifest_path,
        root=root,
        enable_fourier=enable_fourier,
        surface_only=surface_only,
        max_surface_points=eval_surface_points,
        max_volume_points=eval_volume_points,
        sampling_mode=eval_sampling_mode,
    )
    test_dataset = DrivAerMLCaseDataset(
        list(manifest["surface_splits"]["test"]),
        manifest_path=manifest_path,
        root=root,
        enable_fourier=enable_fourier,
        surface_only=surface_only,
        max_surface_points=eval_surface_points,
        max_volume_points=eval_volume_points,
        sampling_mode=eval_sampling_mode,
    )
    stats = prepare_drivaerml.surface_stats_from_normalizers(train_dataset.store)
    target_stats = TargetTransformStats(
        y_mean=None if stats is None else stats.get("y_mean"),
        y_std=None if stats is None else stats.get("y_std"),
        geometry_center=None if stats is None else stats.get("geometry_center"),
        geometry_scale=None if stats is None else stats.get("geometry_scale"),
    )
    base_surface_dim = prepare_drivaerml.SURFACE_X_DIM
    base_volume_dim = 0 if surface_only else prepare_drivaerml.VOLUME_X_DIM
    fourier_extra = 24 if enable_fourier else 0
    return DatasetBundle(
        train_dataset=train_dataset,
        val_datasets={"val_surface": val_dataset},
        test_datasets={"test_surface": test_dataset},
        spec=DatasetSpec(
            name="drivaerml",
            space_dim=3,
            surface_input_dim=base_surface_dim + fourier_extra,
            surface_output_dim=prepare_drivaerml.SURFACE_Y_DIM,
            volume_input_dim=base_volume_dim + fourier_extra if base_volume_dim else 0,
            volume_output_dim=0 if surface_only else prepare_drivaerml.VOLUME_Y_DIM,
            pressure_output_index=0,
            default_metric="surface_rel_l2_pct",
            notes=[
                "DrivAerML defaults to the repaired public 400/34/50 surface split for the paper sprint.",
                "Surface-first mode is the default; the volume subset stays optional.",
                "Paper-facing evaluation follows AB-UPT's average per-case relative-L2 contract on unnormalized targets.",
                "When DrivAerML train point limits are set, each epoch repeats a case ceil(N / points_per_view) times.",
                "When DrivAerML eval point limits are set, val/test cover every point exactly once via strided case views.",
            ],
        ),
        target_stats=target_stats,
        sample_weights=torch.ones(len(train_dataset), dtype=torch.float32),
    )


def build_dataset_bundle(
    dataset_name: str,
    *,
    debug: bool = False,
    tandem_manifest: str | Path = DEFAULT_TANDEM_MANIFEST,
    tandem_stats: str | Path = DEFAULT_TANDEM_STATS,
    tandemfoil_paper_task: str = "cruise_random_uniform",
    tandemfoil_paper_manifest: str | Path = DEFAULT_TANDEM_PAPER_MANIFEST,
    tandemfoil_paper_stats: str | Path = DEFAULT_TANDEM_PAPER_STATS,
    airfrans_manifest: str | Path = DEFAULT_AIRFRANS_MANIFEST,
    airfrans_root: str | Path | None = None,
    airfrans_task: str = "full",
    drivaerml_manifest: str | Path = DEFAULT_DRIVAERML_MANIFEST,
    drivaerml_root: str | Path | None = None,
    drivaerml_surface_only: bool = True,
    drivaerml_train_surface_points: int = 0,
    drivaerml_eval_surface_points: int = 0,
    drivaerml_train_volume_points: int = 0,
    drivaerml_eval_volume_points: int = 0,
    enable_fourier: bool = False,
    enable_te_coord_frame: bool = False,
    enable_cp_panel: bool = False,
    enable_wake_deficit: bool = False,
    enable_wake_angle: bool = False,
    enable_vortex_panel_velocity: bool = False,
) -> DatasetBundle:
    if dataset_name in {"tandemfoil", "tandemfoilset"}:
        return build_tandem_bundle(
            manifest_path=tandem_manifest,
            stats_path=tandem_stats,
            debug=debug,
            enable_fourier=enable_fourier,
            enable_te_coord_frame=enable_te_coord_frame,
            enable_cp_panel=enable_cp_panel,
            enable_wake_deficit=enable_wake_deficit,
            enable_wake_angle=enable_wake_angle,
            enable_vortex_panel_velocity=enable_vortex_panel_velocity,
        )
    if dataset_name in {"tandemfoil_paper", "tandemfoilset_paper"}:
        return build_tandem_paper_bundle(
            task=tandemfoil_paper_task,
            manifest_path=tandemfoil_paper_manifest,
            stats_path=tandemfoil_paper_stats,
            debug=debug,
            enable_fourier=enable_fourier,
            enable_wake_deficit=enable_wake_deficit,
            enable_wake_angle=enable_wake_angle,
        )
    if dataset_name == "airfrans":
        return build_airfrans_bundle(
            manifest_path=airfrans_manifest,
            root=airfrans_root,
            task=airfrans_task,
            debug=debug,
            enable_fourier=enable_fourier,
            enable_cp_panel=enable_cp_panel,
        )
    if dataset_name == "drivaerml":
        return build_drivaerml_bundle(
            manifest_path=drivaerml_manifest,
            root=drivaerml_root,
            surface_only=drivaerml_surface_only,
            enable_fourier=enable_fourier,
            train_surface_points=drivaerml_train_surface_points,
            eval_surface_points=drivaerml_eval_surface_points,
            train_volume_points=drivaerml_train_volume_points,
            eval_volume_points=drivaerml_eval_volume_points,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
