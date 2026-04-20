# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import json
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


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TANDEM_MANIFEST = ROOT_DIR / "tandemfoil/data/split_manifest_tandemfoilset_v2.json"
DEFAULT_TANDEM_STATS = ROOT_DIR / "tandemfoil/data/split_stats.json"
DEFAULT_AIRFRANS_MANIFEST = ROOT_DIR / "airfrans/data/split_manifest_airfrans.json"
DEFAULT_DRIVAERML_MANIFEST = ROOT_DIR / "drivaerml/data/split_manifest_drivaerml.json"


def _read_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _stats_from_json(stats_path: str | Path) -> TargetTransformStats:
    raw = _read_json(stats_path)
    return TargetTransformStats(
        y_mean=torch.tensor(raw["y_mean"], dtype=torch.float32),
        y_std=torch.tensor(raw["y_std"], dtype=torch.float32),
    )


class TandemFoilCaseDataset(Dataset):
    def __init__(
        self,
        split_indices: list[int],
        manifest_path: str | Path = DEFAULT_TANDEM_MANIFEST,
        *,
        debug: bool = False,
        enable_fourier: bool = False,
        enable_cp_panel: bool = False,
        enable_wake_deficit: bool = False,
        enable_wake_angle: bool = False,
    ):
        manifest = _read_json(manifest_path)
        cache_size = -1 if debug else 0
        self.base = prepare_multi.MultiFieldDataset(
            prepare_multi._resolve_pickle_paths(manifest),
            cache_size=cache_size,
        )
        self.indices = list(split_indices)
        self.augment = dict(
            enable_fourier=enable_fourier,
            enable_cp_panel=enable_cp_panel,
            enable_wake_deficit=enable_wake_deficit,
            enable_wake_angle=enable_wake_angle,
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
            case_id=f"tandem_{base_idx}",
            dataset_name="tandemfoilset",
            space_dim=2,
            surface_x=surface_x,
            surface_y=surface_y,
            volume_x=volume_x,
            volume_y=volume_y,
            metadata={"base_idx": base_idx},
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
    ):
        manifest = _read_json(manifest_path)
        dataset_root = prepare_airfrans._resolve_root(manifest, override_root=root)
        cache_size = -1 if debug else 0
        self.base = prepare_airfrans.AirfRANSDataset(
            root=dataset_root,
            case_ids=case_ids,
            include_nut=False,
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
    ):
        self.store = prepare_drivaerml.DrivAerMLCaseStore(manifest_path=manifest_path, root=root)
        self.case_ids = list(case_ids)
        self.surface_only = surface_only
        self.augment = dict(
            enable_fourier=enable_fourier,
            enable_cp_panel=False,
            enable_wake_deficit=False,
            enable_wake_angle=False,
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> CaseSample:
        case = self.store.load_case(self.case_ids[idx])
        sample = CaseSample(
            case_id=case.case_id,
            dataset_name="drivaerml",
            space_dim=3,
            surface_x=case.surface_x,
            surface_y=case.surface_y,
            volume_x=None if self.surface_only else case.volume_x,
            volume_y=None if self.surface_only else case.volume_y,
            metadata=dict(case.metadata),
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
    enable_cp_panel: bool = False,
    enable_wake_deficit: bool = False,
    enable_wake_angle: bool = False,
) -> DatasetBundle:
    manifest = _read_json(manifest_path)
    train_indices = list(manifest["splits"]["train"])
    val_names = list(manifest.get("val_splits", []))
    train_dataset = TandemFoilCaseDataset(
        train_indices,
        manifest_path=manifest_path,
        debug=debug,
        enable_fourier=enable_fourier,
        enable_cp_panel=enable_cp_panel,
        enable_wake_deficit=enable_wake_deficit,
        enable_wake_angle=enable_wake_angle,
    )
    val_datasets = {
        name: TandemFoilCaseDataset(
            list(manifest["splits"][name]),
            manifest_path=manifest_path,
            debug=debug,
            enable_fourier=enable_fourier,
            enable_cp_panel=enable_cp_panel,
            enable_wake_deficit=enable_wake_deficit,
            enable_wake_angle=enable_wake_angle,
        )
        for name in val_names
    }
    group_sizes = {name: len(indices) for name, indices in manifest["domain_groups"].items()}
    idx_to_group = {}
    for group_name, indices in manifest["domain_groups"].items():
        for idx in indices:
            idx_to_group[idx] = group_name
    sample_weights = torch.tensor(
        [1.0 / group_sizes[idx_to_group[idx]] for idx in train_indices],
        dtype=torch.float32,
    )
    base_dim = prepare_multi.X_DIM
    augmented_dim = base_dim + (16 if enable_fourier else 0) + (1 if enable_cp_panel else 0) + (2 if enable_wake_deficit else 0) + (1 if enable_wake_angle else 0)
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
            notes=["TandemFoilSet grouped into surface and volume tokens from the overset point cloud."],
        ),
        target_stats=_stats_from_json(stats_path),
        sample_weights=sample_weights,
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
    train_dataset = AirfRANSCaseDataset(
        list(manifest["splits"][train_name]),
        manifest_path=manifest_path,
        root=root,
        debug=debug,
        enable_fourier=enable_fourier,
        enable_cp_panel=enable_cp_panel,
    )
    val_dataset = AirfRANSCaseDataset(
        list(manifest["splits"][val_name]),
        manifest_path=manifest_path,
        root=root,
        debug=debug,
        enable_fourier=enable_fourier,
        enable_cp_panel=enable_cp_panel,
    )
    _, _, stats, _ = prepare_airfrans.load_data(
        manifest_path=manifest_path,
        task=task,
        root=root,
        debug=debug,
        include_nut=False,
        cache_size=-1 if debug else 0,
    )
    base_dim = prepare_airfrans.X_DIM
    augmented_dim = base_dim + (16 if enable_fourier else 0) + (1 if enable_cp_panel else 0)
    return DatasetBundle(
        train_dataset=train_dataset,
        val_datasets={val_name: val_dataset},
        spec=DatasetSpec(
            name="airfrans",
            space_dim=2,
            surface_input_dim=augmented_dim,
            surface_output_dim=3,
            volume_input_dim=augmented_dim,
            volume_output_dim=3,
            pressure_output_index=2,
            notes=["AirfRANS uses repo-local VTK parsing and the official task split lists."],
        ),
        target_stats=TargetTransformStats(y_mean=stats["y_mean"], y_std=stats["y_std"]),
        sample_weights=torch.ones(len(train_dataset), dtype=torch.float32),
    )


def build_drivaerml_bundle(
    *,
    manifest_path: str | Path = DEFAULT_DRIVAERML_MANIFEST,
    root: str | Path | None = None,
    surface_only: bool = True,
    enable_fourier: bool = False,
) -> DatasetBundle:
    manifest = _read_json(manifest_path)
    train_dataset = DrivAerMLCaseDataset(
        list(manifest["surface_splits"]["train"]),
        manifest_path=manifest_path,
        root=root,
        enable_fourier=enable_fourier,
        surface_only=surface_only,
    )
    val_dataset = DrivAerMLCaseDataset(
        list(manifest["surface_splits"]["val"]),
        manifest_path=manifest_path,
        root=root,
        enable_fourier=enable_fourier,
        surface_only=surface_only,
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
        spec=DatasetSpec(
            name="drivaerml",
            space_dim=3,
            surface_input_dim=base_surface_dim + fourier_extra,
            surface_output_dim=prepare_drivaerml.SURFACE_Y_DIM,
            volume_input_dim=base_volume_dim + fourier_extra if base_volume_dim else 0,
            volume_output_dim=0 if surface_only else prepare_drivaerml.VOLUME_Y_DIM,
            pressure_output_index=0,
            notes=[
                "DrivAerML defaults to the packaged public surface split for the paper sprint.",
                "Surface-first mode is the default; the volume subset stays optional.",
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
    airfrans_manifest: str | Path = DEFAULT_AIRFRANS_MANIFEST,
    airfrans_root: str | Path | None = None,
    airfrans_task: str = "full",
    drivaerml_manifest: str | Path = DEFAULT_DRIVAERML_MANIFEST,
    drivaerml_root: str | Path | None = None,
    drivaerml_surface_only: bool = True,
    enable_fourier: bool = False,
    enable_cp_panel: bool = False,
    enable_wake_deficit: bool = False,
    enable_wake_angle: bool = False,
) -> DatasetBundle:
    if dataset_name in {"tandemfoil", "tandemfoilset"}:
        return build_tandem_bundle(
            manifest_path=tandem_manifest,
            stats_path=tandem_stats,
            debug=debug,
            enable_fourier=enable_fourier,
            enable_cp_panel=enable_cp_panel,
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
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
