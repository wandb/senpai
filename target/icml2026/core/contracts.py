# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DatasetSpec:
    name: str
    space_dim: int
    surface_input_dim: int
    surface_output_dim: int
    volume_input_dim: int = 0
    volume_output_dim: int = 0
    pressure_output_index: int | None = None
    default_metric: str = "mae"
    notes: list[str] = field(default_factory=list)


@dataclass
class TargetTransformStats:
    y_mean: torch.Tensor | None = None
    y_std: torch.Tensor | None = None
    geometry_center: torch.Tensor | None = None
    geometry_scale: torch.Tensor | None = None


@dataclass
class CaseSample:
    case_id: str
    dataset_name: str
    space_dim: int
    surface_x: torch.Tensor
    surface_y: torch.Tensor | None
    volume_x: torch.Tensor | None = None
    volume_y: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def surface_pos(self) -> torch.Tensor:
        return self.surface_x[:, : self.space_dim]

    @property
    def volume_pos(self) -> torch.Tensor | None:
        if self.volume_x is None:
            return None
        return self.volume_x[:, : self.space_dim]


@dataclass
class GroupedBatch:
    case_ids: list[str]
    dataset_name: str
    space_dim: int
    surface_x: torch.Tensor
    surface_y: torch.Tensor | None
    surface_mask: torch.Tensor
    volume_x: torch.Tensor | None
    volume_y: torch.Tensor | None
    volume_mask: torch.Tensor | None
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def to(self, device: torch.device | str) -> "GroupedBatch":
        return GroupedBatch(
            case_ids=list(self.case_ids),
            dataset_name=self.dataset_name,
            space_dim=self.space_dim,
            surface_x=self.surface_x.to(device),
            surface_y=None if self.surface_y is None else self.surface_y.to(device),
            surface_mask=self.surface_mask.to(device),
            volume_x=None if self.volume_x is None else self.volume_x.to(device),
            volume_y=None if self.volume_y is None else self.volume_y.to(device),
            volume_mask=None if self.volume_mask is None else self.volume_mask.to(device),
            metadata=list(self.metadata),
        )


@dataclass
class ABUPTBatch:
    case_ids: list[str]
    dataset_name: str
    space_dim: int
    geometry_position: torch.Tensor
    geometry_supernode_idx: torch.Tensor
    geometry_batch_idx: torch.Tensor | None
    surface_anchor_position: torch.Tensor
    surface_anchor_target: torch.Tensor | None
    volume_anchor_position: torch.Tensor | None
    volume_anchor_target: torch.Tensor | None
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def to(self, device: torch.device | str) -> "ABUPTBatch":
        return ABUPTBatch(
            case_ids=list(self.case_ids),
            dataset_name=self.dataset_name,
            space_dim=self.space_dim,
            geometry_position=self.geometry_position.to(device),
            geometry_supernode_idx=self.geometry_supernode_idx.to(device),
            geometry_batch_idx=None if self.geometry_batch_idx is None else self.geometry_batch_idx.to(device),
            surface_anchor_position=self.surface_anchor_position.to(device),
            surface_anchor_target=None if self.surface_anchor_target is None else self.surface_anchor_target.to(device),
            volume_anchor_position=None if self.volume_anchor_position is None else self.volume_anchor_position.to(device),
            volume_anchor_target=None if self.volume_anchor_target is None else self.volume_anchor_target.to(device),
            metadata=list(self.metadata),
        )


@dataclass
class DatasetBundle:
    train_dataset: torch.utils.data.Dataset
    val_datasets: dict[str, torch.utils.data.Dataset]
    spec: DatasetSpec
    target_stats: TargetTransformStats
    sample_weights: torch.Tensor | None = None

