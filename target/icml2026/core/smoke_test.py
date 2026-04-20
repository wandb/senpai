# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, Dataset

from core.architectures import ABUPTReference, ReferenceTransolver, SenpaiTransolver
from core.contracts import CaseSample
from core.datasets import ABUPTCollate, collate_grouped


class SyntheticCaseDataset(Dataset):
    def __init__(self, *, space_dim: int, input_dim: int, output_dim: int, length: int = 4):
        self.space_dim = space_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> CaseSample:
        surface_n = 11 + idx
        volume_n = 17 + idx
        return CaseSample(
            case_id=f"synthetic_{idx}",
            dataset_name="synthetic",
            space_dim=self.space_dim,
            surface_x=torch.randn(surface_n, self.input_dim),
            surface_y=torch.randn(surface_n, self.output_dim),
            volume_x=torch.randn(volume_n, self.input_dim),
            volume_y=torch.randn(volume_n, self.output_dim),
            metadata={"idx": idx},
        )


def main() -> None:
    dataset = SyntheticCaseDataset(space_dim=2, input_dim=24, output_dim=3)
    grouped_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_grouped)
    grouped_batch = next(iter(grouped_loader))

    ref = ReferenceTransolver(
        space_dim=2,
        surface_input_dim=24,
        surface_output_dim=3,
        volume_input_dim=24,
        volume_output_dim=3,
    )
    ref_out = ref(
        surface_x=grouped_batch.surface_x,
        surface_mask=grouped_batch.surface_mask,
        volume_x=grouped_batch.volume_x,
        volume_mask=grouped_batch.volume_mask,
    )

    sen = SenpaiTransolver(
        space_dim=2,
        surface_input_dim=24,
        surface_output_dim=3,
        volume_input_dim=24,
        volume_output_dim=3,
        pressure_output_index=2,
        surface_pressure_prior_idx=23,
        volume_pressure_prior_idx=23,
    )
    sen_out = sen(
        surface_x=grouped_batch.surface_x,
        surface_mask=grouped_batch.surface_mask,
        volume_x=grouped_batch.volume_x,
        volume_mask=grouped_batch.volume_mask,
    )

    abupt_loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=ABUPTCollate(
            geometry_points=8,
            geometry_supernodes=4,
            surface_anchor_points=6,
            volume_anchor_points=6,
        ),
    )
    abupt_batch = next(iter(abupt_loader))
    abupt = ABUPTReference(
        space_dim=2,
        surface_output_dim=3,
        volume_output_dim=3,
    )
    abupt_out = abupt(
        geometry_position=abupt_batch.geometry_position,
        geometry_supernode_idx=abupt_batch.geometry_supernode_idx,
        geometry_batch_idx=abupt_batch.geometry_batch_idx,
        surface_anchor_position=abupt_batch.surface_anchor_position,
        volume_anchor_position=abupt_batch.volume_anchor_position,
    )

    summary = {
        "grouped_surface_shape": list(grouped_batch.surface_x.shape),
        "grouped_volume_shape": list(grouped_batch.volume_x.shape),
        "reference_surface_shape": list(ref_out["surface_preds"].shape),
        "senpai_surface_shape": list(sen_out["surface_preds"].shape),
        "abupt_geometry_shape": list(abupt_batch.geometry_position.shape),
        "abupt_surface_shape": list(abupt_out["surface_preds"].shape),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
