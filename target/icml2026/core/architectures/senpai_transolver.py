# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import torch
import torch.nn as nn

from .transolver_reference import ReferenceTransolver


class ZeroInitSurfaceRefinementHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, mlp_hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor, base_pred: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([hidden, base_pred], dim=-1))


class SenpaiTransolver(ReferenceTransolver):
    """Reference Transolver plus the small durable Senpai mechanisms.

    The clean retained mechanism set is intentionally narrow:
    - residual prediction from a designated input prior channel
    - zero-init surface refinement head
    """

    def __init__(
        self,
        *,
        pressure_output_index: int | None,
        surface_refine: bool = True,
        surface_pressure_prior_idx: int | None = None,
        volume_pressure_prior_idx: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pressure_output_index = pressure_output_index
        self.surface_pressure_prior_idx = surface_pressure_prior_idx
        self.volume_pressure_prior_idx = volume_pressure_prior_idx
        self.surface_refine = surface_refine and self.surface_output_dim > 0
        self.surface_head = (
            ZeroInitSurfaceRefinementHead(self.n_hidden, self.surface_output_dim)
            if self.surface_refine
            else None
        )

    def _apply_pressure_residual(
        self,
        preds: torch.Tensor | None,
        x: torch.Tensor | None,
        prior_idx: int | None,
    ) -> torch.Tensor | None:
        if (
            preds is None
            or x is None
            or prior_idx is None
            or self.pressure_output_index is None
            or prior_idx >= x.shape[-1]
        ):
            return preds
        preds = preds.clone()
        preds[..., self.pressure_output_index] = preds[..., self.pressure_output_index] + x[..., prior_idx]
        return preds

    def forward(
        self,
        *,
        surface_x: torch.Tensor,
        surface_mask: torch.Tensor,
        volume_x: torch.Tensor | None = None,
        volume_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = super().forward(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
        surface_preds = self._apply_pressure_residual(
            outputs["surface_preds"],
            surface_x,
            self.surface_pressure_prior_idx,
        )
        if self.surface_head is not None and surface_preds is not None:
            surface_preds = surface_preds + self.surface_head(outputs["surface_hidden"], surface_preds)
            surface_preds = surface_preds * surface_mask.unsqueeze(-1)
        volume_preds = self._apply_pressure_residual(
            outputs["volume_preds"],
            volume_x,
            self.volume_pressure_prior_idx,
        )
        outputs["surface_preds"] = surface_preds
        outputs["volume_preds"] = volume_preds
        return outputs
