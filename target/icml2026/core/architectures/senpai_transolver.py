# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import torch
import torch.nn as nn

from .transolver_reference import ReferenceTransolver


class ZeroInitSurfaceRefinementHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        *,
        mlp_hidden_dim: int = 128,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = hidden_dim + output_dim
        for layer_idx in range(max(num_hidden_layers, 1)):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else mlp_hidden_dim, mlp_hidden_dim))
            layers.append(nn.LayerNorm(mlp_hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(mlp_hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor, base_pred: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([hidden, base_pred], dim=-1))


class ANPSurfaceDecoder(nn.Module):
    """Deterministic ANP-style surface decoder used by the #2379 lineage."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
        context_extra_dim: int = 5,
        attn_dim: int | None = None,
        num_heads: int = 4,
    ):
        super().__init__()
        self.attn_dim = attn_dim or hidden_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim + context_extra_dim, self.attn_dim),
            nn.GELU(),
            nn.Linear(self.attn_dim, self.attn_dim),
        )
        self.pre_attn_norm = nn.LayerNorm(self.attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(self.attn_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim),
            nn.GELU(),
            nn.Linear(self.attn_dim, output_dim),
        )
        nn.init.zeros_(self.output_mlp[-1].weight)
        nn.init.zeros_(self.output_mlp[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        cp_panel: torch.Tensor | None,
        coords: torch.Tensor,
        saf_vec: torch.Tensor,
        is_surface: torch.Tensor,
        is_tandem: torch.Tensor,
        fore_mask: torch.Tensor,
        aft_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, _ = hidden.shape
        output_dim = self.output_mlp[-1].out_features
        correction = torch.zeros(batch_size, num_points, output_dim, device=hidden.device, dtype=hidden.dtype)

        for batch_idx in range(batch_size):
            surface_idx = is_surface[batch_idx].nonzero(as_tuple=True)[0]
            if surface_idx.numel() == 0:
                continue

            hidden_local = hidden[batch_idx, surface_idx]
            coords_local = coords[batch_idx, surface_idx]
            saf_local = saf_vec[batch_idx, surface_idx]
            cp_local = (
                cp_panel[batch_idx, surface_idx]
                if cp_panel is not None
                else torch.zeros(surface_idx.numel(), 1, device=hidden.device, dtype=hidden.dtype)
            )
            context_input = torch.cat([hidden_local, cp_local, coords_local, saf_local], dim=-1)
            context = self.pre_attn_norm(self.context_encoder(context_input))

            attn_mask = None
            if bool(is_tandem[batch_idx].item()):
                fore_local = fore_mask[batch_idx, surface_idx].nonzero(as_tuple=True)[0]
                aft_local = aft_mask[batch_idx, surface_idx].nonzero(as_tuple=True)[0]
                if fore_local.numel() > 0 and aft_local.numel() > 0:
                    attn_mask = torch.zeros(surface_idx.numel(), surface_idx.numel(), device=hidden.device, dtype=context.dtype)
                    attn_mask[fore_local.unsqueeze(1), aft_local.unsqueeze(0)] = float("-inf")

            attn_out, _ = self.cross_attn(
                context.unsqueeze(0),
                context.unsqueeze(0),
                context.unsqueeze(0),
                attn_mask=attn_mask,
            )
            attn_out = attn_out.squeeze(0)
            fused = self.post_attn_norm(attn_out + hidden_local)
            correction[batch_idx, surface_idx] = self.output_mlp(fused).to(correction.dtype)

        return correction


class SenpaiTransolver(ReferenceTransolver):
    """Grouped-domain Transolver plus the retained Senpai refinement mechanisms."""

    def __init__(
        self,
        *,
        pressure_output_index: int | None,
        surface_refine: bool = True,
        surface_refine_hidden_dim: int = 128,
        surface_refine_layers: int = 2,
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
            ZeroInitSurfaceRefinementHead(
                self.n_hidden,
                self.surface_output_dim,
                mlp_hidden_dim=surface_refine_hidden_dim,
                num_hidden_layers=surface_refine_layers,
            )
            if self.surface_refine
            else None
        )

    def _apply_pressure_prior_addition(
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
        surface_preds = self._apply_pressure_prior_addition(
            outputs["surface_preds"],
            surface_x,
            self.surface_pressure_prior_idx,
        )
        if self.surface_head is not None and surface_preds is not None:
            surface_preds = surface_preds + self.surface_head(outputs["surface_hidden"], surface_preds)
            surface_preds = surface_preds * surface_mask.unsqueeze(-1)
        volume_preds = self._apply_pressure_prior_addition(
            outputs["volume_preds"],
            volume_x,
            self.volume_pressure_prior_idx,
        )
        outputs["surface_preds"] = surface_preds
        outputs["volume_preds"] = volume_preds
        return outputs
