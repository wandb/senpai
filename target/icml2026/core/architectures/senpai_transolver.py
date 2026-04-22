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


class GlobalContextBlock(nn.Module):
    """BERT [CLS]-style global context token that aggregates global flow field
    information via cross-attention and broadcasts it back to all nodes."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.global_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.global_token, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        g = self.global_token.expand(B, -1, -1)
        g_updated, _ = self.cross_attn(
            query=self.norm1(g),
            key=self.norm2(x),
            value=x,
        )
        g = g + g_updated
        global_bias = self.proj(g)
        return x + global_bias


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
        global_context_token: bool = False,
        global_context_token_interval: int = 2,
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
        self.use_global_context = global_context_token
        if global_context_token:
            n_blocks = len(self.backbone.blocks)
            n_heads = self.n_hidden // 64 if self.n_hidden >= 64 else 1
            # Place one GlobalContextBlock after every K transformer blocks
            self.global_ctx_interval = global_context_token_interval
            n_ctx_blocks = (n_blocks + global_context_token_interval - 1) // global_context_token_interval
            self.global_ctx_blocks = nn.ModuleList(
                [GlobalContextBlock(dim=self.n_hidden, num_heads=n_heads) for _ in range(n_ctx_blocks)]
            )
        else:
            self.global_ctx_interval = global_context_token_interval
            self.global_ctx_blocks = nn.ModuleList()

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

    def _forward_backbone_with_global_ctx(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run backbone blocks, injecting GlobalContextBlock after every K blocks."""
        ctx_idx = 0
        for block_idx, block in enumerate(self.backbone.blocks):
            hidden = block(hidden, attn_mask=attn_mask)
            # Apply global context after every K-th block (1-indexed)
            if (block_idx + 1) % self.global_ctx_interval == 0 and ctx_idx < len(self.global_ctx_blocks):
                hidden = self.global_ctx_blocks[ctx_idx](hidden)
                ctx_idx += 1
        # Apply any remaining global context blocks after the final backbone block
        if ctx_idx < len(self.global_ctx_blocks):
            hidden = self.global_ctx_blocks[ctx_idx](hidden)
        return hidden

    def forward(
        self,
        *,
        surface_x: torch.Tensor,
        surface_mask: torch.Tensor,
        volume_x: torch.Tensor | None = None,
        volume_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not self.use_global_context:
            outputs = super().forward(
                surface_x=surface_x,
                surface_mask=surface_mask,
                volume_x=volume_x,
                volume_mask=volume_mask,
            )
        else:
            # Replicate ReferenceTransolver.forward with global context injection
            surface_hidden = self._group_hidden(surface_x, self.project_surface_features, self.surface_bias)
            hidden_parts = [surface_hidden]
            mask_parts = [surface_mask]
            volume_hidden = None
            if volume_x is not None and volume_mask is not None and self.volume_output_dim > 0:
                volume_hidden = self._group_hidden(volume_x, self.project_volume_features, self.volume_bias)
                hidden_parts.append(volume_hidden)
                mask_parts.append(volume_mask)
            hidden = torch.cat(hidden_parts, dim=1) + self.placeholder
            attn_mask = torch.cat(mask_parts, dim=1)
            hidden = self._forward_backbone_with_global_ctx(hidden, attn_mask)
            raw_output = self.out(self.norm(hidden))

            surface_tokens = surface_x.shape[1]
            surface_preds = raw_output[:, :surface_tokens, : self.surface_output_dim]
            volume_preds = None
            if self.volume_output_dim > 0 and volume_x is not None:
                volume_tokens = volume_x.shape[1]
                start = self.surface_output_dim
                volume_preds = raw_output[:, surface_tokens : surface_tokens + volume_tokens, start : start + self.volume_output_dim]

            surface_preds = surface_preds * surface_mask.unsqueeze(-1)
            if volume_preds is not None and volume_mask is not None:
                volume_preds = volume_preds * volume_mask.unsqueeze(-1)

            outputs = {
                "surface_hidden": hidden[:, :surface_tokens],
                "surface_preds": surface_preds,
                "volume_hidden": None if volume_hidden is None else hidden[:, surface_tokens:],
                "volume_preds": volume_preds,
            }

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
