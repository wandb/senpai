# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_rope_nd(x: torch.Tensor, positions: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Apply N-dimensional Rotary Position Embeddings using continuous spatial coordinates.

    x: [..., D] tensor (query or key)
    positions: [..., space_dim] continuous coordinates
    rope_dim: number of head dimensions to rotate (rest pass through unchanged)
    """
    D = x.shape[-1]
    space_dim = positions.shape[-1]
    dims_per_axis = (rope_dim // space_dim) & ~1
    if dims_per_axis < 2:
        return x
    total_rope = dims_per_axis * space_dim
    half = dims_per_axis // 2
    freq_seq = torch.arange(half, device=x.device, dtype=torch.float32) / half
    freqs = 1.0 / (10000.0 ** freq_seq)
    angles = positions.unsqueeze(-1) * freqs
    angles = angles.flatten(-2)
    cos_a = angles.cos().to(x.dtype)
    sin_a = angles.sin().to(x.dtype)
    x_rope = x[..., :total_rope]
    x_rest = x[..., total_rope:]
    x1 = x_rope[..., 0::2]
    x2 = x_rope[..., 1::2]
    y1 = cos_a * x1 - sin_a * x2
    y2 = sin_a * x1 + cos_a * x2
    x_rotated = torch.stack([y1, y2], dim=-1).flatten(-2)
    if x_rest.shape[-1] > 0:
        return torch.cat([x_rotated, x_rest], dim=-1)
    return x_rotated


class LinearProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.project = nn.Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_linear(self.project)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, max_wavelength: int = 10_000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.max_wavelength = max_wavelength
        padding = hidden_dim % input_dim
        dim_per_axis = (hidden_dim - padding) // input_dim
        sincos_padding = dim_per_axis % 2
        self.padding = padding + sincos_padding * input_dim
        effective_dim_per_axis = (hidden_dim - self.padding) // input_dim
        if effective_dim_per_axis <= 0:
            raise ValueError("hidden_dim must be large enough for the requested input dimension")
        arange = torch.arange(0, effective_dim_per_axis, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (arange / effective_dim_per_axis))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        out = coords.unsqueeze(-1) * self.omega
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        emb = emb.flatten(start_dim=-2)
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, padding], dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.net.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpActDownMlp(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_dim)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransolverAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, dropout: float = 0.0, rope_dim: int = 0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads
        self.num_slices = num_slices
        self.dropout = dropout
        self.rope_dim = rope_dim

        self.temperature = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.5))
        self.in_project_x = LinearProjection(hidden_dim, hidden_dim)
        self.in_project_fx = LinearProjection(hidden_dim, hidden_dim)
        self.in_project_slice = LinearProjection(self.dim_head, num_slices)
        self.qkv = LinearProjection(self.dim_head, self.dim_head * 3, bias=False)
        self.proj = LinearProjection(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def create_slices(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = x.shape
        fx_mid = self.in_project_fx(x).view(batch_size, num_tokens, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).view(batch_size, num_tokens, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = F.softmax(slice_logits, dim=-1)
        if attn_mask is not None:
            mask = attn_mask[:, None, :, None].float()
            slice_weights = slice_weights * mask
        slice_norm = slice_weights.sum(dim=2, keepdim=False).unsqueeze(-1)
        slice_tokens = torch.einsum("bhnc,bhns->bhsc", fx_mid, slice_weights) / (slice_norm + 1e-5)
        return slice_tokens, slice_weights

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, coords: torch.Tensor | None = None) -> torch.Tensor:
        slice_tokens, slice_weights = self.create_slices(x, attn_mask=attn_mask)
        qkv = self.qkv(slice_tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        if self.rope_dim > 0 and coords is not None:
            slice_mass = slice_weights.sum(dim=2)
            centroids = torch.einsum("bhns,bnp->bhsp", slice_weights, coords)
            centroids = centroids / (slice_mass.unsqueeze(-1) + 1e-8)
            q = apply_rope_nd(q, centroids, self.rope_dim)
            k = apply_rope_nd(k, centroids, self.rope_dim)
        out_slice = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out_x = torch.einsum("bhsc,bhns->bhnc", out_slice, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.hidden_dim)
        return self.proj_dropout(self.proj(out_x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int | float,
        num_slices: int,
        dropout: float = 0.0,
        rope_dim: int = 0,
    ):
        super().__init__()
        mlp_hidden_dim = int(math.ceil(hidden_dim * mlp_expansion_factor))
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = TransolverAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            dropout=dropout,
            rope_dim=rope_dim,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = UpActDownMlp(hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, coords: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attn_mask=attn_mask, coords=coords)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int | float,
        num_slices: int,
        dropout: float = 0.0,
        rope_dim: int = 0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_expansion_factor=mlp_expansion_factor,
                    num_slices=num_slices,
                    dropout=dropout,
                    rope_dim=rope_dim,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, coords: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, coords=coords)
        return x


class ReferenceTransolver(nn.Module):
    """Local grouped-domain Transolver port based on the active milieu_cfd contract."""

    def __init__(
        self,
        *,
        space_dim: int,
        surface_input_dim: int,
        surface_output_dim: int,
        volume_input_dim: int = 0,
        volume_output_dim: int = 0,
        n_layers: int = 3,
        n_hidden: int = 192,
        dropout: float = 0.0,
        n_head: int = 3,
        mlp_ratio: int = 4,
        slice_num: int = 96,
        rope_dim: int = 0,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.surface_output_dim = surface_output_dim
        self.volume_output_dim = volume_output_dim
        self.surface_extra_dim = max(0, surface_input_dim - space_dim)
        self.volume_extra_dim = max(0, volume_input_dim - space_dim)

        self.pos_embed = ContinuousSincosEmbed(hidden_dim=n_hidden, input_dim=space_dim)
        self.surface_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)
        self.volume_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)
        if self.surface_extra_dim > 0:
            self.project_surface_features = LinearProjection(self.surface_extra_dim, n_hidden)
        else:
            self.project_surface_features = None
        if self.volume_extra_dim > 0:
            self.project_volume_features = LinearProjection(self.volume_extra_dim, n_hidden)
        else:
            self.project_volume_features = None

        self.placeholder = nn.Parameter(torch.rand(1, 1, n_hidden) / n_hidden)
        self.backbone = Transformer(
            depth=n_layers,
            hidden_dim=n_hidden,
            num_heads=n_head,
            mlp_expansion_factor=mlp_ratio,
            num_slices=slice_num,
            dropout=dropout,
            rope_dim=rope_dim,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        self.out = LinearProjection(n_hidden, surface_output_dim + volume_output_dim)
        self.n_hidden = n_hidden

    def _group_hidden(self, x: torch.Tensor, projector: nn.Module | None, bias: nn.Module) -> torch.Tensor:
        pos = x[:, :, : self.space_dim]
        hidden = self.pos_embed(pos)
        if projector is not None and x.shape[-1] > self.space_dim:
            hidden = hidden + projector(x[:, :, self.space_dim :])
        return bias(hidden)

    def forward(
        self,
        *,
        surface_x: torch.Tensor,
        surface_mask: torch.Tensor,
        volume_x: torch.Tensor | None = None,
        volume_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        surface_hidden = self._group_hidden(surface_x, self.project_surface_features, self.surface_bias)
        hidden_parts = [surface_hidden]
        mask_parts = [surface_mask]
        coord_parts = [surface_x[:, :, : self.space_dim]]
        volume_hidden = None
        if volume_x is not None and volume_mask is not None and self.volume_output_dim > 0:
            volume_hidden = self._group_hidden(volume_x, self.project_volume_features, self.volume_bias)
            hidden_parts.append(volume_hidden)
            mask_parts.append(volume_mask)
            coord_parts.append(volume_x[:, :, : self.space_dim])
        hidden = torch.cat(hidden_parts, dim=1) + self.placeholder
        attn_mask = torch.cat(mask_parts, dim=1)
        coords = torch.cat(coord_parts, dim=1)
        hidden = self.backbone(hidden, attn_mask=attn_mask, coords=coords)
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

        return {
            "surface_hidden": hidden[:, :surface_tokens],
            "surface_preds": surface_preds,
            "volume_hidden": None if volume_hidden is None else hidden[:, surface_tokens:],
            "volume_preds": volume_preds,
        }
