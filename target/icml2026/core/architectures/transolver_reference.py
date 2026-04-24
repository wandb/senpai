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
    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, dropout: float = 0.0, num_kv_heads: int = 0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads
        self.num_slices = num_slices
        self.dropout = dropout
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.kv_group_size = num_heads // self.num_kv_heads
        self.use_gqa = self.num_kv_heads < num_heads

        self.temperature = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.5))
        self.in_project_x = LinearProjection(hidden_dim, hidden_dim)
        self.in_project_slice = LinearProjection(self.dim_head, num_slices)
        self.proj = LinearProjection(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)

        if self.use_gqa:
            kv_hidden = self.num_kv_heads * self.dim_head
            self.in_project_fx = LinearProjection(hidden_dim, kv_hidden)
            self.q_proj = LinearProjection(self.dim_head, self.dim_head, bias=False)
            self.kv_proj = LinearProjection(self.dim_head, self.dim_head * 2, bias=False)
        else:
            self.in_project_fx = LinearProjection(hidden_dim, hidden_dim)
            self.qkv = LinearProjection(self.dim_head, self.dim_head * 3, bias=False)

    def create_slices(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = x.shape
        fx_mid = self.in_project_fx(x).view(batch_size, num_tokens, self.num_heads if not self.use_gqa else self.num_kv_heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).view(batch_size, num_tokens, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = F.softmax(slice_logits, dim=-1)
        if attn_mask is not None:
            mask = attn_mask[:, None, :, None].float()
            slice_weights = slice_weights * mask
        if not self.use_gqa:
            slice_norm = slice_weights.sum(dim=2, keepdim=False).unsqueeze(-1)
            slice_tokens = torch.einsum("bhnc,bhns->bhsc", fx_mid, slice_weights) / (slice_norm + 1e-5)
            return slice_tokens, slice_weights
        return (x_mid, fx_mid), slice_weights

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        slices_out, slice_weights = self.create_slices(x, attn_mask=attn_mask)

        if not self.use_gqa:
            slice_tokens = slices_out
            qkv = self.qkv(slice_tokens)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            x_mid, fx_mid = slices_out
            q_norm = slice_weights.sum(dim=2).unsqueeze(-1)
            q_slice_tokens = torch.einsum("bhnc,bhns->bhsc", x_mid, slice_weights) / (q_norm + 1e-5)
            q = self.q_proj(q_slice_tokens)

            kv_slice_weights = slice_weights.view(
                slice_weights.shape[0], self.num_kv_heads, self.kv_group_size,
                slice_weights.shape[2], slice_weights.shape[3],
            ).mean(dim=2)
            kv_norm = kv_slice_weights.sum(dim=2).unsqueeze(-1)
            kv_slice_tokens = torch.einsum("bhnc,bhns->bhsc", fx_mid, kv_slice_weights) / (kv_norm + 1e-5)
            kv = self.kv_proj(kv_slice_tokens)
            k, v = kv.chunk(2, dim=-1)
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

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
        num_kv_heads: int = 0,
    ):
        super().__init__()
        mlp_hidden_dim = int(math.ceil(hidden_dim * mlp_expansion_factor))
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = TransolverAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            dropout=dropout,
            num_kv_heads=num_kv_heads,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = UpActDownMlp(hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attn_mask=attn_mask)
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
        num_kv_heads: int = 0,
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
                    num_kv_heads=num_kv_heads,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
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
        n_kv_head: int = 0,
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
            num_kv_heads=n_kv_head,
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
        volume_hidden = None
        if volume_x is not None and volume_mask is not None and self.volume_output_dim > 0:
            volume_hidden = self._group_hidden(volume_x, self.project_volume_features, self.volume_bias)
            hidden_parts.append(volume_hidden)
            mask_parts.append(volume_mask)
        hidden = torch.cat(hidden_parts, dim=1) + self.placeholder
        attn_mask = torch.cat(mask_parts, dim=1)
        hidden = self.backbone(hidden, attn_mask=attn_mask)
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
