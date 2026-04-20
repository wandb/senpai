# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:, None, :, :]
    out = torch.view_as_real(x_complex * freqs).flatten(start_dim=3)
    return out.type_as(x)


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, dim: int, ndim: int, max_wavelength: int = 10_000):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        ndim_padding = dim % ndim
        dim_per_axis = (dim - ndim_padding) // ndim
        sincos_padding = dim_per_axis % 2
        self.padding = ndim_padding + sincos_padding * ndim
        effective_dim_per_axis = (dim - self.padding) // ndim
        arange = torch.arange(0, effective_dim_per_axis, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (arange / effective_dim_per_axis))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        out = coords.unsqueeze(-1) * self.omega
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1).flatten(start_dim=-2)
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, padding], dim=-1)
        return emb


class RopeFrequency(nn.Module):
    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10_000.0):
        super().__init__()
        ndim_padding = dim % ndim
        dim_per_axis = (dim - ndim_padding) // ndim
        sincos_padding = dim_per_axis % 2
        self.padding = ndim_padding + sincos_padding * ndim
        effective_dim_per_axis = (dim - self.padding) // ndim
        arange = torch.arange(0, effective_dim_per_axis, 2, dtype=torch.float32)
        self.ndim = ndim
        self.register_buffer("omega", 1.0 / max_wavelength ** (arange / effective_dim_per_axis))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = coords.float().unsqueeze(-1) * self.omega
        out = out.flatten(start_dim=-2)
        if self.padding > 0:
            padding = torch.zeros(*out.shape[:-1], self.padding // 2, device=coords.device, dtype=out.dtype)
            out = torch.cat([out, padding], dim=-1)
        return torch.polar(torch.ones_like(out), out)


class Mlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DotProductAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(q, freqs)
        k = rope(k, freqs)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        return self.proj(out)


class AnchorAttention(DotProductAttention):
    def forward(self, x: torch.Tensor, freqs: torch.Tensor, num_anchor_tokens: int | None = None) -> torch.Tensor:
        if num_anchor_tokens is None or num_anchor_tokens >= x.shape[1]:
            return super().forward(x, freqs)
        anchors = x[:, :num_anchor_tokens]
        queries = x[:, num_anchor_tokens:]

        b, qa, _ = anchors.shape
        _, qq, _ = queries.shape
        q_all = self.qkv(torch.cat([anchors, queries], dim=1))
        q_all = q_all.view(b, qa + qq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = q_all[0]
        k = q_all[1][:, :, :num_anchor_tokens]
        v = q_all[2][:, :, :num_anchor_tokens]
        q = rope(q, freqs)
        k = rope(k, freqs[:, :num_anchor_tokens])
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, qa + qq, self.dim)
        return self.proj(out)


class SharedweightsSplitattnAttention(DotProductAttention):
    def forward(self, x: torch.Tensor, freqs: torch.Tensor, split_size: list[int]) -> torch.Tensor:
        if len(split_size) == 1:
            return super().forward(x, freqs)
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(q, freqs)
        k = rope(k, freqs)
        q_parts = q.split(split_size, dim=2)
        k_parts = k.split(split_size, dim=2)
        v_parts = v.split(split_size, dim=2)
        out_parts = [F.scaled_dot_product_attention(qp, kp, vp) for qp, kp, vp in zip(q_parts, k_parts, v_parts, strict=True)]
        out = torch.cat(out_parts, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        return self.proj(out)


class SharedweightsCrossattnAttention(DotProductAttention):
    def forward(self, x: torch.Tensor, freqs: torch.Tensor, split_size: list[int]) -> torch.Tensor:
        if len(split_size) == 1:
            return x
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(q, freqs)
        k = rope(k, freqs)
        q_parts = q.split(split_size, dim=2)
        k_parts = k.split(split_size, dim=2)
        v_parts = v.split(split_size, dim=2)
        out_parts: list[torch.Tensor] = []
        for i in range(len(split_size)):
            partner = 1 - i if len(split_size) == 2 else i
            if len(split_size) == 2:
                out_parts.append(F.scaled_dot_product_attention(q_parts[i], k_parts[partner], v_parts[partner]))
            else:
                out_parts.append(q_parts[i])
        out = torch.cat(out_parts, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        return self.proj(out)


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, q_freqs: torch.Tensor, k_freqs: torch.Tensor) -> torch.Tensor:
        b, q_len, _ = q.shape
        kv_len = kv.shape[1]
        q = self.q(q).view(b, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(kv).view(b, kv_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = rope(q, q_freqs)
        k = rope(k, k_freqs)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, q_len, self.dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_ctor: type[nn.Module] = DotProductAttention):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attn_ctor(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor, attn_kwargs: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), **(attn_kwargs or {}))
        x = x + self.mlp(self.norm2(x))
        return x


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = PerceiverAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_kwargs: dict | None = None) -> torch.Tensor:
        q = q + self.attn(q=self.norm_q(q), kv=self.norm_kv(kv), **(attn_kwargs or {}))
        q = q + self.mlp(self.norm2(q))
        return q


class SupernodePoolingPosonly(nn.Module):
    def __init__(self, hidden_dim: int, ndim: int, k: int = 16, mode: str = "relpos"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.k = k
        self.mode = mode
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        if mode == "relpos":
            self.rel_pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim + 1)
            message_input_dim = hidden_dim
        elif mode == "abspos":
            self.rel_pos_embed = None
            message_input_dim = hidden_dim * 2
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")
        self.message = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def _messages_for_one(
        self,
        input_pos: torch.Tensor,
        supernode_local_idx: torch.Tensor,
    ) -> torch.Tensor:
        super_pos = input_pos[supernode_local_idx]
        dists = torch.cdist(super_pos, input_pos)
        k = min(self.k, input_pos.shape[0])
        neighbor_idx = dists.topk(k, largest=False).indices
        neighbor_pos = input_pos[neighbor_idx]
        super_pos_expanded = super_pos[:, None, :]
        if self.mode == "relpos":
            rel = super_pos_expanded - neighbor_pos
            mag = rel.norm(dim=-1, keepdim=True)
            message_in = self.rel_pos_embed(torch.cat([rel, mag], dim=-1))
        else:
            src = self.pos_embed(neighbor_pos)
            dst = self.pos_embed(super_pos_expanded.expand_as(neighbor_pos))
            message_in = torch.cat([src, dst], dim=-1)
        message = self.message(message_in).mean(dim=1)
        return self.proj(torch.cat([message, self.pos_embed(super_pos)], dim=-1))

    def forward(
        self,
        input_pos: torch.Tensor,
        supernode_idx: torch.Tensor,
        batch_idx: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if batch_idx is None:
            return self._messages_for_one(input_pos[:, : self.ndim], supernode_idx).unsqueeze(0)

        outputs: list[torch.Tensor] = []
        counts = torch.bincount(batch_idx, minlength=batch_size)
        num_supernodes = supernode_idx.numel() // batch_size
        point_offset = 0
        super_offset = 0
        for b in range(batch_size):
            num_points = int(counts[b].item())
            points = input_pos[point_offset : point_offset + num_points, : self.ndim]
            local_supernodes = supernode_idx[super_offset : super_offset + num_supernodes] - point_offset
            outputs.append(self._messages_for_one(points, local_supernodes))
            point_offset += num_points
            super_offset += num_supernodes
        return torch.stack(outputs, dim=0)


class ABUPTReference(nn.Module):
    """Local AB-UPT port with bridge-aligned batched geometry flattening."""

    def __init__(
        self,
        *,
        space_dim: int,
        surface_output_dim: int,
        volume_output_dim: int = 0,
        hidden_dim: int = 192,
        num_heads: int = 3,
        geometry_depth: int = 1,
        blocks: str = "pscscs",
        num_surface_blocks: int = 2,
        num_volume_blocks: int = 2,
        supernode_k: int = 16,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.surface_output_dim = surface_output_dim
        self.volume_output_dim = volume_output_dim
        self.rope = RopeFrequency(dim=hidden_dim // num_heads, ndim=space_dim)
        self.encoder = SupernodePoolingPosonly(hidden_dim=hidden_dim, ndim=space_dim, k=supernode_k, mode="relpos")
        self.geometry_blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(geometry_depth)])
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=space_dim)
        self.surface_bias = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.volume_bias = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

        self.blocks = nn.ModuleList()
        for block in blocks:
            if block == "s":
                ctor = partial(TransformerBlock, attn_ctor=SharedweightsSplitattnAttention)
            elif block == "c":
                ctor = partial(TransformerBlock, attn_ctor=SharedweightsCrossattnAttention)
            elif block == "p":
                ctor = PerceiverBlock
            else:
                raise NotImplementedError(f"Unknown AB-UPT block '{block}'")
            self.blocks.append(ctor(hidden_dim, num_heads))
        self.surface_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, attn_ctor=AnchorAttention) for _ in range(num_surface_blocks)]
        )
        self.volume_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, attn_ctor=AnchorAttention) for _ in range(num_volume_blocks)]
        )
        self.surface_decoder = nn.Linear(hidden_dim, surface_output_dim)
        self.volume_decoder = nn.Linear(hidden_dim, volume_output_dim) if volume_output_dim > 0 else None

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)

    def forward(
        self,
        *,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor | None,
        surface_anchor_position: torch.Tensor,
        volume_anchor_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size = surface_anchor_position.shape[0]
        num_supernodes = geometry_supernode_idx.numel() // batch_size
        geometry_encoding = self.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
            batch_size=batch_size,
        )
        geometry_supernode_position = geometry_position[geometry_supernode_idx].view(batch_size, num_supernodes, self.space_dim)
        geometry_rope = self.rope(geometry_supernode_position)
        for block in self.geometry_blocks:
            geometry_encoding = block(geometry_encoding, attn_kwargs={"freqs": geometry_rope})

        surface_tokens = self.surface_bias(self.pos_embed(surface_anchor_position))
        if volume_anchor_position is not None:
            volume_tokens = self.volume_bias(self.pos_embed(volume_anchor_position))
            token_seq = torch.cat([surface_tokens, volume_tokens], dim=1)
            split_size = [surface_anchor_position.shape[1], volume_anchor_position.shape[1]]
            rope_all = self.rope(torch.cat([surface_anchor_position, volume_anchor_position], dim=1))
        else:
            volume_tokens = None
            token_seq = surface_tokens
            split_size = [surface_anchor_position.shape[1]]
            rope_all = self.rope(surface_anchor_position)

        geometry_attn_kwargs = {"q_freqs": rope_all, "k_freqs": geometry_rope}
        for block in self.blocks:
            if isinstance(block, PerceiverBlock):
                token_seq = block(q=token_seq, kv=geometry_encoding, attn_kwargs=geometry_attn_kwargs)
            else:
                token_seq = block(token_seq, attn_kwargs={"freqs": rope_all, "split_size": split_size})

        if volume_anchor_position is not None:
            surface_tokens, volume_tokens = token_seq.split(split_size, dim=1)
        else:
            surface_tokens = token_seq

        surface_rope = self.rope(surface_anchor_position)
        for block in self.surface_blocks:
            surface_tokens = block(surface_tokens, attn_kwargs={"freqs": surface_rope})
        surface_preds = self.surface_decoder(surface_tokens)

        volume_preds = None
        if volume_anchor_position is not None and volume_tokens is not None and self.volume_decoder is not None:
            volume_rope = self.rope(volume_anchor_position)
            for block in self.volume_blocks:
                volume_tokens = block(volume_tokens, attn_kwargs={"freqs": volume_rope})
            volume_preds = self.volume_decoder(volume_tokens)

        return {
            "surface_preds": surface_preds,
            "volume_preds": volume_preds,
        }
