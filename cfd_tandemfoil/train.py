# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Train Transolver with structured benchmark splits.

Reads a split manifest and stats file produced by data/split.py,
then trains with four separate val tracks:
  val_in_dist          — raceCar_single random holdout (interpolation sanity)
  val_tandem_transfer  — raceCar_tandem Part2 (unseen tandem front foil)
  val_ood_cond         — cruise Part1+3 frontier 20% (extreme AoA/gap/stagger)
  val_ood_re           — cruise Part2 Re=4.445M (fully OOD Reynolds number)

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"

KNOWN LIMITATIONS (inherited from read-only prepare.py):
  - Only NACA[0] and AoA[0] are encoded in x. Foil 2 is implicit in dsdf/saf.
  - SURFACE_IDS=(5,6) misses boundary ID 7 (foil 2 surface in tandem data).
    Tandem surface loss is therefore underweighted.
"""

import os
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from dataclasses import dataclass, asdict
from einops import rearrange
from timm.layers import trunc_normal_
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
import simple_parsing as sp

from data.utils import visualize
from data.prepare_multi import X_DIM, pad_collate, load_data, VAL_SPLIT_NAMES, set_extra_features_mode, get_extra_dim

torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# Transolver model (inlined so students can
# modify architecture and training script in a single file)
# ---------------------------------------------------------------------------

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class GatedMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, act='gelu'):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.gate_proj = nn.Linear(n_input, n_hidden)
        self.up_proj = nn.Linear(n_input, n_hidden)
        self.down_proj = nn.Linear(n_hidden, n_output)
        self.act = act_fn()
    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.act(self.up_proj(x)))


class GatedMLP2(nn.Module):
    """GatedMLP with a residual second gated layer."""
    def __init__(self, n_input, n_hidden, n_output, act='gelu'):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.gate1 = nn.Linear(n_input, n_hidden)
        self.up1 = nn.Linear(n_input, n_hidden)
        self.gate2 = nn.Linear(n_hidden, n_hidden)
        self.up2 = nn.Linear(n_hidden, n_hidden)
        self.down = nn.Linear(n_hidden, n_output)
        self.act = act_fn()

    def forward(self, x):
        h = torch.sigmoid(self.gate1(x)) * self.act(self.up1(x))
        h = h + torch.sigmoid(self.gate2(h)) * self.act(self.up2(h))
        return self.down(h)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError
        act_fn = ACTIVATION[act]
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class DomainLayerNorm(nn.Module):
    """Domain-specific LayerNorm: separate weight/bias for single-foil vs tandem (Phase 3 R10)."""
    def __init__(self, dim, zeroinit=False):
        super().__init__()
        self.ln_single = nn.LayerNorm(dim)
        self.ln_tandem = nn.LayerNorm(dim)
        if not zeroinit:
            self.ln_tandem.weight.data.copy_(self.ln_single.weight.data)
            self.ln_tandem.bias.data.copy_(self.ln_single.bias.data)
        # zeroinit: tandem defaults to weight=1, bias=0 (LayerNorm default) — identical to copy

    def forward(self, x, is_tandem=None):
        if is_tandem is None:
            return self.ln_single(x)
        mask_t = is_tandem.view(-1, 1, 1).expand_as(x)
        return torch.where(mask_t, self.ln_tandem(x), self.ln_single(x))


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics attention for irregular meshes in 1D/2D/3D space."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
                 linear_no_attention=False, learned_kernel=False,
                 decouple_slice=False, zone_temp=False, prog_slices=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.tandem_temp_offset = nn.Parameter(torch.zeros(1, heads, 1, 1))
        self.linear_no_attention = linear_no_attention
        self.learned_kernel = learned_kernel
        self.decouple_slice = decouple_slice
        self.zone_temp = zone_temp
        self.prog_slices = prog_slices
        if prog_slices:
            # Buffer for masking inactive slices; updated per-epoch by training loop
            self.register_buffer('slice_mask', torch.zeros(slice_num))

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        if decouple_slice:
            # Separate slice projection for tandem samples
            self.in_project_slice_tandem = nn.Linear(dim_head, slice_num)
            torch.nn.init.orthogonal_(self.in_project_slice_tandem.weight)
        if zone_temp:
            # Zone-aware temperature: learned offset from [is_tandem, gap_mag, re_feat]
            self.zone_temp_proj = nn.Linear(3, heads)
            nn.init.zeros_(self.zone_temp_proj.weight)
            nn.init.zeros_(self.zone_temp_proj.bias)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.slice_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.attn_scale = nn.Parameter(torch.ones(1, self.heads, 1, 1) * 10.0)
        if learned_kernel:
            self.kernel_mlp = nn.Sequential(
                nn.Linear(2 * dim_head, dim_head), nn.GELU(),
                nn.Linear(dim_head, 1),
            )

    def forward(self, x, spatial_bias=None, tandem_mask=None, zone_features=None):
        bsz, num_points, _ = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .reshape(bsz, num_points, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(bsz, num_points, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        temp = self.temperature
        if self.zone_temp and zone_features is not None:
            # zone_features: [B, 3] → per-head offset [B, heads] → [B, heads, 1, 1]
            zone_offset = self.zone_temp_proj(zone_features).reshape(bsz, self.heads, 1, 1)
            temp = temp + zone_offset
        if tandem_mask is not None:
            temp = (temp + self.tandem_temp_offset * tandem_mask).clamp(min=1e-4)
        temp = temp.clamp(min=1e-4)
        if self.decouple_slice and tandem_mask is not None:
            std_logits = self.in_project_slice(x_mid) / temp
            tan_logits = self.in_project_slice_tandem(x_mid) / temp
            is_tan = (tandem_mask > 0.5)  # [B, 1, 1, 1]
            slice_logits = torch.where(is_tan.expand_as(std_logits), tan_logits, std_logits)
        else:
            slice_logits = self.in_project_slice(x_mid) / temp
        if spatial_bias is not None:
            slice_logits = slice_logits + 0.1 * spatial_bias.unsqueeze(1)
        if self.prog_slices:
            # Apply slice mask: 0 for active slices, -1e9 for inactive (updated each epoch)
            slice_logits = slice_logits + self.slice_mask
        slice_weights = self.softmax(slice_logits)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        if self.linear_no_attention:
            out_slice_token = slice_token
        else:
            q_slice_token = self.to_q(slice_token)
            slice_token_kv = slice_token.mean(dim=1, keepdim=True)
            k_slice_token = self.to_k(slice_token_kv).expand(-1, self.heads, -1, -1)
            v_slice_token = self.to_v(slice_token_kv).expand(-1, self.heads, -1, -1)
            if self.learned_kernel:
                B, H, S, D = q_slice_token.shape
                q_exp = q_slice_token.unsqueeze(-2).expand(B, H, S, S, D)
                k_exp = k_slice_token.unsqueeze(-3).expand(B, H, S, S, D)
                qk_cat = torch.cat([q_exp, k_exp], dim=-1)
                attn_logits = self.kernel_mlp(qk_cat).squeeze(-1)
                attn_weights = F.softmax(attn_logits, dim=-1)
            else:
                q_norm = F.normalize(q_slice_token, dim=-1)
                k_norm = F.normalize(k_slice_token, dim=-1)
                attn_logits = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * self.attn_scale
                attn_weights = F.softmax(attn_logits, dim=-1)
            out_slice_token = torch.matmul(attn_weights, v_slice_token)
            out_slice_token = out_slice_token + self.slice_residual_scale * slice_token

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        linear_no_attention=False,
        learned_kernel=False,
        field_decoder=False,
        adaln_output=False,
        soft_moe=False,
        adaln_all=False,
        adaln_cond_dim=2,
        adaln_zero_init=True,
        film_cond=False,
        decouple_slice=False,
        zone_temp=False,
        domain_layernorm=False,
        dln_zeroinit=False,
        domain_velhead=False,
        prog_slices=False,
        pressure_first=False,
        pressure_no_detach=False,
        pressure_deep=False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.field_decoder = field_decoder
        self.domain_velhead = domain_velhead
        self.pressure_first = pressure_first
        self.pressure_no_detach = pressure_no_detach
        self.adaln_output = adaln_output
        self.soft_moe = soft_moe
        self.adaln_all = adaln_all
        self.film_cond = film_cond
        self.domain_layernorm = domain_layernorm
        _LN = (lambda d: DomainLayerNorm(d, zeroinit=dln_zeroinit)) if domain_layernorm else nn.LayerNorm
        self.ln_1 = _LN(hidden_dim)
        self.attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            linear_no_attention=linear_no_attention,
            learned_kernel=learned_kernel,
            decouple_slice=decouple_slice,
            zone_temp=zone_temp,
            prog_slices=prog_slices,
        )
        if adaln_all:
            # AdaLN-Zero: cond → (scale1, bias1, scale2, bias2) for ln_1 and ln_2
            self.adaln_net = nn.Sequential(
                nn.Linear(adaln_cond_dim, 128), nn.SiLU(),
                nn.Linear(128, hidden_dim * 4),
            )
            if adaln_zero_init:
                nn.init.zeros_(self.adaln_net[-1].weight)
                nn.init.zeros_(self.adaln_net[-1].bias)
        if film_cond:
            # FiLM: cond → (gamma, beta) applied after SE layer
            self.film_net = nn.Sequential(
                nn.Linear(2, 64), nn.SiLU(),
                nn.Linear(64, hidden_dim * 2),
            )
            nn.init.zeros_(self.film_net[-1].weight)
            nn.init.zeros_(self.film_net[-1].bias)
        self.ln_2 = _LN(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        self.spatial_bias = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, slice_num),
        )
        nn.init.zeros_(self.spatial_bias[-1].weight)
        nn.init.zeros_(self.spatial_bias[-1].bias)
        self.ln_1_post = _LN(hidden_dim)
        self.ln_2_post = _LN(hidden_dim)
        self.se_fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.se_fc2 = nn.Linear(hidden_dim // 4, hidden_dim)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.se_fc2.bias)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            if soft_moe:
                self.gate_net = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Softmax(dim=-1))
                self.expert1 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
                )
                self.expert2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
                )
            elif pressure_first:
                # Pressure-first: predict p, then condition v on p (takes priority over domain_velhead/field_decoder)
                if pressure_deep:
                    self.pres_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                        nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    )
                else:
                    self.pres_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 1)
                    )
                # Velocity head conditioned on predicted pressure: input is hidden_dim + 1
                self.vel_head_conditioned = nn.Sequential(
                    nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
                )
            elif domain_velhead:
                # Domain-specific output heads: separate for single-foil vs tandem
                self.velhead_single = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
                )
                self.velhead_tandem = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
                )
            elif field_decoder:
                self.vel_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
                )
                self.pres_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 1)
                )
            elif adaln_output:
                self.cond_net = nn.Sequential(nn.Linear(2, hidden_dim * 2), nn.GELU())
                self.mlp2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
                )
            else:
                self.mlp2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
                )

    def forward(self, fx, raw_xy=None, tandem_mask=None, condition=None, zone_features=None):
        sb = self.spatial_bias(raw_xy) if raw_xy is not None else None
        # DomainLayerNorm helper: pass is_tandem when enabled, else plain call
        dln_it = (tandem_mask.view(-1) > 0.5) if (self.domain_layernorm and tandem_mask is not None) else None
        if self.domain_layernorm:
            def _ln(m, x): return m(x, is_tandem=dln_it)
        else:
            def _ln(m, x): return m(x)
        if self.adaln_all and condition is not None:
            cond_out = self.adaln_net(condition)  # [B, H*4]
            s1, b1, s2, b2 = cond_out.chunk(4, dim=-1)  # each [B, H]
            fx_norm = _ln(self.ln_1, fx) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1)
            fx = _ln(self.ln_1_post, self.attn(fx_norm, spatial_bias=sb, tandem_mask=tandem_mask, zone_features=zone_features) + fx)
            fx_norm = _ln(self.ln_2, fx) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1)
            fx = _ln(self.ln_2_post, self.mlp(fx_norm) + fx)
        else:
            fx = _ln(self.ln_1_post, self.attn(_ln(self.ln_1, fx), spatial_bias=sb, tandem_mask=tandem_mask, zone_features=zone_features) + fx)
            fx = _ln(self.ln_2_post, self.mlp(_ln(self.ln_2, fx)) + fx)
        se = fx.mean(dim=1, keepdim=True)
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        fx = fx * se
        if self.film_cond and condition is not None:
            film_out = self.film_net(condition)  # [B, H*2]
            gamma, beta = film_out.chunk(2, dim=-1)  # each [B, H]
            fx = gamma.unsqueeze(1) * fx + beta.unsqueeze(1)
        if self.last_layer:
            fx_ln = self.ln_3(fx)
            if self.soft_moe:
                gate = self.gate_net(fx_ln)  # [B, N, 2]
                return gate[:, :, 0:1] * self.expert1(fx_ln) + gate[:, :, 1:2] * self.expert2(fx_ln)
            elif self.pressure_first:
                # Pressure-first: predict p, then condition v on p
                p_pred = self.pres_head(fx_ln)  # [B, N, 1]
                p_cond = p_pred if self.pressure_no_detach else p_pred.detach()
                vel_input = torch.cat([fx_ln, p_cond], dim=-1)  # [B, N, H+1]
                v_pred = self.vel_head_conditioned(vel_input)  # [B, N, 2]
                return torch.cat([v_pred, p_pred], dim=-1)
            elif self.domain_velhead:
                out_s = self.velhead_single(fx_ln)
                out_t = self.velhead_tandem(fx_ln)
                if tandem_mask is not None:
                    is_tan = (tandem_mask.view(-1) > 0.5).view(-1, 1, 1)
                    return torch.where(is_tan.expand_as(out_s), out_t, out_s)
                return out_s
            elif self.field_decoder:
                return torch.cat([self.vel_head(fx_ln), self.pres_head(fx_ln)], dim=-1)
            elif self.adaln_output and condition is not None:
                cond = self.cond_net(condition)  # [B, 2*H]
                scale, shift = cond.chunk(2, dim=-1)  # [B, H]
                fx_ln = fx_ln * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                return self.mlp2(fx_ln)
            else:
                return self.mlp2(fx_ln)
        return fx


class Transolver(nn.Module):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        output_fields: list[str] | None = None,
        output_dims: list[int] | None = None,
        linear_no_attention=False,
        learned_kernel=False,
        field_decoder=False,
        adaln_output=False,
        soft_moe=False,
        uncertainty_loss=False,
        adaln_all_blocks=False,
        adaln_4cond=False,
        adaln_nozero=False,
        film_cond=False,
        adaln_decouple=False,
        adaln_zone_temp=False,
        domain_layernorm=False,
        dln_zeroinit=False,
        domain_velhead=False,
        prog_slices=False,
        pressure_first=False,
        pressure_no_detach=False,
        pressure_deep=False,
    ):
        super().__init__()
        self.__name__ = "UniPDE_3D"
        self.pressure_first = pressure_first
        self.ref = ref
        self.unified_pos = unified_pos
        self.adaln_output = adaln_output
        self.adaln_all_blocks = adaln_all_blocks
        self.adaln_4cond = adaln_4cond
        self.film_cond = film_cond
        self.adaln_zone_temp = adaln_zone_temp
        if output_fields is None or output_dims is None:
            raise ValueError("output_fields and output_dims must be provided")
        if len(output_fields) != len(output_dims):
            raise ValueError("output_fields and output_dims must have the same length")
        if out_dim != sum(output_dims):
            raise ValueError("out_dim must equal sum(output_dims)")
        self.output_fields = output_fields
        self.output_dims = output_dims
        if uncertainty_loss:
            self.log_sigma_vol = nn.Parameter(torch.zeros(1))
            self.log_sigma_surf_ux = nn.Parameter(torch.zeros(1))
            self.log_sigma_surf_uy = nn.Parameter(torch.zeros(1))
            self.log_sigma_surf_p = nn.Parameter(torch.zeros(1))

        if self.unified_pos:
            self.preprocess = MLP(
                fun_dim + self.ref * self.ref * self.ref,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = GatedMLP2(fun_dim + space_dim, n_hidden * 2, n_hidden)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.feature_cross = nn.Linear(fun_dim + space_dim, fun_dim + space_dim, bias=False)
        nn.init.eye_(self.feature_cross.weight)  # start as identity
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(idx == n_layers - 1),
                    linear_no_attention=linear_no_attention,
                    learned_kernel=learned_kernel,
                    field_decoder=field_decoder if (idx == n_layers - 1) else False,
                    adaln_output=adaln_output if (idx == n_layers - 1) else False,
                    soft_moe=soft_moe if (idx == n_layers - 1) else False,
                    adaln_all=adaln_all_blocks,
                    adaln_cond_dim=4 if adaln_4cond else 2,
                    adaln_zero_init=not adaln_nozero,
                    film_cond=film_cond,
                    decouple_slice=adaln_decouple,
                    zone_temp=adaln_zone_temp,
                    domain_layernorm=domain_layernorm,
                    dln_zeroinit=dln_zeroinit,
                    domain_velhead=domain_velhead if (idx == n_layers - 1) else False,
                    prog_slices=prog_slices,
                    pressure_first=pressure_first if (idx == n_layers - 1) else False,
                    pressure_no_detach=pressure_no_detach,
                    pressure_deep=pressure_deep,
                )
                for idx in range(n_layers)
            ]
        )
        # Separate pressure pathway (pressure_separate_last_block):
        # Independent MLP + pres_head that processes shared hidden features
        self._pressure_separate = False  # set from Config after construction
        self.pressure_sep_mlp = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden * 2), nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, 1),
        )
        self.initialize_weights()
        self.out_skip = nn.Linear(n_hidden, out_dim)
        nn.init.zeros_(self.out_skip.weight)
        nn.init.zeros_(self.out_skip.bias)
        self.skip_gate = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        nn.init.constant_(self.skip_gate[0].bias, -2.0)  # starts nearly closed
        self.placeholder_scale = nn.Parameter(torch.ones(n_hidden))
        self.placeholder_shift = nn.Parameter(torch.zeros(n_hidden))
        self.re_head = nn.Sequential(nn.Linear(n_hidden, 32), nn.GELU(), nn.Linear(32, 1))
        self.aoa_head = nn.Sequential(nn.Linear(n_hidden, 32), nn.GELU(), nn.Linear(32, 1))
        self.fourier_freqs_fixed = torch.tensor([0.5, 2.0, 8.0, 32.0])  # non-learnable
        self.fourier_freqs_learned = nn.Parameter(torch.tensor([1.0, 3.0, 6.0, 16.0]))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.weight.dim() >= 2:
                nn.init.orthogonal_(module.weight, gain=1.0)
            else:
                nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_grid(self, my_pos):
        batchsize = my_pos.shape[0]
        device = my_pos.device
        dtype = my_pos.dtype

        gridx = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=dtype)
        gridx = gridx.view(1, self.ref, 1, 1, 1).repeat(batchsize, 1, self.ref, self.ref, 1)
        gridy = torch.linspace(0, 2, self.ref, device=device, dtype=dtype)
        gridy = gridy.view(1, 1, self.ref, 1, 1).repeat(batchsize, self.ref, 1, self.ref, 1)
        gridz = torch.linspace(-4, 4, self.ref, device=device, dtype=dtype)
        gridz = gridz.view(1, 1, 1, self.ref, 1).repeat(batchsize, self.ref, self.ref, 1, 1)
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).reshape(batchsize, self.ref**3, 3)

        pos = torch.sqrt(((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2).sum(dim=-1))
        return pos.reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()

    def _unpack_inputs(self, data, pos=None, condition=None):
        if not isinstance(data, Mapping):
            raise TypeError("Model input must be a Mapping with keys: x, pos, condition")
        x = data.get("x")
        pos = data.get("pos", pos)
        condition = data.get("condition", condition)
        return x, pos, condition

    def _validate_output_dims(self, preds):
        if sum(self.output_dims) != preds.shape[-1]:
            raise ValueError("Sum of output_dims must match preds last dimension")

    def forward(self, data, pos=None, condition=None):
        x, pos, condition = self._unpack_inputs(data, pos=pos, condition=condition)
        if x is None:
            raise ValueError("Missing required input tensor: x")
        if condition is not None:
            raise ValueError("Transolver does not support conditioning inputs")

        # Compute internal condition before feature_cross (indices are stable here)
        use_cond = self.adaln_all_blocks or self.film_cond
        if use_cond:
            cond_2 = x[:, 0, 13:15]  # Re, AoA [B, 2]
            if self.adaln_4cond:
                gap_feat = x[:, 0, 21:22]  # gap feature [B, 1]
                # surf_frac: fraction of nodes near a surface (curvature at index 24)
                surf_frac = (x[:, :, 24].abs() > 0.01).float().mean(dim=1, keepdim=True)  # [B, 1]
                block_condition = torch.cat([cond_2, gap_feat, surf_frac], dim=-1)  # [B, 4]
            else:
                block_condition = cond_2  # [B, 2]
        else:
            block_condition = None

        # Compute zone features for zone-aware temperature
        if self.adaln_zone_temp:
            is_tandem_scalar = (x[:, 0, 21].abs() > 0.01).float()  # [B]
            gap_mag = x[:, 0, 21].abs()  # [B]
            re_feat = x[:, 0, 13]  # [B]
            zone_features = torch.stack([is_tandem_scalar, gap_mag, re_feat], dim=-1)  # [B, 3]
        else:
            zone_features = None

        if self.unified_pos:
            if pos is None:
                raise ValueError("Missing required input tensor: pos")
            new_pos = self.get_grid(pos)
            x = torch.cat((x, new_pos), dim=-1)

        x_cross = x * self.feature_cross(x)
        x = x + 0.1 * x_cross  # residual with small scale
        raw_xy = torch.cat([x[:, :, :2], x[:, :, 24:26]], dim=-1)  # x, y, curvature, dist

        # Detect tandem samples via gap feature (index 21); shape [B,1,1,1] for broadcasting
        is_tandem = (x[:, 0, 21].abs() > 0.01).float()[:, None, None, None]

        fx = self.preprocess(x)
        fx_pre = fx  # save for skip
        fx = fx * self.placeholder_scale[None, None, :] + self.placeholder_shift[None, None, :]

        for block in self.blocks[:-1]:
            fx = block(fx, raw_xy=raw_xy, tandem_mask=is_tandem, condition=block_condition, zone_features=zone_features)

        # Auxiliary Re prediction from pre-output-head hidden representation
        re_pred = self.re_head(fx.mean(dim=1))  # [B, 1]
        aoa_pred = self.aoa_head(fx.mean(dim=1))

        # Last block: use adaln_all condition if enabled, else fallback to adaln_output
        last_condition = block_condition if use_cond else (x[:, 0, 13:15] if self.adaln_output else None)

        if self._pressure_separate and self.pressure_first:
            # Separate pressure pathway: independent MLP processes pre-last features
            fx_for_pressure = fx  # save for separate pressure branch
            p_sep = self.pressure_sep_mlp(fx_for_pressure)  # [B, N, 1]
            # Main last block produces vel only (pressure_first still active but p comes from separate branch)
            fx = self.blocks[-1](fx, raw_xy=raw_xy, tandem_mask=is_tandem, condition=last_condition, zone_features=zone_features)
            # Override: replace the pressure channel from the last block with the separate branch's output
            fx = torch.cat([fx[:, :, :2], p_sep], dim=-1)
        else:
            fx = self.blocks[-1](fx, raw_xy=raw_xy, tandem_mask=is_tandem, condition=last_condition, zone_features=zone_features)

        gate = self.skip_gate(fx_pre)
        fx = fx + gate * self.out_skip(fx_pre)
        self._validate_output_dims(fx)
        return {"preds": fx, "re_pred": re_pred, "aoa_pred": aoa_pred}


# ---------------------------------------------------------------------------
# End Transolver model
# ---------------------------------------------------------------------------


MAX_TIMEOUT = 180.0  # minutes
MAX_EPOCHS = 500


@dataclass
class Config:
    lr: float = 1.5e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    surf_weight: float = 20.0
    manifest: str = "data/split_manifest.json"
    stats_file: str = "data/split_stats.json"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    # Schedule params (tuned for 3-hour / 500-epoch runs)
    warmup_total_iters: int = 20
    warmup_start_factor: float = 0.2
    cosine_T_max: int = 230
    cosine_eta_min: float = 1e-5
    ema_start_epoch: int = 140
    ema_decay: float = 0.998
    temp_anneal_epoch: int = 50
    vol_ramp_epochs: int = 40
    tandem_curriculum_epochs: int = 10
    noise_anneal_epochs: int = 60
    scheduler_type: str = "sequential"  # "sequential", "warm_restarts", "onecycle"
    cosine_T_0: int = 50       # warm_restarts only
    cosine_T_mult: int = 2     # warm_restarts only
    onecycle_max_lr: float = 3e-3        # onecycle only
    onecycle_epochs: int = 200           # onecycle only
    onecycle_pct_start: float = 0.15    # onecycle only
    onecycle_div_factor: float = 10.0   # onecycle only
    onecycle_final_div_factor: float = 100.0  # onecycle only
    use_lookahead: bool = True
    # Architecture flags (one per GPU)
    linear_no_attention: bool = False  # GPU0: skip Q/K/V in slice attention
    field_decoder: bool = False        # GPU1: separate vel/pres output heads
    learned_kernel: bool = False       # GPU2: MLP attention kernel
    uncertainty_loss: bool = False     # GPU3: Kendall uncertainty weighting
    swad: bool = False                 # GPU4: SWAD weight averaging
    boundary_aware: bool = False       # GPU5: upweight near-wall volume nodes
    adaln_output: bool = False         # GPU6: AdaLN on output head
    soft_moe: bool = False             # GPU7: Soft MoE output
    # Phase 2 R4: AdaLN-Zero all blocks
    n_hidden: int = 192                # model width (override default)
    adaln_all_blocks: bool = False     # AdaLN-Zero on ALL TransolverBlocks
    adaln_4cond: bool = False          # use 4-dim condition (Re, AoA, gap, surf_frac)
    adaln_decouple: bool = False       # decoupled slice assignment for tandem
    adaln_nozero: bool = False         # ablation: no zero-init on adaln projection
    adaln_sam: bool = False            # SAM optimizer in last 25% of training
    film_cond: bool = False            # FiLM conditioning (simpler alternative to AdaLN)
    adaln_zone_temp: bool = False      # zone-aware temperature modulation
    # Phase 2 R5: tandem warm-in combinations
    tandem_ramp: bool = False          # gradual tandem surface loss warm-in (0→1 over epochs 10-50)
    foil2_dist: bool = False           # explicit foil-2 distance feature (from secondary dsdf)
    slice_num: int = 48                # slice count (default 48, GPU6: 96)
    # Phase 3: training dynamics experiments
    swa: bool = False             # GPU 0/6: uniform SWA weight averaging
    swa_start_epoch: int = 200   # epoch to start SWA (GPU 0: 200, GPU 6: 160)
    grad_accum_steps: int = 1    # GPU 2: gradient accumulation (step every N batches)
    half_target_noise: bool = False  # GPU 3: reduce target noise by 50%
    no_target_noise: bool = False    # Phase 4: completely disable target noise injection
    use_lion: bool = False        # GPU 4: Lion optimizer instead of AdamW
    rdrop: bool = False           # GPU 7: R-drop regularization
    rdrop_alpha: float = 1.0     # R-drop consistency loss weight
    # Phase 3 R3: normalization/prediction-space experiments
    no_perstd: bool = False           # GPU 0: remove per-sample std norm entirely
    no_perstd_p: bool = False         # GPU 1: remove per-sample std for pressure only
    unified_clamps: bool = False      # GPU 2: unified clamps (0.2, 0.2, 0.7) for all
    high_p_clamp: bool = False        # GPU 3: higher pressure clamp (2.0)
    multiply_std: bool = False        # GPU 4: multiply instead of divide per-sample std
    raw_targets: bool = False         # GPU 5: skip physics norm, raw target space
    tight_denorm_clamps: bool = False  # GPU 6: tighter denorm clamps [-5,5]/[-10,10]
    log_pressure: bool = False        # GPU 7: log-transform Cp pressure channel
    # Phase 3: compound experiments
    seed: int = -1                     # random seed (-1 = no seeding)
    n_layers: int = 2                  # number of TransolverBlocks (default 2)
    # Phase 3: data augmentation (training-only)
    aug: str = "none"  # none|yflip|jitter|featdrop|mixup|scale|flip_jitter|aoa_perturb|cutmix
    aug_scale_range: float = 0.05   # half-range for scale augmentation (default ±5%)
    aug_start_epoch: int = 0        # delay augmentation onset until this epoch
    aug_full_dsdf_rot: bool = False  # also rotate DSDF gradient pairs in aoa_perturb
    # Phase 3 R10: DomainLayerNorm compounds
    domain_layernorm: bool = False     # domain-specific LayerNorm for single vs tandem
    dln_zeroinit: bool = False         # zero-init tandem LN weights (else copy from single)
    domain_velhead: bool = False       # domain-specific output heads for single vs tandem
    prog_slices: bool = False          # progressive slice warmup
    prog_slices_end: int = 128         # max slice count for prog_slices
    prog_slices_epochs: int = 100      # epochs to ramp slice_num → prog_slices_end
    # Phase 3 R11: SWA / snapshot ensemble / EMA tuning
    swa_cyclic: bool = False           # GPU 0/1: SWA with cyclic LR warm restarts
    swa_cyclic_T: int = 40             # warm-restart cycle period in epochs
    swa_cyclic_start: int = 100        # epoch to switch from cosine to cyclic schedule
    two_phase_lr: bool = False         # GPU 5: lr=3e-4 for phase1, then lr=1e-4
    two_phase_switch_epoch: int = 100  # epoch at which to switch phases
    two_phase_lr_1: float = 3e-4       # phase 1 LR (overrides cfg.lr when active)
    two_phase_lr_2: float = 1e-4       # phase 2 LR
    snapshot_ensemble: bool = False    # GPU 6: average checkpoints at fixed epochs
    snapshot_epochs_str: str = "120,160,200"  # comma-separated snapshot epochs
    # Phase 4: throughput optimization
    val_every: int = 1                  # validate every N epochs (1 = every epoch)
    disable_pcgrad: bool = False        # skip PCGrad dual-backward, use simple combined loss
    vol_subsample_frac: float = 1.0     # fraction of volume nodes in loss after vol_ramp (0.8 = 80%)
    compile_mode: str = "default"       # torch.compile mode: "default", "max-autotune", "reduce-overhead"
    num_workers: int = 4                # data loader workers
    # Phase 4: Pressure-first sequential prediction
    pressure_first: bool = False        # predict p first, then condition v on p
    pressure_no_detach: bool = False    # allow gradient from vel back to pres head
    pressure_deep: bool = False         # 3-layer pressure head instead of 2
    pressure_separate_last_block: bool = False  # separate last TransolverBlock for pressure
    # Phase 5: feature engineering + conditional mixup
    extra_features: str = "none"        # none|wall_dist|wall_dist_dir|all — extra geometric features
    conditional_mixup: bool = False     # conditional mixup: mix samples with same geometry, different flow
    mixup_alpha: float = 0.4           # Beta distribution parameter for conditional mixup


cfg = sp.parse(Config)

if cfg.seed >= 0:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

if cfg.debug:
    MAX_EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG MODE]" if cfg.debug else ""))

# Phase 5: set extra features mode before data loading
if cfg.extra_features != "none":
    set_extra_features_mode(cfg.extra_features)
    _extra_feat_dim = get_extra_dim()
    print(f"Phase 5 extra features: mode={cfg.extra_features}, +{_extra_feat_dim} dims")
else:
    _extra_feat_dim = 0

train_ds, val_splits, stats, sample_weights = load_data(
    cfg.manifest, cfg.stats_file, debug=cfg.debug,
)
stats = {k: v.to(device) for k, v in stats.items()}


def _umag_q(y, mask):
    """Per-sample reference velocity and dynamic pressure from mean velocity.

    Uses mean velocity of actual (unpadded) nodes as Umag proxy. For CFD flow
    over airfoils, the domain-mean velocity tracks the freestream magnitude
    across Re numbers (surface no-slip nodes reduce the mean slightly, but
    consistently across all samples).

    Returns:
        Umag: [B, 1, 1], dynamic velocity magnitude, clamped ≥ 1.0
        q:    [B, 1, 1], dynamic pressure = 0.5 * Umag^2
    """
    n_nodes = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    Ux_mean = (y[:, :, 0] * mask.float()).sum(dim=1, keepdim=True) / n_nodes  # [B, 1]
    Uy_mean = (y[:, :, 1] * mask.float()).sum(dim=1, keepdim=True) / n_nodes  # [B, 1]
    Umag = (Ux_mean ** 2 + Uy_mean ** 2).sqrt().clamp(min=1.0).unsqueeze(-1)  # [B, 1, 1]
    q = 0.5 * Umag ** 2
    return Umag, q


def _phys_norm(y, Umag, q):
    """Normalize Ux→Ux/Umag, Uy→Uy/Umag, p→p/q (Cp)."""
    y_p = y.clone()
    y_p[:, :, 0:1] = y[:, :, 0:1] / Umag
    y_p[:, :, 1:2] = y[:, :, 1:2] / Umag
    y_p[:, :, 2:3] = y[:, :, 2:3] / q
    return y_p


def _phys_denorm(y_p, Umag, q):
    """Reverse physics normalization: Ux/Umag→Ux, Uy/Umag→Uy, Cp→p."""
    y = y_p.clone()
    y[:, :, 0:1] = y_p[:, :, 0:1].clamp(-10, 10) * Umag
    y[:, :, 1:2] = y_p[:, :, 1:2].clamp(-10, 10) * Umag
    y[:, :, 2:3] = y_p[:, :, 2:3].clamp(-20, 20) * q
    return y

loader_kwargs = dict(collate_fn=pad_collate, num_workers=cfg.num_workers, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    # Avoid sampler/length mismatch when train_ds is truncated
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(subset, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, subset in val_splits.items()
}

# --- Physics normalization stats (computed over training set) ---
# Compute mean/std of Cp-normalized targets so the model sees O(1) values.
print("Computing physics normalization stats...")
_phys_sum = torch.zeros(3, device=device)
_phys_sq_sum = torch.zeros(3, device=device)
_phys_n = 0.0
_stats_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
with torch.no_grad():
    for _x, _y, _is_surf, _mask in tqdm(_stats_loader, desc="Phys stats", leave=False):
        _y, _mask = _y.to(device), _mask.to(device)
        _Um, _q = _umag_q(_y, _mask)
        _yp = _phys_norm(_y, _Um, _q)
        if cfg.log_pressure:
            _yp = _yp.clone()
            _yp[:, :, 2:3] = _yp[:, :, 2:3].abs().add(1).log() * _yp[:, :, 2:3].sign()
        _m = _mask.float().unsqueeze(-1)  # [B, N, 1]
        _phys_sum += (_yp * _m).sum(dim=(0, 1))
        _phys_sq_sum += (_yp ** 2 * _m).sum(dim=(0, 1))
        _phys_n += _mask.float().sum().item()
_pmean = (_phys_sum / _phys_n).float()
_pstd = ((_phys_sq_sum / _phys_n - _pmean ** 2).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
phys_stats = {"y_mean": _pmean, "y_std": _pstd}
print(f"  Cp stats — mean: {_pmean.tolist()}, std: {_pstd.tolist()}")

if cfg.raw_targets:
    print("Computing raw target stats (no physics normalization)...")
    _raw_sum = torch.zeros(3, device=device)
    _raw_sq_sum = torch.zeros(3, device=device)
    _raw_n = 0.0
    with torch.no_grad():
        for _x, _y, _is_surf, _mask in tqdm(_stats_loader, desc="Raw stats", leave=False):
            _y, _mask = _y.to(device), _mask.to(device)
            _m = _mask.float().unsqueeze(-1)
            _raw_sum += (_y * _m).sum(dim=(0, 1))
            _raw_sq_sum += (_y ** 2 * _m).sum(dim=(0, 1))
            _raw_n += _mask.float().sum().item()
    _raw_mean = (_raw_sum / _raw_n).float()
    _raw_std = ((_raw_sq_sum / _raw_n - _raw_mean ** 2).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
    raw_stats = {"y_mean": _raw_mean, "y_std": _raw_std}
    print(f"  Raw stats — mean: {_raw_mean.tolist()}, std: {_raw_std.tolist()}")
else:
    raw_stats = None

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2 + 2 + (1 if cfg.foil2_dist else 0) + 32 + _extra_feat_dim,  # +curv, +dist, [+foil2dist], +32 fourier PE, +extra
    out_dim=3,
    n_hidden=cfg.n_hidden,
    n_layers=cfg.n_layers,
    n_head=3,
    slice_num=cfg.prog_slices_end if cfg.prog_slices else cfg.slice_num,
    mlp_ratio=2,
    dropout=0.05 if cfg.rdrop else 0.0,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
    linear_no_attention=cfg.linear_no_attention,
    learned_kernel=cfg.learned_kernel,
    field_decoder=cfg.field_decoder,
    adaln_output=cfg.adaln_output,
    soft_moe=cfg.soft_moe,
    uncertainty_loss=cfg.uncertainty_loss,
    adaln_all_blocks=cfg.adaln_all_blocks,
    adaln_4cond=cfg.adaln_4cond,
    adaln_nozero=cfg.adaln_nozero,
    film_cond=cfg.film_cond,
    adaln_decouple=cfg.adaln_decouple,
    adaln_zone_temp=cfg.adaln_zone_temp,
    domain_layernorm=cfg.domain_layernorm,
    dln_zeroinit=cfg.dln_zeroinit,
    domain_velhead=cfg.domain_velhead,
    prog_slices=cfg.prog_slices,
    pressure_first=cfg.pressure_first,
    pressure_no_detach=cfg.pressure_no_detach,
    pressure_deep=cfg.pressure_deep,
)

model = Transolver(**model_config).to(device)
model._pressure_separate = cfg.pressure_separate_last_block
torch._functorch.config.donated_buffer = False  # required for retain_graph=True in PCGrad
model = torch.compile(model, mode=cfg.compile_mode)
_base_model = model._orig_mod if hasattr(model, '_orig_mod') else model

from copy import deepcopy
ema_model = None
swad_initial_val = None
swad_prev_val = float("inf")
swad_checkpoints: list = []
swad_collecting = False
swad_done = False
swa_model = None
swa_n = 0
swa_cyclic_model = None
swa_cyclic_n = 0
swa_cyclic_scheduler = None
snapshot_avg_model = None
snapshot_n = 0
snapshot_epoch_list = [int(e) for e in cfg.snapshot_epochs_str.split(",")] if cfg.snapshot_ensemble else []

n_params = sum(p.numel() for p in model.parameters())


class SAM:
    """Sharpness-Aware Minimization (Foret et al., 2021).

    Usage:
        sam.perturb()          # perturb params in gradient direction
        recompute loss/backward
        sam.restore_and_step() # restore params, then call base_optimizer.step()
    """
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        norms = [
            p.grad.norm(2)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.stack(norms).norm(2) if norms else torch.tensor(0.0)

    def perturb(self):
        scale = self.rho / (self._grad_norm() + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p._sam_e_w = (p.grad * scale).detach()
                    p.data.add_(p._sam_e_w)

    def restore(self):
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, '_sam_e_w'):
                    p.data.sub_(p._sam_e_w)
                    del p._sam_e_w


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023). Sign-based updates, ~2x less memory than AdamW.

    Use lr ~3-10x lower than AdamW (e.g. lr=3e-4 instead of 1.5e-3).
    """
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                b1, b2 = group['betas']
                update = exp_avg * b1 + p.grad * (1 - b1)
                p.data.add_(update.sign(), alpha=-group['lr'])
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                exp_avg.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


class Lookahead:
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.slow_params = [
            [p.data.clone() for p in group['params']]
            for group in base_optimizer.param_groups
        ]
        self.step_count = 0

    def step(self):
        self.base_optimizer.step()
        self.step_count += 1
        if self.step_count % self.k == 0:
            for slow, group in zip(self.slow_params, self.base_optimizer.param_groups):
                for s, p in zip(slow, group['params']):
                    s.data.add_(self.alpha * (p.data - s.data))
                    p.data.copy_(s.data)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


attn_params = [p for n, p in model.named_parameters() if any(k in n for k in ['Wqkv', 'temperature', 'slice_weight', 'attn_scale', 'spatial_bias'])]
other_params = [p for n, p in model.named_parameters() if not any(k in n for k in ['Wqkv', 'temperature', 'slice_weight', 'attn_scale', 'spatial_bias'])]
_base_lr = cfg.two_phase_lr_1 if cfg.two_phase_lr else cfg.lr
if cfg.use_lion:
    base_opt = Lion([
        {'params': attn_params, 'lr': _base_lr * 0.5},
        {'params': other_params, 'lr': _base_lr}
    ], weight_decay=cfg.weight_decay)
    optimizer = base_opt  # Lion has its own momentum; skip Lookahead
else:
    base_opt = torch.optim.AdamW([
        {'params': attn_params, 'lr': _base_lr * 0.5},
        {'params': other_params, 'lr': _base_lr}
    ], weight_decay=cfg.weight_decay)
    if cfg.use_lookahead:
        optimizer = Lookahead(base_opt, k=10, alpha=0.8)
    else:
        optimizer = base_opt

sam_optimizer = SAM(base_opt, rho=0.05) if cfg.adaln_sam else None
if cfg.scheduler_type == "warm_restarts":
    _warmup = torch.optim.lr_scheduler.LinearLR(base_opt, start_factor=0.1, total_iters=10)
    _restarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        base_opt, T_0=cfg.cosine_T_0, T_mult=cfg.cosine_T_mult, eta_min=cfg.cosine_eta_min
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        base_opt, schedulers=[_warmup, _restarts], milestones=[10]
    )
elif cfg.scheduler_type == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        base_opt,
        max_lr=[cfg.onecycle_max_lr * 0.5, cfg.onecycle_max_lr],
        epochs=cfg.onecycle_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.onecycle_pct_start,
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
    )
else:  # sequential (default)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        base_opt, start_factor=cfg.warmup_start_factor, total_iters=cfg.warmup_total_iters
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt, T_max=cfg.cosine_T_max, eta_min=cfg.cosine_eta_min
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        base_opt, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_total_iters]
    )
step_scheduler_per_batch = (cfg.scheduler_type == "onecycle")

# --- wandb ---
run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    project=os.environ.get("WANDB_PROJECT", "senpai-v1"),
    group=cfg.wandb_group,
    name=cfg.wandb_name,
    tags=[cfg.agent] if cfg.agent else [],
    config={
        **asdict(cfg),
        "model_config": model_config,
        "n_params": n_params,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
        "split_manifest": cfg.manifest,
    },
    mode=os.environ.get("WANDB_MODE", "online"),
)

# All metrics use global_step (gradient updates) as the x-axis.
wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
for _sname in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_sname}/*", step_metric="global_step")
wandb.define_metric("lr", step_metric="global_step")
wandb.define_metric("epoch_time_s", step_metric="global_step")
wandb.define_metric("val_predictions", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

best_val = float("inf")
ema_val_loss = float("inf")
ema_decay_val = 0.9
best_metrics = {}
global_step = 0
train_start = time.time()
prev_vol_loss = 1.0
prev_surf_loss = 0.2  # initial ratio ~5 (clamped minimum)
running_tandem_loss = 0.05
running_nontandem_loss = 0.05

for epoch in range(MAX_EPOCHS):
    elapsed_min = (time.time() - train_start) / 60.0
    if elapsed_min >= MAX_TIMEOUT:
        print(f"Wall-clock limit reached ({elapsed_min:.1f} min >= {MAX_TIMEOUT} min). Stopping.")
        break

    t0 = time.time()

    # Adaptive surface weight: loss-ratio based, clamped [5, 50]
    surf_weight = max(5.0, min(50.0, prev_vol_loss / max(prev_surf_loss, 1e-8)))

    # --- Train ---
    model.train()
    epoch_vol = 0.0
    epoch_surf = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [train]", leave=False)
    if cfg.grad_accum_steps > 1:
        optimizer.zero_grad()
    for batch_idx, (x, y, is_surface, mask) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Phase 5: extract extra features before augmentation/standardization
        _extra_feats = None
        if _extra_feat_dim > 0:
            _extra_feats = x[:, :, X_DIM:]  # [B, N, extra_dim]
            x = x[:, :, :X_DIM]  # trim to core 24 features

        # --- Data augmentation (training-only, applied before normalization) ---
        if model.training and cfg.aug != "none" and epoch >= cfg.aug_start_epoch:
            if cfg.aug in ("yflip", "flip_jitter"):
                _flip = torch.rand(x.size(0), 1, 1, device=x.device) < 0.5
                x[:, :, 1:2] = torch.where(_flip, -x[:, :, 1:2], x[:, :, 1:2])
                y[:, :, 1:2] = torch.where(_flip, -y[:, :, 1:2], y[:, :, 1:2])
                for _idx in [3, 5, 7, 9]:
                    x[:, :, _idx:_idx+1] = torch.where(_flip, -x[:, :, _idx:_idx+1], x[:, :, _idx:_idx+1])
            if cfg.aug in ("jitter", "flip_jitter"):
                x[:, :, :2] = x[:, :, :2] + 0.001 * torch.randn_like(x[:, :, :2])
            if cfg.aug == "featdrop":
                _B_aug = x.size(0)
                for _b in range(_B_aug):
                    _drop = torch.randperm(22, device=x.device)[:2] + 2
                    x[_b, :, _drop] = 0.0
            if cfg.aug == "mixup":
                _B_aug = x.size(0)
                _beta_dist = torch.distributions.Beta(torch.tensor(0.2), torch.tensor(0.2))
                _lam = _beta_dist.sample((_B_aug,)).to(x.device).view(_B_aug, 1, 1)
                _mix_idx = torch.randperm(_B_aug, device=x.device)
                x = _lam * x + (1 - _lam) * x[_mix_idx]
                y = _lam * y + (1 - _lam) * y[_mix_idx]
                mask = mask & mask[_mix_idx]
            if cfg.aug == "scale":
                _lo = 1.0 - cfg.aug_scale_range
                _scale = torch.rand(x.size(0), 1, 1, device=x.device) * (2 * cfg.aug_scale_range) + _lo
                x[:, :, :2] = x[:, :, :2] * _scale
            if cfg.aug == "aoa_perturb":
                _angle_deg = torch.rand(x.size(0), device=x.device) * 2.0 - 1.0
                _angle_rad = _angle_deg * (torch.pi / 180.0)
                _cos_a = torch.cos(_angle_rad).view(-1, 1, 1)
                _sin_a = torch.sin(_angle_rad).view(-1, 1, 1)
                _xc, _yc = x[:, :, 0:1].clone(), x[:, :, 1:2].clone()
                x[:, :, 0:1] = _cos_a * _xc - _sin_a * _yc
                x[:, :, 1:2] = _sin_a * _xc + _cos_a * _yc
                _Ux, _Uy = y[:, :, 0:1].clone(), y[:, :, 1:2].clone()
                y[:, :, 0:1] = _cos_a * _Ux - _sin_a * _Uy
                y[:, :, 1:2] = _sin_a * _Ux + _cos_a * _Uy
                if cfg.aug_full_dsdf_rot:
                    # Rotate DSDF gradient pairs (x,y components at indices 2-9)
                    for _xi, _yi in [(2, 3), (4, 5), (6, 7), (8, 9)]:
                        _dx = x[:, :, _xi:_xi+1].clone()
                        _dy = x[:, :, _yi:_yi+1].clone()
                        x[:, :, _xi:_xi+1] = _cos_a * _dx - _sin_a * _dy
                        x[:, :, _yi:_yi+1] = _sin_a * _dx + _cos_a * _dy
            if cfg.aug == "cutmix":
                _B_aug = x.size(0)
                _cut_idx = torch.randperm(_B_aug, device=x.device)
                _x0 = x[:, :, 0]
                _x0_lo = _x0.min(dim=1).values
                _x0_hi = _x0.max(dim=1).values
                _x_start = _x0_lo + torch.rand(_B_aug, device=x.device) * (_x0_hi - _x0_lo - 0.3).clamp(min=0)
                _x_end = _x_start + 0.3
                for _b in range(_B_aug):
                    _in_region = (_x0[_b] >= _x_start[_b]) & (_x0[_b] <= _x_end[_b])
                    x[_b, _in_region] = x[_cut_idx[_b], _in_region]
                    y[_b, _in_region] = y[_cut_idx[_b], _in_region]
                    is_surface[_b, _in_region] = is_surface[_cut_idx[_b], _in_region]

        # Phase 5: conditional mixup — mix targets of samples with same domain
        if model.training and cfg.conditional_mixup and x.size(0) >= 2:
            _B_mix = x.size(0)
            # Identify domain: single (gap~0) vs tandem (gap!=0)
            _gap_vals = x[:, 0, 21]  # gap feature (same for all nodes in a sample)
            _is_tandem_mix = (_gap_vals.abs() > 0.01)  # [B]
            # Find pairs within same domain
            _mix_idx = torch.arange(_B_mix, device=x.device)
            for _b in range(_B_mix):
                _same_domain = (_is_tandem_mix == _is_tandem_mix[_b])
                _same_domain[_b] = False  # don't mix with self
                if _same_domain.any():
                    _candidates = _same_domain.nonzero(as_tuple=True)[0]
                    _mix_idx[_b] = _candidates[torch.randint(len(_candidates), (1,))]
            _lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample((_B_mix,)).to(x.device).view(_B_mix, 1, 1)
            # Mix only flow conditions (Re, AoA at indices 11-14) and targets
            # Keep geometry (pos, saf, dsdf) from original sample
            x[:, :, 11:15] = _lam * x[:, :, 11:15] + (1 - _lam) * x[_mix_idx, :, 11:15]
            y = _lam * y + (1 - _lam) * y[_mix_idx]
            # Also mix mask to handle different padding
            mask = mask & mask[_mix_idx]

        raw_dsdf = x[:, :, 2:10]  # original dsdf before standardization
        dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
        dist_feat = torch.log1p(dist_surf * 10.0)  # log-scale for better gradient flow
        x = (x - stats["x_mean"]) / stats["x_std"]
        # Curvature proxy: norm of first 4 dsdf channels (gradient magnitude) for surface nodes
        curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
        if cfg.foil2_dist:
            foil2_dist_feat = torch.log1p(raw_dsdf[:, :, 4:8].abs().min(dim=-1, keepdim=True).values * 10.0)
            x = torch.cat([x, curv, dist_feat, foil2_dist_feat], dim=-1)
        else:
            x = torch.cat([x, curv, dist_feat], dim=-1)
        # Fourier positional encoding: append sin/cos of (x,y) at 4 learnable frequencies
        raw_xy = x[:, :, :2]
        # Normalize xy to [0,1] per-sample for consistent Fourier encoding
        xy_min = raw_xy.amin(dim=1, keepdim=True)
        xy_max = raw_xy.amax(dim=1, keepdim=True)
        xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
        freqs = torch.cat([model.fourier_freqs_fixed.to(device), model.fourier_freqs_learned.abs()])
        xy_scaled = xy_norm.unsqueeze(-1) * freqs  # [B, N, 2, 4]
        fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)  # [B, N, 16]
        x = torch.cat([x, fourier_pe], dim=-1)
        # Phase 5: append extra features after standardization/fourier
        if _extra_feats is not None:
            x = torch.cat([x, _extra_feats], dim=-1)
        if model.training and epoch < cfg.noise_anneal_epochs:
            noise_scale = 0.05 * (1 - epoch / cfg.noise_anneal_epochs)
            x[:, :, 2:25] = x[:, :, 2:25] + noise_scale * torch.randn_like(x[:, :, 2:25])
        Umag, q = _umag_q(y, mask)
        if cfg.raw_targets:
            y_norm = (y - raw_stats["y_mean"]) / raw_stats["y_std"]
        else:
            y_phys = _phys_norm(y, Umag, q)
            if cfg.log_pressure:
                y_phys = y_phys.clone()
                y_phys[:, :, 2:3] = y_phys[:, :, 2:3].abs().add(1).log() * y_phys[:, :, 2:3].sign()
            y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
        if model.training and not cfg.no_target_noise:
            noise_progress = min(1.0, epoch / max(cfg.noise_anneal_epochs, 1))
            if cfg.half_target_noise:
                vel_noise = 0.0075 * (1 - noise_progress) + 0.0015 * noise_progress
                p_noise = 0.004 * (1 - noise_progress) + 0.0005 * noise_progress
            else:
                vel_noise = 0.015 * (1 - noise_progress) + 0.003 * noise_progress
                p_noise = 0.008 * (1 - noise_progress) + 0.001 * noise_progress
            noise_scale = torch.tensor([vel_noise, vel_noise, p_noise], device=device)
            y_norm = y_norm + noise_scale * torch.randn_like(y_norm)

        # Per-sample std normalization: skip tandem samples (gap feature index 21)
        raw_gap = x[:, 0, 21]
        is_tandem = raw_gap.abs() > 0.5
        B = y_norm.shape[0]
        sample_stds = torch.ones(B, 1, 3, device=device)
        if not cfg.no_perstd and not cfg.raw_targets:
            if cfg.unified_clamps:
                channel_clamps = tandem_clamps = torch.tensor([0.2, 0.2, 0.7], device=device)
            elif cfg.high_p_clamp:
                channel_clamps = torch.tensor([0.1, 0.1, 2.0], device=device)
                tandem_clamps = torch.tensor([0.3, 0.3, 2.0], device=device)
            else:
                channel_clamps = torch.tensor([0.1, 0.1, 0.5], device=device)
                tandem_clamps = torch.tensor([0.3, 0.3, 1.0], device=device)
            if model.training:
                for b in range(B):
                    valid = mask[b]
                    if cfg.no_perstd_p:
                        # Normalize velocity only; pressure keeps std=1
                        vc = (tandem_clamps[:2] if is_tandem[b] else channel_clamps[:2])
                        sample_stds[b, 0, :2] = y_norm[b, valid, :2].std(dim=0).clamp(min=vc)
                    elif is_tandem[b]:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                    else:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)
        if model.training and not cfg.no_perstd and not cfg.raw_targets:
            if cfg.multiply_std:
                y_norm = y_norm * sample_stds
            else:
                y_norm = y_norm / sample_stds

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model({"x": x})
            pred = out["preds"]
            re_pred = out["re_pred"]
            aoa_pred = out["aoa_pred"]
        pred = pred.float()
        re_pred = re_pred.float()
        aoa_pred = aoa_pred.float()
        if model.training and not cfg.no_perstd and not cfg.raw_targets:
            if cfg.multiply_std:
                pred = pred * sample_stds
            else:
                pred = pred / sample_stds
        sq_err = (pred - y_norm) ** 2
        abs_err = (pred - y_norm).abs()
        if cfg.tandem_ramp:
            pass  # no hard curriculum; tandem_weight applied via tandem_boost below
        elif epoch < cfg.tandem_curriculum_epochs:
            is_tandem_curr = (x[:, :, -8:].abs().sum(dim=(1, 2)) > 0.01)
            sample_mask = (~is_tandem_curr).float()[:, None, None]
            abs_err = abs_err * sample_mask
        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface

        # Progressive resolution: subsample volume nodes in loss early in training
        # Ramps from 10% → 100% of volume nodes over first 40 epochs
        if epoch < cfg.vol_ramp_epochs:
            vol_keep_ratio = 0.05 + 0.95 * (epoch / cfg.vol_ramp_epochs)
            vol_indices = vol_mask.nonzero(as_tuple=False)
            n_vol = vol_indices.shape[0]
            n_keep = max(int(n_vol * vol_keep_ratio), 1)
            perm = torch.randperm(n_vol, device=vol_mask.device)[:n_keep]
            vol_mask_train = torch.zeros_like(vol_mask)
            if n_keep > 0:
                vol_mask_train[vol_indices[perm, 0], vol_indices[perm, 1]] = True
        elif cfg.vol_subsample_frac < 1.0:
            vol_indices = vol_mask.nonzero(as_tuple=False)
            n_vol = vol_indices.shape[0]
            n_keep = max(int(n_vol * cfg.vol_subsample_frac), 1)
            perm = torch.randperm(n_vol, device=vol_mask.device)[:n_keep]
            vol_mask_train = torch.zeros_like(vol_mask)
            if n_keep > 0:
                vol_mask_train[vol_indices[perm, 0], vol_indices[perm, 1]] = True
        else:
            vol_mask_train = vol_mask

        if cfg.boundary_aware:
            vol_dist = dist_feat[:, :, 0]  # [B, N], log1p-scaled dist-to-surface
            valid_dists = vol_dist.masked_select(vol_mask_train)
            if valid_dists.numel() > 10:
                threshold = valid_dists.quantile(0.1)
                near_wall = vol_mask_train & (vol_dist < threshold)
                node_weight = (1.0 + near_wall.float()).unsqueeze(-1)  # 2x near-wall, 1x else
                vol_loss = (abs_err * node_weight * vol_mask_train.float().unsqueeze(-1)).sum() / \
                           (node_weight.squeeze(-1) * vol_mask_train.float()).sum().clamp(min=1)
            else:
                vol_loss = (abs_err * vol_mask_train.unsqueeze(-1)).sum() / vol_mask_train.sum().clamp(min=1)
        else:
            vol_loss = (abs_err * vol_mask_train.unsqueeze(-1)).sum() / vol_mask_train.sum().clamp(min=1)
        is_tandem_batch = (x[:, 0, 21].abs() > 0.01)
        surf_per_sample = (abs_err[:, :, 2:3] * surf_mask.unsqueeze(-1)).sum(dim=(1, 2)) / surf_mask.sum(dim=1).clamp(min=1).float()
        tandem_err = surf_per_sample[is_tandem_batch].mean().item() if is_tandem_batch.any() else running_tandem_loss
        nontandem_err = surf_per_sample[~is_tandem_batch].mean().item() if (~is_tandem_batch).any() else running_nontandem_loss
        running_tandem_loss = 0.9 * running_tandem_loss + 0.1 * tandem_err
        running_nontandem_loss = 0.9 * running_nontandem_loss + 0.1 * nontandem_err
        # Asymmetric hard-node mining for non-tandem samples after epoch 30 (vectorized)
        if epoch >= 30:
            surf_pres = abs_err[:, :, 2:3]  # pressure errors [B, N, 1]
            surf_pres_flat = surf_pres[:, :, 0]  # [B, N]
            surf_pres_masked = surf_pres_flat.masked_fill(~surf_mask, float('nan'))
            thresh = torch.nanmedian(surf_pres_masked, dim=1).values  # [B]
            thresh = thresh.nan_to_num(float('inf'))  # safe: inf → no hard nodes
            hard_mask = (~is_tandem_batch)[:, None] & surf_mask & (surf_pres_flat >= thresh[:, None])
            hard_weights = (hard_mask.float() * 0.5 + 1.0).unsqueeze(-1)  # 1.5 hard, 1.0 else
            surf_per_sample = (surf_pres * hard_weights * surf_mask.unsqueeze(-1)).sum(dim=(1, 2)) / surf_mask.sum(dim=1).clamp(min=1).float()
        adaptive_boost = max(1.0, min(4.0, running_tandem_loss / max(running_nontandem_loss, 1e-8)))
        if cfg.tandem_ramp:
            tandem_weight = min(1.0, max(0.0, (epoch - 10) / 40.0))
            tandem_boost = torch.where(is_tandem_batch,
                                       torch.tensor(adaptive_boost * tandem_weight, device=device),
                                       torch.ones(B, device=device))
        else:
            tandem_boost = torch.where(is_tandem_batch, adaptive_boost, 1.0).to(device)
        surf_loss = (surf_per_sample * tandem_boost).mean()
        if cfg.uncertainty_loss:
            bm = _base_model
            surf_ux_loss = (abs_err[:, :, 0:1] * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            surf_uy_loss = (abs_err[:, :, 1:2] * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            surf_p_loss  = (abs_err[:, :, 2:3] * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = (vol_loss    * torch.exp(-2 * bm.log_sigma_vol)    / 2 + bm.log_sigma_vol +
                    surf_ux_loss * torch.exp(-2 * bm.log_sigma_surf_ux) / 2 + bm.log_sigma_surf_ux +
                    surf_uy_loss * torch.exp(-2 * bm.log_sigma_surf_uy) / 2 + bm.log_sigma_surf_uy +
                    surf_p_loss  * torch.exp(-2 * bm.log_sigma_surf_p)  / 2 + bm.log_sigma_surf_p)
        else:
            loss = vol_loss + surf_weight * surf_loss

        # Multi-scale loss: coarse spatial pooling
        _coarse_loss = None
        coarse_pool_size = 64
        B, N, C = pred.shape
        n_groups = N // coarse_pool_size
        if n_groups > 1:
            # Sort by x-coordinate for spatially coherent groups
            raw_x_coord = x[:, :, 0]  # x-coordinate
            sort_idx = raw_x_coord.argsort(dim=1)
            pred_sorted = torch.gather(pred, 1, sort_idx.unsqueeze(-1).expand_as(pred))
            y_sorted = torch.gather(y_norm, 1, sort_idx.unsqueeze(-1).expand_as(y_norm))
            mask_sorted = torch.gather(mask, 1, sort_idx)
            # Pool predictions and targets over groups of 64 nodes
            pred_trunc = pred_sorted[:, :n_groups * coarse_pool_size]
            y_trunc = y_sorted[:, :n_groups * coarse_pool_size]
            mask_trunc = mask_sorted[:, :n_groups * coarse_pool_size]

            mask_trunc_f = mask_trunc.float().reshape(B, n_groups, coarse_pool_size).unsqueeze(-1)  # [B, G, P, 1]
            pred_g = pred_trunc.reshape(B, n_groups, coarse_pool_size, C)
            y_g = y_trunc.reshape(B, n_groups, coarse_pool_size, C)
            pred_coarse = (pred_g * mask_trunc_f).sum(dim=2) / mask_trunc_f.sum(dim=2).clamp(min=1)
            y_coarse = (y_g * mask_trunc_f).sum(dim=2) / mask_trunc_f.sum(dim=2).clamp(min=1)
            mask_coarse = mask_trunc.reshape(B, n_groups, coarse_pool_size).any(dim=2)

            coarse_err = (pred_coarse - y_coarse).abs()
            coarse_loss = (coarse_err * mask_coarse.unsqueeze(-1)).sum() / mask_coarse.sum().clamp(min=1)
            _coarse_loss = coarse_loss
            loss = loss + 1.0 * coarse_loss

        log_re_target = x[:, 0, 13:14]  # log(Re) from input features (same for all nodes)
        re_loss = F.mse_loss(re_pred, log_re_target)
        loss = loss + 0.01 * re_loss
        aoa_target = x[:, 0, 14:15]  # AoA0_rad from normalized input
        aoa_loss = F.mse_loss(aoa_pred.float(), aoa_target)
        loss = loss + 0.01 * aoa_loss

        # R-drop: second forward pass with different dropout mask for consistency
        rdrop_loss = torch.tensor(0.0, device=device)
        if cfg.rdrop and model.training:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                rdrop_out = model({"x": x})
                rdrop_pred = rdrop_out["preds"].float() / sample_stds
            valid_mask = mask.float().unsqueeze(-1)
            rdrop_loss = ((pred - rdrop_pred) ** 2 * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            loss = loss + cfg.rdrop_alpha * rdrop_loss

        # PCGrad: in-dist (Group A) vs all-OOD (Group B) gradient projection
        # Group B = tandem + extreme-Re (>1σ) + extreme-AoA (>1σ), Group A = rest
        is_ood_pcgrad = is_tandem_batch | (x[:, 0, 13] > 1.0) | (x[:, 0, 14].abs() > 1.0)
        is_indist_pcgrad = ~is_ood_pcgrad
        use_pcgrad = (not cfg.disable_pcgrad) and is_indist_pcgrad.any() and is_ood_pcgrad.any()

        if use_pcgrad:
            n_a = is_indist_pcgrad.float().sum().clamp(min=1)
            n_b = is_ood_pcgrad.float().sum().clamp(min=1)
            vol_mask_a = vol_mask_train & is_indist_pcgrad.unsqueeze(1)
            vol_mask_b = vol_mask_train & is_ood_pcgrad.unsqueeze(1)
            vol_loss_a = (abs_err * vol_mask_a.unsqueeze(-1)).sum() / vol_mask_a.sum().clamp(min=1)
            vol_loss_b = (abs_err * vol_mask_b.unsqueeze(-1)).sum() / vol_mask_b.sum().clamp(min=1)
            surf_loss_a = (surf_per_sample * is_indist_pcgrad.float() * tandem_boost).sum() / n_a
            surf_loss_b = (surf_per_sample * is_ood_pcgrad.float() * tandem_boost).sum() / n_b
            coarse_shared = _coarse_loss * 0.5 if _coarse_loss is not None else 0.0
            loss_a = vol_loss_a + surf_weight * surf_loss_a + coarse_shared + 0.005 * re_loss + 0.005 * aoa_loss
            loss_b = vol_loss_b + surf_weight * surf_loss_b + coarse_shared + 0.005 * re_loss + 0.005 * aoa_loss

            optimizer.zero_grad()
            loss_a.backward(retain_graph=True)
            grads_a = [p.grad.clone() if p.grad is not None else None
                       for p in model.parameters()]
            optimizer.zero_grad()
            loss_b.backward()

            ga_flat = torch.cat([g.view(-1) for g in grads_a if g is not None])
            gb_flat = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            dot_ab = (ga_flat @ gb_flat).item()
            gb_ns = float((gb_flat @ gb_flat).item()) + 1e-8
            ga_ns = float((ga_flat @ ga_flat).item()) + 1e-8
            for p, ga in zip(model.parameters(), grads_a):
                gb = p.grad
                if ga is None and gb is None:
                    continue
                if ga is None:
                    pass  # keep gb
                elif gb is None:
                    p.grad = ga
                elif dot_ab < 0:
                    p.grad = ((ga - (dot_ab / gb_ns) * gb) + (gb - (dot_ab / ga_ns) * ga)) * 0.5
                else:
                    p.grad = (ga + gb) * 0.5
        else:
            if cfg.grad_accum_steps <= 1:
                optimizer.zero_grad()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        sam_active = sam_optimizer is not None and epoch >= int(MAX_EPOCHS * 0.75)
        _should_step = (cfg.grad_accum_steps <= 1 or
                        (batch_idx + 1) % cfg.grad_accum_steps == 0 or
                        batch_idx == len(train_loader) - 1)
        if sam_active and not use_pcgrad and _should_step:
            # SAM first step: perturb parameters toward gradient direction
            sam_optimizer.perturb()
            sam_optimizer.zero_grad()
            # Recompute forward at perturbed parameters (simplified loss, no coarse/pcgrad)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out2 = model({"x": x})
                pred2 = out2["preds"].float() / sample_stds
                re_pred2 = out2["re_pred"].float()
                aoa_pred2 = out2["aoa_pred"].float()
            abs_err2 = (pred2 - y_norm).abs()
            vol_loss2 = (abs_err2 * vol_mask_train.unsqueeze(-1)).sum() / vol_mask_train.sum().clamp(min=1)
            surf_ps2 = (abs_err2[:, :, 2:3] * surf_mask.unsqueeze(-1)).sum(dim=(1, 2)) / surf_mask.sum(dim=1).clamp(min=1).float()
            surf_loss2 = (surf_ps2 * tandem_boost).mean()
            re_loss2 = F.mse_loss(re_pred2, log_re_target)
            aoa_loss2 = F.mse_loss(aoa_pred2, aoa_target)
            loss2 = vol_loss2 + surf_weight * surf_loss2 + 0.01 * re_loss2 + 0.01 * aoa_loss2
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            sam_optimizer.restore()
        if use_pcgrad or _should_step:
            optimizer.step()
            if cfg.grad_accum_steps > 1 and not use_pcgrad:
                optimizer.zero_grad()
        if step_scheduler_per_batch and (use_pcgrad or _should_step):
            try:
                scheduler.step()
            except ValueError:
                pass
        if epoch >= cfg.ema_start_epoch and not cfg.swad and not cfg.swa and not cfg.swa_cyclic and not cfg.snapshot_ensemble:
            if ema_model is None:
                ema_model = deepcopy(_base_model)
            else:
                with torch.no_grad():
                    for ep, mp in zip(ema_model.parameters(), _base_model.parameters()):
                        ep.data.mul_(cfg.ema_decay).add_(mp.data, alpha=1 - cfg.ema_decay)
        global_step += 1
        wandb.log({"train/loss": loss.item(), "train/surf_weight": surf_weight, "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1
        pbar.set_postfix(vol=f"{vol_loss.item():.3f}", surf=f"{surf_loss.item():.3f}")

    if not step_scheduler_per_batch:
        if cfg.swa_cyclic and epoch >= cfg.swa_cyclic_start:
            if swa_cyclic_scheduler is None:
                swa_cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    base_opt, T_0=cfg.swa_cyclic_T, eta_min=cfg.cosine_eta_min
                )
            swa_cyclic_scheduler.step()
            # At cycle minimum (end of each T period), save checkpoint to running average
            if (epoch - cfg.swa_cyclic_start + 1) % cfg.swa_cyclic_T == 0:
                snap = {k: v.cpu().clone() for k, v in _base_model.state_dict().items()}
                if swa_cyclic_model is None:
                    swa_cyclic_model = deepcopy(_base_model)
                    swa_cyclic_model.load_state_dict(snap)
                    swa_cyclic_n = 1
                else:
                    with torch.no_grad():
                        cs = swa_cyclic_model.state_dict()
                        for k in snap:
                            cs[k].mul_(swa_cyclic_n / (swa_cyclic_n + 1)).add_(snap[k].to(device) / (swa_cyclic_n + 1))
                    swa_cyclic_n += 1
        else:
            scheduler.step()
    # Two-phase LR: at switch epoch, reset optimizer LR and replace scheduler
    if cfg.two_phase_lr and epoch + 1 == cfg.two_phase_switch_epoch:
        lrs = [cfg.two_phase_lr_2 * 0.5, cfg.two_phase_lr_2]
        for pg, new_lr in zip(base_opt.param_groups, lrs):
            pg['lr'] = new_lr
            pg['initial_lr'] = new_lr
        remaining = max(1, cfg.cosine_T_max - cfg.two_phase_switch_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=remaining, eta_min=cfg.cosine_eta_min
        )
    if epoch >= cfg.temp_anneal_epoch:
        with torch.no_grad():
            _base_model.blocks[0].attn.temperature.data.clamp_(max=0.25)
    if cfg.prog_slices:
        # Progressive slice warmup: ramp active slices from cfg.slice_num → prog_slices_end
        if epoch < cfg.prog_slices_epochs:
            active = int(cfg.slice_num + (cfg.prog_slices_end - cfg.slice_num) * epoch / cfg.prog_slices_epochs)
        else:
            active = cfg.prog_slices_end
        with torch.no_grad():
            for _blk in _base_model.blocks:
                if hasattr(_blk.attn, 'slice_mask'):
                    _blk.attn.slice_mask.zero_()
                    _blk.attn.slice_mask[active:].fill_(-1e9)
    epoch_vol /= n_batches
    epoch_surf /= n_batches
    prev_vol_loss = epoch_vol
    prev_surf_loss = epoch_surf
    # Snapshot ensemble: save running average at specified epochs
    if cfg.snapshot_ensemble and (epoch + 1) in snapshot_epoch_list:
        snap = {k: v.cpu().clone() for k, v in _base_model.state_dict().items()}
        snapshot_n += 1
        if snapshot_avg_model is None:
            snapshot_avg_model = deepcopy(_base_model)
            snapshot_avg_model.load_state_dict(snap)
        else:
            with torch.no_grad():
                sa = snapshot_avg_model.state_dict()
                for k in snap:
                    sa[k].mul_((snapshot_n - 1) / snapshot_n).add_(snap[k].to(device) / snapshot_n)

    # --- Validate across all splits ---
    _do_val = (epoch + 1) % cfg.val_every == 0 or epoch == 0 or epoch == MAX_EPOCHS - 1
    if not _do_val:
        dt = time.time() - t0
        wandb.log({
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "epoch_time_s": dt,
            "lr": scheduler.get_last_lr()[0],
            "global_step": global_step,
        })
        print(f"Epoch {epoch+1:3d} ({dt:.0f}s)  train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  [val skipped]")
        continue

    if cfg.swa_cyclic and swa_cyclic_model is not None:
        eval_model = swa_cyclic_model
    elif cfg.snapshot_ensemble and snapshot_avg_model is not None:
        eval_model = snapshot_avg_model
    elif cfg.swa and swa_model is not None:
        eval_model = swa_model
    elif ema_model is not None:
        eval_model = ema_model
    else:
        eval_model = model
    eval_model.eval()
    model.eval()
    val_metrics_per_split: dict[str, dict] = {}
    val_loss_sum = 0.0

    for split_name, vloader in val_loaders.items():
        val_vol = 0.0
        val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = torch.zeros(3, device=device)
        n_vol = torch.zeros(3, device=device)
        n_vbatches = 0

        with torch.no_grad():
            for x, y, is_surface, mask in tqdm(
                vloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [{split_name}]", leave=False
            ):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                # Phase 5: extract extra features
                _val_extra = None
                if _extra_feat_dim > 0:
                    _val_extra = x[:, :, X_DIM:]
                    x = x[:, :, :X_DIM]

                raw_dsdf = x[:, :, 2:10]  # original dsdf before standardization
                dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
                dist_feat = torch.log1p(dist_surf * 10.0)  # log-scale for better gradient flow
                x = (x - stats["x_mean"]) / stats["x_std"]
                # Curvature proxy: norm of first 4 dsdf channels (gradient magnitude) for surface nodes
                curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
                if cfg.foil2_dist:
                    foil2_dist_feat = torch.log1p(raw_dsdf[:, :, 4:8].abs().min(dim=-1, keepdim=True).values * 10.0)
                    x = torch.cat([x, curv, dist_feat, foil2_dist_feat], dim=-1)
                else:
                    x = torch.cat([x, curv, dist_feat], dim=-1)
                # Fourier positional encoding: append sin/cos of (x,y) at 4 learnable frequencies
                raw_xy = x[:, :, :2]
                # Normalize xy to [0,1] per-sample for consistent Fourier encoding
                xy_min = raw_xy.amin(dim=1, keepdim=True)
                xy_max = raw_xy.amax(dim=1, keepdim=True)
                xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
                freqs = torch.cat([model.fourier_freqs_fixed.to(device), model.fourier_freqs_learned.abs()])
                xy_scaled = xy_norm.unsqueeze(-1) * freqs  # [B, N, 2, 4]
                fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)  # [B, N, 16]
                x = torch.cat([x, fourier_pe], dim=-1)
                # Phase 5: append extra features
                if _val_extra is not None:
                    x = torch.cat([x, _val_extra], dim=-1)
                Umag, q = _umag_q(y, mask)
                if cfg.raw_targets:
                    y_norm = (y - raw_stats["y_mean"]) / raw_stats["y_std"]
                else:
                    y_phys = _phys_norm(y, Umag, q)
                    if cfg.log_pressure:
                        y_phys = y_phys.clone()
                        y_phys[:, :, 2:3] = y_phys[:, :, 2:3].abs().add(1).log() * y_phys[:, :, 2:3].sign()
                    y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                # Per-sample std normalization: skip tandem samples
                raw_gap = x[:, 0, 21]
                is_tandem = raw_gap.abs() > 0.5
                B = y_norm.shape[0]
                sample_stds = torch.ones(B, 1, 3, device=device)
                if not cfg.no_perstd and not cfg.raw_targets:
                    if cfg.unified_clamps:
                        channel_clamps = tandem_clamps = torch.tensor([0.2, 0.2, 0.7], device=device)
                    elif cfg.high_p_clamp:
                        channel_clamps = torch.tensor([0.1, 0.1, 2.0], device=device)
                        tandem_clamps = torch.tensor([0.3, 0.3, 2.0], device=device)
                    else:
                        channel_clamps = torch.tensor([0.1, 0.1, 0.5], device=device)
                        tandem_clamps = torch.tensor([0.3, 0.3, 1.0], device=device)
                    for b in range(B):
                        valid = mask[b]
                        if cfg.no_perstd_p:
                            vc = (tandem_clamps[:2] if is_tandem[b] else channel_clamps[:2])
                            sample_stds[b, 0, :2] = y_norm[b, valid, :2].std(dim=0).clamp(min=vc)
                        elif is_tandem[b]:
                            sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                        else:
                            sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)
                if cfg.multiply_std:
                    y_norm_scaled = y_norm * sample_stds
                else:
                    y_norm_scaled = y_norm / sample_stds

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = eval_model({"x": x})["preds"]
                pred = pred.float()
                if cfg.multiply_std:
                    pred_loss = pred * sample_stds
                else:
                    pred_loss = pred / sample_stds
                sq_err = (pred_loss - y_norm_scaled) ** 2
                abs_err = (pred_loss - y_norm_scaled).abs()
                abs_err = abs_err.nan_to_num(0.0)

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += min(
                    (abs_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item(),
                    1e6
                )
                val_surf += min(
                    (abs_err[:, :, 2:3] * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item(),
                    1e6
                )
                n_vbatches += 1

                # Denormalize: phys_stats → Cp space → original scale
                if cfg.raw_targets:
                    pred_orig = pred * raw_stats["y_std"] + raw_stats["y_mean"]
                else:
                    pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
                    if cfg.log_pressure:
                        pred_phys = pred_phys.clone()
                        pred_phys[:, :, 2:3] = pred_phys[:, :, 2:3].sign() * (pred_phys[:, :, 2:3].abs().exp() - 1)
                    if cfg.tight_denorm_clamps:
                        _pd = pred_phys.clone()
                        _pd[:, :, 0:1] = pred_phys[:, :, 0:1].clamp(-5, 5) * Umag
                        _pd[:, :, 1:2] = pred_phys[:, :, 1:2].clamp(-5, 5) * Umag
                        _pd[:, :, 2:3] = pred_phys[:, :, 2:3].clamp(-10, 10) * q
                        pred_orig = _pd
                    else:
                        pred_orig = _phys_denorm(pred_phys, Umag, q)
                y_clamped = y.clamp(-1e6, 1e6)
                err = (pred_orig - y_clamped).abs()
                finite = err.isfinite()
                err = err.where(finite, torch.zeros_like(err))
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += (surf_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
                n_vol += (vol_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()

        val_vol /= max(n_vbatches, 1)
        val_surf /= max(n_vbatches, 1)
        val_vol = float(torch.tensor(val_vol).nan_to_num(0.0).clamp(max=1e6))
        val_surf = float(torch.tensor(val_surf).nan_to_num(0.0).clamp(max=1e6))
        split_loss = val_vol + cfg.surf_weight * val_surf
        mae_surf /= n_surf.clamp(min=1)
        mae_vol /= n_vol.clamp(min=1)

        val_metrics_per_split[split_name] = {
            f"{split_name}/vol_loss":    val_vol,
            f"{split_name}/surf_loss":   val_surf,
            f"{split_name}/loss":        split_loss,
            f"{split_name}/mae_vol_Ux":  mae_vol[0].item(),
            f"{split_name}/mae_vol_Uy":  mae_vol[1].item(),
            f"{split_name}/mae_vol_p":   mae_vol[2].item(),
            f"{split_name}/mae_surf_Ux": mae_surf[0].item(),
            f"{split_name}/mae_surf_Uy": mae_surf[1].item(),
            f"{split_name}/mae_surf_p":  mae_surf[2].item(),
        }
        val_loss_sum += split_loss

    # 4-split val/loss (all splits) — used for checkpoint selection
    _checkpoint_names = VAL_SPLIT_NAMES  # all 4 splits instead of _3split_names
    _checkpoint_losses = [val_metrics_per_split[n][f"{n}/loss"] for n in _checkpoint_names
                          if not (torch.tensor(val_metrics_per_split[n][f"{n}/loss"]).isnan() or
                                  torch.tensor(val_metrics_per_split[n][f"{n}/loss"]).isinf())]
    val_loss_3split = sum(_checkpoint_losses) / max(len(_checkpoint_losses), 1)
    ema_val_loss = val_loss_3split if ema_val_loss == float("inf") else ema_decay_val * ema_val_loss + (1 - ema_decay_val) * val_loss_3split

    if cfg.swad:
        if swad_initial_val is None:
            swad_initial_val = val_loss_3split
        if not swad_collecting and not swad_done:
            if val_loss_3split < swad_initial_val * 0.5:
                swad_collecting = True
        if swad_collecting and not swad_done:
            if val_loss_3split > swad_prev_val:
                swad_done = True
                if swad_checkpoints:
                    avg_state = {k: torch.stack([c[k].float() for c in swad_checkpoints]).mean(0).to(device)
                                 for k in swad_checkpoints[0]}
                    if ema_model is None:
                        ema_model = deepcopy(_base_model)
                    ema_model.load_state_dict(avg_state)
            else:
                snap = {k: v.cpu().clone() for k, v in _base_model.state_dict().items()}
                swad_checkpoints.append(snap)
                if len(swad_checkpoints) > 20:
                    swad_checkpoints.pop(0)
            swad_prev_val = val_loss_3split

    # SWA: uniform weight averaging after swa_start_epoch
    if cfg.swa and epoch >= cfg.swa_start_epoch:
        if swa_model is None:
            swa_model = deepcopy(_base_model)
            swa_n = 1
        else:
            with torch.no_grad():
                for sp, mp in zip(swa_model.parameters(), _base_model.parameters()):
                    sp.data = (sp.data * swa_n + mp.data) / (swa_n + 1)
            swa_n += 1

    # 4-split val/loss (all splits including ood_re)
    _4split_losses = [val_metrics_per_split[n][f"{n}/loss"] for n in VAL_SPLIT_NAMES
                      if not (torch.tensor(val_metrics_per_split[n][f"{n}/loss"]).isnan() or
                              torch.tensor(val_metrics_per_split[n][f"{n}/loss"]).isinf())]
    val_loss_4split = sum(_4split_losses) / max(len(_4split_losses), 1)

    dt = time.time() - t0

    # --- Log to wandb ---
    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": val_loss_3split,
        "val/loss_3split": val_loss_3split,
        "val/loss_4split": val_loss_4split,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
    }
    for split_metrics in val_metrics_per_split.values():
        metrics.update(split_metrics)
    metrics["global_step"] = global_step
    learned_freqs = model.fourier_freqs_learned.abs().detach().cpu().tolist()
    for i, f in enumerate(learned_freqs):
        metrics[f"fourier_freq_{i}"] = f
    wandb.log(metrics)

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_mem_gb = 0.0

    tag = ""
    if ema_val_loss < best_val:
        best_val = ema_val_loss
        best_metrics = {"epoch": epoch + 1, "val_loss": val_loss_3split}
        for split_metrics in val_metrics_per_split.values():
            for k, v in split_metrics.items():
                best_metrics[f"best_{k}"] = v
        if cfg.swa and swa_model is not None:
            save_model = swa_model
        elif ema_model is not None:
            save_model = ema_model
        else:
            save_model = _base_model
        torch.save(save_model.state_dict(), model_path)
        tag = f" * -> {model_path}"

    split_summary = "  ".join(
        f"{name}={val_metrics_per_split[name][f'{name}/loss']:.4f}"
        for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_mem_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[{split_summary}]{tag}"
    )


# --- Final summary ---
total_time = (time.time() - train_start) / 60.0
print("\n" + "=" * 70)
print(f"TRAINING COMPLETE ({total_time:.1f} min)")
print("=" * 70)
if best_metrics:
    print(f"Best model at epoch {best_metrics['epoch']}  (val/loss={best_metrics['val_loss']:.4f})")
    for split_name in VAL_SPLIT_NAMES:
        k_p = f"best_{split_name}/mae_surf_p"
        k_l = f"best_{split_name}/loss"
        if k_p in best_metrics:
            print(f"  {split_name:30s}  loss={best_metrics[k_l]:.4f}  mae_surf_p={best_metrics[k_p]:.1f}")
else:
    print("No completed epochs (timeout too short?).")

wandb.summary.update({"total_epochs": epoch + 1, "total_time_min": total_time})
if best_metrics:
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    print("\nGenerating flow field plots...")
    try:
        if cfg.swa_cyclic and swa_cyclic_model is not None:
            vis_model = swa_cyclic_model
        elif cfg.snapshot_ensemble and snapshot_avg_model is not None:
            vis_model = snapshot_avg_model
        elif cfg.swa and swa_model is not None:
            vis_model = swa_model
        elif ema_model is not None:
            vis_model = ema_model
        else:
            vis_model = _base_model
        _sd = torch.load(model_path, map_location=device, weights_only=True)
        # Strip _orig_mod. prefix from torch.compile state_dict keys if needed
        _sd = {k.removeprefix("_orig_mod."): v for k, v in _sd.items()}
        vis_model.load_state_dict(_sd)
        vis_model.eval()
        plot_dir = Path("plots") / run.id
        n = 1 if cfg.debug else 4
        for split_name, split_ds in val_splits.items():
            samples = []
            for i in range(min(n, len(split_ds))):
                x, y_true, is_surface = split_ds[i]
                with torch.no_grad():
                    x_dev = x.unsqueeze(0).to(device)
                    y_dev = y_true.unsqueeze(0).to(device)
                    is_surf_dev = is_surface.unsqueeze(0).to(device)
                    mask = torch.ones(1, x_dev.shape[1], dtype=torch.bool, device=device)
                    # Phase 5: extract extra features
                    _vis_extra = None
                    if _extra_feat_dim > 0:
                        _vis_extra = x_dev[:, :, X_DIM:]
                        x_dev = x_dev[:, :, :X_DIM]
                    raw_dsdf = x_dev[:, :, 2:10]
                    dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
                    dist_feat = torch.log1p(dist_surf * 10.0)
                    x_n = (x_dev - stats["x_mean"]) / stats["x_std"]
                    curv = x_n[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surf_dev.float().unsqueeze(-1)
                    x_n = torch.cat([x_n, curv, dist_feat], dim=-1)
                    # Fourier PE (must match training loop)
                    raw_xy = x_n[:, :, :2]
                    xy_min = raw_xy.amin(dim=1, keepdim=True)
                    xy_max = raw_xy.amax(dim=1, keepdim=True)
                    xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
                    freqs = torch.cat([vis_model.fourier_freqs_fixed.to(device), vis_model.fourier_freqs_learned.abs()])
                    xy_scaled = xy_norm.unsqueeze(-1) * freqs
                    fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)
                    x_n = torch.cat([x_n, fourier_pe], dim=-1)
                    if _vis_extra is not None:
                        x_n = torch.cat([x_n, _vis_extra], dim=-1)
                    Umag, q = _umag_q(y_dev, mask)
                    pred = vis_model({"x": x_n, "mask": mask})["preds"].float()
                    if cfg.raw_targets:
                        y_pred = (pred * raw_stats["y_std"] + raw_stats["y_mean"]).squeeze(0).cpu()
                    else:
                        pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
                        if cfg.log_pressure:
                            pred_phys = pred_phys.clone()
                            pred_phys[:, :, 2:3] = pred_phys[:, :, 2:3].sign() * (pred_phys[:, :, 2:3].abs().exp() - 1)
                        if cfg.tight_denorm_clamps:
                            _pd = pred_phys.clone()
                            _pd[:, :, 0:1] = pred_phys[:, :, 0:1].clamp(-5, 5) * Umag
                            _pd[:, :, 1:2] = pred_phys[:, :, 1:2].clamp(-5, 5) * Umag
                            _pd[:, :, 2:3] = pred_phys[:, :, 2:3].clamp(-10, 10) * q
                            y_pred = _pd.squeeze(0).cpu()
                        else:
                            y_pred = _phys_denorm(pred_phys, Umag, q).squeeze(0).cpu()
                samples.append((x[:, :2], y_true, y_pred, is_surface))
            images = visualize(samples, out_dir=plot_dir / split_name)
            if images:
                wandb.log({f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images], "global_step": global_step})
    except Exception as e:
        print(f"Warning: flow field visualization failed: {e}")
        wandb.alert(title="Vis failed", text=str(e)[:200], level="WARN")

wandb.finish()
