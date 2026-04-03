#!/usr/bin/env python3
"""Ensemble evaluation for 8 trained CFD surrogate models.

Loads each model checkpoint one at a time (to avoid OOM), runs inference on
all validation splits, collects fully denormalized predictions in original
physical space, averages them, and reports surface MAE for Ux, Uy, p.

Usage:
    CUDA_VISIBLE_DEVICES=0 python ensemble_eval.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections.abc import Mapping

# ── Ensure we can import from cfd_tandemfoil ───────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from data.prepare_multi import X_DIM, pad_collate, load_data, VAL_SPLIT_NAMES

# ── Config ─────────────────────────────────────────────────────────────────
MANIFEST   = str(SCRIPT_DIR / "data/split_manifest.json")
STATS_FILE = str(SCRIPT_DIR / "data/split_stats.json")
MODELS_DIR = SCRIPT_DIR / "models"
BATCH_SIZE = 4
NUM_WORKERS = 4
ASINH_SCALE = 0.75

MODEL_IDS = [
    "tpe30wbe",   # standard s42
    "15ao7p73",   # 48 slices
    "e49brvyy",   # srf256
    "ndesai9p",   # srf4L
    "zje7sq3p",   # 64 slices
    "wp1o95ep",   # srfponly
    "xkarmlqj",   # standard s43
    "ggbkmp56",   # standard s44
]

# slice_num per model (from config.yaml)
MODEL_SLICE_NUM = {
    "tpe30wbe": 96,
    "15ao7p73": 48,
    "e49brvyy": 96,
    "ndesai9p": 96,
    "zje7sq3p": 64,
    "wp1o95ep": 96,
    "xkarmlqj": 96,
    "ggbkmp56": 96,
}

# ── Inline model architecture (copied from train.py) ───────────────────────

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
    """Domain-specific LayerNorm: separate weight/bias for single-foil vs tandem."""
    def __init__(self, dim, zeroinit=False):
        super().__init__()
        self.ln_single = nn.LayerNorm(dim)
        self.ln_tandem = nn.LayerNorm(dim)
        if not zeroinit:
            self.ln_tandem.weight.data.copy_(self.ln_single.weight.data)
            self.ln_tandem.bias.data.copy_(self.ln_single.bias.data)

    def forward(self, x, is_tandem=None):
        if is_tandem is None:
            return self.ln_single(x)
        mask_t = is_tandem.view(-1, 1, 1).expand_as(x)
        return torch.where(mask_t, self.ln_tandem(x), self.ln_single(x))


class Physics_Attention_Irregular_Mesh(nn.Module):
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
            self.register_buffer('slice_mask', torch.zeros(slice_num))
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        if decouple_slice:
            self.in_project_slice_tandem = nn.Linear(dim_head, slice_num)
            torch.nn.init.orthogonal_(self.in_project_slice_tandem.weight)
        if zone_temp:
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
            zone_offset = self.zone_temp_proj(zone_features).reshape(bsz, self.heads, 1, 1)
            temp = temp + zone_offset
        if tandem_mask is not None:
            temp = (temp + self.tandem_temp_offset * tandem_mask).clamp(min=1e-4)
        temp = temp.clamp(min=1e-4)
        if self.decouple_slice and tandem_mask is not None:
            std_logits = self.in_project_slice(x_mid) / temp
            tan_logits = self.in_project_slice_tandem(x_mid) / temp
            is_tan = (tandem_mask > 0.5)
            slice_logits = torch.where(is_tan.expand_as(std_logits), tan_logits, std_logits)
        else:
            slice_logits = self.in_project_slice(x_mid) / temp
        if spatial_bias is not None:
            slice_logits = slice_logits + 0.1 * spatial_bias.unsqueeze(1)
        if self.prog_slices:
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
        num_heads, hidden_dim, dropout, act="gelu", mlp_ratio=4,
        last_layer=False, out_dim=1, slice_num=32,
        linear_no_attention=False, learned_kernel=False,
        field_decoder=False, adaln_output=False, soft_moe=False,
        adaln_all=False, adaln_cond_dim=2, adaln_zero_init=True,
        film_cond=False, decouple_slice=False, zone_temp=False,
        domain_layernorm=False, dln_zeroinit=False,
        domain_velhead=False, prog_slices=False,
        pressure_first=False, pressure_no_detach=False, pressure_deep=False,
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
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
            linear_no_attention=linear_no_attention,
            learned_kernel=learned_kernel,
            decouple_slice=decouple_slice,
            zone_temp=zone_temp,
            prog_slices=prog_slices,
        )
        if adaln_all:
            self.adaln_net = nn.Sequential(
                nn.Linear(adaln_cond_dim, 128), nn.SiLU(),
                nn.Linear(128, hidden_dim * 4),
            )
            if adaln_zero_init:
                nn.init.zeros_(self.adaln_net[-1].weight)
                nn.init.zeros_(self.adaln_net[-1].bias)
        if film_cond:
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
                self.vel_head_conditioned = nn.Sequential(
                    nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
                )
            elif domain_velhead:
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
        dln_it = (tandem_mask.view(-1) > 0.5) if (self.domain_layernorm and tandem_mask is not None) else None
        if self.domain_layernorm:
            def _ln(m, x): return m(x, is_tandem=dln_it)
        else:
            def _ln(m, x): return m(x)
        if self.adaln_all and condition is not None:
            cond_out = self.adaln_net(condition)
            s1, b1, s2, b2 = cond_out.chunk(4, dim=-1)
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
            film_out = self.film_net(condition)
            gamma, beta = film_out.chunk(2, dim=-1)
            fx = gamma.unsqueeze(1) * fx + beta.unsqueeze(1)
        if self.last_layer:
            fx_ln = self.ln_3(fx)
            if self.soft_moe:
                gate = self.gate_net(fx_ln)
                return gate[:, :, 0:1] * self.expert1(fx_ln) + gate[:, :, 1:2] * self.expert2(fx_ln)
            elif self.pressure_first:
                p_pred = self.pres_head(fx_ln)
                p_cond = p_pred if self.pressure_no_detach else p_pred.detach()
                vel_input = torch.cat([fx_ln, p_cond], dim=-1)
                v_pred = self.vel_head_conditioned(vel_input)
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
                cond = self.cond_net(condition)
                scale, shift = cond.chunk(2, dim=-1)
                fx_ln = fx_ln * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                return self.mlp2(fx_ln)
            else:
                return self.mlp2(fx_ln)
        return fx


class SurfaceRefinementHead(nn.Module):
    def __init__(self, n_hidden: int, out_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, p_only: bool = False):
        super().__init__()
        self.p_only = p_only
        actual_out = 1 if p_only else out_dim
        layers: list[nn.Module] = []
        in_dim = n_hidden + out_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, actual_out))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor, base_pred: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([hidden, base_pred], dim=-1)
        correction = self.mlp(inp)
        if self.p_only:
            zeros = torch.zeros(correction.shape[0], base_pred.shape[-1] - 1,
                                device=correction.device, dtype=correction.dtype)
            correction = torch.cat([zeros, correction], dim=-1)
        return correction


class Transolver(nn.Module):
    def __init__(
        self,
        space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
        n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
        slice_num=32, ref=8, unified_pos=False,
        output_fields=None, output_dims=None,
        linear_no_attention=False, learned_kernel=False,
        field_decoder=False, adaln_output=False, soft_moe=False,
        uncertainty_loss=False, adaln_all_blocks=False, adaln_4cond=False,
        adaln_nozero=False, film_cond=False, adaln_decouple=False,
        adaln_zone_temp=False, domain_layernorm=False, dln_zeroinit=False,
        domain_velhead=False, prog_slices=False,
        pressure_first=False, pressure_no_detach=False, pressure_deep=False,
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
        self.output_fields = output_fields
        self.output_dims = output_dims
        if uncertainty_loss:
            self.log_sigma_vol = nn.Parameter(torch.zeros(1))
            self.log_sigma_surf_ux = nn.Parameter(torch.zeros(1))
            self.log_sigma_surf_uy = nn.Parameter(torch.zeros(1))
            self.log_sigma_surf_p = nn.Parameter(torch.zeros(1))
        self.preprocess = GatedMLP2(fun_dim + space_dim, n_hidden * 2, n_hidden)
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.feature_cross = nn.Linear(fun_dim + space_dim, fun_dim + space_dim, bias=False)
        nn.init.eye_(self.feature_cross.weight)
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
        self._pressure_separate = False
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
        nn.init.constant_(self.skip_gate[0].bias, -2.0)
        self.placeholder_scale = nn.Parameter(torch.ones(n_hidden))
        self.placeholder_shift = nn.Parameter(torch.zeros(n_hidden))
        self.re_head = nn.Sequential(nn.Linear(n_hidden, 32), nn.GELU(), nn.Linear(32, 1))
        self.aoa_head = nn.Sequential(nn.Linear(n_hidden, 32), nn.GELU(), nn.Linear(32, 1))
        self.fourier_freqs_fixed = torch.tensor([0.5, 2.0, 8.0, 32.0])
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

    def forward(self, data, pos=None, condition=None):
        x = data.get("x") if isinstance(data, Mapping) else data
        use_cond = self.adaln_all_blocks or self.film_cond
        if use_cond:
            cond_2 = x[:, 0, 13:15]
            if self.adaln_4cond:
                gap_feat = x[:, 0, 21:22]
                surf_frac = (x[:, :, 24].abs() > 0.01).float().mean(dim=1, keepdim=True)
                block_condition = torch.cat([cond_2, gap_feat, surf_frac], dim=-1)
            else:
                block_condition = cond_2
        else:
            block_condition = None
        if self.adaln_zone_temp:
            is_tandem_scalar = (x[:, 0, 21].abs() > 0.01).float()
            gap_mag = x[:, 0, 21].abs()
            re_feat = x[:, 0, 13]
            zone_features = torch.stack([is_tandem_scalar, gap_mag, re_feat], dim=-1)
        else:
            zone_features = None
        x_cross = x * self.feature_cross(x)
        x = x + 0.1 * x_cross
        raw_xy = torch.cat([x[:, :, :2], x[:, :, 24:26]], dim=-1)
        is_tandem = (x[:, 0, 21].abs() > 0.01).float()[:, None, None, None]
        fx = self.preprocess(x)
        fx_pre = fx
        fx = fx * self.placeholder_scale[None, None, :] + self.placeholder_shift[None, None, :]
        for block in self.blocks[:-1]:
            fx = block(fx, raw_xy=raw_xy, tandem_mask=is_tandem, condition=block_condition, zone_features=zone_features)
        fx_deep = fx
        re_pred = self.re_head(fx.mean(dim=1))
        aoa_pred = self.aoa_head(fx.mean(dim=1))
        last_condition = block_condition if use_cond else (x[:, 0, 13:15] if self.adaln_output else None)
        if self._pressure_separate and self.pressure_first:
            fx_for_pressure = fx
            p_sep = self.pressure_sep_mlp(fx_for_pressure)
            fx = self.blocks[-1](fx, raw_xy=raw_xy, tandem_mask=is_tandem, condition=last_condition, zone_features=zone_features)
            fx = torch.cat([fx[:, :, :2], p_sep], dim=-1)
        else:
            fx = self.blocks[-1](fx, raw_xy=raw_xy, tandem_mask=is_tandem, condition=last_condition, zone_features=zone_features)
        gate = self.skip_gate(fx_pre)
        fx = fx + gate * self.out_skip(fx_pre)
        return {"preds": fx, "re_pred": re_pred, "aoa_pred": aoa_pred, "hidden": fx_deep}


# ── Physics normalization helpers ───────────────────────────────────────────

def _umag_q(y, mask):
    n_nodes = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    Ux_mean = (y[:, :, 0] * mask.float()).sum(dim=1, keepdim=True) / n_nodes
    Uy_mean = (y[:, :, 1] * mask.float()).sum(dim=1, keepdim=True) / n_nodes
    Umag = (Ux_mean ** 2 + Uy_mean ** 2).sqrt().clamp(min=1.0).unsqueeze(-1)
    q = 0.5 * Umag ** 2
    return Umag, q


def _phys_norm(y, Umag, q):
    y_p = y.clone()
    y_p[:, :, 0:1] = y[:, :, 0:1] / Umag
    y_p[:, :, 1:2] = y[:, :, 1:2] / Umag
    y_p[:, :, 2:3] = y[:, :, 2:3] / q
    return y_p


def _phys_denorm(y_p, Umag, q):
    y = y_p.clone()
    y[:, :, 0:1] = y_p[:, :, 0:1].clamp(-10, 10) * Umag
    y[:, :, 1:2] = y_p[:, :, 1:2].clamp(-10, 10) * Umag
    y[:, :, 2:3] = y_p[:, :, 2:3].clamp(-20, 20) * q
    return y


# ── Preprocessing: input features (same as validation loop in train.py) ─────

def preprocess_batch(x, y, is_surface, mask, stats, device, model):
    """Prepare input features exactly as train.py validation loop does."""
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    is_surface = is_surface.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    raw_dsdf = x[:, :, 2:10]
    dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
    dist_feat = torch.log1p(dist_surf * 10.0)
    _raw_aoa = x[:, 0, 14:15]  # AoA before standardization

    x = (x - stats["x_mean"]) / stats["x_std"]
    curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
    x = torch.cat([x, curv, dist_feat], dim=-1)  # foil2_dist=False for all models

    # Fourier positional encoding
    raw_xy = x[:, :, :2]
    xy_min = raw_xy.amin(dim=1, keepdim=True)
    xy_max = raw_xy.amax(dim=1, keepdim=True)
    xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
    freqs = torch.cat([model.fourier_freqs_fixed.to(device), model.fourier_freqs_learned.abs()])
    xy_scaled = xy_norm.unsqueeze(-1) * freqs
    fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)
    x = torch.cat([x, fourier_pe], dim=-1)

    return x, y, is_surface, mask, _raw_aoa


def denorm_prediction(pred, sample_stds, _v_freestream, phys_stats, Umag, q):
    """Reverse the full normalization pipeline to get predictions in original space."""
    # pred here is in per-sample-std-scaled space; undo per-sample std
    # (multiply_std=False for all these models, so: pred_loss = pred / sample_stds
    #  and pred = pred_loss * sample_stds to get back to phys_norm + zscore space)
    # At this point pred is already the result after sample_stds multiply-back.

    # Add freestream (residual_prediction=True for all 8 models)
    if _v_freestream is not None:
        pred = pred + _v_freestream

    # Undo z-score
    pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]

    # Undo asinh pressure (asinh_pressure=True, asinh_scale=0.75)
    pred_phys = pred_phys.clone()
    pred_phys[:, :, 2:3] = torch.sinh(pred_phys[:, :, 2:3]) / ASINH_SCALE

    # Physics denorm
    pred_orig = _phys_denorm(pred_phys, Umag, q)
    return pred_orig


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(run_id: str, device: torch.device):
    """Load Transolver + SurfaceRefinementHead for a given run_id."""
    model_dir = MODELS_DIR / f"model-{run_id}"
    slice_num = MODEL_SLICE_NUM[run_id]

    model = Transolver(
        space_dim=2,
        fun_dim=56,
        out_dim=3,
        n_hidden=192,
        n_layers=3,
        n_head=3,
        slice_num=slice_num,
        mlp_ratio=2,
        dropout=0.0,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
        linear_no_attention=False,
        learned_kernel=False,
        field_decoder=True,
        adaln_output=True,
        soft_moe=False,
        uncertainty_loss=False,
        adaln_all_blocks=False,
        adaln_4cond=False,
        adaln_nozero=False,
        film_cond=False,
        adaln_decouple=False,
        adaln_zone_temp=False,
        domain_layernorm=True,
        dln_zeroinit=False,
        domain_velhead=True,
        prog_slices=False,
        pressure_first=True,
        pressure_no_detach=False,
        pressure_deep=True,
    ).to(device)
    model._pressure_separate = False

    ckpt_path = model_dir / "checkpoint.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    # Strip _orig_mod. prefix if present (from compiled models)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        cleaned[new_k] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # SurfaceRefinementHead — all 8 models have refine_head.pt
    # Infer architecture from checkpoint to handle different hidden_dim / n_layers
    refine_path = model_dir / "refine_head.pt"
    rh_state = torch.load(refine_path, map_location=device)
    rh_cleaned = {}
    for k, v in rh_state.items():
        new_k = k.replace("_orig_mod.", "")
        rh_cleaned[new_k] = v
    # Infer hidden_dim, n_layers, and p_only from state_dict shapes
    _rh_first_w = rh_cleaned["mlp.0.weight"]  # shape [hidden_dim, n_hidden+out_dim]
    _rh_hidden_dim = _rh_first_w.shape[0]
    _rh_weight_keys = [k for k in rh_cleaned if "weight" in k]
    # Each hidden layer: linear.weight + LN.weight = 2 weight keys; output: 1 weight key
    _rh_n_layers = (len(_rh_weight_keys) - 1) // 2
    # Infer p_only: output weight shape [1, hidden_dim] means p_only=True, [3, hidden_dim] means False
    _rh_last_w_key = _rh_weight_keys[-1]
    _rh_out_dim = rh_cleaned[_rh_last_w_key].shape[0]
    _rh_p_only = (_rh_out_dim == 1)
    refine_head = SurfaceRefinementHead(n_hidden=192, out_dim=3, hidden_dim=_rh_hidden_dim, n_layers=_rh_n_layers, p_only=_rh_p_only).to(device)
    refine_head.load_state_dict(rh_cleaned, strict=True)
    refine_head.eval()

    return model, refine_head


# ── Per-model inference: collect denorm'd predictions ──────────────────────

@torch.no_grad()
def run_model_inference(run_id, model, refine_head, val_loaders, stats, phys_stats, device):
    """Run inference for one model across all val splits.

    Returns: dict[split_name -> list of (pred_orig [B,N,3], y [B,N,3], is_surface [B,N], mask [B,N])]
    """
    results = {split: [] for split in val_loaders}

    for split_name, vloader in val_loaders.items():
        for x_raw, y_raw, is_surface_raw, mask_raw in tqdm(
            vloader, desc=f"  [{run_id}] {split_name}", leave=False
        ):
            x, y, is_surface, mask, _raw_aoa = preprocess_batch(
                x_raw, y_raw, is_surface_raw, mask_raw, stats, device, model
            )

            Umag, q = _umag_q(y, mask)

            # Target normalization (phys norm + asinh + zscore)
            y_phys = _phys_norm(y, Umag, q)
            y_phys = y_phys.clone()
            y_phys[:, :, 2:3] = torch.asinh(y_phys[:, :, 2:3] * ASINH_SCALE)
            y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

            # Freestream (residual_prediction=True)
            _aoa = _raw_aoa
            _fs_phys = torch.zeros(y_norm.shape[0], 1, 3, device=device)
            _fs_phys[:, 0, 0] = torch.cos(_aoa.squeeze(-1))
            _fs_phys[:, 0, 1] = torch.sin(_aoa.squeeze(-1))
            _v_freestream = (_fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
            y_norm = y_norm - _v_freestream

            # Per-sample std
            raw_gap = x[:, 0, 21]
            is_tandem = raw_gap.abs() > 0.5
            B = y_norm.shape[0]
            channel_clamps = torch.tensor([0.1, 0.1, 0.5], device=device)
            tandem_clamps  = torch.tensor([0.3, 0.3, 1.0], device=device)
            sample_stds = torch.ones(B, 1, 3, device=device)
            for b in range(B):
                valid = mask[b]
                if is_tandem[b]:
                    sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                else:
                    sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)
            # y_norm_scaled = y_norm / sample_stds  (not needed for prediction)

            # Forward pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model({"x": x})
                pred = out["preds"]
                hidden = out["hidden"]
            pred = pred.float()
            hidden = hidden.float()

            # pred_loss = pred / sample_stds  (multiply_std=False)
            pred_loss = pred / sample_stds

            # Surface refinement (non-context version)
            surf_idx = is_surface.nonzero(as_tuple=False)
            if surf_idx.numel() > 0:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    surf_hidden = hidden[surf_idx[:, 0], surf_idx[:, 1]]
                    surf_pred   = pred_loss[surf_idx[:, 0], surf_idx[:, 1]]
                    correction  = refine_head(surf_hidden, surf_pred).float()
                pred_loss = pred_loss.clone()
                pred_loss[surf_idx[:, 0], surf_idx[:, 1]] += correction

            # Back-compute pred for denorm (multiply_std=False, so pred = pred_loss * sample_stds)
            pred = pred_loss * sample_stds

            # Denormalize to original physical space
            pred_orig = denorm_prediction(pred, sample_stds, _v_freestream, phys_stats, Umag, q)

            # Store CPU tensors to save GPU memory
            results[split_name].append((
                pred_orig.cpu(),
                y.cpu(),
                is_surface.cpu(),
                mask.cpu(),
            ))

    return results


# ── MAE computation ─────────────────────────────────────────────────────────

def compute_surface_mae(all_results_per_model, split_names):
    """
    Compute per-split surface MAE for each individual model and the ensemble.

    all_results_per_model: list of dicts (one per model)
        each dict: split_name -> list of (pred_orig, y, is_surface, mask)

    Returns:
        individual_mae: list of dicts model_id -> split -> {Ux, Uy, p}
        ensemble_mae: dict split -> {Ux, Uy, p}
    """
    n_models = len(all_results_per_model)
    n_batches_per_split = {
        split: len(all_results_per_model[0][split]) for split in split_names
    }

    individual_mae = []
    for m_idx in range(n_models):
        mae_dict = {}
        for split in split_names:
            mae_surf = torch.zeros(3)
            n_surf   = torch.zeros(3)
            for pred_orig, y, is_surface, mask in all_results_per_model[m_idx][split]:
                surf_mask = mask & is_surface
                y_clamped = y.clamp(-1e6, 1e6)
                err = (pred_orig - y_clamped).abs()
                finite = err.isfinite()
                err = err.where(finite, torch.zeros_like(err))
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf   += (surf_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
            mae_surf /= n_surf.clamp(min=1)
            mae_dict[split] = {"Ux": mae_surf[0].item(), "Uy": mae_surf[1].item(), "p": mae_surf[2].item()}
        individual_mae.append(mae_dict)

    # Ensemble: average predictions per batch, then compute MAE
    ensemble_mae = {}
    for split in split_names:
        mae_surf = torch.zeros(3)
        n_surf   = torch.zeros(3)
        n_batches = n_batches_per_split[split]
        for b_idx in range(n_batches):
            # Stack predictions from all models
            preds = torch.stack([all_results_per_model[m][split][b_idx][0] for m in range(n_models)], dim=0)
            ensemble_pred = preds.mean(dim=0)  # [B, N, 3]
            # Ground truth and masks from model 0 (same for all)
            _, y, is_surface, mask = all_results_per_model[0][split][b_idx]
            surf_mask = mask & is_surface
            y_clamped = y.clamp(-1e6, 1e6)
            err = (ensemble_pred - y_clamped).abs()
            finite = err.isfinite()
            err = err.where(finite, torch.zeros_like(err))
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf   += (surf_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
        mae_surf /= n_surf.clamp(min=1)
        ensemble_mae[split] = {"Ux": mae_surf[0].item(), "Uy": mae_surf[1].item(), "p": mae_surf[2].item()}

    return individual_mae, ensemble_mae


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data...")
    train_ds, val_splits, stats, _ = load_data(MANIFEST, STATS_FILE, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(
        collate_fn=pad_collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loaders = {
        name: DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
        for name, subset in val_splits.items()
    }

    # ── Compute phys_stats over training set (once) ──────────────────────────
    print("Computing phys_stats over training set...")
    _phys_sum    = torch.zeros(3, device=device)
    _phys_sq_sum = torch.zeros(3, device=device)
    _phys_n      = 0.0
    _stats_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    # Use a dummy model just to get fourier freqs for stats computation is not needed
    # (stats computation doesn't use model); but we need fourier_freqs_learned for val input
    # — we compute stats before loading any model

    with torch.no_grad():
        for _x, _y, _is_surf, _mask in tqdm(_stats_loader, desc="Phys stats", leave=False):
            _y, _mask = _y.to(device), _mask.to(device)
            _Um, _q = _umag_q(_y, _mask)
            _yp = _phys_norm(_y, _Um, _q)
            _yp = _yp.clone()
            _yp[:, :, 2:3] = torch.asinh(_yp[:, :, 2:3] * ASINH_SCALE)
            _m = _mask.float().unsqueeze(-1)
            _phys_sum    += (_yp * _m).sum(dim=(0, 1))
            _phys_sq_sum += (_yp ** 2 * _m).sum(dim=(0, 1))
            _phys_n      += _mask.float().sum().item()

    _pmean = (_phys_sum / _phys_n).float()
    _pstd  = ((_phys_sq_sum / _phys_n - _pmean ** 2).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
    phys_stats = {"y_mean": _pmean, "y_std": _pstd}
    print(f"  Cp stats — mean: {_pmean.tolist()}, std: {_pstd.tolist()}")

    # ── Per-model inference ───────────────────────────────────────────────────
    all_results_per_model = []
    split_names = list(val_splits.keys())

    for run_id in MODEL_IDS:
        print(f"\n{'='*60}")
        print(f"Model: {run_id}  (slice_num={MODEL_SLICE_NUM[run_id]})")
        t0 = time.time()

        model, refine_head = load_model(run_id, device)
        results = run_model_inference(run_id, model, refine_head, val_loaders, stats, phys_stats, device)
        all_results_per_model.append(results)

        # Peak GPU memory
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        print(f"  Done in {time.time()-t0:.1f}s  |  peak GPU mem: {mem_gb:.2f} GB")

        # Free model from GPU
        del model, refine_head
        torch.cuda.empty_cache()

    # ── Compute MAE ───────────────────────────────────────────────────────────
    print("\n\nComputing MAE...")
    individual_mae, ensemble_mae = compute_surface_mae(all_results_per_model, split_names)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RESULTS — Surface MAE in original physical space")
    print("="*80)

    # Per split, print each model + ensemble
    for split in split_names:
        print(f"\n{'─'*60}")
        print(f"Split: {split}")
        print(f"{'─'*60}")
        print(f"  {'Model':<14}  {'Ux MAE':>10}  {'Uy MAE':>10}  {'p MAE':>12}")
        print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*12}")
        for m_idx, run_id in enumerate(MODEL_IDS):
            m = individual_mae[m_idx][split]
            print(f"  {run_id:<14}  {m['Ux']:>10.6f}  {m['Uy']:>10.6f}  {m['p']:>12.6f}")
        e = ensemble_mae[split]
        print(f"  {'ENSEMBLE':<14}  {e['Ux']:>10.6f}  {e['Uy']:>10.6f}  {e['p']:>12.6f}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY — Ensemble Surface MAE")
    print("="*80)
    print(f"  {'Split':<28}  {'Ux MAE':>10}  {'Uy MAE':>10}  {'p MAE':>12}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*12}")
    for split in split_names:
        e = ensemble_mae[split]
        print(f"  {split:<28}  {e['Ux']:>10.6f}  {e['Uy']:>10.6f}  {e['p']:>12.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
