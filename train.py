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
from data.prepare_multi import X_DIM, pad_collate, load_data, VAL_SPLIT_NAMES

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


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics attention for irregular meshes in 1D/2D/3D space."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.tandem_temp_offset = nn.Parameter(torch.zeros(1, heads, 1, 1))

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.slice_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.attn_scale = nn.Parameter(torch.ones(1, self.heads, 1, 1) * 10.0)

    def forward(self, x, spatial_bias=None, tandem_mask=None):
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
        if tandem_mask is not None:
            temp = (temp + self.tandem_temp_offset * tandem_mask).clamp(min=1e-4)
        slice_logits = self.in_project_slice(x_mid) / temp
        if spatial_bias is not None:
            slice_logits = slice_logits + 0.1 * spatial_bias.unsqueeze(1)
        slice_weights = self.softmax(slice_logits)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        slice_token_kv = slice_token.mean(dim=1, keepdim=True)  # shared K,V: (bsz, 1, slice_num, dim_head)
        k_slice_token = self.to_k(slice_token_kv).expand(-1, self.heads, -1, -1)
        v_slice_token = self.to_v(slice_token_kv).expand(-1, self.heads, -1, -1)
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
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        self.spatial_bias = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, slice_num),
        )
        nn.init.zeros_(self.spatial_bias[-1].weight)
        nn.init.zeros_(self.spatial_bias[-1].bias)
        self.ln_1_post = nn.LayerNorm(hidden_dim)
        self.ln_2_post = nn.LayerNorm(hidden_dim)
        self.se_fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.se_fc2 = nn.Linear(hidden_dim // 4, hidden_dim)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.se_fc2.bias)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, raw_xy=None, tandem_mask=None):
        sb = self.spatial_bias(raw_xy) if raw_xy is not None else None
        fx = self.ln_1_post(self.attn(self.ln_1(fx), spatial_bias=sb, tandem_mask=tandem_mask) + fx)
        fx = self.ln_2_post(self.mlp(self.ln_2(fx)) + fx)
        se = fx.mean(dim=1, keepdim=True)
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        fx = fx * se
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
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
    ):
        super().__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        if output_fields is None or output_dims is None:
            raise ValueError("output_fields and output_dims must be provided")
        if len(output_fields) != len(output_dims):
            raise ValueError("output_fields and output_dims must have the same length")
        if out_dim != sum(output_dims):
            raise ValueError("out_dim must equal sum(output_dims)")
        self.output_fields = output_fields
        self.output_dims = output_dims

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
                )
                for idx in range(n_layers)
            ]
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
            fx = block(fx, raw_xy=raw_xy, tandem_mask=is_tandem)

        # Auxiliary Re prediction from pre-output-head hidden representation
        re_pred = self.re_head(fx.mean(dim=1))  # [B, 1]
        aoa_pred = self.aoa_head(fx.mean(dim=1))

        fx = self.blocks[-1](fx, raw_xy=raw_xy, tandem_mask=is_tandem)
        gate = self.skip_gate(fx_pre)
        fx = fx + gate * self.out_skip(fx_pre)
        self._validate_output_dims(fx)
        return {"preds": fx, "re_pred": re_pred, "aoa_pred": aoa_pred}


# ---------------------------------------------------------------------------
# End Transolver model
# ---------------------------------------------------------------------------


MAX_TIMEOUT = 30.0  # minutes
MAX_EPOCHS = 100


@dataclass
class Config:
    lr: float = 2.6e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    surf_weight: float = 20.0
    manifest: str = "data/split_manifest.json"
    stats_file: str = "data/split_stats.json"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


cfg = sp.parse(Config)

if cfg.debug:
    MAX_EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG MODE]" if cfg.debug else ""))

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

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
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
        _m = _mask.float().unsqueeze(-1)  # [B, N, 1]
        _phys_sum += (_yp * _m).sum(dim=(0, 1))
        _phys_sq_sum += (_yp ** 2 * _m).sum(dim=(0, 1))
        _phys_n += _mask.float().sum().item()
_pmean = (_phys_sum / _phys_n).float()
_pstd = ((_phys_sq_sum / _phys_n - _pmean ** 2).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
phys_stats = {"y_mean": _pmean, "y_std": _pstd}
print(f"  Cp stats — mean: {_pmean.tolist()}, std: {_pstd.tolist()}")

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2 + 2 + 32,  # +1 curv, +1 dist_feat, +32 fourier PE
    out_dim=3,
    n_hidden=192,  # regime-w: full width with finer routing
    n_layers=1,       # was 2 — 1 layer for maximum epochs in 30 min
    n_head=3,
    slice_num=48,  # regime-h: more slices for finer spatial decomposition
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
torch._functorch.config.donated_buffer = False  # required for retain_graph=True in PCGrad
model = torch.compile(model, mode="default")
_base_model = model._orig_mod if hasattr(model, '_orig_mod') else model

from copy import deepcopy
ema_model = None
ema_start_epoch = 40
ema_decay = 0.998

n_params = sum(p.numel() for p in model.parameters())


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
base_opt = torch.optim.AdamW([
    {'params': attn_params, 'lr': cfg.lr * 0.5},
    {'params': other_params, 'lr': cfg.lr}
], weight_decay=cfg.weight_decay)
optimizer = Lookahead(base_opt, k=10, alpha=0.8)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(base_opt, start_factor=0.2, total_iters=10)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=62, eta_min=5e-5)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    base_opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10]
)

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
    for x, y, is_surface, mask in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x = (x - stats["x_mean"]) / stats["x_std"]
        # Curvature proxy: norm of first 4 dsdf channels (gradient magnitude) for surface nodes
        curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
        dist_surf = x[:, :, 2:10].abs().min(dim=-1, keepdim=True).values  # [B, N, 1]
        dist_feat = torch.log1p(dist_surf * 10.0)  # log-scale for better gradient flow
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
        if model.training and epoch < 60:
            noise_scale = 0.05 * (1 - epoch / 60)
            x[:, :, 2:25] = x[:, :, 2:25] + noise_scale * torch.randn_like(x[:, :, 2:25])
        Umag, q = _umag_q(y, mask)
        y_phys = _phys_norm(y, Umag, q)
        y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
        if model.training:
            noise_progress = min(1.0, epoch / 60)
            vel_noise = 0.015 * (1 - noise_progress) + 0.003 * noise_progress
            p_noise = 0.008 * (1 - noise_progress) + 0.001 * noise_progress
            noise_scale = torch.tensor([vel_noise, vel_noise, p_noise], device=device)
            y_norm = y_norm + noise_scale * torch.randn_like(y_norm)

        # Per-sample std normalization: skip tandem samples (gap feature index 21)
        raw_gap = x[:, 0, 21]
        is_tandem = raw_gap.abs() > 0.5
        B = y_norm.shape[0]
        sample_stds = torch.ones(B, 1, 3, device=device)
        channel_clamps = torch.tensor([0.1, 0.1, 0.5], device=device)
        tandem_clamps = torch.tensor([0.3, 0.3, 1.0], device=device)
        if model.training:
            for b in range(B):
                valid = mask[b]
                if is_tandem[b]:
                    sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                else:
                    sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)
            y_norm = y_norm / sample_stds

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model({"x": x})
            pred = out["preds"]
            re_pred = out["re_pred"]
            aoa_pred = out["aoa_pred"]
        pred = pred.float()
        re_pred = re_pred.float()
        aoa_pred = aoa_pred.float()
        if model.training:
            pred = pred / sample_stds
        sq_err = (pred - y_norm) ** 2
        abs_err = (pred - y_norm).abs()
        if epoch < 10:
            is_tandem_curr = (x[:, :, -8:].abs().sum(dim=(1, 2)) > 0.01)
            sample_mask = (~is_tandem_curr).float()[:, None, None]
            abs_err = abs_err * sample_mask
        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface

        # Progressive resolution: subsample volume nodes in loss early in training
        # Ramps from 10% → 100% of volume nodes over first 40 epochs
        if epoch < 40:
            vol_keep_ratio = 0.05 + 0.95 * (epoch / 40)
            vol_indices = vol_mask.nonzero(as_tuple=False)
            n_vol = vol_indices.shape[0]
            n_keep = max(int(n_vol * vol_keep_ratio), 1)
            perm = torch.randperm(n_vol, device=vol_mask.device)[:n_keep]
            vol_mask_train = torch.zeros_like(vol_mask)
            if n_keep > 0:
                vol_mask_train[vol_indices[perm, 0], vol_indices[perm, 1]] = True
        else:
            vol_mask_train = vol_mask

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
        tandem_boost = torch.where(is_tandem_batch, adaptive_boost, 1.0).to(device)
        surf_loss = (surf_per_sample * tandem_boost).mean()
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

            pred_coarse = pred_trunc.reshape(B, n_groups, coarse_pool_size, C).mean(dim=2)
            y_coarse = y_trunc.reshape(B, n_groups, coarse_pool_size, C).mean(dim=2)
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

        # PCGrad: in-dist (Group A) vs all-OOD (Group B) gradient projection
        # Group B = tandem + extreme-Re (>1σ) + extreme-AoA (>1σ), Group A = rest
        is_ood_pcgrad = is_tandem_batch | (x[:, 0, 13] > 1.0) | (x[:, 0, 14].abs() > 1.0)
        is_indist_pcgrad = ~is_ood_pcgrad
        use_pcgrad = is_indist_pcgrad.any() and is_ood_pcgrad.any()

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
            optimizer.zero_grad()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch >= ema_start_epoch:
            if ema_model is None:
                ema_model = deepcopy(_base_model)
            else:
                with torch.no_grad():
                    for ep, mp in zip(ema_model.parameters(), _base_model.parameters()):
                        ep.data.mul_(ema_decay).add_(mp.data, alpha=1 - ema_decay)
        global_step += 1
        wandb.log({"train/loss": loss.item(), "train/surf_weight": surf_weight, "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1
        pbar.set_postfix(vol=f"{vol_loss.item():.3f}", surf=f"{surf_loss.item():.3f}")

    scheduler.step()
    if epoch >= 50:
        with torch.no_grad():
            _base_model.blocks[0].attn.temperature.data.clamp_(max=0.25)
    epoch_vol /= n_batches
    epoch_surf /= n_batches
    prev_vol_loss = epoch_vol
    prev_surf_loss = epoch_surf

    # --- Validate across all splits ---
    eval_model = ema_model if ema_model is not None else model
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

                x = (x - stats["x_mean"]) / stats["x_std"]
                # Curvature proxy: norm of first 4 dsdf channels (gradient magnitude) for surface nodes
                curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
                dist_surf = x[:, :, 2:10].abs().min(dim=-1, keepdim=True).values  # [B, N, 1]
                dist_feat = torch.log1p(dist_surf * 10.0)  # log-scale for better gradient flow
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
                Umag, q = _umag_q(y, mask)
                y_phys = _phys_norm(y, Umag, q)
                y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                # Per-sample std normalization: skip tandem samples
                raw_gap = x[:, 0, 21]
                is_tandem = raw_gap.abs() > 0.5
                B = y_norm.shape[0]
                sample_stds = torch.ones(B, 1, 3, device=device)
                channel_clamps = torch.tensor([0.1, 0.1, 0.5], device=device)
                tandem_clamps = torch.tensor([0.3, 0.3, 1.0], device=device)
                for b in range(B):
                    valid = mask[b]
                    if is_tandem[b]:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                    else:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)
                y_norm_scaled = y_norm / sample_stds

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = eval_model({"x": x})["preds"]
                pred = pred.float()
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
                pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
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
    # weights = [in_dist, tandem, ood_cond, ood_re]: tandem 2x, ood_re 1.5x
    _checkpoint_names = VAL_SPLIT_NAMES  # all 4 splits instead of _3split_names
    _split_weights = {"val_in_dist": 1.0, "val_tandem_transfer": 2.0, "val_ood_cond": 1.0, "val_ood_re": 1.5}
    _weighted_loss_sum = 0.0
    _weight_total = 0.0
    for n in _checkpoint_names:
        v = val_metrics_per_split[n][f"{n}/loss"]
        if not (torch.tensor(v).isnan() or torch.tensor(v).isinf()):
            w = _split_weights[n]
            _weighted_loss_sum += v * w
            _weight_total += w
    val_loss_3split = _weighted_loss_sum / max(_weight_total, 1e-8)
    ema_val_loss = val_loss_3split if ema_val_loss == float("inf") else ema_decay_val * ema_val_loss + (1 - ema_decay_val) * val_loss_3split

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
        save_model = ema_model if ema_model is not None else model
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

if best_metrics:
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    print("\nGenerating flow field plots...")
    vis_model = ema_model if ema_model is not None else model
    vis_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
                x_n = (x_dev - stats["x_mean"]) / stats["x_std"]
                curv = x_n[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surf_dev.float().unsqueeze(-1)
                dist_surf = x_n[:, :, 2:10].abs().min(dim=-1, keepdim=True).values
                dist_feat = torch.log1p(dist_surf * 10.0)
                x_n = torch.cat([x_n, curv, dist_feat], dim=-1)
                Umag, q = _umag_q(y_dev, mask)
                pred = vis_model({"x": x_n})["preds"].float()
                pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
                y_pred = _phys_denorm(pred_phys, Umag, q).squeeze(0).cpu()
            samples.append((x[:, :2], y_true, y_pred, is_surface))
        images = visualize(samples, out_dir=plot_dir / split_name)
        if images:
            wandb.log({f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images], "global_step": global_step})

wandb.finish()
