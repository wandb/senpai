# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Train Transolver with structured benchmark splits.

This is the primary training script for the structured-split experiment track.
(The root train.py is a separate, simpler single-dataset script for earlier work.)

Reads a split manifest and stats file produced by structured_split/split.py,
then trains with four separate val tracks:
  val_in_dist          — raceCar_single random holdout (interpolation sanity)
  val_tandem_transfer  — raceCar_tandem Part2 (unseen tandem front foil)
  val_ood_cond         — cruise Part1+3 frontier 20% (extreme AoA/gap/stagger)
  val_ood_re           — cruise Part2 Re=4.445M (fully OOD Reynolds number)

Run (manifest and stats default to the committed files):
  python structured_split/structured_train.py --agent <name> --wandb_name "<name>/<desc>"

KNOWN LIMITATIONS (inherited from read-only prepare.py):
  - Only NACA[0] and AoA[0] are encoded in x. Foil 2 is implicit in dsdf/saf.
  - SURFACE_IDS=(5,6) misses boundary ID 7 (foil 2 surface in tandem data).
    Tandem surface loss is therefore underweighted.
"""

import sys
from pathlib import Path

# Reach repo root so we can import prepare, transolver, utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
from collections.abc import Mapping

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

from utils import visualize
sys.path.insert(0, str(Path(__file__).parent))  # structured_split dir
from prepare_multi import X_DIM, pad_collate, load_structured_split, VAL_SPLIT_NAMES


# ---------------------------------------------------------------------------
# Transolver model (inlined from repo root transolver.py so students can
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

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
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
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dropout_p = self.dropout.p if self.training else 0.0
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token,
            k_slice_token,
            v_slice_token,
            dropout_p=dropout_p,
            is_causal=False,
        )

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
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
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
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
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
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
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

        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)
        self._validate_output_dims(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# End Transolver model
# ---------------------------------------------------------------------------


MAX_TIMEOUT = 30.0  # minutes
MAX_EPOCHS = 100


@dataclass
class Config:
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 20.0
    manifest: str = "structured_split/split_manifest.json"
    stats_file: str = "structured_split/split_stats.json"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


cfg = sp.parse(Config)

if cfg.debug:
    MAX_EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG MODE]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_structured_split(
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
    y[:, :, 0:1] = y_p[:, :, 0:1].clamp(-50, 50) * Umag
    y[:, :, 1:2] = y_p[:, :, 1:2].clamp(-50, 50) * Umag
    y[:, :, 2:3] = y_p[:, :, 2:3].clamp(-100, 100) * q
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
    fun_dim=X_DIM - 2,  # X_DIM=24; fun_dim + space_dim must equal x.shape[-1]
    out_dim=3,
    n_hidden=128,
    n_layers=1,       # was 2 — 1 layer for maximum epochs in 30 min
    n_head=4,
    slice_num=32,  # was 64 — fewer slices for faster attention, more epochs
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - 5)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
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
best_metrics = {}
global_step = 0
train_start = time.time()

for epoch in range(MAX_EPOCHS):
    elapsed_min = (time.time() - train_start) / 60.0
    if elapsed_min >= MAX_TIMEOUT:
        print(f"Wall-clock limit reached ({elapsed_min:.1f} min >= {MAX_TIMEOUT} min). Stopping.")
        break

    t0 = time.time()

    # Dynamic surface weight: linear ramp from 5 → 30 over training
    sw_start, sw_end = 5.0, 30.0
    progress = epoch / MAX_EPOCHS
    surf_weight = sw_start + (sw_end - sw_start) * progress

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
        Umag, q = _umag_q(y, mask)
        y_phys = _phys_norm(y, Umag, q)
        y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
        if model.training:
            y_norm = y_norm + 0.01 * torch.randn_like(y_norm)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model({"x": x})["preds"]
        pred = pred.float()
        sq_err = (pred - y_norm) ** 2
        abs_err = (pred - y_norm).abs()
        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (abs_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + surf_weight * surf_loss

        # Multi-scale loss: coarse spatial pooling
        coarse_pool_size = 64
        B, N, C = pred.shape
        n_groups = N // coarse_pool_size
        if n_groups > 1:
            # Pool predictions and targets over groups of 64 nodes
            pred_trunc = pred[:, :n_groups * coarse_pool_size]
            y_trunc = y_norm[:, :n_groups * coarse_pool_size]
            mask_trunc = mask[:, :n_groups * coarse_pool_size]

            pred_coarse = pred_trunc.reshape(B, n_groups, coarse_pool_size, C).mean(dim=2)
            y_coarse = y_trunc.reshape(B, n_groups, coarse_pool_size, C).mean(dim=2)
            mask_coarse = mask_trunc.reshape(B, n_groups, coarse_pool_size).any(dim=2)

            coarse_err = (pred_coarse - y_coarse).abs()
            coarse_loss = (coarse_err * mask_coarse.unsqueeze(-1)).sum() / mask_coarse.sum().clamp(min=1)
            loss = loss + 2.0 * coarse_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "train/surf_weight": surf_weight, "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1
        pbar.set_postfix(vol=f"{vol_loss.item():.3f}", surf=f"{surf_loss.item():.3f}")

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # --- Validate across all splits ---
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
                Umag, q = _umag_q(y, mask)
                y_phys = _phys_norm(y, Umag, q)
                y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model({"x": x})["preds"]
                pred = pred.float()
                sq_err = (pred - y_norm) ** 2
                abs_err = (pred - y_norm).abs()

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += min(
                    (abs_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item(),
                    1e12
                )
                val_surf += (abs_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
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

    # val/loss = mean across finite splits; NaN-robust for checkpoint selection
    finite_losses = [val_metrics_per_split[name][f"{name}/loss"]
                     for name in VAL_SPLIT_NAMES
                     if not (torch.tensor(val_metrics_per_split[name][f"{name}/loss"]).isnan() or
                             torch.tensor(val_metrics_per_split[name][f"{name}/loss"]).isinf())]
    mean_val_loss = sum(finite_losses) / max(len(finite_losses), 1)

    dt = time.time() - t0

    # --- Log to wandb ---
    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": mean_val_loss,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
    }
    for split_metrics in val_metrics_per_split.values():
        metrics.update(split_metrics)
    metrics["global_step"] = global_step
    wandb.log(metrics)

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_mem_gb = 0.0

    tag = ""
    if mean_val_loss < best_val:
        best_val = mean_val_loss
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss}
        for split_metrics in val_metrics_per_split.values():
            for k, v in split_metrics.items():
                best_metrics[f"best_{k}"] = v
        torch.save(model.state_dict(), model_path)
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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    plot_dir = Path("plots") / run.id
    # Visualize from val_in_dist — same distribution as original val_ds
    images = visualize(model, val_splits["val_in_dist"], stats, device,
                       n_samples=2 if cfg.debug else 4, out_dir=plot_dir)
    if images:
        wandb.log({"val_predictions": [wandb.Image(str(p)) for p in images], "global_step": global_step})

wandb.finish()
