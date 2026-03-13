# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Train Transolver on full-field airfoil flow prediction with separate surface/volume losses."""

import os
import time
import torch
import wandb
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset
import simple_parsing as sp

from prepare import FullFieldDataset, pad_collate, DATA_ROOT
from transolver import Transolver
from utils import visualize, dataset_stats


MAX_TIMEOUT = 5.0 # minutes
MAX_EPOCHS = 20
@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    dataset: str = "raceCar_single_randomFields"
    wandb_group: str | None = None  # group related runs (e.g. iterations on the same idea)
    wandb_name: str | None = None  # name for this specific run
    agent: str | None = None  # agent name (e.g. pepe) for filtering in W&B
    debug: bool = False


cfg = sp.parse(Config)

if cfg.debug:
    MAX_EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG MODE]" if cfg.debug else ""))

# Eager cache — all 899 samples preprocessed into RAM (~13GB)
ds = FullFieldDataset([DATA_ROOT / f"{cfg.dataset}.pickle"], cache_size=0)
train_ds, val_ds = random_split(ds, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

if cfg.debug:
    train_ds = Subset(train_ds, range(min(8, len(train_ds))))
    val_ds = Subset(val_ds, range(min(4, len(val_ds))))

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

stats = {k: v.to(device) for k, v in dataset_stats[cfg.dataset].items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)

model_config = dict(
    space_dim=2,
    fun_dim=16,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)


model = Transolver(
    **model_config
).to(device)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)


# --- wandb ---
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "senpai"),
    group=cfg.wandb_group,
    name=cfg.wandb_name,
    tags=[cfg.agent] if cfg.agent else [],
    config={**asdict(cfg), "model_config": model_config, "n_params": n_params, "train_samples": len(train_ds), "val_samples": len(val_ds)},
    mode="offline" if cfg.debug else "online",
)

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True)
model_path = model_dir / f"checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

best_val = float("inf")
best_metrics = {}
train_start = time.time()

for epoch in range(MAX_EPOCHS):
    # Check wall-clock timeout
    elapsed_min = (time.time() - train_start) / 60.0
    if elapsed_min >= MAX_TIMEOUT:
        print(f"Wall-clock limit reached ({elapsed_min:.1f} min >= {MAX_TIMEOUT} min). Stopping.")
        break

    t0 = time.time()

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
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        pred = model({"x": x})["preds"]
        sq_err = (pred - y_norm) ** 2

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss
        wandb.log({"train/loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1
        pbar.set_postfix(vol=f"{vol_loss.item():.3f}", surf=f"{surf_loss.item():.3f}")

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # --- Validate ---
    model.eval()
    val_vol = 0.0
    val_surf = 0.0
    mae_surf = torch.zeros(3, device=device)
    mae_vol = torch.zeros(3, device=device)
    n_surf = 0
    n_vol = 0
    n_val = 0

    with torch.no_grad():
        for x, y, is_surface, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [val]", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]

            pred = model({"x": x})["preds"]
            sq_err = (pred - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
            val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
            n_val += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            err = (pred_orig - y).abs()
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += surf_mask.sum().item()
            n_vol += vol_mask.sum().item()

    val_vol /= n_val
    val_surf /= n_val
    val_loss = val_vol + cfg.surf_weight * val_surf
    mae_surf /= max(n_surf, 1)
    mae_vol /= max(n_vol, 1)

    dt = time.time() - t0

    # --- Log to wandb ---
    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/vol_loss": val_vol,
        "val/surf_loss": val_surf,
        "val/loss": val_loss,
        "val/mae_vol_Ux": mae_vol[0].item(),
        "val/mae_vol_Uy": mae_vol[1].item(),
        "val/mae_vol_p": mae_vol[2].item(),
        "val/mae_surf_Ux": mae_surf[0].item(),
        "val/mae_surf_Uy": mae_surf[1].item(),
        "val/mae_surf_p": mae_surf[2].item(),
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
    }
    wandb.log(metrics, commit=False)

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_mem_gb = 0.0

    tag = ""
    if val_loss < best_val:
        best_val = val_loss
        best_metrics = {
            "mae_vol_Ux": mae_vol[0].item(),
            "mae_vol_Uy": mae_vol[1].item(),
            "mae_vol_p": mae_vol[2].item(),
            "mae_surf_Ux": mae_surf[0].item(),
            "mae_surf_Uy": mae_surf[1].item(),
            "mae_surf_p": mae_surf[2].item(),
            "epoch": epoch + 1,
            "val_loss_loss": val_loss,
        }
        torch.save(model.state_dict(), model_path)
        tag = f" * -> {model_path}"

    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_mem_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[vol={val_vol:.4f} surf={val_surf:.4f}]  "
        f"mae_vol=[Ux:{mae_vol[0]:.2f} Uy:{mae_vol[1]:.2f} p:{mae_vol[2]:.1f}]  "
        f"mae_surf=[Ux:{mae_surf[0]:.2f} Uy:{mae_surf[1]:.2f} p:{mae_surf[2]:.1f}]{tag}"
    )


# --- Final summary ---
total_time = (time.time() - train_start) / 60.0
print("\n" + "=" * 70)
print(f"TRAINING COMPLETE ({total_time:.1f} min)")
print("=" * 70)
if best_metrics:
    print(f"Best model at epoch {best_metrics['epoch']}")
    print(f"  Val total loss: {best_metrics['val_loss_loss']:.4f}")
    print(f"  Volume  MAE:  Ux={best_metrics['mae_vol_Ux']:.2f}  Uy={best_metrics['mae_vol_Uy']:.2f}  p={best_metrics['mae_vol_p']:.1f}")
    print(f"  Surface MAE:  Ux={best_metrics['mae_surf_Ux']:.2f}  Uy={best_metrics['mae_surf_Uy']:.2f}  p={best_metrics['mae_surf_p']:.1f}")
else:
    print("No completed epochs (timeout too short?).")

if best_metrics:
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    # Generate visualizations with best model
    print("\nGenerating flow field plots...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    plot_dir = Path("plots") / run.id
    images = visualize(model, val_ds, stats, device, n_samples=2 if cfg.debug else 4, out_dir=plot_dir)
    if images:
        wandb.log({"val_predictions": [wandb.Image(str(p)) for p in images]})

wandb.finish()
