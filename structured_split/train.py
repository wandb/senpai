# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Train Transolver with structured benchmark splits.

Structured variant of the root train.py. Reads a split manifest and stats file
produced by structured_split/split.py, then trains with four separate val tracks:
  val_in_dist          — raceCar_single random 10% holdout (interpolation sanity)
  val_tandem_transfer  — raceCar_tandem Part2 (unseen tandem front foil)
  val_ood_cond         — cruise Part1+3 frontier 20% (extreme AoA/gap/stagger)
  val_ood_re           — cruise Part2 Re=4.445M (fully OOD Reynolds number)

Run:
  python structured_split/train.py \
      --manifest structured_split/split_manifest.json \
      --stats_file structured_split/split_stats.json

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
import torch
import wandb
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
import simple_parsing as sp

from transolver import Transolver
from utils import visualize
sys.path.insert(0, str(Path(__file__).parent))  # structured_split dir
from prepare_multi import X_DIM, pad_collate, load_structured_split, VAL_SPLIT_NAMES


MAX_TIMEOUT = 30.0  # minutes
MAX_EPOCHS = 50


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
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

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,  # X_DIM=24; fun_dim + space_dim must equal x.shape[-1]
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

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
        n_surf = 0
        n_vol = 0
        n_vbatches = 0

        with torch.no_grad():
            for x, y, is_surface, mask in tqdm(
                vloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [{split_name}]", leave=False
            ):
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
                n_vbatches += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

        val_vol /= max(n_vbatches, 1)
        val_surf /= max(n_vbatches, 1)
        split_loss = val_vol + cfg.surf_weight * val_surf
        mae_surf /= max(n_surf, 1)
        mae_vol /= max(n_vol, 1)

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

    # val/loss = mean across all splits; used for checkpoint selection
    mean_val_loss = val_loss_sum / len(val_loaders)

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
