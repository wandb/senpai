#!/usr/bin/env python3
"""Pre-compute ensemble soft targets for knowledge distillation.

Loads model checkpoints one at a time, runs inference on the training set,
and averages predictions in physical space. Saves per-sample soft targets
indexed by train_ds Subset position.

Usage:
    cd cfd_tandemfoil
    python precompute_ensemble.py \
        --run_ids j9w7d1r7 mc4jvgqj cbbvhl62 bigqfn3k bqhg6lq8 5ukk7wv6 xlnhwuqc ii1tz4vv \
        --output ensemble_soft_targets.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.prepare_multi import load_data, pad_collate
from eval_ensemble import load_model_and_refine, _umag_q, _phys_norm, _phys_denorm


def precompute_soft_targets(run_ids, output_path, asinh_scale=0.75, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Models: {len(run_ids)}")

    # Load training data
    print("Loading data...")
    train_ds, _, stats, _ = load_data("data/split_manifest.json", "data/split_stats.json")
    x_stats = {k: stats[k].to(device) for k in ["x_mean", "x_std"]}

    # Compute physics normalization stats (same as training)
    print("Computing physics normalization stats...")
    stats_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=pad_collate, num_workers=4, pin_memory=True)
    phys_sum = torch.zeros(3, device=device, dtype=torch.float64)
    phys_sq_sum = torch.zeros(3, device=device, dtype=torch.float64)
    phys_n = 0.0
    with torch.no_grad():
        for x, y, is_surface, mask in tqdm(stats_loader, desc="Phys stats"):
            y, mask = y.to(device), mask.to(device)
            Um, q = _umag_q(y, mask)
            yp = _phys_norm(y, Um, q)
            yp = yp.clone()
            yp[:, :, 2:3] = torch.asinh(yp[:, :, 2:3] * asinh_scale)
            m = mask.float().unsqueeze(-1)
            phys_sum += (yp * m).sum(dim=(0, 1))
            phys_sq_sum += (yp ** 2 * m).sum(dim=(0, 1))
            phys_n += mask.float().sum().item()
    phys_mean = (phys_sum / phys_n).float()
    phys_std = ((phys_sq_sum / phys_n - phys_mean ** 2).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
    phys_stats = {"y_mean": phys_mean, "y_std": phys_std}

    # Sequential inference: one model at a time to save VRAM
    n_models = len(run_ids)
    # We'll accumulate predictions in physical space per sample
    # Store as list of (N_i, 3) tensors
    n_train = len(train_ds)
    pred_sums = [None] * n_train  # Will be populated on first model

    for model_idx, run_id in enumerate(run_ids):
        print(f"\n[{model_idx+1}/{n_models}] Loading model {run_id}...")
        model, refine_head = load_model_and_refine(run_id, device)

        # Run inference on training set in order
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=pad_collate, num_workers=4, pin_memory=True)

        sample_offset = 0
        with torch.no_grad():
            for x_raw, y, is_surface, mask in tqdm(loader, desc=f"Inference {run_id}", leave=False):
                x_raw = x_raw.to(device)
                y = y.to(device)
                is_surface = is_surface.to(device)
                mask = mask.to(device)
                B, N = mask.shape
                Umag, q_val = _umag_q(y, mask)

                # Preprocessing (exact copy from eval_ensemble.py)
                raw_dsdf = x_raw[:, :, 2:10]
                dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
                dist_feat = torch.log1p(dist_surf * 10.0)
                _raw_aoa = x_raw[:, 0, 14:15]

                x = (x_raw - x_stats["x_mean"]) / x_stats["x_std"]
                curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
                x = torch.cat([x, curv, dist_feat], dim=-1)

                # Target normalization
                y_phys = _phys_norm(y, Umag, q_val)
                y_phys = y_phys.clone()
                y_phys[:, :, 2:3] = torch.asinh(y_phys[:, :, 2:3] * asinh_scale)
                y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                # Residual prediction
                _aoa = _raw_aoa
                _fs_phys = torch.zeros(B, 1, 3, device=device)
                _fs_phys[:, 0, 0] = torch.cos(_aoa.squeeze(-1))
                _fs_phys[:, 0, 1] = torch.sin(_aoa.squeeze(-1))
                _v_freestream = (_fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
                y_norm = y_norm - _v_freestream

                # Per-sample std
                raw_gap = x[:, 0, 21]
                is_tandem = raw_gap.abs() > 0.5
                sample_stds = torch.ones(B, 1, 3, device=device)
                channel_clamps = torch.tensor([0.1, 0.1, 2.0], device=device)
                tandem_clamps = torch.tensor([0.3, 0.3, 2.0], device=device)
                for b in range(B):
                    valid = mask[b]
                    if is_tandem[b]:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                    else:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)

                # Model forward pass
                raw_xy = x[:, :, :2]
                xy_min = raw_xy.amin(dim=1, keepdim=True)
                xy_max = raw_xy.amax(dim=1, keepdim=True)
                xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
                freqs = torch.cat([model.fourier_freqs_fixed.to(device),
                                   model.fourier_freqs_learned.abs()])
                xy_scaled = xy_norm.unsqueeze(-1) * freqs
                fourier_pe = torch.cat([xy_scaled.sin().flatten(-2),
                                        xy_scaled.cos().flatten(-2)], dim=-1)
                x_aug = torch.cat([x, fourier_pe], dim=-1)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model({"x": x_aug})
                    pred = out["preds"].float()
                    hidden = out["hidden"].float()

                # Undo per-sample std
                pred_loss = pred / sample_stds

                # Surface refinement
                if refine_head is not None:
                    surf_idx = is_surface.nonzero(as_tuple=False)
                    if surf_idx.numel() > 0:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            surf_hidden = hidden[surf_idx[:, 0], surf_idx[:, 1]]
                            surf_pred = pred_loss[surf_idx[:, 0], surf_idx[:, 1]]
                            correction = refine_head(surf_hidden, surf_pred).float()
                        pred_loss = pred_loss.clone()
                        pred_loss[surf_idx[:, 0], surf_idx[:, 1]] += correction
                    pred = pred_loss * sample_stds

                # Add freestream back
                pred = pred + _v_freestream

                # Denormalize to physical space
                pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
                pred_phys = pred_phys.clone()
                pred_phys[:, :, 2:3] = torch.sinh(pred_phys[:, :, 2:3]) / asinh_scale
                pred_orig = _phys_denorm(pred_phys, Umag, q_val)

                # Accumulate per-sample predictions
                for b in range(B):
                    idx = sample_offset + b
                    if idx >= n_train:
                        break
                    n_valid = mask[b].sum().item()
                    pred_sample = pred_orig[b, :n_valid].cpu()
                    if pred_sums[idx] is None:
                        pred_sums[idx] = pred_sample
                    else:
                        pred_sums[idx] = pred_sums[idx] + pred_sample
                sample_offset += B

        # Free model memory
        del model, refine_head
        torch.cuda.empty_cache()
        print(f"  Model {run_id} done. Processed {sample_offset} samples.")

    # Average predictions
    print(f"\nAveraging {n_models} model predictions...")
    soft_targets = {}
    for i in range(n_train):
        if pred_sums[i] is not None:
            soft_targets[i] = pred_sums[i] / n_models
        else:
            print(f"  WARNING: sample {i} has no predictions!")

    print(f"Saving {len(soft_targets)} soft targets to {output_path}")
    torch.save(soft_targets, output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ids", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="ensemble_soft_targets.pt")
    parser.add_argument("--asinh_scale", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    precompute_soft_targets(args.run_ids, args.output, args.asinh_scale, args.batch_size)
