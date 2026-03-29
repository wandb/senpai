# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Greedy Model Soup — Weight-Space Averaging (Wortsman et al., ICML 2022).

Given N trained checkpoints (different seeds, same architecture), greedily
average their weights starting from the best seed. Add each seed's weights
to the running average only if it improves val/loss.

Usage:
    python model_soup.py --checkpoints <dir1> <dir2> ... --manifest data/split_manifest.json --stats data/split_stats.json
"""

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from train.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from data.prepare_multi import X_DIM, pad_collate, load_data, VAL_SPLIT_NAMES


def load_model_from_checkpoint(ckpt_dir, device):
    """Load a model from a checkpoint directory (config.yaml + checkpoint.pt)."""
    config_path = Path(ckpt_dir) / "config.yaml"
    ckpt_path = Path(ckpt_dir) / "checkpoint.pt"

    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    # Import model class
    from train import Transolver
    model = Transolver(**model_config).to(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model, model_config


def evaluate_model(model, val_loaders, stats, phys_stats, device, cfg_flags):
    """Evaluate a model on all validation splits. Returns val/loss (4-split average)."""
    from train import _umag_q, _phys_norm

    model.eval()
    split_losses = {}

    with torch.no_grad():
        for split_name, vloader in val_loaders.items():
            val_vol = 0.0
            val_surf = 0.0
            n_batches = 0

            for x, y, is_surface, mask in tqdm(vloader, desc=split_name, leave=False):
                x, y = x.to(device), y.to(device)
                is_surface = is_surface.to(device)
                mask = mask.to(device)

                raw_dsdf = x[:, :, 2:10]
                dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
                dist_feat = torch.log1p(dist_surf * 10.0)
                x = (x - stats["x_mean"]) / stats["x_std"]
                curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
                x = torch.cat([x, curv, dist_feat], dim=-1)

                raw_xy = x[:, :, :2]
                xy_min = raw_xy.amin(dim=1, keepdim=True)
                xy_max = raw_xy.amax(dim=1, keepdim=True)
                xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
                freqs = torch.cat([model.fourier_freqs_fixed.to(device), model.fourier_freqs_learned.abs()])
                xy_scaled = xy_norm.unsqueeze(-1) * freqs
                fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)
                x = torch.cat([x, fourier_pe], dim=-1)

                Umag, q = _umag_q(y, mask)
                y_phys = _phys_norm(y, Umag, q)
                y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                # Residual prediction
                if cfg_flags.get("residual_prediction"):
                    _aoa = x[:, 0, 14:15]
                    _fs_phys = torch.zeros(y_norm.shape[0], 1, 3, device=device)
                    _fs_phys[:, 0, 0] = torch.cos(_aoa.squeeze(-1))
                    _fs_phys[:, 0, 1] = torch.sin(_aoa.squeeze(-1))
                    _freestream = (_fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
                    y_norm = y_norm - _freestream

                B = y_norm.shape[0]
                sample_stds = torch.ones(B, 1, 3, device=device)
                channel_clamps = torch.tensor([0.1, 0.1, 2.0], device=device)
                tandem_clamps = torch.tensor([0.3, 0.3, 2.0], device=device)
                raw_gap = x[:, 0, 21]
                is_tandem = raw_gap.abs() > 0.5
                for b in range(B):
                    valid = mask[b]
                    if is_tandem[b]:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                    else:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)
                y_norm_scaled = y_norm / sample_stds

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model({"x": x})["preds"]
                pred = pred.float()
                pred_loss = pred / sample_stds
                abs_err = (pred_loss - y_norm_scaled).abs().nan_to_num(0.0)

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += min(
                    (abs_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item(), 1e6
                )
                val_surf += min(
                    (abs_err[:, :, 2:3] * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item(), 1e6
                )
                n_batches += 1

            val_vol /= max(n_batches, 1)
            val_surf /= max(n_batches, 1)
            split_losses[split_name] = val_vol + 20.0 * val_surf  # default surf_weight

    avg_loss = sum(split_losses.values()) / len(split_losses)
    return avg_loss, split_losses


def average_state_dicts(sd1, sd2, alpha=0.5):
    """Average two state dicts: alpha * sd1 + (1-alpha) * sd2."""
    result = {}
    for k in sd1:
        result[k] = alpha * sd1[k].float() + (1 - alpha) * sd2[k].float()
    return result


def greedy_soup(checkpoints, val_loaders, stats, phys_stats, device, cfg_flags):
    """Greedy model soup: start with best, add others if they improve."""
    print(f"\n{'='*60}")
    print(f"GREEDY MODEL SOUP — {len(checkpoints)} checkpoints")
    print(f"{'='*60}\n")

    # Step 1: Evaluate all individual checkpoints
    individual_results = []
    for i, ckpt_dir in enumerate(checkpoints):
        print(f"\nEvaluating checkpoint {i}: {ckpt_dir}")
        model, model_config = load_model_from_checkpoint(ckpt_dir, device)
        loss, splits = evaluate_model(model, val_loaders, stats, phys_stats, device, cfg_flags)
        individual_results.append((loss, ckpt_dir, model_config))
        print(f"  val/loss = {loss:.4f}  splits = {', '.join(f'{k}={v:.4f}' for k, v in splits.items())}")
        del model

    # Step 2: Sort by val/loss (best first)
    individual_results.sort(key=lambda x: x[0])
    print(f"\nBest individual: {individual_results[0][1]} (val/loss={individual_results[0][0]:.4f})")

    # Step 3: Greedy soup
    best_loss, best_dir, model_config = individual_results[0]
    soup_model, _ = load_model_from_checkpoint(best_dir, device)
    soup_sd = {k: v.float() for k, v in soup_model.state_dict().items()}
    n_in_soup = 1
    soup_dirs = [best_dir]

    print(f"\nStarting greedy soup from {best_dir}")
    for loss, ckpt_dir, _ in individual_results[1:]:
        print(f"\nTrying to add {ckpt_dir} (individual loss={loss:.4f})...")
        candidate, _ = load_model_from_checkpoint(ckpt_dir, device)
        candidate_sd = {k: v.float() for k, v in candidate.state_dict().items()}

        # Average: (n * soup + 1 * candidate) / (n + 1)
        trial_sd = {}
        for k in soup_sd:
            trial_sd[k] = (n_in_soup * soup_sd[k] + candidate_sd[k]) / (n_in_soup + 1)

        soup_model.load_state_dict(trial_sd)
        trial_loss, trial_splits = evaluate_model(
            soup_model, val_loaders, stats, phys_stats, device, cfg_flags
        )
        print(f"  Trial loss = {trial_loss:.4f} (current best = {best_loss:.4f})")

        if trial_loss < best_loss:
            print(f"  ACCEPTED — improves by {best_loss - trial_loss:.4f}")
            soup_sd = trial_sd
            best_loss = trial_loss
            n_in_soup += 1
            soup_dirs.append(ckpt_dir)
        else:
            print(f"  REJECTED — would degrade by {trial_loss - best_loss:.4f}")
            soup_model.load_state_dict(soup_sd)

        del candidate

    print(f"\n{'='*60}")
    print(f"SOUP COMPLETE — {n_in_soup} models in soup")
    print(f"Soup members: {soup_dirs}")
    print(f"{'='*60}")

    # Final evaluation
    soup_model.load_state_dict(soup_sd)
    final_loss, final_splits = evaluate_model(
        soup_model, val_loaders, stats, phys_stats, device, cfg_flags
    )
    print(f"\nFinal soup val/loss = {final_loss:.4f}")
    print(f"Splits: {', '.join(f'{k}={v:.4f}' for k, v in final_splits.items())}")

    # Save soup model
    soup_path = Path("models/model-soup/checkpoint.pt")
    soup_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup_sd, soup_path)
    print(f"Saved soup model to {soup_path}")

    return soup_model, final_loss, final_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Model checkpoint directories")
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--stats", default="data/split_stats.json")
    parser.add_argument("--residual_prediction", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_splits, stats, _ = load_data(args.manifest, args.stats, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    val_loaders = {
        name: DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=4, pin_memory=True)
        for name, subset in val_splits.items()
    }

    # Compute physics normalization stats
    from train import _umag_q, _phys_norm
    train_ds, _, _, _ = load_data(args.manifest, args.stats, debug=False)
    print("Computing physics normalization stats...")
    _phys_sum = torch.zeros(3, device=device)
    _phys_sq_sum = torch.zeros(3, device=device)
    _phys_n = 0.0
    _stats_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=pad_collate, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for _x, _y, _is_surf, _mask in tqdm(_stats_loader, desc="Phys stats", leave=False):
            _y, _mask = _y.to(device), _mask.to(device)
            _Um, _q = _umag_q(_y, _mask)
            _yp = _phys_norm(_y, _Um, _q)
            _m = _mask.float().unsqueeze(-1)
            _phys_sum += (_yp * _m).sum(dim=(0, 1))
            _phys_sq_sum += (_yp ** 2 * _m).sum(dim=(0, 1))
            _phys_n += _mask.float().sum().item()
    _pmean = (_phys_sum / _phys_n).float()
    _pstd = ((_phys_sq_sum / _phys_n - _pmean ** 2).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
    phys_stats = {"y_mean": _pmean, "y_std": _pstd}

    cfg_flags = {"residual_prediction": args.residual_prediction}

    greedy_soup(args.checkpoints, val_loaders, stats, phys_stats, device, cfg_flags)
