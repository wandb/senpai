#!/usr/bin/env python3
"""Prediction-Level Ensemble — Average model outputs from independently trained seeds.

Unlike weight-space interpolation (model soups), this averages model PREDICTIONS,
which sidesteps permutation symmetry issues and is guaranteed to reduce variance.

Usage:
  python ensemble_eval.py --model_dirs dir1 dir2 dir3 ...
"""

import argparse
import ast
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import rearrange
from timm.layers import trunc_normal_

sys.path.insert(0, str(Path(__file__).parent))
from data.prepare_multi import X_DIM, pad_collate, load_data, VAL_SPLIT_NAMES


# ---------------------------------------------------------------------------
# Extract model classes from train.py without executing the training loop
# ---------------------------------------------------------------------------
def _load_model_classes():
    train_path = Path(__file__).parent / "train.py"
    source = train_path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    needed_classes = {"GatedMLP", "GatedMLP2", "MLP", "DomainLayerNorm",
                      "Physics_Attention_Irregular_Mesh", "TransolverBlock",
                      "Transolver", "SurfaceRefinementHead", "SurfaceRefinementContextHead"}

    segments = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segments.append((node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.ClassDef) and node.name in needed_classes:
            segments.append((node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ACTIVATION":
                    segments.append((node.lineno - 1, node.end_lineno))

    code_lines = []
    for start, end in segments:
        code_lines.extend(lines[start:end])
        code_lines.append("\n")

    ns = {
        "__builtins__": __builtins__,
        "torch": torch,
        "nn": nn,
        "F": F,
        "rearrange": rearrange,
        "trunc_normal_": trunc_normal_,
    }
    exec(compile("".join(code_lines), str(train_path), "exec"), ns)
    return ns


_ns = _load_model_classes()
Transolver = _ns["Transolver"]
SurfaceRefinementHead = _ns["SurfaceRefinementHead"]


# ---------------------------------------------------------------------------
# Helper functions (inlined from train.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Prediction-Level Ensemble Evaluation")
    parser.add_argument("--model_dirs", nargs="+", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--manifest", type=str, default="data/split_manifest.json")
    parser.add_argument("--stats_file", type=str, default="data/split_stats.json")
    parser.add_argument("--wandb_name", type=str, default="frieren/prediction-ensemble")
    parser.add_argument("--agent", type=str, default="frieren")
    return parser.parse_args()


def build_model(config, device):
    model = Transolver(**config).to(device)
    model._pressure_separate = False
    return model


def build_refine_head(config, device):
    return SurfaceRefinementHead(
        n_hidden=config["n_hidden"],
        out_dim=3,
        hidden_dim=192,
        n_layers=3,
        p_only=False,
    ).to(device)


def load_checkpoint(model_dir, model, refine_head, device):
    sd = torch.load(Path(model_dir) / "checkpoint.pt", map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)

    refine_path = Path(model_dir) / "refine_head.pt"
    if refine_path.exists():
        rsd = torch.load(refine_path, map_location=device, weights_only=True)
        rsd = {k.removeprefix("_orig_mod."): v for k, v in rsd.items()}
        refine_head.load_state_dict(rsd)


def preprocess_batch(x, y, is_surface, mask, stats, model, device, cfg_flags):
    """Preprocess a batch: feature engineering, normalization, etc.
    Returns preprocessed x, y_norm, y_norm_scaled, sample_stds, Umag, q, is_tandem, _v_freestream, phys_stats.
    """
    raw_dsdf = x[:, :, 2:10]
    dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
    dist_feat = torch.log1p(dist_surf * 10.0)
    _raw_aoa = x[:, 0, 14:15]
    x = (x - stats["x_mean"]) / stats["x_std"]
    curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
    x = torch.cat([x, curv, dist_feat], dim=-1)

    # Fourier PE
    raw_xy = x[:, :, :2]
    xy_min = raw_xy.amin(dim=1, keepdim=True)
    xy_max = raw_xy.amax(dim=1, keepdim=True)
    xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
    freqs = torch.cat([model.fourier_freqs_fixed.to(device), model.fourier_freqs_learned.abs()])
    xy_scaled = xy_norm.unsqueeze(-1) * freqs
    fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)
    x = torch.cat([x, fourier_pe], dim=-1)

    return x, _raw_aoa


@torch.no_grad()
def forward_single_model(model, refine_head, x_preprocessed, is_surface, mask, sample_stds, device):
    """Forward pass through one model + refinement head.
    Returns pred_loss (predictions in per-sample-std-normalized space).
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model({"x": x_preprocessed})
        pred = out["preds"]
        hidden = out["hidden"]
    pred = pred.float()
    hidden = hidden.float()
    pred_loss = pred / sample_stds

    # Surface refinement
    surf_idx = is_surface.nonzero(as_tuple=False)
    if surf_idx.numel() > 0:
        surf_hidden = hidden[surf_idx[:, 0], surf_idx[:, 1]]
        surf_pred = pred_loss[surf_idx[:, 0], surf_idx[:, 1]]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            correction = refine_head(surf_hidden, surf_pred).float()
        pred_loss = pred_loss.clone()
        pred_loss[surf_idx[:, 0], surf_idx[:, 1]] += correction

    return pred_loss


@torch.no_grad()
def evaluate_ensemble(model_dirs, model_config, val_loaders, stats, phys_stats,
                      device, cfg_flags, weights=None):
    """Evaluate an ensemble by sequentially loading each model, accumulating predictions.

    Args:
        model_dirs: list of model directories to ensemble
        weights: optional per-model weights (softmax-normalized). None = uniform.
    """
    n_models = len(model_dirs)
    if weights is None:
        weights = [1.0 / n_models] * n_models

    all_metrics = {}
    split_losses = []

    for split_name, vloader in val_loaders.items():
        # Collect all batches first (so we can iterate multiple times for each model)
        batches = []
        for batch in vloader:
            batches.append(tuple(t.to(device, non_blocking=True) for t in batch))

        # Initialize accumulators for ensemble predictions
        # pred_loss_accum[batch_idx] = accumulated weighted pred_loss
        pred_loss_accum = [torch.zeros_like(b[1]) for b in batches]  # [B, N, 3]

        # For each model, forward all batches and accumulate
        for m_idx, model_dir in enumerate(model_dirs):
            model = build_model(model_config, device)
            refine_head = build_refine_head(model_config, device)
            load_checkpoint(model_dir, model, refine_head, device)
            model.eval()
            refine_head.eval()

            w = weights[m_idx]

            for b_idx, (x_raw, y, is_surface, mask) in enumerate(batches):
                # Preprocess (needs model-specific Fourier freqs)
                x_proc, _raw_aoa = preprocess_batch(
                    x_raw.clone(), y, is_surface, mask, stats, model, device, cfg_flags
                )

                # Compute per-sample stds (same for all models since it depends on y, not model)
                Umag, q_val = _umag_q(y, mask)
                y_phys = _phys_norm(y, Umag, q_val)
                y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                _v_freestream = None
                if cfg_flags.get("residual_prediction", False):
                    _aoa = _raw_aoa
                    _fs_phys = torch.zeros(y_norm.shape[0], 1, 3, device=device)
                    _fs_phys[:, 0, 0] = torch.cos(_aoa.squeeze(-1))
                    _fs_phys[:, 0, 1] = torch.sin(_aoa.squeeze(-1))
                    _v_freestream = (_fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
                    y_norm = y_norm - _v_freestream

                raw_gap = x_proc[:, 0, 21]
                is_tandem = raw_gap.abs() > 0.5
                B = y_norm.shape[0]
                sample_stds = torch.ones(B, 1, 3, device=device)
                channel_clamps = torch.tensor([0.1, 0.1, 2.0], device=device)
                tandem_clamps = torch.tensor([0.3, 0.3, 2.0], device=device)
                for b in range(B):
                    valid = mask[b]
                    if is_tandem[b]:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                    else:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)

                # Forward pass
                pred_loss = forward_single_model(model, refine_head, x_proc, is_surface, mask, sample_stds, device)
                pred_loss_accum[b_idx] += w * pred_loss

            # Free this model's GPU memory
            del model, refine_head
            torch.cuda.empty_cache()

        # Now compute metrics from accumulated ensemble predictions
        val_vol = 0.0
        val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = torch.zeros(3, device=device)
        n_vol = torch.zeros(3, device=device)
        n_vbatches = 0

        for b_idx, (x_raw, y, is_surface, mask) in enumerate(batches):
            # Recompute normalization quantities (same as above)
            Umag, q_val = _umag_q(y, mask)
            y_phys = _phys_norm(y, Umag, q_val)
            y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

            _v_freestream = None
            _raw_aoa = x_raw[:, 0, 14:15]
            if cfg_flags.get("residual_prediction", False):
                _aoa = _raw_aoa
                _fs_phys = torch.zeros(y_norm.shape[0], 1, 3, device=device)
                _fs_phys[:, 0, 0] = torch.cos(_aoa.squeeze(-1))
                _fs_phys[:, 0, 1] = torch.sin(_aoa.squeeze(-1))
                _v_freestream = (_fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
                y_norm = y_norm - _v_freestream

            # Recompute sample_stds (x_proc[:, 0, 21] uses standardized x, need to match)
            # Since sample_stds depends on y_norm, not the model, it's the same
            x_std = (x_raw - stats["x_mean"]) / stats["x_std"]
            raw_gap = x_std[:, 0, 21]
            is_tandem = raw_gap.abs() > 0.5
            B = y_norm.shape[0]
            sample_stds = torch.ones(B, 1, 3, device=device)
            channel_clamps = torch.tensor([0.1, 0.1, 2.0], device=device)
            tandem_clamps = torch.tensor([0.3, 0.3, 2.0], device=device)
            for b in range(B):
                valid = mask[b]
                if is_tandem[b]:
                    sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                else:
                    sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)
            y_norm_scaled = y_norm / sample_stds

            # Ensemble prediction (already accumulated)
            pred_loss = pred_loss_accum[b_idx]
            pred = pred_loss * sample_stds

            abs_err = (pred_loss - y_norm_scaled).abs().nan_to_num(0.0)
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            val_vol += min(
                (abs_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item(),
                1e6)
            val_surf += min(
                (abs_err[:, :, 2:3] * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item(),
                1e6)
            n_vbatches += 1

            # Denormalize for MAE
            if cfg_flags.get("residual_prediction", False) and _v_freestream is not None:
                pred = pred + _v_freestream
            pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
            pred_orig = _phys_denorm(pred_phys, Umag, q_val)
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
        surf_weight = 20.0
        split_loss = val_vol + surf_weight * val_surf
        mae_surf /= n_surf.clamp(min=1)
        mae_vol /= n_vol.clamp(min=1)

        all_metrics[f"{split_name}/loss"] = split_loss
        all_metrics[f"{split_name}/vol_loss"] = val_vol
        all_metrics[f"{split_name}/surf_loss"] = val_surf
        all_metrics[f"{split_name}/mae_surf_Ux"] = mae_surf[0].item()
        all_metrics[f"{split_name}/mae_surf_Uy"] = mae_surf[1].item()
        all_metrics[f"{split_name}/mae_surf_p"] = mae_surf[2].item()
        all_metrics[f"{split_name}/mae_vol_Ux"] = mae_vol[0].item()
        all_metrics[f"{split_name}/mae_vol_Uy"] = mae_vol[1].item()
        all_metrics[f"{split_name}/mae_vol_p"] = mae_vol[2].item()

        if not (torch.tensor(split_loss).isnan() or torch.tensor(split_loss).isinf()):
            split_losses.append(split_loss)

        # Free batch accumulators
        del pred_loss_accum

    all_metrics["val/loss"] = sum(split_losses) / max(len(split_losses), 1)
    return all_metrics


def fmt(label, metrics):
    vl = metrics.get("val/loss", float("nan"))
    p_in = metrics.get("val_in_dist/mae_surf_p", float("nan"))
    p_oodc = metrics.get("val_ood_cond/mae_surf_p", float("nan"))
    p_tan = metrics.get("val_tandem_transfer/mae_surf_p", float("nan"))
    p_re = metrics.get("val_ood_re/mae_surf_p", float("nan"))
    return f"| {label:45s} | {vl:.4f} | {p_in:5.1f} | {p_oodc:5.1f} | {p_tan:5.1f} | {p_re:5.1f} |"


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')

    model_dirs = [Path(d) for d in args.model_dirs]
    print(f"\n{'='*70}")
    print(f"PREDICTION-LEVEL ENSEMBLE — {len(model_dirs)} models")
    print(f"{'='*70}")
    for d in model_dirs:
        print(f"  {d}")

    # Load data
    print("\nLoading data...")
    train_ds, val_splits, stats, _ = load_data(args.manifest, args.stats_file)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    val_loaders = {
        name: DataLoader(subset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, subset in val_splits.items()
    }

    # Compute physics normalization stats
    print("Computing physics normalization stats...")
    _stats_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    _phys_sum = torch.zeros(3, device=device)
    _phys_sq_sum = torch.zeros(3, device=device)
    _phys_n = 0.0
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
    print(f"  Cp stats — mean: {_pmean.tolist()}, std: {_pstd.tolist()}")
    del _stats_loader

    # Load model config
    with open(model_dirs[0] / "config.yaml") as f:
        model_config = yaml.safe_load(f)
    print(f"\nModel config loaded from {model_dirs[0]}")

    cfg_flags = {"high_p_clamp": True, "residual_prediction": True}

    # First, evaluate each individual model to get val/loss for ranking and weighting
    print(f"\n{'='*70}")
    print("Evaluating individual models for ranking...")
    print(f"{'='*70}")

    individual_losses = []
    for i, d in enumerate(model_dirs):
        print(f"\n  Model {i+1}/{len(model_dirs)}: {d.name}")
        m = evaluate_ensemble([d], model_config, val_loaders, stats, phys_stats, device, cfg_flags)
        individual_losses.append(m["val/loss"])
        print(f"    val/loss = {m['val/loss']:.4f}")

    # Rank by val/loss (ascending)
    ranked_indices = sorted(range(len(model_dirs)), key=lambda i: individual_losses[i])
    ranked_dirs = [model_dirs[i] for i in ranked_indices]
    ranked_losses = [individual_losses[i] for i in ranked_indices]

    print(f"\nRanked models:")
    for i, (d, loss) in enumerate(zip(ranked_dirs, ranked_losses)):
        print(f"  {i+1}. {d.name}: val/loss={loss:.4f}")

    # --- Ensemble configurations ---
    configs = [
        ("Best individual (1 model)", ranked_dirs[:1], None),
        ("Best 2 ensemble", ranked_dirs[:2], None),
        ("Best 4 ensemble", ranked_dirs[:4], None),
        ("Best 6 ensemble", ranked_dirs[:6], None),
        ("All 8 ensemble", ranked_dirs[:8], None),
    ]

    # Weighted ensemble: softmax on negative val/loss (lower loss = higher weight)
    neg_losses = torch.tensor([-l for l in ranked_losses[:8]])
    softmax_weights = torch.softmax(neg_losses * 100, dim=0).tolist()  # temperature=0.01 to sharpen
    configs.append(("Weighted ensemble (all 8)", ranked_dirs[:8], softmax_weights))

    # Also try gentler temperature
    softmax_weights_gentle = torch.softmax(neg_losses * 10, dim=0).tolist()
    configs.append(("Weighted ensemble (gentle, all 8)", ranked_dirs[:8], softmax_weights_gentle))

    # --- Run all ensemble evaluations ---
    hdr = f"| {'Config':45s} | val/loss | p_in  | p_oodc | p_tan  | p_re  |"
    sep = f"|{'-'*47}|----------|-------|--------|--------|-------|"

    results = []
    for config_name, dirs, weights in configs:
        print(f"\n{'='*70}")
        print(f"Evaluating: {config_name} ({len(dirs)} models)")
        if weights:
            print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
        print(f"{'='*70}")

        metrics = evaluate_ensemble(dirs, model_config, val_loaders, stats, phys_stats,
                                    device, cfg_flags, weights=weights)
        results.append((config_name, metrics))
        print(fmt(config_name, metrics))

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(hdr)
    print(sep)
    for name, m in results:
        print(fmt(name, m))

    # --- W&B logging ---
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "senpai-v1"),
        group="phase5/prediction-ensemble",
        name=args.wandb_name,
        tags=[args.agent, "ensemble"],
        config={
            "n_models": len(model_dirs),
            "model_dirs": [str(d) for d in model_dirs],
            "individual_losses": individual_losses,
            "softmax_weights_sharp": softmax_weights,
            "softmax_weights_gentle": softmax_weights_gentle,
        },
    )

    for name, m in results:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        for k, v in m.items():
            wandb.summary[f"{safe_name}/{k}"] = v

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
