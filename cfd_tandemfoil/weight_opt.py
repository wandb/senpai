#!/usr/bin/env python3
"""
weight_opt.py: Post-hoc ensemble weight optimisation.

Evaluates:
  - Equal-weight ensemble (baseline)
  - Scipy-optimised weights (SLSQP, minimise surface-p MAE on held-out in_dist)
  - Best N-of-8 subsets for N = 4, 5, 6, 7

Inference is run one model at a time so peak GPU memory is ~38 GB.

Usage:
  python weight_opt.py \
      --run_ids rboyvjeo h0uog211 kwt8tw52 5j26p5v1 rmump7ke ujt9cu0l 7fw8ksxq 0lsry8km \
      --asinh_scale 0.75 \
      --wandb_name "askeladd/ensemble-weight-opt"
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import wandb
from scipy.optimize import minimize
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.prepare_multi import load_data, pad_collate

# Re-use helpers already defined in eval_ensemble.py
from eval_ensemble import (
    _extract_classes_from_train,
    _umag_q,
    _phys_norm,
    _phys_denorm,
    compute_phys_stats,
    load_model_and_refine,
    print_results,
)

_ns = {}
exec(_extract_classes_from_train(), _ns)


# ---------------------------------------------------------------------------
# Core inference: collect per-model predictions for every split
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    run_ids, val_loaders, phys_stats, x_stats, device,
    asinh_pressure=True, asinh_scale=0.75,
):
    """Run inference for each model one at a time; return collected batches.

    Returns
    -------
    split_batches : dict[split_name -> list of batch_dicts]
        Each batch_dict has:
          'preds'     : Tensor [n_models, B, N, 3]  (physical space, CPU)
          'target'    : Tensor [B, N, 3]             (physical space, CPU)
          'surf_mask' : BoolTensor [B, N]
          'valid_mask': BoolTensor [B, N]
    """
    n_models = len(run_ids)
    # Initialise storage: split -> [batch_idx] -> list of model preds (filled in order)
    split_model_preds = {name: {} for name in val_loaders}   # {split: {bid: [model_pred…]}}
    split_meta       = {name: {} for name in val_loaders}    # {split: {bid: (target, surf, mask)}}

    for model_idx, rid in enumerate(run_ids):
        print(f"\n[{model_idx+1}/{n_models}] Loading model {rid}...")
        model, refine_head = load_model_and_refine(rid, device)

        for split_name, loader in val_loaders.items():
            for bid, (x_raw, y, is_surface, mask) in enumerate(
                    tqdm(loader, desc=f"  {split_name}", leave=False)):
                x_raw       = x_raw.to(device)
                y           = y.to(device)
                is_surface  = is_surface.to(device)
                mask        = mask.to(device)
                B, N        = mask.shape
                Umag, q     = _umag_q(y, mask)

                # --- preprocessing (mirrors eval_ensemble.py exactly) ---
                raw_dsdf  = x_raw[:, :, 2:10]
                dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
                dist_feat = torch.log1p(dist_surf * 10.0)
                _raw_aoa  = x_raw[:, 0, 14:15]

                x = (x_raw - x_stats["x_mean"]) / x_stats["x_std"]
                curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
                x = torch.cat([x, curv, dist_feat], dim=-1)

                y_phys = _phys_norm(y, Umag, q)
                if asinh_pressure:
                    y_phys = y_phys.clone()
                    y_phys[:, :, 2:3] = torch.asinh(y_phys[:, :, 2:3] * asinh_scale)
                y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                _aoa       = _raw_aoa
                _fs_phys   = torch.zeros(B, 1, 3, device=device)
                _fs_phys[:, 0, 0] = torch.cos(_aoa.squeeze(-1))
                _fs_phys[:, 0, 1] = torch.sin(_aoa.squeeze(-1))
                _v_freestream = (_fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
                y_norm    = y_norm - _v_freestream

                raw_gap   = x[:, 0, 21]
                is_tandem = raw_gap.abs() > 0.5
                sample_stds     = torch.ones(B, 1, 3, device=device)
                channel_clamps  = torch.tensor([0.1, 0.1, 2.0], device=device)
                tandem_clamps   = torch.tensor([0.3, 0.3, 2.0], device=device)
                for b in range(B):
                    valid = mask[b]
                    if is_tandem[b]:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                    else:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)

                # --- single-model forward ---
                raw_xy = x[:, :, :2]
                xy_min = raw_xy.amin(dim=1, keepdim=True)
                xy_max = raw_xy.amax(dim=1, keepdim=True)
                xy_norm_f = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
                freqs = torch.cat([model.fourier_freqs_fixed.to(device),
                                   model.fourier_freqs_learned.abs()])
                xy_scaled  = xy_norm_f.unsqueeze(-1) * freqs
                fourier_pe = torch.cat([xy_scaled.sin().flatten(-2),
                                        xy_scaled.cos().flatten(-2)], dim=-1)
                x_aug = torch.cat([x, fourier_pe], dim=-1)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out    = model({"x": x_aug})
                    pred   = out["preds"].float()
                    hidden = out["hidden"].float()

                pred_loss = pred / sample_stds
                if refine_head is not None:
                    surf_idx = is_surface.nonzero(as_tuple=False)
                    if surf_idx.numel() > 0:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            surf_hidden  = hidden[surf_idx[:, 0], surf_idx[:, 1]]
                            surf_pred_in = pred_loss[surf_idx[:, 0], surf_idx[:, 1]]
                            correction   = refine_head(surf_hidden, surf_pred_in).float()
                        pred_loss = pred_loss.clone()
                        pred_loss[surf_idx[:, 0], surf_idx[:, 1]] += correction
                    pred = pred_loss * sample_stds

                pred = pred + _v_freestream
                pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
                if asinh_pressure:
                    pred_phys = pred_phys.clone()
                    pred_phys[:, :, 2:3] = torch.sinh(pred_phys[:, :, 2:3]) / asinh_scale
                pred_orig = _phys_denorm(pred_phys, Umag, q)

                # Store on CPU
                if bid not in split_model_preds[split_name]:
                    split_model_preds[split_name][bid] = []
                    split_meta[split_name][bid] = (
                        y.cpu(),
                        (mask & is_surface).cpu(),
                        mask.cpu(),
                    )
                split_model_preds[split_name][bid].append(pred_orig.cpu())

        del model, refine_head
        torch.cuda.empty_cache()

    # Assemble final structure
    split_batches = {}
    for split_name in val_loaders:
        batches = []
        for bid in sorted(split_model_preds[split_name]):
            preds = torch.stack(split_model_preds[split_name][bid])  # [n_models, B, N, 3]
            target, surf_mask, valid_mask = split_meta[split_name][bid]
            batches.append({
                "preds":      preds,
                "target":     target,
                "surf_mask":  surf_mask,
                "valid_mask": valid_mask,
            })
        split_batches[split_name] = batches
    return split_batches


# ---------------------------------------------------------------------------
# MAE computation helpers
# ---------------------------------------------------------------------------

def compute_mae_surf_p(weights, batches):
    """Compute weighted-ensemble surface-pressure MAE over a list of batches."""
    w = torch.tensor(weights, dtype=torch.float32)  # [n_models]
    total_err = 0.0
    total_n   = 0
    for batch in batches:
        # preds: [n_models, B, N, 3]
        pred_avg = (w.view(-1, 1, 1, 1) * batch["preds"]).sum(0)  # [B, N, 3]
        err       = (pred_avg[:, :, 2] - batch["target"][:, :, 2]).abs()
        surf_valid = batch["surf_mask"] & batch["valid_mask"]
        total_err += err[surf_valid].sum().item()
        total_n   += surf_valid.sum().item()
    return total_err / max(total_n, 1)


def compute_all_split_metrics(weights, split_batches):
    """Return dict of {split_name: {mae_surf_p, mae_surf_Ux, mae_surf_Uy, mae_vol_p}}."""
    w = torch.tensor(weights, dtype=torch.float32)
    results = {}
    for split_name, batches in split_batches.items():
        mae_surf = torch.zeros(3)
        mae_vol  = torch.zeros(3)
        n_surf   = torch.zeros(3)
        n_vol    = torch.zeros(3)
        for batch in batches:
            pred_avg = (w.view(-1, 1, 1, 1) * batch["preds"]).sum(0)
            target   = batch["target"]
            err      = (pred_avg - target).abs()
            finite   = err.isfinite()
            err      = err.where(finite, torch.zeros_like(err))
            surf_mask = batch["surf_mask"] & batch["valid_mask"]
            vol_mask  = batch["valid_mask"] & ~batch["surf_mask"]
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            mae_vol  += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf   += (surf_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
            n_vol    += (vol_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
        mae_surf /= n_surf.clamp(min=1)
        mae_vol  /= n_vol.clamp(min=1)
        results[split_name] = {
            "mae_surf_Ux": mae_surf[0].item(),
            "mae_surf_Uy": mae_surf[1].item(),
            "mae_surf_p":  mae_surf[2].item(),
            "mae_vol_p":   mae_vol[2].item(),
        }
    return results


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def optimise_weights(opt_batches, n_models):
    """Find weights that minimise surface-p MAE on opt_batches using SLSQP."""
    def objective(w):
        return compute_mae_surf_p(w, opt_batches)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds      = [(0.0, 1.0)] * n_models
    x0          = np.ones(n_models) / n_models

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-8, "maxiter": 500, "disp": True})
    return result.x, result.fun


def best_n_of_k(n, batches_per_split, opt_batches, n_models):
    """Find the N-model subset (equal weights within subset) minimising surf-p MAE on opt_batches."""
    best_mae   = float("inf")
    best_combo = None
    for combo in combinations(range(n_models), n):
        w = np.zeros(n_models)
        w[list(combo)] = 1.0 / n
        mae = compute_mae_surf_p(w, opt_batches)
        if mae < best_mae:
            best_mae   = mae
            best_combo = combo
    return best_combo, best_mae


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ids",     nargs="+", required=True)
    parser.add_argument("--asinh_scale", type=float, default=0.75)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--opt_frac",    type=float, default=0.8,
                        help="Fraction of in_dist batches used for weight optimisation")
    parser.add_argument("--wandb_name",  type=str,   default="askeladd/ensemble-weight-opt")
    parser.add_argument("--wandb_project", type=str, default="senpai-v1")
    parser.add_argument("--wandb_entity",  type=str, default="wandb-applied-ai-team")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models  = len(args.run_ids)
    print(f"Device: {device}  |  Models: {n_models}")

    # --- W&B init ---
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config={"run_ids": args.run_ids, "asinh_scale": args.asinh_scale,
                "n_models": n_models, "opt_frac": args.opt_frac},
    )

    # --- Data ---
    print("Loading data...")
    train_ds, val_splits, stats, _ = load_data("data/split_manifest.json",
                                                "data/split_stats.json")
    x_stats = {k: v.to(device) for k, v in stats.items()}
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=4, pin_memory=True)
        for name, ds in val_splits.items()
    }

    print("Computing phys stats...")
    phys_stats = compute_phys_stats(train_ds, device,
                                    asinh_pressure=True, asinh_scale=args.asinh_scale)
    phys_stats = {k: v.to(device) for k, v in phys_stats.items()}

    # --- Collect per-model predictions ---
    print("\n=== Collecting per-model predictions ===")
    split_batches = collect_predictions(
        args.run_ids, val_loaders, phys_stats, x_stats, device,
        asinh_pressure=True, asinh_scale=args.asinh_scale,
    )

    # --- Equal-weight baseline ---
    print("\n=== Equal-weight ensemble (baseline) ===")
    eq_weights = np.ones(n_models) / n_models
    eq_results = compute_all_split_metrics(eq_weights, split_batches)
    print_results("Equal-weight (1/8 each)", eq_results)

    # --- Split in_dist for optimisation ---
    in_dist_batches = split_batches["val_in_dist"]
    n_opt           = max(1, int(len(in_dist_batches) * args.opt_frac))
    opt_batches     = in_dist_batches[:n_opt]
    heldout_batches = in_dist_batches[n_opt:]
    print(f"\nIn_dist batches: total={len(in_dist_batches)}, "
          f"opt={n_opt}, heldout={len(heldout_batches)}")

    # --- Optimise weights ---
    print("\n=== Optimising ensemble weights ===")
    opt_weights, opt_mae = optimise_weights(opt_batches, n_models)
    print(f"Optimised weights: {np.round(opt_weights, 4).tolist()}")
    print(f"Opt in_dist surface-p MAE (on optimisation set): {opt_mae:.4f}")

    opt_results = compute_all_split_metrics(opt_weights, split_batches)
    print_results("Optimised weights", opt_results)

    # --- N-of-8 best subsets ---
    all_in_dist_batches = in_dist_batches  # optimise over full in_dist for subset search
    subset_results = {}
    print("\n=== N-of-8 best subsets (optimise on full in_dist) ===")
    for n in [4, 5, 6, 7]:
        combo, mae = best_n_of_k(n, split_batches, all_in_dist_batches, n_models)
        w = np.zeros(n_models)
        w[list(combo)] = 1.0 / n
        metrics = compute_all_split_metrics(w, split_batches)
        subset_results[n] = {"combo": combo, "mae": mae, "metrics": metrics}
        run_ids_str = [args.run_ids[i] for i in combo]
        print(f"\nBest {n}-of-{n_models}: indices={list(combo)}, run_ids={run_ids_str}")
        print_results(f"Best {n}-of-{n_models}", metrics)

    # --- Summary table ---
    splits_order = ["val_in_dist", "val_ood_cond", "val_tandem_transfer", "val_ood_re"]
    key_map      = {"val_in_dist": "p_in", "val_ood_cond": "p_oodc",
                    "val_tandem_transfer": "p_tan", "val_ood_re": "p_re"}

    print("\n\n" + "="*70)
    print("SUMMARY  (surface pressure MAE)")
    print("="*70)
    header = f"{'Config':30s}" + "".join(f"  {k:>8s}" for k in key_map.values())
    print(header)
    print("-"*70)

    def row(label, res):
        vals = "".join(f"  {res.get(s, {}).get('mae_surf_p', float('nan')):8.2f}"
                       for s in splits_order)
        print(f"{label:30s}{vals}")

    row(f"Equal-weight (8/{n_models})", eq_results)
    row("Optimised weights",           opt_results)
    for n, sr in subset_results.items():
        row(f"Best {n}-of-{n_models}",  sr["metrics"])

    # --- W&B logging ---
    def flat(prefix, res):
        d = {}
        for s, m in res.items():
            k = key_map.get(s, s)
            d[f"{prefix}/{k}"] = m.get("mae_surf_p", float("nan"))
        return d

    log_dict = {}
    log_dict.update(flat("equal_weight", eq_results))
    log_dict.update(flat("opt_weights",  opt_results))
    for n, sr in subset_results.items():
        log_dict.update(flat(f"best_{n}of{n_models}", sr["metrics"]))

    for i, w in enumerate(opt_weights):
        log_dict[f"opt_weight/model_{i}_{args.run_ids[i]}"] = float(w)

    for n, sr in subset_results.items():
        log_dict[f"best_subset/{n}_of_{n_models}_indices"] = str(list(sr["combo"]))
        log_dict[f"best_subset/{n}_of_{n_models}_run_ids"] = str(
            [args.run_ids[i] for i in sr["combo"]])

    wandb.log(log_dict)
    print(f"\nW&B run: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
