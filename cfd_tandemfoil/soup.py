#!/usr/bin/env python3
"""Greedy Checkpoint Soup — Weight Interpolation of independently trained models.

References:
  - Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models
    improves accuracy without increasing inference cost", ICML 2022.
  - Rame et al., "Diverse Weight Averaging for Out-of-Distribution Generalization",
    NeurIPS 2022 (DiWA).

Usage:
  python soup.py --model_dirs dir1 dir2 dir3 ...
  python soup.py   # auto-discover from models/ directory
"""

import argparse
import ast
import importlib
import os
import sys
import textwrap
import time
import types
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import rearrange
from timm.layers import trunc_normal_

sys.path.insert(0, str(Path(__file__).parent))
from data.prepare_multi import X_DIM, pad_collate, load_data, VAL_SPLIT_NAMES


# ---------------------------------------------------------------------------
# Extract model classes from train.py without executing the training loop.
# We parse the AST, collect class/function definitions up to the first
# non-class/function/import/assignment at module level, then exec them.
# ---------------------------------------------------------------------------
def _load_model_classes():
    """Load Transolver, SurfaceRefinementHead, and helpers from train.py."""
    train_path = Path(__file__).parent / "train.py"
    source = train_path.read_text()
    tree = ast.parse(source)

    # Collect all top-level class/function defs and imports/assignments
    # that appear BEFORE the training loop starts
    lines = source.splitlines(keepends=True)
    keep_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef,
                             ast.Assign, ast.AnnAssign)):
            keep_nodes.append(node)
        else:
            # Stop at first executable statement that isn't a class/func/import
            # (i.e., when the training loop begins)
            # But actually we should keep going to catch all classes — they may
            # appear after some assignments. Let's just collect ALL class/func defs
            # plus imports.
            pass

    # Actually simpler: collect everything we need by line range
    # We need: imports, ACTIVATION dict, all class defs, and the helper functions
    # Everything from line 1 to end of class Transolver + SurfaceRefinementHead
    # Let's find the end of the last class we need

    # Find line ranges for classes and functions we need
    needed_classes = {"GatedMLP", "GatedMLP2", "MLP", "DomainLayerNorm",
                      "Physics_Attention_Irregular_Mesh", "TransolverBlock",
                      "Transolver", "SurfaceRefinementHead", "SurfaceRefinementContextHead"}
    needed_funcs = set()  # _umag_q etc are redefined inline below

    segments = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segments.append((node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.ClassDef) and node.name in needed_classes:
            segments.append((node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.Assign):
            # Capture ACTIVATION dict
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ACTIVATION":
                    segments.append((node.lineno - 1, node.end_lineno))

    # Build the code to exec
    code_lines = []
    for start, end in segments:
        code_lines.extend(lines[start:end])
        code_lines.append("\n")

    code = "".join(code_lines)

    # Create a module-like namespace
    ns = {
        "__builtins__": __builtins__,
        "torch": torch,
        "nn": nn,
        "F": F,
        "rearrange": rearrange,
        "trunc_normal_": trunc_normal_,
    }
    exec(compile(code, str(train_path), "exec"), ns)

    return ns


_ns = _load_model_classes()
Transolver = _ns["Transolver"]
SurfaceRefinementHead = _ns["SurfaceRefinementHead"]


# ---------------------------------------------------------------------------
# Helper functions (inlined to avoid importing from train.py)
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
# Core soup functions
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Greedy Checkpoint Soup")
    parser.add_argument("--model_dirs", nargs="+", type=str, default=None,
                        help="Paths to model directories (each with checkpoint.pt, refine_head.pt, config.yaml)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--manifest", type=str, default="data/split_manifest.json")
    parser.add_argument("--stats_file", type=str, default="data/split_stats.json")
    parser.add_argument("--wandb_name", type=str, default="frieren/checkpoint-soup-eval")
    parser.add_argument("--agent", type=str, default="frieren")
    return parser.parse_args()


def discover_model_dirs() -> list[Path]:
    """Find all model directories with checkpoints."""
    models_root = Path("models")
    if not models_root.exists():
        print("No models/ directory found")
        return []
    dirs = []
    for d in sorted(models_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "checkpoint.pt").exists():
            continue
        if not (d / "config.yaml").exists():
            continue
        dirs.append(d)
    print(f"Found {len(dirs)} model directories with checkpoints")
    return dirs


def build_model(config: dict, device: torch.device):
    model = Transolver(**config).to(device)
    model._pressure_separate = False
    return model


def build_refine_head(config: dict, device: torch.device):
    return SurfaceRefinementHead(
        n_hidden=config["n_hidden"],
        out_dim=3,
        hidden_dim=192,
        n_layers=3,
        p_only=False,
    ).to(device)


def load_checkpoint(model_dir: Path, model, refine_head, device: torch.device):
    sd = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)

    refine_path = model_dir / "refine_head.pt"
    if refine_path.exists():
        rsd = torch.load(refine_path, map_location=device, weights_only=True)
        rsd = {k.removeprefix("_orig_mod."): v for k, v in rsd.items()}
        refine_head.load_state_dict(rsd)


def interpolate_state_dicts(sd1: dict, sd2: dict, alpha: float) -> dict:
    return {k: alpha * sd1[k].float() + (1 - alpha) * sd2[k].float() for k in sd1}


@torch.no_grad()
def evaluate(model, refine_head, val_loaders, stats, phys_stats, device, cfg_flags):
    model.eval()
    refine_head.eval()

    all_metrics = {}
    split_losses = []

    for split_name, vloader in val_loaders.items():
        val_vol = 0.0
        val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = torch.zeros(3, device=device)
        n_vol = torch.zeros(3, device=device)
        n_vbatches = 0

        for x, y, is_surface, mask in tqdm(vloader, desc=f"  [{split_name}]", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

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

            raw_gap = x[:, 0, 21]
            is_tandem = raw_gap.abs() > 0.5
            B = y_norm.shape[0]
            sample_stds = torch.ones(B, 1, 3, device=device)
            if cfg_flags.get("high_p_clamp", False):
                channel_clamps = torch.tensor([0.1, 0.1, 2.0], device=device)
                tandem_clamps = torch.tensor([0.3, 0.3, 2.0], device=device)
            else:
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
                _eval_out = model({"x": x})
                pred = _eval_out["preds"]
                _eval_hidden = _eval_out["hidden"]
            pred = pred.float()
            _eval_hidden = _eval_hidden.float()
            pred_loss = pred / sample_stds

            # Surface refinement
            surf_idx = is_surface.nonzero(as_tuple=False)
            if surf_idx.numel() > 0:
                surf_hidden = _eval_hidden[surf_idx[:, 0], surf_idx[:, 1]]
                surf_pred = pred_loss[surf_idx[:, 0], surf_idx[:, 1]]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    correction = refine_head(surf_hidden, surf_pred).float()
                pred_loss = pred_loss.clone()
                pred_loss[surf_idx[:, 0], surf_idx[:, 1]] += correction
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

    # Discover model directories
    if args.model_dirs:
        model_dirs = [Path(d) for d in args.model_dirs]
    else:
        model_dirs = discover_model_dirs()

    if len(model_dirs) < 2:
        print(f"Need at least 2 model directories, found {len(model_dirs)}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"CHECKPOINT SOUP — {len(model_dirs)} models")
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

    # Load model config from first checkpoint
    with open(model_dirs[0] / "config.yaml") as f:
        model_config = yaml.safe_load(f)
    print(f"\nModel config: {model_config}")

    cfg_flags = {"high_p_clamp": True, "residual_prediction": True}

    # --- Evaluate each individual model ---
    print(f"\n{'='*70}")
    print("PHASE 1: Individual model evaluation")
    print(f"{'='*70}")

    individual_results = []
    individual_main_sds = []
    individual_refine_sds = []

    for i, model_dir in enumerate(model_dirs):
        print(f"\nEvaluating model {i+1}/{len(model_dirs)}: {model_dir.name}")
        model = build_model(model_config, device)
        refine_head = build_refine_head(model_config, device)
        load_checkpoint(model_dir, model, refine_head, device)

        main_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        refine_sd = {k: v.cpu().clone() for k, v in refine_head.state_dict().items()}
        individual_main_sds.append(main_sd)
        individual_refine_sds.append(refine_sd)

        metrics = evaluate(model, refine_head, val_loaders, stats, phys_stats, device, cfg_flags)
        individual_results.append((model_dir.name, metrics))

        del model, refine_head
        torch.cuda.empty_cache()

    # Sort by val/loss ascending
    ranked = sorted(range(len(individual_results)), key=lambda i: individual_results[i][1]["val/loss"])

    hdr = f"| {'Model':45s} | val/loss | p_in  | p_oodc | p_tan  | p_re  |"
    sep = f"|{'-'*47}|----------|-------|--------|--------|-------|"
    print(f"\n{hdr}\n{sep}")
    for idx in ranked:
        name, m = individual_results[idx]
        print(fmt(name, m))

    best_idx = ranked[0]
    best_name, best_metrics = individual_results[best_idx]

    # --- Greedy Soup ---
    greedy_results = {}
    for alpha in [0.5, 0.7]:
        print(f"\n{'='*70}")
        print(f"GREEDY SOUP (alpha={alpha})")
        print(f"{'='*70}")

        soup_main = {k: v.clone() for k, v in individual_main_sds[best_idx].items()}
        soup_refine = {k: v.clone() for k, v in individual_refine_sds[best_idx].items()}

        model = build_model(model_config, device)
        refine_head = build_refine_head(model_config, device)
        model.load_state_dict({k: v.to(device) for k, v in soup_main.items()})
        refine_head.load_state_dict({k: v.to(device) for k, v in soup_refine.items()})
        soup_metrics = evaluate(model, refine_head, val_loaders, stats, phys_stats, device, cfg_flags)
        soup_loss = soup_metrics["val/loss"]
        members = [best_name]

        print(f"Starting with: {best_name} (val/loss={soup_loss:.4f})")

        for idx in ranked[1:]:
            cand_name = individual_results[idx][0]
            print(f"\nTrying: {cand_name}...")

            trial_main = interpolate_state_dicts(soup_main, individual_main_sds[idx], alpha)
            trial_refine = interpolate_state_dicts(soup_refine, individual_refine_sds[idx], alpha)

            model.load_state_dict({k: v.to(device) for k, v in trial_main.items()})
            refine_head.load_state_dict({k: v.to(device) for k, v in trial_refine.items()})
            trial_metrics = evaluate(model, refine_head, val_loaders, stats, phys_stats, device, cfg_flags)
            trial_loss = trial_metrics["val/loss"]

            if trial_loss < soup_loss:
                print(f"  ACCEPTED: {soup_loss:.4f} -> {trial_loss:.4f}")
                soup_main = trial_main
                soup_refine = trial_refine
                soup_loss = trial_loss
                soup_metrics = trial_metrics
                members.append(cand_name)
            else:
                print(f"  REJECTED: would be {trial_loss:.4f} (best {soup_loss:.4f})")

        greedy_results[alpha] = (members, soup_metrics)
        print(f"\nGreedy soup alpha={alpha}: {len(members)} models included")
        print(fmt(f"Greedy soup (a={alpha}, {len(members)} models)", soup_metrics))

        del model, refine_head
        torch.cuda.empty_cache()

    # --- Uniform Average ---
    print(f"\n{'='*70}")
    print("UNIFORM AVERAGE (all models)")
    print(f"{'='*70}")

    n = len(individual_main_sds)
    uniform_main = {k: torch.stack([sd[k].float() for sd in individual_main_sds]).mean(0)
                    for k in individual_main_sds[0]}
    uniform_refine = {k: torch.stack([sd[k].float() for sd in individual_refine_sds]).mean(0)
                      for k in individual_refine_sds[0]}

    model = build_model(model_config, device)
    refine_head = build_refine_head(model_config, device)
    model.load_state_dict({k: v.to(device) for k, v in uniform_main.items()})
    refine_head.load_state_dict({k: v.to(device) for k, v in uniform_refine.items()})
    uniform_metrics = evaluate(model, refine_head, val_loaders, stats, phys_stats, device, cfg_flags)
    print(fmt(f"Uniform average (all {n})", uniform_metrics))

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(hdr)
    print(sep)
    print(fmt(f"Best individual ({best_name})", best_metrics))
    for alpha, (members, m) in greedy_results.items():
        print(fmt(f"Greedy soup (a={alpha}, {len(members)} models)", m))
    print(fmt(f"Uniform average (all {n})", uniform_metrics))

    # --- W&B logging ---
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "senpai-v1"),
        group="phase5/checkpoint-soup-eval",
        name=args.wandb_name,
        tags=[args.agent, "soup"],
        config={
            "n_models": len(model_dirs),
            "model_dirs": [str(d) for d in model_dirs],
            "alpha_values": [0.5, 0.7],
        },
    )
    for i, (name, m) in enumerate(individual_results):
        for k, v in m.items():
            wandb.log({f"individual/{name}/{k}": v, "model_idx": i})
    for k, v in uniform_metrics.items():
        wandb.summary[f"uniform_avg/{k}"] = v
    for k, v in best_metrics.items():
        wandb.summary[f"best_individual/{k}"] = v
    for alpha, (members, m) in greedy_results.items():
        for k, v in m.items():
            wandb.summary[f"greedy_soup_a{alpha}/{k}"] = v
        wandb.summary[f"greedy_soup_a{alpha}/n_members"] = len(members)
        wandb.summary[f"greedy_soup_a{alpha}/members"] = members
    wandb.finish()

    del model, refine_head
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
