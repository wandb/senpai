#!/usr/bin/env python3
"""Post-hoc ensemble evaluation: average predictions from multiple seed checkpoints.

Usage:
    python eval_ensemble.py --run_ids id1 id2 id3 ... --asinh_scale 0.75
"""
import argparse
import ast
import sys
import textwrap
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# --- Extract model classes from train.py without executing training code ---
def _extract_classes_from_train():
    """Parse train.py AST and extract model classes + dependencies without running training."""
    src = Path(__file__).parent / "train.py"
    text = src.read_text()
    tree = ast.parse(text)

    model_classes = {"GatedMLP", "GatedMLP2", "MLP", "DomainLayerNorm",
                     "Physics_Attention_Irregular_Mesh", "TransolverBlock",
                     "SurfaceRefinementHead", "SurfaceRefinementContextHead", "Transolver"}

    code_blocks = []
    past_config = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            past_config = True
            continue
        if past_config and not isinstance(node, ast.ClassDef):
            continue  # skip runtime code after Config

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(text, node)
            if seg:
                code_blocks.append(seg)
        elif isinstance(node, ast.Assign) and not past_config:
            seg = ast.get_source_segment(text, node)
            if seg:
                code_blocks.append(seg)
        elif isinstance(node, ast.ClassDef) and node.name in model_classes:
            seg = ast.get_source_segment(text, node)
            if seg:
                code_blocks.append(seg)

    return "\n\n".join(code_blocks)


# Build namespace with model classes
_ns = {}
_code = _extract_classes_from_train()
exec(_code, _ns)
Transolver = _ns["Transolver"]
SurfaceRefinementHead = _ns["SurfaceRefinementHead"]

# Now import data loading
sys.path.insert(0, str(Path(__file__).parent))
from data.prepare_multi import load_data, pad_collate


def _phys_denorm(y_p, Umag, q):
    y = y_p.clone()
    y[:, :, 0:1] = y_p[:, :, 0:1].clamp(-10, 10) * Umag
    y[:, :, 1:2] = y_p[:, :, 1:2].clamp(-10, 10) * Umag
    y[:, :, 2:3] = y_p[:, :, 2:3].clamp(-20, 20) * q
    return y


def _umag_q(y, mask):
    n_nodes = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    Ux_mean = (y[:, :, 0] * mask.float()).sum(dim=1, keepdim=True) / n_nodes
    Uy_mean = (y[:, :, 1] * mask.float()).sum(dim=1, keepdim=True) / n_nodes
    Umag = (Ux_mean**2 + Uy_mean**2).sqrt().clamp(min=1.0).unsqueeze(-1)
    q = 0.5 * Umag**2
    return Umag, q


def load_model_and_refine(run_id: str, device: torch.device):
    """Load a Transolver model + optional refine head from a checkpoint dir."""
    ckpt_dir = Path(f"models/model-{run_id}")
    cfg_dict = yaml.safe_load((ckpt_dir / "config.yaml").read_text())

    model = Transolver(**cfg_dict).to(device)
    sd = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    refine_head = None
    refine_path = ckpt_dir / "refine_head.pt"
    if refine_path.exists():
        rsd = torch.load(refine_path, map_location=device, weights_only=True)
        rsd = {k.removeprefix("_orig_mod."): v for k, v in rsd.items()}
        n_hidden = cfg_dict["n_hidden"]
        hidden_dim = rsd["mlp.0.weight"].shape[0]
        # Each hidden layer has Linear + LayerNorm + GELU (3 submodules), then final Linear
        # Count Linear layers (have 2D weight tensors)
        linear_keys = sorted([k for k in rsd if k.endswith(".weight") and rsd[k].dim() == 2])
        n_layers = len(linear_keys) - 1  # subtract final projection
        # Detect p_only from output layer shape
        out_shape = rsd[linear_keys[-1]].shape[0]
        p_only = (out_shape == 1)
        refine_head = SurfaceRefinementHead(
            n_hidden=n_hidden, out_dim=3,
            hidden_dim=hidden_dim, n_layers=n_layers, p_only=p_only,
        ).to(device)
        refine_head.load_state_dict(rsd)
        refine_head.eval()

    return model, refine_head


def compute_phys_stats(train_ds, device, asinh_pressure=False, asinh_scale=1.0):
    """Recompute physics normalization stats (must match training exactly)."""
    loader = DataLoader(train_ds, batch_size=4, shuffle=False,
                        collate_fn=pad_collate, num_workers=4, pin_memory=True)
    phys_sum = torch.zeros(3, device=device, dtype=torch.float64)
    phys_sq_sum = torch.zeros(3, device=device, dtype=torch.float64)
    phys_n = 0.0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            y, mask = y.to(device), mask.to(device)
            Umag, q = _umag_q(y, mask)
            yp = y.clone()
            yp[:, :, 0:1] = y[:, :, 0:1] / Umag
            yp[:, :, 1:2] = y[:, :, 1:2] / Umag
            yp[:, :, 2:3] = y[:, :, 2:3] / q
            if asinh_pressure:
                yp = yp.clone()
                yp[:, :, 2:3] = torch.asinh(yp[:, :, 2:3] * asinh_scale)
            m = mask.float().unsqueeze(-1)
            phys_sum += (yp * m).sum(dim=(0, 1))
            phys_sq_sum += (yp ** 2 * m).sum(dim=(0, 1))
            phys_n += mask.float().sum().item()

    mean = (phys_sum / phys_n).float()
    std = ((phys_sq_sum / phys_n - mean ** 2).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
    return {"y_mean": mean, "y_std": std}


@torch.no_grad()
def evaluate_ensemble(models_and_heads, val_loaders, phys_stats, stats, device,
                      asinh_pressure=False, asinh_scale=1.0, residual_prediction=True,
                      high_p_clamp=False):
    """Run ensemble evaluation: average denormalized predictions, compute MAE."""
    results = {}
    n_models = len(models_and_heads)

    for split_name, loader in val_loaders.items():
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = torch.zeros(3, device=device)
        n_vol = torch.zeros(3, device=device)

        for x, y, is_surface, mask in loader:
            x, y = x.to(device), y.to(device)
            is_surface, mask = is_surface.to(device), mask.to(device)
            B, N = mask.shape
            Umag, q = _umag_q(y, mask)

            # Physics-normalize targets for per-sample std computation
            y_phys = y.clone()
            y_phys[:, :, 0:1] = y[:, :, 0:1] / Umag
            y_phys[:, :, 1:2] = y[:, :, 1:2] / Umag
            y_phys[:, :, 2:3] = y[:, :, 2:3] / q
            if asinh_pressure:
                y_phys_t = y_phys.clone()
                y_phys_t[:, :, 2:3] = torch.asinh(y_phys_t[:, :, 2:3] * asinh_scale)
            else:
                y_phys_t = y_phys
            y_norm = (y_phys_t - phys_stats["y_mean"]) / phys_stats["y_std"]

            # Per-sample std
            channel_clamps = torch.tensor([0.05, 0.05, 0.02], device=device) if high_p_clamp else torch.tensor([0.05, 0.05, 0.05], device=device)
            tandem_clamps = torch.tensor([0.03, 0.03, 0.01], device=device)
            sample_stds = torch.ones(B, 1, 3, device=device)
            is_tandem = (x[:, 0, 21].abs() > 0.01)  # raw x before normalization
            for b in range(B):
                valid = mask[b]
                if valid.any():
                    if is_tandem[b]:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=tandem_clamps)
                    else:
                        sample_stds[b, 0] = y_norm[b, valid].std(dim=0).clamp(min=channel_clamps)

            # Freestream for residual prediction (use raw AoA before normalization)
            _v_freestream = None
            if residual_prediction:
                _raw_aoa = x[:, 0, 14:15]  # AoA0_rad [B, 1] — raw, before normalization
                fs_phys = torch.zeros(B, 1, 3, device=device)
                fs_phys[:, 0, 0] = _raw_aoa.cos().squeeze(-1)
                fs_phys[:, 0, 1] = _raw_aoa.sin().squeeze(-1)
                _v_freestream = (fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

            # Accumulate denormalized predictions
            pred_orig_sum = torch.zeros(B, N, 3, device=device)

            # Preprocess x: dist_feat, normalize, curv, fourier PE (must match training exactly)
            raw_dsdf = x[:, :, 2:10]
            dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
            dist_feat = torch.log1p(dist_surf * 10.0)
            x_normed = (x - stats["x_mean"].to(device)) / stats["x_std"].to(device)
            curv = x_normed[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
            x_prep = torch.cat([x_normed, curv, dist_feat], dim=-1)  # [B, N, 26]

            for model, refine_head in models_and_heads:
                # Add per-model Fourier PE (learned freqs differ per seed)
                raw_xy = x_prep[:, :, :2]
                xy_min = raw_xy.amin(dim=1, keepdim=True)
                xy_max = raw_xy.amax(dim=1, keepdim=True)
                xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
                freqs = torch.cat([model.fourier_freqs_fixed.to(device), model.fourier_freqs_learned.abs()])
                xy_scaled = xy_norm.unsqueeze(-1) * freqs
                fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)
                x_aug = torch.cat([x_prep, fourier_pe], dim=-1)  # [B, N, 58]

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

                # Freestream
                if residual_prediction and _v_freestream is not None:
                    pred = pred + _v_freestream

                # Denormalize
                pred_phys = pred * phys_stats["y_std"] + phys_stats["y_mean"]
                if asinh_pressure:
                    pred_phys = pred_phys.clone()
                    pred_phys[:, :, 2:3] = torch.sinh(pred_phys[:, :, 2:3]) / asinh_scale
                pred_orig = _phys_denorm(pred_phys, Umag, q)

                pred_orig_sum += pred_orig

            # Average
            pred_avg = pred_orig_sum / n_models

            # MAE
            y_clamped = y.clamp(-1e6, 1e6)
            err = (pred_avg - y_clamped).abs()
            finite = err.isfinite()
            err = err.where(finite, torch.zeros_like(err))

            surf_mask = mask & is_surface
            vol_mask = mask & ~is_surface

            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += (surf_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
            n_vol += (vol_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()

        mae_surf /= n_surf.clamp(min=1)
        mae_vol /= n_vol.clamp(min=1)

        results[split_name] = {
            "mae_surf_Ux": mae_surf[0].item(),
            "mae_surf_Uy": mae_surf[1].item(),
            "mae_surf_p": mae_surf[2].item(),
            "mae_vol_p": mae_vol[2].item(),
        }

    return results


def print_results(label, results):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    p_in = results.get("val_in_dist", {}).get("mae_surf_p", float("nan"))
    p_tan = results.get("val_tandem_transfer", {}).get("mae_surf_p", float("nan"))
    p_oodc = results.get("val_ood_cond", {}).get("mae_surf_p", float("nan"))
    p_re = results.get("val_ood_re", {}).get("mae_surf_p", float("nan"))
    print(f"  p_in={p_in:.1f}  p_oodc={p_oodc:.1f}  p_tan={p_tan:.1f}  p_re={p_re:.1f}")
    for split, metrics in results.items():
        print(f"  {split:30s}  mae_surf_p={metrics['mae_surf_p']:.1f}  mae_vol_p={metrics['mae_vol_p']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation")
    parser.add_argument("--run_ids", nargs="+", required=True, help="W&B run IDs")
    parser.add_argument("--asinh_scale", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ensemble: {len(args.run_ids)} models")

    # Load data
    print("Loading data...")
    train_ds, val_splits, stats, _ = load_data("data/split_manifest.json", "data/split_stats.json")
    val_loaders = {
        name: DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=4, pin_memory=True)
        for name, subset in val_splits.items()
    }

    # Phys stats
    print("Computing phys stats...")
    phys_stats = compute_phys_stats(train_ds, device, asinh_pressure=True, asinh_scale=args.asinh_scale)
    print(f"  mean: {phys_stats['y_mean'].tolist()}, std: {phys_stats['y_std'].tolist()}")

    # Load models
    print("Loading models...")
    models_and_heads = []
    for run_id in args.run_ids:
        print(f"  {run_id}...", end=" ", flush=True)
        model, refine = load_model_and_refine(run_id, device)
        models_and_heads.append((model, refine))
        print("OK")

    stats = {k: v.to(device) for k, v in stats.items()}
    eval_kwargs = dict(
        val_loaders=val_loaders, phys_stats=phys_stats, stats=stats, device=device,
        asinh_pressure=True, asinh_scale=args.asinh_scale,
        residual_prediction=True, high_p_clamp=True,
    )

    # Individual models
    print("\nEvaluating individual models...")
    for i, run_id in enumerate(args.run_ids):
        results = evaluate_ensemble([models_and_heads[i]], **eval_kwargs)
        print_results(f"Single model: {run_id} (seed {42 + i})", results)

    # 4-seed ensemble
    if len(args.run_ids) >= 4:
        print("\nEvaluating 4-seed ensemble (seeds 42-45)...")
        results_4 = evaluate_ensemble(models_and_heads[:4], **eval_kwargs)
        print_results("4-Seed Ensemble (42-45)", results_4)

    # 8-seed ensemble
    if len(args.run_ids) >= 8:
        print("\nEvaluating 8-seed ensemble (seeds 42-49)...")
        results_8 = evaluate_ensemble(models_and_heads[:8], **eval_kwargs)
        print_results("8-Seed Ensemble (42-49)", results_8)


if __name__ == "__main__":
    main()
