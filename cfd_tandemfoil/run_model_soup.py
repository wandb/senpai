"""Greedy Model Soup: weight-space averaging of multiple seed checkpoints.

Algorithm (Wortsman et al., ICML 2022):
1. Rank models by val/loss
2. Start with best model's weights
3. Greedily try adding each other model (average weights)
4. Keep the addition only if val/loss improves

Also computes a uniform soup (simple average of all weights) for comparison.
"""
import torch
import yaml
import sys
import copy
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from data.prepare_multi import load_data, pad_collate, X_DIM, VAL_SPLIT_NAMES

# --- Extract model classes from train.py (avoid triggering training loop) ---
_train_src = Path('train.py').read_text()
_lines = _train_src.split('\n')
_end_idx = None
for i, line in enumerate(_lines):
    if '# End Transolver model' in line:
        _end_idx = i + 1
        break
assert _end_idx is not None
_model_src = '\n'.join(_lines[:_end_idx])
_ns = {}
exec(_model_src, _ns)

_helper_lines = []
for i, line in enumerate(_lines):
    if line.startswith('def _umag_q') or line.startswith('def _phys_norm') or line.startswith('def _phys_denorm'):
        j = i + 1
        while j < len(_lines) and (_lines[j].startswith('    ') or _lines[j].strip() == ''):
            j += 1
        _helper_lines.extend(_lines[i:j])
exec('\n'.join(_helper_lines), _ns)

Transolver = _ns['Transolver']
_umag_q = _ns['_umag_q']
_phys_norm = _ns['_phys_norm']
_phys_denorm = _ns['_phys_denorm']


SURF_WEIGHT = 20.0  # must match train.py default


def load_state_dict_from_checkpoint(checkpoint_path, device='cpu'):
    """Load state dict from checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema_model" in ckpt:
        return ckpt["ema_model"]
    elif "model" in ckpt:
        return ckpt["model"]
    else:
        return ckpt


def load_model_config(checkpoint_path):
    """Load model config from config.yaml next to checkpoint."""
    config_path = Path(checkpoint_path).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(model_config, device):
    """Create a Transolver model from config."""
    return Transolver(**model_config).to(device)


def average_state_dicts(sd_list):
    """Average a list of state dicts (uniform weight)."""
    avg = {}
    n = len(sd_list)
    for key in sd_list[0]:
        tensors = [sd[key].float() for sd in sd_list]
        avg[key] = sum(tensors) / n
    return avg


def evaluate_model(model, val_splits, stats, phys_stats, device):
    """Evaluate model on all val splits. Returns val/loss and per-split metrics.

    Replicates the validation logic from train.py exactly.
    """
    model.eval()
    x_mean = stats["x_mean"].to(device)
    x_std = stats["x_std"].to(device)

    val_metrics = {}
    split_losses = []

    for split_name, vloader_ds in val_splits.items():
        loader = DataLoader(vloader_ds, batch_size=4, collate_fn=pad_collate, shuffle=False, num_workers=2)

        val_vol = 0.0
        val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = torch.zeros(3, device=device)
        n_vol = torch.zeros(3, device=device)
        n_vbatches = 0

        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x, y = x.to(device), y.to(device)
                is_surface = is_surface.to(device)
                mask = mask.to(device)

                # Feature processing (same as train.py)
                raw_dsdf = x[:, :, 2:10]
                dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
                dist_feat = torch.log1p(dist_surf * 10.0)
                _raw_aoa = x[:, 0, 14:15]
                x_proc = (x - x_mean) / x_std
                curv = x_proc[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
                x_proc = torch.cat([x_proc, curv, dist_feat], dim=-1)

                # Fourier PE
                raw_xy = x_proc[:, :, :2]
                xy_min = raw_xy.amin(dim=1, keepdim=True)
                xy_max = raw_xy.amax(dim=1, keepdim=True)
                xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
                freqs = torch.cat([model.fourier_freqs_fixed.to(device), model.fourier_freqs_learned.abs()])
                xy_scaled = xy_norm.unsqueeze(-1) * freqs
                fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)
                x_proc = torch.cat([x_proc, fourier_pe], dim=-1)

                # Physics normalization of targets
                Umag, q = _umag_q(y, mask)
                y_phys = _phys_norm(y, Umag, q)
                y_norm = (y_phys - phys_stats["y_mean"]) / phys_stats["y_std"]

                # Residual prediction: subtract freestream
                _fs_phys = torch.zeros(y_norm.shape[0], 1, 3, device=device)
                _fs_phys[:, 0, 0] = torch.cos(_raw_aoa.squeeze(-1))
                _fs_phys[:, 0, 1] = torch.sin(_raw_aoa.squeeze(-1))
                _v_freestream = (_fs_phys - phys_stats["y_mean"]) / phys_stats["y_std"]
                y_norm = y_norm - _v_freestream

                # Per-sample std normalization (default config)
                raw_gap = x[:, 0, 21]  # use original x for gap detection
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

                # Forward pass
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model({"x": x_proc})["preds"]
                pred = pred.float()
                pred_loss = pred / sample_stds

                # Compute normalized loss
                abs_err = (pred_loss - y_norm_scaled).abs().nan_to_num(0.0)
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

                # Denormalize for surface MAE
                pred_with_fs = pred + _v_freestream
                pred_phys = pred_with_fs * phys_stats["y_std"] + phys_stats["y_mean"]
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
        split_loss = val_vol + SURF_WEIGHT * val_surf
        mae_surf /= n_surf.clamp(min=1)

        split_losses.append(split_loss)
        val_metrics[split_name] = {
            "loss": split_loss,
            "p": mae_surf[2].item(),
            "Ux": mae_surf[0].item(),
            "Uy": mae_surf[1].item(),
        }

    val_loss = sum(split_losses) / max(len(split_losses), 1)
    return val_loss, val_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoints = [
        ('s42', 'models/model-focedl6i/checkpoint.pt'),
        ('s43', 'models/model-3bhhwhnv/checkpoint.pt'),
        ('s44', 'models/model-6jjejon2/checkpoint.pt'),
        ('s45', 'models/model-hu72tx3d/checkpoint.pt'),
        ('s46', 'models/model-u8m6vddj/checkpoint.pt'),
        ('s47', 'models/model-xbt4j9on/checkpoint.pt'),
        ('s48', 'models/model-bkjl5yrt/checkpoint.pt'),
        ('s49', 'models/model-cel9w3k4/checkpoint.pt'),
    ]

    # Load data
    _, val_splits, stats, _ = load_data('data/split_manifest.json', 'data/split_stats.json', debug=False)

    # Compute phys_stats
    train_ds, _, _, _ = load_data('data/split_manifest.json', 'data/split_stats.json', debug=False)
    stats_loader = DataLoader(train_ds, batch_size=8, collate_fn=pad_collate, shuffle=False, num_workers=4)

    print("Computing phys_stats...")
    _phys_sum = torch.zeros(3, device=device)
    _phys_sq_sum = torch.zeros(3, device=device)
    _phys_n = 0.0
    with torch.no_grad():
        for _x, _y, _is_surf, _mask in tqdm(stats_loader, desc="Phys stats"):
            _y, _mask = _y.to(device), _mask.to(device)
            _U, _q = _umag_q(_y, _mask)
            _yp = _phys_norm(_y, _U, _q)
            _m = _mask.float().unsqueeze(-1)
            _phys_sum += (_yp * _m).sum(dim=(0, 1))
            _phys_sq_sum += (_yp ** 2 * _m).sum(dim=(0, 1))
            _phys_n += _mask.float().sum().item()
    _pm = (_phys_sum / _phys_n).float()
    _ps = ((_phys_sq_sum / _phys_n - _pm ** 2).clamp(min=0).sqrt()).clamp(min=1e-6).float()
    phys_stats = {'y_mean': _pm, 'y_std': _ps}
    print(f"  phys_stats: mean={_pm.tolist()}, std={_ps.tolist()}")

    # Load model config (same for all seeds)
    model_config = load_model_config(checkpoints[0][1])

    # Load all state dicts
    print("\nLoading state dicts...")
    state_dicts = {}
    for name, path in checkpoints:
        state_dicts[name] = load_state_dict_from_checkpoint(path)
        print(f"  Loaded: {name} ({path})")

    # Step 1: Evaluate each individual model
    print("\n" + "=" * 60)
    print("STEP 1: Evaluate individual models")
    print("=" * 60)
    individual_results = {}
    for name, path in checkpoints:
        model = create_model(model_config, device)
        model.load_state_dict(state_dicts[name])
        val_loss, metrics = evaluate_model(model, val_splits, stats, phys_stats, device)
        individual_results[name] = {"val_loss": val_loss, "metrics": metrics}
        p_in = metrics.get("val_in_dist", {}).get("p", 0)
        p_oodc = metrics.get("val_ood_cond", {}).get("p", 0)
        p_tan = metrics.get("val_tandem_transfer", {}).get("p", 0)
        p_re = metrics.get("val_ood_re", {}).get("p", 0)
        print(f"  {name}: val/loss={val_loss:.4f}  p_in={p_in:.2f}  p_oodc={p_oodc:.2f}  p_tan={p_tan:.2f}  p_re={p_re:.2f}")
        del model
        torch.cuda.empty_cache()

    # Rank by val/loss
    ranked = sorted(individual_results.items(), key=lambda x: x[1]["val_loss"])
    ranking_str = ', '.join(f'{n}({r["val_loss"]:.4f})' for n, r in ranked)
    print(f"\nRanking: {ranking_str}")

    # Step 2: Uniform soup (simple average of all weights)
    print("\n" + "=" * 60)
    print("STEP 2: Uniform soup (all 8 models)")
    print("=" * 60)
    all_sds = [state_dicts[name] for name, _ in checkpoints]
    uniform_sd = average_state_dicts(all_sds)
    model = create_model(model_config, device)
    model.load_state_dict(uniform_sd)
    uniform_loss, uniform_metrics = evaluate_model(model, val_splits, stats, phys_stats, device)
    p_in = uniform_metrics.get("val_in_dist", {}).get("p", 0)
    p_oodc = uniform_metrics.get("val_ood_cond", {}).get("p", 0)
    p_tan = uniform_metrics.get("val_tandem_transfer", {}).get("p", 0)
    p_re = uniform_metrics.get("val_ood_re", {}).get("p", 0)
    print(f"  Uniform soup: val/loss={uniform_loss:.4f}  p_in={p_in:.2f}  p_oodc={p_oodc:.2f}  p_tan={p_tan:.2f}  p_re={p_re:.2f}")
    del model
    torch.cuda.empty_cache()

    # Step 3: Greedy soup
    print("\n" + "=" * 60)
    print("STEP 3: Greedy soup")
    print("=" * 60)

    best_name = ranked[0][0]
    soup_members = [best_name]
    soup_sd = copy.deepcopy(state_dicts[best_name])
    best_loss = ranked[0][1]["val_loss"]
    print(f"  Starting with {best_name} (val/loss={best_loss:.4f})")

    for candidate_name, candidate_result in ranked[1:]:
        # Try averaging current soup with candidate
        n_current = len(soup_members)
        trial_sd = {}
        for key in soup_sd:
            trial_sd[key] = (soup_sd[key].float() * n_current + state_dicts[candidate_name][key].float()) / (n_current + 1)

        model = create_model(model_config, device)
        model.load_state_dict(trial_sd)
        trial_loss, trial_metrics = evaluate_model(model, val_splits, stats, phys_stats, device)
        del model
        torch.cuda.empty_cache()

        if trial_loss < best_loss:
            soup_sd = trial_sd
            soup_members.append(candidate_name)
            p_in = trial_metrics.get("val_in_dist", {}).get("p", 0)
            p_oodc = trial_metrics.get("val_ood_cond", {}).get("p", 0)
            print(f"  + Added {candidate_name}: val/loss={trial_loss:.4f} (improved from {best_loss:.4f})")
            best_loss = trial_loss
        else:
            print(f"  - Skipped {candidate_name}: val/loss={trial_loss:.4f} (worse than {best_loss:.4f})")

    print(f"\n  Greedy soup members ({len(soup_members)}): {', '.join(soup_members)}")

    # Final evaluation of greedy soup
    model = create_model(model_config, device)
    model.load_state_dict(soup_sd)
    greedy_loss, greedy_metrics = evaluate_model(model, val_splits, stats, phys_stats, device)
    del model
    torch.cuda.empty_cache()

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    best_single = ranked[0]
    print(f"\nBest single model ({best_single[0]}):")
    print(f"  val/loss={best_single[1]['val_loss']:.4f}")
    for sn, m in best_single[1]['metrics'].items():
        print(f"  {sn}: p={m['p']:.2f}, Ux={m['Ux']:.4f}, Uy={m['Uy']:.4f}")

    print(f"\nUniform soup (8 models):")
    print(f"  val/loss={uniform_loss:.4f}")
    for sn, m in uniform_metrics.items():
        print(f"  {sn}: p={m['p']:.2f}, Ux={m['Ux']:.4f}, Uy={m['Uy']:.4f}")

    print(f"\nGreedy soup ({len(soup_members)} models: {', '.join(soup_members)}):")
    print(f"  val/loss={greedy_loss:.4f}")
    for sn, m in greedy_metrics.items():
        print(f"  {sn}: p={m['p']:.2f}, Ux={m['Ux']:.4f}, Uy={m['Uy']:.4f}")

    # Save greedy soup checkpoint
    soup_path = "models/greedy_soup_checkpoint.pt"
    torch.save(soup_sd, soup_path)
    print(f"\nGreedy soup saved to {soup_path}")


if __name__ == "__main__":
    main()
