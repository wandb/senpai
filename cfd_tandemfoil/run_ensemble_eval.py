"""Self-contained ensemble evaluation — does NOT import train.py (which has no __main__ guard)."""
import torch
import yaml
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from data.prepare_multi import load_data, pad_collate, X_DIM

# --- Extract model classes from train.py lines 1-702 + helper functions ---
# We exec() only the model definitions to avoid triggering the training loop.
_train_src = Path('train.py').read_text()
_lines = _train_src.split('\n')

# Find the "End Transolver model" marker
_end_idx = None
for i, line in enumerate(_lines):
    if '# End Transolver model' in line:
        _end_idx = i + 1
        break
assert _end_idx is not None, "Could not find model boundary in train.py"

# Build namespace with only model code (imports + classes up to line _end_idx)
_model_src = '\n'.join(_lines[:_end_idx])
_ns = {}
exec(_model_src, _ns)

# Also exec the helper functions
_helper_lines = []
for i, line in enumerate(_lines):
    if line.startswith('def _umag_q') or line.startswith('def _phys_norm') or line.startswith('def _phys_denorm'):
        # Grab the function (up to next blank line or def)
        j = i + 1
        while j < len(_lines) and (_lines[j].startswith('    ') or _lines[j].strip() == ''):
            j += 1
        _helper_lines.extend(_lines[i:j])
exec('\n'.join(_helper_lines), _ns)

Transolver = _ns['Transolver']
_umag_q = _ns['_umag_q']
_phys_norm = _ns['_phys_norm']
_phys_denorm = _ns['_phys_denorm']

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_path = Path(checkpoint_path).parent / "config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    model = Transolver(**model_config).to(device)
    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, model_config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoints = [
        'models/model-focedl6i/checkpoint.pt',  # s42
        'models/model-3bhhwhnv/checkpoint.pt',  # s43
        'models/model-6jjejon2/checkpoint.pt',  # s44
        'models/model-hu72tx3d/checkpoint.pt',  # s45
        'models/model-u8m6vddj/checkpoint.pt',  # s46
        'models/model-xbt4j9on/checkpoint.pt',  # s47
        'models/model-bkjl5yrt/checkpoint.pt',  # s48
        'models/model-cel9w3k4/checkpoint.pt',  # s49
    ]

    # Load data
    _, val_splits, stats, _ = load_data('data/split_manifest.json', 'data/split_stats.json', debug=False)

    # Compute phys_stats from training data
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

    # Load all models
    print(f"\nLoading {len(checkpoints)} models...")
    models = []
    for ckpt_path in checkpoints:
        model, _ = load_model(ckpt_path, device)
        models.append(model)
        print(f"  Loaded: {ckpt_path}")

    x_mean = stats["x_mean"].to(device)
    x_std = stats["x_std"].to(device)

    # Evaluate each split
    for split_name, vloader_ds in val_splits.items():
        loader = DataLoader(vloader_ds, batch_size=4, collate_fn=pad_collate, shuffle=False, num_workers=2)

        mae_surf_individual = [torch.zeros(3, device=device) for _ in models]
        mae_surf_ensemble = torch.zeros(3, device=device)
        # Also track top-4 ensemble
        mae_surf_top4 = torch.zeros(3, device=device)
        n_surf = torch.zeros(3, device=device)

        # Top-4 models by val/loss (seeds 42,45,43,48 based on individual metrics)
        top4_indices = [0, 3, 1, 6]  # s42, s45, s43, s48 (lowest val/loss)

        with torch.no_grad():
            for x, y, is_surface, mask in tqdm(loader, desc=split_name, leave=False):
                x, y = x.to(device), y.to(device)
                is_surface = is_surface.to(device)
                mask = mask.to(device)

                surf_mask = mask & is_surface
                if surf_mask.sum() == 0:
                    continue

                # Process input features (same as train.py)
                raw_dsdf = x[:, :, 2:10]
                dist_surf = raw_dsdf.abs().min(dim=-1, keepdim=True).values
                dist_feat = torch.log1p(dist_surf * 10.0)
                x_proc = (x - x_mean) / x_std
                curv = x_proc[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface.float().unsqueeze(-1)
                x_proc = torch.cat([x_proc, curv, dist_feat], dim=-1)

                # Fourier PE (per-model, since learned freqs differ)
                raw_xy = x_proc[:, :, :2]
                xy_min = raw_xy.amin(dim=1, keepdim=True)
                xy_max = raw_xy.amax(dim=1, keepdim=True)
                xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)

                Umag, q = _umag_q(y, mask)

                # Get predictions from each model
                all_pred_orig = []
                for i, model in enumerate(models):
                    freqs_i = torch.cat([model.fourier_freqs_fixed.to(device),
                                         model.fourier_freqs_learned.abs()])
                    xy_scaled_i = xy_norm.unsqueeze(-1) * freqs_i
                    fpe_i = torch.cat([xy_scaled_i.sin().flatten(-2), xy_scaled_i.cos().flatten(-2)], dim=-1)
                    x_i = torch.cat([x_proc, fpe_i], dim=-1)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pred = model({"x": x_i})["preds"]
                    pred = pred.float()

                    # Residual prediction: add freestream back
                    _aoa = x[:, 0, 14]  # AoA0_rad
                    _fs = torch.zeros(pred.shape[0], 1, 3, device=device)
                    _fs[:, 0, 0] = torch.cos(_aoa)
                    _fs[:, 0, 1] = torch.sin(_aoa)
                    _v_fs = (_fs - phys_stats['y_mean']) / phys_stats['y_std']
                    pred = pred + _v_fs

                    # Denormalize
                    pred_phys = pred * phys_stats['y_std'] + phys_stats['y_mean']
                    pred_orig = _phys_denorm(pred_phys, Umag, q)
                    all_pred_orig.append(pred_orig)

                    # Individual MAE
                    err_i = (pred_orig - y).abs()
                    finite = err_i.isfinite()
                    err_i = err_i.where(finite, torch.zeros_like(err_i))
                    mae_surf_individual[i] += (err_i * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))

                # Ensemble-8 average
                ensemble_pred = torch.stack(all_pred_orig, dim=0).mean(dim=0)
                err_ens = (ensemble_pred - y).abs()
                finite = err_ens.isfinite()
                err_ens = err_ens.where(finite, torch.zeros_like(err_ens))
                mae_surf_ensemble += (err_ens * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))

                # Top-4 ensemble
                top4_preds = torch.stack([all_pred_orig[i] for i in top4_indices], dim=0).mean(dim=0)
                err_t4 = (top4_preds - y).abs()
                finite_t4 = err_t4.isfinite()
                err_t4 = err_t4.where(finite_t4, torch.zeros_like(err_t4))
                mae_surf_top4 += (err_t4 * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))

                n_surf += (surf_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()

        # Report
        n_surf_safe = n_surf.clamp(min=1)
        print(f"\n=== {split_name} ===")
        for i in range(len(models)):
            mae_i = mae_surf_individual[i] / n_surf_safe
            print(f"  Model {i} (seed {42+i}): Ux={mae_i[0]:.4f}, Uy={mae_i[1]:.4f}, p={mae_i[2]:.2f}")
        mae_ens = mae_surf_ensemble / n_surf_safe
        print(f"  ENSEMBLE-8 (all models): Ux={mae_ens[0]:.4f}, Uy={mae_ens[1]:.4f}, p={mae_ens[2]:.2f}")
        mae_t4 = mae_surf_top4 / n_surf_safe
        print(f"  ENSEMBLE-4 (top-4):      Ux={mae_t4[0]:.4f}, Uy={mae_t4[1]:.4f}, p={mae_t4[2]:.2f}")

if __name__ == "__main__":
    main()
