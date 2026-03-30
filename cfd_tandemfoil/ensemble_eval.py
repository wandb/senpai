"""Ensemble evaluation: average predictions from 8 seed checkpoints.

Loads 8 trained models (with surface refinement heads), runs inference on all
validation splits, averages predictions in physical space, and computes MAE.
"""
import os
import sys
import re
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from data.prepare_multi import X_DIM, pad_collate, load_data, VAL_SPLIT_NAMES

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 4

# Model directories (seed 42-49)
MODEL_DIRS = [
    Path("models/model-qni2ym4v"),  # seed 42
    Path("models/model-9t9t3es1"),  # seed 43
    Path("models/model-1cswlc05"),  # seed 44
    Path("models/model-7wnctpdt"),  # seed 45
    Path("models/model-lzjqzvd6"),  # seed 46
    Path("models/model-4joddf3y"),  # seed 47
    Path("models/model-69f4uwhu"),  # seed 48
    Path("models/model-pr98phk7"),  # seed 49
]

# --- Import model classes ---
train_source = Path("train.py").read_text()
end_marker = "# End Transolver model"
model_defs = train_source[:train_source.index(end_marker) + len(end_marker)]
srh_match = re.search(r'(class SurfaceRefinementHead.*?)(?=class \w)', train_source, re.DOTALL)
ns = {}
exec("import torch; import torch.nn as nn; import torch.nn.functional as F; from einops import rearrange; from timm.layers import trunc_normal_; from collections.abc import Mapping", ns)
exec(model_defs, ns)
exec(srh_match.group(1), ns)
Transolver = ns['Transolver']
SurfaceRefinementHead = ns['SurfaceRefinementHead']

# --- Load all 8 models ---
print(f"Loading {len(MODEL_DIRS)} models...")
models = []
refine_heads = []
for i, mdir in enumerate(MODEL_DIRS):
    with open(mdir / "config.yaml") as f:
        mc = yaml.safe_load(f)
    m = Transolver(**mc).to(DEVICE)
    sd = {k.removeprefix("_orig_mod."): v for k, v in
          torch.load(mdir / "checkpoint.pt", map_location=DEVICE, weights_only=True).items()}
    m.load_state_dict(sd)
    m.eval()
    rh = SurfaceRefinementHead(n_hidden=mc['n_hidden'], out_dim=3, hidden_dim=192, n_layers=3).to(DEVICE)
    rh_sd = {k.removeprefix("_orig_mod."): v for k, v in
             torch.load(mdir / "refine_head.pt", map_location=DEVICE, weights_only=True).items()}
    rh.load_state_dict(rh_sd)
    rh.eval()
    models.append(m)
    refine_heads.append(rh)
    print(f"  [{i}] {mdir.name} loaded")

# Use first model for shared attributes (fourier freqs)
ref_model = models[0]

# --- Load data ---
train_ds, val_splits, stats, _ = load_data("data/split_manifest.json", "data/split_stats.json", debug=False)
stats = {k: v.to(DEVICE) for k, v in stats.items()}
loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# --- Physics helpers ---
def _umag_q(y, mask):
    n = mask.float().sum(1, True).clamp(min=1.)
    ux = (y[:,:,0]*mask.float()).sum(1,True)/n
    uy = (y[:,:,1]*mask.float()).sum(1,True)/n
    U = (ux**2+uy**2).sqrt().clamp(min=1.).unsqueeze(-1)
    return U, 0.5*U**2

def _phys_denorm(y_p, Umag, q):
    y = y_p.clone()
    y[:,:,0:1] = y_p[:,:,0:1].clamp(-10,10)*Umag
    y[:,:,1:2] = y_p[:,:,1:2].clamp(-10,10)*Umag
    y[:,:,2:3] = y_p[:,:,2:3].clamp(-20,20)*q
    return y

# --- Compute phys stats ---
print("Computing physics normalization stats...")
_phys_sum = torch.zeros(3, device=DEVICE)
_phys_sq_sum = torch.zeros(3, device=DEVICE)
_phys_n = 0.0
with torch.no_grad():
    for _x, _y, _, _m in tqdm(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs), desc="Phys stats", leave=False):
        _y, _m = _y.to(DEVICE), _m.to(DEVICE)
        U, q = _umag_q(_y, _m)
        yp = _y.clone(); yp[:,:,0:1]/=U; yp[:,:,1:2]/=U; yp[:,:,2:3]/=q
        m = _m.float().unsqueeze(-1)
        _phys_sum += (yp*m).sum((0,1))
        _phys_sq_sum += (yp**2*m).sum((0,1))
        _phys_n += _m.float().sum().item()
pm = (_phys_sum/_phys_n).float()
ps = ((_phys_sq_sum/_phys_n - pm**2).clamp(min=0.).sqrt()).clamp(min=1e-6).float()
phys = {"y_mean": pm, "y_std": ps}
print(f"  Cp stats: mean={pm.tolist()}, std={ps.tolist()}")

# --- Ensemble evaluation ---
print("\n" + "="*70)
print("ENSEMBLE EVALUATION (8-seed prediction averaging)")
print("="*70)

results = {}
for split_name, split_ds in val_splits.items():
    loader = DataLoader(split_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    mae_surf_ensemble = torch.zeros(3, device=DEVICE)
    mae_surf_single = {i: torch.zeros(3, device=DEVICE) for i in range(8)}
    n_surf = torch.zeros(3, device=DEVICE)

    with torch.no_grad():
        for x_raw, y, is_surface, mask in tqdm(loader, desc=split_name):
            x_raw, y = x_raw.to(DEVICE), y.to(DEVICE)
            is_surface, mask = is_surface.to(DEVICE), mask.to(DEVICE)
            B = x_raw.shape[0]

            # Preprocess (shared across models)
            raw_dsdf = x_raw[:,:,2:10]
            dist_feat = torch.log1p(raw_dsdf.abs().min(dim=-1,keepdim=True).values*10.)
            _raw_aoa = x_raw[:,0,14:15]
            x = (x_raw - stats["x_mean"]) / stats["x_std"]
            curv = x[:,:,2:6].norm(dim=-1,keepdim=True) * is_surface.float().unsqueeze(-1)
            x = torch.cat([x, curv, dist_feat], dim=-1)
            rxy = x[:,:,:2]
            xymin, xymax = rxy.amin(1,True), rxy.amax(1,True)
            xyn = (rxy-xymin)/(xymax-xymin+1e-8)
            freqs = torch.cat([ref_model.fourier_freqs_fixed.to(DEVICE), ref_model.fourier_freqs_learned.abs()])
            fpe = torch.cat([(xyn.unsqueeze(-1)*freqs).sin().flatten(-2),
                             (xyn.unsqueeze(-1)*freqs).cos().flatten(-2)], dim=-1)
            x = torch.cat([x, fpe], dim=-1)

            Umag, q = _umag_q(y, mask)
            # Freestream for residual prediction
            fs = torch.zeros(B,1,3,device=DEVICE)
            fs[:,0,0] = torch.cos(_raw_aoa.squeeze(-1))
            fs[:,0,1] = torch.sin(_raw_aoa.squeeze(-1))
            vf = (fs - phys["y_mean"])/phys["y_std"]

            # Per-sample std (from ground truth, same as training)
            y_phys = y.clone(); y_phys[:,:,0:1]/=Umag; y_phys[:,:,1:2]/=Umag; y_phys[:,:,2:3]/=q
            y_norm = (y_phys - phys["y_mean"])/phys["y_std"] - vf
            sample_stds = torch.ones(B,1,3,device=DEVICE)
            ch_clamp = torch.tensor([0.1,0.1,2.0],device=DEVICE)
            tan_clamp = torch.tensor([0.3,0.3,2.0],device=DEVICE)
            raw_gap = x[:,0,21]; is_tan = raw_gap.abs() > 0.5
            for b in range(B):
                v = mask[b]; c = tan_clamp if is_tan[b] else ch_clamp
                sample_stds[b,0] = y_norm[b,v].std(dim=0).clamp(min=c)

            # Get predictions from all 8 models
            pred_orig_all = []
            surf_idx = is_surface.nonzero(as_tuple=False)

            for mi in range(8):
                # Each model has its own fourier freqs — recompute x with this model's freqs
                freqs_i = torch.cat([models[mi].fourier_freqs_fixed.to(DEVICE), models[mi].fourier_freqs_learned.abs()])
                fpe_i = torch.cat([(xyn.unsqueeze(-1)*freqs_i).sin().flatten(-2),
                                   (xyn.unsqueeze(-1)*freqs_i).cos().flatten(-2)], dim=-1)
                x_i = torch.cat([x[:,:,:-32], fpe_i], dim=-1)  # replace last 32 (fourier PE)

                out = models[mi]({"x": x_i})
                pred = out["preds"].float()
                hidden = out["hidden"].float()

                # Apply refinement
                pred_loss = pred / sample_stds
                if surf_idx.numel() > 0:
                    sh = hidden[surf_idx[:,0], surf_idx[:,1]]
                    sp = pred_loss[surf_idx[:,0], surf_idx[:,1]]
                    corr = refine_heads[mi](sh, sp).float()
                    pred_loss = pred_loss.clone()
                    pred_loss[surf_idx[:,0], surf_idx[:,1]] += corr

                # Denormalize
                pred_denorm = pred_loss * sample_stds + vf
                pred_phys = pred_denorm * phys["y_std"] + phys["y_mean"]
                pred_orig = _phys_denorm(pred_phys, Umag, q)
                pred_orig_all.append(pred_orig)

                # Single model MAE
                surf_mask = mask & is_surface
                y_clamped = y.clamp(-1e6, 1e6)
                err_i = (pred_orig - y_clamped).abs()
                finite = err_i.isfinite()
                err_i = err_i.where(finite, torch.zeros_like(err_i))
                mae_surf_single[mi] += (err_i * surf_mask.unsqueeze(-1)).sum(dim=(0,1))

            # Ensemble: average predictions in physical space
            ensemble_pred = torch.stack(pred_orig_all, dim=0).mean(dim=0)

            # Ensemble MAE
            surf_mask = mask & is_surface
            y_clamped = y.clamp(-1e6, 1e6)
            err_ens = (ensemble_pred - y_clamped).abs()
            finite = err_ens.isfinite()
            err_ens = err_ens.where(finite, torch.zeros_like(err_ens))
            mae_surf_ensemble += (err_ens * surf_mask.unsqueeze(-1)).sum(dim=(0,1))
            n_surf += (surf_mask.unsqueeze(-1) * finite).sum(dim=(0,1)).float()

    # Normalize
    mae_ens = (mae_surf_ensemble / n_surf.clamp(min=1)).cpu().numpy()
    mae_singles = np.array([(mae_surf_single[i] / n_surf.clamp(min=1)).cpu().numpy() for i in range(8)])

    results[split_name] = {
        "ensemble": {"Ux": mae_ens[0], "Uy": mae_ens[1], "p": mae_ens[2]},
        "single_mean": {"Ux": mae_singles[:,0].mean(), "Uy": mae_singles[:,1].mean(), "p": mae_singles[:,2].mean()},
        "single_std": {"Ux": mae_singles[:,0].std(), "Uy": mae_singles[:,1].std(), "p": mae_singles[:,2].std()},
    }

    print(f"\n{split_name}:")
    print(f"  Ensemble MAE surf_p:  {mae_ens[2]:.2f}")
    print(f"  Single mean±std p:    {mae_singles[:,2].mean():.2f} ± {mae_singles[:,2].std():.2f}")
    print(f"  Improvement:          {(1 - mae_ens[2]/mae_singles[:,2].mean())*100:.1f}%")

# --- Summary ---
print("\n" + "="*70)
print("ENSEMBLE RESULTS SUMMARY")
print("="*70)
print(f"\n{'Split':30s} {'Ensemble p':>12s} {'Single mean p':>14s} {'Single std p':>13s} {'Improvement':>12s}")
print("-"*85)
for sn in VAL_SPLIT_NAMES:
    r = results[sn]
    ens_p = r["ensemble"]["p"]
    sm_p = r["single_mean"]["p"]
    ss_p = r["single_std"]["p"]
    imp = (1 - ens_p/sm_p)*100
    print(f"{sn:30s} {ens_p:12.2f} {sm_p:14.2f} {ss_p:13.2f} {imp:+11.1f}%")

# Baseline comparison
print(f"\n{'Metric':>10s} {'Ensemble':>10s} {'Single mean±std':>20s} {'Baseline (PR)':>18s}")
print("-"*62)
for sn, bl in [("val_in_dist", 12.95), ("val_tandem_transfer", 30.01),
               ("val_ood_cond", 8.31), ("val_ood_re", 6.70)]:
    r = results[sn]
    print(f"{'p_'+sn.split('_',1)[1]:>10s} {r['ensemble']['p']:10.2f} {r['single_mean']['p']:10.2f}±{r['single_std']['p']:.2f} {bl:18.2f}")
