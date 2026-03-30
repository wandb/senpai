"""Pre-compute ensemble soft labels for offline distillation.

Runs 8 teacher models on all training samples, averages predictions in
the model's output space (z-score normalized residual), and saves per-sample
soft labels as a .pt file.

The soft labels are stored in the model's RAW output space (before sample_stds
division and refinement), so they can be used directly with any sample_stds
during training.

Usage:
    cd cfd_tandemfoil
    CUDA_VISIBLE_DEVICES=0 python precompute_soft_labels.py [--output soft_labels.pt]
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
from data.prepare_multi import X_DIM, pad_collate, load_data

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 4

# The 8 ensemble checkpoints
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
    for p in m.parameters():
        p.requires_grad_(False)
    rh = SurfaceRefinementHead(n_hidden=mc['n_hidden'], out_dim=3, hidden_dim=192, n_layers=3).to(DEVICE)
    rh_sd = {k.removeprefix("_orig_mod."): v for k, v in
             torch.load(mdir / "refine_head.pt", map_location=DEVICE, weights_only=True).items()}
    rh.load_state_dict(rh_sd)
    rh.eval()
    for p in rh.parameters():
        p.requires_grad_(False)
    models.append(m)
    refine_heads.append(rh)
    print(f"  [{i}] {mdir.name} loaded")

ref_model = models[0]

# --- Load data ---
train_ds, val_splits, stats, _ = load_data("data/split_manifest.json", "data/split_stats.json", debug=False)
stats = {k: v.to(DEVICE) for k, v in stats.items()}
loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

# --- Physics helpers ---
def _umag_q(y, mask):
    n = mask.float().sum(1, True).clamp(min=1.)
    ux = (y[:,:,0]*mask.float()).sum(1,True)/n
    uy = (y[:,:,1]*mask.float()).sum(1,True)/n
    U = (ux**2+uy**2).sqrt().clamp(min=1.).unsqueeze(-1)
    return U, 0.5*U**2

def _phys_norm(y, Umag, q):
    y_p = y.clone()
    y_p[:,:,0:1] /= Umag; y_p[:,:,1:2] /= Umag; y_p[:,:,2:3] /= q
    return y_p

# --- Compute phys stats ---
print("Computing physics normalization stats...")
_phys_sum = torch.zeros(3, device=DEVICE)
_phys_sq_sum = torch.zeros(3, device=DEVICE)
_phys_n = 0.0
with torch.no_grad():
    for _x, _y, _, _m in tqdm(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs), desc="Phys stats", leave=False):
        _y, _m = _y.to(DEVICE), _m.to(DEVICE)
        U, q = _umag_q(_y, _m)
        _yp = _phys_norm(_y, U, q)
        m = _m.float().unsqueeze(-1)
        _phys_sum += (_yp*m).sum((0,1))
        _phys_sq_sum += (_yp**2*m).sum((0,1))
        _phys_n += _m.float().sum().item()
pm = (_phys_sum/_phys_n).float()
ps = ((_phys_sq_sum/_phys_n - pm**2).clamp(min=0.).sqrt()).clamp(min=1e-6).float()
phys = {"y_mean": pm, "y_std": ps}
print(f"  Cp stats: mean={pm.tolist()}, std={ps.tolist()}")

# --- Pre-compute soft labels ---
# We'll process training data sequentially (no shuffle) and save per-sample
# Each sample gets the averaged ensemble prediction in LOSS SPACE
# (after sample_stds division and refinement)
print(f"\nPre-computing ensemble soft labels for {len(train_ds)} training samples...")

# We need to process in order and track sample boundaries within batches
# Use sequential DataLoader (no shuffle), track cumulative sample count
sequential_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

all_soft_labels = []  # list of [N_i, 3] tensors (one per sample, unpadded)
sample_idx = 0

with torch.no_grad():
    for x_raw, y, is_surface, mask in tqdm(sequential_loader, desc="Soft labels"):
        x_raw, y = x_raw.to(DEVICE), y.to(DEVICE)
        is_surface, mask = is_surface.to(DEVICE), mask.to(DEVICE)
        B = x_raw.shape[0]

        # Preprocess (shared)
        raw_dsdf = x_raw[:,:,2:10]
        dist_feat = torch.log1p(raw_dsdf.abs().min(dim=-1,keepdim=True).values*10.)
        _raw_aoa = x_raw[:,0,14:15]
        x = (x_raw - stats["x_mean"]) / stats["x_std"]
        curv = x[:,:,2:6].norm(dim=-1,keepdim=True) * is_surface.float().unsqueeze(-1)
        x = torch.cat([x, curv, dist_feat], dim=-1)
        rxy = x[:,:,:2]
        xymin, xymax = rxy.amin(1,True), rxy.amax(1,True)
        xyn = (rxy-xymin)/(xymax-xymin+1e-8)

        # Physics normalization of ground truth (for sample_stds computation)
        Umag, q = _umag_q(y, mask)
        y_phys = _phys_norm(y, Umag, q)
        y_norm = (y_phys - phys["y_mean"]) / phys["y_std"]

        # Freestream subtraction
        fs = torch.zeros(B,1,3,device=DEVICE)
        fs[:,0,0] = torch.cos(_raw_aoa.squeeze(-1))
        fs[:,0,1] = torch.sin(_raw_aoa.squeeze(-1))
        vf = (fs - phys["y_mean"])/phys["y_std"]
        y_norm = y_norm - vf

        # Sample stds (clean, no target noise)
        raw_gap = x[:,0,21]; is_tan = raw_gap.abs() > 0.5
        sample_stds = torch.ones(B,1,3,device=DEVICE)
        ch_clamp = torch.tensor([0.1,0.1,2.0],device=DEVICE)
        tan_clamp = torch.tensor([0.3,0.3,2.0],device=DEVICE)
        for b in range(B):
            v = mask[b]; c = tan_clamp if is_tan[b] else ch_clamp
            sample_stds[b,0] = y_norm[b,v].std(dim=0).clamp(min=c)

        # Get ensemble prediction for this batch
        teacher_preds = []
        for mi in range(8):
            # Per-teacher fourier PE
            freqs_i = torch.cat([models[mi].fourier_freqs_fixed.to(DEVICE), models[mi].fourier_freqs_learned.abs()])
            fpe_i = torch.cat([(xyn.unsqueeze(-1)*freqs_i).sin().flatten(-2),
                               (xyn.unsqueeze(-1)*freqs_i).cos().flatten(-2)], dim=-1)
            x_no_fpe = torch.cat([x, torch.zeros_like(fpe_i)], dim=-1)  # placeholder
            x_i = torch.cat([x, fpe_i], dim=-1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = models[mi]({"x": x_i})
            pred = out["preds"].float()
            hidden = out["hidden"].float()

            # Apply sample_stds (same as training)
            pred_loss = pred / sample_stds

            # Apply refinement head
            surf_idx = is_surface.nonzero(as_tuple=False)
            if surf_idx.numel() > 0 and refine_heads[mi] is not None:
                sh = hidden[surf_idx[:,0], surf_idx[:,1]]
                sp = pred_loss[surf_idx[:,0], surf_idx[:,1]]
                corr = refine_heads[mi](sh, sp).float()
                pred_loss = pred_loss.clone()
                pred_loss[surf_idx[:,0], surf_idx[:,1]] += corr

            teacher_preds.append(pred_loss)

        # Average in loss space
        ensemble_pred = torch.stack(teacher_preds, dim=0).mean(dim=0)  # [B, N, 3]

        # Extract per-sample (unpadded) and save to CPU
        for b in range(B):
            n_valid = mask[b].sum().item()
            soft_label = ensemble_pred[b, :n_valid].cpu()  # [N_i, 3]
            all_soft_labels.append(soft_label)
            sample_idx += 1

print(f"  Computed {len(all_soft_labels)} soft labels")
print(f"  Shapes: min={min(s.shape[0] for s in all_soft_labels)}, max={max(s.shape[0] for s in all_soft_labels)}")

# Save
output_path = sys.argv[1] if len(sys.argv) > 1 else "soft_labels.pt"
torch.save(all_soft_labels, output_path)
print(f"  Saved to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
