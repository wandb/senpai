<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-04-05 (Noam)

Generated after reviewing 127 merged experiments, 1509 non-merged runs, and the full
Phase 6 baseline code. These two ideas target the hardest remaining problem: p_tan (tandem
transfer pressure, currently 28.60 on 2-seed avg).

## Current Baseline (for reference)

Run command:
```bash
cd cfd_tandemfoil && python train.py --agent <name> \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias
```

2-seed baseline (seeds 42, 73; PRs #2130 + #2139):

| Metric | 2-seed target |
|--------|--------------|
| p_in   | 13.05        |
| p_oodc | 7.70         |
| p_tan  | **28.60**    |
| p_re   | 6.55         |

---

## Idea 1 (for fern): SRF Head Differential Learning Rate — Accelerating the Zero-Initialized Correctors

### What it is

A one-line learning rate change that gives the surface refinement heads (both `refine_head`
and `aft_srf_head`) a **3x higher learning rate** than the main Transolver trunk.

### Why it might help

The SRF heads are zero-initialized — their parameters start at exactly zero and must
accumulate meaningful correction signal over training. The trunk, by contrast, has a
complex, already-informative random initialization (orthogonal init) and has been developing
representations from epoch 1. By epoch 160, the trunk's parameters are well-optimized
while the SRF heads may still be under-fitted — they're playing catch-up with a handicapped
optimizer.

This is structurally identical to the principle behind **layer-wise adaptive rate scaling
(LARS)** and **differential learning rates** in transfer learning (e.g., fine-tuning the
classification head faster than the backbone in BERT or ViT). In those settings, 3-10x
higher LR for the newly-added head relative to the pretrained backbone is standard practice.

The current code already creates separate param groups for attention parameters (0.5x LR)
and other parameters (1x LR). The SRF heads are added as a third param group at 1x LR
(see lines ~1447-1460 in train.py). The only change needed is raising that group to 3x LR.

No architecture change. No new parameters. No new hyperparameters to sweep (start at 3x;
if it works, we know the direction for further tuning).

**Why this targets p_tan specifically:** The aft-foil SRF head (`aft_srf_head`) is the
head most directly responsible for p_tan — it fires only on aft-foil nodes in tandem
configurations. If the SRF heads are under-fitted, the aft-foil head suffers most because
it has the fewest training samples (tandem is a minority of the training set) and the
hardest task (predicting wake-interaction pressure). Giving it more per-step learning
signal directly targets this bottleneck.

**Known risk:** Too-high LR on zero-initialized layers can cause instability if the first
few gradient signals are noisy (oscillation around zero). The safest approach is to warm
up the SRF head LR from 1x to 3x over the first 20 epochs before jumping to full 3x.
Start with no warmup for simplicity; if that destabilizes, add warmup.

### Papers and prior work

- LARS: "Large Minibatch SGD: Training ImageNet in 1 Hour" (Goyal et al., 2017)
- Layer-wise LR in fine-tuning: standard in HuggingFace transformers, CLIP fine-tuning
- Zero-init heads with separate LR: Masked Autoencoders (He et al., 2021) uses separate
  LR schedules for the decoder (new) vs encoder (pretrained) heads
- Our own programme: #622 "Differential LR: attention 1.5e-3, rest 3e-3" was a WINNER —
  confirms that parameter-group LR separation works in this codebase

### Implementation

**Step 1: Find the SRF param group addition in train.py.**

Around line 1447:
```python
# Add refinement head params to optimizer if enabled
if refine_head is not None:
    _refine_params = list(refine_head.parameters())
    base_opt.add_param_group({'params': _refine_params, 'lr': _base_lr})
    ...

if aft_srf_head is not None:
    _aft_params = list(aft_srf_head.parameters())
    base_opt.add_param_group({'params': _aft_params, 'lr': _base_lr})
```

**Step 2: Add a config flag.**

In `Config` dataclass:
```python
srf_lr_multiplier: float = 1.0  # LR multiplier for SRF heads relative to trunk (1.0=same)
```

**Step 3: Change the LR for the SRF param groups.**

```python
if refine_head is not None:
    _refine_params = list(refine_head.parameters())
    _srf_lr = _base_lr * cfg.srf_lr_multiplier
    base_opt.add_param_group({'params': _refine_params, 'lr': _srf_lr})
    print(f"Added {sum(p.numel() for p in _refine_params):,} refinement head params "
          f"to optimizer (lr={_srf_lr:.2e}, {cfg.srf_lr_multiplier}x trunk)")

if aft_srf_head is not None:
    _aft_params = list(aft_srf_head.parameters())
    _srf_lr = _base_lr * cfg.srf_lr_multiplier
    base_opt.add_param_group({'params': _aft_params, 'lr': _srf_lr})
    print(f"Added {sum(p.numel() for p in _aft_params):,} aft-foil SRF head params "
          f"to optimizer (lr={_srf_lr:.2e}, {cfg.srf_lr_multiplier}x trunk)")
```

**Step 4: Add `--srf_lr_multiplier` to argparse.** The `simple_parsing` library reads
from the Config dataclass automatically — no explicit argparse change needed.

**Step 5: Run the sweep.**

Run 3 multiplier values × 2 seeds = 6 experiments:

```bash
# Multiplier 2x, seed 42
cd cfd_tandemfoil && python train.py --agent fern \
  --wandb_name "fern/srf-lr-2x-s42" --wandb_group phase6/srf-lr-mult \
  --srf_lr_multiplier 2.0 --seed 42 \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias

# Multiplier 2x, seed 73 (same + --seed 73, wandb_name "fern/srf-lr-2x-s73")

# Multiplier 3x, seed 42 (same + --srf_lr_multiplier 3.0, wandb_name "fern/srf-lr-3x-s42")
# Multiplier 3x, seed 73

# Multiplier 5x, seed 42 (same + --srf_lr_multiplier 5.0, wandb_name "fern/srf-lr-5x-s42")
# Multiplier 5x, seed 73
```

### What to report

- Surface MAE (p_in, p_oodc, p_tan, p_re) for all 6 runs in a table with W&B run IDs
- 2-seed average per multiplier value
- Whether any run diverged early (a sign of too-high LR)
- Log `srf_head/grad_norm` if possible to see whether the SRF head gradients were
  previously small (validating the hypothesis) or already large

### Expected impact

p_tan -0.5% to -1.5% (best case: -2%). Confidence is moderate: the mechanism is clean
and theoretically sound, but the SRF heads may already be learning adequately at 1x LR.
The worst case is neutral (no change) — the zero-init means the heads can't actively hurt
at startup, so catastrophic failure is unlikely.

---

## Idea 2 (for askeladd): Surface Normal Pressure Gradient Auxiliary Loss (Physics-Informed Constraint)

### What it is

An auxiliary loss term that penalizes predicted pressure gradient in the surface-normal
direction at airfoil surface nodes. At a no-slip wall, the pressure gradient normal to
the surface must be zero (inviscid Euler: dp/dn = 0 from the momentum equation normal
to the wall). Currently the model learns this implicitly from data — we can make it
explicit.

### Why it might help

The val_tandem_transfer problem is a **geometric generalization** problem: the model
sees NACA2412 and NACA9412 front foils during training, and must generalize to NACA6416.
The pressure distribution on any airfoil is strongly constrained by the normal gradient
condition dp/dn=0 at the wall. A model that reliably satisfies this constraint should
extrapolate better to new foil geometries — the pressure distribution must respect the
geometry, not just match training samples.

We have the surface normal proxy already in the input: the SAF gradient features at
`x[:,:,2:4]` (foil-1) and `x[:,:,6:8]` (foil-2) are the signed distance function
gradients, which point in the wall-normal direction at surface nodes. These are included
in the input but the model has no loss incentive to make its pressure prediction respect
the normal-gradient condition they encode.

**What we're adding:** At surface nodes, compute the numerical gradient of the predicted
pressure in the surface-normal direction (approximated using the model's own pressure
predictions at nearby nodes + the SAF normal direction), and penalize deviations from
zero. This adds physical structure without requiring any new labels — it's a constraint
derived from physics, applied purely to the model's own predictions.

**Why this is genuinely new:** The programme has tried representation-level regularization
(contrastive tandem, domain LayerNorm), data augmentation, architectural changes, and
loss reweighting. But it has NEVER tried a physics-constraint auxiliary loss that directly
penalizes violation of the Navier-Stokes equations at the surface. This is a first-principles
physics-informed ML approach (PINN-style) applied to the surface nodes only.

**Why not a full PINN:** Full PINNs on this architecture would require computing gradients
of predictions with respect to spatial coordinates (autograd through the model for spatial
Jacobians), which is expensive and difficult with the irregular mesh. The simplified version
here operates on the hidden space normal to the surface — much cheaper and still captures
the key constraint.

### Implementation approach

**Simplified version (recommended for initial experiment):**

Rather than computing actual spatial Jacobians of the predicted pressure field, use a
self-supervised smoothness constraint in the surface-normal direction: for each pair of
adjacent surface nodes, penalize the rate of change of predicted pressure in the direction
normal to the surface, weighted by how close those nodes are to each other.

More precisely:

**Step 1: Add config flag.**
```python
pde_surf_normal_weight: float = 0.0  # weight for surface normal dp/dn=0 loss
```

**Step 2: Build the auxiliary loss in the training loop.**

After computing `pred` (the model's predictions) and before the backward pass:

```python
if cfg.pde_surf_normal_weight > 0.0 and model.training:
    # Surface node indices
    surf_idx = is_surface.nonzero(as_tuple=False)  # [M, 2]
    if surf_idx.numel() > 0:
        # Pressure predictions at surface nodes (in normalized space, before denorm)
        surf_p = pred[surf_idx[:, 0], surf_idx[:, 1], 2:3]  # [M, 1]
        
        # Surface normals from SAF gradient at surface nodes
        # x[:,:,2:4] = dsdf gradient (x, y components) for foil-1
        # x[:,:,6:8] = dsdf gradient (x, y components) for foil-2
        # Use foil-1 for single-foil and fore-foil nodes; foil-2 for aft-foil nodes
        
        # After standardization: x[:,:,2:4] is the normalized SAF gradient
        raw_saf = x[surf_idx[:, 0], surf_idx[:, 1], 2:4]  # [M, 2] — already normalized
        raw_saf2 = x[surf_idx[:, 0], surf_idx[:, 1], 6:8]  # [M, 2] — foil-2 saf
        # Use foil-2 SAF only for nodes with nonzero foil-2 SAF (aft-foil nodes)
        foil2_active = raw_saf2.norm(dim=-1, keepdim=True) > 0.1  # [M, 1]
        normal_dir = torch.where(foil2_active, raw_saf2, raw_saf)  # [M, 2]
        normal_dir = F.normalize(normal_dir, dim=-1)  # unit surface normal [M, 2]
        
        # Spatial coordinates of surface nodes (normalized)
        surf_xy = x[surf_idx[:, 0], surf_idx[:, 1], :2]  # [M, 2]
        
        # For each surface node, find the surface-normal pressure gradient:
        # Approximate: for nearby pairs of surface nodes (same sample), compute
        # (p_j - p_i) / |x_j - x_i| projected onto the average normal direction.
        # We want this to be small (dp/dn ≈ 0 at the wall).
        
        # Group by sample (loop over batch, or vectorize if feasible)
        pde_loss = torch.tensor(0.0, device=device)
        pde_count = 0
        
        for b in range(B):
            b_surf_mask = (surf_idx[:, 0] == b)  # [M] bool
            if b_surf_mask.sum() < 2:
                continue
            b_xy = surf_xy[b_surf_mask]   # [M_b, 2]
            b_p  = surf_p[b_surf_mask]    # [M_b, 1]
            b_n  = normal_dir[b_surf_mask]  # [M_b, 2]
            
            # Pairwise differences among surface nodes in this sample
            # For efficiency, sample K random pairs rather than all O(M^2) pairs
            M_b = b_xy.shape[0]
            K_pairs = min(M_b * 4, 200)  # at most 200 pairs per sample
            i_idx = torch.randint(M_b, (K_pairs,), device=device)
            j_idx = torch.randint(M_b, (K_pairs,), device=device)
            valid = i_idx != j_idx
            if valid.sum() == 0:
                continue
            i_idx, j_idx = i_idx[valid], j_idx[valid]
            
            dx = b_xy[j_idx] - b_xy[i_idx]  # [K, 2]
            dp = b_p[j_idx] - b_p[i_idx]    # [K, 1]
            dist = dx.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [K, 1]
            
            # Normal component of dx (how much the pair separation is in the normal direction)
            avg_n = F.normalize(b_n[i_idx] + b_n[j_idx], dim=-1)  # [K, 2]
            normal_proj = (dx * avg_n).sum(dim=-1, keepdim=True) / dist  # [K, 1] ∈ [-1, 1]
            
            # dp/dn ≈ (dp / dist) * sign(normal_proj) when normal_proj is large
            # We only penalize pairs that are primarily separated in the normal direction
            weight = normal_proj.abs()  # [K, 1] — weight by how normal this pair is
            dp_dn = (dp / dist) * weight  # [K, 1]
            pde_loss = pde_loss + (dp_dn ** 2 * weight).sum()
            pde_count += (weight > 0.1).sum().item()
        
        if pde_count > 0:
            pde_loss = pde_loss / max(pde_count, 1)
            loss = loss + cfg.pde_surf_normal_weight * pde_loss
            
            # Log
            if batch_idx % 50 == 0:
                wandb.log({
                    "pde/surf_normal_loss": pde_loss.item(),
                    "pde/pair_count": pde_count,
                }, step=global_step)
```

**Step 3: Run sweep.** The weight needs careful tuning — too large will dominate the
primary loss and hurt all metrics, too small has no effect.

```bash
# weight=0.001 (conservative), seed 42
cd cfd_tandemfoil && python train.py --agent askeladd \
  --wandb_name "askeladd/pde-dpdn-w001-s42" --wandb_group phase6/pde-surf-normal \
  --pde_surf_normal_weight 0.001 --seed 42 \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias

# weight=0.001, seed 73

# weight=0.01 (moderate), seed 42
cd cfd_tandemfoil && python train.py --agent askeladd \
  --wandb_name "askeladd/pde-dpdn-w01-s42" --wandb_group phase6/pde-surf-normal \
  --pde_surf_normal_weight 0.01 --seed 42 \
  [same flags]

# weight=0.01, seed 73

# weight=0.05 (strong), seed 42
cd cfd_tandemfoil && python train.py --agent askeladd \
  --wandb_name "askeladd/pde-dpdn-w05-s42" --wandb_group phase6/pde-surf-normal \
  --pde_surf_normal_weight 0.05 --seed 42 \
  [same flags]

# weight=0.05, seed 73
```

### Critical implementation notes

1. **The dp/dn loss must only activate during training** — never during validation. It
   penalizes the model's own predictions, not ground truth. Wrap it in `if model.training`.

2. **Pair sampling is critical for speed.** Computing all O(M^2) surface-node pairs
   per sample (M ~300-500) would be ~100k pairs per batch, each requiring distance
   computation. Limit to K_pairs=200 random pairs per sample (sampled without replacement
   within each batch item). This reduces to ~800 pairs per batch of 4 — negligible overhead.

3. **The normal direction proxy.** The SAF gradient `x[:,:,2:4]` is the 2D gradient of
   the signed distance function — at a smooth surface, this is the inward surface normal.
   After standardization, the magnitude is not 1 (it has been rescaled by `x_std`), so
   always normalize to unit length before using as a direction.

4. **Pressure is in normalized space during loss computation.** The loss is applied in
   the model's normalized prediction space (after physics norm + asinh + per-sample std
   division), not in physical units. This is correct — dp/dn=0 is scale-invariant. The
   penalized gradients are in normalized units but the constraint holds regardless of scale.

5. **Only penalize pairs with large normal projection** (`normal_proj.abs() > 0.1`).
   Surface node pairs that are separated tangentially (along the surface) don't test the
   normal constraint and would add noise.

6. **PCGrad interaction.** Since the PDE loss is added to `loss` (the overall combined
   loss), not to `loss_a` or `loss_b` individually, it will be shared across the 3-way
   PCGrad groups. This is the correct behavior — the physics constraint applies to all
   samples. If it creates gradient conflicts, an alternative is to add it to all three
   task losses proportionally.

### What to report

- Surface MAE (p_in, p_oodc, p_tan, p_re) for all 6 runs, 2-seed avg per weight
- `pde/surf_normal_loss` at convergence — is it being minimized?
- Whether higher weight helps p_tan but hurts p_in (tradeoff check)
- VRAM — random pair sampling is CPU-efficient; overhead should be <1% per epoch

### Expected impact

p_tan -0.5% to -2% at the optimal weight value. The mechanism is well-motivated by
physics but the proxy (pairwise surface-node gradient approximation using random pairs
+ SAF normals) is an approximation of the true dp/dn constraint — results will depend
on whether the SAF normals are clean enough after standardization to provide useful
directional information. Confidence: moderate. The idea is theoretically sound; the
implementation approximation introduces uncertainty.

---

## Summary

| Student | Idea | Mechanism | Risk | Expected p_tan gain |
|---------|------|-----------|------|---------------------|
| fern | SRF Head Differential LR | 3x LR for SRF correction heads | Low (neutral worst case) | -0.5% to -1.5% |
| askeladd | Surface Normal dp/dn=0 Loss | Physics-informed constraint on pressure gradient | Medium (weight tuning needed) | -0.5% to -2% |

Both ideas are genuinely untested in this programme and target p_tan through different
mechanisms: the first accelerates the SRF head optimization; the second adds a physics
prior that should help geometric generalization to unseen foil shapes (NACA6416 front foil).

The SRF LR idea (fern) is the lower-risk option — clean, minimal code change, theoretically
grounded in differential LR practice. The PDE loss (askeladd) is higher-risk but higher
potential reward if the physics constraint actively regularizes toward better geometric
generalization.
