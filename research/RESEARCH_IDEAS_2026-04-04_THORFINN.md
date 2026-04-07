# SENPAI Research Ideas — 2026-04-04 (Thorfinn Assignment)

Generated for idle student thorfinn after PR #2112 (Mesh-Density Weighted L1) closed as dead-end.

**Current single-model baseline (PR #2104, +aft_foil_srf, 8-seed mean, seeds 42-49):**
p_in=13.19 ± 0.33, p_oodc=7.92 ± 0.17, p_tan=30.05 ± 0.36, p_re=6.45 ± 0.07

**Active WIP (do NOT duplicate):** fern #2117 (fore-foil SRF head), frieren #2107 (dual-frame coords),
nezuko #2115 (gap/stagger perturbation), tanjiro #2114 (gradient centralization),
edward #2113 (Huber loss), askeladd #2106 (Fourier PE), alphonse #2116 (Charbonnier loss)

---

## Ranked Hypotheses

### 1. boundary-id-onehot — Boundary-ID One-Hot as Sideband Input Feature

**Priority: HIGH. #1 in official priority queue. ~12 LoC. Expected -3 to -8% p_tan. MEDIUM risk.**

**What it is:** Append a 3-dimensional one-hot vector encoding node boundary type (single-foil surface ID=5, fore-foil ID=6, aft-foil ID=7) as an explicit input feature to every mesh node. Non-surface nodes get the zero vector.

**Why it helps p_tan:** The model currently has to infer which boundary a surface node belongs to from geometric cues (SAF distance, position). The dedicated aft-foil SRF head (PR #2104, -0.8% p_tan) confirmed that boundary-type specialization is valuable. But that head only acts after the Transolver trunk has already processed the node without knowing its boundary type. Making the boundary ID explicit in the input allows all 3 Transolver blocks to route computation differently for aft-foil nodes from the very first layer — a much earlier and stronger conditioning signal. The single-foil/tandem asymmetry (p_tan=30.05 vs p_in=13.19, a 2.3x gap) is largely a boundary-type generalization problem; explicit boundary type removes a major source of ambiguity.

**Implementation — exactly where to add code:**

The one-hot must be constructed from RAW (pre-standardization) features, then appended AFTER the Fourier PE. The three proxy formulas, using already-computed variables:

```python
# Immediately AFTER the Fourier PE append (after line ~1569 in train.py):
# Build boundary-ID one-hot [B, N, 3] from raw pre-standardized signals
# (These signals are available even if aft_foil_srf is False, as raw_dsdf and _is_tandem
#  can be recomputed here from the already-standardized x and raw backup)
#
# IMPORTANT: _raw_saf_norm and _is_tandem must be computed from raw x BEFORE line 1552.
# Add the following block just before the standardization call (line 1552):
#
#   _raw_saf_norm_bid = x[:, :, 2:4].norm(dim=-1)         # [B, N]
#   _is_tandem_bid = (x[:, 0, 22].abs() > 0.01)           # [B]
#   _bid_single   = is_surface & ~_is_tandem_bid.unsqueeze(1)                                        # ID=5
#   _bid_fore     = is_surface & (_raw_saf_norm_bid <= 0.005) & _is_tandem_bid.unsqueeze(1)          # ID=6
#   _bid_aft      = is_surface & (_raw_saf_norm_bid  > 0.005) & _is_tandem_bid.unsqueeze(1)          # ID=7
#   _bid_onehot   = torch.stack([_bid_single, _bid_fore, _bid_aft], dim=-1).float()  # [B, N, 3]
#
# Then after the Fourier PE append:
#   x = torch.cat([x, fourier_pe, _bid_onehot], dim=-1)  # appended last
```

Add a config flag `boundary_id_onehot: bool = False` and wrap the block. Input dim increases by 3 (44 → 47 or 45 → 48); the model input projection is dynamic so no other changes needed.

**Suggested experiment:**
```
--boundary_id_onehot
```
Run 2 seeds for initial validation. If p_tan improves, run 8 seeds.

**Key insight from dead-ends:** FiLM conditioning on gap/stagger (#2104) caused catastrophic p_oodc regression (+41.6%) because it changed the condition dynamically. One-hot is a static input feature — it doesn't alter the loss landscape, it just removes information ambiguity. The mechanism is completely different and the risk profile is lower.

**Confidence:** Strong — direct address of the primary bottleneck (boundary-type ambiguity). Infrastructure already exists for boundary proxy detection. Expected to compound with aft-foil SRF head already merged.

---

### 2. pcgrad-3way — Tandem-Specific PCGrad 3-Way Task Split

**Priority: HIGH. ~15 LoC. Expected -2 to -5% p_tan. MEDIUM risk.**

**What it is:** Re-enable PCGrad (`--enable_pcgrad`, currently disabled) with a 3-way task split: (A) single-foil samples, (B) tandem "normal" samples, (C) tandem extreme-Re/AoA samples. Gradient surgery is applied across all three groups rather than the current 2-group split.

**Why it helps p_tan:** PCGrad was disabled in PR #1846 because it added ~18% memory overhead without improvement. However, the current baseline has changed significantly since then — we now have residual prediction, surface refinement, and aft-foil SRF. The gradient conflict between single-foil (p_in) and tandem (p_tan) is the most likely explanation for the 2.3x performance gap. PCGrad's gradient surgery directly addresses this conflict by projecting out gradient components that would harm the other task. The 3-way split adds explicit isolation of the hardest OOD tandem cases, which the 2-way split would lump into the same group as easier tandem samples.

**Implementation:**
```python
# In train.py, re-enable pcgrad with a new 3-way group assignment:
# Group A: ~is_tandem (single-foil)
# Group B: is_tandem & ~is_extreme (tandem, normal Re/AoA)
# Group C: is_tandem & is_extreme (tandem, extreme Re/AoA — top/bottom 10% of Re distribution)
# is_extreme: x[:, 0, re_feature_idx].abs() > threshold (to be calibrated from data)
```

New flags: `--pcgrad_3way` (enables 3-way split), `--pcgrad_extreme_threshold 0.8` (percentile cutoff for "extreme" Re/AoA).

**Caution:** PCGrad was confirmed to add memory overhead. With --aft_foil_srf now in baseline, test memory usage first. If OOM, reduce slice_num temporarily for the sweep.

**Confidence:** Moderate. The 2-way PCGrad failed before surface_refine was added. The mechanism is sound; the question is whether the current baseline is already handling gradient conflict through other means (disable_pcgrad being the current state).

---

### 3. langevin-noise — Langevin Gradient Noise (SGLD-Style)

**Priority: MEDIUM-HIGH. ~10 LoC. Expected -1 to -3% p_in. LOW-MEDIUM risk.**

**What it is:** After each Lion optimizer step, add isotropic Gaussian noise to model parameters (or equivalently to gradients before the update) at a temperature that decays during training. This implements Stochastic Gradient Langevin Dynamics (SGLD) and encourages exploration of flat basins in the loss landscape.

**Why it helps:** Lion is a sign-based optimizer that makes discrete step-direction decisions. Unlike SGD, Lion has no inherent stochasticity from minibatch noise at the gradient level (the sign is either +1 or -1). Adding calibrated Langevin noise restores exploration dynamics, helping escape sharp local minima that Lion's sign-quantization might trap the model in. The effect is most visible for p_in (in-distribution) where the model has the most headroom to improve.

**Implementation:**
```python
# After optimizer.step() and before ema update:
if cfg.langevin_noise > 0.0:
    noise_scale = cfg.langevin_noise * (1.0 - epoch / cfg.cosine_T_max)  # anneal to 0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(torch.randn_like(p.data) * noise_scale)
```

New flags: `--langevin_noise 1e-4` (start value). Sweep: {5e-5, 1e-4, 3e-4}. Anneal to zero by end of training so EMA averaging captures a stable final basin.

**Key constraint:** EMA starts at epoch 140 and averages the noisy-then-stabilized weights. The annealing schedule must reach near-zero before epoch 140 so EMA isn't averaging noisy checkpoints.

**Confidence:** Moderate. Langevin noise is well-established in Bayesian deep learning (Welling & Teh, 2011). The interaction with Lion's sign-based updates is theoretically complementary but empirically untested in this setting.

---

### 4. fore-aft-loss-split — Tandem Fore/Aft Decoupled Loss Weighting

**Priority: MEDIUM. ~8 LoC. Expected -1 to -3% p_tan. LOW risk.**

**What it is:** Apply separate loss weights to fore-foil (ID=6) and aft-foil (ID=7) surface nodes rather than treating all tandem surface nodes uniformly. Specifically: upweight aft-foil nodes by a factor of 1.5-2.0 in the L1 surface loss, while keeping fore-foil weight at 1.0.

**Why it helps p_tan:** The aft-foil SRF head (PR #2104) confirmed that aft-foil nodes have distinct error patterns worth specializing for. The surface loss currently weights all tandem surface nodes equally. But aft-foil pressure is harder to predict (sits in the wake of the fore-foil, highly sensitive to gap/stagger geometry). Upweighting aft-foil in the main loss directly pushes the trunk's representations to be more informative for aft-foil nodes, complementing the dedicated SRF head correction.

**Implementation:**
```python
# In the surface loss computation, after building surface_mask:
if cfg.aft_foil_loss_weight > 1.0 and _aft_foil_mask is not None:
    node_weights = torch.ones(B, N, device=device)
    node_weights[_aft_foil_mask] = cfg.aft_foil_loss_weight
    surface_loss = (l1_per_node * node_weights.unsqueeze(-1))[surface_mask].mean()
```

New flag: `--aft_foil_loss_weight 1.5` (sweep: {1.5, 2.0, 3.0}).

**Distinction from dead-ends:** PR #1893 (Foil-2 Loss Upweighting) was tried before dedicated SRF heads existed and used a shared head — marginal result. This experiment uses the existing _aft_foil_mask infrastructure (same proxy that drives the merged SRF head) to apply a separate weight to aft-foil nodes specifically, and is run WITH the aft-foil SRF head already in baseline. The combination of loss upweighting + dedicated SRF head has not been tested.

**Confidence:** Moderate-low. The interaction between loss upweighting and the existing dedicated SRF head is unknown — they may be partially redundant, or they may compound. The risk of p_in/p_oodc regression is real if the weight is too aggressive.

---

### 5. pressure-poisson — Precomputed Pressure-Poisson Soft Constraint

**Priority: MEDIUM. ~65 LoC. Expected -2 to -4% p_tan. MEDIUM-HIGH risk.**

**What it is:** Add an auxiliary loss that penalizes violations of the pressure Poisson equation: ∇²p ≈ -ρ(u·∇)u. Using precomputed finite-difference Laplacian stencils on the mesh, compute a soft physics consistency loss between predicted pressure and predicted velocity divergence.

**Why it helps p_tan:** p_tan involves tandem-foil flow with complex pressure distributions in the inter-foil channel. The model currently learns pressure from data alone with no physics constraint. Adding a Poisson residual loss directly encodes the governing equation, which should regularize predictions toward physically consistent solutions — especially valuable in the OOD tandem regime where pure data-driven extrapolation is least reliable.

**Implementation sketch:**
```python
# Precompute: for each mesh point, identify k nearest neighbors (k=6) and precompute
# finite-diff weights for ∇² using mesh connectivity (done once per batch)
# At training time:
laplacian_p = compute_mesh_laplacian(pred_p, mesh_coords, precomputed_weights)  # [B, N]
divergence_rhs = -rho * compute_advection(pred_ux, pred_uy, mesh_coords)         # [B, N]
poisson_loss = F.l1_loss(laplacian_p[interior_mask], divergence_rhs[interior_mask])
loss = main_loss + cfg.poisson_weight * poisson_loss
```

New flags: `--poisson_loss`, `--poisson_weight 0.01` (sweep: {0.001, 0.01, 0.1}).

**Implementation challenges:** Mesh connectivity varies between samples (irregular mesh). Precomputing neighbor indices each batch is expensive. The mesh coordinate features are standardized, requiring careful denormalization for physics-correct stencil weights. This is the most implementation-heavy idea on this list and the most likely to have subtle bugs.

**Confidence:** Moderate in principle (physics-informed losses are well-established: Raissi et al. PINN 2019). Low in practice for this specific mesh topology and training setup. Recommend only if simpler ideas (#1-4) are exhausted.

---

## Summary Table

| Rank | Slug | Code size | Risk | Expected p_tan gain | Priority |
|------|------|-----------|------|--------------------:|----------|
| 1 | boundary-id-onehot | ~12 LoC | MEDIUM | -3 to -8% | HIGH |
| 2 | pcgrad-3way | ~15 LoC | MEDIUM | -2 to -5% | HIGH |
| 3 | langevin-noise | ~10 LoC | LOW-MEDIUM | -1 to -3% | MEDIUM-HIGH |
| 4 | fore-aft-loss-split | ~8 LoC | LOW | -1 to -3% | MEDIUM |
| 5 | pressure-poisson | ~65 LoC | MEDIUM-HIGH | -2 to -4% | MEDIUM |

**Recommended assignment for thorfinn: `boundary-id-onehot`** — highest expected impact, #1 in official priority queue, directly attacks the p_tan bottleneck via a mechanism confirmed by multiple prior experiments (coordinate-frame sensitivity, dedicated-head gains), and uses already-validated proxy detection infrastructure from PR #2104.
