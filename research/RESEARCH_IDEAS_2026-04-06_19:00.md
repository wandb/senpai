# SENPAI Research Ideas — 2026-04-06 19:00

Generated after reviewing 1817 experiment PRs and targeted literature search.
Target: p_tan < 28.0 (current baseline PR #2184: p_tan=28.502).

**Current baseline (PR #2184, 2-seed avg, seeds 42 and 73):**
p_in=13.205, p_oodc=7.816, p_tan=28.502, p_re=6.453

**Active WIP (do NOT duplicate):**
- #2211: Surface Pressure Gradient Loss — penalize dp/ds mismatch along surface (alphonse)
- #2210: Arc-Length Surface Loss Reweighting — fix non-uniform mesh density bias (fern)
- #2209: Attention Register Tokens — learnable global slots in Physics-Attention (thorfinn)
- #2208: Iterative SRF Heads (RAFT-style) — N=3 correction passes on surface nodes (askeladd)
- #2207: TE Coordinate Frame — trailing-edge-relative input features for wake coupling (edward)
- #2205: NOBLE Nonlinear Low-Rank Branches in TransolverBlock FFN Retry (nezuko)
- #2199: Spectral Conditioning of Attention (SCA) to prevent OOD collapse (frieren)
- #2197: Geometry-adaptive curvature loss weighting on surface nodes (tanjiro)

---

## Ranked Hypotheses

### 1. analytical-cp-delta — Thin-Airfoil Analytical Cp Baseline + Learned Delta

**Priority: HIGH. ~30 LoC. Expected -2 to -5% p_tan. LOW-MEDIUM risk.**

**What it is:** For each surface node on both foils, compute a physics-derived baseline pressure coefficient Cp_baseline using the thin-airfoil linearization: Cp ≈ -2 * (dY/dX) for the thin airfoil in attached flow (first-order panel approximation). Train the aft-foil SRF head (and optionally the main surface refine head) to predict only the delta: pred_p_final = Cp_baseline + aft_srf(delta). This is NOT the same as the existing `--residual_prediction`, which predicts correction relative to freestream uniform flow (p_freestream = 0.5 * rho * V^2). The thin-airfoil Cp directly encodes surface curvature geometry.

**Why it might help p_tan:** The current SRF head must learn both the baseline pressure shape (suction peak on upper surface, stagnation at LE) AND the wake-induced perturbation for aft-foil. By subtracting the analytical baseline, the delta the network must learn is smaller in magnitude and more tightly localized around wake-interaction anomalies — exactly where aft-foil predictions fail. The delta has near-zero variance in the far field and suction peak region, concentrating the training signal where it matters. This is the DeltaPhi principle (residual learning on physics baseline, not on freestream) applied at the surface level.

**Key distinction from dead-ends:** `--residual_prediction` predicts correction to p=0 (uniform freestream). Panel Cp residual target (#2169, failed) computed Cp from a full panel solver at inference time — this was slow and noisy. This approach instead uses the closed-form thin-airfoil dY/dX approximation, which is purely geometric (zero cost), and applies it only inside the existing `AftFoilRefinementHead` correction term.

**Implementation — exactly where to add code:**

The 24-dim input `x` already contains DSDF geometry features. The node's local surface tangent dY/dX can be approximated from the 2D surface normal features (which are part of the DSDF input). Specifically, feature indices 4:8 encode the DSDF gradient directions for both foils.

```python
# In AftFoilRefinementHead.forward() or just before the surface_refine call:
# Compute thin-airfoil baseline: Cp_ta ≈ -2 * sin(theta) where theta = surface normal angle
# x_raw[:, :, 4:6] = DSDF1 gradient direction (unit normal for foil 1 / fore-foil)
# x_raw[:, :, 6:8] = DSDF2 gradient direction (unit normal for aft-foil)
# For aft-foil nodes: surface_tangent_y = dsdf2_grad_x (perpendicular to normal)
# Cp_baseline = -2.0 * dsdf2_grad_y / (1.0 + eps)  # thin-airfoil, small-angle
# Detach: Cp_baseline = Cp_baseline.detach()
# SRF predicts: delta = aft_srf_mlp(features)  # [B, N_aft, 3]
# Only apply Cp_baseline to pressure channel (dim 2):
# pred_p_aft[:, :, 2] = Cp_baseline[aft_mask] + delta[:, :, 2]
```

New flags: `--analytical_cp_delta` (bool, default False). No parameter changes to SRF head shape. The only new parameter is a clamp on `Cp_baseline` to [-3, 0.5] to prevent numerical artifacts near TE singularity.

**Sweep:** Start with single seed (42). If p_tan improves, run 2 seeds.

**Confidence:** Moderate-strong. The thin-airfoil delta decomposition is a classical technique in aerodynamics (Lighthill, 1951; Abbott & von Doenhoff textbook). The residual-on-physics-baseline principle has strong deep learning support (ResNets, DeltaPhi-NO). The risk is that DSDF gradient features do not faithfully encode the surface tangent angle at every node; verify with a small diagnostic before full run.

---

### 2. wake-deficit-feature — Analytical Wake Deficit Injection for Aft-Foil Input

**Priority: HIGH. ~25 LoC. Expected -2 to -4% p_tan. LOW risk.**

**What it is:** For each tandem sample, compute a scalar "wake deficit strength" at the aft-foil leading edge using a closed-form model of trailing-edge velocity deficit: u_deficit ≈ C_D_approx * chord / (2 * pi * gap), where gap is the inter-foil distance and C_D_approx is estimated from the fore-foil AoA via a simple drag polar lookup. Append this scalar (plus its log) as two additional input features to all aft-foil surface nodes. Non-aft-foil nodes receive zero.

**Why it might help p_tan:** The aft-foil pressure distribution is dominated by the incoming velocity deficit in the fore-foil's wake. The current model must infer this deficit from (gap, stagger, AoA, Re) features but has no direct representation of the strength of the incoming disturbance. A physics-motivated scalar encoding the expected deficit magnitude removes a major source of ambiguity for aft-foil surface nodes specifically. The gap_stagger_spatial_bias (PR #2130, merged, -3.0% p_tan) showed that injecting tandem geometry into slice routing is valuable; injecting a physics-derived velocity deficit is a stronger signal than raw gap/stagger because it already accounts for AoA and Re effects.

**Key distinction from dead-ends:** Inter-foil distance feature (#2148, failed) added raw Euclidean distance between foil centroids — a purely geometric signal with no aerodynamic interpretation. The wake deficit feature here is AoA/Re/chord-aware and represents a physical quantity the aft-foil actually experiences. This is NOT fore-aft cross-attention (which is an architectural change that failed).

**Implementation:**

The 24-dim input already encodes: feature index 22 = gap (inter-foil chordwise distance), index 23 = stagger (perpendicular offset), plus Re and AoA as scalar condition features. The analytical wake model:

```python
# Computed once per sample in the batch collation loop (or in train.py's feature construction):
# Using pre-standardized raw values:
#   aoa_rad = raw_aoa * pi / 180.0
#   cl_fore ≈ 2 * pi * sin(aoa_rad)  (thin-airfoil lift coefficient)
#   cd_fore ≈ 0.01 + cl_fore^2 / (pi * 6.0)  (simple Oswald drag polar, AR=6 approximation)
#   gap_raw = raw_gap  # chord units
#   wake_deficit = cd_fore / (2 * pi * (gap_raw + 0.1))  # +0.1 for numerical stability
#   log_wake_deficit = log(wake_deficit + 1e-6)
#
# Append [wake_deficit, log_wake_deficit] to aft-foil surface nodes only (2 features).
# All other nodes receive [0.0, 0.0].
# Input dim increases by 2.
```

New flags: `--wake_deficit_feature` (bool, default False). Single-seed initial validation. The feature is purely a function of existing input data — no new training data required.

**Sweep:** Try one seed (42). If promising, 2 seeds. Consider ablating log_wake_deficit vs. raw only.

**Confidence:** Moderate. The aerodynamic motivation is sound (wake deficit models are standard in wind turbine array modeling: Jensen top-hat model, 1983; Bastankhah Gaussian model, 2014). The main risk is that the analytical model is too approximate for the NACA6416 aft-foil geometry. However, even an imprecise wake deficit proxy should outperform raw gap/stagger since it captures the AoA dependence.

---

### 3. tandem-curriculum-ramp — Progressive Tandem Sample Weight Curriculum

**Priority: MEDIUM-HIGH. ~15 LoC. Expected -1 to -3% p_tan. LOW risk.**

**What it is:** During the first N_warmup=40 epochs, train almost exclusively on single-foil in-distribution samples (tandem weight=0.05). Then linearly ramp the tandem training weight from 0.05 to 1.0 over epochs 40-80, stabilizing at 1.0 for the final 80 epochs. This is a curriculum learning schedule inspired by the observation that tandem OOD pressure patterns are fundamentally harder to learn and may confuse the network when learned simultaneously with single-foil patterns from epoch 0.

**Why it might help p_tan:** The model currently trains single-foil and tandem samples with equal sampling weight from epoch 0. However, the tandem transfer p_tan=28.50 vs. p_in=13.21 shows a 2.16x performance gap — the model's learned representations appear dominated by single-foil geometry. A warmup curriculum lets the model first learn a robust pressure field representation for the simpler single-foil case, then progressively adapt to the tandem case starting from a strong prior. This mirrors how curriculum learning in NLP first trains on short/simple sequences before long/complex ones (Bengio et al., 2009, ICML).

**Key distinction from dead-ends:** Tandem-focused training with 3x oversampling (PR #2044, failed) — that experiment aggressively oversampled tandem data from epoch 0, which hurt p_in significantly. The curriculum approach starts with under-weighting tandem (not over-weighting single-foil) and ramps — the key difference is the trajectory matters, not just the endpoint weight. The existing `--tandem_ramp` flag in the baseline is NOT a curriculum — it ramps the tandem loss contribution, not the training data sampling weight.

**Implementation:**

```python
# In the DataLoader sampler or batch loss weighting:
# epoch_t = current_epoch / total_epochs
# if epoch_t < warmup_frac:  # warmup_frac = 40/160 = 0.25
#     tandem_sample_weight = cfg.curriculum_tandem_min  # default 0.05
# elif epoch_t < ramp_end_frac:  # ramp_end_frac = 80/160 = 0.5
#     ramp_progress = (epoch_t - warmup_frac) / (ramp_end_frac - warmup_frac)
#     tandem_sample_weight = cfg.curriculum_tandem_min + ramp_progress * (1.0 - cfg.curriculum_tandem_min)
# else:
#     tandem_sample_weight = 1.0
#
# Apply as a per-sample loss multiplier OR as a DataLoader WeightedRandomSampler weight.
# The simpler approach: multiply tandem-sample losses by tandem_sample_weight before mean.
```

New flags: `--tandem_curriculum` (bool), `--curriculum_tandem_min 0.05`, `--curriculum_warmup_epochs 40`, `--curriculum_ramp_epochs 40`.

**Caution:** This interacts with `--tandem_ramp` (which ramps tandem loss weight). Run WITH the existing `--tandem_ramp` flag and verify the two ramps are additive, not conflicting. Log the effective tandem weight per epoch to W&B for diagnostic.

**Confidence:** Moderate. Curriculum learning has strong theoretical grounding (Bengio et al. 2009; Platanios et al. 2019 — competence-based curriculum). The CFD application is novel but the motivation is direct. Risk: if the warmup period is too long, the model overfits to single-foil representations and the ramp-phase tandem adaptation is insufficient.

---

### 4. boundary-integral-aggregation — Fore-Foil Boundary Integral Features for Aft-Foil Nodes

**Priority: MEDIUM-HIGH. ~40 LoC. Expected -2 to -5% p_tan. MEDIUM risk.**

**What it is:** For each aft-foil surface node, compute a fixed-size aggregated representation of all fore-foil surface nodes using a learned kernel. Specifically, for each aft-foil node i and each fore-foil node j, compute:

  K(i,j) = exp(-||r_i - r_j||^2 / sigma^2) * w_mlp(concat(r_i - r_j, gap, stagger))

Then aggregate: foil1_context_i = sum_j K(i,j) * p_j_hat_prev

Where p_j_hat_prev is the SRF head's current pressure prediction for fore-foil node j (from the main trunk, before aft-foil SRF correction). This gives each aft-foil node a summary of "what pressure distribution is incoming from the fore-foil."

**Why it might help p_tan:** The aft-foil surface pressure is governed by two contributions: its own geometry in freestream, and the perturbation from the fore-foil wake. The current model must infer the fore-foil wake effect from (gap, stagger, AoA) scalars alone — there is no direct path for the aft-foil's SRF head to "see" the fore-foil's predicted pressure distribution. This boundary-integral aggregation creates a direct fore-to-aft information channel, inspired by the Fredholm Integral Equation formulation of BVPs (FIE-NO, 2024): boundary conditions at one surface influence fields at another surface via a Green's function kernel.

**Key distinction from dead-ends:** Fore-aft cross-attention in SRF (#2167, failed) was a full attention mechanism between fore and aft SRF tokens — this failed due to computational overhead and training instability. The boundary-integral approach here is simpler: a fixed-sigma Gaussian kernel with a tiny learned scalar MLP (no attention, no new sequence dimension), applied only at the aft-foil SRF head level as a 1D context vector. The kernel is purely spatial (distance-based), not learned query/key/value.

**Implementation:**

```python
# In AftFoilRefinementHead.forward(), after obtaining fore-foil predicted pressures:
# fore_p_hat = model_out["surface_pred"][fore_foil_mask, 2].detach()  # [N_fore]
# fore_coords = x_raw[fore_foil_mask, 0:2]  # [N_fore, 2]
# aft_coords  = x_raw[aft_foil_mask, 0:2]   # [N_aft, 2]
#
# Pairwise distances: D = ||aft_coords[:, None] - fore_coords[None, :]||^2  # [N_aft, N_fore]
# sigma = 0.5 (in normalized coords; sweep {0.2, 0.5, 1.0})
# K = exp(-D / (2 * sigma^2))  # [N_aft, N_fore]
# K = K / K.sum(dim=1, keepdim=True)  # normalize rows
#
# geo_feat = (aft_coords[:, None] - fore_coords[None, :]).reshape(N_aft*N_fore, 2)
# w = small_mlp(geo_feat).reshape(N_aft, N_fore)  # [N_aft, N_fore]
# K_weighted = K * softmax(w, dim=1)
#
# fore_context = (K_weighted * fore_p_hat[None, :]).sum(dim=1)  # [N_aft]
# Concat fore_context to aft-foil SRF input features.
```

New flags: `--bia_fore_context` (bool), `--bia_sigma 0.5`. The small_mlp is a 2-layer MLP (2 → 16 → 1), ~50 parameters — effectively zero overhead.

**Caution:** The fore-foil pressure predictions used as input are from the main trunk (before SRF correction) — they are imperfect. This creates a noisy signal that the aft-foil SRF must use. Use `.detach()` to prevent gradient flow through the fore-foil predictions into the aft-foil SRF kernel (cleaner optimization).

**Sweep:** Start with 1 seed (42). If promising, ablate: fixed vs. learned kernel weights, detach vs. no-detach, sigma values.

**Confidence:** Moderate. The FIE-NO approach to BVPs (Kovachki et al. 2024) demonstrates that boundary integral representations improve OOD generalization for PDEs by building in the correct causal structure. The simplification to a spatial Gaussian kernel + tiny MLP is conservative enough to avoid the instability that killed fore-aft cross-attention.

---

### 5. stagnation-pe — Stagnation-Point Relative Positional Encoding for Surface Nodes

**Priority: MEDIUM. ~20 LoC. Expected -1 to -3% p_tan. LOW risk.**

**What it is:** For each surface node on the aft-foil, compute its signed arc-length distance from the estimated stagnation point, and inject this as an additional positional feature. The stagnation point is estimated analytically: for a symmetric airfoil at AoA, the stagnation point location along the chord is approximately s_stag ≈ -chord * sin(AoA) * 0.5 (with sign convention: positive = upper surface, negative = lower). For the aft-foil, the effective AoA is perturbed by the fore-foil wake, modeled as AoA_eff = AoA + delta_AoA_wake where delta_AoA_wake = -C_L_fore * chord / (2 * pi * gap) (downwash formula).

**Why it might help p_tan:** The suction peak (lowest pressure, highest gradient) migrates with the stagnation point. The current model receives DSDF features (distance to nearest surface point) but no explicit encoding of how far along the surface a node is from the stagnation point. On the aft-foil, the stagnation point is shifted by the fore-foil's downwash — a wake-dependent shift that gap/stagger/AoA features partially encode but only implicitly. Explicit stagnation-relative coordinates give the model direct knowledge of where the leading-edge suction peak should be, which is the region with the largest pressure errors in tandem cases.

**Key distinction from dead-ends:** Arc-Length Surface Loss Reweighting (PR #2210, in-flight) reweights the LOSS near the stagnation point but does not change the model's INPUT representation. SE(2) AoA-aligned spatial bias (PR #2149, failed) rotated the entire coordinate frame by AoA — a global transformation that changed the representation for ALL nodes. This idea only modifies the input to surface nodes and only encodes distance from the stagnation point, not a rotation.

**Implementation:**

```python
# After the Fourier PE append, for surface nodes only:
# aoa_rad = raw_aoa  # already in radians or convert
# cl_fore = 2 * pi * sin(aoa_rad)
# gap_raw = raw_gap  # chord units
# downwash = -cl_fore / (2 * pi * (gap_raw + 0.1))  # von Karman downwash
# aoa_eff_aft = aoa_rad + downwash  # effective AoA at aft-foil
# s_stag = -0.5 * sin(aoa_eff_aft)  # stagnation point chord fraction, range [-0.5, 0.5]
#
# For each aft-foil surface node with arc-length coordinate s (from DSDF / mesh connectivity):
# s_rel = s - s_stag  # signed distance from stagnation point
# stag_pe = [sin(2*pi*s_rel), cos(2*pi*s_rel), s_rel]  # 3-dim encoding
# Append stag_pe to aft-foil surface node features; zeros for all other nodes.
```

New flags: `--stagnation_pe` (bool). Input dim increases by 3 (all-zeros for non-aft-foil nodes). Requires access to arc-length coordinate (s) — which can be approximated from DSDF features or precomputed from mesh coordinates.

**Implementation challenge:** Getting the arc-length coordinate s for each aft-foil node requires ordered surface node connectivity. The DSDF2 features (indices 6:8 in the 24-dim input) give the gradient direction but not arc-length position. One approach: use the angle of the DSDF2 gradient vector as a proxy for arc-length position (valid for convex airfoils). This avoids needing mesh connectivity explicitly.

**Confidence:** Moderate-low. The aerodynamic motivation is strong (stagnation-relative coordinates are the natural frame for leading-edge pressure). The implementation challenge (clean arc-length proxy) introduces risk. If DSDF gradient angle is a good arc-length proxy, the implementation simplifies significantly.

---

### 6. ema-dual-rate — Dual-Rate EMA: Fast EMA for Loss, Slow EMA for Evaluation

**Priority: MEDIUM. ~8 LoC. Expected -0.5 to -1.5% uniformly. LOW risk.**

**What it is:** Run two EMA averaging streams simultaneously: a fast EMA (decay=0.99, updated every epoch) and a slow EMA (decay=0.9999, also updated every epoch). At validation time, evaluate BOTH and report the better of the two. The fast EMA tracks recent weight changes more closely (useful early in training), while the slow EMA smooths out more of the noise (useful near convergence). The final checkpoint uses the slow EMA for submission.

**Why it might help:** The current EMA uses a fixed decay=0.999 (set at epoch 140). This is a single hyperparameter that affects the entire training trajectory. For p_tan specifically — where the model makes slow, irregular improvements — a slow EMA may continue improving the tandem metric even when the fast EMA has plateaued. The idea is analogous to Polyak-Ruppert averaging with two different lookback windows. The mechanism is well-studied: slower EMA converges to a lower-variance estimate of the final parameter distribution at the cost of slower adaptation.

**Key distinction from dead-ends:** SWAD (#2019, failed as snapshot ensemble) averaged widely-separated checkpoints. This approach averages WITHIN a single gradient trajectory at two timescales — the slow EMA is essentially a flatter basin estimate of the same converged solution. EMA-based ensembling was NOT tried in this specific form.

**Implementation:**

```python
# In Trainer.__init__ or after ema_model is constructed:
if cfg.dual_rate_ema:
    slow_ema_model = copy.deepcopy(model)
    slow_ema_decay = cfg.slow_ema_decay  # default: 0.9999

# In the training loop, alongside the existing EMA update:
if cfg.dual_rate_ema and epoch >= cfg.ema_start_epoch:
    for sp, p in zip(slow_ema_model.parameters(), model.parameters()):
        sp.data.mul_(slow_ema_decay).add_(p.data * (1.0 - slow_ema_decay))

# At validation time:
# val_loss_fast = evaluate(ema_model)   # existing
# val_loss_slow = evaluate(slow_ema_model)  # new
# best_val_loss = min(val_loss_fast, val_loss_slow)
# Report both to W&B.
```

New flags: `--dual_rate_ema` (bool), `--slow_ema_decay 0.9999`. Memory overhead: one extra model copy (~same size as current EMA model, ~0.5GB for our model size — well within 96GB).

**Sweep:** Try decay values {0.9995, 0.9999, 0.99995} against the current decay=0.999 fast EMA.

**Confidence:** Moderate-strong. Two-timescale EMA is theoretically motivated (Polyak & Juditsky, 1992). The simplicity of the change and zero-risk profile (the fast EMA path is unchanged) make this a low-downside experiment. The question is whether the training dynamics near convergence are different enough across timescales to matter — the 2.3x p_tan/p_in gap suggests the tandem regime has high variability that slower averaging might help.

---

### 7. slice-temperature-anneal — Learnable Per-Slice Temperature with Cosine Annealing

**Priority: MEDIUM. ~10 LoC. Expected -1 to -3% p_tan. MEDIUM risk.**

**What it is:** The Physics-Attention in Transolver assigns points to slices via softmax over a learned affinity. Currently the temperature tau (sharpness) is fixed at 1.0 (implicitly) or a single learnable scalar. Replace this with per-block, per-head learnable temperatures that are initialized high (tau=2.0, soft assignments) and annealed toward lower temperature (tau=0.5, sharper assignments) following the cosine schedule. The motivation: early training benefits from soft assignments to allow gradient flow across slices; later training benefits from sharper assignments that specialize slices for specific physical states.

**Why it might help p_tan:** The tandem-foil case has a qualitatively different physical state space than single-foil: the inter-foil wake coupling creates a third "regime" (aft-foil-in-wake) that may not be well-served by the default tau. The gap_stagger_spatial_bias (PR #2130, merged) showed that conditioning slice routing on (gap, stagger) improved p_tan by -3.0%. Annealing the slice temperature is orthogonal to this — it controls not WHAT inputs the routing attends to, but HOW SHARPLY it commits to slice assignments.

**Key distinction from dead-ends:** Spectral conditioning of attention (SCA, PR #2199, in-flight) modifies how frequency-domain features condition attention. Temperature annealing operates on the softmax sharpness directly — a different axis of the Physics-Attention mechanism.

**Implementation:**

The Physics-Attention softmax in Transolver is: A_ij = softmax(q_i * k_j / tau). The current tau is fixed at 1.0 (standard dot-product attention). To make it learnable per block:

```python
# In TransolverBlock.__init__:
if cfg.slice_temp_anneal:
    self.slice_temp = nn.Parameter(torch.tensor(cfg.slice_temp_init))  # default 2.0

# In TransolverBlock.forward, before the Physics-Attention softmax:
if cfg.slice_temp_anneal:
    # Cosine anneal: at epoch t, tau = tau_final + 0.5*(tau_init-tau_final)*(1+cos(pi*t/T))
    current_tau = softplus(self.slice_temp)  # ensure positive
    attn_logits = attn_logits / current_tau

# Alternatively: pass epoch as a forward argument and compute tau externally.
```

New flags: `--slice_temp_anneal` (bool), `--slice_temp_init 2.0`, `--slice_temp_final 0.5`.

**Implementation challenge:** The epoch needs to be accessible inside the forward pass (or a separate scheduler call sets self.current_tau). The cleanest approach: add a `model.set_slice_temp(tau)` method called from the training loop at each epoch.

**Confidence:** Moderate. Temperature annealing in attention/clustering is well-studied (VQ-VAE training, soft-to-hard EM algorithms for GMMs). The novel element is applying it to the Physics-Attention slice routing specifically for OOD tandem regime. The risk is that SHARP assignments early in training hurt single-foil performance (the model can't route single-foil nodes correctly with a shared tau).

---

## Summary Table

| Rank | Slug | Code size | Risk | Expected p_tan gain | Priority |
|------|------|-----------|------|--------------------:|----------|
| 1 | analytical-cp-delta | ~30 LoC | LOW-MEDIUM | -2 to -5% | HIGH |
| 2 | wake-deficit-feature | ~25 LoC | LOW | -2 to -4% | HIGH |
| 3 | tandem-curriculum-ramp | ~15 LoC | LOW | -1 to -3% | MEDIUM-HIGH |
| 4 | boundary-integral-aggregation | ~40 LoC | MEDIUM | -2 to -5% | MEDIUM-HIGH |
| 5 | stagnation-pe | ~20 LoC | LOW | -1 to -3% | MEDIUM |
| 6 | ema-dual-rate | ~8 LoC | LOW | -0.5 to -1.5% | MEDIUM |
| 7 | slice-temperature-anneal | ~10 LoC | MEDIUM | -1 to -3% | MEDIUM |

**Recommended next assignments:**
- Next idle student A: `analytical-cp-delta` — targets the precise mechanism (suction peak delta on aft-foil), directly leverages existing SRF head infrastructure.
- Next idle student B: `wake-deficit-feature` — purely additive input feature, zero architectural risk, strong aerodynamic motivation.
- Next idle student C: `ema-dual-rate` — trivially small change, can be stacked with any other experiment as a free win if validated.
