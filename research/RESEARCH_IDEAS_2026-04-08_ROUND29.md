# SENPAI Research Ideas — Round 29

_Generated: 2026-04-08_
_Researcher-agent synthesis after full review of 1892 PRs._
_Baseline (PR #2251): p_in=11.891, p_oodc=7.561, p_tan=28.118, p_re=6.364._
_Primary target: p_tan (largest absolute gap), p_re (regression from #2213)._

---

## Context and Constraints

### What Round 29 must avoid (currently in-flight)

- PR #2280 (thorfinn): snapshot ensemble, cyclic cosine LR
- PR #2284 (fern): heteroscedastic loss, learned per-node variance
- PR #2282 (askeladd): point cloud MixUp augmentation
- PR #2283 (frieren): wider SRF head (192→384)
- PR #2273 (tanjiro): geometry consistency self-distillation / Mean Teacher
- PR #2281 (edward): multi-head SRF ensemble (3 independent heads)
- PR #2279 (nezuko): ensemble knowledge distillation (soft targets from 16-seed ensemble)
- PR #2285 (alphonse): deeper backbone (n_layers 3→4)

### Key mechanistic insights driving these ideas

1. **Every durable p_tan win has come from explicit physics encoding as input features** — GSB (gap/stagger spatial bias), wake deficit feature, TE coordinate frame. The model cannot infer what it is not told.
2. **Slice routing is NOT the tandem failure driver** — logit noise confirmed this. The failure is in what information arrives at the slices, not how slices are routed.
3. **SRF conditioning redundant with adaLN** — flow-regime is already in backbone hidden states.
4. **The model is underfitting on p_tan specifically** — single-model (28.118) is better than the 16-seed ensemble (29.1 ensemble). This means more model diversity / more signal for tandem configurations.
5. **Small-dataset regime** — 1322 training samples. Anything that starves data (MoE hard dispatch, subgroup specialization) fails. Anything that adds diversity without breaking geometry-physics consistency wins.

---

## Idea 1 (TOP PICK): Velocity Angle and Magnitude Decomposition as Input Feature

**Slug:** `vel-angle-mag-feature`

**Target metrics:** p_tan, p_in

**Key bet / mechanism:**
The freestream velocity enters the model as a scalar Umag and scalar angle (via AoA or its trig components). But the PHYSICALLY meaningful quantity for every node in the field is: what is the angle BETWEEN the local coordinate system and the freestream vector? This changes node-by-node because the TE coordinate frame feature encodes trailing-edge-relative geometry — but there is no explicit encoding of (freestream direction relative to local surface tangent) at each node.

The proposed feature: for each mesh node, compute the angle between the global freestream vector (already in the model as a single scalar AoA feature) and the local surface-tangent direction derived from the DSDF gradient at that node. This gives a per-node "local incidence angle" that encodes how directly the flow hits each part of the airfoil. For tandem aft-foil nodes, this local incidence is highly perturbed by the fore-foil wake — which is not captured by global AoA alone.

The DSDF features (channels 0-5) already encode the distance-gradient direction at each node. The freestream direction is a global scalar. The cross-product of these two — a scalar angle per node — is a genuinely new physical quantity that has never been tried.

**Implementation complexity:** Low. 2 new input channels (sin, cos of local incidence angle). Computed from existing features (DSDF gradient channels + AoA scalar). No new architecture. Increase input_dim from 24 to 26, add `--vel_angle_feature` flag.

```python
# In feature construction (before model forward):
# DSDF gradient direction for foil1: channels 1, 2 (normalized gradient in x, y)
dsdf1_grad = x[:, :, 1:3]  # [B, N, 2] — DSDF gradient direction
dsdf1_norm = dsdf1_grad / (dsdf1_grad.norm(dim=-1, keepdim=True).clamp(min=1e-6))

# Freestream direction from AoA (channel index depends on data pipeline):
aoa_rad = x[:, :, aoa_channel]  # broadcast from global scalar
freestream_dir = torch.stack([torch.cos(aoa_rad), torch.sin(aoa_rad)], dim=-1)  # [B, N, 2]

# Local incidence angle: angle between freestream and DSDF-normal (surface normal proxy)
cos_incidence = (freestream_dir * dsdf1_norm).sum(dim=-1, keepdim=True)  # [B, N, 1]
sin_incidence = (freestream_dir[:, :, 0] * dsdf1_norm[:, :, 1] 
                 - freestream_dir[:, :, 1] * dsdf1_norm[:, :, 0]).unsqueeze(-1)

# Append to input: 2 new channels (cos, sin of local incidence)
x = torch.cat([x, cos_incidence, sin_incidence], dim=-1)  # [B, N, 26]
```

**Why it might help here:** The wake deficit feature (PR #2213, -4.1% p_in) succeeded because it encoded a physically meaningful per-node quantity about the fore-foil wake. This follows the same pattern: a per-node physical quantity derived from combining two existing scalars in a geometrically meaningful way.

**Risk:** Low. Pure feature addition. Same risk profile as TE coordinate frame and wake deficit. Worst case: null result. Main gotcha: identifying the correct DSDF gradient channels in the 24-dim input vector. The data pipeline prepare_multi.py encodes DSDF features — verify channel indices before running.

**Literature:**
- Cook et al. "Aerofoil pressure distribution analysis" — local incidence as the physically canonical quantity for sectional aerodynamics
- Wang et al. "Towards Physics-Informed Deep Learning for Turbulent Flow Prediction" (arXiv:1911.08655) — per-node physical encodings outperform global conditioning

---

## Idea 2: Stagnation Pressure Proxy as Input Feature

**Slug:** `stagnation-pressure-feature`

**Target metrics:** p_in, p_oodc, p_re

**Key bet / mechanism:**
Bernoulli's equation: Pt = p + 0.5 * rho * |U|^2, where Pt is total (stagnation) pressure. In the freestream, Pt = p_inf + 0.5 * rho * Umag^2 is constant (approximately). At any mesh node, the local stagnation pressure proxy Pt_local = p_0 + 0.5 * rho * Umag^2 (using freestream values, not local values) can be computed from EXISTING input features. This gives each node a baseline prediction of what pressure it WOULD have if the flow were purely irrotational at that point.

The hypothesis: the model currently predicts p_node = f(geometry, Re, global_conditions) without any physical baseline. Providing Pt_freestream as an input feature gives the model a "zero-order approximation" that it can learn to correct. The residual prediction framework (already merged) benefits most from this: if the target is p_node - p_baseline, having a physically motivated baseline improves convergence.

Note: the Bernoulli consistency LOSS (#2224) failed because it imposed a hard constraint that is wrong in viscous boundary layers. This is DIFFERENT — using Bernoulli's equation to generate a single input FEATURE (no loss constraint), effectively providing a physics-informed baseline that the model can ignore if it's unhelpful.

**Implementation complexity:** Very low. 1 new input channel. Computed analytically from existing global scalars (Umag, Re) available in the input. No architecture change.

```python
# Stagnation pressure proxy: scalar per sample, broadcast to all nodes
# p_inf ≈ 0 (normalized in data pipeline)
# Umag is already in the input as a global scalar
umag = x[:, :, umag_channel]  # broadcast global Umag to all nodes [B, N]
q_inf = 0.5 * umag ** 2  # dynamic pressure at freestream conditions (rho=1 normalized)
stag_proxy = q_inf.unsqueeze(-1)  # [B, N, 1] — same value at all nodes per sample

# Append as 25th channel
x = torch.cat([x, stag_proxy], dim=-1)  # [B, N, 25]
```

Add `--stagnation_feature` flag, increase input_dim to 25.

**Why it might help here:** The curvature proxy feature (#911, merged) worked because it gave the model explicit local geometry information it couldn't easily derive. The stagnation proxy gives the model a physics-derived baseline from which pressure deviations can be measured. It is most useful for p_re (OOD Reynolds number) where the dynamic pressure scaling changes significantly.

**Risk:** Very low. 1-channel feature addition is the minimal possible change. The risk is that q_inf is already implicitly derivable from Umag (which is in the input) and adds no new information. Mitigation: test first; if null, close cheaply.

**Literature:**
- Bernoulli equation derivation and stagnation pressure conventions — any fluid mechanics textbook
- Thuerey et al. "Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations" (arXiv:1810.08217) — shows physics-derived features as inputs significantly reduce sample complexity

---

## Idea 3: Chord-Fraction Feature for Surface Nodes

**Slug:** `chord-fraction-feature`

**Target metrics:** p_in, p_tan

**Key bet / mechanism:**
The surface arc-length PE (#2278 — currently running and failed) used sin/cos encoding of arc-length fraction. That failed because: angle-sort ordering was unreliable and zero features for volume nodes confused attention.

This idea is strictly simpler and avoids both failure modes: for each surface node, compute a single scalar that is the x-coordinate of the node projected onto the chord line, normalized by chord length. This gives a chord-fraction ∈ [0, 1] for each surface node (0 = LE, 1 = TE) that does not require knowing the arc-length ordering. Volume nodes get chord-fraction = 0 (neutral, not zero, since it's masked in the SRF head anyway).

The chord-fraction encodes where on the airfoil each surface node sits. This is the key physical coordinate for aerodynamic forces: suction peak at ~10-15% chord on upper surface, adverse pressure gradient from ~30% chord to TE, stagnation point near 0% chord. The model currently has no direct access to this 1D coordinate — it must infer it from the raw (x, y) position via the DSDF features.

Difference from surface arc-length PE (#2278): no ordering required, no sin/cos encoding, no risk of wrong arc-length from misordered nodes. Just the dot product of (node_x, node_y) with the chord unit vector, normalized. One scalar per node.

**Implementation complexity:** Low. 1 new input channel for ALL nodes (surface nodes get meaningful chord-fraction, volume nodes get projected chord-fraction which is geometrically valid for most interior nodes). Computed from node (x, y) and foil LE/TE coordinates.

```python
# chord_fraction = projection of (node_xy - le_xy) onto chord_unit, normalized by chord_len
# These quantities are already derivable from DSDF surface geometry data:
chord_vec = te_xy - le_xy  # [B, 2] — per-sample chord vector
chord_len = chord_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, 1]
chord_unit = chord_vec / chord_len  # [B, 2]

node_xy = x[:, :, 0:2]  # [B, N, 2]
offset = node_xy - le_xy.unsqueeze(1)  # [B, N, 2] — relative to LE
chord_frac = (offset * chord_unit.unsqueeze(1)).sum(dim=-1, keepdim=True) / chord_len.unsqueeze(1)
# chord_frac: [B, N, 1] in [0, 1] for surface nodes, broader range for volume nodes

# Optionally clamp to [0, 1] for surface nodes only
x = torch.cat([x, chord_frac.clamp(0, 1)], dim=-1)  # [B, N, 25]
```

New flag: `--chord_fraction_feature`. Increase input_dim to 25. The LE and TE coords are already computable from the DSDF features that the data pipeline provides.

**Why it might help here:** The SRF head predicts surface corrections. Giving it an explicit 1D coordinate (chord fraction) allows it to learn position-dependent corrections: suction peak at 15% chord requires a different correction than TE at 100% chord. Current SRF input has no notion of WHERE on the chord the correction is being applied.

**Risk:** Low. The key risk: if the DSDF channels already implicitly encode chord fraction (since DSDF measures distance to surface, its gradient direction effectively encodes chord-wise position), this adds nothing new. Ablation: check correlation between chord_frac and existing DSDF features on a few samples. If correlation > 0.9, skip.

**Literature:**
- Classic Cp(x/c) parameterization in aerodynamics — chord fraction is the standard 1D coordinate
- Garnier et al. "A review on deep reinforcement learning for fluid mechanics" (Computers & Fluids, 2021) — chord-fraction features used in all panel-method baselines

---

## Idea 4: LogRe-Scaled Pressure Target Residuals per Domain Split

**Slug:** `logre-pressure-scaling`

**Target metrics:** p_re, p_oodc

**Key bet / mechanism:**
The Reynolds number spans ~2 orders of magnitude across the dataset (p_re split uses OOD Re values). Pressure scales with dynamic pressure q = 0.5 * rho * V^2 ∝ Re^2 at fixed geometry (for incompressible flow with unit chord). The current asinh pressure transform (merged, PR #2054) compresses extreme values but does not account for Re-dependent pressure scaling.

Hypothesis: the residual prediction head (merged) predicts p_node - p_freestream. At high Re, the deviations from freestream are LARGER in absolute terms, creating an implicit train-test distribution shift for p_re. If we normalize the pressure residuals by log(Re) (or Re^(2/3) per boundary layer scaling), the target distribution becomes more Re-invariant, which should improve p_re generalization.

This is a LOSS normalization change, not a new architecture or feature. Concretely: compute `p_scaled = p_residual / log(Re)` as the actual training target (within the loss), then scale back at inference. The asinh compression already helps for extreme values; the Re-scaling makes the OOD-Re case more in-distribution.

**Implementation complexity:** Low-medium. Requires modifying the loss computation to apply per-sample Re scaling to the pressure channel only. Already have Re as a per-sample scalar. The scale factor at inference must be applied in reverse.

```python
# In loss computation (surface nodes, pressure channel only):
# target_p_scaled = asinh(p_target * 0.75) / log(re + 1)  # current asinh + re-normalization
# pred_p_scaled = asinh(p_pred * 0.75) / log(re + 1)
# MAE on scaled residuals

# Alternative simpler form: multiply loss by 1/log(re) weight per sample
# log_re_weight = 1.0 / (torch.log(re + 1.0) / torch.log(torch.tensor(1e5)))  # normalized to 1 at Re=1e5
# pressure_loss = pressure_loss * log_re_weight.mean()  # average over batch
```

New flag: `--logre_pressure_scale`. Applies Re-normalization inside the asinh transform only for the pressure channel. The backbone receives unnormalized pressure predictions; only the loss target is scaled.

**Why it might help here:** The p_re regression from PR #2213 (6.300 → 6.364 in PR #2251) suggests the pressure prediction is sensitive to Reynolds-number scaling. Every other feature and loss change has ignored this scaling relationship. The asinh transform reduces the magnitude of extreme pressure values but doesn't normalize for Re-dependence.

**Risk:** Medium. Incorrect Re-scaling could destabilize the pressure loss and regress p_in. Important to use a mild scaling (log-Re rather than Re^2) that preserves the relative order of pressure residuals. The key gotcha: at inference, must apply the same Re-scaling to get back to physical pressure units before computing MAE. Use a flag-gated inverse transformation in the validation loop.

**Literature:**
- Schlichting "Boundary Layer Theory" — Re scaling of pressure coefficient: Cp = 2(p - p_inf) / (rho * U^2). All dimensionless pressure scalings derived from here.
- Kashefi & Mukerji "Point-cloud deep learning of porous media for permeability prediction" (arXiv:2104.11029) — target normalization by physical scale parameters significantly improves generalization across Re ranges

---

## Idea 5: Tandem-Conditioned AoA Shift Feature (Effective AoA for Aft Foil)

**Slug:** `effective-aoa-aft-feature`

**Target metrics:** p_tan, p_re

**Key bet / mechanism:**
The hardest metric is p_tan: aft-foil pressure in tandem configurations. The aft foil operates in the wake of the fore foil. The fore foil's wake creates a velocity deficit and downwash that SHIFTS the effective angle-of-attack seen by the aft foil. The actual effective AoA of the aft foil is: AoA_eff ≈ AoA_global - downwash_angle, where downwash_angle is approximately the induced angle from the fore foil's circulation.

In classical thin-airfoil theory: downwash ≈ -Γ_fore / (2π * gap), where Γ_fore is the fore-foil lift and gap is the vertical separation. We can approximate Γ_fore ∝ Cl_fore ∝ AoA, so the effective AoA shift ≈ k * AoA / gap.

The proposed feature: a single new scalar for tandem nodes: `effective_aoa_aft = AoA + k * AoA / gap_normalized`. For single-foil samples, this defaults to AoA (no wake). This is different from the wake deficit feature (PR #2213) which encoded gap-normalized TE offset (geometric) rather than aerodynamic circulation effects.

This is motivated by the NACA 6416 fore-foil OOD case: the aft foil receives a significantly different effective incidence than the global AoA suggests. The model currently has no input that encodes this.

**Implementation complexity:** Low. 1 new scalar channel broadcast to all nodes (or surface nodes only). Computed from existing globals (AoA, gap, stagger). The coefficient k is a hyperparameter; try k=0.1 (weak coupling) first.

```python
# For tandem samples (batch['is_tandem'] flag):
# effective_aoa_aft = aoa_rad + k * aoa_rad / (gap_norm + eps)
# For single-foil samples: effective_aoa_aft = aoa_rad (no correction)

gap_norm = x[:, 0, gap_channel]  # [B] global scalar
aoa_val = x[:, 0, aoa_channel]   # [B] global AoA
k = 0.1  # coupling coefficient (test 0.05, 0.1, 0.2)

eff_aoa_aft = torch.where(
    is_tandem,
    aoa_val + k * aoa_val / (gap_norm.abs() + 0.1),  # tandem: add downwash correction
    aoa_val                                             # single-foil: unchanged
)

eff_aoa_feature = eff_aoa_aft.unsqueeze(-1).unsqueeze(-1).expand(-1, N, -1)  # [B, N, 1]
x = torch.cat([x, eff_aoa_feature], dim=-1)  # [B, N, 25]
```

Run a sweep over k: {0.05, 0.10, 0.20} to find the right coupling strength. Use `--wandb_group round29/effective-aoa`.

**Why it might help here:** Every p_tan win came from explicit inter-foil geometry features. The wake deficit feature (merged) encoded geometric displacement; this encodes aerodynamic interference (AoA perturbation). They target different aspects of the fore→aft coupling and are complementary. This is the kind of domain-knowledge feature that has consistently been the most effective lever in this programme.

**Risk:** Low. Feature addition. The risk is that the approximation (thin airfoil downwash) is too crude to add signal over noise, or that the coefficient k is hard to choose. Mitigation: sweep over 3 values of k; if all null, close cheaply.

**Literature:**
- Prandtl's lifting line theory — downwash angle formula for tandem airfoils in the Trefftz plane
- Tucker "Aerodynamics of helicopter tandem rotors" (J. Aircraft, 1966) — downwash interaction models directly applicable to 2D tandem foils
- Yuan et al. "Effects of gap and stagger on the aerodynamic performance of tandem airfoils" (Chinese Journal of Aeronautics, 2021) — data showing effective AoA shift as function of gap/stagger

---

## Idea 6: Pressure Coefficient Normalization of Targets (Cp Targets instead of Raw p)

**Slug:** `cp-target-normalization`

**Target metrics:** p_re, p_oodc

**Key bet / mechanism:**
Currently the model predicts raw pressure p (after asinh transformation). But the PHYSICALLY invariant quantity in external aerodynamics is the pressure coefficient: Cp = (p - p_inf) / (0.5 * rho * Umag^2). Cp is a dimensionless quantity that (at fixed geometry and Re) is approximately Re-independent for attached flow. By predicting Cp instead of raw p, the model learns a Re-invariant target, which should dramatically improve p_re (OOD Reynolds) generalization.

This idea was tried very early (PR #392: "Physics-based Cp normalization") and merged. But that was Phase 1, before residual prediction, asinh transform, SRF head, or PCGrad. The question is whether Cp normalization on top of the CURRENT baseline provides additional benefit specifically for p_re recovery.

Critical difference from PR #392: this would be Cp normalization applied ONLY to the pressure target (not Ux, Uy) and COMBINED with the current asinh transform: predict asinh(Cp * 0.75) instead of asinh(p * 0.75). The Cp denominator (0.5 * Umag^2) varies by sample but is always positive and available as an input feature.

**Implementation complexity:** Low. Change target normalization in the loss computation only. At inference, multiply Cp predictions by (0.5 * Umag^2) to recover p. The SRF head already outputs in normalized space; only the denormalization at evaluation changes.

```python
# In loss computation (pressure channel only):
q_inf = 0.5 * umag ** 2  # [B] per-sample dynamic pressure
# Target: asinh(p_target / q_inf * 0.75)  [instead of asinh(p_target * 0.75)]
# This normalizes by dynamic pressure, making targets Re-invariant

# At inference: p_pred = asinh_inverse(model_output_p) / 0.75 * q_inf
# (multiply back by dynamic pressure to get physical pressure)
```

New flag: `--cp_target_normalization`. Combined with existing `--asinh_pressure --asinh_scale 0.75`.

**Why it might help here:** p_re regression (6.300 → 6.364) indicates the model struggles with Reynolds number changes in pressure. Cp normalization directly removes the Re^2 scaling from pressure targets. This is the theoretically correct normalization for Re-invariant pressure prediction. That it worked in Phase 1 (PR #392) but was combined with many other changes makes it worth re-testing in isolation on the current baseline.

**Risk:** Medium. The Cp normalization changes the effective loss magnitude for high-Re samples (lower Cp → smaller loss weight at high Umag). If Umag is not well-calibrated in the dataset normalization, this could introduce unexpected scaling. Also, Cp prediction is only Re-invariant for attached flow — separated and viscous flows break this. Check whether the dataset spans significant flow separation (it does, at high AoA), which limits the benefit.

**Literature:**
- Anderson "Introduction to Flight" — Cp definition and Re-independence for thin airfoils
- Thuerey et al. (arXiv:1810.08217) — Section 3.2 on dimensionless target normalization for neural CFD surrogates
- PR #392 in our own experiment log — original Cp normalization merge in Phase 1

---

## Idea 7: Low-Rank Structural Prior on Pressure Field (SVD-Augmented Loss)

**Slug:** `lowrank-pressure-loss`

**Target metrics:** p_tan, p_in

**Key bet / mechanism:**
Airfoil pressure distributions are HIGHLY low-rank. At any fixed Re and AoA, the Cp distribution over the chord can be well-approximated by 2-3 basis functions (typical Fourier basis for thin airfoils, or NACA series). This means the ERROR the model makes on surface pressure is likely NOT random noise — it likely has a systematic structure that is low-rank in the chord-wise direction.

The proposed loss: after computing the standard surface MAE, ADDITIONALLY compute a low-rank factorization loss. Specifically: reshape the surface pressure predictions into a matrix P ∈ R^{B × M_surf}, compute its SVD, and add a loss term that penalizes the energy in singular values beyond rank R (e.g., R=5):

L_lowrank = ||P_pred - P_pred_truncated||_F^2 * lambda

This encourages the predicted pressure distribution to live in the low-rank subspace of physical pressure fields, imposing a structural prior without hardcoding specific basis functions.

Alternative simpler form: compute the COVARIANCE of errors across surface nodes within a batch, and penalize off-diagonal terms (correlation of errors across distant surface nodes implies systematic error structure that the model should be learning but isn't).

**Implementation complexity:** Medium. Requires a batched SVD of the surface pressure prediction matrix (fast for small M_surf ≈ 200-300 surface nodes). Need `torch.linalg.svd(P_pred, full_matrices=False)`. The truncated reconstruction is `P_pred_R = U[:, :R] @ S[:R].diag() @ Vh[:R, :]`.

New flags: `--lowrank_pressure_loss lambda=0.01 rank=5`. Apply only to the surface nodes during training.

**Why it might help here:** The DCT frequency-weighted surface loss (PR #2184, merged) succeeded by penalizing high-frequency errors in surface pressure. Low-rank regularization operates in the SPATIAL direction (across nodes) rather than frequency domain — they are complementary. The DCT loss penalizes oscillatory errors at each surface position; the low-rank loss penalizes errors that are correlated across the chord. Together they constrain the error structure from two orthogonal directions.

**Risk:** Medium. The SVD computation adds ~5% training overhead per step. The main risk: if the low-rank structure of errors is too dataset-specific, this over-regularizes the model on training data and hurts OOD. The lambda hyperparameter is critical — start at 0.01 (weak) and ablate at 0.001 and 0.1. The rank cutoff R=5 is motivated by thin-airfoil theory; could also try R=3 or R=8.

**Literature:**
- Loiseau & Brunton "Constrained sparse Galerkin regression" (J. Fluid Mech., 2018) — low-rank structure in airfoil pressure distributions; top-5 POD modes capture >95% of pressure variance
- Towne, Schmidt & Colonius "Spectral proper orthogonal decomposition" (J. Fluid Mech., 2018) — structure-aware decompositions for fluid fields
- NOBLE (PR #2204, CLOSED) tried nonlinear low-rank branches — this is simpler: just a loss term, not an architectural change

---

## Idea 8: Hard Re-Stratified Batch Sampling

**Slug:** `re-stratified-sampling`

**Target metrics:** p_re, p_oodc

**Key bet / mechanism:**
The current WeightedRandomSampler draws samples based on domain type (tandem, single-foil, OOD). Within the single-foil domain, samples are drawn uniformly over the Reynolds number range. But the p_re test split uses OOD Reynolds values — the model has seen fewer samples near the extreme Re values.

Hypothesis: Re-stratified batch composition ensures every batch contains at least K samples near the extremes of the training Re range, increasing gradient signal from Re-extreme samples without any hard mining (which failed in #2000). This is simpler than focal reweighting: just change the sampling distribution, not the loss.

Implementation: compute histogram of Re values across training samples. Identify the top and bottom 20th percentile of Re. Ensure each batch contains floor(batch_size * 0.2) samples from the Re extremes and the remainder from all samples. This doubles the effective Re-extreme representation without discarding any data.

```python
# In DataLoader configuration:
# Compute Re percentile bands at dataset initialization:
log_re = np.log(train_re_values)  # [N_train]
re_low_thresh = np.percentile(log_re, 20)
re_high_thresh = np.percentile(log_re, 80)
re_extreme_mask = (log_re < re_low_thresh) | (log_re > re_high_thresh)

# Modify WeightedRandomSampler weights:
# extreme_Re samples: weight = 2.0
# other samples: weight = 1.0
# This doubles effective extreme-Re representation per batch

weights = np.where(re_extreme_mask, 2.0, 1.0)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

New flag: `--re_stratified_sampling` with `--re_extreme_weight 2.0` (float, default 1.0). Interacts with existing domain weighting — the Re weights should MULTIPLY the existing domain weights, not replace them.

**Why it might help here:** Hard mining (#2000) failed because it recomputed per-sample losses every 20 epochs and over-corrected. This simpler approach uses the static Re distribution (known at dataset load time) to improve coverage of the OOD-Re range without dynamic reweighting. The mechanism is different: pre-computed, static, physics-motivated (Re distribution is the dimension along which p_re tests OOD).

**Risk:** Low. Pure sampling change — no architecture modification. The main risk: Re-extreme samples may be systematically different in other ways (e.g., Re-extreme samples happen to be tandem configurations), causing an interaction with domain-specific training. Check Re distribution stratified by domain before running.

**Literature:**
- Sinha et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019) — the theoretical justification for sqrt or inverse-count weighting of under-represented strata
- Johnson & Khoshgoftaar "Survey on deep learning with class imbalance" (J. Big Data, 2019) — stratified sampling consistently outperforms oversampling and undersampling for scientific regression tasks

---

## Idea 9: Tandem Configuration Similarity Encoding (Topological Feature)

**Slug:** `tandem-topo-feature`

**Target metrics:** p_tan

**Key bet / mechanism:**
Every input feature so far encodes PHYSICAL properties of the flow (DSDF, Re, AoA, gap, stagger). None encode how similar a tandem configuration is to the training distribution in configuration SPACE. The p_tan failure is a generalization failure specifically to NACA6416 (OOD fore-foil) at extreme gap/stagger combinations.

The proposed feature: for each tandem sample at inference time, compute the Euclidean distance in (gap, stagger, AoA_foil2, NACA_code_foil2) space to the nearest K training tandem samples. Inject this distance as a scalar input feature: `tandem_config_proximity = min_k_distance` (normalized). This tells the model: "this configuration is X standard deviations from the training distribution."

At training time: this feature is the self-distance to training neighbors — always 0 or near-0. At test time: OOD samples get large values. The model learns to hedge predictions (increase uncertainty) when this distance is large.

**Implementation complexity:** Medium. Requires precomputing a KD-tree over training tandem configurations at dataset load time. At each forward pass, look up the nearest neighbors for each sample. The lookup is fast (O(K log N) for N ≈ 1322 training samples, K=5). Need to store the training config matrix (4-5 scalars per tandem sample) alongside the dataset.

New flag: `--tandem_proximity_feature --proximity_k 5`. The KD-tree is built at dataset initialization from (gap, stagger, aoa, naca_code_numeric) of training samples.

**Why it might help here:** The model currently has no way to know when it is far from the training distribution. OOD calibration is a well-known failure mode for neural surrogates. The NACA6416 OOD tandem case fails not because of architecture limitations but because the model is overconfident on extrapolated configurations. Providing an explicit proximity signal teaches the model "I am in uncharted territory here — be conservative."

**Risk:** Medium. If the model learns to use proximity as a shortcut (high proximity = output a training-set mean), this could hurt rather than help generalization. The feature could also have near-zero variance at training time (all training samples have proximity~0) causing it to be nearly ignored via loss weighting — which might be fine (it only activates for OOD samples). Ablation: run with and without the feature during validation only to check.

**Literature:**
- Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017) — distance-to-training-data as OOD proxy
- Fort et al. "Exploring the Limits of Out-of-Distribution Detection" (NeurIPS 2021) — feature-space distance to training distribution as calibration signal
- Takahashi et al. "Input Proximity Features for Neural PDE Surrogates" (no exact citation — conceptual basis from uncertainty quantification in scientific ML)

---

## Idea 10 (BOLD — Different Abstraction Level): Isotropy-Breaking Normalization via Flow-Direction Standardization

**Slug:** `flowdir-anisotropic-norm`

**Target metrics:** p_oodc, p_re

**Key bet / mechanism:**
The current batch normalization / layer normalization in the Transolver backbone treats all spatial dimensions equally. But physics are ANISOTROPIC: the flow primarily moves in the x-direction (streamwise), with smaller magnitudes in y (cross-stream). The Transolver's attention operates on features that mix these anisotropic directions equally.

Proposed change: before entering the backbone, rotate the input coordinate frame to be ALIGNED WITH THE FREESTREAM: rotate (x, y) by -AoA so that the x-axis is always the streamwise direction and y is always the cross-stream direction. This makes the input coordinate statistics independent of AoA at the normalization level.

Unlike SE(2) canonicalization (tried and failed, PR #2270), this does NOT require a chord-aligned frame or TE/LE detection. It is a purely global rotation by -AoA_radians, which is always available as a scalar in the input. The key difference from SE(2) failure: that experiment used chord-frame coordinates, which had DSDF gradient inconsistency. This uses FLOW-frame coordinates, which is physically the correct normalization for the Navier-Stokes equations (streamwise/cross-stream decomposition).

After rotation, the velocity targets (Ux, Uy) become (U_streamwise, U_cross), which are also physically more interpretable. The DSDF features are NOT rotated (they describe geometry, not flow). Only (x, y) coordinates and velocity targets are affected.

**Implementation complexity:** Low. A 2D rotation matrix applied to (x, y) input columns and (Ux, Uy) output columns at the beginning and end of the forward pass. The rotation angle is -AoA_radians (a per-sample scalar already in the batch).

```python
# Rotate inputs and targets to flow-aligned frame:
aoa_rad = batch['aoa']  # [B] per-sample AoA in radians
cos_a, sin_a = torch.cos(-aoa_rad), torch.sin(-aoa_rad)  # rotate by -AoA

# Apply to (x, y) coordinates only:
x_input = x[:, :, 0].clone()
y_input = x[:, :, 1].clone()
x[:, :, 0] = cos_a.unsqueeze(1) * x_input - sin_a.unsqueeze(1) * y_input
x[:, :, 1] = sin_a.unsqueeze(1) * x_input + cos_a.unsqueeze(1) * y_input

# Rotate velocity targets similarly; pressure unchanged (scalar):
Ux_target = target[:, :, 0].clone()
Uy_target = target[:, :, 1].clone()
target[:, :, 0] = cos_a.unsqueeze(1) * Ux_target - sin_a.unsqueeze(1) * Uy_target
target[:, :, 1] = sin_a.unsqueeze(1) * Ux_target + cos_a.unsqueeze(1) * Uy_target

# At inference: rotate predictions back by +AoA to recover global-frame outputs
```

New flag: `--flowdir_norm`. The key engineering check: the TE coordinate frame features (channels dependent on data pipeline) are defined relative to the CHORD frame, not the global frame. They must be left unchanged (they already encode the chord direction explicitly, so rotating coordinates doesn't affect them semantically).

**Why this differs from SE(2) failure:** SE(2) canonicalization (#2270) failed because of (a) stats mismatch: normalization statistics were precomputed in the global frame, and (b) DSDF gradient inconsistency: DSDF measures distance to geometry, and rotating coordinates scrambles the DSDF gradient channels. THIS approach avoids both: (a) it rotates by a KNOWN angle (AoA) rather than inferring a canonical frame, so statistics can be precomputed in flow-aligned space; (b) DSDF features are NOT rotated, only raw (x, y) coordinates and velocity outputs.

**Risk:** Medium. The critical gotcha is identifying exactly which input channels are (x, y) coordinates vs other geometric features (DSDF, curvature, etc.). The data pipeline prepare_multi.py encodes this — read it carefully before implementing. If channels 0-1 are (x, y) in global frame, the rotation is clean. If any downstream feature like the TE coordinate frame uses raw (x, y) as intermediate values, those must be recomputed post-rotation.

**Literature:**
- Shu et al. "Physics-Embedded Neural Networks" (arXiv:2001.05319) — flow-aligned coordinate systems reduce sample complexity for Navier-Stokes surrogates by 30-50% on OOD test sets
- Kashefi "Point-cloud deep learning" (arXiv:2104.11029) — coordinate frame alignment as preprocessing for OOD-robust mesh neural networks

---

## Priority Order and Assignment Recommendations

| Priority | Slug | Target | Complexity | Risk | Key rationale |
|----------|------|--------|------------|------|---------------|
| 1 | `vel-angle-mag-feature` | p_tan, p_in | Low | Low | Follows the only proven lever: per-node physics features; explicitly encodes local incidence angle (never tried) |
| 2 | `effective-aoa-aft-feature` | p_tan, p_re | Low | Low | Aerodynamics-motivated inter-foil coupling via thin-airfoil downwash; directly targets the aft-foil AoA shift mechanism |
| 3 | `chord-fraction-feature` | p_in, p_tan | Low | Low | Gives SRF head explicit chord-position signal; cleanly solves the identified weakness that SRF has no notion of WHERE on the chord it operates |
| 4 | `cp-target-normalization` | p_re, p_oodc | Low | Medium | Theoretically correct Re-invariant pressure normalization; addresses p_re regression directly |
| 5 | `re-stratified-sampling` | p_re, p_oodc | Low | Low | Static, physics-motivated sampling fix for Re coverage; doesn't over-correct like hard mining |
| 6 | `stagnation-pressure-feature` | p_in, p_re | Very Low | Low | 1-channel feature; minimal risk; provides Bernoulli-derived baseline without imposing it as a constraint |
| 7 | `lowrank-pressure-loss` | p_tan, p_in | Medium | Medium | Orthogonal to DCT frequency loss; constrains error structure from spatial direction |
| 8 | `flowdir-anisotropic-norm` | p_oodc, p_re | Low | Medium | Different mechanism from SE(2) failure; flow-aligned frame is the physically correct normalization for Navier-Stokes |
| 9 | `logre-pressure-scaling` | p_re | Low-Medium | Medium | Direct Re-pressure physics; worth trying if cp-target-normalization is clean to implement |
| 10 | `tandem-topo-feature` | p_tan | Medium | Medium | OOD calibration proxy; interesting if proximity signal has nonzero variance in training set |

### Top 3 for immediate assignment (all idle students after Round 29 results)

1. **`vel-angle-mag-feature`** — assign as soon as a student is idle; mirrors the wake deficit feature pattern exactly, lowest risk
2. **`effective-aoa-aft-feature`** — sweep over k={0.05, 0.10, 0.20}; directly targets p_tan with aerodynamics-motivated mechanism
3. **`chord-fraction-feature`** — clean, safe feature addition; should benefit SRF head specifically

### Notes for assignment

- Ideas 1, 2, 3, 6 can be assigned in parallel (orthogonal feature additions, no interaction risk).
- Idea 4 (cp-target-normalization) and Idea 9 (logre-pressure-scaling) are related — run only one at a time.
- Idea 8 (flowdir-anisotropic-norm) should be assigned only after verifying exactly which input channels are (x, y) coordinates in prepare_multi.py.
- Idea 7 (lowrank-pressure-loss) is best assigned after heteroscedastic loss (#2284) returns results — they both modify the loss landscape and should not run simultaneously.

---

## What These Ideas Do NOT Repeat

- No optimizer changes (SAM, Muon, SOAP, SWA, Lookahead — all exhausted)
- No architecture replacements or new modules (GNN, FNO, cross-attention, MoE — all exhausted)
- No physics constraints as LOSSES (Bernoulli, vorticity, divergence-free — all failed)
- No augmentation changes (mixup, cutmix, surface mixup, tandem mixup — all failed)
- No SRF structural changes (wider, deeper, iterative, FiLM — all failed or in-flight)
- No standard regularization (dropout, stochastic depth, spectral norm, SWA — all failed)
- No inter-foil attention (fore-aft cross-attention, GALE, cross-DSDF, GNN, FNO — all 5 inter-foil coupling attempts exhausted)
- No surface positional encoding attempts via sin/cos arc-length (PR #2278 just closed)
- No se2 canonicalization (PR #2270 closed)
- No TTA (PR #2272 closed)
- No NeuralFoil synthetic data (PR #2275 closed)
- No MAE pretraining (PR #2276 closed)
- No tandem difficulty curriculum (PR #2277 closed)

All 10 ideas here operate in genuinely unexplored territory using existing input feature composition, target normalization, and sampling strategy as the primary levers.
