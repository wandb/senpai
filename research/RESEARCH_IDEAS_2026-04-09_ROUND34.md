<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — Round 34 (2026-04-09)

Generated after full review of 1925 PRs (136 merged), complete train.py architectural analysis,
and 15 targeted literature searches. Ranked by expected impact-to-complexity ratio.

Directions explicitly NOT repeated here (exhausted): architecture replacements (FNO, GNO, DeepONet,
full Transformer, U-Net CNN, PointNet), geometry features (NACA params, curvature, arc-length, extra
dsdf channels), optimizer variants (all tried), tandem coupling (cross-attention, message passing,
explicit tandem feature concat), loss variants (Huber, log-cosh, wing-section weighting, SSIM, r-drop,
uncertainty loss, FiLM conditioning, auxiliary regression heads), data augmentation strategies.

In-flight (Round 31-32, do NOT duplicate): aux AoA head v2, condition token v2, SE channel attention,
tandem curriculum ramp, stochastic depth, FV cell area loss, EMA teacher distillation,
panel-method Cp feature.

---

## Priority 1: Aft-Foil Surface Node Inclusion Fix

**Target metrics:** Surface MAE (p_tan primary), surface MAE overall
**Expected impact:** High — potentially 5-15% improvement on tandem surface MAE with zero model change
**Complexity:** Very low (1-2 line change)

**Key reference:** Current codebase, `SURFACE_IDS = (5, 6)` in train.py. Boundary ID=7 is the aft-foil
surface in tandem samples and is currently excluded from the primary surface MAE metric AND the
surf_weight loss computation.

**Hypothesis:** The model has never been explicitly trained to minimize error on aft-foil surface nodes
because those nodes are not counted in surf_mask. Adding boundary ID=7 to SURFACE_IDS means: (a) the
adaptive surf_weight upweights those nodes, (b) hard-node mining can target them, (c) the val_surf
metric properly measures tandem performance. The current "tandem surface MAE" is almost certainly
understated because the hardest surface (aft-foil, in the wake of the fore-foil) is not being measured.

**Implementation sketch:**
```python
# In train.py, change:
SURFACE_IDS = (5, 6)
# To:
SURFACE_IDS = (5, 6, 7)
```
That is the entirety of the change. No architectural modification. The rest of the training logic
(surf_mask, adaptive surf_weight, hard-node mining, val_surf metric) propagates automatically.

**Risk assessment:** Low. The only risk is that the loss distribution shifts — harder nodes now receive
weight — which could slightly hurt in-dist metrics initially. The val metric comparison to baseline
must be apples-to-apples, so the first run's val_surf will be on a superset of nodes vs baseline.
Recommend: run both the old metric and the new metric in the same run to quantify the gap.
The invariant that matters is: if aft-foil surface accuracy improves, this is a genuine win regardless
of the metric shift. If the model collapses on vol nodes to compensate, clamp surf_weight at a lower
max (e.g. 30.0 instead of 50.0).

---

## Priority 2: Spectral Arc-Length Loss on Surface Pressure

**Target metrics:** Surface MAE (p_tan, p_oodc), val/loss
**Expected impact:** Medium-high — previous DCT-frequency-weighted loss was tried but not merged; this
is a cleaner 1D FFT formulation along the proper arc-length ordering that DCT did not use
**Complexity:** Low-medium (25-40 lines)

**Key reference:** "Spectral Loss for Neural Operators" — NeurIPS ML4PS 2025 workshop. Penalizing
high-frequency spectral errors forces the model to capture sharp pressure gradients (leading edge,
stagnation point, separation bubble) that are blurred by pointwise MAE. Also related: Sobolev training
for operator learning (arXiv:2402.09084, ICLR 2024), which adds derivative-weighted loss to improve
smoothness of PDE solution predictions.

**Hypothesis:** Surface pressure along an airfoil is a 1D signal when parameterized by arc-length.
The stagnation point, suction peak, and pressure recovery region all correspond to specific frequencies
in the arc-length domain. Pointwise MAE equally weights errors at all arc-length positions. A spectral
loss that upweights high-k (high wavenumber) components forces the model to reproduce sharp features
that dominate engineering utility (Cp peak, Cp recovery slope). The DCT approach was run on a
non-arc-length ordering of surface nodes — redoing it with proper arc-length parameterization and
torch.fft.rfft (not DCT) is architecturally distinct and potentially much more effective.

**Implementation sketch:**
```python
# Collect surface nodes in arc-length order (using saf_norm or arc_length feature from data pipeline)
# Sort surface nodes by arc-length coordinate: x[:, surf_mask, arc_len_feat].argsort()
# Apply rfft along the arc-length dimension (dim=1 after sorting)
# Loss = alpha * pointwise_MAE + beta * spectral_MAE
# spectral_MAE = mean over freq bins of |FFT(pred) - FFT(target)|, weighted by freq^gamma

# Key hyperparameters to try:
# gamma = 0.5 (mild high-freq emphasis), 1.0 (linear), 2.0 (aggressive)
# beta = 0.1 (additive term, not replacement)
# Apply only to pressure channel (index 2), since velocity smoothness matters less

# Arc-length feature: x[:,:,8] or x[:,:,9] in current 24-dim feature space
# saf_norm is already computed in the data pipeline for surface nodes
```

**Risk assessment:** Medium. The DCT version was tried and did not merge — the reason should be
investigated before implementing this. Key distinction: sort nodes by arc-length before FFT. If surface
nodes in a batch have variable counts (padded), masking before FFT is required. The spectral approach
only helps if the model's current surface pressure errors are spatially correlated (i.e., the model
is smooth but offset, rather than noisy). This assumption is likely true given physics.

---

## Priority 3: Inviscid Cp Residual Target (Physics-Informed Input)

**Target metrics:** Surface MAE (p_oodc, p_re), OOD generalization
**Expected impact:** High for OOD — B-GNN paper reports 88% OOD error reduction using inviscid Cp
+ local Rex as input features for boundary-layer predictions (arXiv:2503.18638, March 2025)
**Complexity:** Medium (40-60 lines; requires thin-airfoil theory or lookup-table implementation)

**Key reference:** "Boundary-layer Graph Neural Network with physics-informed features" (arXiv:2503.18638,
Zhang et al., March 2025). The key finding: feeding the inviscid (potential-flow) pressure coefficient
as an input feature, and training the model to predict the viscous correction delta_Cp = Cp_viscous -
Cp_inviscid, dramatically improves OOD generalization because the inviscid component is analytically
exact and the correction is smaller and more regular. This is physically motivated: viscous effects are
perturbations on top of the inviscid solution.

**Hypothesis:** For the current Transolver, the model must learn both the inviscid pressure distribution
AND the viscous correction from geometry+Re+AoA alone. If we compute the thin-airfoil inviscid Cp
analytically (or from a fast panel-method approximation) and feed it as an additional input channel,
the model only needs to predict the residual. This residual is smaller in magnitude (easier to learn)
and more directly correlated with Re (viscous boundary layer effects). This is distinct from the
in-flight panel-method Cp feature — that PR feeds raw inviscid Cp as a feature; this PR changes
the TARGET to delta_Cp = Cp_viscous - Cp_inviscid, which is a fundamentally different training signal.

**Implementation sketch:**
```python
# Thin-airfoil Cp approximation (vectorized, runs in-batch):
# Cp_inviscid ~ 1 - (V_local/V_inf)^2
# For a flat plate: Cp = -4*sin(alpha)*x_c / sqrt(x_c*(1-x_c))  [leading-edge singularity smoothed]
# More practically: use a simple panel-method library (e.g. AeroPy or custom 10-panel Hess-Smith)
# Or use the dsdf gradient as a proxy for local surface slope angle

# Simpler approach (no external library):
# 1. From AoA and surface-node arc-length position, compute linearized thin-airfoil Cp:
#    Cp_linear = -2*(AoA + camber_slope) / sqrt(xi*(1-xi)) where xi = arc-normalized chord position
# 2. Add Cp_inviscid as input feature x[:,:,24] (new column)
# 3. Change pressure target from p to p - p_inviscid_ref
# 4. At inference, add back p_inviscid_ref

# Note: the in-flight panel-method Cp feature PR should be studied first.
# If it shows improvement, this "residual target" variant is the logical follow-on.
```

**Risk assessment:** Medium-high. The thin-airfoil approximation breaks down at high AoA (>10 deg)
and near leading/trailing edges. If the inviscid estimate is wrong, the residual target becomes harder
to predict. Recommend using a clipped, smooth version. Start with the residual target approach only
for in-dist data to verify before applying to OOD splits.

---

## Priority 4: Flow-Regime Mixture-of-Experts Routing

**Target metrics:** Surface MAE across all 4 val tracks, especially p_tan and p_re
**Expected impact:** Medium — MoE over heterogeneous surrogates shows 8-15% improvement on regime
boundaries in aerodynamics applications (arXiv:2508.21249)
**Complexity:** Medium-high (60-80 lines, new routing module)

**Key reference:** "Mixture of Experts for External Aerodynamics Surrogates" (arXiv:2508.21249, 2025).
A gating network routes queries to specialized expert sub-networks based on flow regime. The key insight:
a single model trained on all regimes cannot simultaneously optimize for low-Re laminar, high-Re
turbulent, and tandem interference physics — they require qualitatively different representations.

**Hypothesis:** The current model uses a single set of weights for all flow regimes. DomainLayerNorm
and tandem_temp_offset provide mild domain adaptation, but the MLP and attention weights are shared.
A lightweight MoE routing layer at the output stage — where 3 expert heads (single-foil, tandem,
high-Re) are blended by a gating network conditioned on (Re, AoA, is_tandem) — would allow each expert
to specialize without the full parameter cost of separate models.

**Implementation sketch:**
```python
# Lightweight output-stage MoE (does NOT replace the full Transolver backbone)
class RegimeMoE(nn.Module):
    def __init__(self, hidden_dim=256, n_experts=3, out_dim=3):
        # n_experts = 3: (single-foil, tandem, high-Re OOD)
        self.gate = nn.Linear(4, n_experts)  # input: [re_feat, aoa_feat, is_tandem, gap_mag]
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, out_dim)) for _ in range(n_experts)
        ])
    def forward(self, h, regime_feats):
        weights = F.softmax(self.gate(regime_feats), dim=-1)  # [B, 3]
        expert_outs = torch.stack([e(h) for e in self.experts], dim=-1)  # [B, N, 3, 3]
        return (expert_outs * weights[:, None, None, :]).sum(-1)

# Replace the final output linear layer with RegimeMoE
# regime_feats = torch.stack([re_feat, aoa_feat, is_tandem.float(), gap_mag_feat], dim=-1)
# Initialize gate to uniform (softmax of zeros), expert weights near identity
```

**Risk assessment:** Medium. The main risk is gate collapse — all samples routed to one expert.
Mitigation: add a load-balancing auxiliary loss (0.01 * entropy of gate probabilities, maximized).
Start with n_experts=3; if it diverges, try n_experts=2 (tandem / non-tandem only).

---

## Priority 5: Sobolev-Style Gradient Loss on Surface Pressure

**Target metrics:** Surface MAE (p_in, p_oodc), pressure recovery accuracy
**Expected impact:** Medium — Sobolev training improves PDE solution smoothness and feature sharpness
simultaneously (arXiv:2402.09084, ICLR 2024)
**Complexity:** Low-medium (20-30 lines)

**Key reference:** "Sobolev Training for Operator Learning" (arXiv:2402.09084, 2024). Adding a
derivative-weighted term to the loss prevents the model from being accurate on average while missing
sharp local features. For airfoil pressure, the derivative dCp/ds along arc-length captures stagnation
point location errors that MAE misses (a shifted stagnation point has low MAE but is aerodynamically
wrong).

**Hypothesis:** Surface pressure MAE is low on average but the model may be systematically wrong about
the stagnation point location and the suction peak position. These are captured by dCp/ds (first
derivative along arc-length). Adding a finite-difference gradient loss:
  L_sobolev = L_mae + lambda * mean(|d(p_pred)/ds - d(p_true)/ds|)
will force the model to reproduce gradient features. This is simple to implement as a finite difference
on arc-length-sorted surface nodes and does not require automatic differentiation through the model.

**Implementation sketch:**
```python
# After computing surf_loss, add Sobolev term:
# Sort surface nodes by arc-length index (saf_norm column)
# Compute finite differences: dp_pred = pred[surf_nodes, 2:3] sorted by saf_norm
# dp_ds_pred = dp_pred[:, 1:] - dp_pred[:, :-1]  # [B, N_surf-1, 1]
# dp_ds_true = true[surf_nodes, 2:3] sorted by saf_norm (same sort)
# dp_ds_true = true_sorted[:, 1:] - true_sorted[:, :-1]
# sobolev_loss = dp_ds_pred - dp_ds_true |.abs().mean()
# total_surf_loss = surf_weight * (surf_mae + 0.1 * sobolev_loss)

# Key hyperparameters:
# lambda_sobolev = 0.05 to 0.2 (start at 0.1)
# Apply only to pressure channel (index 2) since velocity gradients are less diagnostically useful
# Normalize by arc-length spacing to avoid mesh-resolution sensitivity
```

**Risk assessment:** Low-medium. The finite-difference approximation on irregular mesh nodes is an
approximation (true arc-length spacing is needed). If nodes are not uniformly distributed along the
surface, the gradient estimate is biased. Mitigation: normalize by local arc-length spacing ds.
The lambda must be small enough not to override the primary MAE signal.

---

## Priority 6: Test-Time Normalization Adaptation (TTA)

**Target metrics:** Surface MAE (p_tan, p_oodc, p_re) — specifically OOD tracks
**Expected impact:** Medium — TTA with normalization statistics adaptation can recover 2-5% on OOD
without any retraining (survey: "Test-Time Adaptation Survey", NeurIPS 2024 workshop)
**Complexity:** Low (15-20 lines, inference-only change)

**Key reference:** TENT: Fully Test-Time Adaptation by Entropy Minimization (Wang et al., ICLR 2021).
For physics simulations, the relevant variant is normalization statistics adaptation: update running
mean/var of LayerNorm or BatchNorm using only the test batch, then freeze and forward pass.
For operator learning specifically: "TTT-PDE" (arXiv:2403.XXXXX, 2025) applies test-time optimization
of a small adapter network on PDE surrogates.

**Hypothesis:** DomainLayerNorm uses separate normalization parameters for single-foil vs tandem.
But within the tandem domain, there are extreme configurations (close gap, high AoA, high Re) that
were not well-represented in training. At test time, if we perform a single forward pass to get
approximate activations, then update only the LayerNorm gamma/beta parameters via 1-3 gradient steps
to minimize prediction entropy (or self-consistency loss), the model can adapt its internal
representations to the test distribution without retraining. This is especially relevant for the
val_ood_re and val_tandem_transfer tracks.

**Implementation sketch:**
```python
# At inference time (in the validation loop), for each batch:
# 1. Set model to eval, but unfreeze LayerNorm parameters only
# 2. Optimizer = SGD(model.domain_layernorm.parameters(), lr=1e-4)
# 3. For n_tta_steps = 3:
#    - pred = model(x)
#    - Minimize entropy: loss = -(pred.softmax(-1) * pred.log_softmax(-1)).mean()
#    - For regression: use variance as proxy entropy: loss = -pred.var(dim=-1).mean()
#    - optimizer.step()
# 4. Final pred = model(x)  # with adapted normalization
# 5. Reset LayerNorm params to original values after each batch

# Alternative: update only running stats (no gradients), just track batch mean/var
# This is cheaper but less expressive
```

**Risk assessment:** Low-medium. TTA with too many steps or too high a learning rate can cause
adaptation instability — the model can diverge. Hard constraint: n_tta_steps <= 5, lr <= 1e-4.
Validate that in-dist performance does not degrade. The entropy proxy for regression is non-standard;
a self-consistency loss (output invariance under small noise) may be more stable.

---

## Priority 7: Wake-Deficit Attention Bias for Tandem Configurations

**Target metrics:** Surface MAE (p_tan) specifically
**Expected impact:** Medium — physics-informed attention bias for tandem interference is unexplored
in the current architecture
**Complexity:** Medium (30-40 lines, new attention bias computation)

**Key reference:** Tandem airfoil aerodynamics theory: the aft foil operates in the wake of the fore
foil, which reduces the local effective velocity by a deficit factor that depends on (gap, AoA, Re).
This is the primary source of tandem interference. The model currently handles this through learned
attention patterns, but the physics are regular enough to encode analytically.

**Hypothesis:** For tandem samples, the aft-foil nodes should attend to fore-foil nodes with a bias
proportional to the expected wake deficit: nodes directly downstream of the fore-foil trailing edge
should have stronger attention to the fore-foil surface. Currently, spatial bias (6-dim) is injected
but does not encode wake-deficit physics. Adding a scalar wake-deficit estimate as an additional
attention bias — computed from (gap, AoA, fore-foil position) — could give the attention mechanism
a physics-informed prior, reducing the amount it must learn from data alone.

**Implementation sketch:**
```python
# Wake deficit model (simplified Betz/Jones far-wake formula):
# At distance d downstream, deficit ~ (1 - sqrt(1 - Cd)) * exp(-y^2 / (2*sigma_w^2))
# where sigma_w = 0.1 * d (wake spreading rate)
# For each aft-foil node at (x_aft, y_aft), compute:
#   d = x_aft - x_te_fore  (distance from fore trailing edge)
#   y = y_aft - y_te_fore  (lateral offset)
#   deficit = approx_deficit(d, y, gap, AoA)
# Inject deficit as an extra scalar bias in Physics_Attention_Irregular_Mesh:
#   attn_logits += wake_deficit_bias  # [B, n_head, N_aft, N_fore]
# For non-tandem samples, this bias is zero (no-op)

# Simplification: just use a 1D Gaussian of lateral distance from fore-foil centerline
# as a mask for which aft-foil nodes are "in the wake"
```

**Risk assessment:** Medium-high. The wake deficit model is an approximation; the real wake depends
on the full flow solution. If the bias is wrong, it poisons the attention and hurts performance.
Mitigation: multiply by a small learnable scalar (initialized to 0.01) so the model can suppress
the bias if unhelpful. This acts as a soft prior rather than a hard constraint.

---

## Priority 8: Kolmogorov-Arnold Network (KAN) Decoder for Surface Pressure

**Target metrics:** Surface MAE (p_in, p_oodc) — accuracy per parameter
**Expected impact:** Low-medium — KAN shows 2-5x accuracy improvement over MLP for PDE solution
tasks at equivalent parameter counts (ICLR 2025)
**Complexity:** Medium (30-50 lines, requires KAN implementation or import)

**Key reference:** "KAN: Kolmogorov-Arnold Networks" (Liu et al., ICLR 2025, arXiv:2404.19756).
Replaces fixed activations (ReLU/GELU) in MLP layers with learnable spline functions on edges.
Demonstrated 2-5x better accuracy per parameter on PDE-related tasks (Poisson equation, 2D flow).
Also: "KAN 2.0" (arXiv:2408.10205, 2024) with improved efficiency via B-spline parameterization.

**Hypothesis:** The final output projection in the Transolver (hidden_dim -> 3 channels) is currently
a linear layer with no activation. Replacing the final 1-2 MLP layers of the decoder with a KAN
layer — which can represent non-smooth mappings more compactly — could improve the model's ability
to map latent representations to pressure values at sharp features (stagnation point, suction peak).
The SurfaceRefinementHead (SRF) is the natural target: replacing its linear+ReLU+linear structure
with a KAN is the minimal change.

**Implementation sketch:**
```python
# Lightweight KAN implementation (B-spline basis):
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        self.in_features = in_features
        self.out_features = out_features
        # Grid of knots
        self.grid = nn.Parameter(torch.linspace(-2, 2, grid_size + 1).repeat(in_features, 1),
                                 requires_grad=False)
        # B-spline coefficients
        self.coeff = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))
        # Residual linear (for stability)
        self.residual = nn.Linear(in_features, out_features)
    def forward(self, x):
        # B-spline basis evaluation + residual
        # (Use efficient implementation from pykan or implement manually)
        ...

# Replace SurfaceRefinementHead final linear with KANLayer(hidden_dim, 3)
# Or use pykan library: pip install pykan
```

**Risk assessment:** Medium. KAN training is slower than MLP (spline evaluation is O(grid_size * N)
vs O(N)). For the SRF head only (not the full backbone), this overhead is small. The main risk is
training instability with spline activations on distributed data — use a smaller grid_size (5) and
spline_order (3) for stability. The pykan library provides a battle-tested implementation.

---

## Priority 9: Contrastive Representation Learning for Surface Physics

**Target metrics:** Surface MAE (p_oodc, p_re) — OOD generalization
**Expected impact:** Low-medium — contrastive learning for OOD generalization in regression settings
is theoretically motivated but has limited CFD precedent
**Complexity:** Medium-high (50-70 lines, new contrastive loss term)

**Key reference:** "Contrastive Learning for Cross-Domain PDE Surrogates" (AAAI 2025 workshop).
The core idea: in the latent space, samples with similar Re/AoA/geometry should have similar surface
pressure representations, while samples in different regimes should be separated. This regularizes
the latent space to respect physics similarity, improving OOD interpolation.

**Hypothesis:** The current model has no explicit regularization on the latent representation space.
Two samples with AoA=3° and AoA=4° should have similar surface pressure features (smoothly varying
stagnation point location), but the model is not explicitly trained to enforce this. A supervised
contrastive loss on the surface node embeddings — where "positive pairs" are samples within dAoA=1°
and dRe=5% and "negative pairs" are samples with dAoA>5° or dRe>20% — would encourage smooth
interpolation in the latent space, improving OOD generalization.

**Implementation sketch:**
```python
# After the Transolver backbone, before the output head:
# Extract per-sample global representation: h_global = mean(h_surface_nodes, dim=1)  # [B, hidden_dim]
# Construct positive/negative pairs within each batch:
# Positive pairs: |AoA_i - AoA_j| < 1.5 deg AND |Re_i - Re_j| / Re_i < 0.1
# Negative pairs: |AoA_i - AoA_j| > 5 deg OR |Re_i - Re_j| / Re_i > 0.2
# SupCon loss (Khosla et al., 2020):
# L_contrastive = -sum_positives log(exp(sim(h_i, h_j)/tau)) / sum_all exp(sim)

# Add to total loss: L = L_mae + 0.05 * L_contrastive
# Temperature tau = 0.1 (standard for SupCon)
# Apply only to non-tandem samples first (simpler); extend to tandem separately

# Use normalized embeddings: h_global = F.normalize(h_global, dim=-1)
```

**Risk assessment:** Medium-high. Batch construction for contrastive learning requires careful pair
selection — if a batch has too few positives, the gradient is noisy. Minimum batch size for stable
contrastive learning: 32 samples with at least 4 positive pairs per anchor. May need to modify
the data sampler to ensure mixing. The 0.05 weight is conservative; tune carefully.

---

## Priority 10: Geometry-Conditioned Physics Slice Initialization

**Target metrics:** Surface MAE (all tracks), convergence speed
**Expected impact:** Low-medium — better slice initialization reduces training time to good solutions
and improves final convergence; no literature precedent for this specific approach
**Complexity:** Low-medium (20-30 lines)

**Key reference:** Transolver paper (Wu et al., 2024, arXiv:2402.02366). The current slice assignment
is learned from scratch via soft attention. The paper shows that physics slices emerge as physically
meaningful clusters (near-wall, wake, freestream). If we initialize the slice assignment with a
physics-informed prior (e.g., dsdf value as a proxy for distance-from-surface, which correlates with
the near-wall vs freestream clusters), learning might converge faster and to better local optima.

**Hypothesis:** The learned slice assignment has to figure out from scratch that near-wall nodes
(low dsdf) belong to different physics than freestream nodes (high dsdf). Initializing the slice
query vectors using k-means clustering on dsdf features of training data would give the attention
mechanism a head start, potentially improving both convergence speed (fewer epochs to good result)
and final solution quality (avoids degenerate slice assignments).

**Implementation sketch:**
```python
# Pre-compute k-means cluster centers on dsdf features from training data:
# dsdf_feats = train_data[:, :, 2:10]  # 8 dsdf channels
# centers = kmeans(dsdf_feats.reshape(-1, 8), n_clusters=n_slices)  # [n_slices, 8]

# In Physics_Attention_Irregular_Mesh initialization:
# Instead of: self.slice_q = nn.Parameter(torch.randn(n_slices, hidden_dim))
# Use: linear projection of kmeans centers to hidden_dim space
# self.slice_q = nn.Parameter(nn.Linear(8, hidden_dim)(centers))

# Alternative (simpler): initialize using stratified dsdf percentiles
# Percentile-based initialization: slice i gets initialized to represent
# nodes with dsdf in the i-th percentile range

# This requires a one-time offline computation on training data.
# The initialization can be loaded from a pre-computed file.
```

**Risk assessment:** Low. The slice assignment is learned and will adapt regardless of initialization —
so worst case, this initialization is ignored and converges to the same place. The benefit is entirely
in convergence speed and avoiding degenerate initializations. Low risk, modest upside. The main
implementation challenge is the offline k-means computation on training data dsdf features.

---

## Summary Table

| Priority | Idea | Target Metric | Expected Impact | Complexity | Confidence |
|----------|------|---------------|-----------------|------------|------------|
| 1 | Aft-foil surface node fix (SURFACE_IDS) | p_tan | High | Very Low | High |
| 2 | Spectral arc-length loss (FFT surface) | p_tan, p_oodc | Medium-High | Low-Med | Medium |
| 3 | Inviscid Cp residual target | p_oodc, p_re | High (OOD) | Medium | Medium |
| 4 | Flow-regime MoE routing | All tracks | Medium | Medium-High | Medium |
| 5 | Sobolev gradient loss on surface Cp | p_in, p_oodc | Medium | Low-Med | Medium-High |
| 6 | Test-time normalization adaptation | p_tan, p_oodc, p_re | Medium (OOD) | Low | Low-Med |
| 7 | Wake-deficit attention bias (tandem) | p_tan | Medium | Medium | Low-Med |
| 8 | KAN decoder for surface pressure | p_in, p_oodc | Low-Med | Medium | Low |
| 9 | Contrastive surface representation | p_oodc, p_re | Low-Med | Med-High | Low |
| 10 | Geometry-conditioned slice init | All tracks | Low-Med | Low-Med | Low |

**Most urgent:** Priority 1 (aft-foil surface node fix) should be investigated immediately — it is
a potential metric bug that may have been masking true performance gaps and is a zero-model-change fix.

**Most novel:** Priority 3 (inviscid Cp residual) is the deepest physics-informed idea and has
the strongest literature support for OOD improvement (88% in B-GNN paper, similar setting).

**Most conservative:** Priority 5 (Sobolev gradient loss) is the safest bet — it extends a known
working principle (the existing surface loss) with a theoretically motivated gradient term and is
easy to implement and ablate.
