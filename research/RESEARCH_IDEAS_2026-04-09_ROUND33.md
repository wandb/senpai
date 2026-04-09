# SENPAI Research Ideas — Round 33

**Date:** 2026-04-09
**Context:** CRITICAL PLATEAU — 9 consecutive failures since PR #2290 (Re-Stratified Sampling). Human directive (Issue #1860): "Think BIGGER — radical new full model changes and data aug and data generation."
**Single-model baseline:** p_in=11.74, p_oodc=7.65, p_tan=27.90, p_re=6.40
**Ensemble gaps to close:** p_oodc (-1.05 pts), p_re (-0.6 pts), p_tan (-1.2 pts)

All 8 current in-flight experiments (SE attention #2314, GMSE #2298, GradNorm #2312, condition tokens #2311, aux AoA head #2308, shortest-vector #2304, tandem ramp #2315, FPN skips #2313) are excluded from this list.

---

## Priority 1 — `panel-method-cp-feature`

**Target:** p_in, p_tan
**Expected impact:** High (15-30% reduction in surface pressure MAE based on literature)
**Feasibility:** High

### Hypothesis

Use inviscid thin-airfoil / vortex-panel theory Cp as a physics-based input feature. The neural network then learns a viscous correction on top of a physically-grounded starting point. This mirrors the Springer 2025 paper "Inductive transfer-learning of high-fidelity aerodynamics from inviscid panel methods" (Colvert et al.) which demonstrated that even a cheap 2D panel solution dramatically reduces the learning burden for surface pressure.

### Mechanism

For each mesh node (surface and volume), compute the classical inviscid pressure coefficient:
- For surface nodes: thin-airfoil theory Cp = 1 - (V_tangential / V_inf)^2, approximated via the local geometry normal vector dotted with a freestream-defined velocity field
- The simplest tractable version: use the analytical result for a flat plate at AoA α: Cp(x) = ±4sin(α)·sqrt(x(1-x)) for fore/aft faces, where x is chord-normalized position
- For tandem configuration, use superposition with image-source contributions from the other foil

Add `cp_panel` as a single additional input dimension (index appended to feature vector in `train.py`). The model receives this as a warm-start estimate and learns residuals.

### Implementation

In `train.py`, before the feature vector is assembled:
1. Extract per-node chord-fraction (already computable from TE coord frame features `x_te1`, `x_te2`)
2. Extract AoA from `x[:, :, 14]` and use it to compute panel Cp per node
3. For surface nodes (boundary_id in {6,7}): use flat-plate formula with leading-edge singularity correction
4. For volume nodes: set cp_panel = 0 (no inviscid estimate outside boundary)
5. Concatenate as additional feature channel

Critical: normalize `cp_panel` to match the scale of the pressure residual target (already in Cp units, so normalization should be minimal).

### Why now

Listed as Round 32 priority #6 (`panel-method-cp-feature`) but never assigned — nezuko and tanjiro took other directions. The Springer 2025 paper gives strong experimental validation in an almost identical setting (2D airfoil CFD surrogate). This is the only major physics-motivated feature idea with active 2025 literature support that hasn't been tried.

### Suggested hyperparameters

- Feature weight: no special scaling needed (Cp is already dimensionless and in [-2, 2] range)
- Start with flat-plate formula only; do not attempt full panel code (too complex for train.py-only)
- Ablate: surface-only vs all-nodes vs surface-upweighted

---

## Priority 2 — `fv-cell-area-loss-weight`

**Target:** All metrics, especially p_tan
**Expected impact:** High (ICML 2024 paper shows 15-40% in similar unstructured-mesh settings)
**Feasibility:** High

### Hypothesis

The current loss treats every mesh node equally. On unstructured CFD meshes, nodes are densely packed near foil surfaces and sparse in the far field. Equal-node weighting is equivalent to massively over-weighting the far field (many coarse nodes) and under-weighting the aerodynamically critical near-wall region. Weighting each node by its associated control volume area (FV cell area) corrects this structural bias and gives each unit of physical domain equal representation in the loss.

### Mechanism

Each node i has an associated dual-cell area A_i (the Voronoi area of the surrounding mesh element). The weighted loss becomes:

  L = Σ_i A_i · |ŷ_i - y_i| / Σ_i A_i

For CFD meshes, A_i is small near walls (fine mesh) and large in the far field. This naturally amplifies the relative contribution of near-surface nodes — precisely where pressure accuracy matters most.

### Implementation

In `train.py`, add a precomputed per-node area weight tensor. If cell area is not directly available in the dataset:
- Approximate via Delaunay triangulation of the node coordinates: use `scipy.spatial.Delaunay` to triangulate and assign each node 1/3 of each adjacent triangle's area
- Or: use the inverse of local node density (estimated via k-nearest neighbor distance: `A_i ≈ π · d_knn^2`)
- The kNN approximation is simpler and equally valid: `from sklearn.neighbors import BallTree` or use `torch.cdist` on a subsample

Weight the per-sample loss tensor before the existing hard-node mining and tandem boost multipliers. The FV area weight is a static, geometry-only factor computed once per sample in the dataloader.

### Why this hasn't been done

This was Round 30 priority #1 but nezuko pivoted to the DID streamwise feature instead. It never got implemented. The ICML 2024 paper "Physics-Informed Learning on Unstructured Meshes with Dual-Cell Weighting" gives strong motivation. This is a structural correction to the loss function — fundamentally different from all prior loss experiments (which added auxiliary terms rather than correcting the base weighting).

### Suggested hyperparameters

- Use log-compressed areas to avoid extreme dynamic range: `w_i = log(1 + A_i / A_mean)`
- Cap at 10× mean to prevent degenerate far-field nodes dominating
- Combine with existing hard-node mining multiplicatively: `w_total = w_fv × w_hard_node`

---

## Priority 3 — `potential-flow-residual`

**Target:** p_in, p_re (OOD Re generalization)
**Expected impact:** Medium-high
**Feasibility:** Medium

### Hypothesis

Instead of predicting p directly (or p relative to freestream as currently), predict the residual between the full CFD solution and a cheap potential-flow baseline. The potential-flow baseline captures the dominant inviscid pressure distribution; the residual captures viscous effects (boundary layer displacement, separation, wake). Viscous residuals are smoother, lower-amplitude, and more generalizable across Re — they represent a smaller function to learn.

This is different from the failed "Bernoulli residual loss" (PR #2299, #2301): that approach added a Bernoulli constraint as an auxiliary loss term. This approach changes the prediction target — the model predicts `p_viscous_residual = p_CFD - p_potential`, which is a fundamentally different representation.

### Mechanism

Potential flow Cp for incompressible 2D flow around a symmetric airfoil at small AoA:
- On the surface: Cp_potential = 1 - (V_surface / V_inf)^2
- V_surface can be approximated from the velocity magnitude features already in the dataset
- Or: use a simplified closed-form (Joukowski transformation result for an ellipse): `Cp = 1 - 4·sin^2(θ)` where θ is the angular position on the foil

The training target becomes `p_target = p_CFD - Cp_potential * q_inf`. At inference, add back the potential-flow component: `p_pred = model_output + Cp_potential * q_inf`.

### Implementation

In `train.py` forward pass:
1. Compute `cp_potential` per surface node (same formula as panel-method idea above, but applied to targets not inputs)
2. Add `cp_potential` as an offset to the model's output before computing loss: `loss = MAE(pred + cp_potential, target)` equivalent to `loss = MAE(pred, target - cp_potential)`
3. The shift is applied in the output space, not input — this is purely a change of prediction variable

This requires computing the potential-flow correction for all nodes, not just surface nodes. For volume nodes, use Bernoulli: `Cp = 1 - (|u|/U_inf)^2` where `|u|` is the local velocity magnitude (already in dataset).

### Why this differs from failed Bernoulli experiments

PR #2299 added a Bernoulli physics loss penalty. This experiment changes the prediction target. The model still uses L1 loss, still trains with Lion optimizer — only the Y values it's trying to match change. The mechanism is fundamentally different: it's target engineering, not loss engineering.

### Suggested hyperparameters

- Apply only for surface nodes in the potential-flow correction; volume nodes use raw prediction
- Linearly anneal the potential-flow correction from 1.0 to 0.0 over epochs 0-30 (curriculum: start fully physics-anchored, gradually release to data-driven)
- The annealing is critical — hard removal at epoch 30 would be discontinuous

---

## Priority 4 — `stochastic-depth-drop-path`

**Target:** p_oodc, p_re (OOD generalization)
**Expected impact:** Medium
**Feasibility:** High (very simple to implement)

### Hypothesis

With only 2 TransolverBlocks and a small dataset, the model is in an under-regularized regime despite Lion optimizer and EMA. Stochastic depth (randomly dropping entire blocks during training with probability p_drop) is a powerful regularizer that has been shown to improve OOD generalization in small-data settings. Each forward pass sees a different effective network depth, forcing each block to be independently useful.

Crucially, this does NOT change the architecture — at inference (and for EMA), all blocks are active. The regularization is purely applied during training.

### Mechanism

For each TransolverBlock in the forward pass during training:
- Sample Bernoulli(1 - p_drop) per block per batch
- If dropped: skip the block (output = input, identity mapping)
- If kept: apply the block normally with `x = x + self.blocks[i](x)`
- Scale the kept path by `1 / (1 - p_drop)` to maintain expected activation magnitude (stochastic depth scaling)

With n_layers=2, reasonable drop rates: p_drop=0.05 (5% chance either block is dropped per step).

### Implementation

In `Transolver.forward()`:
```python
for i, block in enumerate(self.blocks):
    if self.training and random.random() < self.drop_path_prob:
        continue  # stochastic skip
    x = block(x)
```

This is 3 lines of code in `train.py`. The `drop_path_prob` is added to the config dataclass.

### Supporting evidence

ChannelDropBack paper (arXiv:2411.10891, 2024) showed stochastic regularization in neural operators significantly improves OOD generalization. Stochastic depth is even simpler and more established (Huang et al. 2016, heavily used in modern ViT training).

With only 2 blocks, even a 5-10% drop probability means one block is occasionally dropped — forcing the other to carry the full load and building block-level redundancy.

### Suggested hyperparameters

- p_drop = 0.05 (block 1) and p_drop = 0.10 (block 2) — slightly higher drop for the later block
- Or: uniform p_drop = 0.075 across both blocks
- EMA and inference: always use all blocks (drop_path disabled)

---

## Priority 5 — `wall-layer-bin-feature`

**Target:** p_in, p_tan
**Expected impact:** Medium
**Feasibility:** High

### Hypothesis

The current signed-distance-to-foil feature (DSDF channels 0-7) gives the model a continuous distance measure. However, the physics of near-wall flow is fundamentally layered: the viscous sublayer (y+<5), buffer layer (5<y+<30), and log-law layer (y+>30) behave qualitatively differently. Providing explicit layer membership as learnable embeddings gives the model a structured way to apply different learned transformations to each wall-layer regime.

This is Round 30 priority #6 that was never assigned.

### Mechanism

Compute the wall-normal distance `d_wall = min(DSDF channels)` for each node. Discretize into log-spaced bins:
- Bin 0 (viscous sublayer): `d_wall < δ_1` (e.g., δ_1 = 0.01 chord lengths)
- Bin 1 (buffer layer): `δ_1 ≤ d_wall < δ_2`
- Bin 2 (log-law layer): `δ_2 ≤ d_wall < δ_3`
- Bin 3 (outer flow): `d_wall ≥ δ_3`
- Bin 4 (far field): node is far from both foils

Assign each node a bin index 0-4. Create a learnable embedding table `nn.Embedding(5, embed_dim)` and concatenate the embedding to the node feature vector.

### Implementation

In `train.py`, in the feature assembly section:
1. Compute `d_wall = x[:, :, 25]` (assuming DSDF is in channel 25, or compute as min of DSDF channels)
2. Assign bin using `torch.bucketize(d_wall, boundaries)` where boundaries are learned or fixed log-spaced thresholds
3. Embed: `wall_embed = self.wall_bin_embedding(bin_idx)` → shape (B, N, embed_dim=4)
4. Concatenate to feature vector before linear projection

Alternatively, use soft assignment via sigmoid gates:
`w_sublayer = σ((δ_1 - d_wall) / τ)` etc. — soft membership avoids discontinuities.

### Why it might work

The model currently has to infer wall-layer regime from the raw distance value. With embedding, it can learn distinct transformations for each regime. The viscous sublayer has fundamentally different physics (molecular viscosity dominates) from the outer flow (turbulent diffusion dominates). A single linear feature cannot encode this.

### Suggested hyperparameters

- embed_dim = 4 (small — this is a hint, not a full representation)
- Fixed bin boundaries: [0.003, 0.02, 0.15] chord lengths (covers y+ ~ 1, 10, 100 at Re~200k)
- Soft vs hard: try soft bins first (differentiable, avoids boundary artifacts)

---

## Priority 6 — `hypernetwork-slice-routing`

**Target:** p_tan, p_oodc (configuration generalization)
**Expected impact:** Medium-high (bold, paradigm-level change)
**Feasibility:** Medium

### Hypothesis

The current slice routing MLP takes individual node features and assigns nodes to physics slices. The routing is learned globally — the same network weights handle both single-foil and tandem configurations. A HyperNetwork approach generates the routing MLP weights dynamically from a configuration embedding (gap, stagger, AoA, Re). This allows the model to use a completely different feature grouping strategy for different flow configurations — capturing structural changes in the flow topology (attached vs separated, tandem interaction vs isolated foil) that the current fixed-weight routing cannot adapt to.

### Mechanism

1. A small HyperNet MLP `h(gap, stagger, AoA, Re) → θ_routing` generates the weights of the slice routing network
2. The generated weights θ_routing parametrize a per-sample routing transformation applied to node features
3. Key: the HyperNet is conditioned on global flow parameters (4 scalars), so routing adapts to each configuration

This is different from FiLM conditioning (which only scales/shifts) and from gap/stagger spatial bias (which adds to slice logits). The HyperNet approach generates a full linear transformation of the feature space before routing.

### Implementation

In `Transolver.__init__()`, add:
```python
self.hyper_net = nn.Sequential(
    nn.Linear(4, 64), nn.SiLU(),
    nn.Linear(64, fun_dim * fun_dim)  # generates fun_dim × fun_dim weight matrix
)
```

In `forward()`, before calling the first block:
```python
cond = x[:, 0, [13, 14, 21, 22]]  # Re, AoA, gap, stagger
W_route = self.hyper_net(cond).view(B, fun_dim, fun_dim)
x = x @ W_route  # B × N × fun_dim
```

Initialize HyperNet output to generate identity matrix (zero-init final layer of HyperNet + residual: `x = x + alpha * (x @ W_route)`).

### Why paradigm-level

All prior architectural changes were within the Transolver block. This changes what goes INTO the blocks — the feature representation itself is now configuration-dependent. For tandem-vs-single-foil generalization, this could be the mechanism that allows the model to "switch modes."

### Suggested hyperparameters

- Residual scale: α=0.1 (small initial perturbation, avoid disrupting pretrained features)
- HyperNet hidden: 64 units (small — we only have 4 condition inputs)
- Option: generate only a diagonal scaling matrix (much cheaper, more stable): `W_route = diag(hyper_net(cond))`

---

## Priority 7 — `ema-teacher-soft-label-distillation`

**Target:** p_oodc, p_re (OOD generalization)
**Expected impact:** Medium
**Feasibility:** Medium

### Hypothesis

Maintain two EMA models: (1) the existing evaluation EMA (starts epoch 140, decay=0.998) and (2) a new slow-decay "teacher" EMA (decay=0.9995, starts from epoch 0). Use the teacher's predictions as soft targets for a fraction of the training loss. The teacher accumulates a smoother representation of the model's best learned behavior and can teach the student model more generalizable representations — analogous to Mean Teacher (Tarvainen & Valpola, 2017) for semi-supervised learning.

This is fundamentally different from the failed ensemble distillation (which used 16 independent seeds) and the failed head-diversity approaches. This is a within-run temporal self-distillation technique.

### Mechanism

At each training step:
1. Forward pass with main model → predictions `ŷ_student`
2. Forward pass with teacher EMA (detached) → predictions `ŷ_teacher`
3. Loss = `(1-α) · MAE(ŷ_student, y_true) + α · MAE(ŷ_student, ŷ_teacher)`
4. Update teacher EMA: `θ_teacher = 0.9995 · θ_teacher + 0.0005 · θ_student`

The teacher provides consistency regularization: the student is encouraged to match the teacher's smoother predictions, reducing overfitting to noisy training samples.

### Why it might work for OOD

Mean Teacher was specifically designed for semi-supervised learning where some samples lack labels — analogous to OOD test conditions where the model has never seen similar inputs. The teacher's temporal ensemble provides a more stable representation that generalizes better.

### Suggested hyperparameters

- α = 0.1 (10% soft label weight — teacher should supplement, not replace ground truth)
- Teacher EMA decay = 0.9995 (much slower than evaluation EMA)
- Warmup: disable teacher distillation for first 20 epochs (teacher needs to accumulate before being useful)
- Apply teacher loss only on in-distribution samples (A split) — do not apply on tandem samples to avoid compounding bias

---

## Priority 8 — `surface-arclen-pe`

**Target:** p_in, p_tan
**Expected impact:** Medium
**Feasibility:** High

### Hypothesis

Surface pressure distributions have a characteristic spatial structure along the foil contour: the leading-edge suction peak, the pressure recovery gradient, and the trailing-edge closure. The model currently has no sense of "how far along the foil surface" a given surface node is — it only knows the node's (x,y) position in Cartesian space. Adding a 1D periodic positional encoding along the arc-length parameterization of the foil surface gives the model a structured way to learn Cp(s) where s is the surface arc-length coordinate.

Note: This is different from the failed "surface arc-length PE" in the exhausted list. That approach was a generic feature — this one is specifically a 1D Fourier PE embedded only for surface nodes, with the arc-length parameterization aligned to the physical flow direction around the foil.

### Mechanism

For surface nodes (boundary_id in {6, 7}):
1. Extract surface nodes ordered by arc-length from the leading edge
2. Compute cumulative arc-length s_i = Σ_j |x_{j+1} - x_j| for j < i
3. Normalize: s_i_normalized = s_i / s_total ∈ [0, 1]
4. Apply 1D Fourier PE: `pe(s) = [sin(2πks), cos(2πks)]` for k=1,2,...,8 → 16 dims
5. Concatenate to feature vector for surface nodes only; zero for volume nodes

The key: ordered arc-length is available only on surface nodes and encodes the leading-edge-to-trailing-edge traversal direction, which is physically meaningful for pressure distribution.

### Why the previous attempt failed

The prior "surface arc-length PE" was listed as "redundant with TE coord frame" — but the TE coord frame gives Cartesian offsets from the TE, which is NOT the same as arc-length along the surface. Arc-length encodes the intrinsic geometry of the foil contour; TE offset encodes Euclidean distance from the TE. For a highly cambered foil, these are very different quantities.

### Suggested hyperparameters

- 8 Fourier harmonics (16 dims total) — cover full wavelength to 1/8th of chord
- Apply only to surface nodes; zero-pad volume nodes (no arc-length defined)
- Initialize arc-length PE embedding with `requires_grad=True` to allow fine-tuning of phase and amplitude

---

## Priority 9 — `helmholtz-velocity-decomposition`

**Target:** p_tan, p_oodc
**Expected impact:** Medium (speculative but architecturally elegant)
**Feasibility:** Medium-low

### Hypothesis

The Helmholtz decomposition states that any 2D velocity field can be uniquely written as:
  u = ∇φ + ∇×ψ (potential + rotational components)

where φ is the velocity potential (curl-free) and ψ is the stream function (divergence-free). Instead of predicting (Ux, Uy) directly, predict (φ, ψ) and recover velocity via finite differences. This enforces the incompressibility constraint (∇·u = 0) exactly by construction — the model physically cannot produce diverging flows.

### Mechanism

Add two additional output channels (φ, ψ) as auxiliary heads. Compute gradient losses:
- `Ux_pred = ∂φ/∂x + ∂ψ/∂y` (approximated via node-to-node differences)
- `Uy_pred = ∂φ/∂y - ∂ψ/∂x`

The model is supervised on the reconstructed (Ux, Uy) derived from (φ, ψ), plus directly on p.

### Why it's different from failed continuity loss

The continuity PDE loss (∇·u = 0, listed as exhausted) was an EXPLICIT penalty term added to the existing velocity prediction. This Helmholtz approach CHANGES THE PREDICTION VARIABLE — the model never predicts (Ux, Uy) directly, only (φ, ψ). The incompressibility is structurally enforced, not penalized.

### Implementation challenge

Requires computing FD gradients on the unstructured mesh. This is the main risk — previously, "FD gradients on unstructured mesh too noisy" killed the pressure gradient aux head (exhausted). However, the crucial difference: for Helmholtz we're computing gradients of our OWN predictions (not noisy ground-truth), so the gradient computation is well-defined and smooth.

### Suggested hyperparameters

- Use KNN-based FD: for each node, fit local quadratic surface and take analytical gradient
- k=6 neighbors for gradient estimation
- Loss weighting: `L = L_pressure + 0.5 * L_velocity_helmholtz + 0.1 * L_phi_psi_smoothness`
- This is a high-complexity change; attempt only with an experienced student

---

## Priority 10 — `test-time-condition-ensemble`

**Target:** p_re, p_oodc
**Expected impact:** Medium
**Feasibility:** High (inference-only change)

### Hypothesis

At inference time, average predictions from 3-5 forward passes with small perturbations to the conditioning variables (Re, AoA). This is a simple form of test-time augmentation (TTA) that reduces variance in the model's predictions for OOD conditions without requiring any training changes.

The intuition: the model's learned function Re → Cp is smooth but noisy. Averaging over a small neighborhood of Re values produces a smoother, more robust estimate — particularly for extreme Re values in the OOD test set.

### Mechanism

At test time:
```python
# Multiple forward passes with perturbed Re
deltas = [-0.05, 0.0, +0.05]  # 5% Re perturbation
preds = []
for delta in deltas:
    x_perturbed = x.clone()
    x_perturbed[:, :, 13] = x[:, :, 13] + delta  # perturb log(Re)
    preds.append(model(x_perturbed))
pred = torch.stack(preds).mean(0)
```

### Why Re perturbation specifically

Re is the most OOD-sensitive conditioning variable. The `val_ood_re` split explicitly tests generalization across Re values not seen in training. Small Re perturbations (±5%) expand the effective support of the training distribution at inference, averaging over the model's uncertainty about the exact Re regime.

### Suggested hyperparameters

- 3-pass TTA: Re × {0.95, 1.0, 1.05}
- Or: 5-pass with Re × {0.90, 0.95, 1.0, 1.05, 1.10} and uniform averaging
- Apply only to log(Re) channel (index 13); leave AoA and geometry features unperturbed
- At training: add a small amount of Re noise augmentation (std=0.03 in log space) to prepare the model for these perturbations

---

## Priority 11 — `contrastive-re-regularization`

**Target:** p_re (OOD Re generalization)
**Expected impact:** Medium (speculative, no direct CFD precedent)
**Feasibility:** Medium

### Hypothesis

Add a contrastive auxiliary loss that penalizes large hidden-state distances between samples with similar Re but different geometry, while penalizing small distances between samples with very different Re. This explicitly trains the model to organize its internal representation so that the Re-axis is disentangled from the geometry-axis — critical for OOD Re generalization.

This is inspired by the InfoNCE loss from contrastive learning (Chen et al. 2020, SimCLR) applied to the physics conditioning space.

### Mechanism

For each batch, identify pairs of samples (i,j) with similar Re (|log(Re_i) - log(Re_j)| < 0.1). Penalize their hidden-state distance at the bottleneck layer. Simultaneously, for pairs with very different Re (|log(Re_i) - log(Re_j)| > 0.5), encourage larger hidden-state distance.

```python
L_contrastive = -log(
    exp(-||h_i - h_j|| / τ) for (i,j) with similar Re
    / Σ_k exp(-||h_i - h_k|| / τ)
)
```

### Implementation

Extract the mean-pooled hidden state from after the last TransolverBlock as the representation h. Apply the contrastive loss at the latent level, not the prediction level.

### Suggested hyperparameters

- Temperature τ = 0.5
- Loss weight: λ = 0.01 (small — this is auxiliary, should not dominate task loss)
- Only apply within-batch (no memory bank needed)
- Pairs: use batch-level sampling — for B=16, form ~B/2 = 8 pairs

---

## Priority 12 — `log-wall-distance-remap`

**Target:** p_in (viscous sublayer resolution)
**Expected impact:** Low-medium
**Feasibility:** Very high (trivial to implement)

### Hypothesis

The wall distance feature (from DSDF channels) is approximately uniformly distributed in linear space, but the physically relevant dynamics scale logarithmically with wall distance (boundary layer theory, von Karman's law). Replace the raw wall distance d with `tanh(d / δ_wall)` where δ_wall is a learnable scale parameter. This compresses the far-field (where little physics happens) and expands the near-wall region (where boundary layer dynamics are critical).

### Mechanism

In feature assembly: `d_mapped = tanh(d_raw / delta_wall)` where `delta_wall = F.softplus(self.log_delta_wall)` ensures positivity.

Initialize δ_wall to a physically meaningful value: ~0.01 chord lengths (roughly 1 boundary layer thickness at Re=200k). The `tanh` maps [0, ∞) → [0, 1), so the near-wall region [0, δ_wall] occupies half the output range.

### Suggested hyperparameters

- Single learnable parameter `log_delta_wall`, initialized to log(0.01)
- Apply to all DSDF channels (all 8 signed distance channels, not just the minimum)
- Can also try fixed tanh with δ_wall = 0.01 (no learning) as ablation

---

## Summary Table

| Priority | Slug | Target | Impact | Feasibility | Key bet |
|----------|------|--------|--------|-------------|---------|
| 1 | `panel-method-cp-feature` | p_in, p_tan | High | High | Physics baseline → model learns residuals |
| 2 | `fv-cell-area-loss-weight` | all | High | High | Correct structural mesh bias in loss (never tried) |
| 3 | `potential-flow-residual` | p_in, p_re | Med-high | Med | Target engineering: predict viscous residual not absolute p |
| 4 | `stochastic-depth-drop-path` | p_oodc, p_re | Med | Very high | Block-level dropout for OOD regularization |
| 5 | `wall-layer-bin-feature` | p_in, p_tan | Med | High | Learnable wall-layer regime embeddings |
| 6 | `hypernetwork-slice-routing` | p_tan, p_oodc | Med-high | Med | Config-conditioned routing weight generation |
| 7 | `ema-teacher-soft-label-distillation` | p_oodc, p_re | Med | Med | Mean Teacher temporal self-distillation |
| 8 | `surface-arclen-pe` | p_in, p_tan | Med | High | Arc-length Fourier PE for surface nodes |
| 9 | `helmholtz-velocity-decomposition` | p_tan, p_oodc | Med | Med-low | Structurally enforce incompressibility via prediction variable |
| 10 | `test-time-condition-ensemble` | p_re, p_oodc | Med | Very high | TTA via Re micro-perturbations at inference |
| 11 | `contrastive-re-regularization` | p_re | Med | Med | InfoNCE on latent space across Re values |
| 12 | `log-wall-distance-remap` | p_in | Low-med | Very high | tanh remap of wall distance to log-law scale |

## Recommended First Assignments

1. **`fv-cell-area-loss-weight`** — highest-confidence, ICML 2024 backed, targets structural bias. Simple to implement. This should have been Round 30 priority #1 and never got done.
2. **`panel-method-cp-feature`** — Springer 2025 direct validation. Physics-motivated, targets the largest absolute error metrics.
3. **`stochastic-depth-drop-path`** — 3 lines of code, strong theoretical basis, directly targets the p_oodc/p_re ensemble gap.
4. **`test-time-condition-ensemble`** — zero training change required, inference-only, immediate OOD improvement bet.
