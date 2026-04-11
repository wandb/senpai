# Bold Research Ideas — Round 25

_Generated: 2026-04-08_
_Responding to Issue #1860 directive: radical, bold, paradigm-shifting ideas only._
_No T_max adjustments. No loss weight tuning. No incremental tweaks._

---

## Context

Current best: val_loss ~0.3994 (PR #2213, wake deficit feature).
Hardest metrics: p_tan ~28.34 MAE (2.19x harder than p_in), p_oodc ~7.643.

What has consistently failed (do NOT repeat):
- Deeper/wider backbone, SRF head sizing
- Standard regularization: SAM, SWA, stochastic depth (early), dropout, MixStyle
- Optimizer tricks: SGDR, Huber, OHNM, Muon/Gram-NS, gradient centralization
- Feature additions without new physics: surface normals, curvature, arc-length PE, LE frame
- Cross-attention between foils: GALE, fore-aft cross-attn, cross-DSDF
- Panel-method inviscid Cp input (regressed p_tan)
- Bernoulli consistency loss, vorticity auxiliary target
- Multi-scale slice hierarchy, register tokens, MoE FFN, AdaLN
- Iterative refinement, deep supervision
- GEPS test-time adaptation, multi-resolution hash grid
- Hard sample replay, focal reweighting

What works: TE frame, wake deficit feature, PCGrad, asinh pressure transform, DCT frequency loss,
domain-specific heads (AftSRF), spatial bias (gap/stagger), DomainLayerNorm, GatedMLP.

The 8 ideas below operate at fundamentally different levels than anything attempted so far.

---

## Idea 1: Continuous Normalizing Flow over Surface Pressure — Generative Surrogate for p_tan

**Slug:** `cnf-surface-pressure`

**Target metric:** p_tan, p_oodc

**Rationale:**
The entire research programme treats this as a deterministic regression problem: one input geometry
maps to one output field. But the hardest metric — p_tan — is the pressure distribution on the
aft foil in tandem configurations. The difficulty here is not fundamentally about model capacity:
it is about the *conditional distribution* of aft-foil pressure given (geometry, Re, gap, stagger)
being multimodal or high-variance. When the model is forced to predict a single point estimate,
it minimizes MAE by regressing to the conditional mean — which may be far from any physically
realizable solution.

A Continuous Normalizing Flow (CNF) replaces the deterministic surface head with a flow-based
model that learns the full conditional distribution p(pressure_surface | backbone_hidden). At
inference time, sampling from this distribution and taking the mean (or the mode) of N=16 samples
produces a statistically consistent estimate that is more robust to multi-modality than the
point-estimate SRF. This is the approach used in protein structure prediction (AlphaFold2's
structure module uses a diffusion head for per-residue coordinates), now applied to surface
pressure.

This is NOT a full-field generative model — that would be computationally prohibitive. The CNF
is applied ONLY to the surface node outputs (M_s << N_total nodes), keeping computation tractable.

**Implementation:**
Replace `SurfaceRefinementHead` with a small CNF conditioned on the backbone hidden states:

1. Use a 4-layer ODE-driven coupling flow (e.g. RealNVP or a simple continuous-time flow) with
   backbone hidden as conditioning input. The flow operates in the 3D output space (Ux, Uy, p).
2. During training: maximize log-likelihood of the true (Ux, Uy, p) under the conditional flow.
   This replaces the current SRF MAE loss on surface nodes.
3. During inference: draw N=16 samples from the flow, compute their mean, use as the surface
   prediction. EMA applies to the flow parameters.

Simpler alternative (lower risk, recommended first pass): use a **flow matching** approach
(Lipman et al., 2022) — train a small MLP to predict the conditional velocity field mapping
noise -> target surface pressure. This is easier to train than full CNF and supports torch.compile.

```python
class SurfaceFlowHead(nn.Module):
    """Flow matching head: MLP that predicts velocity field for ODE."""
    def __init__(self, n_hidden=192, out_dim=3, t_embed_dim=16):
        # Input: [backbone_hidden (192), noisy_target (3), t_embedding (16)] -> velocity (3)
        self.net = nn.Sequential(
            nn.Linear(n_hidden + out_dim + t_embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )
    def forward(self, h, x_t, t):
        # h: [M_s, n_hidden], x_t: [M_s, 3] (noisy), t: [M_s] in [0,1]
        t_emb = fourier_time_embed(t, t_embed_dim)  # [M_s, t_embed_dim]
        return self.net(torch.cat([h, x_t, t_emb], dim=-1))

# Training loss: flow matching objective
def flow_matching_loss(head, h, y_true, n_t=4):
    t = torch.rand(h.shape[0], device=h.device)  # random timesteps
    x0 = torch.randn_like(y_true)   # noise
    x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * y_true  # linear interpolation
    v_pred = head(h, x_t, t)
    v_target = y_true - x0  # target velocity
    return F.mse_loss(v_pred, v_target)
```

New flag: `--cnf_surface_pressure` with `--cnf_n_samples 16` (inference samples).

**Risk:** Medium. Flow matching is well-tested in protein, molecule, and image domains. The
primary risk is that pressure surface predictions are NOT multimodal given backbone hidden state
(in which case the distribution collapses to a point and matches the SRF). If the backbone already
captures all the conditioning information needed for a sharp conditional, this adds computation
without gain. Check: does the SRF residual variance correlate with p_tan errors? If yes, this
is the right intervention.

**Literature:**
- Lipman et al. "Flow Matching for Generative Modeling" (2022, arXiv:2210.02747) — flow matching
  objective that avoids simulation of ODE during training; directly applicable.
- Tong et al. "Improving and Generalizing Flow-Matching" (2023, arXiv:2302.00482) — conditional
  flow matching for scientific domains.
- AlphaFold3 (Abramson et al., 2024) — uses diffusion head instead of deterministic regression
  for structure coordinates; directly analogous to our surface pressure problem.

---

## Idea 2: Graph Neural Network Boundary Layer Module — Physics-Informed Local Propagation

**Slug:** `gnn-boundary-layer`

**Target metric:** p_tan, p_in (surface near-wall accuracy)

**Rationale:**
The Transolver slice attention is a global operator — it pools nodes into slices, computes slice
interactions, and broadcasts back. This is powerful for global flow patterns but structurally
wrong for boundary layer physics: near-wall pressure/velocity gradients are dominated by LOCAL
propagation along the wall, not global attention. The boundary layer physics is a PDE with
characteristic length scales orders of magnitude smaller than the global domain.

A small Graph Neural Network (GNN) module operating ONLY on surface and near-surface nodes
would implement local message-passing that respects the boundary layer's locality. This is
fundamentally different from every cross-attention or global attention approach tried so far —
it is local, physics-respecting propagation.

Concretely: after the Transolver backbone produces hidden states for all nodes, apply 2-3 rounds
of GraphSAGE message passing among (a) surface nodes and (b) their k=4 nearest volume neighbors.
This lets near-wall information propagate locally before the SRF head applies its correction.

This is directly motivated by B-GNNs (arXiv:2503.18638) which showed 85% error reduction using
GNN message passing with local physics-informed inputs on airfoil meshes.

**Implementation:**
```python
class BoundaryLayerGNN(nn.Module):
    """Local GNN operating on surface + near-surface nodes."""
    def __init__(self, n_hidden=192, n_layers=2, k_neighbors=4):
        self.layers = nn.ModuleList([
            GraphSAGELayer(n_hidden, n_hidden) for _ in range(n_layers)
        ])

class GraphSAGELayer(nn.Module):
    def forward(self, h, edge_index):
        # Aggregate neighbor features, concatenate with self, project
        agg = scatter_mean(h[edge_index[1]], edge_index[0], dim=0)
        return F.silu(self.linear(torch.cat([h, agg], dim=-1)))
```

The edge graph (surface + k=4 volume neighbors) is precomputed from node coordinates per sample.
Critically: do NOT use torch_geometric — implement scatter_mean manually for torch.compile
compatibility.

New flag: `--gnn_boundary_layer` with `--gnn_layers 2` and `--gnn_k_neighbors 4`.

**Risk:** Medium. torch.compile compatibility requires avoiding dynamic shapes. The edge index
graph has variable size per sample — use dense adjacency matrices with masking rather than
sparse edge indices. Main risk: the 2 GNN layers add forward-pass computation but are
structurally compatible with existing PCGrad setup.

**Literature:**
- Baque et al. "B-GNNs: Boundary-Informed GNNs" (arXiv:2503.18638, 2025) — 85% error reduction
  on airfoil boundary layer prediction using local physics-informed GNN message passing.
- Hamilton et al. "Inductive Representation Learning on Large Graphs" (GraphSAGE, 2017,
  NeurIPS) — the specific aggregation scheme recommended here.
- Eliasof et al. "PDENets..." (2023) — GNN message passing as PDE discretization.

---

## Idea 3: Fourier Neural Operator Layer as Tandem Inter-Foil Coupling Block

**Slug:** `fno-inter-foil-coupling`

**Target metric:** p_tan

**Rationale:**
Every inter-foil coupling mechanism tried so far (fore-aft cross-attention, GALE, cross-DSDF)
uses attention between foil node features. These all failed. The fundamental reason may be that
attention lacks the right inductive bias for wake coupling physics: the wake is a spatially
structured, frequency-domain phenomenon. The fore foil sheds vortices at specific spatial
frequencies determined by its chord and angle of attack — and those frequencies interact with
the aft foil's leading edge in frequency space.

A Fourier Neural Operator (FNO) layer applied to the tandem gap region encodes this inter-foil
interaction as spectral convolution — exactly the right inductive bias for a periodic, structured
wake. Critically, this is NOT an FNO on the full field (that would require a uniform grid). It
is applied only in a narrow "coupling zone" between the two foils, where we can extract a 1D
chord-wise profile of hidden states and apply 1D spectral convolution.

This is a genuinely novel combination: Transolver for global field, FNO for localized spectral
inter-foil coupling. No prior experiment has attempted spectral coupling in the physical space
between the foils.

**Implementation:**
```python
class FNOCouplingLayer(nn.Module):
    """1D FNO applied to chord-wise hidden state profile in tandem gap."""
    def __init__(self, n_hidden=192, n_modes=16):
        self.n_modes = n_modes
        # Fourier weights: complex [n_hidden, n_hidden, n_modes]
        self.fourier_weight = nn.Parameter(
            torch.randn(n_hidden, n_hidden, n_modes, dtype=torch.cfloat) * 0.02
        )
        self.bypass = nn.Linear(n_hidden, n_hidden)  # residual bypass

    def forward(self, hidden, gap_node_mask, x_coords):
        # For tandem samples only:
        # 1. Extract hidden states for nodes in the inter-foil gap region
        # 2. Bin them into G=32 chord-wise grid cells (interpolate via nearest neighbor)
        # 3. Apply 1D spectral convolution in the chord direction
        # 4. Scatter corrected features back to gap nodes
        if not is_tandem:
            return hidden  # no-op for single-foil
        gap_hidden = hidden[gap_node_mask]  # [M_gap, n_hidden]
        # ... spectral convolution ...
        return hidden + correction  # residual update

# Add between TransolverBlock 1 and 2 (after backbone block 1, before block 2)
```

For torch.compile compatibility: the gap region grid is fixed size G=32 (pad if fewer nodes,
truncate if more — gap region node count is relatively stable across tandem geometries).

New flag: `--fno_inter_foil_coupling` with `--fno_modes 16` and `--fno_gap_bins 32`.
Apply only when `gap_magnitude > 0` (tandem samples).

**Risk:** Medium-high. The chord-wise binning introduces approximation error for highly refined
meshes near the foil surfaces. The key bet is that spectral modes in the gap region encode
wake frequency information that attention cannot capture. Risk mitigation: zero-initialize
`fourier_weight` so the module starts as identity.

**Literature:**
- Li et al. "Fourier Neural Operator for Parametric PDEs" (arXiv:2010.08895, 2020) — the
  foundational FNO paper; spectral convolution as the right inductive bias for periodic PDE solutions.
- Guibas et al. "Adaptive Fourier Neural Operators" (AFNO, arXiv:2111.13587, 2021) — efficient
  token-mixing via Fourier transform; applied to weather prediction.
- Hao et al. "GNOT" (arXiv:2208.00592, 2022) — combined GNN + Fourier for irregular mesh PDE.

---

## Idea 4: Consistency Training via Geometry Augmentation Self-Distillation

**Slug:** `geometry-consistency-distill`

**Target metric:** p_oodc, p_tan

**Rationale:**
The model generalizes poorly to OOD geometries (p_oodc). The core issue: the training set
contains limited geometric diversity, so the model overfits to the specific coordinate
distributions seen during training. Standard augmentation has been tried (chord-ratio, chord
flip, etc.) and failed.

A fundamentally different approach: **consistency training via self-distillation on augmented
geometry views**. For each training sample, create a geometrically augmented view by applying
small random perturbations to the non-airfoil-surface mesh nodes (volume nodes only — surface
shape is preserved). The model must produce CONSISTENT predictions for both the original and
augmented view. The augmented-view predictions are supervised by the EMA model's predictions
on the original view (not the ground truth labels) — this is the Mean Teacher / BYOL pattern.

This forces the model to learn representations that are invariant to volume mesh topology
changes — exactly what we need for OOD geometry generalization. The geometry augmentation
is volume-node coordinate jitter (σ=0.005 chord lengths), which changes the input features
without changing the underlying physical solution.

This is inspired by CutMix/MixUp-style consistency in computer vision and Mean Teacher
semi-supervised learning (Tarvainen & Valpola, 2017), but applied to mesh geometry rather
than image pixels.

**Implementation:**
```python
# In training loop, after computing base loss:
if cfg.geometry_consistency_distill and epoch > 30:
    # Create augmented view: jitter volume node coordinates
    vol_mask = ~is_surface  # volume nodes only
    x_aug = x.clone()
    x_aug[:, vol_mask, :2] += torch.randn_like(x_aug[:, vol_mask, :2]) * 0.005
    # Recompute DSDF features for augmented coords (can use approximate update)
    x_aug_features = recompute_dsdf_approx(x_aug)  # fast approximation

    # Forward pass through student model with augmented view
    pred_aug = model(x_aug_features, ...)
    # Targets: EMA model predictions on ORIGINAL view (no gradients)
    with torch.no_grad():
        pred_orig_ema = ema_model(x_features, ...)
    # Consistency loss on SURFACE nodes only (where predictions matter)
    loss_consistency = F.mse_loss(pred_aug[surf_mask], pred_orig_ema[surf_mask].detach())
    loss = loss + cfg.consistency_weight * loss_consistency  # weight=0.1
```

New flags: `--geometry_consistency_distill`, `--consistency_weight 0.1`,
`--consistency_start_epoch 30`, `--consistency_jitter_std 0.005`.

**Risk:** Medium. The key risk is that `recompute_dsdf_approx` is expensive. Approximation:
use the existing DSDF features with only the volume-node coordinate differences updated (not
a full DSDF recomputation). Alternatively, drop DSDF recomputation and only jitter the raw (x,y)
coordinates — simpler and still valid for the consistency objective.

**Literature:**
- Tarvainen & Valpola "Mean Teachers are Better Role Models" (2017, NeurIPS) — the foundation
  of EMA-based self-distillation for regularization; exactly the pattern here.
- Sohn et al. "FixMatch" (2020, NeurIPS) — consistency training with strong augmentation;
  motivates the two-view consistency setup.
- Wang et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization" (2020) — test-time
  self-consistency; our approach is the training-time version.
- Raonic et al. "Convolutional Neural Operators" (2023) — shows mesh-invariant training
  improves PDE surrogate OOD generalization.

---

## Idea 5: Pressure Gradient Field as Auxiliary Prediction Target (with Shared Encoder)

**Slug:** `pressure-gradient-aux-head`

**Target metric:** p_tan, p_in

**Rationale:**
The model currently predicts (Ux, Uy, p). The pressure Poisson equation for incompressible flow
states: ∇²p = -ρ(∂u_i/∂x_j)(∂u_j/∂x_i). This means the pressure field is NOT independent of
velocity — it is fully determined by the velocity gradient tensor. By adding an auxiliary
prediction target of (∂p/∂x, ∂p/∂y) — the pressure gradient — and supervising it against finite
differences of the ground-truth pressure field, we force the model to learn pressure
representations consistent with the velocity field through the coupling imposed by the shared
encoder.

This is fundamentally different from the Bernoulli consistency loss (tried, failed) because:
1. Bernoulli is a point-wise algebraic constraint; pressure gradient is a spatial differential operator
2. The gradient field captures the direction and magnitude of pressure variation — exactly the
   information needed to predict the suction peak (LE pressure gradient) and pressure recovery
   (TE pressure gradient) that dominate p_tan errors
3. The auxiliary head operates on volume nodes (where gradients are meaningful), not surface nodes

The gradient supervision provides a "gradient matching" signal that is orthogonal to the MAE
loss on pressure values — it penalizes errors in the *shape* of the pressure field, not just
its magnitude at individual nodes.

**Implementation:**
```python
class PressureGradientHead(nn.Module):
    """Auxiliary head predicting (dp/dx, dp/dy) from backbone hidden state."""
    def __init__(self, n_hidden=192):
        self.net = nn.Sequential(
            nn.Linear(n_hidden, 64), nn.SiLU(), nn.Linear(64, 2)
        )

# Ground truth gradients (precomputed per batch from GT pressure):
def finite_diff_pressure_gradient(p_gt, pos, neighbor_idx):
    """Approximate ∇p at each node via neighbor finite differences."""
    dp_dx, dp_dy = [], []
    for k in range(K_neighbors):
        dx = pos[neighbor_idx[:, k], 0] - pos[:, 0]
        dy = pos[neighbor_idx[:, k], 1] - pos[:, 1]
        dp = p_gt[neighbor_idx[:, k]] - p_gt
        # Weighted least squares normal equations
        ...
    return torch.stack([dp_dx, dp_dy], dim=-1)  # [N_vol, 2]

# Training loss:
grad_pred = pressure_grad_head(hidden[vol_mask])  # [N_vol, 2]
grad_target = finite_diff_pressure_gradient(p_gt[vol_mask], pos[vol_mask], neighbors)
loss_grad = F.l1_loss(grad_pred, grad_target) * cfg.grad_weight  # grad_weight=0.05
```

New flags: `--pressure_gradient_aux`, `--grad_weight 0.05`.
Gradient targets are computed per-batch from ground truth (no precomputation needed).
PCGrad: add this as a 4th gradient task alongside tandem/single-foil/extreme-Re.

**Risk:** Low-medium. The auxiliary head is structurally isolated — its gradients pass through
the shared encoder but the head itself can be zeroed out at inference with no impact on the
main prediction. The finite-difference approximation introduces noise but the L1 loss is
robust to this. Main risk: the pressure gradient magnitude varies enormously across the flow field
(near-wall vs far-field), so normalization is critical. Use asinh transform on gradient targets too.

**Literature:**
- Baque et al. "DeepSDF for CFD" — pressure gradient fields as auxiliary supervision in mesh-based
  surrogates.
- Stachenfeld et al. "Learned Simulators for Turbulence" (2021, DeepMind) — physics-consistent
  auxiliary targets that constrain the shared encoder.
- Pan et al. "Physics-Informed PointNet" (2022) — gradient-based auxiliary losses improve
  generalization on irregular meshes.
- Fathony et al. "Multiplicative Filter Networks" (2021) — spectral auxiliary targets for PDE solutions.

---

## Idea 6: Learned Per-Sample Physics Scaling (Hypernetwork over Re, Gap, Stagger)

**Slug:** `hypernetwork-physics-scaling`

**Target metric:** p_oodc, p_re, p_tan

**Rationale:**
The current model handles multi-condition generalization via: (1) PCGrad separating gradients
across domains, (2) gap/stagger spatial bias in slice routing, (3) DomainLayerNorm (tandem vs
single). These are all fixed, pre-specified mechanisms. A fundamentally different approach:
train a small **hypernetwork** that generates the weight perturbations for the SRF head
(and optionally the backbone output LayerNorm) conditioned on the physics scalars (Re, gap, stagger,
is_tandem).

The hypernetwork learns to produce sample-specific affine transforms (scale + bias) for the SRF
head's internal activations — this is equivalent to conditioning the SRF on the physics regime
without sharing parameters between regimes. For the extreme-Re OOD case, the hypernetwork can
generate a different "adaptation" of the SRF than for in-distribution Re.

This is directly inspired by HyperNetworks (Ha et al., 2016) and FILM (Feature-wise Linear
Modulation, Perez et al., 2018) — but applied to physics scalars rather than language conditioning.
The prior DomainLayerNorm only gave 2 modes (tandem/single); the hypernetwork gives a continuous
family of adaptations over the full (Re, gap, stagger) parameter space.

**Implementation:**
```python
class PhysicsHyperNetwork(nn.Module):
    """Generates (scale, bias) for SRF head layers from physics scalars."""
    def __init__(self, physics_dim=4, srf_hidden=192, n_srf_layers=3):
        # physics_dim: [log(Re/1e5), gap_norm, stagger_norm, is_tandem]
        # Generates scale + bias for each of the n_srf_layers linear layers
        self.net = nn.Sequential(
            nn.Linear(physics_dim, 32), nn.SiLU(),
            nn.Linear(32, n_srf_layers * srf_hidden * 2)  # scale + bias per layer
        )

    def forward(self, physics_scalars):
        # Returns dict: {layer_i: (scale [srf_hidden], bias [srf_hidden])}
        params = self.net(physics_scalars)  # [B, n_layers * hidden * 2]
        ...

# Apply in SRF head forward:
class SurfaceRefinementHead(nn.Module):
    def forward(self, x, physics_params=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if physics_params is not None:
                scale, bias = physics_params[i]  # [n_hidden]
                x = x * (1 + scale) + bias  # residual scaling
            x = F.silu(x)
        return self.layers[-1](x)
```

Initialize hypernetwork to produce near-zero scale and bias (so initial behavior matches
existing SRF). Physics scalars: [log(Re/1e5), gap_normalized, stagger_normalized, is_tandem_float].

New flag: `--hypernetwork_physics_scaling` with `--hypernet_hidden 32`.

**Risk:** Medium. The hypernetwork introduces a small parameter overhead (~5K params for
32-hidden hypernet) and is conditionally applied per batch. Interaction with PCGrad: the
hypernetwork parameters receive gradients from all task losses — this is desirable (it learns
a universal physics-conditioning). Main risk: the hypernet may overfit to the Re/gap distribution
seen in training, failing to generalize to truly OOD conditions.

**Literature:**
- Ha et al. "HyperNetworks" (2016, arXiv:1609.09106) — generating weight matrices from
  conditioning networks; the foundational approach.
- Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" (2018) — affine
  conditioning on external scalars; directly applicable here.
- Garnelo & Shanahan "Reconciling deep learning with symbolic AI" (2019) — conditioning on
  physics scalars for better compositional generalization.
- BlendedNet++ (arXiv:2512.03280) — FiLMNet variant applied to CFD surrogates; showed strong
  performance on multi-condition aerodynamics benchmarks.

---

## Idea 7: Stochastic Tandem Augmentation — Random Gap/Stagger Interpolation

**Slug:** `tandem-geom-interpolation`

**Target metric:** p_tan

**Rationale:**
The tandem training set has finite coverage of (gap, stagger) space. The model has never seen
gap=0.7c, stagger=0.15c during training, so it interpolates imperfectly. A targeted data
augmentation that synthesizes new (gap, stagger) values by interpolating between existing tandem
training samples would dramatically increase coverage of the tandem configuration space.

This is NOT generic mixup (tried, failed). It is *physics-preserving interpolation* for the
specific case of tandem configurations: given two tandem training samples A (gap=g1, stagger=s1)
and B (gap=g2, stagger=s2), create a synthetic sample C by:
1. Interpolating the global flow scalars: gap_C = α*g1 + (1-α)*g2, stagger_C = α*s1 + (1-α)*s2
2. Updating the TE coordinate frame features and wake deficit feature for the interpolated geometry
   (these are computed from gap/stagger directly — no mesh re-meshing needed)
3. Using the INTERPOLATED labels: y_C = α*y_A + (1-α)*y_B (since pressure varies smoothly
   with gap/stagger for subsonic flow)
4. Keeping all other features (DSDF, surface shape) from sample A

The physics justification: for subsonic, attached-flow tandem configurations, the aft-foil
pressure distribution varies SMOOTHLY and nearly linearly with small changes in gap and stagger
— this is a fundamental result from linearized thin-airfoil theory for tandem arrangements.
Interpolation produces physically plausible labels within the smooth manifold.

**Implementation:**
```python
# In training loop, with probability p_aug=0.3 for tandem batches:
if is_tandem and random.random() < cfg.tandem_aug_prob:
    # Sample another tandem example from the batch or from a replay buffer
    idx_b = random.choice(tandem_indices)
    alpha = random.uniform(0.2, 0.8)
    # Interpolate flow scalars
    gap_aug = alpha * gap[b] + (1-alpha) * gap[idx_b]
    stagger_aug = alpha * stagger[b] + (1-alpha) * stagger[idx_b]
    # Recompute gap/stagger-dependent features (TE frame, wake deficit, spatial bias)
    x_aug = recompute_tandem_features(x[b], gap_aug, stagger_aug)  # fast, no DSDF
    y_aug = alpha * y[b] + (1-alpha) * y[idx_b]  # interpolated labels
    # Randomly swap one of the batch examples with the augmented one
    x[b], y[b] = x_aug, y_aug
```

New flags: `--tandem_geom_interpolation`, `--tandem_aug_prob 0.3`, `--tandem_aug_start_epoch 20`.

Key difference from prior tandem mixup (PR #2246): that approach mixed features+labels globally
across ALL tandem samples with no geometric correction. This approach ONLY mixes gap/stagger
scalars and updates the dependent features (TE frame, wake deficit) consistently — preserving
the physical coherence of the sample.

**Risk:** Low-medium. The physical interpolation assumption (linear variation with gap/stagger)
is valid for subsonic attached flow — the regime most of the training data is in. It breaks
for separated flow at extreme angles of attack. Start with `--tandem_aug_prob 0.2` to limit
exposure and validate on val_tandem_transfer first.

**Literature:**
- Garnier et al. "A review on deep reinforcement learning for fluid mechanics" (2021) — discusses
  physics-respecting augmentation for CFD data.
- Benton et al. "A Simple Data Augmentation Algorithm for Improving" (2020) — interpolation-based
  augmentation preserving class/physics manifold structure.
- Kochkov et al. "Machine learning-accelerated computational fluid dynamics" (2021, PNAS) —
  augmentation strategies for CFD training data; geometry interpolation validated for NS flows.
- Thin-airfoil theory for tandem foils (classical aerodynamics) — analytical validation that
  pressure is smooth/linear in gap and stagger for attached subsonic flow.

---

## Idea 8: Spectral Whitening of Input Features — Decorrelating the Feature Covariance

**Slug:** `spectral-feature-whitening`

**Target metric:** p_oodc, p_tan

**Rationale:**
The current input feature space (24 dimensions) is highly correlated: DSDF features, TE frame
distances, wake deficit, Re are all functions of a small number of underlying physical parameters
(chord, angle, Re, gap, stagger). This redundancy creates a poorly conditioned optimization
landscape where the model must first "undo" correlations before learning the physics mapping.

Per-foil whitening (PR #2261, in-flight) applies standard per-feature normalization. This is
incremental. A fundamentally stronger approach: **ZCA whitening** of the full 24-dim input
feature covariance matrix. ZCA (Zero-phase Component Analysis) decorrelates ALL features
simultaneously using the covariance matrix eigenvectors — producing a feature space where all
directions are equally informative, which is the optimal conditioning for gradient-based
optimization.

The insight: ZCA whitening is NOT just normalization. It is a rotation in feature space that
aligns the feature axes with the principal components of variation in the training data. For
our dataset, the dominant variation is (Re, angle of attack) — ZCA will implicitly weight these
variations appropriately. For OOD conditions, the ZCA transform ensures that unusual feature
combinations (OOD Re, unusual gap) are represented in a space where the model has been trained
to handle ALL directions equally.

This is computationally free at inference time (linear transform applied to inputs) and requires
no architectural changes. It is a change to the DATA REPRESENTATION level, which is the level
at which most of our successful improvements have operated.

**Implementation:**
```python
# Precompute ZCA matrix from training data (once, before training):
# Collect all training samples' input features
X_train = []  # [N_total, 24]
for batch in train_loader:
    X_train.append(batch.x.reshape(-1, 24))
X = torch.cat(X_train)  # [N_total_nodes, 24]

# Compute covariance and ZCA transform
mu = X.mean(0)  # [24]
X_centered = X - mu
C = (X_centered.T @ X_centered) / len(X_centered)  # [24, 24] covariance
U, S, Vt = torch.linalg.svd(C)
eps = 1e-5
W_zca = U @ torch.diag(1.0 / (S + eps).sqrt()) @ U.T  # [24, 24] ZCA matrix

# Apply at input (replaces current per-feature normalization):
x_whitened = (x - mu) @ W_zca.T  # [B, N, 24]
```

New flag: `--zca_whitening`. The ZCA matrix is precomputed from training data and saved as a
buffer. It replaces the existing per-feature normalization.

**Risk:** Low. ZCA is a linear transform — it cannot make predictions worse (the model can
learn to invert it if ZCA is suboptimal). The risk is numerical: if any principal component
has near-zero variance, the whitening amplifies noise in that direction. Fix: use eps=1e-5
for the singular value denominator. Also: ZCA changes the learning rate sensitivity — may
need to reduce lr slightly if training is unstable.

**Literature:**
- Bell & Sejnowski "An Information-Maximization Approach to Blind Separation" (1995) — ZCA
  origins; the theoretical foundation for decorrelating input features.
- LeCun et al. "Efficient BackProp" (1998) — whitening inputs is the optimal preprocessing
  for gradient descent; dramatically improves conditioning.
- Huang & LeCun "Large-scale Learning with SVM and Convolutional for Generic Object
  Categorization" — ZCA whitening as standard preprocessing in deep learning.
- Cogswell et al. "Reducing Overfitting in Deep Networks by Decorrelating Representations"
  (2015) — decorrelation as regularization; the DeCov loss; shows decorrelated features
  generalize better, directly supporting use for p_oodc.

---

## Summary

| Idea | Slug | Target | Risk | Key bet |
|------|------|--------|------|---------|
| 1 | `cnf-surface-pressure` | p_tan, p_oodc | Medium | Surface pressure is multimodal → generative head beats point estimate |
| 2 | `gnn-boundary-layer` | p_tan, p_in | Medium | Local GNN propagation respects boundary layer locality better than global attention |
| 3 | `fno-inter-foil-coupling` | p_tan | Med-High | Wake coupling is a spectral phenomenon → FNO has the right inductive bias |
| 4 | `geometry-consistency-distill` | p_oodc, p_tan | Medium | Volume mesh jitter + EMA self-distillation forces mesh-invariant representations |
| 5 | `pressure-gradient-aux-head` | p_tan, p_in | Low-Med | Gradient field supervision teaches the model the *shape* of pressure, not just values |
| 6 | `hypernetwork-physics-scaling` | p_oodc, p_re, p_tan | Medium | Continuous Re/gap conditioning via hypernet generalizes beyond 2-mode DomainLayerNorm |
| 7 | `tandem-geom-interpolation` | p_tan | Low-Med | Physics-preserving gap/stagger interpolation synthesizes new tandem configurations |
| 8 | `spectral-feature-whitening` | p_oodc, p_tan | Low | ZCA input decorrelation is optimal gradient conditioning; free at inference |

**Priority for assignment:**
1. `spectral-feature-whitening` — lowest risk, data-representation level, directly in the success lineage
2. `pressure-gradient-aux-head` — auxiliary head isolated from main prediction; low regression risk
3. `tandem-geom-interpolation` — physics-motivated augmentation targeting the hardest metric
4. `hypernetwork-physics-scaling` — continuous physics conditioning, extends DomainLayerNorm
5. `gnn-boundary-layer` — motivated by B-GNNs 85% error reduction; needs torch.compile care
6. `geometry-consistency-distill` — self-distillation for OOD; higher implementation complexity
7. `cnf-surface-pressure` — highest novelty; use flow matching variant (simpler than full CNF)
8. `fno-inter-foil-coupling` — most architecturally invasive; highest risk
