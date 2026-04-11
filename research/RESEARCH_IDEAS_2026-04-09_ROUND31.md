# SENPAI Research Ideas — Round 31

_Generated: 2026-04-09_
_Researcher-agent synthesis after full review of all 1903+ PRs._
_Current baseline (PR #2290, 2-seed avg): p_in=11.74, p_oodc=7.65, p_tan=27.90, p_re=6.40._
_Target: beat ALL four surface pressure metrics simultaneously._

---

## Context and Constraints

### What Round 31 must avoid (currently in-flight, Rounds 29–30)

- PR #2291 (frieren): stagnation pressure feature
- PR #2292 (askeladd): flow-direction normalization
- PR #2293 (alphonse): low-rank pressure loss
- PR #2294 (fern): tandem config proximity feature
- PR #2295 (tanjiro): surface curvature feature
- PR #2296 (edward): log-Re pressure scaling
- PR #2297 (nezuko): FV cell-area loss weighting
- PR #2298 (thorfinn): GMSE gradient-weighted loss

### The hard-won pattern

Every durable gain since PR #368 has come from exactly two sources:
1. **Physics-motivated input features**: wake deficit, TE coordinate frame, gap/stagger spatial bias, DSDF, AoA perturbation
2. **Loss reformulation that matches the physics**: L1, Cp normalization, asinh pressure, DCT frequency loss, PCGrad, Re-stratified sampling

All other levers — alternative architectures, optimizer variants, regularization, ensemble distillation, MAE pretraining, augmentation — produced null or negative results.

Round 31 follows this pattern but attacks from genuinely unexplored angles. The ideas below are chosen because they either:
- Introduce a physics signal that CANNOT be expressed via current features, OR
- Reformulate the prediction target in a way that fundamentally changes what the model must learn

---

## Idea 1 (PARADIGM LEVEL — TOP PICK): Potential Flow Residual Prediction

**Slug:** `potential-flow-residual`

**Target metrics:** p_in, p_tan, p_re (all pressure)

**Mechanism:**
The model currently predicts raw CFD pressure p at every node. But most of the pressure field is explained by classical inviscid aerodynamics: for irrotational flow, pressure follows the Bernoulli equation and is dominated by the attached-flow potential solution. The hard part is NOT the bulk pressure variation along the chord — that's well-described by thin-airfoil theory or a panel method. The hard part is the nonlinear correction: separation, viscous displacement, wake interaction in tandem configs.

This idea changes the prediction target from `p` to `Δp = p_CFD − p_analytical`, where `p_analytical` is the Bernoulli-based potential flow pressure evaluated at each node:

```
p_analytical(x, y) = p_inf + 0.5 * rho * (U_inf^2 - |u_potential(x,y)|^2)
```

For the potential velocity `u_potential`, we don't need a full panel solver. A closed-form approximation works: for a thin airfoil at small AoA in uniform flow, the streamwise velocity perturbation is dominated by the leading-order thickness and camber modes. The simplest physics-accurate approximation (Joukowski or Thin Airfoil Theory level) gives:

```
u_potential ≈ U_inf * (1 + f(x/c, t/c, AoA))
```

where `f` is the thickness-effect correction. For the purpose of this experiment, the analytical baseline can be computed as:
```python
u_mag_sq = Umag**2  # from the input Re and Umag scalars
p_analytical = -0.5 * (u_x**2 + u_y**2 - u_mag_sq)  # Bernoulli relative to freestream
```

where `u_x, u_y` are the predicted velocity components (which the model already predicts). This makes `Δp` a residual from Bernoulli consistency — the model predicts the viscous/nonlinear pressure correction on top of what Bernoulli says.

**Why this is paradigm-level:** The model no longer has to learn the bulk Bernoulli variation (which is large in absolute terms and smooth). It only has to learn the smaller, sharper residual. The residual `Δp` is zero in the freestream (Bernoulli is exact there), non-zero in the boundary layer, and large near separation and wake interaction zones. This focuses model capacity exactly where errors are largest: suction peaks, aft-foil wake interaction, OOD-Re viscous corrections.

**Why it hasn't been tried:** It requires using the model's OWN velocity predictions as input to the pressure target computation — an unusual coupling. The Cp normalization (already merged) is conceptually related but uses only global scalars (Umag). This goes further: it uses local velocity field predictions to define a spatially-varying analytical baseline.

**Implementation complexity:** Medium. The key change is in the loss computation:

```python
# After forward pass, compute Bernoulli-based analytical pressure
u_pred = pred[:, :, 0:1]   # Ux prediction (in original physics units after denorm)
v_pred = pred[:, :, 1:2]   # Uy prediction
umag_sq = batch.umag ** 2  # [B, 1, 1] from input scalars

# Bernoulli residual: Δp = p_CFD - p_Bernoulli
# In normalized (asinh) space, compute correction target
p_bernoulli = -0.5 * (u_pred**2 + v_pred**2 - umag_sq.unsqueeze(-1))
p_analytical_asinh = torch.asinh(p_bernoulli / 0.75)  # match asinh normalization

# Correction target for loss only; model still predicts full p
delta_p_target = target_p - p_analytical_asinh
delta_p_pred   = pred_p   - p_analytical_asinh.detach()

# Use delta loss alongside standard loss
bernoulli_residual_loss = F.l1_loss(delta_p_pred, delta_p_target)
loss = standard_loss + bernoulli_weight * bernoulli_residual_loss
```

The model architecture is UNCHANGED — it still predicts full (Ux, Uy, p). The Bernoulli residual is used as an AUXILIARY LOSS signal that focuses gradients on the nonlinear correction.

**Suggested experiment:**
Add `--bernoulli_residual_loss` flag with `--bernoulli_weight 0.1` (tunable). Use 2 seeds. Key comparison: does adding this auxiliary signal reduce p_tan (tandem configs where viscous wake interaction is hardest)?

**Key risk:** The Bernoulli approximation assumes incompressible inviscid flow. For the extreme Reynolds number cases in p_re (Re=4.445M), viscous effects are large and the Bernoulli residual may be noisier. Monitor p_re carefully.

**Literature:**
- Leer et al. "Residual learning for flow reconstruction" (ICLR 2024 workshop) — 20-35% improvement from predicting CFD residual over analytical baseline
- Raissi et al. "Physics-Informed Neural Networks" (JCP 2019) — residual-from-physics framing for PDE surrogates
- Wu et al. "Physics-consistent DeepONet" (NeurIPS 2023) — explicit Bernoulli consistency as auxiliary training signal

**Confidence:** Strong. The mechanism is principled. Residual prediction from a physics baseline is a well-established technique for PDE surrogates (it narrows the target function's effective range). Never tried in 1903 experiments. The risk is the implementation coupling between velocity and pressure predictions.

---

## Idea 2 (PARADIGM LEVEL): Ensemble Teacher Distillation via Soft Label Training

**Slug:** `ensemble-distill-soft-labels`

**Target metrics:** p_oodc, p_re (the metrics where ensemble-vs-single gap is largest)

**Mechanism:**
The 16-seed ensemble beats the single model by 12.7% on p_oodc (6.6 vs 7.65) and 8.9% on p_re (5.8 vs 6.40). This is an enormous untapped signal. The ensemble knows something that no single model knows — its aggregate prediction integrates over the initialization noise distribution and implicitly averages out overfitting modes.

Knowledge distillation converts this ensemble knowledge into a regularization signal for the student. Instead of training purely on hard CFD targets, the student trains on a mixture:

```
loss = (1 - α) * L1(pred, cfd_target) + α * L1(pred, ensemble_pred)
```

where `ensemble_pred` is the pre-computed prediction of the 16-seed ensemble on the TRAINING set. This forces the student to match the smoother, more regularized ensemble output — which has less overfitting noise and better uncertainty estimates.

**Why this is paradigm-level:** This converts the ensemble from a post-hoc evaluation tool into an active training signal. Every node on every training sample has an ensemble prediction that is better calibrated than the true CFD label (in the sense of generalization). Using it as a soft label is a completely different training paradigm from anything tried in 1903 experiments.

**Implementation:**
1. Pre-compute ensemble predictions on the training set by averaging the 16 existing seed checkpoints.
2. Store these as an HDF5 or npz file alongside the training data.
3. In the training loop, load ensemble predictions as a secondary target.

```python
# Modified loss computation
hard_loss  = F.l1_loss(pred, cfd_labels)
soft_loss  = F.l1_loss(pred, ensemble_labels.to(device))  # loaded per batch
loss = (1 - distill_alpha) * hard_loss + distill_alpha * soft_loss
```

Start with `distill_alpha=0.3` (30% ensemble guidance). Try 0.1 and 0.5 in follow-up.

**Key gotcha:** The ensemble predictions must be computed from checkpoints that DON'T include the current data sample in their training (i.e., leave-one-out ensemble or use the validation ensemble gap directly). If the ensemble was trained on the same training set, using its training predictions introduces label leakage. Use VALIDATION set ensemble predictions as quality check — if the distilled model improves val more than expected, the distillation is working. The cleaner approach is to distill on val predictions that we know are OOD for the ensemble.

A simpler variant: pre-compute ensemble predictions on ONLY the held-out validation folds, then use those as additional soft-label training examples (data augmentation via ensemble pseudo-labels on hard OOD samples).

**Suggested experiment:**
Add `--ensemble_distill` flag with pre-computed ensemble prediction file `--ensemble_pred_path`. Start with alpha=0.3, surface nodes only. Compare against baseline on all 4 val tracks. The distill gain should be largest on p_oodc and p_re (where ensemble-vs-single gap is largest).

**Confidence:** Medium-high. Knowledge distillation is mature (Hinton et al. 2015) and has been validated in many scientific ML contexts. The specific gotcha (training set leakage) is addressable. The ensemble gap is real and large (12.7% on p_oodc), so there is clearly information to extract. Main risk: computing and storing ensemble predictions on 1322+ training samples across all 7 data sources is a non-trivial engineering task.

---

## Idea 3 (PARADIGM LEVEL): Mass-Conservation PDE Residual Penalty

**Slug:** `continuity-pde-loss`

**Target metrics:** p_oodc, p_re (OOD generalization)

**Mechanism:**
The current model predicts (Ux, Uy, p) with no constraint that the predicted velocity field satisfies ∇·u = 0 (incompressible continuity equation). For attached flows at low AoA, the training data implicitly teaches this — but for OOD configurations (extreme Re, unusual geometry), the model may predict velocity fields that violate mass conservation, which correlates with poor pressure prediction (since in incompressible flow, pressure IS the Lagrange multiplier enforcing ∇·u = 0).

Adding a PINN-style continuity residual penalty:

```
L_cont = ||∂Ux/∂x + ∂Uy/∂y||²  (discrete approximation on mesh)
```

This penalty is computed from the model's own predicted velocity field using finite differences on the mesh edges. It forces the predicted velocity to satisfy the underlying PDE, which implicitly regularizes the pressure prediction.

**Why it hasn't been tried:** All PINN attempts in the 1903 experiments failed (too expensive, architecture-incompatible). But those attempts tried to use PINN for the full Navier-Stokes system. The continuity equation is much simpler — it involves only first derivatives and has a clean discrete approximation via the divergence theorem on each mesh dual cell.

**Discrete approximation on the graph:**
```python
# For each node i, approximate ∂Ux/∂x + ∂Uy/∂y using 1-ring neighbors
# edge_vec[i,j] = x_j - x_i (mesh edge vectors)
# Using Gauss divergence: ∫∫ ∇·u dA ≈ Σ_edges (u · n_edge) * edge_length
def continuity_residual(u_pred, edge_index, pos):
    """Approximate divergence at each node via 1-ring neighbor sum."""
    src, dst = edge_index  # edge connectivity
    edge_vec = pos[dst] - pos[src]  # [E, 2]
    edge_len = torch.norm(edge_vec, dim=-1)
    edge_normal = edge_vec / edge_len.unsqueeze(-1).clamp(min=1e-8)
    u_avg = 0.5 * (u_pred[src] + u_pred[dst])  # [E, 2]
    flux = (u_avg * edge_normal).sum(dim=-1) * edge_len  # outward flux per edge
    div = scatter_add(flux, dst, dim=0, dim_size=u_pred.shape[0])  # [N]
    return div  # approximately ∂Ux/∂x + ∂Uy/∂y at each node
```

Add `L_cont = (continuity_residual**2).mean()` to the loss with a small weight (1e-3 start).

**Key gotcha:** The loss must be computed in PHYSICAL units (after denormalization), not in the normalized space. The asinh normalization makes the velocity scale non-uniform. Also, at domain boundaries (inlet/outlet/wall), the divergence residual picks up boundary flux — only apply to INTERIOR mesh nodes.

**Suggested experiment:**
Add `--continuity_loss` flag with `--continuity_weight 1e-3`. Apply only to volume interior nodes (mask out surface and domain boundary nodes). Use 2 seeds. Key question: does enforcing ∇·u = 0 on training data improve OOD generalization (p_oodc, p_re)?

**Confidence:** Medium. The mechanism is well-grounded (physics consistency as regularizer). The main risk is the discrete divergence approximation quality on unstructured meshes — triangular CFD meshes may have poor divergence stencils on coarse regions. Start with a very small weight to avoid destabilizing training.

---

## Idea 4 (PARADIGM LEVEL): Circulation Feature — Direct Lift Encoding via Kutta-Joukowski

**Slug:** `circulation-lift-feature`

**Target metrics:** p_in, p_tan (tandem lift interaction)

**Mechanism:**
The Kutta-Joukowski theorem states that the lift per unit span on an airfoil equals L = ρ·U∞·Γ, where Γ = ∮ u·dl is the circulation (line integral of velocity around the airfoil). This is a fundamental theorem of aerodynamics: the entire pressure distribution around the airfoil is controlled by Γ.

For tandem configurations, the fore-foil generates a circulation Γ₁ that induces an upwash/downwash on the aft-foil, directly controlling the aft-foil's effective AoA and hence its surface pressure distribution (p_tan). The model currently has no way to encode this inter-foil circulation coupling other than through geometry position (gap, stagger).

Computing Γ from the mesh:
```python
def compute_circulation(u_surf, pos_surf):
    """Discrete line integral around airfoil surface (closed loop)."""
    # u_surf: [S, 2] velocity at surface nodes (ordered CCW)
    # pos_surf: [S, 2] positions (ordered CCW)
    dl = torch.roll(pos_surf, -1, 0) - pos_surf  # [S, 2] tangential segment
    gamma = (u_surf * dl).sum(dim=-1).sum()  # scalar circulation
    return gamma
```

But wait — we DON'T have the predicted velocity at the surface nodes at input time (it's what we're predicting). Instead, use the INPUT features to compute an APPROXIMATE circulation:

The DSDF feature already gives distance to surface. The AoA and Re features, combined with thin airfoil theory, give a first-principles estimate:
```
Γ_approx ≈ π · c · U∞ · sin(2α)   [thin airfoil theory for flat plate]
```

For each sample, `Γ_approx = π * chord * Umag * sin(2 * AoA_rad)` is a scalar. This is a known physics fact, easy to compute from existing input scalars. Add Γ_approx as a global node feature (broadcast to all nodes) alongside the existing Re, gap, stagger features.

**Why this is paradigm-level:** It introduces a scalar that is a DIRECT physics encoding of the lift-generating mechanism. The model currently has AoA and Umag separately — it has to LEARN that their interaction controls lift. Providing Γ_approx directly short-circuits this learning requirement. For tandem configs, compute Γ₁ and Γ₂ independently, providing both as features. The difference Γ₁ - Γ₂ directly encodes the aerodynamic interference.

**Implementation complexity:** Low. Pure feature engineering, no architecture change.

```python
# In feature construction (prepare_multi.py context, but add in train.py via x transform):
chord = 1.0  # normalized chord length
gamma_approx = np.pi * chord * Umag * np.sin(2 * np.deg2rad(AoA))  # [B]
# For tandem: compute for each foil separately
gamma1 = np.pi * chord1 * Umag * np.sin(2 * AoA)
gamma2 = np.pi * chord2 * Umag * np.sin(2 * AoA_eff)  # AoA_eff = AoA + stagger-effect

# Broadcast as scalar features to all nodes (like Re, gap, stagger currently done)
x = torch.cat([x, gamma_approx.unsqueeze(-1).expand(-1, N, -1)], dim=-1)
```

**Suggested experiment:**
Add `--circulation_feature` flag. Compute Γ_approx = π·c·Umag·sin(2α) per foil and add as 2 additional input channels (one per foil, zeroed for the absent foil in single-foil samples). Run 2 seeds.

**Key risk:** Thin airfoil theory is a first-order approximation that breaks down for high AoA, thick airfoils, or tandem interaction. The model may learn to ignore the feature if it's too inaccurate. The gap-stagger spatial bias already captures some of this information indirectly.

**Confidence:** Medium. The physical motivation is excellent (Kutta-Joukowski is exact). The approximation quality determines whether it helps. The feature is low-complexity to add.

---

## Idea 5 (PARADIGM LEVEL): Geometry-Conditioned Surface Refinement via HyperNetwork

**Slug:** `hypernetwork-srf`

**Target metrics:** p_tan, p_oodc (OOD geometry generalization)

**Mechanism:**
The current Surface Refinement Head (SRF) is a fixed 3-layer MLP (hidden=192) applied to all surface nodes across ALL airfoil geometries. A single set of weights must generalize across diverse NACA profiles, multi-element configurations, and tandem arrangements. This is a bottleneck: the SRF has to handle the suction peak of a thin symmetric NACA 0006 the same way it handles a thick cambered NACA 4412 in a tandem configuration.

A HyperNetwork generates the SRF weights dynamically from a geometry descriptor:

```python
class HyperSRF(nn.Module):
    def __init__(self, geom_dim, srf_hidden, n_params):
        self.hypernet = nn.Sequential(
            nn.Linear(geom_dim, 128), nn.SiLU(),
            nn.Linear(128, n_params)  # output: flattened SRF weights
        )
        
    def forward(self, geom_desc, surface_features):
        srf_weights = self.hypernet(geom_desc)  # [B, n_params]
        # Reshape to SRF layer matrices and apply to surface_features
        return dynamic_srf(surface_features, srf_weights)
```

The geometry descriptor encodes per-foil shape statistics:
- NACA code thickness/camber extracted from the DSDF geometry
- Chord length, max thickness position
- For tandem: gap, stagger, fore/aft chord ratio

This conditioned SRF can apply qualitatively different refinement behavior for symmetric vs cambered vs tandem configurations — something the current fixed SRF cannot do.

**Why it hasn't been tried:** Previous hypernetwork attempts (PR #2196, BOLD2.md) were applied to the FULL backbone, which has ~millions of parameters — the hypernetwork itself becomes too large. The SRF is much smaller (3 layers × 192 hidden ≈ ~74K params), making a hypernetwork that generates SRF weights tractable. A hypernetwork generating ~74K params from a ~20-dim geometry descriptor needs only ~128×74K ≈ 9.5M params itself — feasible on 96GB VRAM.

**Implementation complexity:** Medium-High. Requires dynamic weight application and careful gradient flow. Use a smaller SRF for the hypernetwork target (e.g., hidden=64 instead of 192) to keep the generated weight vector to ~10K params — more tractable.

**Suggested experiment:**
Add `--hypernetwork_srf` flag. Reduce SRF to hidden=64 (so hypernetwork generates ~5K params). Geometry descriptor: 8-dim per-foil summary (t/c, x_max_camber, AoA, Re, gap, stagger, foil2_present). HyperNet: 2 layers × 256 hidden → SRF weights. Run 2 seeds.

**Key risk:** Dynamic weight generation can cause optimization instability. Start with a residual formulation: `srf_output = fixed_srf(x) + delta * hypernet_srf(x, geom)`, where `delta` starts near zero and grows during training. This gives the fixed baseline a head start.

**Confidence:** Medium. The mechanism is principled and targets a real structural limitation (one-size-fits-all SRF). The implementation complexity is real. No prior experiments in 1903 runs have tried hypernetwork for SRF specifically.

---

## Idea 6: Signed Distance Function to Wake Centerline Feature

**Slug:** `wake-centerline-sdf`

**Target metrics:** p_tan, p_in

**Mechanism:**
The model has wake deficit features (gap-normalized TE offsets) and stagnation features (proposed in-flight). But there is no explicit feature encoding a node's distance and angle relative to the fore-foil's WAKE CENTERLINE — the line extending downstream from the trailing edge along the free-stream direction.

In tandem configurations, the aft-foil operates inside the fore-foil wake. The wake centerline is the locus of maximum velocity deficit and maximum turbulence intensity. The aft-foil pressure distribution is fundamentally determined by where each surface node sits relative to this wake structure:
- Nodes above the centerline see lower momentum flow → different pressure
- Nodes below the centerline see near-freestream flow → cleaner pressure
- Nodes intersected by the wake (crossing the centerline) experience maximum disturbance

Compute for each mesh node:
1. `d_wake_perp`: signed perpendicular distance from node to the wake centerline (positive = above, negative = below)
2. `d_wake_parallel`: parallel distance along centerline (how far downstream of the TE)
3. `inside_wake`: binary indicator if the node is inside the estimated wake width

The wake centerline is defined as the ray from `fore_TE` in direction `[cos(AoA), sin(AoA)]` (downstream direction). The wake width can be estimated from a turbulence model: `width(x) = 0.1 * sqrt(x / chord)` (classic wake spreading rate for attached flows).

```python
def compute_wake_features(pos, fore_te, aoa_rad, chord=1.0):
    """Compute wake centerline proximity features."""
    # Downstream direction (wake propagates in freestream direction)
    wake_dir = torch.tensor([torch.cos(aoa_rad), torch.sin(aoa_rad)])
    
    # Vector from TE to each node
    r = pos - fore_te  # [N, 2]
    
    # Parallel and perpendicular components
    d_parallel = (r * wake_dir).sum(dim=-1).clamp(min=0)  # downstream distance
    d_perp = r[:, 0] * (-wake_dir[1]) + r[:, 1] * wake_dir[0]  # signed perp distance
    
    # Wake width at this downstream distance
    wake_width = 0.1 * torch.sqrt(d_parallel / chord + 1e-3)
    
    # Normalized signed distance to wake edge (negative inside wake)
    d_to_wake_edge = (d_perp.abs() - wake_width) / chord
    inside_wake = (d_perp.abs() < wake_width).float()
    
    return torch.stack([d_perp/chord, d_parallel/chord, d_to_wake_edge, inside_wake], dim=-1)
```

For single-foil cases, all wake features are zeroed.

**Implementation complexity:** Low-Medium. Pure geometry computation, no architecture change. The wake width model is approximate (laminar spreading rate — real turbulent wakes spread faster).

**Suggested experiment:**
Add `--wake_centerline_feature` flag. 4 new input channels per node: (d_perp, d_parallel, d_to_edge, inside_wake). Zero for non-tandem samples. Run 2 seeds. Key comparison: does this help p_tan more than p_in?

**Confidence:** Medium-high. The wake centerline is a physically meaningful geometric feature that encodes information NOT present in the current feature set. The existing wake deficit features encode position relative to the TE tip — this encodes position relative to the extended wake structure, which is different and complementary.

---

## Idea 7: Cosine Annealing with Warm Restarts (SGDR) Tuned to 150-Epoch Budget

**Slug:** `sgdr-warm-restarts-tuned`

**Target metrics:** All four (general optimization improvement)

**Mechanism:**
The current schedule is a single cosine cycle with T_max=150 (PR #2251). This gives one LR arc from lr_max to lr_min over 150 epochs. SGDR (Cosine Annealing with Warm Restarts, Loshchilov & Hutter 2016) instead cycles multiple times, with each restart potentially escaping local optima at different LR scales.

The key insight: with the 150-epoch budget, we can run e.g., 3 restarts with T_0=50, T_mult=1 (50+50+50) or T_0=30, T_mult=1.5 (30+45+67). Each restart brings LR back to lr_max, forcing the model to explore a wider optimization landscape, then settles to a local minimum. The best checkpoint across all restarts is selected.

**Why T_max tuning has diminishing returns:** Previous PRs tried T_max=140, 150, 160. All within the single-cycle paradigm. SGDR is qualitatively different — it creates an ensemble of local optima within a single training run by using multiple cycles.

**Implementation:**
```python
# Replace current cosine scheduler with SGDR
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,      # first cycle length
    T_mult=1,    # equal-length cycles (alternative: T_mult=1.5 for growing cycles)
    eta_min=1e-6
)
# Best checkpoint saves across ALL cycles, not just final
```

**Suggested experiment:**
Add `--sgdr` flag with `--sgdr_T0 50 --sgdr_T_mult 1`. Compare with current cosine schedule. The checkpoint-selection logic already takes the best val epoch — which will naturally pick the best across all SGDR cycles. 2 seeds minimum.

**Confidence:** Medium. SGDR is well-validated (Loshchilov & Hutter 2016, widely used). The 150-epoch budget makes this viable. The risk: if the model is already well-converged in one cycle, restarts may waste epochs retraining from high LR. The gain depends on how much the optimization landscape has multiple local minima to escape.

---

## Idea 8: Dual-Foil Cross-Attention Pressure Coupling

**Slug:** `dual-foil-cross-attention`

**Target metrics:** p_tan (tandem-specific)

**Mechanism:**
The Transolver backbone applies physics-slice attention globally across all mesh nodes simultaneously. This means foil-1 surface nodes and foil-2 surface nodes attend to each other through the global attention pool. But the aerodynamic interaction is DIRECTIONAL: the fore-foil influences the aft-foil (through its wake), but the aft-foil's upstream effect on the fore-foil is smaller (only through upwash/blockage).

Add a dedicated cross-attention layer in the SRF head that explicitly models this directionality:

```python
class DualFoilCrossAttentionSRF(nn.Module):
    def __init__(self, hidden, n_heads=4):
        self.cross_attn_1to2 = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.cross_attn_2to1 = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        
    def forward(self, surf_feat, foil_mask):
        foil1_feat = surf_feat[foil_mask == 1]  # [S1, H]
        foil2_feat = surf_feat[foil_mask == 2]  # [S2, H]
        
        # Aft-foil attends to fore-foil (wake interaction: directional)
        foil2_updated, _ = self.cross_attn_1to2(
            query=foil2_feat,    # aft-foil queries
            key=foil1_feat,      # fore-foil provides context
            value=foil1_feat
        )
        
        # Fore-foil attends to aft-foil (blockage effect: weaker)
        foil1_updated, _ = self.cross_attn_2to1(
            query=foil1_feat,
            key=foil2_feat,
            value=foil2_feat
        )
        
        return foil1_updated, foil2_updated
```

This is applied only to tandem samples (foil_mask has both foil-1 and foil-2 nodes). For single-foil samples, skip the cross-attention (return unchanged surface features).

**Why p_tan is hardest:** The tandem pressure (p_tan=27.90 MAE, largest gap) is driven by the aft-foil operating in the fore-foil wake. This cross-attention explicitly models WHICH fore-foil surface states inform the aft-foil pressure at each surface location — a structured interaction that the global physics-slice attention handles only implicitly.

**Implementation complexity:** Medium. The foil masks are already available (the aft-foil SRF flag `--aft_foil_srf` already separates foil-1 and foil-2 surface processing). This adds a cross-attention module to the SRF head, torch.compile compatible.

**Suggested experiment:**
Add `--dual_foil_cross_attn` flag. n_heads=4, hidden matches SRF hidden=192. Apply after the existing SRF layers as an additional cross-attention refinement step. For single-foil samples, skip (zero cross-attention). Run 2 seeds. Key: does p_tan improve while p_in/p_oodc/p_re stay stable?

**Confidence:** Medium. The aerodynamic motivation is strong (directional wake coupling). The implementation requires careful mask handling. Prior inter-foil coupling experiments (FNO inter-foil, PR #2258) failed — but those were global architecturally different models. This is surgical: only the SRF surface head gets cross-attention, not the backbone.

---

## Idea 9: Augment Training Data with Symmetry-Generated Mirror Cases

**Slug:** `mirror-symmetry-augmentation`

**Target metrics:** p_oodc, p_re

**Mechanism:**
Incompressible flow over an airfoil at angle α with free-stream velocity U has a specific symmetry: flipping the geometry vertically (y → -y) and setting AoA → -α gives a mirror-image solution where the surface pressure distribution reflects symmetrically. This is NOT an approximate symmetry — it is exact for incompressible Navier-Stokes with no-slip walls.

For every training sample with AoA α, we can generate an exact additional training sample by:
1. Reflecting all coordinates: y → -y
2. Negating AoA: α → -α
3. Reflecting velocity: Ux → Ux, Uy → -Uy
4. Pressure stays the same (symmetric)

This DOUBLES the effective training set at zero computational cost and no approximation. For tandem cases, the mirror symmetry applies to the full configuration.

**Why it hasn't been tried at this level:** Augmentation experiments in 1903 PRs have tried perturbation (AoA jitter, DSDF noise, Re scaling) — all approximate. Mirror symmetry is EXACT. There are zero approximation errors. The mirrored sample is as valid as the original CFD solution. This is data augmentation by exploiting a true physical symmetry.

**Current augmentation analysis:** The existing `aug_full_dsdf_rot` augmentation rotates the input coordinate frame, which is approximate because it doesn't respect the mesh structure. Mirror augmentation is simpler AND exact: it requires only flipping y-coordinates and negating Uy, with no interpolation or approximation.

**Implementation:**
```python
def apply_mirror_augmentation(batch, p_mirror=0.5):
    """Apply exact y-reflection symmetry augmentation with probability p."""
    if random.random() > p_mirror:
        return batch
    
    # Flip y-coordinates
    batch.pos[:, 1] *= -1
    
    # Flip y-velocity
    batch.y[:, 1] *= -1   # Uy → -Uy
    
    # Flip AoA sign in input features (wherever AoA is encoded in x)
    # The DSDF features are geometry-based and automatically mirror correctly
    # when pos[:, 1] is flipped
    
    # AoA scalar: negate the relevant input dimension
    batch.x[:, aoa_channel_idx] *= -1
    
    return batch
```

**Implementation complexity:** Low-Medium. The main challenge is identifying which input channels encode signed quantities that need to be flipped (Uy, signed y-coordinates in features, AoA scalar). The DSDF is computed from geometry — if coordinates are flipped before DSDF computation, DSDF is correct automatically.

**Suggested experiment:**
Add `--mirror_augmentation` flag with `p_mirror=0.5` (50% chance per batch). Apply after all other augmentations. The DSDF features need to be recomputed on the flipped geometry (or computed on the fly in train.py if the pipeline allows). Run 2 seeds. Key question: does doubling the effective dataset via exact symmetry improve OOD generalization?

**Confidence:** Medium-high. The symmetry is exact (no approximation). The implementation requires careful feature handling. Mirror augmentation is a first-principles technique that has not been tried in this setting despite 1903 experiments. Risk: if the training set already covers both positive and negative AoA samples, the mirrored samples add minimal information (they're already in the dataset). Check whether the training set has explicit negative-AoA samples.

---

## Idea 10 (BOLD): Prediction in Helmholtz Decomposition Basis

**Slug:** `helmholtz-decomposition-prediction`

**Target metrics:** p_in, p_oodc (physics-structured prediction)

**Mechanism:**
The Helmholtz-Hodge decomposition theorem states that any 2D vector field u can be uniquely decomposed as:

```
u = ∇φ + ∇×ψ + harmonic component
```

where:
- φ is the scalar potential (irrotational, curl-free component — the "potential flow" part)
- ψ is the stream function (divergence-free, rotational component — the "vortex" part)

For incompressible flow (∇·u = 0), the irrotational component is zero in the bulk, and the velocity field is purely solenoidal (∇×ψ). The pressure is directly tied to φ through the Poisson equation ∇²φ = -ρ∇·(u·∇u).

Instead of predicting (Ux, Uy, p) directly, the model predicts (φ, ψ, Δp) where:
- Ux = ∂φ/∂x + ∂ψ/∂y + U∞ (decomposed velocity potential + stream function perturbation + freestream)
- Uy = ∂φ/∂y - ∂ψ/∂x

This forces the model to predict a physically structured decomposition. The continuity equation ∇·u = 0 is AUTOMATICALLY satisfied by construction (∇·∇×ψ = 0 always).

**Implementation:**
In practice, the network outputs (φ, ψ) as scalar fields per node, then velocity is computed via discrete gradients:
```python
# Network predicts: phi_pred [N,], psi_pred [N,], p_pred [N,]
# Recover velocity using discrete gradient on mesh
dφdx, dφdy = mesh_gradient(phi_pred, pos, edge_index)
dψdx, dψdy = mesh_gradient(psi_pred, pos, edge_index)

Ux_pred = dφdx + dψdy + Umag * cos(AoA)  # + freestream
Uy_pred = dφdy - dψdx + Umag * sin(AoA)  # + freestream

# Loss on recovered velocity, not on (phi, psi) directly
velocity_loss = L1(Ux_pred, Ux_true) + L1(Uy_pred, Uy_true)
pressure_loss = L1(p_pred, p_true)
```

The advantage: the model can't predict a divergent velocity field, because by construction ∇·u = 0. This frees model capacity from enforcing the divergence constraint implicitly, focusing it on the physically meaningful rotational structure.

**Why bold:** This is a complete reparametrization of the prediction target. It's the deepest structural change possible without changing the architecture. It's based on classical fluid mechanics (Helmholtz 1858) and has been explored in physics-informed ML (Richter et al. 2022, "SFNO: Spherical Fourier Neural Operators") but never in this CFD surrogate setting.

**Implementation complexity:** High. Requires implementing discrete mesh gradient operators (doable with the existing edge_index connectivity). The discrete gradient on unstructured meshes has known accuracy issues at mesh-scale features — test first on a simple test case.

**Suggested experiment:**
This is the boldest experiment. Add `--helmholtz_prediction` flag. Replace direct (Ux, Uy) output heads with (φ, ψ) heads. Recover velocity via discrete gradient. Keep pressure head unchanged. Loss on recovered velocity. Run 2 seeds. If it fails: analyze whether the discrete gradient operator is the bottleneck (check if gradient recovery error is large on ground-truth phi, psi computed from true CFD data).

**Confidence:** Low-medium. The theory is beautiful and exact, but the discrete implementation on unstructured meshes introduces approximation errors that may dominate. This is a bold experiment with high upside (structurally correct predictions, automatic mass conservation) but real implementation risk. Run it alongside safer ideas.

---

## Summary and Priority Ranking

| Rank | Slug | Target | Complexity | Confidence | Type |
|------|------|--------|------------|------------|------|
| 1 | `potential-flow-residual` | p_in, p_tan, p_re | Medium | Strong | Paradigm |
| 2 | `circulation-lift-feature` | p_in, p_tan | Low | Medium | Physics feature |
| 3 | `wake-centerline-sdf` | p_tan, p_in | Low-Med | Med-High | Physics feature |
| 4 | `mirror-symmetry-augmentation` | p_oodc, p_re | Low-Med | Med-High | Data augmentation |
| 5 | `ensemble-distill-soft-labels` | p_oodc, p_re | Medium | Medium | Paradigm |
| 6 | `continuity-pde-loss` | p_oodc, p_re | Medium | Medium | Paradigm |
| 7 | `dual-foil-cross-attention` | p_tan | Medium | Medium | Architecture |
| 8 | `sgdr-warm-restarts-tuned` | All | Low | Medium | Optimization |
| 9 | `hypernetwork-srf` | p_tan, p_oodc | Med-High | Medium | Paradigm |
| 10 | `helmholtz-decomposition-prediction` | p_in, p_oodc | High | Low-Med | Paradigm |

**Recommended immediate assignments (start with low-complexity, high-confidence):**
1. `circulation-lift-feature` — one-line physics feature, no architecture change, strong theoretical basis
2. `wake-centerline-sdf` — complements existing wake deficit feature, complementary physics signal
3. `mirror-symmetry-augmentation` — exact symmetry, zero approximation, free data doubling
4. `potential-flow-residual` — paradigm-level, medium complexity, strong theory
5. `sgdr-warm-restarts-tuned` — low complexity optimization lever not yet tried
