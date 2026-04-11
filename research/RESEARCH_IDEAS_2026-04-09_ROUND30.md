# SENPAI Research Ideas — Round 30

_Generated: 2026-04-09_
_Researcher-agent synthesis after full review of 1903 PRs._
_Baseline (PR #2251): p_in=11.891, p_oodc=7.561, p_tan=28.118, p_re=6.364._
_Primary target: p_tan (largest absolute gap), p_re (regression from #2213 where p_re=6.300)._

---

## Context and Constraints

### What Round 30 must avoid (Round 29 currently in-flight)

- PR #2288 (thorfinn): chord fraction feature — per-node chord-wise position [0,1]
- PR #2294 (fern): tandem config proximity feature — KD-tree OOD distance signal
- PR #2292 (askeladd): flow-direction normalization — rotate coords by -AoA to streamwise frame
- PR #2291 (frieren): stagnation pressure feature — q_inf = 0.5*Umag^2 as input channel
- PR #2295 (tanjiro): surface curvature feature — discrete Menger curvature at surface nodes
- PR #2296 (edward): log-Re pressure scaling — Re-normalize loss for OOD-Re generalization
- PR #2290 (nezuko): Re-stratified sampling — 2x weight for extreme-Re training samples
- PR #2293 (alphonse): low-rank pressure loss — SVD structural prior on surface error

### KEY PATTERN from all 1903 experiments

**Every durable improvement came from: (a) physics-motivated input features, or (b) loss reformulation.**
All other approaches — architecture changes, augmentation strategies, regularization, optimizer variants, ensemble methods — have produced null or negative results.

This round follows the same pattern but targets UNSEEN angles:
1. **Global geometry descriptors** not yet tried (Finite Volume features from ICML 2024)
2. **Cell-area / mesh-density-aware loss weighting** (not tried in any of 1903 experiments)
3. **Flow-regime indicator features** (local field invariants: Q-criterion proxy, strain rate)
4. **Gradient-weighted loss** targeting high-gradient regions automatically (GMSE, arXiv:2411.17059)
5. **Shortest-vector geometry representation** — vector from each node to nearest foil surface point

---

## Idea 1 (TOP PICK): Finite Volume Cell-Area Loss Weighting

**Slug:** `fv-cell-area-loss-weight`

**Target metrics:** p_in, p_tan, p_re (all surface metrics)

**Key bet / mechanism:**
The current loss treats every mesh node equally. But CFD meshes are deliberately non-uniform: near the airfoil surface, cells are tiny (capturing boundary layer). Far from the foil, cells are large (coarse wake/freestream). When loss is unweighted, a single large farfield cell contributes equally to gradient signal as ten boundary-layer cells — this biases training toward coarse freestream regions.

Finite Volume theory says the correct numerical integration weight for a node is the VOLUME (2D: AREA) of its control volume / dual mesh cell. Surface MAE is measured only on surface nodes — which sit in the smallest cells. By weighting loss by 1/cell_area (emphasizing small boundary-layer cells) or by cell_area (matching true physics integration), we can focus gradient signal where surface MAE is measured.

The TandemFoilSet mesh has cell area encoded in the data pipeline. From `prepare_multi.py`, the feature vector `x` already has access to node-level properties. The cell area can be computed from the mesh connectivity (already available as edge indices in the data).

This is one of the main findings of the Finite Volume Features paper (ICML 2024, arXiv:2402.02367): using cell volume as both edge/node feature AND as loss weight dramatically improves PDE surrogate accuracy on non-uniform meshes.

**Why it hasn't been tried:** The current hard-node mining multiplies loss by 1.5 for above-median error nodes. That's error-based weighting, not geometry-based weighting. Cell-area weighting is orthogonal — it targets the structural bias from mesh non-uniformity, not from error magnitude.

**Implementation complexity:** Low. One new loss weight tensor, no architecture change. Compute dual cell area from mesh edge connectivity during data loading (or once per sample at forward time from batch data).

```python
# In training loop, after computing node-level losses:
# node_areas: [B, N] — dual control volume area for each mesh node
# Use 1/sqrt(area) weighting to up-weight small boundary-layer cells

# Option A: inverse-sqrt weighting (emphasize fine boundary layer cells)
area_weight = 1.0 / (node_areas.clamp(min=1e-8).sqrt())
area_weight = area_weight / area_weight.mean(dim=-1, keepdim=True)  # normalize to mean 1

# Apply to volume loss ONLY (surface loss already targets surface nodes)
vol_loss_weighted = (vol_loss_per_node * area_weight).mean()

# Option B: cell-area normalization for surface loss (match integration)
surf_area = node_areas[surface_mask]  # [B, S]
surf_area_weight = surf_area / surf_area.sum(dim=-1, keepdim=True)  # normalize per sample
surf_loss_weighted = (surf_loss_per_node * surf_area_weight * surf_area.shape[-1]).mean()
```

**Key gotcha:** The data pipeline (`prepare_multi.py`) may not currently expose node areas. If not, approximate cell area from the mean distance to k-nearest neighbors: `approx_area = (knn_dists[:, :, :k].mean(dim=-1)) ** 2`. This gives a proxy that's monotone with true cell area.

**Suggested experiment:** Add `--fv_area_loss_weight` flag. Weight VOLUME loss (not surface loss) by 1/sqrt(cell_area). Keep surface loss unweighted (hard-node mining already handles it). Run 2 seeds to confirm. Compare against baseline on all 4 val tracks.

**Literature:**
- Lam et al. "Finite Volume Features, Global Geometry Representations, and Residual Training for Deep Learning-Based CFD Simulation" (ICML 2024, arXiv:2402.02367) — direct evidence that cell-volume-weighted loss + FV edge features improves surrogate accuracy by 15-40% on non-uniform meshes
- Hsieh et al. "Learning Physical Simulation with Message Passing" (NeurIPS 2023) — FV-style integration as loss weighting

**Confidence:** Strong. The mechanism is principled (addresses a real structural bias in current loss formulation), the paper has direct ablations showing the benefit of volume weighting, and it has NEVER been tried in 1903 experiments.

---

## Idea 2: GMSE — Gradient-Weighted Loss (Auto-Target High-Variance Regions)

**Slug:** `gmse-gradient-loss`

**Target metrics:** p_tan, p_in

**Key bet / mechanism:**
The current loss has several manual weighting components: surface weighting (5-50x adaptive), hard-node mining (1.5x above-median pressure), DCT frequency loss. But none of these explicitly weight by the SPATIAL GRADIENT of the predicted field. Yet in CFD, the physically interesting — and hardest to predict — regions are exactly where ∇p and ∇u are large: leading edge stagnation, trailing edge separation, suction peak, wake-foil interface in tandem configs.

GMSE (Gradient Mean Squared Error, arXiv:2411.17059) weights each node's loss by the local field gradient magnitude: `loss_GMSE = (||∇f_pred||₂ + epsilon) * (f_pred - f_true)^2`. This automatically amplifies gradient signal in high-gradient regions without requiring any manual identification of "hard" regions. For tandem configs specifically, the slot between fore and aft foil creates strong pressure gradient signatures that the model consistently underfits (p_tan = 28.118, the largest surface error).

Key difference from current hard-node mining: hard-node mining weights by ERROR magnitude. GMSE weights by FIELD GRADIENT magnitude. These are complementary — a low-error node in a high-gradient region still deserves more gradient signal because small position errors in high-gradient regions have large physical consequences.

**Implementation complexity:** Medium. Requires computing local gradient of predicted field (finite difference or graph-Laplacian on mesh edges). Can approximate with 1-ring neighbor differences.

```python
# Compute spatial gradient magnitude of predicted pressure field
# pred_p: [B, N] — predicted pressure at all nodes
# edge_index: [2, E] — mesh edges (from batch.edge_index)
# pos: [B, N, 2] — node positions

def field_gradient_magnitude(pred, pos, edge_index, eps=1e-6):
    """Approximate gradient magnitude via finite differences on mesh edges."""
    src, dst = edge_index[0], edge_index[1]
    dp = pred[dst] - pred[src]                         # [E] scalar field difference
    dx = pos[dst] - pos[src]                           # [E, 2] position difference
    dist = dx.norm(dim=-1).clamp(min=eps)              # [E]
    grad_edge = dp.abs() / dist                        # [E] gradient magnitude along edge
    
    # Scatter max to nodes (use max to capture sharp gradients)
    grad_node = torch.zeros_like(pred)
    grad_node.scatter_reduce_(0, src, grad_edge, reduce='amax')
    return grad_node  # [B*N] gradient magnitude at each node

# In training loss:
with torch.no_grad():
    grad_mag = field_gradient_magnitude(pred_p.flatten(), pos.flatten(0,1), batch_edge_index)
    grad_mag = grad_mag.view(B, N)
    grad_weight = (grad_mag + 0.1).clamp(max=5.0)     # cap at 5x, floor at 0.1
    grad_weight = grad_weight / grad_weight.mean(dim=-1, keepdim=True)  # normalize

# Apply ONLY to pressure component of volume loss (Ux/Uy less affected by sharp gradients)
p_loss = (pred_p - true_p).abs()
p_loss_gmse = (p_loss * grad_weight).mean()
```

**Why it might help here:** The tandem inter-foil slot is a high-gradient-pressure region. The current loss gives it the same weight as freestream nodes (unless those nodes happen to have high error this batch). GMSE guarantees they receive 3-5x more gradient signal every batch.

**Risk:** Medium. Gradient estimation is noisy on unstructured meshes. The pressure gradient aux head experiment (PR #2267) FAILED but that was a separate output head computing FD gradients as a supervisory target — this is different: using gradient as a loss WEIGHT, not as a prediction target.

**Suggested experiment:** Add `--gmse_pressure_weight` flag (default 0). Apply gradient weighting only to the pressure component of vol_loss. Start with `lambda_gmse=0.3` (interpolate: `loss_p = (1-0.3)*l1_loss + 0.3*gmse_loss`). Verify gradient magnitude computation is correct on a single batch before running full training.

**Literature:**
- Bhatt et al. "GMSE: Gradient-Weighted Mean Squared Error for Improved Learning of Complex Physical Fields" (arXiv:2411.17059, 2024) — 5-23% MAE reduction on heat/flow fields vs standard MSE
- Obiols-Sales et al. "CFDNet: A deep learning-based accelerator for fluid simulations" (2020) — boundary region weighting in CFD surrogates

**Confidence:** Medium-high. The mechanism is sound and the paper has ablations. Main risk is the gradient estimation quality on unstructured mesh edges.

---

## Idea 3: Shortest-Vector Geometry Descriptor (Global Geometry Awareness)

**Slug:** `shortest-vector-feature`

**Target metrics:** p_tan, p_oodc

**Key bet / mechanism:**
The current DSDF features encode DISTANCE to the nearest foil surface (scalar) and optionally the DIRECTION to the nearest point (via DSDF gradient channels). But DSDF is computed separately for each foil — the model has `dsdf1` and `dsdf2` as separate channel groups. What is missing is: for each mesh node, what is the VECTOR TO THE NEAREST POINT ON ANY FOIL SURFACE (foil1 or foil2, whichever is closer)?

This is the "Shortest Vector" (SV) representation from the Finite Volume Features paper (ICML 2024): for each node, compute the full 2D displacement vector [dx, dy] to the nearest surface point across all foils. For tandem configurations, this gives the model an explicit encoding of which foil is the nearest geometric reference for each node — and in what direction — without relying on the model to infer this from two separate DSDF scalars.

Why this matters for p_tan: in tandem configs, nodes in the slot between fore and aft foils have comparable DSDF to both foils. The model must disambiguate "am I closer to the pressure side of the aft foil or the suction side of the fore foil?" Currently it does this from two DSDF scalars + the gap/stagger scalars. An explicit 2D vector to the nearest foil surface makes this disambiguation geometrically unambiguous.

**Implementation complexity:** Low-Medium. Computed from existing mesh positions and surface node positions. Can be done in `prepare_multi.py` style (add to feature vector) or at forward time.

```python
# Compute shortest vector to nearest foil surface
# pos: [B, N, 2] — all node positions  
# surface_pos_foil1: [B, S1, 2] — foil 1 surface node positions
# surface_pos_foil2: [B, S2, 2] — foil 2 surface node positions (tandem) or zeros (single)

def shortest_vector_to_surface(pos, surf_pos_foil1, surf_pos_foil2, has_foil2):
    """[B, N, 2] shortest vector from each node to nearest point on any foil surface."""
    B, N, _ = pos.shape
    
    # Distances to foil 1 surface
    # pos: [B, N, 1, 2], surf1: [B, 1, S1, 2]
    diff1 = pos.unsqueeze(2) - surf_pos_foil1.unsqueeze(1)  # [B, N, S1, 2]
    dist1 = diff1.norm(dim=-1)                               # [B, N, S1]
    min_dist1, min_idx1 = dist1.min(dim=-1)                  # [B, N]
    sv1 = diff1[torch.arange(B)[:, None], torch.arange(N)[None, :], min_idx1]  # [B, N, 2]
    
    if has_foil2:
        diff2 = pos.unsqueeze(2) - surf_pos_foil2.unsqueeze(1)
        dist2 = diff2.norm(dim=-1)
        min_dist2, min_idx2 = dist2.min(dim=-1)
        sv2 = diff2[torch.arange(B)[:, None], torch.arange(N)[None, :], min_idx2]
        # Choose whichever foil is closer
        use_foil1 = (min_dist1 <= min_dist2).unsqueeze(-1)
        sv = torch.where(use_foil1, sv1, sv2)  # [B, N, 2]
    else:
        sv = sv1
    
    # Normalize by chord length (or clamp) to make scale-invariant
    sv_norm = sv / (sv.norm(dim=-1, keepdim=True).clamp(min=1e-6) + 1.0)  # soft normalization
    return sv_norm  # [B, N, 2]

# Append as 2 new channels to input x
```

**Key gotcha:** This is O(N * S) to compute. With N~10K nodes and S~300 surface nodes, this is 3M comparisons per sample. Use torch.cdist for batch efficiency. Compute once at data loading, not at forward time.

**Why not redundant with DSDF?** DSDF gives scalar distances to each foil separately. The shortest vector gives the DIRECTION to the closest surface point across ALL foils — a 2D vector that encodes both which foil is geometrically dominant AND in what direction. This is the key missing quantity for inter-foil disambiguation in the slot region.

**Literature:**
- Lam et al. "Finite Volume Features..." (ICML 2024, arXiv:2402.02367) — SV representation: explicitly showed that global geometry encodings (SV, DID) reduce MSE on Airfoil and ShapeNet datasets by 18-32% beyond DSDF alone
- Bonnet et al. "AirfRANS: High Fidelity Computational Fluid Dynamics Dataset..." (NeurIPS 2023) — surface proximity vector as key input feature

**Confidence:** Medium-high. The SV representation is theoretically sound and has empirical support from the FVF paper. Key uncertainty is whether the tandem mesh already provides adequate disambiguation through existing gap/stagger features.

---

## Idea 4: Directional Integrated Distance (DID) Feature

**Slug:** `did-feature`

**Target metrics:** p_oodc, p_re

**Key bet / mechanism:**
The Directional Integrated Distance (DID) is the second global geometry representation from the ICML 2024 Finite Volume Features paper. For each mesh node, DID encodes how "far inside" the near-wake region the node is, measured along the direction of flow (AoA-aligned streamwise direction). Specifically: for each node, integrate the distance along a ray cast in the freestream direction until the ray exits the computational domain or hits a geometry boundary.

Why this helps for p_oodc (OOD conditions: extreme AoA/gap/stagger): at extreme AoA, the separation point moves dramatically — early separation at high AoA changes which nodes are in the wake vs attached flow. The DID feature encodes a per-node "how far am I in the wake region along this flow direction?" that adapts per sample based on AoA. This is a stronger OOD signal than the current global AoA scalar broadcast to all nodes.

Simplified version: instead of full ray casting, approximate DID as the dot product of the node position (relative to foil centroid) with the unit freestream direction vector:
`DID_approx = (pos - foil_centroid) · [cos(AoA), sin(AoA)]`

This gives a scalar per node that is positive for downstream nodes and negative for upstream nodes in the AoA-aligned frame. Combined with DSDF, this gives the model a "position along the streamline" at each node.

**Implementation complexity:** Very Low. Pure computation from existing features. 1 new channel. No architecture change.

```python
# Approximate DID: projection of node position onto freestream direction
# pos: [B, N, 2] normalized node coordinates
# aoa: [B] angle of attack in radians (from x[:, 0, aoa_channel])
# foil_centroid: [B, 2] mean position of all surface nodes

aoa = x[:, 0, aoa_channel]                            # [B] global scalar
freestream_dir = torch.stack([
    torch.cos(aoa), torch.sin(aoa)
], dim=-1)                                             # [B, 2]

# Compute foil centroid from surface node positions
foil_centroid = pos[is_surface].mean(dim=...)          # [B, 2] approx

# DID: projection of (pos - centroid) onto freestream
pos_rel = pos - foil_centroid.unsqueeze(1)             # [B, N, 2]
did = (pos_rel * freestream_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)  # [B, N, 1]

# Normalize to [-1, 1] range
did_norm = did / (did.abs().max(dim=1, keepdim=True).values + 1e-6)

# Append as 1 new channel
x = torch.cat([x, did_norm], dim=-1)
```

**Why not redundant with AoA?** AoA is a global scalar — same value broadcast to every node. DID is a per-node scalar that encodes that node's position relative to the flow direction. A node 2 chord-lengths downstream in the wake has a very different DID than a node 2 chord-lengths upstream, even though both see the same global AoA.

**Suggested experiment:** Start with the simplified dot-product approximation. Add `--did_feature` flag. 1-2 new channels (DID for foil1 centroid, and optionally DID for foil2 centroid in tandem configs). Verify that values are physically interpretable on a sample before training.

**Literature:**
- Lam et al. "Finite Volume Features..." (ICML 2024, arXiv:2402.02367) — DID shown to improve AirfRANS velocity prediction by 12% over DSDF-only baseline
- Stachenfeld et al. "Learned Coarse Models for Efficient Turbulence Simulation" (2021) — streamwise distance as structural prior

**Confidence:** Medium. Mechanism is sound and grounded in literature. Key uncertainty: the simplified dot-product DID may not capture enough signal compared to true ray-cast DID. But it is zero-cost to try and the full ray-cast version can follow if it works.

---

## Idea 5: Q-Criterion Proxy Feature (Flow Regime Indicator)

**Slug:** `q-criterion-proxy-feature`

**Target metrics:** p_tan, p_oodc

**Key bet / mechanism:**
The Q-criterion in fluid mechanics is Q = 0.5 * (||Ω||² - ||S||²) where Ω is the vorticity tensor and S is the strain-rate tensor. Q > 0 identifies vortex-dominated regions (separation, wake vortices, slot vortices in tandem). Q < 0 identifies strain-dominated regions (freestream, boundary layer far field).

At TRAINING time, we have the full CFD velocity field. From Ux and Uy predictions/ground truth on the mesh, we can compute approximate Q at each node as: `Q_approx = -(dUx/dx * dUy/dy) + 0.5 * (dUy/dx - dUx/dy)^2` using finite differences on mesh edges. The Q-criterion sign divides the flow into qualitatively different zones.

The proposal: at TRAINING and INFERENCE time, compute a Q-criterion PROXY from the INPUT FEATURES (not from predicted fields). Using the DSDF gradient (which encodes the geometry-induced deformation field near the foil) and the freestream velocity direction, approximate a geometric Q-proxy that identifies nodes likely to be in vortex-dominated vs strain-dominated zones. A binary or continuous feature `q_proxy = (dsdf_curl_approx)` captures whether a node is inside the expected wake/separation zone.

Simpler approximation: the cross-product of DSDF gradient with freestream direction gives an approximation to the local vorticity proxy for each node.

```python
# DSDF gradient approximation to vorticity proxy
# dsdf1_grad: [B, N, 2] — DSDF gradient direction for foil 1 (channels from prepare_multi.py)
dsdf1_grad = x[:, :, 1:3]  # foil 1 DSDF gradient x,y components

# Freestream direction
aoa = x[:, 0, aoa_channel]
u_inf = torch.stack([torch.cos(aoa), torch.sin(aoa)], dim=-1)  # [B, 2]

# Approximate vorticity proxy: cross product (dsdf1_grad x u_inf)
# This is a scalar: positive in upper wake region, negative below foil
vort_proxy = (dsdf1_grad[:, :, 0] * u_inf[:, 1].unsqueeze(1) 
              - dsdf1_grad[:, :, 1] * u_inf[:, 0].unsqueeze(1))  # [B, N]

# Scale by DSDF magnitude (closer to surface = stronger boundary layer vorticity)
dsdf1_dist = x[:, :, 0]  # DSDF scalar (distance to foil 1)
q_proxy = vort_proxy * torch.exp(-dsdf1_dist * 3.0)  # [B, N] decays away from surface

x = torch.cat([x, q_proxy.unsqueeze(-1)], dim=-1)  # 1 new channel
```

**Why it matters for p_tan:** Tandem aft-foil errors concentrate in the wake-interaction zone (slot and downstream wake). This zone is characterized by strong vorticity from the fore-foil trailing edge vortex. A feature that explicitly flags "this node is in a fore-foil wake vortex zone" gives the model information it currently lacks.

**Motivation from literature:** The RITA paper (arXiv:2504.06758) shows that local flow invariants (Q, R, velocity gradient tensor) classify flow regime (attached, separated, wake) with high accuracy from geometry features alone. Using these as inputs rather than as supervised targets is the proposed approach here.

**Risk:** Medium. The Q-criterion proxy is an approximation. The DSDF gradient is a geometry-driven quantity, not a velocity gradient — the proxy is an analog, not a true Q-criterion. Main risk: too noisy to be informative. Mitigation: apply sigmoid smoothing and test on a few samples before full training.

**Confidence:** Medium. Novel for this codebase. Motivated by aerodynamic physics and RITA literature. Not tried in 1903 experiments.

---

## Idea 6: Wall-Layer Binning Feature (Turbulence Zone Indicator)

**Slug:** `wall-layer-bin-feature`

**Target metrics:** p_in, p_tan

**Key bet / mechanism:**
Turbulent boundary layer theory identifies distinct physical zones by wall distance (in wall units y+): viscous sublayer (y+ < 5), buffer layer (5 < y+ < 30), log-law layer (30 < y+ < 300), wake region (y+ > 300). Different physics govern each zone — the model is asked to predict the SAME type of flow variable across radically different physical regimes.

Currently, DSDF provides raw distance to wall. But raw distance encodes POSITION, not PHYSICAL REGIME. The log-law layer exists at very different absolute distances for high-Re vs low-Re flows. Proposal: encode wall-proximity as DISCRETE ZONE MEMBERSHIP using Re-scaled distance thresholds, providing the model with explicit zone indicators.

Approximate wall units: y+ ≈ y * u_tau / nu, where u_tau ≈ sqrt(Cf/2) * Umag and Cf ≈ 0.026 * Re^(-1/7) (turbulent flat plate approximation). This lets us compute approximate y+ from existing features (DSDF, Re, Umag).

```python
# Wall-layer zone indicator from DSDF + Re + Umag
# dsdf: [B, N] — distance to nearest foil surface (raw, unnormalized)
# Re: [B] — Reynolds number (from x[:, 0, re_channel])  
# Umag: [B] — freestream velocity magnitude

Re = x[:, 0, re_channel].unsqueeze(1)              # [B, 1]
dsdf = x[:, :, dsdf_channel]                       # [B, N]

# Approximate friction velocity from Re (turbulent flat plate Cf formula)
Cf_approx = 0.026 * Re ** (-1.0/7.0)              # [B, 1]
u_tau_approx = torch.sqrt(Cf_approx / 2.0)        # [B, 1] non-dimensional

# Approximate y+: y * u_tau / nu = dsdf * u_tau * Re (all non-dimensional)
y_plus = dsdf * u_tau_approx * Re.sqrt()           # [B, N] rough approximation

# Zone bins (log scale):
sublayer  = (y_plus < 5.0).float()                # viscous sublayer
buffer    = ((y_plus >= 5.0) & (y_plus < 30.0)).float()   # buffer layer
log_layer = ((y_plus >= 30.0) & (y_plus < 300.0)).float()  # log-law layer
wake_zone = (y_plus >= 300.0).float()             # outer wake

zone_feats = torch.stack([sublayer, buffer, log_layer, wake_zone], dim=-1)  # [B, N, 4]
x = torch.cat([x, zone_feats], dim=-1)  # 4 new binary channels
```

**Why it might help here:** The model sees boundary-layer nodes (y+<5) and freestream nodes (y+>1000) in the same physics slice. Providing explicit zone membership allows the Transolver slice-routing mechanism to route similar-physics nodes to the same slice — improving physics coherence of the attention.

**Risk:** Low-Medium. Pure feature addition. Main concern: the y+ approximation is coarse (assumes flat-plate turbulence). Near stagnation points and separation zones the formula breaks down. Could add noise rather than signal. Mitigation: use soft binning (Gaussian RBF on log(y+) with 4 centers) rather than hard binary indicators.

**Suggested experiment:** Add `--wall_layer_feature` flag. Implement soft Gaussian binning of log(y+) with 4 centers at log(y+) = 0, 1, 2, 3. Compare against hard binary version. Run 1 seed first to check if the y+ approximation gives physically reasonable values before committing 2 seeds.

**Confidence:** Medium. Grounded in turbulence physics. Novel for this codebase. The y+ approximation is coarse but the qualitative zone separation (within 1 chord of surface vs freestream) should still be meaningful.

---

## Idea 7: Bernoulli Residual Feature (Physics-Baseline Correction Target)

**Slug:** `bernoulli-residual-feature`

**Target metrics:** p_in, p_re

**Key bet / mechanism:**
Note: The Bernoulli CONSISTENCY LOSS (PR #2224) failed — it imposed a hard constraint incorrect in viscous regions. This is a COMPLETELY DIFFERENT approach: using Bernoulli as an INPUT FEATURE that provides a per-node physics baseline, which the model then corrects.

Bernoulli's equation for irrotational flow: p + 0.5 * rho * |U|^2 = p_t (constant along streamline). In the freestream: p_inf + 0.5 * U_inf^2 = const. The PRESSURE COEFFICIENT Cp = (p - p_inf) / (0.5 * rho * U_inf^2). The model already predicts Cp (via `_phys_norm`). But what if we give the model an INITIAL ESTIMATE of Cp from potential flow theory?

Potential flow theory (thin-airfoil, inviscid) predicts Cp from the surface geometry via conformal mapping. An approximation: for surface nodes, thin-airfoil theory gives Cp ≈ -2 * alpha * sin(theta) where theta is the surface angle to the chord line and alpha is AoA (small angle). For volume nodes, Bernoulli with freestream gives Cp ≈ 0 everywhere. The model is asked to correct the viscous residual: delta_Cp = Cp_true - Cp_potential.

This is a specialization of the already-merged RESIDUAL PREDICTION (PR #1927) which gave p_oodc -4.7%. That used a global baseline. This uses a PHYSICS-DERIVED per-node baseline. The model would learn: delta_Cp = f(geometry, Re, wake effects) — a much smoother target than raw Cp.

```python
# Thin-airfoil potential flow Cp approximation as input feature
# surface_theta: angle of surface tangent to chord line, from TE coord frame
# For surface nodes: use TE frame angles (already available if te_coord_frame is on)
# For volume nodes: Cp_potential ≈ 0 (freestream) as baseline

# Simplified: use AoA and local DSDF gradient angle
dsdf_grad_angle = torch.atan2(
    x[:, :, dsdf_grad_y_channel], 
    x[:, :, dsdf_grad_x_channel]
)  # [B, N] — local surface normal direction angle
aoa = x[:, 0, aoa_channel].unsqueeze(1)               # [B, 1]

# Thin-airfoil Cp proxy: -2*sin(surface_angle - AoA) for surface nodes
cp_potential_proxy = -2.0 * torch.sin(dsdf_grad_angle - aoa)  # [B, N]

# Zero this out for volume nodes (DSDF > threshold means not near surface)
near_surface_mask = (x[:, :, dsdf_channel] < 0.05).float()
cp_potential_proxy = cp_potential_proxy * near_surface_mask

x = torch.cat([x, cp_potential_proxy.unsqueeze(-1)], dim=-1)  # 1 new channel
```

**Why it might help here:** The model currently predicts Cp from scratch. Providing an initial estimate shifts the learning problem from "predict Cp" to "correct the inviscid Cp". Viscous effects are smaller-magnitude, smoother corrections — easier to learn. This is exactly why residual prediction (PR #1927) worked, just with a better physics-grounded baseline.

**Risk:** Low. The potential flow formula can be wrong (especially near stagnation and separation) but wrong features just contribute noise, which the model can suppress with near-zero weights. The worst case is null effect, not degradation.

**Key distinction from PR #2224 (failed Bernoulli loss):** This adds Bernoulli as an INPUT channel. The model can choose to use or ignore it. PR #2224 added Bernoulli as a CONSTRAINT in the loss — which was too hard and wrong in boundary layers.

**Confidence:** Medium-high. Residual prediction (PR #1927) proved that baseline-correction framing works here. This is the same bet with a physics-derived baseline per node. Main uncertainty: whether the thin-airfoil approximation is close enough to be useful as a starting point.

---

## Idea 8: Tandem Slot Indicator Feature (Per-Node Slot Channel Membership)

**Slug:** `tandem-slot-indicator`

**Target metrics:** p_tan

**Key bet / mechanism:**
The current tandem features are: (1) gap value (scalar per sample), (2) stagger value (scalar per sample), (3) binary tandem flag (is_tandem), (4) wake deficit (per-node, fore-foil boundary conditions). None of these encode which nodes are INSIDE THE SLOT CHANNEL between the two foils.

In tandem airfoil aerodynamics, the slot channel (region between fore and aft foils) is the defining aerodynamic feature. Nodes inside the slot experience: (a) accelerated flow from the converging gap, (b) strong pressure gradient from fore-foil pressure recovery, (c) aft-foil boundary layer interaction. The model consistently fails on p_tan = 28.118 — and tandem errors almost certainly concentrate in the slot region.

A slot indicator feature: for each node, compute its signed distance to the slot midline (geometric midpoint between the trailing edge of the fore-foil and the leading edge of the aft-foil). For single-foil samples, this feature is 0 everywhere.

```python
# Slot channel indicator: per-node signed distance to slot midline
# gap: [B] — vertical gap between foils (x[:, 0, gap_channel])
# stagger: [B] — horizontal offset (x[:, 0, stagger_channel])
# Foil1 TE position and Foil2 LE position from surface nodes

# For tandem: define slot midline as midpoint between foil1 TE and foil2 LE
# Slot channel = region bounded by foil1 chord line and foil2 chord line in the gap
# Approximate: node is "in slot" if DSDF2_dist < slot_height/2 AND x_pos in [foil1_TE, foil2_LE]

gap = x[:, 0, gap_channel].unsqueeze(1)            # [B, 1]
stagger = x[:, 0, stagger_channel].unsqueeze(1)    # [B, 1]
is_tandem = (gap.abs() > 0.5).float()              # [B, 1]

# Node vertical position relative to slot midline
pos_y = pos[:, :, 1]                               # [B, N] y-coordinates
slot_midline_y = gap / 2.0                         # rough midline (depends on normalization)

# Soft slot membership: Gaussian centered on slot midline, width = gap/2
slot_dist = (pos_y - slot_midline_y).abs()
slot_indicator = torch.exp(-(slot_dist / (gap.abs() / 2.0 + 0.1)) ** 2)
slot_indicator = slot_indicator * is_tandem        # zero for single-foil samples

x = torch.cat([x, slot_indicator.unsqueeze(-1)], dim=-1)  # 1 new channel
```

**Why it hasn't been tried:** The gap/stagger features (PR #2130) gave p_tan -3.0% by encoding the configuration as scalars. This extends to PER-NODE encoding of slot membership — which nodes are actually inside the slot channel, and therefore experience the most anomalous physics.

**Why it might help:** The model needs to learn that nodes inside the slot channel are aerodynamically special (accelerated flow, strong pressure gradient). Currently it infers this from global gap/stagger scalars. An explicit per-node "slot membership" feature makes this distinction explicit.

**Risk:** Low. Single-channel feature addition. May be partially redundant with gap + DSDF2. But DSDF2 encodes distance to foil2 surface, not membership in the inter-foil slot region — these are distinct geometric regions.

**Confidence:** Medium. Grounded in aerodynamic reasoning. The slot channel is THE key physical feature of tandem airfoil aerodynamics. Very low complexity. If gap/stagger encoding (PR #2130) gave -3%, explicit slot membership should help further.

---

## Idea 9: Log-Wall-Distance Feature (Better DSDF Encoding)

**Slug:** `log-dsdf-feature`

**Target metrics:** p_in, p_tan

**Key bet / mechanism:**
The current DSDF features include raw signed distance to the foil surface. But boundary layer physics has a LOGARITHMIC dependence on wall distance (the law of the wall: u+ = (1/kappa) * ln(y+) + B). Velocity gradients in the boundary layer are proportional to 1/y near the wall. Using raw linear DSDF as an input means the network must learn to "undo" the logarithmic physics before it can reason about wall-normal profiles.

Proposal: add log(1 + DSDF) as an additional channel alongside the existing linear DSDF. This is analogous to positional encoding providing both raw and log-transformed position information. The model can use the log-transformed distance for boundary-layer physics reasoning and the linear distance for geometry reasoning.

This is architecturally zero-cost (just a monotone transformation of an existing feature) but gives the model both scales simultaneously.

```python
# Log-transformed wall distance channels
# dsdf1: [B, N] — raw signed distance to foil 1 (channels dsdf1_channel from data pipeline)
# dsdf2: [B, N] — raw signed distance to foil 2

dsdf1 = x[:, :, dsdf1_channel].clamp(min=0.0)     # unsigned distance (clamp negative)
dsdf2 = x[:, :, dsdf2_channel].clamp(min=0.0)

log_dsdf1 = torch.log1p(dsdf1 * 20.0) / torch.log1p(torch.tensor(20.0))  # log(1+20*d)/log(21)
log_dsdf2 = torch.log1p(dsdf2 * 20.0) / torch.log1p(torch.tensor(20.0))  # normalized to [0,1]

# Also add rank-based distance: 1/(1 + dsdf) — emphasizes near-surface nodes
inv_dsdf1 = 1.0 / (1.0 + dsdf1 * 10.0)           # [B, N]
inv_dsdf2 = 1.0 / (1.0 + dsdf2 * 10.0)           # [B, N]

new_feats = torch.stack([log_dsdf1, log_dsdf2, inv_dsdf1, inv_dsdf2], dim=-1)  # [B, N, 4]
x = torch.cat([x, new_feats], dim=-1)
```

**Why this is different from existing DSDF encoding:** The model currently sees 8 DSDF channels (scalar + gradient components for foil1 and foil2). But ALL are linear in the distance. The logarithmic wall-distance physics is forced to be learned from data. Providing log(d) directly shortcuts this learning and may help especially for OOD Reynolds numbers where the log-layer location shifts.

**Why it might help for p_re:** p_re regression is from 6.300 (PR #2213) to 6.364. Reynolds number scales wall-normal distances by Re scaling. Log-DSDF provides a Re-invariant representation of the boundary layer structure. When combined with the existing log(Re) feature, the model can directly compute Re-scaled wall distances.

**Risk:** Very Low. Pure feature addition. Adding log-transformed channels cannot hurt more than adding noise. If not useful, the model will zero out the weights on these channels.

**Confidence:** High. The law of the wall is one of the most established results in turbulence — log(y) dependence is fundamental. This is the simplest possible way to bake this physics into the input. Surprised it hasn't been tried in 1903 experiments.

---

## Idea 10: Freestream Momentum Deficit Feature

**Slug:** `momentum-deficit-feature`

**Target metrics:** p_re, p_oodc

**Key bet / mechanism:**
The wake deficit feature (PR #2213, -4.1% p_in, one of the biggest wins) encodes the VELOCITY deficit in the fore-foil wake for tandem configurations. It is computed analytically from the fore-foil geometry and flight conditions.

This idea extends that success to SINGLE-FOIL cases: encode the BOUNDARY LAYER MOMENTUM DEFICIT as a per-node feature. In viscous flow, the boundary layer has a momentum deficit compared to freestream. The momentum thickness theta is: theta = integral_0^inf (u/U_inf) * (1 - u/U_inf) dy. This can be APPROXIMATED from geometry + Re without solving the boundary layer equations.

For each mesh node, the approximate momentum deficit compared to freestream is:
`deficit_approx = 1.0 - exp(-dsdf * u_tau_approx * Re)`

where u_tau_approx is the friction velocity estimate (from Cf approximation in Idea 6). This gives a per-node "how much slower is this node expected to be vs freestream, based on boundary layer theory alone?"

```python
# Approximate momentum deficit per node from BL theory
# dsdf1: [B, N] — wall distance
# Re: [B] — Reynolds number  
# Umag: [B] — freestream velocity magnitude (=1 after normalization)

Re_val = x[:, 0, re_channel]                       # [B]
dsdf1 = x[:, :, dsdf1_channel].clamp(min=0)        # [B, N]

# Approximate friction velocity (turbulent flat plate)
Cf = 0.026 * Re_val ** (-1.0/7.0)                  # [B]
u_tau = torch.sqrt(Cf / 2.0)                       # [B] non-dimensional

# Approximate BL momentum deficit: exp(-y/delta_99) where delta_99 ~ Re^{-1/5}
delta_99 = 0.37 * Re_val ** (-0.2)                 # [B] turbulent BL thickness (Prandtl)

# Deficit: 1 near wall (strong deficit), 0 far from wall (freestream)
# clamp dsdf1 to avoid overflow
y_over_delta = dsdf1 / (delta_99.unsqueeze(1) + 1e-6)
momentum_deficit = torch.exp(-y_over_delta * 3.0)  # [B, N] — smooth decay

# Apply only near the body (far-wake nodes are different physics)
near_body_mask = (dsdf1 < 0.5).float()
momentum_deficit = momentum_deficit * near_body_mask

x = torch.cat([x, momentum_deficit.unsqueeze(-1)], dim=-1)  # 1 new channel
```

**Why it helps for p_re:** The Reynolds number determines the THICKNESS of the boundary layer and the distribution of momentum deficit with wall distance. For OOD Reynolds numbers (Re=4.445M vs training range), the boundary layer structure is physically different. By encoding a Re-scaled momentum deficit estimate, the model gets an explicit signal about WHERE the boundary layer should be for this specific Re value — reducing the extrapolation gap.

**Why it extends wake deficit success:** PR #2213 showed that an explicit per-node physics estimate (even an approximate one) of a physically relevant quantity can significantly improve accuracy. Momentum deficit is the same bet applied to single-foil boundary layer physics and OOD-Re generalization.

**Risk:** Low-Medium. The BL approximation is coarse but directionally correct. Main risk: confounding signal — the model already has dsdf + log(Re), so it can approximate this relationship internally. But providing it explicitly shortcuts this learning and may help in the OOD-Re setting where the model needs to generalize to unseen Reynolds numbers.

**Confidence:** Medium. Motivated by the strong success of wake deficit feature (PR #2213) and grounded in turbulence physics. Different target regime (single-foil BL thickness vs inter-foil wake interaction). Coarse approximation may add noise.

---

## Priority Rankings

| Rank | Slug | Target | Key bet | Complexity | Risk | Confidence |
|------|------|--------|---------|------------|------|------------|
| 1 | `fv-cell-area-loss-weight` | p_in, p_tan, p_re | Cell-area-weighted loss corrects structural mesh bias | Low | Low | High |
| 2 | `log-dsdf-feature` | p_in, p_tan, p_re | Log(d) shortcut for law-of-the-wall physics | Very Low | Very Low | High |
| 3 | `bernoulli-residual-feature` | p_in, p_re | Per-node inviscid Cp baseline for residual correction | Low | Low | Med-High |
| 4 | `tandem-slot-indicator` | p_tan | Explicit per-node slot channel membership | Low | Low | Medium |
| 5 | `did-feature` | p_oodc, p_re | Streamwise position encoding (AoA-adaptive) | Very Low | Low | Medium |
| 6 | `gmse-gradient-loss` | p_tan, p_in | Gradient-weighted loss: auto-targets high-gradient zones | Medium | Medium | Med-High |
| 7 | `shortest-vector-feature` | p_tan, p_oodc | Vector to nearest foil surface: explicit inter-foil disambiguation | Low-Med | Low | Med-High |
| 8 | `momentum-deficit-feature` | p_re, p_oodc | BL momentum deficit estimate: Re-scaled boundary layer signal | Low | Medium | Medium |
| 9 | `wall-layer-bin-feature` | p_in, p_tan | Turbulence zone indicator (y+ binning) | Low | Medium | Medium |
| 10 | `q-criterion-proxy-feature` | p_tan, p_oodc | Flow regime indicator: vortex vs strain zone per node | Low | Medium | Medium |

### Assignment priority for next 8 students

Given 8 idle student slots and 8 Round 29 WIP:
- Assign ideas 1-8 in order, with `fv-cell-area-loss-weight` and `log-dsdf-feature` as highest priority due to strong theoretical backing and very low complexity/risk.
- Hold `wall-layer-bin-feature` and `q-criterion-proxy-feature` as alternates if ideas 1-8 are assigned.

### Notes for implementation

1. **`fv-cell-area-loss-weight`**: First check if `batch.face_area` or equivalent is available in the data collation. If not, approximate from KNN distances. This is load-bearing for the idea so verify before assigning.
2. **`log-dsdf-feature`**: Must identify exact channel indices for DSDF scalars in `prepare_multi.py`. There are 8 DSDF channels total — need to identify which 2 are the scalar distances to foil1 and foil2.
3. **`gmse-gradient-loss`**: The gradient computation must be implemented carefully. Test on a single batch first. If edge-based gradient is noisy, fall back to 1/sigma^2 normalization using running statistics.
4. **All ideas**: Each adds at most 4 new channels to the 24-32-channel input. The `Transolver` input_dim must be updated accordingly via `--extra_features` flag or direct input_dim override.
