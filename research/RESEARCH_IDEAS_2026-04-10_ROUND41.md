<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas for Round 41 — Bold Experiments

Generated: 2026-04-10
Baseline (2-seed avg): p_in=11.872, p_oodc=7.459, p_tan=26.319, p_re=6.229
Primary target: lower p_tan (tandem transfer surface MAE)

Round 40 WIP (do NOT duplicate): edward=#2362 (Viscous Residual), fern=#2363 (Global Cl/Cd SRF),
frieren=#2365 (FFD Geometry Aug), tanjiro=#2366 (MoE Domain-Expert FFN),
askeladd=#2367 (Biot-Savart Cross-Foil Attention), alphonse=#2368 (Sobolev Surface Gradient Loss)

---

## Idea 1: Surface-Node All-to-All Global Pressure Communication Layer

### Hypothesis
Adding a dedicated all-to-all attention layer that operates exclusively on surface nodes — after the main Transolver blocks but before the final output projection — will improve surface pressure accuracy by enforcing global incompressibility and Kutta condition constraints implicitly. For tandem cases, both foil surfaces attend to each other, enabling direct aerodynamic interference modeling.

### Scientific justification
The B-GNN paper (Jena et al., arXiv 2503.18638, 2025) shows that the key architectural insight for high-fidelity surface pressure prediction is global communication among all surface nodes. Standard message-passing or local attention misses long-range pressure correlations (e.g., leading-edge suction peak coupling to trailing-edge pressure recovery). Their model achieves 88% OOD error reduction over baselines specifically because all surface nodes communicate simultaneously. In tandem foils, the aft-foil leading edge and fore-foil trailing edge must "know" about each other — currently this signal travels through many Transolver layers with no direct path. A surface-only cross-attention block costs O(S²) where S ≈ few hundred surface nodes per batch, negligible compared to full O(N²) volume attention.

### Papers
- Jena et al., "B-GNN: Boundary Graph Neural Network for Aerodynamic Flow Prediction" arXiv 2503.18638 (2025) — surface-only all-to-all approach, 88% OOD reduction
- NeurIPS ML4CFD Competition 2025 winners — geometric inductive biases and physical consistency as key factors

### Implementation guidance
After the final `TransolverBlock`, extract surface node embeddings using `is_surface` mask. Run one standard multi-head self-attention over them (4 heads, dim=192, no positional encoding beyond existing node features). For tandem samples, include both fore- and aft-foil surface nodes in a single joint attention; for single-foil samples, include only fore-foil surface nodes. Project back and add as residual to the surface node positions in the full embedding before the output MLP. The surface attention should use the existing `DomainLayerNorm` for normalization. Total new parameters: ~4 * 192^2 * 3 ≈ 450K.

Key implementation detail: use a separate `nn.MultiheadAttention(embed_dim=192, num_heads=4, batch_first=True)` module. Gather surface nodes per sample via `is_surface` mask (pad to max surface count per batch), run attention, scatter back to full node positions. Do NOT add positional encoding — rely on existing features.

### Expected impact
- p_tan: -5% to -15% (direct foil-to-foil surface communication)
- p_in: -2% to -5% (global pressure self-consistency on single foil)
- p_oodc: -2% to -5%
- p_re: -1% to -3%

### Confidence
Medium-High. The B-GNN paper directly validates this mechanism. The implementation is moderate complexity but self-contained. Risk: surface node count varies per sample, requiring careful batched gather/scatter — student should benchmark memory overhead.

---

## Idea 2: Local Reynolds Number Re_x as Per-Node Input Feature

### Hypothesis
Adding a per-node local Reynolds number Re_x = (Re * |x_from_LE|) / chord as an input feature — where x_from_LE is the arc-length distance from the leading edge along the surface — will improve surface boundary layer modeling by directly encoding where laminar-to-turbulent transition is likely to occur. This single scalar feature should reduce surface pressure error especially for the OOD Reynolds number split.

### Scientific justification
The B-GNN paper (Jena et al. 2025) explicitly uses local Reynolds number Re_x as a key per-node feature, and it is listed as one of the primary reasons for the model's superiority over physics-agnostic baselines. Re_x encodes the local boundary layer state: values below ~5×10^5 indicate laminar flow, above indicate turbulent. This directly affects skin friction and pressure distribution near transition. Currently our model uses global Re as a scalar condition but does not give each surface node a local sense of "how far along the boundary layer am I." For OOD Re (p_re split), this should help generalize since Re_x scales predictably. This is a pure feature engineering change — one additional dimension to x — with no architectural modification.

### Papers
- Jena et al., arXiv 2503.18638 (2025) — local Re_x identified as key feature contributing to 83% model size reduction and 88% OOD error reduction
- Classic boundary layer theory (Schlichting, 1979) — Re_x as the fundamental dimensionless parameter for boundary layer thickness and transition

### Implementation guidance
Compute in `prepare_multi.py`... wait, that file is read-only. Instead compute inline in `train.py` during the feature assembly step. After extracting `raw_xy` and the global `Re_raw` (from `x[:,:,20]` or wherever the Re channel lives), compute:
```python
# Arc-length along surface from leading edge (approximate as cumulative chord distance)
# x_from_le: [B, N] — use |x_coord - x_leading_edge| as first approximation
x_le = x_coords * is_surface.float() + 1e6 * (~is_surface).float()
le_x = x_le.min(dim=1)[0]  # [B] — leading edge x per sample
x_from_le = (x_coords - le_x.unsqueeze(1)).clamp(min=0) * is_surface.float()
Re_x = Re_raw.unsqueeze(1) * x_from_le  # [B, N]
Re_x_log = torch.log1p(Re_x) / 15.0  # normalize; log1p+scale
```
Concatenate `Re_x_log` as an additional feature column in `x` before the input projection. This adds 1 dim to input size; update `in_channels` accordingly.

For aft-foil in tandem: use x_from_le relative to aft-foil's own leading edge (where saf_norm > 0.005 identifies aft-foil surface nodes).

### Expected impact
- p_re: -5% to -12% (directly encodes Re-dependent boundary layer physics)
- p_in: -2% to -5%
- p_oodc: -3% to -8%
- p_tan: -2% to -5%

### Confidence
High. The feature has strong physical basis and direct empirical support from B-GNN. Implementation is low complexity. Risk: leading-edge detection may be imprecise for highly cambered airfoils — using minimum x rather than minimum pressure side may require adjustment.

---

## Idea 3: Streamline-Curvature Pressure Gradient Feature

### Hypothesis
Adding a per-node centripetal pressure gradient estimate dp/dn ≈ ρ V²/R (where R is the local streamline radius of curvature approximated from velocity field geometry) as an input feature will improve surface pressure prediction by encoding the inviscid Euler normal-to-streamline momentum equation directly as a feature. This is especially valuable at the leading-edge suction peak where curvature is maximum.

### Scientific justification
The Euler equation normal to a streamline is dp/dn = ρ V²/R, where R is the local radius of curvature of the streamline. This is a first-principles relationship between pressure gradient and velocity curvature that the neural network currently has to learn implicitly. On the airfoil surface, the surface curvature (already available from feature #911) combines with the local velocity (estimated from surface tangent + boundary condition) to give a direct pressure gradient estimate. This is the classical laminar boundary layer pressure gradient parameter β = (θ²/ν)(dU/ds) that separates attached vs. separated flow. Our existing surface curvature feature captures the geometric part; adding the velocity-weighted version captures the aerodynamic part. Note this is different from the existing Vortex-Panel induced velocity (#2357) which gives the inviscid velocity magnitude, not its streamwise derivative.

### Papers
- Lighthill (1963) — streamline curvature and boundary layer separation; classical result
- Drela & Giles (1987), MISES solver — integrated boundary layer solver using curvature-pressure coupling
- NeurIPS ML4CFD 2025 competition analysis — pressure gradient features ranked among top-performing feature sets

### Implementation guidance
Approximate the streamline curvature at each surface node by combining the existing surface geometry curvature (κ from existing feature) with the panel-method Cp feature. The centripetal pressure gradient estimate: `dp_dn_est = -0.5 * rho * V_inf^2 * Cp_gradient_tangential`. Compute the tangential gradient of the existing `cp_panel` feature along the surface: for each surface node i, `dCp_ds ≈ (Cp[i+1] - Cp[i-1]) / (2 * arc_length_step)`. This requires sorting surface nodes by arc-length, which can be approximated by angle from centroid. Output: one scalar per node. Log-scale normalize. Add to input features.

Simpler fallback if arc-length ordering is hard: use finite differences on Cp in the x-direction on surface nodes only: `dCp_dx = gradient(Cp_panel, x_coord)` using PyTorch autograd or a simple neighbor lookup. This is a signed feature — preserve sign.

### Expected impact
- p_in: -3% to -8% (leading-edge suction peak accuracy)
- p_tan: -3% to -6% (fore-foil wake pressure coupling)
- p_oodc: -2% to -5%
- p_re: -2% to -5%

### Confidence
Medium. The physics is sound but implementation requires careful surface node ordering. May overlap partially with existing curvature feature. Recommend implementing as a standalone feature addition with ablation.

---

## Idea 4: Tandem-Specific Stagger-Normalized Inter-Foil Distance Feature

### Hypothesis
Adding per-node signed distance features relative to the aft-foil leading edge (for all nodes, not just surface) normalized by the stagger distance, will improve tandem transfer generalization by encoding the aerodynamic interference zone geometry explicitly. Nodes in the inter-foil gap region have distinctive pressure fields that the model currently can't localize well without knowing their position relative to the aft foil's leading edge.

### Scientific justification
Currently the model has wake deficit features (gap-normalized distance from fore-foil TE, PR #2213) but lacks the symmetric counterpart: distance to the aft-foil leading edge. The aft-foil leading edge is a stagnation point — the most important pressure feature of the tandem configuration. The inter-foil gap region is an aerodynamic "channel" where the stagnation pressure of the incoming flow is partially recovered, and the shape of this channel (set by stagger and overlap) determines aft-foil performance. The model currently must learn this geometry implicitly from coordinates. Explicit signed distance to aft-foil LE (positive = upstream, negative = downstream), normalized by stagger, gives the model a dimensionless quantity that generalizes across different gap/stagger combinations (which vary in tandem transfer split). This directly addresses p_tan — the OOD tandem geometry cases.

### Papers
- Jena et al., arXiv 2503.18638 (2025) — gap and stagger as key conditioning variables for tandem configurations
- Classical tandem airfoil aerodynamics: Schlichting & Truckenbrodt (1979) — stagnation point location as primary determinant of fore/aft coupling strength

### Implementation guidance
In `train.py`, after extracting `raw_xy` and identifying aft-foil surface nodes (saf_norm > 0.005), find the aft-foil leading edge (minimum x-coord on aft-foil surface):
```python
aft_surf = is_surface & (saf_norm > 0.005)
# Leading edge: minimum x on aft-foil surface
aft_x_safe = x_coords * aft_surf.float() + 1e6 * (~aft_surf).float()
aft_le_idx = aft_x_safe.min(dim=1)[1]  # [B]
aft_le_x = x_coords.gather(1, aft_le_idx.unsqueeze(1)).squeeze(1)  # [B]
aft_le_y = y_coords.gather(1, aft_le_idx.unsqueeze(1)).squeeze(1)  # [B]

# Stagger-normalized offsets from aft-foil LE (for ALL nodes)
stagger = (aft_le_x - fore_te_x).clamp(min=0.05)  # [B]
dx_aft_le = (x_coords - aft_le_x.unsqueeze(1)) / stagger.unsqueeze(1)
dy_aft_le = (y_coords - aft_le_y.unsqueeze(1)) / stagger.unsqueeze(1)
r_aft_le = (dx_aft_le**2 + dy_aft_le**2).sqrt()

# Zero for single-foil
is_tandem_flag = aft_surf.any(dim=1).float().unsqueeze(1)
dx_aft_le = dx_aft_le * is_tandem_flag
dy_aft_le = dy_aft_le * is_tandem_flag
r_aft_le = r_aft_le * is_tandem_flag
```
Add dx_aft_le, dy_aft_le, r_aft_le as 3 new input features. Update `in_channels`.

### Expected impact
- p_tan: -5% to -12% (direct geometric encoding of tandem interference zone)
- p_in: neutral (feature is zero for single-foil)
- p_oodc: neutral to slight improvement
- p_re: neutral

### Confidence
Medium-High. Feature is self-contained, physically motivated, and complementary to existing wake deficit features. Specifically targets p_tan which is our hardest metric. Low implementation risk.

---

## Idea 5: Asymmetric Surface Loss — Suction-Peak Weighted MAE

### Hypothesis
Replacing the current surface MAE with an asymmetrically weighted version that applies 3× weight to nodes with predicted Cp < -0.5 (suction peaks) will improve leading-edge accuracy by forcing the model to be especially precise at the most physically important and aerodynamically sensitive regions. These nodes currently contribute disproportionately to surface MAE errors.

### Scientific justification
The pressure distribution on an airfoil is not uniform in importance: the leading-edge suction peak (Cp ≈ -2 to -6 for typical cases) has the largest absolute error and drives lift coefficient accuracy. A uniform MAE treats a 0.1 error on a flat pressure recovery region the same as a 0.1 error at the suction peak — but the latter is physically catastrophic (it determines flow separation onset). Kaggle-style loss engineering shows that asymmetric loss functions consistently improve performance on structured prediction problems with sparse important regions. This is analogous to focal loss in object detection: down-weight easy examples, up-weight hard ones. For tandem cases, the aft-foil suction peak is the hardest prediction because the effective inflow angle depends on the fore-foil wake — exactly what p_tan measures.

Papers also suggest that pressure-weighted training objectives consistently outperform uniform objectives in CFD surrogate training. The Sobolev loss (#2368 WIP, alphonse) addresses derivative smoothness — this proposal addresses value-space weighting.

### Papers
- Lin et al. (2017) "Focal Loss for Dense Object Detection" — asymmetric weighting for rare important regions
- Drela (2014) "Flight Vehicle Aerodynamics" — suction peak as primary determinant of lift and separation
- Bhatnagar et al. (2019) "Prediction of aerodynamic flow fields using convolutional neural networks" — weighted loss on high-gradient regions

### Implementation guidance
In `train.py`, modify the surface loss computation. Currently the training loop computes surface MAE uniformly over all `is_surface` nodes. Replace with:
```python
# In training forward pass, compute predicted Cp proxy
# Pressure channel is index 2 of the target (p)
p_pred_surface = pred[is_surface, 2]
p_target_surface = target[is_surface, 2]

# Suction peak weight: 3× for strongly negative pressure
# Use target pressure to define weights (not predicted, to avoid gradient issues)
suction_weight = torch.ones_like(p_target_surface)
suction_mask = p_target_surface < -0.5  # strong suction
suction_weight = torch.where(suction_mask, torch.full_like(suction_weight, 3.0), suction_weight)

# Apply stagnation point up-weighting too (near Cp=1)
stagnation_mask = p_target_surface > 0.7
suction_weight = torch.where(stagnation_mask, torch.full_like(suction_weight, 2.0), suction_weight)

surface_loss = (suction_weight * torch.abs(p_pred_surface - p_target_surface)).mean()
```
Apply this only to the pressure channel of the surface loss. Keep Ux/Uy surface loss uniform. Adjust total loss weight balance so the new surface_loss term has the same total contribution as before (divide by mean(suction_weight) ≈ 1.3).

### Expected impact
- p_in: -3% to -8% (better suction peak accuracy)
- p_tan: -5% to -10% (aft-foil suction peaks under fore-foil wake)
- p_oodc: -3% to -8%
- p_re: -2% to -5%

### Confidence
Medium-High. Simple loss modification with clear physical motivation. No architectural changes. Risk: if suction peak threshold (-0.5) is wrong for this dataset's pressure scale, may overfit. Student should check the pressure value distribution in the dataset first and adjust threshold accordingly.

---

## Idea 6: Circulation-Conservation Auxiliary Loss for Tandem Pairs

### Hypothesis
Adding an auxiliary loss that penalizes the deviation of the predicted circulation around each foil (∮ v·dl) from a panel-method estimate will improve tandem pressure accuracy by enforcing a global constraint that's physically exact for inviscid flow. For tandem cases, the total circulation must sum to a consistent value with the freestream, providing a cross-foil constraint not present in the current pointwise loss.

### Scientific justification
Kelvin's circulation theorem states that for inviscid, irrotational flow, the circulation around any closed contour enclosing a lifting body is constant and equal to Γ = L / (ρ V∞) by Kutta-Joukowski theorem. This is a global integral constraint linking velocity to lift. Enforcing it as a soft loss term means the model can't simultaneously over-predict lift on fore-foil and under-predict it on aft-foil — which is exactly the type of error that causes p_tan failures. The panel-method Cp feature (already in the model) gives us a reference Γ_reference ≈ Cp_panel integrated around the surface. Penalizing |Γ_pred - Γ_reference| adds a global aerodynamic consistency constraint. Unlike point-wise physics losses (which have often failed in our history), this is a global integral constraint — aggregated, not local — making it much more robust to noise.

### Papers
- Kelvin's circulation theorem — classical fluid mechanics
- Jena et al. arXiv 2503.18638 (2025) — global incompressibility enforcement as key to B-GNN performance
- #2302 (Circulation Lift PR — never ran) — our own idea that was proposed but never executed; worth finally testing

### Implementation guidance
At the end of each forward pass, for surface nodes only, compute the predicted circulation per foil:
```python
def compute_circulation(pred_uv, surface_xy, surface_tangent, is_fore, is_aft):
    """Approximate ∮ v·dl = sum_i (u_i * tx_i + v_i * ty_i) * ds_i"""
    # surface_tangent: pre-computed unit tangent vectors along surface [B, N, 2]
    # ds: arc-length element between consecutive surface nodes [B, N]
    v_tangential = (pred_uv * surface_tangent).sum(-1)  # [B, N_surf]
    gamma_fore = (v_tangential * is_fore * ds).sum(-1)  # [B]
    gamma_aft = (v_tangential * is_aft * ds).sum(-1)    # [B]
    return gamma_fore, gamma_aft

# Panel-method Γ reference from cp_panel feature (already computed)
gamma_ref_fore = (cp_panel_fore * ds_fore).sum(-1)  # [B]

circ_loss = F.mse_loss(gamma_fore, gamma_ref_fore.detach())
# For tandem: penalize circulation ratio deviation from stagger-gap prediction
if is_tandem_sample:
    circ_ratio_loss = F.mse_loss(gamma_aft / (gamma_fore + 1e-6), expected_ratio)
    circ_loss = circ_loss + 0.5 * circ_ratio_loss

total_loss = main_loss + 0.01 * circ_loss
```
Pre-compute surface node arc-length spacing and tangent vectors during data loading (in the existing train.py feature assembly). The `cp_panel` feature already encodes the panel-method solution, so `gamma_ref_fore` is naturally available.

### Expected impact
- p_tan: -6% to -15% (cross-foil circulation consistency)
- p_in: -2% to -5%
- p_oodc: -3% to -6%
- p_re: -1% to -3%

### Confidence
Medium. The physical principle is sound and more principled than local residual losses (which failed in our history). The global integral form should be more robust. Implementation requires careful arc-length computation on unstructured surface meshes. Risk: panel-method Γ reference may have different calibration than CFD Γ — use only as a soft constraint with small weight (0.01).

---

## Idea 7: Learnable Per-Slice Physics Token Initialization

### Hypothesis
Instead of initializing the Transolver physics slice tokens from scatter-pooled node features, initialize them from a small MLP applied to the global flow condition vector (Re, AoA, gap, stagger). This gives each attention head's slice tokens a physics-aware starting point rather than a data-driven pool, potentially improving convergence and OOD generalization by anchoring the latent representation to known aerodynamic regimes.

### Scientific justification
The current `Physics_Attention_Irregular_Mesh` creates slice tokens via `slice_token = einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)` — a pure data-driven aggregation. The tokens carry no explicit knowledge of the macroscopic flow conditions. But in aerodynamics, we know that the flow structure varies dramatically with Re (laminar vs turbulent), AoA (attached vs separated), and gap/stagger (tandem coupling strength). A physics-aware initialization using a 2-layer MLP mapping [Re_norm, AoA_norm, gap_norm, stagger_norm] → [n_heads × slice_num × dim_head] would give each slice a flow-regime-specific prototype. This is analogous to conditioning the attention keys on a global context — similar to cross-attention conditioning in diffusion models (which has proven effective at enabling OOD generalization). For tandem transfer (p_tan), where gap and stagger are OOD, this would let the model anchor its internal representation to the known global parameters even when the local geometry is novel.

### Papers
- Rombach et al. (2022) "High-Resolution Image Synthesis with Latent Diffusion Models" — cross-attention conditioning for OOD generalization
- Transolver (Wu et al., 2024) — original slice-token design; this idea extends it with physics conditioning
- FiLM (Perez et al., 2018) — feature-wise linear modulation for conditioning; directly applicable here

### Implementation guidance
In `Physics_Attention_Irregular_Mesh.__init__`, add:
```python
# Global condition: [Re_norm, AoA_norm, gap_norm, stagger_norm] → 4 dims
self.cond_mlp = nn.Sequential(
    nn.Linear(4, 64), nn.SiLU(),
    nn.Linear(64, heads * slice_num * dim_head)
)
self.cond_scale = nn.Parameter(torch.zeros(1))  # learnable blend weight
```
In the forward pass, after computing `slice_token` via einsum, add:
```python
if global_cond is not None:  # [B, 4]: [Re_norm, AoA_norm, gap_norm, stagger_norm]
    cond_tokens = self.cond_mlp(global_cond)  # [B, H*S*D]
    cond_tokens = cond_tokens.view(B, heads, slice_num, dim_head)
    slice_token = slice_token + self.cond_scale.sigmoid() * cond_tokens
```
Pass `global_cond` through the TransolverBlock chain. Extract global conditions from the `x` feature tensor (Re, AoA etc. are already in channels 20-23 approximately). Initialize `cond_scale` near zero (torch.zeros) so the modification starts as a small perturbation.

### Expected impact
- p_tan: -5% to -10% (regime-aware slice initialization for OOD gap/stagger)
- p_oodc: -3% to -7% (OOD condition generalization)
- p_re: -3% to -6%
- p_in: -1% to -3%

### Confidence
Medium. Architecture modification with reasonable complexity. The "start near identity" initialization (cond_scale=0) ensures safe training behavior. Risk: the global condition vector may already be implicitly encoded in node features, making this redundant. Student should check gradient flow through cond_mlp.

---

## Idea 8: Dual-Branch Tandem Encoder with Explicit Foil Interaction Module

### Hypothesis
Processing fore-foil and aft-foil node embeddings through separate encoder branches (sharing weights) before a lightweight cross-attention interaction layer will improve tandem transfer accuracy by explicitly modeling the two-body aerodynamic coupling as a structured interaction rather than relying on the main Transolver blocks to discover it incidentally.

### Scientific justification
The core difficulty with p_tan (tandem transfer, our worst metric at 26.319) is that the aerodynamic coupling between fore and aft foils involves a fundamentally two-body problem: the fore-foil generates a wake that modifies the aft-foil's effective inflow angle, and the aft-foil's upwash modifies the fore-foil's circulation. Current architecture treats all nodes as a flat set with no special structure for the fore/aft split. A dual-branch encoder explicitly mirrors the physical structure: branch 1 encodes fore-foil node embeddings, branch 2 encodes aft-foil node embeddings, then a cross-attention module lets each foil "read" the other's summary representation. This is architecturally analogous to multi-agent attention in robotics (each agent = one foil) and molecule interaction encoding in drug discovery GNNs (each molecule = one foil). For single-foil samples, branch 2 produces a zero vector and the cross-attention is a no-op. Weight sharing between branches enforces symmetry (any foil could be fore or aft given different configurations).

### Papers
- Jiang et al. (2020) "Multi-Agent Graph Convolutional Networks" — explicit agent-level cross-attention
- Schütt et al. (2021) "Equivariant message passing for the prediction of tensorial properties" — two-body interaction modules in molecular property prediction
- B-GNN (Jena et al. 2025) — shows that direct inter-foil communication is critical for tandem accuracy

### Implementation guidance
After the initial input projection (before TransolverBlocks), split node embeddings by foil membership:
```python
fore_mask = is_surface & (saf_norm <= 0.005) | (~is_surface & ~is_aft_volume)  # approximate
aft_mask = ~fore_mask & is_tandem_sample  # aft-foil nodes

# Pool foil-level summary vectors
fore_summary = x_embed[fore_mask].mean(0).unsqueeze(0)  # [B, 1, D]
aft_summary = x_embed[aft_mask].mean(0).unsqueeze(0)   # [B, 1, D]

# Cross-attention: fore reads aft, aft reads fore
fore_from_aft = cross_attn(query=fore_summary, key=aft_summary, value=aft_summary)
aft_from_fore = cross_attn(query=aft_summary, key=fore_summary, value=fore_summary)

# Broadcast back to all fore/aft nodes as an additive bias
x_embed[fore_mask] = x_embed[fore_mask] + scale_fore * fore_from_aft.expand_as(x_embed[fore_mask])
x_embed[aft_mask] = x_embed[aft_mask] + scale_aft * aft_from_fore.expand_as(x_embed[aft_mask])
```
Use a single shared `nn.MultiheadAttention(192, 4)` for both cross-attention directions. Scale parameters initialized near zero. For single-foil samples, skip the interaction entirely (is_tandem guard). Note: this modifies embeddings before all TransolverBlocks, so the effect propagates through all subsequent attention layers.

Alternative simpler approach: instead of node-level split, just use mean-pooled foil embeddings (two summary tokens per tandem sample) and run one cross-attention layer, then add the result as a bias to all nodes of each respective foil. This is cleaner and less error-prone.

### Expected impact
- p_tan: -8% to -18% (explicit foil coupling; directly addresses the core tandem failure mode)
- p_in: neutral (single-foil samples skip interaction)
- p_oodc: neutral to slight improvement
- p_re: neutral to slight improvement

### Confidence
Medium-High. Mechanistically well-motivated. The dual-branch structure is novel for this codebase and targets the exact failure mode we're trying to fix (p_tan). Main risk: properly defining "fore vs aft" node membership for non-surface volume nodes may require careful implementation. Student should consider whether to apply dual-branch to surface nodes only (simpler, safer) or all nodes.

---

## Priority ranking for assignment

1. **Idea 8** (Dual-Branch Tandem Encoder) — highest expected impact on p_tan, novel architecture, directly addresses root cause
2. **Idea 2** (Local Reynolds Number Re_x) — highest confidence, simplest implementation, empirically validated by B-GNN paper
3. **Idea 1** (Surface All-to-All Attention) — strong theoretical basis, moderate complexity, validated mechanism
4. **Idea 4** (Stagger-Normalized Aft-LE Distance Feature) — simple feature, directly targets p_tan OOD
5. **Idea 5** (Suction-Peak Weighted Loss) — loss-only change, high confidence, simple implementation
6. **Idea 7** (Learnable Slice Physics Conditioning) — architectural, medium confidence, good OOD potential
7. **Idea 3** (Streamline Curvature Pressure Gradient) — good physics but higher implementation complexity
8. **Idea 6** (Circulation-Conservation Auxiliary Loss) — interesting but requires careful arc-length preprocessing
