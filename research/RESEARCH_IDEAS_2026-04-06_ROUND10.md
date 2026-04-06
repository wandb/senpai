<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Ideas — Round 10
# Date: 2026-04-06 ~14:30 UTC
# Author: researcher-agent
# Baseline: p_in=13.21, p_oodc=7.82, p_tan=28.50 (PRIMARY TARGET), p_re=6.45

## Context

All 8 students are active. Round 10 ideas are for the next wave of idle students. The primary target
remains p_tan < 28.0 (current single-model best beats the 16-seed ensemble 28.50 vs 29.1).

**Hard constraints for Round 10:**
- Do NOT duplicate any of the 8 WIP experiments.
- Do NOT duplicate any unassigned ideas from R5–R9.
- All ideas must be novel relative to the complete PR history (1810 PRs reviewed).
- Every idea must modify only `cfd_tandemfoil/train.py`.

**Key research patterns recalled:**
- Works: DSDF magnitude aug, aft-foil SRF, 2-way PCGrad, tandem-geometry-aware routing (GSB),
  DCT spectral loss. All exploit tandem-specific physics signals.
- Fails: Feature distribution manipulation (3 catastrophic failures), fore-foil SRF (4 failures),
  iterative 2-pass refinement (+6.6%), inter-foil distance feature (+2.4%).
- The gap between p_oodc=7.82 and p_tan=28.50 is a 3.6× ratio. This gap is structural, not
  primarily a training-dynamics issue. The model needs better representations of the tandem
  wake interaction, not just better optimization.

**Sources for Round 10:**
- Transolver++: arXiv 2502.02414 — Ada-Temp (per-point adaptive temperature) + Rep-Slice (Gumbel
  reparameterization). 13% mean gain on 6 benchmarks; 62% error reduction on aircraft surfaces.
- GeoMPNN: arXiv 2412.09399 — Multi-frame coordinates (LE + TE frames), NeurIPS 2024 ML4CFD
  Best Student award. +3–5 OOD-score pts from trailing-edge coordinate frame alone.
- Transolver: arXiv 2402.02366 — baseline architecture reference.
- GAOT: arXiv 2505.18781 — geometry-aware operator transformer, multi-scale GNO encoder.

---

## IDEA 1 (HIGHEST PRIORITY): Transolver++ Adaptive Slice Temperature (Ada-Temp + Rep-Slice)

**Slug:** `transolver-plus-ada-temp`

**Hypothesis:**
The current Transolver uses a fixed, shared softmax temperature for slice weight computation.
Transolver++ (arXiv 2502.02414, ICML 2025) shows that replacing this with a *per-point learnable
temperature* (Ada-Temp) and Gumbel-Softmax reparameterization (Rep-Slice) prevents slice-weight
homogenization — the pathological state where all nodes route to the same slice, erasing physics
distinctions. The paper reports 62% surface-error reduction from Ada-Temp alone on aircraft datasets
and 13% average improvement across 6 PDE benchmarks. For our tandem problem, homogenization is
plausible: the 96 slices are trained overwhelmingly on single-foil and NACA0012 tandem samples;
for NACA6416 OOD inputs, the routing may collapse to "average" states that lack the
tandem-coupling specificity needed for accurate aft-foil pressure.

This is distinct from the WIP SCA (frieren #2199), which adds a learnable diagonal scale D to
attention *logits* after the slice tokens are computed. Ada-Temp acts *before* slice aggregation,
sharpening which nodes contribute to each physical state — a different and complementary mechanism.

**Why p_tan specifically:**
Sharp, distinguishable slice assignments let the backbone represent the NACA6416 wake mode as a
distinct physics state rather than a smeared average. If slices are currently near-uniform for
OOD inputs, this is the single change most likely to fix it.

**Implementation — exactly what to change:**

In `Physics_Attention_Irregular_Mesh.__init__`, the existing slice weight computation uses a
fixed temperature τ₀ embedded in the softmax. Add:
```python
# Ada-Temp: per-point learnable temperature offset
if cfg.ada_temp:
    self.temp_proj = nn.Linear(n_hidden, 1)   # projects node features → scalar τ_offset
    self.tau0 = cfg.ada_temp_tau0             # base temperature (e.g. 0.1)
```

In `Physics_Attention_Irregular_Mesh.forward`, where `slice_weights = softmax(...)`, replace with:
```python
if self.ada_temp:
    tau = self.tau0 + self.temp_proj(x).squeeze(-1)  # [B, N]  per-point temp
    tau = tau.unsqueeze(-1).clamp(min=0.01)           # [B, N, 1]
    logits = self.slice_weight_proj(x) / tau          # [B, N, slice_num]
    if self.rep_slice and self.training:
        # Gumbel noise: add -log(-log(U)) with U ~ Uniform(0,1)
        gumbel = -torch.log(-torch.log(
            torch.empty_like(logits).uniform_(1e-7, 1 - 1e-7)
        ))
        logits = (logits + gumbel) / tau
    slice_weights = F.softmax(logits, dim=-1)  # [B, N, slice_num]
else:
    slice_weights = F.softmax(self.slice_weight_proj(x) / cfg.tau0, dim=-1)
```

Add config flags:
```python
ada_temp: bool = False           # Transolver++ per-point adaptive slice temperature
ada_temp_tau0: float = 0.1       # base temperature (same as original Transolver default)
rep_slice: bool = False          # Transolver++ Gumbel reparameterization of slice weights
```

**Suggested command:**
```bash
--ada_temp --ada_temp_tau0 0.1 --rep_slice
```
Run 2 seeds. Try `--ada_temp_tau0 0.05` in a follow-up if first run shows improvement.

**Critical implementation notes:**
- The `temp_proj` output is an *offset* from τ₀, not τ itself. Initialize `temp_proj.bias`
  to 0.0 (not negative — we want the initial temperature to be exactly τ₀ = 0.1).
- During eval (`.eval()` mode), skip Gumbel noise — it is training-only.
- torch.compile: `torch.empty_like(...).uniform_()` is compatible. The `clamp(min=0.01)`
  prevents division by near-zero temperature.
- Interaction with EMA: Ada-Temp's `temp_proj` weights are included in EMA as normal.
- Interaction with GSB: the `gap_stagger_spatial_bias` module feeds into `raw_xy` which feeds
  into the slice weight projection. Ada-Temp multiplies by a per-point scalar *after* this
  projection — fully orthogonal, no conflict.

**Source:** Transolver++, arXiv 2502.02414, ICML 2025. Code: github.com/thuml/Transolver_plus

**Confidence:** HIGH. Direct adaptation from the immediate successor to our backbone architecture.
The paper uses the same Transolver Physics-Attention as our model. Aircraft surface datasets in
ablations are directly comparable to our tandem foil surface task. The only uncertainty: our
model is already well-tuned (GSB, PCGrad, EMA) and the marginal gain may be smaller than in the
paper's vanilla baseline setting. Expected p_tan gain: -1% to -4%.

---

## IDEA 2 (HIGH PRIORITY): Trailing-Edge-Relative Coordinate Frame as Additional Input Features

**Slug:** `te-coord-frame`

**Hypothesis:**
The current input features use raw (x, y) Cartesian coordinates plus DSDF distances. For
aerodynamic pressure prediction, the *trailing edge* is the most critical geometric reference
point: it is where the Kutta condition is enforced, where pressure recovery terminates, and
where wake formation begins. Predictions on NACA6416 fail partly because the TE location and
shape differ significantly from training foils (NACA0012/2412/4412), yet the model's spatial
coordinate frame takes no special account of the TE.

GeoMPNN (arXiv 2412.09399, NeurIPS 2024 ML4CFD Best Student) uses *both* a leading-edge and
trailing-edge coordinate frame as input features, showing +3–5 OOD score points from the TE
frame alone (ablation in Table 2). The key insight: representing each node's position relative
to the trailing edge captures the aerodynamic coupling between a node and the wake origin in a
geometry-invariant way — the distance from a node to the TE encodes "how far into the recovery
region am I?" regardless of the absolute foil shape.

This is distinct from:
- Chord-normalized coords (R8 idea 4): that normalizes x/c, y/c by chord length — a global
  normalization. TE-relative coordinates use the TE as a local origin, preserving the signed
  distance to the separation point.
- Inter-foil distance feature (DEAD, #2195): that added distance to foil-2 center, which
  overfits tandem patterns. This adds distance to *own* foil TE — geometry-intrinsic.
- Fourier pos embed (R8 idea 10): that encodes (x, y) with sinusoidal functions. This changes
  the coordinate *origin*, not the frequency representation.

**Implementation — where to add code:**

In `Transolver.forward`, before the `preprocess` GatedMLP2, add TE-frame features:
```python
if cfg.te_coord_frame:
    # For each foil in the sample, find the trailing edge node:
    # TE = argmax of x_coordinate among boundary_id in {5,6} (fore) or {7} (aft)
    # Use scatter_max or topk over surface nodes per sample.
    fore_surf = (boundary_id == 5) | (boundary_id == 6)   # [B, N] bool
    aft_surf  = (boundary_id == 7)                         # [B, N] bool
    x_coords  = x[:, :, 0]                                 # [B, N]
    # Fore-foil TE: max x among fore surface nodes
    INF = 1e6
    fore_te_x = (x_coords * fore_surf.float() - INF * (~fore_surf).float()).max(dim=1)[0]  # [B]
    fore_te_y_idx = (x_coords * fore_surf.float() - INF * (~fore_surf).float()).argmax(dim=1)
    fore_te_y = x[:, :, 1].gather(1, fore_te_y_idx.unsqueeze(1)).squeeze(1)               # [B]
    aft_te_x  = (x_coords * aft_surf.float()  - INF * (~aft_surf).float()).max(dim=1)[0]   # [B]
    aft_te_y_idx = (x_coords * aft_surf.float() - INF * (~aft_surf).float()).argmax(dim=1)
    aft_te_y  = x[:, :, 1].gather(1, aft_te_y_idx.unsqueeze(1)).squeeze(1)                # [B]

    # Compute signed offset from fore-foil TE and aft-foil TE for each node:
    dx_fore_te = x_coords - fore_te_x[:, None]      # [B, N]
    dy_fore_te = x[:,:,1] - fore_te_y[:, None]      # [B, N]
    dx_aft_te  = x_coords - aft_te_x[:, None]       # [B, N]
    dy_aft_te  = x[:,:,1] - aft_te_y[:, None]       # [B, N]
    # Optionally: polar distance + angle from each TE
    r_fore_te  = (dx_fore_te**2 + dy_fore_te**2).sqrt().clamp(min=1e-6)   # [B, N]
    r_aft_te   = (dx_aft_te**2  + dy_aft_te**2).sqrt().clamp(min=1e-6)    # [B, N]
    # For single-foil samples: aft TE features → 0
    is_tandem = (x[:, 0, 21].abs() > 0.01).float()[:, None]   # [B, 1]
    dx_aft_te  = dx_aft_te  * is_tandem
    dy_aft_te  = dy_aft_te  * is_tandem
    r_aft_te   = r_aft_te   * is_tandem

    # Append as 6 new input channels: [dx_fore, dy_fore, r_fore, dx_aft, dy_aft, r_aft]
    te_feats = torch.stack([dx_fore_te, dy_fore_te, r_fore_te,
                             dx_aft_te,  dy_aft_te,  r_aft_te], dim=-1)  # [B, N, 6]
    x = torch.cat([x, te_feats], dim=-1)  # X_DIM += 6
```

Add config flags:
```python
te_coord_frame: bool = False   # Trailing-edge relative coordinate frame as input features
```

Update `fun_dim` in the `preprocess` GatedMLP2 input dimension accordingly: if `cfg.te_coord_frame`,
add 6 to `fun_dim`.

**Suggested command:**
```bash
--te_coord_frame
```
Run 2 seeds. No wandb group needed for first trial; add if promising to sweep variants
(polar-only vs Cartesian+polar, fore-only vs fore+aft).

**Critical implementation notes:**
- The `argmax` over masked surface nodes must be compile-compatible. Using `topk(1)` is safer
  than `argmax` for torch.compile. Alternatively, compute TE positions from DSDF channels:
  DSDF of foil-1 near 0 and x is max → fore TE.
- For single-foil samples (p_in, p_oodc, p_re tracks), `aft_surf` will be all-False. The
  `is_tandem` mask sets those features to 0 — treated as a sentinel for single-foil samples.
- Normalization: TE-relative offsets are in the same coordinate space as (x, y). The model
  may benefit from normalizing by chord length: `dx_fore_te / (fore_te_x - min_fore_x)`.
  Try unnormalized first (simpler, less fragile).
- torch.compile: all operations used (topk, gather, sqrt, clamp) are compile-compatible.

**Source:** GeoMPNN, arXiv 2412.09399, NeurIPS 2024. This idea directly applies the TE
coordinate frame contribution from the ML4CFD competition Best Student submission.

**Confidence:** MEDIUM-HIGH. Domain-motivated and empirically validated in a directly comparable
airfoil aerodynamics setting. The key uncertainty: our model already has DSDF features which
encode approximate distance to foil surfaces, so TE-relative offsets may be partially redundant.
However, DSDF gives distance to the surface, not direction from the TE specifically — those are
different signals. Expected p_tan gain: -1% to -3%.

---

## IDEA 3 (MEDIUM-HIGH): Arc-Length-Weighted Surface Loss — Compensate for Non-Uniform Mesh Density

**Slug:** `arclength-surface-loss`

**Hypothesis:**
The surface MAE is computed as a uniform mean over all boundary nodes. But surface meshes in
CFD are *not* uniformly spaced: the leading edge and trailing edge have many tightly-packed
nodes (to resolve sharp pressure gradients), while the mid-chord region has fewer, widely-spaced
nodes. This means the current L1 loss is implicitly downweighted at LE/TE (they appear many
times in the mean, so each contributes less per unit arc length) and overweighted at mid-chord
(fewer nodes → each one counts more). The consequence: the model is gradient-optimized primarily
for mid-chord accuracy, not for LE/TE accuracy.

Arc-length reweighting corrects this by weighting each node by `ds_i / sum(ds)`, where `ds_i`
is the approximate arc-length element at node `i` (half the sum of distances to adjacent nodes
along the surface). This makes the loss approximate the continuous L1 norm `∫|pred-true| ds`,
which is the geometrically correct measure regardless of mesh density.

This is complementary to and distinct from:
- Curvature loss weighting (WIP tanjiro #2197): weights by `|kappa_i|`, concentrating on
  high-curvature nodes. Arc-length weighting compensates for mesh density without assuming
  high curvature = high importance.
- DCT freq loss (merged #2184): weights high-frequency Fourier components. Operating in
  frequency space, not physical space.
- OHEM hard mining (DEAD #2169): selects the highest-error nodes. Arc-length is fixed geometry,
  not adaptive to errors.

**Implementation — where to add code:**

In `train.py`, in the surface loss computation section, compute arc-length weights once per
batch (cheap, geometry is constant):
```python
if cfg.arclength_surface_loss:
    # x_surf: [N_surf, 2] surface node positions (x, y) ordered along the boundary.
    # If nodes are NOT pre-sorted, sort by angle from foil centroid first.
    # Assume nodes within each boundary_id group are ordered consecutively (verify from
    # prepare_multi.py — this is typically true for CFD surface meshes).
    
    # Compute ds_i ≈ (|p_i - p_{i-1}| + |p_i - p_{i+1}|) / 2 for each surface node.
    # Use roll to get neighbors; mask the join points (first/last nodes):
    xy_surf = surf_positions   # [B, N_surf, 2] — extract from x[:, surf_mask, :2]
    ds_prev = (xy_surf - xy_surf.roll(1, dims=1)).norm(dim=-1)   # [B, N_surf]
    ds_next = (xy_surf - xy_surf.roll(-1, dims=1)).norm(dim=-1)  # [B, N_surf]
    ds = (ds_prev + ds_next) / 2.0                               # [B, N_surf]
    # Normalize: each node's weight = ds_i / sum(ds)
    arc_weights = ds / ds.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N_surf]
    # Apply to surface loss: weighted_MAE = sum(arc_weights * |pred - target|)
    surf_loss = (arc_weights * (pred_surf - target_surf).abs()).sum(dim=-1).mean()
else:
    surf_loss = (pred_surf - target_surf).abs().mean()
```

Add config flag:
```python
arclength_surface_loss: bool = False   # Weight surface loss by arc-length element ds_i
```

**Suggested command:**
```bash
--arclength_surface_loss
```
Run 2 seeds. If p_tan improves ≥0.5%, run 4 more seeds and also test combining with
`--curvature_loss_weight` (if tanjiro's #2197 has landed by then).

**Critical implementation notes:**
- The `roll`-based neighbors assume surface nodes are sequentially ordered within each
  boundary_id group. Verify this assumption from `prepare_multi.py` before implementing.
  If nodes are NOT ordered, sort by angle: `theta_i = atan2(y_i - cy, x_i - cx)` where
  `(cx, cy)` is the foil centroid.
- The first and last nodes in the sequence will have artifically large `ds_prev` or `ds_next`
  (wrapping around). For a closed surface (foil), `roll` is exact and no masking is needed.
- Normalize arc-length weights SEPARATELY for each foil (fore vs aft), then combine. This
  ensures that both foils contribute equally regardless of their mesh density ratio.
- torch.compile: `roll`, `norm`, `sum` are all compile-compatible.
- The val/loss metric and surface MAE metric are computed separately from training loss —
  ensure that `arclength_surface_loss` only affects the *training loss*, not the val metric.

**Confidence:** MEDIUM. Principled and theoretically correct. The key uncertainty is how much
node density varies at LE/TE: if density is already fairly uniform (well-resolved mesh), the
gain will be small. If LE/TE have 3–5× more nodes than mid-chord (typical in CFD), the
reweighting could be significant. Expected p_tan gain: -0.5% to -2%.

---

## IDEA 4 (MEDIUM): Polar Coordinate Augmentation for Spatial Bias MLP

**Slug:** `polar-coord-bias`

**Hypothesis:**
The current `spatial_bias` MLP receives `(x, y)` Cartesian coordinates (and optionally gap,
stagger scalars from GSB) to compute per-node slice routing logits. Cartesian coordinates are
a poor basis for the rotationally-periodic pressure patterns around airfoil surfaces: the model
must implicitly learn that "upper leading-edge" = small r, moderate θ, upper half. Adding
*polar coordinates relative to each foil centroid* — `(r_fore, θ_fore, r_aft, θ_aft)` — as
additional spatial_bias inputs gives the routing MLP a geometry-natural basis for distinguishing
upper/lower surface nodes, LE vs TE, and inter-foil gap regions, without changing the backbone
attention mechanism.

This differs from Fourier position embedding (R8 idea 10) which applies multi-scale sinusoidal
features to raw (x, y). Polar features are a *coordinate transformation* — they change the
reference frame — not a multi-scale Fourier encoding.

GeoMPNN (arXiv 2412.09399) demonstrates that polar coordinates with 4 rotation angles (+3 OOD
score pts in ablation) substantially outperform pure Cartesian representations in airfoil
aerodynamics. They apply this to the full input features; we apply it only to the spatial_bias
MLP (which is the key routing mechanism in our Transolver backbone), keeping the change
surgical.

**Implementation — where to add code:**

In `Transolver.forward`, compute polar features before the `spatial_bias` MLP:
```python
if cfg.polar_coord_bias:
    xy = x[:, :, :2]   # [B, N, 2] raw coordinates

    # Foil centroids: mean of surface nodes for each foil
    fore_mask = ((boundary_id == 5) | (boundary_id == 6)).float()[:, :, None]  # [B, N, 1]
    aft_mask  = (boundary_id == 7).float()[:, :, None]
    
    n_fore = fore_mask.sum(dim=1).clamp(min=1)   # [B, 1, 1]
    n_aft  = aft_mask.sum(dim=1).clamp(min=1)
    cx_fore = (xy * fore_mask).sum(dim=1) / n_fore.squeeze(-1)   # [B, 2]
    cx_aft  = (xy * aft_mask).sum(dim=1)  / n_aft.squeeze(-1)    # [B, 2]

    # Polar relative to fore-foil centroid
    d_fore = xy - cx_fore[:, None, :]                   # [B, N, 2]
    r_fore = d_fore.norm(dim=-1, keepdim=True)           # [B, N, 1]
    theta_fore = torch.atan2(d_fore[:,:,1:2], d_fore[:,:,0:1])  # [B, N, 1]
    sin_t_fore = theta_fore.sin()
    cos_t_fore = theta_fore.cos()

    # Polar relative to aft-foil centroid (zero for single-foil samples)
    is_tandem = (x[:, 0, 21].abs() > 0.01).float()[:, None, None]
    d_aft  = (xy - cx_aft[:, None, :]) * is_tandem
    r_aft  = d_aft.norm(dim=-1, keepdim=True)
    theta_aft  = torch.atan2(d_aft[:,:,1:2], d_aft[:,:,0:1]) * is_tandem
    sin_t_aft  = theta_aft.sin()
    cos_t_aft  = theta_aft.cos()

    # Append [r_fore, sin_fore, cos_fore, r_aft, sin_aft, cos_aft] to raw_xy
    polar_feats = torch.cat([r_fore, sin_t_fore, cos_t_fore,
                              r_aft, sin_t_aft, cos_t_aft], dim=-1)   # [B, N, 6]
    # raw_xy is currently x[:,:,:2] + optionally gap/stagger (4-6 dims total)
    # Extend to include polar_feats → raw_xy becomes [B, N, 10-12 dims]
    raw_xy_extended = torch.cat([raw_xy, polar_feats], dim=-1)
```

The `spatial_bias` MLP first linear changes from `Linear(4 or 6, 64)` to
`Linear(4+6 or 6+6, 64)` — i.e., 10 or 12 input dims. Add config flag:
```python
polar_coord_bias: bool = False   # Add polar coordinates to spatial_bias MLP input
```

**Suggested command:**
```bash
--polar_coord_bias
```
Run 2 seeds.

**Critical notes:**
- The centroid computation for aft foil will be noisy for single-foil samples (all zeros),
  but the `is_tandem` mask sets those features to 0 — safe sentinel.
- torch.compile: `atan2`, `norm` are compile-compatible. Avoid Python conditionals in
  the forward pass; use the `is_tandem` float mask instead.
- The spatial_bias MLP input dimension must be updated consistently with `--gap_stagger_spatial_bias`
  (which currently changes it from 4 to 6). With `--polar_coord_bias`, add 6 more.
- r_fore is in raw coordinate units — normalize by something sensible (chord ≈ 1.0 in our
  data). The raw magnitude is fine for a first test; if it diverges, normalize by `r_fore.clamp(0.1, 10.0)`.

**Confidence:** MEDIUM. Well-motivated by GeoMPNN ablations and standard aerodynamics practice.
The risk is that DSDF features already encode much of the same information as polar coordinates
(DSDF ≈ distance to surface, which correlates with r_fore). Polar coordinates add signed angular
information (θ) which DSDF does not provide. Expected p_tan gain: -0.5% to -2%.

---

## IDEA 5 (MEDIUM): Separate Batch Statistics for Tandem vs Single-Foil in Surface Refinement Head

**Slug:** `domain-split-srf-norm`

**Hypothesis:**
The AftFoilRefinementHead applies LayerNorm and outputs predictions in the same statistical
space regardless of whether it is processing a tandem sample or single-foil. But the aft-foil
pressure distribution in tandem configurations has a systematically different mean and variance
from the same foil in isolation: the wake-induced velocity deficit compresses the pressure range,
and the inter-foil channel creates a high-pressure region not present in isolation. This means
the SRF head's internal LayerNorm is implicitly averaging over two structurally different
distributions, which may prevent it from specializing optimally for each.

The `domain_layernorm` flag (already merged) addresses this at the backbone level: separate
learned scale/shift per domain for backbone norms. This idea extends the same principle
*specifically to the AftFoilRefinementHead* — giving the SRF head separate learned mean/var
rescaling parameters per domain. This is a 2-parameter addition per SRF LayerNorm (tandem
scale + tandem bias), initialized to the same as the default.

This is explicitly distinct from `domain_layernorm` (which conditions backbone LayerNorms on a
domain embedding) because: (a) it targets only the SRF head, not the backbone; (b) it uses a
hard domain indicator (is_tandem bool) rather than a soft domain embedding; (c) it is an
independent addition that stacks with `domain_layernorm`.

**Implementation — where to add code:**

In `AftFoilRefinementHead.__init__`, replace the standard `nn.LayerNorm(hidden_dim)` with a
new `DomainConditionedLayerNorm`:
```python
class DomainConditionedLayerNorm(nn.Module):
    """LayerNorm with separate affine parameters for tandem vs single-foil."""
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=True)
        # Tandem-specific delta: initialized to zeros so training starts from baseline
        self.tandem_scale = nn.Parameter(torch.zeros(dim))  # delta scale
        self.tandem_bias  = nn.Parameter(torch.zeros(dim))  # delta bias
    
    def forward(self, x, is_tandem):
        # is_tandem: [B, 1] or scalar float
        out = self.ln(x)   # [B, N_aft, dim]
        # Add tandem-specific correction (only active for tandem samples)
        out = out + is_tandem * (x * self.tandem_scale + self.tandem_bias)
        return out
```

In `AftFoilRefinementHead.forward`, pass `is_tandem` to each `DomainConditionedLayerNorm`.
Add config flag:
```python
domain_split_srf_norm: bool = False   # Separate LN affine params in AftFoilRefinementHead
```

Apply to all LayerNorms within the aft-foil SRF head MLP tower (typically 1-2 LN layers
in the 3-layer MLP with n_layers=3).

**Suggested command:**
```bash
--domain_split_srf_norm
```
Run 2 seeds.

**Critical notes:**
- Zero-initialization of `tandem_scale` and `tandem_bias` means the initial forward pass is
  identical to the current baseline — no regression risk at initialization.
- The `is_tandem` indicator must be the same one already used for the `tandem_ramp` flag,
  typically `x[:, 0, 21].abs() > threshold`.
- torch.compile: parameterized LN with a conditional float scaling is fully compatible.
- Do NOT apply to the main SRF head (only aft-foil SRF). The aft-foil head is the most
  impactful for p_tan and has the clearest distribution shift.

**Confidence:** MEDIUM. The reasoning is principled and the risk is low (zero-init). The
question is whether the existing `domain_layernorm` (backbone) already captures enough of
this effect, leaving little residual signal for an SRF-specific version. Estimated p_tan
gain: -0.5% to -2%.

---

## IDEA 6 (MEDIUM): Learnable Global Sink Tokens in Physics-Attention — "Register Tokens"

**Slug:** `attention-register-tokens`

**Hypothesis:**
Recent work on Vision Transformers (arXiv 2309.16588, "Vision Transformers Need Registers",
NeurIPS 2023) shows that adding a small number of learnable "register tokens" — global
memory slots that do not correspond to any input position — to the attention computation
eliminates pathological attention patterns (attention sinks, artifact high-norm tokens) and
improves OOD generalization. The mechanism: without registers, information that "doesn't
belong" to any position creates high-norm tokens that act as dumps; with registers, these
tokens are explicitly allocated for global state.

In Transolver's Physics-Attention, slice tokens (`slice_token = einsum("bnm,bnh->bmh", slice_weights, x)`)
are soft aggregates from mesh nodes. For OOD inputs (NACA6416), some slice tokens may
become "dump tokens" that absorb OOD signals without contributing to predictions. Adding K=4
learnable global register tokens that participate in the slice-token self-attention — and are
discarded before deslicing — provides explicit slots for global physics state, preventing slice
collapse to "average" modes.

This is distinct from eidetic states in Transolver++ (which changes the slice weight computation
mechanism). Register tokens are added to the *attention sequence* after slicing, not before.

**Implementation — exactly where to add code:**

In `Physics_Attention_Irregular_Mesh.__init__`, add:
```python
if cfg.register_tokens:
    self.register_tokens = nn.Parameter(
        torch.randn(cfg.register_k, n_hidden) * 0.02
    )  # [K, n_hidden], small init
```

In `Physics_Attention_Irregular_Mesh.forward`, after computing `slice_token = [B, S, n_hidden]`,
append register tokens:
```python
if self.register_tokens is not None:
    regs = self.register_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, K, n_hidden]
    augmented = torch.cat([slice_token, regs], dim=1)            # [B, S+K, n_hidden]
    attended = self_attn(augmented)                              # [B, S+K, n_hidden]
    slice_token = attended[:, :S, :]                             # [B, S, n_hidden] — discard registers
else:
    slice_token = self_attn(slice_token)
```

Add config flags:
```python
register_tokens: bool = False   # Add learnable global register tokens to Physics-Attention
register_k: int = 4             # Number of register tokens (paper recommends 4-8)
```

`register_k = 4` → 4 × 192 × 3 blocks = 2304 extra params (negligible).

**Suggested command:**
```bash
--register_tokens --register_k 4
```
Run 2 seeds. Try `--register_k 8` as a follow-up if first run improves.

**Critical notes:**
- The self-attention in `Physics_Attention_Irregular_Mesh` currently operates over
  `[B, slice_num, n_hidden]` = `[B, 96, 192]`. With K=4 registers, it becomes `[B, 100, 192]`.
  This is a fixed-size sequence → torch.compile compatible.
- Register tokens must be discarded (`[:, :S, :]`) before the deslice step. Do not pass
  register outputs back to node features.
- Initialize with small random init (0.02 std), NOT zeros — zero-init registers would
  collapse to zero gradient in first step.
- The `multi_head` attention in Physics_Attention uses `torch.einsum` rather than
  `nn.MultiheadAttention`. Verify the sequence dimension is correctly extended; the einsum
  pattern for `[B, S+K, n_hidden]` should be identical to `[B, S, n_hidden]`.

**Source:** "Vision Transformers Need Registers" (arXiv 2309.16588, NeurIPS 2023). Direct
application of register tokens to Physics-Attention slice tokens is novel; not tried in any
prior round of this research programme.

**Confidence:** MEDIUM. Strong theoretical motivation (attention sinks are a known issue in
transformers) and empirical support from ViT-based tasks. The application to Transolver's
slice-token attention is novel and the mechanism is plausible. Risk: with only 96 slices and
3 blocks, slice collapse may already be limited; the gain may be smaller than in ViTs with
thousands of patch tokens. Expected p_tan gain: -0.5% to -2%.

---

## IDEA 7 (MEDIUM-LOW): Pressure-Conditioned Attention Temperature (PCAT) — Sharper Routing for High-Pressure Nodes

**Slug:** `pressure-conditioned-attn-temp`

**Hypothesis:**
For pressure prediction, the most critical nodes are those with large pressure magnitude (LE,
TE, suction peak). But in the current attention routing, all nodes are treated equally
regardless of their likely pressure value. A physics-motivated improvement: condition the slice
routing temperature on an estimate of the node's pressure magnitude. Nodes likely to have
high pressure (indicated by geometric proximity to LE/TE, or by a fast single-pass MLP estimate
from DSDF features) should have *lower* temperature (sharper routing → more specialized
representation), while mid-chord nodes with smoother pressure can afford higher temperature
(softer routing → shared representations).

This is a step beyond Ada-Temp (Idea 1), which uses the full n_hidden hidden state as the
temperature input. PCAT uses a *physics-prior* to initialize the temperature estimate:
specifically, a precomputed weight proportional to `walldist_proxy^(-alpha)` (nodes near the
wall have lower wall distance → likely high pressure → get lower temperature → sharper routing).
The temperature is then fine-tuned by gradient descent from this physics-informed prior.

**Implementation — compact variant:**

This is a minimal version of Ada-Temp with a physics-informed initialization. In the spatial
bias MLP, which already computes per-node routing features from `(x, y, gap, stagger)`, add a
temperature output head:
```python
# In Physics_Attention_Irregular_Mesh:
self.temp_head = nn.Linear(64, 1)   # 64 = spatial_bias_hidden_dim output
nn.init.constant_(self.temp_head.bias, 0.0)
nn.init.normal_(self.temp_head.weight, std=0.01)
```

Then use `spatial_bias_out = spatial_bias_mlp(raw_xy)` (already computed) to get both the
routing logits AND a per-node temperature:
```python
temp_delta = self.temp_head(spatial_bias_out[:, :, :-1]).squeeze(-1)  # [B, N]
tau = (cfg.ada_temp_tau0 + temp_delta).clamp(min=0.02)                # [B, N]
slice_logits = spatial_bias_out[:, :, -1:] + x_proj   # existing routing logits
slice_weights = F.softmax(slice_logits / tau.unsqueeze(-1), dim=-1)
```

Note: this is ONLY recommended if Idea 1 (Ada-Temp) is NOT running simultaneously. If Ada-Temp
is assigned first and proves effective, PCAT can be retired. If Ada-Temp fails, PCAT is a
lightweight alternative: it repurposes the already-computed spatial_bias_out instead of adding
a full `nn.Linear(n_hidden, 1)` projection through the entire hidden state.

Add config flag:
```python
pcat: bool = False           # Pressure-conditioned attention temperature via spatial bias
```

**Suggested command:**
```bash
--pcat
```
Run 2 seeds. Only assign if Ada-Temp (#1 above) is not already running.

**Confidence:** LOW-MEDIUM. Related to Ada-Temp but with a physics-prior initialization angle.
The key risk is that the spatial_bias_out features are a low-dimensional, position-based signal
that may not contain enough information to predict which nodes need sharper routing. Expected
p_tan gain: -0.3% to -1.5%.

---

## Summary Table

| Rank | Slug | Lines of code | Risk | Expected p_tan | Source |
|------|------|--------------|------|----------------|--------|
| 1 | `transolver-plus-ada-temp` | ~25 | LOW-MED | -1% to -4% | Transolver++ arXiv 2502.02414 |
| 2 | `te-coord-frame` | ~30 | LOW-MED | -1% to -3% | GeoMPNN arXiv 2412.09399 |
| 3 | `arclength-surface-loss` | ~15 | LOW | -0.5% to -2% | First-principles geometry |
| 4 | `polar-coord-bias` | ~25 | LOW | -0.5% to -2% | GeoMPNN arXiv 2412.09399 |
| 5 | `domain-split-srf-norm` | ~20 | LOW | -0.5% to -2% | Domain adaptation theory |
| 6 | `attention-register-tokens` | ~15 | LOW | -0.5% to -2% | arXiv 2309.16588 (ViT Registers) |
| 7 | `pressure-conditioned-attn-temp` | ~15 | LOW-MED | -0.3% to -1.5% | Physics-motivated Ada-Temp variant |

**Recommended assignment order:**
1. `transolver-plus-ada-temp` — highest expected gain, strong prior from direct architecture
   successor. Assign to first idle student.
2. `te-coord-frame` — second highest, domain-validated in directly comparable airfoil CFD.
   Assign to second idle student.
3. `arclength-surface-loss` — low complexity, principled, easy to implement correctly.
4. `attention-register-tokens` — more novel, clean implementation, clear failure mode.
5. `polar-coord-bias` — lower priority due to DSDF partial-overlap, but low risk.
6. `domain-split-srf-norm` — assign after domain_layernorm is confirmed still in baseline.
7. `pcat` — only if Ada-Temp is not running; retire if Ada-Temp is assigned.

**Non-duplicates verified (checked against all R5-R9 ideas and all WIP PRs):**
- `transolver-plus-ada-temp`: not in any prior round. SCA (WIP #2199) is a different mechanism.
- `te-coord-frame`: not in any prior round. Different from chord-normalized coords (R8 idea 4)
  and inter-foil distance (DEAD #2195).
- `arclength-surface-loss`: not in any prior round. Different from curvature loss (WIP #2197).
- `polar-coord-bias`: not in any prior round. Different from Fourier pos embed (R8 idea 10).
- `domain-split-srf-norm`: not in any prior round. Different from backbone domain_layernorm
  (merged).
- `attention-register-tokens`: not in any prior round. Different from eidetic states (R7 idea
  3, still unassigned — that is slice reparameterization, not register tokens).

**Unassigned backlog reminder (R5–R9, still viable, NOT included in Round 10):**
The following ideas are still unassigned and should be considered if Round 10 ideas are all
assigned and additional students become idle:
- `iterative-srf` (R8), `wake-deficit-feature` (R8), `asinh-scale-anneal` (R8),
  `slice-diversity-reg` (R8), `chord-normalized-coords` (R8), `kutta-condition-loss` (R8),
  `tandem-biased-stochastic-depth` (R8), `tandem-feature-cross` (R8), `fourier-pos-embed` (R8),
  `geotransolver-gale` (R9), `piratenets-adaptive-residuals` (R9), `moon-optimizer` (R9),
  `mhc-residuals` (R9), Hopfield Memory Bank (R6/R7), Eidetic States (R7 idea 3),
  Adaptive Boundary Layer Sampling (R7), Learned Anisotropic Attention Kernel (R7),
  Transient Re Conditioning (R7), Vorticity Input Feature (R8 note).
