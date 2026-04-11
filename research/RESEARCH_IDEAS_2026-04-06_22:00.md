<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Research Ideas — 2026-04-06 22:00

Generated after reviewing 1,827 PRs (134 merged, 1,545 ran-not-merged). Current baseline:
p_in=11.979 | p_oodc=7.643 | p_tan=28.341 | p_re=6.300 (PR #2213, Wake Deficit Feature).

Merge thresholds: p_in < 11.98 | p_oodc < 7.65 | p_tan < 28.34 | p_re < 6.30.

In-flight (do not duplicate): #2216 (GALE), #2217 (Fore-SRF Skip), #2218 (LE coord frame),
#2219 (Fore→Aft cross-attn AftSRF), #2220 (Slice diversity reg).

---

## Hypothesis 1 (HIGHEST PRIORITY): Stagnation-Point Coordinate Frame

**Slug**: `stagnation-point-coord-frame`

**Rationale**: The TE coordinate frame (PR #2207, -5.4% p_in) and wake deficit feature (PR #2213,
-4.1% p_in) both demonstrated that explicitly encoding aerodynamically meaningful reference
points as input features is the highest-ROI intervention at this stage of training. Both are
trailing-edge-anchored. The leading-edge stagnation point is the other critical aerodynamic
feature: it is where the boundary layer starts, where the Cp peak occurs, and where pressure
gradients are largest. The stagnation point location shifts with AoA (for a NACA0012 at AoA=5deg
it moves ~3-5% chord from the geometric leading edge). Encoding each node's signed arc-length
distance and (dx, dy) offset from the stagnation point — estimated as the surface node with the
highest predicted pressure on the previous pass — would give the model direct access to the
pressure-gradient structure rather than requiring it to infer this from DSDF alone.

Critically, for OOD generalization (p_oodc, p_tan): stagnation point location is relatively
stable across Reynolds numbers and mild geometry changes, making this a more OOD-robust anchor
than trailing-edge position. The stagnation displacement encodes the effective AoA directly.

**Implementation**:
Add a new input feature set: for each mesh node, compute (dx, dy, dist) offsets from the
approximate stagnation point of each foil. The stagnation point can be approximated as the
surface node with minimum DSDF from each foil. Alternatively, use the geometric leading edge
(min-x surface node) as a fixed proxy — simpler and no inference loop required. This is
analogous to the TE coord frame but anchored at the LE/stagnation region rather than the wake.

Add to train.py:
- `--stagnation_coord_frame`: compute (dx_fore_le, dy_fore_le, dist_fore_le, dx_aft_le,
  dy_aft_le, dist_aft_le) = 6 new channels, using geometric leading edge (min-x surface node)
  as stagnation proxy. Zero aft-foil channels for single-foil samples.
- These 6 channels are appended to the 24-dim input, bringing input to 30 dims.
- The TE coord frame helper already exists in train.py from PR #2207 — mirror it for the LE.

**Baseline command addition** (add after `--wake_deficit_feature`):
```
--stagnation_coord_frame
```

**Expected gains**: -3% to -6% p_in (analogous to TE frame), likely p_oodc improvement since
stagnation region drives the highest-pressure spike. p_tan may benefit if LE location is more
stable across foil geometries than TE.

**Why not tried yet**: TE frame was so recent (same day as this write-up) that the natural
follow-on (LE frame) hasn't been assigned. PR #2218 is in-flight testing a "LE coordinate
frame" — IF #2218 is testing the same thing (LE as reference point), assign this as a
complement. If #2218 is the camber-line LE, this is distinct (stagnation-point-relative vs
geometric-LE-relative). Check PR #2218 body to confirm.

**Risk**: Medium. LE location is fairly invariant, so the feature should generalize well. The
main risk is saturation — the model already has DSDF channels encoding distance-to-foil-surface
which partially encode LE proximity.

---

## Hypothesis 2 (HIGH PRIORITY): Camber-Line Arc-Length Coordinate Feature

**Slug**: `camber-line-arc-coord`

**Rationale**: The DSDF channels encode distance to the foil surface but carry no information
about position *along* the surface. The TE and LE coordinate frames add reference-point offsets.
What's missing is a smooth parameterization of position along the camber line — the 1D skeleton
of the airfoil. Aerodynamics textbooks parameterize surface pressure as Cp(s) where s is
arc-length. If the model could access an approximate arc-length coordinate s ∈ [0, 1] (TE to TE
going suction side vs pressure side), it would have direct access to the variable that Cp is
natively a function of.

For OOD foils (NACA6416 tandem case): the camber-line arc-length is geometry-invariant in the
sense that it normalizes the parameterization to [0,1] regardless of chord length, camber, or
thickness. This is exactly the kind of invariant feature that helps OOD generalization.

**Implementation**:
For surface nodes: compute approximate arc-length position along each foil's surface. Surface
nodes are ordered (boundary ID = 6 for fore-foil, 7 for aft-foil). Sort by arc-length from
geometric stagnation (min-x). Assign s ∈ [0, 1] using cumulative chord length normalized by
total perimeter. For volume nodes: set to 0 (or the DSDF-weighted nearest-surface s value).

Add `--camber_arc_feature`: computes 2 channels (s_fore, s_aft) — arc-length position along
fore-foil and aft-foil surface, [0,1] normalized. Zero aft channels for single-foil samples.
Volume nodes get their nearest-surface s value.

This is a light, geometry-native feature with no hyperparameters.

**Expected gains**: -2% to -4% p_tan (camber-invariant parameterization helps OOD foil geometry
in the tandem case). Some p_in benefit from sharper surface pressure structure. p_oodc uncertain.

**Why not tried**: The feature space exploration has focused on reference-point offsets (TE, LE)
and distance fields (DSDF). Arc-length parameterization is a distinct axis — it's the natural
coordinate for 1D surface pressure, not a 3D distance field.

---

## Hypothesis 3 (HIGH PRIORITY): Pressure-Coefficient-Informed Surface Loss Reweighting

**Slug**: `cp-informed-surface-loss-weight`

**Rationale**: The current surface loss weights every surface node's MAE equally (after the
DCT frequency-weighted auxiliary term). But Cp distributions are highly non-uniform: the
leading-edge suction peak and trailing-edge separation region concentrate almost all the
aerodynamic interest and almost all the error. A node near the mid-chord of the pressure side
contributes the same gradient signal as a node in the suction peak, despite the peak being
harder to predict and more aerodynamically important.

The DCT loss (PR #2184) partially addresses this via frequency weighting (high-frequency = LE/TE
features), but operates in Fourier space on surface-ordered nodes — it doesn't adapt to the
*spatial* location of peaks.

Proposal: add a per-node loss weight that is proportional to the local pressure magnitude (or
gradient) estimated from the EMA model's current predictions. Nodes where |p_pred| is large or
where |dp/ds| is large get upweighted. This is a form of adaptive importance sampling that
concentrates gradient signal on the hardest, most important nodes.

**Implementation**:
During training, maintain a running estimate of per-node pressure magnitude. Use EMA smoothing
(τ=0.99) on |p_pred - p_freestream| to compute node-wise weights w_i. Normalize weights to
mean=1.0 within each sample. Surface loss becomes: L_surf = mean_i(w_i * |pred_i - target_i|).

Add `--cp_adaptive_weight --cp_adaptive_tau 0.99 --cp_adaptive_pct 1.5` where pct=1.5 is the
maximum weight multiplier for the highest-pressure nodes.

Alternative (simpler): use the ground-truth pressure magnitude as the weight (w_i = 1 +
alpha * |p_true_i - p_mean| / p_std). This has no additional parameters.

**Expected gains**: -2% to -5% p_in (suction peak accuracy). Potential p_tan improvement if
the model focuses more on the LE suction structure of the OOD foil.

**Why promising**: Adaptive per-sample loss upweighting (PR #2169) was "never ran" — it targeted
*sample*-level mining, not *node*-level. This is a different, finer-grained intervention.
Arc-length surface loss reweighting (PR #2210) is also never-ran but focuses on mesh density
correction, not pressure magnitude. This is distinct.

---

## Hypothesis 4 (HIGH PRIORITY): Velocity-Pressure Physics Consistency Loss

**Slug**: `bernoulli-consistency-loss`

**Rationale**: The model currently predicts (Ux, Uy, p) independently conditioned on geometry.
But these fields are coupled by the steady Bernoulli equation for incompressible flow along a
streamline: p + 0.5 * rho * (Ux^2 + Uy^2) = constant. This isn't enforced anywhere in the
current loss. Adding a soft consistency regularizer penalizing deviation from Bernoulli would
couple the velocity and pressure heads, potentially improving both — especially near the
surface where the boundary condition p + 0.5*|u|^2 = p_total is approximately satisfied even
in viscous flow (thin boundary layer approximation).

This is distinct from the failed "Normal-Velocity Hard Constraint" (PR #2187) which tried to
*hard-enforce* u_n = 0 at the wall. A soft auxiliary loss is much less likely to destabilize
training. The failed "Panel Cp Residual Target" (PR #2186) used analytical panel-method output
as a supervision target — this is instead a loss between the *model's own outputs*, requiring no
external solver.

**Implementation**:
Add a Bernoulli consistency auxiliary loss on surface nodes only:
  L_bern = mean( (p_pred + 0.5*(Ux_pred^2 + Uy_pred^2) - C)^2 )
where C is the inlet stagnation pressure (which equals Umag^2 * 0.5 in normalized units).
This is computed from the *predicted* values, not from ground truth, so it measures internal
consistency of the model's output.

Weight: `--bernoulli_loss --bernoulli_weight 0.01` (keep very small to avoid overriding MAE).

**Expected gains**: -1% to -3% on p_in and surface velocity (Ux, Uy on surface). The coupling
may help the pressure-first decoder (already in baseline) better condition velocity prediction.

**Why this works in principle**: The pressure-first sequential decoder (--pressure_first) already
conditions velocity on pressure prediction. A Bernoulli consistency loss makes this coupling
*explicit* at the loss level, providing gradient signal that couples the two heads.

**Risk**: If the pressure normalization (asinh transform) makes the Bernoulli equation non-trivial
to express, need to operate in the un-transformed (physical) space. The asinh transform is
applied to targets; the internal prediction operates in normalized space. This needs care.

---

## Hypothesis 5 (MEDIUM-HIGH): Reynolds-Scaled DSDF Feature Normalization

**Slug**: `re-scaled-dsdf-features`

**Rationale**: The current DSDF (distance-signed distance field) channels encode absolute
distances in mesh coordinates. But the physically relevant length scale for boundary layer
development is the viscous length δ ~ chord / sqrt(Re). At Re=1e6, the boundary layer is
much thinner than at Re=1e4 — but the model sees the same absolute DSDF values for both,
forcing it to learn the Re-dependence implicitly from the scalar Re input alone.

Explicitly scaling DSDF channels by 1/δ(Re) = sqrt(Re)/chord would give the model distance
inputs in "viscous units" rather than geometric units. This is the wall-normal coordinate used
in turbulence modeling (y+ scaling). The model would then see the same normalized DSDF structure
for the same Re, making the relationship between DSDF and boundary layer state more learnable.

**Implementation**:
For each training sample, compute a viscous length scale: delta = 1.0 / sqrt(Re). Multiply all
DSDF channels by delta (or log(Re)/log(1e6) as a softer normalization). The scalar Re is still
passed as a separate input feature.

Add `--re_scaled_dsdf` flag. No new parameters — pure input preprocessing.

**Expected gains**: -2% to -4% on p_re (OOD Reynolds number generalization, which is the
strongest signal for this feature). Possible p_oodc benefit.

**Why not tried**: DSDF augmentation (sigma scaling) has been tried (#2143, never ran). Feature
normalization by Reynolds number hasn't been tried. The physical motivation is strong — viscous
scaling is standard in CFD non-dimensionalization.

**Risk**: Low. This is a simple multiplicative rescaling of existing features. If it doesn't
help, it's trivially reverted.

---

## Hypothesis 6 (MEDIUM): Wake Centerline Coordinate Feature

**Slug**: `wake-centerline-coord`

**Rationale**: The wake deficit feature (PR #2213) encodes each node's position relative to the
fore-foil's trailing edge, gap-normalized. This captures the *source* of the wake. The wake
propagates downstream as a velocity deficit with a roughly Gaussian profile. Nodes near the
wake centerline (the horizontal line extending from the fore-TE) experience different local
flow than nodes displaced vertically. Encoding each node's signed vertical distance from the
wake centerline (dy_from_wake = y - y_fore_TE, normalized by gap) would give the model direct
access to its wake-relative position.

This is complementary to the existing wake_deficit_feature (dx/gap, dy/gap from fore-TE): the
existing feature encodes 2D offset from the TE point, while this would add a single feature
specifically encoding the vertical separation from the wake axis, which is the physically
dominant direction for wake-induced interference.

**Implementation**:
Add `--wake_axis_feature`: a single new channel encoding (y_node - y_fore_TE) / gap for all
nodes. This is already nearly derivable from existing wake_deficit_feature channels but makes
the wake axis separation *explicit*. Zero for single-foil samples.

Alternatively: encode the angle from the TE to the node relative to the freestream direction.
This is a single trigonometric feature encoding whether a node is inside the wake cone or not.

**Expected gains**: -1% to -3% p_in, -1% to -2% p_tan. The wake-axis separation is the primary
source of interference lift for tandem foils — explicitly encoding it reduces what the model
must learn implicitly.

---

## Hypothesis 7 (MEDIUM): Signed Pressure-Side / Suction-Side Label Feature

**Slug**: `suction-side-label-feature`

**Rationale**: The model must infer whether a surface node is on the suction side or pressure
side from DSDF features and coordinate geometry. This distinction is fundamental to airfoil
aerodynamics — the suction side carries most of the lift and has the strongest adverse pressure
gradient. An explicit binary (or signed) label indicating suction vs pressure side would give
the model a direct "which side of the foil am I on?" signal.

For standard airfoils (NACA0012 at positive AoA): suction side = upper surface (y > camber
line). This can be computed from the surface normal direction relative to the freestream.

**Implementation**:
Add `--side_label_feature`: for each surface node, compute sign(y_node - y_camber_line) where
y_camber_line is the camber-line y at the same x. For volume nodes, use the signed DSDF to the
nearest foil surface (already available). This is a single scalar feature.

More robustly: use the dot product of the outward surface normal with the freestream direction
to assign a "leeward/windward" label that rotates with AoA.

**Expected gains**: -1% to -3% p_in. The suction/pressure side distinction is the primary
driver of lift, and making it explicit should sharpen the surface pressure head.

---

## Hypothesis 8 (MEDIUM): Adversarial OOD Augmentation via AoA + Re Joint Perturbation

**Slug**: `ood-joint-augmentation`

**Rationale**: The current augmentation applies AoA perturbation (small Gaussian noise on angle)
and gap/stagger perturbation (σ=0.02) independently. Reynolds number perturbation was proposed
(PR #2125, "never ran") but as *independent* Re noise. The OOD challenge is that p_oodc tests
*combinations* of flow conditions not seen at training — not just extreme Re or extreme AoA
independently, but their joint distribution.

Proposal: during training, augment a fraction (20%) of in-distribution samples by jointly
sampling from a wider joint (AoA, Re) distribution that includes the test condition range.
This is a simple form of domain randomization applied specifically to the variables where we
have OOD splits.

**Implementation**:
Add `--ood_joint_aug --ood_joint_aoa_sigma 3.0 --ood_joint_re_scale 1.5`: for 20% of
in-distribution samples per batch, replace (AoA, Re) with samples drawn from N(AoA,
3deg^2) × U(Re/1.5, Re*1.5). The geometry is unchanged; only the flow condition is perturbed.
The DSDF features are not changed (they are geometry-dependent only), but the scalar Re and AoA
inputs to the model are augmented.

This is a very cheap augmentation — no geometric transformation needed.

**Expected gains**: -2% to -5% p_oodc (the primary OOD condition metric). Minimal cost.

**Risk**: If the current DSDF features encode flow information (they shouldn't — they are purely
geometric distance fields), this augmentation could create mismatches. Since DSDF is geometry-
only, it should be clean.

---

## Hypothesis 9 (MEDIUM): Smooth Chord-Normalized Coordinate Frame

**Slug**: `chord-normalized-coord-frame`

**Rationale**: The current coordinate system uses raw mesh coordinates (x, y) normalized by
a global scale. The chord length varies across foil shapes in the dataset. For OOD foil
shapes (different camber, thickness), the chord length is different — the model sees a NACA6416
foil at one physical scale and must recognize it as structurally similar to the NACA0012 it
trained on. Normalizing node positions by chord length (c) — so that the foil always spans
from (0,0) to (1,0) in the coordinate frame — would make the spatial encoding geometry-invariant
with respect to scale. This is exactly the coordinate normalization used in airfoil aerodynamics
textbooks (x/c parameterization).

**Implementation**:
Add `--chord_normalized_xy`: replace the raw (x, y) features passed to the spatial bias MLP
with (x - x_le) / c, (y - y_le) / c where x_le is the leading-edge x-coordinate and c is the
chord length. Compute these per-foil (separate normalization for fore and aft foil). Keep the
original (x, y) features too — just add the normalized versions as additional channels.

This is 2 new channels (x/c, y/c for the nearest foil). Straightforward preprocessing.

**Expected gains**: -2% to -4% p_tan. The chord normalization makes the spatial structure
of the NACA6416 test foil more similar to the NACA0012 training foils.

---

## Hypothesis 10 (SPECULATIVE): Gradient-Boosted Surface Correction (Post-Hoc Stacking)

**Slug**: `xgboost-surface-correction`

**Rationale**: This is a Kaggle-style idea: train a gradient-boosted model (XGBoost/LightGBM)
as a post-hoc corrector on the neural network's surface pressure residuals. The neural network
produces surface pressure predictions. On the training set, compute the residuals (pred - true).
Train a gradient-boosted model to predict these residuals from features: (x, y, AoA, Re, gap,
stagger, DSDF, TE features, arc-length s, suction-side label). At test time, add the gradient-
boosted correction to the neural network predictions.

This approach exploits the fact that gradient-boosted models excel at tabular regression where
neural networks underfit structured patterns — and surface pressure is essentially a tabular
regression problem (per-node, given per-node features).

**Implementation**:
After training completes, run an inference pass on the training set to get (pred, true) for all
surface nodes. Train an XGBoost regressor on (features → residual). Add `--gbm_correction`
flag. At inference, apply the correction.

**Complexity**: High. This would need to be integrated into train.py or as a post-processing
step. The student would need to implement XGBoost fitting within the training pipeline.

**Expected gains**: Uncertain. Could be 2-5% if the neural network's residuals are structured
(e.g., consistently underpredicting the suction peak). If residuals are random noise, no benefit.

**Risk**: High complexity vs uncertain gain. Should be deprioritized unless the simpler ideas
are exhausted.

---

## Summary Ranking (by expected impact)

| Rank | Slug | Target metric | Expected gain | Complexity |
|------|------|---------------|---------------|------------|
| 1 | stagnation-point-coord-frame | p_in, p_oodc | -3% to -6% p_in | Low |
| 2 | camber-line-arc-coord | p_tan, p_in | -2% to -4% p_tan | Low |
| 3 | cp-informed-surface-loss-weight | p_in, p_tan | -2% to -5% p_in | Low-Med |
| 4 | bernoulli-consistency-loss | p_in, surface vel | -1% to -3% p_in | Medium |
| 5 | re-scaled-dsdf-features | p_re, p_oodc | -2% to -4% p_re | Low |
| 6 | wake-centerline-coord | p_in, p_tan | -1% to -3% p_in | Low |
| 7 | suction-side-label-feature | p_in | -1% to -3% p_in | Low |
| 8 | ood-joint-augmentation | p_oodc | -2% to -5% p_oodc | Low |
| 9 | chord-normalized-coord-frame | p_tan | -2% to -4% p_tan | Low |
| 10 | xgboost-surface-correction | p_in, p_tan | 2-5% (uncertain) | High |
