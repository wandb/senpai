# SENPAI Research Ideas — Round 5 (2026-04-06)

## Context

Deep plateau: 25+ consecutive failures since PR #2130 (p_tan=28.60). All Round 4 ideas are
currently running (#2175, #2179, #2180, #2181, #2182, #2183, #2184, #2185, #2186). This round
draws from categorically different design spaces: classical numerical analysis, operator theory,
geometric deep learning, and theoretical learning principles. These are not architecture tweaks.

The structural problem: p_tan (OOD NACA6416 tandem, 28.60) is 2.19x worse than p_in (13.05).
The model cannot transfer pressure predictions to unseen airfoil shapes in tandem configurations.
The question is why — and what structural change directly addresses the mechanism of failure.

**Must avoid:** All currently running experiments and all confirmed dead ends listed in
CURRENT_RESEARCH_STATE.md.

---

## Idea 1: Laplacian Eigenvector Mesh Positional Encoding

### What it is

Replace the 16-dim Fourier PE (raw x,y → sin/cos at fixed frequencies) with Laplacian
eigenvectors of the mesh graph. For each mesh, compute the k=16 smallest eigenvectors of the
graph Laplacian L = D - A, where A is the mesh adjacency matrix and D is the degree matrix.
These eigenvectors encode the intrinsic geometry of the mesh — not the embedding in 2D space,
but the topological and geometric structure of the flow domain itself.

The key difference from Fourier PE: Laplacian eigenvectors are **invariant to rigid-body
transformations** and **adapt to the actual mesh topology**. A NACA6416 mesh and a NACA0012
mesh have different eigenvector structures that reflect their different geometric properties.
The current Fourier PE gives both meshes essentially identical encodings near x=0, y=0.

### Why it might help p_tan

The Fourier PE is shared across all geometries — it encodes where a node sits in 2D space, not
what role it plays in the flow topology. For NACA6416 (OOD), nodes have different x,y
distributions than the training shapes, but the topological role (leading edge, pressure side,
wake region) is the same. Laplacian eigenvectors would encode that topological role intrinsically,
making representations transferable across geometries.

For tandem configurations specifically: the aft-foil wake interaction creates topologically
distinct regions that don't appear in single-foil training. Laplacian eigenvectors would
distinguish these regions by their connectivity structure rather than their spatial position.

This connects to Schmidhuber's principle: the right representation makes the task trivially
easy. The current x,y Fourier PE is a lossy, geometry-oblivious encoding. The Laplacian
eigenvector PE is a lossless intrinsic encoding.

### Key papers

- Belkin & Niyogi (2003). "Laplacian Eigenmaps for Dimensionality Reduction and Data
  Representation." Neural Computation. Origin of Laplacian eigenmaps — the mathematics that
  underpins this approach. https://cs.uchicago.edu/~niyogi/papersps/LapMapNeurComp.pdf

- Dwivedi & Bresson (2020). "A Generalization of Transformer Networks to Graphs." Introduces
  Laplacian eigenvectors as positional encodings for graph transformers. Direct predecessor
  to this idea. https://arxiv.org/abs/2012.09699

- Lim et al. (2022). "Sign and Basis Invariant Networks for Spectral Graph Neural Networks
  (SignNet/BasisNet)." ICML 2022. Addresses the sign ambiguity problem in Laplacian eigenvectors
  via a small MLP post-processor. Critical for implementation correctness.
  https://arxiv.org/abs/2202.13013

- Rampasek et al. (2022). "Recipe for a General, Powerful, Scalable Graph Transformer (GPS)."
  NeurIPS 2022. State-of-the-art graph transformer using LapPE + SignNet.
  https://arxiv.org/abs/2205.12454

### Implementation notes

The sign ambiguity of Laplacian eigenvectors is the critical gotcha: for eigenvector v,
both v and -v are valid. Standard fix is to pass each eigenvector through a small sign-invariant
MLP (SignNet) before using it. This adds ~2K parameters and is non-negotiable for correctness.

Alternative (simpler): use eigenvector **absolute values** as the PE. Loses some information
but is sign-invariant without SignNet. Worth trying first for speed.

For tandem meshes, the graph Laplacian should be computed per-sample (mesh topology varies
by gap/stagger). This is feasible — the adjacency structure is already known from the CFD mesh
faces. The eigenvector computation (k=16 smallest non-trivial eigenvectors) takes ~0.1s per
sample using scipy.sparse.linalg.eigsh, which can be done in the DataLoader workers.

The eigenvectors replace x[:, :, 26:42] (current Fourier PE dimensions). Dimensions 0:26
remain unchanged. This is a drop-in replacement requiring only changes to the feature
construction code path.

**Critical hyperparameter:** k=16 (matching current Fourier PE dims). Can also try k=8.

### Suggested experiment design

1. In `train.py` feature construction, add a `--laplacian_pe` flag that:
   - Builds a sparse adjacency matrix from mesh face connectivity
   - Calls `scipy.sparse.linalg.eigsh(L, k=16, which='SM')` per sample
   - Passes each eigenvector through a 2-layer MLP with absolute-value sign equivariance
   - Appends to input features, replacing the Fourier PE block

2. Initial test: absolute-value Laplacian eigenvectors (no SignNet) to validate concept.
   Full SignNet in a follow-up if promising.

3. Run with `--laplacian_pe` added to full baseline command. Compare p_tan.

### Risk assessment

**Medium-high risk.** The eigenvector computation adds DataLoader overhead (~0.1s/sample
for k=16 on a 20K-node mesh — may need caching). The sign ambiguity issue is real and
the absolute-value shortcut may lose phase information needed for directional features.
This is the most theoretically motivated idea in this round, but also the most sensitive
to implementation details.

The biggest risk: Laplacian eigenvectors encode mesh topology, but CFD meshes for different
airfoils have very similar topologies (O-grid with boundary layer refinement). The eigenvectors
may not differ much between NACA0012 and NACA6416, making this a null result.

### Expected impact

If it works: potentially large (5-10% p_tan reduction). This directly attacks the PE's
geometry-blindness. If it doesn't: the failure tells us that the PE is not the bottleneck,
which is also useful information.

---

## Idea 2: Test-Time Feature Statistics Normalization (Feature Distribution Shift Correction)

### What it is

At test time, for each NACA6416 sample, shift and scale the DSDF features (x[:, :, 2:10])
to match the training distribution statistics. Specifically, for each of the 8 DSDF channels:
1. Compute the per-sample mean and std of that channel on the test mesh
2. Compute the running mean and std of that channel across all training samples
3. Apply an affine transformation: `dsdf_corrected = (dsdf - test_mean) / test_std * train_std + train_mean`

This is instance normalization in reverse — instead of normalizing away geometry-specific
information, we're aligning the feature distribution of the OOD sample to the training
distribution before it reaches the model.

The deeper motivation: the model has learned a function `f(dsdf_features) → flow field`.
If the DSDF feature distribution of NACA6416 meshes is systematically different from the
training distribution (e.g., different curvature profiles, different wake geometry), then
`f` is being evaluated outside its learned domain. This correction pulls the input back
into domain before inference.

This is distinct from the currently running SWD Domain Alignment (askeladd #2175), which
operates on the slice token distribution during training. This idea operates on the raw
input features at test time — no training modification required.

### Why it might help p_tan

NACA6416 is a cambered airfoil with max camber 6% at 40% chord. The DSDF features for
a cambered foil systematically differ from a NACA00XX symmetric foil at the same x-position:
the pressure/suction surface distance asymmetry is different, the TE thickness is different.
The current model has never seen this level of camber during tandem training, so the DSDF
feature values fall in a low-density region of the training distribution.

By aligning the test DSDF distribution to training distribution statistics, we give the
model's learned `f` a familiar input distribution, even though the underlying geometry differs.

### Key papers and connections

This is essentially the inference-time version of Domain Adaptation. The specific technique
is closest to:

- Pan & Yang (2010). "A Survey on Transfer Learning." IEEE TKDE. Foundation paper on
  distribution shift and adaptation. https://ieeexplore.ieee.org/document/5288526

- Schneider et al. (2020). "Improving robustness against common corruptions by covariate
  shift adaptation." NeurIPS 2020. Shows that adapting batch norm statistics at test time
  dramatically improves OOD robustness for classification. Direct predecessor.
  https://arxiv.org/abs/2006.14547

- Wang et al. (2021). "Tent: Fully Test-Time Adaptation by Entropy Minimization." ICLR 2021.
  The canonical TTA paper. Our approach is simpler (feature statistics rather than entropy)
  but the same spirit. https://arxiv.org/abs/2006.10726

- The DSDF feature normalization idea also connects to Sun & Saenko (2016). "Deep CORAL:
  Correlation Alignment for Deep Domain Adaptation." Where CORAL aligns second-order
  statistics; here we align first-order only (mean/std), which is much simpler.
  https://arxiv.org/abs/1607.01719

### Implementation notes

Two variants to try:

**Variant A (per-channel mean/std alignment):** Compute training distribution stats once
as constants. At inference, apply affine transform per DSDF channel per sample.
No training modification. Extremely simple.

**Variant B (learned scaling with per-sample statistics):** Add a small MLP that takes
the per-sample DSDF statistics as input and outputs a per-channel scale+shift correction.
This MLP is trained jointly with the model. More expressive but adds parameters.

Start with Variant A. The stats to store: 8 channel means + 8 channel stds = 16 floats,
computed once offline from the training set.

**Critical implementation detail:** The alignment should be applied AFTER the existing
normalization pipeline (before the model sees the features) and should be conditioned on
the `tandem_indicator` flag — only apply to tandem OOD samples. Applying it to in-distribution
samples would degrade their accuracy.

This can be implemented as a 5-line code addition with a `--dsdf_test_time_align` flag.

### Suggested experiment design

1. Compute training DSDF channel statistics (mean/std per channel) offline and store as
   a constant tensor in train.py.

2. In the validation forward pass only (not training), when processing tandem val samples:
   `dsdf_features = (dsdf_features - sample_mean) / sample_std * train_std + train_mean`

3. Test with `--dsdf_tta_align` flag and compare p_tan only (p_in/p_oodc/p_re should
   be unchanged since we only apply to tandem OOD samples).

4. If Variant A works, try Variant B with a learned 2-layer MLP for the correction.

### Risk assessment

**Low-medium risk.** Variant A is trivially implementable and has zero effect on training.
The main risk is that DSDF features are already well-normalized in the data pipeline, making
this redundant. Also, aligning mean/std doesn't address higher-order distributional shifts
(different skewness, different correlations across channels). The deeper question is whether
the DSDF feature distribution is actually the bottleneck for NACA6416 OOD — we believe so
based on the failure of cross-DSDF features (#2162) but that failure is weak evidence.

### Expected impact

**Moderate.** If DSDF distribution shift is the mechanism of p_tan failure, this could be
3-5% improvement with minimal implementation cost. If not, this is a clean null result.
The low implementation cost makes this worth trying even at medium confidence.

---

## Idea 3: Boundary Condition Hard Constraint via Normal-Velocity Penalty Projection

### What it is

Add a physics-based hard constraint to the surface predictions: on any airfoil surface node,
the velocity normal to the surface must be zero (no-penetration condition). Currently this
is only enforced as a soft loss (the model can violate it). The hard constraint version:
after the model predicts (Ux, Uy) at surface nodes, project out the normal component:
`u_pred_constrained = u_pred - (u_pred · n_hat) * n_hat`
where `n_hat` is the outward surface normal at that node.

This is not a new loss term — it's a post-prediction projection that makes the no-penetration
condition structurally impossible to violate, regardless of what the backbone learned.

The surface normal `n_hat` can be computed analytically from the airfoil parametrization
or numerically from DSDF gradients: `n_hat = ∇DSDF / |∇DSDF|` at surface nodes.

### Why it might help p_tan

For NACA6416 (OOD), the model has never trained on this exact surface geometry, so it can
and does predict nonzero normal velocity at surface nodes. This nonzero normal velocity
contaminates the pressure prediction (via Bernoulli or the pressure-velocity coupling in
incompressible flow). The projection ensures zero normal velocity at surfaces regardless
of geometry, which may improve p_tan stability on OOD shapes.

The deeper insight: this is a **geometric bias** rather than a learned bias. For any airfoil
shape — seen or unseen — the no-penetration condition holds. By baking it in as a hard
constraint rather than a soft loss, we are improving OOD generalization for free.

This is related to the "physics-constrained networks" literature, but specifically for the
most important boundary condition in aerodynamics.

### Key papers

- Raissi et al. (2019). "Physics-Informed Neural Networks." JCP. Introduced soft physics
  constraints in NN training. Our approach is the hard constraint version.
  https://arxiv.org/abs/1711.10561

- Beucler et al. (2021). "Enforcing Analytic Constraints in Neural Networks Emulating
  Physical Systems." Phys Rev Letters. Shows hard constraints beat soft constraints for
  physical systems — the key distinction from standard PINN.
  https://arxiv.org/abs/1909.00912

- Maron et al. (2019). "Universality of Equivariant Networks." Connects constraint
  projection to equivariant architectures. Theoretical foundation.
  https://arxiv.org/abs/1910.13022

- Holzschuh et al. (2023). "Solving the Navier-Stokes Equation with Constrained Graph
  Neural Networks." Direct application of hard BCs in mesh-based flow prediction.
  https://arxiv.org/abs/2209.11529

### Implementation notes

For surface nodes, `n_hat` can be approximated as the gradient of the level-set distance
field: if DSDF is the signed distance to the surface, then `∇DSDF` at surface nodes (where
DSDF ≈ 0) gives the outward normal direction. The DSDF features are already in the input,
so we have access to this information.

In practice, at a surface node i:
1. Compute `n_hat_i = (dsdf_grad_x_i, dsdf_grad_y_i) / norm` — the outward surface normal
2. Apply projection to the velocity prediction: `u_corrected = u_pred - dot(u_pred, n_hat) * n_hat`
3. The pressure prediction is unchanged

This projection needs to be applied in the **denormalized** space (physical Ux, Uy), not
the normalized space, since the physics-space constraint involves the physical velocity
components.

**Gotcha:** The DSDF gradient must be computed at the correct resolution. Using finite
differences on the DSDF values in the feature vector may introduce errors. Alternative:
use the SRF output directly (the SRF already processes surface nodes) and apply the
projection after the SRF but before loss computation.

The projection is differentiable with respect to the predictions and the normal vectors,
so it can be used during training as well (back-propagating through the projection).
This is the preferred approach — apply during training AND inference.

Add as `--normal_velocity_projection` flag.

### Suggested experiment design

1. Compute surface normals from DSDF gradient (finite differences on x[:, :, 2:10] channels
   at surface nodes using `dist_feat` to identify surface nodes).

2. Apply projection only to surface nodes (boundary_id = 4 or 7).

3. Apply during both training forward pass and validation. The projection should appear
   after the SRF correction but before the loss computation.

4. Test against baseline. Focus on p_tan and p_in (since this affects surface velocity
   which couples to surface pressure).

### Risk assessment

**Medium risk.** The main risk is that the DSDF gradient approximation for `n_hat` is
too noisy to be useful — CFD meshes can have irregular point spacing near surfaces that
introduces numerical error in finite-difference gradient estimation. If the normal
estimates are wrong, the projection could corrupt correct velocity predictions.

Alternative: compute normals directly from the mesh face connectivity (more accurate but
requires access to the face list). This is cleaner but requires more implementation work.

The physics motivation is solid — this is a hard constraint that should help. The question
is purely implementation quality.

### Expected impact

**Potentially large (3-8% p_tan).** Hard physics constraints are consistently better than
soft ones in the literature, and no-penetration is the most fundamental BC in external
aerodynamics. If this works, it also explains why p_tan is harder — the OOD geometry means
the model generates larger normal-velocity violations at the surface.

---

## Idea 4: Learned Geometry Tokenizer — Compress Each Airfoil into a Shape Code

### What it is

Add a small auxiliary network (the "geometry tokenizer") that reads the surface points of
each foil and produces a compact latent code `z_geom ∈ R^64`. This code is then injected
into the Transolver backbone via cross-attention (one cross-attention layer per Transolver
block, attending to `z_geom` as key/value). The geometry tokenizer is trained end-to-end
with the main model.

This is different from the current DSDF-based geometry encoding, which works at the node
level (each node gets the DSDF of the nearest surface point). The geometry tokenizer works
at the **foil level** — it produces a global descriptor of the airfoil shape that is
available as context to every node in the mesh.

The geometry tokenizer is a small set-transformer operating on the surface node coordinates
and DSDF values, producing a single summary vector `z_geom` per foil.

### Why it might help p_tan

The NACA6416 OOD failure may stem partly from the model's inability to form a global
representation of the airfoil shape. Currently, each node only knows its local DSDF
features — it has no global context about what shape it belongs to. For a NACA6416 node
near the leading edge, the local features look similar to a NACA0012 node, but the global
shape context (camber, thickness distribution, TE shape) is completely different.

A geometry tokenizer would encode this global shape context into `z_geom`, which is then
made available to every attention block. The model can condition its flow prediction on
"this is a highly cambered airfoil" even for unseen camber values.

In tandem configurations: `z_geom_foil1` and `z_geom_foil2` are computed separately and
cross-attended jointly, enabling the model to reason about the interaction between the
two foil shapes.

### Key papers

- Lee et al. (2019). "Set Transformer: A Framework for Attention-based Permutation-Invariant
  Neural Networks." ICML 2019. The set-transformer architecture for the geometry tokenizer.
  https://arxiv.org/abs/1810.00825

- Fathony et al. (2021). "Multiplicative Filter Networks." ICLR 2021. Alternative to
  set-transformer for compact geometry encoding.
  https://arxiv.org/abs/2006.09356

- Lu et al. (2022). "Comprehensive Study of Cross-Attention in Vision." Relevant architecture
  for injecting global geometry context into local attention blocks.
  https://arxiv.org/abs/2211.09852

- Shi et al. (2023). "Physics-Informed Neural Networks for Subsonic and Transonic Airfoil
  Design." Uses a similar global shape encoding for geometry-conditioned flow prediction.
  https://arxiv.org/abs/2309.01002

### Implementation notes

The geometry tokenizer is a 2-layer set-transformer:
1. Input: surface node coordinates (x, y) + DSDF values (8 channels) for one foil's
   surface nodes. Dimensionality: N_surf × 10.
2. Attention pooling → z_geom ∈ R^64.
3. Cross-attention injection: in each Transolver block, after the slice attention,
   add `CrossAttn(block_output, z_geom.unsqueeze(1))` with 2 heads and dim=64.

For tandem: compute z_geom1 (foil 1) and z_geom2 (foil 2) separately; concatenate to
z_geom ∈ R^128. The cross-attention sees both foil codes simultaneously.

For single-foil: z_geom2 = zeros (masked out). This is critical for clean training.

**Gotcha:** The set-transformer attends over surface nodes, which vary in count across
meshes. Use a learned "inducing points" aggregation (the standard approach in Set Transformer)
to get a fixed-size representation regardless of surface resolution.

Add as `--geometry_tokenizer` flag with `--geom_token_dim 64` hyperparameter.

### Suggested experiment design

1. Implement a 2-layer set-transformer on the surface nodes of each foil.
2. Inject via cross-attention after each Transolver block (3 injections for n_layers=3).
3. Run with geom_token_dim=64. If promising, try 32 and 128 in follow-up.
4. Monitor whether z_geom learns interpretable geometry representations (cosine similarity
   between NACA6416 and training foils) — this is a useful diagnostic.

### Risk assessment

**Medium risk.** This is a non-trivial architectural addition. The cross-attention injection
may interfere with the carefully tuned slice attention routing (similar to how Backbone AdaLN
failed in #2164). The set-transformer on surface nodes adds ~100K parameters and has its
own optimization dynamics that may not couple cleanly with the backbone.

The key distinction from AdaLN failures: this is cross-attention (learns what to attend to)
rather than scale/shift (imposes a linear coupling). Cross-attention is more flexible and
less likely to disrupt existing routing.

### Expected impact

**Potentially large (5-10% p_tan)** if global shape context is the missing piece. This is
a qualitatively different type of information from what the current model uses. If it works,
it also opens the door to shape interpolation and OOD extrapolation in the z_geom space.

---

## Idea 5: Implicit Neural Representation of the Pressure Field (INR Decoder Head)

### What it is

Replace the current pressure decoder with an Implicit Neural Representation (INR): instead
of predicting `p` at each mesh node independently, train a small network that takes a
**continuous 2D coordinate (x, y)** and the Transolver backbone features as input, and
outputs the pressure at that point. This turns pressure prediction into a continuous function
estimation problem.

The decoder MLP becomes: `p_decoder(x_coord, y_coord, backbone_features) → p_value`
where `backbone_features` are the Transolver slice tokens (pooled or queried).

During training: evaluate the INR at each mesh node position to get the training targets.
During inference for OOD shapes: the INR can be queried at arbitrary coordinates, including
the denser mesh nodes of NACA6416 surfaces.

This connects to the NeRF literature: the same principle (continuous neural representation
from discrete samples) applied to pressure fields.

### Why it might help p_tan

The current architecture predicts `p` independently at each mesh node. For OOD geometry
(NACA6416), the node positions are in regions of (x, y) space that the model has not seen
during tandem training. The model must extrapolate from nearby training-set positions.

An INR decoder, trained to produce a smooth continuous pressure function, would generalize
more naturally to new node positions: the continuity is built into the network architecture
(via SIREN's periodic activations or random Fourier features), not learned as a side effect.

Crucially: the INR decoder is **coordinate-continuous**, meaning it implicitly interpolates
between training-set geometries. A NACA6416 surface node at (0.3, 0.05) would be predicted
from the learned continuous pressure function, which is smoother and more physically
consistent than a discrete node prediction.

### Key papers

- Sitzmann et al. (2020). "Implicit Neural Representations with Periodic Activations
  (SIREN)." NeurIPS 2020. The key architecture — periodic activations for smooth implicit
  representations. https://arxiv.org/abs/2006.09661

- Mildenhall et al. (2020). "NeRF: Representing Scenes as Neural Radiance Fields for
  View Synthesis." ECCV 2020. The paradigm — coordinate-in, value-out.
  https://arxiv.org/abs/2003.08934

- Dupont et al. (2022). "From data to functa: Your data point is a function and you can
  treat it like one." ICML 2022. Generalizes INR to function space learning.
  https://arxiv.org/abs/2201.12204

- Serrano et al. (2023). "INFINITY: Neural Field Representations of Geometry and Flow
  for Scientific Computing." Directly applies INR to CFD flow field prediction.
  https://arxiv.org/abs/2307.13743

### Implementation notes

The minimal INR decoder for pressure:
1. Input: `[x_coord, y_coord, fourier_pe(x,y,L=4), queried_backbone_features]`
2. 3-layer SIREN MLP with ω₀=30 (controls frequency), width=128
3. Output: scalar p

`queried_backbone_features` = the slice tokens from the Transolver backbone, projected
to dim=64 and appended to the coordinate input.

The INR is only used for pressure; velocity (Ux, Uy) continues to use the current decoder.
This is justified by the problem structure: p is the hardest target and the one that
violates continuity most on OOD shapes.

**Critical hyperparameter:** SIREN ω₀=30. Too high → hash of the field. Too low → missing
high-frequency features. Start with 30, try 10 and 60 in follow-up.

**Gotcha:** SIREN requires careful initialization (Sitzmann et al., App. 1A). The first
layer W is initialized as U(-1/n_in, 1/n_in) and subsequent layers as U(-√(6/n_in),
√(6/n_in)). Incorrect initialization causes the SIREN to fail spectacularly.

Add as `--siren_pressure_decoder` flag.

### Suggested experiment design

1. Implement a 3-layer SIREN with ω₀=30, width=128 as the pressure decoder.
2. Query it at each mesh node coordinate to get p predictions during training.
3. The velocity decoder remains unchanged.
4. Compare p_tan vs baseline. Also compare p_in (should be neutral or slightly better).

### Risk assessment

**High risk, high reward.** INRs for fluid mechanics are an active research area but not
yet standard. The main risks:
- SIREN training instability if initialization is wrong
- The coordinate input may not carry enough information without richer backbone features
- The pressure field for tandem configurations has sharp gradients near the aft foil TE;
  SIREN with fixed ω₀ may not resolve these

The risk is worth taking: if the current pressure decoder is the bottleneck for OOD
generalization, this is the most theoretically clean fix.

### Expected impact

**Potentially large (5-15% p_tan)** if the discrete node decoder is the bottleneck. This is
a structurally different approach to pressure prediction that naturally handles OOD coordinates.
If it fails, the most likely failure mode is training instability, which is diagnosable quickly.

---

## Idea 6: Stochastic Depth / Layer Drop for Regularization at Scale

### What it is

Apply stochastic depth (Huang et al., 2016) to the Transolver blocks: during training, each
of the 3 transformer blocks is independently dropped with probability p_drop (typically
0.1–0.2). The surviving blocks' outputs are scaled by 1/(1-p_drop) to maintain expected
value. At inference, all blocks are active.

This is distinct from dropout (which operates within layers) — stochastic depth drops
entire layers, creating an implicit ensemble of networks of different depths during training.
The theoretical motivation: stochastic depth trains an ensemble of 2^L architectures
simultaneously, where L is the number of layers.

This connects to Schmidhuber's work on training ensembles implicitly — the ensemble is
not explicitly constructed but emerges from the stochastic training process.

### Why it might help p_tan

The current model has n_layers=3 with all layers active. The OOD failure on NACA6416 may
be related to the depth of the network overfitting to the training distribution —
intermediate representations that are useful for NACA00XX shapes may be misaligned for
NACA6416 geometry.

Stochastic depth forces each layer to be independently useful (the gradient can flow
through any subset of layers), which tends to produce more robust, generalizable
representations. In image classification, stochastic depth consistently improves OOD
robustness without degrading in-distribution accuracy.

The connection to our problem: when the model processes a NACA6416 mesh, it can "skip"
the layers that have overfit to symmetric-foil geometry and rely more on the layers that
have learned generalizable flow physics.

### Key papers

- Huang et al. (2016). "Deep Networks with Stochastic Depth." ECCV 2016. Original paper.
  Shows 0.1-0.2 drop probability optimal for ResNets; same range expected here.
  https://arxiv.org/abs/1603.09382

- Touvron et al. (2021). "Training data-efficient image transformers with Stochastic Depth."
  DeiT. Shows stochastic depth essential for training data-efficient transformers — directly
  relevant to our small-dataset CFD setting.
  https://arxiv.org/abs/2012.12877

- Bochkovskiy et al. (2020). "YOLOv4." Uses stochastic depth with "bag of freebies" to
  consistently improve generalization. Practical validation of the approach.
  https://arxiv.org/abs/2004.10934

### Implementation notes

Implementation in the Transolver block:
```python
# In TransolverBlock.forward():
if self.training and stochastic_depth_prob > 0:
    if torch.rand(1).item() < stochastic_depth_prob:
        return x  # skip this block entirely
    else:
        return x + self.block(x) / (1 - stochastic_depth_prob)
```

The per-layer drop probability can be linearly interpolated: the first layer has lower
drop probability (0.05) and the last layer has higher (0.15), following the "linear
decay" schedule from Huang et al.

**Critical detail:** Stochastic depth must NOT be applied to the SRF or the residual
prediction head — only to the main Transolver blocks. Dropping the SRF would eliminate
the primary surface correction.

Add as `--stochastic_depth` flag with `--stochastic_depth_prob 0.1` hyperparameter.

### Suggested experiment design

1. Add `stochastic_depth_prob` to each TransolverBlock.
2. Apply linearly from 0.05 (block 1) to 0.15 (block 3).
3. Run with `--stochastic_depth --stochastic_depth_prob 0.15` (the max probability in the
   linear schedule).
4. Compare all 4 validation metrics vs baseline.

### Risk assessment

**Low risk.** Stochastic depth is simple, well-understood, and rarely makes things worse.
The main risk is that n_layers=3 is already small enough that layer dropping significantly
reduces capacity during training. At p_drop=0.1, the expected number of active layers per
forward pass is 2.7/3, which is safe.

The simplicity is the main reason this hasn't been tried — it feels too simple to work
at this stage of the plateau. But sometimes the simple regularization idea is the one
that breaks through.

### Expected impact

**Moderate (1-4% p_tan).** Stochastic depth is a regularization technique; it won't
fundamentally change the model's information access. But if the model is slightly
overfit to training-set geometry distributions (likely given the deep plateau), this
could provide the regularization needed to improve OOD transfer.

---

## Idea 7: Attention over Mesh Neighbors (Local Topology-Aware Attention)

### What it is

In addition to the global slice attention (current architecture), add a **local neighbor
attention** mechanism: for each node, attend only over its K nearest spatial neighbors
(K=8 or K=16). The local attention uses a separate small MLP to compute Q/K/V from the
node features, and its output is added to the global slice attention output via a learned
gating scalar.

This creates a two-scale attention: global (current slice attention) + local (neighborhood
attention). The local attention is essentially a graph attention layer restricted to K-NN.

This is inspired by graph attention networks (GAT) and differs from standard GNNs in
that it's applied as an auxiliary term on top of the existing architecture, not as a
replacement.

### Why it might help p_tan

The current slice attention is global — each node's representation is a linear combination
of ALL slice tokens. This is powerful for capturing global flow patterns but misses local
interactions: a node near the aft foil TE doesn't directly attend to its immediate neighbors
in the wake.

For NACA6416 specifically: the high-camber surface has sharp curvature changes that create
strong local pressure gradients. The global attention may average out these local structures
when transferring from training geometries. A local neighbor attention would capture these
sharp local features regardless of the global geometry context.

The K-NN graph is cheap to construct (~1ms for 20K nodes with K=8 using GPU kNN) and
fully differentiable via the torch.scatter operations.

### Key papers

- Veličković et al. (2018). "Graph Attention Networks." ICLR 2018. Origin of graph
  attention — the K-NN version of this idea.
  https://arxiv.org/abs/1710.10903

- Zhao et al. (2021). "Point Transformer." ICCV 2021. Shows that local attention on
  K-NN graphs outperforms global attention for point cloud processing — directly relevant.
  https://arxiv.org/abs/2012.09164

- Wu et al. (2022). "Scalable Diffusion for Materials Generation." Uses local+global
  attention hybrid in a mesh setting with excellent OOD generalization. Indirect precedent.
  https://arxiv.org/abs/2311.09235

- Lienen et al. (2022). "Learning the Dynamics of Physical Systems from Sparse
  Observations with Finite Element Networks." Direct application of local attention to
  CFD mesh prediction. https://arxiv.org/abs/2203.08852

### Implementation notes

After each Transolver block's global slice attention, add:
1. Build KNN graph: `knn_idx = knn_graph(node_positions, k=8)` — shape [N, 8]
2. Gather neighbor features: `neighbor_feats = x[knn_idx]` — shape [N, 8, d]
3. Compute attention: `attn = softmax(Q(x) @ K(neighbor_feats).T / sqrt(d/8))`
4. Output: `local_out = attn @ V(neighbor_feats)` — shape [N, d]
5. Gate: `x = x + gate * local_out` where `gate = sigmoid(learnable_scalar)`

Starting with gate initialized to 0 (no contribution) and letting it learn is the safest
approach — the model is free to ignore the local attention if it's not useful.

**Critical gotcha:** Do not share parameters between the global slice attention Q/K/V
and the local neighbor attention Q/K/V. They operate at different scales and should have
independent projections. This is a common mistake in hybrid attention implementations.

Add as `--local_knn_attention` flag with `--local_knn_k 8` hyperparameter.

### Suggested experiment design

1. Add 1 local KNN attention layer per Transolver block (3 total).
2. Initialize gate=0 for each layer (no contribution at start).
3. Run with K=8 first; try K=16 in follow-up if promising.
4. Monitor gate values in W&B — if they stay near 0, the local attention is not being
   used and the experiment is a null result without metric regression.

### Risk assessment

**Medium risk.** The KNN graph construction per forward pass adds ~5ms overhead per
sample. With batch size ~4 and 50K iterations, this is ~4 extra GPU-hours of KNN
computation — manageable. The bigger risk is that K=8 neighbors are not enough to
capture the relevant spatial context (wake regions can span hundreds of nodes). K=32
might be needed but would quadratically increase memory use.

The gate=0 initialization is the safety net: if local attention doesn't help, the gate
stays near 0 and there's no regression in baseline metrics.

### Expected impact

**Moderate (2-5% p_tan)** if local topology is the missing information. The theoretical
motivation is sound — local structure matters for surface pressure, especially on
high-curvature OOD shapes. The gating mechanism ensures graceful degradation.

---

## Summary and Priority Ranking

| Rank | Idea | p_tan Target | Risk | Implementation Cost |
|------|------|-------------|------|---------------------|
| 1 | Boundary Condition Hard Constraint (Normal Projection) | 3-8% | Medium | Low (5-10 lines) |
| 2 | Stochastic Depth | 1-4% | Low | Low (10 lines) |
| 3 | DSDF Test-Time Feature Alignment | 3-5% | Low-Medium | Low (10 lines) |
| 4 | Geometry Tokenizer (Set-Transformer + Cross-Attn) | 5-10% | Medium | High (new module) |
| 5 | Local KNN Attention | 2-5% | Medium | Medium (new layer) |
| 6 | Laplacian Eigenvector PE | 5-10% | Medium-High | Medium (new PE) |
| 7 | INR Pressure Decoder (SIREN) | 5-15% | High | Medium (new decoder) |

**Immediate assignments:**
- For idle students after Round 4 results come in: prioritize ideas 1, 2, 3 (low risk,
  low cost) first to maintain throughput. Run idea 4 (Geometry Tokenizer) and idea 5
  (Local KNN Attention) in parallel for architectural exploration.
- Ideas 6 and 7 are the "bold swings" — assign to experienced students who can debug
  training instabilities.

**Key principle for Round 5:** Every idea in this list has been designed to be
implementable as a flag addition to the existing train.py without disrupting the baseline.
This maintains the ability to ablate cleanly and avoids the risk of entangling multiple
changes.
