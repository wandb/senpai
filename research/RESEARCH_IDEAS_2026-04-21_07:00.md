<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-04-21 07:00

**Context**: This document covers ideas NOT already in the 04:20 document (which covers
per-channel pressure weighting, asinh transform, SWA, multi-seed ensemble, SpiderSolver
cross-attention, SDF features). Focus here is on: DrivAerML epoch budget hacks,
architecture mechanisms from 2025-2026 literature, optimization improvements, data
augmentation, and cross-benchmark transfer strategies. Current bests: TandemFoil
val_primary=82.65 (PR #2473); AirfRANS val_primary=0.00935 (PR #2737, gc=1.5);
DrivAerML surface_rel_l2_pct=5.027% (PR #2648, 4L/320d). External targets:
AirfRANS 0.0043 (SpiderSolver), DrivAerML 3.71% (Transolver-3).

All ideas annotated: **Impact** (high/medium/low), **Risk** (low/medium/high).
Ideas are ordered by confidence-weighted expected gain.

---

## Idea 1: DrivAerML — model_slices Reduction for More Epochs

**Slug**: `drivaerml-slices-reduction`

**What to change.** Add `--model-slices 32` (or 48, 64) to the DrivAerML training
command. No other changes. Compare throughput and validation loss versus the default
96 slices.

**Specific flags**:
```
--model-slices 32   # fastest, most epochs
--model-slices 48   # intermediate
--model-slices 64   # conservative
```

**Why it should help.** DrivAerML at default slices=96 completes only 2 epochs in the
30-minute timeout. The TandemFoil breakthrough (val_primary from 197 to 82, -58%)
came primarily from slices reduction (from 96 to 64 or 48) enabling more gradient
updates. DrivAerML is the same architecture and the same timeout — the identical
mechanism applies. At slices=32 DrivAerML should reach 6-8 epochs, tripling the
gradient updates. This is the single most obvious untested lever for DrivAerML.

**Mechanism.** model_slices controls the number of parallel attention groups in the
Transolver. Fewer slices = smaller attention matrix = faster forward/backward pass =
more epochs per wall-clock minute. The accuracy tradeoff at very low slices is model
expressiveness, but empirically on TandemFoil the throughput gain dominates.

**Impact**: High. **Risk**: Low (already validated the mechanism on TandemFoil).

**Never-ran PR**: #2487. Highest priority for DrivAerML.

---

## Idea 2: T_max=5 for TandemFoil (Extrapolating the Monotonic T_max Trend)

**Slug**: `tandemfoil-tmax5`

**What to change.**
```
--cosine-t-max 5
```
Applied to the current TandemFoil winner config (Fourier+physics+no-EMA, slices=64
or 48, lr=3e-4).

**Why it should help.** PR #2445 established a strict monotonic ranking across
T_max values: T_max=10 > T_max=20 > T_max=30 > T_max=50, all else equal. The
winning T_max=10 at 750 steps/epoch completes 75 full cosine cycles per epoch.
T_max=5 would complete 150 cycles, doubling the LR exploration frequency. The
mechanism is that more frequent LR oscillations act as implicit annealing — the
optimizer escapes local minima more often. The monotonic trend has not been tested
below T_max=10. The risk is that T_max=5 causes too-rapid LR changes that prevent
convergence, but given the shape of the trend, T_max=5 is the most likely next
improvement. Also worth trying T_max=3 if T_max=5 wins.

**Impact**: Medium-high. **Risk**: Low (small change, easy to test).

**PR to create**: New, not previously proposed.

---

## Idea 3: AirfRANS — Fourier + Physics + no-EMA Triple Compound

**Slug**: `airfrans-fourier-physics-noema`

**What to change.** Combine the three independently validated improvements on
AirfRANS:
```
--fourier-features          # from PR #2457 winner
--asinh-pressure            # physics feature
--residual-prediction       # physics feature
# disable EMA (already default after PR #2454 merged)
--lr 7e-4                   # from AirfRANS winner config
--cosine-t-max 10           # from TandemFoil transfer
```

**Why it should help.** TandemFoil PR #2473 showed that Fourier + physics + no-EMA
compound synergistically — the gain is more than additive (val went from 197 to 82,
-58%). AirfRANS has independently validated: no-EMA (#2454/2465), Fourier (#2457,
#2474), and asinh+residual (#2459). The combination has never been tested on
AirfRANS. The mechanism is that Fourier PE provides better positional information,
asinh normalizes pressure magnitude, and residual prediction removes the trivial
freestream component — these address orthogonal bottlenecks simultaneously.

**Impact**: High. **Risk**: Low (all three components individually validated).

**Never-ran PR**: #2492.

---

## Idea 4: Gradient Accumulation for Effective Larger Batch on DrivAerML

**Slug**: `drivaerml-grad-accumulation`

**What to change.**
```
--gradient-accumulation-steps 4   # effective batch = 4 × default batch
```
If not already a flag, add it via the training loop: accumulate gradients over 4
mini-batches before calling `optimizer.step()`. Combine with lr scaled by sqrt(4)=2x.

**Why it should help.** DrivAerML is a 3D dataset with large surface meshes — batch
size is already small (likely 1-2) due to memory. Small batches produce high-variance
gradient estimates. Gradient accumulation simulates a larger effective batch without
increasing memory. On large-mesh problems, this is often the only way to stabilize
training without reducing batch size further. The technique is universally used in
LLM training where memory limits effective batch size; the same principle applies to
mesh-based CFD surrogates with per-case memory overhead of tens of MB.

**Critical interaction.** LR should scale as `lr_new = lr_base * sqrt(N_accum)` or
linearly `lr_base * N_accum` — try both. With N_accum=4, try lr=1.6e-3 (linear
scaling from 8e-4) or lr=1.1e-3 (sqrt scaling).

**Impact**: Medium. **Risk**: Medium (requires code change, LR scaling not obvious).

---

## Idea 5: Transolver++ Local Adaptive Physical-State Clustering

**Slug**: `transolver-plusplus-local-adaptive`

**What to change.** Replace the global slice-based attention partitioning with a
local adaptive version that clusters nodes by their local physical state (velocity
magnitude, pressure gradient) rather than globally. Concretely, modify the slice
assignment to use k-means on `[x, y, |u|, |∇p|]` rather than on `[x, y]` alone.

**Why it should help.** Transolver++ (ICML 2025, Liu et al., PMLR 267) achieved
13% average gain over vanilla Transolver on 6 PDE benchmarks by making the slice
assignment physically adaptive. The mechanism: global slicing by position groups
nodes by spatial proximity, but flow features at similar positions can differ
dramatically (e.g., boundary layer vs freestream at the same radial distance from
the airfoil). Physical-state clustering puts nodes in the same attention group when
their flow state is similar, enabling more semantically coherent attention patterns.
The key finding in Transolver++ is that local physical state (not global position)
is the right grouping criterion for Transolver attention.

**Implementation.** At each forward pass, replace the current positional slice
assignment with a clustering step on the combined position+physical-state features.
The clustering can use differentiable soft k-means or a straight-through estimator.
Alternatively, at data loading time, precompute clusters from input features alone
(no ground-truth needed).

**Key paper**: Liu et al., "Transolver++: An Accurate Neural Solver for PDEs on
Million-Scale Meshes", ICML 2025. https://arxiv.org/abs/2502.10452

**Impact**: High. **Risk**: High (requires architectural change; may take student
effort to implement correctly).

---

## Idea 6: Cosine Annealing with Warm Restarts, Multiple T_max Values in Sequence

**Slug**: `cosine-multi-cycle-schedule`

**What to change.** Instead of a fixed T_max, use a schedule where T_max decreases
across training: start with T_max=20 for the first half of training, then switch
to T_max=5 for the second half. Implemented as two chained CosineAnnealingWarmRestarts
schedulers.

**Specific implementation**:
```python
# Phase 1: epochs 1-5, T_max=20 (exploration)
# Phase 2: epochs 6-11, T_max=5 (fine-grained exploitation)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
# after epoch 5, reset with T_0=5
```

**Why it should help.** The monotonic T_max trend shows that faster cycling is
always better at our epoch budget, but very fast cycling (T_max=3) might destabilize
early training when the model is still far from any reasonable minimum. A two-phase
schedule: coarser exploration early, finer cycling late, may get the best of both.
This is the SGDR warm restarts strategy (Loshchilov & Hutter, 2017) extended with
phase-dependent T_max. T_mult>1 in CosineAnnealingWarmRestarts achieves the same
effect automatically — try T_mult=0.5 to decrease T_max over restarts.

**Impact**: Medium. **Risk**: Low (standard scheduler, easy to implement).

---

## Idea 7: DrivAerML — Surface-Weighted Loss (Upweight Surface Nodes in Volume Loss)

**Slug**: `drivaerml-surface-weighted-loss`

**What to change.** In the DrivAerML loss computation, apply a weight factor of
10-50× to loss terms from surface nodes versus volume nodes. Current loss treats all
nodes equally; the primary metric is surface-only.

**Specific flags**:
```
--surface-loss-weight 20    # new flag, multiplies surface node loss by 20
```

**Why it should help.** DrivAerML primary metric is `surface_rel_l2_pct` — computed
exclusively on surface nodes. Volume nodes dominate the mesh count (typically 95%+
of nodes are interior). If the loss is uniform over all nodes, 95% of the gradient
signal comes from interior nodes that don't affect the primary metric. Upweighting
surface nodes by 20× concentrates the optimization budget on the quantity that
actually matters. This is directly analogous to the per-channel pressure weighting
for AirfRANS (Rank 1 in 04:20 ideas doc) which targets the same kind of metric
mismatch between training loss and evaluation metric.

**Impact**: High. **Risk**: Low (trivial code change, strong theoretical basis).

---

## Idea 8: Layer-Wise Learning Rate Decay (LLRD) for Deeper Models

**Slug**: `layer-wise-lr-decay`

**What to change.** When using 4L/256d capacity, apply layer-wise LR decay: earlier
layers receive lower LR than later layers. Classic LLRD formula:
`lr_layer_k = lr_base * decay^(num_layers - k)`, with decay=0.8-0.9.

**Specific implementation**:
```python
# 4-layer model, decay=0.85, base_lr=8e-4
lr_schedule = {
    "layer0": 8e-4 * (0.85 ** 3),  # = 4.9e-4
    "layer1": 8e-4 * (0.85 ** 2),  # = 5.8e-4
    "layer2": 8e-4 * (0.85 ** 1),  # = 6.8e-4
    "layer3": 8e-4 * (0.85 ** 0),  # = 8e-4
}
```

**Why it should help.** LLRD (Howard & Ruder, 2018, ULMFiT; Devlin et al., 2019,
BERT fine-tuning) is standard in NLP fine-tuning and increasingly used in ViT
fine-tuning. The mechanism: earlier layers capture low-level features that converge
quickly and are easily over-written; later layers need larger updates for
task-specific adaptation. At our epoch budget (2-11 epochs depending on dataset),
early-layer over-writing is a real risk. LLRD prevents catastrophic forgetting of
general geometric features in the first layers while allowing fast task adaptation
in the last layers.

**Impact**: Medium. **Risk**: Low (established technique, easy to implement with
parameter groups in PyTorch).

---

## Idea 9: Random Mesh Subsampling as Data Augmentation (DropNode)

**Slug**: `dropnode-augmentation`

**What to change.** During training, randomly drop 10-20% of non-surface mesh nodes
from each batch, forcing the model to predict the full field from a subsampled mesh.
At inference, use the full mesh.

**Specific flags**:
```
--dropnode-rate 0.15    # drop 15% of volume nodes during training
```

**Why it should help.** This is the mesh equivalent of DropPath/Dropout applied at
the node level. Three benefits:
1. Regularization: prevents overfitting to specific mesh topology.
2. Robustness: model learns to interpolate, not memorize.
3. Throughput: slightly smaller effective mesh per batch → marginal speedup.

The key insight is that CFD meshes from simulation have a fixed topology per case,
so the model risks over-fitting to mesh-specific patterns rather than learning the
underlying physics. Dropping nodes forces generalization. This is related to the
DropEdge technique (Rong et al., 2020) for graph networks and has shown 5-8%
improvements on mesh-based problems in structural mechanics (MeshGraphNet literature).

**Surface nodes must NOT be dropped** — they carry boundary condition information
that is part of the input signal, not the prediction target (on DrivAerML).

**Impact**: Medium. **Risk**: Low (non-invasive augmentation; easy to ablate).

---

## Idea 10: Geometry-Conditioned Normalization (GeoNorm)

**Slug**: `geonorm-conditioning`

**What to change.** Replace LayerNorm in the Transolver blocks with a geometry-
conditioned affine transform: the scale and shift parameters are predicted by a
small MLP from the geometry encoding (AoA + chord + geometry descriptors), rather
than being fixed learned scalars.

**Specific form**:
```python
class GeoNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma_mlp = nn.Linear(cond_dim, dim)  # predict scale
        self.beta_mlp = nn.Linear(cond_dim, dim)   # predict shift

    def forward(self, x, cond):
        x_norm = self.norm(x)
        gamma = 1 + self.gamma_mlp(cond).unsqueeze(1)
        beta = self.beta_mlp(cond).unsqueeze(1)
        return gamma * x_norm + beta
```

**Why it should help.** Each CFD case has different operating conditions (AoA, Re,
chord). Standard LayerNorm applies the same affine transform regardless of operating
condition — the model must encode the condition implicitly through the hidden state.
Geometry-conditioned normalization allows the model to explicitly shift and scale
its internal representations based on the operating condition at each layer. This is
the core mechanism behind Adaptive Layer Norm (DiT, Peebles & Xie 2023) which
drove major improvements in image generation. The same mechanism has been validated
in NovaSolver (2026) for PDE operators conditioned on equation parameters.

**Impact**: Medium. **Risk**: Medium (requires modifying norm layers throughout).

---

## Idea 11: TandemFoil — Geometric Augmentation (Random AoA Perturbation)

**Slug**: `tandemfoil-aoa-augmentation`

**What to change.** During training, randomly perturb the AoA input feature by
±0.5 degrees, and correspondingly rotate the predicted velocity field before
computing loss. This is an equivariance-enforcing augmentation.

**Specific flags**:
```
--aoa-augment-std 0.5    # Gaussian perturbation of AoA, std=0.5 deg
```

**Why it should help.** TandemFoil has 4 validation splits; the hardest is
`val_re_rand` (OOD Reynolds number) and `val_geom_camber_rc/cruise` (OOD geometry).
Augmenting by small AoA perturbations teaches the model that nearby AoA values
produce physically consistent velocity fields (related by a rotation). This is a
form of physics-consistent data augmentation. At the short epoch budget, the model
sees each case only 2-11 times — augmentation effectively multiplies the training
dataset without requiring new cases. The expected gain on OOD validation splits
is larger than on in-distribution splits.

**Critical note.** The velocity augmentation requires rotating (u_x, u_y) by the
same angle as the AoA perturbation. Pressure is a scalar and does not rotate. Only
apply to TandemFoil, not AirfRANS or DrivAerML.

**Impact**: Medium. **Risk**: Medium (requires careful velocity rotation; physics
contract must be maintained).

---

## Idea 12: GIST-Style Spectral Mesh Embeddings

**Slug**: `spectral-mesh-embedding`

**What to change.** Replace the Fourier positional encoding (sinusoidal functions
of x, y coordinates) with Laplacian eigenvector embeddings: precompute the first
k=16 eigenvectors of the mesh graph Laplacian, and use them as positional features
alongside x, y.

**Why it should help.** The GIST paper (Gauge-Invariant Spectral Transformer, 2026)
showed that spectral embeddings from the mesh Laplacian capture topology and
geometry more faithfully than coordinate-based Fourier features, achieving SOTA on
automotive CFD benchmarks. Fourier features encode absolute position well but miss
mesh connectivity — two nodes with the same (x, y) coordinates but on different
sides of a thin boundary (e.g., the airfoil surface) get identical Fourier
embeddings. Laplacian eigenvectors are topology-aware: they naturally encode
proximity through the mesh graph, not Euclidean distance. For the airfoil surface
case, this distinction matters — the boundary layer forms along the surface in
mesh-topology space, not Euclidean space.

**Key paper**: "GIST: Gauge-Invariant Spectral Transformer for Physical Simulation",
2026. Check arxiv for the most recent version.

**Compute overhead.** One-time eigendecomposition per case at data loading time
(O(N^1.5) for sparse Laplacian, feasible offline). No additional compute at training
time.

**Impact**: Medium-high. **Risk**: Medium (eigendecomposition infrastructure needed).

---

## Idea 13: Mixed Precision Training (bf16) for DrivAerML Throughput

**Slug**: `drivaerml-bf16-throughput`

**What to change.** Enable bfloat16 mixed precision training for DrivAerML:
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(batch)
```
Combined with `GradScaler` for numerical stability.

**Why it should help.** DrivAerML is epoch-limited (2 epochs in 30 min at default
settings). bf16 reduces memory bandwidth by 2× compared to fp32, which is often the
bottleneck on large-mesh cases. Expected throughput gain: 30-50% more epochs per
timeout. On A100/H100 hardware (which CoreWeave provides), bf16 uses Tensor Cores
and provides 2-4× throughput vs fp32. Combined with slices reduction (Idea 1), this
could enable DrivAerML to reach 10-15 epochs in 30 minutes.

**Why bf16 over fp16.** bf16 has the same exponent range as fp32 (no overflow risk),
only reduced mantissa. This is important for CFD surrogates where pressure/velocity
values can span several orders of magnitude.

**Impact**: Medium. **Risk**: Low (standard practice, well-understood tradeoffs).

---

## Idea 14: Multi-Task Learning — AirfRANS + TandemFoil Joint Training

**Slug**: `multitask-airfrans-tandemfoil`

**What to change.** Train a single model simultaneously on both AirfRANS and
TandemFoil batches, with dataset-specific output heads. Alternate between batches
from each dataset within each epoch. Use a shared backbone (first 2 of 3 Transolver
layers) with dataset-specific final layer.

**Specific implementation**:
```python
# Shared backbone (2 layers)
# Dataset-specific heads (1 layer each + output projection)
# Training: alternate AirfRANS batch, TandemFoil batch, repeat
# Loss: task-specific losses, weighted by 1/num_nodes (normalize by case size)
```

**Why it should help.** Both datasets are 2D airfoil flow around similar geometries
(single airfoil vs tandem foil, both NACA-family shapes, both external flow). The
shared backbone should learn transferable flow representations — boundary layer
dynamics, pressure-velocity coupling, wake formation — that benefit both tasks.
This is the standard multi-task learning hypothesis (Caruana, 1997; Ruder, 2017).
The key question is whether the representations truly transfer. Evidence: the same
Fourier PE + physics features improve both AirfRANS and TandemFoil, suggesting
shared underlying structure.

**Impact**: Medium. **Risk**: High (multi-task training is complex to stabilize;
gradient interference between tasks is a known failure mode).

---

## Idea 15: Lookhead + No-EMA Revisit with Correct Inner-LR Interaction

**Slug**: `lookahead-noema-correct-lr`

**What to change.** Prior lookahead ablation (PR #2451) tested lookahead with a
specific inner LR. With no-EMA as the new default, revisit lookahead at its own
optimal inner LR. Specifically:
```
--use-lookahead True
--lr 3e-4                # Lion inner LR
--lookahead-steps 5      # default
--lookahead-alpha 0.5    # default
# NO EMA (already default)
```
Then try lr=2e-4 and lr=1e-4 for the inner optimizer (lookahead effectively
multiplies the outer step, so inner LR should be smaller).

**Why it should help.** Lookahead (Zhang et al., 2019) maintains a slow outer
weight that interpolates toward the fast inner weight every k steps. It tends to
smooth the loss landscape and improve generalization — similar in spirit to EMA but
with a fundamentally different update rule (interpolation vs exponential average).
PR #2451 tested lookahead WITH EMA, which created a double-averaging effect that
likely hurt performance. With EMA now removed, lookahead at a correctly calibrated
inner LR may be beneficial. The optimal inner LR for lookahead is typically 0.5-0.3×
the optimal direct LR, meaning for TandemFoil (optimal lr=3e-4), lookahead inner
lr should be tested at 1e-4 to 2e-4.

**Impact**: Medium. **Risk**: Low (already tested lookahead, just need correct LR).

---

## Idea 16: Residual Feature Scaling (Learned Magnitude Initialization)

**Slug**: `residual-scale-init`

**What to change.** Initialize the final output projection layer of the Transolver
with a scale factor of 0.01 instead of the default initialization. This makes the
model's initial output near-zero, so early training is dominated by the residual
connection (freestream prediction), not random model output.

**Specific change**:
```python
# In model initialization
nn.init.normal_(output_projection.weight, std=0.01)
nn.init.zeros_(output_projection.bias)
```

**Why it should help.** At initialization, a randomly initialized network produces
predictions with O(1) magnitude noise. The freestream baseline (residual prediction)
has O(1) physical magnitude. When the model's random output competes with the
freestream at initialization, gradient updates are chaotic. By initializing the
output projection near-zero, the model starts close to the pure freestream
prediction (a physically reasonable initialization) and learns corrections
incrementally. This is the μP / small-init principle (Yang et al., 2022) applied
to PDE surrogates. The same principle is used in ResNet's zero-init BN (He et al.,
2016, which gave 0.2% CIFAR-10 improvement from initialization alone).

**Impact**: Low-medium. **Risk**: Low (one-line initialization change; easily ablated).

---

## Idea 17: AirfRANS — Reynolds Number as Explicit Sinusoidal Feature

**Slug**: `reynolds-sinusoidal-embedding`

**What to change.** Encode Reynolds number (Re) as a set of sinusoidal features
analogous to Fourier PE for spatial coordinates:
```python
re_features = [sin(Re / 1e5), cos(Re / 1e5), sin(Re / 1e6), cos(Re / 1e6),
               sin(Re / 1e4), cos(Re / 1e4)]
```
Concatenate these to the global condition vector alongside AoA and chord.

**Why it should help.** The AirfRANS OOD task `reynolds` requires generalization to
Reynolds numbers not seen in training. Current encoding likely uses raw Re (or
normalized Re) as a scalar feature — a single scalar is a poor representation for a
quantity that controls the physics of boundary layer transition nonlinearly. Fourier
encoding of Re creates multiple periodic representations at different scales, enabling
the model to represent both the coarse (laminar vs turbulent transition) and fine-
grained (boundary layer thickness) effects of Re variation. This is the same
principle as Fourier PE for spatial coordinates, but applied to the operating
condition space.

**Key reference.** Neural Fourier Operator conditioning (Li et al., 2023): parameter
encoding as Fourier features improved generalization on parametric PDEs.

**Impact**: Medium. **Risk**: Low (additive feature; easily ablated).

---

## Idea 18: DrivAerML — Cosine T_max=10 Transfer from TandemFoil

**Slug**: `drivaerml-tmax10`

**What to change.**
```
--cosine-t-max 10
--lr 8e-4
```
Combined with slices=48 or 64 (Idea 1) to enable enough epochs for T_max=10 to
be effective (need at least 5-10 epochs for multiple cosine cycles).

**Why it should help.** The T_max=10 finding from TandemFoil (PR #2445, -14.5%)
has been tested on AirfRANS (#2469) but NEVER on DrivAerML. DrivAerML currently
runs with default T_max=150, which means the LR barely completes one cycle in 2
epochs. T_max=10 at slices=48 (allowing ~6 epochs) would complete 60 cosine cycles
in 6 epochs of DrivAerML training. The transfer is not guaranteed but the mechanism
(more frequent LR restarts = more optimization landscape exploration) is dataset-
agnostic. PR #2480 tested T_max on DrivAerML but without slices reduction — the
two must be combined to see the effect.

**Impact**: Medium. **Risk**: Low (hyperparameter change only, no code modification).

---

## Idea 19: Post-Training Quantization-Aware Averaging (QAA) at Inference

**Slug**: `quantization-aware-averaging`

**What to change.** At inference time, for each case, run the model N=8 times with
different random seeds in the dropout layers (or artificially added tiny Gaussian
noise to activations), then average predictions. This is Monte Carlo inference
averaging — different from ensemble in that it uses one trained model, not N models.

**Specific flags**:
```
--mc-inference-samples 8     # number of stochastic forward passes
--mc-noise-std 1e-3          # noise level on activations
```

**Why it should help.** With the current 30-minute budget and limited epochs, the
model converges to a sharp local minimum with high posterior uncertainty. MC averaging
over small noise perturbations samples the local neighborhood of that minimum and
averages out the prediction variance. Expected gain: similar to ensemble but at 1/N
the training cost. The tradeoff is N× inference cost, which is acceptable for
validation-time evaluation (not training time). Closely related to Test-Time
Augmentation (TTA) in computer vision — proven to give consistent 1-3% gains in
Kaggle competition settings without additional training.

**Impact**: Low-medium. **Risk**: Low (inference only, no training change needed).

---

## Idea 20: AirfRANS — 6L/256d Deep-Wide Model at Reduced Slices

**Slug**: `airfrans-6l256d-slices48`

**What to change.**
```
--num-layers 6
--hidden-dim 256
--model-slices 48    # reduce from 96 to maintain throughput
--lr 5e-4
--cosine-t-max 10
```

**Why it should help.** AirfRANS at 30 min/session gets ~41 epochs with 3L/192d.
Reducing slices to 48 restores similar throughput for the 6L/256d model. The
hypothesis is that deeper networks have a representational advantage for capturing
the multi-scale structure of turbulent flow: the boundary layer (thin, high-gradient
region) and the far field (nearly uniform) require very different representations.
A 6-layer network can develop hierarchical representations where early layers capture
fine-scale boundary layer features and later layers integrate them into global pressure
distributions. Prior attempts at 4L/256d showed modest improvement on AirfRANS
(PR #2474). 6L goes deeper, which has not been tested.

**Impact**: Medium. **Risk**: Medium (more layers may slow training; need slices
compensation to maintain throughput).

---

## Summary Table

| Rank | Slug | Dataset | Impact | Risk | Status |
|------|------|---------|--------|------|--------|
| 1 | drivaerml-slices-reduction | DrivAerML | High | Low | Never-ran #2487 |
| 2 | tandemfoil-tmax5 | TandemFoil | Med-High | Low | New |
| 3 | airfrans-fourier-physics-noema | AirfRANS | High | Low | Never-ran #2492 |
| 4 | drivaerml-grad-accumulation | DrivAerML | Medium | Medium | New |
| 5 | transolver-plusplus-local-adaptive | All | High | High | New |
| 6 | cosine-multi-cycle-schedule | All | Medium | Low | New |
| 7 | drivaerml-surface-weighted-loss | DrivAerML | High | Low | New |
| 8 | layer-wise-lr-decay | All | Medium | Low | New |
| 9 | dropnode-augmentation | All | Medium | Low | New |
| 10 | geonorm-conditioning | All | Medium | Medium | New |
| 11 | tandemfoil-aoa-augmentation | TandemFoil | Medium | Medium | New |
| 12 | spectral-mesh-embedding | All | Med-High | Medium | New |
| 13 | drivaerml-bf16-throughput | DrivAerML | Medium | Low | New |
| 14 | multitask-airfrans-tandemfoil | AirfRANS+TF | Medium | High | New |
| 15 | lookahead-noema-correct-lr | TandemFoil | Medium | Low | New |
| 16 | residual-scale-init | All | Low-Med | Low | New |
| 17 | reynolds-sinusoidal-embedding | AirfRANS | Medium | Low | New |
| 18 | drivaerml-tmax10 | DrivAerML | Medium | Low | New |
| 19 | quantization-aware-averaging | All | Low-Med | Low | New |
| 20 | airfrans-6l256d-slices48 | AirfRANS | Medium | Medium | New |

**Top 3 immediate priorities** (clear mechanism, low risk, highest expected gain):
1. `drivaerml-slices-reduction` (Idea 1) — most obvious untested lever for DrivAerML
2. `airfrans-fourier-physics-noema` (Idea 3) — validated components, never combined
3. `drivaerml-surface-weighted-loss` (Idea 7) — metric mismatch fix, trivial to implement
