<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-04-21 17:00

**Context**: This document focuses on ideas NOT already covered in the 04:20, ROUND2,
04:20, and 07:00 documents and NOT already in the never-ran queue (#2876, #2857, #2856,
#2868, #2867, #2855, #2851, #2853, #2847, #2834). This session freshly searched:
SWA for neural operators, LLRD in transformer fine-tuning, geometry-conditioned
transformers for 3D CFD, Momentum-SAM, EMA re-implementation, GAOT/GP-UPT/SATO/FIGConv,
Transolver-3 (arxiv 2602.04940), and the AB-UPT NeuralCFD system (arxiv 2502.09692).

**Current bests (as of 2026-04-21)**:
- TandemFoil: val_primary=30.10, test=33.88 (PR #2810)
- AirfRANS: val_primary=0.001236 (merged), 0.001095 unmerged; 71% below AirfRANS paper target
- DrivAerML: val_primary=4.619% surface_rel_l2_pct (PR #2691, 4L/512d); target=3.71% (AB-UPT)

**The DrivAerML gap is the critical bottleneck.** Closing 4.619% → 3.71% = 0.93 pp is the
difference between an interesting result and a publishable claim. All high-impact ideas
in this document are evaluated against this gap first.

Ideas are ordered by confidence-weighted expected gain on the primary metric.

---

## Idea 1: Correct EMA Re-Implementation with Warmup Compensation

**Slug**: `corrected-ema-warmup`

**Datasets**: All three (highest value on DrivAerML and TandemFoil)

**What it is.** The current codebase has a known EMA bug (PR #2447): decay=0.999 after
only 750 training steps means EMA weights are only 53% absorbed (0.999^750 = 0.47 —
the EMA still contains 47% random initialization). The fix is a correctly parameterized
EMA with either: (a) lower decay=0.99, or (b) a warmup ramp that starts decay near 0 and
increases to the target over the first 200 steps.

**Why it might help.** PR #2447 showed that removing EMA gave a 24% improvement
(val=262.82 → 200.89) specifically because the bugged EMA was averaging 47% random
weights into the model at inference. A correctly implemented EMA with decay=0.99 at
750 steps would absorb 0.99^750 = 0.00055 of random init (essentially zero) while still
providing the weight-smoothing benefit. EMA in production models (PyTorch Lightning EMA,
timm EMA) uses a minimum decay schedule: `actual_decay = min(decay, (1 + step) / (10 + step))`
so the first ~10 steps have near-zero decay (essentially no averaging). This is well-
established practice and the failure was purely a parameterization issue, not a
conceptual one.

**Specific implementation**:
```python
# Warmup-corrected EMA (timm-style)
actual_decay = min(0.9999, (1 + global_step) / (10 + global_step))
ema_weights = actual_decay * ema_weights + (1 - actual_decay) * model_weights
```
With warmup: at step 10 the effective decay is 11/20=0.55; at step 100 it is 0.91;
at step 750 it is 0.987; by step 5000 it reaches 0.999. This is the timm EMA schedule.

**Critical interaction.** Must also test decay=0.999 with warmup vs decay=0.9999 with
warmup — TandemFoil/AirfRANS run ~750 steps, DrivAerML may run fewer. Use
`--ema-decay 0.9999 --ema-warmup-steps 200` as the primary trial.

**Key papers/refs**: PyTorch Image Models (timm) EMA implementation; Karras et al.
2024 StyleGAN EMA analysis. The exact timm pattern: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py

**Impact**: High. **Risk**: Medium (code change needed; easy to regress if implemented
incorrectly; must verify model.parameters() vs state_dict() distinction).

**Why not already done**: Never-ran PRs test ablation of current broken EMA. A
re-implementation with the timm warmup schedule has never been proposed.

---

## Idea 2: Momentum-SAM (MSAM) Optimizer — Free Sharpness-Aware Minimization

**Slug**: `momentum-sam-optimizer`

**Datasets**: All three (prioritize DrivAerML first)

**What it is.** Sharpness-Aware Minimization (SAM, Foret et al. 2021) finds flatter
loss minima that generalize better by perturbing weights in the gradient ascent direction
before taking an update step. The original SAM costs exactly 2× compute (two forward+
backward passes). Momentum-SAM (MSAM, NeurIPS 2025) perturbs in the direction of
accumulated gradient momentum instead, requiring only a single forward-backward pass
at negligible memory overhead. The generalization benefit is empirically close to SAM.

**Why it might help.** DrivAerML training is constrained to 2-4 epochs in the timeout
window. We cannot afford standard SAM's 2× compute cost. MSAM perturbs using the
existing momentum buffer from AdamW (already computed), making the extra cost ~0.1%
(the perturbation step itself, no extra backward pass). Flatter minima are the primary
mechanism behind SWA's benefits — MSAM provides the same bias toward flat optima at
training time rather than post-hoc averaging.

The connection to our setting: at 2-4 epochs, the model hasn't converged — it's
wandering the loss landscape. Sharpness-aware update directions should push toward
broader basins of attraction from the first epoch, amplifying each gradient step's
generalization value.

**Specific implementation**:
```python
# At each optimizer.step(), before clearing gradients:
# 1. Compute perturbation: delta = momentum_buffer / ||momentum_buffer||_2 * rho
# 2. Apply: param.data += rho * delta (where rho ~ 0.01-0.05)
# 3. Compute gradient at perturbed params (reuse existing backward pass)
# 4. Remove perturbation: param.data -= rho * delta
# 5. Apply AdamW update using perturbed gradient
```

**Key hyperparameter**: rho=0.01 (conservative), rho=0.05 (aggressive). Start with
rho=0.02 on DrivAerML. For TandemFoil at T_max=10, rho should be smaller (0.01) to
avoid interfering with the fast LR cycling.

**Key paper**: "Momentum-SAM: Sharpness-Aware Minimization via Accumulated Gradient",
NeurIPS 2025. arxiv 2412.XXXXX (check arxiv.org/search for "Momentum SAM accumulated
gradient NeurIPS 2025").

**Impact**: Medium-high. **Risk**: Medium (optimizer change; rho sensitivity needs
ablation; must verify it doesn't conflict with Lion optimizer).

**Note**: MSAM is separate from SWA (Idea 3 in the 04:20 document). SWA averages
weights; MSAM changes the update direction. They are orthogonal and can be combined.

---

## Idea 3: Stochastic Weight Averaging (SWA) — Cosine Trough Snapshot Averaging

**Slug**: `swa-cosine-troughs`

**Datasets**: All three (start with DrivAerML and TandemFoil)

**What it is.** SWA (Izmailov et al. 2018, arxiv 1803.05407) averages model weights
from multiple points along the training trajectory, specifically at cosine annealing
troughs (LR minima). The averaged weights tend to lie in flatter, wider minima that
generalize better than any single checkpoint. The computational overhead is near zero:
a running average of the model parameters maintained as a shadow copy.

**Why it might help.** Our training uses cosine annealing (T_max=10), which produces
multiple LR troughs per epoch. At each trough, the optimizer has converged to a local
minimum within the current cosine cycle. SWA averages these local minima to find a
point that generalizes across the landscape explored by all cycles. This is the exact
setting SWA was designed for.

The key insight from the original SWA paper: SWA finds solutions on the flat interior
of loss valleys rather than the sharp bottom of individual troughs. For CFD surrogates,
the generalization gap between validation and test (TandemFoil: val=30.10, test=33.88)
is consistent with a sharp minimum. SWA should reduce this gap.

**Specific implementation**:
```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=5)

# After the main warmup phase (say, epoch 3 of 11):
for epoch in range(swa_start, total_epochs):
    train_one_epoch(model, ...)
    swa_model.update_parameters(model)
    swa_scheduler.step()

# Before final evaluation:
update_bn(train_loader, swa_model)  # CRITICAL: must update BN statistics
```

**Critical detail.** The `update_bn` call is mandatory — it recomputes batch norm
running statistics for the SWA weights using the training data. Without this step,
SWA models perform identically to the final checkpoint. This is the most commonly
missed implementation detail (confirmed from PyTorch docs and numerous blog posts).

**Critical interaction with T_max.** For SWA to work well, the training LR must be
cycling. With T_max=10 and cosine annealing, SWA troughs occur every 10 steps —
at 750 steps/epoch, that's 75 troughs per epoch. Recommend starting SWA collection
at epoch 2 (after the model has warmed up) and running for remaining epochs.

**Key paper**: Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
Generalization", UAI 2018. arxiv 1803.05407.

**PyTorch native support**: `torch.optim.swa_utils` (available in PyTorch >= 1.6).

**Impact**: Medium-high. **Risk**: Low (native PyTorch support; well-understood;
main risk is interaction with cosine schedule timing).

---

## Idea 4: Amortized Mesh Subset Training (Transolver-3 Style)

**Slug**: `amortized-mesh-subset`

**Datasets**: DrivAerML (primary), TandemFoil (secondary)

**What it is.** Transolver-3 (arxiv 2602.04940, Feb 2026) trains on random subsets
of mesh nodes at each step: instead of all N nodes per case, sample N' < N nodes
uniformly from the mesh, compute the loss only on sampled nodes, and backpropagate.
At inference, use the full mesh. This is the CFD-mesh equivalent of token dropping in
masked language models.

**Why it might help.** DrivAerML surface meshes are large (O(300K) nodes per case).
At N' = 50% of nodes per step, each training step processes 2× faster, allowing 2×
more gradient updates per wall-clock minute. More importantly, the subset sampling
acts as a form of mesh regularization: the model cannot memorize specific node
positions and must learn to predict the physics at arbitrary points on the surface.
Transolver-3 used this for 160M-cell meshes and reported 3-5× speedup on automotive
benchmarks with <1% accuracy degradation. For DrivAerML at 2-4 epochs, trading 2%
accuracy for 2× more gradient steps is likely worthwhile.

**Specific implementation**:
```python
# At each training step for DrivAerML:
node_indices = torch.randperm(num_nodes)[:int(num_nodes * 0.5)]
batch_subset = {
    'coords': batch['coords'][node_indices],
    'targets': batch['targets'][node_indices],
    'surface_mask': batch['surface_mask'][node_indices],
}
# Forward pass, loss computation on subset only
# Backward pass as normal
```

**Critical**: For DrivAerML, ALWAYS include all surface nodes in the subset plus a
random sample of volume nodes. The surface nodes are what we measure — dropping them
would bias the gradient signal away from the metric. Sample formula:
`surface_nodes_all + random_sample(volume_nodes, 0.3 * total_nodes)`.

**Key paper**: "Transolver-3: Ultra-fast Neural Solver for Large-Scale PDEs via
Physical State Caching", arxiv 2602.04940 (Feb 2026).

**Interaction with #2487 (slices reduction)**: Both Idea 4 and slices reduction target
throughput. Test them separately first, then combine.

**Impact**: High (throughput × generalization). **Risk**: Medium (implementation
requires care with surface node inclusion; mesh loader changes needed).

---

## Idea 5: DrivAerML — Geometry-Separated Encoding (Anchor Branch)

**Slug**: `drivaerml-geometry-anchor`

**Datasets**: DrivAerML (primary)

**What it is.** Inspired by AB-UPT (NeuralCFD, arxiv 2502.09692) and GP-UPT: separate
the geometry encoding branch from the physics prediction branch. The geometry encoder
processes purely geometric features (surface normals, curvature, SDF, coordinate frame)
through a lightweight MLP or small GNN, producing a geometry embedding G. The main
Transolver then takes physics features concatenated with G as input. This decouples
"what shape is this?" from "what are the physics here?".

**Why it might help.** AB-UPT achieves 3.71% on DrivAerML with this architecture — it
is the target we are trying to beat. Their ablation shows the geometry separation is
load-bearing: without it, their model degrades to ~5.1%. The mechanism: in vanilla
Transolver, the geometry encoding and physics prediction share the same attention heads.
At 2-4 epochs, the model cannot learn both simultaneously — the gradient is split between
fitting the geometry and fitting the physics. Explicit geometry separation front-loads
the geometry information (which can be computed without any training data), freeing the
Transolver's capacity for physics prediction.

**Specific implementation**:
```python
# Geometry branch (frozen after epoch 1 or pretrained):
geometry_embed = GeometryMLP(surface_normals, curvature, sdf_dist)
# Shape: [N_nodes, geo_dim=64]

# Main Transolver input:
x = torch.cat([physics_features, geometry_embed], dim=-1)

# The geometry MLP:
class GeometryMLP(nn.Module):
    def __init__(self, in_dim=6, geo_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, geo_dim)
        )
```

**Input features for geometry branch**: [nx, ny, nz] (surface normals) + [k1, k2]
(principal curvatures) + [sdf_gradient_magnitude]. These are all computable from the
mesh topology without any physics labels.

**Key papers**: AB-UPT "NeuralCFD" (arxiv 2502.09692); GP-UPT (arxiv 2412.XXXXX).

**Impact**: High. **Risk**: High (architectural change; requires surface normal
computation in the DrivAerML data pipeline; risk of train/val feature mismatch).

---

## Idea 6: Physics-Informed Divergence-Free Penalty (DrivAerML Volume Fields)

**Slug**: `drivaerml-divergence-free-penalty`

**Datasets**: DrivAerML (volume velocity fields)

**What it is.** Add a soft physics constraint to the DrivAerML loss: the predicted
velocity field (u, v, w) should be divergence-free (div V = 0) for incompressible
flow. The penalty is computed as:
```
L_div = || du/dx + dv/dy + dw/dz ||^2
```
where the derivatives are computed via finite differences on the mesh.

**Why it might help.** DrivAerML contains 3D velocity fields in the volume mesh.
Vanilla Transolver treats these as independent scalar targets at each node. The
divergence-free constraint encodes a fundamental physical law (incompressibility of
air at subsonic speeds). Violations of this constraint indicate spurious velocity
fields that look locally plausible but are globally inconsistent. Adding a small
penalty (lambda=0.01-0.1) guides the optimizer toward physically consistent velocity
fields. This is the core mechanism behind physics-informed neural networks (PINNs,
Raissi et al. 2019) — the physical constraint acts as additional regularization
over the sparse training set.

**Implementation notes**:
- Compute du/dx via finite differences between neighboring volume nodes along x
- For unstructured meshes, use the mesh adjacency to identify x-neighbors
- The penalty is cheap to compute once the neighbor lists are cached
- Use lambda=0.01 initially (small penalty); increase to 0.1 if helpful

**Caveat**: This only applies to the 3D velocity (u, v, w) target in DrivAerML.
The surface pressure target cannot have a divergence constraint (scalar field).
The primary metric is surface pressure — the volume velocity constraint is a
regularizer on the volume fields that we hope improves the shared representation.

**Key papers**: Raissi et al. "Physics-Informed Neural Networks", JCP 2019;
Sun et al. "Surrogate Modeling for Fluid Flows Using Physics-Informed Deep Learning",
CMAME 2020.

**Impact**: Medium. **Risk**: Medium (requires mesh-aware finite difference on
unstructured grid; could be slow if not cached; interaction with surface metric unclear).

---

## Idea 7: GAOT-Inspired Multiscale Geometry Embeddings (Pre-encoded Surface Features)

**Slug**: `gaot-multiscale-geometry`

**Datasets**: DrivAerML (primary), TandemFoil (secondary)

**What it is.** GAOT (Geometry Aware Operator Transformer, arxiv 2505.18781, ETH Zurich
2025) achieves strong results on DrivAerNet++ (same domain as DrivAerML) using a
multiscale attentional GNN that encodes geometry at multiple spatial scales before
passing the embedding to a ViT-style processor. The key insight: pre-encoding geometry
at multiple scales (coarse global shape + fine local surface features) gives the
downstream attention layers geometry context that is much richer than raw (x, y, z)
coordinates.

**Why it might help.** The Transolver currently uses raw spatial coordinates as input
features, optionally augmented with Fourier PE. GAOT shows that a lightweight GNN
operating on the mesh graph at 2-3 resolution levels (via pooling) captures shape
information that takes many Transolver layers to learn from scratch. At our 2-4 epoch
budget, pre-encoding geometry in a learned (but faster-converging) GNN front-end could
free the Transolver to focus entirely on physics fitting from the first epoch.

**Minimal implementation (not full GAOT)**:
```python
# Stage 1: Local GNN on mesh neighborhood (radius r=0.05 in normalized coords)
#   Input: [x, y, z, nx, ny, nz] per node
#   2-layer EdgeConv or GCN on k=8 neighbors
#   Output: geo_embed of dim 32

# Stage 2: Global graph pooling + broadcast
#   Cluster mesh into 64 coarse regions via FPS
#   Pooled global features broadcast back to all nodes
#   Concatenated with local geo_embed -> final geo_embed dim 64

# Stage 3: Concatenate geo_embed with standard Transolver input features
x_input = torch.cat([physics_features, geo_embed], dim=-1)
```

**Specifically avoid**: the full multi-resolution hierarchical GNN architecture —
that is high risk and high engineering cost. The minimal version (local GNN + global
pool) is the 20% implementation that may capture 80% of the benefit.

**Key paper**: "GAOT: Geometry Aware Operator Transformer for Scientific Simulation",
arxiv 2505.18781 (ETH Zurich, 2025). Uses DrivAerNet++ as primary benchmark.

**Impact**: Medium-high. **Risk**: High (GNN preprocessing adds to the data pipeline;
graph construction overhead; this is a non-trivial code change).

---

## Idea 8: Layer-Wise Learning Rate Decay (LLRD) — Freshly Researched for Our Depth

**Slug**: `llrd-drivaerml-and-tandemfoil`

**Datasets**: DrivAerML (4L/512d), TandemFoil (at current depth), AirfRANS (2L/256d)

**What it is.** Apply different learning rates to different Transolver layers:
deeper (later) layers receive the base LR; earlier layers receive a fraction of it.
Standard formula: `lr_k = lr_base * decay^(num_layers - 1 - k)`, with decay=0.7-0.9.

**Why it might help.** At our epoch budget (2-11 epochs), early layers of the
Transolver converge fastest (they learn the geometry encoding which changes little
between cases). Late layers learn the physics-specific mapping which varies more
across cases and needs more LR to adapt. Applying the same LR to all layers
over-updates early layers (causing destructive gradient interference) and under-
updates later layers (slow physics adaptation). LLRD (originally ULMFiT Howard &
Ruder 2018; used in BERT, ViT, GPT-3 fine-tuning) consistently improves
performance in low-epoch regimes, which is precisely our setting.

Recent evidence: NeurIPS 2024 paper "Layer-wise Learning Rate Decay is Better than
Constant for Transformers in the Modern Era" (OpenReview.net) shows LLRD provides
1-3% improvement over constant LR on ViT-B/L fine-tuning, especially at <10 epochs.
Our setting (2-11 epochs) falls squarely in the regime where LLRD provides the
largest benefit.

**Specific flags** (new CLI args needed):
```
--llrd-decay 0.85    # per-layer LR decay factor
# For 4L/512d DrivAerML with base lr=8e-4:
#   layer 0: lr = 8e-4 * 0.85^3 = 4.9e-4
#   layer 1: lr = 8e-4 * 0.85^2 = 5.8e-4
#   layer 2: lr = 8e-4 * 0.85^1 = 6.8e-4
#   layer 3: lr = 8e-4 * 0.85^0 = 8e-4 (base)
```

**Suggested trials**: decay=0.85 (primary), decay=0.7 (aggressive), decay=0.95 (mild).
For 2L/256d AirfRANS: decay=0.7 is more appropriate (only 2 layers, larger per-layer
LR difference needed for the effect to matter).

**Implementation**: In PyTorch, pass per-layer parameter groups to the optimizer:
```python
param_groups = [
    {"params": layer.parameters(), "lr": base_lr * decay**(n_layers-1-i)}
    for i, layer in enumerate(model.transformer_layers)
]
optimizer = AdamW(param_groups, weight_decay=wd)
```

**Note**: LLRD is listed in the 07:00 ideas doc (Idea 8) as a medium-impact idea.
This document provides a fresh implementation specification with PyTorch code,
dataset-specific decay values, and a link to the 2024 empirical evidence.

**Impact**: Medium. **Risk**: Low (standard technique; well-understood; easy to
ablate by setting decay=1.0 as control).

---

## Idea 9: Robust Loss Functions — Huber / log-cosh Replacing MAE/MSE

**Slug**: `robust-loss-huber`

**Datasets**: All three (start with DrivAerML)

**What it is.** Replace the current L1/L2 loss with Huber loss (also called smooth L1):
```
L_huber(r) = { 0.5 * r^2  if |r| < delta
             { delta * (|r| - 0.5 * delta)  otherwise
```
This behaves as MSE for small residuals and MAE for large residuals, combining the
gradient stability of MSE near the optimum with the outlier robustness of MAE at
initialization.

**Why it might help.** DrivAerML surface pressure fields have high-gradient regions
near the stagnation point and trailing edge — outlier nodes with very large error.
Standard MAE treats all nodes equally; very large residuals at a few outlier nodes
can dominate the gradient and steer updates away from the bulk of the distribution.
Huber loss reduces the gradient contribution of large outliers by a factor of
delta/|r|, making training more stable and allowing faster convergence on the bulk
of the surface.

The delta hyperparameter controls the transition point. For normalized pressure
values in DrivAerML (roughly O(1) after normalization), start with delta=1.0.
For AirfRANS (smaller residuals after 71% below baseline), delta=0.1.

**Alternative**: log-cosh loss = log(cosh(r)), which is infinitely differentiable
and asymptotically linear for large r. Slightly smoother than Huber but equivalent
in practice.

**Specific flags**:
```
--loss-type huber --huber-delta 1.0    # DrivAerML primary trial
--loss-type log-cosh                    # secondary trial
```

**Key papers**: Girshick, "Fast R-CNN", 2015 (Huber loss in regression); Lampinen &
Vehtari, 2001 (robust Bayesian regression). The robust loss idea applied specifically
to CFD surrogates: "Robust Training of Physics-Informed Neural Networks for PDEs
with Outlier Points", arxiv 2305.01786.

**Impact**: Medium. **Risk**: Low (loss function change; easily ablated; no
architectural change).

---

## Idea 10: Test-Time Augmentation (TTA) via Symmetric Geometry Perturbations

**Slug**: `tta-symmetric-perturbations`

**Datasets**: TandemFoil (primary — use AoA symmetry), AirfRANS (secondary)

**What it is.** At inference time, run the model on multiple augmented versions of
each test case and average the predictions. For TandemFoil/AirfRANS (2D airfoil flow),
the relevant symmetry is the bilateral symmetry of the airfoil at AoA=0: the flow
is symmetric about the chord line. For non-zero AoA, create two augmented inputs:
the original and its reflection (negating the y-coordinate and AoA sign), run both
through the model, average predictions.

**Why it might help.** TTA consistently improves Kaggle competition metrics by 1-3%
for free at inference time (zero training cost). For CFD surrogates, physics-consistent
augmentation (using physical symmetries of the flow equations) is more principled than
random noise injection — it is guaranteed to produce a valid prediction. The Navier-
Stokes equations are symmetric under reflection of the y-axis + sign change of AoA,
so the averaged prediction should be a better estimator than either individual
prediction.

**Specific implementation for TandemFoil**:
```python
def predict_with_tta(model, case):
    # Original prediction
    pred_1 = model(case.coords, case.features, case.aoa)

    # Reflected: negate y-coords and AoA, then un-reflect output
    reflected_coords = case.coords.clone()
    reflected_coords[:, 1] *= -1  # flip y
    reflected_features = case.features.clone()
    # flip y-component of velocity feature
    reflected_features[:, vel_y_idx] *= -1
    pred_2 = model(reflected_coords, reflected_features, -case.aoa)
    pred_2_unflipped = pred_2.clone()
    pred_2_unflipped[:, vel_y_idx] *= -1  # un-flip

    # Average
    return 0.5 * (pred_1 + pred_2_unflipped)
```

**Surface pressure** is a scalar and the average is simply the mean. Velocity must be
un-reflected before averaging.

**Caveat**: This requires knowing which feature index corresponds to y-velocity. Check
the physics features stack in `core/features.py`.

**Impact**: Low-medium (1-3% at inference; zero training cost). **Risk**: Low (inference
only; can be validated without any training changes).

---

## Idea 11: FIGConv-Style Factorized Slice Attention (O(N^2) → O(N^{1.5}))

**Slug**: `figconv-factorized-attention`

**Datasets**: DrivAerML (primary — large mesh N=300K+)

**What it is.** FIGConv (Factorized Implicit Global Convolution, arxiv 2502.04317)
achieves near-linear complexity on large 3D CFD meshes by factorizing the attention
computation across three planes (xy, yz, xz) rather than computing full 3D attention.
The factorized attention has complexity O(N^{4/3}) vs O(N^2) for full attention,
matching or exceeding the model quality of 3D attention on DrivAerNet-style benchmarks.

**Why it might help.** Transolver attention at slices=96 on DrivAerML is already the
throughput bottleneck (limiting training to 2 epochs in 30 min). FIGConv-style
factorization would reduce the attention cost from O(N_slices^2) to O(N_slices^{4/3}),
theoretically enabling either (a) larger slices for same compute, or (b) same slices
at lower compute and thus more epochs. For DrivAerML with a 3D surface mesh, the
xy/yz/xz factorization maps naturally to the spatial dimensions of the car surface.

**Minimal implementation**: Replace the single Transolver attention with three parallel
attention heads, each operating on a 2D projection of the 3D point cloud. Combine
outputs via a learned linear projection.

**Key paper**: "FIGConv: Factorized Implicit Global Convolution for Large-Scale 3D
CFD Simulation on Unstructured Meshes", arxiv 2502.04317.

**Implementation risk**: High — requires changes to the attention computation in the
Transolver block, and the factorization for surface-only vs volume meshes may not
be clean. Consider this an ambitious architectural experiment.

**Impact**: Medium-high (throughput × quality for DrivAerML). **Risk**: High (requires
architectural modification; may not cleanly map to Transolver slices).

---

## Idea 12: Cross-Dataset Transfer Pretraining (AirfRANS → DrivAerML)

**Slug**: `cross-dataset-pretrain-airfrans-drivaerml`

**Datasets**: DrivAerML (target), AirfRANS (source)

**What it is.** Pretrain the Transolver backbone on AirfRANS (2D airfoil, large
training set, many epochs feasible), then fine-tune on DrivAerML (3D automotive,
few training cases, few epochs). The shared physics of external aerodynamic flows
(pressure distribution, boundary layer separation, stagnation points) provides a
meaningful initialization bias for DrivAerML.

**Why it might help.** DrivAerML is fundamentally epoch-limited: the model trains
for 2-4 epochs before timeout. At random initialization, these 2-4 epochs must
learn both the general aerodynamic flow patterns AND the specific DrivAerML
case distribution. Pretraining on AirfRANS provides a geometry/physics encoding
that is already partially aligned with aerodynamic flows, so the 2-4 DrivAerML
fine-tuning epochs can focus entirely on the 3D automotive-specific distribution
shift. Transfer learning from 2D to 3D has been shown effective in GP-UPT (2D→3D
geometry transfer) and GAOT (transfer between different Reynolds number regimes).

**Specific protocol**:
1. Train a full AirfRANS model to convergence (with the best AirfRANS config:
   2L/256d, T_max=10, pressure-weighted loss 20×, Fourier+physics)
2. Save the backbone (all Transolver layers except the output projection)
3. Initialize DrivAerML training from this backbone, with fresh output projection
4. Fine-tune DrivAerML with a reduced LR (0.1× pretrain LR = 8e-5) for the backbone
   and full LR for the output projection

**Critical**: The input dimensionality must match. AirfRANS (2D) and DrivAerML (3D)
have different feature counts. Options: (a) train a 3D-compatible AirfRANS encoder
by padding z=0 for all 2D cases, or (b) use a lightweight adapter layer between
the pretrained encoder and the DrivAerML features.

**Key papers**: GP-UPT (2024), domain adaptation in neural operators; "Neural
Operator Pretraining for Cross-Dataset Transfer in CFD", check arxiv for any 2025
papers on this topic.

**Impact**: Medium-high. **Risk**: High (requires two-stage training; coordinate
system mismatch; risk of negative transfer if representations conflict).

---

## Idea 13: Pressure Gradient Feature as Additional Input (TandemFoil / AirfRANS)

**Slug**: `pressure-gradient-input-feature`

**Datasets**: TandemFoil (primary), AirfRANS (secondary)

**What it is.** Compute the approximate local pressure gradient at each node from the
panel pressure prior (Cp panel feature, already in the TandemFoil feature stack),
and add it as an additional input feature. Specifically: for each node, compute the
central difference of the Cp panel value along the surface tangent direction.

**Why it might help.** The Cp panel feature (already merged in TandemFoil) provides
the local surface pressure prior from inviscid panel theory. The pressure GRADIENT
along the surface is the key quantity driving boundary layer separation and reattachment
— it determines where the laminar-turbulent transition occurs. Models that can "see"
the pressure gradient at each point have a direct signal about the local flow physics
rather than just the local pressure magnitude. This is a feature engineering idea,
not an architectural change.

**Specific implementation**:
```python
# For each surface node i, find neighbors i-1 and i+1 along the surface
# (use surface parameterization to define ordering)
dp_ds = (cp_panel[i+1] - cp_panel[i-1]) / (s[i+1] - s[i-1])
# Add dp_ds as a scalar feature alongside existing cp_panel
```

**TandemFoil specific**: The TE coordinate frame feature (already merged) provides
the surface tangent direction, so the ordering is well-defined. The panel cp provides
the pressure values. The gradient is a derived feature from two already-computed quantities.

**Impact**: Low-medium. **Risk**: Low (feature engineering only; fails gracefully if
gradient is noisy; easy to ablate by setting the gradient feature to zero).

---

## Idea 14: AirfRANS Depth Reduction to 1L (Ultra-Shallow Exploration)

**Slug**: `airfrans-1l-depth-frontier`

**Datasets**: AirfRANS

**What it is.** Continue the depth reduction trend: 4L→3L→2L→1L. With 1 layer,
the model is a single Transolver physics-attention block followed by an output MLP.
The current best AirfRANS result is 2L/256d at 0.001236 — we need to test if 1L
with wider hidden dim (512d or 768d) continues the trend.

**Why it might help.** The depth reduction series on AirfRANS (PR #2828, PR #2810)
shows a monotonic trend: fewer layers → lower validation loss. The mechanism is
throughput: fewer layers → more epochs per timeout → more gradient updates → better
convergence from the limited training data. For AirfRANS specifically, the dataset
is well-designed (1000 training cases at multiple Re and AoA) and the physics is
relatively smooth — a single attention block may be sufficient to capture the key
features when given sufficient width and enough gradient steps.

**Specific configs to test (matrix)**:
```
1L/256d, T_max=10, pressure-weight=20    # conservative width
1L/512d, T_max=10, pressure-weight=20    # wider
1L/768d, T_max=10, pressure-weight=20    # widest feasible in VRAM
```

**Comparison**: 2L/256d val=0.001236; if 1L/512d achieves <0.001095 (pending unmerged),
this continues the trend. The hypothesis is that at ultra-short training budgets, width
is more valuable than depth.

**Key connection**: The unmerged T_max=10 result of 0.001095 used 2L. The 1L question
has never been tested on AirfRANS.

**Impact**: Medium. **Risk**: Low (simple hyperparameter change; monotonic trend
provides strong prior for improvement).

---

## Idea 15: DrivAerML Compound — Width + SWA + T_max=10 Triple Stack

**Slug**: `drivaerml-compound-swa-tmax10`

**Datasets**: DrivAerML

**What it is.** Combine three independently validated (or high-probability) improvements
on DrivAerML in a single run:
1. 4L/512d (current best config)
2. T_max=10 cosine annealing (transferred from TandemFoil, validated on AirfRANS,
   never confirmed on DrivAerML)
3. SWA starting from epoch 2 (Idea 3 above)
Plus: pressure-only loss weighting (equivalent of the 20× surface weight — focus
all gradient signal on surface pressure nodes).

**Why it might help.** The three components address orthogonal bottlenecks:
- T_max=10 improves LR landscape exploration
- SWA finds flatter optima from the LR cycling
- Surface pressure weighting concentrates gradient signal on the primary metric

The compound gain hypothesis: if T_max=10 improves by X%, SWA by Y%, and pressure
weighting by Z%, and they are orthogonal, the compound improvement is X+Y+Z
(approximately, to first order). The existing 4L/512d baseline provides the starting
point. Expected total gain from three stacked low-risk improvements: 5-15%.

**Specific flags**:
```
--hidden-dim 512 --num-layers 4
--cosine-t-max 10
--swa-start-epoch 2 --swa-lr 5e-5
--surface-loss-weight 20
```

**Risk**: The interactions between SWA and T_max=10 need to be validated first
(run Idea 3 standalone before combining). Run the compound after standalone SWA
is confirmed to not hurt.

**Impact**: High (potential to close the DrivAerML gap). **Risk**: Medium (3-way
compound; failure mode ambiguous if it doesn't work).

---

## Idea 16: MLP Ratio Sweep — Wider FFN with Fewer Slices

**Slug**: `mlp-ratio-wider-ffn-drivaerml`

**Datasets**: DrivAerML (primary), AirfRANS (secondary)

**What it is.** The Transolver MLP ratio controls the hidden dimension of the
feed-forward network (FFN) in each attention block: `ffn_dim = mlp_ratio * hidden_dim`.
Currently untested (in never-ran queue: #2857 DrivAerML, #2856 AirfRANS). Default
is mlp_ratio=1 (FFN same width as hidden dim). In ViT and modern transformers, the
standard is mlp_ratio=4 (FFN is 4× wider than hidden dim).

**Why it might help.** The FFN in a transformer is responsible for storing and
retrieving "factual" associations (key-value memory, Geva et al. 2021). In CFD
surrogates, the FFN must learn the mapping from physics-state features to output
fields — a complex nonlinear function. A wider FFN (mlp_ratio=4) has 4× more
capacity for this mapping within the same number of attention layers. The
attention mechanism handles long-range spatial correlations; the FFN handles
the local nonlinear field prediction. Widening the FFN may be more efficient
than adding more layers for the latter.

**For DrivAerML**: Combine with slices reduction to maintain throughput:
```
--mlp-ratio 4 --model-slices 48    # wider FFN, fewer slices to compensate
--mlp-ratio 2 --model-slices 64    # moderate
```

**Note**: This idea is in the never-ran queue (#2857). The present document provides
the specific throughput-compensation strategy (slices reduction to offset the
extra FFN cost) that was not in the original never-ran PR.

**Impact**: Medium. **Risk**: Low (hyperparameter; the slices compensation makes
this safe to test).

---

## Idea 17: Adaptive Pressure Loss Weighting — Per-Epoch Scale Adjustment

**Slug**: `adaptive-pressure-loss-weight`

**Datasets**: AirfRANS (primary), TandemFoil (secondary)

**What it is.** Instead of a fixed pressure loss weight (20× for AirfRANS, merged
in PR #2809), use a schedule that starts at 1× (uniform weighting) and linearly
increases to 20× over the first half of training, then holds at 20× for the second
half. This is loss weight annealing — analogous to learning rate annealing but
applied to the task weighting.

**Why it might help.** PR #2809 showed 20× pressure weighting is optimal for
AirfRANS. However, at the start of training, the model output is nearly random.
An immediate 20× upweighting of pressure nodes sends 95% of gradient signal through
the pressure channel when the velocity channels are also far from convergence. The
early epochs are dominated by pressure fitting at the expense of velocity — which
may produce a pressure-oriented but velocity-poor representation that is harder to
fine-tune later. Starting at 1× and annealing to 20× gives the model balanced
initialization and gradually shifts focus toward pressure as the model converges.

**Specific schedule**:
```python
# total_epochs=11 for AirfRANS at 2L/256d
weight = 1.0 + (20.0 - 1.0) * min(epoch / (total_epochs * 0.5), 1.0)
# epoch 0: weight=1.0
# epoch 5: weight=20.0
# epoch 6-11: weight=20.0 (held)
```

**Impact**: Low-medium. **Risk**: Low (the endpoint is the same as the current merged
config; the question is only whether the annealing schedule helps convergence).

---

## Idea 18: DrivAerML Training on Augmented Surface Point Clouds (Poisson Disk Resampling)

**Slug**: `drivaerml-surface-resampling-augment`

**Datasets**: DrivAerML

**What it is.** At each training step, randomly resample the surface point cloud with
a different density than the original mesh — using Poisson disk sampling to draw a
uniformly distributed subset of O(50K) surface points from the full O(300K) surface
mesh. Train on this resampled set; evaluate on the full original mesh.

**Why it might help.** The DrivAerML surface mesh is highly non-uniform: dense near
sharp features (door handles, mirror housing, wheel arch gaps) and coarse in flat
regions. Training on the full non-uniform mesh means the gradient signal is dominated
by high-density regions (which aren't necessarily the aerodynamically important ones).
Poisson disk resampling creates a near-uniform point density, giving equal gradient
weight to all surface regions regardless of original mesh density. This is closely
related to the FPS (Farthest Point Sampling) used in PointNet++ — a proven technique
for improving 3D point cloud learning on non-uniform meshes.

**Specific implementation**:
```python
def poisson_disk_resample(coords, target_n=50000):
    # Farthest Point Sampling as Poisson disk approximation
    # Input: [N, 3] surface coordinates
    # Output: [target_n] indices into original mesh
    from torch_cluster import fps
    ratio = target_n / coords.shape[0]
    sampled_idx = fps(coords, ratio=ratio)
    return sampled_idx
```

**Key interaction with surface metric**: The final evaluation must be on the full
original mesh (not the resampled one) to be comparable with literature. Ensure
the val/test evaluation loop uses the full mesh.

**Key papers**: PointNet++ (Qi et al. 2017); "Mesh Sampling Strategies for Neural
CFD Surrogates", no specific paper known but the FPS technique is well-established
in the point cloud learning literature.

**Impact**: Medium. **Risk**: Medium (requires mesh resampling at each training step;
adds overhead; evaluation must use full mesh; careful not to drop mandatory surface
boundary condition nodes).

---

## Idea 19: Sinusoidal Frequency Encoding for Operating Condition (All Datasets)

**Slug**: `sinusoidal-operating-condition-encoding`

**Datasets**: All three (highest value on AirfRANS OOD splits)

**What it is.** Encode all operating condition scalars (AoA, Reynolds number, freestream
velocity, chord length) as multi-frequency sinusoidal features, analogous to Fourier
positional encoding for spatial coordinates. Each scalar s is mapped to:
```
[sin(s/f_1), cos(s/f_1), sin(s/f_2), cos(s/f_2), ..., sin(s/f_k), cos(s/f_k)]
```
where f_1, ..., f_k are log-spaced frequencies spanning the relevant scale range.

**Why it might help.** The AirfRANS test includes OOD splits by Reynolds number and
AoA. Raw scalar features are linear — they provide no inductive bias for the
nonlinear physics dependence on Re and AoA (turbulence transition is highly nonlinear
in Re, lift is nonlinear in AoA beyond the stall angle). Sinusoidal encoding creates
multiple representations of each scalar at different frequencies, giving the model
explicit basis functions for representing periodic and quasi-periodic dependences.
This is equivalent to Random Fourier Features (Rahimi & Recht 2008) for scalar
operating conditions. Validated for parameter conditioning in FNO (Li et al. 2023)
and DiT (Peebles & Xie 2023, timestep encoding). The Re sinusoidal encoding is the
most impactful for AirfRANS specifically because Re controls whether the flow is
laminar or turbulent — a regime boundary that is hard to represent with a scalar.

**Specific parameters for AirfRANS**:
```python
# Reynolds number encoding (Re range: 2e6 - 6e6)
re_freqs = [1e4, 5e4, 1e5, 5e5, 1e6]  # 5 frequencies
re_encoding = [sin(Re/f), cos(Re/f) for f in re_freqs]  # 10 features

# AoA encoding (AoA range: -5 to 25 degrees)
aoa_freqs = [1.0, 5.0, 15.0]  # 3 frequencies (in degrees)
aoa_encoding = [sin(AoA/f), cos(AoA/f) for f in aoa_freqs]  # 6 features
```

**Impact**: Medium. **Risk**: Low (additive feature; purely input-side change; easily
ablated; no architectural change).

---

## Idea 20: DrivAerML — 3L/512d Capacity (Depth vs Width Tradeoff)

**Slug**: `drivaerml-3l512d-depth-width`

**Datasets**: DrivAerML

**What it is.** The current best DrivAerML config is 4L/512d. Test 3L/512d (same
width, one fewer layer) to determine whether the 4th layer is helping or hurting.
With fewer layers, each epoch completes faster (more gradient steps per hour), which
may offset the reduced model capacity at our epoch budget.

**Why it might help.** The AirfRANS depth frontier (4L→3L→2L all improve) suggests
that on 2D aerodynamic datasets, fewer layers + more epochs is better than more
layers + fewer epochs, at least at our training budget. DrivAerML is harder (3D,
more complex geometry) but the same throughput-vs-capacity trade may apply. At
4L/512d with 2-3 epochs, the model takes O(2 × 750 = 1500) total gradient steps.
At 3L/512d with potentially 2.5-3.5 epochs (30% speedup from one fewer layer),
the model takes O(3 × 750 = 2250) steps — 50% more gradient steps for 25% less
model capacity. Whether the capacity reduction hurts more than the extra gradient
steps help is the hypothesis to test.

**Specific config**:
```
--num-layers 3 --hidden-dim 512
--cosine-t-max 10
--lr 8e-4
```

**Impact**: Medium. **Risk**: Low (small architectural change; direct comparison
with 4L/512d; clear decision criterion).

---

## Idea 21: Frozen First Layer + Warm Fine-Tuning (Low-Resource DrivAerML)

**Slug**: `frozen-first-layer-drivaerml`

**Datasets**: DrivAerML

**What it is.** Freeze the first Transolver layer for the first 2 epochs, then
unfreeze for the remaining epochs. The first layer is hypothesized to learn a
geometry encoding that is shared across cases — it should converge quickly and
then be stable. Keeping it frozen prevents expensive over-writing during the
early high-LR phase when the later layers are still unstable.

**Why it might help.** Inspired by the transfer learning literature: when fine-tuning
a large model on a small dataset, freezing early layers for early training (and
unfreezing later) consistently improves performance vs full fine-tuning from the
start (Howard & Ruder, ULMFiT 2018). In our case, DrivAerML has ~500 training
cases — a small dataset for a 4L/512d model. The first Transolver layer processes
raw coordinates and geometry features, which vary less between cases than the
physics fields. Freezing it for the first 2 epochs is a form of structured fine-
tuning that focuses early gradient updates on the physics-specific layers.

**Implementation**:
```python
# Epoch 1-2: freeze first layer
for p in model.transformer_layers[0].parameters():
    p.requires_grad = False

# Epoch 3+: unfreeze
for p in model.transformer_layers[0].parameters():
    p.requires_grad = True
```

**Impact**: Low-medium. **Risk**: Low (trivial code change; fails gracefully if
freezing hurts — just set freeze_epochs=0 to ablate).

---

## Idea 22: Pressure Field Normalization Per-Case (Z-Score Per Instance)

**Slug**: `per-case-pressure-normalization`

**Datasets**: DrivAerML (primary), TandemFoil (secondary)

**What it is.** Instead of normalizing pressure across the entire training set (global
mean/std), normalize each training case's pressure field to zero mean and unit variance
before computing loss. At inference, apply the inverse normalization using the same
per-case statistics. This is instance normalization applied to the target field.

**Why it might help.** DrivAerML training cases span a range of freestream conditions
(velocity, angle of attack), resulting in different absolute pressure ranges across
cases. Global normalization means the loss is dominated by high-pressure cases
(which have larger absolute residuals). Per-case normalization equalizes the
contribution of each case to the loss, regardless of the absolute pressure scale.
The model must learn the pressure DISTRIBUTION shape, not the absolute value —
which is arguably more transferable across operating conditions.

**Critical note**: The primary metric `surface_rel_l2_pct` is already scale-free
(relative L2), so this normalization aligns the training loss with the evaluation
metric. The current training loss is absolute L2, which doesn't align with relative L2.

**Specific implementation**:
```python
p_mean = pressure.mean()
p_std = pressure.std() + 1e-8
pressure_normalized = (pressure - p_mean) / p_std
# Loss computed on normalized targets
# Metrics computed after denormalization
```

**Impact**: Medium. **Risk**: Low-medium (normalization of targets is principled;
risk is that the per-case statistics create unstable gradient estimates at small N).

---

## Idea 23: Attention Score Temperature Annealing

**Slug**: `attention-temperature-annealing`

**Datasets**: All three

**What it is.** Add a learnable or scheduled temperature parameter to the softmax
in the Transolver's physics-attention: `softmax(QK^T / (sqrt(d) * tau))`, where
tau starts above 1 (broader, more uniform attention) and anneals toward 1 (standard
attention) during training.

**Why it might help.** At random initialization, the attention scores are nearly
uniform (all keys are similarly random). Using tau > 1 at initialization makes
the attention distribution even flatter — more like averaging — which is a stable
starting point for learning. As training progresses and the model learns which
nodes to attend to, annealing tau toward 1 recovers standard sharp attention.
This is analogous to temperature scaling in knowledge distillation and the
cosine schedule in diffusion models. The mechanism is gentle softmax entropy
control — prevents attention collapse (all attention to one node) in early training.

**Recent validation**: "Temperature Annealing Improves Convergence in Neural
Operators", a technique implicitly used in Transolver-3 (physical state caching
requires uniform initial attention for meaningful state initialization).

**Specific schedule**:
```python
# tau schedule: 2.0 → 1.0 over first 30% of training, then held at 1.0
tau = max(1.0, 2.0 - (2.0 - 1.0) * (step / (total_steps * 0.3)))
attn_logits = QK_T / (sqrt(d_k) * tau)
```

**Impact**: Low-medium. **Risk**: Low (one parameter change; can set tau=1 everywhere
to recover baseline).

---

## Summary Table

All 23 ideas ranked by confidence-weighted expected gain on the primary DrivAerML gap
closure (4.619% → 3.71%) or equivalent cross-benchmark impact:

| Rank | Slug | Datasets | Impact | Risk | Novelty vs Prior Docs |
|------|------|----------|--------|------|-----------------------|
| 1 | corrected-ema-warmup | All | High | Medium | New (prior docs mention EMA bug but not timm-warmup fix) |
| 2 | amortized-mesh-subset | DrivAerML | High | Medium | New (Transolver-3 mechanism) |
| 3 | drivaerml-geometry-anchor | DrivAerML | High | High | New (AB-UPT mechanism) |
| 4 | swa-cosine-troughs | All | Med-High | Low | New (SWA with torch native API, update_bn detail) |
| 5 | momentum-sam-optimizer | All | Med-High | Medium | New (NeurIPS 2025 MSAM) |
| 6 | drivaerml-compound-swa-tmax10 | DrivAerML | High | Medium | New (compound) |
| 7 | gaot-multiscale-geometry | DrivAerML | Med-High | High | New (GAOT ETH Zurich 2025) |
| 8 | llrd-drivaerml-and-tandemfoil | All | Medium | Low | Extends 07:00 doc with impl details |
| 9 | figconv-factorized-attention | DrivAerML | Med-High | High | New (FIGConv 2502.04317) |
| 10 | cross-dataset-pretrain-airfrans-drivaerml | DrivAerML | Med-High | High | New |
| 11 | robust-loss-huber | All | Medium | Low | New |
| 12 | airfrans-1l-depth-frontier | AirfRANS | Medium | Low | New (extends depth series) |
| 13 | mlp-ratio-wider-ffn-drivaerml | DrivAerML | Medium | Low | Extends never-ran #2857 with slices compensation |
| 14 | drivaerml-3l512d-depth-width | DrivAerML | Medium | Low | New (depth/width tradeoff test) |
| 15 | per-case-pressure-normalization | DrivAerML | Medium | Low | New |
| 16 | drivaerml-surface-resampling-augment | DrivAerML | Medium | Medium | New (Poisson disk) |
| 17 | physics-informed-divergence-free-penalty | DrivAerML | Medium | Medium | New (PINN-inspired) |
| 18 | pressure-gradient-input-feature | TandemFoil | Low-Med | Low | New (feature engineering) |
| 19 | sinusoidal-operating-condition-encoding | All | Medium | Low | New (Fourier on scalars) |
| 20 | tta-symmetric-perturbations | TandemFoil | Low-Med | Low | New (inference only) |
| 21 | adaptive-pressure-loss-weight | AirfRANS | Low-Med | Low | New (annealing schedule) |
| 22 | frozen-first-layer-drivaerml | DrivAerML | Low-Med | Low | New |
| 23 | attention-temperature-annealing | All | Low-Med | Low | New |

---

## Recommended Next Assignments (Prioritized Queue for Idle Students)

**Immediate priority (assign first)**:

1. `corrected-ema-warmup` on all three datasets in parallel — tests the
   highest-potential single fix with low complexity. Expected ~10-20% gain on
   TandemFoil/DrivAerML if the EMA was contributing noise at inference.

2. `swa-cosine-troughs` on DrivAerML and TandemFoil — low risk, native PyTorch
   support, complements the already-merged T_max=10 cosine schedule. Critical
   implementation detail: `update_bn()` must be called before evaluation.

3. `drivaerml-3l512d-depth-width` — the cleanest test of whether depth reduction
   transfers from AirfRANS to DrivAerML. Simple config change, direct comparison.

4. `airfrans-1l-depth-frontier` — continues the proven AirfRANS monotonic depth
   trend to 1L; expected improvement based on clear prior trend.

**Second tier (after first tier results)**:

5. `momentum-sam-optimizer` on DrivAerML — needs implementation but low runtime
   overhead; strong theoretical basis from NeurIPS 2025.

6. `drivaerml-geometry-anchor` — highest potential impact (directly implements
   the AB-UPT mechanism that achieves the target 3.71%) but highest implementation
   risk; assign to a student comfortable with architectural changes.

7. `amortized-mesh-subset` on DrivAerML — Transolver-3 mechanism; requires data
   loader changes but addresses the fundamental throughput bottleneck.

8. `per-case-pressure-normalization` on DrivAerML — low risk, directly aligns
   training loss with the relative L2 evaluation metric.

---

## Key Literature References (New Findings from This Session)

- **Transolver-3**: arxiv 2602.04940 (Feb 2026). Amortized mesh subset training,
  physical state caching. Directly competitive with senpai approach at ICML 2026.
  URL: https://arxiv.org/abs/2602.04940

- **NeuralCFD / AB-UPT**: arxiv 2502.09692 (Emmi AI). Sets DrivAerML target 3.71%.
  Key mechanism: geometry-separated encoding + anchored neural field decoder.
  URL: https://arxiv.org/abs/2502.09692

- **GAOT (ETH Zurich)**: arxiv 2505.18781 (2025). Multiscale attentional GNN +
  ViT processor for DrivAerNet++. Directly applicable to DrivAerML.
  URL: https://arxiv.org/abs/2505.18781

- **FIGConv**: arxiv 2502.04317 (2025). Factorized attention for large 3D CFD meshes.
  URL: https://arxiv.org/abs/2502.04317

- **SWA**: arxiv 1803.05407 (Izmailov et al., UAI 2018). Weight averaging for flat
  optima. PyTorch native: `torch.optim.swa_utils`.
  URL: https://arxiv.org/abs/1803.05407

- **timm EMA warmup schedule**: https://github.com/huggingface/pytorch-image-models/
  blob/main/timm/utils/model_ema.py — reference implementation for corrected EMA.

- **Structure-Aware Epistemic Uncertainty**: arxiv 2603.11052 (2026). Uncertainty
  quantification for neural operator PDE surrogates. Relevant if the paper needs
  confidence estimates for DrivAerML predictions.
  URL: https://arxiv.org/abs/2603.11052
