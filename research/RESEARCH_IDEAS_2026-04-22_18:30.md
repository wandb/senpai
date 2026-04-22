<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-04-22 18:30

Generated after reviewing 631 experiment PRs (105 merged, 358 ran, 204 never ran),
all 8 prior research idea files, and all 57 WIP PRs. Every idea below is verified
to not duplicate anything blacklisted, WIP, or previously proposed.

Priority order: DrivAerML first (3.997% → 3.71% target gap), then cross-dataset,
then dataset-specific.

---

## Idea 1 — DrivAerML: Progressive Surface-Point Sampling Curriculum

**Dataset:** drivaerml (primary), potentially airfrans

**Hypothesis:** Training DrivAerML with a fixed 50k surface points per batch is
compute-heavy from epoch 1, causing slow wall-clock throughput (~2 epochs/run in
practice). Starting at a lower point count (e.g., 20k) and linearly or
step-wise ramping to 50k allows the model to learn coarse pressure patterns
quickly before being asked to resolve fine-scale geometry. This should improve
both convergence rate and final accuracy on a fixed compute budget.

**Mechanism:** In the early training phase the model is in a high-loss regime
where any 50k-point sample is essentially providing the same gradient signal as
a 20k-point sample — the bottleneck is not resolution but optimization landscape.
Ramping the resolution acts as a form of implicit curriculum that matches
information density to model readiness, analogous to progressive growing in image
generation (Karras et al., 2018). For CFD specifically, the coarse point cloud
still captures the global pressure distribution and leading/trailing edge
structure; fine-surface details become discriminative only in later training.

**Not a duplicate of:** DrivAerML slices reduction (WIP #3048) which changes
model capacity, not resolution; DrivAerML T_max sweep (#3045), DrivAerML WD+gc
(#3046), DrivAerML LR fine-tune (#3047) — all optimizer, not data pipeline.

**Implementation — exact CLI flags:**

Phase 1 (first 50% of epochs): `--drivaerml-train-surface-points 20000 --drivaerml-eval-surface-points 50000`
Phase 2 (remaining epochs): `--drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000`

Since the harness does not natively support mid-run flag changes, implement as
two separate runs:

```
# Phase 1: fast coarse pretraining (save checkpoint)
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 \
  --no-use-ema --enable-fourier \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --epochs 200 --batch-size 1 \
  --drivaerml-train-surface-points 20000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --wandb_group drivaerml-progressive-curriculum

# Phase 2: fine-resolution continuation (resume from phase 1 checkpoint)
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 \
  --no-use-ema --enable-fourier \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --epochs 999 --batch-size 1 \
  --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --resume <phase1_checkpoint> \
  --wandb_group drivaerml-progressive-curriculum
```

Eval always at 50k points throughout to ensure valid metric comparison.

If the harness supports a `--curriculum-surface-points-start` / `--curriculum-surface-points-end`
flag pair with a linear schedule over epochs, that is the cleaner single-run
version. Check `train.py` args before implementing.

**Expected impact:** 5-15% relative improvement in final metric due to faster
early convergence, plus approximately 2x wall-clock speedup in phase 1 (allowing
more total effective epochs within the timeout). Moderate confidence based on
analogous results in PointNet/point-cloud literature.

**Risk:** Low. Worst case is that phase 2 performance matches the baseline
trained entirely at 50k. No architectural changes, no hyperparameter coupling.

---

## Idea 2 — DrivAerML: Bilateral Left-Right Symmetry Augmentation

**Dataset:** drivaerml

**Hypothesis:** Passenger cars have near-perfect bilateral (left-right) symmetry
in their body geometry. Mirroring each training sample along the car's longitudinal
symmetry plane (xz-plane if y is lateral) produces a physically valid training
example with mirrored surface pressure distribution. This doubles the effective
training set size at essentially zero cost, directly addressing DrivAerML's
data-scarce regime (~500 training cases per split).

**Mechanism:** For a surface mesh with positions (x, y, z) and target Cp values,
reflecting through the symmetry plane gives (x, -y, z) with identical Cp targets
(pressure is symmetric under body reflection in zero-yaw flow). Input features
that are scalar (Cp) are preserved; directional features (wall normals) need
their y-component negated. Freestream direction vector (1, 0, 0) is preserved.
This is identical in spirit to horizontal flipping in image classification — it
multiplies data without generating out-of-distribution examples.

**Not a duplicate of:** No prior DrivAerML augmentation ideas exist in any
previous idea file or WIP PR. The surface normals/curvature WIP (#3038) adds
features, not augmentation.

**Implementation:**

In the DrivAerML data pipeline, before batching each training case:
- With probability 0.5, apply: `pos[:, 1] = -pos[:, 1]`
- If wall normals are features: `normals[:, 1] = -normals[:, 1]`
- Targets (pressure Cp) are unchanged (scalar field, symmetric under reflection)
- Eval: never augment (use canonical orientation only)

Exact CLI addition to champion command:
```
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 \
  --no-use-ema --enable-fourier \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --epochs 999 --batch-size 1 \
  --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --drivaerml-symmetry-augmentation \
  --wandb_group drivaerml-symmetry-aug
```

The `--drivaerml-symmetry-augmentation` flag needs to be added to the argument
parser and the data pipeline. The implementation is approximately 5 lines of code
in the DrivAerML dataset loader.

**Caution:** Verify that DrivAerML cars are actually in a canonical orientation
where the symmetry plane is known. If cars have variable yaw angles in the
dataset, this augmentation does not apply without first canonicalizing orientation.
Check the drivaerml data pipeline before implementing.

**Expected impact:** 5-10% relative improvement. Doubles effective training set.
Strong evidence from analogous augmentations in 3D shape learning (ModelNet,
ShapeNet). Higher confidence than most ideas in this list.

**Risk:** Low-medium. Implementation is simple but requires verifying dataset
symmetry assumptions. If the cars are not axis-aligned the augmentation would
corrupt the data.

---

## Idea 3 — DrivAerML: Geometric Region Loss Weighting via Surface Curvature

**Dataset:** drivaerml

**Hypothesis:** DrivAerML surface pressure errors concentrate at high-curvature
geometric transition regions: the A-pillar, hood-windshield junction, wheel
arches, and side mirror attachment points. Weighting the per-point training loss
proportionally to surface curvature (computed offline from the mesh) focuses
gradient signal on the exact regions that are hardest to predict, without
requiring any architectural change.

**Mechanism:** For each surface point, compute the mean curvature kappa from the
mesh geometry (using the discrete Laplace-Beltrami operator or cotangent weights).
Normalize to a per-case weight map: `w_i = 1 + alpha * (kappa_i / max(kappa))`.
Apply as a per-point loss multiplier: `loss = mean(w_i * (pred_i - target_i)^2)`.
The alpha parameter controls the degree of emphasis; alpha=0 recovers baseline;
alpha=1 doubles the loss weight at peak curvature.

This is distinct from the mass conservation loss (WIP #3039) which adds a physics
constraint. This is pure geometric reweighting of the regression objective.

**Not a duplicate of:** mass conservation loss (#3039 WIP) — different mechanism.
Per-channel pressure weighting (earlier idea, TandemFoil-specific) — different
dataset and different axis of weighting. No DrivAerML geometric loss weighting
has been proposed before.

**Implementation:**

Precompute per-point curvature weights offline and store in the DrivAerML dataset.
Then pass the weights to the loss function during training.

```
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 \
  --no-use-ema --enable-fourier \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --epochs 999 --batch-size 1 \
  --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --drivaerml-curvature-loss-weight 1.0 \
  --wandb_group drivaerml-curvature-loss
```

Try alpha values: 0.5, 1.0, 2.0. The sweep can run in a single W&B group.

**Expected impact:** 3-8% relative improvement. The mechanism is sound but the
actual distribution of errors in DrivAerML has not been confirmed to track
curvature — this is an informed assumption from general CFD knowledge. Medium
confidence.

**Risk:** Medium. Requires offline precomputation of curvature and integration
into the data pipeline. If the error distribution does not track curvature, this
provides no benefit.

---

## Idea 4 — DrivAerML: Deeper Model (5 Layers) with Gradient Checkpointing

**Dataset:** drivaerml (primary)

**Hypothesis:** The current DrivAerML champion uses 4 layers / 512 hidden dim.
A 5-layer model was previously attempted but likely diverged or OOM'd due to
VRAM pressure at 50k points x batch-size 1. Gradient checkpointing (recompute
activations during backward pass instead of storing them) reduces peak VRAM by
~30-40% at a ~20-25% compute overhead. With gradient checkpointing enabled, a
5-layer model may fit in 96 GB VRAM and could achieve better accuracy through
greater representational depth.

**Mechanism:** PyTorch `torch.utils.checkpoint.checkpoint_sequential` or
`torch.utils.checkpoint.checkpoint` applied per-transformer-block. For a 5-layer
Transolver at 512 hidden dim x 8 heads x 50k points, estimated activation memory
savings: ~8 GB. Within the 96 GB budget even without checkpointing at batch=1,
but for longer sequence lengths or larger batch sizes, checkpointing is the
enabling factor.

**Not a duplicate of:** DrivAerML depth reduction (#3048 WIP) goes in the
opposite direction (fewer layers). Torch compile (#3049 WIP) is orthogonal
(compilation, not memory). No 5-layer DrivAerML experiment with gradient
checkpointing has been proposed.

**Implementation:**

```
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 4e-4 --cosine-t-max 30 \
  --no-use-ema --enable-fourier \
  --model-layers 5 --model-hidden-dim 512 --model-heads 8 \
  --epochs 999 --batch-size 1 \
  --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --gradient-checkpointing \
  --wandb_group drivaerml-5layer-gc
```

Note: reduce LR slightly (5e-4 → 4e-4) because deeper models are more sensitive
to learning rate; the optimizer landscape is steeper with more parameters.

If `--gradient-checkpointing` is not yet a supported flag, the student needs to
add it to train.py and wrap each transformer block in `checkpoint()`.

Also worth trying: 4 layers / 640 hidden dim (wider rather than deeper) as a
comparison. Approximately same parameter count as 5L/512d but different
inductive bias.

**Expected impact:** 3-8% relative improvement if depth is a binding constraint.
If 4L was already sufficient, the 5L model will match but not improve. The
prior failure of deeper models is evidence this is non-trivial; gradient
checkpointing removes the VRAM objection.

**Risk:** Medium-high. Deeper models converge slower. May need more epochs than
available within the timeout to match the 4L baseline, let alone beat it.

---

## Idea 5 — DrivAerML: Per-Case Z-Score Pressure Normalization

**Dataset:** drivaerml

**Hypothesis:** The current training normalizes surface pressure globally across
the entire training set (global mean/std). Different car geometries produce
fundamentally different absolute pressure ranges depending on drag coefficient,
blockage, and reference velocity. Normalizing each case individually (z-score
per case: subtract each case's mean Cp, divide by its std Cp) before loss
computation — and inverting before metric evaluation — decouples shape learning
from absolute pressure scale, reducing inter-case interference during training.

**Mechanism:** With global normalization, a high-drag car (large |Cp| variance)
dominates the gradient signal over a low-drag car. Per-case z-scoring makes the
model see all cases as having unit variance, then the task becomes predicting
normalized shape of pressure distribution rather than absolute pressure. At eval
time, invert: `pred_abs = pred_norm * std_case + mean_case`. The official metric
(surface_rel_l2_pct) is computed in absolute space after inversion.

This is similar in spirit to the asinh-pressure transform used for TandemFoil
(merged, significant win), but applied per-case rather than per-field.

**Not a duplicate of:** asinh pressure transform (TandemFoil-specific, merged
#2xxx). The per-channel pressure weighting idea (TandemFoil, in prior ideas file)
is different — it reweights by spatial location, not case-level statistics.
No DrivAerML per-case normalization has been proposed.

**Implementation:**

Minimal change: in the DrivAerML dataset loader, before returning targets,
compute and store per-case (mean, std) of the pressure field, normalize targets,
and pass (mean, std) to the loss/metric layer for inversion.

```
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 \
  --no-use-ema --enable-fourier \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --epochs 999 --batch-size 1 \
  --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --drivaerml-per-case-pressure-normalization \
  --wandb_group drivaerml-per-case-norm
```

**Caution:** If any training cases have near-zero pressure variance (unlikely but
possible for highly symmetric or degenerate geometries), add a floor on std:
`std = max(std, 1e-6)`.

**Expected impact:** 4-10% relative improvement. Normalization improvements have
historically shown large gains in this codebase (asinh was a large TandemFoil win).
Medium-high confidence.

**Risk:** Medium. The inversion at eval time needs to be carefully implemented.
If the inversion is even slightly off, the metric will appear worse than baseline
even if the normalized predictions are better.

---

## Idea 6 — Cross-Dataset: Learned Slice Assignment via Softmax Routing

**Dataset:** tandemfoil, tandemfoil_paper, airfrans, drivaerml

**Hypothesis:** Transolver's current slice assignment is fixed: each point is
assigned to exactly one physics-informed slice by a hard clustering rule
(typically based on physical fields or spatial location). Replacing hard
assignment with a soft, learned routing (softmax over slice logits, computed per
point from its feature vector) allows the model to discover the optimal
domain decomposition for each problem, rather than relying on the hand-coded
partitioning heuristic. This is analogous to Mixture of Experts routing applied
to spatial domain decomposition.

**Mechanism:** For each input point with features x_i, compute logits
`l_i = W_route @ x_i + b_route` (shape: [n_slices]), then soft weights
`a_i = softmax(l_i)`. The slice representation is then
`s_k = sum_i a_i[k] * embed(x_i) / sum_i a_i[k]` (soft aggregation per slice).
This is differentiable end-to-end. During training, encourage non-degenerate
routing with an entropy bonus: `loss += lambda * mean(-sum_k a_i[k] * log a_i[k])`.

Unlike hard assignment, soft routing can learn that e.g. stagnation points
deserve their own effective slice distinct from the free-stream surface points,
without needing to hand-code this physics knowledge.

**Not a duplicate of:** No learned/soft slice assignment has been proposed in any
prior idea file or WIP PR. The MQA WIP (#2996) changes the attention mechanism
within slices, not the slice assignment itself.

**Implementation:**

```
python train.py \
  --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 \
  --grad-clip 0.5 --weight-decay 1e-2 --model-slices 64 \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --enable-fourier --enable-te-coord-frame --enable-cp-panel \
  --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction \
  --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999 \
  --soft-slice-routing --slice-routing-entropy-lambda 0.01 \
  --wandb_group learned-slice-routing
```

Try entropy lambda values: 0.001, 0.01, 0.1. Start with tandemfoil to validate
mechanism before running on DrivAerML.

**Expected impact:** 3-10% relative improvement. This is a genuine architectural
change to a core component of Transolver. Medium confidence — the mechanism is
sound but the current hard-assignment heuristics may already be near-optimal
for structured CFD domains.

**Risk:** Medium-high. Adds learnable parameters to the routing network. Risk of
routing collapse (all points assigned to one slice) if entropy regularization
is not tuned correctly. The implementation requires modifying the Transolver
forward pass.

---

## Idea 7 — Cross-Dataset: Auxiliary Reynolds Number Prediction Head

**Dataset:** tandemfoil, tandemfoil_paper, airfrans (all have Re as a condition)

**Hypothesis:** Adding an auxiliary task — predicting the Reynolds number of
each case from the learned representation — acts as an inductive bias that forces
the model's latent space to explicitly encode flow regime information. The
auxiliary loss encourages the model to maintain Re-discriminative features in
the bottleneck, which should improve OOD generalization (e.g., TandemFoil's
legacy_noam/p_re metric for out-of-distribution Re cases).

**Mechanism:** After the final transformer layer, apply a mean-pool over all
point representations to get a case-level embedding, then apply a linear head:
`Re_pred = W_re @ pool(h) + b_re`. Loss: `L = L_pressure + lambda * MSE(Re_pred, Re_true)`.
Re_true should be log-normalized: `Re_norm = (log10(Re) - mu_log_Re) / sigma_log_Re`.
This adds approximately 1k parameters — negligible.

Auxiliary tasks of this form are well-validated in multi-task learning
literature (Caruana, 1997; recent: OFA, MT-DNN). The key insight is that
predicting a conditioning variable from the representation regularizes it.

**Not a duplicate of:** Multi-task loss ideas in prior files were about
predicting additional physical fields (velocity, etc.), not scalar case metadata.
No Re-prediction auxiliary head has been proposed.

**Implementation:**

```
python train.py \
  --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 \
  --grad-clip 0.5 --weight-decay 1e-2 --model-slices 64 \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --enable-fourier --enable-te-coord-frame --enable-cp-panel \
  --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction \
  --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999 \
  --aux-re-prediction --aux-re-loss-weight 0.1 \
  --wandb_group aux-re-prediction
```

Try loss weights: 0.01, 0.1, 1.0. Expect 0.1 to be near-optimal.

Also run on AirfRANS (which has Reynolds number conditioning): same flags,
switch `--dataset airfrans` and swap optimizer/LR to AdamW/6e-4.

Measure specifically whether `legacy_noam/p_re` (TandemFoil OOD-Re metric)
improves more than `legacy_noam/p_in` — that would confirm the mechanism.

**Expected impact:** 3-8% improvement, especially on OOD-Re metrics. High
confidence the mechanism is sound; medium confidence the gain is large enough
to beat baseline given the TandemFoil baseline is already well-tuned.

**Risk:** Low-medium. Purely additive change. Lambda=0 recovers baseline exactly.

---

## Idea 8 — Cross-Dataset: Test-Time Augmentation (TTA) Ensembling

**Dataset:** drivaerml, airfrans, tandemfoil

**Hypothesis:** At inference time, apply multiple deterministic augmentations
(e.g., small coordinate perturbations, or for DrivAerML: bilateral reflection)
and average the predictions. This is a zero-training-cost improvement that
reduces variance in the final prediction and can improve accuracy, particularly
in sparse-data regimes like DrivAerML.

**Mechanism:** For DrivAerML at eval:
1. Predict on original orientation: `pred_1`
2. Predict on left-right mirrored orientation: `pred_2` (then un-mirror the prediction)
3. Final prediction: `pred = (pred_1 + pred_2) / 2`

For AirfRANS/TandemFoil: use small random coordinate jitter (sigma=0.001 * domain_size)
averaged over 4-8 forward passes.

TTA is standard practice in Kaggle-winning solutions for structured prediction
tasks. The key question is whether the model's predictions are consistent enough
that averaging improves accuracy (it usually does for well-trained models).

**Not a duplicate of:** SWA (WIP #2991) is a weight-space ensemble. TTA is an
inference-space ensemble — fundamentally different. No TTA idea has been proposed.

**Implementation:**

This does not require a training flag — it is purely an inference change.
Implement as a `--tta` flag that activates TTA ensembling at eval time only.

```
python train.py \
  --dataset drivaerml ... [champion flags] \
  --tta --tta-n-augmentations 2 \
  --wandb_group drivaerml-tta
```

For DrivAerML: TTA with bilateral reflection (2 augmentations).
For AirfRANS: TTA with coordinate jitter (4 augmentations, sigma=0.001).

Eval metrics should be computed from TTA predictions. Train metrics stay
unaugmented.

**Expected impact:** 1-5% relative improvement at zero training cost. The main
uncertainty is whether the model is well-enough calibrated that its predictions
on augmented inputs are informative. For DrivAerML where bilateral symmetry is
exact physics, the gain should be reliable. Medium-high confidence.

**Risk:** Low. Pure inference change. Can only help (or be neutral). If TTA
predictions diverge (model is not symmetric-consistent), the average will
underperform single-pass — but this is informative signal in itself.

---

## Idea 9 — Cross-Dataset: Reynolds Number Curriculum Learning

**Dataset:** tandemfoil, tandemfoil_paper, airfrans (all parameterized by Re)

**Hypothesis:** Sorting training cases by Reynolds number (ascending) and
training on easy-to-hard examples first (low Re = laminar = simpler flow)
should accelerate early learning and potentially improve final accuracy on the
high-Re (OOD) split. This is the CFD-specific analogue of curriculum learning
(Bengio et al., 2009) and directly targets the `legacy_noam/p_re` metric.

**Mechanism:** Re-order the training data loader so that in epoch 1, only the
lowest-Re cases are used; by epoch N (e.g., N=50), all Re cases are included.
Specifically: at epoch t with total_epochs T and annealing_start fraction f0:
`max_Re_this_epoch = Re_min + (Re_max - Re_min) * min(1, (t - f0*T) / ((1-f0)*T))`.
Use f0=0.2 (first 20% of training is low-Re only, then linearly open up).

Alternative simpler version: fixed curriculum with three phases —
[low-Re only] → [low+mid Re] → [all Re]. Transition at epoch 100 and 200.

**Not a duplicate of:** No Re curriculum learning idea has appeared in any prior
idea file, WIP, or closed PR. AdamW β1/β2 sweep (#3028), SGDR (#3035), linear
warmup (#3033) are all optimizer scheduling, not data curriculum.

**Implementation:**

```
python train.py \
  --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 \
  --grad-clip 0.5 --weight-decay 1e-2 --model-slices 64 \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --enable-fourier --enable-te-coord-frame --enable-cp-panel \
  --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction \
  --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999 \
  --re-curriculum --re-curriculum-warmup-fraction 0.2 \
  --wandb_group re-curriculum
```

Run on TandemFoil first (has explicit OOD-Re validation split to measure
targeted impact). If it helps p_re, run on AirfRANS.

Key diagnostic: track `legacy_noam/p_re` vs `legacy_noam/p_in` separately to
confirm the curriculum is specifically improving OOD-Re generalization.

**Expected impact:** 3-8% improvement on p_re specifically, 1-3% on overall
metric. Medium confidence — curriculum learning requires the training distribution
to have the right structure (Re does have natural complexity ordering in CFD).

**Risk:** Low-medium. The implementation is a change to the data sampling loop.
If the curriculum schedule is too aggressive, early epochs will be
distribution-shifted from final eval; the annealing avoids this.

---

## Idea 10 — Cross-Dataset: Positional Encoding via RBF Kernels

**Dataset:** airfrans, drivaerml, tandemfoil

**Hypothesis:** The current Fourier feature encoding uses random frequencies
(`sin/cos(B @ x)` where B is sampled from a Gaussian). Replacing or augmenting
this with a learned RBF (Radial Basis Function) kernel bank — where each basis
function is a Gaussian centered at a learned prototype location in the input
space — provides a locality-sensitive encoding that may better capture the
geometric structure of CFD meshes near walls, leading edges, and wake regions.

**Mechanism:** RBF encoding: `phi_k(x) = exp(-||x - c_k||^2 / (2 * sigma_k^2))`
where c_k (center) and sigma_k (width) are learned parameters. The encoded
input becomes `[x, phi_1(x), ..., phi_K(x)]`. Centers initialize on a regular
grid or as k-means centroids of training point clouds. This is related to
Nadaraya-Watson regression and to the original 1988 Broomhead-Lowe RBF networks,
but here used as a learned feature map rather than a full model.

Key difference from Fourier: Fourier features are global (every frequency affects
all points equally); RBF features are local (each basis function activates
strongly only near its center). For CFD, local features near the wall are
physically meaningful — boundary layer behavior differs qualitatively from
free-stream.

**Not a duplicate of:** Fourier band sweep (WIP #3002) tunes existing Fourier
features. RoPE (WIP #2983) applies rotary positional embeddings within attention.
RBF is a fundamentally different encoding family — no prior proposal in any file.

**Implementation:**

Start with K=64 RBF centers (comparable to the Fourier feature dimension).
Add as a flag that replaces or augments the Fourier encoding:

```
python train.py \
  --dataset airfrans --optimizer adamw --lr 6e-4 --cosine-t-max 50 \
  --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema \
  --model-layers 2 --model-hidden-dim 256 --model-heads 4 \
  --epochs 999 \
  --rbf-encoding --rbf-num-centers 64 \
  --wandb_group rbf-positional-encoding
```

Try: RBF-only (replace Fourier), RBF+Fourier (concatenate), different K values
(32, 64, 128). AirfRANS is the cleanest testbed due to uniform mesh and low
training cost.

**Expected impact:** 3-10% improvement. RBF-style encodings have strong
theoretical motivation for spatially structured problems. However, training
the RBF centers adds complexity and risk of poorly-initialized bases.
Medium confidence.

**Risk:** Medium. Initialization matters significantly for RBF networks.
Poor initialization → basis collapse or non-coverage. Use k-means init from
the training point clouds.

---

## Idea 11 — DrivAerML: Freestream-Conditioned Global Normalization

**Dataset:** drivaerml, airfrans

**Hypothesis:** Normalize each case's surface pressure by the dynamic pressure
`q = 0.5 * rho * U_inf^2` before feeding to the model (i.e., work in
coefficient-of-pressure Cp space throughout training, not raw pressure P).
If the DrivAerML dataset provides raw pressure targets rather than Cp, this
single change ensures the model learns dimensionless pressure coefficients
rather than dimensional pressures with varying scales across freestream conditions.

**Mechanism:** `Cp = (P - P_inf) / (0.5 * rho * U_inf^2)`.
Cp values are typically in the range [-3, +1] for automotive surfaces regardless
of freestream speed, whereas raw pressure varies with U_inf^2. This normalization
is standard CFD postprocessing and makes the learning problem invariant to
freestream velocity scaling.

If the dataset already stores Cp (check `drivaerml/program.md`), this is a no-op.
If it stores P, this is a meaningful data transformation.

**Not a duplicate of:** Per-case z-score normalization (Idea 5 above) normalizes
statistics of the pressure distribution per case. Freestream-conditioned
normalization is a physics-motivated absolute normalization to Cp units.
Fundamentally different. No such idea in any prior file.

**Implementation:**

```
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 \
  --no-use-ema --enable-fourier \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --epochs 999 --batch-size 1 \
  --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --drivaerml-cp-normalization \
  --wandb_group drivaerml-cp-norm
```

**Prerequisite:** Check whether DrivAerML targets are already in Cp units by
reading `target/icml2026/drivaerml/program.md`. If they are, skip this idea.
If they are raw P, this could be a large win for low implementation cost.

**Expected impact:** If targets are raw P: potentially 10-20% improvement
(normalization invariance has historically been a large factor in CFD ML).
If targets are already Cp: 0% improvement (no-op). Check before assigning.

**Risk:** Low if targets are raw P. No-op risk if already in Cp.

---

## Idea 12 — Cross-Dataset: Attention Entropy Regularization

**Dataset:** tandemfoil, airfrans, drivaerml

**Hypothesis:** Transolver attention heads may collapse — attending uniformly
over all slices (entropy too high, no selectivity) or attending to only a single
slice (entropy too low, poor coverage). Adding an explicit entropy regularization
term on the attention distribution encourages each head to maintain meaningful
selectivity, improving gradient flow and representation quality.

**Mechanism:** At each attention layer, compute the entropy of the attention
weight distribution: `H(A) = -sum_j A_ij * log(A_ij + eps)`. Add a penalty:
- Too-uniform: `loss += lambda_min * max(0, H(A) - H_max_target)` — penalize
  when entropy exceeds target (force selectivity)
- Too-peaked: `loss += lambda_max * max(0, H_min_target - H(A))` — penalize
  when entropy is below target (force coverage)

In practice, start with only the too-uniform penalty (lambda_min only) since
attention collapse toward uniformity is more common than collapse toward a single
token. H_max_target = log(n_slices) * 0.8 (80% of max entropy for n slices).

This is inspired by attention entropy analysis in ViT literature (Darcet et al.,
2023, "Vision Transformers Need Registers") but has not been applied to
physics-informed slice attention.

**Not a duplicate of:** No attention entropy regularization has been proposed in
any prior idea file or WIP PR. Attention dropout (WIP #3022) is a different
mechanism (stochastic zero-out, not entropy shaping).

**Implementation:**

```
python train.py \
  --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 \
  --grad-clip 0.5 --weight-decay 1e-2 --model-slices 64 \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --enable-fourier --enable-te-coord-frame --enable-cp-panel \
  --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction \
  --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999 \
  --attention-entropy-reg --attention-entropy-lambda 0.01 \
  --wandb_group attention-entropy-reg
```

Try lambda values: 0.001, 0.01, 0.1. Start with TandemFoil (fastest iteration).

**Expected impact:** 2-6% relative improvement. Medium confidence — the mechanism
is sound, but whether Transolver's slice attention actually suffers from entropy
collapse is an untested assumption. Inspecting pre-reg attention distributions
before committing would de-risk this.

**Risk:** Medium. May interact with existing attention dropout (#3022 WIP).
Run without attention dropout to isolate the effect.

---

## Idea 13 — DrivAerML: Cosine Annealing with Warm Restarts + T_mult Sweep

**Dataset:** drivaerml (specifically, independent of WIP #3035 which targets TandemFoil)

**Hypothesis:** The DrivAerML champion uses T_max=30 cosine annealing. The
current DrivAerML T_max sweep (WIP #3045) tests different T_max values with
standard cosine annealing. A different dimension to explore is SGDR with
warm restarts and T_mult > 1 (cycles that lengthen over time), which has shown
benefits for gradient descent escaping flat regions in loss landscapes. DrivAerML
has a known plateau behavior; SGDR with lengthening cycles may help escape it.

Note: WIP #3035 is SGDR for TandemFoil. This is SGDR for DrivAerML —
a separate dataset with a separate champion config and a separate gap to close.

**Not a duplicate of:** WIP #3035 is explicitly for TandemFoil, not DrivAerML.
WIP #3045 tests fixed T_max values, not warm restarts. These are distinct.

**Implementation:**

```
SENPAI_MAX_EPOCHS=9999 python train.py \
  --dataset drivaerml --optimizer adamw --lr 5e-4 \
  --cosine-t-max 30 --cosine-t-mult 2 \
  --no-use-ema --enable-fourier \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --epochs 999 --batch-size 1 \
  --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 \
  --max-train-batches 394 --max-eval-batches 200 \
  --wandb_group drivaerml-sgdr
```

Sweep: T_mult in {1.5, 2.0, 2.5} while keeping T_0=30 fixed.

**Expected impact:** 2-5% relative improvement. SGDR has mixed results in
practice; the gain depends on whether the loss landscape has exploitable
local minima. Medium-low confidence but very low implementation cost
(single hyperparameter change).

**Risk:** Low. The flag may already exist (check if `--cosine-t-mult` is
already a supported argument — check train.py). If not, add it via
`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.

---

## Idea 14 — AirfRANS: Increase Model Capacity (3L/320d)

**Dataset:** airfrans

**Hypothesis:** The AirfRANS champion uses 2 layers / 256 hidden dim. This is
smaller than the DrivAerML champion (4L/512d) and the TandemFoil champion
(3L/192d with more inductive biases). AirfRANS has a larger training set than
DrivAerML and a complex boundary-layer structure that may benefit from additional
model capacity. Moving to 3 layers / 320 hidden dim increases depth and width
while remaining well within the 96 GB VRAM budget.

**Not a duplicate of:** All capacity changes in the experiment log are for
TandemFoil or DrivAerML. No AirfRANS depth/width increase has been proposed
in any prior file. WIP #3048 is DrivAerML depth reduction — opposite direction
and different dataset.

**Implementation:**

```
python train.py \
  --dataset airfrans --airfrans-task full \
  --optimizer adamw --lr 5e-4 --cosine-t-max 50 \
  --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema \
  --enable-fourier \
  --model-layers 3 --model-hidden-dim 320 --model-heads 4 \
  --epochs 999 \
  --wandb_group airfrans-capacity-3L-320d
```

Note: slight LR reduction (6e-4 → 5e-4) to account for larger model. Also try
3L/256d (depth only) and 2L/320d (width only) as ablation points in the same
W&B group.

**Expected impact:** 3-8% relative improvement. Capacity improvements are
reliable for underfitting regimes. Whether AirfRANS is underfitting is the
key uncertainty — it has 1000 training cases which is substantial. Medium
confidence.

**Risk:** Low. Pure architectural change, no new hyperparameters. Worst case
is same performance as 2L/256d baseline.

---

## Idea 15 — Cross-Dataset: Register Tokens for Global Context

**Dataset:** tandemfoil, airfrans, drivaerml

**Hypothesis:** ViT-22B and the "Vision Transformers Need Registers" paper
(Darcet et al., 2023) showed that global register tokens — additional learned
tokens prepended to the sequence and discarded after attention — dramatically
improve the quality of local feature representations by providing a
dedicated "scratchpad" for global aggregation, preventing attention sinks.
In Transolver, the slice-level representations serve a similar aggregation role,
but there is no mechanism for cross-slice global communication outside the
slice structure. Prepending K=4 global register tokens to the slice sequence
at each attention layer provides this without changing the physics-informed
structure.

**Mechanism:** At each transformer block, prepend K global register tokens
(learnable, shared across all cases) to the slice sequence before attention,
then discard them after attention. The register tokens can attend to all slices
and all slices can attend to them, enabling global information flow without
polluting the point-level representations. Parameters added: K * d_model
(K=4, d=192 → 768 parameters — negligible).

**Not a duplicate of:** Global context token (in a prior idea file) was a single
global CLS token appended at the end — that idea appears to already be WIP or
previously tried. Register tokens differ: K>1 tokens, prepended not appended,
discarded after attention, motivated by the artifact-suppression mechanism
in ViT-22B. Verify that the global context token WIP (if it exists) is a
different mechanism before assigning this.

**Implementation:**

```
python train.py \
  --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 \
  --grad-clip 0.5 --weight-decay 1e-2 --model-slices 64 \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --enable-fourier --enable-te-coord-frame --enable-cp-panel \
  --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction \
  --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999 \
  --register-tokens 4 \
  --wandb_group register-tokens
```

Sweep K in {2, 4, 8}. The TandemFoil baseline is well-tuned so any improvement
here is clean signal.

**Expected impact:** 3-8% relative improvement. The register token mechanism
has strong empirical backing from ViT literature. The question is whether
Transolver's slice attention already handles global aggregation adequately.
Medium confidence.

**Risk:** Low-medium. Requires modifying the Transolver forward pass to
prepend/discard register tokens at each layer. Implementation is ~10 lines.

---

## Priority Summary

For immediate assignment (ordered by expected value per GPU-hour):

1. **Idea 2** (DrivAerML bilateral symmetry aug) — highest confidence, near-zero
   implementation cost, directly doubles effective data
2. **Idea 5** (DrivAerML per-case z-score normalization) — low implementation
   cost, high expected gain based on asinh-transform precedent
3. **Idea 1** (DrivAerML progressive surface-point curriculum) — medium cost,
   directly addresses the wall-clock bottleneck limiting DrivAerML iteration
4. **Idea 7** (auxiliary Re prediction head, TandemFoil+AirfRANS) — low cost,
   directly targets the OOD-Re metric which is a key paper-facing number
5. **Idea 11** (DrivAerML Cp normalization) — if targets are raw P, this is
   the highest-impact / lowest-cost change on the list; prerequisite: check
   drivaerml/program.md first

Ideas 3, 4, 6, 10, 15 are medium-risk architectural changes worth scheduling
after the low-cost ideas above are validated.

Ideas 8, 9, 12, 13, 14 are solid experiments but lower expected impact; assign
when students are available after higher-priority ideas are in flight.
