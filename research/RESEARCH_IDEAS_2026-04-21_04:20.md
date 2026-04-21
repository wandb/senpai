<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# AirfRANS Research Ideas — 2026-04-21

**Context**: Current best `val_primary/surface_mse = 0.01841` (PR #2646, lr=7e-4, T_max=10, 3L/192d, 35 epochs). External target: SpiderSolver `surface_mse = 0.0043` — a 4.3× gap. The 30-minute hard budget (~41 epochs at 3L/192d, ~1.37 ep/min) is the dominant constraint. Hyperparameter tuning (lr, T_max, architecture width/depth) has been exhausted in the local neighborhood. The ideas below operate at different abstraction levels.

All ideas are ranked by expected impact given the 30-minute budget and the known bottleneck: **pressure channel dominates surface MSE** (surface_mse_p accounts for ~99% of surface error at early epochs and remains dominant throughout training).

---

## Rank 1: Per-Channel Pressure-Upweighted MSE Loss

**What it is.** Weight the pressure (`p`) channel 10-50× more heavily in the training MSE compared to the velocity and turbulent-viscosity channels. Standard MSE treats all 4 channels equally, but `p` is the bottleneck.

**Why it should help.** The AirfRANS evaluation metric averages MSE across u_x, u_y, p, nut equally. If p contributes 99% of surface error, the gradient signal from p should dominate training. Equal-channel loss means u_x, u_y, nut gradient noise dilutes the p update. A per-channel weight directly concentrates optimization budget on the failing channel.

**Implementation.** In the training loss computation, replace `mse_loss(pred, target)` with a channel-weighted version:

```python
# weights: [u_x=1, u_y=1, p=20, nut=1], normalized so mean=1
channel_weights = torch.tensor([1.0, 1.0, 20.0, 1.0], device=pred.device) / (23.0/4.0)
loss = (channel_weights * (pred - target).pow(2)).mean()
```

Try p-weights of 10, 20, 50. Start with 20. The validation metric remains unweighted MSE — this only changes the training objective.

**Critical note.** The loss weighting should apply to the training loss only, not the validation metric. The existing metric computation must remain unmodified.

**Compute overhead.** Zero — same compute as current training, just a reweighted sum.

**Confidence.** Strong theoretical basis. Standard practice in multi-task regression when one target dominates the metric. Has not been tried in this codebase. Very high expected impact.

---

## Rank 2: asinh-Pressure Transform + Current Winner Config (T_max=10, lr=7e-4)

**What it is.** Apply an asinh transform to the pressure targets before computing training loss, and invert before computing validation metrics. This compresses the dynamic range of the pressure channel.

**Why it should help.** PR #2492 (asinh-pressure transform) showed 2× reduction in surface_mse_p at 2 epochs, going from ~0.96 to 0.4490. That experiment used an older config. The current winner config (T_max=10, lr=7e-4, Fourier PE, no EMA) has never been combined with this transform. The phase transition that drives the model from ~0.07 to ~0.018 surface_mse operates on normalized pressure — if the transform reduces the dynamic range before normalization, the transition may be sharper, earlier, or more reliable.

**Implementation.** The transform is `asinh(x / c)` where `c` is a scale factor (try `c = 1.0`). Applied to pressure targets after the existing normalization, inverted after prediction before metric computation. Student should check whether PR #2492 / #2459 already has the transform implemented and adapt from there.

**Key interaction to test.** Does the phase transition still occur with the asinh transform? Does it occur at the same epoch? Does it produce a lower post-transition plateau?

**Confidence.** High. The transform is already implemented (PRs #2459, #2492). The combination with the current winner config is a genuine gap. The mechanism is sound — dynamic range compression reduces the gradient magnitude mismatch between high-pressure and low-pressure regions.

---

## Rank 3: Stochastic Weight Averaging (SWA) Across Post-Transition Cosine Troughs

**What it is.** After the phase transition occurs (~epoch 25-35 with T_max=10), save model checkpoints at several cosine LR minima (e.g., epochs 35, 37, 39, 41) and average their weights. Run evaluation on the averaged model.

**Why it should help.** The phase transition is known to be stochastic — the same config produces surface_mse of 0.0248 vs 0.0207 (PR #2617) depending on random seed. Weight averaging across checkpoints near the loss minimum flattens the loss landscape and tends to find lower-variance, more generalizable solutions than any single checkpoint. Hochreiter & Schmidhuber (1997) — flat minima generalize better. SWA (Izmailov et al., 2018) demonstrated this on classification. The mechanism is particularly relevant here because the post-transition loss landscape may be noisy.

**Implementation.** The key change is checkpoint averaging at inference, not during training:

```python
# After training, average weights of checkpoints from epochs 35-41
checkpoints = [load(f"ckpt_epoch_{e}.pt") for e in [35, 37, 39, 41]]
avg_state = {k: sum(ck[k] for ck in checkpoints) / len(checkpoints) 
             for k in checkpoints[0]}
model.load_state_dict(avg_state)
# Run validation on averaged model
```

Alternatively, use PyTorch's `torch.optim.swa_utils.AveragedModel` during training to maintain a running SWA model from epoch 30 onward.

**Compute overhead.** Near zero — requires saving 4-5 extra checkpoints during training, then one additional eval pass.

**Confidence.** Moderate-high. SWA is well-validated in general ML. The specific connection to the phase transition stochasticity makes it plausible. No CFD surrogate application found in literature — this is genuinely novel in this setting.

---

## Rank 4: Multi-Seed Ensemble Prediction (Average Inference, Not Best-of-N)

**What it is.** Train 3-5 models with different random seeds. At test time, average their predictions rather than selecting the best.

**Why it should help.** PR #2649 (currently open) runs 5 seeds and takes the best. This is suboptimal — averaging uncorrelated predictions reduces variance by a factor of n, so MSE reduction is ~1/n in expectation. The phase-transition stochasticity means models trained on different seeds explore different loss basins, giving diverse predictions that benefit from averaging more than a single deterministic model would.

**Implementation.** Train N models independently. At inference:

```python
preds = [model_i(x) for model_i in models]
pred_ensemble = torch.stack(preds).mean(0)
```

3 seeds is the minimum useful ensemble. 5 seeds is preferred if budget allows. With 3L/192d and 30-min budget, a single training run is ~41 epochs. 5 seeds = 5 runs but this can be parallelized across GPUs.

**Expected gain.** If per-seed surface_mse is ~0.018 with stddev ~0.003 (consistent with PR #2617 variance), a 5-model ensemble should push below 0.015 conservatively.

**Confidence.** High. Basic ensemble theory. The variance across seeds is empirically confirmed. This is low-risk, high-reliability improvement.

---

## Rank 5: SpiderSolver-Style Boundary-Interior Cross-Attention Layer

**What it is.** Add an explicit cross-attention mechanism that attends from each interior mesh node to its k=16 nearest surface/boundary nodes, and from each surface node to its k=16 nearest interior nodes. This is the core architectural innovation in SpiderSolver (NeurIPS 2025, Surf MSE=0.0043).

**Why it should help.** The current Transolver architecture uses global attention over learned-slice groups. Surface nodes and interior nodes compete equally for attention slots. SpiderSolver's insight is that boundary conditions (airfoil surface pressure) impose hard constraints on the adjacent interior flow — fine-grained local coupling between surface and near-surface interior nodes is essential for accurate pressure prediction. The global slice attention misses this geometric coupling.

**Implementation.** Add one cross-attention sublayer after each standard self-attention block:

```python
class BoundaryInteriorCrossAttention(nn.Module):
    def __init__(self, dim, heads, k=16):
        # For each node, precompute k nearest neighbors on opposite side
        # (surface->interior, interior->surface)
        # Cross-attention: Q from node, KV from k neighbors
        ...
    
    def forward(self, x, surf_mask, neighbor_idx):
        # surf_mask: bool [N] indicating surface nodes
        # neighbor_idx: [N, k] precomputed nearest neighbors
        ...
```

The neighbor indices can be precomputed at data-loading time from the mesh coordinates. The additional parameters are modest (one MHA layer per Transformer block).

**Key reference.** SpiderSolver repo: https://github.com/Kai-Qi/SpiderSolver — study the tokenization and attention pattern there. The spiderweb tokenization groups boundary-adjacent nodes into "legs" radiating from the airfoil surface.

**Compute overhead.** Moderate — cross-attention over k=16 neighbors per node adds ~15-20% compute per forward pass. This slightly reduces epoch count in the 30-min budget but should be worth it.

**Confidence.** High theoretical basis (SOTA architecture does this). Implementation complexity is non-trivial but contained. This is the most direct path to matching SpiderSolver's performance.

---

## Rank 6: Signed Distance Function (SDF) as Additional Node Feature

**What it is.** Precompute the signed distance from each mesh node to the nearest airfoil surface point, and add it as an additional input feature alongside x, y, AoA, chord, etc.

**Why it should help.** The SDF encodes geometric proximity to the boundary condition in a continuous, differentiable way. Near-surface nodes (SDF close to 0) should receive different treatment than far-field nodes. The current model only has raw x, y coordinates — it must implicitly learn this proximity from data. SpiderSolver uses SDF ranges to define sub-regions for its tokenization scheme. Giving the model explicit SDF as a feature removes a difficult thing it must learn implicitly.

**Expected mechanism.** After adding SDF, the model can directly modulate attention patterns based on boundary proximity. The pressure channel, which carries boundary condition information from the airfoil surface to the flow field, should benefit most.

**Implementation.** At data loading time, for each mesh case:

```python
from scipy.spatial import cKDTree
surf_pts = mesh_coords[surf_mask]  # [N_surf, 2]
tree = cKDTree(surf_pts)
dist, _ = tree.query(mesh_coords)  # [N, 1]
# SDF is positive outside airfoil, negative inside
# For external flow, all nodes are outside — use unsigned distance
sdf_feature = dist[:, None]  # [N, 1], append to node features
```

This is a data-pipeline change — no architectural change required.

**Confidence.** Moderate-high. Geometric feature engineering is well-validated in mesh-based neural networks. The specific gain is hard to predict without running it, but the mechanism is sound and the implementation is cheap.

---

## Rank 7: Surface-Only Primary Loss with Volume Auxiliary Loss

**What it is.** Train with a loss that weights surface nodes more heavily than interior nodes, directly optimizing what the evaluation metric measures.

**Why it should help.** The current loss is computed uniformly over all mesh nodes. The primary evaluation metric (`surface_mse`) only counts surface boundary nodes. Training with equal weight on volume and surface nodes means ~90% of the gradient signal comes from nodes that are not measured. Reweighting to emphasize surface nodes directly aligns training objective with evaluation metric.

**Implementation.** Split the loss into surface and volume components:

```python
surf_loss = mse_loss(pred[surf_mask], target[surf_mask])
vol_loss = mse_loss(pred[~surf_mask], target[~surf_mask])
loss = surf_loss + alpha * vol_loss  # try alpha = 0.1, 0.5, 1.0
```

Start with alpha=0.1 (surface 10× more heavily weighted) and alpha=0.5.

**Critical interaction.** Volume prediction quality helps surface quality via the flow field's physical coupling. Setting alpha too low may hurt pressure at interior nodes near the surface, which in turn degrades surface pressure through attention. Try alpha=0.1 first and check whether volume_mse degrades badly.

**Confidence.** High theoretical basis. Simple change. Some risk of degrading volume MSE but that is a secondary metric. Expected to help surface_mse directly.

---

## Rank 8: Reducing Slice Count for More Epochs (slices=32 vs slices=96)

**What it is.** The current Transolver uses slices=96 by default. Fewer slices means faster forward passes and more epochs in the 30-minute budget. This trades representational capacity per forward pass for more training steps and more opportunities for the phase transition to occur.

**Why it might help.** PR #2617 showed the phase transition is stochastic — it sometimes occurs and sometimes doesn't within 41 epochs. With slices=32, we can run ~120+ epochs in 30 minutes (from PR #2509: slices=48 → ~1.8 min/epoch → 98 epochs; slices=32 would be faster still). More epochs = more opportunities for the transition to fire, and more epochs of post-transition refinement if it fires early.

**Why it might hurt.** The slice groupings are the core expressive mechanism of Transolver. Fewer slices means coarser geometric partitioning. The pressure channel improvements from finer groupings may outweigh the extra-epochs benefit.

**Implementation.** Change `--slices 96` to `--slices 32` in the training command. Keep all other hyperparameters at the current winner config.

**What to measure.** (a) Does the phase transition occur, and at what epoch? (b) What is the post-transition plateau? (c) Does final surface_mse improve vs current best?

**Confidence.** Moderate. The tradeoff is genuine and the outcome is uncertain. Worth testing because the cost is just one run.

---

## Rank 9: Test-Time Augmentation via AoA Geometric Symmetry

**What it is.** For each test case, also compute predictions on the mirror-image geometry (negated angle-of-attack, reflected mesh), then average the two predictions after reflecting the velocity field back.

**Why it should help.** Aerodynamic flows at AoA and -AoA are mirror images (for symmetric airfoils). The u_y velocity component flips sign; u_x, p, nut are symmetric. By averaging predictions over both AoAs, we effectively double the inference-time data, reducing stochastic variance in the prediction without any retraining.

**Implementation.**

```python
# For each test case with AoA = alpha:
pred_forward = model(x)  # normal prediction
# Mirror: negate AoA, reflect y-coordinates, reflect u_y prediction
x_mirror = x.clone()
x_mirror[:, y_coord_dim] = -x_mirror[:, y_coord_dim]
x_mirror[:, aoa_dim] = -x_mirror[:, aoa_dim]
pred_mirror_raw = model(x_mirror)
# Reflect back: u_y should negate, others unchanged
pred_mirror = pred_mirror_raw.clone()
pred_mirror[:, uy_channel] = -pred_mirror[:, uy_channel]
# Also reflect spatial coordinates of predictions
pred_tta = (pred_forward + pred_mirror) / 2
```

**Key assumption.** The airfoils in AirfRANS are NACA symmetric profiles. Verify this before implementing. If the dataset includes cambered airfoils, the u_x symmetry holds approximately but not exactly.

**Compute overhead.** Doubles inference cost only — no additional training.

**Confidence.** Moderate. Depends on the symmetry assumption holding for the AirfRANS dataset profiles. If airfoils are symmetric, this is essentially free improvement. If cambered, the benefit is smaller but still positive (approximate symmetry).

---

## Rank 10: Curriculum Learning by Reynolds Number

**What it is.** Sort training cases by Reynolds number (Re) from lowest to highest and train on them in this order, gradually increasing difficulty. Laminar low-Re flows are simpler (smaller gradients, more regular patterns) than high-Re turbulent flows.

**Why it should help.** The current training samples cases randomly. In early training, the model sees high-Re turbulent cases it cannot yet predict accurately, producing large gradients that may destabilize early learning. By starting with laminar flows and progressively introducing turbulent cases, the model first learns the shared pressure-field structure before encountering high-variance turbulent samples. This mirrors curriculum learning successes in NLP (easy → hard examples) and may accelerate or regularize the phase transition.

**Implementation.** At dataset construction, sort cases by Re. Use a warmup schedule that progressively includes harder (higher Re) cases:

```python
# Epoch 0-10: train only on cases where Re < 33rd percentile
# Epoch 10-20: train on Re < 66th percentile
# Epoch 20+: full dataset
```

This requires access to per-case metadata (Re values). Check if AirfRANS data loader exposes Re per case (it should, as it's a primary conditioning variable).

**Confidence.** Moderate. Curriculum learning shows benefits in settings with high data heterogeneity. The AirfRANS Re range spans laminar to turbulent flows, which is genuinely heterogeneous. However, the phase transition behavior may be disrupted if early training sees only easy cases — the transition dynamics are not well understood. This idea is lower-risk to test than higher-ranked ideas but also has lower expected impact.

---

## Summary Table

| Rank | Idea | Abstraction Level | Compute Overhead | Expected Impact | Confidence |
|------|------|-------------------|------------------|-----------------|------------|
| 1 | Per-channel pressure-upweighted loss | Loss formulation | Zero | Very High | High |
| 2 | asinh-pressure + T_max=10 config | Data representation + training | Zero | High | High |
| 3 | SWA across post-transition checkpoints | Optimization / inference | Near zero | High | Moderate-High |
| 4 | Multi-seed ensemble (average, not best) | Training ensemble | N× training time | High | High |
| 5 | SpiderSolver boundary-interior cross-attention | Architecture | +15-20% per step | Very High | High (but complex) |
| 6 | SDF as additional node feature | Data representation | Near zero | Moderate-High | Moderate-High |
| 7 | Surface-only primary loss | Loss formulation | Zero | Moderate-High | High |
| 8 | slices=32 for more epochs | Training throughput | Zero | Moderate | Moderate |
| 9 | TTA via AoA geometric symmetry | Inference ensemble | 2× inference | Moderate | Moderate |
| 10 | Curriculum by Reynolds number | Training curriculum | Zero | Moderate | Moderate |

**Recommended first assignments** (parallel, independent):

- Student A: Idea 1 (pressure-upweighted loss, p_weight=20)
- Student B: Idea 2 (asinh-pressure + T_max=10, lr=7e-4)
- Student C: Idea 4 (5-seed ensemble average — complement PR #2649 which takes best, not average)
- Student D: Idea 5 (SpiderSolver boundary-interior cross-attention, k=16 neighbors)
- Student E: Idea 6 (SDF feature — cheap to implement, test alongside current winner config)

Ideas 3 (SWA), 7 (surface loss), 8 (slices=32), 9 (TTA), 10 (curriculum) are secondary priorities once the above have results.
