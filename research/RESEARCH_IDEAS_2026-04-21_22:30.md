<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-04-21 22:30

Generated after reviewing 507 experiment PRs (68 merged, 285 ran, 168 never ran)
and a targeted literature search on ML for CFD surrogates, 2024–2026.

## Context and Search Space Summary

### What Has Been Exhaustively Saturated

The following search spaces have been thoroughly explored and should not be
revisited without a compelling new rationale:

- **Learning rate:** 1e-4 to 1e-3, all standard values
- **T_max (cosine annealing period):** 5 to 1000 epochs
- **Architecture depth/width:** 2L–5L, 192d–768d; 4L/640d+ crashes on DrivAerML
- **Gradient clipping:** 0.5–2.0
- **Weight decay:** 1e-4–5e-2; weight_decay=0 tried on DrivAerML (#2630)
- **Surface point count:** 25k–200k for DrivAerML (#2566 tried 100k)
- **Optimizer family:** AdamW, Lion, MSAM (3.7–22.7x worse), SGD with momentum
- **Physics features on DrivAerML:** TE coord frame, Cp panel, asinh pressure,
  residual prediction — all tried on DrivAerML (#2522), showed no gain
- **Attention variants:** Multi-query attention (#2560), surface cross-attention
  (#2559), Kutta condition constraint (#2562)
- **Regularization:** Dropout (already tried on multiple benchmarks), no-EMA
  versus EMA tuning

### What Is Currently In-Flight (Do Not Duplicate)

- SWA (stochastic weight averaging)
- Gradient accumulation
- Huber loss
- LLRD (layer-wise learning rate decay)
- Gradient noise injection
- OneCycleLR
- Cosine warm restarts
- LR warmup + cosine
- torch.compile
- Dropout=0.1
- Pressure-weighted loss

### Key Findings From Literature

- **AB-UPT (TMLR 2025):** Standard self-attention outperforms linear Transolver
  on automotive CFD; their DrivAerML result is 3.71% surface rel-L2, which is
  our target
- **MoE-POT (NeurIPS 2025):** Mixture-of-experts operator transformer yields
  40% error reduction on PDE benchmarks; MoE FFN layers are a drop-in
- **GeoTransolver (arxiv 2512.20399):** GALE attention — cross-attention to
  explicit geometry points plus a global context token — beats Transolver on
  CFD benchmarks
- **SDF + derivative constraints (arxiv 2503.17289):** Signed distance field
  as geometry embedding + continuity loss boosts aerodynamic predictions
- **Prodigy optimizer (ICML 2024):** Parameter-free learning rate adaptation,
  matches or beats hand-tuned Adam/AdamW without any LR search
- **SMART (arxiv 2601.18707):** Mesh-free point-cloud transformer using
  relative L2 as training loss, demonstrating strong DrivAerML results
- **SwiGLU (Noam Shazeer, 2020; adopted by LLaMA-2, Mistral, PaLM-2):**
  Gated linear units universally improve transformer FFN training dynamics with
  zero architectural overhead

---

## Hypotheses Ranked by Expected Value (Impact × Probability)

### Rank 1 — Relative L2 Training Loss

**One-line summary:** Replace MAE/MSE training loss with the relative L2 norm
`||y_pred - y_true||_2 / (||y_true||_2 + eps)` to directly align training
with the DrivAerML evaluation metric.

**Scientific rationale:** DrivAerML's primary metric is `surface_rel_l2_pct`,
but the model currently trains on absolute MAE or MSE. This is a known
objective mismatch — training on MAE penalizes absolute error uniformly,
which over-weights low-magnitude regions (wake, far-field) and under-weights
the high-pressure stagnation zones and suction peaks that drive the relative
metric. A model trained with relative L2 will naturally allocate capacity
toward regions with large signal-to-noise ratio in the normalized target,
which is exactly what the benchmark measures. This is the most direct fix for
the metric-training alignment gap and requires no architectural change. SMART
(2601.18707) explicitly uses this loss for point-cloud CFD and reports strong
results on DrivAerML-class benchmarks.

**Code changes needed:**
- In `train.py` or `core/`, add a new loss option e.g. `--loss rel_l2` or
  `--loss_type rel_l2`
- Loss implementation:
  ```python
  def relative_l2_loss(pred, target, eps=1e-8):
      return (pred - target).norm(dim=-1) / (target.norm(dim=-1) + eps)
  ```
  Average over batch. Can also compute per-field (pressure, velocity
  separately) and sum, or use a single global norm across all output
  channels.
- DrivAerML-only initially; AirfRANS can keep MSE as it's already well below
  target.

**Specific hyperparameters:**
- eps=1e-8 (standard; ablate 1e-6 if training is unstable)
- Compare three variants: (a) global relative L2, (b) per-point relative L2
  (normalize each node independently), (c) mixed: 0.5 * MSE + 0.5 * rel_L2
- Keep all other hyperparameters at DrivAerML defaults (AdamW, lr=5e-4,
  T_max=20, gc=1.0, wd=1e-2, no EMA)

**GPU allocation:**
- DrivAerML: 4 GPUs (primary target, run variants a/b/c in parallel)
- TandemFoil: 2 GPUs (test if relative L2 also helps TF surface pressure MAE)
- AirfRANS: 0 GPUs (already solved; skip)
- DrivAerML ablation (per-field vs. global): remaining 2 GPUs

**Risk:** LOW. Pure loss change, no architecture modification. Worst case it
matches baseline. Most likely direction of improvement given metric alignment.

---

### Rank 2 — SwiGLU FFN Replacement

**One-line summary:** Replace the standard ReLU/GELU feed-forward network in
every Transolver layer with a SwiGLU gated linear unit.

**Scientific rationale:** SwiGLU (Shazeer 2020) is now the default FFN in
LLaMA-2, Mistral, PaLM-2, and most frontier LLMs — it consistently
outperforms GELU with no increase in parameter count if the hidden dimension
is scaled by 2/3 to compensate for the extra gate projection. The mechanism
is a multiplicative gating: `FFN(x) = (xW_1) * sigmoid(xW_3) * W_2` which
produces smoother gradient flow and is empirically more data-efficient. In
scientific ML, SwiGLU has been adopted in protein structure prediction (ESMFold
derivative work) and weather forecasting transformers. The Transolver FFN is
currently a bottleneck that SwiGLU can improve without touching the physics-
encoding slice attention mechanism. This is likely beneficial across all four
benchmarks.

**Code changes needed:**
- In `core/model/` (or wherever the Transolver FFN is defined), add a
  `SwiGLUBlock` class:
  ```python
  class SwiGLUFFN(nn.Module):
      def __init__(self, dim, hidden_dim=None):
          super().__init__()
          hidden_dim = hidden_dim or int(dim * 4 * 2/3)
          self.w1 = nn.Linear(dim, hidden_dim, bias=False)
          self.w2 = nn.Linear(hidden_dim, dim, bias=False)
          self.w3 = nn.Linear(dim, hidden_dim, bias=False)
      def forward(self, x):
          return self.w2(F.silu(self.w1(x)) * self.w3(x))
  ```
- Add `--ffn_type swiglu` flag to `train.py`; default remains `gelu` for
  backward compatibility
- Note: with 2/3 hidden scaling, parameter count stays close to original;
  alternatively use same hidden_dim as GELU FFN and accept slight parameter
  increase (~0.3M for 3L/384d)

**Specific hyperparameters:**
- hidden_dim = 4 * dim * 2/3 (standard SwiGLU recipe)
- No other changes needed; use standard recipe for each benchmark
- If instability is seen in early training, add 1e-5 weight decay on gate
  projections (w1, w3) separately

**GPU allocation:**
- DrivAerML: 3 GPUs
- TandemFoil: 3 GPUs
- AirfRANS: 1 GPU (low priority — already solved)
- TandemFoil Paper: 1 GPU

**Risk:** LOW. Drop-in replacement with strong empirical backing across many
domains. Worst case: negligible change. The mechanism is well-understood.

---

### Rank 3 — Surface Normals and Principal Curvature as Input Features

**One-line summary:** Augment DrivAerML point features with precomputed surface
normal vectors and two principal curvature values at each surface node.

**Scientific rationale:** The largest prediction errors on DrivAerML surface
pressure are consistently found at high-curvature geometric regions — the
A-pillar, wheel arch leading edges, and roof-windshield junction. These are
exactly the regions where pressure gradients are largest and where the model
has the least geometric context from raw XYZ coordinates alone. Surface
normals provide the local surface orientation (essential for pressure recovery:
`Cp ~ f(n · v_inf)` at first order), while principal curvatures encode the
rate of geometric change — a feature the Transolver's point attention has no
other way to infer. This is not the same as the TF-specific TE coordinate
frame (#2522): curvature is a local differential geometry quantity computable
from the mesh at preprocessing time, not a flow-physics heuristic. Literature
in shape analysis and mesh processing consistently shows that normal + curvature
features reduce surface reconstruction error on high-curvature regions.

**Code changes needed:**
- Precompute normals and principal curvatures from the DrivAerML mesh during
  dataset loading. Libraries: `open3d` (fast) or `trimesh` (already likely
  available). For a triangle mesh, per-vertex normals are trivial; principal
  curvatures require fitting a local quadric (see `trimesh.curvature`).
- Add the 5 new features per surface node: `[nx, ny, nz, k1, k2]` (or 4 if
  using mean and Gaussian curvature: `[nx, ny, nz, H, K]`)
- In `train.py` or the DrivAerML data pipeline, add `--drivaerml_surface_curvature`
  flag that activates the augmented feature set
- Optionally normalize curvature values (they span large ranges): apply
  `arcsinh(k)` scaling analogous to the TF asinh pressure transform

**Specific hyperparameters:**
- Curvature normalization: arcsinh scaling with scale=1.0 (ablate 0.1, 10.0)
- Normal vectors: already unit-normalized, no scaling needed
- Keep all other DrivAerML defaults unchanged
- Run with 3L/384d architecture (the current best stable DrivAerML config)

**GPU allocation:**
- DrivAerML: 4 GPUs (main target)
- TandemFoil: 2 GPUs (test curvature on airfoil surfaces — trailing edge
  curvature is physically meaningful)
- AirfRANS: 2 GPUs (2D surface curvature = 1D scalar; easy to add)

**Risk:** MEDIUM. Preprocessing code needs to be written and validated. If
curvature estimates are noisy (common with non-uniform triangle meshes), the
features could add noise rather than signal. Validate by checking curvature
values at known high-curvature regions before training.

---

### Rank 4 — Prodigy / D-Adaptation Optimizer

**One-line summary:** Replace AdamW with Prodigy — a parameter-free optimizer
that adapts its own learning rate, eliminating the LR search that has been
exhausted for DrivAerML.

**Scientific rationale:** The DrivAerML LR sweep has been run extensively
(1e-4 to 1e-3) and shows a frustratingly flat landscape — no single LR value
has broken through. Prodigy (Mishchenko and Defazio, ICML 2024) learns the
effective step size online using a D-Adaptation rule, converging to near-
optimal LR without any grid search. The key advantage here is that Prodigy
does not assume a fixed LR but instead maintains a per-parameter effective
learning rate that adapts to the local loss landscape curvature. On the
DrivAerML objective — which may have extremely ill-conditioned geometry due
to the 3D surface topology — this adaptive scaling could find a better descent
direction than any fixed LR schedule. Prodigy has been validated across vision
transformers, LLMs, and diffusion models at ICML 2024; it consistently matches
or beats hand-tuned Adam within the same epoch budget.

**Code changes needed:**
- `pip install prodigyopt` (lightweight, no non-standard dependencies)
- Add `--optimizer prodigy` branch in the optimizer factory in `train.py`
- Prodigy usage:
  ```python
  from prodigyopt import Prodigy
  optimizer = Prodigy(
      model.parameters(),
      lr=1.0,  # Prodigy ignores this; set to 1.0 by convention
      weight_decay=1e-2,
      safeguard_warmup=True,  # critical: prevents early divergence
      use_bias_correction=True,
  )
  ```
- Keep EMA and gradient clipping at DrivAerML defaults; they are orthogonal

**Specific hyperparameters:**
- `lr=1.0` (conventional placeholder)
- `weight_decay=1e-2` (keep matching AdamW baseline)
- `safeguard_warmup=True` (required for stable early training)
- `use_bias_correction=True`
- `d_coef=1.0` (default; ablate 0.1 if Prodigy diverges)
- Run for the same epoch budget as baseline

**GPU allocation:**
- DrivAerML: 3 GPUs (primary; this is a pure optimizer swap)
- TandemFoil: 3 GPUs (also worth testing given TF's LR sensitivity)
- AirfRANS: 1 GPU
- TandemFoil Paper: 1 GPU

**Risk:** LOW-MEDIUM. Prodigy is well-validated. The main risk is that
`safeguard_warmup` may interact poorly with the cosine schedule — monitor
the effective LR reported by Prodigy and check it is not too large in the
first 10 epochs.

---

### Rank 5 — Global Context Token (GeoTransolver-style)

**One-line summary:** Inject a single learned global-geometry summary token
into the Transolver's slice attention mechanism, giving every slice cross-
attention access to a compact global descriptor.

**Scientific rationale:** The Transolver processes points in slices — groups
of physically similar points attend to each other. This is effective for local
physics but has a fundamental limitation: there is no mechanism for global
information flow across slices without deep layer stacking. In automotive CFD,
global flow features (vehicle frontal area, blockage ratio, overall drag
classification) modulate local pressure everywhere. GeoTransolver (arxiv
2512.20399) addresses this with GALE attention: a learnable global-context
token is updated via cross-attention to geometry points, then injected into
each slice's attention as an additional key-value pair. The result is that
every point in every slice can attend to the global geometry summary, breaking
the locality constraint. This is architecturally motivated by the known
limitation of pure slice-local attention for 3D automotive bodies where far-
field and near-field pressures are correlated. The AB-UPT paper's superior
performance on DrivAerML (they report 3.71%) may partly be attributed to their
standard global self-attention rather than the slice-local Transolver design.

**Code changes needed:**
- Add a `GlobalContextToken` module to the Transolver encoder
- At each attention layer, update the global token via cross-attention to
  the current slice embeddings, then concatenate the global token as an
  additional key-value pair in slice self-attention
- Minimal version: a single trainable `nn.Parameter` of shape `[1, 1, dim]`
  that is updated via a lightweight cross-attention head before each slice
  attention layer
- Add `--global_context_token` flag to `train.py`

**Specific hyperparameters:**
- Global token dimension: same as model dim (dim=384 for 3L/384d)
- Cross-attention heads for global update: 4 (or half of main attention heads)
- Learnable token initialization: zeros or small random normal (N(0, 0.02))
- No change to LR, optimizer, or regularization

**GPU allocation:**
- DrivAerML: 4 GPUs (primary motivation)
- TandemFoil: 3 GPUs (tandem foil interaction is also a global effect)
- AirfRANS: 1 GPU

**Risk:** MEDIUM. Requires non-trivial attention module modification. The key
implementation risk is that the global token adds to VRAM at every layer —
verify no OOM on DrivAerML before scaling. The mechanism is well-motivated
and directly addresses a known architectural gap.

---

### Rank 6 — Mass Conservation Auxiliary Loss (DrivAerML Volume)

**One-line summary:** Add a soft continuity-equation penalty to the DrivAerML
volume prediction — penalize the divergence of the predicted velocity field
at training time.

**Scientific rationale:** Incompressible flow satisfies div(u) = 0 everywhere.
The DrivAerML volume prediction (u, v, w, p) is currently supervised only with
a data-fit loss — the model has no explicit incentive to produce a physically
consistent velocity field. Adding a continuity penalty `lambda * ||div(u_pred)||^2`
forces the model to learn divergence-free velocity fields, which is a strong
physical prior that reduces the solution space. The arxiv 2503.17289 paper
(DeepONet + SDF + continuity loss) demonstrates that even a soft continuity
penalty (lambda=0.01) measurably improves accuracy on aerodynamic volume fields
without requiring an exact solve. On DrivAerML, the surface pressure is
ultimately a function of the volume flow field, so volume fidelity improvements
cascade to the surface metric.

**Code changes needed:**
- During training, compute finite-difference or learned-divergence estimate of
  the predicted velocity field. Simplest: for each batch of volume points,
  use the predicted (u, v, w) and their approximate spatial Jacobian.
  Since points are scattered (not on a grid), use a local polynomial fit or
  a KNN-based finite difference approximation.
- Alternatively, use automatic differentiation if the model can compute
  `d(u_pred)/dx` via `torch.autograd.grad` — this is exact but adds backward
  pass overhead.
- Loss: `total_loss = data_loss + lambda_div * mean(div_u_pred^2)`
- Add `--div_loss_weight` flag (default 0.0 for backward compat)

**Specific hyperparameters:**
- lambda_div: 0.01 (start small; ablate 0.001, 0.1)
- Divergence estimation: KNN with k=8 neighbors for finite difference
- Only apply to volume predictions (not surface); surface pressure is a
  boundary condition not directly constrained by continuity

**GPU allocation:**
- DrivAerML: 4 GPUs (volume prediction target)
- TandemFoil Paper: 2 GPUs (TF also has velocity field predictions)
- AirfRANS: 2 GPUs (2D: div(u,v)=0 is exact for incompressible 2D flow)

**Risk:** MEDIUM-HIGH. The scattered-point divergence estimation is
non-trivial — finite differences on unstructured point clouds can be noisy.
Recommend starting with autograd if the backward pass overhead is acceptable
(typically 20-30% extra compute). If too expensive, use the soft KNN version.
This is a more complex change than the other ideas here and should be staffed
to a strong student.

---

### Rank 7 — Stochastic Depth (DropPath) Regularization

**One-line summary:** Apply layer-level stochastic depth (DropPath) with
probability 0.1–0.2 across all Transolver blocks instead of node-level dropout.

**Scientific rationale:** Standard dropout randomly zeros individual activations,
which is known to interact poorly with LayerNorm and attention mechanisms in
transformers (the normalized residual stream is corrupted in a way that hurts
training). Stochastic depth (Huang et al., 2016; adopted widely in ViT, DeiT,
Swin) instead drops entire residual blocks with probability p_drop, effectively
training an ensemble of different-depth networks and providing strong implicit
regularization. For DrivAerML, where the model has shown flat improvement
despite extensive hyperparameter tuning, stochastic depth is a form of
regularization that operates at a different level of abstraction than weight
decay or dropout — it regularizes the function class rather than individual
weights. This has not been tried on any benchmark in our experiment history.
The `timm` library has a battle-tested DropPath implementation.

**Code changes needed:**
- Import `timm.models.layers.DropPath` or implement the trivial version:
  ```python
  class DropPath(nn.Module):
      def __init__(self, drop_prob=0.0):
          super().__init__()
          self.drop_prob = drop_prob
      def forward(self, x):
          if not self.training or self.drop_prob == 0.0:
              return x
          keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_prob
          return x / (1 - self.drop_prob) * keep
  ```
- Wrap each Transolver block's residual output with DropPath
- Use linear stochastic depth schedule: block i gets
  `p_i = p_max * i / (num_blocks - 1)` (earlier blocks survive more often)
- Add `--drop_path_rate` flag (default 0.0)

**Specific hyperparameters:**
- drop_path_rate: 0.1 (start), 0.2 (higher regularization if 0.1 insufficient)
- Linear depth schedule across blocks (standard practice)
- No other changes needed

**GPU allocation:**
- DrivAerML: 3 GPUs (0.1 and 0.2 rates + one reference)
- TandemFoil: 3 GPUs
- AirfRANS: 1 GPU
- TandemFoil Paper: 1 GPU

**Risk:** LOW. Minimal code change, well-understood mechanism, no interaction
with existing architecture. Worst case: slight training slowdown with no
metric change. This is the lowest-risk "big-regularization" idea available.

---

### Rank 8 — Mixture-of-Experts FFN Layers (MoE-POT style)

**One-line summary:** Replace the dense FFN in the final 1-2 Transolver layers
with a sparse MoE FFN — multiple expert networks with a learned gating function
that routes each token to its top-2 experts.

**Scientific rationale:** MoE-POT (NeurIPS 2025, arxiv 2510.25803) demonstrated
40% error reduction on PDE surrogate benchmarks by replacing dense FFN layers
with sparse MoE. The intuition is that different experts specialize in different
physical regimes: one expert handles high-pressure stagnation regions, another
handles attached boundary layer flow, another handles separated wake. This
specialization is emergent and not manually programmed. For DrivAerML, where
the 3D surface spans dramatically different physics (stagnation zone, attached
flow, separated wake, underbody tunnel), MoE routing could capture
cross-regime variation that a single dense FFN cannot. The key implementation
detail from MoE-POT is that MoE is applied only to the later layers (where
features are already physics-informed) — applying it to early layers (where
geometric encoding is still forming) tends to destabilize routing.

**Code changes needed:**
- Implement a `SparseMoEFFN` class with:
  - `n_experts=8` expert networks (each a standard FFN)
  - Top-2 routing via a learned `nn.Linear(dim, n_experts)` gate
  - Auxiliary load-balancing loss (standard: `lambda_lb * sum(f_i * p_i)`)
- Replace the FFN in the last 1-2 Transolver blocks with SparseMoEFFN
- Add `--moe_layers 0` (default 0 = no MoE, 1 = last layer, 2 = last 2 layers)
- Add `--moe_n_experts 8` and `--moe_load_balance_weight 0.01`

**Specific hyperparameters:**
- n_experts: 8 (MoE-POT default; ablate 4 and 16)
- top-k: 2 (standard; higher k reduces sparsity benefit)
- load_balance_weight: 0.01
- Apply to last 1 layer first; escalate to 2 if improvement is seen
- Keep all other hyperparameters at DrivAerML defaults

**GPU allocation:**
- DrivAerML: 4 GPUs (primary target; high VRAM headroom at 3L/384d)
- TandemFoil: 2 GPUs
- AirfRANS: 2 GPUs

**Risk:** MEDIUM-HIGH. MoE implementation is more complex than other ideas
here. The load-balancing loss weight is a sensitive hyperparameter — too low
causes expert collapse, too high dominates the primary loss. Monitor expert
utilization during training (fraction of tokens routed to each expert should
be roughly uniform). Recommend staffing to a student with strong PyTorch
skills.

---

### Rank 9 — SDF Wall-Distance Feature (3D Geometry Embedding)

**One-line summary:** Add the signed distance from the vehicle surface (SDF)
as a precomputed scalar feature at every volume and surface point in DrivAerML.

**Scientific rationale:** Wall distance is the single most physically meaningful
geometric input for RANS-based flow solvers — the turbulence model (k-omega,
k-epsilon) explicitly depends on it. The DrivAerML volume points currently
only have raw XYZ coordinates; the model must implicitly learn wall proximity
from the distribution of neighboring points. Adding a precomputed SDF gives
the model direct access to this physical prior, which is known to govern
boundary layer development, turbulent viscosity, and near-wall pressure
recovery. The arxiv 2503.17289 paper (DeepONet + SDF) shows that even a
simple global SDF feature improves aerodynamic predictions by ~15% on their
benchmark. For DrivAerML specifically, the underbody tunnel and wheel arch
are regions where wall proximity effects are strong and errors are high.

**Code changes needed:**
- Precompute SDF at every volume and surface point using the vehicle mesh.
  Libraries: `igl` (libigl Python bindings) for exact mesh SDF, or
  `pysdf` for fast approximate SDF computation.
  ```python
  from pysdf import SDF
  sdf_fn = SDF(mesh.vertices, mesh.faces)
  wall_dist = sdf_fn(points)  # shape [N]
  ```
- Append `arcsinh(wall_dist / scale)` as a feature (scale=0.01 for car-scale
  geometry in meters)
- Add `--drivaerml_sdf_feature` flag; precompute and cache as `.npy` file
  alongside the dataset

**Specific hyperparameters:**
- SDF scale: 0.01 m (car-scale; ablate 0.001 and 0.1)
- arcsinh transformation (prevents large values from dominating)
- Cache precomputed SDF to avoid per-epoch overhead

**GPU allocation:**
- DrivAerML: 4 GPUs (volume and surface variants)
- TandemFoil: 2 GPUs (SDF to airfoil chord is physically meaningful)
- AirfRANS: 2 GPUs (2D: exact SDF to airfoil contour is trivial)

**Risk:** MEDIUM. The preprocessing step requires a mesh-based SDF library
and will add to dataset preparation time. The feature itself is low-risk —
a monotone scalar that is easy to validate visually. The main technical risk
is correctness of the SDF sign convention (inside/outside the vehicle body).

---

## Assignment Recommendation

Given 9 idle students and these 9 hypotheses, suggested initial assignment:

| Rank | Hypothesis | Primary Benchmark | Risk |
|------|-----------|------------------|------|
| 1 | Relative L2 Training Loss | DrivAerML | LOW |
| 2 | SwiGLU FFN Replacement | DrivAerML + TandemFoil | LOW |
| 3 | Surface Normals + Curvature | DrivAerML | MEDIUM |
| 4 | Prodigy Optimizer | DrivAerML + TandemFoil | LOW-MED |
| 5 | Global Context Token | DrivAerML + TandemFoil | MEDIUM |
| 6 | Mass Conservation Loss | DrivAerML volume | MED-HIGH |
| 7 | Stochastic Depth | DrivAerML + TandemFoil | LOW |
| 8 | MoE FFN Layers | DrivAerML | MED-HIGH |
| 9 | SDF Wall-Distance Feature | DrivAerML | MEDIUM |

All 9 ideas have been verified as not currently in-flight and not previously
run in the 507-PR experiment history. Ideas 1, 2, 4, and 7 can be coded
and launched by a typical student in under an hour. Ideas 3, 5, 6, 8, 9
require more careful implementation but are all motivated by specific gaps in
the current model identified through both experiment history analysis and
targeted literature review.

The highest-probability quick win for the DrivAerML gap (4.619% → 3.71%) is
**Rank 1 (Relative L2 Loss)** — it requires fewer than 20 lines of code and
directly addresses the metric-training misalignment that is the most obvious
remaining structural flaw in the current approach.
