<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-04-07 23:30

Generated after full review of 134 merged experiments, all failed PRs, and in-flight PRs (#2255, #2258, #2259, #2260, #2261). None of these ideas duplicate previous proposals.

**Current baseline (PR #2213):**
- p_in=11.979, p_oodc=7.643, p_tan=28.341, p_re=6.300
- p_tan is the hardest metric and the primary target for improvement.

**In-flight PRs (do NOT duplicate):**
- #2255: aug annealing schedule
- #2258: decoupled tandem slice (adaln-decouple-tandem)
- #2259: two-pass SRF (srf-two-pass)
- #2260: flow-regime FiLM SRF (srf-flow-film)
- #2261: per-foil target whitening (per-foil-whiten)

---

## Idea 1: Per-Head K/V Slice Attention (`per-head-kv-slice`)

**Hypothesis:** The current Physics_Attention implementation uses a shared K/V pool across all heads: `slice_token_kv = slice_token.mean(dim=1, keepdim=True)`. This means every attention head queries the same global mean of all heads' slice tokens — there is no head specialization in what is stored in K/V. Giving each head its own independent K/V projection (separate linear layers per head, or equivalently using the per-head slice token directly rather than the global mean) allows different heads to learn different physics-relevant groupings. Head 0 might specialize in pressure, head 1 in wake, head 2 in surface boundary layer — something the shared mean actively prevents.

**Mechanism:** In `Physics_Attention_Irregular_Mesh.forward`, replace:
```python
slice_token_kv = slice_token.mean(dim=1, keepdim=True)  # [B, 1, slice_num, d]
K = self.to_k(slice_token_kv)
V = self.to_v(slice_token_kv)
```
with per-head K/V using each head's own slice tokens directly:
```python
# slice_token: [B, H, slice_num, d]  — use each head's own tokens
K = self.to_k(slice_token)  # [B, H, slice_num, d_head] — no mean
V = self.to_v(slice_token)  # [B, H, slice_num, d_head]
```
`to_k` and `to_v` projections remain the same shape; we just remove the `.mean(dim=1, keepdim=True)` collapse. Q already comes from per-head slice tokens, so this makes the full attention symmetric: Q, K, V all per-head.

**Why it might help p_tan:** The tandem interaction involves long-range wake effects from the fore foil reaching the aft. A head that specializes in inter-foil communication needs its own K/V, not one contaminated by the mean of a boundary-layer-focused head.

**Risks:** Parameter count increases modestly (to_k/to_v now project [B, H, slice_num, d] → [B, H, slice_num, d_head] which has the same number of params since d = n_head * d_head). The main cost is computational: attention is computed over H independent K/V sets instead of 1 shared set. With H=3, slice_num=96, N~5000, this is O(N * H * slice_num) — not materially different from baseline. torch.compile compatible.

**Implementation notes:**
- In `Physics_Attention_Irregular_Mesh`, the `in_project_slice` outputs `[B, H, slice_num, d]` where `d = n_hidden // n_head`. Change `to_k` and `to_v` from `nn.Linear(d, d)` to `nn.Linear(d, d)` applied per-head (same shape — just remove the mean).
- The attention computation `attn = einsum('bhnd,bhsd->bhns', Q, K)` already handles the per-head case.
- Zero-init delta approach: if concerned about training stability, initialize the new per-head projection as identity + zero delta: `K = self.to_k(slice_token) + slice_token.detach()` with `to_k` zero-initialized.

**Suggested command additions:**
```
--per_head_kv
```
Boolean flag. Default False. Toggle in `Physics_Attention_Irregular_Mesh.__init__` and `forward`.

**Confidence:** Medium-high. The shared mean is a clear information bottleneck identified by reading the code directly. Whether the gain is large enough to beat baseline is uncertain, but the mechanism is clean and cost is low.

---

## Idea 2: Tandem Geometry Augmentation — Chord Ratio Perturbation (`aug-chord-ratio`)

**Hypothesis:** The current tandem augmentation adds Gaussian noise to gap (index 22) and stagger (index 23) with sigma=0.02. But the DSDF features for foil-2 (indices 6–9) encode the actual foil-2 geometry — there is already `aug_dsdf2_sigma=0.05` for log-normal scale noise on these. What is NOT augmented is the *relative* geometry: specifically, the foil chord ratio and camber relationship between fore and aft foils. In real tandem configurations, the aft foil chord can vary relative to the fore foil. The existing DSDF-scale augmentation scales foil-2's distance field uniformly, which is equivalent to scaling the chord of foil-2 — but it's applied as a multiplicative log-normal scale, which doesn't perturb the interaction asymmetrically. A dedicated "chord ratio perturbation" that differentially scales foil-1 DSDF (indices 2–5) vs foil-2 DSDF (indices 6–9) by independent factors would create more diverse tandem training scenarios and regularize the aft-foil pressure prediction.

**Mechanism:** Add a new augmentation flag `--aug_chord_ratio_sigma` (default 0.0, suggested 0.03). In the augmentation block after the existing `aug_dsdf2_sigma` section:
```python
if cfg.aug_chord_ratio_sigma > 0.0:
    # Independent log-normal scale for foil-1 and foil-2 DSDFs
    _s1 = torch.exp(torch.randn(x.size(0), device=x.device) * cfg.aug_chord_ratio_sigma)
    _s2 = torch.exp(torch.randn(x.size(0), device=x.device) * cfg.aug_chord_ratio_sigma)
    # Only augment tandem samples — single foil samples have zeros in foil-2 channels
    _is_tandem = (x[:, 0, 21].abs() > 0.01)  # same as is_tandem in train
    _s1 = torch.where(_is_tandem, _s1, torch.ones_like(_s1))
    _s2 = torch.where(_is_tandem, _s2, torch.ones_like(_s2))
    x[:, :, 2:6] = x[:, :, 2:6] * _s1.view(-1, 1, 1)
    x[:, :, 6:10] = x[:, :, 6:10] * _s2.view(-1, 1, 1)
```
This creates an independent chord-ratio perturbation for fore vs aft foil, in addition to the existing gap/stagger and foil-2 scale augmentations. The key difference from `aug_dsdf2_sigma` is that this perturbs BOTH foils independently, creating relative chord ratio variation.

**Why it might help p_tan:** p_tan measures aft-foil tangential pressure, which is most sensitive to how wake from the fore foil impinges on the aft. By training on synthetically varied chord ratios, the model learns a more geometry-invariant representation of this interaction.

**Risks:** The DSDF channels are also used for curvature computation (`curv = x[:, :, 2:6].norm(dim=-1, keepdim=True) * is_surface`). Scaling them changes the curvature feature, which could be corrupting rather than augmenting. Mitigate by zeroing the scale perturbation for surface nodes: `x[:, :, 2:6] = x[:, :, 2:6] * torch.where(is_surface.unsqueeze(-1), torch.ones_like(...), _s1.view(-1, 1, 1))`. Alternative: only apply to the `dist_surf` computation path, not the curvature path. The student should check which downstream features are affected.

**Suggested command:**
```
--aug_chord_ratio_sigma 0.03
```

**Confidence:** Medium. The augmentation gap is real (independent chord ratio is not currently covered). The implementation risk around the curvature feature requires care.

---

## Idea 3: Pressure Gradient Feature in Surface Refinement Head (`srf-pressure-gradient`)

**Hypothesis:** The Surface Refinement Head (SRF) currently takes `[hidden_192 || base_pred_3]` as input. For surface pressure prediction, the most physically informative local signal is the pressure gradient along the surface — Bernoulli's equation directly connects dp/ds to velocity, and adverse pressure gradients are the primary driver of separation and stall. We can compute a discrete approximation of the surface pressure gradient from `base_pred` (the backbone prediction) before the SRF, and inject it as an additional feature. This is cheap (no extra parameters beyond a small input expansion), physics-grounded, and directly targets the surface pressure residual that SRF is meant to correct.

**Mechanism:** After computing `base_pred` but before the SRF forward pass, for each surface node identify its arc-length neighbors (using the same surface node ordering that already exists in the data pipeline), compute forward-difference dp/ds, and append to the SRF input:

```python
# In the refinement head forward pass or training loop, before calling refine_head:
# base_pred[:, surf_idx, 2] is the backbone pressure on surface nodes
# surf_idx is sorted by arc-length (use is_surface mask + sorted x-coord as proxy)
# Compute discrete gradient: finite diff on sorted surface nodes
if hasattr(cfg, 'srf_pressure_grad') and cfg.srf_pressure_grad:
    surf_p = base_pred[:, surf_idx, 2]  # [B, N_surf]
    # Sort by x-coordinate as arc-length proxy
    sort_order = x[:, surf_idx, 0].argsort(dim=-1)  # [B, N_surf]
    # ... finite diff dp/ds, scatter back to original surf_idx ordering
    dp_ds = torch.zeros_like(surf_p)
    # central differences for interior, forward/backward for endpoints
    dp_ds[:, 1:-1] = (surf_p[:, 2:] - surf_p[:, :-2]) * 0.5
    dp_ds[:, 0] = surf_p[:, 1] - surf_p[:, 0]
    dp_ds[:, -1] = surf_p[:, -1] - surf_p[:, -2]
    # Append to SRF input: [hidden || base_pred || dp_ds_unsorted]
    srf_input = torch.cat([hidden_surf, base_pred_surf, dp_ds.unsqueeze(-1)], dim=-1)
```

The SRF's first linear layer input size increases from `n_hidden + 3` to `n_hidden + 4`. Zero-initialize the weight column for the new feature to preserve the trained baseline's behavior at the start of fine-tuning.

**Simpler alternative:** Instead of computing dp/ds explicitly, just use the difference between the current node's pressure prediction and the mean surface pressure for that sample: `dp_centered = base_pred_surf[:, :, 2] - base_pred_surf[:, :, 2].mean(dim=1, keepdim=True)`. This is a single scalar centering, trivially zero-init safe, and encodes where each node sits in the pressure distribution.

**Why it might help p_tan:** The aft foil surface pressure has a sharp suction peak followed by adverse gradient — this is exactly what the SRF struggles to capture. Giving the SRF explicit access to the local pressure gradient focuses its correction capacity on high-gradient regions.

**Implementation note:** Use the simpler `dp_centered` variant for the first trial. If it works, the full dp/ds version is a natural follow-up. The weight expansion is compatible with torch.compile as long as sizes are static.

**Suggested command:**
```
--srf_pressure_grad
```

**Confidence:** Medium. The physics motivation is clear. The implementation risk is low with the simplified version. The main uncertainty is whether the SRF already implicitly learns this from the backbone hidden state.

---

## Idea 4: Learned Temperature Annealing for Slice Assignment (`slice-temp-anneal`)

**Hypothesis:** Slice assignment in Physics_Attention uses a softmax with a fixed learned scalar temperature `temp` (initialized to 0.5). The temperature controls how "hard" the assignment is — low temp → sharp, nearly one-hot assignment; high temp → soft, distributed assignment. Early in training, hard assignment is risky because the slices haven't specialized yet, so soft assignment is preferable. Late in training, harder assignment might force better specialization, especially for the rare tandem configurations where the aft-foil wake region needs dedicated slice capacity. A scheduled temperature anneal — starting warm (0.5) and cooling toward a lower target (0.2) over training — could give the best of both worlds without adding a learnable parameter.

**Mechanism:** Replace the fixed `temp` parameter with a temperature that anneals during training. Two options:

Option A (scheduled, no new parameters): Pass the current epoch fraction to `Physics_Attention_Irregular_Mesh.forward` and compute:
```python
# In forward: epoch_frac passed from training loop
temp_start = 0.5  # matches current init
temp_end = 0.15
temp = temp_start - (temp_start - temp_end) * min(epoch_frac, 1.0)
# Use temp instead of self.temp
attn_logit = logit / temp
```

Option B (learned + scheduled lower bound): Keep `self.temp` as a learnable parameter but clip it during forward to `max(self.temp, temp_floor)` where `temp_floor` anneals from 0.5 to 0.2. This prevents the optimizer from collapsing temp to near-zero early.

**Why it might help p_tan:** Tandem samples are a minority. The slice assignment for aft-foil wake nodes may never sharpen sufficiently because there aren't enough tandem examples to drive specialization. Annealing toward harder assignment late in training — when the model has seen many examples — gives these rare regions more dedicated slice capacity.

**Implementation notes:**
- Add `--slice_temp_anneal` flag (boolean). When active, compute `epoch_frac = epoch / MAX_EPOCHS` and pass to the model's forward, or update `model.temp_floor` before each epoch.
- Easiest implementation: add a `set_temp(t)` method to `Physics_Attention_Irregular_Mesh` and call it at the start of each training epoch. No changes to forward signature needed.
- Start with Option A (no extra parameters). If that works, explore Option B.
- Compatible with torch.compile as long as temp is a tensor (not a Python float), since the value changes between epochs but the graph structure is static.

**Suggested hyperparameters:** temp_start=0.5, temp_end=0.15, linear schedule over epochs 0→MAX_EPOCHS.

**Suggested command:**
```
--slice_temp_anneal --slice_temp_end 0.15
```

**Confidence:** Medium. Temperature scheduling is well-established in contrastive learning and mixture models. The specific gain for CFD surrogates is speculative but the mechanism is clean and low-risk.

---

## Idea 5: Mixture-of-Experts FFN for Tandem vs Non-Tandem (`moe-ffn-tandem`)

**Hypothesis:** The FFN in each TransolverBlock is shared across all flow regimes. But the physics of single-foil flows and tandem flows are qualitatively different — single-foil FFN should learn Kutta condition / trailing edge behavior, while tandem FFN should learn wake interaction / downwash. A lightweight Mixture-of-Experts (MoE) with 2 experts per block — one for single-foil, one for tandem — and a hard gate based on the known `is_tandem` flag would allow complete expert specialization without any routing ambiguity. This is not a soft MoE with learned routing; it's a deterministic gate, so it's simple and interpretable.

**Mechanism:** In `TransolverBlock`, replace the single `self.mlp` with two FFNs and a deterministic switch:
```python
# __init__:
self.mlp_single = GatedMLP(n_hidden, n_hidden * ffn_factor, ...)
self.mlp_tandem = GatedMLP(n_hidden, n_hidden * ffn_factor, ...)

# forward(x, is_tandem):  is_tandem: [B] bool
# After attention:
x_single = self.mlp_single(x)  # [B, N, d]
x_tandem = self.mlp_tandem(x)  # [B, N, d]
x = torch.where(is_tandem[:, None, None], x_tandem, x_single)
```

The `is_tandem` flag is already computed in the training loop and could be passed through the model. Initialize `mlp_tandem` as a copy of `mlp_single` weights (both start identical, then diverge).

**Parameter cost:** Doubles the FFN parameters. With n_hidden=192 and ffn_factor=2, each FFN is ~150K params × 3 layers = ~450K. Doubling adds ~450K params total — modest given the 96GB VRAM budget. Total model parameter increase is ~15%.

**Practical concern:** `torch.where` evaluates BOTH branches, so compute cost also doubles for FFN. With 3 blocks each running two FFNs, this is ~2× FFN FLOPS total. FFN is the majority of transformer FLOPS so this roughly doubles epoch time. That may be too expensive for a 150-epoch budget. Mitigate by reducing ffn_factor from 2 to 1.5, or applying MoE to only 1 of the 3 blocks (the last block, which feeds the prediction heads).

**Alternative: Apply MoE only to the last block** — reduces cost by 2/3, targets the most prediction-relevant layer.

**Why it might help p_tan:** Tandem FFN would have dedicated capacity to learn the specific nonlinearities of wake interaction, without interference from single-foil boundary layer patterns.

**Suggested command:**
```
--moe_ffn_tandem --moe_ffn_last_only
```
(Start with last-block only to manage compute cost.)

**Confidence:** Medium-low. The hypothesis is sound but the compute cost is a practical concern. The `torch.where` double-evaluation issue is the same as DomainLayerNorm (which cost 40% epoch time). Recommend last-block-only variant to test the concept cheaply. If it works, expand to all blocks.

---

## Idea 6: Asymmetric Surface Loss — Suction Side vs Pressure Side (`asymmetric-surface-loss`)

**Hypothesis:** The current surface loss treats all surface nodes equally (with hard node mining based on median pressure error, but symmetric in sign). For airfoils, the physics is strongly asymmetric: the suction side (upper surface, negative Cp region) has much larger pressure gradients and is the primary driver of p_tan error. Explicitly up-weighting loss on suction-side nodes — identified by the sign of `base_pred[:, :, 2]` (negative pressure = suction side, Cp < 0) — focuses the model's gradient on the high-pressure-gradient region without adding any architectural complexity.

**Mechanism:** In the loss computation, after computing per-node surface loss:
```python
# After surf_loss_per_node computed:
if cfg.suction_side_weight > 1.0:
    # Suction side: predicted pressure < 0 (after normalization, Cp < 0)
    is_suction = (base_pred[:, surf_nodes, 2] < 0.0).float()
    # Weight: suction_side_weight on suction, 1.0 on pressure side
    ss_weights = 1.0 + (cfg.suction_side_weight - 1.0) * is_suction
    surf_loss = (surf_loss_per_node * ss_weights).mean()
```

This is extremely simple — a 3-line addition to the loss computation. The suction side identification uses the model's own prediction (which is noisy early in training but becomes reliable after ~30 epochs). Could optionally gate it with `epoch >= 30` similar to hard node mining.

**Why it might help p_tan:** The aft-foil suction peak is the hardest feature to predict. The wake from the fore foil impinges on the aft foil's leading edge and modifies the suction peak location and magnitude — this is exactly what p_tan measures. Concentrating loss signal on the suction side directly targets this failure mode.

**Hyperparameter:** `suction_side_weight` — try 1.5 (gentle) and 2.0 (aggressive). Start with 1.5.

**Interaction with existing hard node mining:** The existing hard node mining weights above-median pressure error nodes by 1.5×. This new weighting is multiplicative — a suction-side hard node would get `1.5 × 1.5 = 2.25×` weight. That may be too aggressive. Consider replacing hard node mining with the asymmetric loss rather than stacking.

**Suggested command:**
```
--suction_side_weight 1.5
```

**Confidence:** Medium-high. The physics motivation is the strongest of all ideas in this list — suction side drives separation and is where the model demonstrably fails. The implementation is trivially simple. The main risk is that using `base_pred` to identify suction side introduces a noisy signal early in training; mitigate with the epoch gate.

---

## Idea 7: Weight-Tied Auxiliary Vorticity-Free Continuity Loss (`div-free-surface`)

**Hypothesis:** The existing RANS continuity loss (in the 23:10 file as idea #2 `rans-consistency-loss`) targets volume nodes. This idea is distinct: apply a divergence-free constraint ONLY on surface nodes, using only the velocity components. On a no-slip wall (airfoil surface), the normal velocity must be zero and the tangential velocity must satisfy the no-penetration boundary condition. The discrete version: for adjacent surface nodes at positions (x_i, y_i) and (x_{i+1}, y_{i+1}), the predicted velocity should be approximately tangential to the surface. Specifically: `dot(pred_vel, surface_normal) ≈ 0` at each surface node. This is a hard physical constraint that the model currently only learns implicitly from data.

**Mechanism:** The surface normals can be approximated from the DSDF gradient (which is already in the feature set as columns 2:6):
```python
# Surface normal from foil-1 DSDF gradient (already in x[:, :, 2:4])
# On the surface, the DSDF gradient points outward from the surface
dsdf_grad_x = x[:, :, 2]  # dx component of foil-1 DSDF gradient
dsdf_grad_y = x[:, :, 3]  # dy component of foil-1 DSDF gradient
normal_norm = torch.sqrt(dsdf_grad_x**2 + dsdf_grad_y**2 + 1e-8)
nx = dsdf_grad_x / normal_norm  # [B, N]
ny = dsdf_grad_y / normal_norm  # [B, N]

# Predicted velocity at surface nodes
surf_Ux = pred[:, :, 0][surf_mask]  # after denormalization or in normalized space
surf_Uy = pred[:, :, 1][surf_mask]
# No-penetration loss: dot(vel, normal) should be 0 at surface
no_penetration = (surf_Ux * nx[surf_mask] + surf_Uy * ny[surf_mask]) ** 2
bc_loss = no_penetration.mean()
loss = loss + cfg.bc_loss_weight * bc_loss
```

**Why it might help p_tan:** The aft foil surface in tandem configurations has modified effective AoA due to the fore foil wake. The no-penetration constraint helps anchor the velocity direction to the true surface geometry, which then propagates to more accurate pressure via Bernoulli-like coupling.

**Why this is different from the 23:10 RANS loss:** The 23:10 idea uses finite differences to approximate div(U) on volume nodes. This idea uses the already-available DSDF gradient (a first-order quantity, not a second-order finite difference) to enforce the boundary condition at surface nodes specifically. It is geometrically exact (the DSDF gradient IS the surface normal to first order), requires no finite differencing, and targets surface nodes directly.

**Implementation notes:**
- The DSDF gradient in columns 2:4 may or may not be normalized. Check `prepare_multi.py` comments or print norms for surface nodes — if DSDF is a signed distance field, `||grad DSDF|| ≈ 1` everywhere (Eikonal property), so no normalization needed.
- Gate this loss to epoch >= 20 (when the model produces reasonable velocity predictions).
- Start with `bc_loss_weight=0.1` and try 0.5.
- Apply to BOTH foil-1 and foil-2 surface nodes (use foil-2 DSDF gradient from columns 6:8 for aft foil surface).

**Suggested command:**
```
--bc_loss_weight 0.1
```

**Confidence:** Medium. The physics is correct. The key uncertainty is whether DSDF gradient in the feature set accurately represents the outward normal — this depends on `prepare_multi.py` which is read-only. The student should verify by printing the angle between DSDF gradient and the known foil geometry normal for a few samples before trusting the loss.

---

## Idea 8: Stochastic Slice Dropout During Training (`slice-dropout`)

**Hypothesis:** The 96 slice tokens in Physics_Attention form a learned partition of the flow field. During training, all 96 slices always receive information — there is no pressure on any individual slice to carry robust information independently. Randomly dropping a fraction of slice tokens during training (zero-masking their contribution before the attention softmax) forces each slice to encode information that doesn't rely on the others being present. This is analogous to Dropout for attention heads, or to masked autoencoding applied to the slice token pool. It should increase the robustness of individual slice representations and reduce the co-adaptation between slices that makes the model brittle on rare tandem configurations.

**Mechanism:** In `Physics_Attention_Irregular_Mesh.forward`, after computing `slice_token` and before the K/V projection:
```python
if self.training and cfg.slice_dropout_p > 0.0:
    # slice_token: [B, H, slice_num, d]
    # Randomly zero out slice tokens (per-batch, per-head independently)
    drop_mask = torch.rand(slice_token.shape[0], slice_token.shape[1], 
                           slice_token.shape[2], 1, device=slice_token.device)
    drop_mask = (drop_mask > cfg.slice_dropout_p).float()
    # Rescale to preserve expected sum (inverted dropout)
    slice_token = slice_token * drop_mask / (1.0 - cfg.slice_dropout_p)
```
Apply this ONLY to the K/V side (i.e., to `slice_token_kv`), not to the Q side — we don't want to drop the query representation, only the key-value bank. This makes the model robust to any individual slice being absent.

**Why it might help p_tan:** The aft-foil wake region may be represented by only a few specialized slices. If those slices co-adapt during training, a small distribution shift (new tandem geometry) can cause them all to fail simultaneously. Slice dropout prevents co-adaptation by forcing each slice to be useful independently.

**Hyperparameter:** `slice_dropout_p` — try 0.05 (drop ~5 of 96 slices per forward pass) and 0.10. Start with 0.05. This is very mild dropout compared to standard values.

**Interaction with existing regularization:** The model already has SAM (active after 75% of epochs) and L2 weight decay. Slice dropout adds a different type of regularization targeting the information bottleneck rather than parameter magnitude. They are complementary.

**torch.compile note:** The `torch.rand` inside forward during training is fine for compile as long as shapes are static (they are — slice_num=96 is a fixed config value).

**Suggested command:**
```
--slice_dropout_p 0.05
```

**Confidence:** Medium. Dropout variants for attention mechanisms have mixed results in the literature — sometimes regularizing, sometimes destabilizing. The key risk is that 96 slices are already a severe bottleneck (96 tokens for ~5000 nodes), so dropping even 5 of them may cause information loss during training that never fully recovers. Mitigate by (a) using mild p=0.05, (b) annealing dropout to 0 after epoch 100, (c) applying only to the last 2 layers (not all 3 blocks).

---

## Summary Table

| # | Slug | Primary target | Complexity | Confidence |
|---|------|---------------|------------|------------|
| 1 | `per-head-kv-slice` | p_tan (head specialization) | Low | Medium-high |
| 2 | `aug-chord-ratio` | p_tan (tandem generalization) | Low | Medium |
| 3 | `srf-pressure-gradient` | p_tan (SRF input enrichment) | Low | Medium |
| 4 | `slice-temp-anneal` | p_tan (slice specialization) | Low | Medium |
| 5 | `moe-ffn-tandem` | p_tan (regime separation) | Medium | Medium-low |
| 6 | `asymmetric-surface-loss` | p_tan (suction side focus) | Very low | Medium-high |
| 7 | `div-free-surface` | p_tan (boundary condition) | Low | Medium |
| 8 | `slice-dropout` | All surface (regularization) | Low | Medium |

**Top picks for immediate assignment:**
1. `asymmetric-surface-loss` — highest physics motivation, trivial implementation, no architectural risk
2. `per-head-kv-slice` — targets a clear code-level information bottleneck, clean mechanism
3. `srf-pressure-gradient` — SRF improvements have a good track record in this program; simplified dp_centered version is very low risk
4. `slice-temp-anneal` — novel scheduling idea with clear mechanistic motivation for tandem specialization
