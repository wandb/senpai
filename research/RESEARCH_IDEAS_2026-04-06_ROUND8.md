<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round 8 Research Ideas — 2026-04-06

Target: p_tan < 28.0 (current baseline 28.502, PR #2184)
Constraints: train.py only, torch.compile-compatible, 150-epoch / 180-min timeout, 96 GB VRAM.

---

## Idea 1: Fore-to-Aft Hidden State Injection via Cross-Attention in AftFoilRefinementHead

**Title/Slug:** `fore-aft-crossattn-srf`

**Hypothesis:**
The aft-foil refinement head (AftFoilRefinementHead) currently operates only on the aft-foil's own surface hidden states and base predictions. But the aft-foil's pressure field is fundamentally determined by the wake of the fore-foil — information that lives in the fore-foil's surface hidden states. By adding a lightweight cross-attention step inside the aft-foil SRF head that attends from aft-foil surface nodes to fore-foil surface nodes (both using the final hidden states from the last TransolverBlock), we inject explicit upstream dependency into the most impactful head. This is architecturally cheap (a single multi-head attention layer over a small set of surface nodes), does not affect the Transolver backbone, and targets the exact mechanism responsible for p_tan difficulty.

**Why p_tan specifically:**
p_tan is hard because NACA6416 fore-foil produces a different wake signature than training fore-foils — the Transolver backbone generalizes imperfectly, but the AftFoilRefinementHead currently cannot compensate using wake context. Cross-attention forces explicit fore-aft coupling in the final correction stage.

**Implementation steps:**
1. Add new class `AftFoilCrossAttnRefinementHead` in train.py.
   - Constructor takes `n_hidden, out_dim, hidden_dim=192, n_layers=3, n_heads=4`.
   - Cross-attention: `nn.MultiheadAttention(n_hidden, n_heads, batch_first=True)`.
   - Query = aft-foil surface hidden states `[A, n_hidden]` (unsqueezed to `[1, A, n_hidden]` for batched processing).
   - Key/Value = fore-foil surface hidden states `[F, n_hidden]` (unsqueezed to `[1, F, n_hidden]`).
   - Add a `LayerNorm(n_hidden)` after cross-attention output; apply residual connection.
   - Concatenate with aft base_pred and run through same MLP tower as current AftFoilRefinementHead. Zero-init output layer.
2. In `Transolver.__init__`, add flag `aft_foil_crossattn_srf: bool = False` (and in Config).
3. In `Transolver.forward`, when computing aft-foil SRF correction, identify fore-foil nodes via boundary_id==6, extract their hidden states from `fx_deep`, run the cross-attn head.
4. The `fx_deep` tensor is already computed and available (line 905 of current train.py).
5. For torch.compile: use `torch.nn.functional.scaled_dot_product_attention` directly instead of `nn.MultiheadAttention` to avoid dynamic shapes. Pad fore/aft sequences to a fixed max (e.g., 256 nodes).

**Key hyperparameters:**
- `--aft_foil_crossattn_srf` flag to enable
- `--aft_foil_ca_heads 4` (number of attention heads)
- Start with n_heads=4, hidden_dim=192 (same as current aft SRF)

**Expected impact:** Medium-high. This is the most principled fix for p_tan — it directly models the fore-aft physical dependency at the refinement stage. Risk: cross-attention over variable-size node sets may require padding for torch.compile; must verify compile compatibility.

**Differentiation from prior work:**
PR #1209 tried tandem cross-attention at the Transolver backbone level on a single-layer model (0.8688 val/loss, early Phase 1). That was cross-attention between full node sets (very expensive, didn't survive scaling). This idea targets only the small surface node sets (~100-300 nodes) inside the existing AftFoilRefinementHead, keeping the backbone unchanged. Never tried in the post-SRF, post-aft-foil-head architecture.

---

## Idea 2: Asinh Scale Progressive Annealing — Start Sharper, Finish Compressed

**Title/Slug:** `asinh-scale-anneal`

**Hypothesis:**
The current asinh pressure transform uses a fixed scale of 0.75 (merged PR #2054). The choice of scale is a tradeoff: lower scale (0.5) compresses dynamic range more aggressively (better for OOD high-pressure), higher scale (1.0-1.5) preserves fine-grained gradients near zero. A progressive schedule — starting with high scale (1.5, raw-ish gradients) in early training where the model needs to learn large-scale structure, then annealing to a lower scale (0.5-0.75) in late training where fine-grained OOD correction matters — should get the best of both regimes. This is analogous to curriculum learning in the prediction space.

**Implementation steps:**
1. Add `Config` fields: `asinh_scale_start: float = 1.5`, `asinh_scale_end: float = 0.5`, `asinh_scale_anneal_epochs: int = 80`.
2. In the training loop, compute current asinh scale as a cosine interpolation:
   ```python
   if cfg.asinh_pressure and cfg.asinh_scale_end != cfg.asinh_scale:
       t = min(1.0, epoch / cfg.asinh_scale_anneal_epochs)
       scale_t = cfg.asinh_scale_start + 0.5*(1 - math.cos(math.pi * t)) * (cfg.asinh_scale_end - cfg.asinh_scale_start)
   else:
       scale_t = cfg.asinh_scale
   ```
3. Pass `scale_t` to the target transform and de-normalization at inference time (currently uses `cfg.asinh_scale`).
4. The `stats` dict may need updating if asinh is applied per-batch rather than pre-computed; verify how the current code applies the transform.
5. For val evaluation always use `asinh_scale_end` (the final value) so metrics are comparable.

**Key hyperparameters:**
- `--asinh_scale_start 1.5 --asinh_scale_end 0.5 --asinh_scale_anneal_epochs 80`
- Alternative: `--asinh_scale_start 1.0 --asinh_scale_end 0.5` (less aggressive)

**Expected impact:** Low-medium. This is a cheap 1-line schedule change. Risk is minimal. The key uncertainty is whether the current code applies asinh at the batch level (easy to make dynamic) or pre-computes it in stats (harder — would need to verify).

**Differentiation from prior work:**
Fixed scale 0.75 was tuned empirically (PR #2054). No experiment has tried dynamic scale annealing. Progressive curricula in prediction space are well-established in depth estimation and optical flow.

---

## Idea 3: Slice Diversity Regularization — Prevent Physics State Collapse

**Title/Slug:** `slice-diversity-reg`

**Hypothesis:**
With 96 slices and only 1322 training samples, there is a real risk that many slices collapse to the same representation ("dead slices"), especially for tandem OOD geometries that are underrepresented in training. A diversity loss on the slice tokens — penalizing low cosine distance between pairs of slices — forces the model to maintain orthogonal physics representations. This is analogous to the "slice collapse" problem in Mixture-of-Experts gating. For p_tan, the OOD geometry likely needs a specific slice to represent the modified wake, but if that slice has collapsed with another, the routing cannot differentiate it.

**Implementation steps:**
1. In `Physics_Attention_Irregular_Mesh.forward`, after computing `slice_token = torch.einsum("bnm,bnh->bmh", slice_weights, x)`, optionally compute a diversity loss:
   ```python
   # slice_token: [B, slice_num, n_hidden]
   if self.training and self.diversity_weight > 0:
       st_norm = F.normalize(slice_token, dim=-1)  # [B, S, H]
       gram = torch.bmm(st_norm, st_norm.transpose(1,2))  # [B, S, S]
       eye = torch.eye(gram.shape[-1], device=gram.device).unsqueeze(0)
       diversity_loss = ((gram - eye)**2).sum(dim=(-1,-2)).mean()
   ```
2. Add `slice_diversity_weight: float = 0.0` to Config. Only active when > 0.
3. Accumulate `diversity_loss` into the total loss (weighted by `slice_diversity_weight`). Apply to all 3 blocks or only the last block.
4. In the `Physics_Attention_Irregular_Mesh.__init__`, add `diversity_weight` parameter (default 0.0). Pass via TransolverBlock.
5. For torch.compile: no dynamic control flow; the `if self.training` guard is fine since compile handles it.

**Key hyperparameters:**
- `--slice_diversity_weight 0.001` (start very small; the gram matrix scale grows with slice_num)
- Try applying only to the last block first (cheapest path)
- Weight sweep: 1e-4, 1e-3, 1e-2

**Expected impact:** Speculative but principled. With 96 slices and ~1322 samples, collapse is plausible. If slices are already diverse, this adds only noise; risk is low given small weight. Benefit could be significant for OOD generalization if tandem samples map to under-utilized slices.

**Differentiation from prior work:**
PR #1181 proposed "slice diversity contrastive loss" but was tagged "never ran." The present proposal uses a simpler gram-matrix orthogonality loss (no negatives, no temperature) which is more stable and torch.compile-friendly.

---

## Idea 4: Foil-Relative Chord-Normalized Coordinate Frame as Additional Input Features

**Title/Slug:** `chord-normalized-coords`

**Hypothesis:**
The current input features use raw (x,y) coordinates and DSDF distances. But for surface pressure prediction, the physically meaningful coordinate is position along the chord (x/c) and distance above/below chord line (y/c). This is how aerodynamicists parameterize the pressure distribution — Cp vs x/c. For OOD geometries (NACA6416 with higher camber), the wake/pressure coupling is best described in chord-normalized terms. Adding (x_chord/c, y_chord/c) for each foil as learned or rule-based features gives the model a geometry-invariant coordinate.

**Implementation steps:**
1. The input `x` tensor has shape `[B, N, X_DIM=24+]`. The current DSDF features (dims 2-23) already contain distance information. We need chord-normalized positions.
2. In train.py, after loading the batch (in the training loop or as an augmentation), compute:
   - For fore-foil nodes (boundary_id==6): translate/rotate to the fore-foil chord frame; normalize by chord length c.
   - For aft-foil nodes (boundary_id==7): same for aft-foil.
   - For volume nodes: use distance to nearest foil chord as the feature.
3. Approximate chord frame: leading edge ≈ min(x) of foil surface, trailing edge ≈ max(x). Chord vector = TE - LE. Project all foil surface node positions onto this axis and perpendicular.
4. Append 2 new features (x_chord/c, y_chord/c) to the x tensor. This increases X_DIM by 2.
5. The `preprocess` GatedMLP2 takes `fun_dim + space_dim`; adjust `fun_dim` accordingly, or better: pass the chord features through `raw_xy` for the spatial_bias MLP (already handles 4 or 6 dims, extend to 8).

**Critical note:** The `prepare.py` data loading is read-only. Chord features must be computed inside `train.py` from the existing `x` and boundary_id tensors, using torch operations that are compile-compatible (no Python loops over batches).

**Key hyperparameters:**
- Flag `--chord_coords` to enable
- Whether to include only for surface nodes or all nodes (start with all)
- Whether to use signed or unsigned distance from chord midline

**Expected impact:** Medium. Chord-normalized coordinates are domain knowledge that the model currently lacks. Risk: chord frame estimation from boundary nodes may be noisy for NACA6416 (different camber profile shifts the "chord" estimate). Validate by checking that the computed chord vectors are plausible.

**Differentiation from prior work:**
No experiment has tried chord-normalized coordinates. DSDF features (merged) give distance to the foil surface but not position along the chord. This provides a fundamentally different geometric parameterization.

---

## Idea 5: Wake Deficit Feature — Estimated Upstream Velocity Reduction as Explicit Input

**Title/Slug:** `wake-deficit-feature`

**Hypothesis:**
The aft-foil's pressure distribution is set primarily by the incoming velocity deficit from the fore-foil's wake. The model currently infers this indirectly from DSDF distances and geometry. An explicit "wake deficit proxy" feature — computed as a function of the aft-foil node's position relative to the fore-foil trailing edge — gives the model a physics-informed shortcut for the most important tandem interaction signal. Concretely: for each node, compute the signed distance in the streamwise direction from the fore-foil trailing edge, normalized by the gap, and the perpendicular offset from the wake centerline. These two numbers encode whether a node sits inside the wake core or outside it.

**Implementation steps:**
1. From the input `x` tensor, extract fore-foil surface nodes (boundary_id==6 or equivalently DSDF channel for foil 1 near-zero).
2. Estimate fore-foil trailing edge position: `x_te = max(x_coord) of fore-foil surface nodes per sample`.
3. For each node, compute:
   - `dx_te = x_node - x_te_fore` (streamwise distance from fore-foil TE, normalized by gap)
   - `dy_te = y_node - y_te_fore` (lateral offset from fore-foil TE wake centerline)
4. Append `[dx_te / gap, dy_te / gap]` as 2 new features (clipped to [-3, 3]).
5. For single-foil samples: set these features to a sentinel value (e.g., -10.0) or zero.
6. Implement entirely in torch (no Python loops), using gather/topk over surface nodes to find TE. Compile-compatible.
7. For spatial_bias MLP: include wake features in `raw_xy` (extend from 6 to 8 dims with `--gap_stagger_spatial_bias` active).

**Key hyperparameters:**
- Flag `--wake_deficit_feature`
- Normalization: try gap-normalized vs chord-normalized vs raw
- Whether to use only for aft-foil SRF input or also in the main backbone

**Expected impact:** Medium-high. This is a direct physical feature encoding the key mechanism that makes p_tan hard. The model currently must discover the wake interaction from geometry alone — this makes it explicit. Risk: TE estimation may be noisy; requires robust extraction from DSDF or boundary_id mask.

**Differentiation from prior work:**
No experiment has added an explicit wake position feature. Gap/stagger features (merged, PR #2115) encode the geometry of the tandem configuration but not the wake interaction itself. DSDF features give boundary distances, not wake structure.

---

## Idea 6: Boundary-Condition-Preserving Loss — Enforce Kutta Condition at Trailing Edge

**Title/Slug:** `kutta-condition-loss`

**Hypothesis:**
The Kutta condition requires that the pressure difference across the trailing edge is zero (the flow leaves tangentially from both surfaces). Equivalently, the velocity field must be smooth at the trailing edge — no discontinuity. Currently, the model is not explicitly trained to satisfy this condition. For OOD geometries (NACA6416 with higher camber), the trailing edge location and the geometry of the pressure recovery changes significantly. Adding a soft Kutta condition loss — penalizing pressure discontinuities at the trailing edges — provides a physics constraint that is especially binding for novel geometries, acting as regularization that keeps predictions on the physical manifold.

**Implementation steps:**
1. For each surface node set, identify trailing edge nodes: these are the nodes with maximum x-coordinate among the surface nodes (or equivalently, DSDF ≈ 0 and curvature ≈ large). Use `topk(2)` to get the 2 nodes closest to the trailing edge for each foil surface.
2. Compute Kutta loss: `pressure_upper_TE - pressure_lower_TE` should be ≈ 0.
   - Group surface nodes by foil (use boundary_id). For each foil, split into upper/lower surface by sign of y_chord.
   - Find upper-TE node: max x among upper surface; find lower-TE node: max x among lower surface.
   - Kutta loss = `(p_upper_TE - p_lower_TE)^2`.
3. Apply to the predicted pressure channel (after de-normalizing back to physical space).
4. Add `kutta_loss_weight: float = 0.0` to Config; accumulate into total loss.
5. For torch.compile: use differentiable topk/argmax; avoid Python branching.
6. Apply to both fore and aft foils; weight aft foil loss 2x (more impactful for p_tan).

**Key hyperparameters:**
- `--kutta_loss_weight 0.01`
- Whether to apply in asinh-transformed space or physical space (try physical space first; the gradient is more meaningful there)
- Epochs to activate (delay until epoch 20 to avoid destabilizing early training)

**Expected impact:** Speculative. The Kutta condition is important in aerodynamics but the model may already learn to approximately satisfy it from the data. If predictions violate the Kutta condition on OOD geometries, this could provide meaningful correction. Risk: defining "trailing edge nodes" from the mesh in a compile-compatible way is tricky.

**Differentiation from prior work:**
No experiment has tried a physics-constraint loss from aerodynamics. DCT frequency loss (#2184) constrains spectral content of the surface pressure; Kutta loss constrains the boundary condition at a specific geometric point. These are complementary.

---

## Idea 7: Stochastic Depth with Tandem-Biased Survival Rate

**Title/Slug:** `tandem-biased-stochastic-depth`

**Hypothesis:**
Standard stochastic depth (DropPath) randomly drops Transolver blocks during training with uniform probability. For tandem-foil generalization, the relevant physics requires deeper propagation through all 3 blocks — the aft-foil wake interaction is a higher-order effect that needs multiple passes of attention. By using a lower survival probability for single-foil samples (encouraging shallow early exit, efficient regularization) and a higher survival probability for tandem samples (encourage full depth), we bias the regularization toward preserving the full computational path for the harder OOD cases. This is a form of data-conditional depth regularization.

**Implementation steps:**
1. Add `drop_path_rate: float = 0.0` and `drop_path_tandem_boost: float = 0.1` to Config.
2. In `TransolverBlock.forward`, add DropPath: multiply the block output by a Bernoulli mask.
   ```python
   # is_tandem: [B, 1, 1, 1] boolean
   if self.training and drop_path_rate > 0:
       base_p = drop_path_rate
       survival = 1.0 - base_p + tandem_boost * is_tandem.squeeze()  # [B]
       mask = torch.bernoulli(survival.clamp(0, 1)).view(-1, 1, 1)  # [B, 1, 1]
       fx = fx * mask / survival.view(-1, 1, 1).clamp(min=0.1)  # rescale
   ```
3. This is compile-compatible: `torch.bernoulli` is a valid stochastic op in compile mode.
4. Apply to all 3 blocks with a linearly increasing rate (deeper = higher drop rate, standard stochastic depth schedule).
5. Interaction with EMA: since EMA averages over many steps, the variance from stochastic depth averages out. No conflict.

**Key hyperparameters:**
- `--drop_path_rate 0.1 --drop_path_tandem_boost 0.1` (base survival 0.9, tandem survival 1.0)
- Try `--drop_path_rate 0.15 --drop_path_tandem_boost 0.15` (more aggressive)
- Linear schedule: block 1 rate=0.05, block 2 rate=0.10, block 3 rate=0.0 (last block always survives for output head)

**Expected impact:** Low-medium. Stochastic depth is well-validated as a regularizer for transformers but has never been tested in this exact setting. The tandem-biased variant is novel. Main risk: survival rate calculation must avoid torch.compile graph breaks.

**Differentiation from prior work:**
PR #2196 was assigned "stochastic depth" and was reassigned to local KNN attention (#2200). No stochastic depth experiment has actually run. The tandem-biased survival rate is a novel variant not found in standard literature.

---

## Idea 8: Iterative Refinement — Run the Surface Refinement Head Multiple Times

**Title/Slug:** `iterative-srf`

**Hypothesis:**
The current surface refinement heads (fore-foil SRF and aft-foil SRF) apply a single additive correction to the base prediction. But a single MLP forward pass may be insufficient for large OOD errors. Iterative refinement — running the SRF head N times, feeding each iteration's output as the new "base_pred" input — can make larger corrections with the same parameter count. This is equivalent to unrolled gradient descent in the prediction space, and has been shown effective in optical flow (RAFT), depth estimation, and protein structure prediction (AlphaFold's recycling). For p_tan, where the error is large (28.5 vs 7.8 for p_oodc), multiple passes could compound small per-step corrections.

**Implementation steps:**
1. Add `srf_n_iters: int = 1` and `aft_srf_n_iters: int = 1` to Config.
2. In `Transolver.forward`, where SRF is applied, wrap it in a loop:
   ```python
   current_pred = base_pred
   for _ in range(cfg.srf_n_iters):
       corr = self.surface_refine_head(hidden_surf, current_pred)
       current_pred = current_pred + corr
   pred[surf_mask] = current_pred
   ```
3. Optionally: add an "update gate" — a learned scalar per iteration that controls the step size:
   ```python
   # gate parameter per iteration (initialized to 0.0 = small steps initially)
   self.srf_gates = nn.Parameter(torch.zeros(srf_n_iters))
   step = torch.sigmoid(self.srf_gates[i]) * corr
   ```
4. Apply the same logic to AftFoilRefinementHead.
5. Torch.compile: a Python `for _ in range(n)` loop with fixed `n` is compile-compatible (unrolled at trace time). Do not use dynamic n.

**Key hyperparameters:**
- `--srf_n_iters 3 --aft_srf_n_iters 3` (start with 3; 5 is the maximum before memory pressure)
- Try with and without the update gate (gate = 1/n_iters simplest baseline, then learned)
- `--srf_n_iters 2` minimal viable test

**Expected impact:** Medium-high. Iterative refinement with weight-tied heads is one of the most reliable techniques for improving OOD accuracy in structured prediction. RAFT achieves its gains precisely this way. Risk: if the SRF correction is already well-tuned, more iterations may just amplify noise. Monitor whether corrections decrease in magnitude across iterations (good sign).

**Differentiation from prior work:**
No experiment has tried iterative/recurrent refinement of surface predictions. All current SRF variants (fore-foil, aft-foil, context-augmented) apply exactly one correction pass. This is a direct application of the RAFT/AlphaFold recycling paradigm to the surface refinement problem.

---

## Idea 9: Tandem-Conditioned Feature Cross-Interaction Before Backbone

**Title/Slug:** `tandem-feature-cross`

**Hypothesis:**
The current `feature_cross` module (line 885 of train.py) applies a fixed linear cross-interaction across all input features: `x_cross = x * feature_cross(x)`. This is a domain-agnostic multiplicative interaction. For tandem samples, the physically important cross-interactions are different: DSDF-to-foil-2 × DSDF-to-foil-1 encodes proximity to both foils simultaneously; AoA × gap encodes how the gap-modified wake depends on angle. By adding a tandem-specific feature cross layer (active only when `is_tandem`, else frozen to identity) that mixes DSDF channels from both foils, we give the preprocess stage access to tandem-specific feature interactions before the backbone sees them.

**Implementation steps:**
1. Add a second `feature_cross_tandem: nn.Linear(fun_dim + space_dim, fun_dim + space_dim, bias=False)` initialized to identity, like the existing one.
2. In `Transolver.forward`, blend the two cross-interactions:
   ```python
   is_tandem_float = (x[:, 0, 21].abs() > 0.01).float()[:, None, None]  # [B, 1, 1]
   x_cross_base = x * self.feature_cross(x)
   x_cross_tandem = x * self.feature_cross_tandem(x)
   x_cross = x_cross_base + is_tandem_float * (x_cross_tandem - x_cross_base)
   x = x + 0.1 * x_cross
   ```
3. The tandem cross layer starts as identity — initially no effect. As training proceeds, it learns tandem-specific interactions.
4. Add `--tandem_feature_cross` flag to Config.
5. Torch.compile: straightforward; all ops are standard linear/multiply, no dynamic shapes.

**Key hyperparameters:**
- `--tandem_feature_cross` flag only (no hyperparameter sweep needed for first experiment)
- Optional: `--tandem_cross_weight 0.1` to control mixing coefficient (start with hard 1.0 for tandem samples)

**Expected impact:** Low-medium. The feature cross is a small module. If the model has already learned tandem-relevant interactions implicitly through the backbone, this adds little. But if the preprocess bottleneck is limiting OOD generalization, domain-specific feature interactions could provide a meaningful boost.

**Differentiation from prior work:**
The existing `feature_cross` is domain-agnostic. AdaLN conditioning (#2130 area) modulates layer norms with flow conditions (Re, AoA) but not with tandem geometry-specific feature crosses. This is a pre-backbone tandem-specific multiplicative interaction, never tried.

---

## Idea 10: Harmonic Embedding for Spatial Coordinates — Replace Raw xy with Fourier Features

**Title/Slug:** `fourier-pos-embed`

**Hypothesis:**
The spatial bias MLP receives raw (x, y, curvature, dist) coordinates and must learn to produce slice routing logits. Raw Cartesian coordinates are a poor inductive basis for periodic/oscillatory pressure patterns. Replacing raw (x, y) with multi-scale sinusoidal Fourier features — `[sin(2π f x), cos(2π f x), sin(2π f y), cos(2π f y)]` for f ∈ {1, 2, 4, 8, 16} — gives the spatial bias MLP a richer frequency basis to route slices, analogous to NeRF's positional encoding. For p_tan, the pressure variation on the aft foil has both large-scale (wake deficit) and small-scale (leading-edge stagnation) components; multi-scale features help the routing capture both simultaneously.

**Implementation steps:**
1. Define a fixed Fourier embedding: frequencies `[1, 2, 4, 8, 16]` Hz (5 frequencies × 2 trig functions × 2 dims = 20 features), plus the raw (x, y) (2 features) = 22 features total.
2. Replace the first 2 dims of `raw_xy` (currently `x[:,:,:2]`) with the Fourier embedding:
   ```python
   freqs = torch.tensor([1., 2., 4., 8., 16.], device=x.device)  # [F]
   xy = x[:, :, :2]  # [B, N, 2]
   # [B, N, 2, F] → [B, N, 2F]
   xy_cos = torch.cos(2 * math.pi * xy.unsqueeze(-1) * freqs)
   xy_sin = torch.sin(2 * math.pi * xy.unsqueeze(-1) * freqs)
   fourier_xy = torch.cat([xy_cos, xy_sin], dim=-1).flatten(-2)  # [B, N, 20]
   raw_xy = torch.cat([fourier_xy, x[:, :, 24:26]], dim=-1)  # [B, N, 22]
   ```
3. Adjust `spatial_bias_input_dim` from 4 (or 6) to 22 (or 24 with gap_stagger).
4. The spatial bias MLP first linear is `Linear(spatial_bias_input_dim, 64)` — update input size accordingly.
5. Flag: `--fourier_pos_embed` (and `--fourier_pos_freqs "1,2,4,8,16"` for configurability).
6. Torch.compile: all static ops, no dynamic shapes.

**Key hyperparameters:**
- Frequencies: start with `[1, 2, 4, 8, 16]` (physical scale of ~0.06m chord → ~1m domain)
- Number of frequencies: 5 (20 features total) is sufficient; more risks overfitting with 1322 samples
- Whether to also include in the `preprocess` GatedMLP2 (start with spatial_bias only, less risky)

**Expected impact:** Low-medium. Fourier features for positional encoding are well-established (NeRF, Perceiver, FNO all use them), but the current model already uses DSDF distance features which provide implicit multi-scale spatial information. The marginal gain may be limited. Risk: the spatial bias MLP grows its input; ensure zero-init is preserved for the gap/stagger dims if combined with `--gap_stagger_spatial_bias`.

**Differentiation from prior work:**
PR #1125 tried RoPE (Rotary Position Embedding) in the slice-token attention (0.9097, Phase 1). That was rotary encoding for the attention Q/K, not for the spatial routing. This targets the `raw_xy` input to the spatial bias MLP specifically — a different module, different purpose, never tried in Phase 6 architecture.

---

## Summary Table

| # | Slug | Target | Complexity | Confidence |
|---|------|--------|------------|------------|
| 1 | fore-aft-crossattn-srf | p_tan (fore-aft coupling) | Medium | Medium-High |
| 2 | asinh-scale-anneal | p_tan + p_oodc (prediction space) | Low | Low-Medium |
| 3 | slice-diversity-reg | p_tan (OOD routing) | Low | Speculative |
| 4 | chord-normalized-coords | p_tan (geometric features) | Medium | Medium |
| 5 | wake-deficit-feature | p_tan (physics features) | Medium | Medium-High |
| 6 | kutta-condition-loss | p_tan (physics constraint) | Medium | Speculative |
| 7 | tandem-biased-stochastic-depth | All OOD tracks | Low | Low-Medium |
| 8 | iterative-srf | p_tan (correction accuracy) | Low | Medium-High |
| 9 | tandem-feature-cross | p_tan (feature interaction) | Low | Low-Medium |
| 10 | fourier-pos-embed | All tracks (spatial routing) | Low | Low-Medium |

**Top 3 recommended for immediate assignment (best risk/reward for p_tan):**
1. `iterative-srf` — low complexity, directly extends the proven SRF mechanism, RAFT-style recycling
2. `fore-aft-crossattn-srf` — highest physical motivation, targets the exact p_tan mechanism
3. `wake-deficit-feature` — explicit physics feature for the key tandem interaction signal
