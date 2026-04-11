# SENPAI Research Ideas — 2026-04-05 Round 2

Generated after deep architecture survey of train.py and full review of experiment history through PR #2163.

**Current baseline (PR #2130):** p_in=13.05, p_oodc=7.70, p_tan=28.60, p_re=6.55

**Primary bottleneck:** p_tan=28.60 is 2.19x worse than p_in=13.05. This is an OOD geometry transfer failure — the model trained on single-foil and same-configuration tandem fails to generalize to the NACA6416 OOD tandem configuration. Every idea below addresses this from a different angle.

**Key research pattern from Phase 6:** Only changes that alter the prediction task or inject genuinely new information have yielded improvements. Incremental hyperparameter tuning is exhausted. The Transolver architecture is a strong local optimum. We need structural changes.

**Currently in-flight (do NOT duplicate):**
- fern #2161: fore-foil FiLM SRF
- nezuko #2163: differential LR for SRF/aft-srf heads
- askeladd #2162: tandem cross-DSDF features
- edward #2158: asymmetric-pcgrad
- alphonse #2157: (check current state)
- tanjiro #2156: (check current state)
- thorfinn #2154: (check current state)
- frieren #2153: (check current state)

---

## Idea 1: Backbone-Wide Gap/Stagger Conditioning via AdaLN-All

**Hypothesis:** The gap and stagger scalars (encoding tandem foil spacing and offset) are currently injected only in the output-layer refinement head. Threading this geometric condition through ALL three TransolverBlocks via the existing `adaln_all=True` infrastructure will give the backbone physics-attention mechanism the information it needs to modulate slice assignments differently for different tandem configurations — the root cause of p_tan failures.

**Theoretical motivation:** The current architecture treats the backbone attention as geometry-agnostic and relies on the refinement head to apply tandem corrections post-hoc. But slice assignment (which mesh nodes get grouped into which physics slice) is the core computational primitive. For tandem foils, the inter-foil gap changes the pressure coupling length scale — nodes near the gap should attend to different sets of neighbors depending on gap width. `adaln_all=True` via `adaln_net = Linear(cond_dim → H*4)` is fully wired in `TransolverBlock.__init__` but only ever receives condition from the decoder side. Injecting gap/stagger into all three blocks means the attention heads can modulate their temperature, scale, and bias with full awareness of the tandem geometry from the very first layer.

This is a direct analog to how Vision Transformers conditioned on class tokens outperform class-agnostic backbones for class-specific recognition — the conditioning information needs to be present where the discrimination happens (in the attention), not just where the output is assembled.

**Expected impact:** 
- Primary target: p_tan (OOD tandem) — the gap/stagger scalars directly encode the geometric variation that defines the OOD split
- Secondary target: p_oodc (in-distribution tandem, also uses gap/stagger)
- Expected improvement: 5-15% on p_tan if the backbone was bottlenecked by missing geometric context. Risk of regression on p_in/p_re (single-foil, where condition is zeros) is moderate.

**Implementation complexity:** Medium. The `adaln_all=True` path and `adaln_net` already exist. The only changes needed:
1. Pass condition to the backbone TransolverBlocks (currently it is passed only to the decoder)
2. For single-foil samples, set condition = zeros(2) (gap=0, stagger=0)

**Key risk factors:**
- If the backbone blocks already implicitly learn geometry-conditional behavior through node coordinates, the explicit conditioning adds redundancy with no benefit
- Single-foil samples will see condition=zeros which may create a distribution mismatch if zeros collide with a meaningful tandem configuration
- The `adaln_net` linear map is freshly initialized — it will need the full training budget to learn useful modulation; may need warmup

**Concrete implementation steps:**
1. In `Transolver.forward()`, extract `gap_stagger = x[:, 0, 22:24]` (shape `[B, 2]`). For single-foil samples `tandem_flag=0`, set to zeros: `cond_backbone = gap_stagger * is_tandem.float().unsqueeze(1)`.
2. In the loop over `TransolverBlock`, pass `cond_backbone` as the `condition` argument (it is already accepted by the block signature when `adaln_all=True`).
3. Ensure the blocks are initialized with `adaln_all=True` and `adaln_cond_dim=2`.
4. Keep the existing output-layer conditioning unchanged.
5. Add flag: `--backbone_gap_stagger_cond True`

**Suggested flags:**
```
--backbone_gap_stagger_cond True
--adaln_all True   (should already be default)
```

**Confidence:** Moderate-strong. The mechanism is architecturally sound and the infrastructure is already in place. The gap/stagger scalars have already proven informative for the refinement head (FiLM SRF) — giving the same information to the backbone is a natural and well-motivated extension. The risk is that the backbone already captures this implicitly through spatial coordinates.

---

## Idea 2: Tandem Surface Mixup — Between-Sample Aft-Foil Geometry Augmentation

**Hypothesis:** Swapping aft-foil surface nodes between tandem samples in a batch creates new synthetic tandem configurations (mixed fore-foil geometry + different aft-foil geometry) that the model has never seen, forcing it to generalize the inter-foil interaction rather than memorize specific fore+aft pairs.

**Theoretical motivation:** The p_tan OOD failure is a geometry generalization problem. The model has learned to associate specific (fore-foil shape, aft-foil shape, gap, stagger) combinations with pressure distributions — when a new combination (NACA6416 OOD) appears, the mapping fails. Mixup for graph-structured data is non-trivial (you cannot linearly interpolate node coordinates), but physical surface swapping is geometrically valid: replacing the aft-foil nodes from sample A with the aft-foil nodes from sample B creates a new, physically plausible two-foil configuration with a different combination of geometries.

The key insight is that fore-foil → aft-foil pressure coupling is a function of (fore-foil shape, gap, stagger, aft-foil shape). By mixing these components, the model must learn the coupling mechanism rather than the specific configurations.

This is analogous to CutMix in image classification but for physical node sets: instead of cutting rectangular patches, we cut out the aft-foil mesh and paste in a different one.

**Expected impact:**
- Primary target: p_tan (OOD tandem) — directly attacks the memorization of specific foil pairs
- Secondary: p_oodc (in-distribution tandem generalization)
- Risk: p_in/p_re could regress if mixing disturbs the gradient signal for single-foil samples (guard: only mix within tandem samples, never mix single-foil)

**Implementation complexity:** Medium. Requires identifying aft-foil surface nodes (`x_aft_surf` mask), checking that both samples have the same number of aft-foil surface nodes (required for valid swap), and swapping both `x` features and `y_norm` targets.

**Key risk factors:**
- Aft-foil node count must match between paired samples. If the dataset has variable-size aft-foil meshes, valid pairs may be rare. Guard: `if _aft_a.sum() != _aft_b.sum(): skip`.
- The swapped configuration is geometrically valid but may not be physically realistic if gap/stagger are not updated to reflect the new geometry combination. Simplest approach: keep the original gap/stagger scalars, which creates a slight inconsistency.
- Mixup probability needs tuning — too high destroys the training signal for in-distribution tandem configurations.

**Concrete implementation steps:**
1. Add flags: `--tandem_surface_mixup True`, `--tandem_surface_mixup_prob 0.3`
2. In training loop, after batch assembly, identify tandem-only samples in batch.
3. For each consecutive pair of tandem samples (i, i+1), with probability `tandem_surface_mixup_prob`:
   ```python
   _aft_a = is_aft_surf[i]  # bool mask [N]
   _aft_b = is_aft_surf[i+1]
   if _aft_a.sum() == _aft_b.sum() and _aft_a.sum() > 0:
       # Swap x features for aft-foil surface nodes
       x[i][_aft_a], x[i+1][_aft_b] = x[i+1][_aft_b].clone(), x[i][_aft_a].clone()
       # Swap y_norm targets for aft-foil surface nodes
       y_norm[i][_aft_a], y_norm[i+1][_aft_b] = y_norm[i+1][_aft_b].clone(), y_norm[i][_aft_a].clone()
   ```
4. Apply only when `is_tandem[i] and is_tandem[i+1]`.

**Confidence:** Moderate. The geometric validity argument is sound, and between-sample feature mixing is well-established as a regularization technique. The CFD-specific variant (node-set swapping rather than linear interpolation) is novel and untested. The risk is that the node count guard makes valid pairs rare, reducing effective mixup frequency.

---

## Idea 3: Tandem Pressure Correction MLP with Gated Activation

**Hypothesis:** A dedicated gated MLP that applies a learned correction specifically to pressure predictions for tandem-domain nodes — where the base Transolver systematically under-predicts inter-foil pressure coupling — can recover the residual error that the main model leaves on the table.

**Theoretical motivation:** The pressure field in tandem foil configurations has a physically distinct structure compared to single-foil: the leading edge stagnation of the aft-foil interacts with the wake of the fore-foil through a pressure recovery region. The main Transolver computes a single unified representation and must simultaneously serve single-foil and tandem geometries. A gated correction pathway that is near-zero for single-foil samples (gate bias=-2.0 → sigmoid(gate) ≈ 0.12) and allowed to activate for tandem samples gives the network a separate capacity for tandem-specific pressure corrections.

The gate mechanism (soft binary switch) allows the model to learn when to apply corrections without hard if/else branching, and the zero-init ensures the correction starts as identity (no regression from adding the head).

This is the same principle that has driven multi-task learning advances: shared backbone + task-specific heads consistently outperforms shared backbone + shared head for tasks with fundamentally different output structure. Tandem pressure = different task from single-foil pressure.

**Expected impact:**
- Primary target: p_tan and p_oodc (both tandem pressure)
- p_in/p_re: should be protected by gate bias=-2.0 (near-zero activation for single-foil)
- Expected improvement: 3-8% on p_tan if the systematic residual is large and structured

**Implementation complexity:** Medium. New `TandemPressureHead` module + integration in forward pass.

**Key risk factors:**
- If the main model already captures tandem pressure correctly, the correction head learns zero and provides no benefit
- Gate near-zero init means early training sees essentially no signal through the correction pathway — requires sufficient training budget to warm up
- The correction applies to pressure only (index 2), leaving velocity channels to the main model — this is correct for the physics but means velocity errors in tandem are not addressed

**Concrete implementation steps:**
1. Add flag: `--tandem_pressure_head True`
2. Define `TandemPressureHead(nn.Module)`:
   ```python
   class TandemPressureHead(nn.Module):
       def __init__(self, n_hidden, hidden_dim=64):
           super().__init__()
           self.mlp = nn.Sequential(
               nn.Linear(n_hidden + 1, hidden_dim),  # hidden + base_pressure
               nn.GELU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.GELU(),
               nn.Linear(hidden_dim, 1),  # pressure correction only
           )
           self.gate = nn.Linear(n_hidden, 1)
           # Zero-init last layer (correction starts as identity)
           nn.init.zeros_(self.mlp[-1].weight)
           nn.init.zeros_(self.mlp[-1].bias)
           # Gate near-zero for single-foil: sigmoid(-2) ≈ 0.12
           nn.init.zeros_(self.gate.weight)
           nn.init.constant_(self.gate.bias, -2.0)
       def forward(self, hidden, base_pred):
           # base_pred: [B, N, 3], hidden: [B, N, n_hidden]
           base_p = base_pred[:, :, 2:3]  # pressure channel only
           inp = torch.cat([hidden, base_p], dim=-1)
           gate = torch.sigmoid(self.gate(hidden))  # [B, N, 1]
           corr = self.mlp(inp)  # [B, N, 1]
           return base_pred[:, :, 2:3] + gate * corr
   ```
3. In `Transolver.forward()`, apply to all nodes with `is_tandem=True`:
   ```python
   if self.tandem_pressure_head and is_tandem.any():
       p_corr = self.pressure_head(hidden, pred_out)  # [B, N, 1]
       pred_out[:, :, 2:3] = torch.where(is_tandem_expanded, p_corr, pred_out[:, :, 2:3])
   ```

**Confidence:** Moderate. The gated correction pattern is well-established (cf. mixture-of-experts, residual adapters). The physics motivation (tandem pressure is structurally different) is sound. Main uncertainty is whether the systematic residual is large enough for a separate head to capture.

---

## Idea 4: dp/dn = 0 Physics Constraint — Surface Normal Pressure Gradient Loss

**Hypothesis:** Adding an auxiliary loss term that penalizes nonzero pressure gradients in the wall-normal direction at airfoil surface nodes (derived from the Euler momentum equation at no-slip walls) will force the model to produce physically consistent pressure fields, reducing errors at surface nodes where the physical constraint is strongest.

**Theoretical motivation:** At a solid wall with no-slip boundary condition, the Euler momentum equation in the wall-normal direction reduces to:

    dp/dn = 0   (no-slip wall, inviscid approximation, zero normal velocity)

More precisely for the viscous case: `dp/dn ≈ ρ * κ * v_t^2` where κ is wall curvature and v_t is tangential velocity — but to leading order this is small and the inviscid approximation dp/dn=0 is the dominant constraint.

The model currently has no explicit incentive to enforce this constraint. If the predicted pressure field violates dp/dn=0 at surface nodes, it is physically wrong — and these are exactly the nodes where Surface MAE is evaluated.

The SAF (Signed Arc-length Field) gradient vectors at surface nodes (`x[:,:,2:4]` for foil 1, `x[:,:,6:8]` for foil 2) provide a proxy for the wall-normal direction: `n ≈ normalize(grad_SAF)`. These are already in the input features.

The auxiliary loss: for pairs of adjacent surface nodes (i, j) on the same foil, compute the finite-difference pressure gradient in their average normal direction and penalize its magnitude:

    dp_dn = (p_j - p_i) / ||x_j - x_i|| * |n_avg · (x_j - x_i) / ||x_j - x_i|||
    L_physics = mean(dp_dn^2) * weight

**Expected impact:**
- Primary target: p_in, p_oodc, p_tan — all surface pressure metrics
- Expected improvement: 2-6% if the model currently violates the constraint significantly
- Zero extra parameters — pure loss modification

**Implementation complexity:** Low-medium. ~30 lines of vectorized code. No model changes.

**Key risk factors:**
- The SAF gradient is a proxy for wall normal, not the exact normal. If it is noisy at highly curved surface regions, the loss signal will be incorrect.
- Random pair sampling (to keep cost manageable) introduces variance in the loss estimate. K_pairs=16 per surface per sample is a reasonable starting point.
- The weight must be tuned carefully — too high forces dp/dn=0 at the expense of fitting the actual pressure values (which for viscous flows is not exactly zero).

**Concrete implementation steps:**
1. Add flag: `--pde_surf_normal_weight 0.01`
2. In training loop, after computing main loss, compute physics constraint on surface nodes:
   ```python
   # surf_mask: [B, N] bool, surface nodes only
   # x: [B, N, 24], coords at [:,:,0:2], foil-1 SAF grad at [:,:,2:4]
   # pred: [B, N, 3], predicted (Ux, Uy, p)
   
   K_pairs = 16
   pde_loss = 0.0
   surf_idx = surf_mask.nonzero()  # pairs of surface nodes
   for b in range(B):
       s_idx = surf_mask[b].nonzero(as_tuple=True)[0]
       if len(s_idx) < 2:
           continue
       # Sample K random pairs of surface nodes
       pair_i = torch.randint(len(s_idx), (K_pairs,), device=device)
       pair_j = torch.randint(len(s_idx), (K_pairs,), device=device)
       valid = pair_i != pair_j
       pair_i, pair_j = pair_i[valid], pair_j[valid]
       
       i_idx, j_idx = s_idx[pair_i], s_idx[pair_j]
       xi, xj = x[b, i_idx, :2], x[b, j_idx, :2]
       dx = xj - xi
       dist = dx.norm(dim=-1, keepdim=True).clamp(min=1e-6)
       
       # Wall normal from SAF gradient
       ni = F.normalize(x[b, i_idx, 2:4], dim=-1)
       nj = F.normalize(x[b, j_idx, 2:4], dim=-1)
       avg_n = F.normalize(ni + nj, dim=-1)
       
       # Normal projection of the displacement
       normal_proj = (dx / dist * avg_n).sum(dim=-1, keepdim=True).abs()
       
       # Finite-difference pressure gradient in normal direction
       pi = pred[b, i_idx, 2:3]
       pj = pred[b, j_idx, 2:3]
       dp_dn = ((pj - pi) / dist) * normal_proj
       
       pde_loss = pde_loss + (dp_dn ** 2).mean()
   
   pde_loss = pde_loss / B
   loss = main_loss + cfg.pde_surf_normal_weight * pde_loss
   ```

**Confidence:** Moderate. The physics is correct. The question is whether the model already approximately satisfies dp/dn=0 (in which case the loss adds noise with no benefit) or whether it systematically violates it (in which case the loss provides a genuine additional constraint). Preliminary evidence from the Noam ROUND1 document suggests this is untested and the violation could be significant.

---

## Idea 5: Iterative Prediction Refinement — 2-Pass Forward with Self-Conditioning

**Hypothesis:** Running the model in two passes — first a standard forward pass, then a second forward pass where the predicted outputs from pass 1 are concatenated as additional input channels — gives the model access to its own uncertainty structure and allows it to correct systematic errors on the second pass, with no additional parameters (the same model weights are used for both passes).

**Theoretical motivation:** This is the CFD-surrogate analog of "self-conditioning" used in diffusion models (Chen et al., 2022 "Analog Bits") and iterative refinement in protein structure prediction (AlphaFold2's recycling mechanism). The key insight: the model's own first-pass predictions carry information about its uncertainty and systematic biases that is not present in the input geometry alone. Regions where the model is uncertain (e.g., the inter-foil gap pressure recovery region in tandem configurations) will show higher sensitivity to the second-pass correction.

For the tandem foil problem specifically: the first-pass pressure field will have systematic errors in the inter-foil region (the primary source of p_tan failures). The second pass, conditioned on this first-pass prediction, can learn a correction function `f(x, p_pass1) → p_corrected` that is specialized for fixing these systematic inter-foil errors.

**Why same weights (not a separate corrector network)?** Shared weights force the second pass to be consistent with the first — the model must learn a prediction function that, when applied to (input + own output), is self-consistent. This is a strong regularizer that prevents the corrector from memorizing training-set biases. It is also zero additional parameters.

AlphaFold2's recycling (3 passes, shared weights, structure → structure) is the most successful example of this principle in scientific ML.

**Expected impact:**
- Primary target: p_tan (iterative refinement is most useful when there is a structured residual error)
- All surface metrics: the correction pathway is applied to all surface predictions
- Cost: 2x inference time, 2x training memory for activations; training time increases ~1.8x
- Expected improvement: 5-15% on p_tan if first-pass errors are structured and correctable

**Implementation complexity:** Medium-high. Requires:
1. A second forward pass in the training loop
2. Concatenating the 3-channel output from pass 1 to the 24-dim input (input becomes 27-dim) for pass 2
3. The model must handle variable input_dim (24 or 27)
4. Gradients flow through both passes — memory cost is real

**Key risk factors:**
- 2x memory for activations during training. With 96GB VRAM this is likely manageable but requires verification.
- If first-pass output has high variance (the model is unstable), the second-pass input is noisy and the refinement is unreliable. Use EMA model for first-pass output at inference (already available).
- The input projection `Linear(24, n_hidden)` must be extended to `Linear(27, n_hidden)`. This changes the model architecture slightly — a new Linear layer or a separate "correction projection" that adds to the main projection.
- During early training when pass-1 predictions are garbage, pass-2 conditioning is counterproductive. Warmup strategy: start pass-2 conditioning at epoch 60 (when pass-1 predictions are meaningful).

**Concrete implementation steps:**
1. Add flags: `--iterative_refinement True`, `--iterative_warmup_epoch 60`, `--iterative_stop_grad False`
2. Modify `Transolver.__init__` to add an optional `correction_proj = nn.Linear(out_dim, n_hidden, bias=False)` (zero-init) that adds to the main input projection.
3. In `Transolver.forward()`:
   ```python
   if self.iterative_refinement and epoch >= self.iterative_warmup_epoch:
       with torch.no_grad() if self.iterative_stop_grad else contextlib.nullcontext():
           pass1_out = self._forward_inner(x, fx, ...)  # standard forward
       
       # Concatenate pass-1 predictions as conditioning signal
       pass1_pred = pass1_out["preds"]  # [B, N, 3]
       # Add correction signal via separate projection
       fx_correction = self.correction_proj(pass1_pred)  # [B, N, n_hidden]
       pass2_out = self._forward_inner(x, fx + fx_correction, ...)
       return pass2_out  # train on pass-2 output
   else:
       return self._forward_inner(x, fx, ...)
   ```
4. At inference: always use pass-2 (after warmup epoch).
5. Note: `iterative_stop_grad=True` (detach pass-1 gradients) is cheaper but slightly less accurate. Try both.

**Confidence:** Moderate. Strong theoretical basis from AlphaFold2 recycling and diffusion self-conditioning. The mechanism is genuinely novel in the CFD surrogate setting. Main uncertainty: whether the pass-1 predictions have enough structured error for pass-2 to exploit. The 2x compute cost is the main practical risk.

---

## Summary Table

| Rank | Idea | Core mechanism | Primary target | Code change | Evidence |
|------|------|---------------|----------------|-------------|---------|
| 1 | Backbone-wide AdaLN-All conditioning | Gap/stagger injected into all 3 TransolverBlocks | p_tan (-10%) | Medium | Strong analog from vision transformers |
| 2 | Tandem Surface Mixup | Between-sample aft-foil node swapping | p_tan (-8%) | Medium | Moderate, novel for graph/mesh data |
| 3 | Tandem Pressure Correction MLP | Gated pressure-specific pathway for tandem nodes | p_tan, p_oodc (-5%) | Medium | Moderate, multi-task learning principle |
| 4 | dp/dn=0 Physics Loss | Surface normal pressure gradient auxiliary loss | p_in, p_tan (-3%) | Low | Moderate, physics-motivated |
| 5 | Iterative 2-Pass Refinement | Self-conditioning via pass-1 output (AlphaFold recycling) | p_tan (-10%) | High | Moderate-strong from protein ML |

---

## Assignment Recommendations

These ideas are ready for assignment to fern, nezuko, and askeladd once their current PRs (#2161, #2163, #2162) complete.

**Priority order:**
1. **Idea 1 (Backbone-Wide AdaLN-All)** — highest expected impact, infrastructure already exists, clean test of backbone vs. decoder-only conditioning
2. **Idea 5 (Iterative 2-Pass)** — boldest architectural change, most novel, AlphaFold recycling is the strongest prior in scientific ML
3. **Idea 4 (dp/dn=0 Physics Loss)** — zero parameters, pure loss modification, fast to implement and test
4. **Idea 2 (Tandem Surface Mixup)** — geometric diversity augmentation, medium complexity
5. **Idea 3 (Tandem Pressure Correction MLP)** — well-motivated but more conservative than ideas 1 and 5

## Notes on Interaction with In-Flight Experiments

- **askeladd #2162 (cross-DSDF features)**: Idea 1 (backbone conditioning) is complementary — DSDF features provide local geometry, gap/stagger provides global tandem structure. Can be combined after both are evaluated.
- **fern #2161 (fore-foil FiLM SRF)**: Idea 3 (tandem pressure correction) is downstream of SRF — both improve tandem pressure, but through different mechanisms (SRF corrects after backbone, gated MLP corrects in a separate pathway).
- **nezuko #2163 (differential LR)**: Idea 5 (iterative refinement) changes the architecture — differential LR is orthogonal and can be applied on top.
- **dp/dn=0 physics loss (Idea 4)**: If edward #2158 (asymmetric-pcgrad) changes the gradient balancing, the physics loss weight may need retuning when combined. Test independently first.
