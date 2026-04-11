# SENPAI Research Ideas — 2026-04-06 Round 14

Generated after reviewing full experiment history (~230 PRs) and current research state.

**Current single-model baseline (PR #2184, DCT Freq Loss w=0.05, 2-seed):**
p_in=13.205, p_oodc=7.816, p_tan=28.502, p_re=6.453

**Merge thresholds:** p_tan < 28.50, p_oodc < 7.82, p_in < 13.21, p_re < 6.45

**Active WIP (do NOT duplicate):**
- fern #2210 (arc-length surface loss reweighting)
- alphonse #2211 (surface dp/ds gradient loss)
- thorfinn #2209 (register tokens in physics-attention)
- nezuko #2205 (NOBLE nonlinear low-rank FFN branches)
- frieren #2199 (spectral conditioning of attention SCA)
- edward #2207 (TE coordinate frame, re-running)
- tanjiro #2197 (geometry-adaptive curvature loss)
- askeladd #2212 (analytical Cp delta for SRF correction)

---

## Ranked Hypotheses

### 1. wake-deficit-feature — Fore-Foil Wake Deficit as Explicit Mesh Node Feature

**Priority: HIGH. ~25 LoC. Expected -2 to -5% p_tan. MEDIUM risk.**

**What it is:** For each mesh node, compute the gap-normalized displacement vector from the fore-foil trailing edge: `(dx/gap, dy/gap)` where `(dx, dy)` is the displacement from the fore-foil trailing edge to the node. Append this 2D vector to the 24-dim input `x` for all nodes in tandem samples (zero-pad for single-foil samples), bringing the input dimension from 24 to 26.

**Why it helps p_tan:** The aft-foil is immersed in the pressure-depleted wake of the fore-foil. The fore-foil trailing edge is the origin of the strongest upstream influence on aft-foil surface pressure. Currently, the model must infer this wake-relative position from the combination of raw (x,y) coordinates and DSDF channels — a multi-step inference the network has to do implicitly. Making the wake-relative position explicit removes this indirection entirely. The gap normalization makes the feature invariant to absolute gap size (critical for OOD generalization to NACA6416 tandem configurations). This feature has been in the unassigned hypothesis queue since Round 8. It is mechanistically distinct from the gap/stagger spatial bias (PR #2130, -3.0% p_tan), which encodes global geometry scalars into the slice routing; this feature encodes per-node spatial relationship to the primary wake source.

**Implementation — exactly where to add code:**

```python
# Add config flag:
# wake_deficit_feature: bool = False

# In preprocessing (before standardization, just after building raw x):
# (fore-foil trailing edge position can be recovered from fore-DSDF gradient minimum
#  or from the geometry data — raw x[:, :, 6:8] stores fore-foil DSDF channels,
#  and the trailing edge is the zero-crossing closest to chord=1.0)
#
# Simpler proxy: use the mean position of fore-foil surface nodes (is_tandem & foil_id==6)
# as the "wake source" center, then compute gap-normalized displacement for all nodes.
#
# After building is_tandem and gap from x[:, 0, 22]:
if cfg.wake_deficit_feature:
    gap = x[:, 0, 22:23]                          # [B, 1]
    gap_safe = gap.clamp(min=1e-3)                 # avoid div by zero for single-foil
    # fore-foil trailing edge proxy: mean of nodes where dsdf_fore ≈ 0 and x > 0.5
    # Use a fixed reference point: (1.0, 0.0) in foil-1 coordinates, gap-normalized
    # This requires knowing the raw (unnormalized) xy coordinates
    # dx, dy are computed from raw_xy (stored before standardization) to raw TE (1.0, 0.0)
    raw_xy = x_raw[:, :, 0:2]                     # [B, N, 2] before standardization
    te_ref = torch.zeros(B, 1, 2, device=x.device)
    te_ref[:, 0, 0] = 1.0                          # fore-foil TE at (1.0, 0.0) approximately
    disp = (raw_xy - te_ref) / gap_safe.unsqueeze(-1)   # [B, N, 2] gap-normalized
    is_tandem_flag = (gap.squeeze(-1) > 0.01)     # [B]
    wake_feat = disp * is_tandem_flag.unsqueeze(-1).unsqueeze(-1).float()  # zero for single-foil
    x = torch.cat([x, wake_feat], dim=-1)          # [B, N, 26]
```

**New flag:** `--wake_deficit_feature`

**Implementation notes:**
- The input projection in TransolverBlock is `nn.Linear(in_channels, n_hidden)` and is dynamically sized — appending 2 dims to `x` automatically propagates through.
- The fore-foil TE at (1.0, 0.0) is approximate; a more accurate proxy is the minimum of `x[:, :, 6]` (fore-DSDF channel) for fore-surface nodes. The simple fixed-point proxy is sufficient for a first test.
- The gap normalization is critical — without it, this feature partially duplicates raw (x,y) coordinates and adds no information.
- Zero-pad single-foil samples so gradient flow is identical for both domains.

**Suggested experiment:**
```
--wake_deficit_feature
```
Run 2 seeds (42 and 73) for initial validation. Target: p_tan < 28.0.

**Confidence:** Strong — direct physical mechanism (wake-relative position), mechanistically distinct from all prior geometry encoding experiments, addresses the core OOD failure mode (model lacks explicit wake provenance per node). The gap normalization is the key innovation over naive coordinate features.

---

### 2. deep-supervision — Deep Supervision via fx_deep Intermediate Representation

**Priority: HIGH. ~15 LoC. Expected -1 to -3% p_tan. LOW risk.**

**What it is:** Add a small auxiliary prediction head on `fx_deep` (the intermediate feature map saved just before the final TransolverBlock, exposed as `model_out["hidden"]`), supervised at 10-15% weight on surface nodes only. This auxiliary loss is active only during training; inference uses only the final layer predictions. The auxiliary head is a 2-layer MLP matching the surface refinement head architecture but smaller.

**Why it helps p_tan:** The 3-block Transolver is relatively shallow. The gradient signal from the final surface loss must propagate back through Block 3 before reaching Blocks 1-2, where the key physics attention routing happens. With only 3 blocks, this isn't a severe vanishing gradient problem — but the auxiliary signal directly at `fx_deep` gives Block 1 and Block 2 a supervision signal that doesn't depend on Block 3 being correct. This should improve the quality of the intermediate physics attention in the backbone, which is precisely where the gap/stagger spatial bias operates. Deep supervision has been discussed in the Apr 3 research ideas file but has never been assigned to any student in any PR.

**Key technical detail:** `fx_deep` is already computed in the forward pass and stored in `model_out["hidden"]`. No new computations are needed inside the model. The auxiliary head sits entirely in the training loop.

**Implementation:**

```python
# After model forward pass, in train loop:
if cfg.deep_supervision and "hidden" in model_out:
    fx_deep = model_out["hidden"]                    # [B, N, n_hidden]
    if not hasattr(model, '_deep_sup_head'):
        n_h = fx_deep.shape[-1]
        model._deep_sup_head = nn.Sequential(
            nn.Linear(n_h, n_h // 2),
            nn.GELU(),
            nn.Linear(n_h // 2, 3)                  # predict (Ux, Uy, p)
        ).to(fx_deep.device)
    aux_pred = model._deep_sup_head(fx_deep)         # [B, N, 3]
    aux_loss = F.l1_loss(
        aux_pred[surface_mask],
        targets[surface_mask]
    )
    loss = loss + cfg.deep_supervision_weight * aux_loss
```

**New flags:** `--deep_supervision`, `--deep_supervision_weight 0.12` (sweep: {0.08, 0.12, 0.20})

**Implementation notes:**
- The auxiliary head MUST be added to the optimizer parameter group, otherwise its parameters receive no gradient updates. The easiest way is to add it to `model` as an attribute immediately before the optimizer is constructed, then `model.parameters()` will include it automatically.
- Do NOT apply the auxiliary head at inference time. Gate it behind `model.training`.
- The auxiliary head should NOT use asinh transform on its output — it's predicting raw targets, same as the main head.
- Weight 0.12 is a principled starting point: enough signal to influence Block 2 without dominating the main surface loss.

**Suggested experiment:**
```
--deep_supervision --deep_supervision_weight 0.12
```
Run 2 seeds. If directional improvement, sweep {0.08, 0.20}.

**Confidence:** Moderate-strong. Deep supervision reliably helps in dense prediction tasks (nnUNet, HED). The transfer to mesh-based CFD surrogate with 3-block Transolver is plausible and the implementation cost is very low. The primary risk is the auxiliary head being too large or too small relative to the main head.

---

### 3. pirate-residuals — PirateNets Adaptive Residual Connections

**Priority: HIGH. ~10 LoC. Expected -1 to -3% p_tan. LOW risk.**

**What it is:** Replace the standard residual connection `h = h + f(h)` in each TransolverBlock with the PirateNets gated residual `h = (1 - tanh(s_l)) * h + tanh(s_l) * f(h)`, where `s_l` is a learnable scalar per block, initialized to 0. At initialization, `tanh(0) = 0`, so `h = h * 1 + 0 = h` — the residual is a pure skip at epoch 0. As training proceeds, `s_l` learns to open the gate toward the block's contribution. This was explicitly requested by the human research team in issue #1926 and has been in the unassigned hypothesis queue since Round 9.

**Why it helps p_tan:** The aft-foil flow is inherently more complex than the single-foil flow (pressure deficit from wake, inter-foil channel effects). The standard equal-weight residual may dampen the Transolver block's contribution in tandem cases where the block should contribute more. Learnable gate magnitude per block allows the model to adaptively control the strength of the non-linear transformation vs skip connection on a per-block basis, which may be particularly valuable for the OOD tandem configurations where the standard residual schedule is suboptimal.

**PirateNets reference:** Cho et al., "PirateNets: Physics-Informed Deep Learning with Residual Adaptive Networks", 2024. Originally applied to PINN residual connections for PDE solving, but the mechanism generalizes to any residual network.

**Implementation:**

```python
# In TransolverBlock.__init__, add one learnable scalar per block:
self.residual_gate = nn.Parameter(torch.zeros(1))  # init s_l = 0

# In TransolverBlock.forward, replace:
#   x = x + self.ffn(x)
# with:
gate = torch.tanh(self.residual_gate)
x = (1.0 - gate) * x + gate * self.ffn(x)

# Similarly for the attention residual:
gate_attn = torch.tanh(self.residual_gate_attn)  # second scalar
x = (1.0 - gate_attn) * x + gate_attn * self.attention(x)
```

**New flag:** `--pirate_residuals`

**Implementation notes:**
- Zero-init is mandatory. If `s_l` starts nonzero, the first few batches see a different gradient geometry than the pre-trained residual expects, which can destabilize the EMA warmup.
- Add `residual_gate` and `residual_gate_attn` to the model's parameter list — they are very small (6 scalars for 3 blocks × 2 residuals), so they have negligible parameter count impact.
- The gate applies equally to all nodes in the batch, so it does NOT interact with PCGrad in a way that should cause instability.
- Start with gates on the FFN residuals only (not the attention residual) for a simpler first test.

**Suggested experiment:**
```
--pirate_residuals
```
Run 2 seeds. The change is architecturally conservative (zero-init guarantees identical behavior at epoch 0), so the risk of a catastrophic failure is very low.

**Confidence:** Moderate. PirateNets showed clear improvements in PINN settings. The transfer to Transolver attention blocks is architecturally analogous (both are residual networks with learned non-linear transformations). The primary uncertainty is whether the model has already learned to use the full residual capacity and the gate will stay near zero.

---

### 4. mhc-residuals — Manifold Hyper-Connection (mHC) Residuals

**Priority: MEDIUM-HIGH. ~15 LoC. Expected -1 to -3% p_tan. LOW-MEDIUM risk.**

**What it is:** Replace each TransolverBlock's single scalar residual with a learnable `(alpha, beta)` pair per block, implementing `h = alpha_l * h + beta_l * f(h)`, with alpha and beta initialized to (1.0, 1.0). Unlike PirateNets (which uses a single gate), mHC independently scales the skip path and the transformation path. This was also explicitly requested by the human research team in issue #1926 under the "mHC" label and has been in the unassigned hypothesis queue since Round 9.

**Why it helps p_tan:** The mHC formulation provides strictly more representational freedom than PirateNets: it can amplify the skip path (alpha > 1), attenuate the transformation (beta < 1), or vice versa. For the tandem aft-foil case, where the residual network may need to boost the non-linear component to model the wake interaction, beta > 1 at convergence is physically plausible. The two-parameter version can also implement effective layer skip (beta → 0) or effective layer duplication (alpha + beta ≈ 2.0), giving the optimizer a broader search space.

**Reference:** Zhu et al., "Hyper-Connections", arXiv 2409.19606. The mHC variant uses per-layer (alpha, beta) pairs to implement a learned linear combination of the residual paths.

**Implementation:**

```python
# In TransolverBlock.__init__:
self.hc_alpha = nn.Parameter(torch.ones(1))   # skip-path scale, init 1.0
self.hc_beta  = nn.Parameter(torch.ones(1))   # transform-path scale, init 1.0

# In TransolverBlock.forward, replace:
#   x = x + self.ffn(x)
# with:
x = self.hc_alpha * x + self.hc_beta * self.ffn(x)
```

**New flag:** `--mhc_residuals`

**Implementation notes:**
- Initialize alpha=1.0, beta=1.0 so the model starts at the same point as the standard residual. Do NOT use zero-init for beta — the whole-block contribution starts at standard strength.
- Unlike PirateNets, mHC does not prevent large values at initialization, so add a soft constraint if needed: use `F.softplus(self.hc_alpha)` and `F.softplus(self.hc_beta)` to keep them positive during training.
- Apply to FFN residuals only in the first experiment. Adding to attention residuals is a follow-up.
- The lion optimizer's sign-based updates make (alpha, beta) optimization non-trivial since Lion quantizes gradients to ±1. If mHC converges poorly, try Adam for these 6 parameters only (mixed optimizer) — but this adds complexity, so start with all-Lion.

**Suggested experiment:**
```
--mhc_residuals
```
Run 2 seeds. Compare to pirate-residuals if both are assigned in the same round.

**Confidence:** Moderate. Hyper-connections improved performance on LLM pretraining (Zhu et al.) and the mechanism is theoretically sound for dense prediction. The primary risk is that Lion's sign-based updates are suboptimal for learning (alpha, beta) and the model may converge to (1.0, 1.0) trivially.

---

### 5. tandem-feature-cross — Input-Level Tandem Geometry Gating

**Priority: MEDIUM. ~25 LoC. Expected -1 to -3% p_tan. MEDIUM risk.**

**What it is:** Add a learned sigmoid gate conditioned on tandem-specific scalars (gap, stagger, Re) that multiplicatively modulates the input features for all nodes in tandem samples. A small 3-layer MLP takes `(gap, stagger, Re)` as input and outputs a gate vector of dimension 24 (matching the full input `x`). This gate is applied element-wise: `x_gated = x * sigmoid(MLP(gap, stagger, Re))`. Single-foil samples use a gate of all-ones (identity). This was in the unassigned hypothesis queue since Round 11.

**Why it helps p_tan:** The model currently processes all 24 input features equally regardless of the tandem configuration. For high-gap configurations, the fore-foil DSDF features (indices 6:10) are more distant and less informative; for low-gap, they dominate. The learned gate allows the model to dynamically downweight or upweight input channels based on the specific tandem geometry, improving generalization to the OOD NACA6416 configurations (p_tan is systematically 2.16x worse than p_in). This is an input-level version of the FiLM conditioning that failed catastrophically when applied to the spatial_bias (PR #2104 with gap/stagger FiLM, +41.6% p_oodc regression) — but that failure was due to changing the condition dynamically during training, not due to the gating mechanism per se. The key difference here: the gate modulates the INPUT features (static per sample), not the routing logic.

**Implementation:**

```python
# Add config flag: tandem_feature_cross: bool = False
# Add config: tandem_cross_hidden: int = 16

# In model __init__:
if cfg.tandem_feature_cross:
    self.tandem_cross_gate = nn.Sequential(
        nn.Linear(3, cfg.tandem_cross_hidden),
        nn.GELU(),
        nn.Linear(cfg.tandem_cross_hidden, cfg.tandem_cross_hidden),
        nn.GELU(),
        nn.Linear(cfg.tandem_cross_hidden, 24),   # match input dim
    )
    # Zero-init the last layer so gate starts at sigmoid(0) = 0.5 (attenuating all features equally)
    nn.init.zeros_(self.tandem_cross_gate[-1].weight)
    nn.init.zeros_(self.tandem_cross_gate[-1].bias)

# In forward / preprocessing, after standardization:
if cfg.tandem_feature_cross:
    gap_stagger_re = x[:, 0, [22, 23, re_idx]]     # [B, 3] — global scalars from first node
    raw_gate = self.tandem_cross_gate(gap_stagger_re)   # [B, 24]
    gate = torch.sigmoid(raw_gate)                  # [B, 24], starts at 0.5
    is_tandem = (x[:, 0, 22].abs() > 0.01)         # [B]
    gate = torch.where(
        is_tandem.unsqueeze(-1),
        gate,
        torch.ones_like(gate)                       # identity gate for single-foil
    )
    x = x * gate.unsqueeze(1)                       # [B, N, 24], broadcast over nodes
```

**New flag:** `--tandem_feature_cross`

**Implementation notes:**
- Zero-init of the final linear layer is critical. If the initial gate is far from 1.0, the feature distribution at epoch 0 is very different from the pre-trained baseline and EMA will take longer to stabilize.
- Re index in the 24-dim input: should be identified from `prepare_multi.py`. From the dataset documentation, Reynolds number is stored at a specific index in the global scalar block (approximately index 20-21). Verify this before committing.
- The sigmoid output starts at 0.5 with zero-init — this halves all input features at epoch 0. Use `2 * sigmoid(raw_gate)` instead to start at 1.0 (identity) with zero-init, which is safer.

**Suggested experiment:**
```
--tandem_feature_cross
```
Run 2 seeds. If p_tan improves, sweep `tandem_cross_hidden` {8, 16, 32}.

**Confidence:** Moderate. The mechanism is sound (input-level feature gating conditioned on geometry), but the risk of interaction with the existing gap_stagger_spatial_bias (which also conditions on gap/stagger) is real. The two mechanisms operate at different levels (input features vs slice routing), so they should be complementary rather than redundant — but this is speculative.

---

### 6. fore-srf-additive-skip — Additive Fore-Foil Surface Skip into AftFoilRefinementHead

**Priority: MEDIUM. ~20 LoC. Expected -1 to -3% p_tan. MEDIUM risk.**

**What it is:** Mean-pool the hidden states of fore-foil surface nodes (those with boundary ID=6) and add the resulting vector as a residual input to the AftFoilRefinementHead (AftSRF) before its first linear layer. This gives the aft-foil correction head direct access to a compressed summary of the fore-foil surface representation at the point where correction decisions are made. This is DISTINCT from the failed PR #2202 (askeladd), which replaced the standard SRF head with a cross-attention mechanism. Here, the standard SRF head is kept intact and the fore-foil summary is added additively.

**Why it helps p_tan:** The aft-foil pressure distribution is physically determined by the fore-foil circulation and wake. The AftSRF head currently sees only the aft-foil node's own hidden state (from the trunk) and the gap/stagger context (from PR #2127). It does not have access to what the fore-foil surface looks like in hidden-state space. A mean-pooled fore-foil surface summary is a compact, permutation-invariant representation of the upstream influence. Zero-init the residual projection so the AftSRF starts identically to the current baseline.

**Why it differs from #2202:** PR #2202 replaced `h_aft = standard_srf(h_aft)` with `h_aft = cross_attn(h_aft, h_fore)`. This caused optimization instability because the entire SRF pathway changed. The additive skip approach keeps `h_aft = standard_srf(h_aft) + W_proj * mean_pool(h_fore)`, where `W_proj` is zero-initialized. At epoch 0 the behavior is identical to the current baseline; `W_proj` learns to incorporate fore-foil information only as it improves validation loss.

**Implementation:**

```python
# In AftFoilRefinementHead.__init__:
if cfg.fore_srf_additive_skip:
    self.fore_skip_proj = nn.Linear(n_hidden, n_hidden)
    nn.init.zeros_(self.fore_skip_proj.weight)
    nn.init.zeros_(self.fore_skip_proj.bias)

# In forward, before the SRF correction MLP:
if cfg.fore_srf_additive_skip and fore_surface_mask is not None:
    # h_trunk: [B, N, n_hidden] — full trunk hidden states
    # fore_surface_mask: [B, N] boolean mask for fore-foil surface nodes
    fore_h = h_trunk * fore_surface_mask.unsqueeze(-1).float()  # [B, N, n_hidden]
    fore_count = fore_surface_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # [B,1,1]
    fore_mean = fore_h.sum(dim=1, keepdim=True) / fore_count    # [B, 1, n_hidden]
    fore_skip = self.fore_skip_proj(fore_mean)                   # [B, 1, n_hidden]
    # Add to aft-foil surface nodes only (the AftSRF input)
    aft_input = aft_input + fore_skip                            # [B, N_aft, n_hidden]
```

**New flag:** `--fore_srf_additive_skip`

**Implementation notes:**
- The fore-surface mask uses the same proxy detection as the aft_foil_srf (DSDF magnitude for tandem samples). Re-use the existing `_aft_foil_mask` infrastructure — the complement (is_tandem surface nodes NOT in aft mask) approximates the fore-foil surface.
- Zero-init is non-negotiable here. Non-zero init would immediately perturb the AftSRF from its current optimized state.
- The mean-pool is permutation-invariant, which is important since fore-foil surface node ordering may vary across samples.
- Monitor that the fore_skip_proj weight norm grows during training. If it stays near zero after 80 epochs, the AftSRF is not finding value in the fore-foil summary — close the PR.

**Suggested experiment:**
```
--fore_srf_additive_skip
```
Run 2 seeds. This is a conservative extension of the existing AftSRF infrastructure — low disruption risk.

**Confidence:** Moderate. The physical motivation is clear. The additive+zero-init design avoids the optimization instability that killed #2202. The primary uncertainty is whether the trunk's hidden states already encode enough fore-foil context that this is redundant with existing information pathways.

---

## Summary Table

| Rank | Slug | Code size | Risk | Expected p_tan gain | Priority |
|------|------|-----------|------|--------------------:|----------|
| 1 | wake-deficit-feature | ~25 LoC | MEDIUM | -2 to -5% | HIGH |
| 2 | deep-supervision | ~15 LoC | LOW | -1 to -3% | HIGH |
| 3 | pirate-residuals | ~10 LoC | LOW | -1 to -3% | HIGH |
| 4 | mhc-residuals | ~15 LoC | LOW-MEDIUM | -1 to -3% | MEDIUM-HIGH |
| 5 | tandem-feature-cross | ~25 LoC | MEDIUM | -1 to -3% | MEDIUM |
| 6 | fore-srf-additive-skip | ~20 LoC | MEDIUM | -1 to -3% | MEDIUM |

**Priority rationale:**
- wake-deficit-feature is #1 because it is the only idea in this batch that explicitly encodes the physical mechanism driving p_tan failure (wake-relative position of each node to the fore-foil TE). Every other mechanism is more indirect.
- deep-supervision and pirate-residuals are #2/#3 because they carry very low implementation risk (zero-init guarantees safe initialization), are conceptually clean single-variable tests, and have strong prior-work evidence from adjacent domains.
- mhc-residuals is ranked below pirate-residuals because the 2-parameter version has higher sensitivity to Lion optimizer dynamics.
- tandem-feature-cross and fore-srf-additive-skip are ranked lower because they require more careful implementation (gate initialization, forward pass surgery) and their interaction with existing mechanisms (GSB, AftSRF) is less well-characterized.
