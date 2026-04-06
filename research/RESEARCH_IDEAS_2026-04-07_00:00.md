# SENPAI Research Ideas — Round 15 (2026-04-07 00:00 UTC)

Generated after reviewing all 1,824 experiment PRs (133 merged, 1,544 ran-not-merged, 147 never-ran), 8 active WIP students, and comprehensive dead-ends analysis.

**Current single-model baseline (PR #2207, TE Coordinate Frame):**
p_in=12.490, p_oodc=7.618, p_tan=28.521, p_re=6.411

**Target:** p_tan < 28.0 (delta = -0.52 from baseline)

**Single model p_tan=28.52 already beats 16-seed ensemble (29.1). The model is in elite territory. Next improvement requires qualitatively new information or training signal.**

**8 WIP students — DO NOT DUPLICATE:**
- fern #2210 (arc-length surface loss reweighting)
- nezuko #2217 (fore-SRF skip: zero-init fore-foil mean hidden into AftSRF)
- alphonse #2211 (surface pressure gradient loss dp/ds)
- thorfinn #2216 (GeoTransolver GALE cross-attention)
- tanjiro #2197 (geometry-adaptive curvature loss weighting)
- askeladd #2212 (analytical Cp delta thin-airfoil SRF)
- frieren #2213 (wake deficit feature: gap-normalized fore-TE offset)
- edward #2214 (deep supervision on fx_deep intermediate rep)

**Dead ends — DO NOT REVISIT:** NOBLE/CosNet (#2205), register tokens (#2209), Muon optimizer (#2203), Ada-Temp/Rep-Slice (#2206), SCA (#2199), iterative SRF (#2208, #2165), fore-aft crossattn as REPLACEMENT (#2202), Laplacian PE (#2190).

---

## Idea 1: mHC Learnable Residual Mixing (Carry-Forward from Round 13)

**Slug:** `mhc-residuals`
**Priority: HIGH — human team request (issue #1926), 3+ rounds queued, ~15 LoC**

**What it is:** Replace each fixed residual connection `x + F(x)` in TransolverBlock with `alpha * x + beta * F(x)`, where `alpha` and `beta` are scalar `nn.Parameter`s per sublayer, initialized to (1, 1). This gives the optimizer freedom to independently scale the skip path and the transformation path for each block.

**Why it might help here:**
The baseline uses fixed skip weight. For OOD NACA6416 tandem samples, the optimal blend of identity (skip) and transformation (block output) may differ from in-distribution. Block 1 handles geometry encoding — high skip weight appropriate. Block 3 handles global physics interactions — the skip weight may need to be lower to allow deeper transformation. Fixed (1,1) prevents this calibration. Learnable alpha/beta is strictly more expressive with a safe initialization.

The key theoretical backing: in deep networks, spectral bias causes early layers to overfit low-frequency information. Learnable residual weights allow the optimizer to dynamically control the effective depth for each block — high beta/low alpha means "trust the block," high alpha/low beta means "mostly skip." This is especially useful for OOD generalization where the depth requirement may differ from in-distribution.

**Implementation (~15 LoC in TransolverBlock):**
```python
# In TransolverBlock.__init__:
self.attn_alpha = nn.Parameter(torch.ones(1))   # skip weight for attn sublayer
self.attn_beta = nn.Parameter(torch.ones(1))    # transform weight for attn sublayer
self.mlp_alpha = nn.Parameter(torch.ones(1))    # skip weight for MLP sublayer
self.mlp_beta = nn.Parameter(torch.ones(1))     # transform weight for MLP sublayer

# In TransolverBlock.forward, replace:
#   x = x + self.drop_path(self.attn(norm_x))
#   x = x + self.drop_path(self.fn(norm_x))
# with:
x = self.attn_alpha * x + self.attn_beta * self.drop_path(self.attn(norm_x))
x = self.mlp_alpha * x + self.mlp_beta * self.drop_path(self.fn(norm_x))
```

4 parameters per block x 3 blocks = 12 total new parameters. Negligible impact on parameter count.

**Note on regularization:** No regularization on alpha/beta needed — init (1,1) is neutral. Monitor W&B: if alpha or beta → 0 (one path killed), that's valid but worth flagging. If loss diverges at epoch 1-3, the init at (1,1) may have disturbed EMA — in that case, try init (1, 0.5) for conservative start.

**Flag:** `--mhc_residuals` (boolean). Default False.

**Reference:** Hyper-Connection formulation, DeepMind transformer residual work. Related: PaLM learnable shortcut scaling. Concept explicitly requested by human team in issue #1926.

**Suggested experiment:**
```bash
python train.py ... [baseline flags] --mhc_residuals
# seeds: {42, 73}
# wandb_group: mhc-residuals
```

**Confidence:** Medium. Human-team requested, safe initialization, strictly more expressive than baseline. Main risk: degenerate solutions where one path collapses. Monitor W&B alpha/beta histograms.

---

## Idea 2: Additive Fore-Aft Cross-Attention in AftFoilRefinementHead (Targeted Retry of #2202)

**Slug:** `additive-fore-aft-crossattn-srf`
**Priority: HIGH — targeted fix of a well-motivated direction that failed due to instability**

**What it is:** Add a cross-attention sublayer to the `AftFoilRefinementHead` (AftSRF) that reads from fore-foil backbone hidden states, ADDITIVELY on top of the existing SRF MLP. Zero-init the output projection so the first forward pass is exactly baseline. The cross-attention gradually learns to read the fore-foil pressure state and apply a correction to the aft-foil prediction.

**Why it might help here:**
PR #2202 (fore-aft cross-attention as replacement for SRF MLP) failed: p_tan +2.1%, severe seed variance (s42=28.3, s73=29.9, gap=1.6). The advisor noted "Additive approach worth revisiting." The replacement failed because optimization was unstable — the cross-attention had to simultaneously replace AND learn. The additive variant avoids this: the MLP continues to produce its full prediction; the cross-attention only needs to learn a residual correction, which is a much easier optimization problem.

Physical motivation: the aft-foil pressure loading is directly driven by the fore-foil wake. If the fore-foil prediction has systematic error (especially for OOD NACA6416), the AftSRF currently has no way to account for it. Cross-attention from aft nodes to fore-foil hidden states gives it explicit access to the predicted fore-foil state.

This is the most targeted way to inject fore-foil information into the aft-foil refinement without changing the backbone.

**Implementation (~30 LoC):**
```python
class AftFoilRefinementHead(nn.Module):
    def __init__(self, n_hidden, n_layers=3, hidden_dim=192, additive_crossattn=False):
        ...
        self.additive_crossattn = additive_crossattn
        if additive_crossattn:
            self.fore_cross_attn = nn.MultiheadAttention(
                n_hidden, num_heads=4, batch_first=True, dropout=0.0
            )
            self.fore_cross_norm = nn.LayerNorm(n_hidden)
            # CRITICAL: zero-init out_proj for safe start
            nn.init.zeros_(self.fore_cross_attn.out_proj.weight)
            nn.init.zeros_(self.fore_cross_attn.out_proj.bias)
    
    def forward(self, x_aft, x_fore_hidden=None):
        # x_aft: [B, N_aft, n_hidden] — aft foil surface nodes
        # x_fore_hidden: [B, N_fore, n_hidden] — fore foil backbone output
        
        # Base SRF correction (unchanged)
        correction = self.mlp(x_aft)  # existing MLP path
        
        # Additive cross-attention correction (new, starts at zero)
        if self.additive_crossattn and x_fore_hidden is not None:
            x_aft_norm = self.fore_cross_norm(x_aft)
            cross_out, _ = self.fore_cross_attn(x_aft_norm, x_fore_hidden, x_fore_hidden)
            correction = correction + cross_out  # additive!
        
        return x_aft + correction
```

Fore-foil hidden states are available at backbone output: `fx[fore_surface_mask]` after the 3 TransolverBlocks.

**Flag:** `--additive_fore_aft_crossattn` (boolean). Default False.

**Watch for:** Seed variance on p_tan. If s42 and s73 differ by >1.0, the mechanism is still unstable despite zero-init — in that case try reducing num_heads from 4 to 2.

**Confidence:** Medium-high. The zero-init specifically addresses the instability that killed #2202. The physical motivation is strong. The mechanism is well-validated in other settings (encoder-decoder transformer cross-attention).

---

## Idea 3: Slice Diversity Regularization

**Slug:** `slice-diversity-reg`
**Priority: HIGH — targets OOD attention sink pathology, ~20 LoC, orthogonal to all in-flight**

**What it is:** Add a Gram matrix orthogonality penalty on the physics-attention slice assignment weights to prevent multiple slices from routing identically. Specifically: `L_div = lambda * ||A^T A - I||_F^2` where `A` is the slice assignment matrix `[B, N, slice_num]` (after softmax). Penalizes off-diagonal entries of the Gram matrix — pushes slices to partition the input space more distinctly.

**Why it might help here:**
For OOD NACA6416 inputs, nodes in the tandem wake region may all collapse to the same few physics slices, while most of the 96 slices process nothing useful — the attention sink problem in a mesh context. This leaves most of the model's representational capacity unused for the hardest inputs.

This is complementary to and architecturally orthogonal from register tokens (PR #2209, now closed as a dead end). Register tokens were an architectural fix that failed because the slice-deslice mechanism already provides global aggregates. Diversity regularization is a loss-side fix that doesn't change the architecture — it's a gradient signal that encourages the existing 96 slices to each handle something different.

The Gram matrix regularizer is established in multi-head attention contexts (e.g., subspace regularization in Linformer ablations) and in dictionary learning (RBM/VQ literature). It's mathematically clean, differentiable, and adds no forward-pass compute at inference time.

**Implementation (~20 LoC):**
```python
# Store slice attention weights during forward pass:
# In physics_attention forward, after computing A (softmax slice assignments):
#   model_out["slice_attn_weights"].append(A)  # [B, N, slice_num]

# In training loss computation:
if cfg.slice_diversity_reg:
    diversity_loss = 0.0
    for A in model_out["slice_attn_weights"]:  # per-block
        # Normalize per-node (so the Gram measures slice direction similarity)
        A_norm = F.normalize(A, dim=1, p=2)  # [B, N, slice_num]
        G = torch.bmm(A_norm.transpose(1, 2), A_norm)  # [B, slice_num, slice_num]
        I = torch.eye(G.shape[-1], device=G.device).unsqueeze(0)
        diversity_loss += ((G - I) ** 2).mean()
    loss = loss + cfg.slice_diversity_weight * diversity_loss
```

**Flag:** `--slice_diversity_reg`, `--slice_diversity_weight 0.005`

**Tuning:** Start conservative at `lambda=0.005` — too high and the model fights physically correct co-routing (e.g., all boundary layer nodes should share a slice). Monitor `diversity_loss` independently in W&B to check it's not dominating. If no improvement at 0.005, try 0.02.

**Confidence:** Medium. The attention sink pathology in OOD settings is well-documented in ViT (motivated the register tokens paper arXiv 2309.16588). The Gram regularizer is a principled proxy for slice diversity. Main risk: slice_num=96 may already provide enough coverage that diversity isn't the binding constraint.

---

## Idea 4: Domain-Split SRF Normalization

**Slug:** `domain-split-srf-norm`
**Priority: MEDIUM-HIGH — ~15 LoC, directly targets tandem-specific SRF calibration, safe init**

**What it is:** In the `AftFoilRefinementHead`, replace the shared `nn.LayerNorm` with a domain-conditional normalization: learn separate scale and bias adjustments for tandem vs. non-tandem samples via a zero-initialized `nn.Embedding(2, n_hidden)`. On tandem samples, the LayerNorm output is additively corrected. On non-tandem, the correction is zero (embedding row 0 = zeros). Zero init = first pass identical to baseline.

**Why it might help here:**
The AftSRF sees aft-foil surface nodes from both tandem and non-tandem samples. In tandem, wake impingement creates an adverse pressure gradient region on the aft-foil suction side that is completely absent in single-foil configurations. A single LayerNorm must normalize both pressure distributions with shared scale/bias statistics — a domain mismatch that conditional normalization can resolve.

This is specifically NOT the failed Domain AdaLN (#2164, +5.9-6.8% regression). The key distinction:
- #2164: applied domain conditioning to the BACKBONE (all nodes, all blocks) — catastrophic because it disturbed the slice routing
- This idea: domain conditioning ONLY in the AftSRF head (surface nodes only, post-backbone) — backbone routing is untouched

The AftSRF processes a small, well-defined set of nodes (aft-foil surface only, boundary_id=7) where the tandem vs. single-foil distinction is most physically meaningful.

**Implementation (~15 LoC):**
```python
class AftFoilRefinementHead(nn.Module):
    def __init__(self, n_hidden, domain_split_norm=False, ...):
        ...
        self.norm = nn.LayerNorm(n_hidden)
        self.domain_split_norm = domain_split_norm
        if domain_split_norm:
            self.domain_scale = nn.Embedding(2, n_hidden)  # 0=single, 1=tandem
            self.domain_bias = nn.Embedding(2, n_hidden)
            nn.init.zeros_(self.domain_scale.weight)
            nn.init.zeros_(self.domain_bias.weight)
    
    def forward(self, x, is_tandem_flag):
        # is_tandem_flag: [B] integer tensor, 0 or 1
        x_norm = self.norm(x)
        if self.domain_split_norm:
            ds = self.domain_scale(is_tandem_flag)[:, None, :]  # [B, 1, n_hidden]
            db = self.domain_bias(is_tandem_flag)[:, None, :]
            x_norm = x_norm * (1 + ds) + db  # multiplicative scale + additive bias
        ...
```

`is_tandem_flag` is recoverable from `x`: tandem samples have boundary_id=7 nodes and nonzero foil-2 features.

**Flag:** `--domain_split_srf_norm` (boolean). Default False.

**Confidence:** Medium. The distinction from dead-end #2164 is architectural and well-justified. Domain-conditional normalization is standard in multi-domain learning (AdaIN, MUNIT). Zero-init ensures safety. Main risk: AftSRF processes only ~2% of nodes — the domain signal may be too dilute to learn useful deltas.

---

## Idea 5: Tandem Feature Cross (Input-Level Sigmoid Gate)

**Slug:** `tandem-feature-cross`
**Priority: MEDIUM-HIGH — lightweight, physics-motivated input conditioning, ~25 LoC**

**What it is:** Before the input encoder, compute a configuration-specific gate: `g = sigmoid(MLP([gap, stagger, Re, AoA]))` of shape `[B, n_hidden]`, then modulate all node encodings: `x_enc = x_enc * g`. This allows the model to adapt which input channels are amplified vs. suppressed based on the global flow configuration. Near-identity initialization: final bias → 5.0 so `sigmoid(5) ≈ 0.99`.

**Why it might help here:**
Gap and stagger scalars currently feed only into `--gap_stagger_spatial_bias` (slice routing). This helps routing but doesn't affect the INPUT feature encoding. For NACA6416 OOD: the DSDF camber features signal a different geometric prior, but the model has no mechanism to upweight these features when it detects an OOD configuration. A sigmoid gate parameterized by (gap, stagger, Re, AoA) can learn "for this tandem gap/stagger, the DSDF channel 8 (camber derivative) is more informative than usual — amplify it." The gate is fully differentiable and adds negligible compute.

This is distinct from the failed feature_cross from earlier phases (PR #1436 and relatives) which was a learned node-to-node interaction matrix. This is a GLOBAL gate (same vector broadcast over all N nodes) conditioned on 4 scalars — much lower risk of overfitting.

**Implementation (~25 LoC):**
```python
# In Transolver.__init__:
if cfg.tandem_feature_cross:
    self.tandem_gate = nn.Sequential(
        nn.Linear(4, 32), nn.GELU(),
        nn.Linear(32, n_hidden),
    )
    # Near-identity init: sigmoid(5) ≈ 0.99 → gate ≈ 1 (near-passthrough)
    nn.init.zeros_(self.tandem_gate[-1].weight)
    nn.init.constant_(self.tandem_gate[-1].bias, 5.0)

# In Transolver.forward, after input encoding:
if cfg.tandem_feature_cross:
    config_vec = torch.stack([gap, stagger, re, aoa], dim=-1)  # [B, 4]
    gate = torch.sigmoid(self.tandem_gate(config_vec))          # [B, n_hidden]
    x_enc_out = x_enc_out * gate[:, None, :]                   # [B, N, n_hidden]
```

For non-tandem samples: gap=0, stagger=0, so gate = sigmoid(bias) ≈ 0.99 — effectively identity.

**Flag:** `--tandem_feature_cross` (boolean). Default False.

**Confidence:** Medium. The mechanism is physically motivated. Gap and stagger as input-gate conditioning is novel in this codebase (previously only used in routing). Main risk: redundancy with `--gap_stagger_spatial_bias`. However, routing vs. feature modulation are mechanistically different, so orthogonality is plausible.

---

## Idea 6: Surface Node Arc-Length Positional Encoding

**Slug:** `surface-arc-length-pe`
**Priority: MEDIUM-HIGH — complements TE coord frame, provides local surface position context**

**What it is:** For each surface node (boundary_id in {5, 6, 7}), compute the arc-length distance `s` from the leading edge (x-coord minimum), normalized by chord length. Add this scalar as a new input feature channel. For volume nodes, `s = 0` (or a special out-of-surface indicator). This gives the model explicit knowledge of "where on the airfoil surface is this node" — complementing the TE coordinate frame (PR #2207) which gives global position.

**Why it might help here:**
The TE coordinate frame (PR #2207, -5.4% p_in) showed that explicit geometric reference frame features dramatically help. Arc-length PE provides the orthogonal information: instead of "how far from the TE" (radial), it provides "how far along the surface" (curvilinear). For NACA6416, the leading-edge region has higher curvature and the stagnation point sits at a different chordwise position — arc-length normalized by chord is invariant to chord scale differences between NACA0012 and NACA6416.

Arc-length PE is standard in surface pressure modeling (panel methods use it), and in transformer models for sequences (though here the "sequence" is the surface geometry). It provides the model with a natural parameterization of the pressure distribution as a function of arc-length s, which is how aerodynamicists think about Cp curves.

The current DSDF features encode distance TO the surface but not ALONG the surface — arc-length PE fills this gap.

**Implementation:**
Arc-length computation per sample: sort surface nodes by angle from airfoil centroid (or directly from the x,y coordinates), compute cumulative edge lengths, normalize by total perimeter (chord length proxy). This needs ~20 LoC in the data path, but can be done at training time from the input coordinates `x[:, :, 0:2]` (node x, y positions, which are available in `prepare_multi.py` as the first 2 dims).

For tandem samples: compute separately for fore-foil (boundary_id 5/6) and aft-foil (boundary_id 7). Normalize by respective chord lengths.

**Flag:** `--surface_arc_length_pe` (boolean). Adds 1 input channel → n_x=25.

**Key constraint:** The input projection `Linear(n_x, n_hidden)` must be updated from 24 to 25. New column initialized to near-zero (use `nn.init.normal_` with std=0.01) so the new feature has minimal impact at epoch 0.

**Confidence:** Medium. The TE coord frame success strongly motivates more geometric feature engineering on surface nodes. Arc-length is the canonical surface parameterization in aerodynamics. The main risk is that DSDF features already implicitly encode arc-length information through the curvature tensor.

---

## Idea 7: Panel Method Cp as Input Feature

**Slug:** `panel-method-cp-input`
**Priority: MEDIUM — was proposed as PR #1865 but NEVER RAN, physics prior injection**

**What it is:** Compute the inviscid thin-airfoil panel method Cp at each surface node and inject it as an additional input feature. The panel method solves a 2D potential flow problem analytically (Hess-Smith panel method, O(N^2) but N~150 surface nodes so fast) and gives a physically grounded baseline Cp that already captures leading-edge suction, trailing-edge pressure recovery, and camber effects. The network then predicts the viscous CORRECTION to this inviscid baseline, similar to the residual prediction flag (`--residual_prediction`) but physics-aware.

**Why it might help here:**
This was explicitly proposed as PR #1865 in an earlier phase but was never executed (the student assignment was lost). The core motivation: for NACA6416, the inviscid Cp distribution is qualitatively different from NACA0012 (different camber, different LE suction peak location, different pressure recovery shape). Injecting this as an input feature gives the model a geometry-aware physical prior for EACH airfoil shape, making the OOD generalization easier — the model only needs to learn the viscous correction, not the full potential flow.

The panel method Cp can be precomputed at training time for each surface node from the surface node coordinates (x, y), which are available in the input. It requires:
1. Extracting surface nodes for each foil
2. Running Hess-Smith panel method (~30 LoC of pure NumPy)
3. Injecting the computed Cp as input channel 25 (surface nodes only; volume nodes get 0)

This is related to PR #2212 (askeladd, analytical Cp delta using thin-airfoil theory) but simpler — instead of predicting a DELTA from the analytical Cp, it injects the analytical Cp as an input feature and lets the model learn the correction naturally.

**Implementation notes:**
The Hess-Smith panel method for 2D airfoil:
```python
def hess_smith_cp(x_surf, y_surf, alpha_deg):
    """Compute inviscid Cp at surface nodes using 2D panel method."""
    # x_surf, y_surf: [N_surf] numpy arrays (surface node coordinates)
    # Returns: Cp [N_surf]
    # Implementation: source-doublet panels, solve for surface velocities
    # ~30 LoC using numpy linear algebra
    ...
```
Can be precomputed in `data/prepare.py` or computed lazily in train.py during batch construction.

**Flag:** `--panel_method_cp` (boolean). Adds 1 input channel (n_x=25 for surface nodes, padded with 0 for volume).

**NOTE:** Check that this is truly orthogonal to #2212 (askeladd, analytical Cp delta). If #2212 uses a similar thin-airfoil formula, coordinate with the advisor before implementing. The distinction is: this adds Cp as an INPUT feature; #2212 uses it as a SRF correction BASELINE. Both can be run simultaneously without conflict.

**Reference:** Hess, J.L. and Smith, A.M.O. (1967), "Calculation of Potential Flow about Arbitrary Bodies." Progress in Aerospace Sciences, 8, 1-138. Classical panel method — implemented in <50 LoC for 2D.

**Confidence:** Medium. The idea is well-motivated but never executed. The main uncertainty is implementation complexity (panel method in the training loop). If precomputed offline as part of the dataset, it becomes a simple feature injection — feasible.

---

## Idea 8: Chord-Adaptive LE/TE Loss Weighting

**Slug:** `chord-adaptive-le-te-loss`
**Priority: MEDIUM — precise loss targeting of highest-error surface regions, ~15 LoC**

**What it is:** Apply additional loss weight to surface nodes within the leading-edge (LE) and trailing-edge (TE) regions. Specifically: identify nodes within arc-length distance `epsilon_LE` from the LE (stagnation point) and `epsilon_TE` from the TE, and multiply their loss contribution by a configurable boost factor (default 2.0). Boosting is adaptive: `epsilon_LE` and `epsilon_TE` are set as fractions of chord length (e.g., 5%) so they scale correctly with different airfoil sizes.

**Why it might help here:**
Current surface loss treats all N_surf surface nodes equally. The LE and TE regions have the highest pressure gradients AND are the primary contributors to lift (LE suction peak) and drag (TE pressure recovery). These regions also have the highest prediction error in typical Transolver runs — the model under-resolves the suction peak.

This is DIFFERENT from and COMPLEMENTARY to PR #2210 (fern, arc-length surface loss reweighting), which corrects for non-uniform mesh density. This idea is about TARGETED physics region boosting on top of the density-corrected baseline.

For NACA6416 OOD: the LE suction peak is at a different chordwise location than NACA0012 (NACA6416 has 6% camber, shifting the suction peak aft). By boosting the LE region, the model is forced to get the suction peak location right — critical for p_tan.

**Implementation (~15 LoC):**
```python
# In training loop, after computing per-node loss:
if cfg.chord_adaptive_le_te_loss:
    # Identify LE and TE nodes from surface coordinates
    surf_x = x[is_surface, 0]  # x-coordinates of surface nodes
    le_mask = surf_x < x_chord_min + cfg.le_te_epsilon * chord_length
    te_mask = surf_x > x_chord_max - cfg.le_te_epsilon * chord_length
    le_te_mask = le_mask | te_mask
    surface_loss_weights[is_surface & le_te_mask] *= cfg.le_te_boost_factor

# Defaults:
# --le_te_epsilon 0.05 (5% chord from each end)
# --le_te_boost_factor 2.0
```

Wait for PR #2210 (arc-length reweighting) result before assigning this. If #2210 merges, this becomes an additive refinement. If #2210 doesn't improve, evaluate whether a different loss geometry approach is warranted.

**Flag:** `--chord_adaptive_le_te_loss`, `--le_te_epsilon 0.05`, `--le_te_boost_factor 2.0`

**Confidence:** Medium. Loss region weighting is empirically well-supported (hard mining, curriculum, DCT freq loss all merged successfully). LE/TE targeting is physically principled. Main risk: overlaps with arc-length reweighting #2210 — assign AFTER #2210 result known.

---

## Idea 9: Kutta Condition Auxiliary Loss

**Slug:** `kutta-condition-loss`
**Priority: MEDIUM — aerodynamics physics constraint, novel in this codebase**

**What it is:** Add an auxiliary loss penalizing pressure discontinuity at the trailing edge nodes. The Kutta condition (the fundamental boundary condition that uniquely determines circulation in potential flow) requires that pressure is continuous at the TE: `p_upper_TE = p_lower_TE`. The predicted pressure at the upper and lower TE surface nodes should be equal; penalize their difference.

**Why it might help here:**
The Kutta condition is the core physical constraint that governs lift generation. Violating it means the model is predicting unphysical circulation values, which propagates downstream to the aft-foil wake impingement. For NACA6416, which has a sharp TE (by NACA convention), the condition is cleaner than for blunt-TE geometries.

This is a hard physical constraint that is currently NOT enforced anywhere in the training — the model only sees L1 surface MAE against CFD targets. Adding an explicit Kutta penalty should tighten TE pressure predictions, which directly affects the wake structure and thus aft-foil p_tan.

**Implementation (~20 LoC):**
```python
# In training loop:
if cfg.kutta_condition_loss and is_tandem_sample:
    # Find upper and lower TE nodes (nodes closest to trailing edge on each surface)
    te_upper_idx = find_surface_te_node(x, boundary_id=5, side='upper')
    te_lower_idx = find_surface_te_node(x, boundary_id=5, side='lower')
    p_upper = pred_pressure[te_upper_idx]
    p_lower = pred_pressure[te_lower_idx]
    kutta_loss = cfg.kutta_weight * F.mse_loss(p_upper, p_lower)
    loss = loss + kutta_loss
```

TE identification: find the surface node(s) with maximum x-coordinate on each foil. Upper/lower side: `y > y_TE` is upper, `y < y_TE` is lower.

**Flag:** `--kutta_condition_loss`, `--kutta_weight 0.01`

**Tuning:** Start at lambda=0.01. If the model already approximately satisfies Kutta (CFD solutions do), the loss will be near-zero and harmless. If there's systematic violation, the penalty will correct it.

**Known caveat:** Some NACA profiles use finite-thickness TEs (NACA4-digit series with t>0), in which case `p_upper_TE != p_lower_TE` is allowed — the Kutta condition is enforced by equal velocity, not equal pressure. Check whether the TandemFoilSet uses sharp or blunt TEs before implementing. If blunt TE, use equal VELOCITY magnitude at TE instead.

**Reference:** Anderson, J.D., "Fundamentals of Aerodynamics," Chapter 4 (Kutta condition). Standard aerodynamics textbook.

**Confidence:** Medium. Physically motivated, novel in this codebase. Main uncertainty: the CFD solutions already satisfy Kutta implicitly (CFD enforces it via boundary conditions), so the model may already learn it — making this penalty redundant. Worth testing empirically.

---

## Idea 10: Integrated Lift-Drag Auxiliary Loss

**Slug:** `lift-drag-integral-loss`
**Priority: MEDIUM — global physics consistency constraint on surface pressure**

**What it is:** Add auxiliary losses penalizing errors in the integrated lift coefficient (Cl) and drag coefficient (Cd) computed from the predicted surface pressure. Cl and Cd are computed as surface integrals of `p * n_y` and `p * n_x` respectively (where `n` is the outward surface normal). This gives the model a global physics consistency signal: the predicted pressure field must integrate to correct lift and drag.

**Why it might help here:**
Current training optimizes per-node L1 MAE. Two pressure fields with equal per-node MAE may have very different integrated Cl/Cd — the model could predict high-error TE regions that cancel out in L1 but produce wrong lift. For NACA6416 tandem, the fore-foil generates specific Cl that directly determines wake strength and aft-foil loading. If Cl prediction is wrong, the aft-foil p_tan is wrong regardless of per-node accuracy.

Cl/Cd auxiliary losses provide a GLOBAL constraint that rewards integrated correctness, complementing the LOCAL per-node L1 loss. This is analogous to how the DCT frequency loss (merged, working) provides a frequency-domain constraint complementary to spatial L1.

**Implementation (~25 LoC):**
```python
def compute_cl_cd(pressure, surface_normals, surface_areas, q_inf):
    """Integrate surface pressure to lift and drag coefficients."""
    # pressure: [B, N_surf] predicted pressure
    # surface_normals: [B, N_surf, 2] outward normals (nx, ny)
    # surface_areas: [B, N_surf] panel widths (ds for 2D)
    # q_inf: dynamic pressure (0.5 * rho * U_inf^2)
    Cl_pred = (-pressure * surface_normals[:, :, 1] * surface_areas).sum(-1) / q_inf
    Cd_pred = (-pressure * surface_normals[:, :, 0] * surface_areas).sum(-1) / q_inf
    return Cl_pred, Cd_pred

# Compute for both predicted and target pressure
Cl_pred, Cd_pred = compute_cl_cd(pred_p_surf, normals, areas, q_inf)
Cl_true, Cd_true = compute_cl_cd(true_p_surf, normals, areas, q_inf)

lift_drag_loss = cfg.lift_drag_weight * (
    F.l1_loss(Cl_pred, Cl_true) + F.l1_loss(Cd_pred, Cd_true)
)
loss = loss + lift_drag_loss
```

Surface normals can be computed from consecutive surface node positions `(x_{i+1} - x_i)` rotated 90 degrees. Panel widths are `||x_{i+1} - x_i||`.

**Flag:** `--lift_drag_integral_loss`, `--lift_drag_weight 0.1`

**Note:** Surface normals are NOT currently computed during training. This requires ~15 LoC to compute from surface node coordinates. If `prepare_multi.py` already computes normals, use those directly (check before implementing).

**Confidence:** Medium. Integral loss constraints are well-established in physics-informed learning (energy conservation, mass conservation). Cl/Cd is the engineering quantity of direct interest. Main risk: the integral constraint may be too global to localize the gradient signal to the nodes that need correction most.

---

## Idea 11: Geometry-Moment Global Conditioning

**Slug:** `geometry-moment-conditioning`
**Priority: MEDIUM — ~20 LoC, targets OOD shape generalization with geometric moments**

**What it is:** Compute geometric moments of the fore-foil shape (area, centroid x/y, second moments Ixx, Iyy, Ixy) from the surface DSDF nodes and inject them as a 7-scalar global conditioning vector into the backbone. Specifically: project the 7 scalars to n_hidden via a small MLP and ADD to the backbone input encoding before the first TransolverBlock. Zero-init the output projection.

**Why it might help here:**
The geometric moments uniquely characterize the airfoil shape. For NACA6416, area is larger (6% camber increases enclosed area), centroid is shifted aft, and the second moments encode the asymmetric thickness distribution. The current model has this information implicitly in DSDF values but has no holistic shape summary. A moment-based conditioning vector gives the model a compact, globally consistent shape fingerprint.

This is inspired by shape moments in computer vision (HuMoments for shape description) and geometry-aware conditioning in physical simulation (mesh-to-mesh transfer). It's simpler than the GeoTransolver GALE architecture (#2216, in-flight) — 7 scalars rather than a full geometry encoder — and thus lower risk.

**Implementation (~20 LoC):**
```python
# In Transolver.__init__:
if cfg.geometry_moment_conditioning:
    self.geo_moment_proj = nn.Sequential(
        nn.Linear(7, 32), nn.GELU(),
        nn.Linear(32, n_hidden)
    )
    nn.init.zeros_(self.geo_moment_proj[-1].weight)
    nn.init.zeros_(self.geo_moment_proj[-1].bias)

# In forward, before TransolverBlocks:
if cfg.geometry_moment_conditioning:
    fore_surf_xy = x[fore_surface_mask, 0:2]  # [N_fore_surf, 2]
    moments = compute_shape_moments(fore_surf_xy)  # [B, 7]
    geo_conditioning = self.geo_moment_proj(moments)  # [B, n_hidden]
    fx = fx + geo_conditioning[:, None, :]  # broadcast over N nodes
```

`compute_shape_moments` computes area (using shoelace formula), centroid (x̄, ȳ), and 4 second moments (Ixx, Iyy, Ixy, area moment of inertia) — all standard formulas from solid mechanics.

**Flag:** `--geometry_moment_conditioning` (boolean).

**Confidence:** Medium. Shape moments are a well-established compact geometric descriptor. The key advantage over GALE: much simpler (no cross-attention), easier to implement, lower risk. Assigns after #2216 result known — if GALE works, moments may be redundant; if GALE fails, moments are a simpler alternative.

---

## Idea 12: Node Type Boundary Embedding

**Slug:** `node-type-boundary-embedding`
**Priority: MEDIUM — ~10 LoC, adds type-awareness the model currently lacks**

**What it is:** Add a learned `nn.Embedding(num_boundary_types, n_hidden)` indexed by boundary ID (0=volume, 5=fore-foil surface, 6=aft-foil surface, 7=aft-foil tandem surface, etc.) to the input encoding. This gives each node type a learnable identity vector that the model can use to distinguish node roles. Zero-init all embeddings except the volume embedding, so the first pass is identical to baseline.

**Why it might help here:**
The model currently infers node type implicitly from DSDF features and the `is_surface` indicator. But boundary ID 7 (aft-foil tandem surface) is the hardest metric — p_tan. Giving these nodes an explicit learnable identity (a "tandem aft surface" embedding) gives the SRF head and the backbone a trainable signal that distinguishes them from regular surface nodes (boundary ID 5/6). The optimizer can then learn, for node type 7, to apply different transformation weights — without this, the model must infer this distinction from the DSDF values alone.

Note: this is NOT the same as domain-split SRF norm (Idea 4 above). That conditions the LayerNorm statistics; this provides an additive encoding signal at input level. Both can run simultaneously.

**Implementation (~10 LoC):**
```python
# In Transolver.__init__:
if cfg.node_type_embedding:
    self.node_type_emb = nn.Embedding(num_embeddings=16, embedding_dim=n_hidden)
    nn.init.zeros_(self.node_type_emb.weight)  # zero-init: additive correction

# In forward, after input encoding:
if cfg.node_type_embedding:
    node_type_ids = x[:, :, boundary_id_channel].long()  # [B, N] integer
    type_emb = self.node_type_emb(node_type_ids)  # [B, N, n_hidden]
    x_enc_out = x_enc_out + type_emb
```

Boundary ID is available in the 24-dim input x. Confirm which channel encodes boundary ID from `prepare_multi.py`.

**Flag:** `--node_type_embedding` (boolean).

**Confidence:** Medium. Very low implementation risk. Type embeddings are standard in graph neural networks (node type embeddings in heterogeneous GNNs). The main question is whether DSDF already provides sufficient type distinction — empirically test.

---

## Priority Ranking and Assignment Order

| Rank | Idea | Slug | Code | Confidence | Notes |
|------|------|------|------|------------|-------|
| 1 | mHC Learnable Residuals | `mhc-residuals` | ~15 LoC | Medium | Human-requested #1926, 3 rounds queued |
| 2 | Additive Fore-Aft Cross-Attn SRF | `additive-fore-aft-crossattn-srf` | ~30 LoC | Medium-high | Targeted fix of failed #2202 |
| 3 | Slice Diversity Regularization | `slice-diversity-reg` | ~20 LoC | Medium | Orthogonal to all in-flight |
| 4 | Domain-Split SRF Norm | `domain-split-srf-norm` | ~15 LoC | Medium | Distinct from dead-end #2164 |
| 5 | Tandem Feature Cross | `tandem-feature-cross` | ~25 LoC | Medium | Novel input-gate mechanism |
| 6 | Surface Arc-Length PE | `surface-arc-length-pe` | ~20 LoC | Medium | Extends TE frame success direction |
| 7 | Panel Method Cp Input | `panel-method-cp-input` | ~50 LoC | Medium | Proposed PR #1865, NEVER RAN |
| 8 | Chord-Adaptive LE/TE Loss | `chord-adaptive-le-te-loss` | ~15 LoC | Medium | Assign AFTER #2210 result |
| 9 | Kutta Condition Aux Loss | `kutta-condition-loss` | ~20 LoC | Medium | Novel aerodynamic constraint |
| 10 | Lift-Drag Integral Loss | `lift-drag-integral-loss` | ~25 LoC | Medium | Global physics consistency |
| 11 | Geometry Moment Conditioning | `geometry-moment-conditioning` | ~20 LoC | Medium | Assign AFTER #2216 (GALE) result |
| 12 | Node Type Boundary Embedding | `node-type-boundary-embedding` | ~10 LoC | Medium | Simplest, lowest risk |

---

## Design Principles for These Experiments

1. **Zero-init all new parameters.** Non-negotiable: first forward pass must match baseline. All ideas above include zero-init implementations.

2. **Two seeds {42, 73}, report both.** Baseline seed variance on p_tan is ~0.14. Single-seed results are not actionable.

3. **Watch for degenerate solutions in mHC.** If alpha or beta → 0, log it explicitly — one path killing itself may be valid but needs documentation.

4. **Sequence-dependent ideas:** Panel method Cp (#7) should be implemented AFTER analytical Cp delta (#2212, askeladd) returns results — they're related and the result informs the approach. Chord-adaptive LE/TE loss (#8) should wait for arc-length reweighting (#2210, fern).

5. **Round 15 focus: p_tan below 28.0.** Every experiment should be analyzed for whether it specifically improves the tandem OOD split. An experiment that improves p_in and p_oodc but leaves p_tan flat is interesting but not a priority merge.

6. **The TE coordinate frame direction (PR #2207, +5.4% p_in) remains the most productive recent thread.** Ideas 6 (arc-length PE), 9 (Kutta), 10 (lift-drag), 11 (geometry moments) all build on the principle that explicit geometric/physical structure injection helps OOD generalization. Prioritize in this cluster.
