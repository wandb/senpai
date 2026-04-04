# SENPAI Research Ideas — 2026-04-04 (Nezuko Assignment)

Generated for idle student nezuko after PR #2115 (gap/stagger perturbation augmentation) merged.

**Current single-model baseline (PR #2104, +aft_foil_srf, 8-seed mean, seeds 42-49):**
p_in=13.19 ± 0.33, p_oodc=7.92 ± 0.17, p_tan=30.05 ± 0.36, p_re=6.45 ± 0.07

**Active WIP (do NOT duplicate):**
- #2125 (thorfinn): Re perturbation augmentation — Gaussian noise on log_Re feature index 13
- #2126: Foil-2 DSDF magnitude augmentation — log-normal scale on channels 6:10, tandem-only
- #2127: KNN volume context for AftFoilRefinementHead — new AftFoilRefinementContextHead, k=8 neighbors
- #2128 (edward): FiLM conditioning on (Re, AoA) indices 13:15 for SurfaceRefinementHead

---

## Ranked Hypotheses

### 1. gap-stagger-spatial-bias — Gap/Stagger-Conditioned Spatial Bias

**Priority: HIGH. #1 recommendation for nezuko. ~18 LoC. Expected -3 to -7% p_tan. MEDIUM risk.**

**What it is:** Extend the `raw_xy` input to the `spatial_bias` MLP in each `TransolverBlock` from 4 dimensions (x, y, curvature, dist-to-surface) to 6 dimensions by appending the global inter-foil geometry: `gap` (feature index 21) and `stagger` (feature index 22), broadcast to all nodes. This conditions Transolver's slice routing on the tandem configuration geometry at every attention layer.

**Why this targets p_tan — the core mechanism:**

Transolver's slice routing assigns each mesh node to a soft mixture of physics slices via `slice_logits = ...+ 0.1 * spatial_bias(raw_xy)`. Currently, `raw_xy` is purely local: (x_coord, y_coord, curvature, dist-to-surface). The spatial_bias MLP has no knowledge of the tandem inter-foil geometry. For single-foil samples, gap=0 and stagger=0 — so the new inputs are always zero for those samples, meaning the change has NO effect on p_in/p_oodc. For tandem samples, different gap/stagger values fundamentally change the flow topology (wake interaction, channel flow, suction-side shielding), but all tandem configurations currently route through identical slice assignment logic regardless of their inter-foil separation.

The gap/stagger augmentation in PR #2115 showed that adding geometric perturbation during training improves tandem generalization. The underlying mechanism is that the model needs to be robust to geometric variation. This experiment takes the complementary action: make the model's internal routing explicitly AWARE of that geometry. Rather than regularizing by varying gap/stagger during training (what #2115 does), we route computation differently based on gap/stagger (what this idea does). These are orthogonal and complementary mechanisms.

**Why this is novel — it has never been tried:**

Every prior experiment has either (a) added global geometry as an input feature in the first linear projection (early embedding), (b) used global geometry for loss weighting or auxiliary heads (post-trunk), or (c) used it for FiLM conditioning on the refinement heads (#2128 does this for Re/AoA). No prior experiment has made the Transolver's core slice routing attention geometry-aware at the inter-foil scale. The spatial_bias pathway is the unique mechanism that determines HOW physics are distributed across learned physics slices — conditioning it on global tandem geometry is a new pathway for information to flow.

**Exact code changes — three locations:**

**Location 1: `TransolverBlock.__init__` (line ~330)**

```python
# CURRENT:
self.spatial_bias = nn.Sequential(
    nn.Linear(4, 64), nn.GELU(),
    nn.Linear(64, 64), nn.GELU(),
    nn.Linear(64, slice_num),
)
nn.init.zeros_(self.spatial_bias[-1].weight)
nn.init.zeros_(self.spatial_bias[-1].bias)

# PROPOSED: add spatial_bias_input_dim parameter to __init__ (default=4):
_sb_dim = getattr(cfg_or_passed_arg, 'spatial_bias_input_dim', 4)
# OR pass it as a direct argument: def __init__(self, ..., spatial_bias_input_dim=4):
self.spatial_bias = nn.Sequential(
    nn.Linear(spatial_bias_input_dim, 64), nn.GELU(),
    nn.Linear(64, 64), nn.GELU(),
    nn.Linear(64, slice_num),
)
nn.init.zeros_(self.spatial_bias[-1].weight)
nn.init.zeros_(self.spatial_bias[-1].bias)
# NOTE: Do NOT zero-init the new input weights — leave them at default kaiming_uniform.
# The output layer is already zero-init, so the new inputs start with zero effect.
# Alternative (safer): zero-init the entire first layer's extra 2 columns explicitly.
```

**Location 2: `Transolver.forward` (line ~833) — raw_xy construction**

```python
# CURRENT:
raw_xy = torch.cat([x[:, :, :2], x[:, :, 24:26]], dim=-1)  # x, y, curvature, dist — [B, N, 4]

# PROPOSED (flag-guarded):
if cfg.gap_stagger_spatial_bias:
    gap_stagger = x[:, 0:1, 21:23].expand(-1, x.shape[1], -1)  # [B, N, 2], broadcast from first node
    raw_xy = torch.cat([x[:, :, :2], x[:, :, 24:26], gap_stagger], dim=-1)  # [B, N, 6]
else:
    raw_xy = torch.cat([x[:, :, :2], x[:, :, 24:26]], dim=-1)  # [B, N, 4]
```

NOTE: `x[:, 0, 21]` is the gap feature and `x[:, 0, 22]` is the stagger feature. These are global (same for all nodes in a sample), so `x[:, 0:1, 21:23].expand(-1, N, -1)` is correct and torch.compile safe (no dynamic shapes, constant expand).

NOTE: This `x` is already post-standardization (line 833 is inside the model forward after standardization has been applied). The standardized gap/stagger values are in range [-2, 2] typically, which is appropriate as spatial_bias input. NO denormalization needed.

**Location 3: `Transolver.__init__` — pass dim to TransolverBlock**

```python
# When constructing TransolverBlock instances inside Transolver.__init__,
# pass spatial_bias_input_dim=6 if cfg.gap_stagger_spatial_bias else 4:
for i in range(n_layers):
    block = TransolverBlock(
        ...,
        spatial_bias_input_dim=6 if cfg.gap_stagger_spatial_bias else 4,
    )
    self.blocks.append(block)
```

**New config flag:**
```python
gap_stagger_spatial_bias: bool = False
```
CLI: `--gap_stagger_spatial_bias`

**Parameter count change:** Negligible. Each TransolverBlock gains 2 extra input weights in the first Linear of spatial_bias (4→6 means 2 × 64 = 128 additional scalars per block). With 3 blocks, that is 384 extra scalars total — essentially zero overhead.

**torch.compile safety:** Fully safe. `x[:, 0:1, 21:23].expand(-1, x.shape[1], -1)` is a static-shape operation. The `if cfg.gap_stagger_spatial_bias` branch is resolved at trace time (cfg is a frozen dataclass). No dynamic shapes anywhere.

**Interaction with merged improvements:**
- PR #2115 (gap/stagger augmentation) complements this: augmentation makes the model robust to gap/stagger variation during training; this makes slice routing aware of gap/stagger values. Orthogonal pathways that both serve tandem generalization.
- PR #2104 (aft-foil SRF head) complements this: the SRF head corrects post-trunk predictions; this change improves the trunk representations feeding the SRF head.
- These changes compound without interference.

**Expected impact analysis:**
The p_tan gap (30.05 vs p_in=13.19, factor 2.3x) is the single largest metric gap in the programme. Every layer of the Transolver currently assigns physics slices based only on local geometry. Adding 2 dimensions that carry inter-foil configuration information — and that change drastically across tandem samples (gap ranges 0.5-2.0 chord lengths, stagger -0.5 to 0.5 chord) — is a qualitatively new signal in a pathway that directly controls how physics are learned. The expected gain is 3-7% p_tan with minimal risk to p_in/p_oodc (single-foil samples always have gap=stagger=0 in standardized space, so the new dimensions add no information for those samples).

**Risk analysis:**
- LOW risk of p_in regression: gap/stagger are zero (or near-zero in standardized space) for non-tandem samples, so the new input dimensions carry no signal for single-foil routing.
- MEDIUM risk of p_tan being unstable: if the model over-relies on the gap/stagger route signal early in training, it might fail to learn general pressure features. Mitigate: zero-init the 2 new columns of the first Linear's weight matrix explicitly (see implementation note below).
- LOW risk of p_oodc regression: OOD cases are single-foil (same as p_in reasoning applies).

**Safer initialization option (recommended):**
```python
# After constructing spatial_bias, zero-init only the extra input columns:
if cfg.gap_stagger_spatial_bias:
    with torch.no_grad():
        for block in self.blocks:
            # spatial_bias[0] is the nn.Linear(6, 64)
            block.spatial_bias[0].weight[:, 4:].zero_()  # zero the 2 new input cols
```
This ensures the new inputs start with exactly zero contribution, same as the zero-init output layer. Training turns on the new signal gradually.

**Suggested experiment:**
```
--gap_stagger_spatial_bias
```
Run 2 seeds (42-43) for initial validation. If p_tan improves by >1% relative, run 8 seeds.

**Confidence:** Strong. The mechanism is direct — global geometry enters a pathway (slice routing) that was previously blind to it. No prior experiment has touched this pathway with global geometry. The zero-init guard removes initialization risk. Single-foil samples are unaffected by design.

---

### 2. boundary-id-onehot — Boundary-ID One-Hot as Sideband Input Feature

**Priority: HIGH. ~12 LoC. Expected -3 to -8% p_tan. MEDIUM risk.**

**What it is:** Append a 3-dimensional one-hot vector encoding node boundary type (single-foil surface ID=5, fore-foil ID=6, aft-foil ID=7) as an explicit input feature to every mesh node. Non-surface nodes get the zero vector.

**Why it helps p_tan:** The model must currently infer which boundary a surface node belongs to from geometric cues (SAF distance, position). The dedicated aft-foil SRF head (PR #2104, -0.8% p_tan) confirmed that boundary-type specialization is valuable, but that head only acts after the Transolver trunk has already processed the node without knowing its boundary type. Making boundary ID explicit in the input allows all 3 Transolver blocks to route computation differently for aft-foil nodes from layer 1 — an earlier, stronger conditioning signal.

**Implementation — proxy detection from raw features:**

```python
# BEFORE standardization (line ~1552 in train.py), add:
_raw_saf_norm_bid = x[:, :, 2:4].norm(dim=-1)           # [B, N] — SAF distance proxy
_is_tandem_bid = (x[:, 0, 21].abs() > 0.01)              # [B]
_bid_single = is_surface & ~_is_tandem_bid.unsqueeze(1)                                       # ID=5
_bid_fore   = is_surface & (_raw_saf_norm_bid <= 0.005) & _is_tandem_bid.unsqueeze(1)         # ID=6
_bid_aft    = is_surface & (_raw_saf_norm_bid  > 0.005) & _is_tandem_bid.unsqueeze(1)         # ID=7
_bid_onehot = torch.stack([_bid_single, _bid_fore, _bid_aft], dim=-1).float()  # [B, N, 3]

# AFTER Fourier PE append (line ~1584):
x = torch.cat([x, fourier_pe, _bid_onehot], dim=-1)  # total input dim +3
```

Add config flag `boundary_id_onehot: bool = False`. Input dim increases by 3; the model's input projection is dynamic so no other changes needed.

**Confidence:** Strong — direct address of boundary-type ambiguity. The same proxy detection infrastructure used by aft-foil SRF (PR #2104) applies here.

---

### 3. tandem-slice-specialization — Tandem-Specific Slice Carve-Out

**Priority: MEDIUM-HIGH. ~20 LoC. Expected -2 to -5% p_tan. MEDIUM risk.**

**What it is:** Reserve a dedicated subset of physics slices exclusively for tandem samples during training. With `slice_num=48` (or 96 in the 8-seed ensemble), partition slices into a "shared" pool and a "tandem-only" pool (e.g., 40 shared + 8 tandem-only). For single-foil samples, the softmax over all slices has a large negative bias applied to the tandem-only slice logits. For tandem samples, no bias is applied. This carves out dedicated capacity in the physics token space for tandem-specific flow patterns.

**Why it helps p_tan:** The current model uses the same 48 physics slices for all samples. Single-foil samples may be co-opting slice representations that would otherwise specialize for tandem inter-foil physics. By explicitly reserving slices for tandem use, we create dedicated representational capacity for the tandem-specific patterns (wake interaction, gap channel flow, fore-foil induced suction) without harming single-foil capacity.

**Implementation sketch:**
```python
# In TransolverBlock.forward, after computing slice_logits [B, N, num_slices]:
if cfg.tandem_slice_carveout > 0 and tandem_mask is not None:
    n_reserved = cfg.tandem_slice_carveout  # e.g. 8
    # Large negative bias on reserved slices for non-tandem samples
    is_nontandem = (1.0 - tandem_mask.squeeze())  # [B]
    carveout_bias = is_nontandem[:, None, None] * (-100.0)  # [B, 1, 1]
    reserved_bias = torch.zeros_like(slice_logits)
    reserved_bias[:, :, -n_reserved:] = carveout_bias
    slice_logits = slice_logits + reserved_bias
```

New flag: `--tandem_slice_carveout 8` (sweep: {4, 8, 16}).

**Confidence:** Moderate. The slice capacity argument is sound, but the implementation interacts with the existing softmax normalization in ways that require careful tuning. The -100 bias may be too aggressive and cause gradient issues — consider a learned gate instead.

---

### 4. fore-aft-loss-split — Tandem Fore/Aft Decoupled Loss Weighting

**Priority: MEDIUM. ~8 LoC. Expected -1 to -3% p_tan. LOW risk.**

**What it is:** Apply separate loss weights to fore-foil (ID=6) and aft-foil (ID=7) surface nodes: upweight aft-foil nodes by 1.5-2.0x in the L1 surface loss, while keeping fore-foil weight at 1.0. This pushes trunk representations toward being more informative for aft-foil nodes.

**Why it helps p_tan:** The aft-foil SRF head (PR #2104) confirmed aft-foil nodes have distinct error patterns. The surface loss currently weights all tandem surface nodes equally. Upweighting aft-foil loss directly pushes the main trunk toward better aft-foil representations, complementing the dedicated SRF head correction. The combination of loss upweighting + dedicated SRF head has not been tested together.

**Implementation:**
```python
# In the surface loss computation, after building surface_mask:
if cfg.aft_foil_loss_weight > 1.0 and _aft_foil_mask is not None:
    node_weights = torch.ones(B, N, device=device)
    node_weights[_aft_foil_mask] = cfg.aft_foil_loss_weight
    surface_loss = (l1_per_node * node_weights.unsqueeze(-1))[surface_mask].mean()
```

New flag: `--aft_foil_loss_weight 1.5` (sweep: {1.5, 2.0, 3.0}).

**Confidence:** Moderate-low. PR #1893 (Foil-2 Loss Upweighting) failed before dedicated SRF heads existed. The combination with #2104 is untested and may compound positively or be partially redundant.

---

### 5. pressure-poisson-soft — Precomputed Pressure-Poisson Soft Constraint

**Priority: MEDIUM. ~65 LoC. Expected -2 to -4% p_tan. MEDIUM-HIGH risk.**

**What it is:** Add an auxiliary loss penalizing violations of the pressure Poisson equation: ∇²p ≈ -ρ(u·∇)u. Using precomputed finite-difference stencils on the mesh, compute a soft physics consistency loss between predicted pressure and predicted velocity fields.

**Why it helps p_tan:** In the tandem configuration, pressure distributions in the inter-foil channel are highly sensitive to the exact foil geometry and flow conditions. The model currently learns pressure from data alone with no physics constraint. A Poisson residual loss directly encodes the governing equation, regularizing predictions toward physically consistent solutions — especially valuable in OOD tandem cases where pure data-driven extrapolation is least reliable.

**Confidence:** Moderate in principle (PINNs: Raissi et al. 2019). Low in practice for this mesh topology. Recommend only if ideas 1-4 are exhausted — this is the highest implementation complexity item on the list.

---

## Summary Table

| Rank | Slug | Code size | Risk | Expected p_tan gain | Priority |
|------|------|-----------|------|--------------------:|----------|
| 1 | gap-stagger-spatial-bias | ~18 LoC | MEDIUM | -3 to -7% | HIGH |
| 2 | boundary-id-onehot | ~12 LoC | MEDIUM | -3 to -8% | HIGH |
| 3 | tandem-slice-specialization | ~20 LoC | MEDIUM | -2 to -5% | MEDIUM-HIGH |
| 4 | fore-aft-loss-split | ~8 LoC | LOW | -1 to -3% | MEDIUM |
| 5 | pressure-poisson-soft | ~65 LoC | MEDIUM-HIGH | -2 to -4% | MEDIUM |

**Recommended assignment for nezuko: `gap-stagger-spatial-bias`** — highest confidence of the novel ideas, directly targets the core representational bottleneck (slice routing has no global geometry signal), complements already-merged improvements (#2115 aug, #2104 aft-SRF), no overlap with any in-flight experiment, and is architecturally clean with well-understood initialization properties.

The key insight: PR #2115's gap/stagger augmentation implicitly pressured the model to become robust to inter-foil geometry variation. This experiment is the complementary signal: making the model's attention routing explicitly aware of which inter-foil geometry it is currently processing. These are not competing mechanisms — they are two sides of the same coin, and their combination has the potential to push p_tan below 25.
