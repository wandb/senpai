# SENPAI Research Ideas — Round 16 (2026-04-06 21:30 UTC)

Generated after reviewing all merged experiments from PR #368 through #2215, all failed experiments, and the full Round 15 idea queue.

**Current single-model baseline (PR #2207, TE Coordinate Frame):**
p_in=12.490, p_oodc=7.618, p_tan=28.521, p_re=6.411

**Target for p_tan:** < 28.0 (delta = -0.52 from baseline; single model already beats 16-seed ensemble 29.1)

**Context:** The model is in elite territory. The nearest improvement margins on p_tan are now sub-1%. Standard incremental ideas (loss weighting, input features) are approaching saturation. The next tier of improvement requires either: (a) higher-quality gradient signal on the surface, (b) new structural inductive bias that transfers OOD, or (c) training dynamics changes that let the existing architecture find a better basin.

**In-flight — DO NOT DUPLICATE:**
- #2210 fern: arc-length surface loss reweighting
- #2217 nezuko: fore-SRF skip (fore-foil mean hidden into AftSRF)
- #2219 alphonse: additive fore-aft cross-attention in AftSRF
- #2216 thorfinn: GeoTransolver GALE cross-attention
- #2218 tanjiro: LE coordinate frame input features
- #2220 askeladd: slice diversity regularization (Gram matrix)
- #2213 frieren: wake deficit feature
- #2214 edward: deep supervision on fx_deep

**Already in Round 15 queue — DO NOT DUPLICATE:**
mhc-residuals, domain-split-srf-norm, tandem-feature-cross, surface-arc-length-pe, panel-method-cp-input, chord-adaptive-le-te-loss, kutta-condition-loss, lift-drag-integral-loss, geometry-moment-conditioning, node-type-boundary-embedding

---

## Idea 1: Stochastic Depth Curriculum (Progressive Block Dropping)

**Slug:** `stochastic-depth-curriculum`
**Confidence:** MEDIUM-HIGH
**Complexity:** LOW (~20 LoC)

**What it is:** During training, randomly drop later TransolverBlocks with a linearly decaying probability schedule. In epoch 1, block 3 is dropped with p=0.3 and block 2 with p=0.15. By epoch 80, all probabilities decay to 0. At epoch 80+, all blocks are always active (standard inference). This is "stochastic depth" (Huang et al., 2016) but applied as a curriculum rather than a fixed rate.

**Why it might help here:** The current 3-layer Transolver shows signs of the early-layer bottleneck: block 1 handles geometry, block 3 handles global physics, and if block 3 is noisy early in training, it corrupts the clean geometry encoding from block 1. Progressive block dropping forces blocks 1-2 to develop robust representations before block 3 is trained on top of them. This is the training dynamics reason that PirateNets (PR #2215) was merged — the adaptive residual gating addresses the same pathology from the parameter side; stochastic depth addresses it from the gradient side.

For NACA6416 OOD generalization specifically: the model's failure to generalize tandem pressure likely originates in block 1-2 (where geometry and wake features are extracted), not block 3. Stochastic depth pressure-tests the earlier blocks to be independently predictive, which regularizes against block-3 overfitting.

**Key difference from dropout:** Standard dropout on activations is already absent (no `--dropout` flag). Stochastic depth is a structural dropout at the block level — a much stronger regularizer for deep networks.

**Implementation (~20 LoC in TransolverBlock + training loop):**
```python
# In TransolverBlock.forward:
def forward(self, fx, T, B, N, ...):
    if self.training and self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob:
        return fx  # skip this block entirely
    return ... (normal forward pass)

# In Transolver.__init__, set drop_prob per block:
# block 0 (first): drop_prob = 0.0
# block 1 (second): drop_prob = stochastic_depth_rate * 0.5
# block 2 (third): drop_prob = stochastic_depth_rate
# stochastic_depth_rate decays linearly from --sd_max_rate to 0.0 over --sd_ramp_epochs

# In training loop:
if cfg.stochastic_depth:
    progress = min(1.0, epoch / cfg.sd_ramp_epochs)
    current_rate = cfg.sd_max_rate * (1.0 - progress)
    for i, block in enumerate(model.blocks):
        block.drop_prob = current_rate * (i / len(model.blocks))
```

**Flags:** `--stochastic_depth`, `--sd_max_rate 0.3`, `--sd_ramp_epochs 80`

**Key hyperparameter:** sd_max_rate=0.3 is a reasonable starting point (Huang et al. used 0.5 for ResNets; Transolver has only 3 blocks so be more conservative). sd_ramp_epochs=80 (half the training budget of ~160 epochs) ensures all blocks are active for the second half of training.

**Reference:** Huang et al., "Deep Networks with Stochastic Depth," ECCV 2016. https://arxiv.org/abs/1603.09382. Used in ViT training (DeiT) and physics simulators.

**Suggested experiment:**
```bash
python train.py ... [baseline flags] --stochastic_depth --sd_max_rate 0.3 --sd_ramp_epochs 80
# seeds: {42, 73}
# wandb_group: stochastic-depth-curriculum
```

---

## Idea 2: Pressure Field Laplacian Smoothness Auxiliary Loss

**Slug:** `pressure-laplacian-loss`
**Confidence:** MEDIUM-HIGH
**Complexity:** LOW (~25 LoC)

**What it is:** Add a graph-Laplacian smoothness auxiliary loss on the SURFACE pressure predictions. For each pair of adjacent surface nodes (i, j) on the same foil, penalize `(p_pred_i - p_pred_j)^2 / ||x_i - x_j||^2` scaled by a small weight. This enforces that predicted surface pressure varies smoothly along the surface, which is physically correct everywhere except at the stagnation point and separation point (where gradients are high, but bounded).

**Why it might help here:** The model has no explicit smoothness constraint. For OOD NACA6416 inputs, the surface pressure predictions can develop high-frequency oscillations (ringing artifacts) that increase MAE without violating the L1 loss strongly. A smoothness prior directly constrains the pressure to behave like an aerodynamic pressure distribution — continuous, slowly varying between mesh nodes, with sharp features only at stagnation.

This is distinct from and complementary to the DCT frequency loss (PR #2184, merged): DCT freq loss penalizes high-frequency content in sorted-node space (which has sign/direction issues at LE). Laplacian smoothness uses actual spatial distances between adjacent nodes — it is topology-aware and does not require an arbitrary node ordering.

**Physical motivation:** Inviscid pressure distributions are harmonic functions (Laplace equation). Even in viscous flows, surface pressure is smooth except at separation bubbles. The mesh has ~150 surface nodes per airfoil with ~3-4 adjacent neighbors each — the adjacency graph is a 1D ring, so the Laplacian reduces to a simple finite difference.

**Implementation (~25 LoC):**
```python
# Precompute adjacency: for surface nodes sorted by arc-length, each node
# has index i-1 and i+1 as neighbors (ring topology on airfoil surface)
# In train_step:
if cfg.pressure_laplacian_loss and is_surface.any():
    surf_p_pred = pred_p[is_surface]          # [N_surf]
    surf_xy = x[is_surface, 0:2]              # [N_surf, 2]
    
    # Sort by angle from centroid (consistent ring ordering)
    centroid = surf_xy.mean(dim=0, keepdim=True)
    angles = torch.atan2(surf_xy[:, 1] - centroid[0, 1],
                          surf_xy[:, 0] - centroid[0, 0])
    order = angles.argsort()
    surf_p_pred_sorted = surf_p_pred[order]
    surf_xy_sorted = surf_xy[order]
    
    # Finite difference Laplacian (ring topology)
    dp = surf_p_pred_sorted[1:] - surf_p_pred_sorted[:-1]
    ds = (surf_xy_sorted[1:] - surf_xy_sorted[:-1]).norm(dim=-1)
    grad_sq = (dp / (ds + 1e-5)) ** 2
    laplacian_loss = cfg.laplacian_weight * grad_sq.mean()
    loss = loss + laplacian_loss
```

**Flags:** `--pressure_laplacian_loss`, `--laplacian_weight 0.01`

**Tuning notes:** Start at lambda=0.01. The smoothness loss should be ~0.1-1.0x the magnitude of the main surface loss. Monitor `train/laplacian_loss` in W&B. If the loss dominates (>5x surface loss), reduce lambda. If it's near zero (model already smooth), increase lambda to 0.05.

**Reference:** Graph Laplacian regularization is standard in semi-supervised learning (Zhou et al. 2004) and physics-informed neural networks (where it enforces PDE smoothness on the interior). Applied to surface aerodynamic predictions in GNN-based surrogate models.

---

## Idea 3: Learnable Input Channel Uncertainty Weights (Aleatoric Feature Gating)

**Slug:** `input-channel-uncertainty`
**Confidence:** MEDIUM**
**Complexity:** LOW (~15 LoC)**

**What it is:** Add a learnable 24-dimensional weight vector `w` (one per input feature channel), initialized to all-ones, that multiplies the input features before standardization: `x_gated = x * w[None, None, :]`. Regularize with an L2 penalty on `(w - 1)^2` to prevent degenerate solutions. The optimizer learns which input channels are most informative for pressure prediction, and can down-weight noisy or redundant channels.

**Why it might help here:** The 24-dimensional input includes coordinates (2), DSDF (8), boundary ID encodings, Reynolds number, AoA, gap, stagger, Fourier PE. Not all are equally informative for OOD pressure. In particular, for NACA6416 OOD: the foil-2 DSDF channels (indices 6-9) encode the foil-2 geometry, but for single-foil OOD (p_re, p_oodc), these channels are zero-padded and constitute dead weight in the input projection.

Learnable channel weights allow the model to discover this structure automatically. A weight near 0 for foil-2 channels on single-foil inputs would effectively zero out irrelevant features — a learned form of the domain-aware gating that PCGrad achieves from the gradient side.

This is directly analogous to feature selection in classical ML (LASSO, attention-based feature weighting) but applied at the input level of a neural network. Related to "Learned Input Masking" in feature-robust models.

**Implementation (~15 LoC):**
```python
# In Transolver.__init__:
if cfg.input_channel_uncertainty:
    self.input_channel_weights = nn.Parameter(torch.ones(n_x))

# In Transolver.forward, before input_encode:
if cfg.input_channel_uncertainty:
    # Apply channel weights (broadcast over batch and nodes)
    x_proc = x * self.input_channel_weights[None, None, :]

# In loss computation:
if cfg.input_channel_uncertainty:
    channel_reg = cfg.channel_reg_weight * ((model.input_channel_weights - 1.0) ** 2).sum()
    loss = loss + channel_reg
```

**Flags:** `--input_channel_uncertainty`, `--channel_reg_weight 0.001`

**Monitoring:** Log `model.input_channel_weights` to W&B at each epoch. The pattern of which channels get high/low weights is scientifically interesting in its own right — it reveals which input features the model finds most/least useful.

**Note:** This is complementary to `--tandem_feature_cross` (Round 15 queue) which gates by CONFIGURATION (gap, stagger) — this gates by CHANNEL (feature type). Both are safe-init (ones vs sigmoid(5)).

---

## Idea 4: Fore-Foil Stagnation Point Feature

**Slug:** `fore-stagnation-feature`
**Confidence:** MEDIUM-HIGH
**Complexity:** MEDIUM (~30 LoC)

**What it is:** For each mesh node, compute the angular distance to the fore-foil stagnation point (the node with maximum local pressure, or equivalently minimum velocity). Add this as an input feature (+1 channel). The stagnation point is the aerodynamic "anchor" — pressure distributions are organized relative to it, and for NACA6416 (with 6% camber), it shifts to a different location than for NACA0012.

**Why it might help here:** The TE coordinate frame (PR #2207, merged) showed that explicit geometric reference points dramatically help OOD generalization. The trailing edge is one reference; the LEADING EDGE / STAGNATION POINT is the complementary reference. But the LE coordinate frame (#2218, in-flight for tanjiro) uses the geometric LE (minimum x-coord), which is not the same as the aerodynamic stagnation point for cambered airfoils like NACA6416 (where stagnation shifts below the geometric LE).

The stagnation point is estimated from the input at training time: it is the surface node with the smallest velocity magnitude in the target field. Alternatively, it can be approximated as the node where the DSDF gradient direction aligns most closely with the freestream direction. This provides the model with an AoA-corrected, camber-corrected geometric reference that is aerodynamically meaningful.

**Key insight for OOD:** NACA6416's 6% camber shifts the stagnation point from the geometric LE by ~2-3% chord at typical angles of attack. The LE frame (#2218) uses the geometric LE and will therefore mislocalize the stagnation point for NACA6416 — the stagnation feature uses the AERODYNAMIC stagnation point.

**Implementation:**
```python
# At batch construction time:
# For each sample, find the surface node with min(||u_target||):
u_surf = target_uvp[is_surface, 0:2]  # [N_surf, 2]
vel_mag = u_surf.norm(dim=-1)          # [N_surf]
stag_idx = vel_mag.argmin()
stag_xy = x[is_surface][stag_idx, 0:2]  # [2]

# For each node, compute angular distance to stagnation:
dx = x[:, 0:2] - stag_xy[None, :]   # [N, 2]
stag_dist = dx.norm(dim=-1)          # [N] — Euclidean distance to stagnation

# Append stag_dist as new input channel (normalized by chord length)
x_augmented = torch.cat([x, stag_dist[:, None] / chord_length], dim=-1)
```

**Important caveat:** This uses the TARGET velocity to find the stagnation point — which is valid at training time (targets are available) and at test time (targets are available for validation). At production inference without targets, a different stagnation point estimator is needed (panel method or AoA-based formula). For validation purposes this is fine.

**Flags:** `--fore_stagnation_feature` (boolean). Adds 1 input channel (n_x=25).

**Reference:** Aerodynamics stagnation point estimation: Anderson "Fundamentals of Aerodynamics" Ch. 3. The stagnation point as a coordinate reference is used in thin-airfoil pressure distribution formulas.

---

## Idea 5: Randomized Fourier Features for Boundary Condition Encoding

**Slug:** `rff-boundary-encoding`
**Confidence:** MEDIUM
**Complexity:** LOW (~20 LoC)

**What it is:** Replace the current fixed Fourier positional encoding (sin/cos of coordinates) with **Randomized Fourier Features (RFF)** — a set of randomly sampled frequencies from a Gaussian distribution, fixed after initialization. The key difference: standard Fourier PE uses a fixed frequency grid (harmonics), while RFF samples frequencies from a data-driven distribution that can be fit to the spatial frequency content of the CFD fields.

**Why it might help here:** The current Fourier PE is applied to the raw (x, y) coordinates. For NACA6416 OOD, the relevant spatial frequencies of the pressure distribution differ from NACA0012 (higher camber → different dominant harmonics in the pressure field). Fixed harmonic PE cannot adapt to this. RFF with frequencies sampled from a distribution fit to the training pressure field's spatial power spectrum would provide a more adaptive encoding.

Specifically: fit a Gaussian to the spatial frequency content of the pressure field across the training set. Sample RFF frequencies from this distribution. The RFF approximates a kernel function over the node positions, which is more expressive than a fixed harmonic grid.

**Implementation (~20 LoC):**
```python
# Precompute: fit frequency distribution from training data (offline, once)
# OR: use a fixed isotropic Gaussian with sigma=1/median_mesh_spacing
# 
# During training:
class RandomFourierEncoder(nn.Module):
    def __init__(self, n_input_dims=2, n_rff=64, sigma=10.0):
        super().__init__()
        # Fixed after init (not learned)
        W = torch.randn(n_input_dims, n_rff) * sigma
        b = torch.rand(n_rff) * 2 * math.pi
        self.register_buffer('W', W)
        self.register_buffer('b', b)
    
    def forward(self, x_coords):
        # x_coords: [B, N, 2]
        proj = x_coords @ self.W + self.b[None, None, :]
        return torch.cat([proj.cos(), proj.sin()], dim=-1) * (2 / self.W.shape[1]) ** 0.5
```

**Note on the current PE:** If `train.py` already applies a fixed Fourier PE to coordinates, this would REPLACE that component. Inspect the current PE implementation first — if it's already randomized, skip this idea.

**Flags:** `--rff_boundary_encoding`, `--rff_n_features 64`, `--rff_sigma 10.0`

**Reference:** Rahimi & Recht, "Random Features for Large-Scale Kernel Machines," NeurIPS 2007. Applied to neural fields in Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions," NeurIPS 2020 (NeRF paper).

---

## Idea 6: Adaptive Surface Loss Masking — Drop Low-Gradient Nodes

**Slug:** `surface-loss-adaptive-mask`
**Confidence:** MEDIUM
**Complexity:** LOW (~20 LoC)

**What it is:** During training, progressively focus the surface loss on the nodes with the highest prediction error. After each epoch, compute per-node surface pressure error, keep the top-K% highest-error nodes "active" in the surface loss, and downweight (but don't zero) the rest. Start with 100% active at epoch 0 and reduce to 60% active by epoch 80, then hold at 60% through epoch 160. This is a curriculum version of hard example mining applied at the node level.

**Why it might help here:** The surface has ~150 nodes per airfoil, but the majority of the MAE is concentrated in ~20-30 nodes near the LE, TE, and suction peak. By focusing the surface loss gradient on these high-error regions progressively, the model gets a stronger gradient signal where it matters most. The "ramp" from 100% to 60% ensures early training uses all nodes (stability), then transitions to hard mining.

This is complementary to arc-length reweighting (#2210, in-flight) which corrects for mesh density bias — this idea is purely error-based, not geometry-based. It targets nodes that are CURRENTLY hard to predict, which adapts over training.

**Distinction from other loss ideas:**
- #2210 (arc-length): geometry-based weighting (corrects mesh density)
- #2184 (DCT freq): frequency-domain weighting (pushes on high-freq features)  
- This: error-based adaptive masking (focuses on current worst predictions)
- Round 15 chord-adaptive-le-te-loss: hardcoded LE/TE zones

**Implementation (~20 LoC):**
```python
# After computing per-node surface loss (before mean reduction):
if cfg.surface_loss_adaptive_mask:
    # p_error: [B, N_surf] per-node pressure absolute error
    p_error = (pred_p_surf - true_p_surf).abs().detach()
    
    # Active fraction ramp: 1.0 at epoch 0 → 0.6 at epoch 80 → constant 0.6
    active_frac = max(0.6, 1.0 - 0.4 * min(1.0, epoch / 80))
    
    # Per-sample, keep top active_frac by error
    k = int(p_error.shape[-1] * active_frac)
    threshold, _ = p_error.kthvalue(p_error.shape[-1] - k + 1, dim=-1, keepdim=True)
    weights = (p_error >= threshold).float() * (1.0 / active_frac)  # normalize
    
    surface_loss = (surface_loss_per_node * weights).mean()
```

**Flags:** `--surface_loss_adaptive_mask`, `--slam_min_active 0.6`, `--slam_ramp_epochs 80`

**Risk:** If hard mining starts too early (before the model has learned basic pressure distributions), it might destabilize training by focusing on noise. The ramp from 100% mitigates this.

**Reference:** Online Hard Example Mining (OHEM), Shrivastava et al., CVPR 2016. Curriculum learning, Bengio et al. 2009.

---

## Idea 7: TE Velocity Coupling — Wake Velocity at TE as Auxiliary Target

**Slug:** `te-velocity-coupling`
**Confidence:** MEDIUM
**Complexity:** MEDIUM (~30 LoC)

**What it is:** Add an auxiliary loss that specifically targets the prediction of velocity (Ux, Uy) at the trailing-edge node of the fore-foil. The TE velocity is the "outlet condition" for the boundary layer and determines wake strength — it is the single most important quantity for aft-foil loading in tandem configurations.

**Why it might help here:** The current surface loss weights all surface nodes equally. For tandem configurations, the fore-foil TE velocity is geometrically the most influential prediction: the wake shed from the fore-TE impinges directly on the aft-foil. An error in TE velocity prediction propagates multiplicatively to p_tan.

Rather than applying an extra weight, this creates a DEDICATED mini-loss term for the TE velocity of the fore-foil, separate from the standard surface loss. This isolates the gradient signal for the hardest causal point in the tandem flow interaction.

**Implementation:**
```python
# In train_step, for tandem samples only:
if cfg.te_velocity_coupling and is_tandem:
    # Find fore-foil TE node (max x-coord among fore-foil surface nodes)
    fore_surf_mask = (boundary_id == 5) | (boundary_id == 6)
    fore_surf_x = x[fore_surf_mask, 0]
    te_idx_local = fore_surf_x.argmax()
    
    pred_te_vel = pred_uvp[fore_surf_mask][te_idx_local, 0:2]   # [2]
    true_te_vel = target_uvp[fore_surf_mask][te_idx_local, 0:2]
    
    te_vel_loss = cfg.te_vel_weight * F.l1_loss(pred_te_vel, true_te_vel)
    loss = loss + te_vel_loss
```

**Flags:** `--te_velocity_coupling`, `--te_vel_weight 0.5`

**Physical motivation:** In thin-airfoil theory, the TE velocity determines the circulation, and the circulation determines the pressure difference across the foil. For viscous flows (as in CFD), TE velocity = wake source strength. Even a small improvement in fore-foil TE velocity prediction should cascade to better aft-foil pressure predictions.

**Caveat:** The TE node is a single point — gradient signal is noisy. Consider using the mean of the 3-5 nodes nearest the TE rather than a single node. The TE coord frame (#2207) already locates the TE — that code can be reused here.

---

## Idea 8: Reynolds Number Curriculum — Train Easy Re First

**Slug:** `re-curriculum`
**Confidence:** MEDIUM
**Complexity:** LOW (~15 LoC)

**What it is:** Apply a curriculum where early epochs emphasize moderate Reynolds number samples (Re in the training distribution center) and gradually add extreme-Re samples. Specifically: weight each sample's loss by `w(Re) = 1 + (max(0, |Re - Re_center| - sigma) / (Re_max - Re_center))^2 * lambda * progress(epoch)`, where `progress(epoch)` ramps from 0 (no extra weight) to 1 over the first 80 epochs, then stays at 1. This is a competency-based curriculum — the model must learn the core pressure physics before being asked to generalize to extreme Reynolds numbers.

**Why it might help here:** The `val_ood_re` split (p_re) evaluates on Reynolds numbers outside the training distribution. The `val_tandem_transfer` split (p_tan) includes extreme-Re tandem cases. Currently all samples have equal loss weight regardless of Re difficulty. A curriculum that starts with easy-Re examples and progressively adds hard-Re examples follows the well-validated curriculum learning principle (Bengio 2009): start with examples the model can learn from, then expose to harder ones.

**Distinction from pcgrad_extreme_pct:** PCGrad (--pcgrad_3way) does gradient surgery for extreme samples; it doesn't modify the ORDER or WEIGHT of sample exposure. This curriculum changes WHEN the model sees hard Re samples in training — orthogonal mechanisms.

**Implementation (~15 LoC):**
```python
# In training loop, per-sample loss weighting:
if cfg.re_curriculum:
    re_vals = x[:, 0, re_feature_idx]  # [B] Reynolds numbers (standardized)
    # Unstandardize: re_phys = re_vals * re_std + re_mean
    re_phys = re_vals * dataset_re_std + dataset_re_mean
    re_center = dataset_re_median
    re_range = dataset_re_std * 2
    
    # Difficulty score per sample
    re_difficulty = torch.clamp((re_phys - re_center).abs() / re_range, 0, 1)
    
    # Curriculum progress (0 to 1 over ramp_epochs)
    progress = min(1.0, epoch / cfg.re_curriculum_ramp_epochs)
    
    # Weight: baseline=1, hard samples boosted by progress*lambda
    sample_weights = 1.0 + re_difficulty ** 2 * progress * cfg.re_curriculum_lambda
    loss = (per_sample_loss * sample_weights).mean()
```

**Flags:** `--re_curriculum`, `--re_curriculum_lambda 2.0`, `--re_curriculum_ramp_epochs 80`

**Reference:** Curriculum Learning, Bengio et al. ICML 2009. Applied in NLP (BERT), RL, and CFD simulation training.

---

## Idea 9: Fore-Foil Pressure Prediction as Aft-Foil Conditioning Input

**Slug:** `fore-pressure-conditioning`
**Confidence:** HIGH
**Complexity:** MEDIUM (~40 LoC)

**What it is:** After the main Transolver forward pass (which predicts pressure for ALL nodes), extract the predicted fore-foil surface pressure and USE IT as an additional conditioning input to the AftSRF head. Specifically: pool the predicted fore-foil surface pressure into a compact vector (mean, std, and 4 DCT coefficients = 6 scalars), then inject these as a learned conditioning offset (via a small MLP projection) into the AftSRF input features before the MLP refinement.

**Why it might help here:** The current AftSRF (#2104) takes aft-foil backbone hidden states as input. These encode geometric and flow features, but the model has NO DIRECT ACCESS to what the fore-foil pressure prediction looks like when correcting the aft-foil. For NACA6416 OOD, the fore-foil Cp distribution is systematically different from training foils — the aft-foil correction should adapt to this.

This is structurally different from the cross-attention approaches (#2219 alphonse — fore backbone hidden states; #2217 nezuko — fore mean hidden states): it conditions on the PREDICTED PRESSURE (a semantically meaningful quantity), not the backbone hidden states (which are high-dimensional and harder to interpret). The 6-scalar DCT summary is compact and provides a physics-meaningful description of the fore-foil Cp curve shape.

**Zero-regression guarantee:** The fore-pressure conditioning is added to the AftSRF input with a zero-initialized projection. At epoch 0, it contributes zero correction. As training proceeds, the optimizer learns to use fore-foil Cp statistics to modulate aft-foil corrections.

**Implementation (~40 LoC):**
```python
# After backbone forward pass (pred_uvp computed):
if cfg.fore_pressure_conditioning:
    # Extract fore-foil surface pressure predictions (post-asinh if applicable)
    fore_surf_mask = (boundary_id == 5) | (boundary_id == 6)
    p_pred_fore = pred_uvp[fore_surf_mask, 2]  # [N_fore_surf] pressure
    
    # Sort by arc-length (use fore-foil surface x-coord as proxy)
    sort_idx = x[fore_surf_mask, 0].argsort()
    p_sorted = p_pred_fore[sort_idx]
    
    # Compact descriptor: mean, std, + first 4 DCT coefficients
    p_mean = p_sorted.mean(keepdim=True)
    p_std = p_sorted.std(keepdim=True)
    dct_coeffs = torch.fft.rfft(p_sorted - p_mean)[:4].real  # [4]
    fore_pressure_descriptor = torch.cat([p_mean, p_std, dct_coeffs])  # [6]
    
    # Inject into AftSRF:
    # AftSRF.fore_p_proj: nn.Linear(6, n_hidden), zero-initialized
    fore_cond = model.aft_foil_srf.fore_p_proj(fore_pressure_descriptor)  # [n_hidden]
    # fore_cond is added to each aft-foil node's input before the MLP
```

**Flags:** `--fore_pressure_conditioning` (boolean)

**Key technical note:** This is a TWO-STAGE forward pass: (1) compute backbone predictions, (2) extract fore-foil pressure, (3) condition AftSRF. The backbone is NOT run twice — only the AftSRF receives the extra conditioning. The fore-foil pressure used is the FINAL backbone prediction (including all SRF corrections for fore nodes), so it's the best available estimate.

---

## Idea 10: Mixup Between Same-Geometry Different-Re Samples

**Slug:** `re-mixup-augmentation`
**Confidence:** MEDIUM
**Complexity:** MEDIUM (~35 LoC)

**What it is:** During training, for a fraction of batches, replace pairs of single-foil samples with the same NACA geometry but different Reynolds numbers with their INTERPOLATED versions: `x_mix = lambda * x_A + (1-lambda) * x_B`, `y_mix = lambda * y_A + (1-lambda) * y_B`, where `lambda ~ Beta(0.4, 0.4)`. This is standard Mixup (Zhang et al., 2018) but applied ONLY between samples that share the same airfoil geometry but have different Re numbers — a physics-constrained Mixup that respects the Re-continuity of pressure distributions.

**Why it might help here:** `val_ood_re` (p_re=6.411) evaluates on Reynolds numbers outside the training distribution. Standard Mixup across random samples creates physically inconsistent interpolations. Physics-constrained Mixup between same-geometry different-Re samples creates physically plausible interpolated training examples that fill gaps in the Re distribution. The pressure field IS approximately linear in small Re changes (at low Mach number), so the interpolated target is physically valid.

For p_re improvement: the current p_re=6.411 is already strong, but OOD-Re generalization is a long-standing challenge. Mixup that explicitly regularizes across the Re axis is a direct attack on this vulnerability.

**Implementation (~35 LoC):**
```python
# In data loader or batch construction:
# 1. Group samples by geometry (NACA code, AoA bucket)
# 2. Within each group, pair samples by nearest Re
# 3. In each batch, randomly replace ~30% of single-foil samples with 
#    Mixup of two same-geometry different-Re samples

# Mixup forward pass:
lambda_mix = beta_distribution.sample()  # lambda ~ Beta(0.4, 0.4)
x_mix = lambda_mix * x_A + (1 - lambda_mix) * x_B
y_mix = lambda_mix * y_A + (1 - lambda_mix) * y_B
# Feed x_mix to model, compute loss against y_mix
```

**Key constraint:** Only mix between the SAME airfoil geometry (same NACA code + AoA bin) to preserve physics consistency. Mixing across geometries would create invalid training targets.

**Flags:** `--re_mixup`, `--re_mixup_prob 0.3`, `--re_mixup_alpha 0.4`

**Reference:** Zhang et al., "Mixup: Beyond Empirical Risk Minimization," ICLR 2018. https://arxiv.org/abs/1710.09412. Physics-constrained Mixup for PDEs: Doan et al. 2022.

---

## Idea 11: FiLM Conditioning of SurfaceRefinementHead on Reynolds Number

**Slug:** `srf-re-film-conditioning`
**Confidence:** MEDIUM-HIGH
**Complexity:** LOW (~20 LoC)

**What it is:** Condition the `SurfaceRefinementHead` (the main srf_head, not AftSRF) on the Reynolds number via Feature-wise Linear Modulation (FiLM): after each MLP layer in the srf_head, apply `x_out = gamma(Re) * x_out + beta(Re)`, where `gamma` and `beta` are learned linear functions of Re. Zero-init both functions so the first pass is baseline-identical.

**Why it might help here:** p_re=6.411 is the OOD Reynolds number metric. The current srf_head applies an identical correction to surface nodes regardless of Reynolds number. But the viscous correction to surface pressure is strongly Re-dependent: at higher Re, the boundary layer is thinner, the suction peak is sharper, and the TE pressure recovery is different. FiLM conditioning on Re gives the srf_head explicit knowledge of which viscous regime it's operating in.

This is a targeted application of conditional normalization (FiLM) to the single component most responsible for p_re accuracy. FiLM is proven in multi-domain adaptation settings (MUNIT, AdaIN) and has a safe zero-initialization.

This is distinct from DomainLayerNorm (backbone, tandem vs. single): FiLM on Re in srf_head is a CONTINUOUS conditioning on a scalar value (not a discrete domain flag), affecting the REFINEMENT phase (not the backbone).

**Implementation (~20 LoC):**
```python
class FiLMSurfaceRefinementHead(nn.Module):
    def __init__(self, n_hidden, n_layers=3, hidden_dim=192, film_re=False):
        ...
        if film_re:
            # One FiLM layer per MLP layer, zero-initialized
            self.film_gamma = nn.Linear(1, hidden_dim)
            self.film_beta = nn.Linear(1, hidden_dim)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)  # gamma=0 → effective scale=1
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)   # beta=0 → no shift
    
    def forward(self, x_surf, re_val):
        # re_val: [B] standardized Reynolds number
        for i, layer in enumerate(self.mlp_layers):
            x_surf = layer(x_surf)
            if self.film_re and i == 1:  # apply after middle layer
                gamma = 1.0 + self.film_gamma(re_val[:, None])  # [B, 1, hidden_dim]
                beta = self.film_beta(re_val[:, None])
                x_surf = gamma * x_surf + beta
        return x_surf
```

**Flags:** `--srf_re_film` (boolean)

**Reference:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018. Zero-init FiLM is used in T2I-Adapter, ControlNet (Stable Diffusion), and domain adaptation for physics models.

---

## Idea 12: Predicted Circulation as Latent Bottleneck (Kutta-Joukowski Auxiliary Task)

**Slug:** `circulation-auxiliary-task`
**Confidence:** MEDIUM
**Complexity:** MEDIUM (~35 LoC)

**What it is:** Add a small auxiliary head that predicts the circulation Gamma from the backbone hidden states. Circulation is computed from the predicted pressure via the Kutta-Joukowski theorem: `Cl = 2 * Gamma / (U_inf * chord)`, and we can compute the "true" circulation from the target pressure via surface integration. The auxiliary head must predict this scalar correctly, which forces the backbone to encode physically meaningful circulation information.

**Why it might help here:** The pressure distribution on an airfoil is dominated by its circulation (Kutta-Joukowski). If the model can accurately predict circulation from the backbone hidden states, it has implicitly learned the most important aerodynamic quantity. For NACA6416, the circulation is higher than for NACA0012 (more camber = more lift = stronger pressure differential), and the model must learn this distinction for OOD transfer.

This is a multi-task learning approach that provides a GLOBAL scalar target (circulation) alongside the LOCAL per-node pressure targets. The scalar task is easy to compute and forces the model to develop a globally coherent pressure representation. It's related to Cl/Cd integral loss (Round 15 queue) but differs in mechanism: Cl/Cd loss backpropagates through the predicted pressure; this predicts circulation from the BACKBONE FEATURES (a different information bottleneck).

**Implementation (~35 LoC):**
```python
# In Transolver:
if cfg.circulation_auxiliary:
    # Pool fore-foil surface hidden states to get global rep
    self.circulation_head = nn.Sequential(
        nn.Linear(n_hidden, 64), nn.GELU(),
        nn.Linear(64, 1)  # predict scalar Gamma/U_inf
    )

# In training forward:
if cfg.circulation_auxiliary:
    fore_surf_hidden = fx[fore_surf_mask].mean(dim=1)  # [B, n_hidden]
    circ_pred = model.circulation_head(fore_surf_hidden)  # [B, 1]
    
    # True circulation from target pressure (surface integral)
    p_surf = target_uvp[fore_surf_mask, 2]  # [B, N_surf]
    normals_x = compute_surface_normals_x(x[fore_surf_mask])  # [B, N_surf]
    ds = compute_arc_length_increments(x[fore_surf_mask])  # [B, N_surf]
    circ_true = -(p_surf * normals_x * ds).sum(dim=-1) / (0.5 * u_inf ** 2)
    
    circ_loss = cfg.circ_weight * F.l1_loss(circ_pred.squeeze(), circ_true)
    loss = loss + circ_loss
```

**Flags:** `--circulation_auxiliary`, `--circ_weight 0.1`

**Reference:** Kutta-Joukowski theorem: classical aerodynamics (Anderson "Fundamentals of Aerodynamics," Ch. 3). Auxiliary task learning with physics quantities: Pathak et al. 2021 "Physics-informed machine learning."

---

## Summary Table

| # | Slug | Target Metric | Confidence | Complexity | Key Mechanism |
|---|------|--------------|------------|------------|---------------|
| 1 | `stochastic-depth-curriculum` | p_tan, p_oodc | MED-HIGH | LOW | Progressive block dropping → better early rep quality |
| 2 | `pressure-laplacian-loss` | p_tan, p_oodc | MED-HIGH | LOW | Graph Laplacian smoothness on surface pressure |
| 3 | `input-channel-uncertainty` | p_re, p_oodc | MED | LOW | Learned per-channel gating → auto-feature selection |
| 4 | `fore-stagnation-feature` | p_tan | MED-HIGH | MED | Aerodynamic stagnation point as input reference |
| 5 | `rff-boundary-encoding` | p_tan, p_oodc | MED | LOW | Random Fourier features vs fixed harmonic PE |
| 6 | `surface-loss-adaptive-mask` | p_tan | MED | LOW | Error-curriculum: focus loss on hardest surface nodes |
| 7 | `te-velocity-coupling` | p_tan | MED | MED | Dedicated loss for fore-foil TE velocity |
| 8 | `re-curriculum` | p_re | MED | LOW | Curriculum on Reynolds number difficulty |
| 9 | `fore-pressure-conditioning` | p_tan | HIGH | MED | Predicted fore-foil Cp as conditioning for AftSRF |
| 10 | `re-mixup-augmentation` | p_re | MED | MED | Physics-constrained Mixup across Re for same geometry |
| 11 | `srf-re-film-conditioning` | p_re | MED-HIGH | LOW | FiLM condition srf_head on Re (zero-init, safe) |
| 12 | `circulation-auxiliary-task` | p_tan | MED | MED | Auxiliary prediction of Kutta-Joukowski circulation |

## Top Picks for Immediate Assignment

**For a student looking to improve p_tan specifically:**
1. **`fore-pressure-conditioning`** (#9) — HIGH confidence, physically very well motivated. The predicted fore-foil Cp directly causes aft-foil loading. Zero-init safe. Orthogonal to all in-flight experiments.
2. **`stochastic-depth-curriculum`** (#1) — MED-HIGH confidence, LOW complexity (~20 LoC). Attacks the block-level training dynamics issue from the gradient side (PirateNets attacked it from the parameter side).
3. **`pressure-laplacian-loss`** (#2) — MED-HIGH confidence, LOW complexity (~25 LoC). Novel in this codebase; provides topology-aware smoothness signal that DCT freq loss lacks.

**For a student looking to improve p_re or p_oodc:**
1. **`srf-re-film-conditioning`** (#11) — MED-HIGH confidence, LOW complexity. Safe zero-init FiLM on the exact component responsible for p_re.
2. **`re-curriculum`** (#8) — MED confidence, LOW complexity. Classical curriculum learning applied to the Re difficulty axis.

**For askeladd (current student):**
Recommend **`fore-pressure-conditioning`** as the next assignment after #2220 (Slice Diversity Reg) completes. It is HIGH confidence, medium complexity, orthogonal to all in-flight work, and directly targets the p_tan gap via a novel conditioning mechanism not yet tried anywhere in the codebase.
