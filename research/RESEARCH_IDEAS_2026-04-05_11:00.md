# Research Ideas — 2026-04-05 11:00

## Context

**Current single-model baseline (PR #2130, GSB + PCGrad, 2-seed avg):**
- p_in = 13.05
- p_oodc = 7.70
- p_tan = 28.60  <-- PRIMARY TARGET
- p_re = 6.55

**What the baseline already uses:**
asinh pressure (s=0.75), residual prediction, SRF MLP (192h, 3L), aft-foil SRF context head (K=8 KNN), gap/stagger aug (σ=0.02), DSDF2 aug (σ=0.05), PCGrad 2-way, gap/stagger spatial bias (GSB).

**In-flight (do NOT duplicate):**
#2150 DSDF2 sigma 0.03/0.08, #2151 EMA start epoch, #2152 Aug annealing, #2153 gap/stagger sigma 0.03, #2154 T_max 140/180, #2149 LR sweep, #2131 Tandem slice carve-out.

**p_tan diagnosis:**
p_tan = 28.60 measures surface MAE on the NACA6416+rear-foil tandem pair (raceCar_tandem Part2). NACA6416 is completely absent from tandem training — the front foil is unseen OOD geometry. The front foil (boundary IDs 5,6) is especially hard to refine because it has unknown camber/thickness profile. The aft foil (boundary ID=7) benefits from the AftSRF context head but is inside a wake shaped by the unseen fore-foil. The fundamental gap: the model must infer NACA6416 surface pressure from DSDF features alone, without having seen that profile in tandem context.

**Critical dead ends (do NOT re-propose variants):**
- Per-foil physics normalization (#2136 — tried, failed: Umag_aft too unstable in tight-gap cases)
- Fore-foil dedicated SRF head (split from shared SRF) (#2117, #2124 — both failed, hurt p_in)
- FiLM on gap/stagger for aft SRF (#2119 — catastrophic p_oodc regression)
- Fourier PE (#2106 — no gain, already in baseline via learnable freqs)
- Boundary ID one-hot input (#2118 — tried, no gain)
- Contrastive tandem-single regularization (#2109 — failed)
- Inverse distance volume weighting (#2112 — failed)

---

## Idea 1 — Foil-Shape Similarity Feature for Spatial Bias (foil_shape_sim_feat)

### What it is

Add a scalar "foil shape similarity" feature to the spatial bias input, extending raw_xy from 6D to 7D. The feature encodes the cosine similarity between the foil-1 DSDF vector and foil-2 DSDF vector, computed at sample level:

```python
# Before normalization, at the raw DSDF computation site:
dsdf1 = x[:, 0, 2:6]   # foil-1 DSDF (first 4 channels), [B, 4]
dsdf2 = x[:, 0, 6:10]  # foil-2 DSDF, [B, 4]
# Cosine similarity: how similar are the foil shapes?
dsdf1_norm = F.normalize(dsdf1, dim=-1)
dsdf2_norm = F.normalize(dsdf2, dim=-1)
shape_sim = (dsdf1_norm * dsdf2_norm).sum(dim=-1, keepdim=True)  # [B, 1]
# Broadcast to all nodes
shape_sim_feat = shape_sim.unsqueeze(1).expand(-1, N, -1)  # [B, N, 1]
```

This scalar tells the spatial routing network: "are the two foils geometrically similar?" For tandem training samples (NACA2412/9412 front, identical to back), similarity is high (~1.0). For p_tan evaluation (NACA6416 front), the shape similarity is lower — the network can use this to route tandem nodes differently when foil shapes diverge.

Currently `raw_xy = [x, y, curvature, dist_to_surface, gap, stagger]`. The new version adds `shape_sim` at index 6, giving `spatial_bias_input_dim=7`.

### Why it might help p_tan

The core difficulty with p_tan is that NACA6416 (6% camber, 16% thickness) is geometrically distinct from NACA2412 and NACA9412 used in training. The gap/stagger scalars already help the GSB condition on configuration geometry. This adds a feature encoding the foil-shape relationship, which is the missing piece: the spatial routing network can detect that "these foils are dissimilar" and adjust which physics slices it routes to. This is a lightweight, zero-param overhead extension of a proven mechanism (the GSB is already in the baseline and works).

For single-foil samples, foil-2 DSDF is zero, so shape_sim will be 0 or undefined — need to guard this: `shape_sim = torch.where(is_tandem_mask, shape_sim, torch.zeros_like(shape_sim))`.

### Code change locations

1. **Config dataclass** (~line 950): Add `foil_shape_sim_bias: bool = False`

2. **Transolver `__init__`** (line 774): Change `spatial_bias_input_dim=6 if gap_stagger_spatial_bias else 4` to:
   ```python
   if cfg.foil_shape_sim_bias:
       _sbi = 7
   elif cfg.gap_stagger_spatial_bias:
       _sbi = 6
   else:
       _sbi = 4
   spatial_bias_input_dim=_sbi
   ```
   Also add zero-init of the new column 6 in `self.blocks[0].spatial_bias[0].weight[:, 6:]`.

3. **Transolver `forward`** (lines 887-892), after computing `gap_stagger`:
   ```python
   if self.foil_shape_sim_bias:
       dsdf1 = x[:, 0:1, 2:6]   # [B, 1, 4]
       dsdf2 = x[:, 0:1, 6:10]  # [B, 1, 4]
       d1_n = F.normalize(dsdf1, dim=-1)
       d2_n = F.normalize(dsdf2, dim=-1)
       shape_sim = (d1_n * d2_n).sum(dim=-1, keepdim=False)  # [B, 1]
       _is_tan_sim = (x[:, 0, 21].abs() > 0.01)
       shape_sim = shape_sim * _is_tan_sim.float()[:, None]
       shape_sim_feat = shape_sim.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, N, 1]
       raw_xy = torch.cat([raw_xy, shape_sim_feat], dim=-1)  # [B, N, 7]
   ```
   Note: `x` here is already the full feature tensor (post `feature_cross`), and DSDF indices are stable relative to the base 24-dim input.

4. **model_config dict** (line 1232): Add `foil_shape_sim_bias=cfg.foil_shape_sim_bias` and pass it to `Transolver.__init__`.

### Hyperparameters and risk

- No new hyperparameters; zero-init on new input column ensures routing is unchanged at initialization
- Risk: LOW. Zero-init guarantees the model starts at the exact baseline; the feature can only help
- Gotcha: x is post-normalization when this runs in `forward()`. The DSDF channels (indices 2:10) are z-score normalized at this point. Cosine similarity is invariant to scale, so normalization doesn't affect the result — but sanity-check that normalized DSDF vectors have reasonable magnitudes
- Single-foil samples will have zero foil-2 DSDF, making shape_sim = 0 — this is correct (the feature is "not applicable" for single foil)

### Expected impact

-1% to -3% p_tan. This is a targeted information injection for the one specific condition that's hardest: dissimilar foil pairs (NACA6416 + rear foil). Low risk due to zero-init.

### LoC estimate

~30 LoC (Config flag + model __init__ + forward change + model_config dict entry).

---

## Idea 2 — DSDF Channel-Level Dropout Augmentation for Tandem (dsdf_channel_dropout)

### What it is

Randomly zero-out individual DSDF channels (indices 2–9) for tandem samples during training with probability p_drop, independently per channel per sample. This is fundamentally different from the DSDF2 magnitude augmentation (PR #2126) which SCALES the foil-2 channels uniformly. Channel dropout destroys specific geometric information, forcing the model to predict surface pressure using incomplete shape information.

```python
# In the augmentation block, after DSDF2 magnitude aug:
if cfg.aug_dsdf_channel_dropout > 0.0:
    _is_tan_drop = (x[:, 0, 21].abs() > 0.01)
    if _is_tan_drop.any():
        _drop_mask = torch.rand(x.size(0), 8, device=x.device) < cfg.aug_dsdf_channel_dropout
        # Zero out dropped channels (broadcast over N)
        for ch in range(8):
            x[:, :, 2 + ch] = torch.where(
                (_drop_mask[:, ch] & _is_tan_drop).unsqueeze(1),
                torch.zeros_like(x[:, :, 2 + ch]),
                x[:, :, 2 + ch]
            )
```

The key question: which channels are foil-1 vs foil-2? From `prepare_multi.py`: `dsdf` channels 0:4 are foil-1 (in-domain), channels 4:8 are foil-2. So indices 2:6 in x = foil-1 DSDF, indices 6:10 = foil-2 DSDF. To specifically target shape-transfer robustness, prioritize foil-1 DSDF dropout (forcing the model to infer foil-1 shape from minimal cues) — this is directly the p_tan problem (NACA6416 is foil-1).

Practically: either (a) drop foil-1 channels only (indices 2:6) with higher probability, or (b) drop all DSDF channels uniformly. Start with foil-1-only dropout.

### Why it might help p_tan

p_tan evaluates on NACA6416 as foil-1. The model has never seen NACA6416's DSDF signature in tandem context. By training with DSDF channel dropout, the model becomes more robust to unseen foil geometries — it learns to predict pressure from partial/noisy shape descriptors rather than memorizing the exact DSDF signature of NACA2412 and NACA9412. At test time, NACA6416's unfamiliar DSDF pattern is "closer" to a corrupted-but-known pattern than to nothing. This is the dropout-as-data-augmentation argument applied to the specific feature modality that matters.

The analogous intuition: if a human aerodynamicist only sees some of the foil cross-section measurements, they still reason from the physics of lift and pressure distribution. DSDF channel dropout forces the model to develop this partial-information robustness.

### Why it's different from tried approaches

- Magnitude aug (PR #2126): scales channels up/down but keeps all of them non-zero — doesn't simulate "I don't recognize this foil shape"
- Global mixup: mixes all features + targets — corrupts geometry-flow correspondence
- DSDF foil-1 magnitude aug (#2133): tried, same mechanism as foil-2 but for foil-1. Mixing with dropout is different — dropout is more aggressive and creates harder examples

### Code change locations

1. **Config** (~line 950): Add `aug_dsdf_channel_dropout: float = 0.0` (probability of dropping each foil-1 DSDF channel)

2. **Augmentation block** (after DSDF2 aug at line ~1655), add:
   ```python
   if cfg.aug_dsdf_channel_dropout > 0.0 and model.training:
       _is_tan_drop = (x[:, 0, 21].abs() > 0.01)  # pre-normalization gap
       if _is_tan_drop.any():
           # Foil-1 DSDF channels: x indices 2:6 (before normalization)
           _drop_mask = torch.rand(x.size(0), 4, device=x.device) < cfg.aug_dsdf_channel_dropout
           _drop_mask = _drop_mask & _is_tan_drop.unsqueeze(1)
           x[:, :, 2:6] = x[:, :, 2:6] * (~_drop_mask).float().unsqueeze(1)
   ```

### Hyperparameters

- `aug_dsdf_channel_dropout`: try 0.1, 0.2, 0.3 (probability of zeroing each foil-1 DSDF channel independently)
- Start with 0.2 (20% per channel = ~59% chance of at least one dropped channel per sample)
- Can combine with existing DSDF2 magnitude aug and gap/stagger aug
- Apply only during training (not validation)

### Risk

MEDIUM-LOW. Worst case: too much dropout makes foil shape unrecoverable from the features, degrading in-dist performance. Mitigate with small p_drop (≤0.3). The model still has foil-2 DSDF, position, curvature, and all condition scalars (Re, AoA, gap, stagger) to reason from.

### Expected impact

-2% to -5% p_tan. This directly addresses the OOD foil shape problem. The foil-1 DSDF is the primary way the model identifies foil-1's geometry — dropout on it forces shape-invariant pressure prediction.

### LoC estimate

~12 LoC.

---

## Idea 3 — FiLM-Conditioned Fore-Foil SRF Head via DSDF1 Statistics (foil1_srf_film_dsdf)

### What it is

The existing aft-foil SRF head (boundary ID=7) succeeded because it got dedicated capacity for a geometrically specific problem. But the FORE foil (boundary IDs 5 and 6) is the actual p_tan bottleneck for NACA6416: the aft foil in p_tan is the SAME aft foil as in training (just different front foil). The front surface pressure is the harder prediction for the unseen geometry.

Previous fore-foil SRF attempts (#2117, #2124) failed because they SPLIT the shared SRF head from the single-foil SRF, creating a training/test mismatch (the single-foil SRF is trained on more data than the tandem fore-foil SRF, splitting dilutes the shared representation).

The key difference here: instead of splitting, ADD an ADDITIVE correction specifically conditioned on DSDF1 statistics (the shape descriptor of the fore foil). The correction is gated to zero for non-tandem samples. This is a small, FiLM-conditioned additive module that fires ONLY for tandem fore-foil nodes, ON TOP of the existing shared SRF.

Architecture:
```python
class ForeFoilSRFFilm(nn.Module):
    """Additive fore-foil correction conditioned on DSDF1 shape statistics.
    
    Fires only for tandem fore-foil nodes (boundary IDs 5,6 in tandem context).
    Conditioned on a 4-dim summary of the foil-1 DSDF vector (mean of first 4 channels
    across fore-foil surface nodes) — this encodes the foil-1 shape fingerprint.
    """
    def __init__(self, n_hidden, out_dim, hidden_dim=128):
        # Shape condition: 4-dim DSDF1 stats
        # Input: hidden (n_hidden) + base pred (out_dim) 
        # FiLM: scale/shift from 4-dim condition
        ...
```

The FiLM conditioning vector is the mean of normalized DSDF1 channels across fore-foil surface nodes (a 4-dim "foil shape fingerprint"). For NACA2412 vs NACA6416, these fingerprints are distinct — the correction head learns to modulate differently per foil shape.

### Why it's different from failed approaches

- #2117 (fore-foil SRF): Split from shared SRF → broke in-dist performance  
- #2124 (stacked fore-foil SRF): Added on top of shared SRF but NOT conditioned on foil shape — the head couldn't distinguish NACA2412 (training) from NACA6416 (test) so it over-fit to training foils
- This approach: conditioned on DSDF1 statistics, so it SEES the foil shape fingerprint and can generalize across foil profiles

### Code change locations

1. **New module** `ForeFoilSRFCondHead` in train.py after AftFoilRefinementContextHead (line ~591):
   ```python
   class ForeFoilSRFCondHead(nn.Module):
       """FiLM-conditioned SRF for fore-foil tandem surface nodes."""
       def __init__(self, n_hidden, out_dim, cond_dim=4, hidden_dim=128, n_layers=2):
           super().__init__()
           in_dim = n_hidden + out_dim
           layers = []
           for i in range(n_layers):
               layers.extend([nn.Linear(in_dim if i==0 else hidden_dim, hidden_dim),
                               nn.LayerNorm(hidden_dim), nn.GELU()])
           layers.append(nn.Linear(hidden_dim, out_dim))
           nn.init.zeros_(layers[-1].weight); nn.init.zeros_(layers[-1].bias)
           self.mlp = nn.Sequential(*layers)
           self.film_scale = nn.Linear(cond_dim, hidden_dim, bias=False)
           self.film_shift = nn.Linear(cond_dim, hidden_dim)
           nn.init.zeros_(self.film_scale.weight)
           nn.init.zeros_(self.film_shift.weight); nn.init.zeros_(self.film_shift.bias)
       
       def forward(self, hidden, base_pred, dsdf1_stats):
           """
           hidden: [M, n_hidden] — fore-foil surface node hidden features
           base_pred: [M, out_dim] — base predictions
           dsdf1_stats: [M, 4] — DSDF1 mean across all fore-foil nodes, broadcast
           """
           x = torch.cat([hidden, base_pred], dim=-1)
           for i, layer in enumerate(self.mlp):
               x = layer(x)
               if i == 2:  # after first LN+GELU
                   x = x * (1 + self.film_scale(dsdf1_stats)) + self.film_shift(dsdf1_stats)
           return x
   ```

2. **Config** (~line 950): Add `fore_foil_srf_cond: bool = False`

3. **Instantiate in training setup** (after aft_srf_head instantiation, ~line 1292):
   ```python
   fore_srf_cond_head = None
   if cfg.fore_foil_srf_cond:
       fore_srf_cond_head = ForeFoilSRFCondHead(
           n_hidden=cfg.n_hidden, out_dim=3, cond_dim=4, hidden_dim=128, n_layers=2
       ).to(device)
       fore_srf_cond_head = torch.compile(fore_srf_cond_head, mode=cfg.compile_mode)
   ```

4. **Apply in training loop** (after aft SRF head application, ~line 1835):
   ```python
   if fore_srf_cond_head is not None and model.training:
       # fore-foil mask: tandem + boundary IDs 5,6 = is_surface & ~aft_foil_mask & is_tandem
       _fore_foil_mask = is_surface & ~_aft_foil_mask & _is_tandem.unsqueeze(1)
       fore_idx = _fore_foil_mask.nonzero(as_tuple=False)
       if fore_idx.numel() > 0:
           _fore_h = hidden[fore_idx[:,0], fore_idx[:,1]]
           _fore_pred = pred[fore_idx[:,0], fore_idx[:,1]]
           # DSDF1 stats: mean of raw foil-1 DSDF across fore-foil nodes per sample
           # Use standardized x DSDF channels (already normalized at this point)
           _dsdf1_per_sample = []
           for b in range(B):
               _b_fore = _fore_foil_mask[b]
               if _b_fore.any():
                   _dsdf1_per_sample.append(x[b, _b_fore, 2:6].mean(dim=0))  # [4]
               else:
                   _dsdf1_per_sample.append(torch.zeros(4, device=device))
           _dsdf1_stats = torch.stack(_dsdf1_per_sample)  # [B, 4]
           _dsdf1_per_node = _dsdf1_stats[fore_idx[:,0]]  # [M, 4]
           with torch.amp.autocast("cuda", dtype=torch.bfloat16):
               _fore_corr = fore_srf_cond_head(_fore_h, _fore_pred, _dsdf1_per_node).float()
           pred = pred.clone()
           pred[fore_idx[:,0], fore_idx[:,1]] += _fore_corr
   ```

### Hyperparameters

- No new hyperparameters beyond enabling flag
- `cond_dim=4` (DSDF1 channels 2:6) — the mean across fore-foil surface nodes is a compact shape fingerprint
- hidden_dim=128 (smaller than aft-foil head which is 192 — fore-foil correction is simpler)

### Risk

MEDIUM. The key failure mode from previous fore-foil SRF attempts was diluting the shared single-foil SRF — this approach is purely additive and tandem-only (zero-init guarantees no change at initialization for tandem, and single-foil samples are completely unaffected). The FiLM conditioning on DSDF1 statistics is the critical differentiator — if DSDF1 statistics are not informative enough to distinguish NACA6416 from NACA2412, the head will not generalize to p_tan. Validate that DSDF1 mean statistics DO differ between NACA profiles (they should — camber and thickness change the gradient magnitude).

### Expected impact

-2% to -5% p_tan. The fore-foil surface in p_tan (NACA6416) has been getting a worse representation than the aft foil because there's no dedicated head for it. This fills that gap with the critical extra: foil-shape conditioning.

### LoC estimate

~60 LoC (new class ~25, instantiation ~15, application ~20).

---

## Idea 4 — Tandem Foil Cross-DSDF Features (tandem_cross_dsdf)

### What it is

For each mesh node, compute three new features encoding the spatial relationship between foil-1 and foil-2 as seen from that node's perspective. These are derived from the DSDF vectors already in the input:

```python
# At feature construction time, for tandem samples only:
dsdf1_node = x[:, :, 2:6]   # [B, N, 4] foil-1 DSDF per node
dsdf2_node = x[:, :, 6:10]  # [B, N, 4] foil-2 DSDF per node

# Feature 1: Ratio of foil-2 distance to foil-1 distance (relative proximity)
dist1 = dsdf1_node.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, N, 1]
dist2 = dsdf2_node.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, N, 1]
dist_ratio = (dist2 / (dist1 + dist2))  # [B, N, 1] — ranges 0 (near foil-1) to 1 (near foil-2)

# Feature 2: Angle between DSDF1 and DSDF2 normals (orientation relationship)
dsdf1_n = F.normalize(dsdf1_node[:,:,:2], dim=-1)  # [B, N, 2] — use first 2 channels as gradient
dsdf2_n = F.normalize(dsdf2_node[:,:,:2], dim=-1)
foil_angle = (dsdf1_n * dsdf2_n).sum(dim=-1, keepdim=True)  # [B, N, 1] — cosine of foil normals

# Feature 3: Magnitude ratio of foil-2 to foil-1 DSDF (shape intensity comparison)
mag_ratio = (dist2 / (dist1 + 1e-6)).clamp(0, 5)  # [B, N, 1] — log would also work

# Zero for single-foil samples
cross_feats = torch.cat([dist_ratio, foil_angle, mag_ratio], dim=-1)  # [B, N, 3]
_is_tan_cross = (x[:, 0, 21].abs() > 0.01)
cross_feats = cross_feats * _is_tan_cross.float().view(-1,1,1)
x = torch.cat([x, cross_feats], dim=-1)  # extend x from 24→27 features (before Fourier PE)
```

These features give every node explicit knowledge of: where it sits in the foil-1 vs foil-2 proximity space (dist_ratio), and how the two foils' local geometry relates at this location (foil_angle).

### Why it might help p_tan

The current model must implicitly derive the inter-foil spatial relationship from the absolute DSDF values at each node. The cross-DSDF features make this explicit: a node in the inter-foil gap "knows" it's midway between both foils (dist_ratio ≈ 0.5), and that the two foils are curved toward each other. For NACA6416 (high camber), foil_angle at specific chord positions differs from NACA2412 (low camber) — this difference is exactly what the model needs to detect to produce a different pressure distribution on the front foil.

This is inspired by the cross-attention approach (#2045, #2134) but much simpler: instead of an attention module, we directly compute geometric features encoding the inter-foil relationship. The cross-attention approaches added O(N^2) computation; this adds 3 features with O(N) computation.

### Why this is different from tried approaches

- #2045 (Cross-Foil Feature Exchange): failed — added a cross-attention module between foil-1 and foil-2 nodes, too slow and architecturally complex
- #2134 (Fore-Foil TE Relative Coords): tried to add inter-foil frame features at the aft SRF head level, not as raw input features
- Current approach: computes cross-foil features at the raw input level, as static per-node features — no new parameters in the main model, no attention overhead

### Code change locations

1. **`prepare_multi.py` equivalent in train.py input construction** (at the feature augmentation site, ~line 1670, after `x = (x - stats["x_mean"]) / stats["x_std"]` but before Fourier PE):
   ```python
   if cfg.tandem_cross_dsdf:
       dsdf1_n = x[:, :, 2:6]  # post-normalization [B, N, 4]
       dsdf2_n = x[:, :, 6:10]  # [B, N, 4]
       dist1 = dsdf1_n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
       dist2 = dsdf2_n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
       dist_ratio = dist2 / (dist1 + dist2)  # [B, N, 1]
       d1_dir = F.normalize(dsdf1_n[:,:,:2], dim=-1)
       d2_dir = F.normalize(dsdf2_n[:,:,:2], dim=-1)
       foil_angle = (d1_dir * d2_dir).sum(dim=-1, keepdim=True)  # [B, N, 1]
       _is_tan_cross = (x[:, 0, 21].abs() > 0.5)  # post-normalization: gap~=0 for single-foil
       cross_feats = torch.cat([dist_ratio, foil_angle], dim=-1) * _is_tan_cross.float()[:,None,None]
       x = torch.cat([x, cross_feats], dim=-1)  # [B, N, X_DIM+2+2+n_fourier]
   ```
   Note: 2 new features (dist_ratio, foil_angle) to start. mag_ratio can be added later.

2. **`model_config` fun_dim adjustment** (~line 1202): `fun_dim` accounts for extra features. Currently:
   ```python
   fun_dim=X_DIM - 2 + 2 + (1 if cfg.foil2_dist else 0) + 32
   ```
   Add `+ (2 if cfg.tandem_cross_dsdf else 0)`.

3. **Config** (~line 950): Add `tandem_cross_dsdf: bool = False`

### Hyperparameters

- No new hyperparameters; fixed geometric features derived from existing DSDF
- Sanity check: print mean dist_ratio for surface nodes in tandem samples to confirm it's ~0 for foil-1 nodes and ~1 for foil-2 nodes

### Risk

LOW-MEDIUM. The features are geometrically derived and deterministic — no optimization is needed for the features themselves. Risk is that post-normalization DSDF values don't have meaningful magnitude structure (they may have been z-scored to mean ~0). Gotcha: the z-scoring in `x = (x - stats["x_mean"]) / stats["x_std"]` may have rotated the DSDF vectors in feature space. Use pre-normalization raw_dsdf for computing these features if post-normalization DSDF norms don't show good separation. The raw_dsdf is already saved at line 1657 (`raw_dsdf = x[:, :, 2:10]`).

Revision: compute cross-DSDF features from `raw_dsdf` (pre-normalization) for cleaner geometry:
```python
if cfg.tandem_cross_dsdf:
    # raw_dsdf is saved at line 1657 before normalization
    _rdist1 = raw_dsdf[:, :, 0:4].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    _rdist2 = raw_dsdf[:, :, 4:8].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dist_ratio = _rdist2 / (_rdist1 + _rdist2)
    d1_dir = F.normalize(raw_dsdf[:, :, 0:2], dim=-1)
    d2_dir = F.normalize(raw_dsdf[:, :, 4:6], dim=-1)
    foil_angle = (d1_dir * d2_dir).sum(dim=-1, keepdim=True)
    _is_tan_cross = (x[:, 0, 22].abs() > 0.01)  # raw x still valid at line 1657
    cross_feats = torch.cat([dist_ratio, foil_angle], dim=-1) * _is_tan_cross.float()[:,None,None]
    x = torch.cat([x, cross_feats], dim=-1)  # add BEFORE stats normalization at line 1669
```
This means the insertion is BEFORE `x = (x - stats["x_mean"]) / stats["x_std"]`, so the normalization will also z-score the cross-features. That's fine — it keeps the input distribution consistent.

### Expected impact

-2% to -4% p_tan. More speculative than ideas 1-3 but directly encodes the missing "inter-foil geometry from the node's perspective" information.

### LoC estimate

~20 LoC.

---

## Idea 5 — Asymmetric PCGrad: OOD Grads Projected onto In-Dist, Not Vice Versa (pcgrad_asymmetric)

### What it is

The current PCGrad implementation (2-way, PR #2139) is SYMMETRIC: when gradients g_A (in-dist) and g_B (OOD) conflict, it projects BOTH gradients to remove the conflicting component. This means in-dist performance is also degraded when conflicts occur.

The asymmetric version: only project g_B onto the normal plane of g_A (remove the component of g_B that conflicts with g_A), but keep g_A unchanged. In-dist learning is never sacrificed. OOD learning only receives updates that don't regress the in-dist direction.

Current symmetric PCGrad (lines 1996-2008):
```python
elif dot_ab < 0:
    # SYMMETRIC: both get projected
    p.grad = ((ga - (dot_ab / gb_ns) * gb) + (gb - (dot_ab / ga_ns) * ga)) * 0.5
else:
    p.grad = (ga + gb) * 0.5
```

Asymmetric PCGrad (proposed):
```python
elif dot_ab < 0:
    # ASYMMETRIC: only project g_B onto normal plane of g_A; keep g_A unchanged
    gb_proj = gb - (dot_ab / ga_ns) * ga  # remove g_A component from g_B
    p.grad = (ga + gb_proj) * 0.5
else:
    p.grad = (ga + gb) * 0.5
```

The theoretical justification: in-dist performance is the "anchor" — we should never sacrifice it for OOD. The current symmetric approach is a compromise between two equal tasks. With the asymmetric version, in-dist is protected and OOD gets as much as it can without hurting in-dist.

### Why it might help pressure metrics

The current baseline p_in=13.05 and p_oodc=7.70 are very good. Asymmetric PCGrad should hold p_in and p_oodc steady (in-dist group A never gets projected away) while potentially improving p_tan further (OOD group B gets cleaner gradients that are guaranteed not to regress in-dist). The symmetric version has been useful but may be leaving p_oodc/p_re performance on the table by also projecting the in-dist gradient.

### Why this hasn't been tried

PCGrad asymmetry is discussed in the original PCGrad paper but not the default recommendation. The literature typically uses symmetric because both tasks are considered "equal" — but in our setting, in-dist and OOD are NOT equal (in-dist defines the physical system; OOD is a generalization test). This asymmetric formulation matches our actual problem structure.

### Code change locations

1. **Config** (~line 950): Add `pcgrad_asymmetric: bool = False`

2. **PCGrad implementation** (~line 2005-2008):
   ```python
   if cfg.pcgrad_asymmetric and dot_ab < 0:
       # Only project g_B; keep g_A pure
       gb_proj = p.grad - (dot_ab / ga_ns) * ga  # remove g_A from g_B
       p.grad = (ga + gb_proj) * 0.5
   elif dot_ab < 0:
       # Original symmetric
       p.grad = ((ga - (dot_ab / gb_ns) * gb) + (gb - (dot_ab / ga_ns) * ga)) * 0.5
   else:
       p.grad = (ga + gb) * 0.5
   ```

### Hyperparameters

- No new hyperparameters; simple boolean flag
- Combine with `--disable_pcgrad False` (PCGrad already enabled in baseline)

### Risk

LOW. This is a 3-line code change to the existing PCGrad implementation. Worst case: asymmetric projection increases gradient variance for the OOD group (g_B can receive zero gradient when it fully conflicts with g_A). Mitigate: if this causes instability, add a small epsilon to keep a minimum g_B contribution.

One subtle issue: the current `ga_ns = (ga_flat @ ga_flat).item()` is computed from flattened gradients. In the per-parameter loop, `ga_ns` is the GLOBAL norm squared but applied per-parameter. This is the existing implementation's approach — the asymmetric version uses the same `ga_ns` scalar, which is correct.

### Expected impact

-1% to -3% across all OOD metrics (p_oodc, p_tan, p_re) while preserving p_in more precisely. The compound of asymmetric PCGrad + existing DSDF2/gap-stagger augmentations may be particularly powerful.

### LoC estimate

~8 LoC (Config flag + 5 lines in PCGrad block).

---

## Idea 6 — Bidirectional DSDF Input: Explicitly Encode Foil Normals as Input (dsdf_normals_feat)

### What it is

The DSDF channels encode the gradient of the signed distance field from each foil. The FIRST two channels of each foil's DSDF (x[:,2:4] for foil-1, x[:,6:8] for foil-2) are approximately the outward-pointing surface normals at nearby surface nodes. For VOLUME nodes far from the surface, these become the gradient of the distance field (pointing toward the foil surface).

Currently the model receives these as raw inputs but never explicitly normalizes them to unit vectors — so the model must implicitly learn to separate "normal direction" (unit vector) from "distance" (magnitude). Adding explicit unit-normal features separates these two pieces of information:

```python
# After raw_dsdf is available (line 1657):
# Foil-1 normals (from first 2 DSDF channels)
_dsdf1_xy = raw_dsdf[:, :, 0:2]  # [B, N, 2] — gradient of foil-1 SDF
_dsdf1_norm = F.normalize(_dsdf1_xy, dim=-1)  # [B, N, 2] unit normal toward foil-1
_dsdf1_mag = _dsdf1_xy.norm(dim=-1, keepdim=True)  # [B, N, 1] distance to foil-1

# Foil-2 normals (from channels 4:6 of dsdf, i.e. x[:,6:8])
_dsdf2_xy = raw_dsdf[:, :, 4:6]  # [B, N, 2] — gradient of foil-2 SDF
_dsdf2_norm = F.normalize(_dsdf2_xy, dim=-1)  # [B, N, 2] unit normal toward foil-2
_dsdf2_mag = _dsdf2_xy.norm(dim=-1, keepdim=True)  # [B, N, 1] distance to foil-2

# For tandem: also add the inter-foil normal (direction from foil-1 surface to foil-2 surface)
# For single: zero-pad foil-2 features
_is_tan_norm = (x[:, 0, 22].abs() > 0.01)  # gap nonzero pre-norm
_dsdf2_norm = _dsdf2_norm * _is_tan_norm.float()[:,None,None]
_dsdf2_mag = _dsdf2_mag * _is_tan_norm.float()[:,None,None]

extra_feats = torch.cat([_dsdf1_norm, _dsdf1_mag, _dsdf2_norm, _dsdf2_mag], dim=-1)  # [B, N, 6]
x = torch.cat([x, extra_feats], dim=-1)  # 24 → 30 features
```

This gives the model 3 explicitly separated representations of each foil's geometry: (1) unit normal direction, (2) distance magnitude, (3) the 8-channel DSDF that it already has. The redundancy is intentional — different scales of the same information feed different parts of the model.

### Why it might help p_tan

For NACA6416 (high-camber front foil), the surface normal direction changes significantly along the chord compared to NACA2412 (low-camber). The raw DSDF channels encode this but mixed with the magnitude. Separating them makes the foil shape fingerprint more explicitly geometric and potentially easier to generalize across foil profiles. The unit normal is a canonical representation of foil geometry — it's invariant to the actual distance (which varies across the mesh) while capturing the local surface orientation.

This is the "disentanglement through explicit decomposition" approach: instead of relying on the network to implicitly factorize DSDF magnitude from direction, we give it both explicitly.

### Code change locations

1. **Config** (~line 950): Add `dsdf_normals_feat: bool = False`

2. **Feature construction** (~line 1657, after raw_dsdf is computed):
   ```python
   if cfg.dsdf_normals_feat:
       _d1xy = raw_dsdf[:, :, 0:2]
       _d2xy = raw_dsdf[:, :, 4:6]
       _d1n = F.normalize(_d1xy, dim=-1)
       _d1m = _d1xy.norm(dim=-1, keepdim=True)
       _d2n = F.normalize(_d2xy, dim=-1)
       _d2m = _d2xy.norm(dim=-1, keepdim=True)
       _is_tan_n = (x[:, 0, 22].abs() > 0.01)  # raw gap pre-norm
       _d2n = _d2n * _is_tan_n.float()[:,None,None]
       _d2m = _d2m * _is_tan_n.float()[:,None,None]
       x = torch.cat([x, _d1n, _d1m, _d2n, _d2m], dim=-1)  # add 6 feats before normalization
   ```

3. **`fun_dim` in model_config** (line 1202): Add `+ (6 if cfg.dsdf_normals_feat else 0)`.

4. **These features should be included in the normalization** since they're appended before `x = (x - stats["x_mean"]) / stats["x_std"]`. But wait — the stats are pre-computed on the ORIGINAL feature layout, so appending new features BEFORE normalization will cause a dimension mismatch. The clean solution: compute `x = (x - stats["x_mean"]) / stats["x_std"]` FIRST, then append the new features using `raw_dsdf` (which is saved before normalization). In this case `fun_dim` needs to account for the extra 6 features which are NOT normalized by the stats (they're derived from `raw_dsdf` directly). This is acceptable — unit normals are already in [-1,1] range and distances will be log1p-scaled.

Revised:
```python
# After x = (x - stats["x_mean"]) / stats["x_std"] at line 1669:
if cfg.dsdf_normals_feat:
    _d1xy = raw_dsdf[:, :, 0:2]  # [B, N, 2]
    _d2xy = raw_dsdf[:, :, 4:6]  # [B, N, 2]
    _d1n = F.normalize(_d1xy, dim=-1)
    _d1m = torch.log1p(_d1xy.norm(dim=-1, keepdim=True) * 5.0)  # log-scale distance
    _d2n = F.normalize(_d2xy, dim=-1)
    _d2m = torch.log1p(_d2xy.norm(dim=-1, keepdim=True) * 5.0)
    _is_tan_n = (x[:, 0, 22].abs() > 0.5)  # post-normalization gap check
    _d2n = _d2n * _is_tan_n.float()[:,None,None]
    _d2m = _d2m * _is_tan_n.float()[:,None,None]
    x = torch.cat([x, _d1n, _d1m, _d2n, _d2m], dim=-1)  # [B, N, X_DIM+2+2+6+n_fourier]
```

### Hyperparameters

- No new hyperparameters
- `fun_dim` must be increased by 6

### Risk

LOW. These features are derived deterministically from existing DSDF channels. No normalization concern (unit normals are bounded [-1,1]; log1p-scaled distances are bounded and smooth). The only risk is compute overhead (6 extra input features add ~3% to the input embedding computation).

### Expected impact

-1% to -3% p_tan. More uncertain than ideas 1-3 — depends on whether the implicit DSDF feature decomposition is actually a bottleneck for the current model.

### LoC estimate

~20 LoC.

---

## Priority Ranking

Given the research programme's primary goal (minimize p_tan while preserving p_in, p_oodc, p_re):

| Rank | Idea | Expected Impact | Risk | LoC |
|------|------|-----------------|------|-----|
| 1 | DSDF Channel Dropout (Idea 2) | -2% to -5% p_tan | LOW | 12 |
| 2 | Fore-Foil SRF FiLM (Idea 3) | -2% to -5% p_tan | MEDIUM | 60 |
| 3 | Foil Shape Similarity Bias (Idea 1) | -1% to -3% p_tan | LOW | 30 |
| 4 | Asymmetric PCGrad (Idea 5) | -1% to -3% all OOD | LOW | 8 |
| 5 | Tandem Cross-DSDF Features (Idea 4) | -2% to -4% p_tan | LOW-MED | 20 |
| 6 | DSDF Normal Features (Idea 6) | -1% to -3% p_tan | LOW | 20 |

**Immediate assignments:**
- Ideas 1, 4, 5, 6 are low-LoC and can be implemented quickly by students returning from hyperparameter sweeps
- Idea 2 (DSDF channel dropout) is the highest expected impact for lowest risk — assign first
- Idea 3 (fore-foil SRF FiLM) is more complex but addresses the core p_tan gap (no dedicated capacity for the unseen fore-foil geometry)

**Compounding:** Ideas 1+2 (foil shape similarity bias + DSDF channel dropout) are likely orthogonal — run as a compound once each is validated individually. Similarly, asymmetric PCGrad (Idea 5) + any augmentation idea should compound well since they operate at different levels.

---

## Notes on Current Architecture State

The main model is near its capacity and complexity ceiling for 3-hour runs. Most gains are now coming from:
1. Better input representation for OOD geometries (Ideas 1, 4, 6)
2. Better training signal for the difficult tandem distribution (Ideas 2, 3)
3. Better gradient routing during optimization (Idea 5)

The plateau in hyperparameter tuning (all 6 current WIP PRs are tuning sweeps) suggests these ideas are the right direction: attack the representation gap rather than the optimization schedule.
