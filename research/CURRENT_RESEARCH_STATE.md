# SENPAI Research State

- **Date:** 2026-04-07 04:30 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2213, +Wake Deficit Feature, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.979** | < 11.98 |
| **p_oodc** | **7.643** | < 7.65 |
| **p_tan** | **28.341** | < 28.34 |
| **p_re** | **6.300** | < 6.30 |

**Latest merge:** PR #2213 (frieren) — Wake Deficit Feature: 2 gap-normalized fore-TE offset channels (dx/gap, dy/gap). Delivers striking -4.1% p_in, -0.6% p_tan, -1.7% p_re. p_oodc marginal miss (+0.3%). W&B: hgml7i2r (s42), qic03vrg (s73).

**Key note:** Explicit physical feature encoding continues to be the most productive direction. Three geometric feature layers now active: DSDF, TE coord frame, wake deficit. Now also exploring loss-level physics coupling (Bernoulli).

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-wake-deficit" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame --wake_deficit_feature
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Single-model p_tan (28.341) **BEATS** ensemble (29.1). p_in (11.979) also beats ensemble (12.1).

## Student Status (2026-04-07 09:15 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2239 | EMA Self-Distillation: use EMA predictions as soft targets | WIP |
| frieren | #2242 | SAM: Sharpness-Aware Minimization for flat minima | WIP (just assigned) |
| edward | #2243 | Spectral Norm SRF: Lipschitz constraint on output heads | WIP (just assigned) |
| fern | #2244 | Higher EMA Decay: 0.9995 for longer-memory averaging | WIP (just assigned) |
| nezuko | #2237 | Manifold Mixup: feature-level interpolation for OOD generalization | WIP |
| alphonse | #2240 | Deeper Backbone: 4 TransolverBlocks for increased capacity | WIP (just assigned) |
| tanjiro | #2218 | LE Coordinate Frame v3: single chordwise ratio le/(le+te) | WIP |
| askeladd | #2241 | Lookahead Optimizer: slow-weight averaging wrapper for Lion | WIP (just assigned) |

**Idle students:** None.

## PRs Ready for Review
None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues. Prior directives still in effect:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5+)
- Issue #1834: Never use raw data files besides assigned training split

## Current Research Focus and Themes

**Theme 1: Explicit Physical Feature Engineering (SATURATED — 7 consecutive failures)**
- TE coord frame (PR #2207): -5.4% p_in ✅
- Wake deficit feature (PR #2213): -4.1% p_in, -1.7% p_re ✅
- FAILED: chord-camber (#2227), Re-walldist (#2228), arc-length PE (#2223), surface normals (#2229), domain-split SRF (#2225), curvature (#2231)
- **Conclusion: DSDF + TE coord frame + wake deficit fully capture useful geometric information. Feature engineering is exhausted.**
- In flight: LE coord frame v3 (#2218 tanjiro) — final feature experiment

**Theme 2: Tandem-Specific Coupling and Conditioning**
- In flight: tandem feature cross (#2226 nezuko) — config-aware sigmoid gate on encoded features
- Domain-split SRF norm (#2225): CLOSED — SRF heads process too few nodes, domain signal too dilute post-backbone
- Domain AdaLN (#2164): CLOSED — disturbed slice routing in backbone
- Fore-aft cross-attention (#2219, #2217, #2202): CLOSED — direction exhausted after 3 attempts

**Theme 3: Training Dynamics**
- In flight: stochastic depth curriculum (#2230 thorfinn)
- Bernoulli consistency loss (#2224): CLOSED — physics wrong at viscous walls
- mHC residuals (#2222): CLOSED — skip-dominant collapse
- Slice diversity reg (#2220): CLOSED — slice collapse is beneficial

**Theme 4: Architecture Dead Ends (DO NOT REVISIT)**
- NOBLE/CosNet, register tokens, Ada-Temp, GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR
- Geometry latent cross-attention (GALE, #2216) — creates competing pathway
- Domain conditioning at SRF heads (#2225) — too late in pipeline

## Potential Next Research Directions

After current wave completes:
1. **Stagnation-point coordinate frame**: explicit LE/stagnation-relative features (complement to TE frame) — expected -3% to -6% p_in
2. **Re-scaled DSDF features**: multiply DSDF by 1/sqrt(Re) to encode viscous length scale — targets p_re
3. **Cp-informed surface loss weight**: upweight nodes with high |p| or high |dp/ds| — adaptive loss mining
4. **Wake centerline coord**: single channel y_node - y_fore_TE normalized by gap — complement to wake deficit
5. **Camber-line arc-length coord**: s ∈ [0,1] along each foil surface — geometry-invariant parameterization
6. **OOD joint augmentation**: joint (AoA, Re) perturbation targeting p_oodc
7. **Researcher-agent**: generate fresh Round 18 hypotheses once current wave completes

## Recent Closed Dead Ends

- PR #2234 (fern): SWA Training — uniform averaging dilutes converged weights; EMA superior; p_in +54.9%
- PR #2238 (frieren): Cosine Warm Restarts — T_0=40 too short, third cycle cut at high LR; all metrics +8-19%
- PR #2233 (edward): Re Input Augmentation — Re is critical signal, σ=0.1 too large; p_re +4.5% (target metric worse)
- PR #2236 (askeladd): Huber Surface Loss — δ=0.5 too large, all nodes in L2 regime, gradient weakening; all metrics +6-50%
- PR #2235 (alphonse): Input Feature Noise Augmentation — uniform noise corrupts geometric features; all metrics +7-14%
- PR #2230 (thorfinn): Stochastic Depth Curriculum — 3 blocks too shallow for block dropout; all metrics +6-34%
- PR #2232 (frieren): Pressure Laplacian Smoothness — catastrophic, smoothness penalty destroys OOD; p_oodc +307%, p_re +291%
- PR #2226 (nezuko): Tandem Feature Cross — global sigmoid gate too blunt, p_tan +1.3%, p_oodc -1.2% but p_re +1.6%
- PR #2231 (askeladd): Surface Curvature Feature — catastrophic, mesh noise in finite-difference curvature; seed 42 diverged +160% p_in
- PR #2229 (alphonse): Surface Normal Features — DSDF already encodes orientation; kNN noise at LE/TE; all metrics +3-7%
- PR #2223 (fern): Surface Arc-Length PE — redundant with TE coord frame; all metrics +3-6%
- PR #2228 (edward): Re-Scaled WallDist — redundant with DSDF + log(Re); all metrics +2-5%
- PR #2227 (frieren): Chord-Camber Distance — geometry-frame upper/lower ≠ flow-frame suction/pressure; all metrics +0.9-5.5%
- PR #2225 (askeladd): Domain-Split SRF Norm — p_in +4.3%, p_re +3.2%; SRF heads too late in pipeline for domain conditioning
- PR #2224 (thorfinn): Bernoulli Consistency Loss — catastrophic +94% p_in, physics wrong at viscous walls
- PR #2219 (alphonse): Additive fore→aft cross-attn — marginal p_oodc only, torch.compile issues, direction exhausted (3 attempts)
- PR #2222 (edward): mHC Learnable Residual Mixing — skip-dominant collapse (alpha≈1.9, beta≈0.1), all metrics regressed
- PR #2221 (frieren): Wake Angle Feature — atan2 redundant with Cartesian (dx/gap, dy/gap), all metrics regressed
- PR #2217 (nezuko): Fore-SRF Skip — mean-pooled fore hidden too coarse, 3/4 metrics worse, high seed variance
- PR #2220 (askeladd): Slice Diversity Reg — forcing slice orthogonality harms all metrics +5-10%, slice collapse is beneficial
- PR #2216 (thorfinn): GeoTransolver GALE — geometry cross-attention creates competing pathway, p_in +29%, all metrics worse
- PR #2214 (edward): Deep Supervision — aux_loss never activated (redundant with --pressure_deep)
- PR #2210 (fern): Arc-Length Surface Loss — conflicts with hard-node mining; p_in +14.2%
- PR #2209 (thorfinn): Attention Register Tokens — p_tan +4.0%, p_in +6.1%
- PR #2205 (nezuko): NOBLE/CosNet — all metrics regressed 5-19%
