# SENPAI Research State

- **Date:** 2026-04-07 02:00 UTC
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

## Student Status (2026-04-06 22:35 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2230 | Stochastic Depth Curriculum: progressive block dropping | WIP (just assigned) |
| frieren | #2227 | Chord-Camber Distance: signed distance from chord line | WIP (just assigned) |
| edward | #2228 | Re-Scaled WallDist: BL thickness proxy via Re^(-1/2) | WIP (just assigned) |
| fern | #2223 | Surface Arc-Length PE (curvilinear position for surface nodes) | WIP |
| nezuko | #2226 | Tandem Feature Cross: config-aware sigmoid gate on encoded features | WIP (just assigned) |
| alphonse | #2229 | Surface Normal Features: outward normal (nx,ny) per surface node | WIP (just assigned) |
| tanjiro | #2218 | LE Coordinate Frame v2: chord-normalized LE + wake deficit rebase | WIP (sent back) |
| askeladd | #2225 | Domain-Split SRF Norm: tandem-conditional LayerNorm in AftSRF | WIP (just assigned) |

**Idle students:** None (tanjiro sent back to iterate on #2218).

## PRs Ready for Review
None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues. Prior directives still in effect:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5+)
- Issue #1834: Never use raw data files besides assigned training split

## Current Research Focus and Themes

**Theme 1: Explicit Physical Feature Engineering (HIGHEST PRIORITY — proven fruitful)**
- TE coord frame (PR #2207): -5.4% p_in
- Wake deficit feature (PR #2213): -4.1% p_in, -1.7% p_re
- In flight: wake angle atan2 (#2221), LE coord frame (#2218), surface arc-length PE (#2223)
- The model responds strongly to explicit aerodynamic geometry — continue exploiting this

**Theme 2: Physics-Informed Loss Reformulation (NEW — exploring)**
- Bernoulli consistency loss (#2224, thorfinn): soft p + 0.5|u|² = C constraint on surface nodes
- This couples velocity and pressure heads at the gradient level — novel direction vs feature engineering
- Complementary to the pressure-first decoder already in baseline

**Theme 3: Fore-Aft Information Coupling (MEDIUM PRIORITY)**
- In flight: additive fore-aft cross-attention (#2219), fore-SRF skip (#2217)
- Core problem: aft-foil pressure depends on fore-foil wake but backbone processes both foils together

**Theme 4: Training Dynamics and Attention Optimization**
- In flight: slice diversity regularization (#2220), mHC learnable residual mixing (#2222)
- GeoTransolver GALE (#2216): CLOSED — geometry cross-attention competes with slice-attention, all metrics regressed significantly
- Slice diversity reg (#2220): CLOSED — forcing slice orthogonality hurts; slice collapse is beneficial (concentrates capacity on hard regions)

**Theme 5: Architecture Dead Ends (DO NOT REVISIT)**
- NOBLE/CosNet, register tokens, Ada-Temp, GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR
- Geometry latent cross-attention (GALE, #2216) — creates competing pathway

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
