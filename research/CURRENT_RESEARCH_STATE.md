# SENPAI Research State

- **Date:** 2026-04-06 22:15 UTC
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

**Key note:** The wake deficit feature success confirms: **explicitly encoding physical aerodynamic quantities as input features is the most productive direction**. The model can learn from geometric proxies it previously had to infer. Three geometric feature layers now active: DSDF, TE coord frame, wake deficit. Focus shifts to: what other physical quantities can be encoded explicitly?

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

Single-model p_tan (28.341) **BEATS** ensemble (29.1) by a large margin. p_in (11.979) also beats ensemble (12.1).

## Student Status (2026-04-06 22:15 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| frieren | #2221 | Wake Angle Feature (atan2 wake direction) | WIP (just assigned) |
| edward | #2222 | mHC Learnable Residual Mixing (alpha/beta per sublayer) | WIP (just assigned) |
| fern | #2223 | Surface Arc-Length PE (curvilinear position for surface nodes) | WIP (just assigned) |
| nezuko | #2217 | Fore-SRF Skip: inject fore-foil mean hidden into AftSRF input | WIP |
| alphonse | #2219 | Additive Fore→Aft Cross-Attention in AftSRF | WIP |
| thorfinn | #2216 | GeoTransolver GALE (geometry-latent cross-attention) | WIP |
| tanjiro | #2218 | LE Coordinate Frame: leading-edge-relative input features | WIP |
| askeladd | #2220 | Slice Diversity Reg: Gram matrix orthogonality on slice attention | WIP |

## PRs Ready for Review
None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues since last check. Prior directives:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5, Phase 6)
- Issue #1834: Never use raw data files besides assigned training split

## Current Research Focus and Themes

**Theme 1: Explicit Physical Feature Engineering (HIGHEST PRIORITY — proving fruitful)**
- TE coord frame (PR #2207): -5.4% p_in via radial position encoding
- Wake deficit feature (PR #2213): -4.1% p_in, -1.7% p_re via wake-relative position
- In flight: wake angle (atan2), LE coord frame, surface arc-length PE
- The model responds strongly to explicit aerodynamic geometry — keep exploiting this

**Theme 2: Fore-Aft Information Coupling (MEDIUM PRIORITY)**
- Core problem: aft-foil pressure depends on fore-foil wake, but the backbone processes both foils together
- In flight: additive fore-aft cross-attention (#2219), fore-SRF skip (#2217)
- Previous approaches that failed: replacement cross-attention (#2202), context-only (#2127 buggy), fore-foil mean hidden

**Theme 3: Training Dynamics and Attention Optimization**
- In flight: slice diversity regularization (#2220), GeoTransolver GALE (#2216)
- In flight: mHC learnable residual mixing (#2222) — 12 parameters, safe init
- Closed dead ends: NOBLE/CosNet, register tokens, Ada-Temp

**Theme 4: Architecture Dead Ends (DO NOT REVISIT)**
- NOBLE/CosNet, register tokens, Muon/Gram-NS, Ada-Temp, SCA, iterative SRF
- Domain AdaLN (backbone), XSA, GNOT/Galerkin/Hierarchical models (all catastrophically failed Phase 5)

## Potential Next Research Directions

After current wave completes:
1. **Wake distance feature**: Euclidean distance from fore-TE to aft node (complement to dx/gap, dy/gap, theta)
2. **Tandem feature cross (global gate)**: sigmoid gate on encoded features parameterized by (gap, stagger, Re, AoA)
3. **Domain-split SRF normalization**: separate LayerNorm scale/bias for tandem vs single-foil in AftSRF (distinct from dead-end #2164)
4. **Panel method Cp as input**: inject inviscid potential-flow Cp as a physics prior input channel — model learns viscous correction only
5. **Pressure Laplacian smoothness loss**: graph-Laplacian penalty on surface pressure predictions (topology-aware, distinct from DCT freq loss)
6. **Stochastic depth curriculum**: block dropping with linear schedule (forces early blocks to be independently predictive)
7. **Researcher-agent hypothesis queue** (in progress): background agent generating fresh ideas informed by new baseline

## Recent Closed Dead Ends

- PR #2214 (edward): Deep Supervision — aux head never activated (aux_loss=0.004-0.008), redundant with --pressure_deep
- PR #2210 (fern): Arc-Length Surface Loss — conflicts with hard-node mining (opposing objectives); p_in +14.2%
- PR #2209 (thorfinn): Attention Register Tokens — p_tan +4.0%, p_in +6.1%
- PR #2205 (nezuko): NOBLE/CosNet — all metrics regressed 5-19%
