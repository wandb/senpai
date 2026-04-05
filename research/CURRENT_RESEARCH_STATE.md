# SENPAI Research State

- **Date:** 2026-04-05 ~02:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2119, PCGrad 2-way, 8-seed mean)

| Metric | 8-seed mean | 2-seed target |
|--------|------------|---------------|
| p_in | **13.20** | < 13.20 |
| p_oodc | **7.91** | < 7.91 |
| **p_tan** | **29.48** | **< 29.48** |
| p_re | **6.50** | < 6.50 |

**Latest merge:** PR #2119 (askeladd) — PCGrad 2-way gradient surgery (8 seeds: tmqq1xlo, g0ukmibf, 1fge4f0m, s0akrj5a, kxs75gcq, 0m1vbsam, 75d4hhzm + rebased s42/s73: jpe1t13t, cdccuyl7).

⚠️ **CRITICAL BUG:** `--aft_foil_srf_context` is a NO-OP — guard bug `if aft_srf_head is not None:` is False when context=True. Context head NEVER applied. PR #2127's improvement was likely seed variance. True pre-PCGrad baseline is PR #2126 (p_tan=30.11). Bug fix in frieren's #2134; awaiting results.

⚠️ **Post-cosine degradation risk:** Some GPUs run faster (~43-45s/epoch vs ~63s), reaching 220+ epochs past cosine_T_max=160, causing OOD metric regression. Validate all runs stop near epoch 160.

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~02:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| askeladd | #2139 | PCGrad + DSDF2 Aug Compound Validation | WIP — just assigned |
| tanjiro | #2137 | EMA Stochastic Weight Perturbation — σ sweep {5e-4, 1e-3, 3e-3} | WIP |
| fern | #2130 | Gap/Stagger Spatial Bias — rebased 2-seed validation | WIP — rebasing |
| alphonse | #2131 | Tandem-Slice Carve-Out K=4 — rebased 2-seed | WIP — rebasing |
| nezuko | #2129 | Supervised Surface Pressure Gradient Aux Loss v2 (per-foil fix) | WIP — revising |
| thorfinn | #2136 | Per-Foil Physics Normalization — Fix Aft-Foil Cp Denominator | WIP |
| frieren | #2134 | Fore-Foil TE Relative Coords + CRITICAL BUG FIX | WIP — awaiting results |
| edward | #2138 | Foil-2 Independent AoA Rotation Aug — Decoupled Tandem Geometry | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-05 ~02:00)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2119 | askeladd | PCGrad 2-way Gradient Surgery (8-seed validation) | **MERGED** | p_tan 30.11→29.48 (-2.1%), p_oodc -3.2%, all metrics beat baseline. 8 seeds confirmed. |
| #2135 | edward | Tandem Self-Distillation (EMA teacher) | **CLOSED** | Post-cosine degradation on most seeds. Only w=0.05 valid; still regressed vs baseline. |
| #2133 | tanjiro | Foil-1 DSDF Magnitude Augmentation | **CLOSED** | All σ values regress p_tan. Front-foil is KNOWN component in val_tandem_transfer — augmenting it hurts. Dead end. |
| #2132 | thorfinn | Tandem DSDF Channel Mixup | **CLOSED** | Mixup between NACA0012 samples creates more NACA0012, not NACA6416. No geometric diversity. Approach unsound. |
| #2131 | alphonse | Tandem-Slice Carve-Out (K=4,8) | **SENT BACK** | K=4 beats control -3.7%, K=8 catastrophic. Rebase onto noam, validate K=4 only. |
| #2130 | fern | Gap/Stagger-Conditioned Spatial Bias | **SENT BACK** | Wrong feature indices (22:24 vs 21:23) fixed. Rebase onto noam for clean validation. |
| #2129 | nezuko | Supervised Surface Pressure Gradient Aux Loss | **SENT BACK** | Cross-foil gradient bug. Sent back for per-foil fix + aft_srf_context rebase. |
| #2128 | edward | Reynolds-Conditional SRF FiLM | **CLOSED** | Null result. AdaLN already handles Re/AoA conditioning. |

## Current Research Focus

### Primary target: p_tan = 29.48 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head (ID=7) — PR #2104
2. `--aug_gap_stagger_sigma 0.02`: tandem scalar domain randomization — PR #2115
3. `--aug_dsdf2_sigma 0.05`: foil-2 DSDF magnitude aug — PR #2126 (p_tan -1.4%)
4. `--aft_foil_srf_context`: KNN volume context (BUG — was no-op, improvement was seed variance) — PR #2127
5. `--pcgrad_3way --pcgrad_extreme_pct 0.15`: PCGrad 2-way gradient surgery — PR #2119 (p_tan -2.1%)
6. `--surface_refine`, `--residual_prediction`, `--pressure_first`, `--pressure_deep`, `--asinh_pressure 0.75`, etc.

**Active experiments (8 students):**
1. **PCGrad + DSDF2 Aug Compound** (askeladd #2139) — Validates PCGrad and DSDF2 gains stack; both were merged sequentially, never tested simultaneously from scratch.
2. **EMA Stochastic Weight Perturbation** (tanjiro #2137) — One-time Gaussian perturbation at EMA start, σ sweep {5e-4, 1e-3, 3e-3}.
3. **Gap/Stagger-Conditioned Spatial Bias** (fern #2130) — Rebasing onto noam with correct feature indices (21:23).
4. **Tandem-Slice Carve-Out K=4** (alphonse #2131) — Rebasing onto noam, K=4 only (K=8 catastrophic).
5. **Supervised Surface Pressure Gradient Aux Loss v2** (nezuko #2129) — Per-foil fix, w=0.05/0.10 sweep.
6. **Per-Foil Physics Normalization** (thorfinn #2136) — Split Cp denominator for fore/aft foil nodes.
7. **Fore-Foil TE Relative Coords + Bug Fix** (frieren #2134) — TE relative coords feature + critical aft_srf_context guard fix.
8. **Foil-2 Independent AoA Rotation Aug** (edward #2138) — Decoupled fore/aft AoA rotation for tandem samples.

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 only), additive specialized correction heads (aft_srf), gradient surgery (PCGrad 2-way), target transforms (asinh), non-local context (KNN wake — pending bug fix validation)
- **What doesn't work:** Foil-1 DSDF aug (hurts known component), fore-foil SRF, Re/AoA FiLM, self-distillation at current training lengths, DSDF mixup between same-type samples
- **Critical interaction:** gap_stagger aug + aft_foil_srf NOT additive alone — helps p_oodc but hurts p_tan by +1.6%
- **Post-cosine degradation:** GPU speed variance causes some runs to overshoot cosine_T_max=160, causing OOD metric collapse. Must validate epoch count.

## Potential Next Research Directions (not yet assigned)

### Top Priority

1. **aft_foil_srf_context bug fix validation** — Frieren #2134 critical. Once TE-coords + true context head results arrive, decide whether to cherry-pick bug fix and re-test KNN context.
2. **Upstream-only KNN context** — Filter vol neighbors to x_vol < x_aft for wake-specific context. Follow-up to PR #2127 (after bug fix).
3. **Learnable distance weighting** — Replace mean K-neighbor aggregation with attention-weighted. Upgrade to PR #2127 mechanism.
4. **Gap/stagger sigma reduction** (0.02→0.01) — reduce p_tan hurt while keeping p_oodc benefit.
5. **aft_foil_tv_loss** — Chord-wise TV regularization on aft-foil pressure predictions. ⚠️ DEPRIORITIZED — nezuko #2129 gradient aux (same family) showed weak results.
6. **foil1-relative-coords** — Add (x,y) relative to fore-foil TE as features to context head (subset of frieren #2134).

### Existing Queue

7. **Combined foil1+foil2 aug at lower σ** — ⚠️ DEAD — Foil-1 DSDF aug regresses at all σ. Only foil-2 aug productive.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Foil-1 DSDF Magnitude Augmentation | #2133 | All σ values regress p_tan. Front-foil is KNOWN component in val_tandem_transfer |
| Tandem DSDF Channel Mixup | #2132 | Mixup between NACA0012 samples adds no geometric diversity |
| Tandem Self-Distillation (EMA teacher) | #2135 | Post-cosine degradation; w=0.05 regressed; GPU speed variance invalidates multi-seed comparison |
| Reynolds-Conditional SRF FiLM | #2128 | Null — AdaLN already handles Re/AoA; FiLM redundant |
| Fore-Foil SRF (all formulations) | #2117, #2124 | Split: +9-11% p_tan. Stacked: +1-4% p_tan, p_oodc degrades |
| Aft-Foil Loss Upweighting | #2121 | p_oodc improves -2% but p_tan regresses +1.5% |
| Fore-Foil Loss Upweighting | #2122 | p_oodc improves -1.8% but p_tan regresses +1.5-1.9% |
| Aft-Foil Coordinate Frame (dual-frame) | #2107 | 3 iterations, 8 runs, p_in +3%, p_tan barely moves |
| Boundary-ID One-Hot (sparse 3-dim input) | #2118 | p_tan +2.1%; sparse features disrupt slice assignment |
| Charbonnier Loss | #2116 | All eps values degrade all metrics |
| Mesh-Density Weighted L1 | #2112 | All metrics regressed 5–16% |
| Smooth L1 / Huber Loss | #2113 | Catastrophic |
| Gradient Centralization (GC-Lion) | #2114 | Incompatible with Lion sign operation |
| Reynolds Number Perturbation Augmentation | #2125 | Null result + p_tan regression |
| Langevin Gradient Noise (SGLD) | #2120 | No improvement; Lion already provides implicit gradient noise |
| Fourier Feature Position Encoding | #2106 | p_oodc +4.8%, p_re +6.2% regression |
| Deep Supervision (aux loss) | #2097 | p_tan +2.7% regression |
| SWAD | #2094 | Catastrophic |
| SGDR warm restarts | #2095 | All T_0 values worse |
| SAM Phase-Only | #2086 | Destabilizes Lion |
| srf4L (deeper SRF) | #2079-2085 | p_tan +5-7% WORSE |
| FiLM on gap/stagger | #2104 | p_oodc +41.6% catastrophe |
| Contrastive tandem-single regularization | #2109 | Hypothesis falsified |
| Model Scale-Up | #2100 | NOT capacity-limited |
| OHEM (hard example mining) | #2101 | No improvement |
| TTA via AoA Perturbation | #2111 | Self-defeating under timeout |
| Progressive surface focus | #2110 | p_in regresses |
| Asymmetric Asinh Scales | #2108 | All metrics worse |
| Weight-Tied Iterative Transolver | #2103 | Monotonic degradation |
| SIREN in SRF Head | #2102 | Monotonic degradation |
| All Phase 5 architectures (GNOT, Galerkin, etc.) | multiple | 5–59% worse |

## Human Researcher Directives

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Responded with Phase 5 report. Current phase focuses on geometric augmentation (DSDF family) and physical inductive biases (KNN context, relative coordinates).
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Confirmed.

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status |
|-------|-------|--------|
| Batch 1 | 42-49 | ✓ BASELINE (combined aft_srf + aug) |
| Batch 2 | 66-73 | ✓ ENSEMBLE |
| Batch 3 | 74-81 | ✓ Trained |
| Batch 4 | 82-89 | ✓ Trained |
| Batch 5 | 90-95 | ✓ Trained |
| Batch 6 | 100-106 | ✓ Trained (available for 23-seed ensemble) |

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
