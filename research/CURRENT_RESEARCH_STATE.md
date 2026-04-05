# SENPAI Research State

- **Date:** 2026-04-05 ~02:20 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2130, GSB + PCGrad, 2-seed)

| Metric | 2-seed avg | 2-seed target |
|--------|-----------|---------------|
| p_in | **13.05** | < 13.05 |
| p_oodc | **7.70** | < 7.70 |
| **p_tan** | **28.60** | **< 28.60** |
| p_re | **6.55** | < 6.55 |

**Latest merge:** PR #2130 (fern) — Gap/Stagger Spatial Bias + PCGrad compound. W&B: d7l91p0x (s42, p_tan=28.9), j9btfx09 (s73, p_tan=28.3). p_tan -3.0% from prior baseline.

⚠️ **RESOLVED BUG (PR #2134):** `--aft_foil_srf_context` guard bug confirmed. When bug fixed and context head actually fires, results are WORSE (p_tan +1.2%, p_in +21% due to KNN overhead → undertrained 132 epochs). Context head officially removed from baseline. PR #2127 improvement was seed variance.

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-gsb-pcgrad" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~02:20 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| askeladd | #2140 | Gap/Stagger Aug Sigma Reduction: 0.02→0.01 | WIP — just assigned |
| tanjiro | #2137 | EMA Stochastic Weight Perturbation — σ sweep {5e-4, 1e-3, 3e-3} | WIP |
| fern | #2145 | Weight Decay Sweep: 5e-5 → {1e-5, 2e-5} on new baseline | WIP — just assigned |
| alphonse | #2131 | Tandem-Slice Carve-Out K=4 — rebased 2-seed | WIP — rebasing |
| nezuko | #2141 | EMA Decay Rate Sweep: 0.999→{0.9995, 0.9998} | WIP — just assigned |
| thorfinn | #2143 | DSDF Spatial Dropout — Zero-Out Foil Shape Features for Random Nodes | WIP — just assigned |
| frieren | #2142 | Cross-Seed Model Soup — Weight-Averaging 3 Seeds | WIP — just assigned |
| edward | #2144 | Input Feature Noise Augmentation — Gaussian on all standardized features | WIP — just assigned |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-05 ~02:20)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2130 | fern | Gap/Stagger Spatial Bias (Round 2 rebased) | **SENT BACK** | GSB+context regressed to p_tan=31.1. Sent back for final PCGrad+GSB validation vs baseline 29.48 |
| #2129 | nezuko | Supervised Surface Pressure Gradient Aux Loss (Round 3) | **CLOSED** | v2 mean p_tan=29.778 misses current baseline 29.48. 3 iterations exhausted. |
| #2119 | askeladd | PCGrad 2-way Gradient Surgery (8-seed validation) | **MERGED** | p_tan 30.11→29.48 (-2.1%), 8 seeds confirmed. New baseline. |
| #2135 | edward | Tandem Self-Distillation (EMA teacher) | **CLOSED** | Post-cosine degradation. Only w=0.05 valid; still regressed. |
| #2133 | tanjiro | Foil-1 DSDF Magnitude Augmentation | **CLOSED** | All σ values regress p_tan. Front-foil DSDF aug hurts. Dead end. |
| #2132 | thorfinn | Tandem DSDF Channel Mixup | **CLOSED** | Mixup between NACA0012 samples adds no geometric diversity. |
| #2131 | alphonse | Tandem-Slice Carve-Out (K=4,8) | **SENT BACK** | K=4 beats control -3.7%, K=8 catastrophic. Rebase + K=4 only. |

## Current Research Focus

### Primary target: p_tan = 29.48 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head — PR #2104
2. `--aug_gap_stagger_sigma 0.02`: tandem scalar domain randomization — PR #2115
3. `--aug_dsdf2_sigma 0.05`: foil-2 DSDF magnitude aug — PR #2126 (p_tan -1.4%)
4. ~~`--aft_foil_srf_context`~~: KNN volume context — **REMOVED** (was no-op due to guard bug; when fixed, harmful due to training slowdown) — PR #2127 retracted
5. `--pcgrad_3way --pcgrad_extreme_pct 0.15`: PCGrad 2-way gradient surgery — PR #2119 (p_tan -2.1%)
6. `--gap_stagger_spatial_bias`: tandem-geometry-aware slice routing (gap+stagger in spatial_bias MLP) — PR #2130 (p_tan -3.0%!)

**Active experiments (7 students + 1 idle):**
1. **Gap/Stagger Sigma Reduction σ=0.01** (askeladd #2140) — Test if smaller aug perturbation preserves p_oodc benefit while reducing p_tan penalty. Expected p_tan -0.5% to -1.5%.
2. **EMA Stochastic Weight Perturbation** (tanjiro #2137) — One-time Gaussian perturbation at EMA start, σ sweep {5e-4, 1e-3, 3e-3}. Expected p_tan -1% to -3%.
3. **Gap/Stagger Spatial Bias + PCGrad** (fern #2130) — Final validation: does GSB compound with PCGrad? GSB showed -0.8% p_tan vs control. Expected compound: ~29.24.
4. **Tandem-Slice Carve-Out K=4** (alphonse #2131) — Rebasing onto noam. Original result K=4: p_tan -3.7% vs control.
5. **Per-Foil Physics Normalization** (thorfinn #2136) — Split Cp denominator for fore/aft foil nodes.
6. **Cross-Seed Model Soup** (frieren #2142) — Train 3 seeds (42, 73, 91), average EMA weights post-training. Weight averaging across independent seeds creates flatter loss basin model. Expected p_tan -1% to -3%.
7. **Foil-2 Independent AoA Rotation Aug** (edward #2138) — Decoupled fore/aft AoA rotation for tandem samples.
8. **EMA Decay Rate Sweep** (nezuko #2141) — Test 0.9995 and 0.9998 vs baseline 0.999. Higher decay = more averaging = flatter basin = better OOD. Expected p_tan -0.5% to -1.5%.

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 only), additive specialized correction heads (aft_srf), gradient surgery (PCGrad 2-way), non-local context (KNN wake — pending bug fix)
- **What doesn't work:** Foil-1 DSDF aug, fore-foil SRF, Re/AoA FiLM, self-distillation, DSDF mixup between same-type samples, surface gradient aux loss (3 iterations, weak signal)
- **Mixed results:** Gap/stagger aug helps p_oodc but hurts p_tan; sigma reduction worth testing
- **Critical interaction:** GSB + aft_foil_srf_context (buggy no-op) gave p_tan=31.1 — unexpectedly bad. Likely a seed issue (42/73 vs 42/43).

## Potential Next Research Directions (not yet assigned)

### High Priority
1. **Nezuko reassignment** — immediate. Options:
   - ~~Upstream-only KNN context~~ — DEAD: context head when working is harmful (PR #2134 confirmed)
   - Checkpoint ensemble averaging (save last 3 checkpoints, average weights) — simple, ~20 LoC, often +0.5-2% OOD
   - Attention sparsification / top-k slice assignment — force sparser routing to prevent single slice domination
   - Adversarial tandem augmentation — generate worst-case gap/stagger perturbations via gradient ascent on validation loss
2. ~~aft_foil_srf_context~~ — **DEAD.** Frieren #2134 confirmed: context head when correctly applied is harmful (p_tan +1.2%, p_in +21%). KNN overhead causes undertraining (132 vs 160 epochs). Whole context head approach dead within current training budget.
3. **Upstream-only KNN context** — after bug fix confirmed.
4. **Learnable distance weighting** — replace mean K-neighbor aggregation with attention-weighted.
5. **AoA stagger flip augmentation** — mirror tandem stagger sign to create novel asymmetric configurations.

### Longer-term
6. **Gap/stagger aug removal** — test if completely removing σ=0.02 (removing p_tan penalty) beats current baseline. If yes, remove from baseline and reclaim the metric headroom.
7. **Combined aft_srf_context + GSB + PCGrad** — once bug fix is available, the true three-way compound.
8. **Tandem carve-out K=4 + PCGrad** — orthogonal mechanisms; if alphonse #2131 rebase passes, these may compound.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Foil-2 Independent AoA Rotation Aug | #2138 | Target inconsistency: rotated geometry but flow not re-simulated. No σ beats baseline |
| Per-Foil Physics Normalization | #2136 | phys_stats mismatch → all metrics +5-19%. Don't change normalization without recomputing stats |
| Fore-Foil TE Relative Coords | #2134 | p_tan +2.6% worse than control. TE relative frame doesn't help |
| AftSRF KNN Context Head (when working) | #2134, #2127 | Context head adds 17% overhead → 132 vs 160 epochs → catastrophic: p_in +21%, p_tan +1.2% |
| Surface Pressure Gradient Aux Loss | #2129 (3 rounds) | p_oodc improves -1.5% but p_tan regresses. 3 iterations with diminishing returns. Signal too weak. |
| Foil-1 DSDF Magnitude Augmentation | #2133 | All σ values regress p_tan. Front-foil is KNOWN component in val_tandem_transfer |
| Tandem DSDF Channel Mixup | #2132 | Mixup between NACA0012 samples adds no geometric diversity |
| Tandem Self-Distillation (EMA teacher) | #2135 | Post-cosine degradation; w=0.05 regressed; GPU speed variance invalidates multi-seed |
| Reynolds-Conditional SRF FiLM | #2128 | Null — AdaLN already handles Re/AoA; FiLM redundant |
| Fore-Foil SRF (all formulations) | #2117, #2124 | Split: +9-11% p_tan. Stacked: +1-4% p_tan, p_oodc degrades |
| Aft-Foil Loss Upweighting | #2121 | p_oodc -2% but p_tan +1.5% |
| Fore-Foil Loss Upweighting | #2122 | p_oodc -1.8% but p_tan +1.5-1.9% |
| Aft-Foil Coordinate Frame (dual-frame) | #2107 | 3 iterations, 8 runs, p_in +3%, p_tan barely moves |
| Boundary-ID One-Hot | #2118 | p_tan +2.1%; sparse features disrupt slice assignment |
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

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Confirmed.

## Ensemble Seed Pool (Complete)

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
