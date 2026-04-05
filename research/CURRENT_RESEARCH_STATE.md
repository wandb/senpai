# SENPAI Research State

- **Date:** 2026-04-05 ~13:30 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2130, GSB + PCGrad, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| p_in | **13.05** | < 13.05 |
| p_oodc | **7.70** | < 7.70 |
| **p_tan** | **28.60** | **< 28.60** |
| p_re | **6.55** | < 6.55 |

**Latest merge:** PR #2130 (fern) — Gap/Stagger Spatial Bias + PCGrad compound. W&B: d7l91p0x (s42, p_tan=28.9), j9btfx09 (s73, p_tan=28.3). p_tan -3.0% from prior baseline.

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

Note: Current single model (p_tan=28.60) already **BEATS** the 16-seed ensemble (29.1) on p_tan.

## Student Status (~13:30 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| askeladd | #2150 | DSDF2 Sigma Optimization: σ={0.03, 0.08} vs baseline 0.05 | WIP |
| tanjiro | #2156 | DSDF-1 Channel Dropout: p={0.2, 0.3} force shape-invariant tandem | WIP — just assigned |
| fern | #2151 | EMA Start Epoch Sweep: {100, 120} vs default ~140 | WIP |
| alphonse | #2157 | Foil Shape Similarity Bias: extend GSB 6D→7D with inter-foil cosine similarity | WIP — just assigned |
| nezuko | #2152 | Augmentation Annealing — linearly decay aug σ over training | WIP |
| thorfinn | #2154 | Cosine T_max Sweep: T_max={140, 180} vs baseline 160 | WIP |
| frieren | #2153 | Gap/Stagger Sigma Increase σ=0.03 — more geometric diversity | WIP |
| edward | #2149 | Learning Rate Sweep: lr={1e-4, 3e-4} vs baseline lr=2e-4 | WIP |

**All 8 students active. Zero idle GPUs.**

⚠️ **tanjiro/slice-count-sweep (#2155):** Merged at 10:03 UTC without result comments. W&B runs still in progress: `fnmekuhn` (slice64-s42), `5yoyekex` (slice64-s73), `c9ev82c9` (slice128-s42), `4vpi753k` (slice128-s73). Monitor — if either slice count beats baseline once converged, update baseline.

## Recently Reviewed (2026-04-05 ~13:30)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2131 | alphonse | Tandem-Slice Carve-Out K=4 (rebased on GSB+PCGrad) | **CLOSED** | p_tan=29.43 vs baseline 28.60 (+2.9%). K=4 redundant with GSB — both address tandem slice specialization, don't compound. |
| #2148 | tanjiro | Gap/Stagger Aug Removal (σ=0) | **CLOSED** | All metrics worse: p_in +3.4%, p_tan +4.0%. |
| #2147 | thorfinn | Actual 3-Way PCGrad | **CLOSED** | All pct values worse than 2-way. |
| #2146 | frieren | Tail EMA Checkpoint Averaging | **CLOSED** | Null result. |
| #2130 | fern | GSB + PCGrad Compound | **MERGED** | p_tan 29.48→28.60 (-3.0%). Current baseline. |

## Current Research Focus

### Primary target: p_tan = 28.60 → push below 28.0

Single model already beats 16-seed ensemble on p_tan. More headroom exists — attacking the **NACA6416 representation gap**.

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf` — dedicated aft-foil SRF head
2. `--aug_gap_stagger_sigma 0.02` — tandem scalar domain randomization
3. `--aug_dsdf2_sigma 0.05` — foil-2 DSDF magnitude aug (p_tan -1.4%)
4. `--pcgrad_3way --pcgrad_extreme_pct 0.15` — 2-way PCGrad gradient surgery (p_tan -2.1%)
5. `--gap_stagger_spatial_bias` — tandem-geometry-aware slice routing (p_tan -3.0%)

**Active experiments (8 students WIP):**
1. **DSDF2 Sigma Optimization** (askeladd #2150) — σ={0.03, 0.08} vs baseline 0.05
2. **DSDF-1 Channel Dropout** (tanjiro #2156) — p={0.2, 0.3} tandem-only dropout of foil-1 DSDF channels; forces shape-invariant wake prediction
3. **EMA Start Epoch Sweep** (fern #2151) — {100, 120} vs default ~140
4. **Foil Shape Similarity Bias** (alphonse #2157) — extend GSB 6D→7D with inter-foil cosine similarity; geometry-type-conditioned routing
5. **Augmentation Annealing** (nezuko #2152) — linearly decay aug σ over training
6. **Cosine T_max Sweep** (thorfinn #2154) — T_max={140, 180} vs baseline 160
7. **Gap/Stagger Sigma Increase** (frieren #2153) — σ=0.03 vs baseline 0.02
8. **Learning Rate Sweep** (edward #2149) — lr={1e-4, 3e-4} vs baseline 2e-4

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 only), specialized correction heads (aft_srf), gradient surgery (2-way PCGrad), tandem-geometry-aware routing (GSB)
- **What doesn't work:** Foil-1 DSDF aug (scaling), fore-foil SRF (unconditioned), tandem slice carve-out (redundant with GSB), 3-way PCGrad, post-hoc weight averaging, flat-minima-seeking (ALL DEAD)
- **Confirmed optimal hyperparams:** ema_decay=0.999, weight_decay=5e-5, aug_gap_stagger_sigma=0.02, aug_dsdf2_sigma=0.05, lr=2e-4

## Critical Finding: PCGrad Flag Logic

⚠️ `--pcgrad_3way` in the baseline runs 2-way PCGrad (correct behavior). The flag requires `--disable_pcgrad` to activate true 3-way, which was tested in PR #2147 and FAILED. Baseline is fine as-is.

## Potential Next Research Directions (not yet assigned)

### High Priority
1. **FiLM-Conditioned Fore-Foil SRF** — additive fore-foil correction conditioned on DSDF1 statistics (shape fingerprint). Key differentiator from failed #2117/#2124: conditioned on foil shape. Expected -2 to -5% p_tan. Medium risk, ~60 LoC.
2. **Tandem Cross-DSDF Features** — explicit inter-foil geometry features (dist_ratio, relative angle) as raw inputs. Expected -2 to -4% p_tan. Low-Med risk, ~20 LoC.
3. **Differential Learning Rates** — higher LR for specialized heads (aft_srf, GSB MLP) vs backbone. Heads may benefit from 5e-4 vs 2e-4 backbone. Low risk.
4. **Fork-then-merge Model Soup** — shared init to epoch 100, branch to 3 seeds, average. Solves loss barrier problem from #2142.
5. **Asymmetric PCGrad** — only project OOD gradients onto in-dist plane. 8 LoC. Expected -1 to -3% OOD.

### Researcher-Agent Ideas
Background researcher-agent running (~13:00 UTC). Ideas will be added when complete.

### Human Researcher Directives
- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. ← Prioritize architecturally novel experiments.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Tandem-Slice Carve-Out (K=4,8) | #2131 | Redundant with GSB; doesn't compound. |
| Gap/Stagger σ=0 (removal) | #2148 | All metrics worse. GSB+aug complementary. |
| Actual 3-Way PCGrad | #2147 | All pct values worse than 2-way. |
| Tail EMA Checkpoint Averaging | #2146 | Null result. Post-hoc weight avg exhausted. |
| Cross-Seed Model Soup | #2142 | Catastrophic. Loss barriers between basins. |
| Foil-2 AoA Rotation Aug | #2138 | Target inconsistency. |
| Per-Foil Physics Normalization | #2136 | +5-19% regression. |
| Fore-Foil TE Relative Coords | #2134 | p_tan +2.6%. |
| AftSRF KNN Context Head | #2134,#2127 | +17% overhead → undertrained. |
| Surface Pressure Gradient Aux Loss | #2129 | 3 rounds, diminishing returns. |
| Foil-1 DSDF Magnitude Aug | #2133 | All σ regress p_tan. |
| Tandem DSDF Channel Mixup | #2132 | No geometric diversity. |
| Flat-minima class | #2086,#2094,#2095,#2120,#2137 | ALL DEAD (SAM/SGLD/SWAD/EMA perturb/SGDR). |
| Loss reformulations | #2112,#2113,#2116 | All worse. |
| Input Feature Noise | #2144 | Catastrophic. |
| DSDF Spatial Dropout | #2143 | Monotonic degradation. |
| Weight Decay 1e-5, 2e-5 | #2145 | 5e-5 confirmed optimal. |
| Gap/Stagger σ=0.01 | #2140 | Worse than 0.02. |
| EMA Decay 0.9995 | #2141 | Regresses with GSB. 0.999 optimal. |
| Reynolds Number Perturbation | #2125 | Null + regression. |
| Fore-foil SRF (all formulations) | #2117,#2124 | All worsen p_tan. |
| Aft/Fore-Foil Loss Upweighting | #2121,#2122 | p_oodc mild benefit, p_tan regression. |
| Various Phase 5 architectures | multiple | 5–59% worse. |

## Ensemble Seed Pool (Complete)

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
