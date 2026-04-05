# SENPAI Research State

- **Date:** 2026-04-06 ~00:15 UTC
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

## Student Status (~00:15 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2161 | FiLM-Conditioned Fore-Foil SRF: shape-aware correction on NACA6416 fore surface | WIP |
| askeladd | #2168 | Tandem Pressure Correction MLP: gated tandem-specific pressure pathway | WIP |
| nezuko | #2169 | Online Hard Example Mining: adaptive per-sample loss upweighting | WIP |
| tanjiro | — | **IDLE** — awaiting new assignment | IDLE |
| alphonse | #2166 | dp/dn=0 Physics Loss: surface normal pressure gradient constraint | WIP |
| thorfinn | #2165 | Iterative 2-Pass Refinement: AlphaFold2-style recycling, zero new params | WIP |
| frieren | — | **IDLE** — awaiting new assignment | IDLE |
| edward | #2167 | Tandem Surface Mixup: between-sample aft-foil node swapping | WIP |

**6 students active, 2 idle (frieren, tanjiro). Researcher-agent generating new hypotheses.**

## Recently Reviewed (2026-04-06 ~00:15)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2164 | frieren | Backbone Gap/Stagger AdaLN | **CLOSED** | adaln_all + gs 4cond p_tan=30.55 (+6.8%), adaln_all Re/AoA p_tan=30.3 (+5.9%). AdaLN disrupts optimized attention routing. |
| #2156 | tanjiro | DSDF-1 Channel Dropout | **CLOSED** | p=0.2 p_tan=30.20 (+5.6%), p=0.3 p_tan=30.40 (+6.3%). Foil-1 DSDF carries critical upstream geometry. p_re improved -9.2% (regularization). |
| #2163 | nezuko | Differential LR | **CLOSED** | Both mult regress p_tan. Uniform LR best. |
| #2162 | askeladd | Tandem Cross-DSDF Features | **CLOSED** | Hand-crafted features add noise. p_tan +4.4%. |
| #2158 | edward | Asymmetric PCGrad | **CLOSED** | All key metrics worse. Symmetric 2-way optimal. |
| #2157 | alphonse | Foil Shape Similarity Bias (GSB 7D) | **CLOSED** | p_tan +3.7%. Sample-level cosine sim too coarse. |
| #2153 | frieren | Gap/Stagger σ=0.03 | **CLOSED** | p_tan +3.3%. σ=0.02 confirmed optimal. |
| #2154 | thorfinn | Cosine T_max Sweep | **CLOSED** | Both +2.8%. T_max=160 confirmed. |
| #2152 | nezuko | Augmentation Annealing | **CLOSED** | p_tan +1.0-2.1%. Constant aug essential. |
| #2130 | fern | GSB + PCGrad Compound | **MERGED** | p_tan 29.48→28.60 (-3.0%). Current baseline. |

## Current Research Focus

### Primary target: p_tan = 28.60 → push below 28.0

Single model already beats 16-seed ensemble on p_tan. More headroom exists — attacking the **NACA6416 representation gap** via input representation and specialized correction heads.

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf` — dedicated aft-foil SRF head
2. `--aug_gap_stagger_sigma 0.02` — tandem scalar domain randomization
3. `--aug_dsdf2_sigma 0.05` — foil-2 DSDF magnitude aug (p_tan -1.4%)
4. `--pcgrad_3way --pcgrad_extreme_pct 0.15` — 2-way PCGrad gradient surgery (p_tan -2.1%)
5. `--gap_stagger_spatial_bias` — tandem-geometry-aware slice routing (p_tan -3.0%)

**Active experiments (6 students WIP):**
1. **FiLM-Conditioned Fore-Foil SRF** (fern #2161) — shape-aware correction on NACA6416 fore surface
2. **Tandem Pressure Correction MLP** (askeladd #2168) — gated tandem-specific pressure pathway
3. **Online Hard Example Mining** (nezuko #2169) — adaptive per-sample loss upweighting
4. **Iterative 2-Pass Refinement** (thorfinn #2165) — AlphaFold2-style recycling, zero new params
5. **dp/dn=0 Physics Loss** (alphonse #2166) — surface normal pressure gradient constraint
6. **Tandem Surface Mixup** (edward #2167) — between-sample aft-foil node swapping

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 only), specialized correction heads (aft_srf), gradient surgery (2-way PCGrad), tandem-geometry-aware routing (GSB), geometry-conditioned mechanisms
- **What doesn't work:** Augmentation annealing, foil-1 aug/dropout, fore-foil SRF (unconditioned), tandem slice carve-out, 3-way PCGrad, flat-minima-seeking, LR changes, earlier EMA starts, DSDF2 sigma variations, backbone AdaLN, cross-DSDF features, shape similarity bias, differential LR, asymmetric PCGrad
- **Confirmed optimal hyperparams:** ema_decay=0.999, ema_start_epoch~140, weight_decay=5e-5, aug_gap_stagger_sigma=0.02, aug_dsdf2_sigma=0.05, lr=2e-4, cosine_T_max=160

## Critical Finding: PCGrad Flag Logic

⚠️ `--pcgrad_3way` in the baseline runs 2-way PCGrad (correct behavior). The flag requires `--disable_pcgrad` to activate true 3-way, which was tested in PR #2147 and FAILED. Baseline is fine as-is.

## Potential Next Research Directions (queue for next idle students)

**Awaiting researcher-agent Round 3 output — new hypotheses needed for frieren and tanjiro.**

### Human Researcher Directives
- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Augmentation Annealing | #2152 | p_tan +1.0-2.1%. Constant aug essential for tandem transfer. |
| EMA Start Epoch Earlier (100, 120) | #2151 | Both regress p_tan +3.7-3.8%. Start ~140 optimal. |
| DSDF2 Sigma (0.03, 0.08) | #2150 | σ=0.05 confirmed optimal. |
| Tandem-Slice Carve-Out (K=4,8) | #2131 | Redundant with GSB. |
| Gap/Stagger σ=0 (removal) | #2148 | All metrics worse. |
| Actual 3-Way PCGrad | #2147 | All pct values worse than 2-way. |
| Tail EMA Checkpoint Averaging | #2146 | Null result. |
| Cross-Seed Model Soup | #2142 | Catastrophic. |
| Foil-2 AoA Rotation Aug | #2138 | Target inconsistency. |
| Per-Foil Physics Normalization | #2136 | +5-19% regression. |
| Fore-Foil TE Relative Coords | #2134 | p_tan +2.6%. |
| AftSRF KNN Context Head | #2134,#2127 | +17% overhead → undertrained. |
| Surface Pressure Gradient Aux Loss | #2129 | 3 rounds, diminishing returns. |
| Foil-1 DSDF Magnitude Aug | #2133 | All σ regress p_tan. |
| Tandem DSDF Channel Mixup | #2132 | No geometric diversity. |
| Flat-minima class | #2086,#2094,#2095,#2120,#2137 | ALL DEAD. |
| Loss reformulations | #2112,#2113,#2116 | All worse. |
| Input Feature Noise | #2144 | Catastrophic. |
| DSDF Spatial Dropout | #2143 | Monotonic degradation. |
| Weight Decay 1e-5, 2e-5 | #2145 | 5e-5 confirmed optimal. |
| Learning Rate ±50% (1e-4, 3e-4) | #2149 | lr=2e-4 confirmed optimal. |
| Cosine T_max {140, 180} | #2154 | Both +2.8% p_tan. T_max=160 confirmed optimal. |
| Asymmetric PCGrad | #2158 | p_in +2.7%, all key metrics worse. Symmetric 2-way optimal. |
| Differential LR (mult=2,3) | #2163 | Both regress p_tan. Uniform LR best. |
| Gap/Stagger σ=0.01 | #2140 | Worse than 0.02. |
| Gap/Stagger σ=0.03 | #2153 | Worse: p_tan +3.3%. σ=0.02 confirmed optimal. |
| EMA Decay 0.9995 | #2141 | Regresses with GSB. 0.999 optimal. |
| Reynolds Number Perturbation | #2125 | Null + regression. |
| Tandem Cross-DSDF Features | #2162 | p_tan +4.4%. Hand-crafted features add noise. |
| Foil Shape Similarity Bias (GSB 7D) | #2157 | p_tan +3.7%. Sample-level cosine sim too coarse. |
| Fore-foil SRF (unconditioned) | #2117,#2124 | Worsen p_tan. |
| Aft/Fore-Foil Loss Upweighting | #2121,#2122 | p_oodc mild benefit, p_tan regression. |
| Various Phase 5 architectures | multiple | 5–59% worse. |
| **Backbone-wide AdaLN** | **#2164** | **p_tan +5.9-6.8%. AdaLN disrupts optimized attention routing.** |
| **DSDF-1 Channel Dropout** | **#2156** | **p_tan +5.6-6.3%. Foil-1 channels need exact values.** |

## Ensemble Seed Pool (Complete)

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
