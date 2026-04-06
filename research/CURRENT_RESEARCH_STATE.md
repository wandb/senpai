# SENPAI Research State

- **Date:** 2026-04-06 ~04:30 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2184, DCT Freq Loss w=0.05, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| p_in | **13.21** | < 13.21 |
| p_oodc | **7.82** | < 7.82 |
| **p_tan** | **28.50** | **< 28.50** |
| p_re | **6.45** | < 6.45 |

**Latest merge:** PR #2184 (nezuko) — DCT frequency-weighted auxiliary loss (w=0.05, gamma=2.0, alpha=1.5). Exploits spectral bias theory to force attention to high-frequency leading-edge/TE features. Absolute DCT coefficient difference is numerically stable (unlike failed BSP #2172). W&B: 6yfv5lio (s42, p_tan=28.432), etepxvjc (s73, p_tan=28.572). p_tan -0.3% from prior baseline.

**Key note:** p_in/p_oodc slightly regressed vs prior baseline (PR #2130). All 4 metrics together represent the current Pareto frontier. Priority is p_tan.

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-dct-freq" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Note: Current single model (p_tan=28.60) already **BEATS** the 16-seed ensemble (29.1) on p_tan.

## Student Status (~05:30 UTC 2026-04-06)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2181 | GEPS Test-Time Low-Rank Adaptation for OOD Tandem | WIP |
| askeladd | #2188 | MixStyle Tandem Feature Regularization for OOD Generalization | WIP |
| nezuko | #2190 | Laplacian Eigenvector Mesh Positional Encoding | WIP |
| tanjiro | #2189 | DSDF Test-Time Feature Alignment for OOD Tandem | WIP |
| alphonse | #2192 | Stochastic Depth: Layer Drop Regularization for Transolver | WIP |
| thorfinn | #2191 | SE(2) AoA-Aligned Spatial Bias: Chord-Frame Slice Routing | WIP |
| frieren | #2183 | Vorticity Auxiliary Target: explicit wake structure learning | WIP |
| edward | #2193 | Curvature-Conditioned Spatial Bias: True Arc-Length Curvature | WIP — just assigned |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2187 | edward | Normal-Velocity Hard Constraint | **CLOSED** | p_tan +3.0% (29.34 vs 28.50). Multi-foil normal bug + constraint already ~satisfied (|u·n|=0.008). p_in improved -3.3% but irrelevant. |
| #2185 | alphonse | MAE Pretraining (self-supervised geometry init) | **CLOSED** | p_tan +38-67% (2-5x worse). MAE objective conflicts with physics prediction; dataset too small for SSL. |
| #2186 | thorfinn | Panel Cp Residual Target | **CLOSED** | p_tan +349% (catastrophic). Panel solver fails for tandem. |
| #2184 | nezuko | DCT Freq-Weighted Loss (w=0.05) | **MERGED** | p_tan 28.60→28.50 (-0.3%). New baseline. |
| #2182 | tanjiro | Ensemble Distillation (alpha=0.3, 0.5) | **CLOSED** | p_tan +1.6-2.8%. Teacher quality gap. |
| #2175 | askeladd | SWD Tandem Domain Alignment | **CLOSED** | w=0.01 neutral, w=0.05 +3.7%. Distributional diff = real physics. |
| #2180 | edward | Multi-Resolution Hash Grid | **CLOSED** | p_tan +12.2%. Per-sample normalization breaks spatial coherence. |

## Current Research Focus

### Primary target: p_tan = 28.50 → push below 28.0

Single model beats 16-seed ensemble on p_tan (28.50 vs 29.1). More headroom exists — attacking the **NACA6416 representation gap** via input representation, physics constraints, and spectral techniques.

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf` — dedicated aft-foil SRF head
2. `--aug_gap_stagger_sigma 0.02` — tandem scalar domain randomization
3. `--aug_dsdf2_sigma 0.05` — foil-2 DSDF magnitude aug (p_tan -1.4%)
4. `--pcgrad_3way --pcgrad_extreme_pct 0.15` — 2-way PCGrad gradient surgery (p_tan -2.1%)
5. `--gap_stagger_spatial_bias` — tandem-geometry-aware slice routing (p_tan -3.0%)
6. `--dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5` — DCT spectral auxiliary loss (p_tan -0.3%)

**Active experiments (8 students WIP):**
1. **GEPS Test-Time Adaptation** (fern #2181) — LoRA context params + continuity residual TTA at inference. Zero training change.
2. **MixStyle Tandem Feature Regularization** (askeladd #2188) — Feature-space style mixing between tandem samples for OOD generalization.
3. **Laplacian Eigenvector Mesh PE** (nezuko #2190) — Replace Fourier PE with intrinsic graph Laplacian eigenvectors. High-potential positional encoding overhaul.
4. **DSDF Test-Time Feature Alignment** (tanjiro #2189) — Align OOD DSDF feature distribution to training stats at inference. 5-line change, zero training cost.
5. **Stochastic Depth** (alphonse #2192) — randomly skip TransolverBlocks during training (p∈{0.05,0.10,0.15}). Standard DeiT/ViT regularizer; forces each layer to be independently useful. 5-line change.
6. **Curvature-Conditioned Spatial Bias** (edward #2193) — Extend spatial_bias MLP from 6→7 inputs by adding true Menger arc-length curvature at surface nodes. The current channel 24 "curvature proxy" is norm(saf+dsdf) ~ constant (~1) everywhere — NOT actual curvature. True κ peaks at LE/TE, zero at mid-chord. Extends biggest historical win (GSB).
7. **Vorticity Auxiliary Target** (frieren #2183) — KNN-computed ω as auxiliary prediction target. Forces explicit wake learning.
8. **SE(2) AoA-Aligned Spatial Bias** (thorfinn #2191) — Rotate (x, y) to AoA-aligned frame before spatial_bias MLP. Makes GSB routing invariant to AoA changes. Extends the biggest historical win. Expected -1 to -2% p_tan.

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 only), specialized correction heads (aft_srf), gradient surgery (2-way PCGrad), tandem-geometry-aware routing (GSB), geometry-conditioned mechanisms
- **What doesn't work:** Augmentation annealing, foil-1 aug/dropout, fore-foil SRF (4 failures), tandem slice carve-out, 3-way PCGrad, flat-minima-seeking, LR changes, earlier EMA starts, DSDF2 sigma variations, backbone AdaLN, cross-DSDF features, shape similarity bias, differential LR, asymmetric PCGrad, OHEM, wider/deeper SRF, more slices, iterative refinement, tandem surface mixup, BSP spectral loss, foil-1 geometry adapter
- **Confirmed optimal hyperparams:** ema_decay=0.999, ema_start_epoch~140, weight_decay=5e-5, aug_gap_stagger_sigma=0.02, aug_dsdf2_sigma=0.05, lr=2e-4, cosine_T_max=160

## Critical Finding: PCGrad Flag Logic

⚠️ `--pcgrad_3way` in the baseline runs 2-way PCGrad (correct behavior). The flag requires `--disable_pcgrad` to activate true 3-way, which was tested in PR #2147 and FAILED. Baseline is fine as-is.

## Potential Next Research Directions (queue for next idle students)

### Round 5 — Unassigned (from `/research/RESEARCH_IDEAS_2026-04-06_ROUND5.md`)
1. ~~**Normal-Velocity Hard Constraint**~~ → edward #2187
2. ~~**DSDF Test-Time Feature Alignment**~~ → tanjiro #2189
3. ~~**Laplacian Eigenvector Mesh PE**~~ → nezuko #2190
4. **Learned Geometry Tokenizer** — compress foil shape into latent code, inject into backbone.
5. **Stochastic Depth** — randomly drop TransolverBlocks during training. Standard regularizer.
6. **Local KNN Attention** — add local attention alongside global slice attention.
7. **SIREN INR Pressure Decoder** — continuous neural field for pressure prediction. Bold swing.

### Round 6 — Researcher-Agent (2026-04-06) — See `/research/RESEARCH_IDEAS_2026-04-06_ROUND6.md`
1. ~~**Boundary ID 7 Surface Loss Fix**~~ — FALSE ALARM: prepare_multi.py already uses SURFACE_IDS_MULTI=(5,6,7). Comment in train.py:19 is stale.
2. ~~**SE(2) Chord-Aligned Slice Routing**~~ → thorfinn #2191
3. **Hopfield Geometry Memory Bank** — k-NN retrieval: find nearest training geometries at inference, retrieve pressure patterns as SRF prior. Targets NACA6416 distribution shift directly.
4. ~~**Stochastic Depth**~~ → alphonse #2192
5. ~~**Curvature-Conditioned Spatial Bias**~~ → edward #2193 (true Menger curvature, not the existing crude proxy)
6. **Tandem Inter-Foil Distance Feature** — log(min distance to opposite foil) per node. Encodes aerodynamic coupling strength.
7. **Geometry-Adaptive Curvature Loss Weighting** — Upweight surface loss at high-curvature nodes (LE, TE). Spatial analog of merged DCT freq loss.

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
| **Iterative 2-Pass Refinement** | **#2165** | **p_tan +6.6%. 1.3x epoch cost, only 131 epochs. Still converging at wall clock. Correction signal doesn't exist for CFD.** |
| **Tandem Surface Mixup** | **#2167** | **p_tan +5.8-5.9%. Physical inconsistency — aft-foil targets coupled to upstream wake.** |
| **FiLM-Conditioned Fore-Foil SRF** | **#2161** | **p_tan +5.8%. 4th fore-foil SRF failure. Correction norm collapses. Direction exhausted.** |
| **Tandem Pressure Correction MLP** | **#2168** | **p_tan +2.8%. Mixed: p_oodc -2.6%, p_re -3.1% but primary target regressed.** |
| **Wider/Deeper SRF (h=256,384)** | **#2170** | **p_tan +4.0-4.7%. More capacity overfits. h=192 confirmed optimal.** |
| **OHEM Hard Sample Mining** | **#2169** | **p_tan +2.2-2.4%. Redundant with existing 3-layer difficulty system.** |
| **Slice Number 128/144** | **#2171** | **p_tan +3.3/3.8%. More slices = more overfitting, fewer epochs. 96 confirmed.** |
| **BSP Spectral Loss** | **#2172** | **w=0.1 catastrophic collapse, w=0.05 all +3-6%. Spectral bias not the bottleneck.** |
| **Foil-1 Geometry Adapter** | **#2173** | **p_tan +2.1-2.4%. DSDF 4-moment stats too coarse, discard spatial structure.** |
| **Attention Temperature Curriculum** | **#2174** | **p_tan +2.7-4.2%. High initial temp disrupts GSB routing, wastes early epochs.** |
| **Smaller SRF Head (h=128/96)** | **#2178** | **p_tan +3.9-4.5%. h=192 confirmed optimal (full sweep: 96<128<256<384<192).** |
| **Spectral Shaping (k=3 filter)** | **#2176** | **p_tan +2.3% avg. Unstable: s42=28.59 (baseline) vs s73=29.95 (+4.7%).** |
| **Coordinated Tandem Ramp** | **#2177** | **p_tan +2.2% avg. Concurrent schedules interfere during tandem warmup.** |
| **dp/dn=0 Physics Loss (6-seed)** | **#2166** | **p_tan neutral (28.97 vs 28.60, within σ=0.67). Regularizer for p_in/p_re, not p_tan.** |
| **Panel Cp as Input Feature** | **#2179** | **p_tan +3.7%. Single-foil solver lacks tandem interaction. p_oodc/p_re improved.** |
| **Panel Cp Residual Target** | **#2186** | **p_tan +349% (57x worse). asinh mismatch + panel error compounds in tandem. Direction fully exhausted.** |
| **SWD Domain Alignment** | **#2175** | **w=0.01 neutral (+0.5%), w=0.05 all worse (+3.7%). Tandem slice token differences encode real physics — forced alignment counterproductive.** |
| **DCT Frequency-Weighted Loss** | **#2184** | **MERGED. w=0.05 p_tan -0.3% (new baseline). w=0.1 unstable (high seed variance).** |
| **Ensemble Distillation** | **#2182** | **p_tan +1.6-2.8%. Teacher quality gap — ensemble pre-dates GSB/PCGrad, weaker than student.** |
| **Multi-Resolution Hash Grid** | **#2180** | **p_tan +12.2%. Per-sample coord normalization breaks spatial coherence. 1.14M extra params overfit, 20s/epoch overhead.** |
| **MAE Pretraining (SSL geometry init)** | **#2185** | **p_tan +38-67% (2-5x worse). MAE reconstruction conflicts with physics prediction objective. Dataset (1322 samples) too small for SSL benefit. Pretraining wastes 10-20 epochs of supervised budget.** |
| **Normal-Velocity Hard Constraint** | **#2187** | **p_tan +3.0% (29.34 vs 28.50). Multi-foil angle-sorting bug corrupts tandem normals. Constraint already near-satisfied implicitly (|u·n|=0.008 = 0.5% of tangential). Hard constraint removes gradient flexibility.** |

## Ensemble Seed Pool (Complete)

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
