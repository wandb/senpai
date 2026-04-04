# SENPAI Research State

- **Date:** 2026-04-04 ~06:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline (PR #2093 — MERGED 2026-04-04)

| Metric | 16-Seed Ensemble (42-49 + 66-73) | Single-model mean (8 seeds) |
|--------|----------------------------------|-----------------------------|
| p_in | **12.1** | 13.03 |
| p_oodc | **6.6** | 7.83 |
| p_tan | **29.1** | 30.29 |
| p_re | **5.8** | 6.45 |

16-seed ensemble beats 8-seed baseline: p_in -0.8%, p_oodc -1.5%. Seeds 100-106 also trained (mean p_in=13.0), available for future 23-seed evaluation.

## Student Status (2026-04-04 ~06:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2101 | OHEM — Online Hard Example Mining | WIP |
| edward | #2094 | SWAD Dense Weight Averaging | WIP |
| askeladd | #2106 | Fourier Feature Position Encoding | WIP — just assigned |
| frieren | #2100 | Model Scale-Up (3L/96s vs 5L/96s vs 3L/160s vs 4L/128s) | WIP |
| fern | #2104 | Dedicated Aft-Foil SRF Branch (ID=7) | WIP |
| nezuko | #2097 | Deep Supervision (aux_w=0.2) — 8-seed validation | WIP (sent back) |
| alphonse | #2102 | Sin Activation in Surface Refinement Head | WIP |
| thorfinn | #2103 | Iterative Weight-Tied Transolver (K=2,3,4) | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-04 ~06:00 UTC)

| PR | Student | Experiment | Decision | Reason |
|----|---------|-----------|---------|--------|
| #2099 | askeladd | Stochastic Depth (DropPath) | CLOSED | Dead end: all metrics worse (p_oodc +9-10%), 3-layer Transolver too shallow for stochastic depth to provide path diversity — DropPath works in 12-24 layer ViTs, not 3-block models |
| #2090 | fern | Knowledge Distillation (Ensemble→Single) | CLOSED | Clean negative: offline regression KD provides no improvement at convergence; ensemble advantage too small for soft targets to capture |
| #2096 | thorfinn | Learnable Asinh Scale | CLOSED | Scale collapse: s → 0.003, trivial shortcut, 100-175% regression |
| #2098 | alphonse | Asinh Velocity Transform | CLOSED | Negative: all scales degrade p_tan +1-3 pts; velocity doesn't have pressure's outlier problem |
| #2097 | nezuko | Deep Supervision (aux loss) | SENT BACK | Promising: aux_w=0.2 shows p_oodc -1.7%, p_re -1.6% vs 8-seed mean; needs 8-seed validation |

## Current Research Focus

### Primary target: p_tan = 29.1 (our weakest metric, 2.5x worse than p_in)

Active experiments attacking p_tan from multiple angles:
1. **OHEM** (tanjiro #2101) — dynamic per-sample hard mining, upweights difficult tandem configs
2. **Deep Supervision** (nezuko #2097) — auxiliary loss on intermediate features; most promising result this round (p_oodc -1.7%, awaiting 8-seed validation)
3. **Iterative Weight-Tied Transolver** (thorfinn #2103) — deeper effective receptive field via block reuse, targets long-range fore→aft signal propagation
4. **Model Scale-Up** (frieren #2100) — larger model backbone (5L, 4L/128s)
5. **Dedicated Aft-Foil SRF Branch** (fern #2104) — separate refinement MLP exclusively for boundary ID=7 (aft foil tandem nodes), with optional FiLM conditioning on gap/stagger

### Representation and signal quality:
6. **Fourier Feature Position Encoding** (askeladd #2106) — Random Fourier Features for spatial coordinates (Tancik et al., NeurIPS 2020), addressing spectral bias and helping model capture high-frequency suction peaks
7. **Sin Activation in srf_head** (alphonse #2102) — periodic activation for oscillatory pressure distributions

### Regularization and optimization:
8. **SWAD** (edward #2094) — flat-basin checkpoint averaging

## Key Research Insights

1. **Architecture changes are absorbed by EMA+Lion** — Phase 5 showed 6 new architectures all failed; the Transolver is a strong local optimum
2. **Target-space transforms cannot be learnable** — Learnable asinh scale (#2096) confirms: joint optimization creates trivial shortcuts
3. **Velocity doesn't need compression** — Asinh velocity (#2098) confirms: velocity channels are well-behaved, pressure asymmetry is the unique bottleneck
4. **Ensembling is near-saturating** — 16→23 seeds gains diminish; single-model improvements are the priority
5. **Deep supervision is promising** — The only bright result this round: auxiliary loss on intermediate features shows consistent improvements across all 4 metrics at aux_w=0.2
6. **Offline regression KD doesn't work** — (#2090) confirms: the ensemble's single-model gap is too small for soft-target distillation to reliably capture at convergence
7. **Stochastic depth requires depth** — DropPath (#2099) is ineffective for 3-block architectures; requires 6+ layers for meaningful path diversity
8. **Per-node adaptive temperature is a known null** — Ada-Temp tried in PRs #1879, #1793, #1615 — all null results; do NOT reassign

## Confirmed Dead Ends (all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Stochastic Depth (DropPath) | #2099 | All metrics worse; 3-block too shallow for path diversity |
| Knowledge Distillation | #2090 | No improvement at convergence; ensemble advantage too small for regression KD |
| SGDR warm restarts | #2095 | All T_0 values worse — restarts disrupt Lion+cosine |
| SAM Phase-Only | #2086 | SAM destabilizes Lion; best ckpt always pre-SAM |
| srf4L | #2079,2081,2083,2085 | p_tan +5-7% WORSE |
| Mesh interpolation | #2066 | Physically invalid for unstructured CFD |
| SOAP/HeavyBall | #2010,2018-2023 | 2-6% WORSE than Lion |
| Muon (full+hybrid) | #2006 | 30-70% worse |
| XSA attention | #2007 | Redundant with orthogonal slices |
| PirateNets RWF | #2008 | Attenuated by LayerNorm + Lion |
| NOBLE | #2011 | Model too small |
| LinearNO | #2033-2038 | All failed |
| Flow matching | #2036 | 60% worse (deterministic) |
| MARIO latent | #2037 | Redundant geometry encoding |
| Inviscid Cp | #2034 | Single-foil wrong for tandem |
| All-to-all surface attn | #2035 | +8% worse |
| Physics losses | #2016,2023 | WLS instability |
| MC Dropout | #2088 | Null result |
| Packed Ensemble | #2082 | Model too small |
| Ensemble Weight Opt | #2089 | Equal weights — no benefit |
| Diverse Hparam Ensemble | #2091 | Seed diversity strictly better |
| Magnitude-Weighted Loss | #2068 | No sweet spot at any alpha |
| Asinh Velocity Transform | #2098 | Hurts p_tan, velocity doesn't have outlier problem |
| Learnable Asinh Scale | #2096 | Scale collapse to ~0.003, trivial shortcut |
| Ada-Temp (per-node) | #1879, #1793, #1615 | Null result across multiple phases |
| Foil-2 (ID=7) Loss Upweighting | #1893 | Marginal improvement (still only one head) |

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status | Run IDs |
|-------|-------|--------|---------|
| Batch 1 | 42-49 | ✓ BASELINE | f59v5aul, 0yurebjv, rdezx8es, ds12ug79, yu1x0dy0, y147zvh1, lc5cbt4l, 7cxu38oh |
| Batch 2 | 66-73 | ✓ BASELINE | j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv |
| Batch 3 | 74-81 | ✓ Trained | 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4 |
| Batch 4 | 82-89 | ✓ Trained | u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2 |
| Batch 5 | 90-95 | ✓ Trained | ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe |
| Batch 6 | 100-106 | ✓ Trained | 9o85duyc, ec7plfg8, zagg4pfs, 6w86plz1, g00kxdva, jt9hwf40, fom4bzro |

**Total trained: 45 models.** 23-seed evaluation (42-49 + 66-73 + 100-106) available at any time.

## Potential Next Research Directions

**Available (not yet assigned), in priority order:**
1. **Precomputed Pressure-Poisson Soft Constraint** — baked finite-diff Laplacian stencil as offline sparse matrix; penalize Poisson residual at training time; distinct from failed WLS (no numerical instability); targets p_tan/p_oodc; complex (~65 lines)
2. **Mesh-Density Weighted L1 Loss** — upweight fine-mesh nodes (leading edge, trailing edge, suction peak) proportional to 1/local_spacing; targets p_in, p_tan; ~15 lines
3. **Contrastive Tandem-Single Regularization** — push apart hidden representations of tandem vs single-foil configurations; targets tandem OOD generalization; ~25 lines
4. **Progressive Surface Focus Schedule** — curriculum learning: ramp surf_weight from 1 to N over training; ~8 lines
5. **Coordinate Frame Normalization per-Foil** — normalize aft foil coordinates in its own local frame; equivariance for tandem; ~20 lines
6. **Geometry-Conditioned AoA Interpolation** — physics-space interpolation between samples at different AoA (distinct from failed feature-space Mixup); targets p_oodc; moderate complexity

Human researcher messages: #1860 (think bigger), #1834 (data integrity) — both acknowledged and incorporated.
