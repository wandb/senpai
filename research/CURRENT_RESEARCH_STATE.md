# SENPAI Research State

- **Date:** 2026-04-03 ~20:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Ensemble Optimization + Variance Reduction

## Current Baseline (PR #2076)

| Metric | 8-Seed Ensemble | Single-model Mean (8 seeds) | Std |
|--------|----------------|-----------------------------|-----|
| p_in | **12.4** | 13.03 | ±0.39 |
| p_oodc | **6.7** | 7.83 | ±0.19 |
| p_tan | **29.4** | 30.29 | ±0.47 |
| p_re | **5.8** | 6.45 | ±0.05 |

Config: Lion lr=2e-4, cosine_T_max=160, 3L Transolver, surface_refine 192/3, asinh s=0.75, residual_prediction, pressure_first.

## Student Status (2026-04-03 ~20:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| frieren | #2086 | SAM Phase-Only (rho sweep) | New — awaiting pickup |
| edward | #2087 | Standard 3L Seeds 74-81 | New — awaiting pickup |
| nezuko | #2088 | MC Dropout in Surface Refine Head | New — awaiting pickup |
| askeladd | #2089 | Ensemble Weight Optimization | New — awaiting pickup |
| fern | #2090 | Knowledge Distillation from Ensemble | New — awaiting pickup |
| tanjiro | #2080 | Ensemble Seeds 66-73 | **DONE — pending 16-seed ensemble eval** |
| thorfinn | #2082 | Packed Ensemble M=2/4/8 | Running (~1.5h left) |
| alphonse | #2068 | Asymmetric/Magnitude-Weighted Surface Loss | Running (~1h left) |

## Key Findings This Session

### srf4L is a Dead End
Two independent experiments (#2079 askeladd, #2083 fern) show 4-layer surface refine **consistently worse**:
- p_tan: **+5.5% to +7.0% regression** (significant)
- Other metrics flat or slightly worse
- Closed 4 srf4L PRs (#2079, #2081, #2083, #2085)
- Added to dead ends list

### Tanjiro's 3L Seeds 66-73 — Consistent with Baseline
8 new standard 3L seeds show same distribution as original 8-seed baseline:
- p_in=13.06±0.34, p_oodc=7.88±0.17, p_tan=30.29±0.63, p_re=6.42±0.09
- Ready for 16-seed ensemble evaluation

## Current Research Strategy

**Exploitation: Ensemble optimization (3 students)**
1. **More seeds** (edward #2087) — seeds 74-81 grow pool to 24
2. **Ensemble weight optimization** (askeladd #2089) — learn non-uniform weights + train 6 more seeds
3. **16-seed ensemble eval** (tanjiro #2080) — pending results

**Exploration: Variance reduction methods (3 students)**
1. **SAM phase-only** (frieren #2086) — flatter minima, lower seed variance
2. **MC Dropout** (nezuko #2088) — stochastic inference from single model
3. **Knowledge distillation** (fern #2090) — distill ensemble into better single model

**Still running:**
- **Packed Ensemble** (thorfinn #2082) — M sub-models in one forward pass
- **Asymmetric Loss** (alphonse #2068) — magnitude-weighted L1, pinball loss

## Confirmed Dead Ends (Phase 6, cumulative)

| Direction | PRs | Finding |
|-----------|-----|---------|
| **srf4L (4-layer surface refine)** | **#2079,2081,2083,2085** | **p_tan +5-7% worse. 12 seeds validated. CLOSED.** |
| Mesh interpolation (synthetic data) | #2066 | Physically invalid for unstructured CFD meshes |
| SOAP/HeavyBall optimizers | #2010,2018-2023 | 2-6% WORSE than Lion |
| Muon (full+hybrid) | #2006 | 30-70% worse |
| XSA attention | #2007 | Redundant with orthogonal slices |
| PirateNets RWF | #2008 | LayerNorm + Lion attenuate mechanism |
| NOBLE | #2011 | Model too small |
| LinearNO | #2033-2038 | All failed |
| Flow matching/generative | #2036 | 60% worse (deterministic problem) |
| MARIO latent geometry | #2037 | Redundant geometry encoding |
| Inviscid Cp precomputed | #2034 | Single-foil wrong for tandem |
| All-to-all surface attention | #2035 | +8% worse |
| Physics losses (vorticity, div-free) | #2016,2023 | WLS gradient instability |

## Key Research Insights

1. **Lion is the optimal optimizer** — all alternatives fail
2. **Transolver is a strong local optimum** — architecture modifications absorbed by EMA
3. **Target-space compression helps** — asinh(p, s=0.75) improved OOD
4. **Ensemble variance reduction is the biggest lever** — 8-seed: -3% to -14%
5. **srf4L hurts** — extra surface refine layers damage tandem transfer generalization
6. **Standard 3L seeds are very consistent** — new seeds match original distribution closely
7. **Throughput matters** — any technique adding >15% step time loses epochs

## Potential Next Research Directions (after current round)

1. Larger ensemble evaluation (24-32 seeds, expected ~40% better than 8-seed)
2. Ensemble of distilled models (if knowledge distillation works)
3. Post-hoc calibration (temperature scaling)
4. HyperP from human issue #1926 (not yet tried)
5. Input feature engineering (curvature, wall distance revisited)
