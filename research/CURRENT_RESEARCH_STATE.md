# SENPAI Research State

- **Date:** 2026-04-03 ~19:40 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Ensemble Expansion + Architectural Exploration

## Current Baseline (PR #2076)

| Metric | 8-Seed Ensemble | Single-model Mean (8 seeds) | Std |
|--------|----------------|-----------------------------|-----|
| p_in | **12.4** | 13.03 | ±0.39 |
| p_oodc | **6.7** | 7.83 | ±0.19 |
| p_tan | **29.4** | 30.29 | ±0.47 |
| p_re | **5.8** | 6.45 | ±0.05 |

Config: Lion lr=2e-4, cosine_T_max=160, 3L Transolver, surface_refine 192/3, asinh s=0.75, residual_prediction, pressure_first.

**Phase 6 wins so far:** T_max=160 (PR #2003), Asinh s=0.75 (PR #2054), 8-Seed Ensemble (PR #2076).

## Student Status (2026-04-03 ~19:40 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| frieren | #2086 | SAM Phase-Only (rho sweep: 0.05/0.1/0.2) | **NEW — just assigned** |
| fern | #2083 | srf4L Ensemble Seeds 50-57 | Running (8/8) |
| thorfinn | #2082 | Packed Ensemble M=2/4/8 | Running (8/8) |
| edward | #2081 | srf4L Ensemble Seeds 42-49 | Running (8/8, first batch crashed) |
| tanjiro | #2080 | Ensemble Seeds 66-73 | Running (8/8, first batch crashed) |
| askeladd | #2079 | srf4L Multi-Seed Validation | Running (8/8, initial group crashed) |
| alphonse | #2068 | Asymmetric/Magnitude-Weighted Surface Loss | Running (8/8) |
| nezuko | #2085 | srf4L Ensemble Seeds 58-65 | Running (8/8) |

## Recent Closed PRs
- #2066 (frieren) — Synthetic data mesh interpolation — **DEAD END** (all metrics 3-10x worse, physically invalid)

## Idle Students
None — all 8 students working.

## Confirmed Dead Ends (Phase 6, cumulative)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Mesh interpolation (synthetic data) | #2066 | Physically invalid for unstructured CFD meshes. All metrics 3-10x worse. |
| SOAP/HeavyBall optimizers | #2010,2018-2023 | SOAP 2-6% WORSE than Lion. False 20% claim from NaN MAE. |
| Muon (full+hybrid) | #2006 | 30-70% worse. Spectral flattening destroys physics signal. |
| XSA attention | #2007 | Redundant with orthogonal slices. |
| PirateNets RWF | #2008 | LayerNorm + Lion attenuate mechanism. |
| NOBLE | #2011 | Model too small, cosine on hidden features unhelpful. |
| Fourier features | #2015 | Marginal. Existing PE sufficient. |
| Deeper model (4-5L) | #2015 | Epoch penalty too large in 180-min timeout. |
| Multi-scale slices | #2017 | 48≈96 slices. Mechanism adapts to any count. |
| Physics losses (vorticity, div-free) | #2016,2023 | WLS gradient instability on unstructured mesh. |
| GeoTransolver cross-attention | #1989 | +9.8% worse, gate near zero. |
| Learned loss weights | #2013 | Collapsed to zero. |
| LinearNO | #2033-2038 | All failed (AAAI paper doesn't transfer). |
| Flow matching/generative | #2036 | 60% worse. Problem is deterministic, not multi-modal. |
| MARIO latent geometry | #2037 | Redundant geometry encoding. |
| Inviscid Cp precomputed | #2034 | Single-foil NeuralFoil wrong for tandem interaction. |
| All-to-all surface attention | #2035 | +8% worse — elliptic coupling handled by existing slices. |
| Asymmetric loss variants | In progress | TBD (alphonse #2068) |

## Current Research Strategy

**Primary direction: Ensemble expansion + quality improvement**
The 8-seed ensemble gave the biggest gains of the programme (-14.4% p_oodc, -10.1% p_re). We are:
1. Building a larger seed pool (16 srf4L seeds training: edward, fern, nezuko)
2. Testing if srf4L (4-layer surface refine) produces better individual models (askeladd #2079)
3. Testing packed ensembles for lower inference cost (thorfinn #2082)
4. Growing the ensemble to 24 seeds (tanjiro #2080)

**Secondary direction: Loss/optimizer improvements**
- SAM phase-only (frieren #2086) — seeks flatter minima for better OOD generalization
- Asymmetric loss (alphonse #2068) — magnitude-weighted L1, pinball loss

## Potential Next Research Directions

1. **Knowledge distillation from ensemble** — distill the 8-seed ensemble into a single model with soft targets. Could improve single-model baseline, making ensembles even better.
2. **MC Dropout in surface refine head** — stochastic inference via K=8 forward passes. Near-zero training overhead, approximates Bayesian model averaging.
3. **HyperP** (from human issue #1926) — Transferable Hypersphere Optimization. Not yet tested.
4. **Larger ensemble evaluation** — if srf4L seeds are better, a 24-seed srf4L ensemble could push metrics significantly lower.
5. **Diverse model ensemble** — ensemble models with different architectures/configs rather than just seeds.
6. **Feature engineering** — local mesh quality metrics, curvature-weighted features.

## Most Recent Human Research Team Directives (from issues)
- Issue #1860: "Think bigger, radical changes" — addressed. Phase 6 trialed all major radical ideas. Only ensemble and target-space tricks worked.
- Issue #1834: "Never use held-out data" — acknowledged and adhered to.
- Issue #1926: All listed ideas tried. HyperP, MSA, mHC not yet attempted.

## Key Research Insights (accumulated)

1. **Lion is the optimal optimizer** — all alternatives (SOAP, Muon, HeavyBall) fail.
2. **Transolver is a strong local optimum** — architecture modifications absorbed by EMA.
3. **Target-space compression helps** — asinh(p, s=0.75) improved OOD metrics.
4. **Ensemble variance reduction is the biggest lever** — 8-seed ensemble: -3% to -14%.
5. **Throughput matters** — any technique adding >15% step time loses epochs and hurts.
6. **Mesh-level operations don't transfer** — unstructured CFD meshes have no spatial correspondence across samples.
