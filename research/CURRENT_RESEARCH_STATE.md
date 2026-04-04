# SENPAI Research State

- **Date:** 2026-04-04 ~00:15 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline (PR #2080 — MERGED 2026-04-03)

| Metric | 8-Seed Ensemble (66-73) | vs prior best (42-49) |
|--------|------------------------|-----------------------|
| p_in | **12.2** | **-1.6%** |
| p_oodc | **6.7** | 0% |
| p_tan | **29.1** | **-1.0%** |
| p_re | **5.8** | 0% |

## Student Status (2026-04-04 ~00:15 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| edward | #2094 | SWAD Dense Weight Averaging | Running (1 run, ~20 min) |
| askeladd | #2095 | SGDR Warm Restarts (T_0 sweep) | Running (8 runs, ~43 min) |
| thorfinn | #2096 | Learnable Asinh Scale | New — just assigned |
| nezuko | #2097 | Multi-Scale Deep Supervision | New — just assigned |
| alphonse | #2098 | Asinh Velocity Transform | New — just assigned |
| frieren | #2086 | SAM Phase-Only (sent back for timeout fix) | WIP — awaiting rerun |
| fern | #2090 | Knowledge Distillation | Teachers done, distillation pending |
| tanjiro | #2093 | 16-Seed Eval + Seeds 100-106 | Retrain done, next phase pending |

## Current Research Direction: Training Improvements

After confirming that:
1. **Seed diversity > hyperparameter diversity** (nezuko #2091)
2. **Ensemble weight optimization = null result** (askeladd #2089)
3. **Magnitude-weighted loss = dead end** (alphonse #2068)
4. **srf4L = dead end** (4 PRs)

We are now pivoting to **training procedure improvements** that could improve individual model quality:

### Active Experiments (Priority Order)
1. **SWAD** (edward #2094) — flat-basin checkpoint averaging, NeurIPS 2021
2. **SGDR Warm Restarts** (askeladd #2095) — cosine restarts for OOD generalization
3. **Learnable Asinh Scale** (thorfinn #2096) — adaptive pressure compression
4. **Deep Supervision** (nezuko #2097) — auxiliary loss on intermediate features
5. **Asinh Velocity** (alphonse #2098) — compression on Ux/Uy channels
6. **SAM Phase-Only** (frieren #2086) — flat minima via SAM (pending timeout fix)
7. **Knowledge Distillation** (fern #2090) — ensemble → single model
8. **16-Seed Combined Eval** (tanjiro #2093) — quantify N-model scaling

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status | Run IDs |
|-------|-------|--------|---------|
| Batch 2 | 66-73 | ✓ BASELINE | j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv |
| Batch 3 | 74-81 | ✓ Trained | 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4 |
| Batch 4 | 82-89 | ✓ Trained | u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2 |
| Batch 5 | 90-95 | ✓ Trained | ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe |
| Batch 1 | 42-49 | Re-trained (tanjiro) | Pending new run IDs |
| Batch 6 | 100-106 | Pending (tanjiro) | Not started |

**Total trained: 30 models** (8+8+8+6). When tanjiro completes: 45 models.

## Confirmed Dead Ends (Phase 6, all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
| srf4L (4-layer surface refine) | #2079,2081,2083,2085 | p_tan +5-7% WORSE |
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

## Key Research Insights

1. **Ensemble variance reduction is the biggest lever** — 30 models trained, pending combined evaluation
2. **Seed diversity > hyperparameter diversity > weight optimization** (confirmed experimentally)
3. **Architecture modifications absorbed by EMA+Lion** — incremental changes don't stick
4. **Next frontier: training procedure changes** (SWAD, SGDR, deep supervision) that alter optimization dynamics rather than model structure
5. **Asinh transform was a big win for pressure** — testing on velocity channels now

## Human Team Directives (from issues #1860, #1834, #1926)
- All ideas from issue #1926 tested (HyperP, MSA, mHC — deprioritized as most exploratory)
- Data constraint from #1834 respected
- Bold architectures tried and failed — training improvements now dominant strategy
