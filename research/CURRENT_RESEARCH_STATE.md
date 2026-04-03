# SENPAI Research State

- **Date:** 2026-04-03 ~20:45 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Ensemble Expansion + Diverse Methods

## Current Baseline (PR #2080 — MERGED 2026-04-03)

| Metric | 8-Seed Ensemble (66-73) | vs prior best (42-49) |
|--------|------------------------|-----------------------|
| p_in | **12.2** | **-1.6%** |
| p_oodc | **6.7** | 0% |
| p_tan | **29.1** | **-1.0%** |
| p_re | **5.8** | 0% |

Seeds: j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv

**Phase 6 wins so far:**
1. T_max=160 (PR #2003) — all metrics improved 1-3.5%
2. Asinh s=0.75 (PR #2054) — OOD improved 4-6%
3. 8-seed ensemble (PR #2076) — ALL metrics -3% to -14%
4. Seeds 66-73 ensemble (PR #2080) — p_in -1.6%, p_tan -1.0%

## Student Status (2026-04-03 ~20:45 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2093 | 16-Seed Combined Ensemble Eval + seeds 100-106 | New |
| thorfinn | #2092 | Standard 3L Seeds 82-89 | New — awaiting pickup |
| nezuko | #2091 | Diverse Hyperparameter Ensemble (lr/T_max/wd sweep) | Running |
| fern | #2090 | Knowledge Distillation from Ensemble | Awaiting pickup |
| askeladd | #2089 | Ensemble Weight Opt + Seeds 90-95 | Running (6/8 seeds) |
| edward | #2087 | Standard 3L Seeds 74-81 | Running (8/8) |
| frieren | #2086 | SAM Phase-Only (rho sweep) | Running (8/8) |
| alphonse | #2068 | Asymmetric Loss v2 (milder α=0.1-0.3) | Sent back, awaiting v2 |

## Current Ensemble Seed Pool

| Batch | Seeds | Status | Notes |
|-------|-------|--------|-------|
| Batch 1 | 42-49 | Done | Original 8-seed ensemble (p_in=12.4) |
| Batch 2 | 66-73 | Done ✓ MERGED | **New best** (p_in=12.2) |
| Batch 3 | 74-81 | Training (edward) | ~1h remaining |
| Batch 4 | 82-89 | New (thorfinn) | Just assigned |
| Batch 5 | 90-95 | Training (askeladd) | ~1h remaining |
| Batch 6 | 100-106 | New (tanjiro) | Just assigned |

**Total seed pool when complete: 48 seeds** (6 batches × 8 seeds)

## Dead Ends This Session (2026-04-03)

| Approach | Finding |
|----------|---------|
| srf4L (4-layer refine) | p_tan +5-7% WORSE (12 seeds confirmed) |
| MC Dropout | Null result — no consistent improvement |
| Packed Ensemble M=4,8 | p_re +6-7% WORSE — insufficient capacity |
| Synthetic data interpolation | 3-10x worse — physically invalid for unstructured meshes |
| Asymmetric loss α=0.5-1.0 | Tradeoff: p_tan -4% but p_in/p_oodc +4% |

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

## Key Research Insights

1. **Ensemble variance reduction is the biggest lever** — keep growing the pool
2. **Different seed batches produce comparable 8-model ensembles** (~12.2-12.4 p_in)
3. **16-seed ensemble should improve further** (1/sqrt(16) vs 1/sqrt(8) = 29% more reduction)
4. **srf4L is a confirmed dead end** — extra surface refine layers hurt tandem transfer
5. **Model too small for packed ensembles** — separate training + averaging strictly better
6. **Architecture modifications absorbed by EMA+Lion** — incremental changes don't stick

## Next Research Priorities

1. **16-seed combined evaluation** (tanjiro #2093) — immediate priority
2. **Knowledge distillation** (fern #2090) — could improve single-model quality
3. **SAM phase-only** (frieren #2086) — could reduce seed variance
4. **Diverse hyperparameter ensemble** (nezuko #2091) — test diversity beyond seed variation
5. **Ensemble weight optimization** (askeladd #2089) — optimize per-model weights post-hoc

## Human Team Directives (from issues #1860, #1834, #1926)
- All ideas from issue #1926 tested (HyperP, MSA, mHC — deprioritized as most exploratory)
- Data constraint from #1834 respected
- Radical architectures tried and failed — ensemble strategy now dominant
