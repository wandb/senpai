# SENPAI Research State

- **Date:** 2026-04-02 (updated after review round)
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold New Architectures & Optimizers

## Current Baseline (PR #2003 MERGED)

| Metric | New Best | vs Phase 5 |
|--------|----------|-----------|
| val/loss | **0.3761** | -1.8% |
| p_in | **12.5** | -3.5% |
| p_oodc | **8.2** | -1.3% |
| p_tan | **29.8** | -0.7% |
| p_re | **6.5** | -3.0% |

New change: `--cosine_T_max 160` (was 180). W&B: `9ysz96ll`. Single-seed — needs multi-seed validation.

## Student Status

| Student | Status | PR | Experiment | Phase |
|---------|--------|----|------------|-------|
| edward | WIP | #2012 | T_max=160 multi-seed + fine sweep (140-170) | 6 |
| tanjiro | WIP | #2013 | Learned pressure weight multiplier (hybrid) | 6 |
| fern | WIP | #2014 | PirateNet adaptive residual gate on surface refine | 6 |
| frieren | WIP | #2008 | PirateNets (RWF) | 6 |
| thorfinn | WIP | #2006 | Muon Optimizer (lower LR sweep after crash fix) | 6 |
| nezuko | WIP | #2010 | HeavyBall optimizers (all crashed — debugging) | 6 |
| alphonse | WIP | #2011 | NOBLE (not started yet — pod issue?) | 6 |
| askeladd | WIP | #2007 | XSA Exclusive Self-Attention (crashed — debugging) | 6 |

## PRs Ready for Review

None — all students working.

## Research Focus

### Completed Phase 6 Loop Items (from Issue #1926)
- [x] **Muon + Gram-NS** → thorfinn (#2006) — debugging, lower LR needed
- [x] **XSA** → askeladd (#2007) — debugging code regression
- [x] **PirateNets (RWF)** → frieren (#2008) — running
- [x] **NOBLE** → alphonse (#2011) — assigned, not started yet
- [x] **HeavyBall** → nezuko (#2010) — all crashed, debugging
- [x] ~~Geosolver~~ → CLOSED (already failed PR #1989)
- [x] ~~MSA~~ → NOT APPLICABLE (wrong problem class)
- [ ] HyperP — wait for Muon results
- [ ] mHC — DEFER (3-layer model too shallow)

### Phase 6 Follow-ups (from Phase 5 results)
- [x] T_max multi-seed validation → edward (#2012)
- [x] Learned pressure weight → tanjiro (#2013)
- [x] PirateNet surface gate → fern (#2014)

## Key Constraints
- Never use raw data files beyond training data (Issue #1834)
- All experiments use `--cosine_T_max 160` (new from PR #2003)
- Baseline to beat: val/loss 0.3761, p_in 12.5, p_oodc 8.2, p_tan 29.8, p_re 6.5

## Potential Next Research Directions
1. If T_max=160 validates → compound with other improvements
2. T_max=140-150 could be even better (edward #2012 will tell us)
3. If PirateNet surface gate works → explore wider/deeper surface refine
4. Muon at very low LR (0.001-0.003) — still running on thorfinn
5. HyperP as Muon follow-up if Muon shows any promise
