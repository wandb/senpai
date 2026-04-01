# SENPAI Research State

- **Date:** 2026-04-01 23:25 UTC
- **Advisor branch:** noam
- **Phase:** Transitioning from Phase 5 (incremental tuning) → Phase 6 (bold new ideas)

## Current Baseline

| Metric | Mean (8 seeds) | Best single |
|--------|---------------|-------------|
| val/loss | 0.383±0.003 | 0.378 |
| p_in | 12.95±0.30 | 12.1 |
| p_oodc | 8.31±0.18 | 8.1 |
| p_tan | 30.01±0.52 | 29.2 |
| p_re | 6.70±0.10 | 6.5 |

Baseline from PR #1935: residual_prediction + surface_refine on Transolver with Lion optimizer.

## Student Status

| Student | Status | PR | Experiment | Phase |
|---------|--------|----|------------|-------|
| frieren | WIP | #1998 | Multi-Exit Ensemble | 5 |
| fern | WIP | #2004 | Noise Schedule Sweep | 5 |
| tanjiro | WIP | #2001 | Learned Loss Weighting (Uncertainty) | 5 |
| nezuko | WIP | #2002 | EMA Decay Sweep | 5 |
| alphonse | WIP | #2000 | OOD-Focused Hard Mining | 5 |
| edward | WIP | #2003 | Warmup & LR Schedule Sweep | 5 |
| thorfinn | WIP | #2006 | **Muon Optimizer + Gram-NS** | **6** |
| askeladd | WIP | #2007 | **XSA Exclusive Self-Attention** | **6** |

## PRs Ready for Review

None currently.

## Research Focus

### Phase 6 Direction (starting now)
The human team has explicitly directed us to move beyond incremental tuning (Issue #1860) and pursue bold new ideas (Issue #1926). Phase 6 experiments should test fundamentally different approaches.

**From Issue #1926 — Ideas to try:**
- [x] **Muon Optimizer + Gram-NS** → assigned to thorfinn (#2006)
- [x] **XSA (Exclusive Self-Attention)** → assigned to askeladd (#2007)
- [ ] NOBLE: Nonlinear Low-Rank Branches for Transformers
- [ ] HyperP: Hypersphere Optimization
- [ ] MSA: Memory Sparse Attention
- [ ] mHC: Hypernetworks
- [ ] PirateNets: Physics-informed architecture
- [ ] Geosolver: Geometry-aware solver
- [ ] HeavyBall optimizers (full collection)

### Phase 5 experiments still running
6 incremental experiments (sweeps of noise, LR, EMA, loss weighting, hard mining, multi-exit) are still WIP. These will be reviewed when complete, but the research direction is shifting to Phase 6.

## Key Constraints
- Never use raw data files beyond assigned training data (Issue #1834)
- Each GPU has 96GB VRAM
- Training capped by SENPAI_MAX_EPOCHS and SENPAI_TIMEOUT_MINUTES
- All experiments must include baseline comparison runs

## Potential Next Research Directions
1. **Muon optimizer variants** — if initial results are promising, explore Moon variant, gradient clipping interactions
2. **Architecture overhauls** — PirateNets (physics-informed), Geosolver, NOBLE
3. **Attention mechanism changes** — XSA, MSA (memory sparse)
4. **New optimizers** — HeavyBall collection, HyperP
5. **Hypernetworks** — mHC for learning per-sample adaptations
6. **Data augmentation** — physics-aware augmentation, synthetic data generation from training data only
