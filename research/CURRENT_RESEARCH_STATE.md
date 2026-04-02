# SENPAI Research State

- **Date:** 2026-04-02 (session start)
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold New Architectures & Optimizers

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
| frieren | WIP | #1998 | Multi-Exit Ensemble (8 parallel) | 5 |
| fern | WIP | #2004 | Noise Schedule Sweep (8 parallel) | 5 |
| tanjiro | WIP | #2001 | Learned Loss Weighting — Kendall Uncertainty (8 parallel) | 5 |
| nezuko | WIP | #2002 | EMA Decay Sweep (8 parallel) | 5 |
| alphonse | WIP | #2000 | OOD-Focused Hard Mining (8 parallel) | 5 |
| edward | WIP | #2003 | Warmup & LR Schedule Sweep (8 parallel) | 5 |
| thorfinn | WIP | #2006 | **Muon Optimizer + Gram-NS** | **6** |
| askeladd | WIP | #2007 | **XSA Exclusive Self-Attention** | **6** |

## PRs Ready for Review

None currently — all students still running.

## Research Focus

### Phase 6 Direction
Phase 6 is about bold, fundamentally new approaches. We are executing ideas from human team Issue #1926 and exploring adjacent fields.

**From Issue #1926 — Progress:**
- [x] **Muon Optimizer + Gram-NS** → thorfinn (#2006) — running
- [x] **XSA (Exclusive Self-Attention)** → askeladd (#2007) — running
- [ ] NOBLE: Nonlinear Low-Rank Branches
- [ ] HyperP: Hypersphere Optimization
- [ ] MSA: Memory Sparse Attention
- [ ] mHC: Hypernetworks
- [ ] PirateNets: Physics-informed architecture
- [ ] Geosolver: Geometry-aware solver
- [ ] HeavyBall optimizers

**Researcher-agent** investigating remaining Phase 6 ideas with deep literature search — results pending.

### Phase 5 experiments finishing up
6 incremental experiments (sweeps of noise, LR, EMA, loss weighting, hard mining, multi-exit) are still WIP as of session start. Will be reviewed on completion.

## Key Constraints
- Never use raw data files beyond assigned training data (Issue #1834)
- Each GPU has 96GB VRAM; each student has 8 GPUs
- Training capped by SENPAI_MAX_EPOCHS and SENPAI_TIMEOUT_MINUTES
- All experiments labeled `phase:6` for Phase 6

## Potential Next Research Directions
1. **PirateNets** — physics-informed residual adaptive networks, directly relevant
2. **NOBLE** — nonlinear low-rank attention branches (architecture improvement)
3. **HeavyBall** — broad optimizer family, drop-in for Lion
4. **Geosolver** — geometry-aware solver, relevant to mesh/airfoil encoding
5. **HyperP** — hypersphere optimization, needs careful LR tuning
6. **MSA** — memory sparse attention, promising for large point clouds
7. **mHC** — hypernetworks for physics-conditioned weight generation
