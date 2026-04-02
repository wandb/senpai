# SENPAI Research State

- **Date:** 2026-04-02 (updated after assignments)
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
| frieren | WIP | #2008 | **PirateNets (Random Weight Factorization)** | **6** |
| alphonse | WIP | #2011 | **NOBLE (Nonlinear Low-Rank Branches)** | **6** |
| nezuko | WIP | #2010 | **HeavyBall Optimizers (SOAP, Cauchy, SF)** | **6** |
| thorfinn | WIP | #2006 | Muon Optimizer + Gram-NS (CRASHED — debugging) | 6 |
| askeladd | WIP | #2007 | XSA Exclusive Self-Attention (CRASHED — debugging) | 6 |
| fern | WIP | #2004 | Noise Schedule Sweep (running, no metrics yet) | 5 |
| tanjiro | WIP | #2001 | Learned Loss Weighting (running, no metrics yet) | 5 |
| edward | WIP | #2003 | Warmup & LR Schedule Sweep (running, no metrics yet) | 5 |

## PRs Ready for Review

None — all students running or debugging.

## Research Focus

### Phase 6 — Active Experiments (from Issue #1926)
- [x] **Muon Optimizer + Gram-NS** → thorfinn (#2006) — CRASHED, debugging
- [x] **XSA (Exclusive Self-Attention)** → askeladd (#2007) — CRASHED, debugging
- [x] **PirateNets** → frieren (#2008) — just assigned
- [x] ~~**Geosolver**~~ → CLOSED (PR #1989 already tried, failed +9.8%)
- [x] **NOBLE** → alphonse (#2011) — just assigned (replaced Geosolver)
- [x] **HeavyBall Optimizers** → nezuko (#2010) — just assigned
- [x] ~~**MSA**~~ — NOT APPLICABLE (wrong problem class, see researcher analysis)
- [ ] HyperP: Hypersphere Optimization (wait for Muon PR #2006 results)
- [ ] mHC: Hyper-Connections (DEFER — too complex for 3-layer model)

### Phase 5 finishing up
3 experiments still running (fern, tanjiro, edward). Will be reviewed when complete, then students reassigned to Phase 6.

## Key Constraints
- Never use raw data files beyond assigned training data (Issue #1834)
- Each GPU has 96GB VRAM; each student has 8 GPUs
- Training capped by SENPAI_MAX_EPOCHS and SENPAI_TIMEOUT_MINUTES
- Throughput is king: >15% step time overhead costs epochs and typically hurts metrics

## Researcher-Agent Key Findings
- **Geosolver/MSA ruled out** — Geosolver already failed (PR #1989), MSA wrong problem class
- **NOBLE is top priority** — cosine nonlinearities match periodic physics patterns
- **mHC deferred** — too complex for 3-layer model, DomainLayerNorm already handles domain specifics
- **HyperP only as Muon follow-up** — don't overlap with PR #2006
- **24-dim input already encodes geometry** — adding more geometry context doesn't help

## Next Assignments (when fern/tanjiro/edward become idle)
1. Need fresh hypotheses from researcher-agent — current issue #1926 list mostly exhausted
2. Consider: data augmentation innovations, loss reformulation, ensemble strategies
3. HyperP only if Muon (#2006) shows promise
