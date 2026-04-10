# SENPAI Research State

- **Date:** 2026-04-10 04:15 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training & Architecture Improvements

## Current Baseline

### Single-Model Baseline (PR #2319 Panel Cp ×0.1, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.709** | < 11.709 |
| **p_oodc** | **7.544** | < 7.544 |
| **p_tan** | **27.402** | < 27.402 |
| p_re | 6.481 | < 6.481 |

## Student Status (2026-04-10 04:15 UTC)

| Student | PR | Experiment | Status | Notes |
|---------|-----|-----------|--------|-------|
| alphonse | #2341 v3 | **Hypernetwork SRF (tandem-only, r=2, α=0.5)** | WIP | v2 beat 3/4 metrics! Iterating for p_in |
| thorfinn | #2351 | **Log-Re-Conditioned Cp** | WIP | Fixes p_re regression from Panel Cp |
| edward | #2350 | **Wake Angle Feature** | WIP | atan2 polar wake direction, 15 LoC |
| fern | #2340 | **Cl/Cd Auxiliary Loss** | WIP | Integral force supervision |
| tanjiro | #2352 | **SRF FiLM Conditioning** | NEWLY ASSIGNED | Re/AoA adaptive surface decoder |
| nezuko | #2353 | **Learnable Cp Scale** | NEWLY ASSIGNED | Self-calibrating physics hint |
| askeladd | #2354 | **Pressure Recovery Feature** | NEWLY ASSIGNED | Inter-foil gap position ratio |
| frieren | #2355 | **Two-Stage SRF** | NEWLY ASSIGNED | Velocity-first then pressure |

## Most Promising Active Experiment

**Hypernetwork SRF v2** (alphonse #2341) showed:
- p_oodc **-2.7%**, p_re **-2.8%**, p_tan **-1.0%** (all beat baseline)
- p_in +1.5% (slight regression, driven by s73)
- v3 with tandem-only activation should fix p_in → potential merge

## Session Results Summary (2026-04-10)

| Round | PR | Experiment | Result | Action |
|-------|-----|-----------|--------|--------|
| 37 | #2341 v1 | Hypernetwork r=8 | p_tan -3.1%, p_in +2.8% | Sent back → v2 |
| 37 | #2341 v2 | Hypernetwork r=2,α=0.5 | 3/4 beat baseline! | Sent back → v3 |
| 37 | #2343 | Arc-Length 1D Conv | 6× slower, p_in +4.2% | Closed |
| 37 | #2332 | Target Noise | diff config, p_tan +4% | Closed |
| 37 | #2339 | Quantile Regression | +92-162% catastrophic | Closed |
| 37 | #2342 | Jacobian Smoothness | +21-34% all metrics | Closed |
| 37 | #2345 | Condition Interpolation | p_in +17.9% | Closed |
| 37 | #2344 | SWA Weight Avg | p_in +41.9% | Closed |
| 37 | #2346 | Focal L1 | p_in +6.2%, p_oodc +11.3% | Closed |
| 37 | #2347 | Sample Curriculum | Neutral | Closed |
| 37 | #2348 | Gumbel MoE SRF | Neutral | Closed |
| 37 | #2349 | Diffusion Decoder | +587-1457% catastrophic | Closed |

## Key Insights Updated

1. **Hypernetwork SRF is the breakthrough direction.** v2 (rank=2, α=0.5) beats baseline on 3/4 metrics. Per-geometry LoRA adaptation works when constrained (weak perturbation principle, same as Panel Cp ×0.1).
2. **Physics-informed features remain the only proven lever.** Panel Cp, wake deficit, TE coord frame all worked. Architecture changes consistently fail or are neutral.
3. **The baseline's loss/training pipeline is extremely well-tuned.** Focal L1, sample curriculum, Gumbel MoE — all neutral or worse. Hard-node mining + PCGrad + tandem boost + Re-stratification cover the loss landscape thoroughly.
4. **Complex architecture changes fail catastrophically.** GRU, diffusion, arc-length conv — all orders of magnitude worse. The Transolver+SRF combo is a strong local optimum.
5. **SWA is redundant with EMA.** The existing EMA (decay=0.999) already provides weight averaging benefits.

## What's Exhausted (DO NOT REVISIT)

*All prior exhausted items plus:*
- Focal L1 surface loss (redundant with hard-node mining)
- Sample difficulty curriculum (redundant with existing multi-layer targeting)
- Gumbel MoE SRF (neutral, per-node routing doesn't help)
- Diffusion surface decoder (per-node diffusion fundamentally limited)
- SWA weight averaging (redundant with EMA)
- Condition-space interpolation (DSDF proxy unreliable)

## Next Research Priorities (Round 38 Ideas)

Already assigned: Log-Re-conditioned Cp (#2351), Wake angle (#2350), FiLM SRF (#2352), Learnable Cp (#2353), Pressure recovery (#2354), Two-stage SRF (#2355)

Unassigned (for future rounds):
1. **Joukowski camber-corrected Cp** (Idea 1) — geometry-aware panel Cp
2. **Vortex-panel induced velocity** (Idea 3) — per-node physics feature
3. **Spectral regularization** (Idea 4) — OOD generalization
4. **Surface-normal coordinate frame** (Idea 7) — BL-physics-aware SRF input
5. **Global aero embedding** (Idea 14) — Cl/Cd embedding for SRF conditioning
