# SENPAI Research State

- **Date:** 2026-04-09 16:50 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training & Architecture Improvements

## Current Baseline

### Single-Model Baseline (PR #2290, +Re-Stratified Sampling, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.74** | < 11.74 |
| p_oodc | 7.65 | < 7.65 |
| **p_tan** | **27.90** | < 27.90 |
| p_re | 6.40 | < 6.40 |

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (2026-04-09 16:50 UTC)

| Student | PR | Experiment | Status | Notes |
|---------|-----|-----------|--------|-------|
| thorfinn | #2319 v3 | **Panel Cp ×0.1 tandem-only** | TRAINING (started 15:24, ETA ~18:24) | Most promising — p_tan/p_oodc gains, testing p_in fix |
| edward | #2333 | **Wider SRF head 256** | TRAINING (started 15:39, ETA ~18:39) | |
| tanjiro | #2332 | **Target Noise Regularization** | RESTARTING (pod restarted 16:42) | Val metrics neutral, will re-run |
| fern | #2328 v2 | **AoA Curriculum warmup=20** | RESTARTING (pod restarted 16:44) | Old runs killed, re-implementing |
| nezuko | #2327 v2 | **Sample Mixup (fixed mask)** | RESTARTING (pod restarted 16:44) | Old runs killed (~1hr lost), re-implementing |
| frieren | #2334 | **Checkpoint Soup** | RESTARTING (pod restarted 16:44) | Was doing debug runs, re-implementing |
| alphonse | #2335 | **Gradient Accumulation 2x** | RESTARTING (pod restarted 16:44) | Newly assigned, will implement fresh |
| askeladd | #2336 | **Panel Cp + AoA Curriculum Combo** | RESTARTING (pod restarted 16:44) | Newly assigned, will implement fresh |

**NOTE:** 6 pods restarted at 16:42-16:44 to fix stale label issue. Some active training runs were interrupted. All students should re-pick up their correct assignments from fresh pod starts.

## Round 35/36 Results So Far

### Completed/Reviewed
| PR | Experiment | p_in | p_oodc | p_tan | p_re | Verdict |
|----|-----------|------|--------|-------|------|---------|
| #2325 | KAN Surface Decoder | +4-10% | +1-3% | +6-10% | +1-3% | **Closed** |
| #2326 | Warmup + Cosine | +1% | +7% | -1% | +1.7% | **Closed** (p_oodc regression) |
| #2329 | Log1p Pressure Target | +1.5% | -2.7% | +3.4% | +1% | **Closed** (p_tan regression) |
| #2330 | BL Proxy Feature | +23% | — | — | — | **Closed** (catastrophic) |
| #2331 | Local Re Feature | +5.3% | -0.7% | +0.6% | +2.7% | **Closed** |
| #2332 | Target Noise σ=0.01 | -0.15% | -0.16% | +0.33% | -0.30% | **Preliminary** (neutral, val only) |

### In Progress (awaiting results)
| PR | Experiment | Expected Completion |
|----|-----------|-------------------|
| #2319 v3 | Panel Cp ×0.1 tandem-only | ~18:24 UTC |
| #2333 | Wider SRF head (192→256) | ~18:39 UTC |
| #2328 v2 | AoA Curriculum warmup=20 | ~20:00 UTC (after restart) |
| #2327 v2 | Mixup fixed surface mask | ~20:00 UTC (after restart) |
| #2334 | Checkpoint Soup | ~20:30 UTC (after restart) |
| #2335 | Gradient Accumulation 2x | ~20:30 UTC (after restart) |
| #2336 | Panel Cp + AoA Curriculum Combo | ~20:30 UTC (after restart) |

## Key Insights from Phase 6

1. **Panel Cp as INPUT feature is the only p_tan winner.** v2 tandem-only: p_oodc -2.8%, p_tan -2.9% but p_in +4.5%. v3 tests ×0.1 scaling to fix p_in.
2. **AoA Curriculum shows strongest p_oodc signal** (-5.1%), but p_tan +3.5%. v2 tests shorter warmup=20.
3. **Physics-informed input features remain the strongest lever.** Every durable win came from features or loss reformulation.
4. **Architecture output modifications are safe; backbone changes are dangerous.** Surface refine head changes are OK; attention/block changes regress p_tan.
5. **Target/loss reformulations have mixed results.** Asinh pressure works; log1p does not. DCT loss works; spectral/Sobolev do not.
6. **Training procedure changes show diminishing returns.** Lion+EMA+cosine is hard to beat. Warmup, noise, and curriculum provide marginal benefits at best.

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 — ACTIVE directive (Morgan McGuire, re-raised 2026-04-08)
**"Too many experiments are incremental tweaks. Think BIGGER."**
- Need radical new model changes, data augmentation, data generation
- Not just training schedule tweaks or hyperparameter tuning

## Current Research Focus (Round 36)

### Themes
1. **Iterate on promising results** — Panel Cp ×0.1 (thorfinn v3), AoA curriculum warmup=20 (fern v2)
2. **Combination testing** — Panel Cp + AoA Curriculum together (askeladd #2336)
3. **Training efficiency** — Gradient accumulation (alphonse), checkpoint soup (frieren)
4. **Data augmentation** — Sample mixup with fixed surface mask (nezuko)
5. **Architecture output** — Wider SRF head (edward)

### What's Needed Next (per Issue #1860)
- **Bold new architectures** — Graph neural networks, neural operators (FNO/DeepONet), point cloud transformers
- **Advanced data augmentation** — Physics-informed augmentations preserving flow constraints
- **Data generation** — Synthetic training data from fast solvers (XFOIL, panel methods)
- **Pre-training** — Train on cheap proxy data, fine-tune on expensive CFD
- **Knowledge distillation** — From ensemble to single model
- **Novel loss techniques** — SAM, SWA, loss flooding
- **Mathematical reformulations** — Fourier features, harmonic analysis, Green's functions

### What's Exhausted (DO NOT REVISIT)

- Architecture replacements, attention modifications, auxiliary heads
- Stochastic depth, EMA teacher, GradNorm, PDE losses
- All optimizer variants (Lion+EMA+cosine is optimal)
- Chord-position features, inter-foil coupling, DID/wake SDF features
- Mirror augmentation, synthetic data, per-foil whitening
- Tandem curriculum ramp, FV cell area loss, TTA variance proxy
- Spectral arc-length loss, Sobolev gradient loss, MoE output routing
- Inviscid Cp residual target, KAN surface decoder, warmup with Lion
- Log1p pressure target, BL proxy feature, local Re feature
- Surface-derivative losses on non-uniform meshes
- Target noise regularization (neutral)
