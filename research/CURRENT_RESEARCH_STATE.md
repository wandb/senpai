# SENPAI Research State

- **Date:** 2026-04-09 12:55 UTC
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

## Student Status (2026-04-09 12:55 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2325 | **KAN Surface Decoder** | TRAINING (2 seeds, started 10:02, ETA ~13:02) |
| thorfinn | #2319 | **Panel-Method Cp Feature v2 (tandem-only)** | TRAINING (2 seeds, started 10:39, ETA ~13:39) |
| edward | #2326 | **Warmup + Cosine Schedule** | IMPLEMENTING |
| nezuko | #2327 | **Sample Mixup Augmentation** | NEWLY ASSIGNED |
| fern | #2328 | **AoA Curriculum Training** | NEWLY ASSIGNED |
| askeladd | #2329 | **Log1p Pressure Target Transform** | NEWLY ASSIGNED |
| alphonse | #2330 | **Boundary Layer Proxy Feature** | NEWLY ASSIGNED |
| frieren | #2331 | **Local Reynolds Number Feature** | NEWLY ASSIGNED |

## Round 34 Results Summary — 7 experiments reviewed

**Critical finding: Only one experiment showed ANY improvement — Panel-Method Cp Feature (p_tan -3.4%).**

| PR | Experiment | p_in | p_oodc | p_tan | p_re | Verdict |
|----|-----------|------|--------|-------|------|---------|
| #2315 | Tandem Curriculum Ramp | -2.1% | +2.7% | +1.3% | +1.3% | **Closed** (trades p_in for OOD) |
| #2319 | Panel-Method Cp Feature | +5.2% | -0.9% | **-3.4%** | +0.8% | **Sent back** (iterate: tandem-only) |
| #2317 | FV Cell Area Loss Weight | 70x worse | — | — | — | **Closed** (catastrophic, 4 iterations) |
| #2322 | TTA Norm Adaptation | +98% | +87% | +25% | +72% | **Closed** (variance min → constant pred) |
| #2320 | Spectral Arc-Length Loss | -1.1% | +1.3% | +1.1% | +1.2% | **Closed** (neutral, redundant w/ DCT) |
| #2321 | Sobolev Gradient Loss | +5.2% | +2.6% | +9.5% | 0% | **Closed** (noisy FD on non-uniform mesh) |
| #2323 | MoE Output Routing | +12.8% | +16.4% | +10.7% | +7.5% | **Closed** (conflicting signals, gate instability) |
| #2324 | Inviscid Cp Residual Target | +26% | +66% | +9.3% | +48% | **Closed** (crude thin-airfoil target) |

## Key Insights from Round 34

1. **Panel Cp as INPUT feature works; as TARGET reformulation fails.** Thorfinn's feature approach improved p_tan -3.4%; frieren's target approach regressed +66% p_oodc. The model benefits from physics hints but needs to learn raw pressure directly.
2. **Surface-derivative losses fail on non-uniform meshes.** Both Sobolev gradient (p_tan +9.5%) and spectral arc-length (neutral, redundant with DCT) added no value. Our existing DCT frequency loss already captures spectral information.
3. **Additional regime-handling mechanisms hurt.** MoE output routing (+10-16%) conflicts with existing domain_velhead + pcgrad_3way + re_stratified_sampling. The model already handles regime diversity sufficiently.
4. **TTA variance minimization is fundamentally broken for regression.** No viable self-supervised signal exists for cheap test-time adaptation.
5. **The strongest signal remains physics-informed input features.** Every durable win came from features (wake deficit, TE coord frame, gap-stagger spatial bias) or loss reformulation (DCT, residual prediction).

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 — ACTIVE directive (Morgan McGuire, re-raised 2026-04-08)
**"Too many experiments are incremental tweaks. Think BIGGER."**

## Current Research Focus (Round 35)

### Themes
1. **Physics-informed input features** (alphonse #2330 BL proxy, frieren #2331 local Re) — follows the proven pattern
2. **Target/loss reformulation** (askeladd #2329 log1p pressure) — changes how the model sees pressure
3. **Training procedure** (edward #2326 warmup, fern #2328 AoA curriculum, nezuko #2327 mixup) — improves optimization/generalization
4. **Architecture output** (tanjiro #2325 KAN decoder) — modifies only the safe output head
5. **Iterating on success** (thorfinn #2319 v2 tandem-only panel Cp) — refining the only p_tan winner

### What's Exhausted (DO NOT REVISIT)

- Architecture replacements, attention modifications, auxiliary heads
- Stochastic depth, EMA teacher, GradNorm, PDE losses
- All optimizer variants (Lion+EMA+cosine is optimal)
- Chord-position features, inter-foil coupling, DID/wake SDF features
- Mirror augmentation, synthetic data, per-foil whitening
- Tandem curriculum ramp (trades p_in for OOD)
- FV cell area loss (destroys optimization)
- TTA with variance proxy (broken for regression)
- Spectral arc-length loss (redundant with DCT freq loss)
- Sobolev gradient loss (noisy on non-uniform mesh)
- MoE output routing (conflicting signals with existing mechanisms)
- Inviscid Cp residual target (thin-airfoil too crude for target)
- Surface-derivative losses on non-uniform meshes (general conclusion)
