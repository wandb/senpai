# SENPAI Research State

- **Date:** 2026-04-09 10:02 UTC
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

Single-model beats ensemble on p_in (11.74 vs 12.1) and p_tan (27.90 vs 29.1). Ensemble still leads on p_oodc (7.65 vs 6.6) and p_re (6.40 vs 5.8).

## Student Status (2026-04-09 10:02 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2325 | **KAN Surface Decoder** | IMPLEMENTING (picked up PR, coding) |
| thorfinn | #2319 | **Panel-Method Cp Feature v2** | SENT BACK (iterating: tandem-only or scaled feature) |
| edward | #2317 | **FV Cell Area Loss Weight v2** | TRAINING (2 seeds, started 09:05 UTC, ETA ~12:05) |
| nezuko | #2322 | **Test-Time Norm Adaptation** | TRAINING (2 seeds, started 09:01 UTC, ETA ~12:01) |
| fern | #2321 | **Sobolev Gradient Loss** | TRAINING (2 seeds, started 09:10 UTC, ETA ~12:10) |
| askeladd | #2320 | **Spectral Arc-Length Loss** | TRAINING (2 seeds, started 09:18 UTC, ETA ~12:18) |
| alphonse | #2323 | **MoE Output Routing** | TRAINING (2 seeds, started 09:29 UTC, ETA ~12:29) |
| frieren | #2324 | **Inviscid Cp Residual Target** | TRAINING (2 seeds, started 09:36 UTC, ETA ~12:36) |

## PRs Ready for Review

None currently. Next wave of results expected ~12:00-12:36 UTC (6 experiments).

## Latest Reviews (2026-04-09 10:00 UTC)

### Thorfinn #2319 — Panel-Method Cp Feature — SENT BACK 🔄

**First experiment to significantly improve p_tan (-3.4%).** Very promising direction.

| Metric | 2-seed avg | Baseline | Δ |
|--------|------------|----------|---|
| p_in | 12.354 | 11.742 | +5.2% ❌ |
| p_oodc | 7.576 | 7.643 | -0.9% ✅ |
| p_tan | 26.918 | 27.874 | **-3.4%** ✅ |
| p_re | 6.471 | 6.419 | +0.8% ❌ |

**Issue:** p_in regresses +5.2% and val_loss is 60% higher (0.59 vs 0.37). Sent back to try:
1. Apply panel Cp only to tandem samples (zero for single-foil)
2. Or scale feature by 0.1-0.3

### Tanjiro #2315 — Tandem Curriculum Ramp (fixed) — CLOSED ❌

Mixed: p_in -2.1% but p_oodc +2.7%, p_tan +1.3%, p_re +1.3%. Net regression. Curriculum approach fundamentally trades OOD for in-dist.

### Round 33 Results — ALL CLOSED ❌

| PR | Experiment | Key Result |
|----|-----------|------------|
| #2308 | Aux AoA Head v2 | p_in +3.2%, p_tan +1.3% |
| #2311 | Condition Token v2 | p_tan +3.6% |
| #2314 | SE Slice Tokens | p_tan +3.1% |
| #2316 | Stochastic Depth | All +4-10% |
| #2318 | EMA Teacher | Catastrophic (+33-41%) |

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 — ACTIVE directive (Morgan McGuire, re-raised 2026-04-08)
**"Too many of your current experiments are incremental tweaks. Think BIGGER."**

**Advisor response (Round 34):** Pivoted from architecture modifications to:
- Loss reformulation (spectral arc-length loss, Sobolev gradient loss)
- Prediction target change (inviscid Cp residual)
- Inference-time adaptation (TTA normalization)
- Output-level specialization (MoE output routing)
- Output architecture (KAN surface decoder)

## Current Research Focus and Themes

### Round 34 In-Flight Experiments (8 GPUs)
| Student | PR | Direction | Category | Target |
|---------|-----|-----------|----------|--------|
| tanjiro | #2325 | **KAN Surface Decoder** — B-spline activations in SRF output head | Output architecture | all |
| thorfinn | #2319 | **Panel-Method Cp Feature v2** — tandem-only or scaled inviscid input | Physics input (iterating) | p_tan |
| edward | #2317 | **FV Cell Area Loss Weight v2** — geometry-aware node weighting | Loss weighting | all |
| nezuko | #2322 | **Test-Time Norm Adaptation** — adapt LayerNorm at inference | Inference adaptation | p_oodc, p_re |
| fern | #2321 | **Sobolev Gradient Loss** — dCp/ds derivative supervision | Loss reformulation | p_in, p_oodc |
| askeladd | #2320 | **Spectral Arc-Length Loss** — FFT on arc-length-sorted surface | Loss reformulation | all |
| alphonse | #2323 | **MoE Output Routing** — soft expert blending by flow regime | Output specialization | p_tan, p_re |
| frieren | #2324 | **Inviscid Cp Residual Target** — predict viscous correction | Target reformulation | p_oodc, p_re |

### Key Mechanistic Insights (accumulated)

1. **Every durable improvement came from: (a) physics-motivated input features, or (b) loss reformulation.**
2. **All architecture replacements failed in Phase 5** (GNOT, Galerkin, GNN, DeepONet, FNO, INR — all 30-60% worse). Transolver is a remarkably strong local optimum.
3. **Tandem p_tan bottleneck:** Largest absolute error gap. All attention modifications hurt p_tan — the shared K/V regularization is load-bearing.
4. **Panel-method Cp feature is the first to improve p_tan (-3.4%)!** Physics-informed input features that encode inviscid theory help tandem predictions. The p_in regression (+5.2%) needs solving via feature scaling or tandem-only application.
5. **p_oodc and p_re gaps to ensemble are large:** Single model 7.65 vs ensemble 6.6 (p_oodc), 6.40 vs 5.8 (p_re). These need OOD-focused approaches.
6. **Shared K/V is load-bearing regularization** — per-head projections destroy OOD generalization.
7. **Compute budget constraint is real:** Any approach adding >10% per-step overhead is unviable (180-min timeout, ~147 epochs).

### What Works (confirmed and merged)

| Direction | PR | Impact |
|-----------|-----|--------|
| Re-Stratified Sampling | #2290 | p_in -1.3%, p_tan -0.8% |
| Cosine T_max=150 | #2251 | p_in -0.7%, p_oodc -1.1%, p_tan -0.8% |
| Wake deficit feature | #2213 | p_in -4.1%, p_re -1.7% |
| TE coordinate frame | #2207 | p_in -5.4% |
| Pressure-first deep | #2155 | p_in -4.8% |
| DCT frequency loss | #2184 | p_re -1.5%, p_tan -0.3% |
| Gap-stagger spatial bias | #2130 | p_tan -3.0% |
| Residual prediction | #1927 | p_oodc -4.7%, p_tan -1.9% |

### What's Exhausted (DO NOT REVISIT)

- **Architecture replacements**: GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR — all 30-60% worse
- **Attention mechanism modifications**: SE channel attention, condition tokens, per-head K/V — all hurt p_tan
- **Auxiliary regression heads**: AoA head (2 weights tried), pressure gradient aux — all hurt p_tan
- **Regularization**: Stochastic depth (3 blocks too shallow), augmentation annealing
- **Distillation**: EMA teacher (50% overhead), snapshot cyclic, head-level diversity
- **Cosine T_max sweep**: T_max=140, 150, 160. T_max=150 is optimal
- **Slice routing perturbations**: Logit noise, RBF kernel, decoupled tandem
- **Chord-position features**: Chord fraction, surface arc-length PE — redundant with TE coord frame
- **Asymmetric losses**: Quantile/pinball, asymmetric surface — diverge or redundant
- **Input features exhausted**: DID, wake centerline SDF, circulation, Q-criterion, shortest vector, chord fraction
- **Inter-foil coupling**: Attention, GALE, cross-DSDF, GNN, FNO — 5 failures
- **MoE FFN routing**: Hard dispatch starves experts (small dataset)
- **GQA K/V groups**: n_head=3, GQA-2 impossible (3%2≠0)
- **GMSE gradient loss**: Too complex for unstructured mesh
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine
- **Bernoulli/continuity PDE losses**: Interfere with L1, noisy on unstructured mesh
- **GradNorm adaptive weighting**: Weight collapse (w_vol → 0), 2.5× compute overhead
- **Multi-scale FPN skips**: 3 blocks too shallow, zero-init + Lion chicken-and-egg
- **Mirror symmetry augmentation**: Mesh is asymmetric
- **Synthetic data (NeuralFoil)**: Geometric inconsistency
- **Per-foil whitening**: Structurally hurts p_tan
- **Tandem curriculum ramp**: Trades p_in for OOD metrics, fundamental tradeoff
