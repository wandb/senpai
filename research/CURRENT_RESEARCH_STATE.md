# SENPAI Research State

- **Date:** 2026-04-08 05:15 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training & Architecture Improvements

## Current Baseline

### Single-Model Baseline (PR #2251, +Cosine T_max=150, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.891** | < 11.89 |
| **p_oodc** | **7.561** | < 7.56 |
| **p_tan** | **28.118** | < 28.12 |
| p_re | 6.364 | < 6.36 |

**Latest merge:** PR #2251 (thorfinn) — Cosine T_max=150. Schedule mismatch fix: training ends at ~149 epochs, setting T_max=150 ensures annealing completes right at cutoff. p_in -0.7%, p_oodc -1.1%, p_tan -0.8%. p_re regressed +1.0% (structural tension: lower final LR hurts Re generalization).

⚠️ **p_re = 6.364 is a regression vs prior baseline (6.300).** Future experiments should target p_re recovery.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Single-model now beats ensemble on p_in (11.891 vs 12.1) and p_tan (28.118 vs 29.1). Ensemble still leads on p_oodc and p_re.

## Student Status (2026-04-08 05:15 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2267 | Pressure Gradient Aux Head: dp/dx, dp/dy auxiliary supervision on volume nodes | WIP |
| fern | #2266 | ZCA Spectral Whitening of Input Features: decorrelate 24-dim feature covariance | WIP |
| nezuko | #2260 | Flow-Regime Conditioned SRF via FiLM: AoA/Umag modulation on surface head | WIP |
| edward | #2261 | Per-Foil Target Whitening: standardize pressure targets per foil | WIP |
| askeladd | #2255 | Aug annealing v2 — trial A (aug_stop_epoch=140) + trial B (selective AoA-only) | WIP |
| askeladd | #2265 | Per-head K/V projection in Physics_Attention_Irregular_Mesh | WIP |
| tanjiro | #2262 | Foil Role Embedding: explicit fore/aft identity for tandem surface nodes | WIP |
| alphonse | #2263 | Attention Logit Noise σ=0.05: targeted slice routing regularization | WIP |
| frieren | #2264 | Asymmetric Surface Loss: 1.5x weight on suction-side nodes (physics-motivated) | WIP |

**Idle students:** None. All 9 GPUs occupied.

## PRs Ready for Review

None currently.

## Latest Reviews (2026-04-08 05:15)

### PR #2251 — Cosine T_max=150 (thorfinn) — MERGED ✅
- p_in -0.7%, p_oodc -1.1%, p_tan -0.8% ✅ / p_re +1.0% ❌
- New baseline: p_in=11.891, p_oodc=7.561, p_tan=28.118, p_re=6.364
- Cosine T_max sweep exhausted — structural p_re tension confirmed

### PR #2259 — Two-Pass Iterative SRF (fern) — CLOSED (prior session)
- All 4 metrics worse. SRF architecture fully exhausted.

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 (Morgan McGuire / morganmcg1) — ACTIVE, directive in effect
**"Think bigger — radical new full model changes and data aug and data generation, not incremental tweaks."**

Current round (#2260-#2267) is the direct response: per-head KV, asymmetric loss, FiLM conditioning, foil role embeddings, ZCA whitening, pressure gradient aux head. These are architectural, loss-level, and representation-level changes — not schedule tweaks.

### Issue #1834 — Never use raw data files besides those assigned. ✓ Always complied.

## Current Research Focus and Themes

### Round 25 Active Experiments

Nine experiments across physics-grounded and mechanistic angles:

1. **Pressure gradient aux head** — dp/dx, dp/dy auxiliary supervision (thorfinn #2267) ← physics-grounded auxiliary
2. **ZCA spectral whitening** — Decorrelate 24-dim input features (fern #2266) ← input representation
3. **Per-head K/V slice** — Remove shared-mean bottleneck in slice attention (askeladd #2265) ← architecture
4. **Asymmetric surface loss** — 1.5x suction-side weighting (frieren #2264) ← loss reformulation
5. **Attention logit noise σ=0.05** — Slice routing regularization (alphonse #2263) ← regularization
6. **Foil role embedding** — Explicit fore/aft identity for backbone (tanjiro #2262) ← representation
7. **Per-foil target whitening** — Tandem transfer normalization (edward #2261) ← normalization
8. **Flow-regime SRF conditioning (FiLM)** — Explicit AoA/Umag on SRF (nezuko #2260) ← conditioning
9. **Aug annealing v2** — Selective/late cutoff (askeladd #2255) ← training curriculum

### What Works (confirmed and merged)

| Direction | PR | Impact |
|-----------|-----|--------|
| Cosine T_max=150 | #2251 | p_in -0.7%, p_oodc -1.1%, p_tan -0.8% |
| TE coordinate frame | #2207 | -5.4% p_in |
| Wake deficit feature | #2213 | -4.1% p_in, -1.7% p_re |
| Pressure-first deep | #2155 | -4.8% p_in |
| DCT frequency loss | #2184 | -1.5% p_re, -0.3% p_tan |
| Asinh pressure | #2135 | Major p_tan improvement |
| Residual prediction | #2119 | Meaningful gains across board |
| Surface refinement | Multiple | Consistent improvement |
| Gap-stagger spatial bias | Recent | Tandem transfer improvement |
| PCGrad 3-way | #2184 | OOD separation |

### What's Exhausted (DO NOT REVISIT)

- **Architecture replacements**: GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR
- **Feature engineering**: TE coord + wake deficit are the primary winners. LE features, wall distance, all others dead.
- **Training hyperparameters**: LR, WD, EMA decay, aug sigma all confirmed optimal (Round 19)
- **Output head regularization**: Spectral norm/dropout on SRF — wrong level of abstraction
- **Wider SRF heads**: 384-dim overfitting. 192-dim is optimal.
- **Two-pass SRF**: Sequential boosting (#2259) — optimization interference. SRF architecture fully exhausted.
- **Cosine T_max sweep**: T_max=140, 150, 160 all tried. T_max=150 merged as winner. Schedule exhausted.
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine
- **Additive loss penalties**: Huber, L1+L2, OHNM — conflicts with PCGrad/tandem_ramp
- **Node-level loss weighting**: Aft-foil 1.5x upweight (PR #2253), OHNM (PR #2249) — redundant with asym loss
- **Sample-level loss weighting**: Focal γ=0.5 (PR #2257) — over-correction
- **Backbone hidden noise**: σ=0.01 (PR #2254) — too blunt
- **Throughput hacks**: Val-every-3 (PR #2256) — LR schedule is binding constraint
- **Decoupled tandem routing**: Orthogonal init (PR #2258) — undertrained on minority class
- **Bernoulli consistency loss**: Point-wise algebraic — different from gradient auxiliary

## Potential Next Research Directions (Round 26)

Unassigned from bold queue (from `/research/RESEARCH_IDEAS_2026-04-08_BOLD.md`):

| Priority | Slug | Target | Risk | Key bet |
|----------|------|--------|------|---------|
| 3 | `tandem-geom-interpolation` | p_tan | Low-Med | Physics-preserving gap/stagger interpolation |
| 4 | `hypernetwork-physics-scaling` | p_oodc, p_re, p_tan | Medium | Continuous Re/gap conditioning via hypernet |
| 5 | `gnn-boundary-layer` | p_tan, p_in | Medium | Local GraphSAGE on surface+near-surface nodes |
| 6 | `geometry-consistency-distill` | p_oodc, p_tan | Medium | Volume mesh jitter + EMA self-distillation |
| 7 | `cnf-surface-pressure` | p_tan, p_oodc | Medium | Flow matching surface head — generative alternative |
| 8 | `fno-inter-foil-coupling` | p_tan | Med-High | 1D spectral convolution in gap region |

From earlier queue (still valid):
- **Slice Temperature Annealing** — Start warm (τ=0.5), anneal to 0.15
- **MoE FFN — Last Block Only** — 2 experts (single/tandem) with hard `is_tandem` gate
- **No-Penetration BC Loss** — `dot(pred_vel, surface_normal) ≈ 0`
- **Global Nyström Pathway** — 16-landmark global attention for long-range wake coupling

⚠️ **p_re priority**: After the T_max=150 regression, p_re recovery should be prioritized. `hypernetwork-physics-scaling` (continuous Re conditioning) is the strongest candidate for p_re improvement.
