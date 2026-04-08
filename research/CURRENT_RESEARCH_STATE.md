# SENPAI Research State

- **Date:** 2026-04-08 07:30 UTC
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

**Latest merge:** PR #2251 (thorfinn) — Cosine T_max=150. p_in -0.7%, p_oodc -1.1%, p_tan -0.8%. p_re regressed +1.0% (structural tension: lower final LR hurts Re generalization).

⚠️ **p_re = 6.364 is a regression vs PR #2213 baseline (6.300).** Future experiments should target p_re recovery.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Single-model now beats ensemble on p_in (11.891 vs 12.1) and p_tan (28.118 vs 29.1). Ensemble still leads on p_oodc and p_re.

## Student Status (2026-04-08 07:30 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2267 | Pressure Gradient Aux Head: dp/dx, dp/dy auxiliary supervision | WIP |
| fern | #2266 | ZCA Spectral Whitening of Input Features: decorrelate 24-dim feature covariance | WIP |
| askeladd | #2265 | Per-head K/V projection in Physics_Attention_Irregular_Mesh | WIP |
| frieren | #2264 | Asymmetric Surface Loss: 1.5x suction-side weighting | WIP |
| tanjiro | #2262 | Foil Role Embedding (v2): fix boundary_id + T_max=150 | WIP (sent back) |
| edward | #2261 | Per-Foil Target Whitening (v2): fore-foil-only + T_max=150 | WIP (sent back) |
| nezuko | #2260 | Flow-Regime Conditioned SRF via FiLM: AoA/Umag modulation | WIP |
| askeladd | #2255 | Augmentation Annealing: disable aug after epoch 120 | WIP |
| alphonse | #2268 | MoE FFN Last Block: tandem-specialized FFN expert in final TransolverBlock | WIP |

## PRs Ready for Review

None.

## Latest Reviews (2026-04-08 07:30)

### PR #2263 (alphonse, attn-logit-noise σ=0.05) — CLOSED ❌
- 2-seed avg vs current baseline: p_in=+3.9%, p_oodc=-0.1%, p_tan=+1.0%, p_re=+1.3%
- All meaningful metrics hurt. Slice routing is not the primary tandem failure driver.

### PR #2262 (tanjiro, foil-role-embed) — REQUEST CHANGES
- p_oodc improved consistently both seeds. Seed 42 p_in=11.76 (below baseline 11.891). Promising.
- Issues: wrong foil identity method (saf_norm vs boundary_id), wrong T_max (160 vs 150).

### PR #2261 (edward, per-foil-whiten) — REQUEST CHANGES
- p_in=-1.6% ✅, p_oodc=-2.1% ✅, p_re=-1.0% ✅ vs current baseline. But p_tan=+2.8% ❌
- Fix: fore-foil-only whitening (skip aft-foil to preserve tandem signal) + T_max=150.

### PR #2251 (thorfinn, T_max=150) — MERGED ✅ (prior session)
- p_in -0.7%, p_oodc -1.1%, p_tan -0.8%. New current baseline.

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 — ACTIVE, directive in effect
**"Think bigger — radical new full model changes and data aug and data generation, not incremental tweaks."**

Current round targets architectural, loss-level, and physics-grounded changes — not schedule tweaks.

## Current Research Focus and Themes

### Round 26 Active Experiments (8/9 GPUs occupied, alphonse being assigned)

| Student | PR | Direction | Target |
|---------|-----|-----------|--------|
| thorfinn | #2267 | Physics: Pressure gradient auxiliary head (dp/dx, dp/dy) | p_tan, p_in |
| fern | #2266 | Input rep: ZCA spectral whitening — decorrelate 24-dim features | p_oodc, generalization |
| askeladd | #2265 | Architecture: Per-head K/V projection (remove shared-mean bottleneck) | p_tan |
| frieren | #2264 | Loss: Asymmetric 1.5x suction-side weighting | p_tan |
| tanjiro | #2262 | Representation: Foil role embedding (v2, boundary_id fix) | p_oodc, p_in |
| edward | #2261 | Normalization: Fore-foil-only target whitening (v2) | p_in, p_oodc |
| nezuko | #2260 | Conditioning: FiLM SRF on flow regime (AoA/Umag) | p_tan, p_oodc |
| askeladd | #2255 | Training: Aug annealing (disable after epoch 120) | p_in |
| alphonse | #2268 | MoE FFN Last Block: tandem-specialized FFN in last TransolverBlock | WIP |

### Key Insights from Round 26 Reviews

1. **Slice routing is NOT the tandem failure driver** (attn-logit-noise confirmed negative). Input feature limitations are more likely the root cause.
2. **Per-foil whitening works for fore-foil** — p_in (-1.6%) and p_oodc (-2.1%) improvements are real. Aft-foil must NOT be normalized (kills tandem signal via high-magnitude error de-emphasis).
3. **Foil identity embedding is promising** — consistent p_oodc improvement across seeds, seed 42 beats baseline on p_in. V2 with correct boundary_id and T_max=150 may merge.

### What Works (confirmed and merged)

| Direction | PR | Impact |
|-----------|-----|--------|
| Cosine T_max=150 | #2251 | p_in -0.7%, p_oodc -1.1%, p_tan -0.8% |
| TE coordinate frame | #2207 | -5.4% p_in |
| Wake deficit feature | #2213 | -4.1% p_in, -1.7% p_re |
| Pressure-first deep | #2155 | -4.8% p_in |
| DCT frequency loss | #2184 | -1.5% p_re, -0.3% p_tan |
| Gap-stagger spatial bias | #2130 | -3.0% p_tan |
| PCGrad 3-way | — | OOD separation |
| Residual prediction | #1927 | p_oodc -4.7%, p_tan -1.9% |

### What's Exhausted (DO NOT REVISIT)

- **Architecture replacements**: GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR — all 30-60% worse
- **Cosine T_max sweep**: T_max=140, 150, 160 tried. T_max=150 is optimal. Schedule exhausted.
- **Slice routing perturbations**: Logit noise (σ=0.05), decoupled tandem routing, RBF kernel — all null/negative
- **Sample-level reweighting**: Focal loss γ=0.5, OHNM — over-correction on top of PCGrad
- **Throughput hacks**: Val-every-3 — LR schedule is binding constraint
- **Backbone hidden noise**: Too blunt, destabilizes training
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine

## Potential Next Research Directions (Round 27)

See `/research/RESEARCH_IDEAS_2026-04-08_04:00.md` for full list. Key picks:

Current queue (from researcher-agent output):

| Priority | Slug | Target | Key bet |
|----------|------|--------|---------|
| 1 | `moe-ffn-last-block` | p_tan | **ASSIGNED to alphonse (#2268)** |
| 2 | `circulation-input-feature` | p_tan, p_in | Fore-foil circulation Γ = ∮(v·dl) as per-foil scalar input — same lineage as wake deficit (#2213) |
| 3 | `tandem-slice-temp` | p_tan | Lower temperature for tandem samples — sharper routing for wake region specialization |
| 4 | `global-nystrom-wake` | p_tan | 16-landmark Nyström attention between fore/aft surface nodes for long-range wake coupling |
| 5 | `hypernetwork-re-scaling` | p_re, p_oodc | Continuous Re/gap conditioning via small hypernet — addresses structural p_re regression |

⚠️ **Removed from queue:** `no-penetration-bc-loss` — already tried as PR #2187, failed.
