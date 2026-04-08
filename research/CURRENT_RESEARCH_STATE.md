# SENPAI Research State

- **Date:** 2026-04-08 01:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2213, +Wake Deficit Feature, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.979** | < 11.98 |
| **p_oodc** | **7.643** | < 7.65 |
| **p_tan** | **28.341** | < 28.34 |
| **p_re** | **6.300** | < 6.30 |

**Latest merge:** PR #2213 (frieren) — Wake Deficit Feature.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Single-model p_tan (28.341) and p_in (11.979) both BEAT the ensemble.

## Student Status (2026-04-08 01:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2251 | Cosine T_max=150 — sent back (T_max=140 promising on p_in/-1.7%, p_oodc/-1.2%) | WIP |
| fern | #2259 | Two-Pass Iterative SRF: sequential boosting of surface corrections | WIP |
| nezuko | #2260 | Flow-Regime Conditioned SRF via FiLM: AoA/Umag modulation on surface head | WIP |
| edward | #2261 | Per-Foil Target Whitening: standardize pressure targets per foil | WIP |
| askeladd | #2255 | Aug annealing — sent back: try aug_stop_epoch=140 AND selective (AoA-only) annealing | WIP |
| tanjiro | #2262 | Foil Role Embedding: explicit fore/aft identity for tandem surface nodes | WIP |
| alphonse | #2263 | Attention Logit Noise σ=0.05: targeted slice routing regularization | WIP |
| frieren | #2264 | Asymmetric Surface Loss: 1.5x weight on suction-side nodes (physics-motivated) | WIP |

**Idle students:** None. All 8 GPUs occupied.

## PRs Ready for Review

None currently.

## Round 24 Reviews Completed (this session, 2026-04-08 ~00:45)

### PR #2258 — Decoupled Tandem Slice Projection (frieren) — CLOSED
- All 4 metrics worse: p_in +11.7%, p_oodc +12.7%, p_tan +1.5%, p_re +10.3%
- **Root cause:** Orthogonal-initialized tandem routing head undertrained (only ~30% tandem samples); existing domain_layernorm/domain_velhead/pcgrad_3way already provide adequate tandem/single separation.
- **Do not revisit** unless warm-starting from shared routing weights.

### PR #2255 — Augmentation Annealing (askeladd) — SENT BACK for follow-up
- p_in improved -1.0% ✓ but p_oodc +2.8% ❌, p_tan +0.7% ❌, p_re +1.7% ❌
- **Direction is real** (clean fine-tuning hypothesis holds for ID), but hard cutoff at epoch 120 hurts OOD too much.
- **Follow-up:** Two trials — (A) aug_stop_epoch=140 (shorter clean phase), (B) selective annealing (only AoA disabled at 120, keep gap/stagger+DSDF).

## Most Recent Research Direction from Human Researcher Team

Prior directives still in effect:
- **Issue #1860:** "Think bigger — radical model changes, not just incremental tweaks." Current round addresses this with architectural changes (per-head KV, MoE FFN forthcoming), attention-level modifications, and physics-grounded loss reformulation.
- **Issue #1834:** Never use raw data files besides those assigned in original data files. ✓ Always complied.

## Current Research Focus and Themes

### Round 24 Active Experiments

Eight experiments across diverse mechanistic angles:

1. **Asymmetric surface loss** — 1.5x suction-side weighting (frieren #2264) ← NEW, high confidence
2. **Per-head KV slice** — Remove shared-mean bottleneck in slice attention (askeladd #2265) ← NEW
3. **Augmentation annealing v2** — Selective/late cutoff (askeladd #2255, sent back)
4. **Flow-regime SRF conditioning (FiLM)** — Explicit AoA/Umag on SRF (nezuko #2260)
5. **Per-foil target whitening** — Tandem transfer normalization (edward #2261)
6. **Foil role embedding** — Explicit fore/aft identity for backbone (tanjiro #2262)
7. **Attention logit noise σ=0.05** — Slice routing regularization (alphonse #2263)
8. **Cosine T_max=150** — Schedule optimization confirmed win (thorfinn #2251)
9. **Two-pass SRF** — Gradient-boosting correction (fern #2259)

### What Works (confirmed and merged)

| Direction | PR | Impact |
|-----------|-----|--------|
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
- **Wider SRF heads**: 384-dim (PR #2252) — overfitting. 192-dim is optimal.
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine
- **Additive loss penalties**: Huber, L1+L2, OHNM — conflicts with PCGrad/tandem_ramp
- **Node-level loss weighting**: Aft-foil 1.5x upweight (PR #2253), OHNM (PR #2249) — redundant
- **Sample-level loss weighting**: Focal γ=0.5 (PR #2257) — over-correction
- **Backbone hidden noise**: σ=0.01 (PR #2254) — too blunt
- **Throughput hacks**: Val-every-3 (PR #2256) — LR schedule is binding constraint
- **Decoupled tandem routing**: Orthogonal init (PR #2258) — undertrained on minority class

## Potential Next Research Directions

When current round completes:

1. **Slice Temperature Annealing** (Round 23 Idea 4) — Start warm (0.5), anneal to 0.15 over training. Forces harder slice specialization late in training when model has seen enough tandem examples.
2. **MoE FFN — Last Block Only** (Round 23 Idea 5) — 2 FFN experts (single-foil / tandem) with hard is_tandem gate, applied only to block 2. Compute-efficient version of domain-specialized FFN.
3. **SRF Pressure Gradient Feature** (Round 23 Idea 3) — Feed centered pressure prediction `dp_centered = base_pred_surf - mean(base_pred_surf)` as extra SRF input feature. Zero-init safe, directly targets suction peak modeling.
4. **No-Penetration Boundary Condition Loss** (Round 23 Idea 7) — Auxiliary loss enforcing `dot(pred_vel, surface_normal) ≈ 0` using DSDF gradient as surface normal proxy. Physics-grounded, OOD robust.
5. **Hard Sample Replay Buffer** — Prioritized resampling frequency (not gradient magnitude) for hard tandem samples. Orthogonal to failed focal reweighting.
6. **Global Nyström Pathway** — 16-landmark global attention in parallel to local slice attention. Target: long-range tandem wake-aft-foil coupling.
