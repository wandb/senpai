# SENPAI Research State

- **Date:** 2026-04-07 23:15 UTC
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

## Student Status (2026-04-07 23:15 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2251 | Cosine T_max=150 (sent back from T_max=140 — p_in/p_oodc beat, try sweet spot) | WIP |
| fern | #2259 | Two-Pass Iterative SRF: sequential boosting of surface corrections | WIP |
| nezuko | #2260 | Flow-Regime Conditioned SRF via FiLM: AoA/Umag modulation on surface head | WIP |
| edward | #2261 | Per-Foil Target Whitening: standardize pressure targets per foil | WIP |
| askeladd | #2255 | Augmentation annealing: disable aug after epoch 120 for clean fine-tuning | WIP |
| alphonse | #2256 | Val-every-3 throughput: validate every 3 epochs for ~20 more training epochs | WIP |
| tanjiro | #2257 | Focal sample reweighting: loss^0.5 upweight on hard samples | WIP |
| frieren | #2258 | Decoupled Tandem Slice Projection: separate routing matrix for tandem samples | WIP |

**Idle students:** None. All 8 GPUs occupied.

## PRs Ready for Review

None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues since last check. Prior directives still in effect:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5+)
- Issue #1834: Never use raw data files besides assigned training split

## Just Completed Reviews (this session)

### PR #2254 — Backbone Hidden Noise (σ=0.01) (edward) — CLOSED
- All 4 metrics worse: p_in +1.1%, p_oodc +2.4%, p_tan +0.4%, p_re +3.8%
- Additive noise across all 3 TransolverBlocks compounds through residual connections, creating instability.
- **Key insight:** Backbone OOD fragility needs targeted intervention (attention-level) not global hidden-state noise.

### PR #2253 — Aft-Foil Surface Loss Upweighting 1.5x (nezuko) — CLOSED
- All 4 metrics worse: p_in +2.2%, p_oodc +0.05%, p_tan +1.1%, p_re +0.4%
- Compounds with existing tandem_ramp + adaptive_boost + PCGrad — gradient budget already saturated.
- **Key insight:** Loss/gradient-level modifications are exhausted. Future improvements need new information or structural changes.

### PR #2251 — Cosine T_max=140 (thorfinn) — SENT BACK (try T_max=150)
- Mixed results: p_in **-1.7%** ✓, p_oodc **-1.2%** ✓, but p_tan +0.4% ✗, p_re +1.1% ✗
- Strong signal that schedule optimization has headroom. T_max=140 too aggressive — try T_max=150.
- **Key insight:** Baseline T_max=160 wastes ~10 epochs of uncompleted annealing. Optimal T_max likely matches actual training length.

## Current Research Focus and Themes

### Round 22 Strategy (current, 2026-04-07)

Eight experiments targeting diverse levers across information flow, prediction targets, training strategy, and regularization:

1. **Schedule optimization (T_max=150)** — Follow-up on clear signal from T_max=140 (thorfinn #2251)
2. **Flow-regime SRF conditioning (FiLM)** — Explicit AoA/Umag signal to SRF head for regime-dependent corrections (nezuko #2260)
3. **Prediction target change (per-foil whitening)** — Per-foil pressure normalization to improve tandem transfer (edward #2261)
4. **Training strategy (aug annealing at epoch 120)** — Clean fine-tuning phase (askeladd #2255)
5. **Throughput (val_every=3)** — More training epochs within wall-clock budget (alphonse #2256)
6. **Sample-level reweighting (focal gamma=0.5)** — Upweight hard samples dynamically (tanjiro #2257)
7. **Decoupled tandem slice routing** — Separate projection for tandem vs single-foil (frieren #2258)
8. **Multi-pass SRF (two-pass)** — Gradient-boosting-style second SRF correction pass (fern #2259)

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

### What's Exhausted (DO NOT REVISIT)

- **Architecture replacements**: GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR, NOBLE/CosNet
- **Feature engineering**: TE coord + wake deficit are the only features that work. LE features, wall distance, all others dead.
- **Training hyperparameters**: LR, WD, EMA decay, aug sigma all confirmed optimal (Round 19)
- **Output head regularization**: Spectral norm/dropout on SRF — wrong level of abstraction
- **Wider SRF heads**: 384-dim (PR #2252) — overfitting. 192-dim is optimal capacity.
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than baseline Lion+EMA+cosine
- **Additive loss penalties**: Huber, L1+L2, OHNM — conflicts with PCGrad/tandem_ramp
- **Node-level loss weighting**: Aft-foil 1.5x upweight (PR #2253) — redundant with existing gradient mechanisms
- **Backbone hidden noise**: σ=0.01 additive (PR #2254) — too blunt, compounds through residual connections

## Potential Next Research Directions

After Round 22 completes:

1. **Ensemble knowledge distillation**: 16-seed ensemble beats single model by 13.4% on p_oodc (6.6 vs 7.6) and 7.9% on p_re (5.8 vs 6.3). Use ensemble predictions as soft targets. Requires precomputing ensemble predictions.
2. **Late stochastic depth (epoch 80+)**: Layer dropping in second half of training only. Prior experiments applied from epoch 0 and failed. Late application forces redundant representations for OOD robustness.
3. **Attention-targeted perturbation**: Noise on attention logits (not hidden states). Mechanistically motivated follow-up from backbone noise experiments — targets the actual source of OOD instability.
4. **Pressure gradient loss (dp/ds)**: Auxiliary loss on surface pressure gradients along the surface. Distinct from DCT freq loss — operates in physical space not spectral.
5. **Wake-relative attention bias**: Use wake deficit features to construct a physics-informed routing prior in slice attention.
6. **Hard sample replay buffer**: Change sampling FREQUENCY (not gradient magnitude) for hard cases. Orthogonal to focal reweighting.
