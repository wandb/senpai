# SENPAI Research State

- **Date:** 2026-04-07 19:45 UTC
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

## Student Status (2026-04-07 19:45 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2251 | Cosine T_max=140: complete LR annealing before training ends | WIP |
| fern | #2252 | Wider SRF heads: 384 hidden dim for more output capacity | WIP |
| nezuko | #2253 | Aft-foil surface loss upweighting: 1.5x on aft-foil nodes | WIP |
| edward | #2254 | Backbone hidden noise: Gaussian noise in TransolverBlock for OOD robustness | WIP |
| askeladd | #2255 | Augmentation annealing: disable aug after epoch 120 for clean fine-tuning | WIP |
| alphonse | #2250 | Blended L1+L2 Surface Loss: MSE auxiliary (alpha=0.1) | WIP |
| frieren | #2249 | Online Hard Node Mining: error-weighted surface loss | WIP |
| tanjiro | #2218 | LE Coordinate Frame v3: single chordwise ratio | WIP |

**Idle students:** None. All 8 GPUs occupied.

## PRs Ready for Review

None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues. Prior directives still in effect:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5+)
- Issue #1834: Never use raw data files besides assigned training split

## Current Research Focus and Themes

### Round 19 Findings (completed 2026-04-07 19:30)

**All training hyperparameters confirmed well-tuned.** Five systematic tests:
- LR 3e-4: all metrics worse (+2-9%), Lion + 2e-4 is optimal
- Weight decay 5e-4 (10x): p_tan improved but p_in regressed, high variance
- EMA 0.9995: dilutes converged weights, 3/4 metrics worse
- Aug sigma 2x: past sweet spot, 3/4 worse
- Spectral norm SRF: **KEY INSIGHT** — OOD failure is in backbone, not output heads (p_in -2.5% but p_tan +1.8%, p_oodc +0.7%)

**Conclusion:** Hyperparameter tuning is exhausted. The baseline is a local optimum on all standard training knobs.

### Round 20 Strategy (current, 2026-04-07 19:45)

Strategy shift per Plateau Protocol. Five diverse experiments targeting different aspects:

1. **Schedule optimization (T_max=140)**: The one schedule parameter never tested — match cosine annealing to actual training length.
2. **Output capacity (wider SRF 384)**: Double the surface refinement head width. SRF is the final bottleneck for surface metrics.
3. **Loss targeting (aft-foil upweight 1.5x)**: Direct p_tan improvement by shifting gradient budget to aft-foil nodes.
4. **Backbone regularization (hidden noise σ=0.01)**: Directly targets the identified OOD failure mechanism — backbone feature instability. Novel approach not tried before.
5. **Training strategy (aug annealing at epoch 120)**: Two-phase training — augmented exploration then clean fine-tuning.

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
- **Feature engineering**: 7 consecutive failures after TE coord + wake deficit. DSDF+TE+wake fully capture useful geometric info.
- **Training hyperparameters**: LR, WD, EMA decay, aug sigma all confirmed optimal (Round 19)
- **Output head regularization**: Spectral norm/dropout on SRF — wrong level of abstraction for OOD
- **Optimizer variants**: SAM, Lookahead, SWA — all worse than baseline Lion+EMA+cosine

## Potential Next Research Directions

After Round 20 completes:

1. **Loss reformulation at a deeper level**: Predict pressure GRADIENTS (dp/ds), Cp normalization, or auxiliary physics-consistent losses that don't conflict with the existing loss.
2. **Backbone OOD regularization**: If backbone hidden noise (Round 20, edward) shows promise, explore it more aggressively — different σ schedules, layer-selective noise, feature-space contrastive learning.
3. **Multi-resolution prediction**: Split prediction into low-frequency (mean field) + high-frequency (local corrections). Different heads for different frequency bands.
4. **Sample-level curriculum/re-weighting**: Train on easy samples first or upweight hard samples. Different from node-level OHNM.
5. **Condition-aware training**: Separate optimization for different flow regimes (single vs tandem, low vs high Re).
6. **Researcher-agent fresh ideas**: Agent running in background, results will inform Round 21+.
