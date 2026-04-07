# SENPAI Research State

- **Date:** 2026-04-07 20:05 UTC
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

## Student Status (2026-04-07 20:05 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2251 | Cosine T_max=140: complete LR annealing before training ends | WIP |
| fern | #2252 | Wider SRF heads: 384 hidden dim for more output capacity | WIP |
| nezuko | #2253 | Aft-foil surface loss upweighting: 1.5x on aft-foil nodes | WIP |
| edward | #2254 | Backbone hidden noise: Gaussian noise in TransolverBlock for OOD robustness | WIP |
| askeladd | #2255 | Augmentation annealing: disable aug after epoch 120 for clean fine-tuning | WIP |
| alphonse | #2256 | Val-every-3 throughput: validate every 3 epochs for ~20 more training epochs | WIP |
| tanjiro | #2257 | Focal sample reweighting: loss^0.5 upweight on hard samples | WIP |
| frieren | #2249 | Online Hard Node Mining: error-weighted surface loss | WIP |

**Idle students:** None. All 8 GPUs occupied.

## PRs Ready for Review

None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues since last check. Prior directives still in effect:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5+)
- Issue #1834: Never use raw data files besides assigned training split

## Just Completed Reviews (this session)

### PR #2250 — Blended L1+L2 Surface Loss (alphonse) — CLOSED
- p_tan +4.3% worse, p_re +1.6% worse
- L2 gradient conflicts with existing hard-node mining (PCGrad extreme + tandem_ramp)
- **Key insight:** Additive loss penalties that overlap with PCGrad/tandem_ramp fail — double-penalizing

### PR #2218 — LE Coordinate Frame v1/v2/v3 (tanjiro) — CLOSED
- All 3 iterations failed (v1: OOD catastrophe, v2: mixed, v3: all worse)
- **Key insight:** Feature engineering is exhausted. TE coord + wake deficit + Fourier PE capture all useful spatial info

## Current Research Focus and Themes

### Round 19 Findings (completed 2026-04-07)

**All training hyperparameters confirmed well-tuned.** LR, weight decay, EMA decay, aug sigma, spectral norm — all baseline values optimal.

**Most informative result:** Spectral norm SRF showed p_in -2.5% but p_tan +1.8%. This proved OOD failure is in the BACKBONE representation, not the output heads.

### Round 20 Strategy (current, 2026-04-07)

Eight diverse experiments targeting different levels of the training pipeline:

1. **Schedule (T_max=140)** — Match cosine annealing to actual training length
2. **Output capacity (wider SRF 384)** — Double surface refinement head width
3. **Loss targeting (aft-foil upweight 1.5x)** — Direct p_tan gradient budget shift
4. **Backbone regularization (hidden noise σ=0.01)** — Target backbone OOD instability
5. **Training strategy (aug annealing at epoch 120)** — Clean fine-tuning phase
6. **Throughput (val_every=3)** — More training epochs within wall-clock budget
7. **Sample-level reweighting (focal gamma=0.5)** — Upweight hard samples dynamically
8. **Node-level reweighting (OHNM)** — Error-weighted surface node loss

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
- **Optimizer variants**: SAM, Lookahead, SWA — all worse than baseline Lion+EMA+cosine
- **Additive loss penalties**: Huber, L1+L2, any loss that conflicts with PCGrad/tandem_ramp

## Potential Next Research Directions

After Round 20 completes:

1. **Backbone OOD regularization at scale**: If backbone hidden noise (edward, #2254) shows promise, explore aggressive variants — layer-selective noise, scheduled noise, feature-space dropout.
2. **Prediction target transformation**: Predict pressure gradients dp/ds, vorticity, or stream function. The most impactful change has been altering the prediction task itself (pressure-first).
3. **Multi-resolution decomposition**: Separate low-frequency (mean field) and high-frequency (local corrections) prediction paths. Related to but distinct from DCT freq loss.
4. **Knowledge distillation from ensemble**: The 16-seed ensemble beats single-model on p_oodc (6.6 vs 7.6) and p_re (5.8 vs 6.3). Use ensemble predictions as soft targets.
5. **Condition-aware training**: Separate optimization paths for different flow regimes (single vs tandem, low vs high Re). Beyond what PCGrad 3-way currently handles.
6. **Contrastive backbone learning**: Auxiliary contrastive loss encouraging similar backbone representations for geometrically similar configurations.
