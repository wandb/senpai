# SENPAI Research State

- **Date:** 2026-04-07 20:45 UTC
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
| frieren | #2258 | Decoupled Tandem Slice Projection: separate routing matrix for tandem samples | WIP |

**Idle students:** None. All 8 GPUs occupied.

## PRs Ready for Review

None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues since last check. Prior directives still in effect:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5+)
- Issue #1834: Never use raw data files besides assigned training split

## Just Completed Reviews (this session)

### PR #2249 — Online Hard Node Mining gamma=1.0 (frieren) — CLOSED
- All 4 metrics worse: p_oodc +7.3%, p_re +7.4%, p_in +1.1%, p_tan +1.6%
- Root cause: baseline already has asymmetric hard-node mining (1.5× at epoch ≥30); stacking OHNM multiplicatively creates 3-5× concentration on small node subsets, hurting OOD generalization
- **Key insight:** Never stack a second hard-node weighting on top of the existing mining. Any future OHNM must replace (not add to) the existing mechanism.

## Current Research Focus and Themes

### Round 19 Findings (completed 2026-04-07)

**All training hyperparameters confirmed well-tuned.** LR, weight decay, EMA decay, aug sigma, spectral norm — all baseline values optimal.

**Most informative result:** Spectral norm SRF showed p_in -2.5% but p_tan +1.8%. This proved OOD failure is in the BACKBONE representation, not the output heads.

### Round 21 Strategy (current, 2026-04-07)

Eight experiments targeting diverse levers:

1. **Schedule (T_max=140)** — Match cosine annealing to actual training length (thorfinn #2251)
2. **Output capacity (wider SRF 384)** — Double surface refinement head width (fern #2252)
3. **Loss targeting (aft-foil upweight 1.5x)** — Direct p_tan gradient budget shift (nezuko #2253)
4. **Backbone regularization (hidden noise σ=0.01)** — Target backbone OOD instability (edward #2254)
5. **Training strategy (aug annealing at epoch 120)** — Clean fine-tuning phase (askeladd #2255)
6. **Throughput (val_every=3)** — More training epochs within wall-clock budget (alphonse #2256)
7. **Sample-level reweighting (focal gamma=0.5)** — Upweight hard samples dynamically (tanjiro #2257)
8. **Decoupled tandem slice routing** — Separate `in_project_slice_tandem` for tandem samples, never tested on Phase 6 baseline (frieren #2258)

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
