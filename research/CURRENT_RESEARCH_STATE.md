# SENPAI Research State

- **Date:** 2026-04-07 23:55 UTC
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

## Student Status (2026-04-07 23:55 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2251 | Cosine T_max=150 — sent back (T_max=140 was promising on p_in/-1.7%, p_oodc/-1.2%, trying sweet spot) | WIP |
| fern | #2259 | Two-Pass Iterative SRF: sequential boosting of surface corrections | WIP |
| nezuko | #2260 | Flow-Regime Conditioned SRF via FiLM: AoA/Umag modulation on surface head | WIP |
| edward | #2261 | Per-Foil Target Whitening: standardize pressure targets per foil | WIP |
| askeladd | #2255 | Augmentation annealing: disable aug after epoch 120 for clean fine-tuning | WIP |
| tanjiro | #2262 | Foil Role Embedding: explicit fore/aft identity for tandem surface nodes | WIP |
| alphonse | #2263 | Attention Logit Noise σ=0.05: targeted slice routing regularization | WIP |
| frieren | #2258 | Decoupled Tandem Slice Projection: separate routing matrix for tandem samples | WIP |

**Idle students:** None. All 8 GPUs occupied.

## PRs Ready for Review

None currently.

## Most Recent Research Direction from Human Researcher Team

No new issues since last check. Prior directives still in effect:
- Issue #1860: "Think bigger — radical model changes, not just incremental tweaks" (addressed Phase 5+)
- Issue #1834: Never use raw data files besides assigned training split

## Round 23 Reviews Completed (this session, 2026-04-07 ~23:45)

### PR #2257 — Focal Sample Reweighting γ=0.5 (tanjiro) — CLOSED
- All 4 metrics worse: p_in +7.1%, p_oodc +5.9%, p_tan +0.7%, p_re +3.2%
- **Key finding:** Existing hard-sample stack (PCGrad 3-way, tandem_ramp, hard-node mining) is already at saturation. Adding focal reweighting over-corrects.
- **Do not revisit:** Per-sample loss weighting while existing mechanisms remain.

### PR #2256 — Val-Every-3 Throughput (alphonse) — CLOSED
- All 4 metrics at/above baseline (159 epochs vs 148). Extra epochs in lowest-LR phase add no signal.
- **Key finding:** LR schedule is the binding constraint, not validation overhead. Checkpoint granularity (val_every=3) costs more than the epoch gain.
- **Follow-up direction:** T_max optimization (thorfinn #2251).

### PR #2251 — Cosine T_max=140 (thorfinn) — SENT BACK for T_max=150
- Mixed: p_in -1.7% ✓, p_oodc -1.2% ✓, p_tan +0.4% ✗, p_re +1.1% ✗
- Student resubmitted T_max=140 data instead of running T_max=150. Sent back with clear instructions.
- **Direction confirmed:** Completing cosine annealing before timeout has headroom. T_max=150 is the sweet spot.

## Current Research Focus and Themes

### Round 23 Strategy (current, 2026-04-07)

Eight experiments targeting diverse angles — information representation, training strategy, schedule optimization, attention regularization:

1. **Schedule optimization (T_max=150)** — Refining confirmed p_in/p_oodc improvement signal (thorfinn #2251)
2. **Flow-regime SRF conditioning (FiLM)** — Explicit AoA/Umag conditioning on SRF head (nezuko #2260)
3. **Per-foil target whitening** — Prediction target normalization per foil for tandem transfer (edward #2261)
4. **Augmentation annealing at epoch 120** — Clean fine-tuning phase (askeladd #2255)
5. **Foil role embedding** — Explicit fore/aft foil identity for backbone (tanjiro #2262) ← NEW
6. **Attention logit noise σ=0.05** — Slice routing regularization, follow-up from PR #2254 autopsy (alphonse #2263) ← NEW
7. **Decoupled tandem slice routing** — Separate projection matrix for tandem vs single-foil (frieren #2258)
8. **Two-pass SRF** — Gradient-boosting-style second SRF correction pass (fern #2259)

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

### What's Exhausted (DO NOT REVISIT)

- **Architecture replacements**: GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR
- **Feature engineering**: TE coord + wake deficit are the primary winners. LE features, wall distance, all others dead.
- **Training hyperparameters**: LR, WD, EMA decay, aug sigma all confirmed optimal (Round 19)
- **Output head regularization**: Spectral norm/dropout on SRF — wrong level of abstraction
- **Wider SRF heads**: 384-dim (PR #2252) — overfitting. 192-dim is optimal.
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine
- **Additive loss penalties**: Huber, L1+L2, OHNM — conflicts with PCGrad/tandem_ramp
- **Node-level loss weighting**: Aft-foil 1.5x upweight (PR #2253) — redundant with existing gradient mechanisms
- **Sample-level loss weighting**: Focal γ=0.5 (PR #2257) — over-correction, existing hard-sample stack saturated
- **Backbone hidden noise**: σ=0.01 additive (PR #2254) — too blunt, compounds through residuals
- **Throughput hacks**: Val-every-3 (PR #2256) — checkpoint granularity loss > epoch gain

## Potential Next Research Directions

When current round completes:

1. **SRF Arc-Length PE** — Sinusoidal position encoding in SRF head (knows where on airfoil surface each node is). Targets p_tan. Orthogonal to in-flight SRF experiments (#2259, #2260).
2. **Dual EMA Checkpoint** — Two shadow models (decay=0.999 fast, 0.996 slow), use fast for p_in/p_re, slow for p_tan/p_oodc. Zero architectural risk.
3. **Hard Sample Replay Buffer** — PrioritizedExperienceReplay-style: change sampling FREQUENCY (not gradient magnitude) for hard tandem samples. Orthogonal to focal reweighting (#2257, failed).
4. **Late Stochastic Depth (epoch 80+)** — Drop entire TransolverBlocks only after epoch 80, forcing redundant representations. Prior failure was from epoch 0 application.
5. **Global Nyström Pathway** — Add 16-landmark global attention in parallel to local slice attention. Target: long-range tandem wake-aft-foil coupling.
6. **RANS Consistency Loss** — Auxiliary mass conservation loss (∇·u ≈ 0) on volume predictions. Physics regularizer for OOD generalization.
