# SENPAI Research State

- **Date:** 2026-04-04 ~10:50 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline (PR #2093 — MERGED 2026-04-04)

| Metric | 16-Seed Ensemble (42-49 + 66-73) | Single-model mean (8 seeds) |
|--------|----------------------------------|-----------------------------|
| p_in | **12.1** | 13.03 |
| p_oodc | **6.6** | 7.83 |
| p_tan | **29.1** | 30.29 |
| p_re | **5.8** | 6.45 |

16-seed ensemble beats 8-seed baseline: p_in -0.8%, p_oodc -1.5%. Seeds 100-106 also trained (mean p_in=13.0), available for future 23-seed evaluation.

## Student Status (2026-04-04 ~10:50 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2114 | Gradient Centralization — Zero-Mean Gradient Updates with Lion | WIP |
| edward | #2113 | Smooth L1 (Huber) Loss — Node-Level Gradient Emphasis | WIP |
| askeladd | #2106 | Fourier Feature Position Encoding | WIP |
| frieren | #2107 | Aft-Foil Coordinate Frame Normalization (dual-frame iteration) | WIP — sent back |
| fern | #2104 | Dedicated Aft-Foil SRF Branch (ID=7) | WIP |
| nezuko | #2115 | Gap/Stagger Perturbation Augmentation — Tandem OOD Robustness | WIP — just assigned |
| alphonse | #2111 | TTA via AoA Perturbation (inference-only) | WIP |
| thorfinn | #2112 | Mesh-Density Weighted L1 Loss | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-04 ~10:50 UTC)

| PR | Student | Experiment | Decision | Reason |
|----|---------|-----------|---------|--------|
| #2110 | nezuko | Progressive Surface Focus Schedule (curriculum) | CLOSED | p_in regressed +0.7% in both variants (40ep, 80ep). Dynamic surf_weight from epoch 0 is already well-tuned. p_tan improved -1.7% with 80ep but doesn't compensate p_in regression. |
| #2109 | tanjiro | Contrastive Tandem-Single Regularization | CLOSED | p_tan did NOT improve: mean 30.4 vs baseline 30.4 across 6 runs. Hypothesis falsified: representational entanglement is NOT the p_tan bottleneck. |
| #2107 | frieren | Aft-Foil Coordinate Frame Normalization (in-place) | SENT BACK | p_tan improved on s73 (29.51, -2.6% vs mean), but p_in regressed severely (13.55-13.83). Direction validated but implementation destroys positional info. **Sent back with dual-frame instruction**: add local-frame coords as ADDITIONAL sideband features. |
| #2108 | edward | Asymmetric Fixed Asinh Scales | CLOSED | Negative: both configs worse on all metrics. Closes asinh direction entirely. |
| #2103 | thorfinn | Iterative Weight-Tied Transolver | CLOSED | Monotonic degradation; blocks learn complementary reps, not iterative refinement. |
| #2102 | alphonse | SIREN Activation in SRF Head | CLOSED | Monotonic degradation with omega; GELU well-calibrated for srf corrections. |
| #2101 | tanjiro | OHEM (hard example mining) | CLOSED | Negative: sample-level reweighting is NOT the p_tan bottleneck. |
| #2100 | frieren | Model Scale-Up (5L, 3L/160s, 4L/128s) | CLOSED | p_tan NOT capacity-limited — similar across all scales. |
| #2097 | nezuko | Deep Supervision (aux loss) | CLOSED | p_in -1.7% but p_tan +2.7% — constrains representational flexibility. |

## Current Research Focus

### Primary target: p_tan = 29.1 (our weakest metric, 2.5x worse than p_in)

**Confirmed bottleneck hypotheses (negative results):**
- NOT capacity-limited (scale-up #2100: p_tan ~30-32 across all scales)
- NOT sample-level reweighting (OHEM #2101: no improvement)
- NOT representational entanglement between tandem/single hidden states (contrastive #2109: hypothesis falsified)
- IS a coordinate-frame problem (local-frame #2107: p_tan -2.6% but p_in regressed — dual-frame retry in progress)

**Active experiments attacking p_tan from multiple angles:**
1. **Dual-Frame Coordinate Features** (frieren #2107) — add local-frame coords as ADDITIONAL features alongside global (non-destructive); directly confirmed p_tan component
2. **Dedicated Aft-Foil SRF Branch** (fern #2104) — separate refinement MLP for boundary ID=7, FiLM conditioning on gap/stagger
3. **Gap/Stagger Perturbation Augmentation** (nezuko #2115) — domain randomization on tandem conditioning features (σ={0.02,0.05,0.10}); most direct attack on p_tan OOD axis
4. **Mesh-Density Weighted L1** (thorfinn #2112) — upweight fine-mesh nodes (leading edge, suction peaks)

**Representation and signal quality:**
5. **Fourier Feature Position Encoding** (askeladd #2106) — Random Fourier Features for spatial coordinates, addressing spectral bias
6. **TTA via AoA Perturbation** (alphonse #2111) — inference-only: average predictions at AoA ± δ for variance reduction

**Loss and optimizer reformulation:**
7. **Smooth L1 (Huber) Loss** (edward #2113) — replace L1 with smooth L1; sweep beta={0.5, 1.0, 2.0}
8. **Gradient Centralization** (tanjiro #2114) — remove DC gradient component before Lion sign operation; novel interaction with sign-based updates; may help tandem gradients compete with single-foil DC offset

## Key Research Insights

1. **Architecture changes are absorbed by EMA+Lion** — Phase 5 showed 6 new architectures all failed; the Transolver is a strong local optimum
2. **Target-space transforms cannot be learnable** — Learnable asinh scale (#2096) confirms: joint optimization creates trivial shortcuts
3. **Velocity doesn't need compression** — Asinh velocity (#2098) confirms: velocity channels are well-behaved, pressure asymmetry is the unique bottleneck
4. **Ensembling is near-saturating** — 16→23 seeds gains diminish; single-model improvements are the priority
5. **Deep supervision improves p_in but hurts p_tan** — (#2097) 8-seed validation: p_in -1.7% but p_tan +2.7%. Auxiliary loss constrains intermediate representations
6. **Offline regression KD doesn't work** — (#2090) confirms: the ensemble's single-model gap is too small for soft-target distillation
7. **Stochastic depth requires depth** — DropPath (#2099) is ineffective for 3-block architectures
8. **Per-node adaptive temperature is a known null** — Ada-Temp tried in PRs #1879, #1793, #1615 — do NOT reassign
9. **Model capacity is NOT the p_tan bottleneck** — Scale-up (#2100): p_tan ~30-32 across 3L/96s, 5L/96s, 3L/160s, 4L/128s
10. **EMA is load-bearing** — SWAD (#2094) catastrophe: disabling EMA → +268% p_in
11. **Transolver blocks learn complementary representations** — Weight-tied iteration (#2103) proves the 3 blocks are NOT doing iterative refinement
12. **SIREN doesn't help surface refinement** — (#2102) srf_head corrections lack oscillatory structure
13. **Asinh transform direction is exhausted** — Symmetric s=0.75 is optimal
14. **Contrastive tandem-single rep separation is NOT the bottleneck** — (#2109) hypothesis cleanly falsified; the model already routes tandem/single differently; forcing separation doesn't help
15. **p_tan has a coordinate-frame component** — (#2107) local-frame normalization improved p_tan -2.6% but cost p_in regression; non-destructive dual-frame approach is the next test
16. **Progressive surface weight scheduling doesn't help** — (#2110) dynamic surf_weight from epoch 0 is already well-tuned; delaying surface focus reduces total surface gradient → p_in regresses; p_tan mild improvement doesn't compensate

## Potential Next Research Directions (available, not yet assigned)

**Priority queue (assign next idle students):**
1. **Separate SRF Heads for Fore-Foil (ID=6) vs Single-Foil (ID=5)** — extend the aft-foil SRF concept to the fore-foil; fore-foil sees aerodynamically different flow in tandem vs single; queue after #2104 results; expected -3 to -6% p_tan; ~30 LoC; low risk
3. **Precomputed Pressure-Poisson Soft Constraint** — baked finite-diff Laplacian stencil; distinct from failed WLS; targets p_tan/p_oodc; ~65 lines
4. **Charbonnier Loss** — `sqrt(x^2 + eps^2) - eps` — smoother than L1/Huber; natural follow-up if Huber (#2113) shows promise
5. **Langevin Gradient Noise** — Gaussian noise to gradients after Lion (SGLD-style); distinct from SAM; targets flat basin finding

**Deferred pending current results:**
- Expand ensemble to 23 seeds (100-106 already trained) — do this after any single-model improvements land
- Aft-foil loss upweighting (only if dual-frame doesn't fix p_tan)

## Confirmed Dead Ends (all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Progressive surface focus (curriculum ramp) | #2110 | p_in regresses +0.7%; dynamic surf_weight already optimal; p_tan mild improvement insufficient |
| Contrastive tandem-single regularization | #2109 | Hypothesis falsified: rep entanglement is NOT the bottleneck; p_tan unchanged across 3 weights |
| Asymmetric Asinh Scales (pos/neg) | #2108 | All metrics worse; symmetric s=0.75 is optimal; **closes entire asinh direction** with #2096, #2098 |
| Weight-Tied Iterative Transolver | #2103 | Monotonic degradation; blocks learn complementary reps, not iterative |
| SIREN Activation in SRF Head | #2102 | Monotonic degradation with omega; GELU is well-calibrated for srf corrections |
| Deep Supervision (aux loss) | #2097 | p_in -1.7% but p_tan +2.7% — constrains representational flexibility for tandem transfer |
| OHEM (hard example mining) | #2101 | No improvement; sample reweighting is not the bottleneck |
| Model Scale-Up (5L, 160s, 4L/128s) | #2100 | No config beats 3L/96s; p_tan NOT capacity-limited |
| SWAD | #2094 | Catastrophic (+268% p_in); suppresses EMA |
| Stochastic Depth (DropPath) | #2099 | All metrics worse; 3-block too shallow |
| Knowledge Distillation | #2090 | No improvement at convergence |
| SGDR warm restarts | #2095 | All T_0 values worse |
| SAM Phase-Only | #2086 | SAM destabilizes Lion |
| srf4L | #2079,2081,2083,2085 | p_tan +5-7% WORSE |
| Mesh interpolation | #2066 | Physically invalid for unstructured CFD |
| SOAP/HeavyBall | #2010,2018-2023 | 2-6% WORSE than Lion |
| Muon (full+hybrid) | #2006 | 30-70% worse |
| XSA attention | #2007 | Redundant with orthogonal slices |
| PirateNets RWF | #2008 | Attenuated by LayerNorm + Lion |
| NOBLE | #2011 | Model too small |
| LinearNO | #2033-2038 | All failed |
| Flow matching | #2036 | 60% worse (deterministic) |
| MARIO latent | #2037 | Redundant geometry encoding |
| Inviscid Cp | #2034 | Single-foil wrong for tandem |
| All-to-all surface attn | #2035 | +8% worse |
| Physics losses (WLS) | #2016,2023 | WLS instability |
| MC Dropout | #2088 | Null result |
| Packed Ensemble | #2082 | Model too small |
| Ensemble Weight Opt | #2089 | Equal weights — no benefit |
| Diverse Hparam Ensemble | #2091 | Seed diversity strictly better |
| Magnitude-Weighted Loss | #2068 | No sweet spot at any alpha |
| Asinh Velocity Transform | #2098 | Hurts p_tan |
| Learnable Asinh Scale | #2096 | Scale collapse to ~0.003 |
| Ada-Temp (per-node) | #1879, #1793, #1615 | Null result across multiple phases |
| Foil-2 (ID=7) Loss Upweighting | #1893 | Marginal improvement (still one head) |

## Human Researcher Directives

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Acknowledged. Phase 5 tested 6 new architectures (all failed); focus now on loss reformulation, coordinate representation, and training dynamics.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Acknowledged and confirmed.

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status | Run IDs |
|-------|-------|--------|---------|
| Batch 1 | 42-49 | ✓ BASELINE | f59v5aul, 0yurebjv, rdezx8es, ds12ug79, yu1x0dy0, y147zvh1, lc5cbt4l, 7cxu38oh |
| Batch 2 | 66-73 | ✓ BASELINE | j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukv7wv6, xlnhwuqc, ii1tz4vv |
| Batch 3 | 74-81 | ✓ Trained | 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4 |
| Batch 4 | 82-89 | ✓ Trained | u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2 |
| Batch 5 | 90-95 | ✓ Trained | ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe |
| Batch 6 | 100-106 | ✓ Trained | 9o85duyc, ec7plfg8, zagg4pfs, 6w86plz1, g00kxdva, jt9hwf40, fom4bzro |

**Total trained: 45 models.** 23-seed evaluation (42-49 + 66-73 + 100-106) available at any time.
