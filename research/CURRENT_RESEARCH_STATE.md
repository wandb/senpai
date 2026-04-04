# SENPAI Research State

- **Date:** 2026-04-04 ~12:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2104 merged — +aft_foil_srf, 8-seed mean, seeds 42-49)

| Metric | 8-seed mean | Target to beat |
|--------|-------------|----------------|
| p_in | 13.19 ± 0.33 | < 13.19 |
| p_oodc | 7.92 ± 0.17 | < 7.92 |
| p_tan | **30.05 ± 0.36** | **< 30.05** |
| p_re | 6.45 ± 0.07 | < 6.45 |

Baseline config now includes `--aft_foil_srf`. All new experiments must use it.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~12:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2117 | Fore-Foil Dedicated SRF Head (ID=6) — Split from Single-Foil | WIP — just assigned |
| tanjiro | #2114 | Gradient Centralization — Zero-Mean Gradient Updates with Lion | WIP |
| edward | #2113 | Smooth L1 (Huber) Loss — Node-Level Gradient Emphasis | WIP |
| askeladd | #2106 | Fourier Feature Position Encoding — Spectral Bias Correction | WIP |
| frieren | #2107 | Aft-Foil Coordinate Frame Normalization (dual-frame iteration) | WIP — sent back |
| nezuko | #2115 | Gap/Stagger Perturbation Augmentation — Tandem OOD Robustness | WIP |
| alphonse | #2116 | Charbonnier Loss — Fully Smooth L1 (eps sweep: 0.05, 0.1, 0.2) | WIP |
| thorfinn | #2112 | Mesh-Density Weighted L1 — Upweight Fine-Mesh Regions | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed / Merged (2026-04-04)

| PR | Student | Experiment | Decision | p_tan result |
|----|---------|-----------|---------|--------------|
| #2104 | fern | Dedicated Aft-Foil SRF Head (ID=7) | **MERGED** | 30.05 ± 0.36 (-0.8% vs 30.29 baseline) |
| #2111 | alphonse | TTA via AoA Perturbation | CLOSED | Self-defeating under timeout; marginal at matching epochs |
| #2110 | nezuko | Progressive Surface Focus Schedule | CLOSED | p_in regresses +0.7%; dynamic surf_weight already optimal |
| #2109 | tanjiro | Contrastive Tandem-Single Regularization | CLOSED | Hypothesis falsified: p_tan unchanged; rep entanglement is NOT bottleneck |
| #2107 | frieren | Aft-Foil Local Frame (in-place) | SENT BACK | p_tan -2.6% but p_in regressed; non-destructive dual-frame retry assigned |
| #2108 | edward | Asymmetric Fixed Asinh Scales | CLOSED | All metrics worse; closes asinh direction entirely |

## Current Research Focus

### Primary target: p_tan = 30.05 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head (ID=7) — p_tan -0.8% (30.29→30.05)
2. `--surface_refine`: dedicated surface MLP (all nodes shared) — massive p_re improvement
3. `--residual_prediction`, `--pressure_first`, `--pressure_deep`, `--asinh_pressure 0.75`, etc.

**Active experiments:**
1. **Fore-Foil SRF Head (ID=6)** (fern #2117) — natural extension of aft-foil SRF; expected -3 to -6% p_tan; LOW RISK; zero-init
2. **Dual-Frame Coordinate Features** (frieren #2107) — add local-frame coords as ADDITIONAL features; p_tan -2.6% previously but p_in regressed with in-place replacement; non-destructive version being tested
3. **Gap/Stagger Perturbation Aug** (nezuko #2115) — domain randomization on tandem geometry features; targets p_tan OOD axis
4. **Mesh-Density Weighted L1** (thorfinn #2112) — upweight fine-mesh nodes (leading edge, suction peaks)
5. **Fourier Feature Position Encoding** (askeladd #2106) — spectral bias correction via RFF
6. **Smooth L1 (Huber) Loss** (edward #2113) — sweep beta={0.5, 1.0, 2.0}
7. **Gradient Centralization** (tanjiro #2114) — remove DC gradient component before Lion sign
8. **Charbonnier Loss** (alphonse #2116) — fully smooth L1; sweep eps={0.05, 0.1, 0.2}

**Confirmed bottleneck findings:**
- NOT capacity-limited (scale-up #2100)
- NOT sample-level reweighting (OHEM #2101)
- NOT representational entanglement (contrastive #2109: hypothesis falsified)
- NOT progressive surface weight scheduling (#2110: p_in regresses)
- HAS coordinate-frame component (local-frame #2107: p_tan -2.6% with right implementation)
- HAS dedicated-head component (aft-foil SRF #2104: p_tan -0.8% confirmed)

## Potential Next Research Directions (not yet assigned)

**Priority queue (assign to next idle students):**
1. **Boundary-ID One-Hot as Sideband Feature** — explicit per-node surface-type conditioning (ID=5/6/7 as 3-dim one-hot input feature); expected -3 to -8% p_tan; ~10 LoC; MEDIUM risk
2. **Tandem-Specific PCGrad 3-Way Split** — split Group B further into tandem vs. extreme-Re/AoA; expected -2 to -5% p_tan; ~15 LoC; MEDIUM risk
3. **Langevin Gradient Noise (SGLD-style)** — Gaussian noise to gradients after Lion update; expected -1 to -3% p_in; ~10 LoC; LOW-MEDIUM risk
4. **Precomputed Pressure-Poisson Soft Constraint** — finite-diff Laplacian stencil as auxiliary loss; targets p_tan/p_oodc

**Deferred pending current results:**
- Expand ensemble to 23 seeds (seeds 100-106 already trained) — do after single-model improvements land
- Aft-foil loss upweighting — deferred now that dedicated SRF head is merged

## Key Research Insights

1. **Architecture changes are absorbed by EMA+Lion** — Phase 5: 6 new architectures all failed
2. **Target-space transforms cannot be learnable** — Learnable asinh scale collapses
3. **Ensembling is near-saturating** — 16→23 seeds gains diminish; single-model improvements priority
4. **Deep supervision improves p_in but hurts p_tan** — (#2097) auxiliary loss constrains intermediate reps
5. **Dedicated SRF heads work for boundary-specific geometry** — #2104 confirms per-boundary-type correction
6. **Model capacity is NOT the p_tan bottleneck** — scale-up (#2100) falsified
7. **Contrastive rep separation is NOT the bottleneck** — (#2109) cleanly falsified
8. **p_tan has a coordinate-frame component** — (#2107) local-frame improved p_tan -2.6% with correct implementation
9. **FiLM conditioning on gap/stagger overfits** — (#2104) catastrophic p_oodc regression with conditioning

## Human Researcher Directives

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Phase 5 tested 6 new architectures (all failed); current focus on loss reformulation, coordinate representation, training dynamics.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Confirmed.

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status | Run IDs |
|-------|-------|--------|---------|
| Batch 1 | 42-49 | ✓ BASELINE (aft_srf) | fctgmn1d, rc40fpuu, ygqo9rom, r5uxnp4b, yxhjfisl, qrbprrli, 9whdgscd, ekdcwekr |
| Batch 2 | 66-73 | ✓ ENSEMBLE | j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv |
| Batch 3 | 74-81 | ✓ Trained | 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4 |
| Batch 4 | 82-89 | ✓ Trained | u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2 |
| Batch 5 | 90-95 | ✓ Trained | ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe |
| Batch 6 | 100-106 | ✓ Trained | 9o85duyc, ec7plfg8, zagg4pfs, 6w86plz1, g00kxdva, jt9hwf40, fom4bzro |

**Total trained: 45 models.** 23-seed evaluation (42-49 + 66-73 + 100-106) available; defer until single-model improvements land.

## Confirmed Dead Ends (all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
| TTA via AoA Perturbation (training-loop) | #2111 | Self-defeating under timeout; marginal at matching epochs |
| Progressive surface focus (curriculum ramp) | #2110 | p_in regresses +0.7%; dynamic surf_weight already optimal |
| Contrastive tandem-single regularization | #2109 | Hypothesis falsified: rep entanglement NOT bottleneck |
| Asymmetric Asinh Scales (pos/neg) | #2108 | All metrics worse; closes asinh direction |
| Weight-Tied Iterative Transolver | #2103 | Monotonic degradation |
| SIREN Activation in SRF Head | #2102 | Monotonic degradation |
| Deep Supervision (aux loss) | #2097 | p_tan +2.7% regression |
| OHEM (hard example mining) | #2101 | No improvement |
| Model Scale-Up (5L, 160s, 4L/128s) | #2100 | NOT capacity-limited |
| SWAD | #2094 | Catastrophic (+268% p_in) |
| Stochastic Depth (DropPath) | #2099 | All metrics worse |
| Knowledge Distillation | #2090 | No improvement |
| SGDR warm restarts | #2095 | All T_0 values worse |
| SAM Phase-Only | #2086 | Destabilizes Lion |
| srf4L (deeper SRF) | #2079-2085 | p_tan +5-7% WORSE |
| Mesh interpolation | #2066 | Physically invalid |
| SOAP/HeavyBall | #2010,2018-2023 | 2-6% WORSE |
| Muon | #2006 | 30-70% worse |
| FiLM on gap/stagger | #2104 | p_oodc catastrophe (+41.6%) |
| Foil-2 Loss Upweighting | #1893 | Marginal (still shared head) |
| Ada-Temp (per-node) | #1879,#1793,#1615 | Null across multiple phases |
