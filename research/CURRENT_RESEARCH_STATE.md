# SENPAI Research State

- **Date:** 2026-04-04 ~18:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PRs #2104 + #2115 merged — +aft_foil_srf +aug_gap_stagger σ=0.02, 8-seed mean, seeds 42-49)

| Metric | 8-seed mean | Target to beat |
|--------|-------------|----------------|
| p_in | 13.19 ± 0.33 | < 13.19 |
| p_oodc | 7.92 ± 0.17 | < 7.92 |
| **p_tan** | **30.05 ± 0.36** | **< 30.05** |
| p_re | 6.45 ± 0.07 | < 6.45 |

**Note:** PR #2123 (alphonse, 8-seed combined validation) in progress — will give accurate combined baseline for aft_foil_srf + aug_gap_stagger together.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~18:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2126 | Foil-2 DSDF Magnitude Augmentation (σ sweep: 0.05/0.10/0.15) | WIP — just assigned |
| frieren | #2127 | Context-Aware AftSRF — KNN Volume Context for Wake Pressure (K=8) | WIP — just assigned |
| nezuko | #2122 | Fore-Foil Loss Upweighting (ID=6) — Symmetric to Aft-Foil Weight | WIP |
| fern | #2124 | Fore-Foil Stacked SRF Head (ID=6) — Additive, Not Split | WIP |
| edward | #2120 | Langevin Gradient Noise (SGLD) — Stochastic Exploration for Lion | WIP |
| askeladd | #2119 | PCGrad 3-Way Task Split — Gradient Surgery (single/tandem-normal/tandem-extreme) | WIP |
| alphonse | #2123 | Combined Baseline 8-Seed Validation (aft_foil_srf + gap/stagger aug, seeds 42-49) | WIP |
| thorfinn | #2125 | Reynolds Number Perturbation Augmentation — OOD-Re Robustness | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-04 ~17:30)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2121 | tanjiro | Aft-Foil Loss Upweighting (w=1.5, w=2.0) | **CLOSED** | p_oodc consistently beats baseline (-2%) but p_tan regresses +1.5-1.7%. Loss weighting helps OOD-C but trades off tandem. |
| #2107 | frieren | Aft-Foil Coordinate Frame (dual-frame v2, 4-seed final) | **CLOSED** | After 3 iterations and 8 runs: 4-seed mean p_in +3.0%, p_tan +0.6%. High p_tan variance (range 1.55). Approach explored. |

## Current Research Focus

### Primary target: p_tan = 30.05 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head (ID=7) — p_tan -0.8%
2. `--aug_gap_stagger_sigma 0.02`: domain randomization on tandem scalars — p_oodc -4.9%, p_re -1.5%
3. `--surface_refine`, `--residual_prediction`, `--pressure_first`, `--pressure_deep`, `--asinh_pressure 0.75`, etc.

**Active experiments (8 students):**
1. **Foil-2 DSDF Magnitude Aug** (tanjiro #2126) — log-normal scale of foil-2 DSDF channels (tandem only); σ sweep {0.05, 0.10, 0.15}; 6 runs total; direct geometric extension of gap/stagger aug; targets p_tan by forcing shape-independent representations
2. **Context-Aware AftSRF** (frieren #2127) — KNN volume context (K=8, zone_id=2 nodes) for aft-foil SRF head; gives wake-state information to the correction head; physically motivated by non-local fore→aft wake dependency
3. **Fore-Foil Loss Upweighting (ID=6)** (nezuko #2122) — upweight fore-foil nodes 1.5–2.0× in surface loss
4. **Fore-Foil Stacked SRF Head (ID=6)** (fern #2124) — additive fore_srf_head on top of shared srf_head
5. **Reynolds Number Perturbation Augmentation** (thorfinn #2125) — add Gaussian noise to log_Re during training; σ sweep {0.05, 0.1, 0.2}; targets p_re < 6.45
6. **PCGrad 3-Way Task Split** (askeladd #2119) — gradient surgery across single/tandem-normal/tandem-extreme-Re
7. **Langevin Gradient Noise / SGLD** (edward #2120) — Gaussian noise after Lion step; sweep {5e-5, 1e-4, 3e-4}
8. **Combined Baseline 8-Seed Validation** (alphonse #2123) — runs seeds 42-49 with aft_foil_srf + aug_gap_stagger_sigma=0.02 combined; gives accurate merge targets

**Key research patterns from recent experiments:**
- **What works:** Data augmentation (gap/stagger aug), dedicated additive heads (aft_srf), target transforms (asinh)
- **What doesn't work:** Loss reweighting by surface region (p_oodc improves but p_tan regresses), input feature engineering (boundary-ID one-hot, coordinate frames), loss function changes (Charbonnier, Huber, GC)
- **Design principle:** SRF specialization must be ADDITIVE (stack on shared head), not REPLACING
- **Design principle:** Boundary-type information must be delivered architecturally (dedicated heads), not as sparse input features

## Potential Next Research Directions (not yet assigned)

**From RESEARCH_IDEAS_2026-04-04_ROUND2.md (high priority):**
1. **DSDF Mixup in Tandem Samples** (Idea 2) — Mixup between two tandem samples in DSDF channel space only; interpolates foil shape encodings; targets p_tan geometric transfer gap
2. **Pressure-Gradient Consistency Aux Loss** (Idea 5) — L1 loss on pressure gradient differences between adjacent aft-foil surface nodes; TV-style physics regularizer; ~40 LoC
3. **Tandem-Aware Temperature Annealing** (Idea 6) — Check if tandem_temp_offset is already wired in Physics_Attention; if not, enable it; soft/hard slice assignment per regime
4. **Foil-2 AoA Rotation Aug** (Idea 7, bold) — Independent AoA perturbation for aft-foil nodes; increases (fore_AoA, aft_AoA) diversity; small perturbation (~0.17 deg)

**From CURRENT_RESEARCH_STATE.md priority queue:**
5. **Precomputed Pressure-Poisson Soft Constraint** — finite-diff Laplacian stencil as auxiliary physics loss; ~65 LoC; MEDIUM-HIGH risk
6. **Fixed per-boundary asinh scale** — different fixed scales for each boundary type (ID=5/6/7)
7. **AoA-consistent augmentation fix** — current aoa_perturb perturbs x but not y proportionally

**Researcher-agent background run:** Launched 2026-04-04 ~17:45 to generate additional fresh hypotheses from literature search.

**Deferred pending current results:**
- Expand ensemble to 23 seeds (seeds 100-106 already trained) — defer until single-model improvements land
- NACA Shape Embedding (Idea 4 ROUND2) — low confidence with only 2 training foil families

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Aft-Foil Loss Upweighting | #2121 | p_oodc improves -2% but p_tan regresses +1.5%; loss weighting benefits OOD-C not tandem |
| Aft-Foil Coordinate Frame (dual-frame) | #2107 | 3 iterations, 8 runs, p_in +3%, p_tan barely moves; high variance |
| Fore-Foil SRF Split (narrowing shared head) | #2117 | +9–11% p_tan regression; loses tandem transfer learning |
| Boundary-ID One-Hot (sparse 3-dim input) | #2118 | p_tan +2.1%; sparse features disrupt slice assignment |
| Charbonnier Loss | #2116 | All eps values degrade all metrics; smooth loss family exhausted |
| Mesh-Density Weighted L1 | #2112 | All metrics regressed 5–16% |
| Smooth L1 / Huber Loss | #2113 | Catastrophic |
| Gradient Centralization (GC-Lion) | #2114 | Incompatible with Lion sign operation |
| Fourier Feature Position Encoding | #2106 | p_oodc +4.8%, p_re +6.2% regression |
| Deep Supervision (aux loss) | #2097 | p_tan +2.7% regression |
| SWAD | #2094 | Catastrophic |
| SGDR warm restarts | #2095 | All T_0 values worse |
| SAM Phase-Only | #2086 | Destabilizes Lion |
| srf4L (deeper SRF) | #2079-2085 | p_tan +5-7% WORSE |
| FiLM on gap/stagger | #2104 | p_oodc +41.6% catastrophe |
| Contrastive tandem-single regularization | #2109 | Hypothesis falsified |
| Model Scale-Up | #2100 | NOT capacity-limited |
| OHEM (hard example mining) | #2101 | No improvement |
| TTA via AoA Perturbation | #2111 | Self-defeating under timeout |
| Progressive surface focus | #2110 | p_in regresses |
| Asymmetric Asinh Scales | #2108 | All metrics worse |
| Weight-Tied Iterative Transolver | #2103 | Monotonic degradation |
| SIREN in SRF Head | #2102 | Monotonic degradation |
| All Phase 5 architectures (GNOT, Galerkin, etc.) | multiple | 5–59% worse |

## Human Researcher Directives

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Phase 5 tested 6 new architectures (all failed); current focus on loss reformulation, coordinate representation, training dynamics.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Confirmed.

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status |
|-------|-------|--------|
| Batch 1 | 42-49 | ✓ BASELINE (aft_srf) |
| Batch 2 | 66-73 | ✓ ENSEMBLE |
| Batch 3 | 74-81 | ✓ Trained |
| Batch 4 | 82-89 | ✓ Trained |
| Batch 5 | 90-95 | ✓ Trained |
| Batch 6 | 100-106 | ✓ Trained (available for 23-seed ensemble) |

**Total trained: 45 models.** 23-seed evaluation (42-49 + 66-73 + 100-106) available; defer until single-model improvements land.
