# SENPAI Research State

- **Date:** 2026-04-04 ~19:30 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PRs #2104 + #2115 merged — combined 8-seed validation PR #2123)

| Metric | Combined 8-seed mean | Target to beat |
|--------|---------------------|----------------|
| p_in | 13.24 ± 0.33 | < 13.24 |
| p_oodc | 7.73 ± 0.22 | < 7.73 |
| **p_tan** | **30.53 ± 0.50** | **< 30.53** |
| p_re | 6.50 ± 0.07 | < 6.50 |

⚠️ **Key interaction finding (PR #2123):** Gap/stagger augmentation + aft_foil_srf is NOT additive.
- p_oodc improves (-2.4% vs aft_srf-only 7.92 ✓)
- p_tan regresses (+1.6% vs aft_srf-only 30.05 ✗ — PRIMARY metric hurt)
- aft_foil_srf-only gave p_tan=30.05; combined gives 30.53

This suggests gap/stagger noise disrupts aft-foil SRF training. All in-flight students run with both flags, so compare against 30.53.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~22:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2130 | Gap/Stagger-Conditioned Spatial Bias — Tandem-Geometry-Aware Slice Routing | WIP — just assigned |
| alphonse | #2131 | Tandem-Slice Carve-Out — Reserved Physics Slices for Tandem Samples | WIP — just assigned |
| tanjiro | #2126 | Foil-2 DSDF Magnitude Augmentation (σ sweep: 0.05/0.10/0.15) | WIP |
| frieren | #2127 | Context-Aware AftSRF — KNN Volume Context for Wake Pressure (K=8) | WIP |
| nezuko | #2129 | Supervised Surface Pressure Gradient Aux Loss (w=0.05/0.10, seeds 42/73) | WIP |
| edward | #2128 | Reynolds-Conditional SRF — FiLM on (Re, AoA) for surface_refine head | WIP |
| askeladd | #2119 | PCGrad 2-Way Validation — 8-seed (seeds 42-49) with gap_stagger aug | WIP (sent back 2x for validation) |
| thorfinn | #2132 | Tandem DSDF Channel Mixup — Synthetic Foil-Shape Interpolation | WIP — just assigned |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-04 ~19:30)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2125 | thorfinn | Reynolds Number Perturbation Augmentation — σ=0.05 | **CLOSED** | Null result — all metrics within noise of baseline. p_re unchanged (6.50 avg), p_tan regressed seed 43 (+1.5). OOD-Re gap too small in log-space for domain randomization to help. |
| #2124 | fern | Fore-Foil Stacked SRF Head (ID=6) — Additive | **CLOSED** | Both 192/3L and 128/2L stacked heads degrade p_oodc by ~0.4; 128/2L s43 catastrophic (p_tan=32.8). Combined with #2117 (split), fore-foil SRF exhausted in all formulations. |
| #2123 | alphonse | Combined Baseline 8-Seed Validation | **CLOSED** | Validation complete. Critical finding: gap_stagger aug + aft_foil_srf not additive — p_tan regresses +1.6%. Baseline updated to combined numbers. |

## Current Research Focus

### Primary target: p_tan = 30.53 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head (ID=7) — p_tan -0.8%
2. `--aug_gap_stagger_sigma 0.02`: domain randomization on tandem scalars — p_oodc -2.4% (vs combined baseline)
3. `--surface_refine`, `--residual_prediction`, `--pressure_first`, `--pressure_deep`, `--asinh_pressure 0.75`, etc.

**Active experiments (8 students):**
1. **Gap/Stagger-Conditioned Spatial Bias** (fern #2130) — Extend raw_xy in TransolverBlock spatial_bias from 4→6 dims by appending gap+stagger; makes slice routing tandem-geometry-aware; zero effect on single-foil by design; ~18 LoC; expected -3 to -7% p_tan; **TOP PRIORITY IDEA**
2. **Tandem-Slice Carve-Out** (alphonse #2131) — Reserve K dedicated physics slices exclusively for tandem samples via large negative bias on reserved slice logits for single-foil; K sweep {4, 8}; ~20 LoC; targets tandem representational capacity
3. **Foil-2 DSDF Magnitude Aug** (tanjiro #2126) — log-normal scale of foil-2 DSDF channels (tandem only); σ sweep {0.05, 0.10, 0.15}; 6 runs total
4. **Context-Aware AftSRF** (frieren #2127) — KNN volume context (K=8, zone_id=2 nodes) for aft-foil SRF head; physically motivated by non-local fore→aft wake dependency
5. **Supervised Surface Pressure Gradient Aux Loss** (nezuko #2129) — L1 on chord-wise pressure gradient finite differences; targets spatial structure of Cp; w=0.05/0.10 sweep
6. **Tandem DSDF Channel Mixup** (thorfinn #2132) — interpolate DSDF channels (x[:,2:10]) between pairs of tandem training samples using Beta(0.7/0.5, 0.4); creates synthetic intermediate foil geometries; α sweep {0.7, 0.5}; targets p_tan via foil-shape generalization
7. **PCGrad 2-Way Validation** (askeladd #2119) — 8-seed validation of 2-way PCGrad (single-foil vs all-tandem); p_oodc signal -1.3% to -5.4% in initial 4 runs
8. **Reynolds-Conditional SRF** (edward #2128) — FiLM conditioning on (Re, AoA) for surface_refine head; targets p_re and p_oodc

**Key research patterns from recent experiments:**
- **What works:** Additive specialized heads (aft_srf), target transforms (asinh), data augmentation (gap/stagger helps p_oodc but not p_tan)
- **What doesn't work:** Fore-foil SRF in any formulation (split or stacked), loss reweighting by surface region, fore-foil coordinate frames, sparse boundary-type features
- **Design principle:** SRF specialization must be ADDITIVE (stack on shared head), not REPLACING
- **Design principle:** Boundary-type information must be delivered architecturally (dedicated heads), not as sparse input features
- **New finding:** Gap/stagger augmentation and aft_foil_srf interact negatively on p_tan — future experiments should consider these carefully
- **Emerging signal:** 2-way PCGrad (single-foil vs tandem) produces consistent p_oodc improvement; awaiting 8-seed validation

## Potential Next Research Directions (not yet assigned)

**From existing queue (DSDF Mixup now assigned to thorfinn #2132):**
1. **Tandem-Aware Temperature Annealing** (Idea 6 ROUND2) — Check if tandem_temp_offset is already wired in Physics_Attention; if not, enable it (softer attention for tandem, sharper for single-foil)
2. **Foil-2 AoA Rotation Aug** (Idea 7 ROUND2, bold) — Independent AoA perturbation for aft-foil nodes; increases (fore_AoA, aft_AoA) diversity; high implementation complexity
3. **Aft-Foil TV Loss** (Idea 5 from 22:00 file) — unsupervised chord-wise total variation regularization on aft-foil predictions; wait for nezuko #2129 results first
4. **EMA Stochastic Weight Perturbation** (Idea 2 from 22:00 file) — inject small noise at EMA start epoch, flat-minima seeking; ~12 LoC; σ sweep {5e-4, 1e-3, 3e-3}
5. **Drop gap_stagger aug from baseline** — Combined validation showed p_tan regresses +1.6% with gap_stagger. Consider testing aft_srf-only to recover the p_tan=30.05 target. (Trade: lose p_oodc=7.73, recover p_tan=30.05)

**Strategic consideration:** Gap/stagger augmentation hurts p_tan but helps p_oodc. If upcoming experiments improve p_tan, gap_stagger may need to be revisited. Alternatively, new ideas that improve p_tan without gap_stagger could use the aft_srf-only baseline (p_tan=30.05) as a starting point.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Fore-Foil SRF (all formulations) | #2117, #2124 | Split: +9-11% p_tan. Stacked: +1-4% p_tan, p_oodc degrades. Fore-foil channel flow is already well-handled by shared srf_head. |
| Aft-Foil Loss Upweighting | #2121 | p_oodc improves -2% but p_tan regresses +1.5%; loss weighting benefits OOD-C not tandem |
| Fore-Foil Loss Upweighting | #2122 | p_oodc improves -1.8% but p_tan regresses +1.5-1.9%; confirms loss-region-weighting is dead |
| Aft-Foil Coordinate Frame (dual-frame) | #2107 | 3 iterations, 8 runs, p_in +3%, p_tan barely moves; high variance |
| Fore-Foil SRF Split (narrowing shared head) | #2117 | +9–11% p_tan regression; loses tandem transfer learning |
| Boundary-ID One-Hot (sparse 3-dim input) | #2118 | p_tan +2.1%; sparse features disrupt slice assignment |
| Charbonnier Loss | #2116 | All eps values degrade all metrics; smooth loss family exhausted |
| Mesh-Density Weighted L1 | #2112 | All metrics regressed 5–16% |
| Smooth L1 / Huber Loss | #2113 | Catastrophic |
| Gradient Centralization (GC-Lion) | #2114 | Incompatible with Lion sign operation |
| Reynolds Number Perturbation Augmentation | #2125 | Null result + p_tan regression (seed 43 +1.5). OOD-Re gap too small in log-space for domain randomization. |
| Langevin Gradient Noise (SGLD) | #2120 | No improvement; Lion sign-based updates already provide implicit gradient noise |
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

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Phase 5 tested 6 new architectures (all failed); current focus on routing/capacity, loss reformulation, novel data augmentation.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Confirmed.

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status |
|-------|-------|--------|
| Batch 1 | 42-49 | ✓ BASELINE (combined aft_srf + aug) |
| Batch 2 | 66-73 | ✓ ENSEMBLE |
| Batch 3 | 74-81 | ✓ Trained |
| Batch 4 | 82-89 | ✓ Trained |
| Batch 5 | 90-95 | ✓ Trained |
| Batch 6 | 100-106 | ✓ Trained (available for 23-seed ensemble) |

**Total trained: 45 models.** 23-seed evaluation (42-49 + 66-73 + 100-106) available; defer until single-model improvements land.
