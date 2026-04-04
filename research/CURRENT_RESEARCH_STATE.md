# SENPAI Research State

- **Date:** 2026-04-04 ~20:45 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PRs #2104 + #2115 + #2126 merged — 2-seed evidence)

| Metric | Current Baseline | vs Prior (combined 8-seed) |
|--------|-----------------|---------------------------|
| p_in | **13.04** | -1.5% |
| p_oodc | **7.66** | -0.9% |
| **p_tan** | **30.11** | **-1.4%** |
| p_re | 6.52 | +0.3% (noise) |

**Latest merge:** PR #2126 (tanjiro) — DSDF2 magnitude aug σ=0.05. W&B: hcc2q68t (s42, p_tan=29.76), e9cri4mt (s73, p_tan=30.46). Metrics fully W&B-verified.

⚠️ **Important:** Current baseline is 2-seed only. Previous combined 8-seed mean was p_tan=30.53. All in-flight experiments should target p_tan < 30.11.

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-dsdf2-aug" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --disable_pcgrad --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~20:45 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2133 | Foil-1 DSDF Magnitude Aug — Front Foil Shape Transfer | WIP — just assigned |
| fern | #2130 | Gap/Stagger-Conditioned Spatial Bias — Tandem-Geometry-Aware Slice Routing | WIP |
| alphonse | #2131 | Tandem-Slice Carve-Out — Reserved Physics Slices for Tandem Samples | WIP |
| frieren | #2127 | Context-Aware AftSRF — KNN Volume Context for Wake Pressure (K=8) | WIP |
| nezuko | #2129 | Supervised Surface Pressure Gradient Aux Loss (w=0.05/0.10) | WIP |
| edward | #2128 | Reynolds-Conditional SRF — FiLM on (Re, AoA) for surface_refine head | WIP |
| askeladd | #2119 | PCGrad 2-Way Validation — 8-seed validation | WIP |
| thorfinn | #2132 | Tandem DSDF Channel Mixup — Synthetic Foil-Shape Interpolation | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-04 ~20:30)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2126 | tanjiro | Foil-2 DSDF Magnitude Augmentation — σ sweep | **MERGED** | σ=0.05 best: p_tan 30.53→30.11 (-1.4%), p_in -1.5%, p_oodc -0.9%. W&B verified all 6 runs. |
| #2125 | thorfinn | Reynolds Number Perturbation Augmentation — σ=0.05 | **CLOSED** | Null result — all metrics within noise of baseline. Re perturbation dead end. |
| #2124 | fern | Fore-Foil Stacked SRF Head (ID=6) — Additive | **CLOSED** | Both 192/3L and 128/2L degrade p_oodc; fore-foil SRF exhausted in all formulations. |
| #2123 | alphonse | Combined Baseline 8-Seed Validation | **CLOSED** | Validation: gap_stagger aug + aft_foil_srf not additive — p_tan regresses +1.6%. |

## Current Research Focus

### Primary target: p_tan = 30.11 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head (ID=7)
2. `--aug_gap_stagger_sigma 0.02`: tandem scalar domain randomization — helps p_oodc, slightly hurts p_tan
3. `--aug_dsdf2_sigma 0.05`: foil-2 DSDF magnitude aug — p_tan -1.4% (PR #2126, just merged)
4. `--surface_refine`, `--residual_prediction`, `--pressure_first`, `--pressure_deep`, `--asinh_pressure 0.75`, etc.

**Active experiments (8 students):**
1. **Foil-1 DSDF Magnitude Aug** (tanjiro #2133) — Mirror of DSDF2 aug for front foil. val_tandem_transfer tests NACA6416 as FRONT foil → foil-1 DSDF channels directly encode this. σ sweep {0.05, 0.10, 0.15}; **HIGH PRIORITY**
2. **Gap/Stagger-Conditioned Spatial Bias** (fern #2130) — Extend spatial_bias 4→6 dims by appending gap+stagger; tandem-geometry-aware slice routing; **TOP ARCHITECTURE IDEA**
3. **Tandem-Slice Carve-Out** (alphonse #2131) — Reserve K dedicated physics slices for tandem via large negative bias on single-foil; K sweep {4, 8}
4. **Context-Aware AftSRF** (frieren #2127) — KNN volume context (K=8) for aft-foil SRF head; non-local fore→aft wake dependency
5. **Supervised Surface Pressure Gradient Aux Loss** (nezuko #2129) — L1 on chord-wise pressure gradient finite differences; w=0.05/0.10 sweep
6. **Tandem DSDF Channel Mixup** (thorfinn #2132) — interpolate DSDF channels between tandem training samples; creates synthetic intermediate foil geometries; α sweep {0.7, 0.5}
7. **PCGrad 2-Way Validation** (askeladd #2119) — 8-seed validation of 2-way PCGrad; prior 4-seed showed p_oodc -1.3% to -5.4%
8. **Reynolds-Conditional SRF** (edward #2128) — FiLM conditioning on (Re, AoA) for surface_refine head; targets p_re and p_oodc

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 confirmed, foil-1 being tested), additive specialized heads (aft_srf), target transforms (asinh)
- **What doesn't work:** Fore-foil SRF (all formulations), loss reweighting, Re perturbation aug, Charbonnier/Smooth-L1 loss families
- **Critical interaction:** gap_stagger aug + aft_foil_srf NOT additive — helps p_oodc but hurts p_tan by +1.6%
- **Emerging mechanism:** Log-normal DSDF magnitude aug forces geometric generalization without disrupting gradient directions. σ=0.05 is the sweet spot.

## Potential Next Research Directions (not yet assigned)

### Top Priority (researcher-agent, 2026-04-04, see RESEARCH_IDEAS_2026-04-04_TANJIRO.md)

1. **foil1-relative-coords** — Add 2 input features for aft-foil nodes: (x,y) relative to fore-foil trailing edge. Physically motivated: wake pressure is controlled by inter-foil jet geometry. ~20 LoC. **Expected p_tan -3% to -7%.** Synergistic with aft_srf head. HIGH PRIORITY.
2. **tandem-selfdistill** — Use EMA model as online teacher for tandem samples only (KD loss after epoch 40). EMA is already on device; adds second, cleaner training signal at near-zero cost. ~25 LoC. **Expected p_tan -2% to -5%.**
3. **foil-shape-ae (stats variant first)** — Inject 16 global shape statistics (mean/std/skew/kurtosis of each DSDF channel over surface nodes) as AdaLN conditioning. ~10 LoC for stats variant. Gives trunk a global shape fingerprint that may interpolate to OOD NACA6416. **Expected p_tan -2% to -6%.**
4. **interfoil-channel-aug** — Morph mesh coordinates between tandem samples (not just scalar gap/stagger). Physically consistent augmentation unlike scalar gap/stagger perturb. ~35 LoC.
5. **surf-node-dropout** — Randomly drop surface nodes during training to force robust volume-to-surface info routing. ~15 LoC, low risk.

### Existing Queue

6. **Combined DSDF1+DSDF2 aug at lower σ** — Simultaneous aug of both foil DSDF channels at σ=0.02-0.03 each; may compound improvements
7. **Gap/stagger sigma reduction** (0.02→0.01) — reduce p_tan hurt while keeping p_oodc benefit
8. **EMA Stochastic Weight Perturbation** — flat-minima seeking; ~12 LoC
9. **Aft-Foil TV Loss** — chord-wise TV regularization on aft-foil predictions; wait for nezuko #2129 first

**Strategic theme:** The research is now attacking the core bottleneck (OOD shape generalization for val_tandem_transfer) from multiple angles: geometric augmentation (DSDF family), coordinate-frame inductive biases (foil1-relative-coords), and training signal quality (self-distillation). If DSDF1 aug succeeds, the next wave should focus on foil1-relative-coords and tandem-selfdistill as complementary, orthogonal levers.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Fore-Foil SRF (all formulations) | #2117, #2124 | Split: +9-11% p_tan. Stacked: +1-4% p_tan, p_oodc degrades. |
| Aft-Foil Loss Upweighting | #2121 | p_oodc improves -2% but p_tan regresses +1.5% |
| Fore-Foil Loss Upweighting | #2122 | p_oodc improves -1.8% but p_tan regresses +1.5-1.9% |
| Aft-Foil Coordinate Frame (dual-frame) | #2107 | 3 iterations, 8 runs, p_in +3%, p_tan barely moves |
| Boundary-ID One-Hot (sparse 3-dim input) | #2118 | p_tan +2.1%; sparse features disrupt slice assignment |
| Charbonnier Loss | #2116 | All eps values degrade all metrics |
| Mesh-Density Weighted L1 | #2112 | All metrics regressed 5–16% |
| Smooth L1 / Huber Loss | #2113 | Catastrophic |
| Gradient Centralization (GC-Lion) | #2114 | Incompatible with Lion sign operation |
| Reynolds Number Perturbation Augmentation | #2125 | Null result + p_tan regression. OOD-Re gap too small in log-space. |
| Langevin Gradient Noise (SGLD) | #2120 | No improvement; Lion already provides implicit gradient noise |
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

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Current phase focuses on geometric augmentation (DSDF family) and routing modifications.
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

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
