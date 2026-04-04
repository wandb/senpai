# SENPAI Research State

- **Date:** 2026-04-04 ~23:45 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2127, +aft_foil_srf_context K=8, 2-seed evidence)

| Metric | 2-seed avg | vs prior (DSDF2 aug) |
|--------|------------|----------------------|
| p_in | **13.02** | -0.2% |
| p_oodc | **7.62** | -0.5% |
| **p_tan** | **29.91** | **-0.7%** |
| p_re | **6.47** | -1.0% |

**Latest merge:** PR #2127 (frieren) — AftSRF KNN Volume Context (K=8). W&B: zosxwjmm (s42, p_tan=29.96), twilqf1x (s73, p_tan=29.87). All 4 metrics beat the prior baseline. Note: run WITHOUT --aug_dsdf2_sigma 0.05.

⚠️ **2-seed only.** Target for merge decisions: p_tan < 29.91, p_oodc < 7.62, p_in < 13.02, p_re < 6.47. VRAM: 69-95GB per seed (dedicated H100 required).

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-aft-srf-ctx" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --disable_pcgrad --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aft_foil_srf_context
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~22:15 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2133 | Foil-1 DSDF Magnitude Aug — Front Foil Shape Transfer | WIP |
| fern | #2130 | Gap/Stagger-Conditioned Spatial Bias — Tandem-Geometry-Aware Slice Routing | WIP |
| alphonse | #2131 | Tandem-Slice Carve-Out — Reserved Physics Slices for Tandem Samples | WIP |
| nezuko | #2129 | Supervised Surface Pressure Gradient Aux Loss — **v2** (per-foil fix + aft_srf_context rebase) | WIP — sent back for revision |
| askeladd | #2119 | PCGrad 2-Way — rebased 2-seed validation (p_tan -1.9% in 8-seed) | WIP — sent back for rebase |
| thorfinn | #2132 | Tandem DSDF Channel Mixup — Synthetic Foil-Shape Interpolation | WIP |
| frieren | #2134 | Fore-Foil TE Relative Coords — Inter-Foil Jet Frame for AftSRF Context | WIP — just assigned |
| edward | #2135 | Tandem Self-Distillation — EMA Teacher for Tandem Samples | WIP — just assigned |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-04 ~23:30)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2129 | nezuko | Supervised Surface Pressure Gradient Aux Loss | **SENT BACK** | All metrics regress vs current baseline. Cross-foil gradient bug identified. Sent back for per-foil fix + aft_srf_context rebase. |
| #2127 | frieren | Context-Aware AftSRF — KNN Volume Context K=8 | **MERGED** | All 4 metrics beat baseline. p_tan 30.11→29.91 (-0.7%), p_oodc -0.5%, p_re -1.0%. W&B verified. |
| #2128 | edward | Reynolds-Conditional SRF — FiLM on (Re, AoA) | **CLOSED** | Null result. FiLM worse than own control on p_oodc (+4%) and p_tan (+1.2%). AdaLN already handles Re/AoA conditioning. |
| #2126 | tanjiro | Foil-2 DSDF Magnitude Augmentation — σ=0.05 | **MERGED** (prior round) | p_tan -1.4%, p_in -1.5%, p_oodc -0.9%. |
| #2125 | thorfinn | Reynolds Number Perturbation Aug | **CLOSED** (prior round) | Null result. |
| #2124 | fern | Fore-Foil Stacked SRF Head | **CLOSED** (prior round) | p_oodc degrades all formulations. |

## Current Research Focus

### Primary target: p_tan = 29.91 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head (ID=7) — PR #2104
2. `--aug_gap_stagger_sigma 0.02`: tandem scalar domain randomization — PR #2115
3. `--aug_dsdf2_sigma 0.05`: foil-2 DSDF magnitude aug — PR #2126 (p_tan -1.4%)
4. `--aft_foil_srf_context`: KNN volume context for aft-foil SRF head — PR #2127 (p_tan -0.7%)
5. `--surface_refine`, `--residual_prediction`, `--pressure_first`, `--pressure_deep`, `--asinh_pressure 0.75`, etc.

**Active experiments (8 students):**
1. **Foil-1 DSDF Magnitude Aug** (tanjiro #2133) — Mirror of DSDF2 aug for front foil. σ sweep {0.05, 0.10, 0.15}. **HIGH PRIORITY** — if foil-1 DSDF also benefits from magnitude aug, the gain may be additive with DSDF2 aug.
2. **Gap/Stagger-Conditioned Spatial Bias** (fern #2130) — Extend spatial_bias 4→6 dims by appending gap+stagger; tandem-geometry-aware slice routing.
3. **Tandem-Slice Carve-Out** (alphonse #2131) — Reserve K dedicated physics slices for tandem via large negative bias on single-foil. K sweep {4, 8}.
4. **Supervised Surface Pressure Gradient Aux Loss** (nezuko #2129) — L1 on chord-wise pressure gradient finite differences; w=0.05/0.10 sweep.
5. **Tandem DSDF Channel Mixup** (thorfinn #2132) — interpolate DSDF channels between tandem training samples; creates synthetic intermediate foil geometries. α sweep {0.7, 0.5}.
6. **PCGrad 2-Way Validation** (askeladd #2119) — 8-seed validation of 2-way PCGrad; prior 4-seed showed p_oodc -1.3% to -5.4%.
7. **Fore-Foil TE Relative Coords** (frieren #2134) — Add (x,y) relative to fore-foil TE as extra features to AftFoilRefinementContextHead. Complementary to KNN context: KNN = what wake carries, rel_coords = where aft-foil sits in wake. Expected p_tan -3% to -7%.
8. **Tandem Self-Distillation** (edward #2135) — EMA model as soft teacher for tandem samples after epoch ema_start_epoch. KD loss (w=0.05/0.10/0.20 sweep). Expected p_tan -2% to -5%.

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 confirmed, foil-1 being tested), additive specialized correction heads (aft_srf), non-local context (KNN wake), target transforms (asinh)
- **What doesn't work:** Fore-foil SRF (all formulations), explicit Re/AoA FiLM on SRF (redundant with AdaLN), Re perturbation aug, Charbonnier/Smooth-L1 loss families
- **Critical interaction:** gap_stagger aug + aft_foil_srf NOT additive alone — helps p_oodc but hurts p_tan by +1.6%
- **Emerging mechanism:** The aft-foil correction head benefits from richer context — volume neighbors (KNN) and now relative coordinate frame (foil1-relative-coords). Physical inductive biases around wake geometry are productive.

## Potential Next Research Directions (not yet assigned)

### Top Priority (researcher-agent, 2026-04-04, see RESEARCH_IDEAS_2026-04-04_22:00.md)

1. **per_foil_pnorm** — Per-foil physics normalization: for tandem samples, split Cp normalization so fore-foil nodes use q_fore and aft-foil nodes use q_aft. Corrects a physical bias baked into all Cp computations since training began. ~25 LoC. **Expected p_tan -2% to -5%. ASSIGN NEXT.**
2. **foil2_aoa_rot_aug** — Independent AoA rotation aug for aft-foil nodes in tandem samples only. Creates novel (fore_AoA, aft_AoA) combinations absent from training data. ~35 LoC. Expected p_tan -2% to -4%.
3. **ema_perturb** — EMA stochastic weight perturbation (σ sweep {5e-4, 1e-3, 3e-3}) at ema_start_epoch to probe flat minima. ~12 LoC. Expected p_tan -1% to -3%.
4. **aft_foil_tv_loss** — Chord-wise TV regularization on aft-foil pressure predictions. ⚠️ DEPRIORITIZED — nezuko #2129 (gradient aux, same family) showed weak results. May revisit if per-foil fix in #2129 v2 shows promise.
5. **Combined DSDF1+DSDF2 aug at lower σ** — Simultaneous aug of both foil DSDF channels at σ=0.02-0.03; may compound. Depends on tanjiro #2133 outcome.

### Existing Queue

6. **Upstream-only KNN context** (frieren suggested) — Filter vol neighbors to x_vol < x_aft for wake-specific context. Follow-up to PR #2127.
7. **Learnable distance weighting** — Replace mean K-neighbor aggregation with attention-weighted (distance-aware). Upgrade to PR #2127 mechanism.
8. **Gap/stagger sigma reduction** (0.02→0.01) — reduce p_tan hurt while keeping p_oodc benefit.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Reynolds-Conditional SRF FiLM | #2128 | Null — AdaLN already handles Re/AoA; FiLM redundant |
| Fore-Foil SRF (all formulations) | #2117, #2124 | Split: +9-11% p_tan. Stacked: +1-4% p_tan, p_oodc degrades |
| Aft-Foil Loss Upweighting | #2121 | p_oodc improves -2% but p_tan regresses +1.5% |
| Fore-Foil Loss Upweighting | #2122 | p_oodc improves -1.8% but p_tan regresses +1.5-1.9% |
| Aft-Foil Coordinate Frame (dual-frame) | #2107 | 3 iterations, 8 runs, p_in +3%, p_tan barely moves |
| Boundary-ID One-Hot (sparse 3-dim input) | #2118 | p_tan +2.1%; sparse features disrupt slice assignment |
| Charbonnier Loss | #2116 | All eps values degrade all metrics |
| Mesh-Density Weighted L1 | #2112 | All metrics regressed 5–16% |
| Smooth L1 / Huber Loss | #2113 | Catastrophic |
| Gradient Centralization (GC-Lion) | #2114 | Incompatible with Lion sign operation |
| Reynolds Number Perturbation Augmentation | #2125 | Null result + p_tan regression |
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

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Responded with Phase 5 report. Current phase focuses on geometric augmentation (DSDF family) and physical inductive biases (KNN context, relative coordinates).
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
