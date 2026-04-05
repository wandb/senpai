# SENPAI Research State

- **Date:** 2026-04-05 ~11:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2130, GSB + PCGrad, 2-seed)

| Metric | 2-seed avg | 2-seed target |
|--------|-----------|---------------|
| p_in | **13.05** | < 13.05 |
| p_oodc | **7.70** | < 7.70 |
| **p_tan** | **28.60** | **< 28.60** |
| p_re | **6.55** | < 6.55 |

**Latest merge:** PR #2130 (fern) — Gap/Stagger Spatial Bias + PCGrad compound. W&B: d7l91p0x (s42, p_tan=28.9), j9btfx09 (s73, p_tan=28.3). p_tan -3.0% from prior baseline.

⚠️ **RESOLVED BUG (PR #2134):** `--aft_foil_srf_context` guard bug confirmed. When bug fixed and context head actually fires, results are WORSE (p_tan +1.2%, p_in +21% due to KNN overhead → undertrained 132 epochs). Context head officially removed from baseline. PR #2127 improvement was seed variance.

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-gsb-pcgrad" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~11:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| askeladd | #2150 | DSDF2 Sigma Optimization: σ={0.03, 0.08} vs baseline 0.05 | WIP |
| tanjiro | #2155 | Slice Count Sweep: slice_num={64, 128} vs baseline 96 | WIP — just assigned |
| fern | #2151 | EMA Start Epoch Sweep: {100, 120} vs default ~140 | WIP |
| alphonse | #2131 | Tandem-Slice Carve-Out K=4 — corrected instructions sent | WIP — rebasing |
| nezuko | #2152 | Augmentation Annealing — linearly decay aug σ over training | WIP |
| thorfinn | #2154 | Cosine T_max Sweep: T_max={140, 180} vs baseline 160 | WIP |
| frieren | #2153 | Gap/Stagger Sigma Increase σ=0.03 — more geometric diversity | WIP |
| edward | #2149 | Learning Rate Sweep: lr={1e-4, 3e-4} vs baseline lr=2e-4 | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed (2026-04-05 ~11:00)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2148 | tanjiro | Gap/Stagger Aug Removal (σ=0) | **CLOSED** | All primary metrics worse: p_in +3.4%, p_tan +4.0%, p_oodc +1.3%. GSB and aug are complementary. σ parameter space fully explored. |
| #2147 | thorfinn | Actual 3-Way PCGrad | **CLOSED** | All pct values worse than 2-way. Tandem-extreme-Re carve-out too small. |
| #2146 | frieren | Tail EMA Checkpoint Averaging | **CLOSED** | Null result: ±0.3% noise. Post-hoc weight avg exhausted. |
| #2142 | frieren | Cross-Seed Model Soup | **CLOSED** | Catastrophic: MAE 50-400x worse due to loss barriers. |
| #2130 | fern | GSB + PCGrad Compound | **MERGED** | p_tan 29.48→28.60 (-3.0%). New baseline. |
| #2131 | alphonse | Tandem-Slice Carve-Out (K=4,8) | **SENT BACK** | K=4 beats control -3.7%. Corrected instructions sent for rebase. |

## Current Research Focus

### Primary target: p_tan = 28.60 → push below 28.0 (single model now beats 16-seed ensemble!)

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head — PR #2104
2. `--aug_gap_stagger_sigma 0.02`: tandem scalar domain randomization — PR #2115
3. `--aug_dsdf2_sigma 0.05`: foil-2 DSDF magnitude aug — PR #2126 (p_tan -1.4%)
4. ~~`--aft_foil_srf_context`~~: KNN volume context — **REMOVED** (was no-op due to guard bug; when fixed, harmful due to training slowdown) — PR #2127 retracted
5. `--pcgrad_3way --pcgrad_extreme_pct 0.15`: PCGrad 2-way gradient surgery — PR #2119 (p_tan -2.1%)
6. `--gap_stagger_spatial_bias`: tandem-geometry-aware slice routing (gap+stagger in spatial_bias MLP) — PR #2130 (p_tan -3.0%!)

**Active experiments (8 students WIP):**
1. **DSDF2 Sigma Optimization** (askeladd #2150) — σ={0.03, 0.08} vs baseline 0.05.
2. **EMA Start Epoch Sweep** (fern #2151) — {100, 120} vs default ~140.
3. **Augmentation Annealing** (nezuko #2152) — Linearly decay aug σ over training.
4. **Cosine T_max Sweep** (thorfinn #2154) — T_max={140, 180} vs baseline 160.
5. **Gap/Stagger Sigma Increase** (frieren #2153) — σ=0.03 vs baseline 0.02.
6. **Learning Rate Sweep** (edward #2149) — lr={1e-4, 3e-4} vs baseline 2e-4.
7. **Tandem-Slice Carve-Out K=4** (alphonse #2131) — Rebasing onto noam.
8. **Slice Count Sweep** (tanjiro #2155) — slice_num={64, 128} vs baseline 96. Tests interaction with GSB's geometry-aware routing.

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 only), additive specialized correction heads (aft_srf), gradient surgery (PCGrad 2-way), tandem-geometry-aware routing (GSB)
- **What doesn't work:** Foil-1 DSDF aug, fore-foil SRF, Re/AoA FiLM, self-distillation, DSDF mixup, surface gradient aux loss, flat-minima-seeking (SAM/SGLD/SWAD/EMA perturb all fail), post-hoc weight averaging (tail avg, model soup), 3-way PCGrad, gap/stagger σ reduction to 0 or 0.01
- **Confirmed optimal hyperparams:** ema_decay=0.999, weight_decay=5e-5, aug_gap_stagger_sigma=0.02
- **Gap/stagger σ sweep complete:** σ={0, 0.01, 0.02(best), 0.03(in-progress)}. σ=0.02 confirmed optimal.

## Critical Finding: PCGrad Flag Logic (2026-04-05 ~05:40)

⚠️ `--pcgrad_3way` in the baseline command is a **NO-OP** because `--disable_pcgrad` is not set. The code has an `if/elif` chain where 2-way PCGrad fires first (when `not cfg.disable_pcgrad`), making the `elif cfg.pcgrad_3way` branch unreachable. The code comment explicitly states: `pcgrad_3way: bool = False  # requires --disable_pcgrad`.

**Impact:** All baseline and experiment runs with `--pcgrad_3way --pcgrad_extreme_pct 0.15` are actually running **2-way PCGrad** (in-dist vs OOD). This is correct — 2-way was validated in PR #2119. But actual 3-way PCGrad (single-foil / tandem-normal / tandem-extreme-Re) has **never been tested**.

**Action:** The baseline is fine as-is (2-way PCGrad works). Testing actual 3-way PCGrad (with `--disable_pcgrad --pcgrad_3way --pcgrad_extreme_pct 0.15`) is a high-priority experiment — assign when a student becomes idle.

## Potential Next Research Directions (not yet assigned)

### High Priority
1. **Tandem carve-out K=4 + current baseline** — Alphonse #2131 showed -3.7% vs control. If it compounds with GSB+PCGrad, could be huge. In progress.
2. **Slice count sweep** — Currently 96 slices, never revisited since architecture expanded with GSB+aft_srf+PCGrad. Try 64 and 128.
3. **Attention sparsification / top-k slice assignment** — force sparser routing to prevent single slice domination.
4. **Fork-then-merge model soup** — Train 1 model to epoch 100, branch into 3 seeds for remaining epochs, then average. Solves loss barrier (shared initialization).
5. **Differential learning rates** — Different LR for backbone vs specialized heads (SRF, aft_srf, spatial_bias). Heads may benefit from higher LR.

### Longer-term
6. **Adversarial tandem augmentation** — gradient ascent on gap/stagger to find worst-case perturbations.
7. **Volume node subsampling** — Reduce vol_subsample_frac to 0.5-0.8 to shift compute budget toward surface.
8. **Extended tandem curriculum** — Increase tandem_curriculum_epochs from 10 to 30-50 to build stronger single-foil foundation.
9. **New ideas from researcher-agent** — awaiting deep literature review for paradigm-shifting approaches.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Foil-2 Independent AoA Rotation Aug | #2138 | Target inconsistency: rotated geometry but flow not re-simulated. No σ beats baseline |
| Per-Foil Physics Normalization | #2136 | phys_stats mismatch → all metrics +5-19%. Don't change normalization without recomputing stats |
| Fore-Foil TE Relative Coords | #2134 | p_tan +2.6% worse than control. TE relative frame doesn't help |
| AftSRF KNN Context Head (when working) | #2134, #2127 | Context head adds 17% overhead → 132 vs 160 epochs → catastrophic: p_in +21%, p_tan +1.2% |
| Surface Pressure Gradient Aux Loss | #2129 (3 rounds) | p_oodc improves -1.5% but p_tan regresses. 3 iterations with diminishing returns. Signal too weak. |
| Foil-1 DSDF Magnitude Augmentation | #2133 | All σ values regress p_tan. Front-foil is KNOWN component in val_tandem_transfer |
| Tandem DSDF Channel Mixup | #2132 | Mixup between NACA0012 samples adds no geometric diversity |
| Tail EMA Checkpoint Averaging | #2146 | Null result: ±0.3% noise. EMA already smooth; snapshot averaging redundant. Post-hoc weight avg exhausted class |
| EMA Decay 0.9995 + GSB | #2141 (Round 2) | All metrics regress. GSB obsoleted the 0.9995 advantage. ema_decay=0.999 confirmed optimal |
| Gap/Stagger σ=0 (removal) | #2148 | All metrics worse: p_in +3.4%, p_tan +4.0%, p_oodc +1.3%. GSB and aug complementary, not redundant |
| Gap/Stagger σ=0.01 | #2140 | Worse than σ=0.02 on p_oodc (+3.3%) and p_tan (+1.5%). σ=0.02 is well-calibrated |
| Weight Decay 1e-5, 2e-5 | #2145 | p_tan regresses +2.1-2.3%. wd=5e-5 confirmed optimal with Lion |
| Input Feature Noise Augmentation | #2144 | Catastrophic: p_in +17% at σ=0.01. Generic input perturbation incompatible with CFD mesh features |
| EMA Stochastic Weight Perturbation | #2137 | All σ values regress p_tan. Flat-minima-seeking confirmed DEAD CLASS (SAM, SGLD, SWAD, EMA perturb all fail) |
| DSDF Spatial Dropout | #2143 | All metrics degrade monotonically (p_in +6.4% at p=0.05). DSDF too information-dense to drop |
| Cross-Seed Model Soup (weight averaging) | #2142 | Catastrophic: MAE 50-400x worse. Loss barriers between independent basins. Only works with shared initialization |
| Tandem Self-Distillation (EMA teacher) | #2135 | Post-cosine degradation; w=0.05 regressed; GPU speed variance invalidates multi-seed |
| Actual 3-Way PCGrad | #2147 | All pct values (0.10, 0.15, 0.20) worse than 2-way. Tandem-extreme-Re carve-out too small for stable gradients |
| Reynolds-Conditional SRF FiLM | #2128 | Null — AdaLN already handles Re/AoA; FiLM redundant |
| Fore-Foil SRF (all formulations) | #2117, #2124 | Split: +9-11% p_tan. Stacked: +1-4% p_tan, p_oodc degrades |
| Aft-Foil Loss Upweighting | #2121 | p_oodc -2% but p_tan +1.5% |
| Fore-Foil Loss Upweighting | #2122 | p_oodc -1.8% but p_tan +1.5-1.9% |
| Aft-Foil Coordinate Frame (dual-frame) | #2107 | 3 iterations, 8 runs, p_in +3%, p_tan barely moves |
| Boundary-ID One-Hot | #2118 | p_tan +2.1%; sparse features disrupt slice assignment |
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

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Confirmed.

## Ensemble Seed Pool (Complete)

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
