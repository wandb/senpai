# Baseline Metrics

## Current Single-Model Baseline (Phase 6 — 2026-04-08, +Re-Stratified Sampling, 2-Seed Evidence, PR #2290)

| Metric | 2-seed avg | vs prior (Cosine T_max=150) | Δ |
|--------|------------|----------------------------|---|
| **p_in** | **11.742** | 11.891 | **-1.3%** ✅ |
| p_oodc | 7.643 | 7.561 | +1.1% (minor regression) |
| **p_tan** | **27.874** | 28.118 | **-0.9%** ✅ |
| p_re | 6.419 | 6.364 | +0.9% (minor regression) |

**PR #2290** (merged 2026-04-08) — Re-Stratified Sampling: 2× weight for top/bottom 20th percentile of log-Re training samples via WeightedRandomSampler (530/1322 = 40.1% of samples upweighted). Weights multiplied with existing domain-balanced sampler. Improves p_in -1.3% and p_tan -0.8%; minor p_oodc and p_re regressions (+1.2%, +0.6%) within noise range. Net aggregate surface MAE improved by -0.24 points (-0.45%). W&B runs: k5qwvce4 (seed 42, p_in=11.60, p_tan=27.7), 7oa5xfhi (seed 73, p_in=11.88, p_tan=28.1).

⚠️ **2-seed only.** For merge decisions: **p_in < 11.74**, p_oodc < 7.64, **p_tan < 27.87**, p_re < 6.42.

---

## Prior Single-Model Baseline (Phase 6 — 2026-04-08, +Cosine T_max=150, 2-Seed Evidence, PR #2251)

| Metric | 2-seed avg | vs prior (Wake Deficit) | Δ |
|--------|------------|-------------------------|---|
| **p_in** | **11.891** | 11.979 | **-0.7%** |
| **p_oodc** | **7.561** | 7.643 | **-1.1%** |
| **p_tan** | **28.118** | 28.341 | **-0.8%** |
| p_re | 6.364 | 6.300 | +1.0% (regression) |

**PR #2251** (merged 2026-04-08) — Cosine T_max=150: changes `--cosine_T_max` from 160 to 150. Training consistently ends at ~149 epochs (30-min timeout), so T_max=150 ensures cosine annealing completes exactly at the training cutoff — giving the model maximum time at moderate LR for generalization while still reaching near-minimum LR at the end. The sweet spot between T_max=140 (too aggressive: helps p_in/p_oodc but hurts p_tan) and T_max=160 (schedule never completes). Delivers -0.7% p_in, -1.1% p_oodc, -0.8% p_tan. p_re regresses +1.0% (cost of tighter convergence on Reynolds generalization). W&B runs: 7jix2jkg (seed 42, p_tan=27.816, p_in=12.019), epkfhxfl (seed 73, p_tan=28.421, p_in=11.763).

⚠️ **2-seed only.** For merge decisions: **p_in < 11.89**, **p_oodc < 7.56**, **p_tan < 28.12**, **p_re < 6.36**.

---

## Prior Single-Model Baseline (Phase 6 — 2026-04-06, +Wake Deficit Feature, 2-Seed Evidence, PR #2213)

| Metric | 2-seed avg | vs prior (TE coord frame) | Δ |
|--------|------------|--------------------------|---|
| **p_in** | **11.979** | 12.490 | **-4.1%** |
| p_oodc | 7.643 | 7.618 | +0.3% (noise) |
| **p_tan** | **28.341** | 28.521 | **-0.6%** |
| **p_re** | **6.300** | 6.411 | **-1.7%** |

**PR #2213** (merged 2026-04-06) — Wake Deficit Feature: adds 2 gap-normalized fore-TE offset channels (dx/gap, dy/gap) encoding each node's dimensionless wake-relative position. Gap normalization makes the geometry invariant across tandem configurations — where the model previously had to infer wake depth indirectly, it now receives it explicitly. Builds on the TE coordinate frame from PR #2207 and refactors TE computation into a shared helper. Zeroed for single-foil samples. Delivers a striking -4.1% improvement on p_in and meaningful improvements on p_tan and p_re. p_oodc is +0.3% (within run-to-run noise). W&B runs: hgml7i2r (seed 42, p_tan=28.733, p_in=11.641), qic03vrg (seed 73, p_tan=27.949, p_in=12.316).

⚠️ **2-seed only.** For merge decisions: **p_in < 11.98**, p_oodc < 7.65, **p_tan < 28.34**, **p_re < 6.30**.

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-wake-deficit" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame --wake_deficit_feature
```

**For merge decisions:** Compare 2-seed avg against: p_in < 11.98, p_oodc < 7.65, p_tan < 28.34, p_re < 6.30.

---

## Prior Single-Model Baseline (Phase 6 — 2026-04-06, +TE Coordinate Frame, 2-Seed Evidence, PR #2207)

| Metric | 2-seed avg | vs prior (DCT freq loss) | Δ |
|--------|------------|--------------------------|---|
| **p_in** | **12.490** | 13.205 | **-5.4%** |
| **p_oodc** | **7.618** | 7.816 | **-2.5%** |
| p_tan | **28.521** | 28.502 | +0.1% (noise) |
| **p_re** | **6.411** | 6.453 | **-0.7%** |

**PR #2207** (merged 2026-04-06) — TE Coordinate Frame: adds 6 new input channels representing trailing-edge-relative coordinates for both the fore-foil TE and aft-foil TE. For each node, computes signed offsets (dx, dy) and Euclidean distance from each foil's trailing edge (max-x surface node). Aft-foil TE features are zeroed for single-foil samples. Based on GeoMPNN (arXiv:2412.09399, NeurIPS 2024 ML4CFD Best Student Paper). Delivers a striking -5.4% improvement on p_in and meaningful improvements on p_oodc and p_re. p_tan is essentially flat (within 2-seed noise). W&B runs: obn1wfja (seed 42, p_tan=28.641, p_in=12.708), 52irfwwg (seed 73, p_tan=28.400, p_in=12.271). Runs completed 148/200 epochs — model still improving at timeout.

⚠️ **2-seed only.** For merge decisions: **p_tan < 28.52**, p_oodc < 7.62, p_in < 12.49, p_re < 6.41.

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-te-coord" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame
```

**For merge decisions:** Compare 2-seed avg against: p_tan < 28.52, p_oodc < 7.62, p_in < 12.49, p_re < 6.41.

---

## Prior Single-Model Baseline (Phase 6 — 2026-04-06, +DCT Frequency-Weighted Loss w=0.05, 2-Seed Evidence, PR #2184)

| Metric | 2-seed avg | vs prior (GSB+PCGrad) | Δ |
|--------|------------|------------------------|---|
| p_in | **13.205** | 13.05 | +1.2% (slight regression) |
| p_oodc | **7.816** | 7.70 | +1.5% (slight regression) |
| **p_tan** | **28.502** | 28.60 | **-0.3%** |
| p_re | **6.453** | 6.55 | **-1.5%** |

**PR #2184** (merged 2026-04-06) — DCT Frequency-Weighted Auxiliary Loss: applies an rfft-based frequency-weighted loss on ordered surface pressure nodes (w_k = 1 + 2.0*(k/N)^1.5) as an auxiliary term with weight 0.05, exploiting spectral bias theory to force attention to high-frequency leading-edge and trailing-edge features. Unlike failed BSP (#2172) which used relative spectral error with DFT, this uses absolute DCT coefficient difference with smooth polynomial weighting — numerically stable. W&B runs: 6yfv5lio (seed 42, p_tan=28.432), etepxvjc (seed 73, p_tan=28.572).

⚠️ **2-seed only.** For merge decisions: **p_tan < 28.50**, p_oodc < 7.82, p_in < 13.21, p_re < 6.45.

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-dct-freq" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5
```

**For merge decisions:** Compare 2-seed avg against: p_tan < 28.50, p_oodc < 7.82, p_in < 13.21, p_re < 6.45.

---

## Prior Single-Model Baseline (Phase 6 — 2026-04-05, +Gap/Stagger Spatial Bias + PCGrad, 2-Seed Evidence, PR #2130)

| Metric | 2-seed avg | vs prior (PCGrad only) | Δ |
|--------|------------|------------------------|---|
| p_in | **13.05** | 13.20 | **-1.1%** |
| p_oodc | **7.70** | 7.91 | **-2.7%** |
| **p_tan** | **28.60** | 29.48 | **-3.0%** |
| p_re | **6.55** | 6.50 | +0.8% (noise) |

**PR #2130** (merged 2026-04-05) — Gap/Stagger-Conditioned Spatial Bias: extends Transolver spatial_bias MLP from 4→6 inputs by appending (gap, stagger) scalars from feature indices 22:24. Makes slice routing tandem-geometry-aware. Zero-init on new input columns ensures identical routing at epoch 0. W&B runs: d7l91p0x (seed 42, p_tan=28.9), j9btfx09 (seed 73, p_tan=28.3). Combined with PCGrad 2-way — the two mechanisms are orthogonal (routing vs gradient surgery) and compound.

⚠️ **2-seed only.** For merge decisions: p_tan < 28.60, p_oodc < 7.70, p_in < 13.05, p_re < 6.55.

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-gsb-pcgrad" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias
```

**For merge decisions:** Compare 2-seed avg against: p_tan < 28.60, p_oodc < 7.70, p_in < 13.05, p_re < 6.55.

---

## Prior Baseline (Phase 6 — 2026-04-05, +PCGrad 2-Way, 10-Seed Evidence, PR #2119)

| Metric | 8-seed mean | Rebased 2-seed | vs prior (aft_srf only, 8-seed) | Δ |
|--------|-------------|----------------|----------------------------------|---|
| p_in | **13.20 ± 0.26** | **12.92** | 13.19 | **-0.1%** |
| p_oodc | **7.91 ± 0.16** | **7.94** | 7.92 | flat |
| **p_tan** | **29.48 ± 0.43** | **29.47** | 30.05 | **-1.9%** |
| p_re | **6.50 ± 0.12** | **6.52** | 6.45 | +0.8% (noise) |

**PR #2119** (merged 2026-04-05) — PCGrad 2-Way: gradient surgery between single-foil and all-tandem batches. Prevents tandem tasks from stealing single-foil gradient capacity. 10 seeds total (8-seed: tmqq1xlo,84aff7cq,23y1pfj5,yw2djp6d,afis6090,c80t1a69,xcmpfkqs,75d4hhzm; rebased 2-seed: jpe1t13t,cdccuyl7). VRAM ~45-46 GB per run (+22% from 3 backward passes).

⚠️ **NOTE:** `--aft_foil_srf_context` flag is currently a **NO-OP** due to a guard bug discovered by frieren (#2134). The actual context head code in `AftFoilRefinementContextHead` was never applied. Fix pending. For merge decisions: use **p_tan < 29.48** (8-seed mean), **p_oodc < 7.91**, **p_in < 13.20**, **p_re < 6.50**.

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-pcgrad2w" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --disable_pcgrad --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02
```

**For merge decisions:** Compare 2-seed avg against: p_tan < 29.48, p_oodc < 7.91, p_in < 13.20, p_re < 6.50.

---

## Prior Baseline (Phase 6 — 2026-04-04, +aft_foil_srf_context K=8 [BUGGY — no-op], 2-Seed Evidence, PR #2127)

| Metric | 2-seed avg | vs prior (DSDF2 aug) | Δ |
|--------|------------|----------------------|---|
| p_in | **13.02** | 13.04 | **-0.2%** |
| p_oodc | **7.62** | 7.66 | **-0.5%** |
| **p_tan** | **29.91** | 30.11 | **-0.7%** |
| p_re | **6.47** | 6.52 | **-1.0%** |

**PR #2127** (merged 2026-04-04) — Context-Aware AftSRF: KNN volume context (K=8 nearest zone-2 volume neighbors) for the aft-foil SRF correction head. Gives the correction MLP direct access to upstream wake hidden states, physically motivated by fore→aft pressure dependency. Note: run WITHOUT `--aug_dsdf2_sigma 0.05` — improvement is independent of DSDF2 aug. W&B runs: zosxwjmm (seed 42, p_tan=29.96), twilqf1x (seed 73, p_tan=29.87).

⚠️ **2-seed only** — statistically solid directional signal (both seeds below baseline). VRAM usage 69-95GB peak per run (dedicated H100 required per seed). Per-epoch overhead ~25% vs baseline. For merge decisions: use p_tan < 29.91, p_oodc < 7.62, p_in < 13.02, p_re < 6.47.

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-aft-srf-ctx" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --disable_pcgrad --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aft_foil_srf_context
```

**For merge decisions:** Compare 2-seed avg against: p_tan < 29.91, p_oodc < 7.62, p_in < 13.02, p_re < 6.47.

---

## Prior Single-Model Baseline (Phase 6 — 2026-04-04, +aft_foil_srf +aug_gap_stagger +aug_dsdf2_sigma=0.05, 2-Seed Evidence, PR #2126)

| Metric | 2-seed avg (σ=0.05) | vs prior combined baseline | Δ |
|--------|---------------------|---------------------------|---|
| p_in | **13.04** | 13.24 | **-1.5%** |
| p_oodc | **7.66** | 7.73 | **-0.9%** |
| **p_tan** | **30.11** | 30.53 | **-1.4%** |
| p_re | 6.52 | 6.50 | +0.3% (noise) |

**PR #2126** (merged 2026-04-04) — Foil-2 DSDF Magnitude Augmentation (σ=0.05). Log-normal scaling of foil-2 DSDF channels (x[:,6:10], tandem samples only) before standardization forces shape-transfer generalization. W&B runs: hcc2q68t (seed 42, p_tan=29.76), e9cri4mt (seed 73, p_tan=30.46).

⚠️ **2-seed only** — Requires 8-seed validation for statistical confidence. Best single run: p_in=13.11, p_oodc=7.70, **p_tan=29.76**, p_re=6.42 (seed 42). For merge decisions: use p_tan < 30.11, p_oodc < 7.66, p_in < 13.04 as targets.

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-dsdf2-aug" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --disable_pcgrad --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05
```

**For merge decisions:** Compare 2-seed avg against: p_tan < 30.11, p_oodc < 7.66, p_in < 13.04, p_re < 6.52.

---

## Prior Single-Model Baseline (Phase 6 — 2026-04-04, +aft_foil_srf +aug_gap_stagger, 8-Seed Combined Validation, Seeds 42-49)

| Metric | Combined 8-seed mean | vs aft_srf-only (PR #2104) | vs Asinh-only |
|--------|---------------------|--------------------------|---------------|
| p_in | **13.24 ± 0.33** | +0.4% | +1.6% |
| p_oodc | **7.73 ± 0.22** | **-2.4%** | -1.3% |
| p_tan | **30.53 ± 0.50** | +1.6% | +0.8% |
| p_re | **6.50 ± 0.07** | +0.8% | +0.8% |

**PR #2123** (validated 2026-04-04) — Combined 8-seed validation of aft_foil_srf + aug_gap_stagger_sigma=0.02. W&B group: `phase6/combined-baseline-8seed`. Run IDs: zapen0x3, 3vuz3adi, g1uhcorj, ea551p6b, fdf1vsi3, al6opl9g, jg15oow3, vzm3s42y.

⚠️ **Key finding:** Gap/stagger augmentation (PR #2115) is NOT additive with aft_foil_srf (PR #2104). The combination helps p_oodc (-2.4%) but **regresses p_tan (+1.6%)**. This suggests gap/stagger noise during training may slightly disrupt the aft-foil SRF head's ability to learn tandem wake corrections.

**Component baselines for reference:**
- aft_foil_srf only (PR #2104, 8-seed): p_in=13.19, p_oodc=7.92, **p_tan=30.05**, p_re=6.45
- gap/stagger only (PR #2115, 2-seed, no aft_srf): p_in≈13.04, p_oodc=7.45, p_tan=30.21, p_re=6.35

**Reproduce (current baseline):**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-aft-srf-aug" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --disable_pcgrad --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02
```

**For merge decisions:** Compare 2-seed avg against combined 8-seed mean targets: p_tan < 30.53, p_oodc < 7.73, p_in < 13.24, p_re < 6.50.

---

## Current Ensemble Baseline (Phase 6 — 2026-04-04, 16-Seed Ensemble from Seeds 42-49 + 66-73)

| Metric | 16-Ensemble | vs 8-Ensemble (66-73) | vs Asinh-only |
|--------|------------|----------------------|---------------|
| p_in | **12.1** | **-0.8%** | -7.2% |
| p_oodc | **6.6** | **-1.5%** | **-15.7%** |
| p_tan | **29.1** | 0% | -3.9% |
| p_re | **5.8** | 0% | **-10.1%** |

**PR #2093** — 16-seed ensemble: re-trained seeds 42-49 (run IDs: f59v5aul, 0yurebjv, rdezx8es, ds12ug79, yu1x0dy0, y147zvh1, lc5cbt4l, 7cxu38oh) + original seeds 66-73 (run IDs: j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv). Beats 8-seed baseline on p_in (-0.8%) and p_oodc (-1.5%), consistent with 1/√N variance-reduction scaling. Additionally, seeds 100-106 are trained and available for future 23-seed ensemble expansion (run IDs: 9o85duyc, ec7plfg8, zagg4pfs, 6w86plz1, g00kxdva, jt9hwf40, fom4bzro).

**Reproduce:**
```bash
python eval_ensemble.py \
  --run_ids f59v5aul 0yurebjv rdezx8es ds12ug79 yu1x0dy0 y147zvh1 lc5cbt4l 7cxu38oh \
            j9w7d1r7 mc4jvgqj cbbvhl62 bigqfn3k bqhg6lq8 5ukk7wv6 xlnhwuqc ii1tz4vv \
  --asinh_scale 0.75
```

⚠️ **Note:** Requires 16x inference cost. Each model uses 38GB VRAM. Run serially or on 16 GPUs in parallel.

---

## Prior Baseline (Phase 6 — 2026-04-03, 8-Seed Ensemble from Seeds 66-73)

| Metric | 8-Ensemble (66-73) | vs prior 8-ensemble (42-49) | vs Asinh-only |
|--------|-------------------|-----------------------------|---------------|
| p_in | **12.2** | **-1.6%** | -6.4% |
| p_oodc | **6.7** | 0% | **-14.4%** |
| p_tan | **29.1** | **-1.0%** | -3.9% |
| p_re | **5.8** | 0% | **-10.1%** |

**PR #2080** — 8-seed ensemble from fresh seeds 66-73 (standard 3L config with asinh s=0.75). Slightly outperforms the original 8-seed ensemble (seeds 42-49) on p_in and p_tan. W&B run IDs: j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv

**Reproduce:**
```bash
# Train 8 seeds
for seed in 66 67 68 69 70 71 72 73; do
  python train.py --agent <name> --wandb_name "<name>/ensemble-seed-${seed}" \
    --wandb_group phase6/ensemble-more-seeds \
    --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
    --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
    --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
    --cosine_T_max 160 --disable_pcgrad --pressure_first --pressure_deep \
    --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
    --seed ${seed} &
done
wait

# Evaluate ensemble
python eval_ensemble.py \
  --run_ids j9w7d1r7 mc4jvgqj cbbvhl62 bigqfn3k bqhg6lq8 5ukk7wv6 xlnhwuqc ii1tz4vv \
  --asinh_scale 0.75
```

⚠️ **Note:** Requires 8x inference cost. Each model uses 38GB VRAM. Can run serially or on 8 GPUs in parallel.

---

## Prior Baseline (Phase 6 — 2026-04-03, 8-Seed Prediction Ensemble, Seeds 42-49)

| Metric | 8-Ensemble | vs Asinh-only baseline |
|--------|-----------|----------------------|
| p_in | **12.4** | -4.8% |
| p_oodc | **6.7** | **-14.4%** |
| p_tan | **29.4** | -2.9% |
| p_re | **5.8** | **-10.1%** |

**PR #2076** — Post-hoc 8-seed ensemble. W&B run IDs: rboyvjeo, h0uog211, kwt8tw52, 5j26p5v1, rmump7ke, ujt9cu0l, 7fw8ksxq, 0lsry8km.

---

## Prior Baseline (Phase 6 — 2026-04-03, Asinh Pressure s=0.75)

| Metric | 8-seed Mean | Std | Best | Improvement |
|--------|------------|-----|------|-------------|
| p_in | **13.03** | 0.39 | 12.2 | ~0% vs prior |
| p_oodc | **7.83** | 0.19 | 7.6 | **-6.1%** |
| p_tan | **30.29** | 0.47 | 29.8 | ~0% vs prior |
| p_re | **6.45** | 0.05 | 6.4 | **-3.9%** |

**PR #2054** — `--asinh_pressure --asinh_scale 0.75`. Validated 8 seeds. W&B group: `phase6/asinh-075-multiseed`.

**Reproduce:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "baseline" \
  --asinh_pressure --asinh_scale 0.75 \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 160 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3
```

---

## Prior Baseline (Phase 6 — 2026-04-03, T_max=160, 8-seed characterization)

| Metric | 8-seed Mean | Std | Best | Worst | 95% CI (±1.96σ) |
|--------|------------|-----|------|-------|------------------|
| val/loss | **0.3829** | 0.003 | 0.377 | 0.392 | ±0.006 |
| p_in | **12.99** | 0.22 | 12.7 | 13.4 | ±0.43 |
| p_oodc | **8.34** | 0.26 | 8.0 | 8.8 | ±0.51 |
| p_tan | **30.13** | 0.45 | 29.5 | 30.9 | ±0.88 |
| p_re | **6.71** | 0.08 | 6.6 | 6.9 | ±0.16 |

**Significance thresholds** (must beat 8-seed mean - 2σ for confident improvement):
- p_in < 12.56, p_oodc < 7.83, p_tan < 29.25, p_re < 6.55

**PR #2003** — cosine_T_max 180→160. W&B group: `phase6/baseline-8seed` (PR #2052).

**Note:** The original single-seed (PR #2003, val/loss=0.3761, p_in=12.5) was a lucky seed. Use the 8-seed mean as the TRUE baseline for merge decisions.

**Reproduce:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 160 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3
```

---

## Previous Baseline (Phase 5 — 2026-03-29, Residual Prediction + Surface Refinement)

| Metric | Mean (8 seeds) | Std | Best single | Worst single |
|--------|---------------|-----|-------------|--------------|
| val/loss | 0.383 | 0.003 | 0.378 | 0.389 |
| p_in | 12.95 | 0.30 | 12.1 | 13.4 |
| p_oodc | 8.31 | 0.18 | 8.1 | 8.7 |
| p_tan | 30.01 | 0.52 | 29.2 | 31.0 |
| p_re | 6.70 | 0.10 | 6.5 | 6.8 |

**Phase 5 improvements merged:**
1. residual_prediction (#1927) — predict correction to freestream. p_oodc -4.7%, p_tan -1.9%.
2. surface_refine (#1935) — dedicated surface-only refinement MLP. p_re -72.7%, p_tan -8.9%, val/loss -3.3%. Manually verified (no target leakage).

Memory: ~38.0 GB. W&B group: `phase5/surface-refine-8seed`.

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 180 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-29 | 0.383±0.003 | 12.95±0.30 | 8.31±0.18 | 30.01±0.52 | 6.70±0.10 | #1935 | **Phase 5: + surface_refine** (p_re -72.7%, p_tan -8.9%) |
| 2026-03-29 | 0.396±0.003 | 12.93±0.22 | 7.98±0.19 | 32.93±0.29 | 24.53±0.08 | #1927 | **Phase 5: + residual_prediction** (p_oodc -4.7%, p_tan -1.9%) |
| 2026-03-28 | 0.404±0.004 | 13.33±0.58 | 8.37±0.22 | 33.57±0.44 | 24.58±0.13 | #1911 | 8-seed characterization |
| 2026-03-28 | 0.401±0.005 | 12.95±0.3 | 8.40±0.4 | 33.8±0.5 | 24.7±0.2 | #1867 | **Phase 4: + pressure_first + pressure_deep** (p_in -4.8%) |
| 2026-03-27 | 0.403±0.003 | 13.6±0.4 | 8.6±0.1 | 33.1±0.6 | 24.7±0.1 | #1846 | Phase 4: + disable_pcgrad (18% memory reduction) |
| 2026-03-27 | 0.4016±0.001 | 13.3±0.2 | 8.3±0.2 | 33.1±0.4 | 24.7±0.2 | #1845 | Phase 4: cosine_T_max=180 (4-seed validated) |
| 2026-03-27 | 0.405±0.004 | 13.6±0.5 | 8.7±0.3 | 33.5±0.6 | 24.7±0.2 | #1836 | Previous baseline (wd=5e-5 ~no-op) |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
