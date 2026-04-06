# SENPAI Research State

- **Date:** 2026-04-06 ~22:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2207, +TE Coordinate Frame, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **12.490** | < 12.49 |
| **p_oodc** | **7.618** | < 7.62 |
| **p_tan** | **28.521** | **< 28.52** |
| **p_re** | **6.411** | < 6.41 |

**Latest merge:** PR #2207 (edward) — TE Coordinate Frame: 6 new input channels (dx, dy, r from fore-foil and aft-foil TEs). Based on GeoMPNN (arXiv 2412.09399). Delivered striking -5.4% on p_in, -2.5% on p_oodc, -0.7% on p_re; p_tan flat (within noise). W&B: obn1wfja (s42, p_tan=28.641), 52irfwwg (s73, p_tan=28.400).

**Key note:** p_tan remains the hardest metric. Single model p_tan=28.52 still beats 16-seed ensemble (29.1). Focus is pushing p_tan below 28.0.

**Reproduce current baseline:**
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

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Single-model p_tan (28.52) already **BEATS** ensemble (29.1). Gap continues to widen.

## Student Status (~22:00 UTC 2026-04-06)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2210 | Arc-Length Surface Loss Reweighting | WIP |
| nezuko | #2205 | NOBLE Nonlinear Low-Rank Branches (Retry) | WIP |
| alphonse | #2211 | Surface Pressure Gradient Loss (dp/ds) | WIP |
| thorfinn | #2209 | Attention Register Tokens | WIP |
| tanjiro | #2197 | Geometry-Adaptive Curvature Loss Weighting | WIP |
| askeladd | #2212 | Analytical Cp Delta (thin-airfoil SRF) | WIP |
| frieren | #2213 | Wake Deficit Feature (gap-normalized fore-TE offset) | WIP (just assigned) |
| edward | #2214 | Deep Supervision on fx_deep intermediate rep | WIP (just assigned) |

**All 8 students active. Zero idle GPUs.**

## Human Research Directives (from GitHub Issues)

- **Issue #1860 (still open):** "Think bigger — too many incremental tweaks." Standing directive: prioritize radical approaches. Transolver absorbs most convergence-dynamic modifications; focus on changes that alter the prediction task or add genuinely new physical information.
- **Issue #1834:** Only use assigned training split data. External datasets permitted if not from TandemFoilSet validation.

## Current Research Themes

### Geometric/Physical Feature Encoding (ACTIVE — high momentum)
TE coordinate frame (PR #2207, merged) showed +5.4% p_in improvement. The hypothesis that explicit geometric reference frames help OOD generalization is confirmed. Follow-ons:
- **#2213 frieren:** Wake deficit feature (gap-normalized dx/dy from fore-TE). Physics: aft-foil p_tan governed by where each node sits in the fore-foil wake, normalized by gap.
- The TE frame direction has been the most productive recently; building on it is the right near-term strategy.

### Gradient Quality / Training Dynamics (ACTIVE — medium confidence)
- **#2214 edward:** Deep supervision on fx_deep intermediate rep (weight 0.12). Gives earlier blocks direct surface pressure signal — reduces gradient path length.
- **#2210 fern:** Arc-length surface loss reweighting (fixes non-uniform mesh density bias).
- **#2211 alphonse:** Surface pressure gradient loss (penalize dp/ds mismatch).

### Architecture Novelty (ACTIVE — mixed results so far)
- **#2209 thorfinn:** Attention register tokens (learnable global slots prevent OOD routing collapse).
- **#2205 nezuko:** NOBLE nonlinear low-rank FFN branches (retry).
- **#2212 askeladd:** Analytical Cp delta (thin-airfoil physics prior as SRF correction baseline).
- **#2197 tanjiro:** Curvature loss weighting.

## Dead Ends (Do Not Revisit)

- Muon/Gram-NS optimizer (#2203) — catastrophic regression, destroys physics gradient geometry
- Ada-Temp/Rep-Slice from Transolver++ (#2206) — three temperature mechanisms fight each other
- Spectral attention conditioning SCA (#2199) — attention spectral collapse is NOT the bottleneck; runs crashed
- Iterative SRF/RAFT-style heads (#2208, #2165) — iterations drift without new info per pass
- Fore-aft cross-attention as SRF REPLACEMENT (#2202) — instability; additive version still viable
- Laplacian PE (#2190) — random eigenvector sign ambiguity; spectral graph theory doesn't transfer OOD

## Potential Next Hypotheses (Round 14 — from researcher-agent)

1. **pirate-residuals** — PirateNets gated adaptive residuals `x = (1-tanh(s))*x + tanh(s)*f(x)`, s init 0. LOW risk, ~10 LoC.
2. **mhc-residuals** — Learnable alpha/beta per block for residual blend. MEDIUM-LOW risk.
3. **tandem-feature-cross** — Sigmoid gate on input features conditioned on (gap, stagger, Re). MEDIUM risk.
4. **fore-srf-additive-skip** — Fore-foil mean surface hidden state as zero-init additive skip into AftSRF. MEDIUM risk.
5. Additional geometry encoding variants building on TE frame success.
