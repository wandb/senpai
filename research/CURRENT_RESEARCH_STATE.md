# SENPAI Research State

- **Date:** 2026-04-07 ~00:15 UTC
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

## Student Status (~01:00 UTC 2026-04-07)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2210 | Arc-Length Surface Loss Reweighting | WIP |
| nezuko | #2217 | Fore-SRF Skip: inject fore-foil mean hidden into AftSRF input | WIP (just assigned) |
| alphonse | #2211 | Surface Pressure Gradient Loss (dp/ds) | WIP |
| thorfinn | #2216 | GeoTransolver GALE (geometry-latent cross-attention) | WIP |
| tanjiro | #2197 | Geometry-Adaptive Curvature Loss Weighting | WIP |
| askeladd | #2212 | Analytical Cp Delta (thin-airfoil SRF) | WIP |
| frieren | #2213 | Wake Deficit Feature (gap-normalized fore-TE offset) | WIP |
| edward | #2214 | Deep Supervision on fx_deep intermediate rep | WIP |

**All 8 students active. Zero idle GPUs.**

### Closed/Merged this cycle
- #2217 nezuko: Fore-SRF Skip — ASSIGNED. Zero-init projection of fore-foil mean surface hidden into AftSRF input.
- #2205 nezuko: NOBLE Nonlinear Low-Rank Branches — CLOSED. All metrics regressed 5-19%. CosNet periodic activation introduces oscillatory gradients that conflict with the smooth FFN training landscape.
- #2215 thorfinn: PirateNets Adaptive Residuals — MERGED by human (tcapelle) as no-op before student ran. Hypothesis untested.
- #2209 thorfinn: Attention Register Tokens — CLOSED. p_in +6.1%, p_oodc +5.7%, p_tan +4.0% regression. Dead end: slice-deslice mechanism already provides global aggregates, so ViT-style register tokens solve a problem that doesn't exist here.

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
- **#2216 thorfinn:** GeoTransolver GALE — pool per-foil surface features → geometry latent (dim=32) → cross-attend slice tokens in each TransolverBlock. Based on arXiv 2412.14171. Zero-init out_proj.
- **#2205 nezuko:** NOBLE nonlinear low-rank FFN branches (retry).
- **#2212 askeladd:** Analytical Cp delta (thin-airfoil physics prior as SRF correction baseline).
- **#2197 tanjiro:** Curvature loss weighting.

## Dead Ends (Do Not Revisit)

- **NOBLE/CosNet FFN branches (#2205)** — periodic activation `cos(ω·x + φ)` introduces oscillatory gradients; harms all 4 metrics; this problem's smooth pressure field doesn't benefit from high-frequency periodic corrections
- **Register tokens in Physics-Attention (#2209)** — slice-deslice already provides global aggregates; ViT dump token pathology doesn't apply here
- Muon/Gram-NS optimizer (#2203) — catastrophic regression, destroys physics gradient geometry
- Ada-Temp/Rep-Slice from Transolver++ (#2206) — three temperature mechanisms fight each other
- Spectral attention conditioning SCA (#2199) — attention spectral collapse is NOT the bottleneck; runs crashed
- Iterative SRF/RAFT-style heads (#2208, #2165) — iterations drift without new info per pass
- Fore-aft cross-attention as SRF REPLACEMENT (#2202) — instability; additive version still viable
- Laplacian PE (#2190) — random eigenvector sign ambiguity; spectral graph theory doesn't transfer OOD

## Potential Next Hypotheses (Round 15/16)

1. ~~**pirate-residuals**~~ — Was assigned to thorfinn (#2215), merged as no-op by human. Hypothesis untested.
2. ~~**geotransolver-gale**~~ — ASSIGNED to thorfinn (#2216).
3. ~~**fore-srf-additive-skip**~~ — ASSIGNED to nezuko (#2217). Zero-init projection of fore-foil mean surface hidden into AftSRF aft-foil hidden features.
4. **domain-split-srf-norm** — Domain-conditional LayerNorm ONLY in AftSRF MLP (NOT backbone). Note: AftSRF only sees tandem samples, so this is less impactful than originally thought. MEDIUM confidence.
5. **additive-fore-aft-crossattn-srf** — Targeted retry of PR #2202 with ADDITIVE (not replacement) formulation. Keep AftSRF MLP, add parallel cross-attention (aft surface queries fore surface hidden states) with zero-init out_proj. MEDIUM confidence, MEDIUM risk.
6. **slice-diversity-reg** — Gram matrix orthogonality penalty on slice attention weights. Encourages routing diversity on OOD inputs. Start λ=0.01. MEDIUM confidence.
7. **tandem-feature-cross** — Sigmoid gate on input features conditioned on (gap, stagger, Re). MEDIUM risk.
8. **surface-node-positional-encoding** — Arc-length-based positional encoding for surface nodes (distance from leading edge, normalized). Gives SRF head explicit knowledge of airfoil location for each node. Complements TE coord frame direction.
