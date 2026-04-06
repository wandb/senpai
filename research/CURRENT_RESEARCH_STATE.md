# SENPAI Research State

- **Date:** 2026-04-07 ~07:30 UTC
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

## Student Status (~07:30 UTC 2026-04-07)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2210 | Arc-Length Surface Loss Reweighting | WIP |
| nezuko | #2217 | Fore-SRF Skip: inject fore-foil mean hidden into AftSRF input | WIP |
| alphonse | #2219 | Additive Fore→Aft Cross-Attention in AftSRF | WIP (just assigned) |
| thorfinn | #2216 | GeoTransolver GALE (geometry-latent cross-attention) | WIP |
| tanjiro | #2218 | LE Coordinate Frame: leading-edge-relative input features | WIP (just assigned) |
| askeladd | #2212 | Analytical Cp Delta (thin-airfoil SRF) | WIP |
| frieren | #2213 | Wake Deficit Feature (gap-normalized fore-TE offset) | WIP |
| edward | #2214 | Deep Supervision on fx_deep intermediate rep | WIP |

**All 8 students active. Zero idle GPUs.**

### Closed/Merged this cycle (2026-04-06 to 2026-04-07)
- #2218 tanjiro: LE Coordinate Frame — ASSIGNED. Leading-edge-relative input features (+6 channels), mirroring successful TE coord frame.
- #2219 alphonse: Fore→Aft Cross-Attention Additive — ASSIGNED. Per-node cross-attention from aft surface to fore surface in AftSRF. Zero-init, additive.
- #2197 tanjiro: Curvature Loss Weighting — CLOSED. 2-seed p_tan=29.00 vs 28.50 (+1.8% regression). Curvature proxy from standardized features is too noisy.
- #2211 alphonse: Surface Pressure Gradient Loss — CLOSED. 2-seed p_tan=29.25 vs 28.50 (+2.6% regression). x-coordinate-sorted dp/ds conflates upper/lower surface nodes at LE.
- #2217 nezuko: Fore-SRF Skip — ASSIGNED. Zero-init projection of fore-foil mean surface hidden into AftSRF input.
- #2205 nezuko: NOBLE Nonlinear Low-Rank Branches — CLOSED. All metrics regressed 5-19%. CosNet periodic activation introduces oscillatory gradients.
- #2215 thorfinn: PirateNets Adaptive Residuals — MERGED by human (tcapelle) as no-op. Hypothesis untested.
- #2209 thorfinn: Attention Register Tokens — CLOSED. p_tan +4.0% regression. Register tokens solve a problem that doesn't exist in Transolver.

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
- ~~**#2211 alphonse:** Surface pressure gradient loss~~ — CLOSED. x-sorted dp/ds fundamentally broken for closed airfoil geometry.

### Architecture Novelty (ACTIVE — mixed results so far)
- **#2216 thorfinn:** GeoTransolver GALE — pool per-foil surface features → geometry latent (dim=32) → cross-attend slice tokens in each TransolverBlock. Based on arXiv 2412.14171. Zero-init out_proj.
- **#2219 alphonse:** Fore→Aft Cross-Attention Additive — per-node cross-attention from aft surface to fore surface in AftSRF. Directly targets p_tan by giving aft nodes spatially-resolved fore-foil context. Complements #2217 (global mean) with fine-grained per-node attention.
- **#2217 nezuko:** Fore-SRF Skip — coarse global mean of fore surface hidden into AftSRF. A weaker version of #2219 to test whether fore-foil info helps at all.
- **#2212 askeladd:** Analytical Cp delta (thin-airfoil physics prior as SRF correction baseline).
- ~~**#2205 nezuko:** NOBLE/CosNet FFN branches~~ — CLOSED. Periodic activations interfere with smooth pressure landscape.
- ~~**#2197 tanjiro:** Curvature loss weighting~~ — CLOSED. Noisy curvature proxy from standardized features.

## Dead Ends (Do Not Revisit)

- **NOBLE/CosNet FFN branches (#2205)** — periodic activation `cos(ω·x + φ)` introduces oscillatory gradients; all metrics regressed 5-19%; smooth pressure field doesn't benefit from high-frequency corrections
- **Register tokens in Physics-Attention (#2209)** — slice-deslice already provides global aggregates; ViT dump token pathology doesn't apply here
- **Curvature loss weighting (#2197)** — curvature proxy from standardized features is too noisy; high seed variance; conceptually sound but implementation pathway unstable
- **Surface pressure gradient loss (dp/ds, #2211)** — x-coordinate ordering conflates upper/lower surface nodes at LE; fundamental implementation flaw for closed airfoil geometry
- Muon/Gram-NS optimizer (#2203) — catastrophic regression, destroys physics gradient geometry
- Ada-Temp/Rep-Slice from Transolver++ (#2206) — three temperature mechanisms fight each other
- Spectral attention conditioning SCA (#2199) — attention spectral collapse is NOT the bottleneck; runs crashed
- Iterative SRF/RAFT-style heads (#2208, #2165) — iterations drift without new info per pass
- Fore-aft cross-attention as SRF REPLACEMENT (#2202) — instability; additive version still viable
- Laplacian PE (#2190) — random eigenvector sign ambiguity; spectral graph theory doesn't transfer OOD

## Potential Next Hypotheses (Round 16/17)

**Assigned this round:**
- ~~**le-coord-frame**~~ → tanjiro #2218
- ~~**additive-fore-aft-crossattn-srf**~~ → alphonse #2219
- ~~**geotransolver-gale**~~ → thorfinn #2216
- ~~**fore-srf-additive-skip**~~ → nezuko #2217

**Ready to assign (from researcher-agent, `/research/RESEARCH_IDEAS_2026-04-07_00:00.md`):**

| Priority | Slug | Idea | Confidence | Notes |
|----------|------|------|------------|-------|
| HIGH | mhc-residuals | Learnable alpha/beta on TransolverBlock residual connections (12 params, init (1,1)) | Medium | Human team requested (issue #1926). Ultra-simple. |
| HIGH | slice-diversity-reg | Gram matrix orthogonality penalty on slice attention weights (λ=0.005) | Medium | Loss-side OOD fix, no architecture change, zero inference cost. |
| MED-HIGH | domain-split-srf-norm | Conditional LayerNorm in AftSRF only (zero-init embeddings) | Medium | NOT same as dead-end #2164 (backbone domain AdaLN). |
| MED-HIGH | tandem-feature-cross | Sigmoid gate on input encodings from (gap, stagger, Re, AoA), bias=5.0 near-identity | Medium | Complementary to gap_stagger_spatial_bias (routing vs features). |
| MED-HIGH | surface-arc-length-pe | Arc-length from LE normalized by chord as input channel (+1 dim) | Medium | Complements TE coord frame. Related to #2218. |
| MEDIUM | panel-method-cp-input | Hess-Smith inviscid Cp as input feature (proposed as #1865, never ran) | Medium | Coordinate with askeladd #2212 (analytical Cp delta). |
| MEDIUM | chord-adaptive-le-te-loss | Extra loss weight on LE/TE regions (5% chord, boost 2x) | Medium | Wait for #2210 (fern) result first. |
| MEDIUM | kutta-condition-loss | Auxiliary loss: L2(p_upper_TE - p_lower_TE) for TE pressure continuity | Medium | Novel physics constraint. Check if TE is sharp or blunt. |
| MEDIUM | lift-drag-integral-loss | Cl/Cd integral loss from predicted surface pressure | Medium | Global physics consistency. |
| MEDIUM | geometry-moment-conditioning | 7-scalar shape moments (area, centroid, Ixx/Iyy/Ixy) as backbone conditioning | Medium | Wait for #2216 (GALE) result. |
| MEDIUM | node-type-boundary-embedding | Learned embedding per boundary_id, zero-init, added to input encoding | Medium | ~10 LoC, explicit node type awareness. |
