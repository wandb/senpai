# SENPAI Research State

- **Date:** 2026-04-10 12:30 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training & Architecture Improvements

## Current Baseline

### Single-Model Baseline (PR #2350 Wake Angle Feature, 2-seed avg)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.90** | < 11.90 |
| **p_oodc** | **7.35** | < 7.35 |
| **p_tan** | **27.20** | < 27.20 |
| **p_re** | **6.40** | < 6.40 |

Reproduce:
```
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way --pressure_first --pressure_deep --residual_prediction --surface_refine --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature
```

## Student Status (2026-04-10 12:30 UTC)

| Student | PR | Experiment | Status | Notes |
|---------|-----|-----------|--------|-------|
| alphonse | #2341 v4 | **Hypernetwork SRF v2 + wake_angle** | WIP | v2 beat 3/4 metrics, v4 validates on new baseline |
| thorfinn | #2351 | **Log-Re-Conditioned Panel Cp** | WIP | Targets p_re regression from Panel Cp |
| edward | #2356 | **Joukowski Camber-Corrected Cp** | WIP | Geometry-aware panel physics |
| fern | #2340 v2 | **Cl/Cd Auxiliary Loss (tandem-only)** | WIP | p_tan -2.9% in v1, v2 adds tandem-only + wake_angle |
| askeladd | #2357 | **Vortex-Panel Induced Velocity** | WIP | Per-node inviscid physics oracle |
| tanjiro | #2358 | **Surface-Normal SRF Frame** | WIP | Training started ~08:09 UTC |
| nezuko | #2359 | **Spectral Regularization** | WIP | λ sweep {1e-5, 1e-4, 1e-3} |
| frieren | #2360 | **Input Consistency Regularization** | WIP | R-Drop style dropout-robust predictions |

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Too many incremental tweaks — go for radical new full model changes, data aug, and data generation. THINK BIGGER."

**Status:** Acknowledged and actioned. Current WIP experiments are the FINAL incremental round. Round 40 will be dedicated entirely to bold, paradigm-level changes. Researcher-agent has generated 10 bold ideas (see `/research/RESEARCH_IDEAS_2026-04-10_BOLD.md`).

## Round 39 Results (PRs closed 2026-04-10)

| PR | Experiment | Result | Verdict |
|----|-----------|--------|---------|
| #2352 (tanjiro) | SRF FiLM Conditioning | p_oodc +2.7% vs new baseline | Closed |
| #2353 (nezuko) | Learnable Cp Scale | +6-42% all metrics | Closed |
| #2354 (askeladd) | Pressure Recovery Ratio | p_in +5.9%, p_tan flat | Closed |
| #2355 (frieren) | Two-Stage SRF | p_in +2.5%, p_oodc -1.9%, net negative | Closed |

## Most Promising Active Experiments

1. **Hypernetwork SRF v4** (alphonse #2341): v2 config (rank=2, α=0.5) + wake_angle_feature. v2 beat 3/4 metrics (p_oodc -2.7%, p_re -2.8%, p_tan -1.0%). Merge candidate if v4 confirms on new baseline.
2. **Cl/Cd Auxiliary Loss v2** (fern #2340): p_tan -2.9% in v1. v2 applies tandem-only + rebased on wake_angle baseline. Strong tandem signal.

## Round 40 Plan — BOLD EXPERIMENTS

Per Morgan's directive, Round 40 will assign ONLY paradigm-level changes. No more physics feature tweaks.

### Top Priority Assignments (first 5 idle students)

| Priority | Hypothesis | Type | Target |
|----------|-----------|------|--------|
| 1 | **Viscous Residual Prediction** — predict delta_p = p_CFD - p_panel instead of raw pressure | Prediction reformulation | p_tan, p_in |
| 2 | **DPOT Pretraining** — load pretrained Transolver backbone, discriminative LR fine-tune | Transfer learning | p_re, p_oodc |
| 3 | **LoRA from Pretrained Operator** — frozen DPOT backbone + rank-8 LoRA adapters | Transfer learning | p_re, p_oodc |
| 4 | **Biot-Savart Cross-Foil Attention Bias** — physics-derived attention structure for tandem coupling | Architecture + physics | p_tan, p_oodc |
| 5 | **Surface-Only Boundary GNN** — replace Transolver+SRF with graph Transformer on surface nodes | Radical architecture | p_in, p_oodc |

### Secondary Assignments (students 6-8)

| Priority | Hypothesis | Type | Target |
|----------|-----------|------|--------|
| 6 | **Global Cl/Cd SRF Conditioning** — two-pass: predict → integrate Cl/Cd → condition SRF | Architecture | p_tan, p_in |
| 7 | **MoE Domain-Expert FFN** — deterministic tandem/single routing with LoRA-style delta experts | Architecture | p_tan |
| 8 | **FFD Geometry Augmentation** — Free-Form Deformation + NeuralFoil synthetic data | Data generation | p_in, p_oodc |

Full details: `/research/RESEARCH_IDEAS_2026-04-10_BOLD.md`

## Key Insights (Updated)

1. **Physics-informed input features are exhausted.** Panel Cp, wake deficit, wake angle, TE coord frame — all merged. Every further feature tweak has been neutral or worse. The feature mining era is over.
2. **The backbone is a strong local optimum.** All 6 alternative architectures tried (GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN) were dramatically worse. BUT these were trained from scratch — **pretrained** backbones (DPOT) have not been tried.
3. **Prediction reformulation is the highest-impact untried lever.** Viscous residual prediction changes the fundamental learning problem. Panel Cp is already in the model — using it as the prediction baseline is a zero-cost reformulation.
4. **Transfer learning is completely unexplored.** No pretrained checkpoints have been used in any of 1966+ experiments. This is the largest gap in our experimental coverage.
5. **Data augmentation beyond simple noise has never been tried.** FFD geometry augmentation with NeuralFoil Cp is a principled approach to expanding the training distribution.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch):
- GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN — all dramatically worse
- FiLM SRF, two-stage SRF, GRU decoder, diffusion decoder, Gumbel MoE SRF
- Per-head KV attention, foil role embeddings, per-foil target whitening, SWA

Loss/training:
- Focal L1, curriculum, quantile regression, Jacobian smoothness, asymmetric surface loss
- Aug annealing, batch size tuning, spectral reg, R-Drop, attention logit noise

Physics features:
- Pressure recovery ratio, learnable Cp scale, arc-length encoding
- SRF FiLM conditioning, surface-normal SRF frame, vortex-panel velocity, log-Re Cp, Joukowski Cp
- Panel Cp, wake deficit, wake angle, TE coord frame (all MERGED)
