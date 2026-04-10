# SENPAI Research State
- **Date:** 2026-04-10 16:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold Round 40/41/42

## Current Baseline

### Single-Model Baseline (PR #2357 Vortex-Panel Induced Velocity, 2-seed avg)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.872** | < 11.872 |
| p_oodc | 7.459 | < 7.459 |
| **p_tan** | **26.319** | < 26.319 |
| **p_re** | **6.229** | < 6.229 |

Reproduce:
```
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way --pressure_first --pressure_deep --residual_prediction --surface_refine --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64
```

## Student Status (2026-04-10 16:00 UTC)

### Round 41/42 Bold Experiments (all WIP)
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| edward | #2374 | **Hard Kutta TE Constraint** | Physics constraint | WIP — just assigned (replaces #2362 which failed) |
| fern | #2363 | **Global Cl/Cd SRF Conditioning** | Architecture (two-pass) | WIP — rebasing onto latest noam |
| alphonse | #2368 | **Sobolev Surface Gradient Loss** | Loss formulation (dp/ds) | WIP — implementing |
| thorfinn | #2369 | **Cross-Foil Autoregressive Decoding** | Architecture (causal AR) | WIP |
| nezuko | #2370 | **Surface-Intrinsic B-GNN** | Architecture (pure surface GNN) | WIP |
| frieren | #2371 | **1D Surface FNO Decoder** | Architecture (spectral surface) | WIP |
| askeladd | #2372 | **Surface-Node Cross-Attention** | Architecture (global surface attn) | WIP |
| tanjiro | #2373 | **Multi-Scale Slice Attention** | Architecture (coarse+fine) | WIP |

### Idle students
None — all 8 GPUs occupied.

### Recently closed (this session)
- **#2362 (edward, Viscous Residual Prediction):** CLOSED. All metrics catastrophic (+22% p_oodc). Flat-plate panel Cp too inaccurate to serve as residual baseline — residuals are MORE variable, not less. Viscous residual approach is exhausted.

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."

**Status:** FULLY ACTIONED. Round 41 is 100% architectural and physics-constraint experiments. Researcher-agent generating Round 42 bold ideas now (background).

## Key Insights (Updated)

1. **Physics-informed input features are now fully exhausted.** Panel Cp, wake deficit, wake angle, vortex-panel velocity — all merged and compounding. This was the primary lever for Rounds 30-40. New territory: constraints, loss formulation, architecture.
2. **Residual prediction (viscous correction) fails.** Panel Cp is too inaccurate for OOD/tandem — subtracting it makes the target HARDER, not easier. Two-pass (predict-then-refine) is incompatible with 30-min training timeout. This direction is closed.
3. **2× forward pass approaches are incompatible** with the 30-min training timeout. Must use stop-gradient or single-pass for any multi-pass approaches.
4. **High seed variance = unstable training signal.** When s42 and s73 diverge >5% on p_in, the approach is learning something spurious.
5. **p_tan = 26.319 is our hardest metric** (2× worse than p_in). The tandem inter-foil interference is the dominant unsolved problem. All Round 41/42 experiments should prioritize p_tan improvement.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem projection, GeoTransolver GALE, MoE-LoRA FFN, Biot-Savart attention bias
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF
Physics features: Panel Cp ✅, wake deficit ✅, wake angle ✅, vortex-panel ✅ (all MERGED). Log-Re Cp, Joukowski Cp, viscous residual baseline (all failed)
Transfer learning: DPOT external checkpoint (AFNO incompatible), self-pretraining denoising (no inductive bias)
Data augmentation: FFD geometry augmentation (label mismatch fatal), re-scaling augmentation

## Next Research Priorities (Round 42+)

Researcher-agent running now (2026-04-10 16:00 UTC) — see `RESEARCH_IDEAS_2026-04-10_ROUND42.md` when complete.

**Pre-identified Tier 1 candidates from Round 42 ideas list:**
1. **Panel Method as Cheap Data Oracle** — generate 5000+ synthetic tandem configs using panel solver as auxiliary training data (self-consistent labels). Infrastructure exists (`--cp_panel`). AIAA 2024 shows 10²× improvement from inviscid pretraining.
2. **Mamba/SSM Surface Sequence Model** — replace SRF attention with Mamba SSM on arc-length-ordered surface nodes. O(n) cost enables 6-8 layer surface decoder. (NOTE: Mamba broke torch.compile in Phase 5 — student should test compile compatibility first and have LRU fallback.)
3. **Neural Process for Pressure Fields** — ANP with context = (surface_coord, cp_panel), predicts CFD pressure as conditional distribution. Novel: no prior CFD application. Targets p_oodc OOD generalization directly.

**Tier 2:**
4. Koopman Operator Lifting (linearize tandem interference in latent space)
5. SE(2)-Equivariant Architecture (orientation-invariant features)
6. Contrastive Geometry Pretraining (UIUC+NACA airfoil library)
