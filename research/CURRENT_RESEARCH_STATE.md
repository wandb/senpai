# SENPAI Research State
- **Date:** 2026-04-11 ~07:00 UTC (updated)
- **Advisor branch:** noam
- **Phase:** Phase 6 — Round 48 BOLD PIVOT

## CRITICAL: ANP Code Was Never Committed

The ANP surface decoder (PR #2379) achieved the biggest improvement in programme history (p_tan -59%, p_in -70%, p_oodc -48%) BUT **the code was never committed to the noam branch**. The merge commit (01cc741) was empty. The W&B runs (metqdxaq, jvfvrs4u) confirm `anp_srf=True` was used and the results are real, but the implementation only existed in frieren's local working directory and was lost on pod restart.

**Consequence:** The actual baseline is the PRE-ANP baseline (PR #2357). All Round 47 experiments targeting ANP's p_re regression have been closed (can't run without ANP code).

**Fix:** alphonse assigned #2399 to re-implement ANP from W&B config + experiment description.

## Current Baseline — PRE-ANP (PR #2357)

| Metric | Value |
|--------|-------|
| **p_in** | **11.872** |
| **p_oodc** | **7.459** |
| **p_tan** | **26.319** |
| **p_re** | **6.229** |

Reproduce:
```
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way --pressure_first --pressure_deep --residual_prediction --surface_refine --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64
```

## Human Researcher Directive (Issue #1860) — ACTIVE

**Morgan McGuire:** "THINK BIGGER. Radical new model changes, data aug, data generation."

**Status:** FULLY ACTIONED. Round 48 bold slate assigned — PIDA chord-scaling aug, flip-symmetry aug, flow matching head, multi-fidelity pretraining. Plus ANP re-implementation (P0 priority).

## Student Status (2026-04-11 ~07:00 UTC)

### Round 48 Active Experiments

| Student | PR | Experiment | Category | Status |
|---------|-----|-----------|----------|--------|
| **alphonse** | **#2399** | **ANP Re-implementation** | P0 Critical | 🆕 ASSIGNED |
| **edward** | **#2400** | **PIDA Chord-Scaling Aug** | Data augmentation (exact physics) | 🆕 ASSIGNED |
| **fern** | **#2401** | **Flip-Symmetry Aug** | Data augmentation (NeuralFoil-validated) | 🆕 ASSIGNED |
| **tanjiro** | **#2402** | **Flow Matching Generative Head** | New model family (rectified flow) | 🆕 ASSIGNED |
| **nezuko** | **#2403** | **Multi-Fidelity Panel Pretraining** | Synthetic data (100k panel solutions) | 🆕 ASSIGNED |
| frieren | #2393 | BL Adaptive Node Weighting | Loss reweighting (OB-GNN style) | 🔄 WIP (training) |
| askeladd | #2391 | Local Re_x BL Feature | Physics feature | 🔄 WIP |
| thorfinn | #2392 | DeltaPhi Residual Prediction | Target reformulation | 🔄 WIP (sent back: run without --anp_srf) |

### Round 47 Closures (ANP code missing — can't run)
| PR | Student | Reason |
|----|---------|--------|
| ❌ #2394 | alphonse | Cp-Space ANP Decoding — --anp_srf doesn't exist |
| ❌ #2395 | tanjiro | ANP Re-Conditional AdaLN — --anp_srf doesn't exist |
| ❌ #2396 | nezuko | ANP Context Dropout — --anp_srf doesn't exist |
| ❌ #2397 | fern | Stochastic Cross-Foil Attention — --anp_srf doesn't exist |
| ❌ #2398 | edward | Re Mixup in Cp Space — --anp_srf doesn't exist |
| ❌ #2374 | edward | Hard Kutta TE v2 — no v2 results, reassigned |

### Pre-ANP Experiment Results (this session)
| PR | Student | Result |
|----|---------|--------|
| ❌ #2387 | fern | Bernoulli — all metrics +10-39% worse (physics wrong at wall) |
| ❌ #2385 | tanjiro | HyPINO — null (LoRA zero-init bug) |
| ❌ #2386 | nezuko | Stagnation — all metrics +8-11% worse (fragile node ID) |

## Key Insights

1. **ANP concept is validated but code is lost.** W&B runs confirm massive improvements with anp_srf=True. Alphonse re-implementing as P0.
2. **Round 48 is BOLD per Morgan's directive.** All new experiments are: new model families (flow matching), synthetic data generation (panel pretraining), exact-physics augmentation (PIDA, flip-sym). Zero incremental tweaks.
3. **Physics constraints on walls are unreliable.** Bernoulli (wrong physics), Stagnation (fragile ID), Kutta (small gain). Surface constraints need wall-specific physics, not free-stream equations.
4. **Pre-ANP baseline is well-tuned.** p_in=11.872, p_oodc=7.459, p_tan=26.319, p_re=6.229 from extensive feature engineering.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, GRU decoder, Gumbel MoE, per-head KV, MoE-LoRA, Biot-Savart, Koopman, Surface B-GNN, 1D FNO, Multi-Scale Slice
Loss/training: Focal L1, curriculum, quantile reg, spectral reg, OHNM, R-Drop, attention noise
Physics constraints: Bernoulli (wall invalid), Stagnation (fragile ID), Kutta TE (small gain only)
Transfer learning: DPOT, contrastive pretrain, denoising pretrain (all 2-8x worse)
Data aug: FFD geometry (label mismatch)

## Next Research Priorities

### When ANP is re-implemented (#2399):
- Stack Round 48 bold ideas ON TOP of ANP baseline
- Re-run PIDA, flip-sym, flow matching with --anp_srf
- Then target p_re with ANP-specific fixes (Round 49)

### Ideas in Reserve
From `RESEARCH_IDEAS_2026-04-11_ROUND48_BOLD.md`:
- AeroDiT Diffusion Head with Re-CFG (Rank 3)
- Tandem Wake Superposition Prior (Rank 4)
- Displacement Thickness δ* Feature (Rank 6)
- Multi-Resolution ANP Context (Rank 7)
- GeoMPNN Bipartite MPNN (Rank 8)
- Divergence-Free Constraint (Rank 10)

### Key Flags for ALL experiments
- Do NOT use `--anp_srf` until #2399 is merged
- Pre-ANP baseline: p_in=11.872, p_oodc=7.459, p_tan=26.319, p_re=6.229
