# SENPAI Research State

- **Date:** 2026-04-10 12:30 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold Round 40

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

## Student Status (2026-04-10 12:30 UTC)

### Round 40 Bold Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| edward | #2362 | **Viscous Residual Prediction** | Prediction reformulation | WIP — implementing |
| fern | #2363 | **Global Cl/Cd SRF Conditioning** | Architecture (two-pass) | WIP — implementing |
| alphonse | #2364 | **DPOT Pretrained Backbone** | Transfer learning | WIP — implementing |
| frieren | #2365 | **FFD Geometry Augmentation** | Data generation | WIP — implementing |
| tanjiro | #2366 | **MoE Domain-Expert FFN** | Architecture (routing) | WIP — implementing |
| askeladd | #2367 | **Biot-Savart Attention Bias** | Physics-informed attention | WIP — implementing |

### Round 39 (still finishing)
| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2351 | **Log-Re-Conditioned Panel Cp** | WIP — awaiting results |
| nezuko | #2359 | **Spectral Regularization** | WIP — awaiting results |

## This Session's Results

### Merged
- **PR #2357 (askeladd):** Vortex-Panel Induced Velocity — p_tan **-3.2%**, p_re **-2.7%**, p_in -0.2%. Biot-Savart physics oracle at every mesh node.

### Closed (Round 39 incremental experiments)
- **PR #2356 (edward):** Joukowski Camber Cp — all worse (+2.3% p_in). A0 correction too small.
- **PR #2340 (fern):** Cl/Cd Aux Loss v2 — p_tan -2.4% but p_in +2.9%. Integral loss disrupts shared weights.
- **PR #2341 (alphonse):** Hypernetwork SRF v4 — only 2/4 beat new baseline. Wake angle subsumes the signal.
- **PR #2360 (frieren):** R-Drop Consistency — catastrophic +54% p_in. 2× forward pass incompatible with timeout.
- **PR #2358 (tanjiro):** SRF Normal Frame — p_oodc +4.1%. Redundant with TE coord frame.

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."

**Status:** ACTIONED. 6 of 8 students now on bold Round 40 experiments covering:
- Prediction reformulation (viscous residual)
- Transfer learning (DPOT pretrained backbone)
- Data generation (FFD geometry augmentation)
- Physics-informed attention (Biot-Savart attention bias)
- Architecture (MoE domain routing, Cl/Cd two-pass conditioning)

## Key Insights (Updated)

1. **Physics-informed input features remain the strongest lever.** Vortex-panel velocity merged with p_tan -3.2%, continuing the pattern of wake deficit, wake angle, panel Cp. The model thrives on explicitly computed physics as input.
2. **The physics feature era is now exhausted.** Panel Cp, wake deficit, wake angle, vortex-panel velocity — all merged. Log-Re Cp, Joukowski Cp, surface normals, pressure recovery all failed. No more surface-level physics features to add.
3. **Transfer learning, data augmentation, and prediction reformulation are COMPLETELY UNEXPLORED.** These are the biggest gaps in 1966+ experiments.
4. **2× forward pass approaches are incompatible** with the 30-min training timeout. R-Drop/consistency regularization cannot work in this setting.
5. **Integral loss terms (Cl/Cd) structurally conflict with per-node MAE** via PCGrad. The information is useful but must be delivered as conditioning, not loss.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, diffusion decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg
Physics features: Panel Cp ✅, wake deficit ✅, wake angle ✅, vortex-panel ✅ (all MERGED). Pressure recovery, learnable Cp, arc-length, log-Re Cp, Joukowski Cp (all failed).

## Next Research Priorities

When Round 40 results come in, remaining unassigned bold ideas for Round 41:
1. **LoRA from Pretrained Operator** (run alongside DPOT — frozen backbone + LoRA adapters)
2. **Surface-Only Boundary GNN** (radical architecture: graph Transformer on surface nodes only)
3. **DSDF-Weighted Physics Features** (distance-to-surface adaptive feature scaling)
4. **Multiphysics Pretraining** (The Well dataset)
