# SENPAI Research State

- **Date:** 2026-04-10 13:45 UTC
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

## Student Status (2026-04-10 13:45 UTC)

### Round 40 Bold Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| edward | #2362 | **Viscous Residual Prediction** | Prediction reformulation | WIP — implementing |
| fern | #2363 | **Global Cl/Cd SRF Conditioning** | Architecture (two-pass) | WIP — implementing |
| alphonse | #2368 | **Sobolev Surface Gradient Loss** | Loss formulation | WIP — just assigned |
| frieren | #2365 | **FFD Geometry Augmentation** | Data generation | WIP — implementing |
| tanjiro | #2366 | **MoE Domain-Expert FFN** | Architecture (routing) | WIP — implementing |
| askeladd | #2367 | **Biot-Savart Attention Bias** | Physics-informed attention | WIP — implementing |

### Round 39 (still finishing)
| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2351 | **Log-Re-Conditioned Panel Cp** | WIP — silent 11+ hrs (last update 02:36 UTC) |
| nezuko | #2359 | **Spectral Regularization** | WIP — silent 6+ hrs (last update 07:45 UTC) |

## This Session's Results

### Merged
- **PR #2357 (askeladd):** Vortex-Panel Induced Velocity — p_tan **-3.2%**, p_re **-2.7%**, p_in -0.2%. Biot-Savart physics oracle at every mesh node.

### Closed (Round 39 incremental experiments)
- **PR #2356 (edward):** Joukowski Camber Cp — all worse (+2.3% p_in). A0 correction too small.
- **PR #2340 (fern):** Cl/Cd Aux Loss v2 — p_tan -2.4% but p_in +2.9%. Integral loss disrupts shared weights.
- **PR #2341 (alphonse):** Hypernetwork SRF v4 — only 2/4 beat new baseline. Wake angle subsumes the signal.
- **PR #2360 (frieren):** R-Drop Consistency — catastrophic +54% p_in. 2× forward pass incompatible with timeout.
- **PR #2358 (tanjiro):** SRF Normal Frame — p_oodc +4.1%. Redundant with TE coord frame.
- **PR #2364 (alphonse):** DPOT/Self-Pretraining — all 36-40% worse. AFNO vs TransolverBlock incompatibility; self-pretraining on same dataset provides no inductive bias.

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."

**Status:** ACTIONED. All 8 students on bold Round 40 experiments:
- Prediction reformulation (viscous residual)
- Transfer learning → FAILED (DPOT incompatible) → replaced with Sobolev gradient loss
- Data generation (FFD geometry augmentation)
- Physics-informed attention (Biot-Savart attention bias)
- Architecture (MoE domain routing, Cl/Cd two-pass conditioning)
- Loss formulation (Sobolev surface gradient loss) ← NEW

## Key Insights (Updated)

1. **Physics-informed input features remain the strongest lever.** Panel Cp, wake deficit, wake angle, vortex-panel velocity — all merged. Physics feature era is now exhausted.
2. **Transfer learning from external checkpoints is eliminated.** DPOT uses AFNO (not Transolver), and self-pretraining on the same dataset provides no inductive bias. The backbone at ~1M params is too small for meaningful pretraining.
3. **2× forward pass approaches are incompatible** with the 30-min training timeout.
4. **Integral loss terms (Cl/Cd) structurally conflict with per-node MAE** via PCGrad. Must be delivered as conditioning, not loss.
5. **Bold Round 40 experiments are the right direction.** Prediction reformulation, data generation, physics-informed attention, and novel loss formulations have not been tried in 1966+ experiments.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, diffusion decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg
Physics features: Panel Cp ✅, wake deficit ✅, wake angle ✅, vortex-panel ✅ (all MERGED). Pressure recovery, learnable Cp, arc-length input, log-Re Cp, Joukowski Cp (all failed).
Transfer learning: DPOT external checkpoint (AFNO incompatible), self-pretraining denoising (no inductive bias)

## Next Research Priorities

When Round 40 results come in, remaining unassigned bold ideas for Round 41:
1. **Surface-Only Transolver / B-GNN** (process only surface nodes — metrics only care about surface)
2. **DSDF-Weighted Physics Features** (distance-to-surface adaptive feature scaling)
3. **Multiphysics Pretraining via The Well dataset** (if architectural incompatibility can be overcome)
4. **LoRA Adapters on Frozen Backbone** (requires solving AFNO vs TransolverBlock mismatch first)
5. **Cross-Foil Autoregressive Prediction** (predict fore foil first, condition aft foil on fore results)
