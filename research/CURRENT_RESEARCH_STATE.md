# SENPAI Research State
- **Date:** 2026-04-10 14:45 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold Round 40/41

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

## Student Status (2026-04-10 14:45 UTC)

### Round 40 Bold Experiments (still WIP)
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| edward | #2362 | **Viscous Residual Prediction** | Prediction reformulation | WIP — implementing |
| fern | #2363 | **Global Cl/Cd SRF Conditioning** | Architecture (two-pass) | WIP — implementing |
| alphonse | #2368 | **Sobolev Surface Gradient Loss** | Loss formulation | WIP — implementing |
| ~~frieren~~ | ~~#2365~~ | ~~FFD Geometry Augmentation~~ | ~~Data generation~~ | CLOSED — p_in +9.4%, p_tan +6.2% |
| tanjiro | #2366 | **MoE Domain-Expert FFN** | Architecture (routing) | WIP — implementing |
| askeladd | #2367 | **Biot-Savart Attention Bias** | Physics-informed attention | WIP — implementing |

### Round 41 Bold Experiments (just assigned)
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| thorfinn | #2369 | **Cross-Foil Autoregressive Decoding** | Architecture (causal AR) | WIP — just assigned |
| nezuko | #2370 | **Surface-Intrinsic B-GNN** | Architecture (pure surface GNN) | WIP — just assigned |
| frieren | #2371 | **1D Surface FNO Decoder** | Architecture (spectral surface) | WIP — just assigned |

## This Session's Actions

### Closed (Round 39 stragglers)
- **PR #2351 (thorfinn):** Log-Re-Conditioned Panel Cp — 2-seed avg fails baseline on 3/4 metrics (p_in +1.7%, p_tan +0.5%, p_re +1.6%); only p_oodc -0.6%. High seed variance. Physics-feature era is exhausted.
- **PR #2359 (nezuko):** Spectral Regularization on FFN Weights — all configurations 10-20% worse across all metrics. Clear dead end; spectral norm restricts expressivity without physics-motivated benefit.
- **PR #2365 (frieren):** FFD Geometry Augmentation — p_in +9.4%, p_tan +6.2%. Label mismatch: deformed geometry with original pressure labels corrupts training.

### New Assignments
- **thorfinn #2369:** Cross-Foil Autoregressive Decoding — predict fore foil pressure first, condition aft foil decoding on fore predictions via cross-attention (stop-gradient). Zero-init output ensures identity at epoch 0. Targets p_tan (tandem causality).
- **nezuko #2370:** Surface-Intrinsic B-GNN — replace SRF head with a pure-PyTorch boundary GNN operating on surface-only k-NN connectivity (arc-length), implementing 4 rounds of message passing with edge features [dx, dy, dist, cp_panel_src, cp_panel_dst]. Based on Jena et al. arXiv:2503.18638.
- **frieren #2371:** 1D Surface FNO Decoder — replace SRF head with a 1D Fourier Neural Operator. Arc-length parameterized surface → uniform 128-point grid → spectral convolutions (16 modes, 4 layers) → interpolate back. Captures multi-scale Cp structure in frequency domain.

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."

**Status:** ACTIONED and sustained. All 8 students on genuinely bold experiments:
- Prediction reformulation (viscous residual)
- Data generation (FFD geometry augmentation)
- Physics-informed attention (Biot-Savart attention bias)
- Architecture (MoE domain routing, Cl/Cd two-pass conditioning, AR decoding, surface B-GNN)
- Loss formulation (Sobolev surface gradient loss)

## Key Insights (Updated)

1. **Physics-informed input features remain the strongest lever.** Panel Cp, wake deficit, wake angle, vortex-panel velocity — all merged. Physics feature era is now exhausted.
2. **Transfer learning from external checkpoints is eliminated.** DPOT uses AFNO (not Transolver), and self-pretraining on the same dataset provides no inductive bias. The backbone at ~1M params is too small for meaningful pretraining.
3. **2× forward pass approaches are incompatible** with the 30-min training timeout. Must use stop-gradient for any multi-pass approaches.
4. **Integral loss terms (Cl/Cd) structurally conflict with per-node MAE** via PCGrad. Must be delivered as conditioning, not loss.
5. **High seed variance = unstable training signal.** When s42 and s73 diverge >5% on p_in, the approach is learning something spurious, not a physics-grounded signal.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, diffusion decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem slice projection, GeoTransolver GALE
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF
Physics features: Panel Cp ✅, wake deficit ✅, wake angle ✅, vortex-panel ✅ (all MERGED). Log-Re Cp, Joukowski Cp, Viscous Residual Baseline Cp, arc-length input (all failed)
Transfer learning: DPOT external checkpoint (AFNO incompatible), self-pretraining denoising (no inductive bias)

## Next Research Priorities (Round 41+)

When Round 40 results come in, remaining bold ideas:
1. **Cross-Foil AR Decoding** — thorfinn #2369 ← IN PROGRESS
2. **Surface-Intrinsic B-GNN** — nezuko #2370 ← IN PROGRESS
3. **1D-FNO on arc-length surface curve** — interpolate surface nodes to uniform arc-length grid, apply 1D FNO, interpolate back
4. **Neural Process for pressure fields** — meta-learning approach: condition on context points (surface coords + panel Cp) to predict pressure at query points
5. **Contrastive geometry pretraining** — self-supervised pretraining with airfoil geometry augmentations (FFD, AoA) as positive pairs
6. **Koopman operator for surface dynamics** — lift surface pressure distribution to Koopman eigenfunction space for linear dynamics
