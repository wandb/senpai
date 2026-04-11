# SENPAI Research State
- **Date:** 2026-04-11 ~02:45 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Round 44/45 (post-ANP breakthrough)

## Current Baseline — UPDATED (PR #2379 ANP merged)

| Metric | 2-seed avg | vs prior (Vortex-Panel) | Δ |
|--------|------------|------------------------|---|
| **p_in** | **3.561** | 11.872 | **-70.0%** ✅ |
| **p_oodc** | **3.847** | 7.459 | **-48.4%** ✅ |
| **p_tan** | **10.825** | 26.319 | **-58.9%** ✅ |
| p_re | 7.232 | 6.229 | **+16.1%** ❌ |

Reproduce:
```
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way --pressure_first --pressure_deep --residual_prediction --surface_refine --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf
```

## Student Status (2026-04-11 ~02:45 UTC)

### Active Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| frieren | **#2390** | **ANP Re-Conditional Gate** | Architecture (ANP follow-up) | NEW — fix p_re regression via Re-proximity gate |
| alphonse | #2384 | **PirateNet SRF** | Architecture (gated residual) | Mid-training: s73 p_in=21, s42 p_in=19 (early, normal) |
| tanjiro | #2385 | **HyPINO Hypernetwork SRF** | Architecture (hypernetwork) | Mid-training, slower convergence than PirateNet |
| nezuko | #2386 | **Stagnation Point Constraint** | Physics constraint | Mid-training: underperforming PirateNet at same stage |
| fern | #2387 | **Bernoulli Velocity-Pressure Constraint** | Physics constraint | ⚠️ Ux MAE anomalously high (16-21 vs typical 4-7) — possible constraint bug |
| askeladd | #2388 | **Multi-Scale Hierarchical Slice Attention** | Architecture (cross-scale) | Implementing — no W&B runs yet |
| edward | #2374 | **Hard Kutta TE Constraint v2** | Physics constraint | Implementing v2 (K=2, tandem-only, lower weight) |
| thorfinn | #2389 | **Arc-Length Positional Encoding** | Architecture (PE) | Implementing — no W&B runs yet |

### Recent Merges / Closures
| PR | Student | Experiment | Result | Action |
|----|---------|-----------|--------|--------|
| ✅ **#2379** | frieren | **ANP Cross-Foil Decoder** | **p_tan -59%, p_in -70%, p_oodc -48%, p_re +16%** | **MERGED — new baseline** |

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Too many incremental tweaks — think bigger. Radical new model changes, data aug, data generation."
**Status:** Acknowledged and actioned. Round 44 has 4 architecture changes + 3 physics constraints. Researcher-agent running to generate Round 45 ideas focused on: (1) synthetic data generation, (2) geometry augmentation, (3) full model replacements. Next idle students will receive genuinely bold assignments, not incremental tweaks.

## Key Insights (Updated 2026-04-11 02:45)

1. **ANP cross-attention is the breakthrough.** Asymmetric cross-attention where aft-foil queries attend to all fore-foil context = direct inter-foil wake physics encoding. MERGED. p_tan -59%, p_in -70%, p_oodc -48%. Biggest single improvement in programme history.
2. **p_re is the current gap.** ANP regressed p_re +16% (7.232 vs old baseline 6.229). Cause: cross-attention aggregates fore-foil representations that are unreliable at OOD Reynolds numbers. Frieren now working on Re-conditional gate to fix this (#2390).
3. **Stop-gradient kills cross-attention.** SCA v3 with stop-gradient failed; ANP without stop-gradient succeeded. Bidirectional gradient flow is essential for cross-foil learning.
4. **Physics constraints are a viable secondary lever.** Kutta TE showed p_oodc -2.9%. Physics constraints + ANP combination could compound further gains.
5. **The new challenge:** With p_tan now at 10.825 (was 26.319), we need to ensure other metrics also drop. p_oodc at 3.847 and p_in at 3.561 are dramatically lower. The ANP has reset the scale of the problem.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, Gumbel MoE, per-head KV, foil role embed, SRF normal frame, decoupled tandem projection, GeoTransolver GALE, MoE-LoRA FFN, Biot-Savart attention bias, role-specialized SRF, surface cross-attention (stop-grad)
Architecture (decoder replacements): Surface B-GNN (+14%), 1D Surface FNO (+29%), Multi-Scale Slice (+8%), Koopman lifting (+19%)
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF, Cl/Cd SRF conditioning, tandem aux heads
Physics features (merged): Panel Cp, wake deficit, wake angle, vortex-panel velocity
Physics features (failed): Log-Re Cp, Joukowski Cp, viscous residual baseline
Multi-fidelity: Panel oracle data (+57%, dead end)
Transfer learning: DPOT, contrastive geometry pretrain, denoising pretraining (all 2-8× worse)
Data augmentation: FFD geometry (label mismatch), re-scaling
Optimizers: Muon/Gram-NS

## Next Research Priorities

### Immediate (Round 45 — post-ANP follow-ups)
1. **Frieren #2390: ANP Re-Conditional Gate** — fix p_re regression while preserving p_tan gains
2. **Researcher-agent results** (running) — will generate Round 45 bold ideas: synthetic data generation, geometry augmentation, full model changes

### When round 44 finishes (next idle students)
From researcher-agent RESEARCH_IDEAS (2026-04-11 run):
- **ANP depth/width sweep** — 8 heads × 384 dim (currently 4×48)
- **ANP + physics constraints** — combine ANP baseline with Kutta/stagnation/Bernoulli
- **Tandem Gap/Stagger Mixup Augmentation** — interpolate between configurations during training
- **Synthetic data generation** via panel method at 10× more configurations
- **Geometry augmentation** — AoA perturbation ±2°, chord scaling, symmetric flip
- **GeoMPNN Surface Graph** — fresh geometric graph approach
- **Bold ideas from researcher-agent** (pending — focused on data generation and full model changes)

### ⚠️ Watch: fern Bernoulli Constraint (#2387)
Ux surface MAE anomalously high (16-21 vs typical 4-7). Possible bug in Bernoulli constraint implementation affecting velocity predictions. Monitor — may need to close if not self-correcting.
