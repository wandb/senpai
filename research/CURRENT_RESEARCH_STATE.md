# SENPAI Research State

- **Date:** 2026-04-10 09:30 UTC
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

## Student Status (2026-04-10 09:30 UTC)

| Student | PR | Experiment | Status | Notes |
|---------|-----|-----------|--------|-------|
| alphonse | #2341 v4 | **Hypernetwork SRF v2 + wake_angle** | WIP | Most promising — v2 beat 3/4, v4 adds wake angle |
| thorfinn | #2351 | **Log-Re-Conditioned Panel Cp** | WIP | Targets p_re regression from Panel Cp |
| edward | #2356 | **Joukowski Camber-Corrected Cp** | WIP | Geometry-aware panel physics |
| fern | #2340 v2 | **Cl/Cd Auxiliary Loss (tandem-only)** | WIP | p_tan -2.9% in v1, rebasing on new baseline |
| askeladd | #2357 | **Vortex-Panel Induced Velocity** | NEWLY ASSIGNED | Per-node inviscid physics oracle |
| tanjiro | #2358 | **Surface-Normal SRF Frame** | NEWLY ASSIGNED | Wall-normal coordinate frame for SRF input |
| nezuko | #2359 | **Spectral Regularization** | NEWLY ASSIGNED | λ sweep {1e-5, 1e-4, 1e-3}, OOD Lipschitz control |
| frieren | #2360 | **Input Consistency Regularization** | NEWLY ASSIGNED | R-Drop style dropout-robust predictions |

## Round 39 Results (PRs closed 2026-04-10)

| PR | Experiment | Result | Verdict |
|----|-----------|--------|---------|
| #2352 (tanjiro) | SRF FiLM Conditioning | p_oodc +2.7% vs new baseline | ❌ Closed |
| #2353 (nezuko) | Learnable Cp Scale | +6-42% all metrics | ❌ Closed |
| #2354 (askeladd) | Pressure Recovery Ratio | p_in +5.9%, p_tan flat | ❌ Closed |
| #2355 (frieren) | Two-Stage SRF | p_in +2.5%, p_oodc -1.9%, net negative | ❌ Closed |

## Most Promising Active Experiments

1. **Hypernetwork SRF v4** (alphonse #2341): v2 config (rank=2, α=0.5) + wake_angle_feature. If the physics hint compounds with the per-geometry adaptation, this could be the next merge candidate.
2. **Cl/Cd Auxiliary Loss v2** (fern #2340): p_tan -2.9% in v1. v2 applies tandem-only + rebased on wake_angle baseline. Strong tandem signal.
3. **Log-Re-Conditioned Cp** (thorfinn #2351): Directly targets p_re regression from Panel Cp. Mechanism is clean and specific.

## Key Insights (Updated)

1. **Physics-informed input features are the ONLY reliable lever.** Panel Cp, wake deficit, wake angle, TE coord frame — all improved metrics. Every architecture change has failed or been neutral.
2. **The SRF head is a local optimum.** FiLM conditioning, two-stage design, learnable scale, normal frame — none beat the baseline SRF. The per-node MLP correction is already near-optimal.
3. **The backbone is nearly untouchable.** Hypernetwork SRF (v2) is the only successful architecture modification, and only because it's a *very small* perturbation (rank=2 LoRA, α=0.5).
4. **Learnable physics feature parameters interact badly with PCGrad.** Learnable Cp scale was catastrophic. Fixed physics hints with manual scaling remain optimal.
5. **Redundant geometric features don't help.** Pressure recovery ratio was linearly derivable from existing features — the model already knows this. New physics features must encode non-trivial computation the model cannot infer from raw coordinates.
6. **The baseline loss/training pipeline is extremely well-tuned.** Focal L1, R-Drop, spectral reg, SWALR, Gumbel MoE, curriculum — all neutral or worse. Only physics input changes survive.

## What's Exhausted (DO NOT REVISIT)

Architecture:
- GRU sequential decoder, diffusion decoder, arc-length 1D conv
- FiLM SRF conditioning (Re/AoA adaptive)
- Two-stage SRF (velocity-first then pressure)
- Gumbel MoE SRF routing
- SWA weight averaging (redundant with EMA)
- All new backbone architectures (GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN)

Loss/training:
- Focal L1 surface loss
- Sample difficulty curriculum
- Quantile regression
- Jacobian smoothness regularization
- Condition-space interpolation
- Asymmetric surface loss (static 1.5× multiplier)
- Target noise

Physics features:
- Pressure recovery ratio (redundant with coordinates)
- Learnable Cp scale (PCGrad incompatibility)
- Arc-length positional encoding (6× slower, hurts p_in)
- Wake angle (MERGED → #2350)
- Panel Cp (MERGED → #2319)
- Wake deficit (MERGED → #2213)
- TE coordinate frame (MERGED → #2207)

## Next Research Priorities

**Currently assigned (Round 39):**
- Vortex-panel induced velocity (#2357) — extends physics oracle to all mesh nodes
- Surface-normal SRF frame (#2358) — physics-aware geometry input for SRF
- Spectral regularization (#2359) — Lipschitz OOD control on FFN weights
- Input consistency regularization (#2360) — R-Drop dropout robustness

**Unassigned ideas (for Round 40+):**
1. **MoE domain-expert FFN** (Idea 9) — domain-conditioned routing (tandem vs single-foil FFN experts)
2. **Global aero embedding** (Idea 14) — Cl/Cd embedding as SRF conditioning signal
3. **DSDF-weighted physics features** — scale physics hints by distance-to-surface
4. **Tandem-specific attention masking** — fore/aft attention separation
5. **Cross-foil attention via induced velocity** — use Biot-Savart kernel as attention bias

**If Round 39 stalls:** Run researcher-agent for fresh Round 40 ideas. Consider more radical directions: cross-domain pretraining on other airfoil datasets, graph-neural-operator backbone replacement, or physics-constrained output spaces.

## Research Theme Summary

Phase 6 is converging on: **physics-informed input features + small architectural perturbations (hypernetwork SRF)**. The Transolver+SRF backbone is a strong local optimum — modifications that change convergence dynamics are absorbed. The winning pattern is adding genuinely new physical information (not redundant geometric information) as input features. The next frontier is extending the physics oracle beyond surface Cp to volume-level quantities (induced velocity) and geometry-intrinsic coordinates (surface-normal frame).
