# SENPAI Research State
- **Date:** 2026-04-11 ~05:00 UTC (updated)
- **Advisor branch:** noam
- **Phase:** Phase 6 — Round 47 (post-ANP p_re fix campaign)

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

## Student Status (2026-04-11 ~05:00 UTC)

### Active & Queued Experiments

All Round 47 assignments target the **p_re regression** — the #1 priority after the ANP breakthrough.

| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| **alphonse** | **#2394** | **Cp-Space ANP Decoding** | Architecture (Re-invariant attention) | 🆕 ASSIGNED — normalize ANP to Cp space before cross-attention |
| **tanjiro** | **#2395** | **ANP Re-Conditional AdaLN** | Architecture (AdaLN on Q/K) | 🆕 QUEUED — waiting for #2385 to finish |
| **nezuko** | **#2396** | **ANP Context Dropout** | Regularization (NP technique) | 🆕 QUEUED — waiting for #2386 to finish |
| **fern** | **#2397** | **Stochastic Cross-Foil Attention** | Regularization (stochastic depth) | 🆕 QUEUED — waiting for #2387 to finish |
| **edward** | **#2398** | **Re Mixup in Cp Space** | Data aug (Re interpolation) | 🆕 QUEUED — waiting for #2374 v2 to finish |
| frieren | #2393 | BL Adaptive Node Weighting | Loss reweighting (OB-GNN style) | 🔄 Pod restarted, picking up #2393 (on ANP baseline) |
| askeladd | #2391 | Local Re_x BL Feature | Physics feature | 🔄 Waiting to finish #2391 queued (on ANP baseline) |
| thorfinn | #2392 | DeltaPhi Residual Prediction | Prediction reformulation | 🔄 Running arc-pe, #2392 queued (on ANP baseline) |

### Pre-ANP experiments still running (finishing within ~60 min)
| Student | Old PR | Experiment | Pre-ANP baseline Δ | Next Assignment |
|---------|--------|-----------|---------------------|----------------|
| tanjiro | #2385 | HyPINO SRF | TBD | → #2395 ANP Re-Conditional AdaLN |
| nezuko | #2386 | Stagnation Constraint | TBD | → #2396 ANP Context Dropout |
| fern | #2387 | Bernoulli Constraint | TBD | → #2397 Stochastic Cross-Foil Attn |
| edward | #2374 | Kutta v2 (3 configs) | TBD | → #2398 Re Mixup in Cp Space |

These will need review when they post results. Evaluate against PRE-ANP baseline (11.872/7.459/26.319/6.229). 

### Recent Closures
| PR | Student | Verdict |
|----|---------|---------|
| ❌ #2384 | alphonse | PirateNet SRF — neutral vs pre-ANP (p_in -1.4%, p_tan +1.2%); no --anp_srf |

### Recent Merges / Closures
| PR | Student | Experiment | Result | Action |
|----|---------|-----------|--------|--------|
| ✅ **#2379** | frieren | **ANP Cross-Foil Decoder** | **p_tan -59%, p_in -70%, p_oodc -48%, p_re +16%** | **MERGED — new baseline** |
| ❌ #2390 | frieren | ANP Re-Conditional Gate | All 3 runs crashed at step 0 (7h idle) | CLOSED — reassigned to BL Adaptive Weighting |
| ❌ #2388 | askeladd | Multi-Scale Hierarchical | Not started | CLOSED — redundant post-ANP |
| ❌ #2389 | thorfinn | Arc-Length PE | Not started | CLOSED — redirected to DeltaPhi Residual |

### Pod Issues
- **frieren** pod was stuck for ~6h (agent saw merged #2379 instead of new #2393, 0% GPU utilization). Restarted at ~04:20 UTC. New pod running.

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Too many incremental tweaks — think bigger. Radical new model changes, data aug, data generation."
**Status:** FULLY ACTIONED. ANP breakthrough (p_tan -59%) delivered the paradigm shift Morgan asked for. Round 46 bold ideas generated (see `RESEARCH_IDEAS_2026-04-11_ROUND46.md`): 12 ideas including competition-winning architectures (MARIO, AeroDiT, GeoMPNN, OB-GNN BL weighting), exact-physics augmentation (PIDA, flip-symmetry), and generative heads (diffusion, flow matching). All pre-ANP experiments will be closed and replaced with this bold slate when they finish. Updated Morgan in issue #1860 comment.

## Key Insights (Updated 2026-04-11 02:45)

1. **ANP cross-attention is the breakthrough.** Asymmetric cross-attention where aft-foil queries attend to all fore-foil context = direct inter-foil wake physics encoding. MERGED. p_tan -59%, p_in -70%, p_oodc -48%. Biggest single improvement in programme history.
2. **p_re is the current gap.** ANP regressed p_re +16% (7.232 vs old baseline 6.229). Cause: cross-attention aggregates fore-foil representations that are unreliable at OOD Reynolds numbers. Re-conditional gate (#2390) crashed — need alternative approach.
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

### Round 47 Active — P_re Fix Campaign (All on ANP baseline)
All 5 new experiments PLUS frieren, askeladd, thorfinn target the p_re regression.

| Priority | Student | PR | Idea | Target |
|----------|---------|-----|------|--------|
| P0 | alphonse | #2394 | **Cp-Space ANP Decoding** | p_re (Re-invariant attention) |
| P0 | tanjiro | #2395 | **ANP Re-Conditional AdaLN** | p_re (surgical Q/K conditioning) |
| P0 | nezuko | #2396 | **ANP Context Dropout** | p_re (NP robustness) |
| P0 | fern | #2397 | **Stochastic Cross-Foil Attn** | p_re (ensemble effect) |
| P0 | edward | #2398 | **Re Mixup in Cp Space** | p_re (data aug) |
| P0 | frieren | #2393 | **BL Adaptive Node Weighting** | surface_mae all |
| P1 | askeladd | #2391 | **Local Re_x BL Feature** | p_re |
| P1 | thorfinn | #2392 | **DeltaPhi Residual** | p_tan |

### Ideas in Reserve (next round)
From `RESEARCH_IDEAS_2026-04-11_04:30.md`:
- **Idea 5:** Spectral Normalization of ANP Q/K
- **Idea 6:** Multi-Resolution ANP Context (coarse + fine)
- **Idea 7:** Tandem Wake Superposition Prior (physics prior in ANP)
- **Idea 9:** Boundary Layer Thickness Feature (Blasius δ, δ*)
- **Idea 10:** Split ANP Heads (single-foil vs tandem)

### ⚠️ Key Flags for ALL future experiments
- MUST include `--anp_srf` (ANP is now the baseline)
- Target: p_re < 6.229 (recover pre-ANP Reynolds generalization)
- p_re is BLOCKED by ANP cross-attention sensitivity to Re-scaled pressure magnitudes
