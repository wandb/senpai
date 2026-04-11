# SENPAI Research State
- **Date:** 2026-04-11 ~04:30 UTC (updated)
- **Advisor branch:** noam
- **Phase:** Phase 6 — Round 45/46 (post-ANP breakthrough, experiments finishing)

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

## Student Status (2026-04-11 ~04:30 UTC)

### Active Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| frieren | **#2393** | **BL Adaptive Node Weighting** | Training (OB-GNN style) | ⚠️ Pod restarted (was stuck 6h, no runs). Should pick up #2393 now. On ANP baseline. |
| alphonse | #2384 | **PirateNet SRF** | Architecture (gated residual) | ✅ FINISHED — neutral result vs pre-ANP. Will be idle soon. |
| tanjiro | #2385 | **HyPINO Hypernetwork SRF** | Architecture (hypernetwork) | ~70% complete, 160 min in (pre-ANP) |
| nezuko | #2386 | **Stagnation Point Constraint** | Physics constraint | ~70% complete, 155 min in (pre-ANP) |
| fern | #2387 | **Bernoulli Velocity-Pressure Constraint** | Physics constraint | ~70% complete, 152 min in (pre-ANP, ⚠️ s42 bernoulli_loss worsening) |
| askeladd | (#2391 queued) | **Running MSHA** from prior round | Architecture | ~60% complete, 126 min in (pre-ANP). #2391 Local Re_x Feature waiting (on ANP). |
| edward | #2374 | **Hard Kutta TE Constraint v2** | Physics constraint | ~65% complete, 142 min in (pre-ANP, 3 configs) |
| thorfinn | (#2392 queued) | **Running Arc-Length PE** from prior round | Architecture | ~45% complete, 93 min in (pre-ANP). #2392 DeltaPhi Residual waiting (on ANP). |

### ⚠️ Note: Pre-ANP experiments
All currently training experiments EXCEPT frieren (#2393) are running WITHOUT `--anp_srf`. Their results cannot beat the ANP baseline. When they finish, evaluate the IDEAS (compare to pre-ANP baseline), then:
- Promising ideas → reassign with `--anp_srf` to test additivity with ANP
- Dead ends → close and assign fresh Round 47 hypotheses on ANP baseline

### Alphonse PirateNet SRF Results (just finished)
| Metric | Seed 42 | Seed 73 | 2-seed avg | Pre-ANP Baseline | Δ |
|--------|---------|---------|------------|------------------|---|
| p_in | 11.784 | 11.621 | 11.70 | 11.872 | -1.4% |
| p_oodc | 7.263 | 7.615 | 7.44 | 7.459 | -0.3% |
| p_tan | 26.587 | 26.673 | 26.63 | 26.319 | +1.2% |
| p_re | 6.256 | 6.341 | 6.30 | 6.229 | +1.1% |
Verdict: Neutral — PirateNet SRF multiplicative residuals didn't help. Close and reassign.

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

### Round 45/46 Active (on ANP baseline)
1. **Frieren #2393: BL Adaptive Node Weighting** — OB-GNN-style loss reweighting (surface ×10, near-wall ×3)
2. **Askeladd #2391: Local Re_x Feature** — arc-length BL Reynolds number, targets p_re (~31% complete)
3. **Thorfinn #2392: DeltaPhi Residual Prediction** — predict viscous correction over panel-method prior (~19% complete)

### When pre-ANP experiments finish → assign Round 46 bold ideas (per Morgan directive)
**alphonse (#2384):** → **PIDA Chord-Scaling Augmentation** (Reynolds similarity, exact physics, targets p_re) [BL weighting now assigned to frieren]
**edward (#2374):** → **MARIO Hypernetwork Re-Conditioning** (ML4CFD 3rd place, targets p_re)
**fern (#2387):** → **AeroDiT Diffusion Head** (NeurIPS 2024, 4-step DDIM with Re-CFG)
**tanjiro (#2385):** → **Flip-Symmetry Augmentation** (exact physics OOD, targets p_oodc and p_re)
**nezuko (#2386):** → **Displacement Thickness δ* Feature** (Thwaites BL integral, targets p_re)
**Bold reserve:** Flow Matching Head, Panel Synth Re-Aug, GeoMPNN Bipartite MPNN, AB-UPT Anchor Tokens, Divergence-Free Constraint
**Round 46 slate:** Generated — see `RESEARCH_IDEAS_2026-04-11_ROUND46.md`

### ⚠️ Key Flags for ALL future experiments
- MUST include `--anp_srf` (ANP is now the baseline)
- Watch fern Bernoulli (#2387): Ux MAE anomalously high — likely close and reassign to DDPM head
