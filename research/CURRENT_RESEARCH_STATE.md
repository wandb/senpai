# SENPAI Research State
- **Date:** 2026-04-11 ~06:20 UTC (updated)
- **Advisor branch:** noam
- **Phase:** Phase 6 — Round 47 (p_re fix campaign) → Round 48 BOLD pivot incoming

## Current Baseline — ANP Merged (PR #2379)

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

## Human Researcher Directive (Issue #1860) — ACTIVE

**Morgan McGuire:** "Too many incremental tweaks. THINK BIGGER. Radical new full model changes, data aug, data generation."

**Status:** ACTIONED — Round 48 bold slate generated and committed (`RESEARCH_IDEAS_2026-04-11_ROUND48_BOLD.md`). 12 radical ideas ranked by priority. Responded to Morgan in issue #1860.

**Round 47 Assessment:** Morgan is correct — Round 47 (AdaLN, context dropout, stochastic attention, Re Mixup) are all incremental ANP modifications. The Round 47 experiments target p_re via small architectural changes. Round 48 will be fundamentally different.

**Round 48 Direction:**
1. Completely new model families (FNO, CNO, diffusion decoder, flow matching)
2. Synthetic data generation using vortex-panel solver at OOD Re 
3. Physics-grounded augmentation (PIDA chord-scaling, AoA interpolation)
4. Radical problem reformulation (predict Cp not p, predict δCp over panel prior)

## Student Status (2026-04-11 ~06:00 UTC)

### Idle Students
- None — all 8 GPUs occupied

### Round 47 Active Experiments — P_re Fix Campaign
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| alphonse | #2394 | Cp-Space ANP Decoding | Re-invariant cross-attention | 🔄 WIP |
| tanjiro | #2395 | ANP Re-Conditional AdaLN | Q/K conditioning on Re | 🔄 WIP |
| nezuko | #2396 | ANP Context Dropout | NP regularization | 🔄 WIP |
| fern | #2397 | Stochastic Cross-Foil Attention | Stochastic depth | 🔄 WIP |
| edward | #2398 | Re Mixup in Cp Space | Re interpolation data aug | 🔄 WIP |
| frieren | #2393 | BL Adaptive Node Weighting | OB-GNN style loss reweighting | 🔄 WIP |
| askeladd | #2391 | Local Re_x BL Feature | Physics feature | 🔄 WIP |
| thorfinn | #2392 | DeltaPhi Residual Prediction | Predict viscous correction | 🔄 WIP |

### Pre-ANP Experiments (reviewed this cycle)
| PR | Student | Verdict |
|----|---------|---------|
| ❌ #2387 | fern | Bernoulli Constraint — all metrics +10-39% worse (Bernoulli invalid at no-slip wall) |
| ❌ #2385 | tanjiro | HyPINO Hypernetwork — null (LoRA zero-init bug → hypernetwork never updated) |
| ❌ #2386 | nezuko | Stagnation Constraint — all metrics +8-11% worse (fragile node ID + normalization) |
| 🔄 #2374 | edward | Kutta TE v2 — still WIP, no v2 results yet |

### Prior Closures/Merges
| PR | Student | Verdict |
|----|---------|---------|
| ✅ **#2379** | frieren | ANP Cross-Foil Decoder — MERGED (p_tan -59%, p_in -70%, p_oodc -48%) |
| ❌ #2390 | frieren | ANP Re-Conditional Gate — all 3 runs crashed at step 0 |
| ❌ #2388 | askeladd | Multi-Scale Hierarchical — closed (redundant post-ANP) |
| ❌ #2389 | thorfinn | Arc-Length PE — closed (redirected to DeltaPhi) |
| ❌ #2384 | alphonse | PirateNet SRF — neutral result |

## Key Insights

1. **ANP cross-attention is the breakthrough.** Asymmetric cross-attention (aft queries fore context) = direct inter-foil wake encoding. p_tan -59%, p_in -70%, p_oodc -48%. Biggest single improvement in programme history. MERGED.
2. **p_re is the current gap.** ANP regressed p_re +16% (7.232 vs 6.229). Cause: cross-attention aggregates fore-foil reps that are unreliable at OOD Re (pressure scales as Re²). Round 47 targets this with incremental fixes. Round 48 will target it radically.
3. **Stop-gradient kills cross-attention.** SCA v3 with stop-grad failed; ANP without stop-grad succeeded. Bidirectional gradient flow essential.
4. **Physics constraints are a viable secondary lever.** Kutta TE showed p_oodc -2.9%.
5. **Bold pivot needed.** Round 47 is surgical fixes. Round 48 must be paradigm shifts — new model families, synthetic data, problem reformulation.

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

### Round 48 — BOLD PIVOT (ready to assign when Round 47 finishes)
Full slate: `/research/RESEARCH_IDEAS_2026-04-11_ROUND48_BOLD.md`

| Rank | Idea | Confidence | Target | Category |
|------|------|-----------|--------|----------|
| 1 | **PIDA Chord-Scaling Aug** | High | p_re | Data aug (exact Reynolds similarity physics) |
| 2 | **Flip-Symmetry Aug** | High | p_oodc | Data aug (NeuralFoil-validated, doubles dataset) |
| 3 | **AeroDiT Diffusion + Re-CFG** | Medium | p_re | New model family (diffusion decoder) |
| 4 | **Tandem Wake Superposition Prior** | Med-High | p_tan | Problem reformulation (panel-initialized decoder) |
| 5 | **Flow Matching Generative Head** | Medium | p_in | New model family (rectified flow decoder) |
| 6 | **Displacement Thickness δ*** | Medium | p_re | Physics feature (Thwaites BL integral) |
| 7 | **Multi-Resolution ANP Context** | Medium | p_re/p_tan | Architecture (hierarchical arc-zone tokens) |
| 8 | **GeoMPNN Bipartite MPNN** | Medium | p_tan | New model family (ML4CFD 4th place) |
| 9 | **Multi-Fidelity Panel Pretraining** | Med-High | p_re | Synthetic data (100k panel solutions) |
| 10 | **Divergence-Free Constraint** | Medium | p_in | Hard physics (Helmholtz projection) |

**Assignment plan:** Ideas 1+2 combined for one student (both pure data aug). Wait for #2391/#2392 results before assigning δ*/Wake Prior.

### ⚠️ Key Flags for ALL future experiments
- MUST include `--anp_srf` (ANP is now the baseline)
- Target: p_re < 6.229 (recover pre-ANP Reynolds generalization)
- Target: all other metrics at or below current baseline
