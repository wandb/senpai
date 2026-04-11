# SENPAI Research State
- **Date:** 2026-04-11 01:25 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Round 43/44

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

## Student Status (2026-04-11 01:25 UTC)

### Active Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| frieren | #2379 | **Attentive Neural Process Decoder** | Architecture (ANP) | **POTENTIAL BREAKTHROUGH** — 97 min, live p_tan=14.3 (-46%), p_oodc=5.5 (-26%), p_in~8 (-33%) but p_re~9.5 (+53%) |
| edward | #2374 | **Hard Kutta TE Constraint v2** | Physics constraint | SENT BACK for v2 (K=2, tandem-only, lower weight). Awaiting implementation. |
| askeladd | #2372 | **Surface Cross-Attention v3** | Architecture | Training 157 min, live p_in=15. Dead end trajectory. |
| fern | #2383 | **Tandem Auxiliary Prediction Heads** | Loss (aux heads) | Training 170 min, live p_in=12.7-12.8. Close to baseline but above. |
| thorfinn | #2382 | **SIREN Surface Decoder (ω₀=10)** | Architecture | Training 119 min, live p_in=25. Dead end. |
| alphonse | #2384 | **PirateNet SRF** | Architecture (gated residual) | NEW — just assigned |
| tanjiro | #2385 | **HyPINO Hypernetwork SRF** | Architecture (hypernetwork) | NEW — just assigned |
| nezuko | #2386 | **Stagnation Point Constraint** | Physics constraint | NEW — just assigned |

### Closed This Session
| PR | Student | Experiment | Result | Why |
|----|---------|-----------|--------|-----|
| #2381 | alphonse | Role-Specialized SRF | p_in +4%, p_tan +6.7% | Fragmenting SRF data hurts more than specialization helps |
| #2380 | nezuko | Koopman Tandem Lifting | All metrics +8-19% | Linear Koopman can't capture nonlinear tandem interference |
| #2378 | tanjiro | Contrastive Pretrain v2 | All metrics 2-8× worse | Pretraining poisoned initialization in 180-min budget |

### ⚠️ PRIORITY: Frieren ANP (#2379) — Potential Breakthrough

**Live metrics at 97 min (volatile, NOT EMA):**

| Metric | s73 | s42 | Approx 2-seed avg | Baseline | Delta |
|--------|-----|-----|-------------------|----------|-------|
| p_in | 7.5 | 8.0 | ~8 | 11.872 | ~-33% |
| p_oodc | 5.5 | 5.6 | ~5.5 | 7.459 | ~-26% |
| p_tan | 14.4 | 14.3 | ~14.3 | 26.319 | ~-46% |
| p_re | 9.0 | 9.6 | ~9.3 | 6.229 | ~+50% |

- p_tan improvement of -46% to -52% (varies by epoch) would be the SINGLE BIGGEST improvement in the research programme
- p_oodc -26% is also extraordinary
- Only p_re regresses significantly (+50%)
- Still 83 min of training left — EMA sweep at end will determine final metrics
- Live metrics are highly volatile at this stage — do not over-interpret single epoch values

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."
**Status:** FULLY ACTIONED. All 8 GPUs running bold experiments. 3 new Round 44 assignments incorporate Morgan's #1926 ideas.

**Issue #1926 Status:**
- NOBLE: Already merged (PR #2204)
- PirateNets: Already merged in backbone (PR #2215), now testing in SRF (alphonse #2384)
- HyperP/mHC: HyPINO hypernetwork assigned to tanjiro (#2385)
- MSA: Queued for Round 44+
- Geosolver: Queued with fresh approach (GeoMPNN surface graph)
- HeavyBall: Queued for Round 44+
- XSA: Not retrying (orthogonal init already provides head diversity)
- Muon/Gram-NS: Conclusively failed (destroys gradient anisotropy for physics)

## Key Insights (Updated 2026-04-11 01:25)

1. **ANP cross-attention is the breakthrough direction.** Frieren's ANP shows live p_tan -46% to -52% by allowing aft-foil surface queries to attend to fore-foil context. The asymmetric attention (aft attends to all, fore only self) is the key.
2. **Physics CONSTRAINTS remain a winning lever.** Kutta TE p_oodc -2.9%, now extending to stagnation points (nezuko #2386).
3. **All decoder REPLACEMENTS and conditioning failed.** B-GNN, FNO, multi-scale, Cl/Cd conditioning, Koopman lifting, role-specialized SRF — 6/6 failed. The Transolver + SRF is a strong local optimum.
4. **Cross-attention at hidden level helps p_tan.** SCA v1 p_tan -3.7%, ANP p_tan -46%+ (live). The common thread: enabling information flow between fore and aft foil surface nodes.
5. **Transfer learning/pretraining is exhausted** in 180-min budget. Contrastive pretrain failed catastrophically even with only 5-min pretrain phase.
6. **p_re is the ANP's weakness.** The cross-attention mechanism appears to overfit to training Reynolds range. If ANP merges, follow-up should target p_re (Re-conditional attention, separate Re-extrapolation head).

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem projection, GeoTransolver GALE, MoE-LoRA FFN, Biot-Savart attention bias, role-specialized SRF
Architecture (decoder replacements): Surface B-GNN (+14%), 1D Surface FNO (+29%), Multi-Scale Slice (+8%), Koopman lifting (+19%)
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF, Cl/Cd SRF conditioning (+25%)
Physics features: Panel Cp, wake deficit, wake angle, vortex-panel velocity (all merged). Log-Re Cp, Joukowski Cp, viscous residual baseline (all failed)
Multi-fidelity: Panel oracle data (+57%, dead end)
Cross-foil sequential: Causal AR decoding (neutral, +1.6% p_tan)
Transfer learning: DPOT, self-pretraining denoising, contrastive geometry pretrain (2-8× worse)
Data augmentation: FFD geometry (label mismatch), re-scaling
Loss/gradient: Sobolev surface gradient dp/ds (conflicts PCGrad)
Optimizers: Muon/Gram-NS (destroys gradient anisotropy)

## Next Research Priorities

### Watch MOST closely
1. **Frieren ANP #2379** — if EMA metrics maintain p_tan -40%+ → IMMEDIATE MERGE candidate, even with p_re regression

### Active Round 44 (just assigned)
2. PirateNet SRF (alphonse #2384) — gated skip connections in SRF
3. HyPINO Hypernetwork (tanjiro #2385) — per-sample SRF weight generation
4. Stagnation Constraint (nezuko #2386) — physics constraint at leading edge
5. Hard Kutta TE v2 (edward #2374) — K=2, tandem-only, lower weight

### Closing soon (dead ends in progress)
6. SIREN ω₀=10 (thorfinn #2382) — p_in=25, dead end
7. Surface Cross-Attention v3 (askeladd #2372) — p_in=15, dead end
8. Tandem Auxiliary Heads (fern #2383) — p_in=12.7, borderline

### Idea Queue (Round 44+, assign when students idle)
From researcher-agent RESEARCH_IDEAS_2026-04-10_ROUND44.md:
- Multi-Scale Hierarchical Attention v2 (differs from failed #2373)
- Hypernetwork Meta-Conditioning (mHC for backbone)
- GeoMPNN Surface Graph (fresh GeoSolver approach)
- Bernoulli Velocity-Pressure Consistency Constraint
- Circulation-Conserving Tandem Feature (Kutta-Joukowski theorem)
- Arc-Length Positional Encoding for Surface Nodes
- Tandem Gap/Stagger Mixup Augmentation
