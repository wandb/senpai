# SENPAI Research State
- **Date:** 2026-04-11 01:50 UTC
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

## Student Status (2026-04-11 01:50 UTC)

### Active Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| frieren | #2379 | **Attentive Neural Process Decoder** | Architecture (ANP) | **POTENTIAL BREAKTHROUGH** — 134 min, s73: p_in=5.4 (-54%), p_oodc=4.6 (-38%), p_tan=13.3 (-49%), p_re=7.2 (+16%) |
| alphonse | #2384 | **PirateNet SRF** | Architecture (gated residual) | Training 33 min, val_in_dist=31-37 (early) |
| tanjiro | #2385 | **HyPINO Hypernetwork SRF** | Architecture (hypernetwork) | Training 11 min, very early warm-up |
| nezuko | #2386 | **Stagnation Point Constraint** | Physics constraint | Training 6 min, very early |
| fern | #2387 | **Bernoulli Velocity-Pressure Constraint** | Physics constraint | Training 3 min, very early |
| askeladd | #2388 | **Multi-Scale Hierarchical Slice Attention** | Architecture (cross-scale) | NEW — just assigned |
| edward | #2374 | **Hard Kutta TE Constraint v2** | Physics constraint | SENT BACK for v2 (K=2, tandem-only, lower weight). Not yet restarted. |
| thorfinn | #2389 | **Arc-Length Positional Encoding** | Architecture (PE) | NEW — just assigned |

### Closed This Session
| PR | Student | Experiment | Result | Why |
|----|---------|-----------|--------|-----|
| #2381 | alphonse | Role-Specialized SRF | p_in +4%, p_tan +6.7% | Fragmenting SRF data hurts more than specialization helps |
| #2380 | nezuko | Koopman Tandem Lifting | All metrics +8-19% | Linear Koopman can't capture nonlinear tandem interference |
| #2378 | tanjiro | Contrastive Pretrain v2 | All metrics 2-8× worse | Pretraining poisoned initialization in 180-min budget |
| #2383 | fern | Tandem Aux Heads | p_in -1.4% but 3 others regress | Aux heads interfere with PCGrad balancing |
| #2372 | askeladd | Surface Cross-Attention v3 | All metrics +2-16% | Stop-gradient severed useful gradient flow |
| #2382 | thorfinn | SIREN Surface Decoder | All metrics +35-75% | SIREN activations incompatible with Transolver+SRF |

### ⚠️ PRIORITY: Frieren ANP (#2379) — Potential Breakthrough

**Live metrics at ~127 min (volatile, NOT EMA):**

| Metric | s73 (jvfvrs4u) | s42 (metqdxaq) | Approx 2-seed avg | Baseline | Delta |
|--------|----------------|----------------|-------------------|----------|-------|
| p_in | 5.44 | 5.63 | ~5.5 | 11.872 | ~-54% |
| p_oodc | 4.61 | 4.83 | ~4.7 | 7.459 | ~-37% |
| p_tan | 13.30 | 13.05 | ~13.2 | 26.319 | ~-50% |
| p_re | 7.20 | 7.95 | ~7.6 | 6.229 | ~+22% |

- p_tan improvement of -50% would be the SINGLE BIGGEST improvement in the research programme
- p_oodc -37% and p_in -54% are also extraordinary
- p_re regression has IMPROVED: was +45% at 127 min, now +22% at 134 min (s73 p_re dropped from 10.09→7.20)
- ~46 min of training left — EMA sweep at end will determine final metrics
- Live metrics trending BETTER over time (p_tan was -46% at 97 min, -52% at 127 min, -50% at 134 min)

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."
**Status:** FULLY ACTIONED. All 8 GPUs running bold experiments. Round 44 assignments incorporate Morgan's #1926 ideas.

**Issue #1926 Status:**
- NOBLE: Already merged (PR #2204)
- PirateNets: Already merged in backbone (PR #2215), now testing in SRF (alphonse #2384)
- HyperP/mHC: HyPINO hypernetwork assigned to tanjiro (#2385)
- MSA: Multi-Scale Hierarchical Attention assigned to askeladd (#2388)
- Geosolver: Queued with fresh approach (GeoMPNN surface graph)
- HeavyBall: Queued for Round 45+
- XSA: Not retrying (orthogonal init already provides head diversity)
- Muon/Gram-NS: Conclusively failed (destroys gradient anisotropy for physics)

## Key Insights (Updated 2026-04-11 01:50)

1. **ANP cross-attention is the breakthrough direction.** Frieren's ANP shows live p_tan -52% by allowing aft-foil surface queries to attend to fore-foil context. The asymmetric attention (aft attends to all, fore only self) is the key. s42 now showing p_tan=11.67 — the best tandem metric EVER by a massive margin.
2. **Stop-gradient kills cross-attention.** askeladd's SCA v3 (#2372) with stop-gradient on fore-foil K,V failed (+2-16% across all metrics), while frieren's ANP without stop-gradient succeeds spectacularly. Bidirectional gradient flow is essential.
3. **Physics CONSTRAINTS remain a promising lever.** Kutta TE p_oodc -2.9%, now extending to stagnation points (nezuko #2386) and Bernoulli consistency (fern #2387).
4. **All decoder REPLACEMENTS and conditioning failed.** B-GNN, FNO, multi-scale, Cl/Cd conditioning, Koopman lifting, role-specialized SRF — 6/6 failed. The Transolver + SRF is a strong local optimum.
5. **Transfer learning/pretraining is exhausted** in 180-min budget. Contrastive pretrain failed catastrophically.
6. **p_re is the ANP's weakness.** The cross-attention mechanism appears to overfit to training Reynolds range. If ANP merges, follow-up should target p_re (Re-conditional attention, separate Re-extrapolation head).

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN (ω₀=10 and ω₀=30)
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem projection, GeoTransolver GALE, MoE-LoRA FFN, Biot-Savart attention bias, role-specialized SRF, surface cross-attention (stop-grad)
Architecture (decoder replacements): Surface B-GNN (+14%), 1D Surface FNO (+29%), Multi-Scale Slice (+8%), Koopman lifting (+19%)
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF, Cl/Cd SRF conditioning (+25%), tandem aux heads (+5%)
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

### Active Round 44
2. PirateNet SRF (alphonse #2384) — gated skip connections in SRF, 33 min
3. HyPINO Hypernetwork (tanjiro #2385) — per-sample SRF weight generation, 11 min
4. Stagnation Constraint (nezuko #2386) — physics constraint at leading edge, 6 min
5. Bernoulli Constraint (fern #2387) — velocity-pressure consistency, 3 min
6. Multi-Scale Hierarchical Attention (askeladd #2388) — coarse-to-fine cross-scale, just assigned
7. Hard Kutta TE v2 (edward #2374) — K=2, tandem-only, lower weight, awaiting restart
8. Arc-Length PE (thorfinn #2389) — 1D surface topology encoding, just assigned

### Idea Queue (Round 45+, assign when students idle)
From researcher-agent RESEARCH_IDEAS_2026-04-10_ROUND44.md:
- Hypernetwork Meta-Conditioning mHC (differs from HyPINO — conditions backbone not SRF)
- GeoMPNN Surface Graph (fresh GeoSolver approach)
- Bernoulli Velocity-Pressure Consistency Constraint (assigned to fern)
- Circulation-Conserving Tandem Feature (Kutta-Joukowski theorem)
- Arc-Length Positional Encoding for Surface Nodes
- Tandem Gap/Stagger Mixup Augmentation

### Post-ANP Follow-up Ideas (if ANP merges)
- Re-conditional cross-attention (scale attention weights by Re distance to training range)
- Separate p_re extrapolation head (auxiliary head that handles OOD-Re separately)
- ANP + physics constraints (combine ANP with Kutta/stagnation/Bernoulli constraints)
- ANP depth/width sweep (currently 4 heads, 192 dim — may be undertrained)
