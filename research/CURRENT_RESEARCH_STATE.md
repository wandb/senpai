# SENPAI Research State
- **Date:** 2026-04-10 21:35 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold Round 41/42

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

## Student Status (2026-04-10 21:35 UTC)

### Round 41/42 Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| askeladd | #2372 | **Surface Cross-Attention v2** | Architecture (tandem-only SCA) | Training 6 configs (~34k steps, nearing completion) |
| fern | #2363 | **Global Cl/Cd SRF Conditioning** | Architecture (two-pass, detach) | Training 2 seeds (~43k steps). p_tan ~33 — looking bad. |
| edward | #2374 | **Hard Kutta TE Constraint** | Physics constraint | Training 3 configs (~48k steps). **hard-soft looking PROMISING: p_oodc=7.28** |
| thorfinn | #2369 | **Cross-Foil AR Decoding** | Architecture (causal AR) | Training 2 seeds (~46k steps). s73 p_tan=26.61 near baseline. |
| tanjiro | #2378 | **Contrastive Geometry Pretraining** | Data (UIUC airfoil library) | Training 2 seeds (~7k steps, still in pretrain phase) |
| frieren | #2379 | **Attentive Neural Process Decoder** | Architecture (cross-foil ANP) | NEW — just assigned. Pod restarted. |
| nezuko | #2380 | **Koopman Tandem Lifting** | Architecture (linear lifting) | NEW — just assigned. Pod restarted. |
| alphonse | #2381 | **Role-Specialized Surface Heads** | Architecture (3 SRF heads) | NEW — just assigned. Pod restarted. |

### Closed This Session
- **#2375 (alphonse, Panel Data Oracle):** CLOSED. Catastrophic: p_tan +57.5%, all metrics +20-57%. Inviscid panel labels poisoned tandem predictions. Dead end.
- **#2377 (frieren, SE(2)-Equivariant Geometry):** MERGED by human. Neutral: p_oodc -0.9% but p_in +2.3%, p_tan +2.7%.
- **#2376 (nezuko, Mamba SSM Surface Decoder):** MERGED by human. No results posted (code changes only to EXPERIMENTS_LOG.md).

### Previous Session Closures
- **#2370 (nezuko, Surface B-GNN Decoder):** CLOSED. All metrics 7-14% above baseline.
- **#2371 (frieren, 1D Surface FNO Decoder):** CLOSED. All metrics 15-29% above baseline.
- **#2373 (tanjiro, Multi-Scale Slice Attention):** CLOSED. All metrics +2-8%, 50% epoch time overhead.

### Key Active Experiment Notes

**Edward #2374 Hard Kutta (MOST PROMISING):**
- hard-soft config: p_in=12.18, p_oodc=**7.28** (beating baseline!), p_tan=26.72, p_re=6.22
- hard-only config: p_in=12.11, p_oodc=7.39, p_tan=27.61, p_re=**6.19** (beating baseline!)
- soft-only config: p_in=12.55, p_oodc=7.38, p_tan=28.04, p_re=6.44
- Still running at ~48k steps — metrics may improve further with late convergence

**Askeladd #2372 Surface Cross-Attention v2:**
- cfgC-s73: p_in=17.89, p_oodc=9.92, p_tan=26.57 (near baseline)
- All 6 configs running but p_in significantly regressed across all configs
- Still running at ~34k steps — late convergence pattern possible

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."

**Status:** FULLY ACTIONED. Round 42 is 100% bold experiments. New Round 43 assignments (ANP, Koopman, Role-Specialized SRF) continue the bold direction. Issue #1860 — no new human messages since last advisor update.

## Key Insights (Updated 2026-04-10 21:35)

1. **Physics-informed input features are fully exhausted.** Panel Cp, wake deficit, wake angle, vortex-panel velocity — all merged and compounding.
2. **All bold decoder REPLACEMENTS failed catastrophically.** B-GNN (+14%), FNO (+29%), multi-scale (+8%), panel oracle (+57%). The Transolver backbone + SRF is an extremely strong local optimum.
3. **Surface cross-attention shows GENUINE tandem benefit.** p_tan -3.7% from global surface-only attention. v2 configs (tandem-conditional SCA) in flight.
4. **Physics CONSTRAINTS are the new frontier.** Edward's Hard Kutta TE constraint is the most promising active direction: p_oodc=7.28 beats baseline. Constraints ADD information without replacing architecture.
5. **Multi-fidelity training with panel data is a dead end.** Panel oracle (alphonse #2375) catastrophic at any weight — inviscid labels systematically wrong for viscous tandem.
6. **Late convergence matters.** Cross-attention went from p_tan=30+ at 50% to 25.3 at completion. Don't judge until runs finish.
7. **p_tan = 26.319 is our hardest metric** (2× worse than p_in). The tandem inter-foil interference is the dominant unsolved problem.
8. **Human researcher merged SE(2) and Mamba PRs** despite not beating baseline. Respect their judgment on code-level benefits.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem projection, GeoTransolver GALE, MoE-LoRA FFN, Biot-Savart attention bias
Architecture (decoder replacements): Surface B-GNN (+14%), 1D Surface FNO (+29%), Multi-Scale Slice (+8%)
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF
Physics features: Panel Cp ✅, wake deficit ✅, wake angle ✅, vortex-panel ✅ (all MERGED). Log-Re Cp, Joukowski Cp, viscous residual baseline (all failed)
Multi-fidelity: Panel oracle data (+57%, dead end)
Transfer learning: DPOT external checkpoint (AFNO incompatible), self-pretraining denoising (no inductive bias)
Data augmentation: FFD geometry augmentation (label mismatch fatal), re-scaling augmentation
Loss/gradient: Sobolev surface gradient dp/ds loss (conflicts with PCGrad 3-way at all weights 0.05-0.2)

## Next Research Priorities

### Immediate (Round 43 — just assigned)
1. **Attentive Neural Process Decoder** (frieren #2379) — cross-foil conditional pressure via ANP
2. **Koopman Tandem Lifting** (nezuko #2380) — linear operator for tandem interference
3. **Role-Specialized Surface Heads** (alphonse #2381) — 3 dedicated SRF heads by foil role

### Watch closely (may become merge candidates)
4. **Hard Kutta TE Constraint** (edward #2374) — p_oodc=7.28 beating baseline, hard-soft config
5. **Surface Cross-Attention v2** (askeladd #2372) — tandem-only SCA, multiple configs testing

### Idea Queue (assign when students become idle)
- Scale-Autoregressive Pressure Decoder (Idea 7) — coarse-to-fine at decoder level
- Diffusion GNN Decoder (Idea 8) — low priority, prior failure #2349
- Need fresh ideas from researcher-agent for Round 44
