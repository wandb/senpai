# SENPAI Research State
- **Date:** 2026-04-10 22:10 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold Round 42/43

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

## Student Status (2026-04-10 22:10 UTC)

### Active Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| askeladd | #2372 | **Surface Cross-Attention v2** | Architecture | Training 6 configs (~175 min, nearing completion). cfgC p_tan~25.6 BEATING BASELINE but p_in +14%. |
| edward | #2374 | **Hard Kutta TE Constraint** | Physics constraint | s42 DONE: p_oodc=7.24 (-2.9%) BEATING BASELINE. s73 just started (~15 min). |
| tanjiro | #2378 | **Contrastive Geometry Pretraining** | Data (UIUC airfoil) | SENT BACK — only 25 epochs due to timeout bug. Rerunning with 180-min budget. |
| frieren | #2379 | **Attentive Neural Process Decoder** | Architecture (ANP) | NEW — implementing |
| nezuko | #2380 | **Koopman Tandem Lifting** | Architecture (lifting) | Training 2 seeds (just started, debug + production) |
| alphonse | #2381 | **Role-Specialized Surface Heads** | Architecture (3 SRF) | Training 2 seeds (~11 min in) |
| thorfinn | #2382 | **SIREN Surface Decoder** | Architecture (sin activations) | NEW — implementing |
| fern | #2383 | **Tandem Auxiliary Prediction Heads** | Loss (aux heads) | NEW — just assigned. Pod restarted. |

### Closed This Session
| PR | Student | Experiment | Result | Why |
|----|---------|-----------|--------|-----|
| #2375 | alphonse | Panel Data Oracle | p_tan +57.5% | Inviscid labels poison tandem predictions |
| #2377 | frieren | SE(2) Geometry | p_oodc -0.9% (neutral) | MERGED by human researcher |
| #2376 | nezuko | Mamba SSM | No results | MERGED by human researcher |
| #2369 | thorfinn | Cross-Foil AR Decoding | p_oodc -0.8% (neutral) | 1-head/64-dim too constrained |
| #2363 | fern | Cl/Cd SRF Conditioning | p_tan +25.1% | Tandem interleaving + info bottleneck |

### Most Promising Active Direction

**Edward #2374 Hard Kutta TE Constraint** — the standout experiment:
- hard-soft-s42: p_oodc=**7.24** (-2.9%), p_re=**6.17** (-0.9%), p_in=12.14 (+2.3%), p_tan=26.70 (+1.5%)
- hard-only-s42: p_re=**6.12** (-1.7%), p_oodc=**7.26** (-2.7%)
- Awaiting s73 seed for 2-seed averages (running, ~15 min in)

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."
**Status:** FULLY ACTIONED. All 8 GPUs running bold experiments. No new human messages.

## Key Insights (Updated 2026-04-10 22:10)

1. **Physics CONSTRAINTS are the new winning lever.** Edward's Kutta TE shows p_oodc beating baseline (-2.9%). Constraints add information without replacing architecture. Next: more physics constraints (stagnation, conservation).
2. **All decoder REPLACEMENTS failed.** B-GNN, FNO, multi-scale, Cl/Cd conditioning — 4/4 failed. The Transolver + SRF is a strong local optimum for this task.
3. **Cross-attention at hidden level is the architecture direction.** Surface cross-attention v1 p_tan -3.7%. v2 (tandem-only + lr scaling) in final training. ANP decoder (frieren) explores this further.
4. **Auxiliary losses beat conditioning.** Cl/Cd as CONDITIONING failed (+25% p_tan). Kutta as CONSTRAINT succeeded. The gradient path matters: constraints flow INTO the backbone, conditioning adds noise ON TOP.
5. **p_tan = 26.319 remains our hardest metric.** 2× worse than p_in. Askeladd's cfgC shows p_tan~25.6 but with p_in regression.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem projection, GeoTransolver GALE, MoE-LoRA FFN, Biot-Savart attention bias
Architecture (decoder replacements): Surface B-GNN (+14%), 1D Surface FNO (+29%), Multi-Scale Slice (+8%)
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF, Cl/Cd SRF conditioning (+25%)
Physics features: Panel Cp, wake deficit, wake angle, vortex-panel velocity (all merged). Log-Re Cp, Joukowski Cp, viscous residual baseline (all failed)
Multi-fidelity: Panel oracle data (+57%, dead end)
Cross-foil sequential: Causal AR decoding (neutral, +1.6% p_tan)
Transfer learning: DPOT, self-pretraining denoising
Data augmentation: FFD geometry (label mismatch), re-scaling
Loss/gradient: Sobolev surface gradient dp/ds (conflicts PCGrad)

## Next Research Priorities

### Watch closely (potential merge candidates)
1. **Hard Kutta TE** (edward #2374) — awaiting s73 for 2-seed avg. If p_oodc beats baseline in avg → MERGE.
2. **Surface Cross-Attention v2** (askeladd #2372) — cfgC tandem-only p_tan beating baseline. Need to assess p_in trade-off.

### Active Round 43 (just assigned)
3. Attentive Neural Process (frieren #2379)
4. Koopman Tandem Lifting (nezuko #2380)
5. Role-Specialized SRF (alphonse #2381)
6. SIREN Surface Decoder (thorfinn #2382)
7. Tandem Auxiliary Heads (fern #2383)
8. Contrastive Pretrain v2 (tanjiro #2378, rerunning)

### Idea Queue (Round 44, assign when students idle)
- Need fresh ideas — researcher-agent hit context limits. Will generate manually.
- Directions to explore: more physics constraints, ensemble/distillation, learned loss weighting
