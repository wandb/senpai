# SENPAI Research State
- **Date:** 2026-04-10 19:20 UTC
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

## Student Status (2026-04-10 18:45 UTC)

### Round 41/42 Experiments
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| askeladd | #2372 | **Surface Cross-Attention v2** | Architecture (tandem-only SCA) | Sent back with tandem-only + lr fix instructions. p_tan -3.7% but p_in +9.2%. |
| fern | #2363 | **Global Cl/Cd SRF Conditioning** | Architecture (two-pass) | Training ~39 min in (2 seeds) |
| alphonse | #2375 | **Panel Data Oracle** | Multi-fidelity data | Training ~33 min in (w=0.1, 2 seeds, after debug iterations) |
| frieren | #2377 | **SE(2)-Equivariant Geometry** | Input features | Training ~24 min in (2 seeds) |
| edward | #2374 | **Hard Kutta TE Constraint** | Physics constraint | Training 3 configs simultaneously (hard-only, soft-only, combined) |
| thorfinn | #2369 | **Cross-Foil AR Decoding** | Architecture (causal AR) | Training ~3 min (2 seeds, after debug) |
| nezuko | #2376 | **Mamba SSM Surface Decoder** | Architecture (sequential SSM) | Implementing (pod restarted, iteration 1) |
| tanjiro | #2378 | **Contrastive Geometry Pretraining** | Data (UIUC airfoil library) | NEW — just assigned. Pretrain geometry encoder on 2000+ airfoil shapes. |

### Idle students
- nezuko (implementing, no W&B runs yet)

### Closed This Session
- **#2370 (nezuko, Surface B-GNN Decoder):** CLOSED. All metrics 7-14% above baseline. Local surface GNN cannot capture global tandem coupling. Dead end.
- **#2371 (frieren, 1D Surface FNO Decoder):** CLOSED. All metrics 15-29% above baseline. Arc-length interpolation destroys fine structure. Dead end.
- **#2373 (tanjiro, Multi-Scale Slice Attention):** CLOSED. All metrics above baseline (+2-8%). Epoch time 103s (vs 68s) — 50% overhead incompatible with training budget.

### Key Review Decisions
- **#2372 (askeladd, Surface Cross-Attention): SENT BACK** for v2. p_tan=-3.7% is the strongest tandem improvement. p_in +9.2% is fixable via tandem-only conditioning (skip SCA for single-foil samples, analogous to cp_panel_tandem_only). Also instructed 0.1× lr for SCA params to address norm explosion (~490).

### Pod Actions This Session
- Restarted thorfinn, fern (stale state) → now training
- Restarted edward, nezuko, thorfinn again (stuck after experiment completion) → now training/implementing

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."

**Status:** FULLY ACTIONED. Round 41/42 is 100% architectural and physics-constraint experiments. Fresh status posted on issue #1860 (2026-04-10 17:00 UTC). Round 42 ideas doc ready for assignment when students become idle.

## Key Insights (Updated 2026-04-10 18:45)

1. **Physics-informed input features are now fully exhausted.** Panel Cp, wake deficit, wake angle, vortex-panel velocity — all merged and compounding. This was the primary lever for Rounds 30-40. New territory: constraints, loss formulation, architecture.
2. **All bold decoder replacements failed catastrophically.** B-GNN (+14%), FNO (+29%), and viscous residual prediction (+22%) all significantly worse than baseline. The Transolver backbone + SRF is an extremely strong local optimum for this task.
3. **Surface cross-attention shows GENUINE tandem benefit.** p_tan -3.7% from global surface-only attention post-backbone. The mechanism works for tandem but hurts single-foil (p_in +9.2%). FIX: tandem-conditional SCA (skip for single-foil). This is the most promising active direction.
4. **Local-only surface methods fail for multi-body aerodynamics.** B-GNN (local k-NN), FNO (per-foil spectral) — any method that severs inter-foil context fails on p_tan. Successful surface decoders MUST preserve global cross-foil information.
5. **Late convergence matters.** Askeladd's cross-attention went from p_tan=30+ at 50% training to 25.3 at completion. Bold experiments may look bad mid-training but converge at the end due to EMA + cosine schedule. Don't judge too early.
6. **High seed variance = unstable training signal.** When s42 and s73 diverge >5% on p_in, the approach is learning something spurious.
7. **p_tan = 26.319 is our hardest metric** (2× worse than p_in). The tandem inter-foil interference is the dominant unsolved problem.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem projection, GeoTransolver GALE, MoE-LoRA FFN, Biot-Savart attention bias
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF
Physics features: Panel Cp ✅, wake deficit ✅, wake angle ✅, vortex-panel ✅ (all MERGED). Log-Re Cp, Joukowski Cp, viscous residual baseline (all failed)
Transfer learning: DPOT external checkpoint (AFNO incompatible), self-pretraining denoising (no inductive bias)
Data augmentation: FFD geometry augmentation (label mismatch fatal), re-scaling augmentation
Loss/gradient: Sobolev surface gradient dp/ds loss (conflicts with PCGrad 3-way at all weights 0.05-0.2)

## Next Research Priorities (Round 42+)

Researcher-agent completed (2026-04-10 16:15 UTC). Full details in `RESEARCH_IDEAS_2026-04-10_ROUND42.md`.

**Ranked by expected p_tan impact (assign in this order as students become idle):**

### Tier 1 — Assign first
1. **Panel Method as Cheap Data Oracle** — ✅ ASSIGNED to alphonse (#2375). Multi-fidelity training with synthetic tandem samples using panel solver. Self-consistent inviscid labels at 0.1× weight. Expected: p_tan -5 to -10%. **CONFIDENCE: HIGH.**
2. **Mamba/SSM Surface Sequence Model** — replace SRF MLP with Mamba SSM on arc-length-ordered surface nodes. Sequential inductive bias for pressure propagation. Run SRF in eager mode (torch.compile breaks Mamba), wrap only backbone. Expected: p_tan -3 to -8%. **CONFIDENCE: MEDIUM-HIGH.** ← **Next assignment for next idle student.**

### Tier 2 — High ceiling, moderate risk
3. **SE(2)-Equivariant Geometry Encoding** — add rotation-invariant relative angle/distance features from surface nodes to both foil centroids. Pure feature engineering, no new libraries. Targets OOD tandem geometry generalization. Expected: p_tan -2 to -5%.
4. **Attentive Neural Process Decoder** — fore-foil nodes as cross-foil context, aft-foil nodes as queries via ANP decoder. Targets both p_tan (interference) and p_oodc (OOD generalization) simultaneously. Expected: p_tan -4 to -8%.
5. **Contrastive Geometry Pretraining** — pretrain geometry encoder on 1600+ UIUC airfoil shapes with InfoNCE contrastive loss. Enriches geometry embeddings beyond the limited training set. Expected: p_tan -3 to -6%.
