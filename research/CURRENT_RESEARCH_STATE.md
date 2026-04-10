# SENPAI Research State
- **Date:** 2026-04-10 15:20 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Bold Round 40/41 (all 8 students WIP)

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

## Student Status (2026-04-10 15:25 UTC)

### Round 40 Bold Experiments (still WIP)
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| edward | #2362 | **Viscous Residual Prediction** | Prediction reformulation | WIP — initial results BAD (p_oodc +23.8%), rebasing onto latest noam |
| fern | #2363 | **Global Cl/Cd SRF Conditioning** | Architecture (two-pass) | WIP — rebasing onto latest noam |
| alphonse | #2368 | **Sobolev Surface Gradient Loss** | Loss formulation | WIP — implementing |
| ~~frieren~~ | ~~#2365~~ | ~~FFD Geometry Augmentation~~ | ~~Data generation~~ | CLOSED — p_in +9.4%, p_tan +6.2% |
| ~~tanjiro~~ | ~~#2366~~ | ~~MoE Domain-Expert FFN~~ | ~~Architecture (routing)~~ | CLOSED — all metrics regress vs current baseline |
| ~~askeladd~~ | ~~#2367~~ | ~~Biot-Savart Attention Bias~~ | ~~Physics-informed attention~~ | CLOSED — p_oodc -1.1% but p_tan +2.7% |

### Round 41 Bold Experiments (just assigned)
| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| thorfinn | #2369 | **Cross-Foil Autoregressive Decoding** | Architecture (causal AR) | WIP — just assigned |
| nezuko | #2370 | **Surface-Intrinsic B-GNN** | Architecture (pure surface GNN) | WIP — just assigned |
| frieren | #2371 | **1D Surface FNO Decoder** | Architecture (spectral surface) | WIP — just assigned |
| askeladd | #2372 | **Surface-Node Cross-Attention** | Architecture (global surface attn) | WIP — just assigned |

## This Session's Actions (2026-04-10 15:25 UTC)

### PR Check
- **PR #2362 (edward):** Initial results posted — all metrics regress badly (p_oodc +23.8%). Student found val-loop bug, rebasing onto latest noam. Watch for updated results.
- **Other 7 PRs:** Students still implementing, no results yet.

### Researcher-Agent Output
- Generated 8 bold Round 42 ideas → `RESEARCH_IDEAS_2026-04-10_15:20.md`
- Top candidates: Panel Data Oracle, Mamba SSM decoder, Hard Kutta constraint, Neural Process, SE(2)-Equivariance

### Prior Session Actions (Round 40/41 setup)
- Closed Round 39 stragglers: #2351 (log-Re Cp), #2359 (spectral reg), #2365 (FFD augmentation), #2367 (Biot-Savart attn bias)
- Assigned Round 41: #2369 (AR decoding), #2370 (surface B-GNN), #2371 (1D FNO decoder), #2372 (surface cross-attn), #2373 (multi-scale slice attn)

## Human Researcher Directive (Issue #1860)

**Morgan McGuire:** "Think bigger — radical new model changes, data aug, data generation."

**Status:** FULLY ACTIONED. Round 41 is 100% architectural and loss-formulation experiments — full decoder replacements (B-GNN, FNO), causal AR decoding, task reformulation (viscous residual), integral force conditioning. None are incremental tweaks.

Researcher-agent launched (2026-04-10 15:20) to generate Round 42 bold ideas: neural processes, Koopman operators, flow matching, data synthesis beyond fixed dataset. Results → `/research/RESEARCH_IDEAS_2026-04-10_15:20.md`.

## Key Insights (Updated)

1. **Physics-informed input features remain the strongest lever.** Panel Cp, wake deficit, wake angle, vortex-panel velocity — all merged. Physics feature era is now exhausted.
2. **Transfer learning from external checkpoints is eliminated.** DPOT uses AFNO (not Transolver), and self-pretraining on the same dataset provides no inductive bias. The backbone at ~1M params is too small for meaningful pretraining.
3. **2× forward pass approaches are incompatible** with the 30-min training timeout. Must use stop-gradient for any multi-pass approaches.
4. **Integral loss terms (Cl/Cd) structurally conflict with per-node MAE** via PCGrad. Must be delivered as conditioning, not loss.
5. **High seed variance = unstable training signal.** When s42 and s73 diverge >5% on p_in, the approach is learning something spurious, not a physics-grounded signal.

## What's Exhausted (DO NOT REVISIT)

Architecture (from scratch): GNOT, Galerkin, HPT, FactFormer, DeepONet, SIREN
Architecture (tweaks): FiLM SRF, two-stage SRF, GRU decoder, diffusion decoder, Gumbel MoE, hypernetwork SRF, per-head KV, foil role embed, SRF normal frame, decoupled tandem slice projection, GeoTransolver GALE
Loss/training: Focal L1, curriculum, quantile reg, Jacobian smooth, asymmetric loss, R-Drop, attention noise, aug annealing, spectral reg, OHNM, two-pass SRF
Physics features: Panel Cp ✅, wake deficit ✅, wake angle ✅, vortex-panel ✅ (all MERGED). Log-Re Cp, Joukowski Cp, Viscous Residual Baseline Cp, arc-length input (all failed)
Transfer learning: DPOT external checkpoint (AFNO incompatible), self-pretraining denoising (no inductive bias)

## Next Research Priorities (Round 42+)

Researcher-agent completed (see `RESEARCH_IDEAS_2026-04-10_15:20.md` for full details). **Ranked by expected impact:**

### Tier 1 — Assign first (highest impact, lowest risk)
1. **Panel Method as Cheap Data Oracle** — generate 5000+ synthetic tandem configs using panel solver; train as auxiliary data at 0.1× loss weight. Infrastructure already exists (`--cp_panel`). Literature: 10² MSE reduction from inviscid pretraining (Springer Nature 2025). Target: p_tan -10–20%.
2. **Mamba/SSM Surface Sequence Model** — replace SRF attention with Mamba SSM; O(n) enables 6–8 layer surface decoder. Mamba Neural Operator (2025) outperforms Transformers on PDE benchmarks. Target: p_in, p_tan -5–10%.
3. **Hard Bernoulli/Kutta Constraint Layer** — hard-enforce Kutta condition at TE (2 lines of differentiable code, parameter-free). Soft Bernoulli coupling as 0.1× loss. Lowest implementation risk. Target: p_tan -5–8%.

### Tier 2 — High ceiling, moderate risk
4. **Neural Process for Pressure Fields** — ANP with context = (surface_coord, cp_panel), target = CFD pressure. No prior CFD application — genuinely novel. Target: p_oodc -10–15%.
5. **SE(2)-Equivariant Architecture** — replace (x,y) coords with SE(2)-invariant features (distance to centroid, curvature) + equivariant vectors (normals). Forces orientation generalization. Target: p_oodc -8–12%.
6. **Contrastive Geometry Pretraining** — pretrain on UIUC+NACA airfoil library (1500+ geometries) with SimCLR/BYOL. Target: p_oodc -8–12%.

### Tier 3 — Schedule after Round 41 closes
7. **Koopman Operator Lifting** — Koopman autoencoder where tandem interference is linear in latent space. Target: p_tan -8–12%.
8. **Flow Matching on Pressure Field** — conditional flow matching from noise→pressure, captures multi-modal tandem wake states. **Wait for PR #2371 (FNO) to close first.** Target: p_tan -10–15%.
