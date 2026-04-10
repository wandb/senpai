# SENPAI Research State

- **Date:** 2026-04-10 00:50 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training & Architecture Improvements

## Current Baseline

### Single-Model Baseline (PR #2319 Panel Cp ×0.1, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.709** | < 11.709 |
| **p_oodc** | **7.544** | < 7.544 |
| **p_tan** | **27.402** | < 27.402 |
| p_re | 6.481 | < 6.481 |

**Reproduce:** `python train.py ... --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1`
W&B runs: h6fqcry4 (s42), cuhoscp9 (s73)

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (2026-04-10 00:50 UTC)

| Student | PR | Experiment | Status | Notes |
|---------|-----|-----------|--------|-------|
| tanjiro | #2346 | **Focal L1 Surface Loss** | NEWLY ASSIGNED | Upweight hard surface nodes with focal gamma |
| nezuko | #2347 | **Sample Difficulty Curriculum** | NEWLY ASSIGNED | Oversample hard training samples |
| askeladd | #2348 | **Gumbel MoE SRF** | NEWLY ASSIGNED | Per-node expert routing K=4 in SRF |
| frieren | #2349 | **Diffusion Surface Decoder** | NEWLY ASSIGNED | DDPM/DDIM iterative denoising refinement |
| alphonse | #2341 v2 | **Hypernetwork SRF (rank=2, α=0.5)** | SENT BACK | v1 showed p_tan -3.1%, iterating with lower rank |
| thorfinn | #2344 | **SWA Weight Averaging** | TRAINING | Stochastic weight averaging |
| edward | #2345 | **Condition-Space Interpolation** | TRAINING | Same-geometry augmentation |
| fern | #2340 | **Cl/Cd Auxiliary Loss** | TRAINING | Integral aerodynamic force supervision |

## Recent Results (Round 36/37)

### Round 37 Review (2026-04-10 00:45)
| PR | Experiment | Result | Verdict |
|----|-----------|--------|---------|
| #2341 | Hypernetwork SRF (rank=8) | p_tan -3.1% ✅, p_in +2.8% ❌ | **Sent back** 🔄 |
| #2343 | Arc-Length 1D Conv | p_in +4.2%, 6× slower | Closed ❌ |
| #2332 | Target Noise (σ=0.01) | p_tan +4%, diff base config | Closed ❌ |
| #2339 | Quantile Regression | +92-162% ALL metrics | Closed ❌ |
| #2342 | Jacobian Smoothness | +21-34% ALL metrics | Closed ❌ |

### Round 36 Review (2026-04-09 23:10)
| PR | Experiment | Result | Verdict |
|----|-----------|--------|---------|
| #2338 | GRU Sequential Decoder | +53-207% (4× slower) | Closed ❌ |
| #2337 | Arc-Length Surface PE | p_in +5.6%, centroid bug | Closed ❌ |

## Key Insights from Phase 6

1. **Panel Cp as INPUT feature is now merged.** ×0.1 scaling prevents backbone disruption.
2. **Hypernetwork SRF shows strongest p_tan signal** (-3.1% at rank=8). Iterating with rank=2 to reduce p_in overfitting.
3. **Architecture output modifications safe; backbone changes dangerous.** SRF changes OK; attention/block changes regress p_tan.
4. **Bigger SRF capacity doesn't help.** 192 hidden is sufficient; wider/KAN/GRU/conv all failed.
5. **Physics-informed input features remain the strongest lever.** Every durable win came from features.
6. **Sequential surface decoders are not viable** — per-sample loops kill throughput (GRU 4×, conv 6× slower).
7. **Loss magnitude matters critically** — quantile regression's 0.5× gradient magnitude caused catastrophic failure. Any loss change must preserve gradient magnitude.
8. **Jacobian smoothness conflicts with CFD physics** — the model needs sensitivity to AoA/Re, not smoothness.

## Current Research Focus (Round 37)

### Promising Directions Being Tested
1. **Hypernetwork SRF v2** (alphonse) — rank=2, α=0.5 to capture p_tan -3.1% without p_in regression
2. **Focal L1 Surface Loss** (tanjiro) — upweight hard nodes at suction peaks/separation
3. **Sample Difficulty Curriculum** (nezuko) — oversample training samples with highest loss
4. **Gumbel MoE SRF** (askeladd) — per-node expert routing for regime specialization
5. **Diffusion Surface Decoder** (frieren) — DDPM/DDIM iterative refinement on SRF residuals
6. **SWA Weight Averaging** (thorfinn) — flatter minima for generalization
7. **Condition-Space Interpolation** (edward) — same-geometry augmentation
8. **Cl/Cd Auxiliary Loss** (fern) — integral force constraint

### Strategy Tiers
- **Tier 1 (Loss/Data):** Focal L1, sample curriculum, Cl/Cd loss — simple, targeted
- **Tier 2 (SRF Architecture):** Hypernetwork v2, Gumbel MoE, diffusion decoder — bold changes
- **Tier 3 (Training Strategy):** SWA, condition interp — regularization/augmentation

### What's Exhausted (DO NOT REVISIT)

- Architecture replacements, attention modifications, auxiliary heads
- Stochastic depth, EMA teacher, GradNorm, PDE losses
- All optimizer variants (Lion+EMA+cosine is optimal)
- Chord-position features, inter-foil coupling, DID/wake SDF features
- Mirror augmentation, synthetic data, per-foil whitening
- Tandem curriculum ramp, FV cell area loss, TTA variance proxy
- Spectral arc-length loss, Sobolev gradient loss, MoE output routing
- Inviscid Cp residual target, KAN surface decoder, warmup with Lion
- Log1p pressure target, BL proxy feature, local Re feature
- Surface-derivative losses on non-uniform meshes
- Target noise regularization (p_tan regression)
- Wider SRF head (192→256) — overfits
- GRU/Mamba sequential decoder — per-sample loop too slow
- Arc-length surface PE — centroid issue in tandem
- Arc-length 1D conv decoder — 6× slower, smooths suction peaks
- Quantile regression — halves gradient magnitude, catastrophic
- Jacobian smoothness — conflicts with CFD physics sensitivity needs
- Gradient accumulation 2x — neutral
- AoA curriculum (all forms) — p_tan regression
- Sample mixup — surface mask issues
- Checkpoint soup — neutral
- Panel Cp + AoA curriculum combo — no synergy
