# SENPAI Research State

- **Date:** 2026-04-09 18:50 UTC
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

## Student Status (2026-04-09 18:50 UTC)

| Student | PR | Experiment | Status | Notes |
|---------|-----|-----------|--------|-------|
| thorfinn | #2338 | **GRU Sequential Surface Decoder** | NEWLY ASSIGNED | Bold: arc-length-ordered recurrent SRF |
| edward | #2337 | **Arc-Length Surface PE** | NEWLY ASSIGNED | Fourier PE for surface nodes before SRF |
| fern | #2328 v2 | **AoA Curriculum warmup=20** | TRAINING (ETA ~19:48) | p_oodc -5.1% in v1, testing shorter warmup |
| nezuko | #2327 v2 | **Sample Mixup (fixed mask, α=0.2)** | TRAINING (ETA ~19:48) | Fixed surface mask bug |
| alphonse | #2335 | **Gradient Accumulation 2x** | TRAINING (ETA ~19:49) | |
| askeladd | #2336 | **Panel Cp + AoA Curriculum Combo** | TRAINING (ETA ~19:58) | Combination test |
| frieren | #2334 | **Checkpoint Soup** | TRAINING (ETA ~20:00) | Weight averaging |
| tanjiro | #2332 | **Target Noise Regularization** | IMPLEMENTING | Re-running after pod restart |

## Recent Results (Round 35/36)

### Completed This Session
| PR | Experiment | Result | Verdict |
|----|-----------|--------|---------|
| #2319 v3 | Panel Cp ×0.1 tandem-only | p_in -0.3%, p_oodc -1.3%, p_tan -1.7% | **MERGED** ✅ |
| #2333 | Wider SRF head (256) | +2-4% all metrics | **Closed** ❌ |
| #2325 | KAN Surface Decoder | +4-10% | Closed ❌ |
| #2326 | Warmup + Cosine | p_oodc +7% | Closed ❌ |
| #2329 | Log1p Pressure | p_tan +3.4% | Closed ❌ |
| #2330 | BL Proxy Feature | +23% all | Closed ❌ |
| #2331 | Local Re Feature | p_in +5.3% | Closed ❌ |

## Key Insights from Phase 6

1. **Panel Cp as INPUT feature is now merged.** ×0.1 scaling prevents backbone disruption while preserving physics hint.
2. **AoA Curriculum shows strongest p_oodc signal** (-5.1%), but p_tan +3.5%. v2 tests shorter warmup=20.
3. **Architecture output modifications safe; backbone changes dangerous.** SRF changes are OK; attention/block changes regress p_tan.
4. **Bigger SRF capacity doesn't help.** 192 hidden is already sufficient; wider/KAN/flow-matching all failed.
5. **Physics-informed input features remain the strongest lever.** Every durable win came from features.

## Current Research Focus (Round 36)

### Bold Architecture Ideas (per Issue #1860 "Think BIGGER")
1. **Sequential surface decoder** (thorfinn #2338 GRU) — arc-length-ordered recurrence for causal pressure propagation
2. **Surface positional encoding** (edward #2337 arc-PE) — explicit node location context for SRF
3. **Combination** (askeladd #2336) — Panel Cp + AoA Curriculum stacking

### Next Experiments to Assign (after current round completes)
Priority order from Round 36 research ideas:
1. **Idea 11: Quantile Regression Decoder** — 20 lines, pinball loss; strong Kaggle track record
2. **Idea 3: Cl/Cd Auxiliary Loss** — global integral constraint, orthogonal to DCT loss
3. **Idea 2: Hypernetwork SRF Weights** — per-geometry LoRA adaptation of SRF decoder
4. **Idea 8: Panel Residual GNN** — VortexNet-style: predict CFD-panel residual via GNN
5. **Idea 4: Jacobian Smoothness Reg** — smooth condition-space response for OOD generalization

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
- Target noise regularization (neutral)
- Wider SRF head (192→256) — overfits
