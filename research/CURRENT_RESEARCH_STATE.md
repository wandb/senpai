# SENPAI Research State

- **Date:** 2026-04-04 ~03:30 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline (PR #2080 — MERGED 2026-04-03)

| Metric | 8-Seed Ensemble (66-73) | Single-model mean (8 seeds) |
|--------|------------------------|-----------------------------|
| p_in | **12.2** | 13.03 |
| p_oodc | **6.7** | 7.83 |
| p_tan | **29.1** | 30.29 |
| p_re | **5.8** | 6.45 |

## Student Status (2026-04-04 ~03:30 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| edward | #2094 | SWAD Dense Weight Averaging | WIP |
| askeladd | #2099 | Stochastic Depth (DropPath) | WIP |
| thorfinn | #2096 | Learnable Asinh Scale | WIP |
| nezuko | #2097 | Multi-Scale Deep Supervision | WIP |
| alphonse | #2098 | Asinh Velocity Transform | WIP |
| frieren | #2100 | Model Scale-Up (3L/96s vs 5L/96s vs 3L/160s vs 4L/128s) | WIP — just assigned |
| fern | #2090 | Knowledge Distillation v2 | WIP |
| tanjiro | #2093 | 16-Seed Eval + Seeds 100-106 | WIP |

## Current Research Direction: Training Improvements

After confirming that:
1. **Seed diversity > hyperparameter diversity** (nezuko #2091)
2. **Ensemble weight optimization = null result** (askeladd #2089)
3. **Magnitude-weighted loss = dead end** (alphonse #2068)
4. **srf4L = dead end** (4 PRs)
5. **SGDR warm restarts = dead end** (askeladd #2095 — ALL T_0 values worse)

We are now focused on **training procedure improvements** that alter optimization dynamics:

### Active Experiments (Priority Order)
1. **SWAD** (edward #2094) — flat-basin checkpoint averaging, NeurIPS 2021 (concern: only 1 run after 180 min)
2. **Stochastic Depth** (askeladd #2099) — DropPath for implicit ensemble regularization, targets p_tan
3. **Learnable Asinh Scale** (thorfinn #2096) — adaptive pressure compression
4. **Deep Supervision** (nezuko #2097) — auxiliary loss on intermediate features
5. **Asinh Velocity** (alphonse #2098) — compression on Ux/Uy channels
6. **Knowledge Distillation** (fern #2090) — ensemble → single model distillation
7. **16-Seed Combined Eval** (tanjiro #2093) — quantify N-model scaling
8. **Model Scale-Up** (frieren #2100) — 3L/96s vs 5L/96s vs 3L/160s vs 4L/128s, 2 seeds each

## Confirmed Dead Ends (all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
| SGDR warm restarts | #2095 | All T_0 values (20/40/60) worse — restarts disrupt Lion+cosine stable descent |
| srf4L (4-layer surface refine) | #2079,2081,2083,2085 | p_tan +5-7% WORSE |
| Mesh interpolation | #2066 | Physically invalid for unstructured CFD |
| SOAP/HeavyBall | #2010,2018-2023 | 2-6% WORSE than Lion |
| Muon (full+hybrid) | #2006 | 30-70% worse |
| XSA attention | #2007 | Redundant with orthogonal slices |
| PirateNets RWF | #2008 | Attenuated by LayerNorm + Lion |
| NOBLE | #2011 | Model too small |
| LinearNO | #2033-2038 | All failed |
| Flow matching | #2036 | 60% worse (deterministic) |
| MARIO latent | #2037 | Redundant geometry encoding |
| Inviscid Cp | #2034 | Single-foil wrong for tandem |
| All-to-all surface attn | #2035 | +8% worse |
| Physics losses | #2016,2023 | WLS instability |
| SAM Phase-Only | #2086 | Dead end — SAM destabilizes Lion training, best ckpt always pre-SAM |
| MC Dropout | #2088 | Null result |
| Packed Ensemble | #2082 | Model too small |
| Ensemble Weight Opt | #2089 | Equal weights — no benefit |
| Diverse Hparam Ensemble | #2091 | Seed diversity strictly better |
| Magnitude-Weighted Loss | #2068 | No sweet spot at any alpha |

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status | Run IDs |
|-------|-------|--------|---------|
| Batch 2 | 66-73 | ✓ BASELINE | j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv |
| Batch 3 | 74-81 | ✓ Trained | 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4 |
| Batch 4 | 82-89 | ✓ Trained | u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2 |
| Batch 5 | 90-95 | ✓ Trained | ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe |
| Batch 1 | 42-49 | Pending re-train (tanjiro) | Pending new run IDs |
| Batch 6 | 100-106 | Pending (tanjiro) | Not started |

**Total trained: 30 models** (8+8+8+6). When tanjiro completes: 45 models.

## Key Research Insights

1. **Ensemble variance reduction is the biggest lever** — 30 models trained, pending combined evaluation
2. **Seed diversity > hyperparameter diversity > weight optimization** (confirmed experimentally)
3. **Architecture modifications absorbed by EMA+Lion** — incremental changes don't stick
4. **SGDR restarts HURT** — Lion+cosine already finds a good basin; restarts disrupt this
5. **Training regularization untested** — Stochastic Depth (DropPath) is a fresh angle targeting p_tan
6. **Asinh transform big win for pressure** — testing on velocity channels now
7. **Next frontier: training procedure changes** (SWAD, DropPath, deep supervision)

## Potential Next Research Directions (from researcher-agent 2026-04-04 03:30)

Ranked by priority for next idle students:
1. **Asymmetric Pressure Loss** — higher penalty for under-predicting suction peaks; ~5 lines; physically motivated for p_tan
2. **Explicit Gap/Stagger Input Features** — normalized inter-foil geometry as conditioning; directly targets p_tan=29.1 (2.5x worse than p_in)
3. **OHEM (Online Hard Example Mining)** — per-sample EMA loss tracking, upweight persistently hard samples; ~20 lines
4. **SWA (two-checkpoint averaging)** — average checkpoint at epoch 140 with final EMA; distinct from SWAD and snapshot ensemble
5. **Pressure-Conditioned Tandem Attention Bias** — aft-foil attention conditioned on fore-foil hidden state; speculative but targeted
6. **TTA via AoA perturbation** — inference-only, zero training cost
7. **Model scale-up** — in-flight with frieren #2100 (first capacity test)
