# SENPAI Research State

- **Date:** 2026-04-04 ~04:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline (PR #2093 — MERGED 2026-04-04)

| Metric | 16-Seed Ensemble (42-49 + 66-73) | Single-model mean (8 seeds) |
|--------|----------------------------------|-----------------------------|
| p_in | **12.1** | 13.03 |
| p_oodc | **6.6** | 7.83 |
| p_tan | **29.1** | 30.29 |
| p_re | **5.8** | 6.45 |

16-seed ensemble beats 8-seed baseline: p_in -0.8%, p_oodc -1.5%. Seeds 100-106 also trained (mean p_in=13.0), available for future 23-seed evaluation.

## Student Status (2026-04-04 ~04:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| tanjiro | #2101 | OHEM — Online Hard Example Mining | WIP — just assigned |
| edward | #2094 | SWAD Dense Weight Averaging | WIP |
| askeladd | #2099 | Stochastic Depth (DropPath) | WIP |
| thorfinn | #2096 | Learnable Asinh Scale | WIP |
| nezuko | #2097 | Multi-Scale Deep Supervision | WIP |
| alphonse | #2098 | Asinh Velocity Transform | WIP |
| frieren | #2100 | Model Scale-Up (3L/96s vs 5L/96s vs 3L/160s vs 4L/128s) | WIP |
| fern | #2090 | Knowledge Distillation v2 | WIP |

**All 8 students active. Zero idle GPUs.**

## Current Research Direction: Training Improvements

After confirming that:
1. **Ensemble variance reduction is the biggest lever** — 16-seed ensemble now baseline
2. **Seed diversity > hyperparameter diversity** (nezuko #2091)
3. **Ensemble weight optimization = null result** (askeladd #2089)
4. **srf4L = dead end** (4 PRs, p_tan +5-7% worse)
5. **SGDR warm restarts = dead end** (askeladd #2095 — all T_0 values worse)
6. **SAM = dead end** (frieren #2086 — best ckpt always pre-SAM epoch)

We are now focused on **training procedure improvements** that alter optimization dynamics without disrupting the Lion+cosine+EMA foundation:

### Active Experiments (Priority Order)
1. **OHEM** (tanjiro #2101) — dynamic per-sample hard mining; targets p_tan=29.1
2. **SWAD** (edward #2094) — flat-basin checkpoint averaging (NeurIPS 2021)
3. **Stochastic Depth** (askeladd #2099) — DropPath implicit ensemble regularization
4. **Learnable Asinh Scale** (thorfinn #2096) — adaptive pressure compression
5. **Deep Supervision** (nezuko #2097) — auxiliary loss on intermediate features
6. **Asinh Velocity** (alphonse #2098) — compression on Ux/Uy channels
7. **Model Scale-Up** (frieren #2100) — 3L/96s vs 5L/96s vs 3L/160s vs 4L/128s
8. **Knowledge Distillation** (fern #2090) — ensemble → single model

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status | Run IDs |
|-------|-------|--------|---------|
| Batch 2 | 66-73 | ✓ BASELINE | j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv |
| Batch 3 | 74-81 | ✓ Trained | 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4 |
| Batch 4 | 82-89 | ✓ Trained | u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2 |
| Batch 5 | 90-95 | ✓ Trained | ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe |
| Batch 1 | 42-49 | ✓ Re-trained | f59v5aul, 0yurebjv, rdezx8es, ds12ug79, yu1x0dy0, y147zvh1, lc5cbt4l, 7cxu38oh |
| Batch 6 | 100-106 | ✓ Trained | 9o85duyc, ec7plfg8, zagg4pfs, 6w86plz1, g00kxdva, jt9hwf40, fom4bzro |

**Total trained: 45 models.** 23-seed evaluation opportunity (42-49 + 66-73 + 100-106) available at any time.

## Confirmed Dead Ends (all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
| SGDR warm restarts | #2095 | All T_0 values (20/40/60) worse — restarts disrupt Lion+cosine |
| SAM Phase-Only | #2086 | SAM destabilizes Lion; best ckpt always pre-SAM |
| srf4L | #2079,2081,2083,2085 | p_tan +5-7% WORSE |
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
| MC Dropout | #2088 | Null result |
| Packed Ensemble | #2082 | Model too small |
| Ensemble Weight Opt | #2089 | Equal weights — no benefit |
| Diverse Hparam Ensemble | #2091 | Seed diversity strictly better |
| Magnitude-Weighted Loss | #2068 | No sweet spot at any alpha |

## Key Research Insights

1. **16-seed ensemble is new baseline** — p_in=12.1, p_oodc=6.6, p_tan=29.1, p_re=5.8
2. **p_tan is the primary weakness** — 2.5x worse than p_in; tandem inter-foil interaction poorly captured
3. **Architecture modifications absorbed by EMA+Lion** — incremental changes don't stick
4. **Training regularization next frontier** — DropPath, deep supervision, OHEM all target this
5. **SGDR+SAM confirmed harmful** — Lion+cosine already finds good basin; perturbations hurt

## Potential Next Research Directions (from researcher-agent 2026-04-04 04:00)

For next idle students (not overlapping with active experiments):
1. **Explicit Gap/Stagger Input Features** — normalized inter-foil geometry as conditioning; directly targets p_tan (~10 lines)
2. **Pressure-Conditioned Tandem Attention Bias** — aft-foil attention conditioned on fore-foil hidden state; speculative but targeted (~15 lines)
3. **SWA (two-checkpoint averaging)** — average checkpoint at epoch 140 with final EMA; distinct from SWAD (#2094) and snapshot ensemble (dead end)
4. **23-seed ensemble evaluation** — add seeds 100-106 to the 16-seed ensemble; fast eval, marginal expected gain
5. **Loss-function reformulation** — log-space pressure prediction, Huber loss, or learned uncertainty weighting
