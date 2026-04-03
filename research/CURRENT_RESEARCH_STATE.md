# SENPAI Research State

- **Date:** 2026-04-03 ~21:15 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Ensemble Expansion + Diverse Methods

## Current Baseline (PR #2080 — MERGED 2026-04-03)

| Metric | 8-Seed Ensemble (66-73) | vs prior best (42-49) |
|--------|------------------------|-----------------------|
| p_in | **12.2** | **-1.6%** |
| p_oodc | **6.7** | 0% |
| p_tan | **29.1** | **-1.0%** |
| p_re | **5.8** | 0% |

Seeds: j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv

**Phase 6 wins so far:**
1. T_max=160 (PR #2003) — all metrics improved 1-3.5%
2. Asinh s=0.75 (PR #2054) — OOD improved 4-6%
3. 8-seed ensemble (PR #2076) — ALL metrics -3% to -14%
4. Seeds 66-73 ensemble (PR #2080) — p_in -1.6%, p_tan -1.0%

## Student Status (2026-04-03 ~23:20 UTC)

| Student | PR | Experiment | Status | ETA |
|---------|-----|-----------|--------|-----|
| edward | #2094 | SWAD Dense Weight Averaging | New — just assigned | ~3h |
| askeladd | #2095 | SGDR Warm Restarts (T_0 sweep) | New — just assigned | ~3h |
| nezuko | #2091 | Diverse Hyperparameter Ensemble | Running ~160 min | ~20 min |
| thorfinn | #2092 | Standard 3L Seeds 82-89 | Running ~151 min | ~30 min |
| fern | #2090 | Knowledge Distillation (teachers) | Running ~158 min | ~22 min |
| tanjiro | #2093 | Re-train seeds 42-49, then eval+100-106 | Running ~145 min | ~35 min |
| alphonse | #2068 | Asymmetric Loss v2 (α=0.1-0.3) | Running ~130 min | ~50 min |
| frieren | #2086 | SAM Phase-Only v3 (restarted) | Running ~21 min | ~159 min |

**Completed this round:**
- edward #2087: Seeds 74-81 trained ✓ (CLOSED — models available for ensemble)
- askeladd #2089: Weight optimization NULL RESULT ✓ (CLOSED — equal weights, need method diversity)

## Current Ensemble Seed Pool

| Batch | Seeds | Status | Notes |
|-------|-------|--------|-------|
| Batch 1 | 42-49 | Done (re-training by tanjiro) | Original 8-seed ensemble (p_in=12.4) |
| Batch 2 | 66-73 | Done ✓ MERGED | **New best** (p_in=12.2) |
| Batch 3 | 74-81 | Done ✓ (edward) | 8 models trained |
| Batch 4 | 82-89 | Training (thorfinn) | ~120 min remaining |
| Batch 5 | 90-95 | Done ✓ (askeladd) | 6 models trained |
| Batch 6 | 100-106 | Pending (tanjiro, after re-train) | ~4h |

**Total seed pool when complete: 48+ seeds**

## Next Research Directions (researcher-agent output 2026-04-03 20:50)

Ranked by priority — to be assigned as current experiments complete:

### Priority 1: SWAD (Stochastic Weight Averaging Densely)
- **Code change:** Flag flip `--swad True` (fully implemented in train.py)
- **Evidence:** NeurIPS 2021 (Cha et al.) — flat basin averaging, strong OOD generalization
- **Why now:** Ensemble variance reduction is our biggest lever. SWAD gives a single model parameter-space averaging without separate seeds. Orthogonal to Lion/EMA.
- **Experiment:** Compare --swad True vs baseline, 8 seeds, seed=42 comparison + sweep

### Priority 2: SGDR Warm Restarts (T_0=40, T_mult=2)
- **Code change:** Flag flip `--scheduler_type warm_restarts --cosine_T_0 40 --cosine_T_mult 2`
- **Evidence:** ICLR 2017 (Loshchilov & Hutter), widely validated
- **Why:** p_tan=29.1 is notably worse than p_in=12.2 — suggests overfitting to single-foil geometry. Warm restarts escape local minima, especially for OOD splits.
- **Experiment:** Compare standard cosine vs warm_restarts at T_0=40/T_mult=2, 8 seeds

### Priority 3: Learnable Asinh Scale
- **Code change:** ~5 lines — `self.asinh_scale = nn.Parameter(torch.tensor(0.75))`
- **Why:** Fixed s=0.75 chosen by grid search. Learnable scale adapts per run to actual pressure distribution. Especially could help p_tan.
- **Extension:** Asymmetric asinh (separate scale for positive/negative pressure)

### Priority 4: Multi-Scale Output Supervision
- **Code change:** ~15 lines — auxiliary head on `fx_deep` (pre-final block features)
- **Evidence:** Deep supervision standard in nnUNet, HED, FNO variants
- **Why:** 3-block Transolver is shallow. Aux loss forces earlier blocks to learn better physics representations.

### Priority 5: Asinh on Velocity Channels
- **Code change:** ~10 lines — apply asinh to Ux/Uy channels with separate scale
- **Why:** Same motivation as pressure asinh — velocity has steep boundary-layer gradients dominating L1 loss

## Dead Ends This Session (2026-04-03)

| Approach | Finding |
|----------|---------|
| srf4L (4-layer refine) | p_tan +5-7% WORSE (12 seeds confirmed) |
| MC Dropout | Null result — no consistent improvement |
| Packed Ensemble M=4,8 | p_re +6-7% WORSE — insufficient capacity |
| Synthetic data interpolation | 3-10x worse — physically invalid for unstructured meshes |
| Asymmetric loss α=0.5-1.0 | Tradeoff: p_tan -4% but p_in/p_oodc +4% |

## Confirmed Dead Ends (Phase 6, all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
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
| MC Dropout | #2088 | Null result |
| Packed Ensemble | #2082 | Model too small |
| Ensemble Weight Opt | #2089 | Equal weights — no benefit from non-uniform weighting |

## Key Research Insights

1. **Ensemble variance reduction is the biggest lever** — keep growing the pool
2. **Different seed batches produce comparable 8-model ensembles** (~12.2-12.4 p_in)
3. **16-seed ensemble should improve further** (1/sqrt(16) vs 1/sqrt(8) = 29% more reduction)
4. **srf4L is a confirmed dead end** — extra surface refine layers hurt tandem transfer
5. **Model too small for packed ensembles** — separate training + averaging strictly better
6. **Architecture modifications absorbed by EMA+Lion** — incremental changes don't stick
7. **Checkpoint persistence** — student pods don't retain checkpoints across sessions; W&B artifacts needed for cross-node ensemble eval

## Human Team Directives (from issues #1860, #1834, #1926)
- All ideas from issue #1926 tested (HyperP, MSA, mHC — deprioritized as most exploratory)
- Data constraint from #1834 respected
- Radical architectures tried and failed — ensemble strategy now dominant
