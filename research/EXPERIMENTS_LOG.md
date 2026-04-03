# SENPAI Research Results

## Phase 6 Experiments (2026-04-01 onwards)

### New Assignments (2026-04-03)

#### 2026-04-03 — PR #2092: Phase 6: Ensemble Seeds 82-89 — thorfinn — NEW
- Branch: `thorfinn/ensemble-seeds-82-89`
- Hypothesis: More standard 3L seeds for ensemble expansion (total pool: 42-49, 66-73, 74-81, 82-89, 90-95)
- Status: WIP — just assigned

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 — tanjiro — **MERGED WINNER**
- Branch: `tanjiro/ensemble-more-seeds`
- W&B group: phase6/ensemble-more-seeds (8 runs finished)
- Results: **NEW BEST.** 8-seed ensemble (66-73) beats prior best (42-49):
  - p_in: 12.2 (was 12.4, **-1.6%**)
  - p_oodc: 6.7 (flat)
  - p_tan: 29.1 (was 29.4, **-1.0%**)
  - p_re: 5.8 (flat)
- Run IDs: j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv
- **MERGED** — Updated baseline. Next: 16-seed combined evaluation.

#### 2026-04-03 — PR #2082: Phase 6: Packed Ensemble — thorfinn — CLOSED
- Branch: `thorfinn/packed-ensemble`
- W&B group: phase6/packed-ensemble (8 runs finished)
- Results: **DEAD END.** Model too small for packed ensembles.
  - M=2: marginal, ~flat vs baseline
  - M=4: p_re +6.4%, p_oodc +2.2% worse
  - M=8: p_re +7.1%, p_oodc +3.6% worse
- Root cause: n_hidden=192 split across M sub-models → insufficient per-sub-model capacity
- Conclusion: At 1.7M params, separate model training + post-hoc averaging strictly dominates. CLOSED.

#### 2026-04-03 — PR #2068: Phase 6: Asymmetric/Magnitude-Weighted Surface Loss — alphonse — SENT BACK
- Branch: `alphonse/asymmetric-loss`
- W&B group: phase6/asymmetric-loss (8 runs finished)
- Results: Mixed tradeoff. magweight α=1.0 improves p_tan -4.3% but worsens p_in +3.7%, p_oodc +3.8%.
  - α=0.5: p_tan -2.0%, other metrics +0.5-1.5% worse
  - α=1.0: p_tan -4.3%, p_in +3.7%, p_oodc +3.8% (tradeoff too steep)
  - pinball τ=0.45: p_tan -2.1%, p_in +2.0%, p_oodc +3.7% (worse)
- Conclusion: Magnitude weighting shifts gradient from OOD to tandem transfer. SENT BACK for milder α sweep (0.1-0.3).

#### 2026-04-03 — PR #2088: Phase 6: MC Dropout in Surface Refine Head — nezuko — CLOSED
- Branch: `nezuko/mc-dropout-surface-refine`
- W&B group: phase6/mc-dropout (8 runs finished)
- Results: NULL RESULT. No consistent improvement across configs.
  - p=0.05 K=16: flat vs baseline (within noise)
  - p=0.05 K=8: mixed (p_in +2.8%, p_re -2.3%)
  - p=0.10 K=8: tradeoff (p_oodc -3.1%, p_re -3.5% BUT p_tan +3.5%)
- Conclusion: MC Dropout stochasticity insufficient for meaningful variance reduction. EMA already provides implicit regularization. CLOSED.

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 — tanjiro — RESULTS IN
- Branch: `tanjiro/ensemble-more-seeds`
- W&B group: phase6/ensemble-more-seeds (8 runs finished)
- Individual model metrics consistent with 8-seed baseline:
  - p_in=13.06±0.34, p_oodc=7.88±0.17, p_tan=30.29±0.63, p_re=6.42±0.09
- Pending: 16-seed ensemble evaluation (combining seeds 42-49 + 66-73)
- Status: Awaiting ensemble eval from student

#### 2026-04-03 — PR #2086: Phase 6: SAM Phase-Only — Flat Minima for OOD Generalization
- Branch: `frieren/sam-phase-only`
- Student: frieren
- Hypothesis: SAM (already in train.py) applied only in last 25% of training (epoch≥120) finds flatter minima. Sweep rho={0.05, 0.1, 0.2} since Lion's sign updates may need different scaling than Adam. Target: reduce seed variance AND improve p_oodc/p_re.
- 8-GPU: 2 baselines (s42/s43) + 3 rho values × 2 seeds = 8 runs
- Status: WIP — assigned

#### 2026-04-03 — PR #2085: Phase 6: srf4L Ensemble Seeds 58-65 (3rd Batch) — nezuko — CLOSED
- Branch: `nezuko/srf4L-ensemble-v3`
- Hypothesis: More srf4L seeds for ensemble
- CLOSED — srf4L hypothesis invalidated (p_tan +5-7% worse per #2079, #2083)

#### 2026-04-03 — PR #2083: Phase 6: srf4L Ensemble Seeds 50-57 — fern — CLOSED
- Branch: `fern/srf4L-ensemble-v2`
- Hypothesis: srf4L seeds 50-57 for ensemble expansion
- W&B group: phase6/srf4L-ensemble-v2 (8 runs finished)
- Results: **DEAD END.** Confirms srf4L regression:
  - p_in: 13.20 (+1.3%), p_oodc: 8.06 (+2.9%), p_tan: **32.43 (+7.0% WORSE)**, p_re: 6.43 (-0.3%)
- CLOSED — srf4L hypothesis invalidated

#### 2026-04-03 — PR #2082: Phase 6: Packed Ensemble — thorfinn
- Branch: `thorfinn/packed-ensemble`
- Hypothesis: M sub-models packed into one forward pass via grouped linear ops. Test M=2,4,8. From Laurent et al. 2023, AirfRANS shows 25% training speedup over deep ensembles.
- 8 GPUs: 2 baseline + 2×M=2 + 2×M=4 + 2×M=8
- Status: WIP, 8 runs running

#### 2026-04-03 — PR #2081: Phase 6: srf4L + 8-Seed Ensemble (Compound) — edward — CLOSED
- Branch: `edward/srf4L-ensemble`
- Hypothesis: srf4L ensemble should beat 3L ensemble
- CLOSED — srf4L hypothesis invalidated (p_tan +5-7% worse per #2079, #2083)

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 (Building Toward 24-Model) — tanjiro
- Branch: `tanjiro/ensemble-more-seeds`
- Hypothesis: Extend ensemble to 24 models (seeds 42-73). If ensemble variance reduction scales with N, 24-model could give another step down.
- Status: WIP, 8 crashed + 8 running (relaunched)

#### 2026-04-03 — PR #2079: Phase 6: srf4L on Asinh Baseline (Multi-Seed) — askeladd — CLOSED
- Branch: `askeladd/srf4L-asinh`
- Hypothesis: Test srf4L (surface_refine_layers=4 vs 3) as a standalone improvement
- W&B group: phase6/srf4L-multiseed (8 runs finished)
- Results: **DEAD END.** srf4L consistently WORSE than 3L baseline:
  - p_in: 13.29 vs 13.30 (flat)
  - p_oodc: 7.83 vs 7.78 (+0.7%)
  - p_tan: **32.01 vs 30.33 (+5.5% WORSE)**
  - p_re: 6.53 vs 6.46 (+1.1%)
- Conclusion: 4th surface refine layer overfits, damages tandem transfer. CLOSED

#### 2026-04-03 — PR #2068: Phase 6: Asymmetric/Magnitude-Weighted Surface Loss — alphonse
- Branch: `alphonse/asymmetric-loss`
- Hypothesis: Replace symmetric L1 with magnitude-weighted L1 (overweight stagnation/suction peaks) and pinball tau=0.45 (penalize underprediction)
- 8 GPUs: 2 baseline + magnitude-weighted α=0.5 + α=1.0 + pinball τ=0.45 × 2 seeds each
- Status: WIP, 8 runs running

#### 2026-04-03 — PR #2066: Phase 6: Synthetic Data Generation via Physics-Consistent Interpolation — frieren
- Branch: `frieren/data-gen-interpolation` (CLOSED)
- Hypothesis: Interpolate between same-condition CFD samples to create synthetic training data
- Results: DEAD END. All metrics 3-10x worse than baseline. Root cause: mesh-level interpolation is physically meaningless for unstructured CFD (no spatial correspondence between node indices across different meshes)
- W&B runs: xjo7mj7s, 5dggj8av, ysg5zqr4, nandekgh, jr5qg1i1, s3lt4tjd, 5d828zol, 1n8k5m0g
- Status: CLOSED

### New Assignments (2026-04-02)

#### 2026-04-02 — PR #2008: Phase 6: PirateNets — Random Weight Factorization
- Branch: `frieren/pirate-nets`
- Student: frieren
- Hypothesis: Apply Random Weight Factorization (RWF) from PirateNets to MLP layers. W = diag(exp(s)) * V where V is fixed random, s is learned. Enables multi-scale frequency learning for better boundary layer / pressure prediction.
- 8-GPU sweep: baseline × 2, RWF uniform × 2, RWF multiscale × 2, RWF multiscale + higher LR × 2
- Status: WIP — assigned

#### 2026-04-02 — PR #2009: Phase 6: Geosolver — Geometry-Aware Feature Encoding
- Branch: `alphonse/geosolver`
- Student: alphonse
- Hypothesis: Add explicit geometric features (wall distance, curvature, normals) as inputs + geometry-conditioned attention bias. Physics-geometry coupling directly encoded.
- 8-GPU sweep: baseline × 2, geo features × 2, geo + attention × 2, geo + attn + higher surf_weight × 2
- Status: WIP — assigned

#### 2026-04-02 — PR #2010: Phase 6: HeavyBall Optimizers
- Branch: `nezuko/heavyball-optimizers`
- Student: nezuko
- Hypothesis: Sweep modern optimizers (SOAP, CauchyAdamW, Schedule-Free AdamW, PaLM-SOAP) as alternatives to Lion. Different optimizers handle CFD's extreme gradient structure differently.
- 8-GPU sweep: SOAP × 2 LR, CauchyAdamW × 2 LR, Schedule-Free × 2 LR, PaLM-SOAP, HB-AdamW
- Status: WIP — assigned

### Previously Assigned

#### 2026-04-01 23:25 — PR #2006: Phase 6: Muon Optimizer + Gram-NS
- Branch: `phase6/muon-optimizer`
- Student: thorfinn
- Hypothesis: Replace Lion with Muon (Newton-Schulz orthogonalized gradients).
- **Status: ALL 8 RUNS CRASHED** — zero metrics logged. Code bug. Advisor commented with fix guidance.

#### 2026-04-01 23:25 — PR #2007: Phase 6: XSA (Exclusive Self-Attention)
- Branch: `phase6/xsa-attention`
- Student: askeladd
- Hypothesis: Replace slice attention with XSA exclusion mechanism.
- **Status: ALL 8 RUNS CRASHED.** Catastrophically bad metrics (val/loss 3.76-10.6 vs baseline 0.383). Code regression. Advisor commented with fix guidance.

  | Run | val/loss | p_in | Status |
  |-----|----------|------|--------|
  | xsa-t0.5-s42 | 3.761 | 296.4 | crashed |
  | baseline-s42 | 3.830 | 299.6 | crashed |

---

## Phase 5 Experiments

### Completed (2026-04-02)

#### PR #2003: LR Schedule Sweep — **MERGED WINNER**
- Student: edward | `cosine_T_max=160` beats baseline on ALL metrics
- val/loss: 0.3761 (−1.8%) | p_in: 12.5 (−3.5%) | p_oodc: 8.2 (−1.3%) | p_tan: 29.8 (−0.7%) | p_re: 6.5 (−3.0%)
- W&B: `9ysz96ll` | **MERGED** — New baseline. Follow-up: multi-seed (#2012)

#### PR #2001: Learned Per-Channel Loss Weighting — **CLOSED (mixed)**
- Student: tanjiro | p_tan −1.7% but p_oodc +3.8%, p_re +4.5%
- Key insight: pressure deserves 1.6x weight (sigma=0.087 vs 0.109-0.111 velocity)
- **CLOSED** → Follow-up: hybrid learned p-weight (#2013)

#### PR #2004: Noise Schedule Sweep — **CLOSED (dead end)**
- Student: fern | all stuck at val/loss 0.66-0.68, far above 0.383
- **CLOSED** → Follow-up: PirateNet surface gate (#2014)

### Closed (Dead Ends)
- PR #1998: Multi-Exit Ensemble (frieren) — **CLOSED.** All 8 crashed, val/loss 4.0-7.5. Auxiliary losses destabilized training.
- PR #2002: EMA Decay Sweep (nezuko) — **CLOSED.** All 8 crashed, val/loss 1.1. Runs terminated too early.
- PR #2000: OOD-Focused Training (alphonse) — **CLOSED.** All 8 crashed, val/loss 4.0-11.5. Hard-mining diverged.

### Winners (Merged — Phase 5)
1. **PR #1927: Residual Prediction** — predict correction from freestream. p_oodc -4.7%, p_tan -1.9%
2. **PR #1935: Surface Refinement Head** — dedicated surface MLP. p_re -72.7%, p_tan -8.9%, val/loss -3.3%
3. **PR #2003: cosine_T_max=160** — faster LR decay. All metrics improved. val/loss -1.8%, p_in -3.5%, p_re -3.0%

### Key Non-Winners (Closed — Phase 5)
- 50+ Phase 5 experiments explored: throughput optimization, data augmentation, architecture variants, loss formulations, ensemble methods, curriculum learning, distillation
- Most incremental tuning exhausted — motivates Phase 6 shift
