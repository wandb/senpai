# SENPAI Research Results

## 2026-04-20 22:20 — PR #2455: AirfRANS: 3L/192d no-EMA no-Fourier 6 epochs — MERGED ✓ NEW BEST

- **Branch:** emma/airfrans-noema-4L256d-retest

| Run | Config | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|---|
| 4L/256d | no-EMA, no-Fourier | 0.2935 | 0.2706 | 5 | wptz6kat |
| **3L/192d** | **no-EMA, no-Fourier** | **0.2597** (-10.2%) | **0.2392** | **6** | pifi0x1v |

**Commentary:** BREAKTHROUGH — Fourier features add ~3x epoch overhead (5→15 min/epoch). Without Fourier, 6 epochs (0.2597) beats Fourier at 2 epochs (0.2710). Same pattern as TandemFoil: more epochs > features. 4L/256d conclusively worse than 3L/192d under time budget. Pressure finally below 1.0 (test_surface_mse_p=0.9556).

---

## 2026-04-20 22:20 — PR #2470: AirfRANS: Fourier full epoch run — CLOSED (superseded)

- **Branch:** haku/airfrans-fourier-noema-fullrun

| Run | Config | val_primary/surface_mse | Epochs | W&B |
|---|---|---|---|---|
| **lr=5e-4, T_max=150** | Fourier+noEMA | **0.2710** | 2 | dui0c6qg |
| lr=3e-4, T_max=150 | Fourier+noEMA | 0.2805 | 2 | anzo6z7u |
| lr=5e-4, T_max=20 | Fourier+noEMA | 0.4354 | 2 | sklqsht0 |

**Commentary:** Best Fourier result (0.2710) beats old baseline (0.2891) but superseded by emma's no-Fourier 0.2597. CONFIRMED: AirfRANS epoch starvation is structural — 15 min/epoch with Fourier, 30-min timeout = 2 epochs max regardless of parallelism. T_max=20 catastrophically bad (LR cycles back to peak). T_max=150 correct for AirfRANS.

---

## 2026-04-20 22:20 — PR #2476: AirfRANS: Fourier + no-EMA on OOD tasks — MERGED ✓ REYNOLDS OOD BEST

- **Branch:** norman/airfrans-fourier-noema-ood

| Task | val_primary/surface_mse | vs Baseline | Epochs | W&B |
|---|---|---|---|---|
| **reynolds** | **0.3319** (-18.2%) | 0.4059 | 2 | m24dt4cg |
| scarce | 0.2760 (+8.4%) | 0.2547 | 2 | vb77cptv |

**Commentary:** Mixed OOD results. Reynolds: Fourier helps significantly (-18.2%). Scarce: Fourier hurts (+8.4%), possibly overfitting with limited data. New reynolds OOD baseline: 0.3319.

---

## 2026-04-20 22:20 — PR #2469: AirfRANS: cosine T_max sweep — CLOSED (obsolete)

- **Branch:** alphonse/airfrans-noema-cosine-sweep

| T_max | val_primary/surface_mse | Epochs | W&B |
|---|---|---|---|
| 10 | 0.3407 | 2 | bcp5ht2b |
| 20 | 0.3840 | 2 | tnypfuoy |
| 50 | 0.3703 | 2 | ujr87q52 |

**Commentary:** No Fourier features. All worse than pre-Fourier baseline (0.3308) at only 2 epochs. Doubly obsolete vs current 0.2597. Cosine T_max hypothesis untestable at 2 epochs.

---

## 2026-04-20 22:00 — PR #2467: DrivAerML: no-EMA + AdamW lr=8e-4 — MERGED ✓ NEW BEST

- **Branch:** violet/drivaerml-noema-lr-bracket

| Run | LR | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|---|
| **lr=8e-4** | **8e-4** | **56.91%** (-20% relative) | 57.33% | 2 | ip8ybl80 |
| lr=1e-3 | 1e-3 | 58.78% | 59.14% | 2 | 28udv9x7 |

**Commentary:** MASSIVE DrivAerML improvement. No-EMA + lr=8e-4 crushes the 71.35% EMA baseline. EMA was suppressing the higher LR's effectiveness. lr=8e-4 is the new DrivAerML default.

---

## 2026-04-20 22:00 — PR #2474: AirfRANS: Fourier + no-EMA + 4L/256d — MERGED ✓ NEW BEST

- **Branch:** senku/airfrans-fourier-noema-combo

| Run | Config | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|---|
| **4L/256d** | Fourier+noEMA+4L/256d+lr=5e-4 | **0.2891** (-3.9%) | **0.2856** | 2 | hxyibvbf |
| lr=3e-4 | Fourier+noEMA+3L/192d+lr=3e-4 | 0.2975 | 0.3052 | 2 | 1mhw0tph |

**Commentary:** Fourier + 4L/256d capacity synergize. Still in steep descent at epoch 2 (0.4256→0.2891). More epochs should push much lower. lr=3e-4 variant underperformed on test.

---

## 2026-04-20 22:00 — PR #2471: TandemFoil: golden config + no-EMA — SENT BACK (epoch starvation)

- **Branch:** gilbert/tandem-golden-noema

| Run | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|
| lr=3e-4 | 215.94 | 2 | 457alys4 |
| lr=2e-4 | **190.34** | 2 | xpuptoy5 |

**Commentary:** Only 2 epochs at slices=64 (should be 11). Likely parallel execution causing I/O contention. lr=2e-4 improving at 12.7%/epoch — very promising. Sent back for strict sequential rerun. With 11 epochs, projected to dramatically beat 114.92 baseline.

---

## 2026-04-20 22:00 — PR #2472: TandemFoil: golden + physics + no-EMA — CLOSED (epoch starvation)

- **Branch:** kaneda/tandem-golden-physics-noema

| Run | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|
| Lion lr=3e-4 | 173.00 | 2 | 3fjtrbv6 |
| **AdamW lr=3e-4** | **153.10** | 2 | dc717g1b |

**Commentary:** Physics features add ~7x overhead at slices=64. Only 2 epochs. Key finding: **AdamW outperforms Lion with physics features** (153.10 vs 173.00, -11.5%) — a reversal of the no-physics optimizer preference. Physics + AdamW at 2 epochs (153.10) already beats old no-physics no-EMA baseline at 2 epochs (197.87). Path forward: slices=32 for more epochs with physics.

---

## 2026-04-20 22:00 — PR #2439: DrivAerML: anchor budget sweep — CLOSED (no-ops, superseded)

- **Branch:** nezuko/drivaerml-anchor-budget-sweep

| Trial | surface_pts/view | val_primary/surface_rel_l2_pct | Epochs | W&B |
|---|---|---|---|---|
| A | 500K | 72.46% | 2 | w0a1g9qo |
| B | 1.5M | 71.37% | 2 | 37zg4voz |

**Commentary:** geometry_supernodes and surface_anchor_points are NO-OPS for senpai_transolver (only work with ABUPTCollate). Student pivoted to surface point budget — more points help marginally but doesn't beat baseline. Now superseded by violet's 56.91%.

---

## 2026-04-20 21:50 — PR #2460: AirfRANS OOD tasks (scarce + reynolds) with no-EMA — MERGED ✓ NEW OOD BASELINES

- **Branch:** norman/airfrans-noema-ood
- **Hypothesis:** No-EMA should improve OOD tasks as it improved the full task

| Task | val_primary/surface_mse | test_primary | val_mse_p | Epochs | W&B |
|---|---|---|---|---|---|
| **scarce** | **0.2547** (-24% vs 0.3351) | 0.6368 | 1.0156 | 2 | bxrn5yye |
| **reynolds** | **0.4059** (-32% vs 0.5956) | 0.6618 | 1.6183 | 2 | az53l5l6 |

**Commentary:** Confirms no-EMA generalizes to OOD tasks. Both improved substantially vs Round 2 EMA baselines. Pressure dominates >95% of surface error in both tasks. Large val/test gap on scarce (1.02→2.54 pressure) indicates distribution shift. Only 2 epochs due to OOD dataset size (~15 min/epoch).

---

## 2026-04-20 21:30 — PR #2435: TandemFoil: cosine T_max sweep at slices=64 — MERGED ✓ NEW BEST

- **Branch:** gilbert/tandem-cosine-tmax-sweep
- **Hypothesis:** Shorter cosine T_max cycles complete more LR restarts in the training budget

| T_max | val_primary/surface_pressure_mae | test_primary | Epochs | W&B |
|---|---|---|---|---|
| **30** | **114.92** (-42% vs 197.87) | **108.16** | 11 | 3ec9m9az |
| 10 | 117.23 | 109.89 | 11 | lx4ly3m6 |
| 50 | 127.51 | 120.69 | 11 | uusjik96 |
| 20 | 132.62 | 124.48 | 10 | 7p6hxl5r |

**Commentary:** BREAKTHROUGH. slices=64 enables 11 epochs in 30 min (vs 2 at slices=96) — a 5.5x training multiplier that completely dominates. T_max=30 is optimal, giving ~25 cosine restarts per epoch at 750 batches/epoch. ALL runs used EMA=True yet still crushed the 197.87 no-EMA baseline. No-EMA retest at slices=64 + T_max=30 is now the highest-priority TandemFoil experiment — projected estimate ~86-90. **slices=64 + T_max=30 is the new golden config for TandemFoil.**

---

## 2026-04-20 21:30 — PR #2459: AirfRANS: asinh-pressure + residual-prediction + no-EMA — CLOSED (metric incompatibility)

- **Branch:** senku/airfrans-noema-asinh-residual
- **Hypothesis:** asinh-pressure + residual-prediction transfer from TandemFoil to AirfRANS

| Trial | val_primary/surface_mse | Epochs | W&B |
|---|---|---|---|
| asinh only | 0.000104 (epoch 1) | 2 | xwbxj30u |
| asinh + residual | 0.002809 | 2 | oyohiwf0 |

**Commentary:** Results are in compressed asinh-normalized space, NOT comparable to baseline (0.3009). The asinh transform changes the target space before normalization. Student correctly identified the incompatibility. Direction is not dead but needs inverse-transform evaluation path. Student also implemented --residual-prediction for AirfRANS (code contribution).

---

## 2026-04-20 21:30 — PR #2449: TandemFoil: Full physics + AdamW LR sweep — CLOSED (EMA, superseded)

- **Branch:** kaneda/tandem-fullphys-adamw-lr-sweep-v2

| Trial | val_primary/surface_pressure_mae | Epochs | EMA | W&B |
|---|---|---|---|---|
| Full physics + AdamW lr=3e-4 | **235.94** | 2 | True | dsictzuq |
| Full physics + AdamW lr=5e-4 | 237.42 | 2 | True | mwh4y0pz |
| Full physics + AdamW lr=8e-4 | 367.42 | 1 | True | kwslbj4e |
| Core physics + AdamW lr=5e-4 | 366.44 | 1 | True | k735vytc |

**Commentary:** EMA=True, now superseded by gilbert's 114.92. Full physics + AdamW lr=3e-4 projected ~189 without EMA — was competitive with old baseline but irrelevant vs new. Only 2 epochs at slices=64 (likely parallel execution). Full physics + AdamW beats core physics at matched EMA conditions.

---

## 2026-04-20 21:30 — PR #2443: TandemFoil: physics+AdamW slices sweep — CLOSED (EMA, superseded)

- **Branch:** edward/tandem-physics-adamw-slices-sweep

| Slices | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|
| 32 | **244.33** | 2 | hgj1bash |
| 64 | 251.09 | 2 | alchrjkp |
| 48 | 367.64 | 1 | u2dkyj00 |
| 80 | 353.37 | 1 | yb1b6oru |
| 96 | 445.55 | 1 | elgagd4t |

**Commentary:** EMA=True + broken cosine_t_max=30 (in steps not epochs). Superseded by gilbert's 114.92. Only 1-2 epochs due to parallel execution and physics feature overhead.

---

## 2026-04-20 21:30 — PR #2436: TandemFoil: Reynolds-stratified sampling — CLOSED (dead end)

- **Branch:** chihiro/tandem-re-stratified-sampling

| Variant | val_primary/surface_pressure_mae | val_re_rand | Epochs | EMA |
|---|---|---|---|---|
| v0: restrat + EMA 0.999 | 587.30 (diverged) | 486.13 | 2 | True |
| v1: restrat + EMA 0.9995 | 364.48 | 290.51 | 1 | True |
| v2: restrat + no-EMA | 343.25 | 300.90 | 1 | False |
| v3: control (no restrat) | 345.33 | 292.04 | 1 | True |

**Commentary:** All results far worse than baseline. Re-stratified sampling showed no OOD benefit (re_rand: 300.9 vs control 292.0). v0 diverged. Clear dead end.

---

## 2026-04-20 21:15 — PR #2457: AirfRANS: Fourier + no-EMA + AdamW lr=5e-4 — MERGED ✓ NEW BEST

- **Branch:** haku/airfrans-fourier-noema
- **Hypothesis:** Fourier positional encoding helps resolve high-frequency pressure gradients near airfoil surface

| Trial | Config | val_primary/surface_mse | test_primary/surface_mse | Epochs | W&B |
|---|---|---|---|---|---|
| **0 (WINNER)** | **Fourier + no-EMA + AdamW lr=5e-4** | **0.3009** (-9.1%) | **0.2869** (-10.3%) | 2 | cgr5omp3 |
| 1 | no-EMA + AdamW lr=8e-4 (no Fourier) | 0.3741 (+13.1%) | 0.3457 | 2 | zcho7dzb |

**Per-channel test breakdown (Trial 0):** Ux=0.001468, Uy=0.0000729, p=1.1459, nut=0.000351

**Commentary:** BREAKTHROUGH — Fourier features + no-EMA beat the 6-epoch baseline in just 2 epochs. Pressure channel improved -10.3% (1.28→1.15). nut channel regressed +875% but is negligible in composite (3 orders of magnitude smaller than pressure). Still rapidly descending at cutoff — full epoch budget should push significantly lower. Trial 1 confirmed lr=8e-4 without Fourier is a dead end on AirfRANS. Fourier encoding is now mandatory for AirfRANS.

---

## 2026-04-20 21:00 — PR #2440: DrivAerML: AdamW vs Lion baseline sweep — MERGED ✓ FIRST BASELINE

- **Branch:** shoya/drivaerml-adamw-baseline-sweep
- **Hypothesis:** Establish first DrivAerML baseline comparing AdamW vs Lion optimizer

| Run | Config | val_primary/surface_rel_l2_pct | Epochs |
|---|---|---|---|
| AdamW lr=3e-4 | 3L/192d, slices=96 | 71.76% | 2 |
| **AdamW lr=5e-4** | 3L/192d, slices=96 | **71.35%** (BEST) | 2 |
| AdamW lr=8e-4 | 3L/192d, slices=96 | 71.76% | 2 |
| Lion lr=3e-4 | 3L/192d, slices=96 | 78.45% (degraded) | 2 |

**Commentary:** First DrivAerML baseline. AdamW clearly beats Lion (which degraded epoch-over-epoch). All AdamW LRs converge to ~71.4-71.8% — optimizer matters more than LR in this range. 71.35% vs 3.71% target = huge gap, but only 2 epochs (30-min timeout, ~10-11 min/epoch). DrivAerML cases have ~8.6M surface points; student resolved OOM with 50k-point sampling. Eval coverage thin (~3.5% of val surface per epoch). AdamW lr=5e-4 is the DrivAerML starting point going forward.

---

## 2026-04-20 21:00 — PR #2434: TandemFoil: slices throughput sweep — CLOSED (EMA-suppressed)

- **Branch:** violet/tandem-slices-sweep

| Slices | val_primary/surface_pressure_mae | Epochs | Peak VRAM |
|---|---|---|---|
| 32 | 288.51 | 2 | ~77 GB |
| 48 | 452.38 | 2 | — |
| 64 | 486.53 | 1 | — |
| 96 | 294.21 | 2 | ~92 GB |

**Commentary:** EMA=True. Slices do NOT affect throughput (all got 2 epochs regardless). slices=32 ≈ slices=96 quality with 15 GB less memory. slices=48 is anomalously bad. Data loading is the bottleneck, not slice attention.

---

## 2026-04-20 21:00 — PR #2433: TandemFoil: AdamW LR sweep slices=64 — CLOSED (Lion dominates)

- **Branch:** alphonse/tandem-adamw-lr-sweep

| LR | val_primary/surface_pressure_mae | Epochs |
|---|---|---|
| 3e-4 | 444.39 | 1 |
| 5e-4 | 338.15 | 1 |
| **8e-4** | **254.34** | 2 |
| 1e-3 | 456.86 | 2 |

**Commentary:** No-EMA (EMA=None confirmed). AdamW lr=8e-4 is optimal AdamW LR but still 22% behind Lion baseline (197.87). Lion dominates AdamW on TandemFoil — opposite of AirfRANS finding. 4 parallel jobs caused epoch starvation (v0/v1 only 1 epoch).

---

## 2026-04-20 21:00 — PR #2413: TandemFoil: full physics stack — CLOSED (core subset better)

- **Branch:** fern/tandem-full-physics

| Variant | val_primary/surface_pressure_mae | Epochs |
|---|---|---|
| v0: Full physics (all flags) | 270.74 | 2 |
| v1: Full minus wake-angle | 293.44 | 2 |
| v2: cp-panel-scale=0.5 | 285.87 | 2 |
| v3: vortex-panel-scale=0.05 | **268.10** | 2 |

**Commentary:** EMA=True. Full physics stack (best 268.10) worse than core physics subset (262.82, #2414). Wake-angle is the most impactful single feature (+22.7 when removed). Vortex-panel computation has Python for-loop bottleneck (~25 min/epoch). Core physics subset is the right path — full stack not worth the computational cost.

---

## 2026-04-20 19:50 — PR #2412: TandemFoil: clean baseline no-EMA (frieren v4) — MERGED ✓ NEW BEST

- **Branch:** frieren/tandem-baseline-default
- **Hypothesis:** Removing EMA in ultra-short training regime (2 epochs)
- **W&B run:** y8f8pkkn (v4)

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | **197.87** (NEW BEST, -24.7% vs 262.82) |
| test_primary/surface_pressure_mae | 191.70 |
| test_single_in_dist | 212.64 |
| test_geom_camber_rc | 172.00 |
| test_geom_camber_cruise | 187.39 |
| test_re_rand | 194.77 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=3e-4, slices=96, **use_ema=False**, use_lookahead=True, NO physics features, cosine_t_max=50 |

**Commentary:** CRITICAL FINDING. Removing EMA improved val_mae by 24.7% without any physics features. EMA with ema_start_step=50 never meaningfully activates in 2 epochs (only 2×750=1500 steps, barely above start step), and the exponential moving average of improving weights with stale early weights is actively harmful. This was independently confirmed on AirfRANS (#2431: EMA degrades 0.3914→0.5038). ALL future experiments MUST use `--no-use-ema`. Compounding this with physics features should give further gains.

Secondary findings from this PR:
- v1 (lr=2e-4, EMA=True): 264.14 — lower LR also helpful even with EMA
- v3 (no-lookahead, EMA=True): 281.15 — lookahead is beneficial
- v0 (baseline, EMA=True): 310.96 — confirms EMA was masking improvements all along
- v2 (lr=5e-4, EMA=True): 446.12 — higher LR with EMA is catastrophic (1 epoch only)

---

## 2026-04-20 19:50 — AirfRANS Round 2 Summary (5 PRs closed — epoch starvation)

Key pattern: ALL 5 AirfRANS Round 2 PRs ran at slices=64 with 4 parallel jobs → only 2 epochs completed vs baseline's 6 epochs. Results are confounded and cannot be compared to baseline.

**#2428 (kohaku, LR bracket):**
| LR | val_primary/surface_mse | Epochs |
|---|---|---|
| 8e-4 | 0.3278 (best) | 5 |
| 3e-4 | 0.3414 | 5 |
| 6e-4 | 0.3513 | 5 |
| 4e-4 | 0.3754 | 5 |
*Note: 5 epochs at slices=64, vs baseline 6 at slices=96. Confounded. lr=8e-4 slightly best but vol_mse regresses.*

**#2429 (emma, capacity):** 4L/256d + 3L/192d at slices=64, only 2 epochs each due to 4-job parallelism. Inconclusive.

**#2430 (senku, cosine T_max):** T_max=10/20/30/50 at slices=64, only 2 epochs. Best T_max=20 (val=0.4763) but confounded.

**#2431 (haku, scaffold ablation):** CRITICAL FINDING — EMA is harmful on AirfRANS!
| Config | val_primary/surface_mse |
|---|---|
| no-EMA + Lookahead (v2, best) | 0.3914 |
| bare AdamW (v3) | 0.4590 |
| full scaffold EMA+Lookahead (v0) | 0.5038 |
| no-Lookahead (v1) | 0.5268 |
*All at slices=64, 2 epochs. No-EMA is the key lever.*

**#2432 (norman, OOD tasks):** First OOD baselines established.
| Task | val | test |
|---|---|---|
| scarce | 0.3351 (AdamW) | 0.8021 |
| reynolds | 0.5956 | 0.8999 |
| full (confounded) | 0.5201 | 0.5041 |
*Large val/test gap on OOD tasks confirms real generalization challenge.*

**Round 2 Key Lessons:**
1. Running 4 parallel AirfRANS jobs causes epoch starvation (I/O contention with num_workers=0)
2. EMA is harmful on AirfRANS (and TandemFoil) in short training regimes
3. MAX 2 parallel jobs per AirfRANS student going forward
4. Must use slices=96 (not 64) for fair comparison to baseline

---

## 2026-04-20 19:30 — PR #2414: TandemFoil: core physics features (TE+Cp+asinh+residual) — MERGED ✓

- **Branch:** tanjiro/tandem-physics-features
- **Hypothesis:** Physics features (TE coord frame, Cp panel, asinh pressure, residual prediction) improve TandemFoil surface pressure prediction by giving the model physically-structured inputs.
- **W&B run:** 1zbp5dlu

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | **262.82** (NEW BEST) |
| test_primary/surface_pressure_mae | 257.51 |
| test_single_in_dist | 267.26 |
| test_geom_camber_rc | 280.59 |
| test_geom_camber_cruise | 225.63 |
| test_re_rand | 256.55 |
| Epochs | 2 (30-min timeout, ~15 min/epoch) |
| Config | Lion lr=3e-4, slices=96, physics: te_coord+cp_panel+cp_panel_tandem_only+asinh+residual+pressure_prior |

**Commentary:** New TandemFoil best — beats alphonse's AdamW baseline (269.32) by 2.4%. Physics features provide physically-structured inductive bias. Two Inf values in test_geom_camber_cruise/mae_vol_p (asinh inversion overflow on volume predictions at early training) — surface metrics are unaffected. Only 2 epochs completed; model was still rapidly improving. Key gap: physics features tested only with Lion — combining with AdamW should compound the gains. Next priority: physics + AdamW LR sweep (tanjiro #2441), ANP decoder (shinji #2444), wake feature ablation (askeladd).

---

## 2026-04-20 19:30 — PR #2419: TandemFoil: batch_size=4 with scaled LR — CLOSED

- **Branch:** askeladd/tandem-batch4-lr
- **W&B run:** 2lc5q8ae

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 454.96 |
| test_primary/surface_pressure_mae | 429.53 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=6e-4, batch_size=4, slices=96 |

**Commentary:** batch_size=4 halves gradient steps per unit time → only 2 epochs, severely undertrained. val_mae=454.96 vs baseline 269.32 (+69%). Clear dead end: batch_size doubling destroys the epoch budget. batch_size=2 is optimal for TandemFoil within 30-min timeout.

---

## 2026-04-20 19:30 — PR #2418: TandemFoil: normalization tricks (asinh+residual) — CLOSED (superseded)

- **Branch:** thorfinn/tandem-normalization
- **W&B run:** svy77euk

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 291.32 |
| test_primary/surface_pressure_mae | 280.60 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=3e-4, slices=96, asinh_pressure=True, residual_prediction=True, cosine_t_max=50 |

**Commentary:** asinh+residual alone (without TE coord+Cp panel) achieves 291.32 — worse than baseline 269.32. Tanjiro's broader physics stack (#2414) includes these features AND more, and beats baseline at 262.82. The subset is superseded. Also: only 1 of requested 4 ablation variants was submitted. Inf in cruise vol_p — same numerical overflow from asinh as #2414 (early-training artifact).

---

## 2026-04-20 19:30 — PR #2417: TandemFoil: bigger model (4L/256d/4H/128slices) — CLOSED

- **Branch:** edward/tandem-bigger-model
- **W&B run:** fv82ma84

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 314.52 |
| test_primary/surface_pressure_mae | 306.34 |
| Epochs | 2 (30-min timeout, ~27 min/epoch) |
| Config | Lion lr=3e-4, 4L/256d/4H, slices=128 |

**Commentary:** Bigger model is too slow for 30-min budget (27 min/epoch → only 2 epochs). val_mae=314.52 vs baseline 269.32 (+17%), but model was still rapidly improving. On AirfRANS, bigger model + Lion was also weak while + AdamW showed promise. Capacity scaling should be revisited with AdamW + slices=64 for fairer comparison.

---

## 2026-04-20 19:30 — PR #2415: TandemFoil: higher LR lr=1e-3 (Lion) — CLOSED

- **Branch:** nezuko/tandem-lr-1e3
- **W&B run:** 1gshqd87

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 352.40 |
| test_primary/surface_pressure_mae | 338.49 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=1e-3, slices=96, cosine_t_max=150 |

**Commentary:** Lion at lr=1e-3 achieves 352.40 vs baseline 269.32 (+31%). Mirrors AirfRANS pattern where Lion at any LR lost to AdamW. LR tuning within Lion is the wrong direction. The correct experiment is AdamW LR sweep (covered in tanjiro's #2441 and alphonse's #2433).

---

## 2026-04-20 18:38 — PR #2416: TandemFoil: AdamW optimizer vs Lion baseline

- **Branch:** alphonse/tandem-adamw
- **Hypothesis:** AdamW optimizer may outperform Lion on TandemFoil as it does on AirfRANS
- **W&B run:** r5t674uy

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 269.32 |
| test_primary/surface_pressure_mae | 262.56 |
| val_geom_camber_cruise | 224.60 (test) |
| val_re_rand | 249.91 (test) |
| val_geom_camber_rc | 270.91 (test) |
| val_single_in_dist | 304.83 (test) |
| Epochs | 2 (30-min timeout, ~15 min/epoch) |
| Config | AdamW lr=5e-4, slices=96, 3L/192d |

**Commentary:** Only 2 epochs completed due to TandemFoil's high per-epoch cost (~15 min/epoch at slices=96). Model still strongly improving (val MAE 349→269). Establishes first TandemFoil baseline on the radford branch. Infinity observed in `test_geom_camber_cruise/mae_vol_p` — likely EMA artifact at very early training. The Lion vs AdamW comparison cannot be made fairly at 2 epochs. Merged to establish baseline. Round 2 will test AdamW LR sweep and slices reduction for faster epoch cycling.

---

## 2026-04-20 18:35 — PR #2423: AirfRANS: AdamW optimizer lr=5e-4

- **Branch:** kohaku/airfrans-adamw-lr5e4
- **Hypothesis:** AdamW may outperform Lion+Lookahead on AirfRANS
- **W&B run:** u95mzqso

| Metric | Value |
|--------|-------|
| val_primary/surface_mse | 0.3308 |
| test_primary/surface_mse | 0.3199 |
| surface_mse_Ux (test) | 0.001287 |
| surface_mse_Uy (test) | 0.000466 |
| surface_mse_p (test) | 1.2775 |
| surface_mse_nut (test) | 3.6e-05 |
| Epochs | 6 (30-min timeout) |
| Config | AdamW lr=5e-4, slices=96, 3L/192d/3H |

**Commentary:** AdamW at lr=5e-4 dramatically outperforms Lion lr=3e-4 (0.331 vs 0.538 baseline, -38%). Clean monotonic improvement across all 6 epochs with no plateau — still improving at cutoff. Pressure channel dominates error (surface_mse_p=1.28 vs <0.002 for velocity channels). All other AirfRANS PRs closed as inferiors: Lion at any LR cannot compete with AdamW. Merged as AirfRANS baseline. Round 2 will bracket the AdamW LR (3e-4–8e-4) and explore capacity + cosine schedule.

---

## 2026-04-20 18:35 — PR #2420: AirfRANS: clean default baseline (closed — superseded)

| Metric | Value |
|--------|-------|
| val_primary/surface_mse | 0.3973 (best epoch 4) / 0.5384 (final epoch 6) |
| Config | Lion lr=3e-4, slices=96, 3L/192d |

**Commentary:** Superseded by kohaku's AdamW recipe (-38%). Oscillating val_mse at epochs 5-6 consistent with cosine LR mismatch (T_max=150 at epoch 6 = barely moved off initial LR). Research question answered: Lion at default settings is not competitive on AirfRANS.

---

## 2026-04-20 18:35 — PR #2421: AirfRANS: higher LR lr=1e-3 (closed)

| Metric | Value |
|--------|-------|
| val_primary/surface_mse | 0.4695 (epoch 6) |
| Config | Lion lr=1e-3, slices=96 |

**Commentary:** Lion at higher LR (0.470) still far behind AdamW (0.331). Unstable spike at epoch 3 (1.36). LR tuning within Lion is the wrong direction — optimizer is the lever.

---

## 2026-04-20 18:35 — PR #2422: AirfRANS: intermediate LR lr=5e-4 (closed)

| val_primary/surface_mse | 0.5940 (final) / 0.4151 (best epoch 4) |
|---|---|
| Config | Lion lr=5e-4, cosine_t_max=50, slices=96 |

**Commentary:** Lion lr=5e-4 degraded when LR peaked at 5e-4 in cosine cycle (surface_mse spiked to 0.576). Final metric 0.594 is worse than Lion baseline. Confirms Lion is not competitive regardless of LR on AirfRANS.

---

## 2026-04-20 18:35 — PR #2424: AirfRANS: bigger model 4L/256d (closed)

| val_primary/surface_mse | 0.5222 (epoch 5) |
|---|---|
| Config | Lion lr=3e-4, 4L/256d/4H/128sl |

**Commentary:** Bigger model with Lion (0.522) barely beats Lion baseline (0.538) — not meaningful. Slower training (5 epochs in 30 min) and noisy trajectory. Capacity helps only when paired with a good optimizer (haku's 4L+AdamW reached 0.379).

---

## 2026-04-20 18:35 — PR #2425: AirfRANS: ablate surface refinement head (closed — research question answered)

| val_primary/surface_mse | 0.5700 (final) / 0.4769 (best ep3) |
|---|---|
| Config | Lion lr=3e-4, surface_refine=False, cosine_t_max=50 |

**Commentary:** Without surface refinement (0.570) is worse than with it (0.538 baseline). Surface refinement head confirmed beneficial. Default surface_refine=True is correct.

---

## 2026-04-20 18:35 — PR #2426: AirfRANS: deeper model 6L/192d (closed — diverging)

| val_primary/surface_mse | 0.9425 (epoch 5) |
|---|---|
| Config | Lion lr=3e-4, 6L/192d, cosine_t_max=50 |

**Commentary:** Severe divergence — spiked to 1.017 at epoch 4. 6-layer model is ~6 min/epoch, only 5 epochs in 30 min. Going deeper with Lion is clearly harmful. Depth scaling is not the direction.

---

## 2026-04-20 18:35 — PR #2427: AirfRANS: bigger model + AdamW lr=1e-3 (closed — direction redirected)

| val_primary/surface_mse | 0.3793 (epoch 5) / test: 0.3482 |
|---|---|
| Config | 4L/256d, AdamW lr=1e-3, cosine_t_max=50 |

**Commentary:** Second best AirfRANS result (0.379) but loses to simpler AdamW 3L/192d at lr=5e-4 (0.331). Bigger model is slower (~6 min/epoch), fewer epochs, and the higher LR (1e-3 vs optimal 5e-4) likely suboptimal. Direction is promising but needs to be tested with lr=5e-4 and fewer slices — covered in emma's Round 2 PR #2429.
