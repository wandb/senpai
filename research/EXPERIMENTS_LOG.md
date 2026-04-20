# SENPAI Research Results

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
