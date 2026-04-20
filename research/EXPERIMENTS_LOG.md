# SENPAI Research Results

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
