# Baseline Metrics — radford branch

## TandemFoilSet

- **Primary metric:** `val_primary/surface_pressure_mae` (= `val_eq4/surface_pressure_mae`)
- **Current best:** 75.59 (val) / 72.12 (test)
- **Best PR:** #2490 (frieren — T_max=10, Fourier + physics + no-EMA, slices=64, Lion lr=3e-4, 14 epochs)
- **Key insight:** T_max=10 creates ~75 cosine cycles per epoch (750 steps/epoch). Extremely rapid LR averaging produces the best minima. T_max=10 > T_max=20 > T_max=15 > T_max=30. Still improving at epoch 14 — longer training should push below 70.

### 2026-04-21 — PR #2490: TandemFoil: Fourier+physics T_max=10 — NEW BEST

- **val_primary/surface_pressure_mae:** 75.59 (-4.1% vs 78.81)
- **test_primary/surface_pressure_mae:** 72.12 (-4.0% vs 75.13)
- **Per-split test MAE:** single_in_dist=72.33, geom_camber_rc=76.01, geom_camber_cruise=70.80, re_rand=69.34
- **W&B run:** 77yoba65 (winner, T_max=10); aiols138 (T_max=15: 80.23); yt60qcd1 (T_max=20: 77.00)
- **Epochs:** 14 (30-min timeout, still improving)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine_t_max 10 --no-use-ema --model_slices 64 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

### 2026-04-21 — PR #2495: TandemFoil: T_max=30 long run (180-min budget) — NEW BEST

- **val_primary/surface_pressure_mae:** 78.81 (-4.6% vs 82.65)
- **test_primary/surface_pressure_mae:** 75.13 (-6.8% vs 80.63)
- **Per-split val MAE:** single_in_dist=97.78, geom_camber_rc=79.50, geom_camber_cruise=65.56, re_rand=72.41
- **Note:** T_max=1000 run (mjihidho) was worse at val=87.80 — falsified slow cosine hypothesis. Scheduler is per-batch; T_max=1000 at 750 steps/epoch gives ~10 cycles not 1. T_max=30 rapid cycling (25 restarts/epoch) continues to win. 14 epochs (180-min budget), still improving at cutoff.
- **W&B run:** 8k0blg8s (winner, T_max=30); mjihidho (T_max=1000 — worse, val=87.80)
- **Epochs:** 14 (180-min budget)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine_t_max 30 --no-use-ema --model_slices 64 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

### 2026-04-21 — PR #2494: TandemFoil: T_max=300 long training — SECONDARY WIN

- **val_primary/surface_pressure_mae:** 80.50 (-2.6% vs old baseline 82.65)
- **test_primary/surface_pressure_mae:** 77.82 (-3.5% vs 80.63)
- **Per-split val MAE:** single_in_dist=91.68, geom_camber_rc=90.20, geom_camber_cruise=62.36, re_rand=77.78
- **Note:** T_max=300 creates ~2.5 cycles/epoch oscillation. LR trough epoch progression: 154→97→86→80.50. Still improving at cutoff. Merged as secondary winner over old 82.65 baseline.
- **W&B run:** 4ie6hkop
- **Epochs:** 14
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine_t_max 300 --no-use-ema --model_slices 64 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

### 2026-04-20 23:15 — PR #2473: TandemFoil: golden + Fourier + physics + no-EMA — NEW BEST

- **val_primary/surface_pressure_mae:** 82.65 (-28.1% vs 114.92)
- **test_primary/surface_pressure_mae:** 80.63
- **Per-split val MAE:** single_in_dist=102.40, geom_camber_rc=88.97, geom_camber_cruise=62.37, re_rand=76.87
- **Note:** 14 epochs at slices=64 with Fourier + core physics + no-EMA. Best IS final epoch — still sharply improving at cutoff (95.63 → 82.65 in last 2 epochs). Run 1 (Fourier only, no physics) also beat baseline at 106.61. Fourier+physics is synergistic.
- **W&B run:** nh380grv
- **Epochs:** 14 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine_t_max 30 --no-use-ema --model_slices 64 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition`

### 2026-04-20 21:30 — PR #2435: TandemFoil: cosine T_max=30 at slices=64 — NEW BEST

- **val_primary/surface_pressure_mae:** 114.92 (-42% vs 197.87)
- **test_primary/surface_pressure_mae:** 108.16
- **Note:** 11 epochs in 30 min at slices=64. Still EMA=True (pre-no-EMA finding). T_max=30 > T_max=10 (117.23) > T_max=50 (127.51) > T_max=20 (132.62). All runs still improving at cutoff.
- **W&B run:** 3ec9m9az
- **Epochs:** 11 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine_t_max 30 --model_slices 64`

### 2026-04-20 19:50 — PR #2412: TandemFoil: clean baseline no-EMA (frieren v4)

- **val_primary/surface_pressure_mae:** 197.87
- **test_primary/surface_pressure_mae:** 191.70
- **Per-split test MAE:** single_in_dist=212.64, geom_camber_rc=172.00, geom_camber_cruise=187.39, re_rand=194.77
- **Note:** v4 variant (no-EMA), Lion lr=3e-4, slices=96, NO physics features, use_lookahead=True. Only 2 epochs. Beats physics-features baseline (262.82) by 24.7% purely by removing EMA.
- **W&B run:** y8f8pkkn
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine_t_max 50 --no-use-ema`

### 2026-04-20 19:30 — PR #2414: TandemFoil: core physics features (TE+Cp+asinh+residual)

- **val_primary/surface_pressure_mae:** 262.82
- **test_primary/surface_pressure_mae:** 257.51
- **Per-split test MAE:** single_in_dist=267.26, geom_camber_rc=280.59, geom_camber_cruise=225.63, re_rand=256.55
- **Note:** Only 2 epochs (30-min timeout, ~15 min/epoch). Physics features: enable_te_coord_frame, enable_cp_panel, enable_cp_panel_tandem_only, asinh_pressure, residual_prediction, enable_pressure_prior_addition. Lion optimizer lr=3e-4.
- **W&B run:** 1zbp5dlu
- **Epochs:** 2 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine_t_max 50 --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition`

### 2026-04-20 18:38 — PR #2416: TandemFoil: AdamW optimizer vs Lion baseline

- **val_primary/surface_pressure_mae:** 269.316
- **test_primary/surface_pressure_mae:** 262.56
- **Per-split test MAE:** eq4=262.56, geom_camber_cruise=224.60, geom_camber_rc=270.91, re_rand=249.91, single_in_dist=304.83
- **Note:** Only 2 epochs (30-min timeout, ~15 min/epoch). Still strongly improving. Infinity observed in test_geom_camber_cruise/mae_vol_p (early-training EMA artifact).
- **W&B run:** r5t674uy
- **Epochs:** 2 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer adamw --lr 5e-4 --cosine_t_max 150`

## AirfRANS

- **Primary metric:** `val_primary/surface_mse`
- **Current best:** 0.0207 (val) / TBD (test)
- **Best PR:** #2617 (kohaku — Fourier+3L/192d + no-EMA + T_max=10, 41 epochs, AdamW lr=5e-4)
- **Key insight:** Phase transition at T_max=10/epoch 40 is reproducible and improving across replications. The transition is stochastic — run-to-run variance exists (0.0207, 0.0248). External target: 0.0043 — **4.8x gap remaining**.

### 2026-04-21 — PR #2617: AirfRANS: T_max=10 replication — NEW BEST

- **val_primary/surface_mse:** 0.0207 (-16.5% vs 0.0248)
- **W&B run:** z7t3ibwi (41 epochs, best at epoch 40 — cosine trough)
- **Epochs:** 41 (same as PR #2556 — phase transition deterministic at epoch 40 for T_max=10)
- **Note:** The phase transition produces different depths run-to-run (stochastic). 0.0207 confirms there is still headroom below 0.0248.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 10 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2556: AirfRANS: Fourier+3L/192d+T_max=10 — NEW BEST (deeper phase transition)

- **val_primary/surface_mse:** 0.0248 (-64.3% vs 0.0696)
- **W&B run:** 7qre8z5x (41 epochs, best at epoch 40 — cosine trough)
- **Epochs:** 41 (T_max=10, best at epoch 40 where cosine LR hits trough)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 10 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2540: AirfRANS: Fourier+3L/192d+T_max=50 — NEW BEST (phase transition breakthrough)

- **val_primary/surface_mse:** 0.0696 (-65.4% vs 0.2015)
- **test_primary/surface_mse:** 0.0877 (-53.6% vs 0.1890)
- **W&B run:** ijwvfcms (winner, lr=5e-4, 23 epochs); km5xxa3n (lr=8e-4, best 0.1048 at epoch 21, unstable)
- **Epochs:** 23 (30-min timeout — model at epoch 22 was 0.19, then phase-transitioned to 0.0696 at epoch 23)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 50 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2538: AirfRANS: Fourier+4L/256d+T_max=50 (compound) — NEW BEST

- **val_primary/surface_mse:** 0.2015 (-14.5% vs 0.2357)
- **test_primary/surface_mse:** 0.1890 (-5.6% vs 0.2002)
- **W&B run:** ty0cmdfz (winner, T_max=50, 14 epochs); 85pabaza (T_max=30, val=0.2195 — also beats baseline but dominated)
- **Epochs:** 14 (still converging at cutoff — downward envelope clear across cosine cycles)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 50 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-21 01:00 — PR #2482: AirfRANS: no-EMA + T_max=50 + lr=5e-4 (24 epochs) — NEW BEST

- **val_primary/surface_mse:** 0.2357 (-1.3% vs 0.2387)
- **test_primary/surface_mse:** 0.2002 (-3.7% vs 0.2079)
- **W&B run:** xmrkwt1y (winner, T_max=50); d057fle1 (lr=8e-4, T_max=150 — unstable final)
- **Epochs:** 24 (180-min budget, no-Fourier so faster epochs)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 50 --no-use-ema --epochs 999`

### 2026-04-21 00:00 — PR #2478: AirfRANS: Fourier + 4L/256d full epoch run — NEW BEST

- **val_primary/surface_mse:** 0.2387 (-17.4% vs 0.2891 prior, -8.1% vs 0.2597)
- **test_primary/surface_mse:** 0.2079
- **full_val/volume_mse:** 0.2933
- **W&B run:** vwb9teqa
- **Epochs:** 8 (180-min budget)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 150 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-20 22:20 — PR #2455: AirfRANS: 3L/192d no-EMA no-Fourier 6 epochs — NEW BEST

- **val_primary/surface_mse:** 0.2597 (-10.2% vs 0.2891)
- **test_primary/surface_mse:** 0.2392 (-16.3% vs 0.2856)
- **Surface MSE breakdown (test):** p=0.9556 (first time below 1.0!)
- **Note:** 6 epochs WITHOUT Fourier. 3L/192d + no-EMA + AdamW lr=5e-4 + T_max=150. Fourier adds ~3x epoch overhead (15→5 min/epoch), so dropping it triples epoch count. Still improving at epoch 6. 4L/256d variant was worse (0.2935, 5 epochs).
- **W&B run:** pifi0x1v
- **Epochs:** 6 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 150 --no-use-ema`

### 2026-04-20 22:00 — PR #2474: AirfRANS: Fourier + no-EMA + 4L/256d

- **val_primary/surface_mse:** 0.2891 (-3.9% vs 0.3009)
- **test_primary/surface_mse:** 0.2856 (-0.5% vs 0.2869)
- **Surface MSE breakdown (val, epoch 2):** Ux=0.000403, Uy=0.000072, p=1.1560, nut=0.000098
- **Note:** 2 epochs only. 4L/256d/4H + Fourier + no-EMA. Steep descent (epoch 1: 0.4256 → epoch 2: 0.2891). More epochs should push much lower. lr=3e-4 variant underperformed on test.
- **W&B run:** hxyibvbf
- **Epochs:** 2 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 150 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4`

### 2026-04-20 21:15 — PR #2457: AirfRANS: Fourier + no-EMA + AdamW lr=5e-4

- **val_primary/surface_mse:** 0.3009 (-9.1% vs 0.3308)
- **test_primary/surface_mse:** 0.2869 (-10.3% vs 0.3199)
- **Surface MSE breakdown (test):** Ux=0.001468, Uy=0.0000729, p=1.1459, nut=0.000351
- **Note:** Only 2 epochs (epoch starvation from parallel jobs). Fourier features resolve high-frequency pressure gradients. nut channel regresses (+875%) but is negligible in composite. Still rapidly improving at cutoff.
- **W&B run:** cgr5omp3
- **Epochs:** 2 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 150 --no-use-ema --enable-fourier`

### 2026-04-20 18:35 — PR #2423: AirfRANS: AdamW optimizer lr=5e-4

- **val_primary/surface_mse:** 0.330816
- **test_primary/surface_mse:** 0.319870
- **Surface MSE breakdown (test):** Ux=0.001287, Uy=0.000466, p=1.2775, nut=3.6e-05
- **val/loss:** 1.074 (train loss at final epoch)
- **W&B run:** u95mzqso
- **Epochs:** 6 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 5e-4 --cosine_t_max 150`

## DrivAerML

- **Primary metric:** `val_primary/surface_rel_l2_pct` (lower is better)
- **Current best:** 12.70% (val) / 13.54% (test)
- **Best PR:** #2593 (shinji — Fourier+4L/256d+no-EMA+T_max=30, 45 epochs, AdamW lr=5e-4)
- **CRITICAL:** Must pass `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200` to avoid OOM and `--model-heads 4` for 256d model
- **External target:** <3.71% (AB-UPT, ~500 epochs) — **3.4x gap remaining**
- **Key insight:** Architecture depth is the critical lever. 4L/256d still converging at 45-epoch cap (SENPAI_MAX_EPOCHS). 5L/256d is WORSE (13.62%) — optimization instability beyond 4 layers. 3L/256d worse than 3L/192d. T_max=30 confirmed optimal.

### 2026-04-21 — PR #2593: DrivAerML: 4L/256d+T_max=30 replication — NEW BEST

- **val_primary/surface_rel_l2_pct:** 12.70% (-2.0% vs 12.96%)
- **test_primary/surface_rel_l2_pct:** 13.54% (-6.0% vs 14.41%)
- **W&B run:** 3aaevlho (45 epochs, hit SENPAI_MAX_EPOCHS=50 cap — NOT timeout. Still converging!)
- **Epochs:** 45 (epoch cap, not time — more training headroom confirmed)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine_t_max 30 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-21 — PR #2550: DrivAerML: Fourier+4L/256d+T_max=30 — NEW BEST

- **val_primary/surface_rel_l2_pct:** 12.96% (-61.5% relative vs 33.65%)
- **test_primary/surface_rel_l2_pct:** 14.41%
- **W&B run:** 8s5i8y06 (winner, T_max=30, 43 epochs); qf8vxows (T_max=50: 13.04%)
- **Epochs:** 43 (still converging — LR=0.000083 at best epoch, not at trough)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine_t_max 30 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-21 — PR #2543: DrivAerML: Fourier+no-EMA+T_max=30 long training replication — NEW BEST

- **val_primary/surface_rel_l2_pct:** 33.65% (-34.5% relative vs 51.35%)
- **test_primary/surface_rel_l2_pct:** 34.00%
- **W&B run:** xm765o85 (6 epochs, 3L/192d, still converging at cutoff)
- **Epochs:** 6 (~5 min/epoch; full 180-min run likely to push well below 30%)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine_t_max 30 --no-use-ema --enable-fourier --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-21 00:05 — PR #2475: DrivAerML: Fourier + no-EMA (T_max=30) — NEW BEST

- **val_primary/surface_rel_l2_pct:** 51.35% (-9.8% relative vs 56.91%)
- **test_primary/surface_rel_l2_pct:** 52.06%
- **W&B run:** 5ncrjm32 (winner, T_max=30); uy73j36s (T_max=150: 52.06%)
- **Epochs:** 2 (still descending at cutoff — clear headroom)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine_t_max 30 --no-use-ema --enable-fourier --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-20 22:00 — PR #2467: DrivAerML: no-EMA + AdamW lr=8e-4 — NEW BEST

- **val_primary/surface_rel_l2_pct:** 56.91% (-20% relative vs 71.35%)
- **test_primary/surface_rel_l2_pct:** 57.33%
- **Note:** 2 epochs. No-EMA unlocks higher LR effectiveness. lr=1e-3 also good (58.78%). AdamW lr=8e-4 + no-EMA is the new DrivAerML default.
- **W&B run:** ip8ybl80
- **Epochs:** 2 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 8e-4 --cosine_t_max 150 --no-use-ema`

### 2026-04-20 21:00 — PR #2440: DrivAerML: AdamW vs Lion baseline sweep (first baseline)

- **val_primary/surface_rel_l2_pct:** 71.35%
- **Note:** Only 2 epochs completed (30-min timeout, ~10-11 min/epoch). AdamW lr=5e-4 best of 4 runs. Lion lr=3e-4 degraded epoch-over-epoch (74.1%→78.5%). Student resolved OOM with 50k surface-point sampling. All AdamW LRs (3e-4, 5e-4, 8e-4) converged to ~71.4-71.8%.
- **W&B run:** kulxytfg
- **Epochs:** 2 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine_t_max 150`
