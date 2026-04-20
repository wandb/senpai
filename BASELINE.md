# Baseline Metrics — radford branch

## TandemFoilSet

- **Primary metric:** `val_primary/surface_pressure_mae` (= `val_eq4/surface_pressure_mae`)
- **Current best:** 197.87 (val) / 191.70 (test)
- **Best PR:** #2412 (frieren — TandemFoil no-EMA, Lion lr=3e-4, no physics features)
- **Key insight:** EMA (ema_start_step=50) is catastrophically harmful at 2 epochs. All future runs must use `--no-use-ema`.

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
- **Current best:** 0.3308 (val) / 0.3199 (test)
- **Best PR:** #2423 (kohaku — AirfRANS AdamW optimizer lr=5e-4)

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
- **Current best:** 71.35% (val)
- **Best PR:** #2440 (shoya — DrivAerML AdamW lr=5e-4, 2 epochs)
- **External target:** <3.71% (AB-UPT, ~500 epochs)

### 2026-04-20 21:00 — PR #2440: DrivAerML: AdamW vs Lion baseline sweep (first baseline)

- **val_primary/surface_rel_l2_pct:** 71.35%
- **Note:** Only 2 epochs completed (30-min timeout, ~10-11 min/epoch). AdamW lr=5e-4 best of 4 runs. Lion lr=3e-4 degraded epoch-over-epoch (74.1%→78.5%). Student resolved OOM with 50k surface-point sampling. All AdamW LRs (3e-4, 5e-4, 8e-4) converged to ~71.4-71.8%.
- **W&B run:** kulxytfg
- **Epochs:** 2 (30-min timeout)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine_t_max 150`
