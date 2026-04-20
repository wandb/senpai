# Baseline Metrics — radford branch

## TandemFoilSet

- **Primary metric:** `val_primary/surface_pressure_mae` (= `val_eq4/surface_pressure_mae`)
- **Current best:** 262.82 (val) / 257.51 (test)
- **Best PR:** #2414 (tanjiro — TandemFoil core physics features: TE+Cp+asinh+residual)

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
