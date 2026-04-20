# Baseline Metrics — radford branch

## TandemFoilSet
- **Primary metric:** `val_primary/surface_pressure_mae` (= `val_eq4/surface_pressure_mae`)
- **Current best:** No baseline established yet (Round 1 in progress)
- **Best PR:** —

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
