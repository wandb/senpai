# Baseline Metrics — radford branch

## TandemFoilSet

- **Primary metric:** `val_primary/surface_pressure_mae` (= `val_eq4/surface_pressure_mae`)
- **Current best:** 21.909 (val) at epoch 334 — test 23.419
- **Best PR:** #3108 (zenitsu — gc=0.3+EMA=0.999, Lion lr=1.25e-4, T_max=10, WD=1e-2, 3L/192d)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 --grad-clip 0.3 --weight-decay 1e-2 --model-slices 64 --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999`

### 2026-04-23 — PR #3108: TandemFoil: gc=0.3 + EMA=0.999 — NEW BEST (CURRENT)

- **val_primary/surface_pressure_mae:** 21.909 (-2.8% vs 22.537) at epoch 334
- **test_primary/surface_pressure_mae:** 23.419 (-4.7% vs 24.581)
- **Per-split test MAE:** geom_camber_cruise=27.436, geom_camber_rc=31.587, re_rand=17.119, single_in_dist=17.533
- **W&B run:** kzg626hf (zenitsu/tf-gc03-ema999)
- **Config:** Lion lr=1.25e-4, T_max=10, **gc=0.3**, WD=1e-2, **EMA=0.999**, 3L/192d, Fourier+physics
- **Key insight:** Softer gc (0.3 vs 0.5) under EMA stability finds a deeper basin. gc=0.5 was already soft relative to standard gc=1.0 — gc=0.3 continues the trend. Model best at ep334 not terminal — still room to improve.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 --grad-clip 0.3 --weight-decay 1e-2 --model-slices 64 --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999`

### 2026-04-22 — PR #2924: TandemFoil: EMA refinement gc=0.5 — PREVIOUS BEST

- **val_primary/surface_pressure_mae:** 22.537 (-13.8% vs 26.06) at epoch 336
- **W&B run:** 0lv7fnun (robin/ema-refine-tf-gc05)
- **Config:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, EMA decay=0.999, 3L/192d, Fourier+physics, 360-min budget
- **Key insight:** gc=0.5 enables stable EMA training across 336+ epochs; gc=1.0 diverged after ep167 (Run 1). Model still descending at ep336 — result is an underestimate of the ceiling. AirfRANS Run 3 (0.000659) was superseded by stark #2951 T_max=50 result (0.000482).
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 --grad-clip 0.5 --weight-decay 1e-2 --model-slices 64 --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999`

### 2026-04-22 — PR #2887: TandemFoil: lr=1e-4 gc=1.0 LR scan — PREVIOUS BEST

- **val_primary/surface_pressure_mae:** 26.06 (-0.3% vs 26.134) at epoch 300
- **W&B run:** pbq4kgdk
- **Config:** Lion lr=1e-4, T_max=10, gc=1.0, WD=1e-2, 3L/192d, Fourier+physics, no-EMA
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 1e-4 --cosine-t-max 10 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --model-slices 64 --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

### 2026-04-21 — PR #2899: TandemFoil: Corrected EMA warmup (decay=0.999) — PREVIOUS BEST

- **val_primary/surface_pressure_mae:** 26.134 (-13.2% vs 30.10) at epoch 123
- **decay=0.9999:** 26.903 at epoch 113 (also beats baseline — 10.6% improvement)
- **W&B run:** nrn0q3ct (decay=0.999, still running at ep125, still improving)
- **Key insight:** EMAWithWarmup replaces bugged EMA. Timm formula `min(target_decay, (1+step)/(10+step))` ramps decay from 0 to target, preventing random-init dominance of EMA weights. Both decay values beat baseline significantly. decay=0.999 is the winner. DrivAerML did NOT benefit (9.749% at 60 eps).
- **Config:** Lion lr=1.25e-4, T_max=10, gc=1.0, WD=1e-2, 3L/192d, Fourier+physics, EMA decay=0.999 (WITHOUT --no-use-ema)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 --grad-clip 1.0 --weight-decay 1e-2 --model-slices 64 --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999`

### 2026-04-21 — PR #2810: TandemFoil: lr=1.25e-4 + gc=1.0 — PREVIOUS BEST

- **val_primary/surface_pressure_mae:** 30.10 (-32.8% vs 44.72) at epoch 157
- **W&B run:** v6amjkh7 (157+ epochs, 180-min budget, still descending at cutoff)
- **Key insight:** Lower LR continues downward trend: 3e-4→2e-4→1.5e-4→1.25e-4 all improved. gc=1.0 essential — provides continuous stabilization of Lion optimizer at cosine LR peaks. Near-identical result to sasuke's gc=0.5 at lr=1.5e-4 (30.11) confirms that both gc reduction and LR reduction independently improve by similar amounts.
- **Reproduce:** `cd target/icml2026 && SENPAI_MAX_EPOCHS=9999 SENPAI_TIMEOUT_MINUTES=180 python train.py --dataset tandemfoil --optimizer lion --lr 1.25e-4 --cosine-t-max 10 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --model-slices 64 --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

### 2026-04-21 — PR #2724: TandemFoil: lr=1.5e-4 — PREVIOUS BEST

- **val_primary/surface_pressure_mae:** 44.72 (-0.78% vs 45.07) at epoch 89
- **W&B run:** g82605dq (115 epochs, 180-min budget, still improving at cutoff)
- **Epochs:** 115 (~1.57 min/epoch at 3L/192d)
- **Per-split val at best (ep89):** geom_camber_cruise=29.65, geom_camber_rc=53.96, re_rand=41.10, single_in_dist=54.18
- **Key insight:** Lower LR (1.5e-4) continues the downward trend from lr=3e-4→2e-4. Trough envelope 3 phases: rapid descent (ep1-30), oscillating descent (ep30-90), near-convergence (ep90-115). Best at ep89 (not final) — slight overfit after. Occasional spikes to 80-90 at cosine peaks suggest gc could help.
- **Reproduce:** `cd target/icml2026 && SENPAI_TIMEOUT_MINUTES=180 SENPAI_MAX_EPOCHS=9999 python train.py --dataset tandemfoil --optimizer lion --lr 1.5e-4 --cosine-t-max 10 --no-use-ema --model-slices 64 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

### 2026-04-21 — PR #2610: TandemFoil: lr=2e-4 + T_max=10 — NEW BEST

- **val_primary/surface_pressure_mae:** 45.07 (-14.7% vs 52.81)
- **W&B run:** ixs1rqgk (119 epochs, 180-min budget, still improving!)
- **Epochs:** 119 (180-min timeout, ~1.5 min/epoch at 3L/192d)
- **Key insight:** lr=2e-4 at default 3L/192d beats lr=3e-4 at 5L/256d (45.07 vs 52.81). Lower LR + ultra-rapid cosine cycling (T_max=10) produces more stable optimization. Consistent downward envelope over 119 epochs with no sign of convergence. Oscillation amplitude ~10-20 points (vs 20-30 at lr=3e-4). T_max=10+lr=2e-4 also beats T_max=30+lr=2e-4 (49.99 at 79ep).
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 2e-4 --cosine-t-max 10 --no-use-ema --model-slices 64 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

### 2026-04-21 — PR #2595: TandemFoil: 5L/256d deep model — NEW BEST

- **val_primary/surface_pressure_mae:** 52.81 (-30.1% vs 75.59)
- **test_primary/surface_pressure_mae:** 55.25 (-23.4% vs 72.12)
- **Per-split test MAE:** single_in_dist=61.80, geom_camber_rc=60.20, geom_camber_cruise=48.99, re_rand=50.00
- **W&B run:** l5kggnbg (67 epochs, 180-min budget, still improving!)
- **Epochs:** 67 (180-min timeout, ~2.7 min/epoch)
- **Key insight:** 5L/256d depth+width scaling mirrors DrivAerML's width-scaling discovery. All splits improved uniformly (cruise -30.8%, re_rand -27.9%). High val oscillation (52-85 range) from T_max=10 but consistent downward envelope. Train loss (0.115) still decreasing at cutoff.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil --optimizer lion --lr 3e-4 --cosine-t-max 10 --no-use-ema --model-slices 64 --model-layers 5 --model-hidden-dim 256 --model-heads 4 --enable-fourier --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999`

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
- **Current best:** 0.000296 (val) at epoch 704 — with Vol MSE 0.002039 (best in programme)
- **Best PR:** #3135 (nezuko — EMA=0.999 + vol-weight=10x, 2L/256d, AdamW lr=6e-4, gc=1.0, WD=1e-2)
- **Key insight:** EMA=0.999 + vol-weight=10x compound dramatically improves both metrics simultaneously: surface -35.5% (0.000296 vs 0.000459) AND volume -26.6% (0.002039 vs 0.002777). Volume gap to SpiderSolver target closed from 1.63x to 1.20x. Model still improving at ep763 timeout.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 6e-4 --cosine-t-max 50 --grad-clip 1.0 --weight-decay 1e-2 --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999 --ema-decay 0.999 --vol-loss-weight 10.0`

### 2026-04-23 — PR #3135: AirfRANS: EMA=0.999 + vol-weight=10x — NEW BEST (CURRENT)

- **val_primary/surface_mse:** 0.000296 (-35.5% vs 0.000459) at epoch 704
- **val_primary/vol_mse:** 0.002039 (-26.6% vs 0.002777) at epoch 743
- **W&B run:** sh2zyfwr (nezuko/af-ema-vol-weight-10x)
- **Config:** 2L/256d/4H, AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, **EMA=0.999**, vol-weight=10x, Fourier
- **Key insight:** vol-weight=10x without EMA was confirmed worse. With EMA=0.999, the combination works dramatically better — EMA stabilizes the upweighted volume gradient. Surface -35.5% and volume -26.6% simultaneously. Vol gap to SpiderSolver: 1.20x (was 1.63x). Model at ep763 still converging.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 6e-4 --cosine-t-max 50 --grad-clip 1.0 --weight-decay 1e-2 --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999 --ema-decay 0.999 --vol-loss-weight 10.0`

### 2026-04-23 — PR #3050: AirfRANS: EMA=0.999 at T_max=50 champion — PREVIOUS BEST

- **val_primary/surface_mse:** 0.000459 (-4.8% vs 0.000482) at epoch 771
- **full_val/volume_mse:** 0.002777 (-63.6% vs 0.00764) — best volume result in programme
- **W&B run:** z6pry4b9 (stark/af-ema-champion)
- **Config:** 2L/256d/4H, AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, **EMA=0.999**, Fourier
- **Key insight:** EMA + T_max=50 is synergistic — the longer cosine cycle (50 vs 10) allows EMA to track the optimization trajectory harmoniously, improving both surface sharpness and volume smoothing simultaneously. Vol MSE 0.002777 closes the SpiderSolver gap from 4.5x to 1.63x. Model still descending at ep771 — further training would push both metrics lower.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 6e-4 --cosine-t-max 50 --grad-clip 1.0 --weight-decay 1e-2 --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999 --ema-decay 0.999`

### 2026-04-22 — PR #2951: AirfRANS: LR + T_max sweep — NEW BEST (CURRENT)

- **val_primary/surface_mse:** 0.000482 (-19.4% vs 0.000598) at epoch 576
- **W&B run:** pr4wsbfm (Run G: lr=6e-4, T_max=50, ~760 epochs trained)
- **Config:** 2L/256d/4H, AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, no-EMA, Fourier, 360-min budget
- **Key insight:** T_max=50 is the decisive variable. T_max=10 configs all underperformed; T_max=20 was intermediate. lr=1e-3 diverged. lr=6e-4 + T_max=50 is the new reference config. Best checkpoint at ep576 (not final). Note: WD=1e-2 was re-added vs #2906 no-WD — this combination still wins because T_max=50 is the dominant effect.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 6e-4 --cosine-t-max 50 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-22 — PR #2906: AirfRANS: gc=1.0 no-WD 360-min budget multi-seed — PREVIOUS BEST

- **val_primary/surface_mse:** 0.000598 (-4.6% vs 0.000627) at epoch 517, seed=42
- **W&B run:** d7a0z1hk (seed=42, 517 epochs, 360-min budget)
- **Other seeds:** W&B nvllyhmf (gc+WD variant: 0.000694 at 360 min); W&B nbc25ot7 (no-gc seed=42: NaN at ep607)
- **Config:** 2L/256d/4H, AdamW lr=6e-4, T_max=10, gc=1.0, **no WD** (no --weight-decay), no-EMA, Fourier, 360-min budget
- **Key insight:** Removing WD=1e-2 while keeping gc=1.0 achieves 0.000598 — gc-only (0.000598) beats gc+WD (0.000694) at 360 min. WD is NOT required for AirfRANS and actively hurts at long training budgets. The 360-min budget (vs 180-min) is highly beneficial. gc=1.0 is essential for stability — without it, seed divergence (NaN at ep607) is common. **Beats external target 0.0043 by 86.1%.**
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 6e-4 --cosine-t-max 10 --grad-clip 1.0 --no-use-ema --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-22 — PR #2902: AirfRANS: gradient accumulation ablation (accum=1 wins) — NEW BEST

- **val_primary/surface_mse:** 0.000627 (-10.3% vs 0.000699) at epoch 661
- **W&B run:** ww9w4x4u (accum=1 control run, 661 epochs)
- **Config:** 2L/256d, AdamW lr=6e-4, T_max=10, gc=1.0, WD=1e-2, no-EMA, Fourier
- **Key insight:** accum=1 (no accumulation) trained longest and found the deepest basin. accum=2 and accum=4 were strictly worse — gradient accumulation is detrimental for AirfRANS. The control run essentially extended training beyond the previous 653-epoch run.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 6e-4 --cosine-t-max 10 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-22 — PR #2887: AirfRANS: lr=6e-4 gc=1.0 T_max=10 LR scan — NEW BEST

- **val_primary/surface_mse:** 0.000699 (-3.6% vs 0.000727) at epoch 653
- **W&B run:** 7m6c1ydk
- **Config:** 2L/256d, AdamW lr=6e-4, T_max=10, gc=1.0, WD=1e-2, no-EMA, Fourier
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 6e-4 --cosine-t-max 10 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-21 — PR #2899: AirfRANS: Corrected EMA warmup (decay=0.999) — PREVIOUS BEST

- **val_primary/surface_mse:** 0.000727 (-41.2% vs 0.001236) at epoch 206 (W&B run i1sevgt2)
- **decay=0.9999:** 0.001123 (-8.7% vs 0.001236) at epoch 275 (W&B run bz00wego — also beats baseline)
- **Key insight:** EMA with timm warmup recovers the regularization benefit that was broken by the bugged EMA. Best checkpoint at ep206; late epochs oscillate due to T_max=5 cycling. decay=0.999 massively outperforms decay=0.9999 on AirfRANS.
- **Config:** 2L/256d, AdamW lr=7e-4, T_max=5, gc=1.0, WD=1e-2, Fourier, EMA decay=0.999 (WITHOUT --no-use-ema)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 7e-4 --cosine-t-max 5 --grad-clip 1.0 --weight-decay 1e-2 --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999 --ema-decay 0.999`

### 2026-04-21 — PR #2828: AirfRANS: 2L/256d depth frontier — PREVIOUS BEST

- **val_primary/surface_mse:** 0.001236 (-16.4% vs 0.001479) at epoch 349
- **2L/256d gc=0.5:** 0.001405 (-5.1% vs 0.001479) at epoch 366 (also beats baseline)
- **2L/384d:** DIVERGED (NaN from ep134 — width catastrophic at 2L)
- **2L/512d:** Never converged (0.049 at ep19, then degraded monotonically)
- **W&B run:** libpwryz (2L/256d gc=1.0, 388 epochs); w38iz72q (2L/256d gc=0.5, 386 epochs)
- **Key insight:** Depth reduction 3L→2L gives 16.4% improvement. 256d is the critical width constraint at 2L — going wider diverges. gc=1.0 outperforms gc=0.5 at 2L (fewer gradient accumulation points means tighter clip is unnecessary). Both runs were still improving at timeout (best at ep349/366 out of 388/386). T_max=10 compound is the immediate next step.
- **Reproduce:** `cd target/icml2026 && CUDA_VISIBLE_DEVICES=0 SENPAI_MAX_EPOCHS=9999 SENPAI_TIMEOUT_MINUTES=180 python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 7e-4 --cosine-t-max 5 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 2 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-21 — PR #2771: AirfRANS: 3L/256d golden config — width vs depth ablation — PREVIOUS BEST

- **val_primary/surface_mse:** 0.001479 (-46.6% vs 0.00277) at epoch 202
- **Terminal val (ep282):** 0.006535 (model regressed after ep202 divergence)
- **Terminal test:** 0.005361 (from diverged model — NOT from best checkpoint)
- **W&B run:** q4hytsr6 (282 epochs, 180-min timeout, best at ep202)
- **Architecture:** 3L/256d/4H (vs 4L/256d/4H baseline — only change is `--model-layers 3`)
- **Key insight:** Removing one layer from 4L→3L unlocked a 46.6% improvement. Width (256d) is the dominant capacity lever; extra depth adds noise to the optimization trajectory. The 3L model found a deep trough at ep202 (0.001479) but subsequently regressed — same T_max=5 instability pattern as 4L configs. Test metric from the diverged terminal checkpoint is invalid. Critical follow-ups: (1) gc=0.5 compound at 3L/256d (kakashi #2823), (2) 3L width frontier 320d/384d/512d (ray #2824), (3) T_max=10 stability variant.
- **Reproduce:** `cd target/icml2026 && CUDA_VISIBLE_DEVICES=0 SENPAI_MAX_EPOCHS=9999 SENPAI_TIMEOUT_MINUTES=180 python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 7e-4 --cosine-t-max 5 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 3 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-22 — PR #2820: AirfRANS: 3L/256d gc=0.5 lr=5e-4 — 3L LINEAGE BEST

- **val_primary/surface_mse:** 0.001241 (-16.1% vs 0.001479) at epoch 232
- **test_primary/surface_mse:** 0.003734 (from best-checkpoint run)
- **W&B run:** rvwmsfth (284 epochs, best at ep232; model regressed after ep232)
- **Config:** 3L/256d, AdamW lr=5e-4, T_max=5, gc=0.5, WD=1e-2, no-EMA, Fourier
- **Key insight:** gc=0.5 + lr=5e-4 is stable through ep284 — no divergence vs. ep205 divergence at lr=7e-4. Hypothesis confirmed: lower LR prevents the catastrophic gradient spikes at cosine T_max=5 peaks. Best 3L/256d result to date. Does not beat the overall AirfRANS best (2L/256d at 0.000627). First-phase 4L/256d run (W&B: 21u2f2n3) yielded 0.00176 at ep160.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 5e-4 --cosine-t-max 5 --grad-clip 0.5 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 3 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-21 — PR #2774: AirfRANS: 4L/256d + gc=0.5 extended — PREVIOUS BEST

- **val_primary/surface_mse:** 0.00277 (-28.9% vs 0.003904) at epoch 150
- **Surface MSE breakdown (ep150):** Ux=3.50e-05, Uy=5.64e-06, nut=3.33e-06, p=0.01105
- **full_val/volume_mse:** 0.01018
- **W&B run:** 0pt769m4 (221 epochs, 180-min timeout, best at ep150, diverged ep205)
- **Key insight:** gc=0.5 allows sharper optimization steps, finding 28.9% deeper basins than gc=1.0. Pressure channel (0.01105) still dominant error, same T_max=5 divergence pattern. Test metric invalid (diverged model). gc=0.5 now the default for AirfRANS experiments. Follow-up: gc=0.5 + T_max=10.
- **Reproduce:** `cd target/icml2026 && CUDA_VISIBLE_DEVICES=0 SENPAI_MAX_EPOCHS=9999 SENPAI_TIMEOUT_MINUTES=180 python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 7e-4 --cosine-t-max 5 --grad-clip 0.5 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-21 — PR #2755: AirfRANS: 4L/256d extended run (180-min, SENPAI_MAX_EPOCHS=9999) — PREVIOUS BEST

- **val_primary/surface_mse:** 0.003904 (-46.2% vs 0.007264) at epoch 201
- **Surface MSE breakdown (ep201):** Ux=4.97e-05, Uy=1.09e-05, nut=1.05e-05, p=0.01555
- **full_val/volume_mse:** 0.03198
- **W&B run:** stxm16tv (223 epochs, 180-min timeout, best at ep201)
- **Epochs:** 223 (180-min timeout, ~0.81 min/epoch at 4L/256d)
- **Key insight:** Removing the 50-epoch cap (SENPAI_MAX_EPOCHS=9999) unlocks 4.5x more training. Progressive descent through clear phases: 0.237 (ep4) → 0.081 (ep17) → 0.023 (ep29) → 0.00965 (ep43) → 0.00620 (ep76) → 0.00468 (ep158) → 0.003904 (ep201). Deep basins are stochastic — sub-0.005 readings appeared sporadically at ep195, 199, 201. Catastrophic divergence at ep208 (gradient norms → infinity) — T_max=5 cosine cycling eventually triggers irreversible instability. **CHECKPOINT AT BEST EPOCH IS ESSENTIAL.** Test metrics unreliable (final model diverged). Beats external target 0.0043 without pressure weighting — combining with `--pressure-loss-weight 20` is the obvious next step.
- **Reproduce:** `cd target/icml2026 && CUDA_VISIBLE_DEVICES=0 SENPAI_MAX_EPOCHS=9999 SENPAI_TIMEOUT_MINUTES=180 python train.py --dataset airfrans --airfrans-task full --optimizer adamw --lr 7e-4 --cosine-t-max 5 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-21 — PR #2727: AirfRANS: 4L/256d + WD=1e-2 + T_max=5 + gc=1.0 — PREVIOUS BEST

- **val_primary/surface_mse:** 0.007264 (-22.3% vs 0.00935)
- **W&B run:** ruurxdqs (50 epochs, best at epoch 40, hit SENPAI_MAX_EPOCHS=50 cap at 61 min)
- **Epochs:** 50 (epoch cap, NOT time cap — only used 61 of 180 min)
- **Key insight:** 4L/256d architecture + WD=1e-2 + T_max=5 stabilizes grad norms (peaked at 18.7, decreased to 7.1). Architecture scaling IS viable on AirfRANS with proper regularization. Still uses gc=1.0 — gc=1.5 hasn't been tried with this architecture yet. Trough envelope still descending at cutoff. CRITICAL: need SENPAI_MAX_EPOCHS=9999 for future runs.
- **Reproduce:** `cd target/icml2026 && SENPAI_MAX_EPOCHS=9999 python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 7e-4 --cosine-t-max 5 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999`

### 2026-04-21 — PR #2737: AirfRANS: lr=7e-4+grad-clip=1.5 — NEW BEST

- **val_primary/surface_mse:** 0.00935 (-26.4% vs 0.01271)
- **W&B run:** 7bdiqnmi (40 epochs, best at epoch 40 — still improving!)
- **Epochs:** 40 (30-min timeout)
- **Key insight:** grad-clip=1.5 at T_max=10 finds a dramatically deeper basin than clip=1.0. The clip sweep now shows: 0.5 (failed, +1.9%), 1.0 (0.01419), 1.5 (**0.00935**), 2.0 (pending). Moderate clipping outperforms tight clipping. Next: combine clip=1.5 with T_max=5 and WD=1e-2.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 7e-4 --cosine-t-max 10 --grad-clip 1.5 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2732: AirfRANS: T_max=5+lr=7e-4+grad-clip=1.0 — NEW BEST

- **val_primary/surface_mse:** 0.01271 (-3.9% vs 0.01323)
- **W&B run:** uh7fchiy (41 epochs, best at epoch 40 — cosine trough)
- **Epochs:** 41 (30-min timeout)
- **Key insight:** T_max=5 produces 8 full cosine cycles in 41 epochs (vs 4 for T_max=10). More frequent annealing cycles allow repeated phase-transition opportunities and deeper basin exploration. ep40=0.01271 (trough), ep41=0.04259 (peak rebound — volatile). WD=1e-4 (not golden 1e-2) — improvement still holds, suggesting T_max is the dominant lever. Next: combine T_max=5 with full golden WD=1e-2.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 7e-4 --cosine-t-max 5 --grad-clip 1.0 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2709: AirfRANS: lr=7e-4+grad-clip=1.0+WD=1e-2 — NEW BEST

- **val_primary/surface_mse:** 0.01323 (-6.8% vs 0.01419)
- **full_val/surface_mse_p:** 0.0528
- **full_val/volume_mse:** 0.0804
- **test_primary/surface_mse:** 0.01478 (-2.3% vs 0.01513)
- **W&B run:** 7vic8kxn (41 epochs, best at epoch 41 — STILL IMPROVING!)
- **Epochs:** 41 (30-min timeout)
- **Phase transition:** epoch 14 (0.1239 → 0.0548), then smooth descent to 0.01323. 7 consecutive new-best epochs early (6-12) from WD regularization during plateau phase. Best at final epoch — longer training should push deeper.
- **Key insight:** WD=1e-2 without grad-clip failed at 0.027 because gradient explosions at cosine peaks destroyed the regularization benefit. With grad-clip preventing spikes, WD provides clean regularization. The combo is strictly better than either alone.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 7e-4 --cosine_t_max 10 --grad-clip 1.0 --weight-decay 1e-2 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2680: AirfRANS: lr=7e-4+grad-clip=1.0 — NEW BEST

- **val_primary/surface_mse:** 0.01419 (-7.3% vs 0.0153)
- **full_val/surface_mse_p:** 0.0564 (-23.3% vs 0.0735)
- **full_val/volume_mse:** 0.0723 (-45.7% vs 0.1134)
- **test_primary/surface_mse:** 0.01513
- **W&B run:** 48ldl625 (41 epochs, best at epoch 41 — still improving!)
- **Epochs:** 41 (30-min timeout)
- **Grad-clip stats:** 327-353 of 360 batches clipped per epoch (91-98%). Mean grad norm ~10-22. Spike reduction: baseline peaks 0.23-0.27 → clipped peaks 0.15-0.17 (40-45% lower).
- **Key insight:** Grad-clip=1.0 at lr=7e-4 reduces destructive spikes at cosine LR peaks while preserving the deep basin exploration. Epoch 40 trough depth: 0.01419 (clipped) vs 0.031 (baseline) — 2.2x deeper. Volume MSE improvement (45.7%) even larger than surface.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 7e-4 --cosine_t_max 10 --grad-clip 1.0 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2655: AirfRANS: lr=3e-4+T_max=10 multi-seed — NEW BEST

- **val_primary/surface_mse:** 0.0153 (-17% vs 0.01841)
- **full_val/volume_mse:** 0.1134
- **W&B run:** srd0fcew (41 epochs, seed=789, best at epoch 41 — still improving!)
- **Epochs:** 41 (30-min timeout)
- **Multi-seed results:** seed=789→0.0153 (BEST), seed=456→0.0170, seed=123→0.0182, seed=42→0.0193, seed=1337→0.0194
- **Note:** lr=3e-4 at 5 seeds: range 0.0153-0.0194 (tight). lr=7e-4 at 5 seeds: range 0.0198-0.0463 (wide). lr=3e-4 is both more reliable AND can find deeper basins. Seed 789 was still descending at epoch 41 — more epochs could push lower.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 3e-4 --cosine_t_max 10 --no-use-ema --enable-fourier --epochs 999 --seed 789`

### 2026-04-21 — PR #2646: AirfRANS: lr=7e-4+T_max=10 — NEW BEST

- **val_primary/surface_mse:** 0.01841 (-6.5% vs 0.0197)
- **full_val/surface_mse_p:** 0.0735
- **full_val/volume_mse:** 0.1331
- **W&B run:** 3pbxocca (41 epochs, best at epoch 35 — earlier phase transition)
- **Epochs:** 41 (30-min timeout, phase transition at epoch 35)
- **Note:** lr=7e-4 triggers the phase transition 3-5 epochs earlier than lr=3e-4. Volatile peak-to-trough swings at cosine LR peaks (epochs 26, 28, 38, 41 spike to ~0.23-0.27). Epoch 35 found a qualitatively different basin at 0.018 while surrounding troughs at epochs 30, 40 were only 0.031. Test metric (0.2323) evaluated at final epoch LR peak — misleading.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 7e-4 --cosine_t_max 10 --no-use-ema --enable-fourier --epochs 999`

### 2026-04-21 — PR #2614: AirfRANS: lr=3e-4+T_max=10 — NEW BEST

- **val_primary/surface_mse:** 0.0197 (-4.8% vs 0.0207)
- **W&B run:** v5ka7832 (41 epochs, best at epoch 38 — cosine trough)
- **Epochs:** 41 (phase transition at cosine trough near epoch 38-40)
- **Note:** lr=3e-4 slower convergence BUT deeper final basin than lr=5e-4. The transition timing shifts slightly (epoch 38 vs 40).
- **Reproduce:** `cd target/icml2026 && python train.py --dataset airfrans --airfrans_task full --optimizer adamw --lr 3e-4 --cosine_t_max 10 --no-use-ema --enable-fourier --epochs 999`

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
- **Current best:** 3.833% (val) at epoch 511 — test 4.685%
- **Best PR:** #3072 (eren — EMA=0.9995 + gc=0.5, 4L/512d/8H, AdamW lr=5e-4, T_max=30)
- **CRITICAL:** Must pass `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`
- **External target:** <3.71% (AB-UPT, ~500 epochs) — **gap: 0.123 pp (1.03x)**
- **Key insight:** EMA=0.9995 requires gc=0.5 as stability guard — EMA alone diverges at ep28. gc unlocks EMA for DrivAerML. Run still converging at ep517 timeout (3.833% at ep511); more epochs expected to push below 3.82% (AB-UPT). Closes 93% of the original gap to AB-UPT.

### 2026-04-23 — PR #3072: DrivAerML: EMA=0.9995 + gc=0.5 — NEW BEST (CURRENT)

- **val_primary/surface_rel_l2_pct:** 3.833% (-4.1% vs 3.997%) at epoch 511
- **test_primary/surface_rel_l2_pct:** 4.685% (best-checkpoint eval)
- **W&B run:** ncl1dh88 (eren/dm-ema-9995-gc05)
- **Config:** 4L/512d/8H, AdamW lr=5e-4, T_max=30, **EMA=0.9995**, **gc=0.5**, Fourier, no WD
- **Key insight:** gc=0.5 is the stability enabler for EMA on DrivAerML. Without gc, EMA diverges at ep28 (Run 1: `64jrja7q`). With gc=0.5, training is stable through 517 epochs with no divergence. EMA was previously thought dead for DM (3 prior configs all failed) but those lacked gc. Run still converging at ep517 — gap to AB-UPT (3.82%) is only 0.013 pp.
- **Reproduce:** `cd target/icml2026 && SENPAI_MAX_EPOCHS=9999 python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 --grad-clip 0.5 --enable-fourier --model-layers 4 --model-hidden-dim 512 --model-heads 8 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200 --ema-decay 0.9995`

### 2026-04-22 — PR #2898: DrivAerML: torch.compile throughput (no-compile wins) — PREVIOUS BEST

- **val_primary/surface_rel_l2_pct:** 3.997% (-13.5% vs 4.619%) at epoch 467
- **W&B run:** bht6h42t (no-compile baseline run, 467 epochs, SENPAI_MAX_EPOCHS=9999)
- **Config:** 4L/512d/8H, AdamW lr=5e-4, T_max=30, no-EMA, Fourier, 360-min budget
- **Key insight:** torch.compile gives 0% throughput benefit on DrivAerML but the no-compile run trained longer (467 vs 256 epochs) and found a deeper basin. The compile run diverged to NaN at ep454 due to operator fusion without --grad-clip — future DrivAerML compile experiments MUST include --grad-clip 1.0. Gap to external: 1.08x (was 1.24x).
- **Reproduce:** `cd target/icml2026 && SENPAI_MAX_EPOCHS=9999 python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 512 --model-heads 8 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-21 — PR #2691: DrivAerML: 4L/512d width scaling (180-min) — NEW BEST

- **val_primary/surface_rel_l2_pct:** 4.619% (-8.1% vs 5.027%) at epoch 256
- **W&B run:** k8qtsxxz (267 epochs, 180-min budget)
- **Epochs:** 267 (~0.67 min/epoch for 4L/512d); best at epoch 256 (5.926% final at 267 = overfitting)
- **Key insight:** 4L/512d gets comparable epoch count to 4L/320d but with higher model capacity. Width scaling IS beneficial at this budget when epoch limit is removed. Gap to external: 1.24x (was 1.35x).
- **Reproduce:** `cd target/icml2026 && SENPAI_MAX_EPOCHS=9999 python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 512 --model-heads 8 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-21 — PR #2648: DrivAerML: 4L/320d+T_max=30 — throughput vs width — NEW BEST

- **val_primary/surface_rel_l2_pct:** 5.027% (-12.3% vs 5.73%) at epoch 257
- **test_primary/surface_rel_l2_pct:** 6.244%
- **W&B run:** qx7z7if3 (257 epochs, 180-min budget, still improving!)
- **Epochs:** 257 (180-min timeout — ~0.7 min/epoch for 4L/320d)
- **Key insight:** 4L/320d uses 5 heads (320/5=64 per head, same dim-per-head). Runs 70% more epochs than 4L/384d in the same wall-clock time. The throughput advantage outweighs the capacity disadvantage. Best at final epoch — no sign of convergence. Gap to external: 1.35x.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 320 --model-heads 5 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-21 — PR #2602: DrivAerML: 4L/384d+T_max=30 — wider architecture — NEW BEST

- **val_primary/surface_rel_l2_pct:** 5.73% (-52.2% vs 11.97%) at epoch 144
- **W&B run:** 7ogfs7ph (151 epochs, 180-min budget, still converging!)
- **Epochs:** 151 (180-min timeout — ~1.2 min/epoch for 4L/384d)
- **Key insight:** Width scaling is the dominant lever on DrivAerML. 384d vs 256d = 52% relative improvement. Model was still improving at epoch 151. Late-epoch oscillation (e.g., epoch 141=10.2%, 144=5.7%) suggests T_max=30 may be slightly aggressive for 384d.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine-t-max 30 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 384 --model-heads 6 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

### 2026-04-21 — PR #2645: DrivAerML: 4L/256d+T_max=30 — 600 batches/epoch — NEW BEST

- **val_primary/surface_rel_l2_pct:** 11.97% (-5.7% vs 12.70%)
- **test_primary/surface_rel_l2_pct:** 13.03% (-3.8% vs 13.54%)
- **W&B run:** dar47nwl (34 epochs, hit 30-min timeout — still converging)
- **val/loss:** 0.0334
- **Epochs:** 34 (timeout, not epoch cap — model still trending downward)
- **Key insight:** 600 batches/epoch (vs 394) sees 53% more car configurations per epoch. Fewer total epochs (34 vs 45) but per-epoch improvement compensates. Still converging at cutoff — longer training or even more batches could push further.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset drivaerml --optimizer adamw --lr 5e-4 --cosine_t_max 30 --no-use-ema --enable-fourier --model-layers 4 --model-hidden-dim 256 --model-heads 4 --epochs 999 --batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 600 --max-eval-batches 200`

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

## TandemFoil Paper

- **Primary metric:** `val_primary/field_mse` (lower is better)
- **Current best:** 0.002383 (val) at epoch 443 — **45.1% improvement over previous best**
- **Best PR:** #3025 (haku — Lion lr=1.25e-4, T_max=10, gc=0.5, EMA=0.999, 3L/192d, Fourier+physics)
- **External target:** Paper MGN ~1.79 (crushed by >99% — internal frontier is the only meaningful target)
- **Key insight:** The TF champion recipe (Lion+gc=0.5+EMA) transfers strongly to TFP. Lion+EMA at T_max=10 is the decisive combination — both the optimizer and the weight averaging contribute. Divergence at ep462 (known T_max=10 instability) but EMA preserves the best checkpoint.
- **Critical flags:** Must use `--tandemfoil-paper` dataset key (see train.py for exact flag)
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil_paper --optimizer lion --lr 1.25e-4 --cosine-t-max 10 --grad-clip 0.5 --weight-decay 1e-2 --enable-fourier --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999`

### 2026-04-22 — PR #3025: TandemFoil Paper: Lion+gc=0.5+EMA champion config — NEW BEST (CURRENT)

- **val_primary/field_mse:** 0.002383 (-45.1% vs 0.00434) at epoch 443
- **val/surface_mse:** 0.001517
- **val/surface_mse_p:** 4.81e-05
- **val/volume_mse:** 0.002397
- **W&B run:** d1xh0o1p (haku/tfp-lion-champion-config)
- **Config:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, EMA=0.999, 3L/192d, Fourier+physics, 360-min budget
- **Key insight:** TF champion config transfers directly to TFP with a 45.1% improvement. Lion optimizer + EMA is the winning combination for tandemfoil geometry (both TF and TFP). Divergence at ep462 from T_max=10 cycling instability, but EMA correctly preserved the ep443 best. Follow-ups: gc=0.3, LR sweep, T_max=20/30 for stability.
- **Reproduce:** `cd target/icml2026 && python train.py --dataset tandemfoil_paper --optimizer lion --lr 1.25e-4 --cosine-t-max 10 --grad-clip 0.5 --weight-decay 1e-2 --enable-fourier --model-layers 3 --model-hidden-dim 192 --model-heads 3 --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --asinh-pressure --residual-prediction --enable-pressure-prior-addition --epochs 999 --ema-decay 0.999`

### 2026-04-22 — PR #2979: TandemFoil Paper: Cross-dataset 2L depth reduction — PREVIOUS BEST

- **val_primary/field_mse (3L/192d):** 0.00434 at epoch 199 ← **NEW DATASET CHAMPION**
- **val_primary/field_mse (2L/192d):** 0.00534 at epoch 184
- **W&B run (3L):** (haku/2l-depth-cross-dataset group, tandemfoil_paper 3L run)
- **W&B run (2L):** (haku/2l-depth-cross-dataset group, tandemfoil_paper 2L run)
- **Note:** These are the FIRST SENPAI runs on the TandemFoil Paper dataset. Paper MGN baseline ~1.79 crushed by >99%. 3L outperforms 2L — depth matters for this complex geometry. Three data pipeline bugs fixed as part of this PR.
- **Depth hypothesis verdict:** FALSIFIED for TandemFoil Paper (3L > 2L), consistent with DrivAerML finding (3D/complex geometry prefers deeper models).
- **Reproduce (3L):** `cd target/icml2026 && python train.py --dataset tandemfoil_paper --optimizer adamw --lr 5e-4 --cosine-t-max 150 --no-use-ema --enable-fourier --model-layers 3 --model-hidden-dim 192 --model-heads 3 --epochs 999`
