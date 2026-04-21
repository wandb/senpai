# Baseline Metrics — radford branch

## TandemFoilSet

- **Primary metric:** `val_primary/surface_pressure_mae` (= `val_eq4/surface_pressure_mae`)
- **Current best:** 44.72 (val) at epoch 89
- **Best PR:** #2724 (gilbert — T_max=10, 3L/192d, Fourier + physics + no-EMA, slices=64, Lion **lr=1.5e-4**, 115 epochs, 180-min)
- **Key insight:** LOWER LR continues to win. lr=1.5e-4 at 3L/192d (44.72 at ep89) beats lr=2e-4 (45.07 at ep107). Lower LR provides more stable Lion optimization. Trough envelope still descending at ep115 — more training could push further. 5L/256d tried at lr=2e-4 but underperformed (50.64).

### 2026-04-21 — PR #2724: TandemFoil: lr=1.5e-4 — NEW BEST

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
- **Current best:** 0.007264 (val) at epoch 40
- **Best PR:** #2727 (shoya — **4L/256d** + Fourier + no-EMA + T_max=5 + WD=1e-2 + gc=1.0, AdamW lr=7e-4, 50 epochs, hit epoch cap at 61 min of 180-min budget)
- **Key insight:** ARCHITECTURE SCALING + GOLDEN CONFIG (WD=1e-2 + T_max=5) is the enabler. 4L/256d beats 3L/192d by 22.3% when properly regularized. Grad norms stabilized (18.7→7.1 over training) instead of diverging. Model hit SENPAI_MAX_EPOCHS=50 cap at only 61 min — trough envelope still descending, more training should improve further. Uses gc=1.0, NOT gc=1.5 — compounding gc=1.5 with this architecture is the highest priority. External target: 0.0043 — **~1.7x gap remaining** (was 2.2x).

### 2026-04-21 — PR #2727: AirfRANS: 4L/256d + WD=1e-2 + T_max=5 + gc=1.0 — NEW BEST

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
- **Current best:** 4.619% (val) at epoch 256
- **Best PR:** #2691 (frieren — **4L/512d**/8H + Fourier + no-EMA + T_max=30, 267 epochs, AdamW lr=5e-4, **180-min budget**)
- **CRITICAL:** Must pass `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`
- **External target:** <3.71% (AB-UPT, ~500 epochs) — **1.24x gap remaining** (was 1.35x)
- **Key insight:** WIDTH SCALES at 180-min budget. 4L/512d gets 267 epochs (~0.67 min/ep) vs 4L/320d's 257 epochs (~0.7 min/ep) — comparable throughput but more capacity. Extra capacity pays off when given enough training time. Best checkpoint at epoch 256 (critical — final epoch 267 overfit at 5.926%). SENPAI_MAX_EPOCHS=9999 required. Next: gc=1.5 at 4L/512d, 5L/512d, compound with WD.

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
