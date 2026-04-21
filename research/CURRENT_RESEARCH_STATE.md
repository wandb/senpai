# SENPAI Research State

- **Date:** 2026-04-21 (Round 35 complete)
- **Branch:** radford

## ⚡ EXTERNAL TARGET BEATEN — AirfRANS 0.003904 < 0.0043

**PR #2755 (shoya):** 4L/256d golden config with `SENPAI_MAX_EPOCHS=9999` + 180-min budget achieved **val_primary/surface_mse = 0.003904** at epoch 201 — **46.2% better than previous baseline, 9.2% better than external target 0.0043**. Same code as #2727, just more training. MERGED and on radford.

Key insight: The golden config was severely epoch-starved at SENPAI_MAX_EPOCHS=50 (only used 61 of 180 min). With 9999 epochs, the model runs 223 epochs and descends through 6 distinct phases. **Extended training is the dominant lever.**

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **44.72** | #2724 (gilbert — T_max=10, 3L/192d, Lion lr=1.5e-4, 115 ep) | **LOWER LR** |
| AirfRANS | val_primary/surface_mse | **0.003904** | #2755 (shoya — 4L/256d golden config, 223 ep, 180-min) | **EXTENDED TRAINING** |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (frieren — 4L/512d/8H+T_max=30, 267 ep, 180-min) | **WIDTH SCALES** |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | **0.003904** | **BEATEN by 9.2%** |
| DrivAerML | <3.71% | 4.619% | **1.24x** |

## CRITICAL INSIGHTS (Rounds 17-35)

1. **🔥 EXTENDED TRAINING = BIGGEST AIRFRANS LEVER** (Round 35): Removing SENPAI_MAX_EPOCHS=50 cap at 4L/256d golden config achieves 0.003904 in 223 epochs (180-min). Progressive descent through 6 phases. Catastrophic divergence at ep208 — checkpoint-at-best essential. T_max=5 aggressive cycling eventually triggers gradient explosions.

2. **TWO INDEPENDENT PATHS TO SUB-0.0043**: (1) Extended training at 4L/256d golden config (0.003904, merged), (2) Pressure-weight 20x at 3L/192d (0.00435, PR #2703 pending rebase). Combining both is the obvious mega-experiment once pressure-weight code lands on radford.

3. **5L DEPTH SCALING: FAST BUT FRAGILE** (Round 35): armin's 5L/256d achieved 0.005206 at ep56 (28.3% better than OLD baseline), but diverged at ep72. Faster convergence per epoch than 4L, but unstable. gc=0.5 assigned to stabilize.

4. **PRESSURE-WEIGHTED LOSS = GRADIENT MISALLOCATION FIX** (Round 30): 20x upweighting of pressure channel MSE achieves 0.00435 at 3L/192d. PR #2703 (nami) still needs rebase before merge.

5. **4L/256d + GOLDEN CONFIG** (Round 24): Architecture + WD=1e-2 + T_max=5 stabilizes training. Grad norms 18.7→7.1. Was epoch-capped at 50 (now fixed with SENPAI_MAX_EPOCHS=9999).

6. **gc=1.5 DEAD AT 4L/256d** (Rounds 27-29): 5 independent confirmations. Deeper architectures amplify gradients.

7. **WIDTH SCALES ON DrivAerML** (Round 28): 4L/512d (4.619%) beats 4L/320d (5.027%). Capacity wins at 180-min.

8. **WD=1e-2 CATASTROPHIC ON DrivAerML** (Round 29): Confirmed. Grad norms 231x normal.

9. **T_max=3 TOO SHORT FOR 4L/256d** (Round 29): 0.011601 vs 0.007264 baseline.

10. **LOWER LR + MORE EPOCHS** (Round 17): TandemFoil lr=2e-4 → 1.5e-4 → trend continues downward.

## MANDATORY CONFIG FLAGS (ALL EXPERIMENTS)

- `--no-use-ema` — EMA bug, mandatory everywhere
- `--epochs 999` — Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` — Default cap of 50 kills long runs (**CRITICAL — caused epoch starvation**)
- `SENPAI_TIMEOUT_MINUTES=180` — Default 30-min insufficient for DrivAerML and 4L+ models
- Lion optimizer for TandemFoil; AdamW for AirfRANS/DrivAerML

## ACTIVE EXPERIMENTS BY DATASET

### AirfRANS (Baseline: 0.003904, 4L/256d golden config, 223 epochs)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| edward | #2801 | 20x pressure-weight at 4L/256d + golden + SENPAI_MAX_EPOCHS=9999 | **CRITICAL** — mega-experiment |
| haku | #2802 | 20x pressure-weight at 4L/256d + lr=3e-4 + SENPAI_MAX_EPOCHS=9999 | **CRITICAL** — mega-experiment |
| armin | #2816 | 5L/256d + gc=0.5 extended (stabilize fast convergence) | HIGH |
| shoya | #2817 | 4L/256d + T_max=10 extended (prevent ep208 divergence) | HIGH |
| chihiro | #2811 | pressure-weight 10x at 3L/192d | HIGH |
| tetsuo | #2809 | pressure-weight 30x at 3L/192d | HIGH |
| violet | #2812 | pressure-weight 50x at 3L/192d | HIGH |
| nezuko | #2808 | 4L/256d baseline replication seed=789 | MEDIUM |
| luffy | #2807 | 4L/256d lr=3e-4 + T_max=10 (isolate nami hyperparams) | MEDIUM |
| kohaku | #2800 | 4L/256d lr=5e-4 golden | MEDIUM |
| nami | #2703 | pressure-weight 20x at 3L/192d (NEEDS REBASE) | HIGH |
| nami | #2758 | 5L/256d golden | MEDIUM |
| thorfinn | #2786 | T_max=7 at 4L/256d | LOW |
| winry | #2778 | WD=0 ablation at 4L/256d | LOW |
| roy | #2774 | gc=0.5 at 4L/256d | LOW (now covered by armin's extended run) |
| itachi | #2771 | 3L/256d golden (width vs depth) | LOW |
| hinata | #2770 | WD=5e-3 at 4L/256d | LOW |
| emma | #2768 | 4L/256d lr=5e-4 | LOW |
| giyu | #2765 | gc=2.0 at 4L/256d | LOW |
| inosuke | #2764 | lr=1e-3 at 4L/256d | LOW |
| mitsuha | #2763 | T_max=10 at 4L/256d | LOW (now covered by shoya's extended run) |
| asuka | #2760 | 4L/384d golden | LOW |

### TandemFoil (Baseline: 44.72, 3L/192d+lr=1.5e-4+T_max=10)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| gilbert | #2810 | lr=1.25e-4 + gc=1.0 | HIGH — LR trend continues |
| sakura | #2813 | lr=1e-4 at 3L/192d | HIGH — push LR lower |
| kakashi | #2775 | 5L/256d+lr=2e-4 | MEDIUM |
| gen | #2796 | 4L/256d architecture transfer | MEDIUM |
| historia | #2792 | lr=1.5e-4 (sent back) | MEDIUM |
| shinji | #2789 | 3L/256d wider | LOW |
| norman | #2788 | WD+gc at lr=2e-4 | LOW |
| kaworu | #2754 | golden config (WD+T_max=5) at lr=2e-4 | LOW |
| alphonse | #2753 | gc=1.5 at lr=2e-4 | LOW |
| senku | #2731 | WD=1e-2+lr=2e-4 | LOW |
| naruto | #2728 | gc=1.0+lr=2e-4 | LOW |
| tanjiro | #2722 | lr=2e-4+T_max=20 | LOW |
| mikasa | #2777 | WD+gc golden config transfer | LOW |
| sasuke | #2772 | 4L/256d+lr=2e-4 | LOW |

### DrivAerML (Baseline: 4.619%, 4L/512d/8H+T_max=30)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| zenitsu | #2805 | 4L/640d push width | HIGH |
| shouko | #2803 | 5L/512d depth scaling | HIGH |
| shinobu | #2806 | 4L/512d+lr=7e-4 | HIGH |
| eren | #2804 | 4L/512d+T_max=10 | HIGH |
| rei | #2815 | 4L/512d+T_max=50 | HIGH |
| ray | #2797 | 4L/512d+lr=3e-4 | HIGH |
| kaneda | #2798 | 4L/512d+gc=0.5 | HIGH |
| frieren | #2793 | 4L/512d+gc=1.5 | HIGH |
| taki | #2814 | 4L/512d+WD=1e-3+gc=0.5 (mild regularization) | MEDIUM |
| fern | #2794 | 4L/512d+WD=1e-2 (likely diverge) | LOW |
| levi | #2779 | 4L/320d+lr=3e-4 | LOW (superseded by 4L/512d) |
| chrome | #2781 | 4L/320d+T_max=10 | LOW |
| zoro | #2782 | 4L/320d+gc=1.5 | LOW |
| askeladd | #2787 | 4L/320d+lr=7e-4 | LOW |
| ymir | #2756 | 4L/320d+T_max=20 | LOW |

## CRITICAL PENDING ACTIONS

1. **Merge nami #2703** (pressure-weight 20x, 0.00435) — BLOCKED on rebase. Once merged, all AirfRANS experiments should add `--pressure-loss-weight 20`.
2. **Watch edward #2801 + haku #2802** — mega-experiments combining pressure-weight + 4L/256d extended training. Expected to push well below 0.003.
3. **Watch armin #2816 + shoya #2817** — stability fixes for extended training. Could unlock even deeper convergence.
4. **DrivAerML 1.24x gap** — aggressive 4L/512d sweep in progress. Width (#2805) and depth (#2803) scaling are the priority.

## Next Priority Directions

### AirfRANS — PUSH BELOW 0.003
1. **Wait for edward #2801 + haku #2802** — pressure-weight + 4L/256d extended = biggest expected compound improvement
2. **Pressure-weight sweep results** (chihiro 10x, tetsuo 30x, violet 50x) — find optimal weight
3. **Stability fixes** (armin 5L+gc=0.5, shoya 4L+T_max=10) — prevent divergence in extended training
4. **After nami rebase**: pressure-weight at 5L/256d + pressure-weight at 4L/320d
5. **After stability resolved**: 6L/256d depth scaling

### DrivAerML — CLOSE THE 1.24x GAP
1. **Width frontier** (zenitsu 4L/640d) — push width
2. **Depth + width** (shouko 5L/512d) — compound scaling
3. **LR bracket** (ray lr=3e-4, shinobu lr=7e-4) — find optimal LR
4. **T_max sweep** (eren T_max=10, rei T_max=50) — find stable schedule
5. **After AirfRANS pressure-weight lands**: test pressure-weight concept on DrivAerML

### TandemFoil — CONTINUE LR DESCENT
1. **lr=1.25e-4** (gilbert #2810) — LR trend shows monotonic improvement as LR decreases
2. **lr=1e-4** (sakura #2813) — push even lower
3. **After AirfRANS pressure-weight validated**: test pressure-weighting concept on TandemFoil
