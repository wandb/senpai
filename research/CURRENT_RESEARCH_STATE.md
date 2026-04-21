# SENPAI Research State

- **Date:** 2026-04-21 (Round 36 complete)
- **Branch:** radford

## ⚡⚡ gc=0.5 BREAKTHROUGH — AirfRANS 0.00277 (35.6% below external target!)

**PR #2774 (roy):** 4L/256d + gc=0.5 + golden config achieved **val_primary/surface_mse = 0.00277** at epoch 150 — **28.9% better than 0.003904 baseline, 35.6% below external target 0.0043**. Only change from baseline: `--grad-clip 0.5`. MERGED.

**gc insight**: Sharper gradient steps (smaller clip threshold) explore the loss landscape more aggressively per epoch, finding 28.9% deeper basins. Same T_max=5 divergence at ep205 (vs ep208 at gc=1.0) — divergence epoch is nearly identical, suggesting it's the LR cycling pattern rather than gc that causes instability.

**gc=0.5 is now the default AirfRANS hyperparameter.**

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **44.72** | #2724 (3L/192d, Lion lr=1.5e-4, T_max=10, 115 ep) | **LOWER LR** |
| AirfRANS | val_primary/surface_mse | **0.00277** | #2774 (4L/256d, gc=0.5, WD=1e-2, T_max=5, 221 ep, 180-min) | **TIGHTER GRADIENT CLIPPING** |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (4L/512d/8H, T_max=30, 267 ep, 180-min) | **WIDTH SCALES** |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | **0.00277** | **CRUSHED by 35.6%** |
| DrivAerML | <3.71% | 4.619% | **1.24x** |

## CRITICAL INSIGHTS (Rounds 17-36)

1. **🔥🔥 gc=0.5 = BIGGEST SINGLE LEVER** (Round 36): 28.9% improvement over gc=1.0 with a single hyperparameter change. Mechanism: sharper gradient steps explore loss landscape more aggressively. Same divergence epoch (~ep205) regardless of gc value — T_max=5 is the divergence cause, not gc level. gc=0.5 is now mandatory default for AirfRANS 4L/256d.

2. **EXTENDED TRAINING = DOMINANT LEVER** (Round 35): SENPAI_MAX_EPOCHS=9999 unlocks 223 epochs at 4L/256d. Progressive descent through 6 phases. Same config as 50-epoch baseline, 2.2x better result.

3. **T_max=5 DIVERGENCE AT ~ep205-208**: All 4L/256d extended runs diverge around ep205-210 regardless of gc value. T_max=5 aggressive LR cycling creates irreversible gradient instability. T_max=10 is the current best hypothesis for prevention.

4. **PRESSURE-WEIGHT CATASTROPHIC AT 4L/256d** (Round 36): 20x weight at 4L/256d → grad norms 300-500, diverged at ep63. 4L/256d amplifies pressure gradients through 4 layers. Lower weight (5x-10x) needed at 4L/256d. 20x works at 3L/192d.

5. **5L DEPTH SCALING: FAST BUT FRAGILE** (Round 35): 5L/256d converges faster but diverges earlier (ep72 vs ep208). gc=0.5 assigned for stability.

6. **4L/256d TOO DEEP FOR TANDEMFOIL** (Round 36): Multiple confirmations. 3L/192d is the sweet spot.

7. **T_max=20 DIVERGES ON DRIVAERML** (Multiple rounds): DrivAerML requires T_max≥30.

## MANDATORY CONFIG FLAGS

- `--no-use-ema` — EMA bug, mandatory everywhere
- `--epochs 999` — Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` — Default cap of 50 kills long runs
- `SENPAI_TIMEOUT_MINUTES=180` — Default 30-min insufficient
- Lion optimizer for TandemFoil; AdamW for AirfRANS/DrivAerML
- **`--grad-clip 0.5` for AirfRANS** — New default after PR #2774 breakthrough

## ACTIVE EXPERIMENTS BY DATASET

### AirfRANS (Baseline: 0.00277, 4L/256d+gc=0.5+WD=1e-2+T_max=5)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| roy | #2818 | gc=0.5 + T_max=10 (prevent divergence) | **CRITICAL** |
| tanjiro | #2819 | gc=0.75 (sweet-spot sweep) | **HIGH** |
| haku | #2820 | gc=0.5 + lr=5e-4 (stability via lower LR) | **HIGH** |
| armin | #2816 | 5L/256d + gc=0.5 extended | HIGH |
| shoya | #2817 | 4L/256d + T_max=10 extended | HIGH |
| edward | #2801 | pressure-weight 20x + 4L/256d + golden | HIGH (may fail like haku) |
| chihiro | #2811 | pressure-weight 10x at 3L/192d | MEDIUM |
| tetsuo | #2809 | pressure-weight 30x at 3L/192d | MEDIUM |
| violet | #2812 | pressure-weight 50x at 3L/192d | MEDIUM |
| nezuko | #2808 | 4L/256d baseline replication seed=789 | LOW |
| luffy | #2807 | lr=3e-4 + T_max=10 at 4L/256d | LOW |
| kohaku | #2800 | lr=5e-4 at 4L/256d | LOW |
| nami | #2703 | pressure-weight 20x at 3L/192d (NEEDS REBASE) | HIGH — unmerged breakthrough |
| nami | #2758 | 5L/256d golden | LOW |
| thorfinn | #2786 | T_max=7 at 4L/256d | LOW |
| winry | #2778 | WD=0 ablation at 4L/256d | LOW |
| itachi | #2771 | 3L/256d golden | LOW |
| hinata | #2770 | WD=5e-3 at 4L/256d | LOW |
| emma | #2768 | lr=5e-4 at 4L/256d | LOW |
| giyu | #2765 | gc=2.0 at 4L/256d | LOW |
| inosuke | #2764 | lr=1e-3 at 4L/256d | LOW |
| mitsuha | #2763 | T_max=10 at 4L/256d | LOW |
| asuka | #2760 | 4L/384d golden | LOW |

### TandemFoil (Baseline: 44.72, 3L/192d+lr=1.5e-4+T_max=10)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| sasuke | #2821 | lr=1.5e-4 + gc=0.5 (transfer gc insight) | **HIGH** |
| gilbert | #2810 | lr=1.25e-4 + gc=1.0 | HIGH — LR trend continues |
| sakura | #2813 | lr=1e-4 at 3L/192d | HIGH — push LR lower |
| kakashi | #2775 | 5L/256d+lr=2e-4 | MEDIUM |
| gen | #2796 | 4L/256d architecture | LOW (4L/256d dead end) |
| historia | #2792 | lr=1.5e-4 (sent back) | MEDIUM |
| shinji | #2789 | 3L/256d wider | LOW |
| norman | #2788 | WD+gc at lr=2e-4 | LOW |
| kaworu | #2754 | golden config lr=2e-4 | LOW |
| alphonse | #2753 | gc=1.5 at lr=2e-4 | LOW |
| senku | #2731 | WD=1e-2+lr=2e-4 | LOW |
| naruto | #2728 | gc=1.0+lr=2e-4 | LOW |
| mikasa | #2777 | WD+gc golden config | LOW |

### DrivAerML (Baseline: 4.619%, 4L/512d/8H+T_max=30)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| ymir | #2822 | 4L/512d + gc=0.5 (transfer gc insight) | **HIGH** |
| zenitsu | #2805 | 4L/640d push width | HIGH |
| shouko | #2803 | 5L/512d depth scaling | HIGH |
| shinobu | #2806 | 4L/512d+lr=7e-4 | HIGH |
| eren | #2804 | 4L/512d+T_max=10 | HIGH |
| rei | #2815 | 4L/512d+T_max=50 | HIGH |
| ray | #2797 | 4L/512d+lr=3e-4 | HIGH |
| kaneda | #2798 | 4L/512d+gc=0.5 (kaneda) | HIGH (now overlaps with ymir) |
| taki | #2814 | 4L/512d+WD=1e-3+gc=0.5 | MEDIUM |
| frieren | #2793 | 4L/512d+gc=1.5 | HIGH |
| fern | #2794 | 4L/512d+WD=1e-2 (likely diverge) | LOW |
| levi | #2779 | 4L/320d+lr=3e-4 | LOW |
| chrome | #2781 | 4L/320d+T_max=10 | LOW |
| zoro | #2782 | 4L/320d+gc=1.5 | LOW |
| askeladd | #2787 | 4L/320d+lr=7e-4 | LOW |

## CRITICAL PENDING ACTIONS

1. **Merge nami #2703** (pressure-weight 20x, 0.00435 at 3L/192d) — BLOCKED on rebase. Note: edward's 20x attempt at 4L/256d may fail too.
2. **Watch roy #2818** — gc=0.5 + T_max=10: if T_max=10 prevents ep205 divergence, model could push below 0.002.
3. **Watch edward #2801** — pressure-weight 20x + golden + extended. May fail like haku (0.023).
4. **DrivAerML 1.24x gap** — massive sweep in progress, ymir's gc=0.5 transfer is the new high-priority test.

## Next Priority Directions

### AirfRANS — PUSH BELOW 0.002
1. **gc=0.5 + T_max=10** (roy #2818) — prevent divergence, extend the deep-basin exploration
2. **gc=0.75 sweep** (tanjiro #2819) — find stable intermediate gc
3. **gc=0.5 + lr=5e-4** (haku #2820) — lower LR as alternative stability mechanism
4. **After stability resolved**: gc=0.5 + 5L/256d compound; gc=0.5 + pressure-weight
5. **Pressure-weight at lower weight** (5x-10x at 4L/256d) — 20x catastrophic, but lower may work

### DrivAerML — CLOSE THE 1.24x GAP
1. **gc=0.5 transfer** (ymir #2822) — highest priority based on AirfRANS success
2. **Width/depth scaling** (zenitsu 4L/640d, shouko 5L/512d)
3. **LR sweep** (ray lr=3e-4, shinobu lr=7e-4)
4. **T_max sweep** (eren T_max=10, rei T_max=50)

### TandemFoil — CONTINUE LR + gc SEARCH
1. **gc=0.5 at lr=1.5e-4** (sasuke #2821) — should suppress cosine peak spikes
2. **lr=1.25e-4 + gc=1.0** (gilbert #2810) — LR descent
3. **lr=1e-4** (sakura #2813) — push lower
