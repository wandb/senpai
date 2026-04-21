# SENPAI Research State

- **Date:** 2026-04-21 (Round 30 complete)
- **Branch:** radford

## ⚠ PRESSURE-WEIGHT BREAKTHROUGH — PENDING MERGE

**PR #2703 (nami):** Pressure-weighted loss (20x) achieved **val_primary/surface_mse = 0.00435** at 3L/192d — **40% better than baseline, MATCHES external target 0.0043.** PR reopened, needs rebase before merge. Two students (edward, haku) assigned to independently implement and test at 4L/256d.

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **45.07** | #2610 (T_max=10, 3L/192d, Lion **lr=2e-4**, 119 ep) | **LOWER LR + MORE EPOCHS** |
| AirfRANS | val_primary/surface_mse | **0.007264** | #2727 (**4L/256d** + T_max=5 + WD=1e-2 + gc=1.0, 50 ep, epoch-capped) | **ARCH SCALING + GOLDEN CONFIG** |
| AirfRANS (unmerged) | val_primary/surface_mse | **0.00435** | #2703 (3L/192d, lr=3e-4, T_max=10, **pressure-weight=20**) | **LOSS REWEIGHTING** |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (4L/**512d**/8H+T_max=30, 267 ep, 180-min) | **WIDTH SCALES AT 180-MIN** |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best (merged) | Our Best (unmerged) | Gap |
|---|---|---|---|---|
| AirfRANS | 0.0043 | 0.007264 | **0.00435** | **≈1.0x (MATCHED!)** |
| DrivAerML | <3.71% | 4.619% | — | **1.24x** |

## CRITICAL INSIGHTS (Rounds 17-30)

1. **🔥 PRESSURE-WEIGHTED LOSS = BIGGEST SINGLE IMPROVEMENT** (Round 30): 20x upweighting of pressure channel MSE achieves 0.00435 at 3L/192d — 40% better than 4L/256d golden config baseline. Fixes fundamental gradient misallocation: pressure dominates composite MSE but gets equal gradient share. Phase transition delayed (ep117 vs ep23) but converges much deeper. EVERY future AirfRANS experiment should use `--pressure-loss-weight 20`.

2. **4L/256d + GOLDEN CONFIG = BREAKTHROUGH** (Round 24): 4L/256d achieves 0.007264 vs 0.00935 at 3L/192d (-22.3%). Grad norms stabilize (18.7→7.1). Hit epoch cap at 61 min — trough still descending.

3. **gc=1.5 DEAD AT 4L/256d** (Rounds 27-29): 5 independent confirmations. Deeper architectures amplify gradients making gc=1.5 harmful.

4. **WIDTH SCALES ON DrivAerML** (Round 28): 4L/512d (4.619%) beats 4L/320d (5.027%). Capacity wins over throughput at 180-min uncapped.

5. **WD=1e-2 DOES NOT TRANSFER TO DrivAerML** (Round 29): Catastrophic divergence. DrivAerML needs milder regularization.

5. **T_max=3 TOO SHORT FOR 4L/256d** (Round 29): 0.011601 vs 0.007264 baseline. T_max hierarchy: T_max=5 ≈ baseline > T_max=3 (1.6x worse).

6. **LOWER LR + MORE EPOCHS** (Round 17): TandemFoil lr=2e-4 at 3L/192d (45.07, 119ep) beats lr=3e-4 at 5L/256d (52.81, 67ep).

7. **WD + GRAD-CLIP COMPOUNDS on AirfRANS** (Round 19): WD=1e-2 alone fails but with gc=1.0 achieves breakthroughs. Clipping prevents gradient explosions so WD provides clean regularization.

8. **30-MIN TIMEOUT KILLS 4L/256d AirfRANS**: Model gets 36-37 epochs vs baseline's 50 (61 min). ALL 4L/256d AirfRANS experiments MUST use SENPAI_TIMEOUT_MINUTES≥60.

## MANDATORY CONFIG FLAGS

- `--no-use-ema` — EMA bug, mandatory everywhere
- `--epochs 999` — Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` — Default cap of 50 kills long runs
- `SENPAI_TIMEOUT_MINUTES=180` — Default 30-min insufficient for DrivAerML and 4L+ models
- Lion optimizer for TandemFoil; AdamW for AirfRANS/DrivAerML

## ACTIVE EXPERIMENTS BY DATASET

### AirfRANS (Baseline: 0.007264, 4L/256d+gc=1.0+WD=1e-2+T_max=5)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| armin | NEW | 5L/256d+golden config | HIGH — depth scaling |
| kohaku | NEW | 4L/256d+lr=5e-4+golden | HIGH — LR sweep |
| emma | #2768 | 4L/256d+lr=5e-4 (sent back for timeout) | HIGH |
| haku | #2791 | gc=1.5+no WD+T_max=10 (cleanest gc=1.5 test) | MEDIUM |
| hinata | #2770 | WD=5e-3 | MEDIUM |
| roy | #2774 | gc=0.5 | MEDIUM |
| chihiro | #2780 | 4L/320d+golden | MEDIUM |
| itachi | #2771 | 3L/256d (width vs depth) | MEDIUM |
| nezuko | #2778 | WD=0 ablation | MEDIUM |
| shinji | #2763 | T_max=10 (gc=1.0) | LOW |
| winry | #2765 | gc=2.0 | LOW |
| giyu | #2764 | lr=1e-3 | LOW |
| edward | #2762 | gc=1.5+T_max=10 | LOW (gc=1.5 dead) |
| norman | #2784 | gc=1.5+WD=5e-3 | LOW (gc=1.5 dead) |
| taki | #2785 | gc=1.5+no WD | LOW (gc=1.5 dead) |
| shouko | #2786 | T_max=7 | LOW |

### TandemFoil (Baseline: 45.07, 3L/192d+lr=2e-4+T_max=10)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| kakashi | #2775 | 5L/256d+lr=2e-4 | HIGH — compound breakthrough |
| sasuke | #2772 | 4L/256d+lr=2e-4 | HIGH — arch transfer |
| gen | #2796 | 4L/256d transfer | HIGH |
| historia | #2792 | lr=1.5e-4 | HIGH — LR sweep |
| mikasa | #2777 | WD=1e-2+gc=1.0 transfer | HIGH |
| sakura | #2773 | T_max=5 | MEDIUM |
| senku | #2788 | WD+gc at lr=2e-4 | MEDIUM |
| violet | #2789 | 3L/256d wider | MEDIUM |

### DrivAerML (Baseline: 4.619%, 4L/512d/8H+T_max=30)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| shouko | NEW | 5L/512d depth scaling | HIGH |
| eren | NEW | 4L/512d+T_max=10 | HIGH |
| zenitsu | NEW | 4L/640d push width | HIGH |
| shinobu | NEW | 4L/512d+lr=7e-4 | HIGH |
| ray | #2797 | 4L/512d+lr=3e-4 | HIGH |
| kaneda | #2798 | 4L/512d+gc=0.5 | HIGH |
| frieren | #2793 | 4L/512d+gc=1.5 | HIGH |
| fern | #2794 | 4L/512d+WD=1e-2 | MEDIUM (WD may diverge) |
| rei | #2795 | 4L/512d+T_max=20 | HIGH |
| levi | #2779 | 4L/320d+lr=3e-4 | LOW (superseded arch) |
| chrome | #2781 | 4L/320d+T_max=10 | LOW |
| zoro | #2782 | 4L/320d+gc=1.5 | LOW |
| askeladd | #2787 | 4L/320d+lr=7e-4 | LOW |

## Next Priority Directions

### AirfRANS — PRESSURE-WEIGHT IS THE DOMINANT LEVER
1. **🔥 Merge PR #2703 (nami pressure-weight)** — BLOCKED on rebase. Once merged, ALL AirfRANS experiments should use `--pressure-loss-weight 20`.
2. **Pressure-weight at 4L/256d** (edward + haku) — combining the biggest single improvement (pressure-weight) with the best architecture (4L/256d+golden). Expected to push well below 0.0043.
3. **Pressure-weight sweep** — after merging: test 10x, 15x, 30x, 50x at 4L/256d
4. **lr=3e-4 + T_max=10 at golden config** (luffy) — test nami's hyperparams without pressure-weight to isolate effects
5. **Seed=789 + golden config replication** (nezuko) — extended baseline with proper timeout/epoch settings
6. **Shoya #2755 extended run** — still the most critical pending result for non-pressure-weight baseline
7. After pressure-weight merge: pressure-weight + 5L/256d, pressure-weight + 4L/320d

### DrivAerML (1.24x from external — aggressive 4L/512d sweep)
1. **Width scaling** (zenitsu: 4L/640d) — push the width frontier
2. **Depth scaling** (shouko: 5L/512d) — depth + width compound
3. **LR sweep** (ray: lr=3e-4, shinobu: lr=7e-4) — bracket optimal LR
4. **T_max sweep** (eren: T_max=10, rei: T_max=20) — faster cycling
5. **Regularization** (kaneda: gc=0.5, frieren: gc=1.5) — gc-only is safer than WD
6. After results: consider pressure-weight transfer to DrivAerML
7. Watch fern #2794 (WD=1e-2) — likely to diverge

### TandemFoil (Focus on sustained improvement)
1. **5L/256d+lr=2e-4** (kakashi #2775) — compound of depth+LR breakthroughs
2. **4L/256d+lr=2e-4** (sasuke #2772, gen #2796) — architecture transfer
3. **lr=1.5e-4** (historia #2792) — push LR even lower
4. **Golden config transfer** (mikasa #2777, senku #2788) — WD+gc from AirfRANS
5. After AirfRANS pressure-weight merge: test pressure-weighting concept on TandemFoil
