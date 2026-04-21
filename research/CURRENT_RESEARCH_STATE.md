# SENPAI Research State

- **Date:** 2026-04-21 (Round 29 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **45.07** | #2610 (T_max=10, 3L/192d, Lion **lr=2e-4**, 119 ep) | **LOWER LR + MORE EPOCHS** |
| AirfRANS | val_primary/surface_mse | **0.007264** | #2727 (**4L/256d** + T_max=5 + WD=1e-2 + gc=1.0, 50 ep, epoch-capped) | **ARCH SCALING + GOLDEN CONFIG** |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (4L/**512d**/8H+T_max=30, 267 ep, 180-min) | **WIDTH SCALES AT 180-MIN** |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.007264 | **1.7x** |
| DrivAerML | <3.71% | 4.619% | **1.24x** |

## CRITICAL INSIGHTS (Rounds 17-29)

1. **4L/256d + GOLDEN CONFIG = BREAKTHROUGH** (Round 24): 4L/256d achieves 0.007264 vs 0.00935 at 3L/192d (-22.3%). Grad norms stabilize (18.7→7.1). Hit epoch cap at 61 min — trough still descending.

2. **gc=1.5 DEAD AT 4L/256d** (Rounds 27-29): 5 independent confirmations. gc=1.5 fails with WD, without WD, at different T_max, at different LR. Deeper architectures amplify gradients making gc=1.5 harmful. gc=1.5 ONLY works at 3L/192d.

3. **WIDTH SCALES ON DrivAerML** (Round 28): 4L/512d (4.619%, 267ep) beats 4L/320d (5.027%, 257ep). At 180-min with no epoch cap, capacity wins over throughput. Old "throughput > width" insight was wrong for uncapped training.

4. **WD=1e-2 DOES NOT TRANSFER TO DrivAerML** (Round 29): Catastrophic divergence at 4L/320d (14.40%, grad norms 231x). DrivAerML's 3D geometry needs much milder regularization (WD≤1e-3 or gc-only).

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
| ray | NEW | 4L/512d+lr=3e-4 | HIGH — LR sweep |
| kaneda | NEW | 4L/512d+gc=0.5 | HIGH — regularization |
| frieren | #2793 | 4L/512d+gc=1.5 | HIGH |
| fern | #2794 | 4L/512d+WD=1e-2 | HIGH (but WD may diverge!) |
| rei | #2795 | 4L/512d+T_max=20 | HIGH |
| levi | #2779 | 4L/320d+lr=3e-4 | LOW (superseded arch) |
| chrome | #2781 | 4L/320d+T_max=10 | LOW (superseded arch) |
| zoro | #2782 | 4L/320d+gc=1.5 | LOW (superseded arch) |
| eren | #2783 | 4L/320d+WD+T_max=20 | LOW (superseded, WD will diverge) |
| askeladd | #2787 | 4L/320d+lr=7e-4 | LOW (superseded arch) |

## Next Priority Directions

### AirfRANS (HIGHEST PRIORITY — 1.7x from external)
1. **Shoya #2755 full 180-min rerun** — most critical pending experiment. Current baseline hit epoch cap at 50 epochs. Full budget could push well below 0.005.
2. **Depth scaling** (armin: 5L/256d) — 3L→4L gave 22.3%, can 4L→5L give another step?
3. **LR sweep at golden config** (kohaku: lr=5e-4, emma: lr=5e-4) — lower LR was TandemFoil breakthrough
4. **Width scaling** (chihiro: 4L/320d) — more capacity at golden config
5. **WD ablation** (hinata: 5e-3, nezuko: WD=0) — is WD=1e-2 optimal or can it be tuned?
6. If gc=1.5 is truly dead at 4L/256d, try gc=0.7 or gc=0.8 — intermediate values

### DrivAerML (1.24x from external — 4L/512d focus)
1. **LR sweep** (ray: lr=3e-4) — lower LR at wider model
2. **gc-only regularization** (kaneda: gc=0.5) — WD=1e-2 catastrophically diverges, gc alone is safer
3. **gc=1.5 transfer** (frieren: #2793) — worked at 3L/192d AirfRANS, might work here
4. **T_max tuning** (rei: T_max=20) — baseline uses T_max=30, faster cycling might help
5. Watch fern #2794 (WD=1e-2) — likely to diverge based on ray's result
6. After initial 4L/512d sweep: try 5L/512d or 4L/640d for even more capacity

### TandemFoil (No external target, focus on sustained improvement)
1. **5L/256d+lr=2e-4** (kakashi #2775) — compound of depth+LR breakthroughs
2. **4L/256d+lr=2e-4** (sasuke #2772, gen #2796) — architecture transfer
3. **lr=1.5e-4** (historia #2792) — push LR even lower
4. **Golden config transfer** (mikasa #2777, senku #2788) — WD+gc from AirfRANS
