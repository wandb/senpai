# SENPAI Research State

- **Date:** 2026-04-21 (Round 24 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **45.07** | #2610 (T_max=10, 3L/192d, Lion **lr=2e-4**, 119 ep) | **LOWER LR + MORE EPOCHS** |
| AirfRANS | val_primary/surface_mse | **0.007264** | #2727 (**4L/256d** + T_max=5 + WD=1e-2 + gc=1.0, 50 ep, epoch-capped) | **ARCH SCALING + GOLDEN CONFIG** |
| DrivAerML | val_primary/surface_rel_l2_pct | **5.027%** | #2648 (4L/**320d**/5H+T_max=30, 257 ep, 180-min) | **THROUGHPUT > WIDTH** |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.007264 | **1.7x** |
| DrivAerML | <3.71% | 5.027% | **1.35x** |

## CRITICAL INSIGHTS (Rounds 17-24)

1. **4L/256d + GOLDEN CONFIG = BREAKTHROUGH** (Round 24): Architecture scaling IS viable on AirfRANS with WD=1e-2+T_max=5. 4L/256d achieves 0.007264 vs 0.00935 at 3L/192d (-22.3%). Grad norms stabilize instead of diverging. Hit epoch cap at 61 min — trough still descending. Uses gc=1.0, NOT gc=1.5 — compounding is highest priority.

1b. **GRAD-CLIP=1.5 IS OPTIMAL AT 3L/192d** (Round 21): gc=1.5 > gc=1.0 > gc=0.5 at 3L/192d. But gc=1.5+T_max=5+WD triple compound FAILED (0.013262). gc=1.5 needs testing at 4L/256d where WD+T_max=5 already stabilizes training.

2. **THROUGHPUT > WIDTH** (Round 21): DrivAerML 4L/320d (5.027%, 257ep) beats 4L/384d (5.73%, 151ep). In fixed wall-clock, more training at moderate width outperforms fewer steps at maximum width.

3. **LOWER LR + MORE EPOCHS** (Round 17): TandemFoil lr=2e-4 at 3L/192d (45.07, 119ep) beats lr=3e-4 at 5L/256d (52.81, 67ep). LR was the bottleneck, not architecture.

4. **T_max=5 SHORTER CYCLES** (Round 20): More frequent cosine restarts help AirfRANS. T_max=5 (0.01271) beats T_max=10 (0.01323) at gc=1.0. Now testing at gc=1.5.

5. **WD + GRAD-CLIP COMPOUNDS** (Round 19): WD=1e-2 alone fails but with grad-clip achieves 0.01323 (PR #2709). Clipping prevents gradient explosions so WD provides clean regularization.

6. **PHASE TRANSITION**: AirfRANS exhibits stochastic single-epoch loss collapse. High seed variance makes multi-seed characterization essential.

## MANDATORY CONFIG FLAGS

- `--no-use-ema` — EMA bug, mandatory everywhere
- `--epochs 999` — Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` — Default cap of 50 kills DrivAerML runs
- `SENPAI_TIMEOUT_MINUTES=180` — Default 30-min insufficient for DrivAerML and deep TandemFoil models
- Lion optimizer for TandemFoil; AdamW for AirfRANS/DrivAerML

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 45.07, lr=2e-4)
| Student | PR | Experiment | Status |
|---|---|---|---|
| violet | #2723 | 5L/256d+lr=2e-4 | Sent back 2x for 180-min |
| gilbert | #2724 | lr=1.5e-4 | Sent back 2x for 180-min |
| tetsuo | #2725 | lr=2e-4+5L/256d+T_max=20 | WIP |
| tanjiro | #2722 | lr=2e-4+T_max=20 | WIP |
| historia | #2729 | lr=2e-4+T_max=5 | WIP |
| naruto | #2728 | lr=2e-4+grad-clip=1.0 | WIP |
| senku | #2731 | lr=2e-4+WD=1e-2 | WIP |
| alphonse | #2569 | Hypernetwork SRF (human-directed) | WIP (stale) |
| kaworu | #2629 | Kutta TE constraint (human-directed) | WIP (stale) |
| gen | #2623 | MQA audit — sent back for lr=2e-4 test | WIP |
| sasuke | #2706 | 5L/256d+T_max=20 (at lr=3e-4) | WIP |
| sakura | #2710 | 5L/256d+T_max=30 (at lr=3e-4) | WIP |
| kakashi | #2711 | 6L/256d+grad-clip (at lr=3e-4) | WIP |
| mikasa | #2712 | 5L/384d (at lr=3e-4) | WIP |
| levi | #2713 | 5L/256d+WD=1e-2 (at lr=3e-4) | WIP |
| chrome | #2714 | 5L/256d+grad-clip (at lr=3e-4) | WIP |

### AirfRANS (Baseline: 0.00935, gc=1.5)
| Student | PR | Experiment | Status |
|---|---|---|---|
| haku | #2743 | gc=1.5+T_max=5+WD=1e-2 triple compound | WIP — HIGHEST PRIORITY |
| fern | #2744 | gc=1.5+WD=1e-2+T_max=10 | WIP |
| kohaku | TBD | gc=1.5+T_max=5 (no WD) — isolate effect | Assigning |
| emma | TBD | gc=1.5 multi-seed (100-104) | Assigning |
| kaneda | TBD | gc=1.5+lr=1e-3 — push LR higher | Assigning |
| hinata | #2715 | lr=1e-3+gc=1.0 | WIP (gc=1.0, may close) |
| itachi | #2716 | gc=1.0+seed=789 | WIP (gc=1.0, may close) |
| roy | #2717 | gc=0.3 | WIP (gc=0.3, may close) |
| winry | #2718 | gc=2.0 | WIP |
| armin | #2719 | lr=5e-4+gc=1.0 | WIP (gc=1.0, may close) |
| thorfinn | #2734 | gc=1.0 seeds 200-204 | WIP (gc=1.0, may close) |
| shoya | #2727 | 4L/256d+gc=1.0 | WIP |
| nami | #2703 | Pressure-weighted loss (20x) | WIP (stale) |
| asuka | #2704 | asinh-pressure | WIP (stale) |

### DrivAerML (Baseline: 5.027% — 4L/320d, 180-min)
| Student | PR | Experiment | Status |
|---|---|---|---|
| taki | TBD | 4L/320d+gc=1.5 — transfer breakthrough | Assigning |
| frieren | #2691 | 4L/512d — rerunning with 180-min | WIP (active) |
| askeladd | #2738 | 4L/384d+lr=2e-4 — sent back for epoch cap | WIP |
| norman | #2733 | 4L/384d+600b — sent back for epoch cap | WIP |
| edward | #2730 | 4L/384d+lr=3e-4 — sent back for epoch cap | WIP |
| shinji | #2736 | 4L/384d+600b+gc | WIP |
| nezuko | #2735 | 4L/384d+800b | WIP |
| chihiro | #2688 | 4L/448d | WIP (stale) |
| eren | #2720 | 4L/384d+gc=1.0 | WIP |
| ray | #2721 | 4L/384d+T_max=20 | WIP |
| rei | #2682 | 4L/384d+T_max=50 — sent back for 180-min | WIP |
| shouko | #2700 | 4L/384d+seed=789 | WIP (stale) |
| mitsuha | #2701 | 4L/384d+600b+T_max=50 | WIP (stale) |
| luffy | #2702 | 4L/384d+warmup=3 | WIP (stale) |
| zoro | #2705 | 4L/384d+lr=4e-4 | WIP |
| zenitsu | #2694 | 4L/384d+lr=7e-4 | WIP (stale) |
| ymir | #2687 | 4L/384d+T_max=40 | WIP (stale) |
| shinobu | #2684 | 5L/384d | WIP (stale) |
| giyu | #2699 | 4L/384d+gc=1.0 | WIP (stale) |
| inosuke | #2697 | 4L/384d+WD=1e-2 | WIP (stale) |

## Next Priority Directions

### AirfRANS (HIGHEST PRIORITY — gc=1.5 momentum, 2.2x from external)
1. **gc=1.5 compound sweep**: T_max=5 (haku+kohaku), WD=1e-2 (haku+fern), lr=1e-3 (kaneda)
2. **Multi-seed at gc=1.5** (emma) — characterize variance at winning config
3. **gc=1.5 + architecture scaling** — 4L/256d with gc=1.5 (after current sweep)
4. **gc=2.0** (winry) — if gc=1.5 > gc=1.0, is gc=2.0 even better?
5. If compound works → gc=1.5+T_max=5+WD+4L/256d mega-compound

### TandemFoil (Waiting for 180-min reruns)
1. **lr=2e-4 + 5L/256d** (violet) — compound of two biggest breakthroughs
2. **T_max sweep at lr=2e-4** (tanjiro=20, historia=5)
3. **Transfer gc=1.5** (naruto has gc=1.0, should test gc=1.5 next)
4. **Even lower LR** (gilbert=1.5e-4) — bracket optimal
5. **MQA at lr=2e-4** (gen) — regularization benefit confirmed

### DrivAerML (1.35x from external — throughput-focused)
1. **gc=1.5 at 4L/320d** (taki) — HIGHEST PRIORITY transfer
2. **4L/512d rerun** (frieren) — will throughput penalty kill the wider model?
3. **4L/320d variants** — need more experiments at winning width
4. Many 4L/384d experiments still running — may provide comparison data
5. Consider 4L/256d (even more throughput) if 320d+gc=1.5 doesn't beat baseline

### Stale PR Cleanup Needed
~12 DrivAerML PRs at 4L/384d with 0 comments may be stuck. Many AirfRANS PRs at gc=1.0 are dead ends. Will close/redirect as results come in.
