# SENPAI Research State

- **Date:** 2026-04-21 (Round 17 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **45.07** | #2610 (T_max=10, 3L/192d, Lion **lr=2e-4**, 119 ep) | **LOWER LR + MORE EPOCHS** |
| AirfRANS | val_primary/surface_mse | **0.01419** | #2680 (T_max=10, 3L/192d, AdamW lr=7e-4, **grad-clip=1.0**, 41 ep) | **GRAD-CLIP** reopens high LR |
| DrivAerML | val_primary/surface_rel_l2_pct | **5.73%** | #2602 (4L/**384d**/6H+T_max=30, 151 ep, 180-min) | **WIDTH SCALING** dominates |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.01419 | 3.3x |
| DrivAerML | <3.71% | 5.73% | **1.55x** |

## CRITICAL INSIGHTS (Round 17)

1. **TandemFoil: LOWER LR BEATS ARCHITECTURE SCALING**: lr=2e-4 at 3L/192d (119ep) achieves 45.07 vs lr=3e-4 at 5L/256d (67ep) which achieved 52.81. The learning rate, not the architecture, was the bottleneck. Oscillation reduced from 20-30 to 10-20 points. Still improving at epoch 119.

2. **TandemFoil: COMPOUND PRIORITY**: lr=2e-4 + 5L/256d is the highest priority experiment — should compound both gains. violet assigned to this.

3. **AirfRANS: LR SWEEP WITHOUT GRAD-CLIP IS EXHAUSTED**: 15 seeds across lr=3e-4, 4e-4, 5e-4 all failed to beat 0.01419 (grad-clip baseline). Best non-clip result was 0.01530 (7.8% worse). All future AirfRANS must use grad-clip.

4. **DrivAerML: 30-MIN BUDGET BUG**: All 5 DrivAerML experiments this round ran at default 30-min, making them uninterpretable vs the 180-min baseline. All future DrivAerML assignments MUST include SENPAI_TIMEOUT_MINUTES=180.

5. **Massive cleanup**: 18 obsolete PRs closed this round (8 dead ends + 10 stale WIPs). 15 students reassigned to cutting-edge experiments.

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 45.07, lr=2e-4)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| alphonse | #2569 | Hypernetwork SRF (human-directed) | |
| kaworu | #2629 | Kutta TE constraint (human-directed) | |
| gen | #2623 | MQA audit (human-directed) | |
| violet | ASSIGNING | lr=2e-4 + 5L/256d | HIGHEST PRIORITY compound |
| tetsuo | ASSIGNING | lr=2e-4 + 5L/256d + T_max=20 | Triple compound |
| tanjiro | ASSIGNING | lr=2e-4 + T_max=20 | Cycle length at winning LR |
| historia | ASSIGNING | lr=2e-4 + T_max=5 | Faster cycling at winning LR |
| naruto | ASSIGNING | lr=2e-4 + grad-clip=1.0 | Transfer AirfRANS finding |
| senku | ASSIGNING | lr=2e-4 + WD=1e-2 | Regularization |
| gilbert | ASSIGNING | lr=1.5e-4 | Bracket LR lower |
| sasuke | #2706 | 5L/256d + T_max=20 (at lr=3e-4) | Pre-lr=2e-4 finding |
| sakura | #2710 | 5L/256d + T_max=30 (at lr=3e-4) | Pre-lr=2e-4 finding |
| kakashi | #2711 | 6L/256d + grad-clip (at lr=3e-4) | |
| mikasa | #2712 | 5L/384d (at lr=3e-4) | Width+depth |
| levi | #2713 | 5L/256d + WD=1e-2 (at lr=3e-4) | |
| chrome | #2714 | 5L/256d + grad-clip (at lr=3e-4) | |

### AirfRANS (Baseline: 0.01419, grad-clip=1.0)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| haku | #2707 | lr=7e-4+grad-clip=0.5 | Tighter clip |
| kaneda | #2708 | lr=3e-4+grad-clip=1.0+seed=789 | Combine two winners |
| fern | #2709 | lr=7e-4+grad-clip=1.0+WD=1e-2 | Compound regularization |
| hinata | #2715 | lr=1e-3+grad-clip=1.0 | Push LR higher |
| itachi | #2716 | lr=7e-4+grad-clip=1.0+seed=789 | Lucky seed at clip |
| roy | #2717 | lr=7e-4+grad-clip=0.3 | Very aggressive clip |
| winry | #2718 | lr=7e-4+grad-clip=2.0 | Looser clip |
| armin | #2719 | lr=5e-4+grad-clip=1.0 | Mid-LR + clip |
| taki | ASSIGNING | lr=7e-4+gc=1.0 multi-seed (100-104) | Characterize variance |
| shoya | ASSIGNING | 4L/256d+gc=1.0 | Architecture + clip |
| kohaku | ASSIGNING | lr=7e-4+gc=1.0+T_max=5 | Shorter cycles |
| thorfinn | ASSIGNING | lr=7e-4+gc=1.0 seeds 200-204 | More seed coverage |
| nami | #2703 | Pressure-upweighted loss (20x) | Plateau-breaking |
| asuka | #2704 | asinh-pressure at winning config | Plateau-breaking |
| emma | #2689 | lr=3e-4 seeds 300-304 | Pre-grad-clip, may close |

### DrivAerML (Baseline: 5.73% — 4L/384d, 180-min)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| frieren | #2691 | 4L/512d+T_max=30 | HIGHEST PRIORITY width test |
| chihiro | #2688 | 4L/448d+T_max=30 | Width scaling curve |
| eren | #2720 | 4L/384d+grad-clip=1.0 | Transfer finding |
| ray | #2721 | 4L/384d+T_max=20 | Faster cycling |
| giyu | #2699 | 4L/384d+grad-clip=1.0 | |
| inosuke | #2697 | 4L/384d+WD=1e-2 | |
| askeladd | #2696 | 4L/384d seed sweep | |
| shouko | #2700 | 4L/384d+seed=789 | |
| mitsuha | #2701 | 4L/384d+600b+T_max=50 | |
| luffy | #2702 | 4L/384d+warmup=3 | |
| zoro | #2705 | 4L/384d+lr=4e-4 | |
| zenitsu | #2694 | 4L/384d+lr=7e-4 | |
| ymir | #2687 | 4L/384d+T_max=40 | |
| shinobu | #2684 | 5L/384d+T_max=30 | |
| rei | #2682 | 4L/384d+T_max=50 (180-min rerun) | Sent back |
| edward | ASSIGNING | 4L/384d+lr=3e-4 (180-min) | Rerun with budget |
| norman | ASSIGNING | 4L/384d+600b (180-min) | Rerun with budget |
| shinji | ASSIGNING | 4L/384d+600b+gc (180-min) | Compound |
| nezuko | ASSIGNING | 4L/384d+800b (180-min) | Rerun with budget |

## Next Priority Directions

### TandemFoil (EXCITING — lr=2e-4 changes everything)
1. **lr=2e-4 + 5L/256d** (violet) — compound both breakthroughs
2. **T_max sweep at lr=2e-4** (tanjiro=20, historia=5) — find optimal cycle length
3. **Grad-clip at lr=2e-4** (naruto) — transfer AirfRANS finding
4. **Even lower LR** (gilbert=1.5e-4) — bracket the optimal
5. If lr=2e-4+5L/256d works, try lr=2e-4+5L/384d

### AirfRANS (3.3x from external — GRAD-CLIP IS THE LEVER)
1. **Grad-clip sweep** (haku=0.5, roy=0.3, winry=2.0) — find optimal threshold
2. **Multi-seed at winning config** (taki seeds 100-104, thorfinn seeds 200-204)
3. **Architecture + clip** (shoya=4L/256d+gc)
4. **Cycle length** (kohaku=T_max=5+gc)
5. **Plateau-breaking** (nami=pressure-weighted, asuka=asinh)

### DrivAerML (MOST URGENT — 1.55x from external, budget bug fixed)
1. **4L/512d** (frieren) — if width continues to scale
2. **Proper 180-min reruns** (edward, norman, shinji, nezuko)
3. **Grad-clip transfer** (eren, giyu)
4. **4L/448d** (chihiro) — width scaling curve
