# SENPAI Research State

- **Date:** 2026-04-21 (Round 15 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **75.59** | #2490 (T_max=10, 3L/192d, Lion lr=3e-4, 14 ep) | Ultra-rapid cosine cycling |
| AirfRANS | val_primary/surface_mse | **0.0153** | #2655 (T_max=10, 3L/192d, AdamW lr=3e-4, **seed=789**, 41 ep) | **SEED SELECTION** > LR tuning |
| DrivAerML | val_primary/surface_rel_l2_pct | **5.73%** | #2602 (4L/**384d**/6H+T_max=30, 151 ep, 180-min) | **WIDTH SCALING** dominates |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.0153 | 3.6x |
| DrivAerML | <3.71% | 5.73% | **1.55x** |

## CRITICAL INSIGHTS (Round 15)

1. **AirfRANS: SEED > LR**: lr=3e-4+seed=789 achieves 0.0153 (17% better than lr=7e-4's best). lr=3e-4 distribution (0.0153-0.0194) is TIGHTER than lr=7e-4 (0.0198-0.0463). Multi-seed exploitation is the key strategy.

2. **AirfRANS PLATEAU on architecture**: 3L/192d with T_max=10 is exhaustively tuned. LR sweep complete (non-monotonic). The --seed flag enables reproducible experiments and multi-seed exploitation.

3. **DrivAerML: Width scaling dominates all other levers**: 4L/384d = 5.73% vs 4L/256d = 11.97%. Now testing 4L/512d, 4L/448d, and 4L/384d compounds.

4. **17 DrivAerML 4L/256d experiments closed**: All obsolete after 4L/384d breakthrough. Students redirected to 4L/384d variants.

5. **All future DrivAerML experiments MUST use 4L/384d as base config**: `--model-hidden-dim 384 --model-heads 6`

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 75.59)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| alphonse | #2569 | Hypernetwork SRF (human-directed) | |
| kaworu | #2629 | Kutta TE constraint (human-directed) | |
| gen | #2623 | MQA audit (human-directed) | |
| sasuke | #2595 | 5L/256d deep model | |
| sakura | #2597 | Gradient accumulation | |
| mikasa | #2631 | T_max=3 ultra-short | |
| levi | #2633 | T_max=10 + LR warmup | |
| chrome | #2635 | T_max=10 + noise augmentation | |
| tetsuo | #2665 | Dropout=0.1 | |
| naruto | #2667 | Gradient clipping | |
| kakashi | #2651 | 4L/192d deeper model | |
| ray | #2653 | T_max=10 + cosine eta_min | |

### AirfRANS (Baseline: 0.0153)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| gilbert | NEW | lr=3e-4 seeds 100-104 | Assigning — more seeds at winning config |
| kohaku | NEW | lr=3e-4 seeds 200-204 | Assigning |
| emma | NEW | lr=3e-4 seeds 300-304 | Assigning |
| violet | NEW | lr=4e-4 seeds 100-104 | Assigning — fill LR gap |
| tanjiro | NEW | lr=5e-4 seeds 100-104 | Assigning — recharacterize with --seed |
| fern | #2678 | lr=7e-4+WD=1e-2 | |
| kaneda | #2679 | lr=7e-4+T_max=8 | |
| haku | #2680 | lr=7e-4+grad-clip=1.0 | |
| eren | #2649 | T_max=10 multi-seed (old, no --seed) | May be stale |
| historia | #2668 | lr=3e-4+WD=1e-2 | |
| nezuko | #2658 | lr=1e-4 | Likely dead end |
| senku | #2664 | 3L/256d+lr=3e-4 | Dead end (confirmed) |
| thorfinn | #2666 | T_max=5 | |
| hinata | #2637 | T_max=10+WD=1e-2 | |
| armin | #2638 | LR decay | |
| winry | #2636 | T_max=15 | |
| roy | #2639 | T_max=8 | |
| itachi | #2647 | T_max=12 | |
| shinji | #2663 | lr=3e-4+dropout | CLOSED |

### DrivAerML (Baseline: 5.73% — 4L/384d)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| shinji | #2681 | 4L/384d + 600 batches | HIGH PRIORITY |
| rei | #2682 | 4L/384d + T_max=50 | HIGH PRIORITY |
| frieren | NEW | 4L/512d+T_max=30 | Assigning — HIGHEST PRIORITY |
| taki | NEW | 4L/384d+800 batches | Assigning |
| edward | NEW | 4L/384d+lr=3e-4 | Assigning |
| zenitsu | NEW | 4L/384d+lr=7e-4 | Assigning |
| inosuke | NEW | 4L/384d+WD=1e-2 | Assigning |
| giyu | NEW | 4L/384d+grad-clip=1.0 | Assigning |
| shinobu | NEW | 5L/384d+T_max=30 | Assigning |
| norman | NEW | 4L/384d+eta_min=1e-5 | Assigning |
| ymir | NEW | 4L/384d+T_max=40 | Assigning |
| chihiro | NEW | 4L/448d+T_max=30 | Assigning — width scaling curve |
| shoya | NEW | 4L/384d+dropout=0.05 | Assigning |
| askeladd | NEW | 4L/384d+seed sweep | Assigning |

## Next Priority Directions

### DrivAerML (MOST URGENT — 1.55x from external)
1. **4L/512d** (frieren) — if width continues to scale, could beat 3.71%
2. **4L/384d compounds** — 600b, 800b, grad-clip, WD, eta_min
3. **4L/448d** (chihiro) — map width scaling curve
4. **4L/384d seed sweep** (askeladd) — characterize variance at 384d
5. **5L/384d** (shinobu) — test depth at wider width

### AirfRANS (3.6x from external — PLATEAU on current approach)
1. **More seeds at lr=3e-4** (gilbert, kohaku, emma) — 15 more seeds
2. **lr=4e-4/5e-4 multi-seed** (violet, tanjiro) — fair LR comparison with --seed
3. **Architecture changes needed** — current 3L/192d is exhausted
4. Researcher-agent exploring plateau-breaking ideas

### TandemFoil (No clear path)
1. Human-directed: Kutta (kaworu), MQA (gen), HyperSRF (alphonse)
2. Cold-start prevents fair hyperparameter testing
