# SENPAI Research State

- **Date:** 2026-04-21 (Round 14 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **75.59** | #2490 (T_max=10, 3L/192d, Lion lr=3e-4, 14 ep) | Ultra-rapid cosine cycling (~75 cycles/epoch) |
| AirfRANS | val_primary/surface_mse | **0.01841** | #2646 (T_max=10, 3L/192d, AdamW lr=7e-4, 41 ep) | Phase transition at epoch 35 (earlier with higher LR) |
| DrivAerML | val_primary/surface_rel_l2_pct | **5.73%** | #2602 (4L/**384d**/6H+T_max=30, 151 ep, 180-min budget) | **WIDTH SCALING** — 384d vs 256d gives 52% improvement |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.01841 | 4.3x |
| DrivAerML | <3.71% | 5.73% | **1.55x** (was 3.2x!) |

## ALL CONFIRMED DEAD ENDS

| Direction | Reason |
|---|---|
| ANP decoder | +5.4% worse |
| EMA | Universally harmful |
| Lion on AirfRANS/DrivAerML | AdamW consistently better |
| AdamW on TandemFoil | Gap widens with training |
| Fourier+physics on AirfRANS | Metric space incompatibility |
| 6L deep model | Diverges |
| Reynolds-stratified sampling | All worse |
| geometry_supernodes/surface_anchor flags | NO-OP |
| DrivAerML T_max=10 (any arch) | 15.47-17.08% — too fast |
| DrivAerML 5L/256d | 13.24-13.62% — instability beyond 4L |
| DrivAerML 5L/256d (giyu old PR) | Stale — closed |
| DrivAerML 3L/256d | Worse than 3L/192d |
| DrivAerML T_max=15/20/25 | Worse than T_max=30 |
| DrivAerML lr=3e-4/4e-4/6e-4/1e-3 | Worse than lr=5e-4 |
| DrivAerML seed=1 (multi-seed) | 43.93% — extreme seed sensitivity |
| DrivAerML 10-ep warmup+lr=3e-4 | 51.15% at 2ep — warmup too long |
| DrivAerML eta_min=1e-5 (4L/256d) | 12.47% — doesn't beat 5.73% new baseline |
| AirfRANS 3L/256d+T_max=10 | 0.0357 — too slow for phase transition |
| AirfRANS 4L/256d+T_max=10 | 0.0881 — too slow per epoch |
| AirfRANS lr=2e-4+T_max=10 | 0.0306 — too conservative |
| AirfRANS dropout=0.1 | 0.029072 — disrupts phase transition |
| TandemFoil slices=32 | No speedup (data loading dominates) |
| TandemFoil SCA | 107.62 — epoch overhead fatal (14→8 ep) |

## CRITICAL INSIGHTS (Round 14)

1. **DrivAerML WIDTH SCALING is the dominant lever**: 4L/384d gives 5.73% vs 4L/256d's 11.97% — a 52% relative improvement. Model ran 151 epochs in 180-min budget. Still improving at cutoff. Width scaling > all other levers tested.

2. **DrivAerML external target within reach**: 5.73% vs external 3.71% = 1.55x gap. With 4L/384d + 600 batches + 180-min + potential T_max tuning, sub-4% is plausible.

3. **4L/384d T_max optimization needed**: Late-epoch oscillation (141=10.2%, 144=5.7%) suggests T_max=30 slightly too aggressive for 384d. Try T_max=50.

4. **4L/512d is the next width step**: If 256→384d gave 52% improvement, 384→512d may push toward the external target. Memory must be monitored.

5. **All WIP DrivAerML (4L/256d) experiments are now obsolete**: Results will be compared against 5.73% instead of 11.97%. Most will fail to beat the new baseline. When they complete, promising architectures should be re-run with 4L/384d.

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 75.59)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| violet | #2675 | WD=1e-2 | In progress |
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

### AirfRANS (Baseline: 0.01841)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| emma | #2673 | lr=6e-4+T_max=10 | |
| kohaku | #2671 | lr=7e-4 multi-seed (5 seeds) | HIGH PRIORITY |
| edward | #2674 | lr=8e-4+T_max=10 | |
| fern | #2678 | lr=7e-4+WD=1e-2 | |
| kaneda | #2679 | lr=7e-4+T_max=8 | |
| haku | #2680 | lr=7e-4+grad-clip=1.0 | |
| gilbert | #2655 | lr=3e-4 multi-seed | |
| eren | #2649 | T_max=10 multi-seed | |
| historia | #2668 | lr=3e-4+WD=1e-2 | |
| nezuko | #2658 | lr=1e-4 | |
| senku | #2664 | 3L/256d+lr=3e-4 | Likely dead end |
| thorfinn | #2666 | T_max=5 | |
| hinata | #2637 | T_max=10+WD=1e-2 | |
| armin | #2638 | LR decay | |
| winry | #2636 | T_max=15 | |
| roy | #2639 | T_max=8 | |
| itachi | #2647 | T_max=12 | |

### DrivAerML (Baseline: **5.73%** — 4L/384d)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| shinji | NEW | 4L/384d + 600 batches | Assigning — HIGH PRIORITY |
| rei | NEW | 4L/384d + T_max=50 | Assigning — HIGH PRIORITY |
| frieren | #2669 | 800 batches (4L/256d) | Will likely not beat 5.73% |
| taki | #2670 | 1000 batches (4L/256d) | Will likely not beat 5.73% |
| norman | #2672 | 600 batch+WD=1e-2 (4L/256d) | Will likely not beat 5.73% |
| shoya | #2676 | 600 batch+grad-clip (4L/256d) | Will likely not beat 5.73% |
| askeladd | #2677 | 600 batch+dropout (4L/256d) | Will likely not beat 5.73% |
| tanjiro | #2641 | Warmup+600 batch (4L/256d) | Will likely not beat 5.73% |
| zenitsu | #2640 | T_max=40 (4L/256d) | Stale baseline |
| luffy | #2650 | Dropout=0.1 (4L/256d) | Stale baseline |
| nami | #2654 | Grad-clip=1.0 (4L/256d) | Stale baseline |
| mitsuha | #2660 | Cosine warmup (4L/256d) | Stale baseline |
| shouko | #2659 | lr=5.5e-4 (4L/256d) | Stale baseline |
| chihiro | #2656 | LR decay (4L/256d) | Stale baseline |
| shoya | #2662 | lr=3e-4 warmup (4L/256d) | CLOSED |
| zoro | #2648 | 4L/320d | Interesting intermediate |
| shinobu | #2634 | Grad-accum=2 (4L/256d) | Stale baseline |
| giyu | #2632 | 25k surface pts (4L/256d) | Stale baseline |
| inosuke | #2630 | WD=0 (4L/256d) | Stale baseline |
| ymir | #2628 | T_max=35 (4L/256d) | Stale baseline |
| asuka | #2652 | eval-batches=400 (4L/256d) | Stale baseline |
| chihiro | #2620 | 4L/256d replication | Stale baseline |

## Next Priority Directions

### DrivAerML (MOST URGENT — 1.55x from external target)
1. **4L/384d + 600 batches** — shinji (assigning) — compound width+data levers
2. **4L/384d + T_max=50** — rei (assigning) — reduce late-epoch oscillation
3. **4L/512d** — next width step — assign when students free up
4. **5L/384d** — test depth at new width (5L/256d diverged but 5L/384d might not)
5. **4L/384d + lr=5e-4 + longer training** — 300-min budget if possible

### AirfRANS (4.3x from external target)
1. **lr=7e-4 multi-seed** (kohaku) — exploit stochastic phase transition at winning LR
2. **lr compounds** — fern, kaneda, haku testing lr=7e-4 + WD/T_max/grad-clip variants
3. **Map lr=6e-4, lr=8e-4** boundary around winning lr=7e-4

### TandemFoil (No clear path except human-directed)
1. Human-directed: Kutta (kaworu), MQA (gen), HyperSRF (alphonse)
2. WD=1e-2 (violet) — promising convergence signal
