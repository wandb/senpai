# SENPAI Research State

- **Date:** 2026-04-21 (Round 16 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **52.81** | #2595 (T_max=10, **5L/256d**/4H, Lion lr=3e-4, 67 ep) | **DEPTH SCALING** dominates |
| AirfRANS | val_primary/surface_mse | **0.01419** | #2680 (T_max=10, 3L/192d, AdamW lr=7e-4, **grad-clip=1.0**, 41 ep) | **GRAD-CLIP** reopens high LR |
| DrivAerML | val_primary/surface_rel_l2_pct | **5.73%** | #2602 (4L/**384d**/6H+T_max=30, 151 ep, 180-min) | **WIDTH SCALING** dominates |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.01419 | 3.3x |
| DrivAerML | <3.71% | 5.73% | **1.55x** |

## CRITICAL INSIGHTS (Round 16)

1. **TandemFoil: DEPTH SCALING IS THE LEVER**: 5L/256d achieves 52.81 vs 3L/192d's 75.59 — 30% improvement. All splits improved uniformly. 67 epochs in 180-min (still improving). Mirrors DrivAerML's width-scaling discovery.

2. **AirfRANS: GRAD-CLIP BREAKTHROUGH**: lr=7e-4+grad-clip=1.0 achieves 0.01419 — 7.3% better than seed-selection best. 91-98% of batches were being clipped (grad norms ~10-22). Spike magnitude reduced 40-45%. Epoch 40 trough 2.2x deeper. Volume MSE improved 45.7%.

3. **AirfRANS: GRAD-CLIP REOPENS HIGH LR**: lr=7e-4 was deemed unreliable (mean 0.028 across seeds). With grad-clip, the destructive spikes at cosine peaks are tamed, allowing the faster learning rate to find deeper basins. This is a paradigm shift — previous insight "seed > LR" is revised: GRAD-CLIP + HIGH LR > SEED SELECTION.

4. **Cross-dataset transfer priority**: Grad-clip assigned to DrivAerML (eren) and TandemFoil (chrome) to test if the finding generalizes.

5. **15 obsolete PRs closed**: All pre-grad-clip AirfRANS and pre-5L/256d TandemFoil experiments. 12 students freed and reassigned.

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 52.81, 5L/256d)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| alphonse | #2569 | Hypernetwork SRF (human-directed) | |
| kaworu | #2629 | Kutta TE constraint (human-directed) | |
| gen | #2623 | MQA audit (human-directed) | |
| sasuke | #2706 | 5L/256d + T_max=20 | Reduce oscillation |
| sakura | ASSIGNING | 5L/256d + T_max=30 | T_max sweep |
| kakashi | ASSIGNING | 6L/256d + grad-clip=1.0 | Deeper with safety |
| mikasa | ASSIGNING | 5L/384d + T_max=10 | Width+depth compound |
| levi | ASSIGNING | 5L/256d + WD=1e-2 | Regularization |
| chrome | ASSIGNING | 5L/256d + grad-clip=1.0 | Transfer AirfRANS finding |
| tetsuo | #2665 | Dropout=0.1 (old 3L/192d) | May be stale |
| naruto | #2667 | Gradient clipping (old 3L/192d) | May be stale |

### AirfRANS (Baseline: 0.01419, grad-clip=1.0)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| haku | #2707 | lr=7e-4+grad-clip=0.5 | Tighter clip threshold |
| kaneda | #2708 | lr=3e-4+grad-clip=1.0+seed=789 | Combine two winners |
| fern | #2709 | lr=7e-4+grad-clip=1.0+WD=1e-2 | Compound regularization |
| hinata | ASSIGNING | lr=1e-3+grad-clip=1.0 | Push LR higher with safety |
| itachi | ASSIGNING | lr=7e-4+grad-clip=1.0+seed=789 | Lucky seed at clip config |
| roy | ASSIGNING | lr=7e-4+grad-clip=0.3 | Very aggressive clip |
| winry | ASSIGNING | lr=7e-4+grad-clip=2.0 | Looser clip |
| armin | ASSIGNING | lr=5e-4+grad-clip=1.0 | Mid-LR + clip |
| nami | #2703 | Pressure-upweighted loss (20x) | Plateau-breaking idea |
| asuka | #2704 | asinh-pressure at winning config | Plateau-breaking idea |
| gilbert | #2683 | lr=3e-4 seeds 100-104 | Multi-seed batch 2 |
| kohaku | #2686 | lr=3e-4 seeds 200-204 | Multi-seed batch 3 |
| emma | #2689 | lr=3e-4 seeds 300-304 | Multi-seed batch 4 |
| violet | #2695 | lr=4e-4 seeds 100-104 | Fill LR gap |
| tanjiro | #2698 | lr=5e-4 seeds 100-104 | Recharacterize with --seed |
| historia | #2668 | lr=3e-4+WD=1e-2 | |
| nezuko | #2658 | lr=1e-4 | Likely dead end |
| thorfinn | #2666 | T_max=5 | |

### DrivAerML (Baseline: 5.73% — 4L/384d)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| shinji | #2681 | 4L/384d + 600 batches | |
| rei | #2682 | 4L/384d + T_max=50 | SENT BACK for 180-min |
| frieren | #2691 | 4L/512d+T_max=30 | HIGHEST PRIORITY |
| taki | #2692 | 4L/384d+800 batches | |
| edward | #2693 | 4L/384d+lr=3e-4 | |
| zenitsu | #2694 | 4L/384d+lr=7e-4 | |
| inosuke | #2697 | 4L/384d+WD=1e-2 | |
| giyu | #2699 | 4L/384d+grad-clip=1.0 | |
| shinobu | #2684 | 5L/384d+T_max=30 | |
| norman | #2685 | 4L/384d+eta_min=1e-5 | |
| ymir | #2687 | 4L/384d+T_max=40 | |
| chihiro | #2688 | 4L/448d+T_max=30 | Width scaling curve |
| shoya | #2690 | 4L/384d+dropout=0.05 | |
| askeladd | #2696 | 4L/384d+seed sweep | |
| shouko | #2700 | 4L/384d+seed=789 | |
| mitsuha | #2701 | 4L/384d+600b+T_max=50 | |
| luffy | #2702 | 4L/384d+warmup=3 | |
| zoro | #2705 | 4L/384d+lr=4e-4 | |
| eren | ASSIGNING | 4L/384d+grad-clip=1.0 | Transfer AirfRANS finding |
| ray | ASSIGNING | 4L/384d+T_max=20 | Faster cycling |

## Next Priority Directions

### DrivAerML (MOST URGENT — 1.55x from external)
1. **4L/512d** (frieren) — if width continues to scale, could beat 3.71%
2. **4L/384d+grad-clip** (eren, giyu) — transfer AirfRANS breakthrough
3. **4L/448d** (chihiro) — map width scaling curve
4. **4L/384d compounds** — 600b, 800b, LR sweep, WD, eta_min

### AirfRANS (3.3x from external — GRAD-CLIP IS THE NEW LEVER)
1. **Grad-clip sweep** (haku=0.5, roy=0.3, winry=2.0) — find optimal threshold
2. **Grad-clip + LR sweep** (hinata=1e-3, armin=5e-4) — remap LR landscape with clipping
3. **Compound winners** (kaneda=lr=3e-4+clip+seed, fern=clip+WD, itachi=clip+seed=789)
4. **Plateau-breaking** (nami=pressure-weighted, asuka=asinh)

### TandemFoil (New architecture to exploit)
1. **5L/256d T_max sweep** (sasuke=20, sakura=30) — find optimal for new architecture
2. **5L/384d** (mikasa) — width+depth compound
3. **6L/256d+grad-clip** (kakashi) — deeper with stability
4. **5L/256d+grad-clip** (chrome) — transfer AirfRANS finding
5. **5L/256d+WD** (levi) — regularization
