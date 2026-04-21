# SENPAI Research State

- **Date:** 2026-04-21 (Round 9 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **75.59** | #2490 (T_max=10, 3L/192d, Lion lr=3e-4, 14 ep) | Ultra-rapid cosine cycling (75 cycles/epoch) |
| AirfRANS | val_primary/surface_mse | **0.0248** | #2556 (T_max=10, 3L/192d, AdamW lr=5e-4, 41 ep) | Phase transition at cosine LR trough (epoch 40) |
| DrivAerML | val_primary/surface_rel_l2_pct | **12.70%** | #2593 (4L/256d+T_max=30, AdamW lr=5e-4, 45 ep) | Architecture depth + model hit epoch cap (still converging!) |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.0248 | 5.8x |
| DrivAerML | <3.71% | 12.70% | 3.4x |

## ALL CONFIRMED DEAD ENDS

| Direction | Reason |
|---|---|
| ANP decoder | +5.4% worse |
| EMA | Universally harmful |
| Lion on AirfRANS/DrivAerML | AdamW consistently better |
| AdamW on TandemFoil | Gap widens with training |
| Physics at slices=96 | 2 epochs max |
| Fourier+physics on AirfRANS | Metric space incompatibility |
| 6L deep model | Diverges |
| batch_size=4 TandemFoil | Destroys epoch budget |
| Reynolds-stratified sampling | All worse |
| geometry_supernodes flag | NO-OP |
| surface_anchor_points flag | NO-OP |
| T_max=150 for AirfRANS (long runs) | T_max=50 > T_max=150 |
| T_max=1000 for TandemFoil | ~10 cycles, worse |
| T_max=60/90 for TandemFoil | High-LR restart spikes |
| T_max=15 for AirfRANS Fourier+4L/256d | Too aggressive (24 cycles/ep) |
| asinh-pressure on DrivAerML | Hurts (38.87% vs 33.19% control) |
| residual-prediction on DrivAerML | NO-OP (TandemFoil only) |
| DrivAerML compound 100k+4L/256d (5 ep) | Slower per-epoch negates capacity gain |
| Timeout override (240 min) | Violates constraint |
| 4L/256d on TandemFoil | TOO SLOW (7-9 epochs vs 14 needed) |
| 3L/256d on DrivAerML | WORSE than 3L/192d (width without depth) |
| DrivAerML T_max=10 (3L/192d) | 15.47% (too fast) |
| Cp panel on AirfRANS | Wrong physics regime (inviscid for viscous) |
| Wake deficit features | Redundant with TE coord frame |
| DrivAerML 5L/256d | 13.62% — optimization instability beyond 4 layers |
| DrivAerML T_max=10+lr=3e-4 compound | 14.90% — TandemFoil hyperparams don't transfer |

## CRITICAL INSIGHT (Round 9)

**DrivAerML 4L/256d hits SENPAI_MAX_EPOCHS=50 cap, not timeout.** At ~4 min/epoch, 45 epochs = ~180 min = full time budget. Model was still converging at epoch 45. This means:
1. The epoch cap and time budget are roughly aligned for DrivAerML 4L/256d at 50k surface points
2. Reducing surface points to 25k (2x faster) could give 90+ epochs in same budget
3. T_max, LR, and WD tuning can potentially push below 12%

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 75.59)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| fern | #2546 | Coarse aux loss v3 | Sent back — v2 reached 75.80 (0.21 away!) |
| haku | #2582 | Weight decay=1e-2 full run | Sent back |
| alphonse | #2569 | Hypernetwork SRF (human-directed) | |
| frieren | #2604 | T_max=10 long run retry | |
| kaneda | NEW | Surface cross-attention (SCA, human) | Assigning |
| kaworu | NEW | Kutta TE constraint (human) | Assigning |
| mikasa | NEW | T_max=3 ultra-short | Assigning |
| levi | NEW | T_max=10 + LR warmup | Assigning |
| chrome | NEW | T_max=10 + input noise augmentation | Assigning |
| askeladd | #2621 | T_max=10 + lr=5e-4 (higher LR) | |
| tetsuo | #2610 | T_max=10 + lr=2e-4 (lower LR) | |
| kaneda | #2583 | T_max=10 + lr=1e-3 (highest LR) | Wait, kaneda now on SCA |
| nezuko | #2611 | T_max=7 interpolation | |
| sakura | #2597 | Gradient accumulation | |
| naruto | #2616 | slices=48 (faster epochs) | |
| sasuke | #2595 | 5L/256d deep model | |
| gen | #2623 | MQA audit (human-directed) | |

### AirfRANS (Baseline: 0.0248)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| kohaku | #2617 | T_max=10 replication | |
| edward | #2612 | 4L/256d + T_max=10 | Architecture + phase transition |
| gilbert | #2614 | lr=3e-4 + T_max=10 | |
| senku | #2615 | lr=1e-3 + T_max=10 | |
| thorfinn | #2613 | T_max=5 | |
| hinata | NEW | T_max=10 + WD=1e-2 | Assigning |
| roy | NEW | T_max=8 interpolation | Assigning |
| winry | NEW | T_max=15 | Assigning |
| armin | NEW | T_max=10 + phase transition monitoring | Assigning |
| hinata | #2567 | T_max sweep (25, 30, 40) | Wait, now on WD test |
| mikasa | #2578 | T_max=100 | |
| levi | #2585 | 5L/256d + T_max=50 | Wait, levi now on TandemFoil warmup |
| eren | #2601 | 4L/256d + T_max=50 | |
| itachi | #2598 | T_max=10 replication | |
| kaworu | #2587 | OOD tasks | Wait, kaworu now on Kutta |
| emma | #2618 | T_max=10 extended | |

### DrivAerML (Baseline: 12.70%)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| shinji | NEW | 4L/256d+T_max=30+lr=4e-4 | Assigning |
| violet | NEW | 4L/256d+T_max=30+lr=6e-4 | Assigning |
| norman | NEW | 4L/256d+T_max=25 | Assigning |
| ymir | NEW | 4L/256d+T_max=35 | Assigning |
| inosuke | NEW | 4L/256d+T_max=30+WD=0 | Assigning |
| giyu | NEW | 4L/256d+T_max=30+25k surface points | Assigning |
| shinobu | NEW | 4L/256d+T_max=30+grad-accum=2 | Assigning |
| zenitsu | NEW | 4L/256d+T_max=40 | Assigning |
| tanjiro | #2606 | 4L/256d+T_max=30+lr=3e-4 | |
| nami | #2599 | 4L/256d+T_max=30+lr=3e-4 | (duplicate of tanjiro?) |
| asuka | #2600 | 4L/256d+T_max=30+lr=8e-4 | |
| rei | #2609 | 4L/256d+T_max=30+lr=1e-3 | |
| mitsuha | #2607 | 4L/256d+T_max=20 | |
| taki | #2608 | 4L/256d+T_max=15 | |
| zoro | #2596 | 4L/256d+T_max=50 | |
| luffy | #2594 | 4L/256d+T_max=10 | |
| kakashi | #2602 | 4L/384d+T_max=30 | |
| ray | #2591 | Weight decay sweep | |
| historia | #2619 | Weight decay=1e-2 | |
| chihiro | #2620 | 4L/256d+T_max=30 replication | |
| shoya | #2605 | 5L/256d+T_max=30 | Likely dead end (5L confirmed worse) |
| shouko | #2622 | 100k surface points | |
| giyu | #2586 | 5L/256d+T_max=30 | Likely dead end |
| violet (old) | #2592 | 5L/256d — CLOSED | |

## Research Themes

1. **DrivAerML 4L/256d fine-tuning**: Anchor at 12.70% with T_max=30+lr=5e-4. Full parameter sweep underway: LR (3e-4, 4e-4, 6e-4, 8e-4, 1e-3), T_max (15, 20, 25, 30, 35, 40, 50), WD (0, 1e-2), surface points (25k, 50k, 100k), grad-accum, 4L/384d width. Goal: break 10%.

2. **AirfRANS phase transition exploitation**: Phase transition at T_max=10/epoch 40 is the key mechanism (0.0248). Now testing: T_max sweep (5, 8, 10, 15, 25, 30), LR sweep (3e-4, 1e-3), 4L/256d+T_max=10, WD=1e-2. Gap to external target: 5.8x.

3. **TandemFoil T_max=10 + historical mechanisms**: Core config locked (T_max=10, Lion lr=3e-4). Human-directed: SCA (kaneda), Kutta (kaworu), HyperSRF (alphonse), MQA (gen), aux loss (fern). Hyperparameter: WD=1e-2 (haku), T_max=3 (mikasa), LR sweep (askeladd, tetsuo, kaneda was kaneda), LR warmup (levi), noise aug (chrome), slices=48 (naruto), T_max=7 (nezuko).

## Next Priority Directions

1. **DrivAerML below 12%**: 45-epoch model still converging — 25k pts (2x epochs) is the key test
2. **AirfRANS 4L/256d + T_max=10** (edward): Combining architecture + phase transition mechanism
3. **Fern's aux loss v3**: 0.21 from TandemFoil baseline — lightest-weight path to improvement
4. **Human mechanisms**: SCA + Kutta now reassigned (kaneda/kaworu)
