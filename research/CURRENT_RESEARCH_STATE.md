# SENPAI Research State

- **Date:** 2026-04-21 (Round 8 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **75.59** | #2490 (T_max=10, 3L/192d, Lion lr=3e-4, 14 ep) | Ultra-rapid cosine cycling (75 cycles/epoch) |
| AirfRANS | val_primary/surface_mse | **0.0248** | #2556 (T_max=10, 3L/192d, AdamW lr=5e-4, 41 ep) | Phase transition at cosine LR trough (epoch 40) |
| DrivAerML | val_primary/surface_rel_l2_pct | **12.96%** | #2550 (4L/256d+T_max=30, AdamW lr=5e-4, 43 ep) | Architecture depth + long training |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.0248 | 5.8x |
| DrivAerML | <3.71% | 12.96% | 3.5x |

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
| DrivAerML T_max=10 (3L/192d) | 15.47% (too fast for 3L) |
| Cp panel on AirfRANS | Wrong physics regime (inviscid for viscous) |
| Wake deficit features | Redundant with TE coord frame |

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 75.59)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| fern | #2546 | Coarse aux loss v3 (lighter weight) | Sent back — v2 reached 75.80 (0.21 away!) |
| frieren | #2604 | T_max=10 long run retry | Ensure 14+ epochs |
| haku | #2582 | Weight decay sweep (WD=1e-2 full run) | Sent back — WD=1e-2 best at 7ep |
| askeladd | #2555 | T_max=5 ultra-short cycles | |
| nezuko | #2611 | T_max=7 interpolation | |
| tetsuo | #2610 | lr=2e-4 (lower LR) | |
| kaneda | #2583 | lr=1e-3 (higher LR) | |
| sakura | #2597 | Gradient accumulation (effective batch) | |
| naruto | #2616 | slices=48 (faster epochs) | |
| sasuke | #2595 | 5L/256d deep model | |
| nezuko | #2564 | Fourier-only (no physics) ablation | |
| alphonse | #2569 | Hypernetwork SRF (historical port) | |
| naruto | #2559 | Surface cross-attention (historical port) | |
| thorfinn | #2560 | MQA audit (historical port) | |
| historia | #2562 | Hard Kutta TE constraint (historical port) | |

### AirfRANS (Baseline: 0.0248)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| kohaku | #2617 | T_max=10 exact replication | Confirm reproducibility |
| edward | #2612 | 4L/256d + T_max=10 | Architecture + phase transition |
| gilbert | #2614 | lr=3e-4 + T_max=10 | Lower LR phase transition |
| senku | #2615 | lr=1e-3 + T_max=10 | Higher LR phase transition |
| thorfinn | #2613 | T_max=5 ultra-short | Even faster transitions? |
| hinata | #2567 | T_max sweep (25, 30, 40) | |
| mikasa | #2578 | T_max=100 longer cycle | |
| armin | #2580 | lr=2e-4 + T_max=50 | |
| roy | #2568 | lr=3e-4 + T_max=50 | |
| winry | #2571 | lr=1e-4 ultra-low | |
| levi | #2585 | 5L/256d + T_max=50 | |
| eren | #2601 | 4L/256d + T_max=50 | |
| itachi | #2598 | T_max=10 replication | |
| kaworu | #2587 | OOD tasks with phase-transition config | |

### DrivAerML (Baseline: 12.96%)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| tanjiro | #2606 | 4L/256d + lr=3e-4 (lower LR) | |
| nami | #2599 | 4L/256d + lr=3e-4 | |
| asuka | #2600 | 4L/256d + lr=8e-4 (higher LR) | |
| rei | #2609 | 4L/256d + lr=1e-3 (highest LR) | |
| mitsuha | #2607 | 4L/256d + T_max=20 | |
| taki | #2608 | 4L/256d + T_max=15 | |
| zoro | #2596 | 4L/256d + T_max=50 | |
| luffy | #2594 | 4L/256d + T_max=10 | |
| shinji | #2593 | 4L/256d + T_max=30 replication | |
| norman | #2603 | 4L/256d + T_max=10 + lr=3e-4 | |
| shoya | #2605 | 5L/256d + T_max=30 | |
| violet | #2592 | 5L/256d + T_max=30 (Fourier variant) | |
| giyu | #2586 | 5L/256d + T_max=30 capacity | |
| kakashi | #2602 | 4L/384d extra-wide | |
| ray | #2591 | Weight decay sweep | |
| zenitsu | #2581 | lr=1e-3 higher LR | |
| chrome | #2589 | lr=3e-4 LR sweep | |
| shinobu | #2588 | T_max=5 ultra-short | |
| inosuke | #2584 | T_max=50 phase transition test | |
| ymir | #2579 | T_max=10 long run | |
| gen | #2590 | 3L/256d width expansion | Known dead end? |
| taki | #2566 | 100k surface points | |
| shoya | #2554 | 3L/192d T_max=30 long run | |
| chihiro | #2537 | T_max=50 long run | |

## Research Themes

1. **TandemFoil T_max=10 refinement**: Core config is locked (T_max=10, Lion lr=3e-4, 3L/192d, 14 ep). Exploring: weight decay (WD=1e-2 promising), even shorter cycles (T_max=5, 7), LR sweep (1e-4 to 1e-3), slices=48 for more epochs, gradient accumulation, coarse aux loss (near miss at 75.80). Historical mechanisms from human guidance: SCA, MQA, Kutta constraint, HyperSRF.

2. **AirfRANS phase transition exploitation**: Phase transition at cosine LR trough is the critical mechanism (0.19→0.0696→0.0248). T_max=10 produced deeper transition than T_max=50. Now testing: replication (kohaku, itachi), LR sweep around transition (gilbert lr=3e-4, senku lr=1e-3), 4L/256d + T_max=10 (edward), even shorter T_max=5 (thorfinn), broader T_max sweep (hinata).

3. **DrivAerML architecture optimization**: 4L/256d+T_max=30 at 12.96% is the anchor. Major parallel exploration: LR sweep (3e-4, 8e-4, 1e-3), T_max fine-tuning (15, 20, 50), 5L/256d depth test, 4L/384d width test, weight decay sweep, replication. Phase transition test with T_max=50.

4. **Cross-dataset insights**: Ultra-short T_max works for both TandemFoil (T_max=10) and AirfRANS (T_max=10 phase transition). DrivAerML optimal at T_max=30 — may benefit from slightly shorter (T_max=15-20 being tested).

## Next Priority Directions

1. **Fern's aux loss** is 0.21 from TandemFoil baseline — lightest-weight variant could break through
2. **AirfRANS 4L/256d + T_max=10** (edward #2612) combines architecture scaling with phase transition — high-impact test
3. **DrivAerML 5L depth test** (shoya, violet, giyu) — if 4L beats 3L by 61%, does 5L continue?
4. **LR optimization around DrivAerML winner** — 5e-4 may not be optimal at 43 epochs
5. **Historical mechanisms** (SCA, MQA, Kutta, HyperSRF) — completely orthogonal to current approach
