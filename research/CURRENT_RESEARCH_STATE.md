# SENPAI Research State

- **Date:** 2026-04-21 (Round 4 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **75.59** | #2490 (T_max=10, Fourier+physics+no-EMA, slices=64, Lion lr=3e-4, 14 ep) | Ultra-rapid cosine cycling (75 cycles/epoch) |
| AirfRANS | val_primary/surface_mse | **0.0696** | #2540 (Fourier+3L/192d+T_max=50, AdamW lr=5e-4, 23 ep) | Phase transition at cosine LR trough |
| DrivAerML | val_primary/surface_rel_l2_pct | **33.65%** | #2543 (Fourier+3L/192d+T_max=30, AdamW lr=5e-4, 6 ep) | Training time dominance |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.0696 | 16x |
| DrivAerML | <3.71% | 33.65% | 9x |

## CRITICAL DISCOVERIES (Round 4)

1. **PHASE TRANSITION on AirfRANS** — At epoch 23, cosine LR near T_max=50 trough causes val to jump from 0.19-0.21 to 0.0696 in a single epoch (-65.4%). The mechanism: very low LR allows optimizer to settle into a sharp narrow minimum. HIGHEST PRIORITY FOLLOW-UP: longer training with this config.

2. **T_max=10 is the new TandemFoil optimal** — 75 cosine cycles per epoch creates ultra-rapid LR averaging. T_max=10 > T_max=20 > T_max=15 > T_max=30. Testing T_max=5 and T_max=3 next.

3. **Training time dominates DrivAerML** — 2ep→51.35%, 6ep→33.65%, luffy WIP at 28.80% (11ep). More training = better, no diminishing returns yet.

4. **Human guidance** (Issue #2545): 4 historical mechanisms being ported: SCA, MQA, Kutta constraint, Hypernetwork SRF. Coarse aux loss already testing (fern #2546).

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

## WIP PRs — ALL STUDENTS

### TandemFoil (In-flight)
| Student | PR | Experiment | Status |
|---|---|---|---|
| fern | #2546 | Coarse spatial-pooling aux loss (HUMAN PRIORITY) | wip |
| haku | #2549 | Wake deficit features | wip |
| edward | #2534 | 4L/256d capacity (sent back, investigate timeout) | wip |
| kaneda | #2536 | T_max=120 extended + T_max=80 | wip |
| sakura | #2505 | Fourier only ablation | wip |
| kakashi | #2524 | slices=48/80 | wip |
| sasuke | #2504 | T_max=150/50 | wip |
| frieren | NEW | T_max=10 long training | assigning |
| askeladd | NEW | T_max=5/3 ultra-short | assigning |
| tetsuo | NEW | T_max=10 + lr sweep (2e-4, 1e-4) | assigning |
| kohaku | NEW | T_max=10 + 4L/256d | assigning |
| nezuko | NEW | T_max=10 no-physics ablation | assigning |
| naruto | NEW | Surface cross-attention (HUMAN #2) | assigning |
| thorfinn | NEW | MQA audit (HUMAN #4) | assigning |
| historia | NEW | Hard Kutta TE constraint (HUMAN #5) | assigning |
| alphonse | NEW | Hypernetwork SRF (HUMAN #6) | assigning |

### AirfRANS (In-flight)
| Student | PR | Experiment | Status |
|---|---|---|---|
| gilbert | #2539 | T_max=25 long run (sent back) | wip |
| norman | #2548 | Cp panel physics feature | wip |
| emma | NEW | Fourier+3L/192d+T_max=50 FULL long run | assigning |
| itachi | NEW | Fourier+4L/256d+T_max=50 long run | assigning |
| hinata | NEW | T_max sweep (25, 30, 40) | assigning |
| roy | NEW | lr=3e-4 + T_max=50 | assigning |
| winry | NEW | lr=1e-4 ultra-low LR | assigning |

### DrivAerML (In-flight)
| Student | PR | Experiment | Status |
|---|---|---|---|
| violet | #2550 | 4L/256d + T_max=30/50 long run | wip |
| chihiro | #2537 | T_max=30 vs T_max=50 long run (sent back) | wip |
| shinji | #2541 | 3L/256d long run (sent back) | wip |
| luffy | #2519 | T_max=50 — **28.80% at ep11!** | wip |
| shoya | NEW | Standard config full 180-min | assigning |
| shouko | NEW | T_max=10 (transfer from TandemFoil) | assigning |
| mitsuha | NEW | lr sweep (3e-4, 8e-4) | assigning |
| taki | NEW | 100k surface points | assigning |
| tanjiro | NEW | T_max=50 full long run | assigning |
| rei | NEW | Phase transition test (T_max=50 long) | assigning |

### Older WIP (may need cleanup)
| Student | PR | Experiment |
|---|---|---|
| armin | #2509 | AirfRANS slices=64/48 |
| asuka | #2521 | DrivAerML T_max=10 |
| chrome | #2515 | DrivAerML no-Fourier long run |
| eren | #2506 | AirfRANS lr=3e-4/8e-4 |
| gen | #2516 | DrivAerML 200k surface points |
| giyu | #2513 | DrivAerML slices=64/48 |
| inosuke | #2512 | DrivAerML 4L/256d |
| kaworu | #2517 | DrivAerML 5L/256d |
| levi | #2510 | AirfRANS Lion optimizer |
| mikasa | #2508 | AirfRANS no-Fourier ablation |
| nami | #2523 | DrivAerML lr=8e-4 |
| ray | #2518 | DrivAerML Lion optimizer |
| shinobu | #2514 | DrivAerML 100k points |
| ymir | #2507 | DrivAerML T_max sweep |
| zenitsu | #2511 | DrivAerML LR sweep |
| zoro | #2520 | DrivAerML T_max=150 |

## Research Themes

1. **AirfRANS phase transition exploitation**: The phase transition at cosine LR trough is the most important discovery. Testing: longer training for repeated transitions (emma), LR optimization around transition (roy, winry), 4L/256d for transition (itachi), T_max sweep (hinata).

2. **TandemFoil T_max=10 refinement + historical mechanisms**: T_max=10 is the new default. Testing: longer training (frieren), even shorter cycles T_max=5/3 (askeladd), lr sweep (tetsuo), 4L/256d capacity (kohaku). In parallel: SCA (naruto), MQA (thorfinn), Kutta (historia), HyperSRF (alphonse), coarse aux loss (fern).

3. **DrivAerML training time exploitation**: The dominant variable is more epochs. Testing: full 180-min runs (shoya, tanjiro), T_max=10 transfer (shouko), lr sweep (mitsuha), phase transition test (rei).

4. **Cross-dataset cosine schedule insight**: Ultra-short T_max works for TandemFoil (T_max=10). Phase transition at T_max=50 trough works for AirfRANS. Testing both patterns on DrivAerML.
