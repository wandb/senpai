# SENPAI Research State

- **Date:** 2026-04-21 (Round 12 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **75.59** | #2490 (T_max=10, 3L/192d, Lion lr=3e-4, 14 ep) | Ultra-rapid cosine cycling (~75 cycles/epoch) |
| AirfRANS | val_primary/surface_mse | **0.01841** | #2646 (T_max=10, 3L/192d, AdamW lr=7e-4, 41 ep) | Phase transition at epoch 35 (earlier with higher LR) |
| DrivAerML | val_primary/surface_rel_l2_pct | **11.97%** | #2645 (4L/256d+T_max=30, AdamW lr=5e-4, 34 ep, 600 batches/ep) | More data per epoch (600 vs 394 batches) |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.01841 | 4.3x |
| DrivAerML | <3.71% | 11.97% | 3.2x |

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
| geometry_supernodes/surface_anchor flags | NO-OP |
| T_max=150 for AirfRANS | T_max=50 > T_max=150 |
| T_max=1000/60/90 for TandemFoil | High-LR spikes or too few cycles |
| T_max=15 for AirfRANS 4L/256d | Too aggressive |
| asinh-pressure on DrivAerML | Hurts |
| residual-prediction on DrivAerML | NO-OP (TandemFoil only) |
| DrivAerML compound 100k+4L/256d (5ep) | Slower negates capacity gain |
| 4L/256d on TandemFoil | TOO SLOW (7-9 ep vs 14 needed) |
| 3L/256d on DrivAerML | WORSE than 3L/192d |
| DrivAerML T_max=10 (3L/192d) | 15.47% (too fast) |
| DrivAerML T_max=10 (4L/256d long run) | 17.08% (confirmed dead end 2nd time) |
| Cp panel on AirfRANS | Wrong physics regime |
| Wake deficit features | Redundant with TE coord frame |
| DrivAerML 5L/256d | 13.62%, 13.24% — optimization instability beyond 4 layers |
| DrivAerML T_max=10+lr=3e-4 compound | 14.90% — hyperparams don't transfer |
| DrivAerML T_max=15 | 13.65% |
| DrivAerML T_max=25 | ~13.1% |
| DrivAerML lr=3e-4 | 13.50% |
| DrivAerML lr=4e-4 | 13.28% |
| DrivAerML lr=6e-4 | 13.42% |
| DrivAerML lr=1e-3 | 12.91% (doesn't beat 12.70%) |
| TandemFoil lr=1e-3 | Catastrophic divergence |
| TandemFoil T_max=10 long runs (cold-start) | Infrastructure: cold filesystem wastes 4-5 epochs |
| TandemFoil slices=32 | 97.23 — no speedup (data loading dominates) |
| AirfRANS 3L/256d+T_max=10 | 0.0357 — too slow for phase transition |
| AirfRANS 4L/256d+T_max=10 | 0.0881 — too slow per epoch (25 epochs in 30 min) |
| AirfRANS T_max=10 extended run | Same epoch count, stochastic — shallow transition |

## CRITICAL INSIGHTS (Round 12)

1. **DrivAerML: More data per epoch is a critical lever**: 600 batches/epoch (vs 394) improved from 12.70% to 11.97%. Model sees 53% more car configs/epoch. Still converging at cutoff (34 ep). Next: test 800 and 1000 batches to find saturation point.

2. **AirfRANS: Higher LR = earlier + deeper phase transition**: lr=7e-4 triggers transition at epoch 35 (vs epoch 38-40 for lr=3e-4/5e-4) and finds deeper basin (0.01841 vs 0.0197). LR sweet spot is 3e-4 to 7e-4. High volatility at cosine peaks but robust troughs.

3. **AirfRANS phase transition is STOCHASTIC**: Same config gives different depths. Running multiple seeds at lr=7e-4 is the highest-value strategy.

4. **TandemFoil cold-start problem persists**: WD=1e-2 converges faster at same epoch count (95.59 vs 103.87 at 7 ep) but can't get enough epochs for fair comparison. slices=32 workaround failed.

5. **DrivAerML compound experiments needed**: Many WIP experiments (eta_min, warmup, grad-clip, dropout) used old 394-batch config. Must be updated to compound with 600 batches.

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 75.59)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| violet | NEW | WD=1e-2 | Assigning |
| haku | #2582 | WD sweep (awaiting response) | Cold-start blocks fair test |
| alphonse | #2569 | Hypernetwork SRF (human-directed) | |
| kaneda | #2627 | SCA surface cross-attention (human-directed) | |
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
| tetsuo | #2610 | lr=2e-4 | May be stale |
| naruto | #2616 | slices=48 | May be stale |

### AirfRANS (Baseline: 0.01841)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| emma | NEW | lr=6e-4+T_max=10 | Assigning |
| kohaku | NEW | lr=7e-4 multi-seed (5 seeds) | Assigning |
| edward | NEW | lr=8e-4+T_max=10 | Assigning |
| gilbert | #2655 | lr=3e-4 multi-seed | |
| eren | #2649 | T_max=10 multi-seed | |
| historia | #2668 | lr=3e-4+WD=1e-2 | |
| fern | #2657 | lr=2e-4 | |
| nezuko | #2658 | lr=1e-4 | |
| senku | #2664 | 3L/256d+lr=3e-4 | Likely dead end (3L/256d confirmed slow) |
| shinji | #2663 | lr=3e-4+dropout | |
| thorfinn | #2666 | T_max=5 | |
| hinata | #2637 | T_max=10+WD=1e-2 | |
| armin | #2638 | LR decay | |
| winry | #2636 | T_max=15 | |
| roy | #2639 | T_max=8 | |
| itachi | #2647 | T_max=12 | |

### DrivAerML (Baseline: 11.97%)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| frieren | NEW | 800 batches/epoch | Assigning |
| taki | NEW | 1000 batches/epoch | Assigning |
| norman | NEW | 600 batches + WD=1e-2 | Assigning |
| tanjiro | #2641 | Warmup (sent back for 600-batch compound) | |
| rei | #2643 | eta_min=1e-5 (sent back for 600-batch compound) | |
| zenitsu | #2640 | T_max=40 | OLD 394 batches |
| luffy | #2650 | Dropout=0.1 | OLD 394 batches |
| nami | #2654 | Grad-clip=1.0 | OLD 394 batches |
| askeladd | #2661 | Multi-seed replication | OLD 394 batches |
| mitsuha | #2660 | 5-epoch cosine warmup | OLD 394 batches |
| shouko | #2659 | lr=5.5e-4 | OLD 394 batches |
| chihiro | #2656 | LR decay | OLD 394 batches |
| shoya | #2662 | lr=3e-4+warmup | OLD 394 batches |
| zoro | #2648 | 4L/320d | OLD 394 batches |
| shinobu | #2634 | Grad-accum=2 | OLD 394 batches |
| giyu | #2632 | 25k surface pts | OLD config |
| inosuke | #2630 | WD=0 | OLD 394 batches |
| ymir | #2628 | T_max=35 | OLD 394 batches |
| asuka | #2652 | eval-batches=400 | OLD 394 batches |
| chihiro | #2620 | 4L/256d replication | OLD 394 batches |

## Next Priority Directions

1. **DrivAerML batches saturation**: Find where more batches/epoch stops helping (800, 1000 tests in flight)
2. **DrivAerML compound experiments**: All WIP DrivAerML experiments used old 394-batch config — when they complete, promising ones should be re-run with 600 batches
3. **AirfRANS lr=7e-4 multi-seed**: Phase transition is stochastic — 5 seeds at winning LR should reliably find deeper basin
4. **AirfRANS LR boundary mapping**: lr=6e-4, lr=8e-4 testing to refine the sweet spot
5. **Human-directed mechanisms**: SCA (kaneda), Kutta (kaworu), HyperSRF (alphonse), MQA (gen) — all still in flight
6. **TandemFoil WD=1e-2**: Promising convergence signal despite cold-start limitation

## Research Themes

### Theme 1: Data Efficiency (DrivAerML)
The 600-batch breakthrough suggests DrivAerML is data-starved per epoch. Testing higher batch counts and compound optimizations.

### Theme 2: Phase Transition Exploitation (AirfRANS)
The stochastic phase transition is the key mechanism. Current strategy: (a) find optimal LR for deepest transitions, (b) run multiple seeds to exploit stochasticity. lr=7e-4 is the current best LR.

### Theme 3: Architectural Mechanisms (TandemFoil)
Human-directed experiments (SCA, Kutta, MQA, HyperSRF) represent the best chance for TandemFoil improvement given the cold-start problem limiting hyperparameter experiments.

### Theme 4: Regularization
WD=1e-2 shows promise on TandemFoil (faster convergence with Lion). Testing on DrivAerML and AirfRANS.
