# SENPAI Research State

- **Date:** 2026-04-21 (Round 10 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **75.59** | #2490 (T_max=10, 3L/192d, Lion lr=3e-4, 14 ep) | Ultra-rapid cosine cycling (~75 cycles/epoch) |
| AirfRANS | val_primary/surface_mse | **0.0207** | #2617 (T_max=10, 3L/192d, AdamW lr=5e-4, 41 ep) | Stochastic phase transition at cosine trough (epoch 40) |
| DrivAerML | val_primary/surface_rel_l2_pct | **12.70%** | #2593 (4L/256d+T_max=30, AdamW lr=5e-4, 45 ep) | Architecture depth + epoch cap (still converging) |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | 0.0207 | 4.8x |
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
| Cp panel on AirfRANS | Wrong physics regime |
| Wake deficit features | Redundant with TE coord frame |
| DrivAerML 5L/256d | 13.62% — optimization instability beyond 4 layers |
| DrivAerML T_max=10+lr=3e-4 compound | 14.90% — hyperparams don't transfer |
| DrivAerML T_max=15 | 13.65% |
| DrivAerML lr=3e-4 | 13.50% |
| DrivAerML lr=1e-3 | 12.91% (doesn't beat 12.70%) |
| TandemFoil lr=1e-3 | Catastrophic divergence |
| TandemFoil T_max=10 long runs (cold-start) | Infrastructure: cold filesystem wastes 4-5 epochs; only 8ep in 30min budget |
| AirfRANS T_max=10 extended run | Same epoch count, different (worse) phase transition depth (stochastic) |

## CRITICAL INSIGHTS (Round 10)

1. **AirfRANS phase transition is STOCHASTIC**: Same config gives different depths (0.0395, 0.0248, 0.0207). The transition at epoch 40 is reliable but the final val varies. Running multiple seeds could exploit this.

2. **DrivAerML T_max and LR landscape fully mapped**: T_max=30 and lr=5e-4 both confirmed optimal. Continuing exploration with novel training tricks (warmup, eta_min, grad-clip, dropout, grad-accum, 25k pts).

3. **TandemFoil cold-start problem**: Three T_max=10 long run attempts all hit the same wall. Workaround: slices=32 to reduce per-epoch time (should get 16-20 epochs even from cold start).

4. **DrivAerML hits SENPAI_MAX_EPOCHS=50 cap, not timeout**: Model still converging at cap. 25k surface points (giyu #2632) could enable 90+ epochs.

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil (Baseline: 75.59)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| fern | #2546 | Coarse aux loss v3 | Near miss at 75.80 |
| haku | #2582 | WD=1e-2 full run | Sent back |
| alphonse | #2569 | Hypernetwork SRF (human-directed) | |
| frieren | #2644 | T_max=10 + slices=32 (cold-start workaround) | NEW |
| kaneda | #2627 | SCA surface cross-attention (human-directed) | |
| kaworu | #2629 | Kutta TE constraint (human-directed) | |
| mikasa | #2631 | T_max=3 ultra-short | |
| levi | #2633 | T_max=10 + LR warmup | |
| chrome | #2635 | T_max=10 + noise augmentation | |
| askeladd | #2621 | T_max=10 + lr=5e-4 | |
| tetsuo | #2610 | T_max=10 + lr=2e-4 | |
| nezuko | #2611 | T_max=7 interpolation | |
| naruto | #2616 | slices=48 | |
| sasuke | #2595 | 5L/256d deep model | |
| sakura | #2597 | Gradient accumulation | |
| gen | #2623 | MQA audit (human-directed) | |
| kakashi | NEW | 4L/192d deeper model | Assigning |
| ray | NEW | T_max=10 + cosine eta_min | Assigning |

### AirfRANS (Baseline: 0.0207)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| kohaku | NEW | 3L/256d+T_max=10 (width expansion) | Assigning |
| emma | NEW | T_max=10 + lr=7e-4 | Assigning |
| itachi | NEW | T_max=12 | Assigning |
| eren | NEW | T_max=10 multi-seed | Assigning |
| edward | #2612 | 4L/256d+T_max=10 (sent back) | Needs more epochs |
| gilbert | #2614 | lr=3e-4+T_max=10 | |
| senku | #2615 | lr=1e-3+T_max=10 | |
| thorfinn | #2613 | T_max=5 | |
| hinata | #2637 | T_max=10+WD=1e-2 | |
| roy | #2639 | T_max=8 | |
| winry | #2636 | T_max=15 | |
| armin | #2638 | T_max=10+lr-decay | |
| kaworu | #2587 | OOD tasks | May be stale |
| mikasa | #2578 | T_max=100 | May be stale |
| levi | #2585 | 5L/256d+T_max=50 | May be stale |
| eren | #2601 | 4L/256d+T_max=50 | May be superseded |

### DrivAerML (Baseline: 12.70%)
| Student | PR | Experiment | Notes |
|---|---|---|---|
| tanjiro | NEW | 4L/256d+T_max=30+lr-warmup | Assigning |
| rei | NEW | 4L/256d+T_max=30+eta_min=1e-5 | Assigning |
| taki | NEW | 4L/256d+T_max=30+600-batches | Assigning |
| zoro | NEW | 4L/320d+T_max=30 | Assigning |
| luffy | NEW | 4L/256d+T_max=30+dropout | Assigning |
| asuka | NEW | 4L/256d+T_max=30+eval400 | Assigning |
| nami | NEW | 4L/256d+T_max=30+grad-clip | Assigning |
| shinji | #2624 | 4L/256d+lr=4e-4 | |
| violet | #2625 | 4L/256d+lr=6e-4 | |
| norman | #2626 | 4L/256d+T_max=25 | |
| ymir | #2628 | 4L/256d+T_max=35 | |
| inosuke | #2630 | 4L/256d+WD=0 | |
| giyu | #2632 | 4L/256d+25k surface pts | KEY: 2x more epochs |
| shinobu | #2634 | 4L/256d+grad-accum=2 | |
| zenitsu | #2640 | 4L/256d+T_max=40 | |
| mitsuha | #2607 | 4L/256d+T_max=20 | |
| asuka (old) | #2600 | 4L/256d+lr=8e-4 | May complete soon |
| kakashi | #2602 | 4L/384d+T_max=30 | Wide model |
| historia | #2619 | 4L/256d+WD=1e-2 | |
| chihiro | #2620 | 4L/256d replication | |
| shouko | #2622 | 4L/256d+100k surface pts | |
| shoya | #2605 | 5L/256d (likely dead end) | |
| ray (old) | #2591 | WD sweep (3L/192d) | Likely stale |

## Next Priority Directions

1. **AirfRANS multi-seed exploitation**: Phase transition is stochastic — running 5 seeds should reliably beat 0.0207
2. **DrivAerML 25k surface points** (giyu): 2x faster epochs = 90+ in budget, model still converging
3. **AirfRANS 3L/256d+T_max=10** (kohaku): More width without depth overhead
4. **TandemFoil cold-start workaround** (frieren slices=32): 16-20 epochs expected
5. **Human-directed mechanisms**: SCA (kaneda), Kutta (kaworu), HyperSRF (alphonse), MQA (gen), aux loss (fern)
