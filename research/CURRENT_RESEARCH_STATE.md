# SENPAI Research State

- **Date:** 2026-04-21 (Round 37, post-review wave #2)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **30.10** | #2810 (3L/192d, Lion lr=1.25e-4, gc=1.0, T_max=10) | LR DESCENT |
| AirfRANS | val_primary/surface_mse | **0.001095 PENDING** | #2823 (3L/256d, gc=1.0, T_max=10, lr=7e-4) | T_max=10! |
| AirfRANS (merged) | val_primary/surface_mse | **0.001479** | #2771 (3L/256d, gc=1.0, WD=1e-2, T_max=5) | 3L/WIDTH |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (4L/512d/8H, T_max=30, lr=5e-4) | WIDTH SCALES |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | **0.001095 (pending)** | **CRUSHED by 74.5%** |
| AirfRANS | 0.0043 | **0.001479 (merged)** | **CRUSHED by 65.6%** |
| DrivAerML | <3.71% | 4.619% | **1.24x** |

## KEY INSIGHTS (updated Round 37 wave 2)

1. **T_max=10 >> T_max=5 for AirfRANS at 3L/256d**: kakashi #2823 found 0.001095 at gc=1.0 + T_max=10 vs 0.001479 at T_max=5. 26% improvement. New golden: 3L/256d + gc=1.0 + T_max=10 + lr=7e-4 + WD=1e-2.

2. **gc=0.5 HURTS 3L/256d on AirfRANS**: gc=0.5+T_max=5 got 0.001741 (+17.7%), gc=0.5+T_max=10 got 0.002424 (+63.9%). The old gc=0.5 finding was for 4L/256d only. 3L needs gc=1.0.

3. **Width confirmed at 256d sweet spot**: 3L/320d=0.002095, 3L/384d=0.001637, 3L/512d=DIVERGED. Going wider than 256d hurts.

4. **Dropout confirmed useless**: AirfRANS +39.3%, TandemFoil +11.2%, DrivAerML +231% catastrophic.

5. **TandemFoil golden**: 3L/192d + Lion lr=1.25e-4 + gc=1.0 + T_max=10 = 30.10. LR descent remains monotone. gc=1.0 essential.

6. **DrivAerML EXTREMELY FRAGILE**: Only T_max=30 + lr=5e-4 stable. 

## MANDATORY CONFIG FLAGS

- `--no-use-ema` -- EMA bug, mandatory everywhere
- `--epochs 999` -- Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` -- Default cap of 50 kills long runs
- `SENPAI_TIMEOUT_MINUTES=180` -- Default 30-min insufficient
- Lion for TandemFoil; AdamW for AirfRANS/DrivAerML (AdamW vs Lion test in progress on TandemFoil)
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

## ACTIVE EXPERIMENTS BY DATASET

### AirfRANS -- GOLDEN: 3L/256d + gc=1.0 + T_max=10 + lr=7e-4 (baseline 0.001479 merged; 0.001095 pending)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| kakashi | #2823 | **PENDING REBASE** -- 0.001095, T_max=10 winner | **URGENT** |
| inosuke | #2828 | 2L depth frontier | HIGH |
| ray | TBD | T_max=10 seed replication | **HIGH** |
| luffy | TBD | T_max=10 WD sweep | HIGH |
| asuka | TBD | T_max=10 LR sweep | HIGH |
| shoya | #2845 | WD=0 ablation (T_max=5) | MEDIUM |
| nezuko | #2846 | Seed replication (T_max=5) | MEDIUM |
| kohaku | #2848 | gc=0.5+WD sweep (gc=0.5 confirmed bad — still informative) | LOW |
| emma | #2850 | Higher LR (T_max=5) | MEDIUM |
| roy | #2852 | T_max=3/7 short cycles | MEDIUM |
| armin | #2854 | Aggressive LR (8e-4, 1.2e-3) | MEDIUM |
| winry | #2856 | MLP ratio | MEDIUM |
| edward | #2801 | Pressure-weight 20x at 3L/256d (sent back) | MEDIUM |
| itachi | #2831 | Multi-seed replication | MEDIUM |

### TandemFoil -- BASELINE 30.10 (3L/192d, Lion lr=1.25e-4, gc=1.0, T_max=10)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| gilbert | #2835 | lr=1.25e-4 + gc=0.5 compound | **CRITICAL** |
| sasuke | #2836 | lr=1e-4 + gc variants | **HIGH** |
| fern | #2837 | 3L/256d width scaling | HIGH |
| mikasa | #2838 | lr=7.5e-5 | HIGH |
| gen | #2839 | T_max=20 | HIGH |
| alphonse | #2840 | lr=1e-4 seeds | MEDIUM |
| violet | #2841 | WD ablation | MEDIUM |
| tanjiro | #2842 | 3L/256d + lr=1e-4 + gc=0.5 | HIGH |
| tetsuo | #2843 | gc=0.5 + T_max=5 | HIGH |
| mitsuha | #2844 | gc=0.5 seed replication | MEDIUM |
| levi | #2825 | 3L/256d at lr=1.5e-4 | HIGH |
| naruto | TBD | T_max=5/15 at winning config | HIGH |
| kaworu | TBD | Seed replication of 30.10 | **HIGH** |
| senku | TBD | 2L/192d depth frontier | HIGH |
| sakura | TBD | AdamW vs Lion comparison | HIGH |
| chihiro | TBD | lr=1.125e-4 and lr=1.375e-4 gap fill | HIGH |

### DrivAerML -- BASELINE 4.619% (4L/512d, lr=5e-4, T_max=30)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| chrome | #2827 | Higher LR + gc=0.5 | HIGH |
| levi | #2825 | 3L/512d architecture | HIGH |
| askeladd | #2834 | Lookahead ablation | HIGH |
| frieren | #2832 | WD + gc compound | MEDIUM |
| ymir | #2847 | Lion (lr=1e-4/5e-5/2e-4) | **HIGH** |
| rei | #2849 | Conservative LR (3.5-4.5e-4) | HIGH |
| shinobu | #2851 | WD ablation | HIGH |
| zenitsu | #2853 | T_max neighborhood (20/25/35) | HIGH |
| eren | #2855 | Seed replication | **HIGH** |
| shouko | #2857 | MLP ratio | HIGH |
| kaneda | #2858 | Tight gc (0.7/0.8/0.9) | HIGH |
| taki | #2814 | gc=0.25/0.1 (sent back) | MEDIUM |
| historia | TBD | 3L/256d architecture | HIGH |
| norman | TBD | 2L/512d depth frontier | HIGH |
| shinji | TBD | gc=0.5 + T_max=25 | HIGH |
| zoro | TBD | Very conservative LR (2-3e-4) | MEDIUM |
| giyu | TBD | Lion at higher LR (3-5e-4) | HIGH |

## PENDING ACTIONS

1. **URGENT**: Merge kakashi #2823 once rebased (0.001095 — new AirfRANS best)
2. **13 new assignments in progress** (assignment agent running)
3. **Watch**: ray T_max=10 seed replication, inosuke 2L depth, gilbert TandemFoil gc compound

## CURRENT RESEARCH THEMES

1. **AirfRANS T_max=10 is the new anchor**: 0.001095 pending merge. Testing WD, LR, seeds at T_max=10.

2. **TandemFoil fine-tuning below 30.10**: LR descent (1.125e-4, 1e-4, 7.5e-5), 2L depth frontier, AdamW test, T_max variants.

3. **DrivAerML crisis recovery**: Lion optimizer, depth reduction (2L/3L), very conservative LR, seed replication, T_max sweep.

4. **2L depth frontier**: Testing 2L across all three benchmarks (inosuke AirfRANS, senku TandemFoil, norman DrivAerML).

## MOST RECENT HUMAN GUIDANCE

Issue #2545: "Revive the strongest missing historical mechanisms." Round 37 update posted. No new directives.
