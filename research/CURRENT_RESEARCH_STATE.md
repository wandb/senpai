# SENPAI Research State

- **Date:** 2026-04-21 (Round 37, post-review wave #2)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **30.10** | #2810 (3L/192d, Lion lr=1.25e-4, gc=1.0, T_max=10) | LR DESCENT |
| AirfRANS | val_primary/surface_mse | **0.001095** | #2823 PENDING REBASE (3L/256d, gc=1.0, T_max=10, lr=7e-4) | T_max=10! |
| AirfRANS (merged) | val_primary/surface_mse | **0.001479** | #2771 (3L/256d, gc=1.0, WD=1e-2, T_max=5) | 3L/WIDTH |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (4L/512d/8H, T_max=30, lr=5e-4) | WIDTH SCALES |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | **0.001095 (pending)** | **CRUSHED by 74.5%** |
| AirfRANS | 0.0043 | **0.001479 (merged)** | **CRUSHED by 65.6%** |
| DrivAerML | <3.71% | 4.619% | **1.24x** |

## KEY INSIGHTS (updated Round 37 wave 2)

1. **T_max=10 > T_max=5 for AirfRANS at 3L/256d**: kakashi #2823 found 0.001095 at gc=1.0 + T_max=10, vs 0.001479 at gc=1.0 + T_max=5. 26% improvement. NEW golden config: 3L/256d + gc=1.0 + T_max=10 + lr=7e-4 + WD=1e-2.

2. **gc=0.5 HURTS 3L/256d on AirfRANS**: gc=0.5+T_max=5 got 0.001741 (+17.7% worse), gc=0.5+T_max=10 got 0.002424 (+63.9% worse). gc=1.0 is the optimal clipping for this architecture. The hypothesis that gc=0.5 was the "biggest lever" does NOT hold at 3L.

3. **Width frontier confirmed at 256d**: ray #2824 — 3L/320d=0.002095, 3L/384d=0.001637, 3L/512d=DIVERGED. 256d is the AirfRANS sweet spot. Going wider hurts.

4. **Dropout confirmed useless**: giyu #2826 — AirfRANS +39.3%, TandemFoil +11.2%, DrivAerML +231% catastrophic. Capacity reduction prevents deep basins.

5. **TandemFoil LR descent**: 3e-4→2e-4→1.5e-4→1.25e-4 all improved. gc=1.0 + T_max=10 is the golden config at 30.10. lr=1e-4 WITHOUT gc only got 41.59 — gc is essential.

6. **DrivAerML EXTREMELY FRAGILE**: Only T_max=30 + lr=5e-4 stable. All other configs crash. Natural grad norms ~0.76-0.85.

## MANDATORY CONFIG FLAGS

- `--no-use-ema` -- EMA bug, mandatory everywhere
- `--epochs 999` -- Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` -- Default cap of 50 kills long runs
- `SENPAI_TIMEOUT_MINUTES=180` -- Default 30-min insufficient
- Lion optimizer for TandemFoil; AdamW for AirfRANS/DrivAerML (testing AdamW on TandemFoil now)
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

## ACTIVE EXPERIMENTS BY DATASET

### AirfRANS -- PENDING BASELINE 0.001095 (3L/256d, lr=7e-4, gc=1.0, T_max=10)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| kakashi | #2823 | 3L/256d + gc=1.0 + T_max=10 — PENDING REBASE (winner!) | **URGENT** |
| inosuke | #2828 | 2L depth frontier: 2L/256d, 2L/384d, 2L/512d | HIGH |
| ray | TBD | 3L/256d + gc=1.0 + T_max=10 seed replication | **HIGH** |
| luffy | TBD | 3L/256d + gc=1.0 + T_max=10 WD sweep | HIGH |
| asuka | TBD | 3L/256d + gc=1.0 + T_max=10 LR sweep | HIGH |
| shoya | #2845 | 3L/256d WD=0 ablation (at T_max=5) | MEDIUM |
| nezuko | #2846 | 3L/256d seed replication (at T_max=5) | MEDIUM |
| kohaku | #2848 | 3L/256d gc=0.5 + WD sweep (gc=0.5 confirmed bad) | LOW |
| emma | #2850 | 3L/256d higher LR (at T_max=5) | MEDIUM |
| roy | #2852 | 3L/256d T_max=3/7 short cycles | MEDIUM |
| armin | #2854 | 3L/256d aggressive LR (8e-4, 1.2e-3) | MEDIUM |
| winry | #2856 | 3L/256d MLP ratio exploration | MEDIUM |
| haku | #2820 | 3L/256d + gc=0.5 + lr=5e-4 (sent back) | LOW |
| thorfinn | #2786 | 3L/256d rerun (sent back) | LOW |
| edward | #2801 | 3L/256d + pressure-weight 20x (sent back) | MEDIUM |
| hinata | #2770 | 3L/256d rerun (sent back) | LOW |
| itachi | #2831 | 3L/256d multi-seed | MEDIUM |

### TandemFoil -- BASELINE 30.10 (3L/192d, lr=1.25e-4, gc=1.0, T_max=10)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| gilbert | #2835 | lr=1.25e-4 + gc=0.5 compound | **CRITICAL** |
| sasuke | #2836 | lr=1e-4 + gc=1.0/0.5 | **HIGH** |
| fern | #2837 | 3L/256d width scaling | **HIGH** |
| mikasa | #2838 | lr=7.5e-5 (LR floor test) | HIGH |
| gen | #2839 | T_max=20 longer cycles | HIGH |
| alphonse | #2840 | lr=1e-4 multi-seed | MEDIUM |
| violet | #2841 | WD ablation | MEDIUM |
| tanjiro | #2842 | 3L/256d + lr=1e-4 + gc=0.5 compound | HIGH |
| tetsuo | #2843 | lr=1.25e-4 + gc=0.5 + T_max=5 | HIGH |
| mitsuha | #2844 | lr=1.5e-4 + gc=0.5 seed replication | MEDIUM |
| levi | #2825 | 3L/256d at lr=1.5e-4 | HIGH |
| naruto | TBD | T_max=5/15 at winning config | HIGH |
| kaworu | TBD | Seed replication of 30.10 result | **HIGH** |
| senku | TBD | 2L/192d depth frontier | HIGH |
| sakura | TBD | AdamW vs Lion at lr=1.25e-4 | HIGH |
| chihiro | TBD | lr=1.125e-4 and lr=1.375e-4 (LR gap fill) | HIGH |

### DrivAerML -- BASELINE 4.619% (4L/512d, lr=5e-4, T_max=30)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| chrome | #2827 | Higher LR + gc=0.5 | HIGH |
| levi | #2825 | 3L/512d architecture transfer | HIGH |
| askeladd | #2834 | Lookahead ablation | HIGH |
| frieren | #2832 | WD + gc compound | MEDIUM |
| ymir | #2847 | Lion optimizer (lr=1e-4, 5e-5, 2e-4) | **HIGH** |
| rei | #2849 | Conservative LR (4e-4, 4.5e-4, 3.5e-4) | HIGH |
| shinobu | #2851 | WD ablation | HIGH |
| zenitsu | #2853 | T_max neighborhood (20, 25, 35) | HIGH |
| eren | #2855 | Seed replication | **HIGH** |
| shouko | #2857 | MLP ratio | HIGH |
| kaneda | #2858 | Tight gc (0.7, 0.8, 0.9) | HIGH |
| taki | #2814 | gc=0.25/0.1 (sent back) | MEDIUM |
| historia | TBD | 3L/256d architecture transfer | HIGH |
| norman | TBD | 2L/512d depth frontier | HIGH |
| shinji | TBD | gc=0.5 + T_max=25 | HIGH |
| zoro | TBD | Very conservative LR (2e-4, 2.5e-4, 3e-4) | MEDIUM |
| giyu | TBD | Lion at higher LR (3e-4, 5e-4, 1e-3) | HIGH |

## PENDING ACTIONS

1. **URGENT: Merge kakashi #2823 when rebased** -- 0.001095 is the new AirfRANS best
2. **Assignments in progress**: 13 students being assigned by background agent
3. **Watch for next big AirfRANS win**: ray (seed replication), inosuke (2L depth), luffy (WD at T_max=10)
4. **DrivAerML**: No experiment has beaten 4.619%. Fresh approaches (Lion, depth reduction, seed replication) are now in flight.

## CURRENT RESEARCH THEMES

1. **AirfRANS T_max=10 golden is the new anchor**: 3L/256d + gc=1.0 + T_max=10 → 0.001095. Testing WD, LR, and seed stability at the new golden.

2. **TandemFoil LR descent below 1.25e-4**: Testing 1.125e-4, 1e-4, 7.5e-5. Also: 3L/256d width transfer, depth frontier (2L), optimizer comparison (AdamW vs Lion).

3. **DrivAerML crisis recovery**: Everything crashed. Lion optimizer, depth reduction (2L/3L), very conservative LR, tight gc, seed replication being tested simultaneously.

4. **Cross-benchmark architecture transfer**: AirfRANS 3L/256d was breakthrough. Testing if 3L transfers to DrivAerML at 3L/256d and 3L/384d sizes.

5. **2L depth frontier**: Never tested on TandemFoil or DrivAerML. inosuke tests 2L on AirfRANS. senku tests 2L on TandemFoil. norman tests 2L on DrivAerML.

## NEXT PRIORITY DIRECTIONS

1. Merge kakashi #2823 (URGENT — 0.001095, just needs rebase)
2. Watch AirfRANS seed replication (ray) — verify 0.001095 is stable
3. TandemFoil gc=0.5 compound (gilbert #2835) — could push below 30.10
4. DrivAerML Lion optimizer (ymir #2847) and seed replication (eren #2855)
5. TandemFoil depth frontier (senku) — first 2L test
6. AirfRANS 2L depth frontier (inosuke #2828) — does going shallower help further?

## MOST RECENT HUMAN GUIDANCE

Issue #2545: "Revive the strongest missing historical mechanisms." Last human response acknowledged Round 37 breakthrough. No new directives since Round 37 update posted.
