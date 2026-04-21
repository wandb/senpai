# SENPAI Research State

- **Date:** 2026-04-21 (Round 37, post-review wave)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **30.10** | #2810 (3L/192d, Lion lr=1.25e-4, gc=1.0, T_max=10) | LR DESCENT |
| AirfRANS | val_primary/surface_mse | **0.001479** | #2771 (3L/256d, gc=1.0, WD=1e-2, T_max=5, AdamW lr=7e-4) | 3L/WIDTH |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (4L/512d/8H, T_max=30, lr=5e-4) | WIDTH SCALES |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | **0.001479** | **CRUSHED by 65.6%** |
| DrivAerML | <3.71% | 4.619% | **1.24x** |

## KEY INSIGHTS (updated Round 37)

1. **TandemFoil LR descent is monotonic**: 3e-4 to 2e-4 to 1.5e-4 to 1.25e-4 all improved. 30.10 is 32.7% better than 44.72. gc=1.0 confirmed essential.

2. **3L/256d > 4L/256d on AirfRANS**: Removing ONE layer from 4L to 3L improved val by 46.6%. Width (256d) is the dominant factor; adding layers hurts.

3. **gc=0.5 = BIGGEST LEVER on AirfRANS**: 28.9% improvement at 4L/256d. Compound with 3L/256d being tested (kakashi #2823).

4. **T_max=5 DIVERGENCE ~ep200**: All extended AirfRANS runs die near ep200-210. T_max=10 made it WORSE (0.002086 vs 0.001479). Ultra-short cycles (T_max=3) being tested.

5. **DrivAerML is EXTREMELY FRAGILE**: Only T_max=30 + lr=5e-4 is stable. ALL other configs crashed. Natural grad norms ~0.76-0.85. gc barely activates. Need fundamentally different approaches.

6. **DrivAerML: LOWER LR FALSIFIED**: Three independent experiments confirmed lr=3e-4 hurts.

7. **lr=7e-4 confirmed optimal for AirfRANS 3L/256d**: Winry's LR sweep showed lower LRs (3e-4, 5e-4) all worse.

## MANDATORY CONFIG FLAGS

- `--no-use-ema` -- EMA bug, mandatory everywhere
- `--epochs 999` -- Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` -- Default cap of 50 kills long runs
- `SENPAI_TIMEOUT_MINUTES=180` -- Default 30-min insufficient
- Lion optimizer for TandemFoil; AdamW for AirfRANS/DrivAerML
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

## ACTIVE EXPERIMENTS BY DATASET

### TandemFoil -- BASELINE 30.10 (3L/192d, lr=1.25e-4, gc=1.0)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| gilbert | #2835 | lr=1.25e-4 + gc=0.5 compound | **CRITICAL** |
| sasuke | #2836 | lr=1e-4 + gc=1.0/0.5 (LR descent continuation) | **HIGH** |
| fern | #2837 | 3L/256d width scaling at lr=1.25e-4 | **HIGH** |
| mikasa | #2838 | lr=7.5e-5 (how low can LR go?) | HIGH |
| gen | #2839 | T_max=20 longer cycles | HIGH |
| alphonse | #2840 | lr=1e-4 multi-seed replication | MEDIUM |
| violet | #2841 | WD ablation at lr=1.25e-4 | MEDIUM |
| tanjiro | #2842 | 3L/256d + lr=1e-4 + gc=0.5 compound | HIGH |
| tetsuo | #2843 | lr=1.25e-4 + gc=0.5 + T_max=5 | HIGH |
| mitsuha | #2844 | lr=1.5e-4 + gc=0.5 seed replication | MEDIUM |
| levi | #2825 | 3L/256d at lr=1.5e-4 (from prior round) | HIGH |
| giyu | #2826 | Dropout=0.1 at 3L/192d | MEDIUM |
| sakura | #2813 | lr=1e-4 at 3L/192d (from prior round) | HIGH |

### AirfRANS -- BASELINE 0.001479 (3L/256d, lr=7e-4, gc=1.0)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| kakashi | #2823 | 3L/256d + gc=0.5/0.25, T_max=5/10 compound | **CRITICAL** |
| ray | #2824 | 3L width frontier: 320d, 384d, 512d | **HIGH** |
| inosuke | #2828 | 2L depth frontier: 2L/256d, 2L/384d, 2L/512d | HIGH |
| giyu | #2826 | Dropout=0.1 at 3L/256d | MEDIUM |
| shoya | #2845 | 3L/256d WD=0 ablation | HIGH |
| nezuko | #2846 | 3L/256d seed replication (seeds 42, 123, 7) | HIGH |
| kohaku | #2848 | 3L/256d gc=0.5 + WD sweep | HIGH |
| emma | #2850 | 3L/256d higher LR (1e-3, 1.5e-3) + gc=0.5 | HIGH |
| roy | #2852 | 3L/256d T_max=3/7 short cycles | HIGH |
| armin | #2854 | 3L/256d aggressive LR (8e-4, 1.2e-3) + gc=0.5 | HIGH |
| winry | #2856 | 3L/256d MLP ratio exploration (2, 8) | HIGH |
| haku | #2820 | 3L/256d + gc=0.5 + lr=5e-4 (sent back) | HIGH |
| thorfinn | #2786 | 3L/256d rerun (sent back) | MEDIUM |
| edward | #2801 | 3L/256d + pressure-weight 20x (sent back) | MEDIUM |
| hinata | #2770 | 3L/256d rerun (sent back) | MEDIUM |
| itachi | #2831 | 3L/256d post-rebase | HIGH |

### DrivAerML -- BASELINE 4.619% (4L/512d, lr=5e-4, T_max=30)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| chrome | #2827 | Higher LR + gc=0.5 (lr=7e-4, 8e-4, 1e-3) | HIGH |
| levi | #2825 | 3L/512d architecture transfer | HIGH |
| giyu | #2826 | Dropout=0.1 at 4L/512d | MEDIUM |
| askeladd | #2834 | Lookahead ablation | HIGH |
| ymir | #2847 | Lion optimizer (lr=1e-4, 5e-5, 2e-4) | **HIGH** |
| rei | #2849 | Conservative LR (4e-4, 4.5e-4, 3.5e-4) | HIGH |
| shinobu | #2851 | WD ablation (0, 5e-3, 2e-2) | HIGH |
| zenitsu | #2853 | T_max neighborhood (20, 25, 35) | HIGH |
| eren | #2855 | Seed replication (42, 123, 7) | HIGH |
| shouko | #2857 | MLP ratio (2, 3, 6) | HIGH |
| kaneda | #2858 | Tight gc (0.7, 0.8, 0.9) | HIGH |
| taki | #2814 | gc=0.25/0.1 (sent back) | MEDIUM |
| frieren | #2832 | New assignment (from prior round) | HIGH |

## PENDING ACTIONS

1. **All 24 new experiments assigned** -- zero idle students from review wave
2. **Watch kakashi #2823**: 3L/256d + gc=0.5 compound is THE highest-priority AirfRANS run
3. **Watch gilbert #2835**: TandemFoil gc=0.5 + lr=1.25e-4 compound
4. **DrivAerML 1.24x gap**: Lion optimizer (ymir #2847) and seed replication (eren #2855) are the key bets
5. **Check prior-round students**: chihiro, historia, naruto, norman, kaworu, senku, shinji, luffy, asuka, zoro may have stale PRs

## CURRENT RESEARCH THEMES

1. **TandemFoil LR descent + compounds**: 1.25e-4 is the new anchor. Testing gc=0.5 compound, further LR descent (1e-4, 7.5e-5), width scaling (3L/256d), schedule variants.

2. **AirfRANS compound optimization at 3L/256d**: gc=0.5+3L compound, width frontier, WD/LR/T_max sweeps at the new architecture. MLP ratio (never tested) added.

3. **DrivAerML crisis recovery**: Everything crashed. New strategy: Lion optimizer, seed replication (verify baseline stability), gentle neighborhood probing (T_max, LR, gc near golden values), MLP ratio, WD ablation.

4. **Cross-benchmark architecture transfer**: 3L architecture being tested on all three datasets. Width scaling at 3L is a shared theme.

5. **Never-tested dimensions**: MLP ratio (default 4), lookahead ablation (default True), dropout (default 0.0) being explored for first time.

## NEXT PRIORITY DIRECTIONS

1. 3L/256d + gc=0.5 compound on AirfRANS (kakashi #2823) -- highest expected gain
2. TandemFoil gc=0.5 + lr=1.25e-4 compound (gilbert #2835) -- could break 30.10
3. DrivAerML Lion optimizer (ymir) -- fundamentally different approach needed
4. 3L width frontier on AirfRANS (ray #2824) -- find width sweet spot
5. TandemFoil 3L/256d (fern #2837, tanjiro #2842) -- cross-benchmark width transfer
6. DrivAerML seed replication (eren) -- verify 4.619% is stable before more experiments

## MOST RECENT HUMAN GUIDANCE

Issue #2545: "Revive the strongest missing historical mechanisms." Last human response acknowledged Round 37 breakthrough. No new directives since.
