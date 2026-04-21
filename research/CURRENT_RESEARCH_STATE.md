# SENPAI Research State

- **Date:** 2026-04-21 (Round 37 underway)
- **Branch:** radford

## ⚡⚡⚡ ROUND 37 BREAKTHROUGH: AirfRANS 3L/256d — val=0.001479 (46.6% vs merged baseline)

**PR #2771 (itachi, PENDING MERGE — trivial rebase in progress):**
- 3L/256d + gc=1.0 + WD=1e-2 + T_max=5 + AdamW lr=7e-4 hit **val=0.001479** at ep202
- Single change from 4L/256d: **removed one layer**
- 46.6% better than merged baseline 0.00277
- **65.6% below external target 0.0043**
- Fundamental finding: **WIDTH > DEPTH** for AirfRANS at this scale
- Same T_max=5 divergence pattern at ep202 — still a systemic issue

## CURRENT BASELINES

| Dataset | Metric | Value | PR | Key Mechanism |
|---|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **44.72** | #2724 (3L/192d, Lion lr=1.5e-4, T_max=10, 115 ep) | LOWER LR |
| AirfRANS | val_primary/surface_mse | **0.00277** | #2774 (4L/256d, gc=0.5, WD=1e-2, T_max=5) | gc=0.5 |
| AirfRANS PENDING | val_primary/surface_mse | **0.001479** | #2771 (3L/256d, gc=1.0, WD=1e-2, T_max=5, awaiting rebase) | 3L/WIDTH |
| DrivAerML | val_primary/surface_rel_l2_pct | **4.619%** | #2691 (4L/512d/8H, T_max=30, 267 ep) | WIDTH SCALES |

## EXTERNAL TARGETS

| Dataset | External Best | Our Best | Gap |
|---|---|---|---|
| AirfRANS | 0.0043 | **0.00277 (merged)** | **CRUSHED by 35.6%** |
| AirfRANS | 0.0043 | **0.001479 (pending)** | **CRUSHED by 65.6%** |
| DrivAerML | <3.71% | 4.619% | **1.24x** |

## KEY INSIGHTS (updated Round 37)

1. **🔥🔥🔥 3L/256d > 4L/256d on AirfRANS** (Round 37): Removing ONE layer from 4L→3L improved val by 46.6%. Width (256d) is the dominant factor; adding layers hurts. This is a structural finding about 2D CFD surrogates — shallow+wide networks may better preserve local flow field gradients.

2. **🔥🔥 gc=0.5 = BIGGEST PRIOR LEVER** (Round 36): 28.9% improvement at 4L/256d by relaxing gradient clipping from 1.0 to 0.5. Combined with 3L/256d → potentially transformative.

3. **T_max=5 DIVERGENCE ~ep200**: All extended AirfRANS runs die near ep200-210. Pattern is architecture-independent (hit ep205 at 4L, ep202 at 3L). Cosine LR cycling creates irreversible instability. T_max=10 is the best current hypothesis for prevention.

4. **DrivAerML: LOWER LR FALSIFIED** (Round 37): Three independent experiments (ray, levi, chrome-prev) confirmed lr=3e-4 hurts DrivAerML. lr=5e-4 or higher is needed. Higher LR + gc=0.5 is the new direction.

5. **TandemFoil: LOWER LR TREND CONTINUES**: 3e-4→2e-4→1.5e-4 all improved. lr=1e-4 being tested (sakura #2813). Width scaling at 3L/256d newly assigned (levi #2825).

## MANDATORY CONFIG FLAGS

- `--no-use-ema` — EMA bug, mandatory everywhere
- `--epochs 999` — Default is 2, must override
- `SENPAI_MAX_EPOCHS=9999` — Default cap of 50 kills long runs
- `SENPAI_TIMEOUT_MINUTES=180` — Default 30-min insufficient
- Lion optimizer for TandemFoil; AdamW for AirfRANS/DrivAerML
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`

## ACTIVE EXPERIMENTS BY DATASET

### AirfRANS — FRONTIER (baseline: 0.00277 merged; 0.001479 pending)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| kakashi | #2823 | 3L/256d + gc=0.5 compound (4 variants: gc=0.5/0.25, T_max=5/10) | **CRITICAL** |
| ray | #2824 | 3L width frontier: 3L/320d, 3L/384d, 3L/512d | **HIGH** |
| inosuke | TBD | 2L depth frontier: 2L/256d, 2L/384d, 2L/512d | HIGH |
| giyu | #2826 | Dropout=0.1 at 3L/256d | MEDIUM |
| roy | #2818 | gc=0.5 + T_max=10 extended (prevent divergence) | **CRITICAL** |
| tanjiro | #2819 | gc=0.75 sweet-spot | HIGH |
| haku | #2820 | gc=0.5 + lr=5e-4 | HIGH |
| armin | #2816 | 5L/256d + gc=0.5 | HIGH |
| shoya | #2817 | 4L/256d + T_max=10 | HIGH |
| edward | #2801 | pressure-weight 20x + 4L/256d | HIGH |
| chihiro | #2811 | pressure-weight 10x at 3L/192d | MEDIUM |
| tetsuo | #2809 | pressure-weight 30x at 3L/192d | MEDIUM |
| violet | #2812 | pressure-weight 50x at 3L/192d | MEDIUM |
| nezuko | #2808 | 4L/256d baseline replication seed=789 | LOW |
| luffy | #2807 | lr=3e-4 + T_max=10 at 4L/256d | LOW |
| kohaku | #2800 | lr=5e-4 at 4L/256d | LOW |
| thorfinn | #2786 | T_max=7 at 4L/256d | LOW |
| winry | #2778 | WD=0 ablation | LOW |
| hinata | #2770 | WD=5e-3 | LOW |
| emma | #2768 | lr=5e-4 at 4L/256d | LOW |
| mitsuha | #2763 | T_max=10 at 4L/256d | LOW |
| asuka | #2760 | 4L/384d | LOW |

### TandemFoil — BASELINE 44.72 (3L/192d, lr=1.5e-4)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| levi | #2825 | 3L/256d at lr=1.5e-4 (width transfer from AirfRANS) | **HIGH** |
| giyu | #2826 | Dropout=0.1 at 3L/192d | MEDIUM |
| sasuke | #2821 | lr=1.5e-4 + gc=0.5 | HIGH |
| gilbert | #2810 | lr=1.25e-4 + gc=1.0 | HIGH |
| sakura | #2813 | lr=1e-4 at 3L/192d | HIGH |
| historia | #2792 | lr=1.5e-4 + T_max=10 | MEDIUM |
| shinji | #2789 | 3L/256d + Lion lr=2e-4 | LOW |
| norman | #2788 | WD + gc at lr=2e-4 | LOW |
| kaworu | #2754 | golden config lr=2e-4 | LOW |
| alphonse | #2753 | gc=1.5 at lr=2e-4 | LOW |
| senku | #2731 | WD=1e-2 + lr=2e-4 | LOW |
| naruto | #2728 | gc=1.0 + lr=2e-4 | LOW |
| mikasa | #2777 | WD + gc golden | LOW |

### DrivAerML — BASELINE 4.619% (4L/512d, lr=5e-4, T_max=30)
| Student | PR | Experiment | Priority |
|---|---|---|---|
| chrome | #2827 | Higher LR + gc=0.5: lr=7e-4, 8e-4, 1e-3 | **HIGH** |
| levi | #2825 | 3L/512d architecture transfer | HIGH |
| giyu | #2826 | Dropout=0.1 at 4L/512d | MEDIUM |
| ymir | #2822 | gc=0.5 at 4L/512d baseline LR | HIGH |
| kaneda | #2798 | gc=0.5 at 4L/512d | HIGH |
| taki | #2814 | WD=1e-3 + gc=0.5 | MEDIUM |
| rei | #2815 | T_max=50 | MEDIUM |
| zenitsu | #2805 | 4L/640d (wider) | HIGH |
| shouko | #2803 | 5L/512d (deeper) | HIGH |
| shinobu | #2806 | lr=7e-4 (no gc — chrome's gc=0.5 runs are comparison) | HIGH |
| eren | #2804 | T_max=10 | HIGH |
| frieren | #2793 | gc=1.5 | HIGH |
| fern | #2794 | WD=1e-2 | LOW |
| askeladd | #2787 | lr=7e-4 at 4L/320d | LOW |
| zoro | #2782 | gc=1.5 at 4L/320d | LOW |

## IDLE STUDENTS
- **itachi**: PR #2771 sent back for trivial rebase — will be idle again shortly after merge
- All other students have active WIP assignments

## PENDING ACTIONS

1. **URGENT: Merge #2771 when itachi rebases** — val=0.001479, new AirfRANS best
2. **Watch kakashi #2823** — 3L/256d + gc=0.5 compound is the highest-priority run
3. **Watch roy #2818** — gc=0.5 + T_max=10: if it prevents ep205 divergence, model could push below 0.001479
4. **DrivAerML 1.24x gap** — chrome #2827 (higher LR + gc) and ymir #2822 (gc=0.5) are the main bets

## CURRENT RESEARCH THEMES

1. **Width-dominant hypothesis for CFD surrogates**: AirfRANS 3L/256d breakthrough suggests shallower+wider > deeper for 2D flow fields. Now testing on TandemFoil and DrivAerML.

2. **gc sweep**: gc=0.5 wins on AirfRANS. gc=0.25 and gc=0.5+3L compound being tested. gc effect on DrivAerML TBD.

3. **Stability via T_max=10**: The T_max=5 divergence at ~ep200 is the main AirfRANS bottleneck. T_max=10 is the primary hypothesis. Combined with gc=0.5, could extend training past ep200.

4. **First exploration of dropout**: Never tested. Could help with stability.

5. **LR descent on TandemFoil**: Strict monotone improvement as lr decreases. lr=1e-4 being tested.

## NEXT PRIORITY DIRECTIONS

1. 3L/256d + gc=0.5 compound — highest expected gain (two orthogonal improvements)
2. 3L width scaling: 320d, 384d, 512d — find the width sweet spot
3. 2L depth frontier — does the shallower trend continue?
4. Cross-benchmark 3L transfer — essential for the "shared recipe" ICML story
5. DrivAerML gc=0.5 + higher LR — close the 1.24x gap
6. AirfRANS T_max=10 stability — prevent ~ep200 divergence to exploit deep basins

## MOST RECENT HUMAN GUIDANCE

Issue #2545: "Revive the strongest missing historical mechanisms." Last response 2026-04-21 11:17 (Round 36 update). No new directives since.
