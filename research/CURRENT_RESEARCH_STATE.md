# SENPAI Research State

- **Date:** 2026-04-25 ~18:35 (cycle 165)
- **Branch:** radford
- **ACTIVE DIRECTIVE:** Issue #3283 — Last Ditch Benchmark Push (TEST metrics only for paper)

## METRIC DISCIPLINE (MANDATORY — from human researcher team, Issue #3283)

**RULE: Only TEST metrics matter for paper-facing comparisons. Val is for internal ranking ONLY.**

- Never compare our val to AB-UPT's test target. They are incomparable.
- All external-facing progress reports MUST cite TEST (full-eval, no --max-eval-batches).
- DM SOTA target: **3.71% TEST** (AB-UPT). Our best: **4.117% TEST** (gojo #3308). Gap: **0.407pp TEST**.

## COST-CONTROL HARVEST (2026-04-25 ~08:20)

Human team (morganmcg1) executed final cost-control harvest pass:
- **42 student pods stopped** — PRs closed before training completed, no results
- **17 pods kept running** for final test-metric harvesting
- **DO NOT reassign idle students** — pods are intentionally stopped

## Current Bests (paper-facing TEST only)

| Dataset | Best TEST (full-eval) | Best Val [internal] | PR |
|---------|----------------------|--------------------|----|
| **DrivAerML** | **4.117%** | **3.521%** (gojo #3380 ep850) | TEST: #3308 / Val: #3380 |
| **TFP** | **0.001712** | 0.001903 | #3346 (yuji 4L lr=4e-5 T_max=15) |
| **AF** | pending | 0.000266 | #3257 (chihiro eval-every-3) |
| **TF** | 22.868 | 21.319 | #3185 — frozen |

DM gap to AB-UPT: **0.407pp TEST** (4.117% vs 3.71%)

## GOJO #3380 — TRAINING COMPLETE, AWAITING FULL-EVAL TEST

**lr=4.8e-4 + T_max=36 MSE-only. W&B run: 5x2to2p8 (FINISHED)**
- **Best val: 3.521% at ep850** — 0.101pp below 3.622% baseline. NEW DM VAL RECORD.
- Final epoch: 865 (600-min timeout hit).
- Trough progression: 3.696→3.644→3.643→3.601→3.598→3.590→3.576→3.555→3.542→**3.521**
- **Student session ended ~09:45Z. Training ran unattended. Student needs to re-activate, run full-eval from best checkpoint (ep850), and post results.**
- **CRITICAL: Full-eval TEST from this run will determine if we can merge and close the gap to AB-UPT.**

## Surviving Fleet (8 open PRs, down from 14)

### DM — Gojo finished, awaiting results
| PR | Student | Config | W&B Result | Notes |
|----|---------|--------|-----------|-------|
| **#3380** | **gojo** | **T_max=36 + lr=4.8e-4 MSE** | **val=3.521% ep850 (FINISHED)** | **AWAITING FULL-EVAL TEST** |

### DM — Other (4)
| PR | Student | Config | Status |
|----|---------|--------|--------|
| #3371 | chopper | 3L/512d + T_max=36 | finished, val=4.647% test=4.213% |
| #3362 | jet | T_max=36 MSE full-eval | finished, val=4.043% test=4.453% |
| #3401 | hinata | lr=4.9e-4 MSE | finished (may be old run) |
| #3381 | usopp | T_max=36 + lr=4.8e-4 + w=0.05 | crashed ep8 |
| #3300 | vegeta | AB-UPT anchored decoder | diverging |

### TFP (2)
| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#3346** | **yuji** | **4L lr=5e-5 T_max=15 sweep** | sent back for next trial |
| #3397 | vash | 3L lr=7.5e-5 T_max=15 | crashed |

### AF (2)
| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#3257** | **chihiro** | **eval-every-3 (val=0.000266 BEST)** | sent back for full-eval TEST |
| #3402 | gohan | eval-every-3 + vol-11x | crashed, diverged |

## Closed This Cycle (4)
- #3396 franky: gc=0.3 — val=3.754%, too aggressive clipping
- #3403 canute: T_max=38 — full-eval val=4.617%, test=4.201%. Massive eval-subset bias.
- #3388 alphonse: T_max=40 — val=3.898%, plateaued early
- #3382 einar: w=0.03 — val=3.782%, test=4.450%. Metric-aware dead at lr=4.8e-4.

## Key Insights

- **DM val record:** gojo #3380 val=3.521% at ep850 (finished). 0.101pp below baseline. CRITICAL: awaiting full-eval TEST.
- **DM compound result:** At lr=4.8e-4+T_max=36, MSE-only is the ONLY successful config. Metric-aware (w=0.03/0.04/0.05), alternate T_max (38/40), and softer gc (0.3) all fail.
- **Eval-subset bias confirmed:** canute showed 0.9pp gap between sampled val (3.705%) and full-eval val (4.617%). All paper-facing metrics MUST use full-eval.
- **TFP 4L BREAKTHROUGH:** yuji test=0.001712 (-4.3% NEW TEST BEST). LR sweep in progress.
- **AF eval-every-3:** chihiro val=0.000266 (-10.1%). Pending full-eval TEST.
