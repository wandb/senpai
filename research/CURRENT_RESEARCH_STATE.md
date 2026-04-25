# SENPAI Research State

- **Date:** 2026-04-25 08:35 (cycle 161 — post-harvest)
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
- **17 experiments kept running** for final test-metric harvesting
- **DO NOT reassign idle students** — pods are intentionally stopped

## Current Bests (paper-facing TEST only)

| Dataset | Best TEST (full-eval) | Best Val [internal] | PR |
|---------|----------------------|--------------------|----|
| **DrivAerML** | **4.117%** | 3.622% | #3308 (gojo lr=4.8e-4 T_max=30) |
| **TFP** | **0.001712** | 0.001903 | #3346 (yuji 4L lr=4e-5 T_max=15) |
| **AF** | pending | 0.000266 | #3257 (chihiro eval-every-3) |
| **TF** | 22.868 | 21.319 | #3185 — frozen |

DM gap to AB-UPT: **0.407pp TEST** (4.117% vs 3.71%)

## Surviving Fleet (17 WIP)

### DM lr=4.8e-4 + T_max=36 compound (8 experiments — HIGHEST PRIORITY)
| PR | Student | Config | Notes |
|----|---------|--------|-------|
| **#3380** | **gojo** | **T_max=36 + lr=4.8e-4 MSE-only** | **THE key experiment** |
| #3381 | usopp | T_max=36 + lr=4.8e-4 + w=0.05 | triple compound |
| #3382 | einar | T_max=36 + lr=4.8e-4 + w=0.03 | |
| #3384 | fern | T_max=36 + lr=4.8e-4 + w=0.04 | w=0.04 best for TEST at T_max=30 |
| #3388 | alphonse | T_max=40 + lr=4.8e-4 MSE | schedule probe above 36 |
| #3396 | franky | T_max=36 + lr=4.8e-4 + gc=0.3 | softer clipping |
| #3401 | hinata | T_max=36 + lr=4.9e-4 MSE | fine LR probe |
| #3403 | canute | T_max=38 + lr=4.8e-4 MSE | fine schedule probe |

### DM other (3 experiments)
| PR | Student | Config |
|----|---------|--------|
| #3362 | jet | T_max=36 MSE full-eval (paper-facing) |
| #3371 | chopper | 3L/512d + T_max=36 (throughput) |
| #3300 | vegeta | AB-UPT-style anchored decoder (breakout) |

### TFP (3 experiments)
| PR | Student | Config |
|----|---------|--------|
| **#3346** | **yuji** | **4L lr=5e-5 T_max=15 (LR sweep — test=0.001712 NEW BEST)** |
| #3377 | haku | 3L lr=8e-5 bracket |
| #3397 | vash | 3L lr=7.5e-5 T_max=15 |

### AF (3 experiments)
| PR | Student | Config |
|----|---------|--------|
| **#3257** | **chihiro** | **eval-every-3 (val=0.000266 NEW BEST, pending full-eval)** |
| #3373 | gilbert | eval-every-3 + vol-12x |
| #3402 | gohan | eval-every-3 + vol-11x (correct AdamW config) |

## Key Insights

- **TFP 4L BREAKTHROUGH:** yuji 4L/192d at lr=4e-5 T_max=15 → test=0.001712 (-4.3% NEW TEST BEST). 4L needs lower LR than 3L. LR sweep (lr=5e-5) in progress.
- **AF eval-every-3:** chihiro val surface_mse=0.000266 (-10.1%). Pending full-eval TEST.
- **DM w=0.04 TEST insight:** canute full-eval TEST=4.126% at w=0.04 vs 4.218% at w=0.05.
- **Compound lr=4.8e-4+T_max=36:** gojo #3380 is THE experiment. Results expected soon.
