# SENPAI Research State

- **Date:** 2026-04-25 ~16:00 (cycle 163 — monitoring)
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
| **DrivAerML** | **4.117%** | **3.598%** (gojo #3380 ep672) | TEST: #3308 / Val: #3380 |
| **TFP** | **0.001712** | 0.001903 | #3346 (yuji 4L lr=4e-5 T_max=15) |
| **AF** | pending | 0.000266 | #3257 (chihiro eval-every-3) |
| **TF** | 22.868 | 21.319 | #3185 — frozen |

DM gap to AB-UPT: **0.407pp TEST** (4.117% vs 3.71%)

## GOJO #3380 — THE KEY EXPERIMENT

**lr=4.8e-4 + T_max=36 MSE-only. W&B run: 5x2to2p8 (running)**
- **Best val: 3.598% at ep672** — BEATS 3.622% val baseline!
- Current: ep710, val=3.745% (cosine peak phase)
- Cosine trough progression: 3.696% → 3.644% → 3.643% → 3.601% → **3.598%**
- Next trough expected ~ep720. Training ongoing.
- Loss spike at ep707 (0.005925) — typical cosine schedule noise.
- **Waiting for student to finish training and post results with full-eval TEST.**

## Surviving Fleet (14 open PRs, 17 pods)

### DM — Still Running (5)
| PR | Student | Config | Latest W&B | Notes |
|----|---------|--------|-----------|-------|
| **#3380** | **gojo** | **T_max=36 + lr=4.8e-4 MSE** | **ep710 val=3.745% best=3.598%** | **HIGHEST PRIORITY** |
| #3382 | einar | T_max=36 + lr=4.8e-4 + w=0.03 | ep703 val=3.935% | |
| #3403 | canute | T_max=38 + lr=4.8e-4 MSE | ep733 val=3.808% | |
| #3396 | franky | T_max=36 + lr=4.8e-4 + gc=0.3 | ep759 val=3.845% | |
| #3388 | alphonse | T_max=40 + lr=4.8e-4 MSE | ep727 val=4.028% | |

### DM — Finished (3, awaiting student results)
| PR | Student | Config | W&B Result | Notes |
|----|---------|--------|-----------|-------|
| #3371 | chopper | 3L/512d + T_max=36 | val=4.647% **test=4.213%** | 2nd best TEST after gojo |
| #3362 | jet | T_max=36 MSE full-eval | val=4.043% test=4.453% | |
| #3401 | hinata | lr=4.9e-4 MSE (may be old run) | val=3.816% test=4.395% | needs verification |

### DM — Problematic (3)
| PR | Student | Config | Status |
|----|---------|--------|--------|
| #3381 | usopp | T_max=36 + lr=4.8e-4 + w=0.05 | crashed at ep8 |
| #3300 | vegeta | AB-UPT anchored decoder | ep5 val=144.86% — diverging |

### TFP (2)
| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#3346** | **yuji** | **4L lr=5e-5 T_max=15 sweep** | sent back for next trial |
| #3397 | vash | 3L lr=7.5e-5 T_max=15 | crashed at ep163 |

### AF (2)
| PR | Student | Config | Status |
|----|---------|--------|--------|
| **#3257** | **chihiro** | **eval-every-3 (val=0.000266 BEST)** | sent back for full-eval TEST |
| #3402 | gohan | eval-every-3 + vol-11x | crashed at ep250, diverged |

## Key Insights

- **DM val breakthrough:** gojo #3380 best val=3.598% at ep672, beating 3.622% baseline. Cosine trough progression steady. CRITICAL: need full-eval TEST to confirm paper-facing improvement.
- **TFP 4L BREAKTHROUGH:** yuji 4L/192d at lr=4e-5 T_max=15 → test=0.001712 (-4.3% NEW TEST BEST). LR sweep (lr=5e-5) in progress.
- **AF eval-every-3:** chihiro val surface_mse=0.000266 (-10.1%). Pending full-eval TEST.
- **Chopper surprise:** 3L/512d test=4.213% — wider 3L nearly matches gojo's 4.117% TEST. Width may partially substitute for depth.
- **Metric-aware loss dead at lr=4.8e-4:** einar w=0.03 (val=3.935%) and all other w variants underperform gojo MSE-only (3.598%).
- **Crashes/divergences:** usopp, vash, gohan, vegeta all failed. Down to ~10 productive experiments.
