# SENPAI Research State

- **Date:** 2026-04-25 08:05 (cycle 158)
- **Branch:** radford
- **ACTIVE DIRECTIVE:** Issue #3283 — Last Ditch Benchmark Push (TEST metrics only for paper)

## METRIC DISCIPLINE (MANDATORY — from human researcher team, Issue #3283)

**RULE: Only TEST metrics matter for paper-facing comparisons. Val is for internal ranking ONLY.**

- Never compare our val to AB-UPT's test target. They are incomparable.
- All external-facing progress reports MUST cite TEST (full-eval, no --max-eval-batches).
- Val figures may appear in internal analysis with explicit "[val only]" labeling.
- DM SOTA target: **3.71% TEST** (AB-UPT). Our best: **4.117% TEST** (gojo #3308). Gap: **0.407pp TEST**.
- Progress is measured in TEST points only.

## Current Bests

| Dataset | Val | Test (full-eval) | PR |
|---------|-----|------|-----|
| **DrivAerML (DM)** | 3.622% | **4.117%** (gojo lr=4.8e-4 T_max=30) | #3308 |
| **TandemFoil Paper (TFP)** | 0.001857 | **0.001789** (haku lr=7.5e-5 T_max=10) | #3352 |
| **AirfRANS (AF)** | **0.000266** (chihiro eval-every-3 — pending full-eval) | pending | #3257 |
| **TandemFoil (TF)** | 21.319 | 22.868 | #3185 — frozen |

External targets: DM AB-UPT = 3.71% TEST (we're at 4.117%, gap = 0.407pp)

## Fleet Status

- **Students:** 59 total. Zero idle. All assigned.
- **PRs ready for review:** 0
- **PRs WIP:** 59 (includes compound fleet + AF eval-every-3 + TFP LR probes)

## HIGHEST PRIORITY — DM lr=4.8e-4 + T_max=36 compound fleet (~18 experiments)

These compound the two best individual findings:
- lr=4.8e-4: best full-eval TEST=4.117% (gojo, T_max=30)
- T_max=36: best val=3.622% (chrome, lr=5e-4)

| PR | Student | Config |
|----|---------|--------|
| #3380 | gojo | T_max=36 + lr=4.8e-4 MSE-only **HIGHEST** |
| #3381 | usopp | T_max=36 + lr=4.8e-4 + w=0.05 |
| #3382 | einar | T_max=36 + lr=4.8e-4 + w=0.03 |
| #3383 | sukuna | T_max=36 + lr=4.8e-4 MSE + seed=42 |
| #3384 | fern | T_max=36 + lr=4.8e-4 + w=0.04 |
| #3385 | mitsuha | T_max=36 + lr=4.6e-4 MSE |
| #3386 | shinobu | T_max=36 + lr=4.8e-4 + w=0.06 |
| #3387 | shoya | T_max=36 + lr=4.8e-4 + w=0.07 |
| #3388 | alphonse | T_max=40 + lr=4.8e-4 MSE |
| #3389 | stark | T_max=36 + lr=4.8e-4 + w=0.05 + seed=0 |
| #3390 | frieren | T_max=36 + lr=4.8e-4 + w=0.02 |
| #3391 | senku | T_max=36 + lr=4.8e-4 + WD=1e-3 MSE |
| #3392 | bulma | T_max=36 + lr=4.8e-4 + w=0.05 + WD=1e-3 |
| #3393 | mugen | T_max=36 + lr=4.8e-4 MSE + seed=13 |
| #3394 | taki | T_max=36 + lr=4.8e-4 + w=0.05 + seed=42 |
| #3395 | norman | T_max=36 + lr=4.8e-4 MSE + seed=0 |
| #3396 | franky | T_max=36 + lr=4.8e-4 + gc=0.3 MSE *(new)* |
| #3399 | levi | T_max=34 + lr=4.8e-4 MSE *(schedule probe)* |

## DM T_max=36 + lr=5e-4 fleet (~18 experiments, all WIP)

PRs #3359–#3378 covering w=0.02–0.07, seeds, WD, gc, LR probes (4.5e-4–5.5e-4), 3L architecture.

Plus: #3398 griffith (lr=5.2e-4 T_max=36 MSE, new).

## AF Direction — eval-every-3 BREAKTHROUGH

Chihiro #3257: AdamW 2L/256d + eval-every-3 = val surface_mse=0.000266 (-10.1% vs 0.000296 baseline).
**NEW AF BEST on primary metric.** Pending full-eval TEST. Sent back requesting test metrics.

AF fleet also covers vol-weight sweep (tanjiro #3368 vol-10x, emma #3372 vol-11x, gilbert #3373 vol-12x)
and extended eval variants (gohan #3356 WD=5e-3, casca #3354 seed=42).

## TFP Direction — LR optimum at 7.5e-5

Champion: lr=7.5e-5, T_max=10, 3L/192d, Lion, EMA=0.999 (val=0.001857, test=0.001789).
In flight:
- haku #3377: lr=8e-5 + lr=7.25e-5 bracket (upper probe)
- nobara #3379: lr=7.5e-5 seed=42+13 (robustness)
- vash #3397: lr=7.5e-5 + T_max=15 (schedule probe, new)
- askeladd #3400: lr=7.5e-5 + T_max=20 + seed=42 (schedule+seed, new)
- yuji #3346: 4L/192d at lr=4e-5 T_max=15 (depth probe)

## Key Insights Learned

- **lr=4.8e-4 CRITICAL:** Gives best DM full-eval TEST=4.117% at T_max=30. -2.4% vs lr=5e-4.
- **T_max=36 CRITICAL:** Gives best DM val=3.622% at lr=5e-4. Clear schedule optimum.
- **Compound lr=4.8e-4+T_max=36 is untested** — results expected within next few hours.
- **AF eval-every-3:** +4% throughput, -10.1% surface_mse. Mechanism: more training epochs in same budget.
- **eval-every-5 unsafe:** Catastrophic divergence at ep745. Stay at eval-every-3.
- **AF wrong-config lesson:** eval-every-3 must use AdamW 2L/256d (not Lion 4L/256d TFP config).
- **DM seed sensitivity:** lr=4.8e-4 at T_max=30 — seeds 0+42 converge to 3.700%, seed=1024 stuck at 4.073%.
- **Checkpoint bug fixed (griffith):** kill-if-no-improvement now saves checkpoint before RuntimeError.

## Potential Next Directions

When compound fleet results arrive:
1. If DM val < 3.622%: merge, update baseline, assign seed robustness and gc variants
2. If chihiro full-eval TEST improves over baseline: merge AF winner, assign seed/vol-weight compounds
3. DM lr fine-tuning around 4.8e-4 (4.7e-4, 4.9e-4)
4. DM extended budget (720-min) if model still improving at 600-min cutoff
5. TFP: T_max probe results from vash/askeladd — if T_max>10 helps, extend schedule range
