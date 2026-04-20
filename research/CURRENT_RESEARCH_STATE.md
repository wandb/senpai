# SENPAI Research State

- **Date:** 2026-04-21 01:30 UTC
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **82.65** | #2473 (Fourier+physics+no-EMA, slices=64, T_max=30, Lion lr=3e-4, 14 epochs) |
| AirfRANS | val_primary/surface_mse | **0.2357** | #2482 (no-Fourier, 3L/192d, T_max=50, lr=5e-4, no-EMA, 24 epochs) ← UPDATED |
| DrivAerML | val_primary/surface_rel_l2_pct | **51.35%** | #2475 (Fourier+no-EMA, T_max=30, AdamW lr=5e-4, 2 epochs) |

## CRITICAL FINDINGS (ALL ROUNDS)

1. **Fourier proven on TandemFoil and DrivAerML** — synergistic with physics on TandemFoil (+28%); +9.8% on DrivAerML. Does NOT transfer to AirfRANS (asinh metric space issue).
2. **EMA IS HARMFUL** — `--no-use-ema` mandatory everywhere.
3. **epochs=2 BUG** — Must pass `--epochs 999` in ALL commands.
4. **180-min budget** — 60-70 epochs (TandemFoil), 18-24 epochs (AirfRANS), 12-15 epochs (DrivAerML).
5. **Optimizer:** Lion for TandemFoil; AdamW for AirfRANS/DrivAerML. AdamW definitively DEAD on TandemFoil.
6. **T_max:** 30 for TandemFoil/DrivAerML (may need rescaling for 180-min); **50 is better than 150 for AirfRANS** at long training.
7. **ANP decoder: dead.** Fourier+physics: dead on AirfRANS. AdamW on TandemFoil: dead.
8. **DrivAerML LR:** With Fourier, lr=5e-4 beats lr=8e-4.
9. **AirfRANS next priority:** Fourier+4L/256d+T_max=50 — combining best architecture (PR #2478) with best schedule (PR #2482).

## WIP PRs

### TandemFoil (14 running)
- frieren #2490: T_max sweep (10, 15, 20) with Fourier+physics
- askeladd #2462: Physics + slices sweep (48/64/96)
- fern #2494: Fourier+physics + T_max=300 long run
- nezuko #2495: Fourier+physics + T_max=30/1000 long run
- haku #2496: Fourier+physics + AdamW optimizer
- tetsuo #2502: Fourier+physics + Lion lr=2e-4/1e-4
- naruto #2503: Fourier+physics + 3L/256d width expansion
- sasuke #2504: Fourier+physics + T_max=150/50
- sakura #2505: Fourier only (ablation)
- kakashi #2524: Fourier+physics + slices=48/80
- edward #NEW: Fourier+physics + 4L/256d (fair 180-min retry)
- kaneda #NEW: T_max=60/90/120 sweep (epoch-scaled for long runs)
- tanjiro #2485: Golden + Lion LR sweep (2e-4, 1e-4) — still WIP if not submitted

### AirfRANS (13 running)
- hinata #2497: Fourier+4L/256d LONG RUN (T_max=150)
- itachi #2498: Fourier+4L/256d + T_max=360/720 alignment
- roy #2500: Fourier+5L/256d and 4L/320d capacity
- winry #2501: Fourier+3L/192d + LR sweep
- eren #2506: Fourier+4L/256d + lr=3e-4/8e-4
- mikasa #2508: no-Fourier long training ablation
- armin #2509: Fourier+4L/256d + slices=64/48
- levi #2510: Fourier+4L/256d + Lion optimizer
- kohaku #NEW: **Fourier+4L/256d+T_max=50** ← MOST CRITICAL
- emma #NEW: Fourier+3L/192d+T_max=50
- senku #NEW: OOD tasks (scarce+reynolds) with Fourier+4L/256d+T_max=50
- gilbert #NEW: Fourier+4L/256d+T_max=25/15 (very short cycles)

### DrivAerML (22 running)
- taki #2493: 1M surface points standardized
- thorfinn #2487: Slices reduction
- mitsuha #2442: Capacity sweep
- shouko #2437: Surface points budget sweep
- norman #2484: Slices reduction
- alphonse #2483: 4L/256d capacity
- shoya #2466: no-EMA retest
- historia #2499: Fourier+no-EMA LONG RUN (most critical)
- ymir #2507: T_max sweep (10, 20, 50)
- zenitsu #2511: LR sweep
- inosuke #2512: 4L/256d capacity
- giyu #2513: slices=64/48 throughput
- shinobu #2514: 100k surface points
- chrome #2515: no-Fourier long run
- gen #2516: 200k surface points
- kaworu #2517: 5L/256d capacity
- ray #2518: Lion optimizer
- luffy #2519: T_max=50 short run
- zoro #2520: T_max=150
- asuka #2521: T_max=10
- nami #2523: lr=8e-4 revisit
- violet #NEW: Replica long run (validate historia)
- tanjiro #NEW: Physics features (asinh-pressure + residual-prediction)
- chihiro #NEW: Fourier+T_max=50 long run
- shinji #NEW: 3L/256d width expansion + T_max=50
- rei #NEW: Compound (4L/256d + 100k pts + T_max=30/50)

## CONFIRMED DEAD ENDS

| Direction | Reason |
|---|---|
| ANP decoder | +5.4% worse |
| EMA | -10% to +29% worse |
| Lion on AirfRANS/DrivAerML | AdamW consistently better |
| AdamW on TandemFoil | DEFINITIVELY dead — gap widens with more training |
| Physics at slices=96 | 2 epochs max |
| Fourier+physics on AirfRANS | Metric space incompatibility, physical-space worse |
| 6L deep model | Diverges |
| batch_size=4 TandemFoil | Destroys epoch budget |
| Reynolds-stratified sampling | All worse |
| geometry_supernodes flag | NO-OP for senpai_transolver |
| surface_anchor_points flag | NO-OP for senpai_transolver |
| T_max=150 for AirfRANS (long runs) | T_max=50 > T_max=150 at 24 epochs |

## Current Research Themes

1. **AirfRANS breakthrough**: Fourier+4L/256d+T_max=50 is the most promising next step. T_max=50 allows ~345 warm restarts vs ~38 for T_max=150 — much better generalization. Kohaku running this now.

2. **TandemFoil: long training optimization**: T_max scaling for 60-70 epoch runs is the key open question. Kaneda testing T_max=60/90/120. Edward testing 4L/256d capacity properly.

3. **DrivAerML: breadth attack**: 51.35% vs 3.71% target — 14x gap. Running the widest possible sweep: T_max, LR, capacity, surface points, physics features. Historia/violet running long training replicas.

4. **Cross-dataset recipe**: Fourier+no-EMA is universal. T_max=50 appears better than 30 for both AirfRANS and DrivAerML at long training. Testing if T_max=50 also improves TandemFoil (sasuke is testing this).

## Potential Next Research Directions

1. **Compound AirfRANS**: After kohaku's result, compound Fourier+4L/256d+T_max=50 with lr optimization.
2. **TandemFoil longer cycles**: If kaneda's T_max=60/90/120 beats T_max=30, update the golden config.
3. **DrivAerML physics**: Tanjiro testing physics features on DrivAerML — if positive, this could be a major insight.
4. **AirfRANS T_max even shorter (< 25)**: Gilbert testing T_max=15/25.
5. **Multi-dataset T_max=50 hypothesis**: If T_max=50 works better on AirfRANS (confirmed), does it also work better on DrivAerML? Chihiro testing this.
