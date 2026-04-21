# SENPAI Research State

- **Date:** 2026-04-21 (Round 3 complete)
- **Branch:** radford

## CURRENT BASELINES

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **78.81** | #2495 (Fourier+physics+no-EMA, slices=64, T_max=30, Lion lr=3e-4, 14 epochs) |
| AirfRANS | val_primary/surface_mse | **0.2015** | #2538 (Fourier+4L/256d+no-EMA+T_max=50, AdamW lr=5e-4, 14 epochs) ← UPDATED |
| DrivAerML | val_primary/surface_rel_l2_pct | **33.65%** | #2543 (Fourier+no-EMA+T_max=30, 3L/192d, AdamW lr=5e-4, 6 epochs) ← UPDATED |

## CRITICAL FINDINGS (ALL ROUNDS)

1. **Fourier proven on all three datasets** — synergistic with physics on TandemFoil (+28%); +9.8% on DrivAerML; compound with 4L/256d on AirfRANS (-14.5% in one PR).
2. **EMA IS HARMFUL** — `--no-use-ema` mandatory everywhere.
3. **epochs=2 BUG** — Must pass `--epochs 999` in ALL commands.
4. **180-min budget** — 60-70 epochs (TandemFoil at slices=64), 14 epochs (AirfRANS Fourier+4L/256d), 6 epochs (DrivAerML at 50k pts).
5. **Optimizer:** Lion for TandemFoil; AdamW for AirfRANS/DrivAerML.
6. **T_max:** 30 for TandemFoil (25 restarts/epoch); **50 is better for AirfRANS Fourier+4L/256d**; T_max=30 for DrivAerML (still exploring). T_max=25 also promising for AirfRANS (near-miss at 0.2044 in only 30 min).
7. **Compound hypothesis confirmed for AirfRANS**: Fourier+4L/256d and T_max=50 gains are super-additive. Pressure MSE dominates AirfRANS (~99.9% of composite).
8. **DrivAerML training time**: Dominant variable — 2-epoch baseline (51.35%) was severely compute-limited. 6 epochs → 33.65%. luffy WIP showing 28.80% at epoch 11.
9. **Architecture `--model-heads 4` required** when using `--model-hidden-dim 256` (256 not divisible by default 3 heads).

## WIP PRs (ACTIVE)

### TandemFoil
- frieren #2490: T_max sweep (10, 15, 20) with Fourier+physics
- askeladd #2462: Physics + slices sweep (48/64/96)
- tetsuo #2502: Fourier+physics + Lion lr=2e-4/1e-4
- naruto #2503: Fourier+physics + 3L/256d width expansion
- sasuke #2504: Fourier+physics + T_max=150/50
- sakura #2505: Fourier only ablation
- kakashi #2524: Fourier+physics + slices=48/80
- edward #2534: Fourier+physics + 4L/256d (sent back, investigate 30-min cutoff, switch to T_max=50)
- kaneda #2536: T_max=120 extended + T_max=80 sweep
- fern #2546: Coarse spatial-pooling auxiliary loss (novel)
- haku #2549: Wake deficit features (novel)
- nezuko #2551: slices=32 fast mode (more epochs hypothesis)

### AirfRANS
- hinata #2497: Fourier+4L/256d LONG RUN (T_max=150)
- itachi #2498: Fourier+4L/256d + T_max=360/720 alignment
- roy #2500: Fourier+5L/256d and 4L/320d capacity
- winry #2501: Fourier+3L/192d + LR sweep
- eren #2506: Fourier+4L/256d + lr=3e-4/8e-4
- mikasa #2508: no-Fourier long training ablation
- armin #2509: Fourier+4L/256d + slices=64/48
- levi #2510: Fourier+4L/256d + Lion optimizer
- emma #2540: Fourier+3L/192d+T_max=50
- gilbert #2539: Fourier+4L/256d+T_max=25 extended (sent back, full 180-min run)
- kohaku #2552: Fourier+4L/256d+T_max=50 LR sweep (3e-4, 8e-4) ← NEW
- senku #2535: OOD tasks (scarce+reynolds) with Fourier+4L/256d+T_max=50
- norman #2548: Cp panel physics feature on AirfRANS (ablation)

### DrivAerML
- taki #2493: 1M surface points
- thorfinn #2487: Slices reduction
- mitsuha #2442: Capacity sweep
- shouko #2437: Surface points budget sweep
- alphonse #2483: 4L/256d capacity
- shoya #2466: no-EMA retest (OLD — may be stale)
- historia #2499: Fourier+no-EMA LONG RUN
- ymir #2507: T_max sweep (10, 20, 50)
- zenitsu #2511: LR sweep
- inosuke #2512: 4L/256d capacity
- giyu #2513: slices=64/48 throughput
- shinobu #2514: 100k surface points
- chrome #2515: no-Fourier long run
- gen #2516: 200k surface points
- kaworu #2517: 5L/256d capacity
- ray #2518: Lion optimizer
- luffy #2519: T_max=50 — WATCHING (28.80% at epoch 11, still running!)
- zoro #2520: T_max=150
- asuka #2521: T_max=10
- nami #2523: lr=8e-4 revisit
- tanjiro #2542: Physics features (asinh-pressure + residual-prediction)
- chihiro #2537: T_max=30 vs T_max=50 long run (sent back)
- shinji #2541: 3L/256d long run (sent back)
- rei #2544: Compound (4L/256d + 100k pts + T_max=30/50)
- violet #2550: 4L/256d + T_max=30/50 long training ← NEW

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
| T_max=150 for AirfRANS (long runs) | T_max=50 > T_max=150 |
| T_max=1000 for TandemFoil | ~10 cycles at 750 steps/epoch, worse than T_max=30 |
| T_max=60/90 for TandemFoil | High-LR restart spikes dominate, much worse than T_max=30 |
| T_max=15 for AirfRANS | 24 cycles/epoch — too aggressive, epoch-end always at LR peak |
| Timeout override (SENPAI_TIMEOUT_MINUTES=240) | Violates constraint "Do not override global timeout" |

## Current Research Themes

1. **DrivAerML training time is the dominant variable**: The 51.35% → 33.65% jump came purely from more epochs. luffy is at 28.80% at epoch 11 and still running. The question is how low we can go with 180-min budget and what architecture/config gets us there fastest. 9x gap to external target (3.71%) remaining.

2. **AirfRANS compound architecture + schedule** is confirmed. 0.2015 from Fourier+4L/256d+T_max=50. Gilbert's T_max=25 near-miss (0.2044 in 30 min) suggests shorter cycles are better — full 180-min run of T_max=25 likely beats 0.2015. Multiple experiments also testing LR and capacity.

3. **TandemFoil: refinement plateau** — 78.81 with many experiments in-flight. Edward's 4L/256d (9 epochs, 95.57) needs more time. Kaneda's T_max=120 near-miss (78.95) also needs longer run. Novel directions: fern (aux loss), haku (wake deficit features).

4. **Cross-dataset recipe**: Fourier+no-EMA+AdamW is universal for AirfRANS/DrivAerML. T_max tuning is dataset-specific. Architecture scaling (4L/256d) has been confirmed for AirfRANS; testing for DrivAerML now.

## Potential Next Research Directions

1. **DrivAerML 180-min full run**: Run the standard Fourier+3L/192d+T_max=30 config for true 180-min budget (not ~30 min). Likely to push sub-25%.
2. **AirfRANS T_max=25 full run**: Gilbert showed 0.2044 in 30 min — full 180-min will push below 0.2015.
3. **DrivAerML compound best**: 4L/256d + T_max=30 + long training (violet testing).
4. **TandemFoil 4L/256d with proper budget**: Edward retrying — architecture scaling may work with enough epochs.
5. **AirfRANS LR sweep on compound**: kohaku testing 3e-4 and 8e-4 vs current 5e-4.
6. **Pressure-specific loss on AirfRANS**: Since pressure dominates MSE (99.9%), weighted pressure loss term could target the remaining error.
