# SENPAI Research State

- **Date:** 2026-04-21 00:30 UTC
- **Branch:** radford

## UPDATED BASELINES (after this review round)

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **82.65** | #2473 (Fourier+physics+no-EMA, slices=64, T_max=30, Lion lr=3e-4, 14 epochs) |
| AirfRANS | val_primary/surface_mse | **0.2387** | #2478 (Fourier+4L/256d+no-EMA, 8 epochs, T_max=150) ← UPDATED |
| DrivAerML | val_primary/surface_rel_l2_pct | **51.35%** | #2475 (Fourier+no-EMA, T_max=30, AdamW lr=5e-4, 2 epochs) ← UPDATED |

## CRITICAL FINDINGS (UPDATED)

1. **Fourier proven on ALL datasets**: TandemFoil (+28% combined with physics), AirfRANS (+17.4% with 4L/256d), DrivAerML (+9.8%).
2. **EMA IS HARMFUL** — `--no-use-ema` mandatory everywhere.
3. **epochs=2 BUG** — CRITICAL: Must pass `--epochs 999` in ALL training commands.
4. **180-min budget** — 60-70 epochs (TandemFoil), 18-22 epochs (AirfRANS), 12-15 epochs (DrivAerML).
5. **Optimizer:** Lion for TandemFoil; AdamW for AirfRANS/DrivAerML.
6. **DrivAerML LR:** With Fourier, lr=5e-4 beats lr=8e-4 (51.35% vs 54.33%).
7. **T_max:** 30 for TandemFoil (needs retest at new scale); 150 for AirfRANS; 30 for DrivAerML+Fourier.
8. **ANP decoder: conclusively negative** — never use.

## Idle Students
None — all 30 assigned in this round.

## PRs Ready for Review
None currently.

## WIP PRs

### TandemFoil
- frieren #2490: T_max sweep (10, 15, 20) with Fourier+physics
- edward #2491: 4L/256d + Fourier+physics capacity
- gilbert #2471: Golden no-EMA (no Fourier, no physics)
- tanjiro #2485: Golden + Lion LR sweep (2e-4, 1e-4)
- shinji #2486: Golden + AdamW vs Lion
- askeladd #2462: Physics + slices sweep (48/64/96)
- fern #2494: Fourier+physics + T_max=300 long run
- nezuko #2495: Fourier+physics + T_max=30/1000 long run
- haku #2496: Fourier+physics + AdamW optimizer
- tetsuo #NEW: Fourier+physics + Lion lr=2e-4/1e-4
- naruto #NEW: Fourier+physics + 3L/256d width
- sasuke #NEW: Fourier+physics + T_max=150/50
- sakura #NEW: Fourier only (ablation)
- kakashi #NEW: Fourier+physics + slices=48/80

### AirfRANS
- kohaku #2492: Fourier+physics+no-EMA (TandemFoil synergy)
- emma #2482: T_max=50 + lr=8e-4
- hinata #2497: Fourier+4L/256d LONG RUN (most critical)
- itachi #2498: Fourier+4L/256d + T_max=360/720 alignment
- roy #2500: Fourier+5L/256d and 4L/320d capacity
- winry #2501: Fourier+3L/192d + LR sweep
- eren #NEW: Fourier+4L/256d + lr=3e-4/8e-4
- mikasa #NEW: no-Fourier long training ablation
- armin #NEW: Fourier+4L/256d + slices=64/48
- levi #NEW: Fourier+4L/256d + Lion optimizer

### DrivAerML
- taki #2493: 1M training points standardized rerun
- thorfinn #2487: Slices reduction
- shoya #2466: no-EMA retest
- mitsuha #2442: Capacity sweep
- shouko #2437: Surface points budget sweep
- norman #2484: Slices reduction for more epochs
- alphonse #2483: 4L/256d capacity
- historia #NEW: Fourier+no-EMA LONG RUN (most critical)
- ymir #NEW: T_max sweep (10, 20, 50) with Fourier
- zenitsu #NEW: LR sweep (3e-4, 4e-4, 6e-4, 7e-4)
- inosuke #NEW: 4L/256d + Fourier capacity
- giyu #NEW: slices=64/48 throughput
- shinobu #NEW: 100k training surface points
- chrome #NEW: no-Fourier long run ablation
- gen #NEW: 200k training surface points
- ray #NEW: Lion optimizer + Fourier
- asuka #NEW: T_max=10 long run
- kaworu #NEW: 5L/256d capacity
- luffy #NEW: T_max=50 long run
- zoro #NEW: T_max=150 long run
- nami #NEW: lr=8e-4 revisit long run

## Current Research Themes

1. **Long-training convergence**: 6x budget increase is the biggest opportunity. Long runs of all 3 baselines assigned as top priority.
2. **Fourier universality**: Fourier proven on all 3 datasets. Now testing at scale — architectures, LRs, training lengths.
3. **DrivAerML breakthrough**: 51.35% vs 3.71% target (14x gap). Long training + surface point sweep + Fourier is the main attack.
4. **Cross-dataset recipe**: Fourier+no-EMA is the universal baseline. Key divergence: optimizer (Lion/AdamW) and T_max.
5. **T_max recalibration**: Old T_max tuned for 14 epochs. Now testing at 60-70+ epoch scale.

## Confirmed Dead Ends

| Direction | Reason |
|---|---|
| ANP decoder | +5.4% worse |
| EMA | -10% to +29% worse |
| Lion on AirfRANS/DrivAerML | AdamW consistently better |
| Physics at slices=96 | 2 epochs max |
| 6L deep model | Diverges |
| batch_size=4 TandemFoil | Destroys epoch budget |
| Reynolds-stratified sampling | All worse |
| geometry_supernodes flag | NO-OP for senpai_transolver |
| surface_anchor_points flag | NO-OP for senpai_transolver |
