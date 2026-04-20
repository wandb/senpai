# SENPAI Research State

- **Date:** 2026-04-20 21:20 UTC
- **Branch:** radford

## THE KEY FINDINGS

1. **EMA IS HARMFUL** — `--no-use-ema` is mandatory everywhere. -24.7% TandemFoil, +29% AirfRANS regression.
2. **Fourier features breakthrough on AirfRANS** — `--enable-fourier` + no-EMA beat 6-epoch baseline in just 2 epochs (-9.1%). Now mandatory for AirfRANS.
3. **Optimizer is dataset-dependent** — Lion for TandemFoil, AdamW for AirfRANS and DrivAerML.

## Student Status

### Idle
- None

### WIP — TandemFoil Round 3 (no-EMA applied)
- tanjiro #2461: Physics + no-EMA + Lion lr sweep (2e-4, 3e-4)
- shinji #2456: **Physics + no-EMA + AdamW** (3e-4, 5e-4) — TRIPLE COMBINATION
- thorfinn #2453: **ANP decoder + no-EMA + physics + AdamW** — HIGH PRIORITY
- askeladd #2462: Physics + no-EMA + slices sweep (48, 64, 96)
- rei #2463: Physics + no-EMA + lookahead ablation + lr=2e-4
- frieren #2464: Physics + no-EMA + cosine T_max sweep (10, 20, 50)
- fern #2468: **Wake-angle + core physics + no-EMA** — tests most impactful single feature

### WIP — AirfRANS Round 3 (no-EMA + Fourier baseline established)
- kohaku #2465: No-EMA + AdamW lr=5e-4/8e-4 clean baseline
- senku #2459: No-EMA + asinh-pressure + residual-prediction — attacks pressure channel
- emma #2455: No-EMA + 4L/256d capacity retest
- norman #2460: No-EMA + OOD tasks (scarce, reynolds)
- haku #2470: **Fourier + no-EMA full epoch run + LR/T_max variants** — HIGHEST PRIORITY
- alphonse #2469: No-EMA + cosine T_max sweep (10, 20, 50)

### WIP — TandemFoil Round 2 (with EMA — results likely suppressed)
- gilbert #2435: cosine T_max sweep (Lion, EMA=True)
- chihiro #2436: RE-stratified sampling (Lion, EMA=True)

### WIP — TandemFoil Round 3 prior batch (EMA=True — predates no-EMA finding)
- edward #2443: Physics + AdamW + slices sweep (EMA=True)
- kaneda #2449: Full physics + AdamW LR sweep (EMA=True)

### WIP — DrivAerML Round 1-2
- shoya #2466: **No-EMA retest** of AdamW lr=5e-4/3e-4 winner
- violet #2467: **No-EMA + higher LR bracket** (8e-4, 1e-3)
- mitsuha #2442: Model capacity sweep (2L/128d, 3L/192d, 4L/256d)
- nezuko #2439: Anchor token budget sweep (4096→16384 supernodes)
- taki #2438: Cosine T_max sweep (10/20/30/50)
- shouko #2437: Surface points budget sweep (0/4k/8k/16k)

## Baselines

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **197.87** | #2412 (no-EMA, Lion lr=3e-4, no physics) |
| AirfRANS | val_primary/surface_mse | **0.3009** | #2457 (haku — Fourier + no-EMA + AdamW lr=5e-4, 2 epochs) |
| DrivAerML | val_primary/surface_rel_l2_pct | **71.35%** | #2440 (shoya — AdamW lr=5e-4, 2 epochs) |

## Key Research Findings

| Finding | Impact | Status |
|---|---|---|
| EMA harmful (2-epoch regime) | -24.7% TandemFoil, +29% AirfRANS regression | **CONFIRMED — mandatory --no-use-ema** |
| **Fourier features on AirfRANS** | **-9.1% at 2 epochs vs 6-epoch baseline** | **CONFIRMED — mandatory --enable-fourier for AirfRANS** |
| Physics features beneficial | -2.4% TandemFoil | Merged (#2414); being combined with no-EMA |
| AdamW > Lion on AirfRANS/DrivAerML | -38% AirfRANS, Lion degrades on DrivAerML | Confirmed; optimizer is dataset-dependent |
| Lion > AdamW on TandemFoil | 197.87 vs 254.34 (22% gap) | Confirmed (#2433) |
| Full physics stack ≤ core subset | 268 vs 262 (core wins) | Closed (#2413); wake-angle is most impactful feature |
| Slices don't affect throughput | All slices values get same epochs | Confirmed (#2434); data loading is bottleneck |
| 4 parallel jobs → epoch starvation | AirfRANS/DrivAerML: fewer epochs | Fixed: max 1-2 jobs per student |
| Lookahead is beneficial | Lion val 281→197 without it | Confirmed; keep lookahead=True |

## Current Research Themes

1. **AirfRANS Fourier + no-EMA is the new frontier** — haku's 2-epoch result (0.3009) beat the 6-epoch baseline. Full epoch budget should push to ~0.26-0.28. This is the highest-priority experiment in the programme (haku #2470).

2. **Physics + no-EMA + Lion for TandemFoil** — 7 students attacking the 197.87 baseline from different angles. Tanjiro's physics+no-EMA LR sweep (#2461) and fern's wake-angle test (#2468) are the most promising.

3. **ANP cross-foil decoder** — HIGH PRIORITY per program.md. Thorfinn's #2453 tests it with no-EMA + physics + AdamW.

4. **DrivAerML no-EMA application** — shoya (#2466) and violet (#2467) applying the no-EMA finding. 4 more students in Round 1. Target: 71.35% → much lower.

5. **AirfRANS should test Fourier + asinh-pressure combination** — Fourier improved pressure resolution, asinh-pressure compresses dynamic range. These may compound.

6. **TandemFoil should test Fourier features** — if Fourier helps AirfRANS pressure, it may help TandemFoil surface_pressure_mae too.

## Next Round Priorities

### TandemFoil
- Once tanjiro finds best physics+no-EMA+Lion LR: full combination
- If fern's wake-angle helps: add to standard physics recipe
- Test Fourier features (proven on AirfRANS) on TandemFoil
- If thorfinn's ANP helps: triple combo (ANP + physics + no-EMA + Lion)

### AirfRANS
- Haku #2470 (Fourier full run) is highest priority — expect major improvement
- Combine Fourier + asinh-pressure (senku's approach + Fourier)
- Test Fourier + 4L/256d (emma's capacity + Fourier)
- All future AirfRANS experiments should include --enable-fourier

### DrivAerML
- No-EMA results from shoya/violet will set new baseline
- Test Fourier on DrivAerML too
- Anchor budget and capacity sweeps (nezuko, mitsuha) inform next round
- Need >>2 epochs to approach 3.71% target
