# SENPAI Research State

- **Date:** 2026-04-20 21:45 UTC
- **Branch:** radford

## THE KEY FINDINGS

1. **slices=64 + T_max=30 is the golden config** — 11 epochs vs 2 at slices=96. More training overwhelms resolution loss. 114.92 val MAE with EMA=True still crushes old baseline (197.87).
2. **EMA IS HARMFUL** — `--no-use-ema` mandatory. -24.7% TandemFoil, +29% AirfRANS regression.
3. **Fourier features breakthrough on AirfRANS** — `--enable-fourier` + no-EMA beat 6-epoch baseline in 2 epochs (-9.1%).
4. **Optimizer is dataset-dependent** — Lion for TandemFoil, AdamW for AirfRANS and DrivAerML.

## Student Status

### Idle
- None

### WIP — TandemFoil "Golden Config" experiments (slices=64, T_max=30)
- gilbert #2471: **Golden config + no-EMA** — HIGHEST PRIORITY, projected ~86-90
- kaneda #2472: Golden config + core physics + no-EMA (Lion + AdamW)
- edward #2473: Golden config + Fourier + no-EMA (transfer AirfRANS finding)

### WIP — TandemFoil Round 3 (no-EMA, slices=96 — may be suboptimal now)
- tanjiro #2461: Physics + no-EMA + Lion lr sweep (2e-4, 3e-4)
- shinji #2456: Physics + no-EMA + AdamW (3e-4, 5e-4) — TRIPLE COMBINATION
- thorfinn #2453: **ANP decoder + no-EMA + physics + AdamW** — HIGH PRIORITY
- askeladd #2462: Physics + no-EMA + slices sweep (48, 64, 96)
- rei #2463: Physics + no-EMA + lookahead ablation + lr=2e-4
- frieren #2464: Physics + no-EMA + cosine T_max sweep (10, 20, 50)
- fern #2468: Wake-angle + core physics + no-EMA

### WIP — AirfRANS Round 3
- haku #2470: **Fourier + no-EMA full epoch run** — HIGHEST AirfRANS PRIORITY
- kohaku #2465: No-EMA + AdamW lr=5e-4/8e-4 baseline
- senku #2474: **Fourier + no-EMA + 4L/256d capacity**
- emma #2455: No-EMA + 4L/256d capacity retest
- norman #2460: No-EMA + OOD tasks (scarce, reynolds)
- alphonse #2469: No-EMA + cosine T_max sweep

### WIP — DrivAerML
- shoya #2466: No-EMA retest of AdamW winner
- violet #2467: No-EMA + higher LR bracket (8e-4, 1e-3)
- chihiro #2475: **Fourier + no-EMA** (transfer AirfRANS finding)
- mitsuha #2442: Model capacity sweep
- nezuko #2439: Anchor token budget sweep
- taki #2438: Cosine T_max sweep
- shouko #2437: Surface points budget sweep

## Baselines

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **114.92** | #2435 (gilbert — slices=64, T_max=30, EMA=True, 11 epochs) |
| AirfRANS | val_primary/surface_mse | **0.3009** | #2457 (haku — Fourier + no-EMA + AdamW lr=5e-4, 2 epochs) |
| DrivAerML | val_primary/surface_rel_l2_pct | **71.35%** | #2440 (shoya — AdamW lr=5e-4, 2 epochs) |

## Key Research Findings

| Finding | Impact | Status |
|---|---|---|
| **slices=64 enables 11 epochs (vs 2 at slices=96)** | **-42% TandemFoil (114.92 vs 197.87)** | **CONFIRMED — slices=64 + T_max=30 is golden config** |
| **cosine_t_max=30 optimal for TandemFoil** | T_max=30 > 10 > 50 > 20 at 11 epochs | Confirmed (#2435) |
| EMA harmful (short training regime) | -24.7% TandemFoil, +29% AirfRANS regression | **CONFIRMED — mandatory --no-use-ema** |
| Fourier features on AirfRANS | -9.1% at 2 epochs vs 6-epoch baseline | **CONFIRMED — mandatory --enable-fourier for AirfRANS** |
| Lion > AdamW on TandemFoil | 197.87 vs 254.34 (22% gap) | Confirmed; needs retest at slices=64 |
| AdamW > Lion on AirfRANS/DrivAerML | -38% AirfRANS, Lion degrades on DrivAerML | Confirmed |
| Full physics stack ≤ core subset | 268 vs 262 at 2 epochs | May change with 11 epochs (kaneda testing) |
| Wake-angle most impactful single feature | +22.7 val MAE when removed from full stack | Confirmed (#2413) |
| asinh-pressure metric incompatibility on AirfRANS | Transforms target space, metrics not comparable | Closed (#2459); needs inverse-transform eval |

## Current Research Themes

1. **TandemFoil golden config + no-EMA** — gilbert #2471 is the HIGHEST PRIORITY experiment. slices=64 + T_max=30 with EMA=True got 114.92. Without EMA, projected ~86-90. This could be a dramatic TandemFoil result.

2. **Golden config + physics and Fourier** — kaneda #2472 tests physics features, edward #2473 tests Fourier features, both on the golden config. Either or both may compound with the epoch advantage.

3. **AirfRANS Fourier full epoch run** — haku #2470 is the highest AirfRANS priority. 2-epoch Fourier already beat 6-epoch baseline. Full 6 epochs should push much lower.

4. **Cross-dataset Fourier transfer** — edward #2473 tests Fourier on TandemFoil, chihiro #2475 tests Fourier on DrivAerML.

5. **DrivAerML needs breakthroughs** — 71.35% is far from 3.71% target. Fourier + no-EMA (chihiro) and model capacity (mitsuha) are the best bets.

6. **Many Round 3 TandemFoil experiments running at slices=96** — these will get only 2 epochs. When they complete, redirect those students to slices=64 golden config variants.

## Next Round Priorities

### TandemFoil
- After gilbert's no-EMA result: combine best config with physics + Fourier + ANP
- All future TandemFoil experiments MUST use slices=64 + T_max=30
- Retest Lion vs AdamW at 11 epochs (prior comparison was at 2 epochs)

### AirfRANS
- Haku full epoch Fourier is top priority
- Fourier + capacity (senku #2474) could push further
- All future AirfRANS must include --enable-fourier

### DrivAerML
- Test Fourier + no-EMA (chihiro #2475)
- Consider slices reduction for more epochs (like TandemFoil breakthrough)
- Anchor budget results (nezuko) will inform next round
