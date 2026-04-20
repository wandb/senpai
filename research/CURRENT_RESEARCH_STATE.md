# SENPAI Research State

- **Date:** 2026-04-20 21:05 UTC
- **Branch:** radford

## THE KEY FINDING: EMA IS HARMFUL — `--no-use-ema` is now mandatory everywhere

With `ema_start_step=50` and only 2 epochs (~1500 steps), EMA barely activates but actively dilutes improving model weights with stale early values. This was independently confirmed on TandemFoil (#2412: -24.7%) and AirfRANS (#2431: +29% regression from EMA). **All future experiments must use `--no-use-ema`.**

## Student Status

### Idle (need new assignments)
- violet — just closed #2434 (slices sweep, EMA-suppressed)
- alphonse — just closed #2433 (AdamW LR sweep — confirmed Lion > AdamW on TandemFoil)
- fern — just closed #2413 (full physics stack — core subset is better)
- shoya — just merged #2440 (DrivAerML first baseline)

### WIP — TandemFoil Round 3 (no-EMA applied)
- tanjiro #2461: Physics + no-EMA + Lion lr sweep (2e-4, 3e-4)
- shinji #2456: **Physics + no-EMA + AdamW** (3e-4, 5e-4) — TRIPLE COMBINATION
- thorfinn #2453: **ANP decoder + no-EMA + physics + AdamW** — HIGH PRIORITY
- askeladd #2462: Physics + no-EMA + slices sweep (48, 64, 96)
- rei #2463: Physics + no-EMA + lookahead ablation + lr=2e-4
- frieren #2464: Physics + no-EMA + cosine T_max sweep (10, 20, 50)

### WIP — AirfRANS Round 3 (no-EMA + max 2 parallel jobs per student)
- kohaku #2465: **No-EMA + AdamW lr=5e-4/8e-4** — should beat 0.3308 baseline
- senku #2459: No-EMA + asinh-pressure + residual-prediction — attacks pressure channel
- emma #2455: No-EMA + 4L/256d capacity retest (single jobs at slices=96)
- haku #2457: No-EMA + Fourier features + lr=8e-4
- norman #2460: No-EMA + OOD tasks (scarce, reynolds)

### WIP — TandemFoil Round 2 (with EMA — results may be suppressed)
- gilbert #2435: cosine T_max sweep (Lion, EMA=True)
- chihiro #2436: RE-stratified sampling (Lion, EMA=True)

### WIP — TandemFoil Round 3 prior batch (EMA=True — predates no-EMA finding)
- edward #2443: Physics + AdamW + slices sweep (EMA=True)
- kaneda #2449: Full physics + AdamW LR sweep (EMA=True)

### WIP — DrivAerML Round 1
- mitsuha #2442: Model capacity sweep (2L/128d, 3L/192d, 4L/256d) + anchor budget
- nezuko #2439: Anchor token budget sweep (4096→16384 supernodes)
- taki #2438: Cosine T_max sweep (10/20/30/50)
- shouko #2437: Surface points budget sweep (0/4k/8k/16k)

## Baselines

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **197.87** | #2412 (frieren v4 — no-EMA, Lion lr=3e-4, no physics) |
| AirfRANS | val_primary/surface_mse | **0.3308** | #2423 (kohaku — AdamW lr=5e-4, EMA=True) |
| DrivAerML | val_primary/surface_rel_l2_pct | **71.35%** | #2440 (shoya — AdamW lr=5e-4, 2 epochs) |

## Key Research Findings So Far

| Finding | Impact | Status |
|---|---|---|
| EMA harmful (2-epoch regime) | -24.7% TandemFoil, +29% AirfRANS regression | **CONFIRMED — mandatory --no-use-ema** |
| Physics features beneficial | -2.4% TandemFoil | Merged (#2414); needs combination with no-EMA |
| AdamW > Lion on AirfRANS | -38% AirfRANS | Merged (#2423); untested on TandemFoil with no-EMA |
| Lion > AdamW on TandemFoil | 197.87 vs 254.34 (22% gap) | Confirmed (#2433); Lion is the TandemFoil optimizer |
| AdamW > Lion on DrivAerML | Lion degraded epoch-over-epoch | Confirmed (#2440); AdamW lr=5e-4 is DrivAerML default |
| 4 parallel jobs → epoch starvation | AirfRANS: 2 epochs vs 6 expected | Fixed: max 2 jobs per student |
| Lookahead is beneficial | Lion val 281→197 without it | Confirmed; keep lookahead=True |
| Full physics stack ≤ core subset | 268 vs 262 (core wins) | Closed (#2413); vortex-panel is bottleneck |
| Slices don't affect throughput | All slices values get same epochs | Confirmed (#2434); data loading is bottleneck |

## Current Research Themes

1. **No-EMA is the dominant lever** — larger than physics features or optimizer choice. All prior results with EMA may be suppressed. We need clean no-EMA baselines across all configurations.

2. **Physics + no-EMA + Lion is the TandemFoil formula** — Lion dominates AdamW on TandemFoil (opposite of AirfRANS/DrivAerML). Tanjiro's #2461 tests the critical physics+no-EMA+Lion combination.

3. **ANP cross-foil decoder** — HIGH PRIORITY per program.md. Thorfinn's #2453 tests it properly for the first time with no-EMA + physics + AdamW.

4. **AirfRANS no-EMA baseline** — kohaku #2465 should beat 0.3308. Senku #2459 attacks the pressure channel bottleneck.

5. **DrivAerML first baseline established at 71.35%** — 5 more students running Round 1 experiments. Need >>10 epochs to approach 3.71% target. No-EMA should be applied in Round 2.

6. **Optimizer is dataset-dependent** — Lion for TandemFoil, AdamW for AirfRANS and DrivAerML. This finding shapes all future experiment design.

## Next Round Priorities

### TandemFoil
- Once tanjiro finds best physics+no-EMA+Lion LR: full combination
- If thorfinn's ANP helps: triple combo (ANP + physics + no-EMA + Lion)
- Wake-angle feature is the most impactful physics feature (#2413)
- slices=32 uses 15GB less memory with same quality — consider for memory-constrained experiments

### AirfRANS
- Once kohaku confirms no-EMA beats baseline: combine with asinh-pressure
- Loss weighting for pressure channel (surface_mse_p dominates error)
- 4L/256d capacity with no-EMA (emma's #2455)

### DrivAerML
- Apply no-EMA to all DrivAerML Round 2 experiments
- Need longer training (current 2 epochs wildly insufficient)
- Anchor token budget (nezuko) and model capacity (mitsuha) are key levers
- Consider increasing timeout if possible
