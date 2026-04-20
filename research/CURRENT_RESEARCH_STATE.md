# SENPAI Research State

- **Date:** 2026-04-20 20:05 UTC
- **Branch:** radford

## THE KEY FINDING: EMA IS HARMFUL — `--no-use-ema` is now mandatory everywhere

With `ema_start_step=50` and only 2 epochs (~1500 steps), EMA barely activates but actively dilutes improving model weights with stale early values. This was independently confirmed on TandemFoil (#2412: -24.7%) and AirfRANS (#2431: +29% regression from EMA). **All future experiments must use `--no-use-ema`.**

## Student Status

### Idle
- None

### WIP — TandemFoil Round 3 (no-EMA applied)
- tanjiro: Physics + no-EMA + Lion lr sweep (2e-4, 3e-4) — PR creation in flight
- shinji #2456: **Physics + no-EMA + AdamW** (3e-4, 5e-4) — TRIPLE COMBINATION
- thorfinn #2453: **ANP decoder + no-EMA + physics + AdamW** — HIGH PRIORITY
- askeladd #2450: Physics + no-EMA + slices sweep (48, 64, 96)
- rei #2451: Physics + no-EMA + lookahead ablation + lr=2e-4
- frieren #2452: Physics + no-EMA + cosine T_max sweep (10, 20, 50)

### WIP — AirfRANS Round 3 (no-EMA + max 2 parallel jobs per student)
- kohaku #2454: **No-EMA + AdamW lr=5e-4/8e-4** — should beat 0.3308 baseline
- senku #2459: No-EMA + asinh-pressure + residual-prediction — attacks pressure channel
- emma #2455: No-EMA + 4L/256d capacity retest (single jobs at slices=96)
- haku #2457: No-EMA + Fourier features + lr=8e-4
- norman: No-EMA + OOD tasks (scarce, reynolds) — PR creation in flight

### WIP — TandemFoil Round 2 (with EMA — results may be suppressed)
- alphonse #2433: AdamW LR sweep at slices=64 (no physics, EMA=True)
- violet #2434: slices sweep (Lion, EMA=True)
- gilbert #2435: cosine T_max sweep (Lion, EMA=True)
- chihiro #2436: RE-stratified sampling (Lion, EMA=True)
- fern #2413: full physics stack (Lion, EMA=True) — W&B shows val≈270.74

### WIP — TandemFoil Round 3 prior batch (EMA=True — predates no-EMA finding)
- edward #2443: Physics + AdamW + slices sweep (EMA=True)
- kaneda #2449: Full physics + AdamW LR sweep (EMA=True)

### WIP — DrivAerML Round 1
- shoya #2440: AdamW vs Lion baseline (first DrivAerML run)
- shouko #2437: Surface points budget sweep (0/4k/8k/16k)
- mitsuha #2442: Model capacity sweep (2L/128d, 3L/192d, 4L/256d) + anchor budget
- taki #2438: Cosine T_max sweep (10/20/30/50)
- nezuko #2439: Anchor token budget sweep (4096→16384 supernodes)

## Baselines

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **197.87** | #2412 (frieren v4 — no-EMA, Lion lr=3e-4, no physics) |
| AirfRANS | val_primary/surface_mse | **0.3308** | #2423 (kohaku — AdamW lr=5e-4, EMA=True) |
| DrivAerML | val_primary/surface_rel_l2_pct | **no baseline yet** | target: <3.71% |

## Key Research Findings So Far

| Finding | Impact | Status |
|---|---|---|
| EMA harmful (2-epoch regime) | -24.7% TandemFoil, +29% AirfRANS regression | **CONFIRMED — mandatory --no-use-ema** |
| Physics features beneficial | -2.4% TandemFoil | Merged (#2414); needs combination with no-EMA |
| AdamW > Lion on AirfRANS | -38% AirfRANS | Merged (#2423); untested on TandemFoil with no-EMA |
| 4 parallel jobs → epoch starvation | AirfRANS: 2 epochs vs 6 expected | Fixed: max 2 jobs per student |
| Lookahead is beneficial | Lion val 281→197 without it (smaller effect than EMA) | Confirmed; keep lookahead=True |

## Current Research Themes

1. **No-EMA is the dominant lever** — larger than physics features or optimizer choice. All prior results with EMA may be suppressed. We need clean no-EMA baselines across all configurations.

2. **Physics + AdamW + no-EMA triple combination** — the most important TandemFoil experiment. Shinji's #2456 tests this. Expected to substantially beat 197.87.

3. **ANP cross-foil decoder** — HIGH PRIORITY per program.md. Thorfinn's #2453 tests it properly for the first time with no-EMA + physics + AdamW.

4. **AirfRANS pressure channel** — surface_mse_p dominates error by orders of magnitude. Senku's #2459 applies asinh-pressure (proven on TandemFoil) directly to this bottleneck.

5. **DrivAerML first results imminent** — 5 students approaching completion. Will establish first surface_rel_l2_pct baselines. Apply no-EMA finding to next DrivAerML round.

6. **AirfRANS epoch starvation fixed** — Round 3 uses max 2 parallel jobs per student at slices=96. Should restore 6+ epochs per run and enable fair comparison to baseline.

## Next Round Priorities

### TandemFoil
- Once shinji finds best physics+no-EMA+AdamW LR: test + slices sweep (get more epochs)
- If thorfinn's ANP helps: triple combo (ANP + physics + no-EMA + AdamW)
- Wake feature ablation with no-EMA (previous ablation had EMA)
- Full physics stack + no-EMA + AdamW (fern's experiment, now with no-EMA)

### AirfRANS
- Once kohaku confirms no-EMA beats baseline: combine with best LR + asinh-pressure
- 4L/256d capacity with no-EMA (emma's #2455) — may finally show capacity benefit
- Loss weighting for pressure channel (if asinh-pressure insufficient)
- Fourier features (haku's #2457) — hypothesis: better high-frequency resolution

### DrivAerML
- Apply no-EMA to all DrivAerML Round 2 experiments
- Anchor token budget is the highest-impact lever (nezuko's sweep)
- Combined uplift: large anchors + no-EMA + best optimizer + LR
