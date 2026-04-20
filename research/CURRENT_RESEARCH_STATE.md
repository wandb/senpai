# SENPAI Research State

- **Date:** 2026-04-20 23:30 UTC
- **Branch:** radford

## THE KEY FINDINGS (UPDATED)

1. **Fourier+physics is the new TandemFoil golden config** — 82.65 val (28.1% below old baseline 114.92). Synergistic: more epochs (14 vs 11) AND better per-epoch quality. Still improving at cutoff.
2. **Epoch count is the universal dominant lever** — But Fourier+physics doesn't reduce epoch count at slices=64 (contrary to prior assumption).
3. **EMA IS HARMFUL** — `--no-use-ema` mandatory across all datasets.
4. **Physics features work when combined with Fourier at slices=64** — The 7x overhead bottleneck observed at slices=96 doesn't manifest at slices=64. This is a critical insight revision.
5. **ANP decoder conclusively negative** — Never use `--anp-srf`.
6. **Fourier alone on AirfRANS: net negative** — 6→2 epochs overhead. Fourier+physics hasn't been tested yet (kohaku running).
7. **Optimizer: Lion for TandemFoil, AdamW for AirfRANS/DrivAerML**
8. **DrivAerML: 1M train surface points >> 50k** — Key hardware finding. Enables much richer gradient signal. Each epoch = ~80 min (1 epoch per 30-min timeout).
9. **DrivAerML T_max=50 is best** for step-level cosine scheduling.

## Student Status

### WIP — TandemFoil (Fourier+physics golden config = NEW STANDARD)
- frieren #2490: **Fourier+physics+no-EMA + T_max sweep (10, 15, 20)** — HIGHEST PRIORITY
- edward #2491: Fourier+physics+no-EMA + 4L/256d capacity
- gilbert #2471: Golden + no-EMA (no Fourier, no physics) — still running
- tanjiro #2485: Golden + no-EMA + Lion lr=2e-4/1e-4
- shinji #2486: Golden + no-EMA + AdamW
- kaneda #2488: Golden + no-EMA + 4L/256d (no Fourier — now superseded by edward)

### WIP — AirfRANS
- kohaku #2492: **Fourier+physics+no-EMA** — new direction from TandemFoil breakthrough
- haku #2470: Fourier + no-EMA full epoch run
- senku #2474: Fourier + no-EMA + 4L/256d
- norman #2460: No-EMA + OOD tasks
- alphonse #2469: No-EMA + cosine T_max sweep

### WIP — DrivAerML
- taki #2493: **1M train + 50k eval + tmx50 + lr=8e-4** — standardized validation
- thorfinn #2487: Slices reduction for more epochs (64, 48, 32)
- shoya #2466: No-EMA retest of AdamW winner
- chihiro #2475: Fourier + no-EMA
- mitsuha #2442: Model capacity sweep
- nezuko #2439: Anchor token budget sweep
- shouko #2437: Surface points budget sweep

## Current Baselines

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **82.65** | #2473 (slices=64, T_max=30, Lion lr=3e-4, no-EMA, Fourier+physics, 14 epochs) |
| AirfRANS | val_primary/surface_mse | **0.2597** | #2455 (3L/192d, no-EMA, no-Fourier, 6 epochs) |
| DrivAerML | val_primary/surface_rel_l2_pct | **56.91%** | #2467 (no-EMA, AdamW lr=8e-4, 2 epochs, 50k pts) |

## Confirmed Dead Ends

| Direction | Result | Reason |
|---|---|---|
| ANP decoder | +5.4% worse | Cross-foil attention harmful |
| Physics at slices=96 | 2 epochs max, ~150-170 val | Overhead at slices=96 blocks |
| Fourier alone on AirfRANS | 2 epochs vs 6 | 3x epoch overhead, net negative |
| Lion on AirfRANS/DrivAerML | Significantly worse | AdamW consistently better |
| EMA (any dataset) | -10% to +29% worse | Short training regime incompatibility |
| 6L deep model | Divergence | Too deep for 30-min regime |
| DrivAerML 50k→baseline | 2 epochs, 56.91% | Low surface point resolution limits learning |

## Current Research Themes

1. **TandemFoil: optimize the Fourier+physics config** — frieren (T_max sweep) and edward (capacity) are the highest priority. 82.65 was still improving at epoch 14. The right T_max and capacity could push below 70.

2. **AirfRANS: test Fourier+physics synergy** — kohaku is testing this. If the same synergy holds (more epochs with Fourier+physics than Fourier alone), this could produce a big AirfRANS breakthrough.

3. **DrivAerML: standardize 1M surface points** — taki's standardized rerun will establish whether 1M pts beats 50k baseline at comparable eval settings. This is critical because it determines the new DrivAerML standard.

4. **DrivAerML: epoch starvation** — thorfinn's slices reduction still pending. Combined with 1M surface points insight, might find a sweet spot.

## Next Round Priorities

### TandemFoil
- After frieren (T_max) and edward (capacity) results: compound best T_max + best capacity + Fourier+physics
- Consider AdamW with Fourier+physics (kohaku/shinji covered AdamW on golden config without Fourier)
- The metric may be approaching ~60-70 range — track whether it's still improving

### AirfRANS
- If kohaku (Fourier+physics) works: immediately scale
- 0.2597 target, external SpiderSolver is 0.0043 (60x off)
- Consider: what physics features are available for AirfRANS? (asinh-pressure, residual-prediction)

### DrivAerML
- Target <3.71% (AB-UPT). Currently 56.91%. 15x gap.
- 1M surface points is the most promising lever
- If taki validates 1M pts: immediately add Fourier, capacity, and T_max tuning on top
