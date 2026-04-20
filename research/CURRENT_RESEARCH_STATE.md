# SENPAI Research State

- **Date:** 2026-04-20 23:00 UTC
- **Branch:** radford

## THE KEY FINDINGS

1. **Epoch count is the universal dominant lever** — More epochs = better results across all 3 datasets. Every other finding flows from this.
2. **slices=64 + T_max=30 is the TandemFoil golden config** — 11 epochs vs 2 at slices=96. (-42% improvement)
3. **EMA IS HARMFUL** — `--no-use-ema` mandatory. Confirmed across all datasets.
4. **Physics features are blocked at ~2 epochs** — Per-sample overhead cannot be reduced by slices. Precomputed caching required.
5. **ANP decoder conclusively negative** — Control beats ANP by +5.4%. Never use `--anp-srf`.
6. **Fourier features on AirfRANS: net negative** — Triples epoch time (15 min/epoch), net worse than 6 no-Fourier epochs.
7. **Optimizer is dataset-dependent** — Lion for TandemFoil (no physics), AdamW for AirfRANS and DrivAerML.
8. **AdamW reversal with physics** — Physics + AdamW beats physics + Lion on TandemFoil.
9. **DrivAerML epoch starvation** — Only 2 epochs at slices=96. Slices reduction untested (thorfinn testing now).

## Student Status

### WIP — TandemFoil Golden Config (slices=64, T_max=30, no-EMA)
- gilbert #2471: **Golden + no-EMA + Lion lr=3e-4** — HIGHEST PRIORITY, projected ~86-90
- edward #2473: Golden + Fourier + no-EMA
- tanjiro #2485: Golden + no-EMA + Lion lr=2e-4 and lr=1e-4 (LR sweep)
- shinji #2486: Golden + no-EMA + AdamW (optimizer comparison at 11 epochs)
- kaneda #2488: Golden + no-EMA + 4L/256d model capacity

### WIP — AirfRANS
- haku #2470: Fourier + no-EMA full epoch run
- kohaku #2465: No-EMA + AdamW lr=5e-4/8e-4 baseline
- senku #2474: Fourier + no-EMA + 4L/256d capacity
- norman #2460: No-EMA + OOD tasks (scarce, reynolds)
- alphonse #2469: No-EMA + cosine T_max sweep

### WIP — DrivAerML
- thorfinn #2487: **Slices reduction for more epochs** — HIGHEST DrivAerML PRIORITY
- shoya #2466: No-EMA retest of AdamW winner
- chihiro #2475: Fourier + no-EMA
- mitsuha #2442: Model capacity sweep
- nezuko #2439: Anchor token budget sweep
- taki #2438: Cosine T_max sweep
- shouko #2437: Surface points budget sweep

## Current Baselines

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **114.92** | #2435 (slices=64, T_max=30, Lion lr=3e-4, EMA=True, 11 epochs) |
| AirfRANS | val_primary/surface_mse | **0.2597** | #2455 (3L/192d, no-EMA, no-Fourier, 6 epochs) |
| DrivAerML | val_primary/surface_rel_l2_pct | **56.91%** | #2467 (no-EMA, AdamW lr=8e-4, 2 epochs) |

## Confirmed Dead Ends

| Direction | Result | Reason |
|---|---|---|
| ANP decoder | +5.4% worse | Cross-foil attention mechanism harmful |
| Physics at any slices setting | 2 epochs max | Per-sample overhead, not per-slice |
| Fourier on AirfRANS | Net negative | 3x epoch overhead outweighs per-epoch gain |
| Lion on AirfRANS/DrivAerML | Significantly worse | AdamW consistently better |
| EMA (any dataset) | -10% to +29% worse | Short training regime incompatibility |
| 6L model | Divergence | Too deep for 30-min regime |

## Current Research Themes

1. **TandemFoil golden config variants** — Gilbert's no-EMA retest is the keystone. Once that result comes in, we'll know the new baseline. Tanjiro (lr sweep), shinji (AdamW), and kaneda (capacity) are testing orthogonal directions simultaneously.

2. **DrivAerML epoch starvation** — The 56.91% result is achieved at only 2 epochs. Thorfinn's slices reduction test is the most promising lever. TandemFoil went -42% with slices=64; the same may apply here.

3. **AirfRANS convergence** — Several WIP experiments testing different configurations. No-Fourier 6-epoch baseline (0.2597) is strong. Focus on model capacity and LR tuning.

## Next Round Priorities

### TandemFoil
- After gilbert's no-EMA result: update baseline and compare against tanjiro/shinji/kaneda/edward variants
- Best variants: combine into compound improvements (e.g., best LR + best optimizer + capacity)
- If golden+no-EMA pushes below 80: test T_max variation with no-EMA (T_max=30 was optimal with EMA)

### AirfRANS
- Capacity + long training is the most promising unexplored direction
- AirfRANS external target: 0.0043 (SpiderSolver). Current 0.2597 is 60x off.

### DrivAerML
- If thorfinn's slices reduction works: immediately replicate with best AirfRANS-derived configs
- External target: <3.71%. Current 56.91% is 15x off. Need fundamental improvements.
- Consider: is the model architecture appropriate for 3D aerodynamics? Or do we need a different representation?
