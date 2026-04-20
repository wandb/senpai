# SENPAI Research State

- **Date:** 2026-04-20 18:50 UTC
- **Branch:** radford

## Student Status

- **Idle:** none
- **WIP (16):**
  - TandemFoil: frieren #2412, fern #2413, tanjiro #2414, nezuko #2415, edward #2417, thorfinn #2418, askeladd #2419 (Round 1, still running)
  - TandemFoil: alphonse #2433, violet #2434, gilbert #2435, chihiro #2436 (Round 2 assigned)
  - AirfRANS: kohaku #2428, emma #2429, senku #2430, haku #2431, norman #2432 (Round 2 assigned)
- **Review-ready:** none

## Baselines (established this session)

| Dataset | Metric | Value | PR |
|---|---|---|---|
| AirfRANS | val_primary/surface_mse | **0.3308** | #2423 (kohaku, AdamW lr=5e-4) |
| TandemFoil | val_primary/surface_pressure_mae | **269.32** | #2416 (alphonse, AdamW lr=5e-4, 2ep) |

## Human Researcher Directives
- Focus on TandemFoilSet and AirfRANS (not DrivAerML for this run)
- Tune and ablate the current stack — no new architecture invention
- Establish strong, defensible recipes on both datasets
- Use all 8 GPUs per student — parallel trials when possible

## Round 1 Key Findings (AirfRANS)

1. **AdamW crushes Lion** — 38% better (0.331 vs 0.538). Optimizer is the dominant lever.
2. **Pressure dominates error** — surface_mse_p (~1.3) is orders of magnitude larger than Ux/Uy/nut
3. **Deeper model (6L) diverges** — depth scaling harmful with current recipe
4. **Bigger model + AdamW promising** (0.379) but slows epoch cycling
5. **Surface refinement head is beneficial** — ablation confirmed
6. **6 epochs is severely training-limited** — all runs still improving at cutoff

## Round 1 Key Findings (TandemFoil)

- Only alphonse's AdamW result (#2416) is in — 2 epochs, strongly improving
- 7 remaining TandemFoil students still WIP (Round 1 in progress)
- TandemFoil training is much slower (~15 min/epoch at slices=96 vs ~4min for AirfRANS)

## Round 2 Assignments (just made)

### AirfRANS (5 students)
| Student | PR | Focus |
|---|---|---|
| kohaku | #2428 | AdamW LR bracket: 3e-4, 4e-4, 6e-4, 8e-4 (all slices=64) |
| emma | #2429 | AdamW + bigger model 4L/256d at slices=64, lr=5e-4 and 8e-4 |
| senku | #2430 | Cosine T_max sweep: 10, 20, 30, 50 (AdamW lr=5e-4, slices=64) |
| haku | #2431 | AdamW scaffold ablation: lookahead/EMA on/off (AdamW lr=5e-4) |
| norman | #2432 | AirfRANS OOD tasks: scarce + reynolds baselines |

### TandemFoil (4 students)
| Student | PR | Focus |
|---|---|---|
| alphonse | #2433 | AdamW LR sweep: 3e-4, 5e-4, 8e-4, 1e-3 (all slices=64) |
| violet | #2434 | Slices sweep: 32, 48, 64, 96 (Lion lr=3e-4) — throughput |
| gilbert | #2435 | Cosine T_max sweep: 10, 20, 30, 50 (Lion lr=3e-4, slices=64) |
| chihiro | #2436 | RE-stratified sampling + EMA ablation (Lion lr=3e-4, slices=64) |

## Current Research Themes

1. **AdamW vs Lion** — AirfRANS: AdamW wins. TandemFoil: unknown (only 2 epochs seen). Round 2 will resolve.
2. **Epoch budget is binding** — reducing slices=96→64 is a free throughput gain being tested on both datasets
3. **Cosine schedule mismatch** — T_max=150 on 5-10 epoch runs means LR never anneals. Testing T_max=10–50.
4. **Pressure as bottleneck** — surface_mse_p dominates AirfRANS error. No direct fix yet (future: loss weighting)
5. **OOD generalization** — `val_re_rand` on TandemFoil and `scarce`/`reynolds` on AirfRANS are key paper metrics

## Potential Next Directions (Round 3+)

- **AirfRANS:** Weight decay sweep (once Round 2 scaffold ablation result is in)
- **AirfRANS:** Pressure-weighted loss (once we confirm AdamW LR sweet spot)
- **AirfRANS:** Fourier features (enable_fourier=True)
- **TandemFoil:** ANP surface decoder (anp_srf=True) — high priority from program.md
- **TandemFoil:** Physics feature stack (waiting for frieren/fern/tanjiro Round 1 results)
- **TandemFoil:** Larger AdamW model with slices=64
- **Cross-dataset:** Test same best recipe on DrivAerML once TandemFoil+AirfRANS recipes converge
