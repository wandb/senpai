# SENPAI Research State

- **Date:** 2026-04-20 19:40 UTC
- **Branch:** radford

## Student Status

### Idle
- None

### WIP — TandemFoil Round 2 (established this round)
- frieren #2412: clean baseline (no physics, Lion)
- fern #2413: full physics stack (Lion) — nearly done, val≈270.74 from W&B group data
- alphonse #2433: AdamW LR sweep at slices=64 (no physics)
- violet #2434: slices sweep (Lion)
- gilbert #2435: cosine T_max sweep at slices=64 (Lion)
- chihiro #2436: RE-stratified sampling ablation (Lion)

### WIP — TandemFoil Round 3 (just assigned)
- tanjiro #2441: **Physics + AdamW LR sweep** (3e-4, 5e-4, 8e-4, slices=64) — KEY EXPERIMENT
- edward #2443: Physics + AdamW + slices sweep (48, 64, 96)
- thorfinn #2445: Physics + AdamW + cosine T_max sweep (10, 20, 30, 50)
- askeladd #2446: Wake feature ablation (core vs +wake_deficit vs +wake_angle vs full stack)
- shinji #2444: **ANP cross-foil decoder** (with/without physics + AdamW) — HIGH PRIORITY
- rei #2447: Weight decay (1e-5, 1e-4, 1e-3) + EMA ablation
- kaneda #2449: Full physics stack + AdamW LR sweep (3e-4, 5e-4, 8e-4) vs core subset control

### WIP — AirfRANS Round 2
- kohaku #2428: AdamW LR bracket (3e-4, 4e-4, 6e-4, 8e-4)
- emma #2429: AdamW + capacity sweep (4L/256d)
- senku #2430: Cosine T_max sweep (10, 20, 30, 50)
- haku #2431: Scaffold ablation (lookahead/EMA)
- norman #2432: OOD tasks (scarce + reynolds)

### WIP — DrivAerML Round 1 (ALL NEW — just assigned)
- shoya #2440: **AdamW vs Lion baseline** (4 LRs) — FIRST DRIVAERML RUN
- shouko #2437: Surface points budget sweep (0/4k/8k/16k)
- mitsuha #2442: Model capacity sweep (2L/128d, 3L/192d, 4L/256d) + large anchors
- taki #2438: Cosine T_max sweep (10, 20, 30, 50)
- nezuko #2439: Anchor budget sweep (4096/8000 vs 8192/16000 vs 16384/16384)

## Baselines

| Dataset | Metric | Value | PR |
|---|---|---|---|
| TandemFoil | val_primary/surface_pressure_mae | **262.82** | #2414 (tanjiro, physics features + Lion lr=3e-4) |
| AirfRANS | val_primary/surface_mse | **0.3308** | #2423 (kohaku, AdamW lr=5e-4) |
| DrivAerML | val_primary/surface_rel_l2_pct | **no baseline yet** | — target: <3.71% |

## Round 2 Key Findings (TandemFoil, closed this round)

| PR | Config | val_mae | vs baseline | Outcome |
|---|---|---|---|---|
| #2414 | Physics features + Lion | **262.82** | -2.4% | **MERGED** (new best) |
| #2418 | asinh+residual only + Lion | 291.32 | +8.2% | Closed (superseded) |
| #2417 | 4L/256d bigger model + Lion | 314.52 | +17% | Closed (too slow) |
| #2415 | Lion lr=1e-3 | 352.40 | +31% | Closed |
| #2419 | batch_size=4 + Lion | 454.96 | +69% | Closed (too slow) |

W&B group insight (from radford-tandem-round1):
- Fern's full physics (val=270.74) is nearly tied with baseline, only 0.5% worse — may flip with more training
- Frieren's clean baseline (val=310.96) establishes clean floor without physics

## Human Researcher Directives
- Focus on all 3 benchmarks: TandemFoilSet, AirfRANS, DrivAerML
- Tune and ablate the current stack — no new architecture invention
- Use all 8 GPUs per student (2-8 parallel trials when possible)
- DrivAerML is NOW in scope — start training immediately (done ✓)

## Current Research Themes

1. **Physics + AdamW compound hypothesis** — the central TandemFoil question for Round 3. Physics features won. AdamW won on AirfRANS. Their combination should compound. Tanjiro's #2441 tests this directly.

2. **ANP cross-foil decoder** — flagged "high priority" in program.md. Tests cross-foil information sharing. Shinji's #2444 isolates its contribution vs physics+AdamW.

3. **Epoch budget is binding** — 30-min timeout is the dominant constraint. slices reduction (96→64→48) trades resolution for more gradient updates. Being tested in violet's #2434 and edward's #2443.

4. **Cosine schedule alignment** — T_max=150 means essentially constant LR in 2-5 epoch runs. T_max=10-30 enables meaningful annealing. Tested in gilbert's #2435 (tandemfoil) and thorfinn's #2445.

5. **DrivAerML bootstrap** — zero baseline → shoya/shouko/mitsuha/taki/nezuko establishing first results across: optimizer (AdamW vs Lion), LR, point budget, model capacity, anchor tokens, cosine schedule.

6. **AirfRANS gap is large** — we're at 0.3308 surface_mse vs SpiderSolver target of 0.0043. That's 77x worse. Round 2 AirfRANS experiments testing LR, capacity, schedule, scaffold ablation — key question: what specifically makes SpiderSolver so much better?

## Potential Next Directions (Round 4+)

### TandemFoil
- **Physics + AdamW + ANP** (triple combination) — if shinji's ANP result is positive
- **Larger model + AdamW + slices=64** — once capacity is manageable with fewer slices
- **RE-stratified sampling + physics + AdamW** — combine chihiro's result with winners
- **Loss weighting** — downweight early-training-unstable vol_p to avoid Inf issues

### AirfRANS
- **Pressure-weighted loss** — pressure channel dominates MSE by orders of magnitude; explicit weighting
- **Fourier features** (enable_fourier=True) — may help high-frequency pressure prediction
- **SpiderSolver gap diagnosis** — need to understand architecturally what makes SpiderSolver 77x better
- **Larger model + AdamW + slices=64** — once AirfRANS LR and schedule are optimized (Round 2 result)

### DrivAerML
- Once shoya establishes baseline: fan out on best optimizer/LR
- Anchor token count likely critical (nezuko's sweep) — AB-UPT uses 16384 vs our 4096
- Full DrivAerML epoch budget: understand min epochs needed (AB-UPT trains 500 epochs!)
- Point budget subsampling for faster epoch cycling
