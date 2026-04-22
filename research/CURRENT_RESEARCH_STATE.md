# SENPAI Research State

- **Date:** 2026-04-22 23:40 (advisor cycle 4 — reviewed PR #2948, closed, assigned #3109)
- **Branch:** radford
- **Idle students:** NONE — all 60 students assigned
- **PRs ready for review:** 0
- **PRs in WIP:** 58 (11 existing + 47 new including #3109)
- **Closed this cycle:** #2948 (guts TFP physics ablation — AdamW 0.003564 val, 49% above baseline, all ablation runs crashed)

## Fleet Status

### Existing WIP (from previous cycle)
- `#3060` levi: DrivAerML bilateral symmetry augmentation
- `#3056` haku: TFP Lion+EMA refinement (T_max/gc/LR sweep)
- `#3051` bulma: DrivAerML 4x case revisits (max-train-batches=1576)
- `#3050` stark: AirfRANS EMA at T_max=50 champion
- `#3048` senku: DrivAerML depth reduction 2L/3L vs 4L/512d
- `#3047` piccolo: DrivAerML LR fine-tuning (4e-4, 4.5e-4, 5.5e-4)
- `#3046` sukuna: DrivAerML WD+gc compound
- `#3045` griffith: DrivAerML T_max cosine period sweep
- `#3044` emma: DrivAerML volume training ablation
- `#3043` einar: DrivAerML gradient accumulation + full-eval baseline
- ~~`#2948` guts: TFP physics-flag ablation~~ CLOSED (0.003564, 49% above baseline)
- `#3109` guts: DrivAerML lr=4e-4 full-eval (lower-LR neighbor)
- `#2947` jin: TFP first field_mse baseline (LR sweep)

### Sent Back This Cycle
- `#2949` vash: TFP depth/width sweep — 4L/192d promising but 25% above baseline; sent back with LR=5e-5 + early stopping instructions

### New Wave (2026-04-22 23:30) — 46 PRs assigned

**DrivAerML (24 students — 52%):**
- `#3063` canute: multi-seed full-eval seed=42 (PAPER-FACING)
- `#3064` casca: multi-seed full-eval seed=123 (PAPER-FACING)
- `#3065` chihiro: multi-seed full-eval seed=456 (PAPER-FACING)
- `#3066` alphonse: 16k surface points
- `#3067` askeladd: 32k surface points
- `#3068` brook: 64k surface points
- `#3069` chopper: 5-epoch linear warmup
- `#3070` chrome: 10-epoch linear warmup
- `#3071` edward: EMA=0.999 at champion
- `#3072` eren: EMA=0.9995
- `#3073` faye: EMA=0.999 + gc=0.5 compound
- `#3074` fern: relative L2 loss
- `#3075` franky: Huber loss (delta=0.1 and 1.0)
- `#3076` frieren: log-cosh loss
- `#3077` gilbert: 5L/512d (deeper)
- `#3078` gohan: 6L/512d (much deeper)
- `#3079` gojo: 4L/640d + gc=0.5 (wider)
- `#3080` himmel: T_max=50
- `#3081` hinata: T_max=20
- `#3082` historia: T_max=40
- `#3083` jet: max-train-batches=600
- `#3084` kakashi: max-train-batches=788
- `#3085` kohaku: larger supernode budget (8192/16000)
- `#3086` megumi: SGDR T_mult=2 (CosineAnnealingWarmRestarts)

**TandemFoil Paper (12 students — 26%):**
- `#3087` mitsuha: T_max=15
- `#3088` mugen: T_max=20
- `#3089` nami: T_max=30
- `#3090` nezuko: gc=0.3
- `#3091` nobara: gc=0.7
- `#3092` norman: EMA=0.9995
- `#3093` rei: EMA=0.99
- `#3094` robin: 4L/192d depth
- `#3095` sanji: 4L/256d depth+width
- `#3096` shinobu: lr=1e-4
- `#3097` shouko: lr=1.5e-4
- `#3098` shoya: clean test evaluation (paper-facing)

**AirfRANS Volume (8 students — 17%):**
- `#3099` spike: 3L/256d for volume
- `#3100` taki: 3L/384d for volume
- `#3101` tanjiro: volume-loss-weight=3x
- `#3102` thorfinn: volume-loss-weight=10x
- `#3103` usopp: T_max=30
- `#3104` vegeta: T_max=100
- `#3105` violet: EMA=0.999
- `#3106` wolfwood: 2L/384d + gc=0.5 + T_max=50

**TandemFoil Parity (2 students — 4%):**
- `#3107` yuji: clean test row (360-min, seed=42)
- `#3108` zenitsu: gc=0.3 + EMA=0.999

## Steering Anchors

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **22.537** (#2924 MERGED) |
| TandemFoil Paper | `val_primary/field_mse` | **0.002383** (#3025 MERGED) |
| AirfRANS | `val_primary/surface_mse` | **0.000482** (#2951 MERGED) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.997%** (#2898 MERGED) |

## Paper-Facing Snapshot

| Dataset | Metric | Current best | External target | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **24.581** | (internal anchor) | Strong |
| TandemFoil Paper | `test_primary/field_mse` | **NO CLEAN ROW YET** | ~0.10-0.36/task | URGENT |
| AirfRANS | `Surf MSE / Vol MSE` | **0.003 / 0.00764** | 0.0043 / 0.0017 | Surface ✓, Volume urgent |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **6.244%** (old config) | 3.71% | Main gap |

## Current Research Focus

### Benchmark Sprint Priorities (ICML phase)

1. **DrivAerML closure** — val=3.997%, need to push test below 5%, ideally toward 3.71%
   - Multi-seed full-eval runs establishing true test baseline
   - Surface points sweep (16k/32k/64k) — directive from human team
   - Loss alignment (relative L2, Huber, log-cosh)
   - Architecture deepening (5L, 6L)
   - Schedule refinement (T_max=20/40/50)
   - EMA exploration at champion config

2. **TFP clean test result** — val=0.002383, need paper-facing test_primary/field_mse
   - shoya #3098: dedicated clean-test run (highest priority TFP)
   - T_max refinement for stable long training (15/20/30)
   - gc and EMA tuning
   - Architecture: 4L/192d per vash's finding
   - LR sweep (1e-4, 1.5e-4)

3. **AirfRANS volume** — Vol MSE=0.00764, target 0.0017 (4.5x gap)
   - Volume-weighted loss (3x and 10x)
   - Deeper architectures (3L/256d, 3L/384d)
   - T_max tuning

4. **TandemFoil anchor** — preserve at 22.537 val, 24.581 test

## Strategy (ICML Final Sprint)

Per human team directive #3020 and ICML sprint guidance:
- **NO cross-dataset default** — each PR targets one benchmark
- DrivAerML is top priority (~50-60% of fleet)
- TFP clean test result is #2 priority
- AirfRANS volume is #3 (surface already excellent)
- Do not let TF absorb large fleet share

## Mandatory Config Rules

- **TF:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **TFP:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **AF:** AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, no-EMA, 2L/256d
- **DM:** AdamW lr=5e-4, T_max=30, NO gc, NO WD, no-EMA, 4L/512d
- `--epochs 999` mandatory
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394`
- Paper-facing DM: NO `--max-eval-batches`

## Known Dead Ends (Do Not Repeat)

DrivAerML: gc+WD compound crashes, T_max=5 diverges, gc=1.5/2.0 diverges, 640d without gc, EMA alone at 9.749%, torch.compile no benefit
AirfRANS: 2L/384d NaN ep134, LR above 6e-4 all worse, accum>1 harmful
TFP: 3L unstable (diverges/degrades), 5L degrades late, all negative novel architectures
Cross-dataset: SAM, PCGrad, LayerScale, sigma-Reparam, GeGLU, SwiGLU, SDF, head scaling — all failed
