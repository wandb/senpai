# SENPAI Research State

- **Date:** 2026-04-23 01:55 (advisor cycle 8 — reviewed #3094/#3091/#3087/#3060, all closed, assigned #3121/#3122/#3123/#3124)
- **Branch:** radford
- **Idle students:** NONE — all 60 students assigned
- **PRs ready for review:** 0
- **PRs in WIP:** 58 (54 continuing + 4 new)
- **Closed this cycle:** #3094 (TFP 4L/192d, Inf), #3091 (TFP gc=0.7, Inf), #3087 (TFP T_max=15, Inf), #3060 (DM symmetry aug, 14.01%)

## Fleet Status

### DrivAerML WIP (~33 students, ~55%)
- `#3063` canute: multi-seed full-eval seed=42 (PAPER-FACING)
- `#3064` casca: multi-seed full-eval seed=123 (PAPER-FACING)
- `#3065` chihiro: multi-seed full-eval seed=456 (PAPER-FACING)
- `#3066` alphonse: 16k surface points
- `#3067` askeladd: 32k surface points
- `#3068` brook: 64k surface points
- `#3069` chopper: 5-epoch linear warmup
- `#3070` chrome: 10-epoch linear warmup
- `#3072` eren: EMA=0.9995
- `#3073` faye: EMA=0.999 + gc=0.5 compound
- `#3074` fern: relative L2 loss
- `#3075` franky: Huber loss (delta=0.1 and 1.0)
- `#3076` frieren: log-cosh loss
- `#3077` gilbert: 5L/512d (deeper)
- `#3078` gohan: 6L/512d (much deeper)
- `#3079` gojo: 4L/640d + gc=0.5 (wider)
- `#3083` jet: max-train-batches=600
- `#3084` kakashi: max-train-batches=788
- `#3085` kohaku: larger supernode budget (8192/16000)
- `#3086` megumi: SGDR T_mult=2 (CosineAnnealingWarmRestarts)
- `#3044` emma: volume training ablation
- `#3046` sukuna: WD+gc compound
- `#3109` guts: lr=4e-4 full-eval
- `#3110` einar: full-eval no max-eval-batches baseline
- `#3111` himmel: larger batch + gc
- `#3112` edward: dual-stage LR (5e-4 → 2.5e-4)
- `#3114` griffith: 4L/384d compact architecture
- `#3115` piccolo: label-smooth L1 loss
- `#3117` bulma: Lion optimizer sweep (lr=5e-5, lr=1e-4)
- `#3118` hinata: weight decay alone (WD=1e-4/5e-4/1e-3)
- `#3119` historia: higher LR + 20-epoch warmup (lr=7e-4/1e-3)
- `#3120` senku: RAdam optimizer (lr=5e-4, lr=7e-4)
- `#3121` levi: dropout regularization (dropout=0.05/0.1)
- `#3122` nobara: polynomial LR decay (no cosine — linear/quadratic)

### TandemFoil Paper WIP (~14 students, ~23%)
- `#3056` haku: Lion+EMA refinement (T_max/gc/LR sweep)
- `#3088` mugen: T_max=20
- `#3089` nami: T_max=30
- `#3090` nezuko: gc=0.3
- `#3092` norman: EMA=0.9995
- `#3093` rei: EMA=0.99
- `#3096` shinobu: lr=1e-4
- `#3098` shoya: clean test evaluation (paper-facing)
- `#2947` jin: first field_mse baseline (LR sweep)
- `#3113` shouko: field_mse full-eval clean test
- `#3116` sanji: 4L/192d champion architecture
- `#2949` vash: depth/width sweep (sent back with LR=5e-5)
- `#3123` mitsuha: shorter T_max (T_max=5, T_max=8)
- `#3124` robin: 3L/256d wider model

### AirfRANS Volume WIP (~8 students, ~13%)
- `#3099` spike: 3L/256d for volume
- `#3100` taki: 3L/384d for volume
- `#3101` tanjiro: volume-loss-weight=3x
- `#3102` thorfinn: volume-loss-weight=10x
- `#3103` usopp: T_max=30
- `#3104` vegeta: T_max=100
- `#3105` violet: EMA=0.999
- `#3106` wolfwood: 2L/384d + gc=0.5 + T_max=50

### TandemFoil Parity WIP (~3 students, ~5%)
- `#3050` stark: AirfRANS EMA at T_max=50 champion
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

1. **DrivAerML closure** — val=3.997%, need test below 5%, ideally toward 3.71%
   - **CLOSED directions:** T_max (only 30 works), depth (4L required), multi-revisit, EMA, bilateral symmetry, LR (only 5e-4 at AdamW)
   - **Active exploration fronts:**
     - Surface points sweep (16k/32k/64k) — human directive
     - Loss alignment (rel L2, Huber, log-cosh, L1-smooth)
     - Architecture (5L/6L deeper, 640d wider, 384d compact)
     - Warmup (5/10/20-epoch linear)
     - Throughput (600/788 batches, supernode budget)
     - Schedule (SGDR warm restarts, polynomial decay)
     - Optimizer (Lion, RAdam)
     - Regularization (WD alone, dropout)
     - Multi-seed full-eval (paper-facing)

2. **TFP clean test result** — val=0.002383, need paper-facing test_primary/field_mse
   - **CLOSED:** T_max>10 (T_max=15 diverged at ep124), gc>0.5 (gc=0.7 diverged at ep142), 4L depth (Inf field_mse), LR=1.5e-4 (worse)
   - **Active:** T_max=5/8 (shorter may stabilize), gc=0.3, EMA=0.99/0.9995, lr=1e-4, 3L/256d width, clean test eval
   - **Key finding:** TFP champion (Lion+T_max=10+gc=0.5+EMA) is a SHARP optimum — deviations in T_max, gc, depth all cause Infinity field_mse

3. **AirfRANS volume** — Vol MSE=0.00764, target 0.0017 (4.5x gap)
   - 8 students covering: volume-loss-weight, architecture, T_max, EMA, compound config

4. **TandemFoil anchor** — preserve at 22.537 val, 24.581 test

## Key Dead Ends (Do Not Repeat)

**DrivAerML:**
- T_max: only 30 works without gc; 15/20/40/50/100 all diverge
- Depth: 4L required; 2L/3L diverge and 2.5-2.8x worse
- EMA: incompatible (9.749%)
- Multi-revisit: 4x/8x diverge without gc
- gc+WD compound: crashes
- gc alone: 1.5/2.0 diverges; best gc result 4.346% still above baseline
- 640d without gc: dead end
- torch.compile: no throughput benefit, diverged
- Bilateral symmetry: aug causes gradient instability (14.01%)
- LR: 4e-4 and 4.5e-4/5.5e-4 all worse; 5e-4 sharp optimum
- Gradient accumulation: harmful (4.860%)

**TandemFoil Paper:**
- T_max=15: diverged ep124 (3.5x earlier than T_max=10 champion)
- gc=0.7: diverged ep142 (3x earlier)
- 4L/192d: Infinity field_mse (pressure overflow)
- 4L/256d: 0.004427 (+85% vs baseline)
- LR=1.5e-4: 0.003199 (+34%)
- 5L degrades late, all negative novel architectures

**AirfRANS:** 2L/384d NaN, LR above 6e-4 all worse, accum>1 harmful
**Cross-dataset:** SAM, PCGrad, LayerScale, sigma-Reparam, GeGLU, SwiGLU, SDF, head scaling — all failed

## Strategy (ICML Final Sprint)

Per human team directive #3020:
- NO cross-dataset default — each PR targets one benchmark
- DrivAerML ~50-60% fleet ✓ (33/60 = 55%)
- TFP ~20-30% ✓ (14/60 = 23%)
- AirfRANS ~10-20% ✓ (8/60 = 13%)
- TF minimal ✓ (3/60 = 5%)

## Mandatory Config Rules

- **TF:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **TFP:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **AF:** AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, no-EMA, 2L/256d
- **DM:** AdamW lr=5e-4, T_max=30, NO gc, NO WD, no-EMA, 4L/512d
- `--epochs 999` mandatory
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394`
- Paper-facing DM: NO `--max-eval-batches`
