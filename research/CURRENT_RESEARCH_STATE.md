# SENPAI Research State

- **Date:** 2026-04-23 01:30 (advisor cycle 7 — reviewed #3082/#3081/#3051/#3048, all closed, assigned #3117/#3118/#3119/#3120)
- **Branch:** radford
- **Idle students:** NONE — all 60 students assigned
- **PRs ready for review:** 0
- **PRs in WIP:** 58 (54 continuing + 4 new #3117-#3120)
- **Closed this cycle:** #3082 (T_max=40, 12.84%), #3081 (T_max=20, 11.06%), #3051 (4x revisits, 9.38%), #3048 (2L/3L depth, 9.99%)

## Fleet Status

### DrivAerML WIP (~31 students, ~52%)
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
- `#3044` emma: DrivAerML volume training ablation
- `#3046` sukuna: DrivAerML WD+gc compound
- `#3109` guts: DrivAerML lr=4e-4 full-eval (lower-LR neighbor)
- `#3110` einar: DrivAerML full-eval no max-eval-batches baseline
- `#3111` himmel: DrivAerML larger batch + gc
- `#3112` edward: DrivAerML dual-stage LR (5e-4 → 2.5e-4)
- `#3114` griffith: DrivAerML 4L/384d compact architecture
- `#3115` piccolo: DrivAerML label-smooth L1 loss
- `#3117` bulma: DrivAerML Lion optimizer sweep (lr=5e-5, lr=1e-4)
- `#3118` hinata: DrivAerML weight decay alone (WD=1e-4/5e-4/1e-3)
- `#3119` historia: DrivAerML higher LR + linear warmup (lr=7e-4/1e-3 + 20ep warmup)
- `#3120` senku: DrivAerML RAdam optimizer (lr=5e-4, lr=7e-4)

### TandemFoil Paper WIP (~14 students, ~23%)
- `#3056` haku: TFP Lion+EMA refinement (T_max/gc/LR sweep)
- `#3087` mitsuha: T_max=15
- `#3088` mugen: T_max=20
- `#3089` nami: T_max=30
- `#3090` nezuko: gc=0.3
- `#3091` nobara: gc=0.7
- `#3092` norman: EMA=0.9995
- `#3093` rei: EMA=0.99
- `#3094` robin: 4L/192d depth
- `#3096` shinobu: lr=1e-4
- `#3097` shouko: lr=1.5e-4 — CLOSED (dead end, assigned #3113)
- `#3098` shoya: clean test evaluation (paper-facing)
- `#3113` shouko: TFP field_mse full-eval clean test
- `#3116` sanji: TFP 4L/192d champion architecture

### AirfRANS Volume WIP (~8 students, ~13%)
- `#3099` spike: 3L/256d for volume
- `#3100` taki: 3L/384d for volume
- `#3101` tanjiro: volume-loss-weight=3x
- `#3102` thorfinn: volume-loss-weight=10x
- `#3103` usopp: T_max=30
- `#3104` vegeta: T_max=100
- `#3105` violet: EMA=0.999
- `#3106` wolfwood: 2L/384d + gc=0.5 + T_max=50

### TandemFoil Parity WIP (~5 students, ~8%)
- `#3060` levi: DrivAerML bilateral symmetry augmentation
- `#2947` jin: TFP first field_mse baseline (LR sweep)
- `#3050` stark: AirfRANS EMA at T_max=50 champion
- `#3107` yuji: clean test row (360-min, seed=42)
- `#3108` zenitsu: gc=0.3 + EMA=0.999

### Recently Closed This Session
- ~~`#3082`~~ historia: T_max=40, 12.84% (+221% vs baseline), diverged ep94
- ~~`#3081`~~ hinata: T_max=20, 11.06% (+177% vs baseline), diverged ep118
- ~~`#3051`~~ bulma: 4x revisits, 9.38% best (+135%), all runs diverged
- ~~`#3048`~~ senku: 2L/3L depth, 9.99% best (3L), both diverged to NaN
- ~~`#3095`~~ sanji: TFP 4L/256d, 0.004427 (+85% vs TFP baseline)
- ~~`#3047`~~ piccolo: DM LR fine-tuning (4e-4/4.5e-4/5.5e-4), all diverged
- ~~`#3045`~~ griffith: DM T_max sweep (15/50/100), all diverged
- ~~`#3097`~~ shouko: TFP lr=1.5e-4, val 0.003199 (+34% vs TFP baseline)
- ~~`#3071`~~ edward: DM EMA=0.999, diverged to NaN
- ~~`#3080`~~ himmel: DM T_max=50, 14.1% best, diverged

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
   - **T_max sweep CLOSED:** T_max=15/20/40/50/100 all diverge without gc. T_max=30 uniquely stable.
   - **Depth sweep CLOSED (4L is necessary):** 2L (11.26%), 3L (9.99%), 4L (3.997%). Monotonic — cannot go shallower.
   - **Multi-revisit CLOSED:** 4x/8x revisits diverge without gc. 1x full-eval too slow (only 36ep).
   - **EMA CLOSED:** EMA incompatible with DM (dead end #3071)
   - **In progress:** Surface points (16k/32k/64k), loss alignment (rel L2/Huber/log-cosh), warmup (5/10ep), architecture (5L/6L/640d), throughput (600/788 batches), spatial budget, SGDR
   - **New (this cycle):** Lion optimizer (lr=5e-5/1e-4), WD alone (1e-4/5e-4/1e-3), higher LR+warmup (7e-4/1e-3), RAdam (lr=5e-4/7e-4)
   - **Key finding:** DM loss landscape is rough — T_max=30 sits in a uniquely narrow stable basin. Monotonic depth preference (4L > 3L > 2L). 4L is NOT over-parameterized.

2. **TFP clean test result** — val=0.002383, need paper-facing test_primary/field_mse
   - shoya #3098: dedicated clean-test run (highest priority TFP)
   - shouko #3113: TFP field_mse full-eval clean test
   - T_max refinement (15/20/30), gc tuning (0.3/0.7), EMA tuning (0.99/0.9995)
   - Architecture: 4L/192d per vash's finding
   - LR: 1e-4 only (1.5e-4 dead end at 0.003199)

3. **AirfRANS volume** — Vol MSE=0.00764, target 0.0017 (4.5x gap)
   - Volume-weighted loss (3x and 10x)
   - Deeper architectures (3L/256d, 3L/384d)
   - T_max tuning (30/100), EMA, wider model + gc

4. **TandemFoil anchor** — preserve at 22.537 val, 24.581 test

## Key Dead Ends (Do Not Repeat)

**DrivAerML specific:**
- T_max: only 30 works without gc; 15/20/40/50/100 all diverge
- Depth: 4L required; 2L/3L both diverge and achieve 2.5-2.8x worse val
- EMA: incompatible (EMA alone = 9.749%)
- Multi-revisit training: 4x/8x diverge without gc
- gc+WD compound: crashes
- gc alone: 1.5/2.0 diverges (2.0 was best gc at 4.346%, still above baseline)
- 640d without gc: dead end
- torch.compile: no throughput benefit, compile run diverged

**AirfRANS:** 2L/384d NaN, LR above 6e-4 all worse, accum>1 harmful
**TFP:** 3L unstable (diverges/degrades), 5L degrades late, LR>1.25e-4 worse, 4L/256d architecture worse
**Cross-dataset:** SAM, PCGrad, LayerScale, sigma-Reparam, GeGLU, SwiGLU, SDF, head scaling — all failed

## Strategy (ICML Final Sprint)

Per human team directive #3020 and ICML sprint guidance:
- **NO cross-dataset default** — each PR targets one benchmark
- DrivAerML is top priority (~50-60% of fleet) ✓
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
