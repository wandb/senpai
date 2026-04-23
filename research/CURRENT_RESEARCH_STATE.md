# SENPAI Research State

- **Date:** 2026-04-23 06:00 (advisor cycle 19 complete — reviewed #3108/#3104/#3103/#3099/#3084/#3079/#3069, merged #3108+#3050, assigned #3137-#3142)
- **Branch:** radford
- **Idle students:** NONE — all 60 students assigned
- **PRs ready for review:** 0
- **PRs in WIP:** 60 (53 continuing + 6 new + 1 sent-back)
- **Merged this cycle:** #3108 (TF gc=0.3 new best), #3050 (AF EMA+T_max=50 new best)
- **Closed this cycle:** #3104 (AF T_max=100), #3103 (AF T_max=30), #3099 (AF 3L/256d), #3084 (DM 788-batches), #3069 (DM 5ep warmup)
- **Sent back this cycle:** #3079 (DM 640d+gc — retry with T_max=60+lr=3e-4)

## Fleet Status

### DrivAerML WIP (~33 students, ~55%)
- `#3063` canute: multi-seed full-eval seed=42 (PAPER-FACING)
- `#3064` casca: multi-seed full-eval seed=123 (PAPER-FACING)
- `#3065` chihiro: multi-seed full-eval seed=456 (PAPER-FACING)
- `#3066` alphonse: 16k surface points
- `#3067` askeladd: 32k surface points
- `#3068` brook: 64k surface points
- `#3072` eren: EMA=0.9995
- `#3073` faye: EMA=0.999 + gc=0.5 compound
- `#3074` fern: relative L2 loss
- `#3075` franky: Huber loss (delta=0.1 and 1.0)
- `#3076` frieren: log-cosh loss
- `#3077` gilbert: 5L/512d (deeper)
- `#3078` gohan: 6L/512d (much deeper) — SENT BACK re-run at max epochs
- `#3079` gojo: 4L/640d + T_max=60 + lr=3e-4 — SENT BACK retry
- `#3083` jet: max-train-batches=600
- `#3085` kohaku: larger supernode budget (8192/16000)
- `#3086` megumi: SGDR T_max=2 (CosineAnnealingWarmRestarts) — SENT BACK
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
- `#3137` chopper: DM warmup+gc=1.0 compound (5-ep warmup + gc=1.0, T_max=30)
- `#3138` kakashi: DM 788 batches + T_max=60 (proportionally scaled cosine)
- `#3141` vegeta: DM 5L/512d + gc=1.0 (depth with stability guard)

### TandemFoil Paper WIP (~14 students, ~23%)
- `#3056` haku: Lion+EMA refinement (T_max/gc/LR sweep)
- `#3088` mugen: T_max=20
- `#3089` nami: T_max=30
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
- `#3135` (pending assign): TFP vol weight 30x

### AirfRANS Volume WIP (~8 students, ~13%)
- `#3100` taki: 3L/384d for volume
- `#3101` tanjiro: volume-loss-weight=3x
- `#3102` thorfinn: volume-loss-weight=10x
- `#3105` violet: EMA=0.999 (at original T_max=10 — baseline)
- `#3106` wolfwood: 2L/384d + gc=0.5 + T_max=50
- `#3139` spike: AF 3L/256d + EMA=0.999 (architecture + EMA compound)

### TandemFoil WIP (~5 students, ~8%)
- `#3050` stark: (MERGED — AirfRANS EMA+T_max=50 champion) → reassigned
- `#3107` yuji: clean test row (360-min, seed=42)
- `#3108` zenitsu: (MERGED — TF gc=0.3) → reassigned as #3142
- `#3136` (assigned): TFP further gc work
- `#3140` usopp: TF gc=0.2 sweep (monotonic trend test — does gc improvement continue?)
- `#3142` zenitsu: TF gc=0.3 + longer budget (480-min)

## Steering Anchors

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **21.909** (#3108 MERGED — gc=0.3) |
| TandemFoil Paper | `val_primary/field_mse` | **0.002383** (#3025 MERGED) |
| AirfRANS | `val_primary/surface_mse` + `vol_mse` | **0.000459 / 0.002777** (#3050 MERGED — EMA+T_max=50) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.997%** (#2898 MERGED) |

## Paper-Facing Snapshot

| Dataset | Metric | Current best | External target | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **23.419** | (internal anchor) | Strong |
| TandemFoil Paper | `test_primary/field_mse` | **NO CLEAN ROW YET** | ~0.10-0.36/task | URGENT |
| AirfRANS | `Surf MSE / Vol MSE` | **0.000459 / 0.002777** | 0.0043 / 0.0017 | Surface ✓✓, Volume 1.63x gap |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **6.244%** (old config) | 3.71% | Main gap |

## Current Research Focus

### Benchmark Sprint Priorities (ICML phase)

1. **DrivAerML closure** — val=3.997%, need test below 5%, ideally toward 3.71%
   - **CLOSED directions:** T_max (only 30 works without gc), depth (4L required), multi-revisit, EMA (3 configs tried — all dead), bilateral symmetry aug, LR (5e-4 sharp optimum), gc alone (best 4.346% still above baseline), gc+WD (crashes), gradient accumulation, 640d without gc, SGDR (2 variants), RAdam, beta2 changes, warmup alone, 6L, torch.compile
   - **Active exploration fronts:**
     - Surface points sweep (16k/32k/64k) — human directive
     - Loss alignment (rel L2, Huber, log-cosh, L1-smooth)
     - Architecture (5L/6L deeper, 640d wider w/ gc, 384d compact)
     - Throughput (600/788+T_max=60 batches, supernode budget)
     - Optimizer (Lion remaining)
     - Regularization (WD alone, dropout)
     - Warmup+gc compound (warmup alone diverges; with gc stability guard may help)
     - 788 batches + T_max=60 (proportionally scaled — kakashi #3138)
     - Multi-seed full-eval (paper-facing)
     - gc=1.0 with 5L/512d compound (vegeta #3141)

2. **TFP clean test result** — val=0.002383, need paper-facing test_primary/field_mse
   - **CLOSED:** T_max>10 (T_max=15/20/30 all diverge), gc>0.5 (diverges), gc<0.5 (gc=0.3 = pressure starvation → Infinity), 4L depth (pressure overflow), LR=1.5e-4 (worse)
   - **Active:** T_max=5/8 (shorter may stabilize), EMA=0.99/0.9995, lr=1e-4, 3L/256d width, clean test eval
   - **Key finding:** TFP champion is a SHARP OPTIMUM — gc=0.5 and T_max=10 are essentially inflection points. LR=1.25e-4 is minimum viable for pressure.

3. **AirfRANS volume** — Vol MSE=0.002777 (1.63x gap from target 0.0017)
   - **New champion:** EMA=0.999 + T_max=50 (both surface AND volume best)
   - **Active:** volume-loss-weight (3x/10x), architecture (3L/384d, 3L/256d+EMA), T_max=50 confirmed optimal
   - **Closed:** T_max=30/100 both worse; 3L/256d without EMA superseded

4. **TandemFoil trend** — gc sweep: 1.0→0.5→0.3 all improve monotonically
   - **Active:** gc=0.2 sweep (does improvement continue?), gc=0.3+480-min budget
   - **Key question:** Is gc=0.1 or gc→0 the floor, or does it invert?

## Key Dead Ends (Do Not Repeat)

**DrivAerML:**
- T_max: only 30 works without gc; 15/20/40/50/100 all diverge
- Depth: 4L required; 2L/3L diverge and 2.5-2.8x worse; 6L also worse
- EMA: incompatible at all tested configs (9.749%)
- Multi-revisit: 4x/8x diverge without gc
- gc+WD compound: crashes
- gc alone: 1.5/2.0 diverges; best gc result 4.346% still above baseline
- 640d without gc: dead end (gojo sent back with gc=1.0 to retry)
- torch.compile: no throughput benefit, diverged
- Bilateral symmetry aug: causes gradient instability (14.01%)
- LR: 4e-4 and 4.5e-4/5.5e-4 all worse; 5e-4 sharp optimum
- Gradient accumulation: harmful (4.860%)
- SGDR (T_mult=1.5 and T_mult=2): both worse than standard cosine
- RAdam: no improvement over AdamW
- beta2=0.99/0.95/0.98: all worse; beta2=0.999 (default) optimal
- Warmup alone (5-ep/10-ep): diverges without gc stability guard

**TandemFoil Paper:**
- T_max>10: T_max=15 diverged ep124, T_max=20 diverged earlier, T_max=30 diverged
- gc=0.7: diverged ep142
- gc=0.3: Infinity all epochs (pressure starvation)
- 4L/192d: pressure overflow
- 4L/256d: 0.004427 (+85% vs baseline)
- LR=1.5e-4: 0.003199 (+34%)

**AirfRANS:**
- 2L/384d NaN
- LR above 6e-4 all worse
- accum>1 harmful
- T_max=30: surface_mse=0.000829 (+72%)
- T_max=100: surface_mse=0.000709 (+47%)
- 3L/256d without EMA: superseded by 2L+EMA champion

**Cross-dataset:** SAM, PCGrad, LayerScale, sigma-Reparam, GeGLU, SwiGLU, SDF, head scaling — all failed

## Strategy (ICML Final Sprint)

Per human team directive #3020:
- NO cross-dataset default — each PR targets one benchmark
- DrivAerML ~50-60% fleet ✓ (33/60 = 55%)
- TFP ~20-30% ✓ (14/60 = 23%)
- AirfRANS ~10-20% ✓ (8/60 = 13%)
- TF minimal ✓ (5/60 = 8%)

## Mandatory Config Rules

- **TF:** Lion lr=1.25e-4, T_max=10, gc=0.3, WD=1e-2, `--ema-decay 0.999`, 3L/192d (**gc updated to 0.3 post #3108**)
- **TFP:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **AF:** AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, `--ema-decay 0.999`, 2L/256d (**EMA now mandatory post #3050**)
- **DM:** AdamW lr=5e-4, T_max=30, NO gc, NO WD, no-EMA, 4L/512d
- `--epochs 999` mandatory
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394`
- Paper-facing DM: NO `--max-eval-batches`
