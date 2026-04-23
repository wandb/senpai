# SENPAI Research State

- **Date:** 2026-04-23 06:30 (advisor cycle 20 complete — reviewed 14 PRs: merged #2974 infra, 9 closed, 5 sent-back, 8 new assigned #3143-#3150)
- **Branch:** radford
- **Idle students:** NONE — all 60 students assigned
- **PRs ready for review:** 0
- **PRs in WIP:** 60

## Fleet Status

### DrivAerML WIP (~35 students, ~58%)
- `#3063` canute: paper-facing full-eval (SENT BACK — two-phase approach)
- `#3064` casca: multi-seed full-eval seed=123 (PAPER-FACING)
- `#3065` chihiro: multi-seed full-eval seed=456 (PAPER-FACING)
- `#3066` alphonse: 16k surface points
- `#3067` askeladd: 32k surface points
- `#3068` brook: 64k surface points
- `#3072` eren: EMA=0.9995
- `#3073` faye: EMA=0.999 + gc=0.5 compound — SENT BACK
- `#3077` gilbert: 5L/512d (deeper)
- `#3078` gohan: 6L/512d (much deeper) — SENT BACK
- `#3079` gojo: 4L/640d + T_max=60 + lr=3e-4 — SENT BACK retry
- `#3083` jet: max-train-batches=600
- `#3085` kohaku: larger supernodes (SENT BACK — retry with gc=1.0)
- `#3076` frieren: log-cosh loss (SENT BACK — retry with gc=1.0)
- `#3046` sukuna: WD+gc compound (SENT BACK — lighter WD=5e-4/1e-4)
- `#3044` emma: volume training ablation
- `#3109` guts: lr=4e-4 full-eval
- `#3110` einar: AdamW beta2=0.99/0.995
- `#3111` himmel: cosine eta_min=1e-6/1e-5
- `#3112` edward: gradient centralization
- `#3113` shouko: attention dropout=0.05
- `#3114` griffith: T_max=50 + gc=1.0
- `#3115` piccolo: batch_size=2 with 25k points
- `#3117` bulma: Lion optimizer sweep
- `#3118` hinata: weight decay alone
- `#3119` historia: higher LR + 20-epoch warmup
- `#3121` levi: dropout regularization
- `#3122` nobara: polynomial LR decay
- `#3125` faye: SWA at cosine troughs
- `#3127` senku: attention heads sweep (4H/16H)
- `#3129` chrome: AF 4L/256d deeper — wait, this is DM? Check: chrome/af-4l-256d — actually chrome is on AF per CURRENT STATE
- `#3131` sanji: OneCycleLR schedule
- `#3132` gohan: eta_min=5e-5 + gc=1.0
- `#3137` chopper: 5-epoch warmup + gc=1.0
- `#3138` kakashi: 788 batches + T_max=60
- `#3141` vegeta: 5L/512d + gc=1.0
- `#3143` thorfinn: Lookahead(AdamW, k=6, alpha=0.5)
- `#3146` taki: top-5 checkpoint averaging
- `#3147` norman: lighter WD + gc (WD=5e-4/1e-4 + gc=1.0)
- `#3148` franky: 10-epoch warmup + gc=1.0
- `#3149` fern: stochastic input feature dropout (p=0.05/0.10)

### TandemFoil Paper WIP (~9 students, ~15%)
- `#3056` haku: Lion+EMA refinement (T_max/gc/LR sweep)
- `#3088` mugen: T_max=20
- `#3089` nami: T_max=30
- `#3098` shoya: clean test evaluation (paper-facing)
- `#2947` jin: first field_mse baseline (LR sweep)
- `#3113` shouko: field_mse full-eval clean test — wait, this is DM per branch name. Reviewing WIP list...
- `#3116` sanji: 4L/192d champion architecture — wait, sanji is on DM (#3131). This might be stale.
- `#3123` mitsuha: shorter T_max (T_max=5, T_max=8)
- `#3124` robin: 3L/256d wider model
- `#3133` shinobu: TFP WD sweep
- `#3135` nezuko: AF EMA=0.999 + vol-weight=10x
- `#2949` vash: depth/width sweep (LR=5e-5)
- `#3145` rei: gc=0.4 boundary test

### AirfRANS WIP (~9 students, ~15%)
- `#3101` tanjiro: vol-loss-weight=1.5 (SENT BACK — retry at 1.5x)
- `#3102` thorfinn: 10x vol-weight — CLOSED; thorfinn now on DM #3143
- `#3106` wolfwood: 2L/384d + gc=0.5 + T_max=50
- `#3129` chrome: 4L/256d deeper architecture
- `#3134` megumi: vol-weight=30x
- `#3135` nezuko: EMA=0.999 + vol-weight=10x
- `#3136` stark: EMA decay sweep (0.9995/0.9999)
- `#3139` spike: 3L/256d + EMA=0.999
- `#3144` violet: vol-weight=2.0 + EMA=0.999

### TandemFoil WIP (~4 students, ~7%)
- `#3107` yuji: CLOSED (old config); reassigned to #3150
- `#3140` usopp: gc=0.2 sweep
- `#3142` zenitsu: gc=0.3 + longer budget (480-min)
- `#3150` yuji: clean test row (gc=0.3 champion, paper-facing)

## Steering Anchors

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **21.909** (#3108 MERGED — gc=0.3+EMA=0.999) |
| TandemFoil Paper | `val_primary/field_mse` | **0.002383** (#3025 MERGED) |
| AirfRANS | `val_primary/surface_mse` + `vol_mse` | **0.000459 / 0.002777** (#3050 MERGED — EMA+T_max=50) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.997%** (#2898 MERGED) |

## Paper-Facing Snapshot

| Dataset | Metric | Current best | External target | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **23.419** (PR #3108) | (internal anchor) | Strong — clean test in-progress (#3150) |
| TandemFoil Paper | `test_primary/field_mse` | **NO CLEAN ROW YET** | ~0.10-0.36/task | URGENT — #3098 in-progress |
| AirfRANS | `Surf MSE / Vol MSE` | **0.000459 / 0.002777** | 0.0043 / 0.0017 | Surface ✓✓, Volume 1.63x gap |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **6.244%** (old config) | 3.71% | Main gap — paper eval in-progress |

## Current Research Focus

### Benchmark Sprint Priorities (ICML phase)

1. **DrivAerML closure** — val=3.997%, need test below 5%, ideally toward 3.71%
   - **CLOSED directions:** T_max (only 30 without gc), depth (4L required), multi-revisit, EMA (3 configs), bilateral symmetry, LR (5e-4 sharp), gc alone (4.346% best), gc+WD (WD=1e-3+gc best 4.44%), gradient accumulation, SGDR, RAdam, beta2 changes, warmup alone, 6L, torch.compile, Huber loss, relative L2 loss
   - **Active frontier:**
     - Surface points sweep (16k/32k/64k) — human directive
     - Loss variants: log-cosh+gc (frieren sent back), label-smooth L1 (piccolo)
     - Architecture: 5L/512d (gilbert), 640d+gc (gojo), 384d compact (griffith), 5L+gc (vegeta)
     - Throughput: 600-batches (jet), 788+T_max=60 (kakashi), supernodes+gc (kohaku sent back)
     - Regularization: WD alone (hinata), dropout (levi), lighter WD+gc (norman), input dropout (fern)
     - Optimizer: Lion (bulma), Lookahead (thorfinn), SWA (faye), OneCycleLR (sanji)
     - Schedule: eta_min sweep (himmel), polynomial (nobara), eta_min+gc (gohan), 5/10-ep warmup+gc (chopper/franky)
     - Special: checkpoint averaging (taki), gradient centralization (edward), GradCentralization (edward), attention heads (senku), attention dropout (shouko)
     - Compound: T_max=50+gc (griffith), WD lighter+gc (sukuna/norman)
     - Multi-seed paper-facing (casca seed=123, chihiro seed=456)

2. **TFP clean test result** — val=0.002383, need paper-facing field_mse
   - **CLOSED:** T_max>10, gc≠0.5 (both directions fail — 0.3 starves, 0.7 destabilizes), EMA≠0.999 (both directions cause sinh overflow), 4L depth, LR=1.5e-4
   - **Active:** T_max=5/8 (mitsuha), 3L/256d width (robin), WD sweep (shinobu), T_max=20/30 (mugen/nami), gc=0.4 boundary test (rei), clean test (shoya)
   - **TFP is a SHARP OPTIMUM** — gc=0.5, T_max=10, EMA=0.999, LR=1.25e-4, 3L/192d all appear to be inflection points

3. **AirfRANS volume** — 1.63x gap to target 0.0017
   - **New lead:** #3101 showed vol-weight=3x gives surface -17% (0.000381!) but volume +41%. Sweep in progress: 1.5x (tanjiro sent back), 2.0x (violet), 30x (megumi)
   - **Active:** EMA decay higher (stark), 3L/256d+EMA (spike), 2L/384d+gc (wolfwood), 4L/256d (chrome), vol+EMA compound (nezuko)

4. **TandemFoil** — gc trend confirmed monotonically improving (1.0→0.5→0.3)
   - **Active:** gc=0.2 (usopp), gc=0.3+480min (zenitsu), clean test gc=0.3 (yuji paper-facing)
   - **Key question:** Does gc=0.2 continue the trend or invert?

## Key Dead Ends (Do Not Repeat)

**DrivAerML:**
- T_max: only 30 works without gc
- Depth: 4L required; 2L/3L diverge; 6L also worse
- EMA: incompatible at all configs
- gc+WD (WD=1e-3): 4.44% — promising but above baseline; lighter WD in-progress
- gc alone: best 4.346% above baseline
- 640d without gc, torch.compile, bilateral symmetry, gradient accumulation
- LR: 5e-4 sharp optimum
- SGDR (T_mult=1.5, T_mult=2), RAdam, beta2≠0.999
- Warmup alone (5/10-ep), Huber loss (both deltas), relative L2 loss (degenerate)

**TandemFoil Paper:**
- T_max>10 (15/20/30 all diverge)
- gc=0.3: pressure starvation → Infinity
- gc=0.7: destabilizes
- EMA=0.99 and EMA=0.9995: sinh overflow from both directions
- 4L/192d: pressure overflow; 4L/256d: +85% worse
- LR=1.5e-4: +34% worse

**AirfRANS:**
- 2L/384d, 3L/384d: both diverge catastrophically
- T_max=30 (+72%), T_max=100 (+47%): T_max=50 optimal
- 3L/256d without EMA: superseded
- EMA=0.99 (+383%), EMA=0.995 (+51%): EMA=0.999 is sweet spot
- Vol-weight=10x: worse on both metrics vs EMA champion

**Cross-dataset:** SAM, PCGrad, LayerScale, sigma-Reparam, GeGLU, SwiGLU, SDF, head scaling — all failed

## Infrastructure
- **PR #2974 MERGED:** Best-checkpoint saving in train.py — all future experiments automatically use best val checkpoint for test eval. Critical: rescued AirfRANS from total NaN failure, 3.4x improvement on DM test metrics.

## Mandatory Config Rules

- **TF:** Lion lr=1.25e-4, T_max=10, gc=0.3, WD=1e-2, `--ema-decay 0.999`, 3L/192d (**gc=0.3 post #3108**)
- **TFP:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **AF:** AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, `--ema-decay 0.999`, 2L/256d (**EMA now mandatory post #3050**)
- **DM:** AdamW lr=5e-4, T_max=30, NO gc, NO WD, no-EMA, 4L/512d
- `--epochs 999` mandatory
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394`
- Paper-facing DM: NO `--max-eval-batches`
