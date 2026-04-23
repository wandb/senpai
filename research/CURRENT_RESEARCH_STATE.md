# SENPAI Research State

- **Date:** 2026-04-23 07:00 (advisor cycle 21 — reviewed 7 PRs: merged #3072 DM new best, closed #3077, 5 sent-back, 2 new assigned #3151/#3152)
- **Branch:** radford
- **Idle students:** NONE — all 60 students assigned
- **PRs ready for review:** 0

## Fleet Status

### DrivAerML WIP (~38 students, ~63%)
- `#3044` emma: volume training ablation
- `#3046` sukuna: WD+gc (SENT BACK — try WD=5e-4/1e-4 + gc=1.0)
- `#3063` canute: paper-facing full-eval (SENT BACK — two-phase approach)
- `#3064` casca: multi-seed seed=123 (SENT BACK — two-phase eval)
- `#3065` chihiro: multi-seed seed=456 (SENT BACK — two-phase eval)
- `#3066` alphonse: 16k surface points
- `#3067` askeladd: 32k surface points (SENT BACK — add max-eval-batches 200)
- `#3068` brook: 64k surface points (SENT BACK — add max-eval-batches 200)
- `#3073` faye: SWA (formerly EMA+gc compound — reassigned)
- `#3076` frieren: log-cosh loss (SENT BACK — retry +gc=1.0)
- `#3079` gojo: 4L/640d + T_max=60 + lr=3e-4 — retry
- `#3083` jet: max-train-batches=600
- `#3085` kohaku: larger supernodes (SENT BACK — retry +gc=1.0)
- `#3109` guts: lr=4e-4 full-eval
- `#3110` einar: AdamW beta2=0.99/0.995
- `#3111` himmel: cosine eta_min sweep
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
- `#3131` sanji: OneCycleLR schedule
- `#3132` gohan: eta_min=5e-5 + gc=1.0
- `#3137` chopper: 5-epoch warmup + gc=1.0
- `#3138` kakashi: 788 batches + T_max=60
- `#3141` vegeta: 5L/512d + gc=1.0
- `#3143` thorfinn: Lookahead(AdamW)
- `#3146` taki: top-5 checkpoint averaging
- `#3147` norman: lighter WD + gc (WD=5e-4/1e-4)
- `#3148` franky: 10-epoch warmup + gc=1.0
- `#3149` fern: stochastic input feature dropout
- `#3151` gilbert: EMA=0.9995 + gc sweep (gc=0.25, gc=1.0)
- `#3152` eren: EMA decay sweep (0.999 vs 0.9995) + extended budget

### TandemFoil Paper WIP (~9 students, ~15%)
- `#2947` jin: first field_mse baseline (LR sweep)
- `#2949` vash: depth/width sweep (LR=5e-5)
- `#3056` haku: Lion+EMA refinement (T_max/gc/LR sweep)
- `#3088` mugen: T_max=20
- `#3089` nami: T_max=30
- `#3098` shoya: clean test evaluation (paper-facing) — URGENT
- `#3123` mitsuha: shorter T_max (T_max=5, T_max=8)
- `#3124` robin: 3L/256d wider model
- `#3133` shinobu: TFP WD sweep
- `#3145` rei: gc=0.4 boundary test

### AirfRANS WIP (~9 students, ~15%)
- `#3101` tanjiro: vol-loss-weight=1.5 (SENT BACK)
- `#3106` wolfwood: 2L/384d+gc+T_max=50 (SENT BACK — add EMA=0.999)
- `#3129` chrome: 4L/256d deeper architecture
- `#3134` megumi: vol-weight=30x
- `#3135` nezuko: EMA=0.999 + vol-weight=10x
- `#3136` stark: EMA decay higher (0.9995/0.9999)
- `#3139` spike: 3L/256d + EMA=0.999
- `#3144` violet: vol-weight=2.0 + EMA=0.999

### TandemFoil WIP (~4 students, ~7%)
- `#3107` yuji: CLOSED (old config) → reassigned #3150
- `#3140` usopp: gc=0.2 sweep
- `#3142` zenitsu: gc=0.3 + longer budget (480-min)
- `#3150` yuji: clean test row (gc=0.3 champion, paper-facing)

## Steering Anchors

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **21.909** (#3108 — gc=0.3+EMA=0.999) |
| TandemFoil Paper | `val_primary/field_mse` | **0.002383** (#3025) |
| AirfRANS | `val_primary/surface_mse` + `vol_mse` | **0.000459 / 0.002777** (#3050 — EMA+T_max=50) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.833%** (#3072 MERGED — EMA=0.9995+gc=0.5) |

## Paper-Facing Snapshot

| Dataset | Metric | Current best | External target | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **23.419** (PR #3108) | (internal) | Strong |
| TandemFoil Paper | `test_primary/field_mse` | **NO CLEAN ROW YET** | ~0.10-0.36/task | URGENT — #3098 in-progress |
| AirfRANS | `Surf MSE / Vol MSE` | **0.000459 / 0.002777** | 0.0043 / 0.0017 | Surface ✓✓, Volume 1.63x gap |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **4.685%** (#3072, partial eval) | 3.71% | Gap closing — paper eval in-progress |

## Current Research Focus

### Benchmark Sprint Priorities (ICML phase)

1. **DrivAerML** — val=3.833%, gap to AB-UPT only 0.013 pp!
   - **BREAKTHROUGH:** EMA=0.9995+gc=0.5 works. gc is the stability enabler for EMA on DM. Run still converging at ep517.
   - **Active EMA follow-ups:** decay sweep 0.999 vs 0.9995 (eren #3152), gc sweep gc=0.25/1.0 (gilbert #3151)
   - **CLOSED:** 5L depth, EMA without gc (diverges), Huber loss, rel-L2 loss, SGDR, RAdam, beta2 changes, warmup alone, bilateral symmetry, LR≠5e-4, WD+gc with heavy WD
   - **Remaining active fronts:** surface points 16k/32k/64k (sent back with eval fix), log-cosh+gc, supernodes+gc, WD+gc lighter, architecture variants, throughput, optimizer (Lion, Lookahead, SWA, OneCycleLR), regularization, checkpoint averaging

2. **TFP clean test result** — val=0.002383
   - **CLOSED:** All EMA deviations (sinh overflow), all T_max>10, gc=0.3 (starvation), gc=0.7 (destabilize), 4L depth
   - **Active:** gc=0.4 boundary test (rei), T_max=5/8 (mitsuha), 3L/256d (robin), WD sweep (shinobu), clean test (shoya URGENT)

3. **AirfRANS volume** — 1.63x gap to target
   - **Lead:** vol-weight=3x gave surface -17% (0.000381!) — 1.5x/2.0x sweep in progress
   - **Active:** EMA higher decay (stark), 3L+EMA (spike), wider model+EMA (wolfwood sent back), vol-weight compound (tanjiro/violet/megumi/nezuko)

4. **TandemFoil** — gc trend: monotonic improvement 1.0→0.5→0.3
   - **Active:** gc=0.2 (usopp), gc=0.3+480min (zenitsu), clean test (yuji PAPER-FACING)

## Key Dead Ends (Do Not Repeat)

**DrivAerML:**
- T_max≠30 without gc (only 30 is stable)
- Depth: 4L required; 5L also worse
- EMA without gc: diverges (confirmed again in #3072 Run 1)
- gc+WD heavy (WD=1e-3): 4.44%, below baseline (lighter WD in progress)
- gc alone (all values): best 4.346%, above baseline
- Bilateral symmetry aug, torch.compile, gradient accumulation
- LR≠5e-4, SGDR, RAdam, beta2≠0.999, warmup alone
- Huber loss, relative L2 loss (degenerate), 6L depth

**TandemFoil Paper:**
- T_max≠10 (all directions diverge)
- gc≠0.5 (0.3=starvation, 0.7=destabilize; 0.4 being tested)
- EMA≠0.999 (0.99 and 0.9995 both cause sinh overflow — very narrow window)
- 4L depth (pressure overflow), LR=1.5e-4 (+34%)

**AirfRANS:**
- 2L/384d and 3L/384d: catastrophic divergence
- EMA<0.999 (0.99, 0.995 both worse)
- T_max≠50 (30 and 100 both worse)
- Vol-weight=10x: worse on both metrics

## Infrastructure
- **PR #2974 MERGED:** Best-checkpoint saving in train.py — all new experiments use best val checkpoint for test eval

## Mandatory Config Rules

- **TF:** Lion lr=1.25e-4, T_max=10, gc=0.3, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **TFP:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **AF:** AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, `--ema-decay 0.999`, 2L/256d
- **DM:** AdamW lr=5e-4, T_max=30, **gc=0.5**, **EMA=0.9995**, no WD, 4L/512d (**UPDATED post #3072**)
- `--epochs 999`, `SENPAI_MAX_EPOCHS=9999` mandatory for DM
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`
- Paper-facing DM: NO `--max-eval-batches`
