# SENPAI Research State

- **Date:** 2026-04-23 12:30 (advisor cycle 31 — closed #3162/#3157, assigning himmel/nami)
- **Branch:** radford
- **Idle students:** 0 (himmel/nami being assigned)
- **PRs ready for review:** 0

## Fleet Status

### DrivAerML WIP (~36 students, ~60%)
- `#3165` senku: EMA+gc+4H heads (128d/head)
- `#3164` faye: paper-facing full eval (two-phase) — PAPER-FACING
- `#3163` shouko: EMA+gc+attn dropout=0.05
- `#NEW` himmel: AGC (adaptive gradient clipping) on EMA champion — BEING ASSIGNED
- `#3161` spike: EMA+gc+eta_min=1e-5 (cosine floor)
- `#3160` griffith: EMA+gc+16H heads
- `#3159` franky: EMA+gc+4L/640d (wider)
- `#3158` bulma: EMA+gc+lr=4e-4
- `#3155` nobara: EMA+gc+longer T_max (45/60)
- `#3154` historia: EMA+gc+monotonic cosine (no restarts)
- `#3153` emma: EMA+gc+light WD (1e-4/5e-4)
- `#3152` eren: EMA decay sweep (0.999 vs 0.9995)
- `#3149` fern: stochastic input feature dropout (p=0.05/0.10)
- `#3147` norman: lighter WD+gc=1.0 (5e-4/1e-4)
- `#3146` taki: top-5 checkpoint averaging
- `#3143` thorfinn: Lookahead(AdamW)
- `#3138` kakashi: 788 batches+T_max=60 (SENT BACK — add gc=0.5+EMA)
- `#3137` chopper: 5-ep warmup+gc=1.0
- `#3132` gohan: eta_min=5e-5+gc=1.0
- `#3131` sanji: OneCycleLR schedule
- `#3121` levi: dropout regularization sweep
- `#3118` hinata: WD alone at champion config
- `#3115` piccolo: bs2+25k pts (SENT BACK — 50k+gc=0.5+EMA)
- `#3110` einar: AdamW beta2=0.99/0.995
- `#3109` guts: lr=4e-4 full-eval
- `#3085` kohaku: larger supernodes (SENT BACK — retry+gc=1.0)
- `#3083` jet: max-train-batches=600
- `#3079` gojo: 4L/640d (SENT BACK — lr=3e-4+T_max=60+EMA)
- `#3076` frieren: log-cosh loss (SENT BACK — retry+gc=1.0)
- `#3068` brook: 64k surface points (SENT BACK — add max-eval-batches 200)
- `#3067` askeladd: 32k surface points (SENT BACK — add max-eval-batches 200)
- `#3066` alphonse: 16k surface points
- `#3065` chihiro: multi-seed s456 (SENT BACK — two-phase eval)
- `#3064` casca: multi-seed s123 (SENT BACK — two-phase eval)
- `#3063` canute: paper-facing full-eval (SENT BACK — two-phase)
- `#3046` sukuna: WD+gc compound (SENT BACK — WD=5e-4/1e-4+gc=1.0)

### TandemFoil Paper WIP (~10 students, ~17%)
- `#NEW` nami: attention heads sweep (4H/16H) — BEING ASSIGNED (moved from TFP)
- `#3145` rei: gc=0.4 boundary test
- `#3133` shinobu: WD sweep (5e-3/2e-2)
- `#3124` robin: 3L/256d wider model
- `#3123` mitsuha: shorter T_max (5/8)
- `#3098` shoya: clean test evaluation — URGENT PAPER-FACING
- `#3088` mugen: T_max=20
- `#3056` haku: Lion+EMA refinement (T_max/gc/LR sweep)
- `#2949` vash: depth/width sweep (LR=5e-5)
- `#NEW` jin: multi-seed paper-facing (seeds 42/123/456) — BEING ASSIGNED

### AirfRANS WIP (~10 students, ~17%)
- `#3156` edward: softer gc=0.5 on EMA champion
- `#3144` violet: vol-weight=2.0+EMA=0.999
- `#3136` stark: EMA decay higher (0.9995/0.9999)
- `#3135` nezuko: EMA=0.999+vol-weight=10x
- `#3134` megumi: vol-weight=30x
- `#3129` chrome: 4L/256d deeper architecture
- `#3106` wolfwood: 2L/384d+gc+T_max=50 (SENT BACK — add EMA=0.999)
- `#3101` tanjiro: vol-loss-weight=1.5 (SENT BACK)
- `#3166` vegeta: vol-weight=5.0/7.0+EMA=0.999 — JUST ASSIGNED
- `#NEW` gilbert: lr=8e-4/1e-3+EMA=0.999 — BEING ASSIGNED

### TandemFoil WIP (~3 students, ~5%)
- `#3150` yuji: clean test row gc=0.3 champion — PAPER-FACING
- `#3142` zenitsu: gc=0.3+longer budget (480-min)
- `#3140` usopp: gc=0.2 sweep

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
| AirfRANS | `Surf MSE / Vol MSE` | **0.000459 / 0.002777** | 0.0043 / 0.0017 | Surface 9.4x better, Volume 1.63x gap |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **4.685%** (#3072, partial eval) | 3.71% | Gap closing — #3164 paper eval in-progress |

## Current Research Focus

### Benchmark Sprint Priorities (ICML phase)

1. **DrivAerML** — val=3.833%, gap to AB-UPT only 0.013 pp!
   - **BREAKTHROUGH:** EMA=0.9995+gc=0.5 works. gc is the stability enabler for EMA. Gap 93% closed.
   - **Confirmed:** gc=0.5 is sharp optimum (0.25 starves, 1.0 insufficient — triple-confirmed via #3072/#3151/#3114)
   - **Phase 2 strategy:** All new DM experiments build on EMA+gc=0.5 champion platform
   - **Active fronts on champion platform:** decay sweep (eren), attention heads (senku/griffith), wider 640d (franky), eta_min (spike), attn dropout (shouko), softer gc=0.3 (himmel), monotonic cosine (historia), WD compound (emma), LR sweep (bulma), longer T_max (nobara)
   - **Paper-facing:** faye #3164 doing full eval, multi-seed (casca/chihiro/canute sent back for two-phase)
   - **Throughput experiments:** jet (600 batches), kakashi (788 batches+T_max=60)
   - **Surface point resolution:** 16k/32k/64k (all sent back with eval fix)
   - **Other active:** Lookahead (thorfinn), OneCycle (sanji), dropout (levi), checkpoint averaging (taki), feature dropout (fern), beta2 (einar), input feature dropout (fern)

2. **TFP clean test result** — val=0.002383
   - **Fully locked config:** Lion lr=1.25e-4, T_max=10, gc=0.5, EMA=0.999, 3L/192d
   - **Every deviation diverges** — very narrow stability window
   - **URGENT:** shoya #3098 clean test evaluation
   - **Active width/depth:** robin (256d), nami (224d)
   - **Active schedule:** mitsuha (T_max=5/8), mugen (T_max=20)
   - **Active regularization:** shinobu (WD sweep), rei (gc=0.4)
   - **Multi-seed:** jin being assigned (seeds 42/123/456)

3. **AirfRANS volume** — 1.63x gap to target (0.002777 vs 0.0017)
   - **Surface already 9.4x better** than target — focus is purely on volume
   - **vol-weight strategy:** 3x good for surface, 10x worse. Now testing 1.5x (tanjiro), 2.0x (violet), 5.0x/7.0x (vegeta), 30x (megumi)
   - **LR sweep:** gilbert being assigned (lr=8e-4, 1e-3 with EMA)
   - **EMA decay:** stark testing 0.9995/0.9999
   - **gc transfer:** edward testing gc=0.5 (from DM success)
   - **Depth:** chrome testing 4L/256d (2L optimal so far, 3L dead)
   - **Width:** wolfwood 2L/384d (sent back to add EMA)

4. **TandemFoil** — gc trend monotonic: 1.0→0.5→0.3
   - **Active:** gc=0.2 (usopp), gc=0.3+480min (zenitsu)
   - **Paper-facing:** yuji #3150 clean test from gc=0.3 champion

## Key Dead Ends (Do Not Repeat)

**DrivAerML:**
- gc≠0.5 with EMA: 0.25 starves (13.1%), 0.3 diverges ep81 (8.816%), 1.0 insufficient (6.21%) — QUAD CONFIRMED
- EMA without gc: diverges (confirmed #3072 Run 1, plus 3 prior attempts)
- gc alone (any value, no EMA): best 4.346%, above baseline ceiling
- Non-EMA regime ceiling: 3.997% (all diverge at cosine restart peaks)
- 5L depth: 5.515% with gc (#3141), 4.172% without (#3104). 6L: 6.37%
- Lion optimizer: all LR variants diverge (5e-5, 1e-4, 5e-5+gc)
- Gradient centralization: fragile at LR restart boundaries (5.492% then diverge)
- Polynomial LR (no cosine troughs): catastrophic — troughs are load-bearing stability features
- SWA: equal-weight averaging poisons across divergent basins (88.98%)
- T_max=50+gc (both 0.5 and 1.0): diverge — T_max=30 only viable period
- 10-ep warmup+gc=1.0: 11.2% then diverged
- 5-ep warmup+gc=1.0: pending (chopper)
- Huber loss, relative L2 loss (degenerate), SGDR, RAdam
- beta2≠0.999, LR≠5e-4 (without EMA), WD+gc heavy (WD=1e-3: 4.44%)
- Bilateral symmetry aug, torch.compile, gradient accumulation
- Attention dropout without EMA/gc: toxic combo with T_max=30
- Cosine eta_min without EMA/gc: diverges (7.255-7.918%)

**TandemFoil Paper:**
- T_max≠10 (all directions diverge)
- gc≠0.5 (0.3=starvation, 0.7=destabilize; 0.4 being tested)
- EMA≠0.999 (0.99 and 0.9995 both cause sinh overflow — very narrow window)
- 4L depth (pressure overflow), LR=1.5e-4 (+34%)
- 3L/224d width: catastrophic divergence (field_mse ~2.2e9, grad explosion ep87)
- T_max=20/30: field_mse never reaches finite values

**AirfRANS:**
- 3L depth (with or without EMA): catastrophic divergence confirmed twice
- 2L/384d and 3L/384d: catastrophic divergence
- EMA<0.999 (0.99, 0.995 both worse)
- T_max≠50 (30 and 100 both worse)
- Vol-weight=10x: worse on both metrics

**TandemFoil:**
- gc≥0.5: monotonically worse than gc=0.3

## Mandatory Config Rules

- **TF:** Lion lr=1.25e-4, T_max=10, gc=0.3, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **TFP:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **AF:** AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, `--ema-decay 0.999`, 2L/256d
- **DM:** AdamW lr=5e-4, T_max=30, **gc=0.5**, **EMA=0.9995**, no WD, 4L/512d (**UPDATED post #3072**)
- `--epochs 999`, `SENPAI_MAX_EPOCHS=9999` mandatory for DM
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`
- Paper-facing DM: NO `--max-eval-batches`

## Infrastructure
- **PR #2974 MERGED:** Best-checkpoint saving in train.py — all new experiments use best val checkpoint for test eval
