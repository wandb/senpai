# SENPAI Research State

- **Date:** 2026-04-23 18:00 (advisor cycle 39 — closed #3068/#3066/#3064, assigned brook/alphonse/casca)
- **Branch:** radford
- **Idle students:** 0
- **PRs ready for review:** 0

## Fleet Status (59 active PRs)

### DrivAerML WIP (30 PRs)
**Noam-pivot experiments (highest priority):**
- `#3192` casca: asinh-pressure alone (scale=0.75/0.5 sweep) — NOAM ABLATION
- `#3188` chihiro: 2H/16H heads sweep on champion — NOAM ABLATION
- `#3181` himmel: asinh+residual on champion — NOAM ABLATION
- `#3175` hinata: full noam stack on champion — NOAM ABLATION

**Pre-pivot champion tuning:**
- `#3173` kakashi: shorter cosine T_max=15/20
- `#3171` sanji: LR warmup (5-ep/10-ep) on champion
- `#3165` senku: 4H heads (128d/head)
- `#3164` faye: paper-facing full eval (two-phase) — PAPER-FACING
- `#3163` shouko: attn dropout=0.05
- `#3161` spike: eta_min=1e-5 (cosine floor)
- `#3160` griffith: 16H heads
- `#3159` franky: 4L/640d wider
- `#3158` bulma: lr=4e-4
- `#3155` nobara: longer T_max (45/60)
- `#3154` historia: monotonic cosine (no restarts)
- `#3152` eren: EMA decay sweep (0.999 vs 0.9995)
- `#3146` taki: top-5 checkpoint averaging
- `#3143` thorfinn: Lookahead(AdamW)
- `#3132` gohan: eta_min=5e-5+gc=1.0
- `#3121` levi: dropout regularization sweep
- `#3115` piccolo: bs2+25k pts (SENT BACK — 50k+gc=0.5+EMA)
- `#3110` einar: beta2=0.99/0.995
- `#3109` guts: lr=4e-4 full-eval
- `#3085` kohaku: larger supernodes (SENT BACK — retry+gc=1.0)
- `#3083` jet: max-train-batches=600
- `#3079` gojo: 4L/640d (SENT BACK — lr=3e-4+T_max=60+EMA)
- `#3076` frieren: log-cosh loss (SENT BACK — retry+gc=1.0)
- `#3067` askeladd: 32k surface points (SENT BACK — add max-eval-batches 200)
- `#3063` canute: paper-facing full-eval (SENT BACK — two-phase)
- `#3046` sukuna: WD+gc compound (SENT BACK — WD=5e-4/1e-4+gc=1.0)

### TandemFoil WIP (6 PRs)
**Noam-pivot experiments (highest priority):**
- `#3189` emma: asinh+physics features (no ANP, no T_max) — NOAM ABLATION
- `#3186` norman: T_max=150 alone (gc=0.3/0.5) — NOAM ABLATION
- `#3185` fern: full noam stack (ANP+physics+T_max=150) — NOAM ABLATION
- `#3180` chopper: ANP decoder alone — NOAM ABLATION

**Other:**
- `#3150` yuji: clean test row gc=0.3 champion — PAPER-FACING
- `#3142` zenitsu: gc=0.3+longer budget (480-min)

### TandemFoil Paper WIP (10 PRs)
**Noam-pivot experiments (highest priority):**
- `#3190` brook: physics features only (wake/vortex/Cp, no ANP) — NOAM ABLATION
- `#3183` rei: ANP+T_max=150 — NOAM ABLATION
- `#3179` usopp: T_max=150+wake+96sl+Lookahead — NOAM ABLATION
- `#3176` mitsuha: full noam stack — NOAM ABLATION

**Other:**
- `#3168` jin: multi-seed paper-facing (seeds 42/123/456)
- `#3133` shinobu: WD sweep (5e-3/2e-2)
- `#3098` shoya: clean test evaluation — URGENT PAPER-FACING
- `#3088` mugen: T_max=20
- `#3056` haku: Lion+EMA refinement (T_max/gc/LR sweep)
- `#2949` vash: depth/width sweep (LR=5e-5)

### AirfRANS WIP (13 PRs)
**Noam-pivot experiments (highest priority):**
- `#3191` alphonse: noam features on base and vol-10x — NOAM ABLATION
- `#3187` stark: asinh+residual on vol-10x champion — NOAM ABLATION
- `#3184` wolfwood: full noam stack — NOAM ABLATION
- `#3177` nezuko: vol-weight 15x/20x + asinh+residual — NOAM ABLATION

**Other:**
- `#3172` robin: LR warmup (5-ep/10-ep)
- `#3169` nami: heads sweep 4H/16H
- `#3167` gilbert: higher LR (8e-4/1e-3)
- `#3166` vegeta: vol-weight=5.0/7.0+EMA=0.999
- `#3156` edward: softer gc=0.5
- `#3144` violet: vol-weight=2.0+EMA=0.999
- `#3134` megumi: vol-weight=30x
- `#3129` chrome: 4L/256d deeper
- `#3101` tanjiro: vol-loss-weight=1.5 (SENT BACK)

## Steering Anchors

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **21.350** (#3140 MERGED — gc=0.2+EMA=0.999) |
| TandemFoil Paper | `val_primary/field_mse` | **0.002383** (#3025) |
| AirfRANS | `val_primary/surface_mse` + `vol_mse` | **0.000296 / 0.002039** (#3135 MERGED — EMA+vol-weight=10x) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.833%** (#3072 MERGED — EMA=0.9995+gc=0.5) |

## Paper-Facing Snapshot

| Dataset | Metric | Current best | External target | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **23.195** (PR #3140) | (internal) | Improving |
| TandemFoil Paper | `test_primary/field_mse` | **NO CLEAN ROW YET** | ~0.10-0.36/task | URGENT — #3098 in-progress |
| AirfRANS | `Surf MSE / Vol MSE` | **0.000296 / 0.002039** | 0.0043 / 0.0017 | Surface 14.5x better, Volume 1.20x gap |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **4.685%** (#3072, partial eval) | 3.71% | Gap closing — #3164 paper eval in-progress |

## Current Research Focus

### STRATEGIC PIVOT — "Think Bigger" (Human directive, issue #3174)

**Key finding:** The noam branch has ~100 merged PRs with winning techniques. Many are ALREADY PORTED to radford's train.py but UNUSED. We've been doing incremental HP sweeps when major architectural features were available all along.

**New priority:** Every new assignment should test noam-ported features or bold new approaches. No more incremental tweaks.

### Available features NOT USED (ready in train.py)
- `--anp-srf` — ANP cross-attention decoder (-58.9% TF!) — TF/TFP only
- `--asinh-pressure --asinh-scale 0.75` — asinh pressure norm (-8% ood) — ALL datasets
- `--residual-prediction` — learned correction to freestream — ALL datasets
- `--enable-cp-panel`, `--enable-te-coord-frame`, `--enable-wake-deficit`, `--enable-wake-angle`, `--enable-vortex-panel-velocity` — physics features — TF/TFP
- `--re-stratified-sampling` — OOD Re robustness — TF/TFP/AF
- `--compile-model` — throughput gain (more epochs in time budget) — ALL
- 96 slices, 3 heads (at 192d), Lookahead — all defaults we haven't tested

### Key config mismatches
- TF/TFP T_max=10 vs noam optimal T_max=150 (15x longer!)
- Radford 8 heads vs noam 3 heads (at 192d)
- Missing: vol_loss_scale, PCGrad, temperature annealing, DomainLayerNorm (need code ports)

### Benchmark Sprint Priorities (revised)

1. **TandemFoil** — HIGHEST PRIORITY for noam feature activation
   - ANP decoder alone was -58.9% on noam. With full stack: potentially 9-12 range vs our 21.909
   - **Next assignments:** full noam stack, ANP alone, T_max=150 alone
   - Paper-facing: yuji #3150 clean test

2. **TandemFoil Paper** — Same noam features apply
   - T_max=150 could break the "fully locked" constraint — it was locked at the WRONG config
   - ANP + physics features could transform results
   - **Next assignments:** full noam stack, ANP + T_max=150

3. **DrivAerML** — val=3.833%, gap to AB-UPT only 0.013 pp
   - **Applicable noam features:** asinh-pressure, residual-prediction, compile-model, 96 slices
   - **Bold changes:** Lion optimizer (noam's final choice), higher EMA decay (0.9999)
   - Existing champion platform still valid as base

4. **AirfRANS volume** — 1.63x gap to target
   - **Applicable noam features:** asinh-pressure, residual-prediction, re-stratified-sampling
   - **Bold changes:** 96 slices, compile for throughput, vol_loss_scale (needs code port)

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
- OneCycleLR (no troughs): 8.04% best, sustained high-LR warmup worse than cosine peaks (without EMA+gc)
- SWA: equal-weight averaging poisons across divergent basins (88.98%)
- T_max=50+gc (both 0.5 and 1.0): diverge — T_max=30 only viable period
- 10-ep warmup+gc=1.0: 11.2% then diverged
- 5-ep warmup+gc=1.0: pending (chopper)
- Huber loss, relative L2 loss (degenerate), SGDR, RAdam
- beta2≠0.999, LR≠5e-4 (without EMA), WD+gc heavy (WD=1e-3: 4.44%)
- Bilateral symmetry aug, torch.compile, gradient accumulation
- Attention dropout without EMA/gc: toxic combo with T_max=30
- Cosine eta_min without EMA/gc: diverges (7.255-7.918%)
- Surface points ≠50k: 16k=11.672%, 32k=7.558%, 64k=12.16% (even with EMA+gc) — 50k only viable count

**TandemFoil Paper:**
- T_max≠10 (all directions diverge)
- gc≠0.5 (0.3=starvation, 0.7=destabilize; 0.4 being tested)
- EMA≠0.999 (0.99 and 0.9995 both cause sinh overflow — very narrow window)
- 4L depth (pressure overflow), LR=1.5e-4 (+34%)
- 3L/224d width: catastrophic divergence (field_mse ~2.2e9, grad explosion ep87)
- 3L/256d width: catastrophic divergence (field_mse 8.61e+24, 27 orders worse)
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
