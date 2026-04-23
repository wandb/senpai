# SENPAI Research State

- **Date:** 2026-04-23 12:30 (advisor cycle 46 — updated per human directive #3174)
- **Branch:** radford
- **Idle students:** 0 (4 being assigned)
- **PRs ready for review:** 0
- **CRITICAL:** TFP champion (0.002383) is unreproducible — 0/3 seeds stay finite (#3168)

## Fleet Status (57 active PRs)

### DrivAerML WIP (27 PRs)
**Innovation track (code ports — new directive):**
- `#3203` vegeta: attention temperature annealing — INNOVATION
- `#3202` senku: GLU preprocess MLP — INNOVATION
- `#3201` jet: DomainLayerNorm — INNOVATION

**Noam-pivot experiments:**
- `#3199` megumi: EMA=0.9999/0.99995 (noam optimal decay) — NOAM ABLATION
- `#3192` casca: asinh-pressure alone (scale=0.75/0.5) — NOAM ABLATION
- `#3193` franky: residual-prediction alone — NOAM ABLATION
- `#3194` bulma: higher LR sweep (5.5e-4/6e-4)
- `#3188` chihiro: 2H/16H heads sweep — NOAM ABLATION
- `#3181` himmel: asinh+residual on champion — NOAM ABLATION
- `#3175` hinata: full noam stack — NOAM ABLATION

**Champion tuning / paper-facing:**
- `#3173` kakashi: shorter cosine T_max=15/20
- `#3171` sanji: LR warmup (5-ep/10-ep)
- `#3164` faye: paper-facing full eval (two-phase) — PAPER-FACING Track 1
- `#3163` shouko: attn dropout=0.05
- `#3161` spike: eta_min=1e-5 (cosine floor)
- `#3160` griffith: 16H heads
- `#3155` nobara: longer T_max (45/60)
- `#3207` historia: DM true monotonic cosine (T_max=393606) — corrected retest
- `#3206` jet: DM 600 batches + gc=0.5 + EMA (stabilized retest)
- `#3152` eren: EMA decay sweep (0.999 vs 0.9995)
- `#3146` taki: top-5 checkpoint averaging
- `#3143` thorfinn: Lookahead(AdamW)
- `#3121` levi: dropout regularization sweep
- `#3110` einar: beta2=0.99/0.995
- `#3109` guts: lr=4e-4 full-eval
- `#3085` kohaku: larger supernodes (SENT BACK)
- `#3076` frieren: log-cosh loss (SENT BACK)
- `#3067` askeladd: 32k surface points (SENT BACK)
- `#3063` canute: paper-facing full-eval (SENT BACK)
- `#3046` sukuna: WD+gc compound (SENT BACK)

### TandemFoil WIP (7 PRs) — GUARDRAIL ONLY per directive
**Noam-pivot experiments:**
- `#3197` gojo: residual-prediction alone — NOAM ABLATION
- `#3196` zenitsu: EMA decay sweep (0.9999/0.99995) — NOAM ABLATION
- `#3189` emma: asinh+physics features — NOAM ABLATION
- `#3186` norman: T_max=150 alone — NOAM ABLATION
- `#3185` fern: full noam stack — NOAM ABLATION
- `#3180` chopper: ANP decoder alone — NOAM ABLATION
- `#3150` yuji: clean test row — PAPER-FACING

### TandemFoil Paper WIP (11 PRs) — STABILIZATION CRISIS
**Stabilization diagnostic (HIGHEST PRIORITY — baseline unreproducible):**
- `#3205` jin: TFP seed sensitivity (seed=0 verify, seed=42 + gc/warmup/asinh probes)
- `#3208` brook: TFP pressure stabilization (base config, lower LR, AdamW vs Lion)

**Noam-pivot experiments:**
- `#3183` rei: ANP+T_max=150 — NOAM ABLATION
- `#3179` usopp: T_max=150+wake+96sl+Lookahead — NOAM ABLATION
- `#3176` mitsuha: full noam stack — NOAM ABLATION

**Other:**
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
- `#3204` gilbert: vol-weight=15x/20x clean control (no extra features) — AF VOLUME FOCUS
- `#3167` gilbert: higher LR (8e-4/1e-3)
- `#3166` vegeta: vol-weight=5.0/7.0+EMA=0.999
- `#3156` edward: softer gc=0.5
- `#3144` violet: vol-weight=2.0+EMA=0.999
- `#3199` megumi: DM EMA=0.9999/0.99995 (noam optimal decay) — NOAM ABLATION
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

### LATEST HUMAN DIRECTIVES — Issue #3174 (2026-04-23 11:08–11:48)

> "We are no longer in a broad discovery phase, and we should stop acting like every benchmark wants the same recipe."
> — morganmcg1, 2026-04-23

**Strategic posture by benchmark (BINDING):**

#### AirfRANS — "Finish the Job" (Volume Closure Mode)
- Surface is already strong. **ALL new AF experiments must target vol_mse < 0.0017 specifically.**
- Surface error is a hard constraint: NO regressions accepted, even for large volume gains.
- No more broad novelty sweeps. This lane is in finish-the-job mode.

#### DrivAerML — Two-Track Approach
- **Track 1 (Stable Champion):** Keep current best config alive with full best-checkpoint test eval. No shortcuts. No `--max-eval-batches` on paper-facing runs.
- **Track 2 (Innovation):** Genuinely new physics-aware and ML ideas — NOT Radford/Noam recipe retuning. Search for benchmark-specific improvements. Code ports from noam: attn temp annealing (#3203), GLU preprocess (#3202), DomainLayerNorm (#3201). Look beyond: mesh-based operators, physics-informed losses, geometry encoders, domain-specific normalizations.
- Bold ideas permitted and encouraged here. Local neighborhood exhausted.

#### TandemFoil Paper — Stabilization First (CRISIS MODE)
- **The failure mode is NaN/inf, not underperformance.** Optimize for finding ONE finite reproducible recipe, not performance.
- Protocol: smoke test (1-3 epochs) → short debug (10-20 epochs) → long run ONLY after both are clean.
- NaN/inf failures get closed immediately, no extensions.
- brook #3200 reveals: base config (no physics) stable at 0.047; `cp_panel_prior_index()` is broken for TFP dataset — **champion may be unreproducible** with current code.
- Next step: fix `cp_panel_prior_index()` bug, then re-run stable configs.

#### TandemFoil — Guardrail Only
- Parity is healthy. Minimal compute sink. Keep as sanity anchor, not a major compute investment.
- No ambitious per-run experiments here.

### Noam Feature Activation (context-dependent per benchmark)
- `--anp-srf` — ANP cross-attention decoder (-58.9% TF!) — TF only (TFP stability-first)
- `--asinh-pressure --asinh-scale 0.75` — asinh pressure norm — DM, AF (not TFP until stable)
- `--residual-prediction` — learned correction to freestream — DM, AF
- Physics features: TF only (TFP pipeline bug: cp_panel/TE/vortex corrupted)
- `--re-stratified-sampling` — OOD Re robustness — TF/AF

### DM Innovation Track (code ports from noam)
- Attention temperature annealing (-11% on noam) — vegeta #3203
- GLU preprocess MLP — senku #3202
- DomainLayerNorm — jet #3201 (merged; awaiting results)
- Still needed: vol_loss_scale, PCGrad (for AF multi-objective)

### Critical Known Bug
- **`cp_panel_prior_index()` broken for tandemfoil_paper** (PR #3200): When `--enable-cp-panel` + `--enable-pressure-prior-addition` are both set, the function returns the wrong index (last Fourier feature, not cp_panel), corrupting all pressure predictions. Fix required before running any long TFP experiments with these flags.
- **TFP champion (#3025, 0.002383) may be unreproducible** with current codebase — this is a CRITICAL issue to resolve.

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
- eta_min=5e-5+gc (any value): gc=1.0→6.311%, gc=0.5→8.617%, both diverge — 3rd confirmation
- Surface points ≠50k: 16k=11.672%, 32k=7.558%, 64k=12.16% (even with EMA+gc) — 50k only viable count
- 4L/640d: dead at ALL tested configs — lr=5e-4+EMA+gc: 8.636% diverge ep82 (#3159); lr=5e-4 no-EMA: 4.516%/6.457%/5.724% all diverge (#3079). Width amplifies gradients beyond gc containment.
- lr<5e-4 under EMA: steep monotonic cliff (4.5e-4=4.134%, 4e-4=5.924%)
- bs=2: Blackwell bf16+bs>1 CUBLAS bug forces fp32, gets only 37% of optimizer steps. 4.373% at bs=2/fp32 vs 3.833% bs=1/bf16. Platform-blocked.
- 4H heads with EMA+gc: 6.650% ep123, diverged ep133. DM heads: 8H optimal, 4H/16H both worse

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
- Vol-weight<10x (2x/5x/7x): all worse on both metrics. 10x+EMA=0.999 is AF sweet spot
- Vol-weight=30x: catastrophic — surface 4.3x worse, vol 3.1x worse
- LR>6e-4: 8e-4 +110% surface, 1e-3 catastrophic divergence

**TandemFoil:**
- gc≥0.5: monotonically worse than gc=0.3
- Longer budget (480-min): no benefit with cosine T_max=10 cycling — more epochs = more independent draws, not monotone descent (22.016 at 480min vs 21.909 at 360min)

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
