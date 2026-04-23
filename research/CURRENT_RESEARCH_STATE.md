# SENPAI Research State

- **Date:** 2026-04-24 08:00 (advisor cycle 70)
- **Branch:** radford
- **Idle students:** 0
- **PRs ready for review:** 0
- **CRITICAL:** TFP champion config is BROKEN — code regression since #3025, seed=0 no longer reproduces (#3205). Waiting for sanji #3209 cp_panel bug fix.
- **Known bug:** `primary_metric_key` shadowing in train.py (line ~1646 local var shadows function on line ~1428). 3 students independently fixed this cycle. Fix: rename local to `best_tracking_metric_key`.

## Fleet Status (58 active PRs)

### DrivAerML WIP (28 PRs)
**Innovation track (new physics-aware/ML ideas per directive):**
- `#3247` brook: EMA periodic reset (100/50-ep intervals) — INNOVATION
- `#3243` jet: prediction-error-weighted surface sampling (hard example mining) — INNOVATION
- `#3242` historia: label smoothing / target noise augmentation — INNOVATION
- `#3235` kakashi: multi-exit prediction (aux losses at intermediate layers) — INNOVATION
- `#3233` gojo: stochastic depth (LayerDrop) regularization — INNOVATION
- `#3231` emma: surface pressure gradient smoothness regularization — INNOVATION (physics-informed)
- `#3251` fern: Pre-LayerNorm architecture ablation (+ RMSNorm variant) — INNOVATION
- `#3252` mitsuha: progressive surface point training (resolution curriculum) — INNOVATION
- `#3221` franky: gradient noise injection (Neelakantan et al.) — INNOVATION
- `#3228` chopper: stochastic weight perturbation at cosine troughs — INNOVATION
- `#3256` alphonse: coordinate noise augmentation (σ=0.001/0.005 input perturbation) — INNOVATION
- `#3253` canute: input feature dropout (Fourier channel dropout 10%/20%) — INNOVATION
- `#3202` senku: GLU preprocess MLP — INNOVATION
- `#3201` jet: DomainLayerNorm — INNOVATION

**Noam-pivot experiments:**
- `#3199` megumi: EMA=0.9999/0.99995 (noam optimal decay) — NOAM ABLATION
- `#3194` bulma: higher LR sweep (5.5e-4/6e-4)
- `#3181` himmel: asinh+residual on champion — NOAM ABLATION

**Champion tuning / paper-facing:**
- `#3209` sanji: TFP cp_panel bug fix + multi-seed retest — STABILIZATION PRIORITY
- `#3244` faye: DM Track 1 paper-facing full eval (champion config, NO --max-eval-batches) — PAPER-FACING Track 1
- `#3160` griffith: 16H heads
- `#3249` shouko: decaying weight decay schedule (WD=1e-4/5e-5 → 0 over 300ep) — INNOVATION
- `#3248` nobara: decaying peak LR at cosine restarts (SGDR eta_mult) — INNOVATION
- `#3152` eren: EMA decay sweep (0.999 vs 0.9995)
- `#3146` taki: top-5 checkpoint averaging
- `#3143` thorfinn: Lookahead(AdamW)
- `#3121` levi: dropout regularization sweep
- `#3110` einar: beta2=0.99/0.995
- `#3109` guts: lr=4e-4 full-eval
- `#3085` kohaku: larger supernodes (SENT BACK)
- `#3076` frieren: log-cosh loss (SENT BACK)
- `#3067` askeladd: 32k surface points (SENT BACK)
- `#3046` sukuna: WD+gc compound (SENT BACK)

### TandemFoil WIP (3 PRs) — GUARDRAIL ONLY per directive
**New champion config:** ANP+full physics+T_max=150+gc=0.2+EMA=0.999+96sl+Lookahead+compile+re-strat (#3185)
**Noam-pivot experiments (still running):**
- `#3196` zenitsu: EMA decay sweep (0.9999/0.99995) — NOAM ABLATION
- `#3150` yuji: clean test row — PAPER-FACING

### TandemFoil Paper WIP (9 PRs) — STABILIZATION CRISIS
**Stabilization status: CODE REGRESSION CONFIRMED — seed=0 broken, waiting for sanji #3209 bug fix**

**Noam-pivot experiments:**
- `#3179` usopp: T_max=150+wake+96sl+Lookahead — NOAM ABLATION

**Other:**
- `#3133` shinobu: WD sweep (5e-3/2e-2)
- `#3098` shoya: clean test evaluation — URGENT PAPER-FACING
- `#3088` mugen: T_max=20
- `#3056` haku: Lion+EMA refinement (T_max/gc/LR sweep)
- `#2949` vash: depth/width sweep (LR=5e-5)

### AirfRANS WIP (14 PRs)
**Volume closure experiments:**
- `#3255` gohan: no-Lookahead/no-compile champion full budget — AF VOLUME FOCUS (critical ablation follow-up)
- `#3257` chihiro: extended training via reduced eval frequency (every 3/5 epochs) — AF VOLUME FOCUS
- `#3241` hinata: T_max=75 + vol-weight=10x/12x on champion — AF VOLUME FOCUS (schedule gap-fill)
- `#3234` jin: lower LR sweep (5e-4/4e-4) on vol-10x champion — AF VOLUME FOCUS
- `#3232` nezuko: vol-weight warm-up schedule (1x→10x linear ramp over 200ep) — AF VOLUME FOCUS

**Noam-pivot experiments (asinh-dependent — likely to fail):**
- `#3187` stark: asinh+residual on vol-10x champion — NOAM ABLATION (asinh confirmed dead for AF)
- `#3184` wolfwood: full noam stack — NOAM ABLATION (asinh confirmed dead for AF)

**Other:**
- `#3172` robin: LR warmup — SENT BACK (confounded config, re-running with correct champion config)
- `#3169` nami: heads sweep 4H/16H
- `#3227` casca: focal-MSE volume loss (upweight hard predictions) — AF VOLUME FOCUS
- `#3254` norman: multi-seed champion run (seeds 42/123/789) — AF VOLUME FOCUS
- `#3238` rei: WD=0 ablation on vol-10x+EMA champion — AF VOLUME FOCUS
- `#3245` tanjiro: per-channel volume loss weighting (upweight nut×4, p×2) — AF VOLUME FOCUS
- `#3211` spike: AF vol_loss_scale learnable scalar (noam port) — AF VOLUME FOCUS
- `#3240` gilbert: joint checkpoint selection + vol-weight 11x/12x/13x fine sweep — AF VOLUME FOCUS
- `#3239` vegeta: additive boundary layer auxiliary volume loss — AF VOLUME FOCUS
- `#3195` piccolo: Re-stratified sampling
- `#3156` edward: softer gc=0.5
- `#3144` violet: vol-weight=2.0+EMA=0.999
- `#3129` chrome: 4L/256d deeper

## Steering Anchors

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **21.319** (#3185 MERGED — full noam stack: ANP+physics+T_max=150+Lookahead+96sl) |
| TandemFoil Paper | `val_primary/field_mse` | **0.002383** (#3025) |
| AirfRANS | `val_primary/surface_mse` + `vol_mse` | **0.000296 / 0.002039** (#3135 MERGED — EMA+vol-weight=10x) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.833%** (#3072 MERGED — EMA=0.9995+gc=0.5) |

## Paper-Facing Snapshot

| Dataset | Metric | Current best | External target | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **22.868** (PR #3185 MERGED) | (internal) | Improving |
| TandemFoil Paper | `test_primary/field_mse` | **NO CLEAN ROW YET** | ~0.10-0.36/task | URGENT — #3098 in-progress |
| AirfRANS | `Surf MSE / Vol MSE` | **0.000296 / 0.002039** | 0.0043 / 0.0017 | Surface 14.5x better, Volume 1.20x gap |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **4.685%** (#3072, partial eval) | 3.71% | Gap closing — #3244 paper eval in-progress (faye) |

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

#### TandemFoil Paper — Stabilization First (CRISIS MODE — UPGRADED)
- **CODE REGRESSION CONFIRMED (#3205):** seed=0 no longer reproduces champion. ALL seeds + ALL stabilization probes → Inf pressure. This is NOT seed sensitivity — it's a broken pressure transform path in current codebase.
- **Base config (no physics) IS stable** across seeds (#3208): floor ~0.047-0.049. Velocity channels healthy. Pathology isolated to sinh inverse pressure transform.
- **Root cause:** cp_panel_prior_index() bug (#3200) + possible eval-time sinh overflow. Sanji #3209 fixing the cp_panel bug — MUST land before any further TFP optimization.
- Protocol: fix bug → verify seed=0 reproduces → then reintroduce pressure transforms one at a time.
- NaN/inf failures get closed immediately, no extensions.

#### TandemFoil — Guardrail Only
- Parity is healthy. Minimal compute sink. Keep as sanity anchor, not a major compute investment.
- No ambitious per-run experiments here.

### Noam Feature Activation (context-dependent per benchmark)
- `--anp-srf` — ANP cross-attention decoder (-58.9% TF!) — TF only (TFP stability-first)
- `--asinh-pressure --asinh-scale 0.75` — asinh pressure norm — DM, AF (not TFP until stable)
- `--residual-prediction` — learned correction to freestream — DM, AF
- Physics features: TF only (TFP pipeline bug: cp_panel/TE/vortex corrupted)
- `--re-stratified-sampling` — OOD Re robustness — TF/AF

### DM Innovation Track (expanded beyond recipe tuning)
**Code ports from noam:**
- Attention temperature annealing (-11% on noam) — vegeta #3203
- GLU preprocess MLP — senku #3202
- DomainLayerNorm — jet #3201 (merged; awaiting results)

**New physics-aware/ML ideas (cycle 48):**
- Fourier encoding of surface normals (angular frequency features) — brook #3217
- Auxiliary gradient prediction (∂p/∂x,y,z as regularization) — kakashi #3214
- Attention distance bias (ALiBi-inspired spatial prior) — canute #3216

**AF volume closure:**
- vol_loss_scale learnable scalar (noam port, -15.9%) — spike #3211
- Per-channel volume loss weighting (upweight nut×4, p×2) — tanjiro #3245

### Critical Known Bug
- **`cp_panel_prior_index()` broken for tandemfoil_paper** (PR #3200): Fix in progress (sanji #3209).
- **TFP champion config BROKEN in current codebase** — code regression confirmed (#3205). Seed=0 no longer reproduces. Base config without physics is stable (#3208). Root cause: pressure transform path (sinh inverse).

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
- T_max≠30: ALL tested — 15→4.406%, 20→4.943%, 30→3.833% CHAMPION, 45→4.638%, 50→diverge, 60→4.409%. T_max=30 is definitive sweet spot
- Noam features on DM: DEFINITIVELY DEAD. Asinh triple-confirmed (4.072%/4.421%/4.475%), residual needs asinh+still→3.999%, full stack→4.447%. Features tuned for T_max=150/3H/no-gc regime
- Attention distance bias (ALiBi): linear diverges, log=5.511% at timeout. Redundant with slice-based spatial grouping
- Auxiliary gradient prediction (∂p/∂x,y,z): kNN targets too noisy — aux_weight=0.1 diverges ep22, aux_weight=0.01 best 5.485% (+43%). Noise dominates once primary loss flattens
- Mixup regularization (buffer-based latent): 85-109% worse, crashed ~ep235. Buffer staleness + non-smooth geometry-specific latent space fundamentally incompatible with bs=1
- Fourier-encoded surface normals: 4.374%/4.454% (4 bands/2 bands). Normals are unit vectors in [-1,1] — don't benefit from Fourier lifting like unbounded coords. More bands = earlier divergence
- Snapshot ensemble (cosine trough checkpoints): test=6.044% single best, WORSE with more snapshots (K=3→6.24%, K=5→6.98%, K=6→7.81%). Quality gradient dominates diversity. EMA already provides implicit smoothing.
- Spectral norm on QKV: 4.035% (+5.3%). σ=1 cap over-restricts attention capacity. Late collapse ep502 from unconstrained FFN layers. gc=0.5 already solves restart stability.
- Self-distillation via EMA teacher: α=0.3→6.446% diverge ep170, α=0.1→4.323% diverge ep400. EMA=0.9995 lag creates destabilizing feedback loop. Same EMA for ckpt+teacher incompatible.
- Attention temperature annealing: 4.024% (+5% vs 3.833%). External annealed τ compounds with existing learnable per-head τ. noam result doesn't transfer
- SAM optimizer (rho=0.05/0.02): 5.36%/9.08%. SAM perturbation + cosine restart = double shock → catastrophic divergence. 2x compute penalty also prohibitive
- True monotonic cosine (T_max=393606, no restarts): 4.086% (+0.253pp). Confirms T_max=30 rapid restarts are core mechanism, not noise. Monotonic decay can't compete
- 600 batches/epoch (stabilized retest): T_max=46 proportional diverged ep61; T_max=30 got 3.887% after MORE total batches than baseline. Data diversity per epoch saturated at 394
- Head count sweep complete: 2H=catastrophic, 4H=6.650%, 8H=3.833% CHAMPION, 16H=4.099%. 8H (64d/head) is definitive
- 10-ep warmup+gc=1.0: 11.2% then diverged
- 5-ep warmup+gc=1.0: pending (chopper)
- LR warmup + EMA+gc=0.5: 5-ep=11.325% diverged, 10-ep=3.918% plateaued (didn't beat 3.833%)
- Huber loss, relative L2 loss (degenerate), SGDR, RAdam
- beta2≠0.999, LR≠5e-4 (without EMA), WD+gc heavy (WD=1e-3: 4.44%)
- Bilateral symmetry aug, torch.compile, gradient accumulation
- Attention dropout=0.05: incompatible at any stability level. Without EMA/gc→12.533%, with EMA+gc→10.118% (delayed divergence ep74 but same fate). Dropout noise compounds faster than gc clips.
- Cosine eta_min without EMA/gc: diverges (7.255-7.918%)
- eta_min+gc (any combination): gc=1.0→6.311%, gc=0.5→8.617%, gc=0.5+EMA→4.202% diverge ep312 — TRIPLE CONFIRMED. gc=0.5 relies on zero-LR trough damping; eta_min removes reset.
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
- asinh-pressure: DEFINITIVELY DEAD — 5 independent confirmations (#3191 T1/T2, #3177 T1/T2/T3). sinh() denormalization exponentially amplifies pressure errors. Non-pressure channels fine. Incompatible with AF pressure distribution.
- PCGrad gradient surgery: 5-6x worse on both metrics (#3212). retain_graph=True disables torch.compile (40% throughput loss). Gradient conflict is NOT the bottleneck.
- GradNorm adaptive loss balancing: surface 1.74x worse, vol 2.64x worse (#3218). retain_graph=True breaks compile (halves throughput). GradNorm converges to w_vol~8x, LOWER than hand-tuned 10x — fights intentional asymmetry
- Volume smoothness regularization (KNN): ALL 4 trials regressed both metrics 44-97% (#3223). KNN Euclidean smoothness penalizes legitimate BL/wake/shear gradients. Even λ=1e-4 harmful
- Residual prediction (no asinh): +55-72% surface worse (#3230). AF has large separated regions — "correction from freestream" IS the entire signal. Re-stratified sampling is no-op (uniform weights)
- Vol-weight curriculum (20x→10x / 15x→10x): collapsed or 38-52% worse on both metrics (#3226). Static 10x validated as optimal — any deviation above hurts
- Lookahead hurts AF surface by 53% at equal epochs (#3236 ablation). Compile also hurts surface despite more epochs. Both are default-on in baseline — ambiguity on synergistic effect at long training
- Proximity-weighted volume loss: 118-2516% worse (#3222). Extreme weight ratios starve far-field gradients while overdriving near-surface. Structural failure at scale=0.1/eps=0.01
- Huber loss on volume channel: no improvement over MSE on either metric (#3215). AF volume residuals are well-behaved, not heavy-tailed — Huber δ threshold adds no benefit
- 3L depth (with or without EMA): catastrophic divergence confirmed twice
- 2L/384d and 3L/384d: catastrophic divergence
- EMA<0.999 (0.99, 0.995 both worse)
- T_max≠50 (30 and 100 both worse)
- Vol-weight<10x (1.5x/2x/3x/5x/7x): ALL worse on both metrics across 6 tested values. 10x+EMA=0.999 is AF sweet spot
- Vol-weight=30x: catastrophic — surface 4.3x worse, vol 3.1x worse
- Vol-weight=15x: surface +19%, vol +67% (#3204). Vol-weight=20x: crashed ep381 (#3204). 10x is definitive AF operating point
- LR>6e-4: 8e-4 +110% surface, 1e-3 catastrophic divergence

**TandemFoil:**
- gc≥0.5: monotonically worse than gc=0.3
- Longer budget (480-min): no benefit with cosine T_max=10 cycling — more epochs = more independent draws, not monotone descent (22.016 at 480min vs 21.909 at 360min)

## Mandatory Config Rules

- **TF:** Lion lr=1.25e-4, **T_max=150**, gc=0.2, WD=1e-2, `--ema-decay 0.999`, 3L/192d, **96 slices**, ANP+full physics+asinh-scale=0.75+residual+Lookahead+compile+re-strat (**UPDATED post #3185**)
- **TFP:** Lion lr=1.25e-4, T_max=10, gc=0.5, WD=1e-2, `--ema-decay 0.999`, 3L/192d
- **AF:** AdamW lr=6e-4, T_max=50, gc=1.0, WD=1e-2, `--ema-decay 0.999`, 2L/256d
- **DM:** AdamW lr=5e-4, T_max=30, **gc=0.5**, **EMA=0.9995**, no WD, 4L/512d (**UPDATED post #3072**)
- `--epochs 999`, `SENPAI_MAX_EPOCHS=9999` mandatory for DM
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`
- Paper-facing DM: NO `--max-eval-batches`

## Infrastructure
- **PR #2974 MERGED:** Best-checkpoint saving in train.py — all new experiments use best val checkpoint for test eval
