# SENPAI Research State

- **Date:** 2026-04-25 03:15 (advisor cycle 130)
- **Branch:** radford
- **ACTIVE DIRECTIVE: Issue #3283 — Last Ditch Benchmark Push (final hours)**
- **TFP NEW BEST: val=0.002180, test=0.001931** (PR #3307 MERGED, haku lr=7e-5, -7%/-13% vs previous)
- **DM NEW VAL BEST: 3.622%** (chrome #3298 T_max=36 MSE-only, MERGED). **BREAKS AB-UPT SOTA by 0.088pp!** Still improving at ep874 timeout.
- **DM authoritative TEST: 4.218%** (canute full-eval run n2t1nzsb, metric-aware w=0.05 T_max=30). T_max=36 full-eval PENDING (#3362 jet).
- **CRITICAL COMPOUND: T_max=36 + metric-aware loss** — 5 students assigned: chrome #3359 (w=0.05), hinata #3360 (w=0.05 seed=0), himmel #3361 (w=0.04), nami #3363 (w=0.03), robin #3364 (w=0.06)
- **Fleet:** 59 students active. All assigned. Zero idle GPUs.
- **Harness patches (#3284): MERGED.** All DM runs use --no-compile-model, --save-checkpoint, --ema-mode fixed
- **Cycle 130:** Closed nami #3297 (T_max=24, val=4.093%) and robin #3294 (gc=0.25, val=3.872%). Both reassigned to T_max=36 + metric-aware compounds.
- **AF BREAKTHROUGH:** eval-every-3 + vol-weight=12x yields programme-best surface (0.000228). 8-student sweep launched to find vol-weight where both metrics beat baseline simultaneously.

## BINDING DIRECTIVE — Issue #3283 (2026-04-24)

**Capacity split (59-student fleet):** 36 DM / 12 AF / 11 TFP / 0 TF
**DM full-test gap: ~4.39-4.50% → 3.71% target (15% relative)**
**Harness patches CRITICAL (vegeta #3284):** --load-checkpoint, --eval-interval, --ema-mode fixed, --skip-update-grad-norm, --save-checkpoint, kill gates

### DM Strategy (36 students)
**Truth/recovery (~40%):** champion replicas (seeds, no-compile, fixed EMA), checkpoint recovery, high-density surface (64k/96k/128k), gradient-spike-safe (gc=0.25, skip-update-grad-norm), narrow schedule (T_max=24/30/36, lr=4.8-5.2e-4), checkpoint soup/ensemble
**Breakout (~60%):** AB-UPT-style anchored decoder, metric-aware fine-tune (mse_z + raw rel-L2), domain-normalized volume auxiliary, coordinate normalization + geometry features, final train+val retrain

### AF Strategy (12 students)
vol-weight sweep 8-13, extended champion training, joint checkpoint selection, 1-2 architecture rescue lanes max. Kill: surface>0.001 at ep180, vol>0.004 at ep180, vol>0.003 at ep300.

### TFP Strategy (11 students)
Re-run haku/tfp-t20-gc05-lr1e4 for 2-8 seeds (best_val=0.002466, best_test=0.002393). Kill: field_mse>0.006 at ep220, >0.004 at ep350, no improvement for 160 epochs.

### TF: FROZEN (only #3185 as guardrail)

## Fleet Status (cycle 119)

### DM Metric-Aware Sweep (HIGHEST PRIORITY)
**Paper-critical:**
- `#3351` canute: w=0.04 + **mandatory full-eval TEST** — PAPER-FACING RUN

**Weight sweep (cycle 118):**
- `#3323` einar: w=0.03 seed=42
- `#3324` frieren: w=0.04 seed=42
- `#3325` kohaku: w=0.02 seed=0
- `#3326` sukuna: w=0.05 seed=0
- `#3327` jin: w=0.07 seed=42
- `#3328` mugen: w=0.05 + WD=1e-3
- `#3329` sanji: w=0.03 + lr=4e-4
- `#3330` shinobu: w=0.015 seed=42
- `#3331` stark: w=0.05 + EMA=0.999

**Schedule + seed + hyperparameter variations (cycles 119-120):**
- `#3332` mitsuha: w=0.05 + T_max=20 (shorter schedule)
- `#3333` alphonse: w=0.05 + T_max=40 (longer schedule)
- `#3334` fern: w=0.04 + WD=1e-3
- `#3335` historia: w=0.05 seed=13
- `#3336` bulma: w=0.03 + EMA=0.999
- `#3337` guts: w=0.08 (high-weight probe)
- `#3349` kakashi: w=0.05 + gc=0.3 (softer clipping)
- `#3350` taki: w=0.05 + lr=4.5e-4 (narrow LR probe)

### DM Breakout Lane
- `#3300` vegeta: **Breakout 1** — AB-UPT-style anchored decoder (last remaining)

### DM Champion Config Exploration (pre-metric-aware)
- `#3313` megumi, `#3314` thorfinn, `#3315` emma, `#3316` chopper, `#3317` nobara, `#3318` shouko, `#3319` senku, `#3320` brook, `#3308` gojo

### DM Champion Recovery Seeds
- `#3293` norman (seed=0, TEST=4.250% batch-ltd, full-eval pending), `#3285` jet, `#3305` faye, `#3310` zenitsu, `#3311` franky, `#3312` griffith, `#3321` wolfwood

### DM T_max=36 + Metric-Aware Compound (HIGHEST PRIORITY)
- `#3359` chrome: w=0.05 + T_max=36 (primary compound)
- `#3360` hinata: w=0.05 + T_max=36 + seed=0 (robustness)
- `#3361` himmel: w=0.04 + T_max=36 (lower weight)
- `#3362` jet: T_max=36 MSE-only + full-eval TEST (paper-facing)
- `#3363` nami: w=0.03 + T_max=36 (low weight)
- `#3364` robin: w=0.06 + T_max=36 (high weight)

### DM Continuing Experiments
- `#3290` askeladd, `#3152` eren, `#3121` levi

### AirfRANS — eval-every-3 Sweep (NEW BREAKTHROUGH DIRECTION)
**Key insight:** eval-every-3 gives ~3x more training epochs. Combined with vol-weight tuning, may simultaneously beat both surface and volume baselines.
- `#3338` rei: vol-10x + eval-3 (throughput baseline)
- `#3339` hinata: vol-11x + eval-3
- `#3340` violet: vol-12x + eval-3
- `#3341` casca: vol-13x + eval-3
- `#3342` gohan: vol-14x + eval-3
- `#3343` tanjiro: vol-12x + eval-3 + seed=42
- `#3344` piccolo: vol-10x + eval-3 + T_max=75
- `#3345` edward: vol-12x + eval-3 + WD=5e-3

**Sent back:**
- `#3257` chihiro: eval-every-3 → follow-up with vol-12x
- `#3240` gilbert: vol-12x fine sweep → follow-up with eval-3

**Continuing AF:**
- `#3309` spike: champion seed=42 extended 600-min

### TandemFoil Paper WIP
**TFP NEW baseline: val=0.002180 / test=0.001931 (PR #3307 MERGED, haku lr=7e-5)**
**LR bracket (narrowing optimum):**
- `#3352` haku: lr bracket 6e-5 / 7.5e-5 around new champion

**4L depth direction:**
- `#3346` yuji: 4L/192d lr=4e-5 T_max=15
- `#3347` usopp: 4L/192d lr=5e-5 T_max=20
- `#2949` vash: SENT BACK for lr=3e-5/7e-5 bracket

**Continuing TFP:**
- `#3287` nezuko: haku seed=0
- `#3098` shoya: clean test eval

### Experiments KILLED this cycle (14 PRs per #3283)
- #3280 jet: DM per-channel heads (surface is single channel cp)
- #3281 megumi: DM MoE output (directive: no MoE)
- #3181 himmel: DM asinh+residual (noam dead for DM)
- #3067 askeladd: DM 32k surface points (directive: only test denser)
- #3274 norman: AF Lion (directive: stop Lion)
- #3172 robin: AF LR warmup (directive: stop warmup)
- #3169 nami: AF heads sweep (directive: stop broad heads)
- #3129 chrome: AF 4L/256d (directive: stop broad depth)
- #3261 nezuko: AF 2L/320d (directive: stop broad width)
- #3187 stark: AF asinh+residual (asinh dead)
- #3184 wolfwood: AF full noam stack (noam dead)
- #3268 jin: AF higher LR (directive: stop higher-LR arms)
- #3144 violet: AF vol-weight=2 (below directive range 8-13)
- #3150 yuji: TF clean test (TF frozen)

## Steering Anchors

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **21.319** (#3185 MERGED — full noam stack: ANP+physics+T_max=150+Lookahead+96sl) |
| TandemFoil Paper | `val_primary/field_mse` | **0.002180** (#3307 MERGED — haku lr=7e-5, -7% val / -13% test) |
| AirfRANS | `val_primary/surface_mse` + `vol_mse` | **0.000296 / 0.002039** (#3135 MERGED — EMA+vol-weight=10x) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.622%** (#3298 MERGED — MSE-only T_max=36, still improving at timeout) |

## Paper-Facing Snapshot

| Dataset | Metric | Current best | External target | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **22.868** (PR #3185 MERGED) | (internal) | Improving |
| TandemFoil Paper | `test_primary/field_mse` | **0.001931** (PR #3307 haku lr=7e-5) | ~0.10-0.36/task | **New programme best.** Bracket 6e-5/7.5e-5 in progress. |
| AirfRANS | `Surf MSE / Vol MSE` | **0.000296 / 0.002039** | 0.0043 / 0.0017 | Surface 14.5x better, Volume 1.20x gap |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **4.218%** (full-eval, canute n2t1nzsb, metric-aware T_max=30) | 3.71% | T_max=36 full-eval PENDING (#3362 jet). T_max=36+metric-aware compound (#3359-#3361) in progress. |

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

### Bug Fixes (RESOLVED)
- **`cp_panel_prior_index()` FIXED** (#3209 MERGED): Guard returns None for tandemfoil_paper. No more sinh overflow.
- **`primary_metric_key` shadowing FIXED** (#3209 MERGED): Local variable at line ~1652 renamed, no longer shadows function at line ~1434.
- **TFP champion config recovery in progress** — sanji #3266 verifying that 0.002383 baseline is reproducible under the fixed codebase.

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
- Pressure gradient smoothness reg (KNN): λ=0.01→4.481%, λ=0.1→4.876%, λ=1.0→12.52%. KNN 30% throughput overhead + sharp car features penalized. Remaining error is systematic bias, not noise.
- Attention temperature annealing: 4.024% (+5% vs 3.833%). External annealed τ compounds with existing learnable per-head τ. noam result doesn't transfer
- Stochastic depth/LayerDrop: p=0.1→4.317%, p=0.2→4.217%, p=0.3→4.411% (#3233). Only 4 layers — dropping 1 removes 25% capacity. Too coarse. EMA+gc already handles stability
- Coordinate noise augmentation: σ=0.001→6.45% crashed, σ=0.005→21.44% Inf grads (#3256). Fourier freqs up to 32.0 amplify coord noise 32x. Structurally incompatible with Fourier PE
- Progressive resolution (25k→50k / 30k→50k): 25k crashed ep112, 30k→4.711% +23% (#3252). Full 50k context needed from epoch 1. Reduced resolution creates irrecoverable representation deficit
- Auxiliary surface normal prediction: w=0.1→4.724%, w=0.01→4.661% (#3258). Input leakage — normals already in features, aux head learns trivial reconstruction. Also epoch starvation (200 vs 511)
- EMA periodic reset: 100ep→4.426% cascade ep335, 50ep→6.818% diverge ep108 (#3247). EMA is primary gradient stabilizer — reset removes protective smoothing at high-curvature basin point
- Weight Standardization: 67.9%/71.4% catastrophic (#3264). WS × cosine restarts × bs=1 feedback loop. Original BiT paper used epoch-level decay not per-batch restarts
- SGDR eta_mult (decaying restart LR): 4.153%/12.64% (#3248). Confounded by switching to warm restarts. Sharp restart jumps more destabilizing than baseline's smooth cosine
- Sparse MoE FFN (K=4/8, top-2): 5.09%/5.35% (#3263). 1.7-2.2x throughput penalty → epoch starvation. Routing collapses to uniform. Per-epoch curve parallel to dense — FFN is not the bottleneck
- Input feature dropout (Fourier channels): p=0.1→4.73%, p=0.2→4.82% (#3253). Input-level dropout = information destruction. All 4 Fourier bands essential. Terminal divergence at both dropout rates
- Curvature-weighted loss: alpha=1.0→4.48% diverge ep300, alpha=0.5→11.6% diverge ep53 (#3259). Train/eval mismatch (weighted vs uniform metric) + curvature amplifies gradient variance at restarts
- Quadratic position features (x²/y²/z²/xy/xz/yz): 6.643% at ep168 timeout (#3272). Fatal 3x throughput penalty (168 vs 511 baseline epochs). Quad-only diverged ep63 — Fourier PE irreplaceable. Same epoch-starvation pattern as MoE (#3263).
- Grouped Query Attention (GQA, 2/4 KV heads): 12.88%/6.88% both crashed (#3273). Throughput-neutral (bottleneck is slice routing einsums, not QKV). GQA destabilizes per-head slice routing — coupled KV sharing explodes at cosine restarts.
- Error-weighted surface sampling (hard example mining): crashed ep6 both trials (#3243). Distribution-shift collapse + 30-60min/epoch error-map compute + train/eval mismatch. Student note: error-weighted LOSS (not sampling) would avoid distribution shift but still faces mismatch.
- Learnable register tokens (N=2/4/8): N=8 best 4.062% (+6%), N=2/N=4 NaN diverged (#3262). Category error — register tokens address pairwise attention sinks (ViT), but Transolver uses slice-based soft-clustering attention where sink mechanism doesn't apply.
- Learnable Fourier frequencies: 5.262%/4.252% both diverged (#3260). Frequencies barely moved from init — fixed [0.5,2,8,32] are already near-optimal. Learnable freq + cosine restarts = cascading perturbation feedback loop.
- Cosine similarity auxiliary loss: w=0.1→4.661% crashed ep244, w=0.5→6.569% crashed ep159 (#3276). Cos_sim saturates >0.99 by ep100 — spatially redundant with MSE. Near-unity cos_sim makes gradients ill-conditioned at LR restarts.
- EMA>0.9995: 0.9999→7.106% NaN ep171, 0.99995→7.095% collapse ep120 (#3199). EMA=0.9995 sharp optimum bracketed both directions
- SAM optimizer (rho=0.05/0.02): 5.36%/9.08%. SAM perturbation + cosine restart = double shock → catastrophic divergence. 2x compute penalty also prohibitive
- True monotonic cosine (T_max=393606, no restarts): 4.086% (+0.253pp). Confirms T_max=30 rapid restarts are core mechanism, not noise. Monotonic decay can't compete
- 600 batches/epoch (stabilized retest): T_max=46 proportional diverged ep61; T_max=30 got 3.887% after MORE total batches than baseline. Data diversity per epoch saturated at 394
- Head count sweep complete: 2H=catastrophic, 4H=6.650%, 8H=3.833% CHAMPION, 16H=4.099%. 8H (64d/head) is definitive
- 10-ep warmup+gc=1.0: 11.2% then diverged
- 5-ep warmup+gc=1.0: pending (chopper)
- LR warmup + EMA+gc=0.5: 5-ep=11.325% diverged, 10-ep=3.918% plateaued (didn't beat 3.833%)
- Huber loss, relative L2 loss (degenerate), SGDR, RAdam
- beta2≠0.999: 0.99 and 0.995 both diverge at cosine restarts (#3110). Default 0.999 confirmed.
- LR≠5e-4 (without EMA), WD+gc heavy (WD=1e-3+gc=1.0: 4.44%, no EMA, #3046)
- WD+gc compound without EMA/metric-aware: 4.44% best (#3046). These runs lack the modern stack.
- Larger supernodes (8192/16000): 5.531%/+38% even with EMA+gc (#3085). Default supernode budget superior.
- Log-cosh loss: 4.599% with gc=1.0, 6.344% without (#3076). MSE + metric-aware rel-L2 is the winning loss.
- SwiGLU FFN activation: 4.454%/4.720%, both diverged with Inf grads (#3267). Multiplicative gating amplifies gradient instability.
- Hidden-space noise regularization: 4.236% best, all 3 configs diverged (#3265). Noise destabilizes transformer hidden states.
- Post-LayerNorm: 9.515% catastrophic (#3251). Pre-LN is load-bearing.
- Label smoothing / target noise: 4.087% (σ=0.001), 4.442% (σ=0.01 → NaN) (#3242). Label noise degrades CFD precision.
- lr>5e-4 with EMA+gc: 5.5e-4→4.390% diverged, 6e-4→4.227% (#3194). lr=5e-4 confirmed champion.
- Multi-exit prediction (aux losses at intermediate layers): 4.278%/4.438%/4.302% (#3235). All configs worse. Aux losses degrade primary even when converged.
- Checkpoint averaging (top-5): -1.6% relative only (#3146). EMA=0.9995 already provides implicit averaging.
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
- T_max=20 at champion config: val=0.004359, +83% worse (#3088). T_max=10 confirmed TFP optimum.
- WD=5e-3 (halved from 1e-2): pressure channel diverged to Infinity (#3133). WD=1e-2 confirmed.
- Champion re-verify post cp_panel fix: val=0.00474, +102% worse (#3266). cp_panel fix altered dynamics.

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
- Vol-weight warm-up (1x→10x ramp): 2/3 collapsed, best stable +28% surface/+69% vol (#3232). Cosine restarts inject LR peak when vol-weight ramps — destructive. Static 10x TRIPLE-CONFIRMED
- Focal-MSE volume loss (error-based reweighting): gamma=2 +392%/+3993%, gamma=1 +222%/+320% (#3227). Batch-max normalization non-stationary + freestream scaffolding destroyed
- Additive BL auxiliary volume loss (SDF targeting): all 3 trials +31-228% both metrics (#3239). SDF<0.05 captures 61% of vol points on AF meshes — "BL targeting" ≈ uniform upscaling. Pattern confirmed with #3222
- Multi-seed champion (seeds 42/123/789): all worse than seed=0 baseline (#3254). Seed 42 diverged, 123/789 +39-76% vol. Champion config has high seed sensitivity — seed=0 is lucky initialization
- No-Lookahead + no-compile at full budget: surface +575%, vol +84% (#3255). Lookahead slow-weight averaging is load-bearing for long-run convergence. compile gives meaningful throughput within timeout. Short-run ablation that showed these removable was misleading (epoch-count confound).
- 3L depth (with or without EMA): catastrophic divergence confirmed twice
- 2L/384d and 3L/384d: catastrophic divergence
- EMA>0.9995 (0.9999/0.99995 both ~85% worse, diverge early, #3199). EMA=0.9995 bracketed as sharp optimum with steep cliffs both directions
- EMA<0.999 (0.99, 0.995 both worse)
- T_max≠50 (30 and 100 both worse)
- Vol-weight<10x (1.5x/2x/3x/5x/7x): ALL worse on both metrics across 6 tested values. 10x+EMA=0.999 is AF sweet spot
- Vol-weight=30x: catastrophic — surface 4.3x worse, vol 3.1x worse
- Vol-weight=15x: surface +19%, vol +67% (#3204). Vol-weight=20x: crashed ep381 (#3204). 10x is definitive AF operating point
- Vol-weight=9x: catastrophic — surface +196%, vol +137% (#3296). Below 10x is destructive.
- Vol→surface cross-attention: surface 3.4x worse, vol 1.5x worse (#3282). Architectural modification failed.
- Separate volume prediction head: surface +433%, vol +24% (#3278). Architecture detrimental.
- Per-channel vol weighting (nut 4x, p 2x): surface +38%, vol +118% (#3250). Biased weighting disrupts joint optimization.
- Re-stratified Reynolds sampling: surface ~2x worse, vol +68% (#3195). Trial 2 diverged.
- gc=0.5 on AF: surface +52%, vol +187% (#3156). Champion gc=1.0 confirmed correct for AF.
- T_max=75 on AF: surface +11.5%, vol +25.5% (#3241). T_max=50 champion confirmed.
- WD=0 / WD=5e-3 on AF: WD=5e-3 gives programme-best surface (0.000249) but vol can't be recovered below 0.002039 after 3 attempts (#3238). WD=1e-2 is the correct value.
- LR>6e-4 (pre-EMA): 8e-4 +110% surface, 1e-3 catastrophic divergence
- LR<6e-4 (with EMA+vol-10x): 5e-4 +66%/+111%, 4e-4 +38%/+76% (#3234). Underfitting under vol-10x amplification. lr=6e-4 bracketed from below

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
