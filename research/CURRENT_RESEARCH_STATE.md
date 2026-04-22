# SENPAI Research State

- **Date:** 2026-04-22 (last polled — Wave 3 + TF Paper baseline wave + DM Lion + AF LR sweep + cross-dataset spatial/physics budget sweeps + Wave 4 cross-dataset code/arch innovations + Wave 5 cross-dataset code/arch innovations + Wave 6 per-step SGDR / coord noise / T_max sweep + Wave 7 attention architecture innovations + usopp #2994 hypernetwork + Wave 8 spike #2995 attention dropout + Wave 9 brook #2999 LayerScale + zenitsu #3000 Spectral Norm + Wave 10 #3001–#3013 (13 new PRs) + Wave 11: griffith sigma-Reparam + casca GeGLU — CLOSED: griffith #2968 Spectral Norm attn, casca #2962 AGC, usopp #3012 AdamW lr ablation (cross-dataset, all worse), mugen #2980 PCGrad Gradient Surgery (torch.compile incompatible) — Wave 12: einar #3021 LayerScale residuals + wolfwood #3022 attn dropout + emma #3023 SDF wall-distance + shoya #3024 LLRD + haku #3025 TFP Lion champion config + usopp #3026 AirfRANS lr fine-tune sweep — Wave 13: mugen #3027 DrivAerML surface-points sweep + jet #3028 AdamW β1/β2 sweep — Wave 14: megumi #3029 best-checkpoint saving cross-dataset — Wave 15: violet #3030 Relative L2 training loss (TF+DM, two variants) — Wave 16: alphonse #3031 Pre-LN + fern #3032 RMSNorm + hinata #3033 linear warmup + kakashi #3034 gradient centralization + tanjiro #3035 SGDR T_mult=2 (thorfinn #3036 AdaFactor never launched — re-assigned as Wave 17 #3041) — Wave 17: faye #3038 surface-normals-curvature + kohaku #3039 mass-conservation-aux-loss + chihiro #3040 moe-ffn-layers + thorfinn #3041 adafactor-optimizer — Wave 18: emma #3042 SAM-optimizer-cross-dataset — CLOSED: megumi #3009 Lion lr sweep cross-dataset, confirmed Lion dead end across all datasets/LRs; violet #2955 Stochastic Depth/DropPath all datasets negative — confirmed dead end)
- **Branch:** radford
- **Fleet status:** 72 WIP PRs — Wave 18 NEW: emma #3042 SAM-optimizer-cross-dataset — Wave 17 NEW: faye #3038 surface-normals-curvature, kohaku #3039 mass-conservation-aux-loss, chihiro #3040 moe-ffn-layers, thorfinn #3041 adafactor-optimizer (re-assignment of stale Wave 16 #3036) — Wave 16: alphonse #3031 Pre-LN normalization, fern #3032 RMSNorm, hinata #3033 linear warmup, kakashi #3034 gradient centralization, tanjiro #3035 SGDR T_mult=2 — violet #3030 Relative L2 training loss TF+DM (Wave 15) — megumi #3029 best-checkpoint-save cross-dataset (Wave 14) — mugen #3027 DrivAerML surface-points sweep (Wave 13) + jet #3028 AdamW β1/β2 sweep (Wave 13) — Wave 12 NEW: haku #3025 TFP Lion+gc=0.5+EMA champion config, usopp #3026 AirfRANS lr sweep (7e-4/8e-4/9e-4) — Wave 12: einar #3021 LayerScale residuals cross-dataset, wolfwood #3022 attn dropout cross-dataset, emma #3023 SDF wall-distance feature cross-dataset, shoya #3024 LLRD cross-dataset — Wave 11: griffith #3016 sigma-Reparam, casca #3017 GeGLU — Wave 7: chrome #2988 MQA, faye #2989 GQA, gojo #2990 head-dim scaling, himmel #2991 SWA, levi #2992 Flash+compile, shoya #2993 sparse top-k; usopp #2994 hypernetwork; Wave 8: spike #2995 attention dropout; Wave 9: brook #2999 LayerScale on Transolver residuals (cross-dataset TF/TFP/AF/DM); Wave 10: canute #3001 slice temp sweep, franky #3002 Fourier freq bands, norman #3003 channel dropout, sanji #3004 long cosine, robin #3005 2L+EMA+gc=0.5, senku #3006 2L+EMA+gc=0.3, shouko #3007 4L+T_max=10, stark #3008 AF depth+T_max, piccolo #3010 DM gc sweep, sukuna #3011 WD sweep, shoya #3013 Fourier ablation. MERGED: zenitsu #2997 Kutta TE v2 (cross-dataset), brook #2998 AF normalization 3-way, haku #2979 TFP first baseline (0.00434). PRs CLOSED: #2968 (griffith, Spectral Norm attn), #2962 (casca, AGC), #2981 (spike, QK attention temp), #2977 (zenitsu, learnable per-head attn temperature), #2996 (brook, MQA draft — superseded), #3012 (usopp, AdamW lr ablation — all worse than baselines), #2980 (mugen, PCGrad — torch.compile incompatible → 75× worse on AF), #3009 (megumi, Lion lr sweep — CONFIRMED DEAD END, catastrophic AF/DM instability at all LRs). NOTE: zenitsu #2983 RoPE dead end was closed. griffith #2889 (3L/512d+gc=1.0 DM) STILL IN FLIGHT.
- **Current relaunch budget:** inherit pod env defaults
  - `SENPAI_TIMEOUT_MINUTES=360`
  - `SENPAI_MAX_EPOCHS=999`

## CORE RESEARCH DIRECTIVE (Human Team Instruction — 2026-04-21)

**CROSS-DATASET GENERALIZATION IS THE PRIMARY CONSTRAINT ON ALL NEW EXPERIMENTS.**

The human research team has explicitly directed: every new hypothesis must be
tested across all relevant datasets in a single PR. Dataset-specific tricks that
do not transfer are not useful. The paper story requires a shared recipe.

All four benchmarks are now active and required:

1. **TandemFoil** — `val_primary/surface_pressure_mae` / `test_primary/surface_pressure_mae`
2. **TandemFoil Paper** — `val_primary/field_mse` / `test_primary/field_mse` ← NEW 4th dataset (human directive 2026-04-21)
3. **AirfRANS** — `val_primary/surface_mse` / `test_primary/surface_mse`
4. **DrivAerML** — `val_primary/surface_rel_l2_pct` / `test_primary/surface_rel_l2_pct`

**Assignment rule (effective immediately):**
- New hyperparameter or architecture hypotheses MUST be tested on ALL four datasets in one PR.
- Single-dataset assignments are only acceptable for dataset-specific ablations (e.g. DrivAerML batch size), TandemFoil Paper baseline runs, or targeted best-checkpoint recovery.
- When in doubt: assign cross-dataset. An idea that only helps one dataset is a dataset hack.

## Paper-Facing Snapshot

| Dataset | Paper-facing metric | Current best | Target / reference | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **33.88** (#2810) | no external scalar | needs test from new EMA best |
| TandemFoil Paper | `test_primary/field_mse` | **not run yet on radford** | `0.10 / 0.18 / 0.36 / 0.13 / 0.14 / 0.21` by task | NEW — baseline run needed |
| AirfRANS | `test_primary/surface_mse` | **0.003** (#2824) | `0.0043` | **BEATEN** — val now 0.000727 |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **6.24%** (#2691) | `3.71%` | **MAIN GAP — 1.68x** |

## Steering Anchors (validation, for experiment decisions)

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **22.537** (#2924 MERGED — Lion lr=1e-4, gc=0.5, EMA=0.999, 3L/192d/3H) |
| TandemFoil Paper | `val_primary/field_mse` | **not established** — baseline run needed urgently |
| AirfRANS | `val_primary/surface_mse` | **0.000482** (#2951 MERGED — AdamW lr=6e-4, T_max=50, no-EMA, 2L/256d/4H) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **3.997%** (#2898 MERGED — AdamW lr=5e-4, T_max=30, no-EMA, 4L/512d/8H) |

### CRITICAL SIGNAL — Best-Checkpoint Saves (2026-04-22)

PR #2895 (mugen, T_mult cosine restarts) found **transient** bests:
- TF: **25.459** @ ep109 (vs 26.06 current) — 2.3% improvement visible in val curve!
- AF: **0.000371** @ ep221 (vs 0.000627 current) — **40.8%** improvement!

Both were erased by post-restart regression. The final checkpoint is worse than baseline.
**The loss landscape has much deeper minima than our current results suggest.**
**Best-checkpoint saving (save whenever val improves) is now a CRITICAL CODE CHANGE.**

## Main Scientific Goal

A shared recipe whose core changes work across all four benchmarks:

- the main TandemFoil parity target
- the new TandemFoil paper-calibration target (Experiment 4, Table 6)
- AirfRANS
- DrivAerML

DrivAerML is still the main gap. Corrected EMA warmup is now the shared recipe
for TF+AF. TandemFoil Paper is a NEW required benchmark added by the human team
— paper-faithful high-Re TandemFoilSet using normalized full-field MSE, 6 tasks.
No baseline has been run yet on radford for this dataset.

## Mandatory Config Rules (UPDATED after EMA merge)

- **TF + AF + TF_paper:** Use `--ema-decay 0.999` (NO --no-use-ema). decay=0.999 > 0.9999 on both.
- **DrivAerML:** Still `--no-use-ema` (EMA alone hurt DM; zenitsu #2925 tests EMA+gc compound)
- `--epochs 999` mandatory
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`
- Lion for TandemFoil; AdamW for AirfRANS/DrivAerML

## Critical Code Change Needed: Best-Checkpoint Saving

**Evidence:** #2895 (mugen T_mult) saw TF=25.459 and AF=0.000371 as transient minima, both clearly beating current baselines, but post-restart LR jumps erased both. The trainer currently saves the **final** checkpoint, not the **best val** checkpoint.

**Impact:** Every experiment with cosine warm restarts (T_mult, SGDR, multi-cycle) is leaving improvement on the table. The minima exist — we just don't capture them.

**Required trainer change:** After each validation epoch, if `val_primary_metric < best_val_so_far`, save `checkpoint_best.pt`. At the end of training, load and evaluate from `checkpoint_best.pt` (not the final epoch).

**Priority:** HIGH — **ASSIGNED to megumi, PR #3029** (`megumi/best-checkpoint-save-cross-dataset`). Wave 14.

## Negative Results (Do Not Repeat)

- **No-Lookahead** (#2834): fatal across all datasets — diverges AF/DM, TF regresses
- **3L/192d on TF or DM** (#2825): too shallow at current scale
- **Pressure-weighted loss at wrong architecture** (#2801): pressure weighting hurts at non-golden depth
- **LR above 5e-4 on DrivAerML** (#2873): 6e-4/5.5e-4/4.5e-4 all worse, LR optimum firmly at 5e-4
- **surface_only_drivaerml is already default** (#2900): use --no-surface-only-drivaerml to test volume
- **lr=9e-4 is above DrivAerML ceiling** (#2907): diverged catastrophically
- **Momentum-SAM (MSAM)** (#2904): 3.7-22.7x worse everywhere; cost is actually 2x
- **gc=1.0+WD=1e-2+cosine compound on DM** (#2908): 6/8 runs crashed
- **4L/640d at lr=5e-4+gc=1.0** (#2917): all 3 DM runs crashed (ep90-153)
- **T_max=5 on DrivAerML** (#2911): all 3 runs diverged — too rapid for 4L/512d scale
- **EMA alone on DrivAerML** (#2899): 9.749% (worse than 4.619% baseline) — EMA+gc compound now being tested
- **gc=1.5/gc=2.0 on DrivAerML** (#2881): all 8 runs diverged. gc=1.5 best 6.066% (+31%), gc=2.0 best 8.834% (+91%). CosineAnnealing restarts amplify instability
- **T_max=10+gc=1.0 on DrivAerML** (#2879 bulma): failed — no sustained improvement over baseline 4.619%
- **T_max=40/50+gc=1.0+WD=1e-2 on DrivAerML** (#2919 wolfwood): failed — longer cosine cycling did not help; WD+gc compound still unstable
- **Gradient noise injection cross-dataset** (#2920 usopp): fatal — systematic catastrophic instability on all datasets; incompatible with cosine annealing
- **AirfRANS gradient accumulation** (#2902 stark accum>1): strictly detrimental — accum=1 (control) trained longest (661 ep) and found best basin; accumulation hurts AF
- **Huber loss on AirfRANS** (#2901 spike, early read): 58.7% WORSE than MSE at same epoch — Huber δ=1.0 is NOT beneficial for AF; final verdict pending completion
- **Learnable per-head QK attention temperature** (#2981 spike, CLOSED): all 4 datasets 200-2844% worse than baseline. Hypothesis falsified — Transolver already has `self.temperature=0.5` for slice assignment (more important than QK scaling); learned QK temperatures converged near 1.0 (±15%), confirming standard 1/√d_k scaling is near-optimal. Mechanism analysis: QK temperature is downstream of the architecturally meaningful slice-assignment temperature.
- **Spectral Norm on attention Q/K/V/O projections** (#2968 griffith, CLOSED): `torch.nn.utils.spectral_norm` on all 4 attention projections — 17.5% worse on TF, 55% worse on AF, DM diverged. SN constrains expressivity, conflicts with Fourier features, incompatible with torch.compile (legacy forward pre-hook API), and DM divergence from SN + cosine cycling instability (sigma collapse/oscillation during LR warm phases). sigma-Reparam (Zhai et al. ICML 2023) is the clean alternative (`W = g * V / ||V||_σ`).
- **Adaptive Gradient Clipping / AGC** (#2962 casca, CLOSED): NFNet-style AGC at clip_factor=0.01 and 0.03 — fundamentally incompatible with Lion optimizer (Lion sign-compression makes g_norm = ||sign(m)|| = sqrt(num_params) constant, bypassing the adaptive component entirely); DrivAerML diverged at both clip_factor values. Global gc=0.5 outperforms AGC everywhere tested.
- **PCGrad Gradient Surgery** (#2980 mugen, CLOSED): `retain_graph=True` (required for dual `.backward()` calls) is incompatible with PyTorch 2.10 torch.compile donated buffer optimization — forces `--no-compile-model` → 6–8× throughput collapse → AF at epoch 23 was 75× worse than baseline (0.036 vs 0.000482), TFP all-NaN. Physical motivation was valid (35% AF / 23% TF conflict rate) but implementation is fatally constrained. Future path: `torch.autograd.grad()` instead of `retain_graph=True`.

## Default Assignment Pattern

Cross-dataset is now the DEFAULT. Every new hypothesis should cover:
- `target/icml2026/tandemfoil/`
- `target/icml2026/tandemfoil_paper/`
- `target/icml2026/airfrans/`
- `target/icml2026/drivaerml/`

Unless there is a strong reason to restrict to a single dataset (ablation,
baseline run, known dataset-specific mechanism), all assignments are
multi-dataset. This is the human team's explicit directive.

When a hypothesis is relevant to TandemFoil generalization or paper
comparability, always include:
- `target/icml2026/tandemfoil/`
- `target/icml2026/tandemfoil_paper/`

## ACTIVE EXPERIMENTS — 72 WIP PRs (updated 2026-04-22 after Wave 18: emma #3042 SAM optimizer cross-dataset)

### Theme 25: Wave 18 — Sharpness-Aware Minimization (SAM) Optimizer Cross-Dataset (NEW — 2026-04-22)

**Scientific rationale:** SAM (Foret et al. ICLR 2021) finds flat minima in the loss landscape by performing a two-step update: first perturb weights to the local worst-case direction (ε = rho * g / ||g||), then compute gradients at the perturbed point and take the actual update step, restoring original weights. Flat minima correlate strongly with better generalization — particularly OOD generalization. CFD surrogates have large distribution shifts between in-distribution (Re, AoA) and OOD splits. SAM costs ~2× compute per step (two forward+backward passes), but on 96 GB VRAM GPUs with our small batch sizes the wall-clock overhead is acceptable. Prior negative result #2904 (MSAM / Momentum-SAM) does NOT apply here — MSAM had catastrophic 3.7–22.7× regressions, but MSAM is a different variant that reuses gradients across steps (cheaper but fundamentally different). This PR tests the original SAM formulation with a bespoke wrapper class and tests two perturbation radii (rho=0.05 / rho=0.1) to find the optimal regularization strength.

**Note on MSAM vs SAM:** MSAM (#2904) failure was due to stale gradient approximation, not SAM itself. Classic SAM with proper two-step (fresh gradients at perturbed point) has not been tested.

| Student | Branch | PR | Hypothesis | Runs |
|---|---|---|---|---|
| emma | `emma/sam-optimizer-cross-dataset` | #3042 | **SAM optimizer** — SAM(Lion) for TF (rho=0.05/0.1), SAM(AdamW) for AF/DM/TFP (rho=0.05/0.1); training loop: clip→first_step→second forward+backward→clip→second_step | 7 runs: TF×2 + AF×2 + DM×2 + TFP×1 (optional) |

**Baselines to beat (all 4 datasets):**
- TF: `val_primary/surface_pressure_mae` = **22.537** (#2924)
- TFP: `val_primary/field_mse` = **0.00434** (#2979/haku baseline)
- AF: `val_primary/surface_mse` = **0.000482** (#2951)
- DM: `val_primary/surface_rel_l2_pct` = **3.997%** (#2898)

**CLI flags needed:** `--optimizer sam_lion`/`--optimizer sam_adamw`, `--sam-rho 0.05` (or 0.1)

**W&B group:** `sam_cross_dataset`

**Key risk:** 2× compute budget means fewer epochs in same wall-clock time. Monitor early epoch counts vs baseline to ensure convergence is not cut short by timeout.

### Theme 24: Wave 17 — Geometry Features, Physics Constraints, Architecture Capacity, Optimizer Alternatives (NEW — 2026-04-22)

**Scientific rationale:** Wave 17 attacks four distinct frontiers simultaneously: (1) enriching the input representation with surface geometry features (normals + curvature) that encode boundary physics not available from raw coordinates; (2) imposing the divergence-free constraint as an auxiliary loss to regularize velocity predictions via mass conservation; (3) increasing model capacity through sparse Mixture-of-Experts FFN layers in the final Transolver blocks; and (4) testing AdaFactor as a memory-efficient optimizer alternative that frees VRAM for larger effective batch sizes. All 4 students run all 4 benchmarks (TF, TFP, AF, DM) in a single PR.

| Student | Branch | PR | Hypothesis | Flag |
|---|---|---|---|---|
| faye | `faye/surface-normals-curvature` | #3038 | **Surface normals + principal curvature** — PCA-based normal estimation (k=16 NN) + trimesh discrete principal curvatures (arcsinh-scaled); zero-init extra weight columns in first linear; GPU split: DM=4, TF=2, TFP=1, AF=1 | `--surface-normals --surface-curvature` |
| kohaku | `kohaku/mass-conservation-aux-loss` | #3039 | **Mass conservation auxiliary loss** — divergence via KNN k=8 Green-Gauss FD; `div_loss = (div_per_point^2).mean()`; applied only to velocity outputs; `total_loss = data_loss + 0.01 * div_loss`; GPU split: DM=4, TFP=2, AF=2 | `--div-loss-weight 0.01` |
| chihiro | `chihiro/moe-ffn-layers` | #3040 | **Sparse MoE FFN in last Transolver blocks** — 8 experts, top-2 routing, load-balance loss; apply to last 1 block only; GPU split: DM=4, TF=2, TFP=1, AF=1 | `--moe-layers 1 --moe-n-experts 8 --moe-load-balance-weight 0.01` |
| thorfinn | `thorfinn/adafactor-optimizer` | #3041 | **AdaFactor optimizer** — factored second-moment (Shazeer & Stern 2018); fixed-LR mode (`scale_parameter=False`, `relative_step=False`); re-assignment of Wave 16 PR #3036 which never launched; GPU split: DM=4, TF=2, TFP=1, AF=1 | `--optimizer adafactor` |

**Baselines to beat (all 4 datasets):**
- TF: `val_primary/surface_pressure_mae` = **22.537** (#2924)
- TFP: `val_primary/field_mse` = **0.00434** (#2979/haku baseline)
- AF: `val_primary/surface_mse` = **0.000340** (#2898)
- DM: `val_primary/surface_rel_l2_pct` = **3.997%** (#2898)

**Per-dataset mandatory config (students must follow exactly):**
- TF: Lion optimizer, `--ema-decay 0.999`, `--cosine-t-max 10`
- TFP: AdamW, `--no-use-ema`, `--cosine-t-max 150`
- AF: AdamW, `--no-use-ema`, `--cosine-t-max 50`, `--grad-clip 1.0`, `--weight-decay 1e-2`
- DM: AdamW, `--no-use-ema`, `--cosine-t-max 30`, `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`; requires `SENPAI_MAX_EPOCHS=9999`

**Note on Wave 16 thorfinn #3036:** This PR was originally assigned in Wave 16 but never launched (no commits, no pod activity). The hypothesis has been re-assigned to thorfinn as Wave 17 PR #3041 with the same AdaFactor hypothesis. #3036 can be considered superseded/stale.

### Theme 23: Wave 16 — Normalization, Warmup, and Optimizer Foundations (NEW — 2026-04-22)

**Scientific rationale:** Wave 16 targets the foundational training mechanics that underpin all prior experiments but have not been systematically explored: transformer normalization placement (Pre-LN vs Post-LN), normalization variant (RMSNorm vs LayerNorm), LR schedule warmup, gradient processing (gradient centralization), cosine restart schedule shape (SGDR T_mult=2 with best-checkpoint saving from megumi #3029), and memory-efficient second-moment optimization (AdaFactor). These are all cross-dataset hypotheses — each student runs all 4 benchmarks (TF, TFP, AF, DM) in a single PR.

| Student | Branch | PR | Hypothesis | Flag |
|---|---|---|---|---|
| alphonse | `alphonse/pre-ln-normalization-order` | #3031 | **Pre-LN normalization** — LayerNorm before residual addition (Pre-LN) instead of after (Post-LN); stabilizes gradient flow in deep transformers | `--pre-ln` |
| fern | `fern/rmsnorm-layer-norm` | #3032 | **RMSNorm** — drop mean-centering from LayerNorm, compute only RMS scale; faster, simpler, used in LLaMA/Gemma | `--rmsnorm` |
| hinata | `hinata/linear-warmup-scheduler` | #3033 | **Linear warmup + cosine** — 10-epoch linear ramp to peak LR, then CosineAnnealing; prevents cold-start instability in first epochs | `--warmup-epochs 10` |
| kakashi | `kakashi/gradient-centralization` | #3034 | **Gradient centralization** — subtract per-tensor gradient mean before optimizer step; smooths loss landscape, acts as implicit weight regularizer | `--grad-centralization` |
| tanjiro | `tanjiro/sgdr-tmult-best-checkpoint` | #3035 | **SGDR T_mult=2** — CosineAnnealingWarmRestarts with doubling restart intervals; uses best-checkpoint saving (megumi #3029) to capture transient minima at each restart | `--cosine-t-mult 2` |
| ~~thorfinn~~ | ~~`thorfinn/adafactor-optimizer`~~ | ~~#3036~~ | ~~**AdaFactor optimizer** — factored second-moment estimation (Shazeer & Stern 2018); memory-efficient alternative to AdamW; run with fixed LR mode~~ (**NEVER LAUNCHED — superseded by Wave 17 PR #3041**) | ~~`--optimizer adafactor`~~ |

**Baselines to beat (all 4 datasets):**
- TF: `val_primary/surface_pressure_mae` = **22.537** (#2924)
- TFP: `val_primary/field_mse` = **0.00434** (#2979/haku baseline)
- AF: `val_primary/surface_mse` = **0.000340** (#2898)
- DM: `val_primary/surface_rel_l2_pct` = **3.997%** (#2898)

**Per-dataset mandatory config (students must follow exactly):**
- TF: Lion optimizer, `--ema-decay 0.999`, `--cosine-t-max 10`
- TFP: AdamW, `--no-use-ema`, `--cosine-t-max 150`
- AF: AdamW, `--no-use-ema`, `--cosine-t-max 50`, `--grad-clip 1.0`, `--weight-decay 1e-2`
- DM: AdamW, `--no-use-ema`, `--cosine-t-max 30`, `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`; requires `SENPAI_MAX_EPOCHS=9999`

**Note on tanjiro #3035 (SGDR T_mult=2):** This experiment requires megumi's best-checkpoint saving (#3029) to land first in order to properly capture transient minima at restart boundaries. If #3029 is still in-flight, tanjiro should wait or run with terminal checkpoint understanding the minima may be missed.

### Theme 22: Wave 15 — Relative L2 Training Loss (violet #3030, NEW — 2026-04-22)

**Scientific rationale:** DrivAerML trains on absolute MSE but evaluates on `surface_rel_l2_pct` — a relative (normalized) L2 metric. This objective mismatch means the model is never directly optimized for the evaluation criterion. Hypothesis: using a relative L2 loss (or a 0.5/0.5 mixed MSE+rel_L2) during training will align gradient signal with the eval metric and reduce DM surface_rel_l2_pct. TandemFoil also sees relative error at evaluation; frieren #2937 (Wave 3) is testing the same idea cross-dataset — violet's Wave 15 is a focused two-dataset variant with more explicit variant exploration.

**Relative L2 formula:** `((pred - target).norm(dim=-1) / (target.norm(dim=-1) + 1e-8)).mean()` — normalizes per-node error by node-level target magnitude; eps=1e-8 prevents division by zero on boundary/zero-velocity nodes.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| violet | #3030 | **Relative L2 training loss** — 4 runs: DM-rel_l2 / DM-mixed (0.5MSE+0.5rel_L2) / TF-rel_l2 / TF-mixed; `--loss rel_l2` or `--loss mixed`; `--wandb_group wave4-rel-l2-loss` | LOW |

**Baselines to beat:**
- TF: `val_primary/surface_pressure_mae` = **22.537** (#2924)
- DM: `val_primary/surface_rel_l2_pct` = **3.997%** (#2898)

**Divergence fallback:** If DM rel_l2 diverges after 20 epochs, add `--grad-clip 1.0`.

**Note:** TandemFoil Paper (TFP) excluded due to data environment issue (DrivAerML-format pickle files on TFP mount causing `ValueError: Expected paper-style tandem AoA to be shared`). AirfRANS excluded because it evaluates on surface_mse (absolute), not a relative metric — no objective mismatch to fix.

### Theme 21: Wave 13 — DrivAerML Targeted Ablations + Optimizer Hyperparameter Sweep (NEW — 2026-04-22)

| Student | PR | Experiment | Risk |
|---|---|---|---|
| mugen | #3027 | **DrivAerML surface-points sweep** — test 16k/32k/64k vs 50k baseline; isolate whether 50k is optimal | LOW/MEDIUM |
| jet | #3028 | **AdamW β1/β2 sweep cross-dataset** — β1∈{0.85,0.9,0.95} × β2∈{0.99,0.999,0.9999}; PyTorch defaults never ablated; covers TFP/AF/DM + TF Lion control | LOW |

**Scientific rationale:**
- mugen #3027: 50k surface points chosen as round-number default; 64k may capture finer geometry (boundary layer mesh clustering), 32k/16k reduce compute. Either improvement or a clear ceiling confirms current default.
- jet #3028: AdamW β parameters are unexplored. CFD mesh regression has heterogeneous gradient scales and cosine LR cycling creates periodic gradient magnitude changes — both interact with β2's second-moment adaptation speed. β2=0.99 (faster adaptation) may help DM's non-stationary loss landscape. Phase-sequential: AF first (fast feedback), then apply winners to TFP and DM.

### Theme 20: Wave 12 — Residual Scaling / Regularization / Features / Optimization (NEW — 2026-04-22)

Four new cross-dataset hypotheses targeting underexplored levers: residual initialization, attention regularization, physics-informed geometry features, and layer-wise LR scheduling. All cover all 4 datasets (TF/TFP/AF/DM).

| Student | PR | Experiment | Risk |
|---|---|---|---|
| einar | #3021 | **LayerScale residuals (cross-dataset)** — per-channel learnable scalar α (init=1e-4) on attention+FFN residuals; `--layer-scale`; cross-dataset TF/TFP/AF/DM | LOW |
| wolfwood | #3022 | **Attention Dropout (p=0.1)** — dropout on Transolver attention weights after softmax; `--attn-dropout 0.1`; cross-dataset TF/TFP/AF/DM | LOW |
| emma | #3023 | **SDF Wall-Distance Feature** — min distance from each mesh node to solid boundary surface as physics input feature; cross-dataset TF/TFP/AF/DM | MEDIUM |
| shoya | #3024 | **Layer-wise LR Decay (LLRD, decay=0.75)** — deeper layers receive lower LR (`base_lr * 0.75^(L-i-1)`); cross-dataset TF/TFP/AF/DM | LOW |

**Scientific rationale:**
- einar #3021 (LayerScale): Touvron et al. 2021 CaiT paper showed per-channel α init at 1e-4 stabilises deep transformer training. Distinct from brook #2999 (which used scalar per-layer, not per-channel). Per-channel gives finer-grained control.
- wolfwood #3022 (Attention Dropout): Current spike #2995 tests standard attn dropout (p=0.1) but may still be in flight. wolfwood tests the same mechanism but cross-dataset via all 4 benchmarks simultaneously, ensuring no dataset-specific regression.
- emma #3023 (SDF Wall-Distance): Wall distance is a classical CFD feature (used in k-ω SST turbulence models). Provides the model with explicit geometric information about proximity to solid boundaries — directly relevant to boundary layer physics. Tested at mesh-node level via chunk-based cdist.
- shoya #3024 (LLRD, decay=0.75): More aggressive than frieren's #2984 (decay=0.8). Earlier layers see LR reduced by 0.75^(L-1) relative to head. Physics intuition: later layers learn task-specific representations (should adapt faster) while earlier layers learn geometry/features (should be more stable).

**Baselines for Wave 12 PRs:**
- TF: val=22.537 (#2924) / test=N/A
- TFP: val=not established
- AF: val=0.000482 (#2951) / test=0.003 (#2824)
- DM: val=3.997% (#2898) / test=6.24% (#2691)

### Theme 19: Wave 11 — sigma-Reparam + GeGLU (NEW — 2026-04-22, post-Wave 9/10 closures)

Two new cross-dataset hypotheses assigned after closing griffith #2968 (Spectral Norm) and casca #2962 (AGC). Both cover all 4 datasets.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| griffith | #3016 | **sigma-Reparam on attention projections** — `W = g * (V / ||V||_σ)` per Zhai et al. ICML 2023; learned scalar g controls spectral norm without hooks; torch.compile compatible; `--sigma-reparam` flag; cross-dataset TF/TFP/AF/DM | LOW |
| casca | #3017 | **GeGLU FFN activation** — `FFN(x) = (xW_1) * GELU(xW_3) * W_2`; distinct from SwiGLU (nezuko #2938 uses SiLU gate); GELU gate has different gradient properties; `--geglu` flag; cross-dataset TF/TFP/AF/DM | LOW |

**Scientific rationale:**
- griffith sigma-Reparam: Directly motivated by griffith's own spectral norm review (he cited Zhai et al. as "a cleaner future alternative"). sigma-Reparam parameterizes W as `g * (V/||V||_σ)` where g is a learned scalar and V an unconstrained matrix — achieves spectral norm control natively in the forward pass, no hook API, no torch.compile issues, and g can be regularized independently. Hypothesis: spectral bound helps attention expressivity when implemented cleanly rather than via SN hooks.
- casca GeGLU: Complementary to SwiGLU (nezuko #2938). The key difference: GELU gate is smoother than SiLU/Swish near zero, with heavier tails. For CFD meshes where many nodes have near-zero physical quantities (pressure, velocity in quiescent regions), GELU gating may provide better sparsity. Dauphin et al. GLU (2017) + Noam Shazeer (2020) GeGLU/SwiGLU survey: both consistently outperform vanilla ReLU FFN.

### Theme 18: Wave 10 — Cross-Dataset Optimization/Architecture/Augmentation Sweep (NEW — 2026-04-22)

Thirteen new cross-dataset hypotheses spanning optimizer hyperparameters, architecture depth/width, regularization, and feature engineering. Wave 10 is a broad sweep to cover underexplored corners of the search space while Wave 9 stability runs are in flight.

| Student | PR | Experiment | Hypothesis |
|---|---|---|---|
| canute | #3001 | **Slice temperature sweep** — Transolver's `self.temperature` is the most architecturally important temperature parameter (slice assignment); systematic sweep to find optimal value | Architecture |
| franky | #3002 | **Extended Fourier frequency bands** — increase Fourier band cutoffs beyond defaults; richer frequency decomposition for complex flow structures | Features |
| norman | #3003 | **Input channel dropout regularization** — randomly zero entire input channels during training; prevents overfit to any single physics feature | Regularization |
| sanji | #3004 | **Long cosine schedule with linear warmup** — extended T_max with slow warmup; test if longer cosine reduces terminal LR overshoot | Optimization |
| robin | #3005 | **2L/192d + gc=0.5 + EMA cross-dataset** — depth meets EMA breakthrough; shallower architecture with gradient clipping and EMA | Architecture |
| senku | #3006 | **2L/192d + gc=0.3 + EMA cross-dataset** — softer clip variant; explore gc=0.3 which may be less restrictive than gc=0.5 for 2L | Architecture |
| shouko | #3007 | **4L/512d + T_max=10 cross-dataset** — shorter cosine for DrivAerML-compatible schedule; test at 4L where gc stability is known | Architecture |
| stark | #3008 | **AirfRANS depth sweep + T_max transfer** — AF-specific depth/T_max sensitivity; also tests cross-dataset transfer of AirfRANS tuning | Optimization |
| ~~megumi~~ | ~~#3009~~ | ~~**Lion lr sweep cross-dataset** — systematic Lion LR search: find optimal lr for TF+AirfRANS+TFP+DM~~ **CLOSED — confirmed dead end; catastrophic AF/DM instability at all LRs (1e-4, 1.25e-4, 1.5e-4, 2e-4)** | ~~Optimization~~ |
| piccolo | #3010 | **DrivAerML grad-clip sweep + gc transfer** — gc sensitivity sweep on DM; test which gc generalises cross-dataset | Optimization |
| sukuna | #3011 | **Weight decay sweep cross-dataset** — wd=5e-3 vs wd=2e-2; explore WD sensitivity (DM WD=0 constraint still applies) | Regularization |
| usopp | #3012 | **AdamW lr ablation** — systematic AdamW LR on TF + standard cross-dataset coverage | Optimization |
| shoya | #3013 | **Fourier feature ablation cross-dataset** — ablate `--enable-fourier` across all datasets to quantify its contribution to baseline | Features |

**Scientific rationale:** Wave 10 is a systematic coverage sweep. After 9 waves of targeted innovations, several hyperparameter axes remain only coarsely explored: slice temperature (most important Transolver parameter per #2981 analysis), Fourier band width, optimizer LR across all datasets simultaneously, and depth variants with EMA. This wave aims to either establish tighter optima or rule out further gains in these directions.

### Theme 17: Wave 9 — Cross-Dataset Residual Stability Innovations (NEW — 2026-04-22)

Two novel residual/regularization hypotheses targeting transformer stability and attention Lipschitz constraint. Both cover all 4 datasets.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| brook | #2999 | **LayerScale on Transolver residuals** — per-layer learnable diagonal scale `ls1/ls2 = nn.Parameter(ones * 1e-6)` on attention+FFN residuals; `--layer-scale`; cross-dataset (TF/TF-paper/AF/DM) | LOW |
| zenitsu | #3000 | **Spectral Norm on attention Q/K/V/O projections** — `torch.nn.utils.spectral_norm` wraps linear projections to bound Lipschitz constant; `--spectral-norm-attn`; cross-dataset (TF/TF-paper/AF/DM) | LOW |

**Scientific rationale:**
- brook #2999 (LayerScale): Touvron et al. 2021 (CaiT/DeiT-III) showed that initialising residual scale at 1e-6 stabilises deep ViT training by preventing gradient explosion in early epochs. Applied to Transolver blocks: `x = x + ls1 * attn(norm1(x))` and `x = x + ls2 * ffn(norm2(x))`. Zero overhead at inference — ls values become near-1.0 after convergence in stable runs. Orthogonal to all Wave 3-8 experiments.
- zenitsu #3000 (Spectral Norm): Miyato et al. 2018 (SNGAN) / Brock et al. 2018 (BigGAN) showed spectral normalisation stabilises GAN training by bounding the largest singular value of each weight matrix to ≤1. For attention projections, this constrains how sensitive the attention map is to small changes in input features — directly relevant to CFD meshes where nearby points are strongly correlated. Should combine with existing recipe (EMA + Lion/AdamW) without conflict.

### Theme 16: Wave 8 — Cross-Dataset Regularization Innovations (NEW — 2026-04-22)

New regularization hypotheses targeting attention and architecture. All cover all 4 datasets.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| spike | #2995 | **Attention Dropout (0.1)** — add `dropout=0.1` to all `nn.MultiheadAttention` calls; forces distributed attention representations; prevents overfitting to dominant local mesh patterns; cross-dataset (TF/TF-paper/AF/DM) | LOW |

**Scientific rationale:**
- spike #2995: Current model uses `dropout=0.0` everywhere. Attention dropout is a well-established regularizer — forces heads to learn redundant, distributed representations. CFD meshes are highly structured (nearby points strongly correlated); dropout may prevent overfitting to local mesh topology. Negligible compute overhead.

### Theme 15: Hypernetwork Condition Encoding (2026-04-22)

A small hypernetwork (~4k params) reads global flow condition scalars (Re, AoA, Mach) and generates per-layer scale+bias offsets for the slice hidden representations. Targets the conditioning pathway — orthogonal to all Wave 3-7 experiments. Motivated by the physics insight that different flow regimes (boundary layer dynamics, separation) should activate different computational paths.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| usopp | #2994 | **Hypernetwork Condition Encoding** — FlowCondHyperNet generates scale+bias applied at transformer entry; `--enable-flow-hypernet`; cross-dataset (TF/TF-paper/AF/DM) | MED |

### Theme 14: Wave 7 — Cross-Dataset Attention Architecture Innovations (NEW — 2026-04-22)

Six novel attention/compute hypotheses, all covering all 4 datasets. Wave 7 targets attention efficiency (MQA/GQA — human team HIGH PRIORITY), attention sparsity, weight averaging, and throughput optimizations.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| chrome | #2988 | **Multi-Query Attention (MQA)** — single KV head shared across all Q heads (`--num-kv-heads 1`); human team flagged HIGH PRIORITY | LOW-MED |
| faye | #2989 | **Grouped-Query Attention (GQA)** — intermediate KV sharing: AF num_kv_heads=2/4H, DM num_kv_heads=4/8H, TF/TF-paper num_kv_heads=1/3H | LOW-MED |
| gojo | #2990 | **Attention Head Dimension Scaling** — per-head dim ablation: TF 2H vs 6H (baseline 3H), AF 2H vs 8H, DM 4H vs 16H, all fixed hidden_dim | LOW-MED |
| himmel | #2991 | **Stochastic Weight Averaging (SWA)** — average checkpoints at cosine LR troughs (`--swa --swa-start 0.75` or Python manual fallback) | LOW-MED |
| levi | #2992 | **Flash Attention + torch.compile** — throughput hypothesis: reduce kernel overhead, more epochs within budget; log steps/sec | LOW |
| shoya | #2993 | **Sparse Top-k Slice Attention** — retain only top-k attention connections per query slice; test k=16 and k=32; OOM savings enable more slices | MED |

**Scientific rationale:**
- MQA/GQA (#2988/#2989): Human team flagged HIGH PRIORITY. KV sharing reduces memory bandwidth by 3-8×, enables larger batches or more epochs. Orthogonal to all other changes.
- Head-dim scaling (#2990): num_heads×head_dim = hidden_dim; changing num_heads trades cross-head diversity for per-head expressivity. Untested in this programme.
- SWA (#2991): Weight averaging at cosine troughs corresponds to flat basin exploration — known to improve generalization in vision/NLP. Orthogonal to optimizer choice.
- Flash+compile (#2992): Pure throughput play. FA-2 + reduce-overhead mode should give 20-50% faster epochs, more training signal within 360-min budget.
- Sparse top-k (#2993): Physics-motivated: leading-edge slices are weakly coupled to wake slices. O(S·k) vs O(S²). k=16/32 vs full k=64.

### Theme 13: Wave 6 — Cross-Dataset Scheduler/Augmentation (NEW — 2026-04-22)

Three new cross-dataset hypotheses targeting scheduler correctness, geometric augmentation, and T_max sensitivity. All cover all 4 datasets.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| gojo | #2985 | **Per-Step SGDR** — `CosineAnnealingWarmRestarts` with step-level T_0=1000, T_mult=2; preserves rapid oscillation regularizer that per-epoch SGDR (#2967) destroyed | LOW-MED |
| shoya | #2986 | **Coordinate Noise Augmentation** — Gaussian noise σ=0.01 on node 3D positions during training only; forces physics-invariant representations (different from point dropout #2970) | LOW |
| chrome | #2987 | **Cosine T_max Cross-Dataset Sweep** — T_max=5 (primary) and T_max=20 across all 4 datasets; first unified T_max comparison vs per-dataset baselines (TF=10, DM=30) | LOW |

**Scientific rationale:**
- gojo #2985: Per-epoch SGDR (#2967) failed because ~750 steps/epoch → LR monotonically decreasing for full epoch → sharp basin → divergence. Per-step T_0=1000 restores the per-step oscillation cycle.
- shoya #2986: Entirely untested in this pipeline. Standard in 3D point cloud literature (PointNet/DGCNN). σ=0.01 is a gentle perturbation that should regularize without distorting physics.
- chrome #2987: TF baseline T_max=10, DM baseline T_max=30 set independently. Cross-dataset optimum has never been measured. Note: T_max=5 previously diverged DM (#2911) — a key data point for comparison.

### Theme 7: Bold New Directions (Wave 3 — recreated after accidental merge of #2928-2936)

10 hypothesis families: loss reformulation, architecture innovations, optimization shifts, physics-informed features, unit-invariant clipping.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| frieren | #2937 | **Relative L2 Training Loss** — align DM training loss with eval metric | LOW |
| nezuko | #2938 | **SwiGLU FFN Replacement** — gated linear units for all Transolver blocks | LOW |
| ~~violet~~ | ~~#2939~~ | ~~**Stochastic Depth (DropPath)** — layer-level regularization 0.1/0.2~~ **CLOSED — negative across all datasets; reassigned to Wave 15 #3030 Relative L2 loss** | ~~LOW~~ |
| gilbert | #2940 | **Prodigy Optimizer** — parameter-free LR adaptation | LOW-MED |
| kohaku | #2941 | **Global Context Token** — break slice-local attention bottleneck | MEDIUM |
| emma | #2942 | **Surface Normals + Curvature** — differential geometry input features | MEDIUM |
| chihiro | #2943 | **Conservation Auxiliary Loss** — div(u)=0 physics regularization | MED-HIGH |
| shoya | #2944 | **MoE FFN Layers** — sparse expert routing for physics-regime specialization | MED-HIGH |
| mitsuha | #2945 | **SDF Wall-Distance Feature** — signed distance field geometry embedding | MEDIUM |
| ~~casca~~ | ~~#2946~~ | ~~**Adaptive Gradient Clipping (AGC)**~~ CLOSED — incompatible with Lion; DM diverged; reassigned to Wave 11 GeGLU | ~~LOW-MED~~ |

### Theme 8: TandemFoil Paper Baseline Wave (NEW — 2026-04-22)

First experiments ever run on `tandemfoil_paper` on the radford programme. Racing to establish the val anchor for `val_primary/field_mse`.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| jin | #2947 | **TF Paper LR Sweep** — Lion lr=1e-4/1.25e-4/1.5e-4/2e-4 at 3L/192d+3L/256d; AdamW refs | LOW |
| guts | #2948 | **TF Paper Physics-Flag Ablation** — 8 runs removing flags one-by-one from full golden config | LOW |
| vash | #2949 | **TF Paper Depth/Width Arch Sweep** — 3L/4L/5L × 192d/256d/384d at Lion lr=1e-4 | LOW |

Paper targets (Experiment 4, Table 6 — MGN best / paper best per task):
- cruise_random_uniform: 1.79 / **0.10**
- cruise_random_aoa_extrap: 2.03 / **0.18**
- cruise_random_re_extrap: 4.85 / **0.36**
- cruise_random_stagger_extrap: 1.74 / **0.13**
- cruise_random_gap_extrap: 1.95 / **0.14**
- racecar_uniform: 0.61 / **0.21**

### Theme 9: DrivAerML + AirfRANS Optimizer Sweeps (NEW — 2026-04-22)

| Student | PR | Experiment | Risk |
|---|---|---|---|
| piccolo | #2950 | **DrivAerML Lion Optimizer** — Lion lr=1e-4/2e-4/3e-4/5e-4 × T_max=30/50 + AdamW refs at 4L/512d | LOW-MED |
| stark | #2951 | **AirfRANS LR+Cosine Sweep** — AdamW lr=7e-4/8e-4/9e-4/1e-3 × T_max=10/20/50 at 2L/256d/4H | LOW |

### Theme 0: EMA Refinement

| Student | PR | Experiment |
|---|---|---|
| robin | #2924 | TF lr=1e-4+EMA, TF gc=0.5+EMA, AF T_max=10+EMA, AF seed=43 |
| zenitsu | #2925 | DrivAerML EMA+gc=1.0, EMA+gc+WD, EMA decay=0.9999, pure gc control |

### Theme 1: AirfRANS Recipe Transfer to DrivAerML

| Student | PR | Experiment |
|---|---|---|
| brook | #2878 | gc=1.0+WD=1e-2 (flagship compound, no EMA) |
| ~~bulma~~ | ~~#2879~~ | ~~T_max=10+gc=1.0~~ CLOSED — failed; reassigned to #2972 cross-dataset spatial sweep |
| canute | #2880 | Full recipe: lr=7e-4+gc=1.0+WD=1e-2 |
| chopper | #2882 | T_max=15+gc=1.0+WD=1e-2 |
| einar | #2883 | gc=1.0+T_max=20+WD=1e-2 |
| yuji | #2922 | WD=5e-3+gc=1.0 moderate, pure gc ablation, WD alone |
| edward | #2916 | lr=6e-4+gc=1.0+WD=1e-2+T_max=10 |
| ~~wolfwood~~ | ~~#2919~~ | ~~T_max=40/50+gc=1.0+WD=1e-2 (longer cycling)~~ CLOSED — failed; reassigned to #2973 cross-dataset spatial sweep |

### Theme 12: Wave 5 — Cross-Dataset Code/Architecture Innovations (NEW — 2026-04-22)

Five novel hypotheses, all covering all 4 datasets. Wave 5 re-runs Wave 4 ideas as TRUE cross-dataset PRs (the Wave 4 assignments #2974-#2978 were single-dataset; Wave 5 is the corrected multi-dataset version).

| Student | PR | Experiment | Risk |
|---|---|---|---|
| mugen | #2980 | **PCGrad Gradient Surgery** — project conflicting surface/volume gradients using PCGrad (Yu et al. 2020); `pcgrad_backward()` helper; all 4 datasets | MED |
| ~~spike~~ | ~~#2981~~ | ~~**Learnable Attention Temperature**~~ CLOSED — all 4 datasets 200-2844% worse; QK temp learning is downstream of the slice-assignment temperature; learned values ≈1.0 confirms standard 1/√d_k near-optimal | ~~LOW-MED~~ |
| taki | #2982 | **Z-Score Pressure Normalization** — replace `--asinh-pressure` with `LearnedPressureNorm` using running_mean/running_var buffers (momentum=0.01); all 4 datasets | LOW |
| zenitsu | #2983 | **RoPE Positional Embeddings** — rotary positional encoding on QK using 3D node coordinates; alongside `--enable-fourier`; all 4 datasets | LOW-MED |
| frieren | #2984 | **LLRD (Layer-wise Learning Rate Decay)** — `get_llrd_param_groups(model, base_lr, decay=0.8)`, test decay=0.8 and 0.9 on TF first; all 4 datasets | LOW |

### Theme 11: Wave 4 — Cross-Dataset Code/Architecture Innovations (2026-04-22, SUPERSEDED by Wave 5)

NOTE: These PRs (#2974-#2978) were originally framed as cross-dataset but were single-dataset implementations. Wave 5 (#2980-#2984) is the corrected multi-dataset version. These PRs may still complete and should be reviewed individually.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| mugen | #2974 | **Best-Checkpoint Saving** — save `checkpoint_best.pt` whenever val improves; load best at end; all 4 datasets | LOW |
| spike | #2975 | **RoPE Positional Embeddings** — rotary positional encoding on QK using 3D node (x,y,z) coordinates; `--rope-dim 32`; all 4 datasets | LOW-MED |
| taki | #2976 | **Z-Score Pressure Normalization** — replace `--asinh-pressure` with per-dataset mean/std `--zscore-pressure`; all 4 datasets | LOW |
| zenitsu | #2977 | **Learnable Attention Temperature** — per-head log-space temperature scalar `nn.Parameter`; `--learnable-attn-temperature`; all 4 datasets | LOW-MED |
| frieren | #2978 | **PCGrad Gradient Surgery** — project conflicting surface/volume gradients; `--pcgrad`; logs `gradient_conflict_rate`; all 4 datasets | MED |

### Theme 10: Cross-Dataset Spatial/Physics Budget Sweeps (NEW — 2026-04-22)

Two systematic sweeps testing the model's sensitivity to physics partition granularity (`model_slices`) and spatial resolution budget (`geometry_supernodes` + `surface_anchor_points`) across all 4 datasets. Neither dimension has ever been swept in the cross-dataset icml2026 format.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| bulma | #2972 | **model_slices cross-dataset sweep** — 48/64/96/128 across all 4 datasets; TF best is 64 (baseline 96 default), never tested on AF/DM/TF-paper | LOW-MED |
| wolfwood | #2973 | **geometry_supernodes + surface_anchor_points budget sweep** — Config A (2048/4000), B default (4096/8000), C (8192/16000) across all 4 datasets | LOW-MED |

**bulma #2972 — model_slices sweep details:**
- 13 runs total: TF×3 (slices 48/96/128; 64 is current TF best), TF-paper×3, AF×4 (48/64/96/128), DM×3 (48/64/128)
- Priority: TF+AF first (fast); DM second (slow — min slices=64 and slices=96)
- Hypothesis: slices=64 TF win may transfer; AF/DM may prefer different granularity

**wolfwood #2973 — spatial resolution budget details:**
- Config A (half): `--geometry-supernodes 2048 --surface-anchor-points 4000`
- Config B (baseline): `--geometry-supernodes 4096 --surface-anchor-points 8000`
- Config C (double): `--geometry-supernodes 8192 --surface-anchor-points 16000`
- DM Config C: OOM risk — fallback to `--geometry-supernodes 8192 --surface-anchor-points 8000` if needed
- Hypothesis: current defaults may under-resolve or over-spend spatial budget; doubling may improve surface fidelity

### Theme 2: DrivAerML LR+gc Exploration

| Student | PR | Experiment |
|---|---|---|
| faye | #2885 | lr=7e-4+gc=1.0 |
| franky | #2886 | lr=4e-4+gc=1.0 |
| gohan | #2887 | gc=1.0+T_max=10 LR scan |
| gojo | #2888 | gc=0.5+T_max=10 |
| ~~casca~~ | ~~#2881~~ | ~~gc=1.5/gc=2.0~~ CLOSED — dead end. Reassigned to #2946 AGC |
| jin | #2947 | TandemFoil Paper first baseline — Lion lr sweep 1e-4/1.25e-4/1.5e-4/2e-4 × 3L/192d+3L/256d + AdamW refs |
| shinobu | #2912 | WD=3e-2/5e-2+gc=1.0 (heavy regularization) |
| sanji | #2918 | gc=0.5+WD=1e-2 (softer clip + regularization compound) |

### Theme 3: DrivAerML Architecture

| Student | PR | Experiment |
|---|---|---|
| griffith | #2889 | 3L/512d+gc=1.0 |
| guts | #2948 | TandemFoil Paper physics-flag ablation — 8 runs removing flags from golden config |
| himmel | #2891 | 5L/512d deeper |
| jet | #2892 | 3L/768d shallow+wide |
| shouko | #2909 | heads=16/4 ablation + gc=1.0 |
| askeladd | #2914 | MLP ratio=6/2 + gc=1.0+WD=1e-2 |
| chrome | #2923 | torch.compile + gc=1.0 (throughput + stability) |

### Theme 4: Scheduler Innovations (CODE CHANGES)

| Student | PR | Experiment |
|---|---|---|
| megumi | #2894 | Linear warmup+cosine |
| mugen | #2895 | CosineAnnealingWarmRestarts T_mult (SENT BACK — found TF=25.459/AF=0.000371 transient but lost due to no best-ckpt save; must add checkpoint saving + retry) |
| vash | #2949 | TandemFoil Paper depth/width arch sweep — 3L/4L/5L × 192d/256d/384d |

### Theme 5: Training Innovations (CODE CHANGES)

| Student | PR | Experiment |
|---|---|---|
| nobara | #2897 | LLRD (layer-wise LR decay) |
| ~~usopp~~ | ~~#2920~~ | ~~Gradient noise injection~~ CLOSED — fatal instability all datasets; reassigned to #2970 Point Dropout |
| sukuna | #2903 | SWA at cosine troughs |
| spike | #2901 | Huber/log-cosh loss (SENT BACK — submitted while still running; AF Huber is 58.7% worse than MSE) |
| stark | #2951 | AirfRANS LR+cosine sweep — lr=7e-4/8e-4/9e-4/1e-3 × T_max=10/20/50 |

### Theme 6: Throughput + Seeds + Ablations

| Student | PR | Experiment |
|---|---|---|
| piccolo | #2950 | DrivAerML Lion optimizer sweep — lr=1e-4/2e-4/3e-4/5e-4 × T_max=30/50 + AdamW refs |
| vegeta | #2906 | 360min multi-seed replication |
| nami | #2896 | Lion higher LR on DrivAerML |
| eren | #2910 | max-train-batches=788 (2x data/epoch) |
| rei | #2913 | surface-points=75k resolution scaling |
| levi | #2915 | no-Fourier ablation (faster epochs) |

### Continuing from Previous Wave

| Student | PR | Dataset | Focus |
|---|---|---|---|
| norman | #2868 | DrivAerML | 2L/512d+3L/512d |
| historia | #2867 | DrivAerML | 3L/256d+3L/384d |
| kakashi | #2823 | AirfRANS | gc=1.0+T_max=10 stabilization |
| thorfinn | #2786 | AirfRANS | gc=1.0+T_max=7 extended |
| taki | #2814 | DrivAerML | Mild regularization |
| tanjiro | #2842 | TandemFoil | 3L/192d+lr=1e-4+gc=0.5 (sent back) |
| alphonse | #2840 | TandemFoil | lr=1e-4+gc=1.0 multi-seed |
| fern | #2837 | TandemFoil | 3L/256d at lr=1.25e-4 |
| senku | #2864 | TandemFoil | 2L/192d+2L/256d depth reduction |
| haku | #2820 | AirfRANS | gc=0.5+lr=5e-4 extended |
| hinata | #2770 | AirfRANS | 4L/256d WD=5e-3 |

Note: jin (#2893), guts (#2890), vash (#2905), piccolo (#2898), stark (#2902) were reassigned to new experiments on 2026-04-22.
Their old PRs remain in-flight but are now listed under their new assignments in Themes 8 and 9 above.

## Research Insights

1. **Corrected EMA (MERGED #2899):** timm-style warmup `min(decay, (1+step)/(10+step))` gives -13.2% TF and -41.2% AF. **This is the shared recipe change.** decay=0.999 > 0.9999 on both datasets.
2. **DrivAerML is fragile to compounds:** gc+WD+cosine crashes (6/8 #2908), 640d crashes (#2917), T_max=5 crashes (#2911). Only gentle perturbations survive at 4L/512d.
3. **Width scaling ceiling at 512d for DM:** 640d is unstable. guts #2890 (768d) and himmel #2891 (5L/512d) will clarify the boundary.
4. **AB-UPT** achieves 3.71% via geometry-separated encoding — escalation if EMA+recipe fails.
5. **TandemFoil Paper baseline urgently needed:** This dataset has never been run on radford. We need a val anchor before we can evaluate any experiment improvements against the paper's Table 6 numbers.

## Current Research Themes and Priorities

### Priority 0: Cross-Dataset Generalization (Human Team Directive — NON-NEGOTIABLE)
- **Every new experiment must test across all 4 datasets.** Ideas that help only one dataset are not acceptable for the shared paper recipe.
- **TandemFoil Paper is now a required 4th benchmark.** All future assignments must include it.
- **Measurement gate:** An experiment is a win only if it does not cause regression on any of the 4 datasets (or shows a cross-dataset improvement).

### Priority 1: TandemFoil Paper Baseline (IN PROGRESS)
- **THREE simultaneous baseline runs launched (#2947 jin, #2948 guts, #2949 vash).** Racing to establish a val anchor on `val_primary/field_mse`.
- jin (#2947) sweeps Lion LR; guts (#2948) ablates physics flags; vash (#2949) sweeps architecture depth/width.
- Paper targets (Table 6 MGN + PRE-RES-FREE+RES-COMB): 0.10/0.18/0.36/0.13/0.14/0.21 per task.
- Once the first result arrives, update BASELINE.md with the new `tandemfoil_paper` section.
- Metric: normalized full-field MSE (`val_primary/field_mse`).

### Priority 2: DrivAerML Gap Closure (4.619% → 3.71%)
- **Loss alignment** (frieren #2928): Most direct fix — training on relative L2 directly matches the evaluation metric
- **Architectural upgrades** (nezuko #2929 SwiGLU, kohaku #2932 global context): Address known Transolver limitations
- **Regularization** (violet #2930 DropPath): New orthogonal regularization dimension
- **Physics-informed features** (emma #2933 curvature, mitsuha #2936 SDF): Give model explicit geometric knowledge it currently must learn implicitly

### Priority 3: Cross-Benchmark Recipe Validation
- Wave 3 experiments should test across all 4 benchmarks by default
- TandemFoil Paper benchmark provides calibration against published numbers
- Ideas that help DM but hurt TF/AF are dataset hacks; we want shared wins across all 4

### Priority 4: Optimization Paradigm Shift
- Prodigy (gilbert #2931): If the LR search is truly exhausted, adaptive optimizers may find new trajectories
- MoE (shoya #2935): Sparse expert specialization is fundamentally different from dense FFN
- Conservation loss (chihiro #2934): Physics constraints as regularization

## Next Priorities

1. **Monitor Wave 16** (alphonse #3031 Pre-LN, fern #3032 RMSNorm, hinata #3033 linear warmup, kakashi #3034 gradient centralization, tanjiro #3035 SGDR T_mult=2, thorfinn #3036 AdaFactor) — six new cross-dataset hypotheses targeting normalization placement, normalization variant, warmup, gradient processing, restart schedule shape, and memory-efficient optimization. All 4 benchmarks mandatory per PR.
2. **Monitor Wave 15** (violet #3030 Relative L2 loss — TF+DM) — if rel_l2 or mixed loss beats DM baseline (3.997%), this is a signal to try relative loss formulations on AF and TFP as well.
3. **Monitor Wave 14** (megumi #3029 best-checkpoint saving) — critical enabler for tanjiro #3035 SGDR T_mult=2; once merged, tanjiro can properly exploit restart transient minima.
4. **Monitor Wave 12** (einar #3021 LayerScale, wolfwood #3022 attn dropout, emma #3023 SDF wall-distance, shoya #3024 LLRD) — four new cross-dataset hypotheses targeting residual scaling, attention regularization, physics features, and LR scheduling.
5. **Monitor Wave 11** (griffith #3016 sigma-Reparam, casca #3017 GeGLU) — two new cross-dataset hypotheses, directly motivated by Wave 9/10 closure analysis.
6. **Monitor Wave 10** (#3001–#3013, 13 PRs) — broad coverage sweep: slice temp, Fourier bands, channel dropout, long cosine, 2L+EMA variants, 4L+T_max=10, AF depth, Lion/AdamW lr sweeps, WD sweep, DM gc sweep, Fourier ablation. All cross-dataset.
7. **Monitor Wave 9** (#2999 brook LayerScale) — lowest-risk stability hypothesis, fully cross-dataset.
8. Watch Wave 3 v3 results (#2953-2962: Relative L2, SwiGLU, DropPath, Prodigy, Global Context, Curvature, Conservation Loss, MoE, SDF, AGC)
9. Watch Wave 6–8 hypotheses (#2985-#2995: per-step SGDR, coord noise, T_max sweep, MQA/GQA/head-dim/SWA/Flash+compile/sparse top-k, hypernetwork, attn dropout)
10. All new assignments: 4-dataset coverage mandatory per human team directive
11. If DM recipe transfer (Theme 1) fails universally → escalate to geometry-separated encoding
12. Check for human team messages on GitHub issues (priority — check very frequently)
