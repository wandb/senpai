# SENPAI Research State

- **Date:** 2026-04-22 (last polled — Wave 3 + TF Paper baseline wave + DM Lion + AF LR sweep + cross-dataset spatial/physics budget sweeps + Wave 4 cross-dataset code/arch innovations + Wave 5 cross-dataset code/arch innovations + Wave 6 per-step SGDR / coord noise / T_max sweep + Wave 7 attention architecture innovations + usopp #2994 hypernetwork + Wave 8 spike #2995 attention dropout. brook #2996 MQA reassignment also WIP.)
- **Branch:** radford
- **Fleet status:** 57 WIP PRs, ALL ASSIGNED (0 idle) — Wave 7: chrome #2988 MQA, faye #2989 GQA, gojo #2990 head-dim scaling, himmel #2991 SWA, levi #2992 Flash+compile, shoya #2993 sparse top-k; usopp #2994 hypernetwork; Wave 8: spike #2995 attention dropout; zenitsu #2983 RoPE; brook #2996 MQA (reassignment from #2878). PRs CLOSED: #2981 (spike, QK attention temp), #2977 (zenitsu, learnable per-head attn temperature — 1.4x–3.0x worse on all datasets, dead end).
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
| TandemFoil | `val_primary/surface_pressure_mae` | **26.06** (#2887 MERGED — Lion lr=1e-4, no-EMA) |
| TandemFoil Paper | `val_primary/field_mse` | **not established** — baseline run needed urgently |
| AirfRANS | `val_primary/surface_mse` | **0.000598** (#2906 MERGED — AdamW lr=6e-4, seed=42, 2L/256d/4H, ep517) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **4.619%** (#2691) |

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

**Priority:** HIGH — assign a student to add this as a code change with a cross-dataset validation run.

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

## ACTIVE EXPERIMENTS — 57 WIP PRs (updated 2026-04-22 after PR #2981 closed, #2995 assigned)

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
| violet | #2939 | **Stochastic Depth (DropPath)** — layer-level regularization 0.1/0.2 | LOW |
| gilbert | #2940 | **Prodigy Optimizer** — parameter-free LR adaptation | LOW-MED |
| kohaku | #2941 | **Global Context Token** — break slice-local attention bottleneck | MEDIUM |
| emma | #2942 | **Surface Normals + Curvature** — differential geometry input features | MEDIUM |
| chihiro | #2943 | **Conservation Auxiliary Loss** — div(u)=0 physics regularization | MED-HIGH |
| shoya | #2944 | **MoE FFN Layers** — sparse expert routing for physics-regime specialization | MED-HIGH |
| mitsuha | #2945 | **SDF Wall-Distance Feature** — signed distance field geometry embedding | MEDIUM |
| casca | #2946 | **Adaptive Gradient Clipping (AGC)** — NFNet-style unit-invariant clipping | LOW-MED |

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

1. ~~**URGENT: Best-checkpoint saving code change**~~ — **IN PROGRESS (#2974 mugen Wave 4).** The T_mult experiment (#2895) proved TF=25.459 and AF=0.000371 exist in the landscape. mugen implementing `checkpoint_best.pt` saving. **Wave 5 launched: #2980-#2984 (PCGrad/AttnTemp/ZScore/RoPE/LLRD) — all true cross-dataset.**
2. **Monitor TF Paper baseline wave** (#2947 jin, #2948 guts, #2949 vash) — update BASELINE.md with new section once first results arrive
3. **Monitor DrivAerML Lion** (#2950 piccolo) — first Lion run on DrivAerML; could be significant
4. **Monitor AirfRANS LR ceiling** (#2951 stark) — tests whether lr>6e-4 helps AF
5. **Monitor spatial/physics budget sweeps** (#2972 bulma model_slices, #2973 wolfwood supernode/anchor) — first systematic spatial resolution sweep in cross-dataset format
6. Watch Wave 3 v3 results (#2953-2962: Relative L2, SwiGLU, DropPath, Prodigy, Global Context, Curvature, Conservation Loss, MoE, SDF, AGC)
7. Watch new Wave 3+ hypotheses (#2963-2971: LayerScale, Multi-Scale Attention, Extended Fourier, Huber, SGDR, SpectralNorm, log1p, PointDropout, LabelSmoothing)
8. All new assignments: 4-dataset coverage mandatory per human team directive
9. If DM recipe transfer (Theme 1) fails universally → escalate to geometry-separated encoding
10. Check for human team messages on GitHub issues (priority — check very frequently)
