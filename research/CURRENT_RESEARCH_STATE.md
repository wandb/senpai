# SENPAI Research State

- **Date:** 2026-04-22 00:15 (Wave 3 Reassigned + casca AGC)
- **Branch:** radford
- **Fleet status:** 60 live students, ALL ASSIGNED (0 idle)
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
| TandemFoil | `val_primary/surface_pressure_mae` | **26.134** (#2899 MERGED — EMA decay=0.999) |
| TandemFoil Paper | `val_primary/field_mse` | **not established** — baseline run needed urgently |
| AirfRANS | `val_primary/surface_mse` | **0.000727** (#2899 MERGED — EMA decay=0.999, ep206) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **4.619%** (#2691) |

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

## ACTIVE EXPERIMENTS — 59 WIP PRs

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

### Theme 0: EMA Refinement

| Student | PR | Experiment |
|---|---|---|
| robin | #2924 | TF lr=1e-4+EMA, TF gc=0.5+EMA, AF T_max=10+EMA, AF seed=43 |
| zenitsu | #2925 | DrivAerML EMA+gc=1.0, EMA+gc+WD, EMA decay=0.9999, pure gc control |

### Theme 1: AirfRANS Recipe Transfer to DrivAerML

| Student | PR | Experiment |
|---|---|---|
| brook | #2878 | gc=1.0+WD=1e-2 (flagship compound, no EMA) |
| bulma | #2879 | T_max=10+gc=1.0 |
| canute | #2880 | Full recipe: lr=7e-4+gc=1.0+WD=1e-2 |
| chopper | #2882 | T_max=15+gc=1.0+WD=1e-2 |
| einar | #2883 | gc=1.0+T_max=20+WD=1e-2 |
| yuji | #2922 | WD=5e-3+gc=1.0 moderate, pure gc ablation, WD alone |
| edward | #2916 | lr=6e-4+gc=1.0+WD=1e-2+T_max=10 |
| wolfwood | #2919 | T_max=40/50+gc=1.0+WD=1e-2 (longer cycling) |

### Theme 2: DrivAerML LR+gc Exploration

| Student | PR | Experiment |
|---|---|---|
| faye | #2885 | lr=7e-4+gc=1.0 |
| franky | #2886 | lr=4e-4+gc=1.0 |
| gohan | #2887 | gc=1.0+T_max=10 LR scan |
| gojo | #2888 | gc=0.5+T_max=10 |
| ~~casca~~ | ~~#2881~~ | ~~gc=1.5/gc=2.0~~ CLOSED — dead end. Reassigned to #2946 AGC |
| jin | #2893 | lr=1e-3+gc=1.0 |
| shinobu | #2912 | WD=3e-2/5e-2+gc=1.0 (heavy regularization) |
| sanji | #2918 | gc=0.5+WD=1e-2 (softer clip + regularization compound) |

### Theme 3: DrivAerML Architecture

| Student | PR | Experiment |
|---|---|---|
| griffith | #2889 | 3L/512d+gc=1.0 |
| guts | #2890 | 4L/768d ultra-wide |
| himmel | #2891 | 5L/512d deeper |
| jet | #2892 | 3L/768d shallow+wide |
| shouko | #2909 | heads=16/4 ablation + gc=1.0 |
| askeladd | #2914 | MLP ratio=6/2 + gc=1.0+WD=1e-2 |
| chrome | #2923 | torch.compile + gc=1.0 (throughput + stability) |

### Theme 4: Scheduler Innovations (CODE CHANGES)

| Student | PR | Experiment |
|---|---|---|
| megumi | #2894 | Linear warmup+cosine |
| mugen | #2895 | CosineAnnealingWarmRestarts T_mult |
| vash | #2905 | OneCycleLR |

### Theme 5: Training Innovations (CODE CHANGES)

| Student | PR | Experiment |
|---|---|---|
| nobara | #2897 | LLRD (layer-wise LR decay) |
| usopp | #2920 | Gradient noise injection (Neelakantan 2015) |
| sukuna | #2903 | SWA at cosine troughs |
| spike | #2901 | Huber/log-cosh loss |
| stark | #2902 | Gradient accumulation |

### Theme 6: Throughput + Seeds + Ablations

| Student | PR | Experiment |
|---|---|---|
| piccolo | #2898 | torch.compile throughput (baseline) |
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

### Priority 1: TandemFoil Paper Baseline
- **No val anchor exists yet for TF Paper.** First student to become idle should run the best current config (EMA decay=0.999, 3L/192d or 4L/512d, Lion lr=1.25e-4 for TF) on tandemfoil_paper to establish a baseline.
- Paper targets (Table 6 MGN + PRE-RES-FREE+RES-COMB): 0.10/0.18/0.36/0.13/0.14/0.21 per task.
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

1. **TandemFoil Paper baseline run** — assign first available idle student
2. Watch Wave 3 results (frieren #2928 rel-L2 is the highest-priority result)
3. Review any WIP PRs that become ready
4. All new assignments: 4-dataset coverage mandatory per human team directive
5. If DM recipe transfer (Theme 1) fails universally → escalate to geometry-separated encoding
6. If Wave 3 bold ideas show promise → double down with follow-up assignments across all 4 datasets
7. Check for human team messages on GitHub issues (priority — check very frequently)
