# SENPAI Research State

- **Date:** 2026-04-21 22:45 (Wave 3 — Bold New Directions)
- **Branch:** radford
- **Fleet status:** 59 live students, ALL ASSIGNED (0 idle)
- **Current relaunch budget:** inherit pod env defaults
  - `SENPAI_TIMEOUT_MINUTES=360`
  - `SENPAI_MAX_EPOCHS=999`

## Paper-Facing Snapshot

| Dataset | Paper-facing metric | Current best | Target / reference | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **33.88** (#2810) | no external scalar | needs test from new EMA best |
| TandemFoil Paper | `test_primary/field_mse` | not run yet on `radford` | `0.10 / 0.18 / 0.36 / 0.13 / 0.14 / 0.21` by task | new calibration lane |
| AirfRANS | `test_primary/surface_mse` | **0.003** (#2824) | `0.0043` | **BEATEN** — val now 0.000727 |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **6.24%** (#2691) | `3.71%` | **MAIN GAP — 1.68x** |

## Steering Anchors (validation, for experiment decisions)

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **26.134** (#2899 MERGED — EMA decay=0.999) |
| AirfRANS | `val_primary/surface_mse` | **0.000727** (#2899 MERGED — EMA decay=0.999, ep206) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **4.619%** (#2691) |

## Main Scientific Goal

A shared recipe whose core changes work across:

- the main TandemFoil parity target
- the new TandemFoil paper-calibration target
- AirfRANS
- DrivAerML

DrivAerML is still the main gap. Corrected EMA warmup is now the shared recipe
for TF+AF, and the paper-calibration Tandem benchmark exists to tell us whether
Tandem-side gains are only helping the parity contract or also the literature-facing one.

## Mandatory Config Rules (UPDATED after EMA merge)

- **TF + AF:** Use `--ema-decay 0.999` (NO --no-use-ema). decay=0.999 > 0.9999 on both.
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

## Default Assignment Pattern

Cross-dataset: TF+AF use EMA decay=0.999, DM still uses --no-use-ema.

When a hypothesis is relevant to TandemFoil generalization or paper
comparability, include both:

- `target/icml2026/tandemfoil/`
- `target/icml2026/tandemfoil_paper/`

## ACTIVE EXPERIMENTS — 59 WIP PRs

### Theme 7: Bold New Directions (Wave 3, 2026-04-21 22:45)

9 new hypothesis families testing genuinely new mechanisms — loss reformulation, architecture innovations, optimization paradigm shifts, and physics-informed features.

| Student | PR | Experiment | Risk |
|---|---|---|---|
| frieren | #2928 | **Relative L2 Training Loss** — align DM training loss with eval metric | LOW |
| nezuko | #2929 | **SwiGLU FFN Replacement** — gated linear units for all Transolver blocks | LOW |
| violet | #2930 | **Stochastic Depth (DropPath)** — layer-level regularization 0.1/0.2 | LOW |
| gilbert | #2931 | **Prodigy Optimizer** — parameter-free LR adaptation | LOW-MED |
| kohaku | #2932 | **Global Context Token** — break slice-local attention bottleneck | MEDIUM |
| emma | #2933 | **Surface Normals + Curvature** — differential geometry input features | MEDIUM |
| chihiro | #2934 | **Conservation Auxiliary Loss** — div(u)=0 physics regularization | MED-HIGH |
| shoya | #2935 | **MoE FFN Layers** — sparse expert routing for physics-regime specialization | MED-HIGH |
| mitsuha | #2936 | **SDF Wall-Distance Feature** — signed distance field geometry embedding | MEDIUM |

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
| casca | #2881 | gc=1.5/gc=2.0 |
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

## Current Research Themes and Priorities

### Priority 1: DrivAerML Gap Closure (4.619% → 3.71%)
- **Loss alignment** (frieren #2928): Most direct fix — training on relative L2 directly matches the evaluation metric
- **Architectural upgrades** (nezuko #2929 SwiGLU, kohaku #2932 global context): Address known Transolver limitations
- **Regularization** (violet #2930 DropPath): New orthogonal regularization dimension
- **Physics-informed features** (emma #2933 curvature, mitsuha #2936 SDF): Give model explicit geometric knowledge it currently must learn implicitly

### Priority 2: Cross-Benchmark Recipe Validation
- Wave 3 experiments test across all 4 benchmarks by default
- TandemFoil Paper benchmark provides calibration against published numbers
- Ideas that help DM but hurt TF/AF are dataset hacks; we want shared wins

### Priority 3: Optimization Paradigm Shift
- Prodigy (gilbert #2931): If the LR search is truly exhausted, adaptive optimizers may find new trajectories
- MoE (shoya #2935): Sparse expert specialization is fundamentally different from dense FFN
- Conservation loss (chihiro #2934): Physics constraints as regularization

## Next Priorities

1. Watch Wave 3 results (frieren #2928 rel-L2 is the highest-priority result)
2. Review any WIP PRs that become ready
3. If DM recipe transfer (Theme 1) fails universally → escalate to geometry-separated encoding
4. If Wave 3 bold ideas show promise → double down with follow-up assignments
5. Check for human team messages on GitHub issues
