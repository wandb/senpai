# SENPAI Research State

- **Date:** 2026-04-21 18:15 (DrivAerML Refocus Relaunch — Wave 2, All Students Assigned)
- **Branch:** radford
- **Fleet status:** 50 live students, ALL ASSIGNED (0 idle)
- **Current relaunch budget:** inherit pod env defaults
  - `SENPAI_TIMEOUT_MINUTES=360`
  - `SENPAI_MAX_EPOCHS=999`

## Paper-Facing Snapshot

| Dataset | Paper-facing metric | Current best | Target / reference | Status |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **33.88** (#2810) | no single packaged external scalar | strong, no longer bottleneck |
| AirfRANS | `test_primary/surface_mse` | **0.003** (#2824) | `0.0043` | **BEATEN** |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **6.24%** (#2691) | `3.71%` | **MAIN GAP — 1.68x** |

## Steering Anchors (validation, for experiment decisions)

| Dataset | Metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **30.10** (#2810) |
| AirfRANS | `val_primary/surface_mse` | **0.001236** merged (#2828), **0.001095** pending (#2823 — unstable) |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **4.619%** (#2691) |

## Main Scientific Goal

A shared recipe whose core changes work across TandemFoil, AirfRANS, and DrivAerML.
DrivAerML is the main gap. All new work is DrivAerML-weighted and cross-dataset.

## Mandatory Config Rules

- `--no-use-ema` mandatory everywhere (robin #2899 tests a fix)
- `--epochs 999` mandatory
- DrivAerML: `--batch-size 1 --drivaerml-train-surface-points 50000 --drivaerml-eval-surface-points 50000 --max-train-batches 394 --max-eval-batches 200`
- Lion for TandemFoil; AdamW for AirfRANS/DrivAerML

## Negative Results (Do Not Repeat)

- **No-Lookahead** (#2834): fatal across all datasets — diverges AF/DM, TF regresses
- **3L/192d on TF or DM** (#2825): too shallow at current scale
- **Pressure-weighted loss at wrong architecture** (#2801): pressure weighting hurts if applied at non-golden depth

## Default Assignment Pattern

Cross-dataset by default: 1 TF + 1 AF + 2-4 DrivAerML + nearby variants per student.

## ACTIVE EXPERIMENTS — 50 WIP PRs

### Theme 1: AirfRANS Recipe Transfer to DrivAerML (HIGHEST PRIORITY)

| Student | PR | Experiment |
|---|---|---|
| brook | #2878 | gc=1.0+WD=1e-2 (flagship compound) |
| bulma | #2879 | T_max=10+gc=1.0 |
| canute | #2880 | Full recipe: lr=7e-4+gc=1.0+WD=1e-2 |
| chopper | #2882 | T_max=15+gc=1.0+WD=1e-2 |
| einar | #2883 | gc=1.0+T_max=20+WD=1e-2 |
| yuji | #2908 | Compound gc=1.0+WD=1e-2+T_max=15 |
| zenitsu | #2911 | T_max=5+gc=1.0+WD=1e-2 (AirfRANS golden scheduler) |
| edward | #2916 | lr=6e-4+gc=1.0+WD=1e-2+T_max=10 (intermediate LR) |

### Theme 2: DrivAerML LR+gc Exploration

| Student | PR | Experiment |
|---|---|---|
| faye | #2885 | lr=7e-4+gc=1.0 |
| franky | #2886 | lr=4e-4+gc=1.0 |
| gohan | #2887 | gc=1.0+T_max=10 LR scan |
| gojo | #2888 | gc=0.5+T_max=10 |
| casca | #2881 | gc=1.5/gc=2.0 |
| wolfwood | #2907 | lr=8e-4+gc=1.0 |
| jin | #2893 | lr=1e-3+gc=1.0 |
| shinobu | #2912 | WD=3e-2/5e-2+gc=1.0 (heavy regularization) |

### Theme 3: DrivAerML Architecture

| Student | PR | Experiment |
|---|---|---|
| griffith | #2889 | 3L/512d+gc=1.0 |
| guts | #2890 | 4L/768d ultra-wide |
| himmel | #2891 | 5L/512d deeper |
| jet | #2892 | 3L/768d shallow+wide |
| shouko | #2909 | heads=16/4 ablation + gc=1.0 |
| askeladd | #2914 | MLP ratio=6/2 + gc=1.0+WD=1e-2 |

### Theme 4: Scheduler Innovations (CODE CHANGES)

| Student | PR | Experiment |
|---|---|---|
| megumi | #2894 | Linear warmup+cosine |
| mugen | #2895 | CosineAnnealingWarmRestarts T_mult |
| vash | #2905 | OneCycleLR |

### Theme 5: Optimizer + Training Recipe Innovations (CODE CHANGES)

| Student | PR | Experiment |
|---|---|---|
| nobara | #2897 | LLRD (layer-wise LR decay) |
| robin | #2899 | Corrected EMA with warmup |
| usopp | #2904 | Momentum-SAM |
| sukuna | #2903 | SWA at cosine troughs |
| spike | #2901 | Huber/log-cosh loss |
| stark | #2902 | Gradient accumulation |

### Theme 6: Throughput + Seeds + Ablations

| Student | PR | Experiment |
|---|---|---|
| piccolo | #2898 | torch.compile throughput |
| sanji | #2900 | surface-only DrivAerML |
| vegeta | #2906 | 360min multi-seed replication |
| nami | #2896 | Lion higher LR on DrivAerML |
| eren | #2910 | max-train-batches=788 (2x data/epoch) |
| rei | #2913 | surface-points=75k resolution scaling |
| levi | #2915 | no-Fourier ablation (faster epochs) |

### Continuing from Previous Wave (~20 students)

| Student | PR | Dataset | Focus |
|---|---|---|---|
| chrome | #2873 | DrivAerML | LR headroom 4.5-6e-4 |
| zoro | #2870 | DrivAerML | Lower LR 2-3e-4 |
| shinji | #2869 | DrivAerML | gc=0.5/0.7+T_max=25/30 |
| norman | #2868 | DrivAerML | 2L/512d+3L/512d |
| historia | #2867 | DrivAerML | 3L/256d+3L/384d |
| kaneda | #2858 | DrivAerML | gc=0.7/0.8/0.9 |
| kakashi | #2823 | AirfRANS | gc=1.0+T_max=10 stabilization (lr=5e-4 + reproduce) |
| inosuke | #2874 | AirfRANS | 2L+T_max=10 compound |
| thorfinn | #2786 | AirfRANS | gc=1.0+T_max=7 extended budget |
| taki | #2814 | DrivAerML | Mild regularization |
| Various | #2835-2876 | TandemFoil | LR/depth/gc fine-tuning |

## Research Insights from Literature

1. **Corrected EMA** (robin #2899): no-EMA was a bug fix, not a design choice. timm-style warmup fixes it.
2. **SWA** (sukuna #2903): weight averaging at cosine troughs → flatter minima.
3. **MSAM** (usopp #2904): flat-minima bias without SAM's 2x cost.
4. **LLRD** (nobara #2897): 1-3% improvement in <10 epoch regime.
5. **AB-UPT** achieves 3.71% via geometry-separated encoding — next escalation if recipe transfer fails.
6. **Transolver-3** (arxiv 2602.04940): amortized mesh subset training for throughput.
7. **Lookahead is load-bearing**: removing it causes catastrophic divergence on AF/DM (confirmed #2834).

## Human Guidance

Issue #2545: Focus on DrivAerML and cross-dataset evidence. No new directives.

## Next Priorities

1. Review PRs as results come in (~30-360 min)
2. **brook #2878** (gc+WD transfer) = THE paper question — first to report
3. **bulma #2879** (T_max=10+gc) = second priority
4. **kakashi #2823** (AirfRANS 0.001095 reproducibility) = AirfRANS question
5. Watch for code-change PRs (robin #2899, megumi #2894, sukuna #2903, usopp #2904) — these need careful review
6. If recipe transfer fails → escalate to geometry-separated encoding (AB-UPT approach)

## Potential Next Research Directions

- **Residual prediction on DrivAerML**: works for TandemFoil (--residual-prediction), untested on DM
- **Geometry-separated encoding**: AB-UPT approach — encode surface geometry separately from flow fields
- **Mesh decimation / adaptive sampling**: sample more points near high-curvature regions
- **Ensemble of 4L/512d seeds**: if variance is ±0.2% and we ensemble 3 seeds, could get deterministic improvement
- **Amortized subset training** (Transolver-3): alternate between subsets of the mesh per step
