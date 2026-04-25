# TandemFoilSet Benchmark Notes

This repo now has **two distinct TandemFoilSet benchmark contracts**. They are
both useful, but they are **not numerically interchangeable**.

## Executive summary

- The **original TandemFoilSet paper** benchmark relevant to our repo is
  Experiment 4 / Table 6.
- That paper-facing contract uses **test MSE on the full flow field**, not a
  surface-only metric.
- Our literature-facing reproduction target is
  [`target/icml2026/tandemfoil_paper/`](../target/icml2026/tandemfoil_paper/program.md).
  Its primary metric is `field_mse`.
- Our main ICML sprint target is
  [`target/icml2026/tandemfoil/`](../target/icml2026/tandemfoil/program.md).
  It intentionally uses a different split family and a different metric:
  denormalized `surface_pressure_mae` on the public `kagent` v2 split design.
- Do **not** compare `tandemfoil/ surface_pressure_mae` directly against the
  paper’s Table 6 `field_mse`.

## 1. Original paper contract

Primary source:

- TandemFoilSet paper: [OpenReview PDF](https://openreview.net/pdf?id=4Z0P4Nbosn)

### What the paper actually reports

The relevant paper statements are:

- datasets are “uniformly sampled in a `8:1:1` ratio for training, validation,
  and test sets, respectively, unless otherwise stated”
- Experiment 4 uses:
  - uniform sampling for `cruise_random_uniform` and `racecar_uniform`
  - top and bottom `5%` tails of a selected variable as `test` for the
    extrapolation tasks
  - the middle `90%` uniformly split into train/val
- the training appendix says the model predicts three fields:
  `[Ux, Uy, p]`
- the training appendix also says a “standard z-score style normalisation” is
  used with the **training-set mean and standard deviation**
- Experiment 4 says “Table 6 shows that the `MSE` test losses are higher...”
- later, the appendix introduces **boundary MSE** separately for the
  three-airfoil extension, which supports reading Table 6 as the overall
  full-field metric rather than a surface-only one

### Paper tasks and reference numbers

| Task | Split style | Paper baseline | Paper best |
| --- | --- | ---: | ---: |
| `cruise_random_uniform` | uniform `8:1:1` | `1.79 ± 1.38` | `0.10 ± 0.13` |
| `cruise_random_aoa_extrap` | lowest/highest `5%` AOA tails to test | `2.03 ± 1.96` | `0.18 ± 0.24` |
| `cruise_random_re_extrap` | lowest/highest `5%` Re tails to test | `4.85 ± 1.82` | `0.36 ± 0.53` |
| `cruise_random_stagger_extrap` | lowest/highest `5%` stagger tails to test | `1.74 ± 1.66` | `0.13 ± 0.17` |
| `cruise_random_gap_extrap` | lowest/highest `5%` gap tails to test | `1.95 ± 1.68` | `0.14 ± 0.20` |
| `racecar_uniform` | uniform `8:1:1` | `0.61 ± 0.51` | `0.21 ± 0.29` |

### Repo reproduction target: `tandemfoil_paper`

Relevant local docs:

- [`target/icml2026/tandemfoil_paper/program.md`](../target/icml2026/tandemfoil_paper/program.md)
- [`target/icml2026/tandemfoil_paper/data/README.md`](../target/icml2026/tandemfoil_paper/data/README.md)
- [`target/icml2026/tandemfoil_paper/data/split_paper_experiment4.py`](../target/icml2026/tandemfoil_paper/data/split_paper_experiment4.py)

Repo contract:

- primary metric: `field_mse`
- definition: normalized full-field MSE over all valid nodes and all three
  channels `[Ux, Uy, p]`
- normalization: task-local train-set `y_mean` / `y_std`
- auxiliary diagnostics:
  - `surface_mse`
  - `volume_mse`
- if training enables an extra target transform such as `--asinh-pressure`,
  paper-facing evaluation must still decode predictions back to raw target space
  and recompute `field_mse` in the paper’s plain z-score space

Current status:

- no clean literature-facing benchmark result yet
- latest memo:
  [`STATUS_2026-04-22-0923_radford_live_status_after_cross_dataset_wave.md`](./STATUS_2026-04-22-0923_radford_live_status_after_cross_dataset_wave.md)
  says the lane is still immature
- best visible debug check there is `test_primary/field_mse = 0.151`

## 2. Packaged parity contract

Primary local docs:

- [`target/icml2026/tandemfoil/program.md`](../target/icml2026/tandemfoil/program.md)
- [`target/icml2026/tandemfoil/data/README.md`](../target/icml2026/tandemfoil/data/README.md)
- [`target/icml2026/tandemfoil/data/split_manifest_tandemfoilset_v2.json`](../target/icml2026/tandemfoil/data/split_manifest_tandemfoilset_v2.json)
- public split rationale:
  [`kagent` SPLITS.md](https://github.com/tcapelle/kagent/blob/main/cfd-competition/organizer/SPLITS.md)

This is the active `tandemfoil/` target used for the ICML sprint parity story.
It follows the public `kagent` split design, not the original paper’s Experiment
4 split family.

### Active split family

Balanced validation and test tracks:

| Split | Cases | What it tests |
| --- | ---: | --- |
| `val_single_in_dist` / `test_single_in_dist` | `100 / 200` | single-foil interpolation sanity check |
| `val_geom_camber_rc` / `test_geom_camber_rc` | `100 / 200` | race-car tandem geometry generalization to unseen front-foil camber |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | `100 / 200` | cruise tandem geometry generalization to unseen front-foil camber |
| `val_re_rand` / `test_re_rand` | `100 / 200` | Reynolds-number generalization across tandem training domains |

Train / val / test totals:

- train: `1499`
- val: `400`
- test: `800`

High-level rationale:

- full-file geometry holdouts give clean unseen-foil-family tests
- a stratified Reynolds-number holdout checks cross-regime generalization
- a random single-foil holdout provides an easier sanity-check track
- all four tracks are balanced so one split does not dominate the summary metric

### Active metric family

Primary metric:

- `val_primary/surface_pressure_mae`
- `test_primary/surface_pressure_mae`

Definition:

- pressure-channel MAE on all valid surface nodes in a split
- computed **after full denormalization back to the original target space**
- aggregated globally over valid surface nodes, not by averaging per-case MAEs

Summary metric:

- `val_eq4/surface_pressure_mae`
- `test_eq4/surface_pressure_mae`

This is the equal-weight mean of:

- `single_in_dist`
- `geom_camber_rc`
- `geom_camber_cruise`
- `re_rand`

Historical note:

- the old legacy split family still exists in the repo as `split_manifest.json`
  and `legacy_noam/p_in`, `p_oodc`, `p_tan`, `p_re`
- those are still useful for historical noam-lineage interpretation
- they are not the same thing as the active v2 manifest

### Current parity status

Latest strong anchor:

- [`STATUS_2026-04-22-0923_radford_live_status_after_cross_dataset_wave.md`](./STATUS_2026-04-22-0923_radford_live_status_after_cross_dataset_wave.md)
  reports `test_primary/surface_pressure_mae = 24.581` for run `nrn0q3ct`

Earlier anchor:

- [`STATUS_2026-04-21-1759_radford_relaunch_status_and_refocus.md`](./STATUS_2026-04-21-1759_radford_relaunch_status_and_refocus.md)
  reported `test_primary/surface_pressure_mae = 33.88`

Interpretation:

- `tandemfoil/` is currently our strongest **internal** TandemFoil story
- but it is not the clean literature-facing Table 6 comparison contract

## 3. What to compare against

Use this rule of thumb:

- if the claim is “how do we compare to the original TandemFoilSet paper?” use
  `tandemfoil_paper/` and compare `field_mse` only
- if the claim is “how strong is our current TandemFoil parity target inside the
  ICML sprint?” use `tandemfoil/` and report `surface_pressure_mae`
- if a table mixes `surface_pressure_mae` and Table 6 `field_mse` without
  relabeling the contracts, it is scientifically misleading

## 4. Bottom line

- The original paper contract is **full-field normalized MSE** on Experiment 4
  tasks.
- The active repo parity contract is **denormalized surface-pressure MAE** on
  the public `kagent` split family.
- Both are worth keeping.
- They must stay clearly separated in the paper, in status memos, and in agent
  instructions.
