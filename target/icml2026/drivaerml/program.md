<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# DrivAerML

DrivAerML is the 3D transfer benchmark for the ICML 2026 sprint.

## Role in the paper

DrivAerML is the surface-first automotive benchmark. It tests whether the shared
stack transfers from 2D airfoil settings to 3D vehicle pressure prediction.

## Primary metrics

Use the **DrivAerML relative-L2 contract** used by AB-UPT and continued in later automotive-surrogate papers such as Transolver-3.

- **Primary benchmark metric for this sprint**
  - `surface pressure relative L2 (%)` on the packaged public validation / test split

- **Exact calculation**
  - For one case with target point cloud `Y` and prediction `Ŷ`, compute:
  - `rel_l2_case = 100 * ||Ŷ - Y||_2 / ||Y||_2`
  - AB-UPT defines the dataset score as the arithmetic mean of those per-case relative-L2 values over the evaluation split.
  - Evaluation is done on **unnormalized** targets and predictions.
  - The shared trainer now follows that same contract for paper-facing DrivAerML metrics and logs the validation scalar as `val_primary/surface_rel_l2_pct`, with the matching test scalar logged as `test_primary/surface_rel_l2_pct`.
  - For debugging and auditability, the trainer also logs the raw ratio before percent-scaling as `surface_rel_l2`.
  - When point-limited DrivAerML sampling is enabled in the shared trainer:
    - training repeats each case `ceil(N / points_per_view)` times per epoch and draws random point subsets with replacement
    - validation/test split each case into deterministic strided point views so every point is evaluated exactly once
    - the reported relative-L2 metric is then re-aggregated back to the exact per-case numerator/denominator over the full case, not averaged over chunk-level scores

- **Field names for literature comparison**
  - Surface:
    - `ps` / surface pressure
    - optionally wall shear stress `tau`
  - Volume:
    - velocity `u`
    - pressure `pv`
    - sometimes vorticity `omega`

- **Current repo sprint contract**
  - The packaged PVC data exposes `surface_cp.npy`, and the current sprint target is surface-first.
  - For this repo, the paper-facing primary metric is therefore:
    - relative L2 on the packaged `surface_cp` target over the packaged public surface split
  - Volume metrics are secondary and should only be reported when a PR explicitly targets the small processed volume subset.
  - For grouped-domain models in this target, evaluation is full-field over all available surface points in each case.

- **Split contract**
  - Default split for this repo target:
    - `394 train / 34 val / 46 test`
  - This is the packaged public processed split present on the PVC.
  - It is slightly smaller than the nominal public split discussed in AB-UPT because `10` processed public cases are absent from the packaged PVC.

- **Important comparison note**
  - The packaged PVC split is not identical to the nominal AB-UPT public split because `10` processed public cases are missing.
  - That split discrepancy must be disclosed whenever we compare against AB-UPT, PhysicsNeMo, or Transolver-3 tables.
  - The local `reference_abupt` model path should be treated as an **AB-UPT-style reference architecture**, not a byte-for-byte reproduction of the published AB-UPT DrivAerML setup.
  - The main remaining differences are:
    - **split**: this target uses the packaged processed split `394 / 34 / 46`, whereas AB-UPT reports the nominal public split `400 / 34 effective val / 50 test`
    - **targets**: the packaged sprint path is surface-first and predicts packaged `surface_cp`, whereas AB-UPT predicts 4 surface variables and 7 volume variables on the full task
    - **token counts**: the AB-UPT paper uses `16384` geometry supernodes, `16384` surface anchors, and `16384` volume anchors for DrivAerML; the local sprint defaults are smaller and configurable for practicality
    - **architecture depth / recipe**: the paper uses the full published AB-UPT block schedule and training recipe (`500` epochs, `bs=1`, Lion with warmup+cosine, mixed precision). The local sprint trainer exposes a shared training loop and does not yet hard-pin every AB-UPT hyperparameter to those paper values
  - The metric path is now aligned for paper-facing reporting, but the model/training path remains a **benchmark-aligned approximation** rather than an exact reproduction.

## Sources

- AB-UPT paper: <https://openreview.net/pdf?id=nwQ8nitlTZ>
- `milieu_cfd` common evaluator: `scripts/eval_drivaerml.py` and `nn_cfd/noether/callbacks.py`
- Transolver-3 paper: <https://arxiv.org/abs/2602.04940>

## Training schedules in related work

- `AB-UPT` on DrivAerML:
  - reported at `500` epochs, `bs=1`, Lion, mixed precision
- `Transolver-3` industrial benchmark section:
  - reports `500` epochs with `bs=1` for the compared industrial aerodynamic benchmarks, including DrivAerML
- `Transolver++`:
  - does **not** use DrivAerML; it reports `200` epochs on `DrivAerNet++`
- original `Transolver`:
  - does **not** use DrivAerML; the released car-design task is `ShapeNetCar` with `200` epochs in the public repo
- `SpiderSolver`:
  - does **not** use DrivAerML; the released public training commands cover `ShapeNetCar` (`200` epochs), `AirfRANS` (`398`), and `BloodFlow` (`500`)

## Code boundaries

- benchmark-local data pipeline: `data/`
- shared training entrypoint: `../train.py`
- shared models and collate contracts: `../core/`
