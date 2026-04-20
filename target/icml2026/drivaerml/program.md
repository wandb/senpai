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
  - The local `reference_abupt` model path now uses full surface anchors at evaluation time, but it still keeps a tractable sampled-geometry branch during training and evaluation. Treat it as an architecture comparison, not a byte-for-byte reproduction of the published AB-UPT training stack.

## Sources

- AB-UPT paper: <https://openreview.net/pdf?id=nwQ8nitlTZ>
- `milieu_cfd` common evaluator: `scripts/eval_drivaerml.py` and `nn_cfd/noether/callbacks.py`
- Transolver-3 paper: <https://arxiv.org/abs/2602.04940>

## Code boundaries

- benchmark-local data pipeline: `data/`
- shared training entrypoint: `../train.py`
- shared models and collate contracts: `../core/`
