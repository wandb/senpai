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
  - Over all valid evaluated entries in the split, compute:
  - `rel_l2 = 100 * sqrt( sum((y_hat - y)^2) / sum(y^2) )`
  - The sums are taken **globally over all valid entries in the split**, not as per-case relative errors averaged afterward.
  - This is the same accumulator style used by the common `milieu_cfd` / Noether DrivAerML evaluator:
    - accumulate `sum_sq_error`
    - accumulate `sum_sq_target`
    - report `sqrt(sum_sq_error / sum_sq_target)`
  - In that evaluator, the comparison tensors follow the packaged `normalizers.json` contract used by the dataset pipeline; the paper tables then report the resulting relative-L2 value in percent.

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

- **Split contract**
  - Default split for this repo target:
    - `394 train / 34 val / 46 test`
  - This is the packaged public processed split present on the PVC.
  - It is slightly smaller than the nominal public split discussed in AB-UPT because `10` processed public cases are absent from the packaged PVC.

- **Important comparison note**
  - The shared local trainer currently logs generic MAE-style diagnostics by default.
  - Those are useful for optimization and debugging.
  - They are **not** the paper-facing benchmark metric for DrivAerML.
  - For literature comparison, use **relative L2 in percent**.

## Sources

- AB-UPT paper: <https://openreview.net/pdf?id=nwQ8nitlTZ>
- `milieu_cfd` common evaluator: `scripts/eval_drivaerml.py` and `nn_cfd/noether/callbacks.py`
- Transolver-3 paper: <https://arxiv.org/abs/2602.04940>

## Code boundaries

- benchmark-local data pipeline: `data/`
- shared training entrypoint: `../train.py`
- shared models and collate contracts: `../core/`
