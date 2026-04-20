<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# AirfRANS

AirfRANS is the 2D transfer benchmark for the ICML 2026 sprint.

## Role in the paper

AirfRANS is the first portability check beyond TandemFoilSet. It lets us test
whether the clean shared stack transfers from tandemfoil geometry to a standard
single-airfoil CFD benchmark with established literature baselines.

## Primary metrics

Use the **official AirfRANS benchmark metric family**. For literature comparisons, this dataset is **not** ranked by relative L2. The published AirfRANS contract uses **MSE** for the field metrics, and SpiderSolver explicitly says it follows that choice.

- **Primary benchmark metrics**
  - `Surf MSE`
  - `Vol MSE`

- **Exact calculation**
  - For each evaluation case, split the mesh into:
    - `Surf`: airfoil boundary nodes
    - `Vol`: non-surface internal mesh nodes
  - Compute unreduced mean squared error separately on those two sets:
    - `loss_surf_var = MSE(y_hat[surf], y[surf]).mean(dim=0)`
    - `loss_vol_var = MSE(y_hat[~surf], y[~surf]).mean(dim=0)`
  - Then average over target channels to get one scalar per case:
    - `loss_surf = loss_surf_var.mean()`
    - `loss_vol = loss_vol_var.mean()`
  - Finally average those case-level scalars over the evaluation split.
  - In the original AirfRANS training / scoring code, these MSEs are computed on the **normalized target tensors** produced by the official `Dataset(...)` loader using the training-set normalization coefficients, not on denormalized physical-unit fields.
  - In the shared trainer, validation metrics are logged on `*_val` splits for model selection, while paper-facing comparison numbers come from the matching `*_test` split. The harness aliases these as `val_primary/surface_mse` and `test_primary/surface_mse`.

- **Official target-field contract**
  - Apples-to-apples AirfRANS leaderboard comparisons use the four official targets:
    - `u_x`
    - `u_y`
    - `p`
    - `nut`

- **Current repo sprint contract**
  - The shared trainer now uses the four-field official target contract by default:
    - `u_x`
    - `u_y`
    - `p`
    - `nut`
  - Paper-facing AirfRANS numbers from this target must therefore come from the four-field MSE scorer, not from any earlier three-field smoke-test path.

- **Comparison contract**
  - When comparing against literature, report:
    - the task name: `full`, `scarce`, `reynolds`, or `aoa`
    - `Surf MSE`
    - `Vol MSE`
  - Do not relabel these as MAE or relative L2.
  - If a run is selected on validation, the final paper number must still be recomputed on the official task test split before citing it.

## Sources

- AirfRANS dataset paper / official benchmark: <https://openreview.net/forum?id=Zp8YmiQ_bDC>
- official AirfRANS repo metric path: `Extrality/AirfRANS` `metrics.py`
- SpiderSolver metric note: <https://openreview.net/pdf/054dcb68b120d4b02b356ca2f357ae4fbd463354.pdf>

## Code boundaries

- benchmark-local data pipeline: `data/`
- shared training entrypoint: `../train.py`
- shared models and collate contracts: `../core/`
