<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai

Autonomous neural network research on CFD surrogates, coordinated through GitHub PRs.

## Problem

We are training a neural network surrogate for CFD (computational fluid dynamics) on the TandemFoilSet dataset. The task is full-field flow prediction: given airfoil geometry and flow conditions, predict velocity (Ux, Uy) and pressure (p) at every mesh node.

## Codebase

- `prepare.py` — dataset loading, preprocessing, collation. **Read-only.**
- `train.py` — training script: hyperparameters, optimizer, training loop, loss formulation. **Modifiable.**
- `transolver.py` — model architecture (Transolver with physics attention). **Modifiable.**
- `utils.py` — visualization. **Read-only** (not relevant to metrics).
- `DATASET_REPORT.md` — dataset documentation.

No new packages beyond `pyproject.toml`.

## Metrics

**The goal: lowest validation losses.** We track:
- **Surface MAE** — mean absolute error on airfoil surface nodes (Ux, Uy, p). **Most important** — these are the quantities engineers care about.
- **Volume MAE** — mean absolute error on volume (field) nodes.
- **val/loss** — the combined validation loss (vol_loss + surf_weight * surf_loss).

Lower is better. Surface accuracy (especially pressure) matters most.

**VRAM**: GPUs have 96GB. Some increase is acceptable for meaningful gains, but should not OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**Timeout**: Each training run is capped at 20 minutes. Do not override this — experiments should be fast iterations, not long runs.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model.