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

- `cfd_tandemfoil/train.py` — **primary training script + model architecture**. **Modifiable.** (Contains the model inline, plus training with 4 val tracks across 7 data sources.)
- `cfd_tandemfoil/data/prepare.py` — dataset loading and collation. **Read-only.**
- `cfd_tandemfoil/data/prepare_multi.py` — extended preprocessing (24-dim x, foil-2 features). **Read-only.**
- `cfd_tandemfoil/data/utils.py` — visualization. **Read-only.**
- `cfd_tandemfoil/data/README.md` — benchmark splits and dataset documentation.

## Metrics

**The goal: lowest validation surface MAE.** We track:
- **Surface MAE** — mean absolute error on airfoil surface nodes (Ux, Uy, p). **Most important** — these are the quantities engineers care about.
- **Volume MAE** — mean absolute error on volume (field) nodes.
- **val/loss** — the combined validation loss.

Lower is better. Surface accuracy (especially pressure) matters most.

**VRAM**: GPUs have 96GB.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**Timeout**: Each training run is capped by time or epochs. Do not override this.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team.