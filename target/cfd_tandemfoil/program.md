<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai

Autonomous neural network research on CFD surrogates, coordinated through GitHub PRs.

## Problem

We are training machine-learning surrogates for computational fluid dynamics. The general task is full-field flow prediction: given geometry, operating conditions, and other problem-specific inputs, predict physically meaningful flow quantities such as velocity and pressure over a mesh, graph, or point cloud.

Problem-specific benchmark details for the current target live in `data/README.md` and the current training code. This file should stay focused on the broader CFD surrogate research objective rather than a single dataset or geometry family.

## Codebase

- `train.py` — **primary training script + model architecture**. **Modifiable.** (Contains the current model, training loop, losses, and validation logic for the active CFD problem.)
- `data/prepare.py` — dataset loading and collation. **Read-only.**
- `data/prepare_multi.py` — problem-specific preprocessing and feature engineering. **Read-only.**
- `data/utils.py` — visualization. **Read-only.**
- `data/README.md` — benchmark splits, dataset assumptions, and problem-local documentation.

## Metrics

**The goal: lowest physically meaningful validation error on the most decision-relevant regions and regimes.** We track:
- **Boundary or surface error** — when applicable, this is usually the most important metric because it is closest to engineering use.
- **Volume or field error** — mean absolute error over the full predicted flow field.
- **Validation loss** — the combined objective optimized during training.
- **Pressure fidelity and problem-specific split or OOD metrics** — whichever additional metrics best capture physical usefulness for the active benchmark.

Lower is better. When multiple metrics exist, prioritize the ones that best reflect physical usefulness and engineering decision quality: usually boundary or surface accuracy, pressure accuracy, and robustness on harder operating regimes.

**VRAM**: GPUs have 96GB.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**Timeout**: Each training run is capped by time or epochs. Do not override this.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team.
