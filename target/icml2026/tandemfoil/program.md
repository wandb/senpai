<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet

TandemFoilSet is the load-bearing benchmark inside the ICML 2026 sprint target.

## Problem

We are training machine-learning surrogates for computational fluid dynamics. The general task is full-field flow prediction: given geometry, operating conditions, and other problem-specific inputs, predict physically meaningful flow quantities such as velocity and pressure over a mesh, graph, or point cloud.

Benchmark details for the tandemfoil dataset live in `data/README.md`. Shared
training now happens via `../train.py`, while the tandemfoil-specific pipeline
stays under `data/`.

## Codebase

- `../train.py` — shared ICML sprint trainer. **Primary editable entrypoint.**
- `data/prepare.py` — tandemfoil dataset loading and collation. **Read-only during normal experiment PRs.**
- `data/prepare_multi.py` — tandemfoil-specific preprocessing and feature engineering. **Read-only during normal experiment PRs.**
- `data/utils.py` — visualization. **Read-only.**
- `data/README.md` — benchmark splits, dataset assumptions, and problem-local documentation.

## Metrics

**The goal: lowest physically meaningful validation error on the most decision-relevant regions and regimes.** We track:
- **Primary validation metric** — `<define this for the active dataset or benchmark; this is the main metric used to rank results and make merge decisions>`
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
