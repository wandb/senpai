<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# ICML 2026 CFD Sprint

Autonomous research target for the ICML 2026 AI for Science workshop sprint.

## Problem

This target packages the experiments for the ICML paper around three CFD surrogate
benchmarks under one harness-compatible problem directory:

- `tandemfoil/` — TandemFoilSet, the load-bearing benchmark and current strongest evidence
- `airfrans/` — AirfRANS, the 2D transfer benchmark
- `drivaerml/` — DrivAerML, the 3D surface-first transfer benchmark

The goal is to compare a clean shared training stack across the three datasets,
with reference comparisons against a vanilla grouped-domain Transolver and an
AB-UPT-style anchor model where appropriate.

## Codebase

- `train.py` — shared ICML sprint trainer and model selector. **Primary editable entrypoint.**
- `core/` — shared dataset contract, features, optimizers, and model definitions.
- `tandemfoil/` — TandemFoilSet-specific data pipeline and benchmark docs.
- `airfrans/` — AirfRANS-specific data pipeline and benchmark docs.
- `drivaerml/` — DrivAerML-specific data pipeline and benchmark docs.
- `data/` — shared split helpers and smoke tests for the multi-dataset target.

## Metrics

The trainer supports multiple datasets, so merge decisions should be based on the
most decision-relevant metric for the selected benchmark:

- `tandemfoil`
  - prioritize structured validation split performance, especially tandem transfer
  - pressure and surface fidelity matter most
- `airfrans`
  - prioritize surface and volume error on the official task split
  - compare against literature-reported Transolver and newer baselines
- `drivaerml`
  - prioritize surface pressure error on the packaged public split
  - treat volume as optional unless a PR explicitly targets it

Lower is better. Prefer improvements that are robust, physically meaningful, and
simple enough to keep maintaining under deadline pressure.

## Dataset subprograms

Read the dataset-specific subprogram before making dataset-specific claims:

- `tandemfoil/program.md`
- `airfrans/program.md`
- `drivaerml/program.md`

## Constraints

- GPUs have 96 GB VRAM.
- Do not override the global timeout or max-epoch cap.
- Keep the implementation readable. We are under paper deadline pressure; clever
  abstractions are not the goal.
