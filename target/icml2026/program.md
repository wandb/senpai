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

For TandemFoilSet specifically, this target should become the reproduction and
extension path for the merged noam lineage through `#2379`, while AirfRANS and
DrivAerML remain on the simpler shared path unless their own parity work is
explicitly requested.

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
  - prioritize two metric families side by side:
  - `legacy_noam/*` for the denormalized historical `p_*` contract
  - `icml2026_v2/*` for the packaged `kagent` split contract
  - pressure and surface fidelity remain the decision-driving quantities
- `airfrans`
  - prioritize surface and volume error on the official task split
  - compare against literature-reported Transolver and newer baselines
- `drivaerml`
  - prioritize surface pressure error on the packaged public split
  - treat volume as optional unless a PR explicitly targets it

Lower is better. Prefer improvements that are robust, physically meaningful, and
simple enough to keep maintaining under deadline pressure.

## Metric Alignment Plan

The code and dataset docs in this target should stay pinned to the
literature-facing contracts we intend to cite in the ICML paper sprint.

Alignment policy:

- use the benchmark split contract actually implemented by the source paper or
  official dataset code when one exists
- use the benchmark metric calculation exactly, including whether evaluation is
  done on normalized or unnormalized targets and whether aggregation is
  per-case or global over the split
- keep hyperparameter-tuning metrics on validation splits, but reserve
  literature-facing comparison numbers for the matching test split
- document any remaining irreducible discrepancy, such as the packaged
  DrivAerML case set being smaller than the nominal public split in AB-UPT
- pin TandemFoilSet parity work to concrete source refs rather than a floating
  branch head:
  - transform, metric, residual, and merged feature-stack contract:
    `origin/noam@d743ba27eb1c561750f55daeefadcbe41e2b8421`
  - ANP cross-foil decoder implementation source:
    `origin/frieren/anp-surface-decoder@7999a2e`
- keep the TandemFoil historical anchor and the clean-target parity target
  distinct:
  - historical best merged single-seed anchor in the report: `#2319`
  - merged parity lineage this target should reproduce and extend:
    `#2319 -> #2350 -> #2357 -> #2379`
- emit both TandemFoil metric regimes explicitly instead of silently mixing
  them:
  - `legacy_noam/p_in`, `legacy_noam/p_oodc`, `legacy_noam/p_tan`,
    `legacy_noam/p_re`
  - `val_eq4/surface_pressure_mae`, `test_eq4/surface_pressure_mae`, and the
    per-split `surface_pressure_mae` values for the v2 manifest

Primary harness metric aliases:

- `tandemfoil`: `val_primary/surface_pressure_mae` and
  `test_primary/surface_pressure_mae`
- `airfrans`: `val_primary/surface_mse` and `test_primary/surface_mse`
- `drivaerml`: `val_primary/surface_rel_l2_pct` and
  `test_primary/surface_rel_l2_pct`

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
