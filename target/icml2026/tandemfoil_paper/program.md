<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet Paper Variant

This subtarget adds a fourth benchmark inside `target/icml2026`: a
paper-faithful high-Re TandemFoilSet variant built around the unambiguous split
rules from Experiment 4 of the TandemFoilSet paper.

## Why This Exists

The main `tandemfoil/` benchmark in this target is intentionally pinned to the
later `kagent` / noam-style parity contract. That is useful for the ICML sprint,
but it does **not** give us a clean literature-facing scalar from the original
TandemFoilSet paper.

This variant exists to answer a simpler question:

- on the paper’s own high-Re `Cruise Random` / `Race Car` tasks, is our model
  actually competitive or not?

## Scope

Only the paper splits that are reasonably unambiguous are included here:

- `cruise_random_uniform`
- `cruise_random_aoa_extrap`
- `cruise_random_re_extrap`
- `cruise_random_stagger_extrap`
- `cruise_random_gap_extrap`
- `racecar_uniform`

We intentionally do **not** treat the low-Re curriculum experiments in Tables
3–5 as this benchmark’s primary contract, because those results depend on the
paper’s curriculum procedure and multi-network decomposition in a way that is
less cleanly comparable to our shared ICML trainer.

## Published Contract

Primary source:

- TandemFoilSet paper: [OpenReview PDF](https://openreview.net/pdf?id=4Z0P4Nbosn)

Paper statements we are reproducing here:

- “the datasets are uniformly sampled in a `8:1:1` ratio for training,
  validation, and test sets, respectively, unless otherwise stated”
- for Cruise Random extrapolation:
  - “the highest and lowest `5%` of the `AOA`, `Re`, `Stagger`, or `Gap` value
    range is used as the test set”
  - “the training and validation sets are uniformly sampled from the middle
    `90%` range”

The paper does **not** publish an exact RNG seed or tie-break rule for samples
on the extrapolation boundary. For reproducibility, this repo uses:

- deterministic RNG seed `42` for uniform train/val sampling
- stable rank ordering by `(value, case_index)` when selecting the top/bottom
  `5%` tails

This preserves the paper’s split semantics even though the exact hidden sample
IDs cannot be recovered from the PDF alone.

## Metrics

The paper reports **normalized full-field MSE** after z-score normalization with
training-set statistics.

This subtarget therefore uses:

- `field_mse`
  - mean squared error over **all valid nodes and all three channels**
    `[Ux, Uy, p]`
  - computed in normalized target space using task-local training-set mean/std
- `surface_mse`
  - normalized-space MSE on surface nodes only
- `volume_mse`
  - normalized-space MSE on non-surface nodes only

Lower is better.

Primary harness aliases:

- `val_primary/field_mse`
- `test_primary/field_mse`

## Paper Reference Numbers

Experiment 4, Table 6:

| Task | MGN baseline | MGN + PRE-RES-FREE + RES-COMB |
|---|---:|---:|
| `cruise_random_uniform` | `1.79 ± 1.38` | `0.10 ± 0.13` |
| `cruise_random_aoa_extrap` | `2.03 ± 1.96` | `0.18 ± 0.24` |
| `cruise_random_re_extrap` | `4.85 ± 1.82` | `0.36 ± 0.53` |
| `cruise_random_stagger_extrap` | `1.74 ± 1.66` | `0.13 ± 0.17` |
| `cruise_random_gap_extrap` | `1.95 ± 1.68` | `0.14 ± 0.20` |
| `racecar_uniform` | `0.61 ± 0.51` | `0.21 ± 0.29` |

Those are the numbers to compare against when deciding whether our shared ICML
stack is competitive on the paper-style TandemFoil evaluation.

## Codebase

- `../train.py` — shared trainer
- `data/split_paper_experiment4.py` — deterministic split and stats generator
- `data/prepare.py` — shared tandem pickle loading re-export
- `data/prepare_multi.py` — shared 24-feature TandemFoil preprocessing re-export
- `data/README.md` — task definitions and artifact materialization details

## Notes

- This variant uses the same TandemFoil raw pickles and 24-feature preprocessing
  stack as `tandemfoil/`, but **not** the noam-style pressure-denorm contract.
- It is a benchmark-calibration target, not a replacement for the main
  `tandemfoil/` ICML sprint target.

