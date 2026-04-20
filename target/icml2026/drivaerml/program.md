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

- surface pressure error on the packaged public split
- volume metrics only when a PR explicitly targets the small processed volume subset

## Code boundaries

- benchmark-local data pipeline: `data/`
- shared training entrypoint: `../train.py`
- shared models and collate contracts: `../core/`
