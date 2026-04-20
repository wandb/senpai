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

- surface error
- volume error
- comparison against official-task Transolver and newer literature baselines

## Code boundaries

- benchmark-local data pipeline: `data/`
- shared training entrypoint: `../train.py`
- shared models and collate contracts: `../core/`
