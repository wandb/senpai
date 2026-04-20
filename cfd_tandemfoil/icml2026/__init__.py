# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""ICML 2026 CFD experiment package.

This package is the clean paper-sprint path extracted from the broader
`cfd_tandemfoil` research tree. It keeps only the experiment contracts needed
for the ICML workshop paper:

- one grouped case contract shared across TandemFoilSet, AirfRANS, and DrivAerML
- one unified trainer entrypoint
- one local reference Transolver implementation
- one local AB-UPT-compatible implementation with bridge-aligned sampling
- a minimal set of Senpai-discovered mechanisms worth rerunning
"""

