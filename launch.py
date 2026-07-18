#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""User-facing entrypoint for every Senpai launch backend."""

# ruff: noqa: E402 -- this wrapper intentionally imports k8s/launch.py by path.

import sys
from pathlib import Path

K8S_DIR = Path(__file__).resolve().parent / "k8s"
sys.path.insert(0, str(K8S_DIR))

from launch import main


if __name__ == "__main__":
    main()
