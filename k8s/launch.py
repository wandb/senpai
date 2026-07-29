#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Backward-compatible Kubernetes launcher path."""

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
main = import_module("senpai.launch.cli").main

if __name__ == "__main__":
    main()
