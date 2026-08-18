"""OpenHands runtime bootstrap shared by the split runner modules."""

# ruff: noqa: E402
# OpenHands imports in child modules intentionally follow this initialization.

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from senpai_agent.weave_monitoring import initialize_weave_monitoring

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WEAVE_PROJECT = initialize_weave_monitoring()
