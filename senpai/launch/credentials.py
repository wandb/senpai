# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""The single .env credential contract used by every Senpai consumer."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from dotenv import dotenv_values

WORKLOAD_REQUIRED_SECRET_NAMES = (
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "EXA_API_KEY",
    "WANDB_API_KEY",
)
SCOUT_REQUIRED_SECRET_NAMES = (
    "GITHUB_TOKEN",
    "WANDB_API_KEY",
    "KUBECONFIG_B64",
)
SCOUT_OPTIONAL_SECRET_NAMES = ("SEMANTIC_SCHOLAR_API_KEY",)
SCOUT_SECRET_NAMES = SCOUT_REQUIRED_SECRET_NAMES + SCOUT_OPTIONAL_SECRET_NAMES
SCOUT_ONLY_SECRET_NAMES = frozenset(SCOUT_SECRET_NAMES).difference(
    WORKLOAD_REQUIRED_SECRET_NAMES
)
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(
            f"ERROR: no credential file at {path}. Create it with `cp example.env .env`."
        )

    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        path.chmod(0o600)
        print(f"Secured {path} permissions (0600).")

    values = {
        key: value
        for key, value in dotenv_values(path, interpolate=False).items()
        if value not in (None, "")
    }
    invalid = sorted(key for key in values if not ENV_NAME.fullmatch(key))
    if invalid:
        raise SystemExit(
            f"ERROR: {path} contains invalid environment variable names: {', '.join(invalid)}"
        )
    return values


def _require(values: dict[str, str], required: tuple[str, ...], path: Path) -> None:
    missing = [name for name in required if not values.get(name, "").strip()]
    if missing:
        raise SystemExit(
            f"ERROR: {path} is missing required secrets: {', '.join(missing)}"
        )


def _workload_secrets(values: dict[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in values.items()
        if name not in SCOUT_ONLY_SECRET_NAMES
    }


def load_workload_secrets(path: Path) -> dict[str, str]:
    """Load credentials shared by advisor and student workers."""
    values = _read_dotenv(path)
    _require(values, WORKLOAD_REQUIRED_SECRET_NAMES, path)
    return _workload_secrets(values)


def load_scout_secrets(path: Path) -> dict[str, str]:
    """Load only credentials declared for the CoreWeave scout workflow."""
    values = _read_dotenv(path)
    _require(values, SCOUT_REQUIRED_SECRET_NAMES, path)
    return {name: values[name] for name in SCOUT_SECRET_NAMES if values.get(name)}


def workload_secret_names(path: Path) -> tuple[str, ...]:
    """Return workload names for redacted dry runs without requiring .env."""
    if not path.exists():
        return WORKLOAD_REQUIRED_SECRET_NAMES
    names = set(WORKLOAD_REQUIRED_SECRET_NAMES)
    names.update(_workload_secrets(_read_dotenv(path)))
    return tuple(sorted(names))
