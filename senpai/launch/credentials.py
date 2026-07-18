# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""The single .env credential contract used by every launch backend."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from dotenv import dotenv_values

REQUIRED_SECRET_NAMES = (
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "EXA_API_KEY",
    "WANDB_API_KEY",
)
SCOUT_ONLY_SECRET_NAMES = frozenset(
    {"KUBECONFIG_B64", "SEMANTIC_SCHOLAR_API_KEY", "SLACK_WEBHOOK_URL"}
)
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_dotenv(path: Path) -> dict[str, str]:
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


def secret_names(path: Path) -> tuple[str, ...]:
    """Return names for redacted dry runs without requiring a populated file."""
    if not path.exists():
        return REQUIRED_SECRET_NAMES
    return tuple(sorted(set(REQUIRED_SECRET_NAMES) | _parse_dotenv(path).keys()))


def load_secrets(
    path: Path, required: tuple[str, ...] = REQUIRED_SECRET_NAMES
) -> dict[str, str]:
    """Load credentials for one consumer from the gitignored .env file."""
    if not path.is_file():
        raise SystemExit(
            f"ERROR: no credential file at {path}. Create it with `cp example.env .env`."
        )

    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        path.chmod(0o600)
        print(f"Secured {path} permissions (0600).")

    values = _parse_dotenv(path)
    missing = [name for name in required if not values.get(name, "").strip()]
    if missing:
        raise SystemExit(
            f"ERROR: {path} is missing required secrets: {', '.join(missing)}"
        )
    return values


def workload_secrets(secrets: dict[str, str]) -> dict[str, str]:
    """Route worker credentials without exposing scout infrastructure access."""
    return {
        name: value
        for name, value in secrets.items()
        if name not in SCOUT_ONLY_SECRET_NAMES
    }
