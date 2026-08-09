"""Credential names shared by Senpai process boundaries."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping, MutableMapping
from pathlib import Path

GITHUB_TOKEN_ENV_NAMES = ("GITHUB_TOKEN", "GH_TOKEN")
GITHUB_TOKEN_FILE_ENV = "SENPAI_GITHUB_TOKEN_FILE"
GITHUB_TOKEN_FD_ENV = "SENPAI_GITHUB_TOKEN_FD"
GITHUB_CREDENTIAL_ENV_NAMES = (
    *GITHUB_TOKEN_ENV_NAMES,
    GITHUB_TOKEN_FILE_ENV,
    GITHUB_TOKEN_FD_ENV,
)
SUPERVISOR_SECRET_DIR_ENV = "SENPAI_SUPERVISOR_SECRET_DIR"
SUPERVISOR_SECRET_ENV_NAMES = frozenset(
    {
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
)
_SECRET_ENV_NAMES = frozenset(
    {
        *GITHUB_CREDENTIAL_ENV_NAMES,
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SSH_AUTH_SOCK",
        "API_KEY",
        "CREDENTIAL",
        "CREDENTIALS",
        "PASSWORD",
        "PRIVATE_KEY",
        "SECRET",
        "TOKEN",
    }
)
_SECRET_ENV_SUFFIXES = (
    "_ACCESS_KEY",
    "_API_KEY",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)


def is_secret_environment_variable(name: str) -> bool:
    """Return whether an environment variable conventionally carries auth."""

    normalized = name.upper()
    return normalized in _SECRET_ENV_NAMES or normalized.endswith(_SECRET_ENV_SUFFIXES)


def scrub_github_credentials(environment: MutableMapping[str, str]) -> None:
    """Remove every GitHub credential handoff from a child environment."""

    for name in GITHUB_CREDENTIAL_ENV_NAMES:
        environment.pop(name, None)


def consume_supervisor_secret_directory(
    environment: Mapping[str, str],
    *,
    required: bool = False,
) -> dict[str, str]:
    """Consume one private pre-exec credential handoff into process memory."""

    hydrated = dict(environment)
    directory_value = hydrated.pop(SUPERVISOR_SECRET_DIR_ENV, None)
    if not directory_value:
        if required:
            raise RuntimeError(f"{SUPERVISOR_SECRET_DIR_ENV} is required")
        return hydrated

    directory = Path(directory_value)
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o077
        or metadata.st_uid != os.geteuid()
    ):
        raise RuntimeError(
            f"{SUPERVISOR_SECRET_DIR_ENV} must be a private owned directory"
        )

    entries = tuple(directory.iterdir())
    unknown = sorted(
        path.name
        for path in entries
        if path.name not in SUPERVISOR_SECRET_ENV_NAMES
    )
    if unknown:
        raise RuntimeError(
            f"{SUPERVISOR_SECRET_DIR_ENV} contains unsupported entries: "
            + ", ".join(unknown)
        )

    for path in entries:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.geteuid()
        ):
            raise RuntimeError(
                f"supervisor secret {path.name} must be a private owned file"
            )
        try:
            value = path.read_text(encoding="utf-8").strip()
        finally:
            path.unlink(missing_ok=True)
        if not value:
            raise RuntimeError(f"supervisor secret {path.name} is empty")
        hydrated[path.name] = value

    directory.rmdir()
    return hydrated
