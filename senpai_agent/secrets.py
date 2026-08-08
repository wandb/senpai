"""Credential names shared by Senpai process boundaries."""

from __future__ import annotations

from collections.abc import MutableMapping

GITHUB_TOKEN_ENV_NAMES = ("GITHUB_TOKEN", "GH_TOKEN")
GITHUB_TOKEN_FILE_ENV = "SENPAI_GITHUB_TOKEN_FILE"
GITHUB_TOKEN_FD_ENV = "SENPAI_GITHUB_TOKEN_FD"
GITHUB_CREDENTIAL_ENV_NAMES = (
    *GITHUB_TOKEN_ENV_NAMES,
    GITHUB_TOKEN_FILE_ENV,
    GITHUB_TOKEN_FD_ENV,
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
