"""Credential names shared by Senpai process boundaries."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence

GITHUB_TOKEN_ENV_NAMES = ("GITHUB_TOKEN", "GH_TOKEN")
GITHUB_TOKEN_FILE_ENV = "SENPAI_GITHUB_TOKEN_FILE"
GITHUB_TOKEN_FD_ENV = "SENPAI_GITHUB_TOKEN_FD"
GITHUB_CREDENTIAL_ENV_NAMES = (
    *GITHUB_TOKEN_ENV_NAMES,
    GITHUB_TOKEN_FILE_ENV,
    GITHUB_TOKEN_FD_ENV,
)
CUSTOM_SECRET_ENV_NAMES_ENV = "SENPAI_CUSTOM_SECRET_ENV_NAMES"
BUILTIN_CONVERSATION_SECRET_ENV_NAMES = ("WANDB_API_KEY", "EXA_API_KEY")
_RESERVED_CUSTOM_SECRET_ENV_NAMES = frozenset(
    {
        # Built-in credentials use separate trust boundaries.
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        *BUILTIN_CONVERSATION_SECRET_ENV_NAMES,
        # The launcher and entrypoints own these names in both roles.
        "ADVISOR_BRANCH",
        "EXTRA_INSTRUCTIONS_B64",
        "GPUS_PER_STUDENT",
        "IS_SANDBOX",
        "PROBLEM_DIR",
        "PVC_MOUNT_PATH",
        "RESEARCH_TAG",
        "STUDENT_NAME",
        "STUDENT_NAMES",
        "TARGET_REPO_BRANCH",
        "TARGET_REPO_REVISION",
        "TARGET_REPO_URL",
        "TARGET_WORKDIR",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        # Custom credentials must not change process startup or executable routing.
        "BASH_ENV",
        "ENV",
        "GIT_ASKPASS",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_EXEC_PATH",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
        "HOME",
        "IFS",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "SHELL",
        "SSH_ASKPASS",
    }
)
_RESERVED_CUSTOM_SECRET_ENV_PREFIXES = ("GH_", "GITHUB_", "SENPAI_")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def validate_custom_secret_env_names(names: Sequence[str]) -> None:
    """Validate custom secret environment-variable names."""

    if any(not name for name in names):
        raise ValueError(
            "custom secret environment variable names must not contain empty names"
        )

    invalid = tuple(name for name in names if _ENV_NAME.fullmatch(name) is None)
    if invalid:
        raise ValueError(
            "invalid custom secret environment variable names: "
            + ", ".join(invalid)
        )

    duplicates = tuple(
        name for name in dict.fromkeys(names) if names.count(name) > 1
    )
    if duplicates:
        raise ValueError(
            "duplicate custom secret environment variable names: "
            + ", ".join(duplicates)
        )

    reserved = tuple(
        name
        for name in names
        if name in _RESERVED_CUSTOM_SECRET_ENV_NAMES
        or name.startswith(_RESERVED_CUSTOM_SECRET_ENV_PREFIXES)
    )
    if reserved:
        raise ValueError(
            "reserved custom secret environment variable names: "
            + ", ".join(reserved)
        )


def configured_custom_secret_env_names(
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    """Parse and validate the configured custom secret names."""

    raw = environment.get(CUSTOM_SECRET_ENV_NAMES_ENV, "")
    if not raw.strip():
        return ()

    names = tuple(part.strip() for part in raw.split(","))
    try:
        validate_custom_secret_env_names(names)
    except ValueError as error:
        raise RuntimeError(f"{CUSTOM_SECRET_ENV_NAMES_ENV}: {error}") from error
    return names


def scrub_github_credentials(environment: MutableMapping[str, str]) -> None:
    """Remove every GitHub credential handoff from a child environment."""

    for name in GITHUB_CREDENTIAL_ENV_NAMES:
        environment.pop(name, None)
