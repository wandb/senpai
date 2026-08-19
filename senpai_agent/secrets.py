"""Credential names and values shared by Senpai process boundaries."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import SecretStr

if TYPE_CHECKING:
    from senpai_agent.openhands.config import RunnerConfig

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
        "TARGET_REPO_URL",
        "TARGET_WORKDIR",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "WANDB_PROJECT",
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


def resolve_api_key(env: Mapping[str, str], key_env: str) -> SecretStr:
    value = env.get(key_env)
    if not value:
        raise RuntimeError(f"{key_env} is required for the OpenHands runtime")
    return SecretStr(value)


def conversation_secrets(
    env: Mapping[str, str],
    *,
    model_api_key_env_names: Sequence[str],
) -> dict[str, str]:
    custom_secret_env_names = configured_custom_secret_env_names(env)
    model_credentials = set(model_api_key_env_names)
    overlap = tuple(
        name for name in custom_secret_env_names if name in model_credentials
    )
    if overlap:
        raise RuntimeError(
            "model credential environment variables cannot also be custom "
            f"secrets: {', '.join(overlap)}"
        )

    custom_secrets = {}
    for name in custom_secret_env_names:
        value = env.get(name)
        if value is None or not value.strip():
            raise RuntimeError(f"configured custom secret {name} is required")
        custom_secrets[name] = value

    return {
        name: value
        for name in BUILTIN_CONVERSATION_SECRET_ENV_NAMES
        if (value := env.get(name))
    } | custom_secrets


def github_token(
    env: Mapping[str, str],
    *,
    required: bool = True,
) -> SecretStr | None:
    token_fd = env.get(GITHUB_TOKEN_FD_ENV)
    if token_fd:
        try:
            descriptor = int(token_fd)
        except ValueError as error:
            raise RuntimeError(f"{GITHUB_TOKEN_FD_ENV} must be an integer") from error
        with os.fdopen(descriptor, encoding="utf-8") as token_stream:
            value = token_stream.read().strip()
        if not value:
            raise RuntimeError(f"{GITHUB_TOKEN_FD_ENV} is empty")
        return SecretStr(value)

    token_file = env.get(GITHUB_TOKEN_FILE_ENV)
    if token_file:
        path = Path(token_file)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise RuntimeError(
                f"{GITHUB_TOKEN_FILE_ENV} must be a private regular file"
            )
        try:
            value = path.read_text(encoding="utf-8")
        finally:
            path.unlink(missing_ok=True)
        value = value.strip()
        if not value:
            raise RuntimeError(f"{GITHUB_TOKEN_FILE_ENV} is empty")
        return SecretStr(value)

    value = next(
        (
            candidate
            for name in GITHUB_TOKEN_ENV_NAMES
            if (candidate := env.get(name, "").strip())
        ),
        None,
    )
    if value is None:
        if required:
            raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required")
        return None
    return SecretStr(value)


def github_repo(env: Mapping[str, str]) -> str:
    value = env.get("GH_REPO", "")
    if len(value.split("/")) != 2 or not all(value.split("/")):
        raise RuntimeError("GH_REPO must use owner/name form")
    return value


def scrub_model_credentials(
    environment: MutableMapping[str, str],
    config: RunnerConfig,
) -> None:
    for key_env in {
        config.api_key_env,
        config.smart_api_key_env,
        config.fast_api_key_env,
        config.frontier_api_key_env,
    }:
        environment.pop(key_env, None)


def scrub_github_credentials(environment: MutableMapping[str, str]) -> None:
    """Remove every GitHub credential handoff from a child environment."""

    for name in GITHUB_CREDENTIAL_ENV_NAMES:
        environment.pop(name, None)
