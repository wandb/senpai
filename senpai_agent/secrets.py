"""Credential names shared by Senpai process boundaries."""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
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
SERVICE_CREDENTIAL_ENV_NAMES = ("WANDB_API_KEY", "EXA_API_KEY")
WANDB_TRAINING_API_KEY_ENV = "SENPAI_WANDB_TRAINING_API_KEY"
MODEL_CREDENTIALS_FD_ENV = "SENPAI_MODEL_CREDENTIALS_FD"
MAX_MODEL_CREDENTIAL_BUNDLE_BYTES = 64 * 1024
SHELL_STARTUP_ENV_NAMES = frozenset(
    {
        "BASHOPTS",
        "BASH_ENV",
        "BASH_XTRACEFD",
        "ENV",
        "MAILCHECK",
        "MAILPATH",
        "PROMPT_COMMAND",
        "PS0",
        "PS1",
        "PS2",
        "PS3",
        "PS4",
        "SHELLOPTS",
        "ZDOTDIR",
    }
)
PRIVATE_CREDENTIAL_FILE_ENVS = {
    "WANDB_API_KEY": "SENPAI_WANDB_API_KEY_FILE",
    "EXA_API_KEY": "SENPAI_EXA_API_KEY_FILE",
    WANDB_TRAINING_API_KEY_ENV: "SENPAI_WANDB_TRAINING_API_KEY_FILE",
}
PRIVATE_CREDENTIAL_FD_ENVS = {
    "WANDB_API_KEY": "SENPAI_WANDB_API_KEY_FD",
    "EXA_API_KEY": "SENPAI_EXA_API_KEY_FD",
    WANDB_TRAINING_API_KEY_ENV: "SENPAI_WANDB_TRAINING_API_KEY_FD",
}
_RESERVED_CUSTOM_SECRET_ENV_NAMES = frozenset(
    {
        # Built-in credentials use separate trust boundaries.
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "WANDB_INFERENCE_API_KEY",
        *SERVICE_CREDENTIAL_ENV_NAMES,
        WANDB_TRAINING_API_KEY_ENV,
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
        *SHELL_STARTUP_ENV_NAMES,
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
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
        "SHELL",
        "SSH_ASKPASS",
    }
)
_RESERVED_CUSTOM_SECRET_ENV_PREFIXES = (
    "GH_",
    "GITHUB_",
    "SENPAI_",
    "WANDB_API_KEY_",
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PR_SET_DUMPABLE = 4


def set_process_nondumpable() -> None:
    """Block same-UID process inspection before this process reads credentials."""

    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def consume_model_credential_fd(
    environment: MutableMapping[str, str],
) -> dict[str, str]:
    """Read delegated model credentials after the child is nondumpable."""

    descriptor_value = environment.pop(MODEL_CREDENTIALS_FD_ENV, None)
    if descriptor_value is None:
        return {}
    try:
        descriptor = int(descriptor_value)
        if descriptor < 0:
            raise ValueError
    except ValueError as error:
        raise RuntimeError(
            f"{MODEL_CREDENTIALS_FD_ENV} must be a nonnegative integer"
        ) from error
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(MAX_MODEL_CREDENTIAL_BUNDLE_BYTES + 1)
    if len(raw) > MAX_MODEL_CREDENTIAL_BUNDLE_BYTES:
        raise RuntimeError("delegated model credential bundle is too large")
    try:
        credentials = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("delegated model credential bundle is invalid") from error
    if (
        not isinstance(credentials, dict)
        or not credentials
        or any(
            not isinstance(name, str)
            or _ENV_NAME.fullmatch(name) is None
            or not isinstance(value, str)
            or not value.strip()
            for name, value in credentials.items()
        )
    ):
        raise RuntimeError("delegated model credential bundle is invalid")
    return credentials


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


def scrub_service_credentials(environment: MutableMapping[str, str]) -> None:
    """Remove service credentials from an agent-controlled child environment."""

    for name in (*SERVICE_CREDENTIAL_ENV_NAMES, WANDB_TRAINING_API_KEY_ENV):
        environment.pop(name, None)
    handoff_names = (
        *PRIVATE_CREDENTIAL_FILE_ENVS.values(),
        *PRIVATE_CREDENTIAL_FD_ENVS.values(),
        MODEL_CREDENTIALS_FD_ENV,
    )
    for name in handoff_names:
        environment.pop(name, None)
