"""Initialize W&B Weave tracing before OpenHands enters the process."""

from __future__ import annotations

import os
from collections.abc import Mapping
from threading import Lock
from uuid import UUID

from weave_openhands import finish as weave_finish
from weave_openhands import init as weave_init

from senpai_agent.secrets import (
    GITHUB_TOKEN_ENV_NAMES,
    is_secret_environment_variable,
)

_initialized = False
_project_name: str | None = None
_content_redactor: SecretRedactor | None = None
TRACE_SECRET_ENV_NAMES = (
    *GITHUB_TOKEN_ENV_NAMES,
    "WANDB_API_KEY",
    "EXA_API_KEY",
    "ANTHROPIC_API_KEY",
)


def _is_secret_env(name: str) -> bool:
    return name in TRACE_SECRET_ENV_NAMES or is_secret_environment_variable(name)


def weave_project_name(env: Mapping[str, str]) -> str | None:
    entity = env.get("WANDB_ENTITY")
    project = env.get("WANDB_PROJECT")
    if not entity and not project:
        return None
    if not entity or not project:
        raise RuntimeError("WANDB_ENTITY and WANDB_PROJECT must be set together")
    return f"{entity}/{project}"


def weave_agent_name(env: Mapping[str, str]) -> str:
    role = env.get("SENPAI_ROLE", "senpai")
    student_name = env.get("STUDENT_NAME")
    if role == "student" and student_name:
        return f"student-{student_name}"
    advisor_name = env.get("ADVISOR_NAME")
    if role == "advisor" and advisor_name and advisor_name != "advisor":
        return f"advisor-{advisor_name}"
    return role


def weave_conversation_url(
    project_name: str | None,
    conversation_id: str | UUID,
) -> str | None:
    if project_name is None:
        return None
    entity, project = project_name.split("/", 1)
    return (
        f"https://wandb.ai/{entity}/{project}/weave/agents/conversations/"
        f"{conversation_id}"
    )


class SecretRedactor:
    def __init__(self, values: set[str]):
        self._lock = Lock()
        self._values = self._sorted(values)

    @staticmethod
    def _sorted(values: set[str]) -> tuple[str, ...]:
        return tuple(sorted(filter(None, values), key=len, reverse=True))

    def register(self, value: str) -> None:
        if not value:
            return
        with self._lock:
            self._values = self._sorted({*self._values, value})

    def __call__(self, content: str) -> str:
        with self._lock:
            values = self._values
        for value in values:
            content = content.replace(value, "<secret-hidden>")
        return content


def secret_redactor(env: Mapping[str, str]) -> SecretRedactor:
    configured_model_key = env.get("SENPAI_OPENHANDS_API_KEY_ENV")
    return SecretRedactor(
        {
            value
            for name, value in env.items()
            if value
            and (
                _is_secret_env(name)
                or (configured_model_key is not None and name == configured_model_key)
            )
        }
    )


def register_trace_secret(value: str) -> None:
    if _content_redactor is not None:
        _content_redactor.register(value)


def initialize_weave_monitoring(
    env: Mapping[str, str] = os.environ,
) -> str | None:
    global _content_redactor, _initialized, _project_name
    if _initialized:
        return _project_name

    project_name = weave_project_name(env)
    if project_name is None:
        return None

    redactor = secret_redactor(env)
    weave_init(
        project_name,
        agent_name=weave_agent_name(env),
        capture_content=True,
        content_transform=redactor,
    )
    _content_redactor = redactor
    _initialized = True
    _project_name = project_name
    return project_name


def finish_weave_monitoring() -> None:
    global _content_redactor, _initialized, _project_name
    if not _initialized:
        return
    try:
        weave_finish()
    finally:
        _content_redactor = None
        _initialized = False
        _project_name = None
