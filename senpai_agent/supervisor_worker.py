"""Run an independent Supervisor in the requesting role's local workspace."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from senpai_agent.delegation import (
    DelegationConfig,
    DelegationManager,
    DelegationRequest,
    OpenHandsChildProcess,
    record_delegated_task_result,
)
from senpai_agent.PROMPTS import SUPERVISOR_ROLE_PROMPT
from senpai_agent.secrets import (
    MODEL_CREDENTIALS_FD_ENV,
    consume_model_credential_fd,
    scrub_github_credentials,
    set_process_nondumpable,
)
from senpai_agent.weave_monitoring import finish_weave_monitoring, register_trace_secret

if TYPE_CHECKING:
    from senpai_agent.openhands_runner import RunnerConfig


class SupervisorResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    resolved: bool = Field(
        description="True only when the requested issue is resolved and verified.",
    )
    repair_summary: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(
        description=(
            "Report the diagnosis, changes and verification. Give actionable feedback "
            "for any remaining issue, including evidence and the next required action."
        ),
    )


class SupervisorProcess(OpenHandsChildProcess):
    def __init__(self, config: DelegationConfig, request: DelegationRequest):
        if (
            request.agent != "supervisor"
            or request.model != "smart"
            or request.parent_context
        ):
            raise ValueError(
                "Supervisor requires its own agent, smart model and clean context"
            )
        super().__init__(config, request)

    @property
    def command(self) -> tuple[str, ...]:
        command = list(super().command)
        command[command.index("-m") + 1] = "senpai_agent.supervisor_worker"
        return tuple(command)

    @property
    def environment(self) -> dict[str, str]:
        environment = super().environment
        environment.pop("SENPAI_PARENT_CONVERSATION_HISTORY_DIR", None)
        return environment


def make_supervisor_manager(
    config: RunnerConfig,
    *,
    event_db_path: Path | None = None,
) -> DelegationManager:
    from senpai_agent.openhands_runner import delegation_config

    child_config = delegation_config(config)
    return DelegationManager(
        child_config,
        lambda request: SupervisorProcess(child_config, request),
        event_db_path=event_db_path,
    )


def supervisor_config(config: RunnerConfig) -> RunnerConfig:
    return replace(
        config,
        agent_name="supervisor",
        instructions=replace(config.instructions, role=SUPERVISOR_ROLE_PROMPT),
        model=config.smart_model,
        api_key_env=config.smart_api_key_env,
        api_key=config.smart_api_key,
        reasoning_effort=config.smart_reasoning_effort,
        github_token=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    from senpai_agent.openhands_runner import (
        parse_runner_args,
        resolve_config,
        run_openhands,
        scrub_model_credentials,
    )

    try:
        try:
            set_process_nondumpable()
            if MODEL_CREDENTIALS_FD_ENV not in os.environ:
                raise RuntimeError(
                    "Supervisor requires the private model credential handoff"
                )
            credentials = consume_model_credential_fd(os.environ)
            for credential in credentials.values():
                register_trace_secret(credential)
            scrub_github_credentials(os.environ)
            os.environ.pop("SENPAI_PARENT_CONVERSATION_HISTORY_DIR", None)
            args = parse_runner_args(argv)
            if not args.child or args.agent != "supervisor":
                raise RuntimeError("Supervisor requires a supervisor child agent")
            prompt = sys.stdin.read()
            if not prompt.strip():
                raise RuntimeError("Supervisor requires a task on stdin")
            runtime_environment = {**os.environ, **credentials}
            config = supervisor_config(resolve_config(args, runtime_environment))
            del runtime_environment, credentials
            scrub_model_credentials(os.environ, config)
            if not config.delegation_task_id:
                raise RuntimeError("Supervisor requires a delegation task")
            results: list[SupervisorResult] = []
            status = run_openhands(
                prompt,
                replace(config, delegation_task_id=None),
                response_schema=SupervisorResult,
                on_structured_result=results.append,
            )
            if status != 0 or not results:
                raise RuntimeError(
                    "Supervisor did not complete with a structured result"
                )
            record_delegated_task_result(
                config.delegation_task_id,
                result=results[0].model_dump_json(),
            )
            return 0
        except BaseException as error:
            if task_id := os.environ.get("SENPAI_DELEGATION_TASK_ID"):
                record_delegated_task_result(
                    task_id,
                    error=f"{type(error).__name__}: {error}",
                )
            raise
    finally:
        finish_weave_monitoring()


if __name__ == "__main__":
    raise SystemExit(main())
