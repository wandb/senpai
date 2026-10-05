"""Queue a merge worker through the normal durable subagent lifecycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import SecretStr

from senpai_agent.delegation import (
    AgentTask,
    AgentTaskState,
    DelegationConfig,
    DelegationRequest,
    OpenHandsChildProcess,
    SpawnAgentsAction,
    configured_delegation_config,
    configured_delegation_manager,
)
from senpai_agent.github.tools.contracts import MergeExperimentAction
from senpai_agent.github.tools.runtime import current_github_credentials

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class MergeWorkerProcess(OpenHandsChildProcess):
    def __init__(
        self,
        config: DelegationConfig,
        request: DelegationRequest,
        action: MergeExperimentAction,
        token: SecretStr,
    ):
        super().__init__(config, request)
        self._action = action
        self._github_token = token

    @property
    def command(self) -> tuple[str, ...]:
        command = list(super().command)
        command[command.index("-m") + 1] = "senpai_agent.github.merge_worker"
        return tuple(command)

    @property
    def environment(self) -> dict[str, str]:
        environment = super().environment
        environment["SENPAI_MERGE_REQUEST_JSON"] = self._action.model_dump_json()
        return environment

    def _child_credentials(self) -> dict[str, str]:
        return {
            **super()._child_credentials(),
            "GITHUB_TOKEN": self._github_token.get_secret_value(),
        }


def queue_merge(
    action: MergeExperimentAction,
    conversation: LocalConversation,
    *,
    event_db_path: Path,
) -> AgentTaskState:
    config = configured_delegation_config()
    credentials = current_github_credentials()
    if config.role != "advisor" or credentials is None:
        raise RuntimeError("merge workers require an authenticated advisor runtime")
    if credentials.repo != config.github_repo:
        raise RuntimeError("merge worker repository differs from the advisor runtime")
    manager = configured_delegation_manager(
        child_runner_factory=lambda request: MergeWorkerProcess(
            config, request, action, credentials.token,
        ),
        event_db_path=event_db_path,
    )
    encoded = action.model_dump_json()
    task = f"Review this exact merge request and complete or block it: {encoded}"
    prior = [
        row for row in manager.registry.rows()
        if row["parent_conversation_id"] == str(conversation.id) and row["task"] == task
    ]
    attempt = len(prior)
    if prior:
        latest = prior[-1]
        if latest["status"] in {"queued", "starting", "running"} or _merged(latest["result"]):
            attempt -= 1
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    return manager.spawn(
        SpawnAgentsAction(
            batch_key=f"merge:{digest}:{attempt}",
            tasks=[AgentTask(
                key="merge", task=task, agent="supervisor", model="smart",
            )],
        ),
        conversation,
    )[0]


def _merged(result: str | None) -> bool:
    if result is None:
        return False
    try:
        outcome = json.loads(result)
    except json.JSONDecodeError:
        return False
    return isinstance(outcome, dict) and outcome.get("state") == "experiment_merged"
