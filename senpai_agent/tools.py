"""Typed OpenHands tools for Senpai's reliable control-plane operations."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.browser_use import BrowserToolSet
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import (
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)
from pydantic import Field

from senpai_agent.delegation import (
    AgentStatusTool,
    AwaitAgentsTool,
    CancelAgentsTool,
    SpawnAgentsTool,
)
from senpai_agent.git_workflow import require_clean_job_worktree
from senpai_agent.github.tools import GitHubWorkflowToolSet
from senpai_agent.hooks import supervised_job_policy
from senpai_agent.monitor import (
    JobMonitorSpec,
    JobMonitorStore,
    MetricMonitorSpec,
    WandbJobStatusSource,
)
from senpai_agent.secrets import is_secret_environment_variable
from senpai_agent.jobs import (
    JobResult,
    JobSpec as SupervisedJobSpec,
    JobState,
    JobSupervisor,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState, LocalConversation


_JOB_RUNTIMES: dict[
    Path,
    tuple[JobSupervisor, JobMonitorStore],
] = {}
_ADVISOR_JOB_MONITOR_STORES: dict[Path, JobMonitorStore] = {}
_BROWSER_ENABLED_STATE_KEY = "senpai.browser_enabled"
_JOB_CONTROL_RESOURCE = "senpai-job-control"


class LoadBrowserAction(Action):
    """Enable the browser tool family for this conversation."""


class LoadBrowserObservation(Observation):
    tools: tuple[str, ...]

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=("Browser tools are now available: " + ", ".join(self.tools) + ".")
            )
        ]


class _LoadBrowserExecutor(ToolExecutor[LoadBrowserAction, LoadBrowserObservation]):
    def __call__(
        self,
        action: LoadBrowserAction,
        conversation: LocalConversation | None = None,
    ) -> LoadBrowserObservation:
        if conversation is None:
            raise ValueError("load_browser requires its parent conversation")
        if conversation.state.agent_state.get(_BROWSER_ENABLED_STATE_KEY):
            names = tuple(
                name
                for name in conversation.agent.tools_map
                if name.startswith("browser_")
            )
            return LoadBrowserObservation(tools=names)

        browser_tools = BrowserToolSet.create(conversation.state)
        conversation.agent.add_runtime_tools(browser_tools)
        conversation.state.agent_state = {
            **conversation.state.agent_state,
            _BROWSER_ENABLED_STATE_KEY: True,
        }
        return LoadBrowserObservation(tools=tuple(tool.name for tool in browser_tools))


class LoadBrowserTool(ToolDefinition[LoadBrowserAction, LoadBrowserObservation]):
    name = "load_browser"

    def declared_resources(self, action: Action) -> DeclaredResources:
        return DeclaredResources(keys=("browser-tools",), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[ToolDefinition]:
        if conv_state.agent_state.get(_BROWSER_ENABLED_STATE_KEY):
            return BrowserToolSet.create(conv_state)
        return [
            cls(
                description=(
                    "Load the full browser tool family for this conversation. "
                    "Use it when interactive web navigation or page inspection is "
                    "needed; the browser operations become available on the next step."
                ),
                action_type=LoadBrowserAction,
                observation_type=LoadBrowserObservation,
                annotations=ToolAnnotations(
                    title="Load browser tools",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_LoadBrowserExecutor(),
            )
        ]


_TASK_TRACKER_DESCRIPTION = """Maintain an optional persisted task list as working memory across turns.

Use it when persistent coordination materially helps: multi-step work,
concurrent workstreams, delegated agents, or long-running jobs. For straightforward
work, proceed directly. Track independent work in parallel, with multiple items
in progress when they are genuinely active. View the current list before
replacing it, keep entries concise, and mark work done only when its required
evidence is complete."""


class SenpaiTaskTrackerTool(TaskTrackerTool):
    """OpenHands task persistence with a concise, concurrency-safe description."""

    name = "task_tracker"

    @classmethod
    def create(cls, conv_state: ConversationState) -> Sequence[ToolDefinition]:
        return [
            tool.model_copy(update={"description": _TASK_TRACKER_DESCRIPTION})
            for tool in super().create(conv_state)
        ]


def job_runtime(
    workspace: Path,
    state_dir: Path,
    *,
    max_timeout_seconds: int | None = None,
) -> tuple[JobSupervisor, JobMonitorStore]:
    key = state_dir.resolve()
    runtime = _JOB_RUNTIMES.get(key)
    if runtime is None:
        runtime = (
            JobSupervisor(
                workspace=workspace,
                state_dir=key,
                max_timeout_seconds=max_timeout_seconds,
            ),
            JobMonitorStore(key / "monitors.sqlite3"),
        )
        _JOB_RUNTIMES[key] = runtime
    return runtime


def close_job_runtimes() -> None:
    for supervisor, monitors in _JOB_RUNTIMES.values():
        supervisor.close()
        monitors.close()
    _JOB_RUNTIMES.clear()
    for monitors in _ADVISOR_JOB_MONITOR_STORES.values():
        monitors.close()
    _ADVISOR_JOB_MONITOR_STORES.clear()


def advisor_job_monitor_store(state_dir: Path) -> JobMonitorStore:
    key = state_dir.resolve()
    if key not in _ADVISOR_JOB_MONITOR_STORES:
        _ADVISOR_JOB_MONITOR_STORES[key] = JobMonitorStore(key / "monitors.sqlite3")
    return _ADVISOR_JOB_MONITOR_STORES[key]


class JobSpec(SupervisedJobSpec):
    """One argv-based process supervised as a durable Senpai job."""

    argv: tuple[str, ...] = Field(
        min_length=1,
        description="Executable and arguments. Shell command strings are not allowed.",
    )
    cwd: Path = Field(description="Working directory inside the role workspace.")
    timeout_seconds: int = Field(
        gt=0,
        description="Hard wall-clock deadline, including process-group shutdown.",
    )
    workspace_access: Literal["read_only", "mutable"] = Field(
        default="mutable",
        description=(
            "Use mutable for builds, training, evaluation, or any process that may "
            "write in the checkout. Use read_only only for passive watchers that "
            "do not modify workspace files."
        ),
    )
    secret_env: tuple[
        Literal[
            "WANDB_API_KEY",
            "MLXFAST_API_TOKEN",
            "YUKON_API_TOKEN",
        ],
        ...,
    ] = Field(
        default=(),
        description=(
            "Registered credentials this process needs. Request WANDB_API_KEY "
            "only for W&B communication, MLXFAST_API_TOKEN only for an "
            "official MLXFast API operation, and YUKON_API_TOKEN only for an "
            "official Yukon API operation."
        ),
    )


class RunJobAction(Action):
    spec: JobSpec = Field(
        description=(
            "Structured argv, workspace directory, and hard timeout for one "
            "long-running process."
        )
    )


class GetJobStatusAction(Action):
    job_id: str = Field(min_length=1, description="Job ID returned by run_job.")


class CancelJobAction(Action):
    job_id: str = Field(
        min_length=1,
        description="Running job ID returned by run_job.",
    )


class MonitorJobAction(Action):
    job_id: str = Field(
        min_length=1,
        pattern=r"^[^:;,#?/\"'\s]+$",
        description="Local job ID returned by run_job.",
    )
    wandb_run_id: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[^:;,#?/\"'\s]+$",
        description=(
            "Exact associated W&B run ID for local metric monitoring. Omit only "
            "when the local job has exactly one associated W&B run."
        ),
    )
    metrics: tuple[MetricMonitorSpec, ...] = Field(
        default=(),
        max_length=3,
        description=(
            "Up to three independent W&B metric policies. Omit to keep "
            "terminal-state monitoring only."
        ),
    )
    poll_interval_seconds: float = Field(
        default=60,
        ge=5,
        allow_inf_nan=False,
        description="Seconds between programmatic monitor polls.",
    )


class AdvisorMonitorJobAction(Action):
    job_id: str = Field(
        min_length=1,
        pattern=r"^[^:;,#?/\"'\s]+$",
        description="W&B run ID in the advisor's configured entity and project.",
    )
    metrics: tuple[MetricMonitorSpec, ...] = Field(
        default=(),
        max_length=3,
        description=(
            "Up to three independent W&B metric policies. Omit to keep "
            "terminal-state monitoring only."
        ),
    )
    poll_interval_seconds: float = Field(
        default=60,
        ge=5,
        allow_inf_nan=False,
        description="Seconds between programmatic monitor polls.",
    )


class MonitorJobObservation(Observation):
    job_id: str
    conversation_id: str
    wandb_run_id: str | None = None
    status: Literal["monitoring"] = "monitoring"

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        binding = (
            f" W&B run: {self.wandb_run_id}."
            if self.wandb_run_id is not None
            else ""
        )
        return [
            TextContent(
                text=(
                    f"Job {self.job_id} is durably monitored.{binding} You may finish "
                    "this turn; the controller will resume this same "
                    f"conversation ({self.conversation_id}) when action is needed."
                )
            )
        ]


class JobResultObservation(Observation):
    job_id: str
    state: JobState
    pid: int | None = None
    process_group_id: int | None = None
    process_start_time: float | None = None
    exit_code: int | None = None
    elapsed_seconds: float
    log_path: str
    wandb_run_ids: tuple[str, ...] = ()
    error_tail: str = ""

    @classmethod
    def from_result(
        cls,
        result: JobResult,
        *,
        mask: object | None = None,
    ) -> Self:
        values = result.model_dump()
        # workspace_access is supervisor metadata rather than model output.
        values.pop("workspace_access", None)
        if callable(mask):
            values["error_tail"] = mask(values["error_tail"])
        return cls.model_validate(values)

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        result = {
            "job_id": self.job_id,
            "state": self.state,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "log_path": self.log_path,
            "wandb_run_ids": self.wandb_run_ids,
        }
        if self.error_tail:
            result["error_tail"] = self.error_tail
        return [
            TextContent(text=json.dumps(result, separators=(",", ":"), default=str))
        ]


class _RunJobExecutor(ToolExecutor[RunJobAction, JobResultObservation]):
    def __init__(
        self,
        supervisor: JobSupervisor,
        monitor_store: JobMonitorStore,
        *,
        role: str,
    ):
        self.supervisor = supervisor
        self.monitor_store = monitor_store
        self.role = role

    def __call__(
        self,
        action: RunJobAction,
        conversation: LocalConversation | None = None,
    ) -> JobResultObservation:
        if conversation is None:
            raise ValueError("run_job requires its parent conversation")
        decision = supervised_job_policy(
            action.spec.argv,
            self.role,
            self.supervisor.workspace,
        )
        if not decision.allowed:
            raise PermissionError(decision.reason)
        if self.role == "student" and action.spec.workspace_access == "mutable":
            require_clean_job_worktree(self.supervisor.workspace)

        environment, redacted_values = _job_environment(
            conversation,
            action.spec.secret_env,
        )
        result = self.supervisor.run_job(
            action.spec,
            env=environment,
            redacted_values=redacted_values,
        )
        try:
            self.monitor_store.register(
                JobMonitorSpec(
                    job_id=result.job_id,
                    conversation_id=conversation.id,
                )
            )
        except BaseException as registration_error:
            try:
                self.supervisor.cancel_job(result.job_id)
            except BaseException as cleanup_error:  # noqa: BLE001
                registration_error.add_note(
                    "job cancellation during monitor-registration rollback failed: "
                    f"{type(cleanup_error).__name__}"
                )
            try:
                self.monitor_store.discard(result.job_id)
            except BaseException as cleanup_error:  # noqa: BLE001
                registration_error.add_note(
                    "monitor discard during registration rollback failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        return _job_observation(result, conversation)

    def close(self) -> None:
        return

    def interrupt(self) -> None:
        # The supervisor is shared by every job. Controller shutdown owns
        # global cancellation; a single interrupted call owns no process.
        return


class _GetJobStatusExecutor(ToolExecutor[GetJobStatusAction, JobResultObservation]):
    def __init__(self, supervisor: JobSupervisor, store: JobMonitorStore):
        self.supervisor = supervisor
        self.store = store

    def __call__(
        self,
        action: GetJobStatusAction,
        conversation: LocalConversation | None = None,
    ) -> JobResultObservation:
        if conversation is None:
            raise ValueError("get_job_status requires its parent conversation")
        _require_owned_job(
            self.store,
            action.job_id,
            conversation,
            "get_job_status",
        )
        result = self.supervisor.get_job_status(action.job_id)
        if result.state is not JobState.RUNNING:
            self.store.record_terminal_and_complete(result)
        return _job_observation(result, conversation)


class _CancelJobExecutor(ToolExecutor[CancelJobAction, JobResultObservation]):
    def __init__(self, supervisor: JobSupervisor, store: JobMonitorStore):
        self.supervisor = supervisor
        self.store = store

    def __call__(
        self,
        action: CancelJobAction,
        conversation: LocalConversation | None = None,
    ) -> JobResultObservation:
        _require_owned_job(self.store, action.job_id, conversation, "cancel_job")
        result = self.supervisor.cancel_job(action.job_id)
        if result.state is JobState.RUNNING:
            raise RuntimeError(
                "cancel_job did not reach a terminal state; "
                "the job monitor remains active"
            )
        self.store.record_terminal_and_complete(result)
        return _job_observation(result, conversation)


def _require_owned_job(
    store: JobMonitorStore,
    job_id: str,
    conversation: LocalConversation | None,
    tool_name: str,
) -> None:
    if conversation is None:
        raise ValueError(f"{tool_name} requires its parent conversation")
    if store.spec(job_id).conversation_id != conversation.id:
        raise PermissionError("job belongs to a different conversation")


def _job_environment(
    conversation: LocalConversation,
    secret_names: Sequence[str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Build a scrubbed child environment and resolve only requested credentials."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if not is_secret_environment_variable(name)
    }

    registry = getattr(getattr(conversation, "state", None), "secret_registry", None)
    redacted_values: list[str] = []
    for name in secret_names:
        value = registry.get_secret_value(name) if registry is not None else None
        if not value:
            raise RuntimeError(f"requested job credential {name} is unavailable")
        environment[name] = value
        redacted_values.append(value)
    return environment, tuple(redacted_values)


def _validate_wandb_run(
    entity: str,
    project: str,
    run_id: str,
    conversation: LocalConversation,
) -> None:
    registry = getattr(getattr(conversation, "state", None), "secret_registry", None)
    api_key = (
        registry.get_secret_value("WANDB_API_KEY") if registry is not None else None
    )
    WandbJobStatusSource(
        entity,
        project,
        api_key=api_key,
    ).get_job_status(run_id)


def _job_observation(
    result: JobResult,
    conversation: LocalConversation | None,
) -> JobResultObservation:
    registry = getattr(getattr(conversation, "state", None), "secret_registry", None)
    mask = registry.mask_secrets_in_output if registry is not None else None
    return JobResultObservation.from_result(result, mask=mask)


class RunJobTool(ToolDefinition[RunJobAction, JobResultObservation]):
    def declared_resources(self, action: Action) -> DeclaredResources:
        return DeclaredResources(keys=(_JOB_CONTROL_RESOURCE,), declared=True)

    @classmethod
    def create(
        cls,
        supervisor: JobSupervisor,
        monitor_store: JobMonitorStore,
        role: str = "advisor",
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Start one supervised long-running process without blocking. "
                    "Use this for training, inference, evaluation, builds, or other "
                    "bounded jobs. Terminal-state monitoring is automatic; use "
                    "monitor_job only to set optional W&B metric policy, and "
                    "get_job_status for one immediate snapshot."
                ),
                action_type=RunJobAction,
                observation_type=JobResultObservation,
                annotations=ToolAnnotations(
                    title="Run job",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=_RunJobExecutor(
                    supervisor,
                    monitor_store,
                    role=role,
                ),
            )
        ]


class GetJobStatusTool(ToolDefinition[GetJobStatusAction, JobResultObservation]):
    def declared_resources(self, action: Action) -> DeclaredResources:
        return DeclaredResources(keys=(_JOB_CONTROL_RESOURCE,), declared=True)

    @classmethod
    def create(
        cls,
        supervisor: JobSupervisor,
        monitor_store: JobMonitorStore,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Read one immediate persisted snapshot for a supervised job."
                ),
                action_type=GetJobStatusAction,
                observation_type=JobResultObservation,
                annotations=ToolAnnotations(
                    title="Get job status",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_GetJobStatusExecutor(supervisor, monitor_store),
            )
        ]


class CancelJobTool(ToolDefinition[CancelJobAction, JobResultObservation]):
    def declared_resources(self, action: Action) -> DeclaredResources:
        return DeclaredResources(keys=(_JOB_CONTROL_RESOURCE,), declared=True)

    @classmethod
    def create(
        cls,
        supervisor: JobSupervisor,
        monitor_store: JobMonitorStore,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Cancel one supervised job, wait for its durable "
                    "terminal state, and retire its monitor. Use this after a stop "
                    "condition or hard monitor signal instead of killing processes "
                    "through the terminal."
                ),
                action_type=CancelJobAction,
                observation_type=JobResultObservation,
                annotations=ToolAnnotations(
                    title="Cancel job",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_CancelJobExecutor(supervisor, monitor_store),
            )
        ]


class _MonitorJobExecutor(ToolExecutor[MonitorJobAction, MonitorJobObservation]):
    def __init__(
        self,
        supervisor: JobSupervisor,
        store: JobMonitorStore,
        *,
        wandb_entity: str | None,
        wandb_project: str | None,
    ):
        self.supervisor = supervisor
        self.store = store
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

    def __call__(
        self,
        action: MonitorJobAction,
        conversation: LocalConversation | None = None,
    ) -> MonitorJobObservation:
        _require_owned_job(self.store, action.job_id, conversation, "monitor_job")
        result = self.supervisor.get_job_status(action.job_id)
        assert conversation is not None
        wandb_run_id = self._resolve_wandb_run_id(action, result)
        if wandb_run_id is not None:
            if not (self.wandb_entity and self.wandb_project):
                raise ValueError(
                    "W&B entity and project are required for metric monitoring"
                )
            _validate_wandb_run(
                self.wandb_entity,
                self.wandb_project,
                wandb_run_id,
                conversation,
            )
        spec = JobMonitorSpec(
            job_id=action.job_id,
            conversation_id=conversation.id,
            wandb_run_id=wandb_run_id,
            metrics=action.metrics,
            poll_interval_seconds=action.poll_interval_seconds,
        )
        self.store.register(spec)
        return MonitorJobObservation(
            job_id=spec.job_id,
            conversation_id=str(spec.conversation_id),
            wandb_run_id=spec.wandb_run_id,
        )

    @staticmethod
    def _resolve_wandb_run_id(
        action: MonitorJobAction,
        result: JobResult,
    ) -> str | None:
        if not action.metrics:
            if action.wandb_run_id is not None:
                raise ValueError("wandb_run_id requires at least one metric policy")
            return None
        if action.wandb_run_id is not None:
            if action.wandb_run_id not in result.wandb_run_ids:
                raise ValueError(
                    f"W&B run {action.wandb_run_id!r} is not associated with job "
                    f"{action.job_id}"
                )
            return action.wandb_run_id
        if len(result.wandb_run_ids) != 1:
            raise ValueError(
                "metric monitoring requires wandb_run_id unless the job has "
                "exactly one associated W&B run"
            )
        return result.wandb_run_ids[0]

    def close(self) -> None:
        return


class MonitorJobTool(ToolDefinition[MonitorJobAction, MonitorJobObservation]):
    def declared_resources(self, action: Action) -> DeclaredResources:
        return DeclaredResources(keys=(_JOB_CONTROL_RESOURCE,), declared=True)

    @classmethod
    def create(
        cls,
        supervisor: JobSupervisor,
        monitor_store: JobMonitorStore,
        *,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Set or replace the monitoring policy for an already-running "
                    "job. run_job automatically registers terminal-state monitoring; "
                    "this tool adds or replaces optional W&B metric gates and "
                    "staleness policy without disabling terminal wakes. Bind metrics "
                    "to an exact associated wandb_run_id; omit it only when the job "
                    "has exactly one associated W&B run. Programmatic polls use no "
                    "model turns."
                ),
                action_type=MonitorJobAction,
                observation_type=MonitorJobObservation,
                annotations=ToolAnnotations(
                    title="Monitor job",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_MonitorJobExecutor(
                    supervisor,
                    monitor_store,
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                ),
            )
        ]


class _AdvisorMonitorJobExecutor(
    ToolExecutor[AdvisorMonitorJobAction, MonitorJobObservation]
):
    def __init__(self, entity: str, project: str, store: JobMonitorStore):
        self.entity = entity
        self.project = project
        self.store = store

    def __call__(
        self,
        action: AdvisorMonitorJobAction,
        conversation: LocalConversation | None = None,
    ) -> MonitorJobObservation:
        if conversation is None:
            raise ValueError("monitor_job requires its advisor conversation")
        _validate_wandb_run(
            self.entity,
            self.project,
            action.job_id,
            conversation,
        )
        spec = JobMonitorSpec(
            job_id=action.job_id,
            conversation_id=conversation.id,
            wandb_run_id=action.job_id if action.metrics else None,
            metrics=action.metrics,
            poll_interval_seconds=action.poll_interval_seconds,
        )
        self.store.register(spec)
        return MonitorJobObservation(
            job_id=spec.job_id,
            conversation_id=str(spec.conversation_id),
            wandb_run_id=spec.wandb_run_id,
        )

    def close(self) -> None:
        return


class AdvisorMonitorJobToolSet(
    ToolDefinition[AdvisorMonitorJobAction, MonitorJobObservation]
):
    """Expose configured-project W&B monitoring to the advisor."""

    name = "monitor_job"

    @classmethod
    def create(
        cls,
        conv_state: object,  # noqa: ARG003
        *,
        state_dir: str | Path,
        wandb_entity: str,
        wandb_project: str,
    ) -> Sequence[Self]:
        store = advisor_job_monitor_store(Path(state_dir))
        return [
            cls(
                description=(
                    "Monitor one W&B job in the configured project. Terminal state "
                    "is always monitored; optionally add up to three independent "
                    "metric policies. Pass the W&B run ID as job_id. Quiet checks "
                    "run in the background without growing model context. Only a "
                    "deduplicated actionable event is prioritized for this "
                    "conversation's next safe turn."
                ),
                action_type=AdvisorMonitorJobAction,
                observation_type=MonitorJobObservation,
                annotations=ToolAnnotations(
                    title="Monitor job",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_AdvisorMonitorJobExecutor(
                    wandb_entity,
                    wandb_project,
                    store,
                ),
            )
        ]


class JobToolSet(ToolDefinition[RunJobAction, JobResultObservation]):
    """Create generic job tools around one process supervisor."""

    @classmethod
    def create(
        cls,
        conv_state: object,
        *,
        state_dir: str | Path,
        max_timeout_seconds: int | None = None,
        role: str | None = None,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ) -> Sequence[ToolDefinition]:
        role = role or os.environ.get("SENPAI_ROLE")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        supervisor, monitor_store = job_runtime(
            Path(conv_state.workspace.working_dir),
            Path(state_dir),
            max_timeout_seconds=max_timeout_seconds,
        )
        tools: list[ToolDefinition] = [
            *RunJobTool.create(
                supervisor=supervisor,
                monitor_store=monitor_store,
                role=role,
            ),
            *GetJobStatusTool.create(
                supervisor=supervisor,
                monitor_store=monitor_store,
            ),
            *CancelJobTool.create(
                supervisor=supervisor,
                monitor_store=monitor_store,
            ),
        ]
        if role == "student":
            tools.extend(
                MonitorJobTool.create(
                    supervisor=supervisor,
                    monitor_store=monitor_store,
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                )
            )
        return tools


class SenpaiTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    """Fail-closed policy wrapper around the native terminal executor."""

    def __init__(
        self,
        delegate: ToolExecutor[TerminalAction, TerminalObservation],
        *,
        role: str,
        workspace: Path,
        foreground_timeout_seconds: int = 600,
    ):
        if foreground_timeout_seconds <= 0:
            raise ValueError("terminal foreground timeout must be positive")
        self.delegate = delegate
        self.role = role
        self.workspace = Path(workspace)
        self.foreground_timeout_seconds = foreground_timeout_seconds

    @property
    def is_pooled(self) -> bool:
        return bool(getattr(self.delegate, "is_pooled", False))

    def __call__(
        self,
        action: TerminalAction,
        conversation: LocalConversation | None = None,
    ) -> TerminalObservation:
        try:
            from senpai_agent.hooks import terminal_policy

            decision = terminal_policy(
                action.command,
                self.role,
                self.workspace,
            )
            if not decision.allowed:
                reason = decision.reason or "No policy reason was provided."
                return _terminal_denial(action, reason)
        except Exception as error:  # noqa: BLE001
            return _terminal_denial(
                action,
                (f"Policy evaluation failed closed ({type(error).__name__})."),
            )
        if (
            action.timeout is not None
            and action.timeout > self.foreground_timeout_seconds
        ):
            action = action.model_copy(
                update={"timeout": float(self.foreground_timeout_seconds)}
            )
        return self.delegate(action, conversation)

    def close(self) -> None:
        self.delegate.close()

    def interrupt(self) -> None:
        self.delegate.interrupt()


def _terminal_denial(
    action: TerminalAction,
    reason: str,
) -> TerminalObservation:
    return TerminalObservation.from_text(
        text=f"Terminal command denied by Senpai policy: {reason}",
        is_error=True,
        command=action.command,
        exit_code=None,
    )


class SenpaiTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    """Create the native terminal behind Senpai's fail-closed policy."""

    @classmethod
    def create(
        cls,
        conv_state: object,
        *,
        role: str | None = None,
    ) -> Sequence[ToolDefinition]:
        role = role or os.environ.get("SENPAI_ROLE")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        try:
            no_change_timeout = int(
                os.environ.get("SENPAI_TERMINAL_NO_CHANGE_TIMEOUT_SECONDS", "600")
            )
            foreground_timeout = int(
                os.environ.get("SENPAI_TERMINAL_FOREGROUND_TIMEOUT_SECONDS", "600")
            )
        except ValueError as error:
            raise RuntimeError(
                "Senpai terminal timeout settings must be integers"
            ) from error
        if min(no_change_timeout, foreground_timeout) <= 0:
            raise RuntimeError("Senpai terminal timeout settings must be positive")
        native = TerminalTool.create(
            conv_state,
            no_change_timeout_seconds=no_change_timeout,
        )[0]
        if native.executor is None:
            raise RuntimeError("native terminal tool has no executor")
        return [
            native.set_executor(
                SenpaiTerminalExecutor(
                    native.executor,
                    role=role,
                    workspace=Path(conv_state.workspace.working_dir),
                    foreground_timeout_seconds=foreground_timeout,
                )
            )
        ]


_TOOLS_REGISTERED = False


def register_senpai_tools() -> None:
    """Register Senpai's serializable OpenHands tool factories once per process."""

    global _TOOLS_REGISTERED
    if _TOOLS_REGISTERED:
        return
    register_tool("senpai_jobs", JobToolSet)
    register_tool("senpai_advisor_job_monitor", AdvisorMonitorJobToolSet)
    register_tool("senpai_github", GitHubWorkflowToolSet)
    register_tool("spawn_agents", SpawnAgentsTool)
    register_tool("await_agents", AwaitAgentsTool)
    register_tool("agent_status", AgentStatusTool)
    register_tool("cancel_agents", CancelAgentsTool)
    register_tool("browser_tool_set", LoadBrowserTool)
    register_tool("load_browser", LoadBrowserTool)
    register_tool("task_tracker", SenpaiTaskTrackerTool)
    register_tool("senpai_terminal", SenpaiTerminalTool)
    _TOOLS_REGISTERED = True
