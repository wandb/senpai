"""Typed OpenHands tools for Senpai's reliable control-plane operations."""

from __future__ import annotations

import json
import os
import threading
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
from openhands.tools.terminal import (
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)
from openhands.tools.task_tracker import TaskTrackerTool
from pydantic import Field, model_validator

from senpai_agent.delegation import (
    AgentStatusTool,
    AwaitAgentsTool,
    CancelAgentsTool,
    DelegateAgentTool,
    SpawnAgentsTool,
)
from senpai_agent.git_workflow import require_clean_training_worktree
from senpai_agent.github.tools import GitHubWorkflowToolSet
from senpai_agent.monitor import MetricGate, MonitorStore, TrainingMonitorSpec
from senpai_agent.training import (
    TrainingResult,
    TrainingSpec,
    TrainingState,
    TrainingSupervisor,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState, LocalConversation


_TRAINING_RUNTIMES: dict[
    Path,
    tuple[TrainingSupervisor, MonitorStore],
] = {}
_BROWSER_ENABLED_STATE_KEY = "senpai.browser_enabled"


class LoadBrowserAction(Action):
    """Enable the browser tool family for this conversation."""


class LoadBrowserObservation(Observation):
    tools: tuple[str, ...]

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=(
                    "Browser tools are now available: "
                    + ", ".join(self.tools)
                    + "."
                )
            )
        ]


class _LoadBrowserExecutor(
    ToolExecutor[LoadBrowserAction, LoadBrowserObservation]
):
    def __call__(
        self,
        action: LoadBrowserAction,  # noqa: ARG002
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

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
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


_TASK_TRACKER_DESCRIPTION = """Maintain an optional persisted task list for complex work.

Use it when a task has several meaningful steps or concurrent workstreams and a
compact todo/in-progress/done record will prevent omissions. Multiple items may
be in progress when the work is genuinely parallel. Skip it for short, atomic,
or purely informational work. View the current list before replacing it, keep
entries concise, and mark work done only when its required evidence is complete."""


class SenpaiTaskTrackerTool(TaskTrackerTool):
    """OpenHands task persistence with a concise, concurrency-safe description."""

    name = "task_tracker"

    @classmethod
    def create(cls, conv_state: ConversationState) -> Sequence[ToolDefinition]:
        return [
            tool.model_copy(update={"description": _TASK_TRACKER_DESCRIPTION})
            for tool in super().create(conv_state)
        ]


def training_runtime(
    workspace: Path,
    state_dir: Path,
    *,
    max_timeout_seconds: int | None = None,
) -> tuple[TrainingSupervisor, MonitorStore]:
    key = state_dir.resolve()
    runtime = _TRAINING_RUNTIMES.get(key)
    if runtime is None:
        runtime = (
            TrainingSupervisor(
                workspace=workspace,
                state_dir=key,
                max_timeout_seconds=max_timeout_seconds,
            ),
            MonitorStore(key / "monitors.sqlite3"),
        )
        _TRAINING_RUNTIMES[key] = runtime
    return runtime


def close_training_runtimes() -> None:
    for training, monitors in _TRAINING_RUNTIMES.values():
        training.close()
        monitors.close()
    _TRAINING_RUNTIMES.clear()


class RunTrainingAction(Action):
    spec: TrainingSpec = Field(
        description=(
            "Structured process argv, assignment-workspace directory, and hard "
            "timeout. Do not pass a shell command string."
        )
    )


class GetTrainingStatusAction(Action):
    training_id: str = Field(
        min_length=1,
        description="Training ID returned by run_training.",
    )


class CancelTrainingAction(Action):
    training_id: str = Field(
        min_length=1,
        description="Running training ID returned by run_training.",
    )


class MonitorTrainingAction(Action):
    training_id: str = Field(
        min_length=1,
        description="Training ID returned by run_training.",
    )
    metric: str | None = Field(
        default=None,
        description="W&B metric to monitor. Omit for terminal process state only.",
    )
    direction: Literal["min", "max"] | None = Field(
        default=None,
        description="Whether lower or higher values are better for change gates.",
    )
    gates: tuple[MetricGate, ...] = Field(
        default=(),
        description=(
            "Metric thresholds or changes that should resume this conversation. "
            "Ordinary polls are programmatic and do not consume model tokens."
        ),
    )
    poll_interval_seconds: float = Field(
        default=60,
        gt=0,
        description="Seconds between programmatic monitor polls.",
    )
    stale_after_seconds: float = Field(
        default=600,
        gt=0,
        description="Notify when the selected metric has not updated this long.",
    )

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_status_filter(cls, value: object) -> object:
        """Resume conversations written before terminal wakes became mandatory."""

        if isinstance(value, dict) and "notify_on_status" in value:
            return {key: item for key, item in value.items() if key != "notify_on_status"}
        return value


class MonitorTrainingObservation(Observation):
    training_id: str
    conversation_id: str
    status: Literal["monitoring"] = "monitoring"

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=(
                    f"Training {self.training_id} is durably monitored. You may "
                    "finish this turn; the controller will resume this same "
                    f"conversation ({self.conversation_id}) when action is needed."
                )
            )
        ]


class TrainingResultObservation(Observation):
    training_id: str
    state: TrainingState
    pid: int | None = None
    process_group_id: int | None = None
    process_start_time: float | None = None
    exit_code: int | None = None
    elapsed_seconds: float
    log_path: str
    wandb_run_ids: tuple[str, ...] = ()
    error_tail: str = ""

    @classmethod
    def from_result(cls, result: TrainingResult) -> Self:
        return cls.model_validate(result.model_dump())

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        result = {
            "training_id": self.training_id,
            "state": self.state,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "log_path": self.log_path,
            "wandb_run_ids": self.wandb_run_ids,
        }
        if self.error_tail:
            result["error_tail"] = self.error_tail
        text = json.dumps(result, separators=(",", ":"), default=str)
        return [TextContent(text=text)]


class _RunTrainingExecutor(ToolExecutor[RunTrainingAction, TrainingResultObservation]):
    def __init__(self, training: TrainingSupervisor, monitor_store: MonitorStore):
        self.training = training
        self.monitor_store = monitor_store
        self._lock = threading.Lock()
        self._in_flight: set[str] = set()
        self._interrupt_generation = 0

    def __call__(
        self,
        action: RunTrainingAction,
        conversation: LocalConversation | None = None,
    ) -> TrainingResultObservation:
        if conversation is None:
            raise ValueError("run_training requires its student conversation")
        require_clean_training_worktree(self.training.workspace)
        with self._lock:
            interrupt_generation = self._interrupt_generation
        result = self.training.run_training(action.spec)
        with self._lock:
            self._in_flight.add(result.training_id)
            interrupted = interrupt_generation != self._interrupt_generation
        try:
            if interrupted:
                self.training.cancel_training(result.training_id)
            self.monitor_store.register(
                TrainingMonitorSpec(
                    training_id=result.training_id,
                    conversation_id=conversation.id,
                )
            )
            return TrainingResultObservation.from_result(result)
        finally:
            with self._lock:
                self._in_flight.discard(result.training_id)

    def close(self) -> None:
        return

    def interrupt(self) -> None:
        with self._lock:
            self._interrupt_generation += 1
            training_ids = tuple(self._in_flight)
        for training_id in training_ids:
            self.training.cancel_training(training_id)


class _GetTrainingStatusExecutor(
    ToolExecutor[GetTrainingStatusAction, TrainingResultObservation]
):
    def __init__(self, training: TrainingSupervisor):
        self.training = training

    def __call__(
        self,
        action: GetTrainingStatusAction,
        conversation: LocalConversation | None = None,
    ) -> TrainingResultObservation:
        return TrainingResultObservation.from_result(
            self.training.get_training_status(action.training_id)
        )


class _CancelTrainingExecutor(
    ToolExecutor[CancelTrainingAction, TrainingResultObservation]
):
    def __init__(self, training: TrainingSupervisor, store: MonitorStore):
        self.training = training
        self.store = store

    def __call__(
        self,
        action: CancelTrainingAction,
        conversation: LocalConversation | None = None,
    ) -> TrainingResultObservation:
        if conversation is None:
            raise ValueError("cancel_training requires its student conversation")
        monitor = self.store.spec(action.training_id)
        if monitor.conversation_id != conversation.id:
            raise PermissionError(
                "training belongs to a different student conversation"
            )
        result = self.training.cancel_training(action.training_id)
        if result.state is TrainingState.RUNNING:
            raise RuntimeError(
                "cancel_training did not reach a terminal state; "
                "the training monitor remains active"
            )
        self.store.complete(action.training_id)
        return TrainingResultObservation.from_result(result)


class RunTrainingTool(ToolDefinition[RunTrainingAction, TrainingResultObservation]):
    @classmethod
    def create(
        cls,
        training: TrainingSupervisor,
        monitor_store: MonitorStore,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Start one supervised training process without blocking and "
                    "automatically monitor its terminal state for this conversation. "
                    "Use monitor_training only to add metric gates or staleness "
                    "policy; use get_training_status for a bounded immediate check."
                ),
                action_type=RunTrainingAction,
                observation_type=TrainingResultObservation,
                annotations=ToolAnnotations(
                    title="Run training",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=_RunTrainingExecutor(training, monitor_store),
            )
        ]


class GetTrainingStatusTool(
    ToolDefinition[GetTrainingStatusAction, TrainingResultObservation]
):
    @classmethod
    def create(
        cls,
        training: TrainingSupervisor,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Read the latest persisted result for one supervised training ID."
                ),
                action_type=GetTrainingStatusAction,
                observation_type=TrainingResultObservation,
                annotations=ToolAnnotations(
                    title="Get training status",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_GetTrainingStatusExecutor(training),
            )
        ]


class CancelTrainingTool(
    ToolDefinition[CancelTrainingAction, TrainingResultObservation]
):
    @classmethod
    def create(
        cls,
        training: TrainingSupervisor,
        monitor_store: MonitorStore,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Cancel one supervised training process, wait for its durable "
                    "terminal state, and retire its monitor. Use this after a stop "
                    "condition or hard monitor signal instead of killing processes "
                    "through the terminal."
                ),
                action_type=CancelTrainingAction,
                observation_type=TrainingResultObservation,
                annotations=ToolAnnotations(
                    title="Cancel training",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_CancelTrainingExecutor(training, monitor_store),
            )
        ]


class _MonitorTrainingExecutor(
    ToolExecutor[MonitorTrainingAction, MonitorTrainingObservation]
):
    def __init__(self, training: TrainingSupervisor, store: MonitorStore):
        self.training = training
        self.store = store

    def __call__(
        self,
        action: MonitorTrainingAction,
        conversation: LocalConversation | None = None,
    ) -> MonitorTrainingObservation:
        if conversation is None:
            raise ValueError("monitor_training requires its student conversation")
        self.training.get_training_status(action.training_id)
        monitor = self.store.spec(action.training_id)
        if monitor.conversation_id != conversation.id:
            raise PermissionError(
                "training belongs to a different student conversation"
            )
        spec = TrainingMonitorSpec(
            training_id=action.training_id,
            conversation_id=conversation.id,
            metric=action.metric,
            direction=action.direction,
            gates=action.gates,
            poll_interval_seconds=action.poll_interval_seconds,
            stale_after_seconds=action.stale_after_seconds,
        )
        self.store.register(spec)
        return MonitorTrainingObservation(
            training_id=spec.training_id,
            conversation_id=str(spec.conversation_id),
        )

    def close(self) -> None:
        return


class MonitorTrainingTool(
    ToolDefinition[MonitorTrainingAction, MonitorTrainingObservation]
):
    @classmethod
    def create(
        cls,
        training: TrainingSupervisor,
        monitor_store: MonitorStore,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Upgrade one training process's automatic terminal monitor "
                    "without model polling. Specify an optional W&B metric, "
                    "direction, threshold/change gates, and stale timeout. Senpai "
                    "resumes this same student conversation when the policy emits "
                    "a signal."
                ),
                action_type=MonitorTrainingAction,
                observation_type=MonitorTrainingObservation,
                annotations=ToolAnnotations(
                    title="Monitor training",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_MonitorTrainingExecutor(training, monitor_store),
            )
        ]


class TrainingToolSet(ToolDefinition[RunTrainingAction, TrainingResultObservation]):
    """Create the training tools around one process supervisor."""

    @classmethod
    def create(
        cls,
        conv_state: object,
        *,
        state_dir: str | Path,
        max_timeout_seconds: int | None = None,
    ) -> Sequence[ToolDefinition]:
        training, monitor_store = training_runtime(
            Path(conv_state.workspace.working_dir),
            Path(state_dir),
            max_timeout_seconds=max_timeout_seconds,
        )
        return (
            *RunTrainingTool.create(
                training=training,
                monitor_store=monitor_store,
            ),
            *GetTrainingStatusTool.create(training=training),
            *CancelTrainingTool.create(
                training=training,
                monitor_store=monitor_store,
            ),
            *MonitorTrainingTool.create(
                training=training,
                monitor_store=monitor_store,
            ),
        )


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
    from senpai_agent.operational_tools import SupervisorOperationTool

    register_tool("senpai_training", TrainingToolSet)
    register_tool("senpai_github", GitHubWorkflowToolSet)
    register_tool("spawn_agents", SpawnAgentsTool)
    register_tool("await_agents", AwaitAgentsTool)
    register_tool("agent_status", AgentStatusTool)
    register_tool("cancel_agents", CancelAgentsTool)
    register_tool("delegate_agent", DelegateAgentTool)
    register_tool("browser_tool_set", LoadBrowserTool)
    register_tool("load_browser", LoadBrowserTool)
    register_tool("task_tracker", SenpaiTaskTrackerTool)
    register_tool("senpai_terminal", SenpaiTerminalTool)
    register_tool("senpai_operations", SupervisorOperationTool)
    _TOOLS_REGISTERED = True
