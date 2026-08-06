"""Typed OpenHands tools for Senpai's reliable control-plane operations."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.terminal import (
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from senpai_agent.delegation import (
    AgentStatusTool,
    AwaitAgentsTool,
    CancelAgentsTool,
    DelegateAgentTool,
    SpawnAgentsTool,
)
from senpai_agent.git_workflow import (
    create_assignment_branch,
    push_assignment_branch,
    require_clean_training_worktree,
)
from senpai_agent.github import PRRetrievalResult, get_prs
from senpai_agent.github_workflow import (
    GitHubWorkflow,
    MutationResult,
    PullHeadMismatchError,
    StaleAssignmentRevisionError,
)
from senpai_agent.models import (
    AssignmentRecord,
    DispositionRecord,
    ExperimentResult,
    render_disposition_marker,
)
from senpai_agent.monitor import MetricGate, MonitorStore, TrainingMonitorSpec
from senpai_agent.training import (
    TrainingResult,
    TrainingSpec,
    TrainingState,
    TrainingSupervisor,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


@dataclass(frozen=True)
class GitHubCredentials:
    repo: str
    token: SecretStr
    trusted_actor: str | None = None


_GITHUB_CREDENTIALS: GitHubCredentials | None = None
_POST_PUSH_HEAD_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0)
_TRAINING_RUNTIMES: dict[
    Path,
    tuple[TrainingSupervisor, MonitorStore],
] = {}


def configure_github_credentials(
    repo: str,
    token: SecretStr,
    *,
    trusted_actor: str | None = None,
) -> None:
    """Hold write auth outside model-facing tool specs and terminal secrets."""

    global _GITHUB_CREDENTIALS
    if len(repo.split("/")) != 2 or not all(repo.split("/")):
        raise ValueError("repo must use owner/name form")
    if not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if not token.get_secret_value().strip():
        raise ValueError("token must not be empty")
    if trusted_actor is not None and not trusted_actor.strip():
        raise ValueError("trusted actor must not be empty")
    _GITHUB_CREDENTIALS = GitHubCredentials(
        repo=repo,
        token=token,
        trusted_actor=trusted_actor,
    )


def clear_github_credentials() -> None:
    global _GITHUB_CREDENTIALS
    _GITHUB_CREDENTIALS = None


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

    def __call__(
        self,
        action: RunTrainingAction,
        conversation: LocalConversation | None = None,
    ) -> TrainingResultObservation:
        if conversation is None:
            raise ValueError("run_training requires its student conversation")
        require_clean_training_worktree(self.training.workspace)
        result = self.training.run_training(action.spec)
        self.monitor_store.register(
            TrainingMonitorSpec(
                training_id=result.training_id,
                conversation_id=conversation.id,
            )
        )
        return TrainingResultObservation.from_result(result)

    def close(self) -> None:
        return

    def interrupt(self) -> None:
        self.training.close()


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


class GetPRsAction(Action):
    repo: str = Field(
        min_length=3,
        description="GitHub repository in owner/name form.",
    )
    numbers: tuple[int, ...] = Field(
        default=(),
        description="Explicit positive PR numbers to include.",
    )
    date_range: tuple[str | date, str | date] | None = Field(
        default=None,
        description="Optional inclusive PR creation-date range.",
    )
    search: str | None = Field(
        default=None,
        description="Optional GitHub issue-search terms or qualifiers.",
    )
    max_inline_prs: int = Field(
        default=5,
        ge=0,
        description=(
            "Maximum PRs returned inline. Do not set this >5 unless explicitly "
            "necessary: more than 5 inline PRs risks severe agent context "
            "pollution. Prefer the returned artifact path."
        ),
    )


class PRManifestObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    head_sha: str
    url: str


class GetPRsObservation(Observation):
    manifest: tuple[PRManifestObservation, ...]
    markdown: str | None = None
    path: str | None = None

    @classmethod
    def from_result(cls, result: PRRetrievalResult) -> Self:
        return cls(
            manifest=tuple(
                PRManifestObservation(
                    number=entry.number,
                    title=entry.title,
                    head_sha=entry.head_sha,
                    url=entry.url,
                )
                for entry in result.manifest
            ),
            markdown=result.markdown,
            path=str(result.path) if result.path is not None else None,
        )

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        if self.markdown is not None:
            return [TextContent(text=self.markdown)]

        manifest = "\n".join(
            (f"- #{entry.number} `{entry.head_sha}` {entry.title} ({entry.url})")
            for entry in self.manifest
        )
        return [
            TextContent(
                text=(
                    f"Full PR context is stored at: {self.path}\n"
                    f"Compact manifest:\n{manifest}"
                )
            )
        ]


class _GetPRsExecutor(ToolExecutor[GetPRsAction, GetPRsObservation]):
    def __init__(
        self,
        get_prs_fn: Callable[..., PRRetrievalResult],
        *,
        credentials: GitHubCredentials | None = None,
        artifact_dir: Path,
        target_workspace: Path,
    ):
        self.get_prs = get_prs_fn
        self.credentials = credentials
        self.artifact_dir = artifact_dir
        self.target_workspace = target_workspace

    def __call__(
        self,
        action: GetPRsAction,
        conversation: LocalConversation | None = None,
    ) -> GetPRsObservation:
        if self.credentials is not None and action.repo != self.credentials.repo:
            raise PermissionError(
                "requested repository does not match configured GitHub credentials"
            )
        auth = {"token": self.credentials.token} if self.credentials is not None else {}
        result = self.get_prs(
            action.repo,
            numbers=action.numbers,
            date_range=action.date_range,
            search=action.search,
            max_inline_prs=action.max_inline_prs,
            artifact_dir=self.artifact_dir,
            target_workspace=self.target_workspace,
            **auth,
        )
        return GetPRsObservation.from_result(result)


class GetPRsTool(ToolDefinition[GetPRsAction, GetPRsObservation]):
    name = "get_prs"

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        *,
        get_prs_fn: Callable[..., PRRetrievalResult] = get_prs,
        state_dir: str | Path | None = None,
        workspace: str | Path | None = None,
    ) -> Sequence[Self]:
        credentials = _GITHUB_CREDENTIALS if get_prs_fn is get_prs else None
        if get_prs_fn is get_prs and credentials is None:
            raise RuntimeError(
                "configure GitHub credentials before initializing get_prs"
            )
        if workspace is None:
            if conv_state is None:
                raise ValueError("get_prs requires its OpenHands workspace")
            workspace = Path(conv_state.workspace.working_dir)
        target_workspace = Path(workspace).resolve()
        artifact_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else Path(tempfile.gettempdir()).resolve() / "senpai-pr-artifacts"
        )
        if artifact_dir == target_workspace or artifact_dir.is_relative_to(
            target_workspace
        ):
            raise ValueError("get_prs state_dir must be outside the target workspace")
        return [
            cls(
                description=(
                    "Retrieve complete PR bodies, comments, reviews, and inline "
                    "comments by number, date range, and/or search. Large results "
                    "are returned as one external Markdown artifact."
                ),
                action_type=GetPRsAction,
                observation_type=GetPRsObservation,
                annotations=ToolAnnotations(
                    title="Get pull requests",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_GetPRsExecutor(
                    get_prs_fn,
                    credentials=credentials,
                    artifact_dir=artifact_dir,
                    target_workspace=target_workspace,
                ),
            )
        ]


class _Transition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str | None = Field(
        default=None,
        min_length=3,
        description=(
            "Optional target repository in owner/name form. When supplied, it "
            "must match the repository bound to this Senpai runtime."
        ),
    )


class ReconcileLabelsTransition(_Transition):
    operation: Literal["reconcile_labels"]
    pr_number: int = Field(gt=0)
    assignment_id: str = Field(min_length=1)
    expected_head_sha: str = Field(min_length=1)
    add: set[str] = Field(default_factory=set)
    remove: set[str] = Field(default_factory=set)


class RequestRevisionTransition(_Transition):
    operation: Literal["request_revision"]
    pr_number: int = Field(gt=0)
    assignment_id: str = Field(min_length=1)
    expected_head_sha: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    comment: str = Field(min_length=1)


class SendAssignmentFeedbackTransition(_Transition):
    operation: Literal["send_assignment_feedback"]
    pr_number: int = Field(gt=0)
    assignment_id: str = Field(min_length=1)
    revision_id: str = Field(
        min_length=1,
        description="Current assignment revision; this operation does not change it.",
    )
    expected_head_sha: str = Field(min_length=1)
    feedback_id: str = Field(
        min_length=1,
        max_length=256,
        description=(
            "Stable ID for this distinct guidance item within the assignment "
            "revision. Exact replay is a no-op; use a new ID for changed guidance."
        ),
    )
    comment: str = Field(
        min_length=1,
        max_length=50_000,
        description="Actionable guidance that does not require a new revision.",
    )


class RespondToIssueTransition(_Transition):
    operation: Literal["respond_to_issue"]
    issue_number: int = Field(gt=0)
    human_message_id: int = Field(
        gt=0,
        description=(
            "Exact numeric ID of the human-authored issue body or comment "
            "being answered."
        ),
    )
    response: str = Field(
        min_length=1,
        max_length=50_000,
        description="Response including the role prefix required by the skill.",
    )


class SubmitResultTransition(_Transition):
    operation: Literal["submit_result"]
    pr_number: int = Field(gt=0)
    branch: str = Field(min_length=1)
    expected_remote_sha: str = Field(
        min_length=1,
        description=(
            "Current remote branch SHA before the push. This is the "
            "force-with-lease precondition, not the local result commit."
        ),
    )
    expected_head_sha: str = Field(
        min_length=1,
        description=(
            "Local commit to push. Must equal result.assignment.expected_head_sha "
            "and result.commit_sha."
        ),
    )
    result: ExperimentResult = Field(
        description=(
            "Result for expected_head_sha. result.assignment.expected_head_sha "
            "and result.commit_sha must both equal expected_head_sha."
        )
    )


class CloseExperimentTransition(_Transition):
    operation: Literal["close_experiment"]
    pr_number: int = Field(gt=0)
    expected_head_sha: str = Field(min_length=1)
    assignment_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MergeExperimentTransition(_Transition):
    operation: Literal["merge_experiment"]
    pr_number: int = Field(gt=0)
    expected_head_sha: str = Field(min_length=1)
    assignment_id: str = Field(min_length=1)
    merge_method: Literal["merge", "squash", "rebase"] = "squash"
    accepted_base_sha: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Exact current SHA of the assignment's base branch. Omit when it "
            "still equals the assignment marker's base SHA. Set only after "
            "reviewing a baseline_advanced event and deliberately accepting "
            "the result against that newer baseline."
        ),
    )


class PushBranchTransition(_Transition):
    operation: Literal["push_branch"]
    branch: str = Field(min_length=1)
    expected_remote_sha: str = Field(min_length=1)
    expected_head_sha: str = Field(
        min_length=1,
        description=(
            "Expected local commit to publish. The transition fails if the "
            "worktree HEAD differs."
        ),
    )


class CreateAssignmentTransition(_Transition):
    operation: Literal["create_assignment"]
    assignment_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    student: str = Field(min_length=1)
    base_branch: str = Field(min_length=1)
    expected_base_sha: str = Field(min_length=1)
    head_branch: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(min_length=1, max_length=50_000)


GitHubTransition = Annotated[
    CreateAssignmentTransition
    | ReconcileLabelsTransition
    | RequestRevisionTransition
    | SendAssignmentFeedbackTransition
    | RespondToIssueTransition
    | SubmitResultTransition
    | CloseExperimentTransition
    | MergeExperimentTransition
    | PushBranchTransition,
    Field(discriminator="operation"),
]


class GitHubTransitionAction(Action):
    transition: GitHubTransition = Field(
        description=(
            "One typed, preconditioned, idempotent GitHub workflow transition. "
            "For submit_result, expected_remote_sha is the current remote branch "
            "SHA before push and expected_head_sha is the local commit to push; "
            "transition.expected_head_sha, result.assignment.expected_head_sha, "
            "and result.commit_sha must be identical."
        )
    )


class GitHubTransitionObservation(Observation):
    changed: bool
    resource_url: str
    state: str
    version: str | None = None

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=json.dumps(
                    self.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ]


class _GitHubTransitionExecutor(
    ToolExecutor[GitHubTransitionAction, GitHubTransitionObservation]
):
    def __init__(
        self,
        workflow: GitHubWorkflow,
        role: str,
        workspace: Path,
        git_token: SecretStr | None = None,
        advisor_branch: str | None = None,
    ):
        self.workflow = workflow
        self.role = role
        self.workspace = workspace
        self.git_token = git_token
        self.advisor_branch = advisor_branch

    def __call__(
        self,
        action: GitHubTransitionAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubTransitionObservation:
        transition = action.transition
        if isinstance(
            transition,
            (
                CreateAssignmentTransition,
                PushBranchTransition,
                ReconcileLabelsTransition,
                RequestRevisionTransition,
                SendAssignmentFeedbackTransition,
                CloseExperimentTransition,
                MergeExperimentTransition,
            ),
        ):
            self._require_role("advisor")
        elif isinstance(transition, SubmitResultTransition):
            self._require_role("student")
        self._require_repo_scope(transition.repo)

        if isinstance(transition, CreateAssignmentTransition):
            branch = create_assignment_branch(
                self.workspace,
                branch=transition.head_branch,
                base_branch=transition.base_branch,
                expected_base_sha=transition.expected_base_sha,
                assignment_id=transition.assignment_id,
                token=self.git_token,
            )
            result = self.workflow.create_assignment(
                AssignmentRecord(
                    repo=self.workflow.repo,
                    assignment_id=transition.assignment_id,
                    revision_id=transition.revision_id,
                    student=transition.student,
                    base_ref=transition.base_branch,
                    base_sha=transition.expected_base_sha,
                    head_ref=transition.head_branch,
                    head_sha=branch.head_sha,
                ),
                title=transition.title,
                body=transition.body,
            )
        elif isinstance(transition, PushBranchTransition):
            if transition.branch != self.advisor_branch:
                raise PermissionError(
                    "push_branch is limited to the configured advisor branch "
                    f"{self.advisor_branch!r}"
                )
            push_options = {
                "branch": transition.branch,
                "expected_remote_sha": transition.expected_remote_sha,
                "expected_local_sha": transition.expected_head_sha,
                "token": self.git_token,
            }
            pushed = push_assignment_branch(self.workspace, **push_options)
            return GitHubTransitionObservation(
                changed=pushed.changed,
                resource_url=f"git:origin/{pushed.branch}",
                state="branch_pushed",
                version=pushed.head_sha,
            )
        elif isinstance(transition, SubmitResultTransition):
            try:
                self.workflow.preflight_submit_result(
                    transition.pr_number,
                    branch=transition.branch,
                    current_head_sha=transition.expected_remote_sha,
                    expected_result_head_sha=transition.expected_head_sha,
                    result=transition.result,
                )
            except StaleAssignmentRevisionError as error:
                if conversation is not None:
                    conversation.state.execution_status = (
                        ConversationExecutionStatus.FINISHED
                    )
                raise ValueError(
                    f"{error} Ending this stale turn so the controller can resume "
                    "the current assignment revision."
                ) from error
            pushed = push_assignment_branch(
                self.workspace,
                branch=transition.branch,
                expected_remote_sha=transition.expected_remote_sha,
                expected_local_sha=transition.expected_head_sha,
                token=self.git_token,
            )
            result = self._submit_result_after_push(transition)
        elif isinstance(transition, ReconcileLabelsTransition):
            result = self.workflow.reconcile_labels(
                transition.pr_number,
                assignment_id=transition.assignment_id,
                add=transition.add,
                remove=transition.remove,
                expected_head_sha=transition.expected_head_sha,
            )
        elif isinstance(transition, RequestRevisionTransition):
            result = self.workflow.request_revision(
                transition.pr_number,
                assignment_id=transition.assignment_id,
                expected_head_sha=transition.expected_head_sha,
                revision_id=transition.revision_id,
                comment=transition.comment,
            )
        elif isinstance(transition, SendAssignmentFeedbackTransition):
            result = self.workflow.send_assignment_feedback(
                transition.pr_number,
                assignment_id=transition.assignment_id,
                revision_id=transition.revision_id,
                expected_head_sha=transition.expected_head_sha,
                feedback_id=transition.feedback_id,
                comment=transition.comment,
            )
        elif isinstance(transition, RespondToIssueTransition):
            result = self.workflow.respond_to_issue(
                transition.issue_number,
                human_message_id=transition.human_message_id,
                response=transition.response,
            )
        elif isinstance(transition, CloseExperimentTransition):
            result = self.workflow.close_experiment(
                transition.pr_number,
                assignment_id=transition.assignment_id,
                expected_head_sha=transition.expected_head_sha,
                marker=render_disposition_marker(
                    DispositionRecord(
                        repo=self.workflow.repo,
                        pr_number=transition.pr_number,
                        assignment_id=transition.assignment_id,
                        head_sha=transition.expected_head_sha,
                    )
                ),
                reason=transition.reason,
            )
        elif isinstance(transition, MergeExperimentTransition):
            result = self.workflow.merge_experiment(
                transition.pr_number,
                expected_head_sha=transition.expected_head_sha,
                assignment_id=transition.assignment_id,
                merge_method=transition.merge_method,
                accepted_base_sha=transition.accepted_base_sha,
            )
        else:
            raise TypeError(
                f"unsupported GitHub transition: {type(transition).__name__}"
            )
        return GitHubTransitionObservation(
            changed=result.changed,
            resource_url=result.resource_url,
            state=result.state,
            version=result.version,
        )

    def _submit_result_after_push(
        self,
        transition: SubmitResultTransition,
    ) -> MutationResult:
        for delay in _POST_PUSH_HEAD_RETRY_DELAYS:
            try:
                return self.workflow.submit_result(
                    transition.pr_number,
                    expected_head_sha=transition.expected_head_sha,
                    result=transition.result,
                )
            except PullHeadMismatchError:
                time.sleep(delay)
        return self.workflow.submit_result(
            transition.pr_number,
            expected_head_sha=transition.expected_head_sha,
            result=transition.result,
        )

    def _require_role(self, expected: str) -> None:
        if self.role != expected:
            raise PermissionError(
                f"{self.role} cannot perform this {expected}-owned transition"
            )

    def _require_repo_scope(self, repo: str | None) -> None:
        if repo is None:
            return
        configured_repo = self.workflow.repo
        if repo != configured_repo:
            raise PermissionError(
                "transition repository does not match the configured GitHub "
                "repository"
            )


class GitHubTransitionTool(
    ToolDefinition[GitHubTransitionAction, GitHubTransitionObservation]
):
    name = "github_transition"

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        workflow: GitHubWorkflow | None = None,
        *,
        role: str | None = None,
        workspace: str | Path | None = None,
        advisor_branch: str | None = None,
    ) -> Sequence[Self]:
        role = role or os.environ.get("SENPAI_ROLE")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        advisor_branch = advisor_branch or os.environ.get("ADVISOR_BRANCH")
        git_token: SecretStr | None = None
        if workflow is None:
            credentials = _GITHUB_CREDENTIALS
            if credentials is None:
                raise RuntimeError(
                    "configure GitHub credentials before initializing workflows"
                )
            workflow = GitHubWorkflow(
                credentials.repo,
                credentials.token,
                trusted_actor=credentials.trusted_actor,
            )
            git_token = credentials.token
        if workspace is None:
            if conv_state is None:
                raise ValueError("github_transition requires its OpenHands workspace")
            workspace = Path(conv_state.workspace.working_dir)
        return [
            cls(
                description=(
                    "Apply one verified GitHub workflow transition. Operations are "
                    "create_assignment, push_branch, reconcile_labels, "
                    "request_revision, send_assignment_feedback, respond_to_issue, "
                    "submit_result, close_experiment, and merge_experiment. Every "
                    "mutation verifies its durable identity and converges on replay."
                ),
                action_type=GitHubTransitionAction,
                observation_type=GitHubTransitionObservation,
                annotations=ToolAnnotations(
                    title="GitHub transition",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_GitHubTransitionExecutor(
                    workflow,
                    role,
                    Path(workspace),
                    git_token,
                    advisor_branch,
                ),
            )
        ]


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
    register_tool("get_prs", GetPRsTool)
    register_tool("github_transition", GitHubTransitionTool)
    register_tool("spawn_agents", SpawnAgentsTool)
    register_tool("await_agents", AwaitAgentsTool)
    register_tool("agent_status", AgentStatusTool)
    register_tool("cancel_agents", CancelAgentsTool)
    register_tool("delegate_agent", DelegateAgentTool)
    register_tool("senpai_terminal", SenpaiTerminalTool)
    register_tool("senpai_operations", SupervisorOperationTool)
    _TOOLS_REGISTERED = True
