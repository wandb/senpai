"""One process-isolated delegation path for every Senpai subagent."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

import psutil
from openhands.sdk.event import ActionEvent, LLMConvertibleEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from pydantic import BaseModel, Field, model_validator

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.processes import terminate_process_group
from senpai_agent.secrets import scrub_github_credentials

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


AgentKind = Literal["general-purpose", "explore", "search", "bash-runner"]
ModelTier = Literal["smart", "fast", "frontier"]
SearchMode = Literal["general-web", "research-publications"]
MAX_PARALLEL_AGENTS = 8
MAX_DELEGATION_DEPTH = 2
MAX_TREE_AGENTS = 8
MAX_SPAWN_BATCH = 8
MAX_AWAIT_SECONDS = 300
MODEL_TIER_TIMEOUT_SECONDS: Mapping[ModelTier, float] = {
    "fast": 600,
    "smart": 1800,
    "frontier": 3600,
}
TaskStatus = Literal[
    "queued",
    "running",
    "finished",
    "failed",
    "cancelled",
]
JoinMode = Literal["all", "first", "quorum"]
TERMINAL_TASK_STATUSES = frozenset({"finished", "failed", "cancelled"})


class AdvisorEventSink(Protocol):
    def enqueue(self, event: AdvisorEvent) -> bool: ...


@dataclass(frozen=True)
class DelegationRequest:
    task_id: str
    parent_conversation_id: str
    parent_context: tuple[Message, ...]
    agent: AgentKind
    model: ModelTier
    search_mode: SearchMode | None
    tree_id: str = ""
    depth: int = 1
    deadline_epoch: float | None = None
    registry_path: Path | None = None
    event_db_path: Path | None = None
    parent_task_id: str | None = None


class ChildAgentRunner(Protocol):
    def start(
        self,
        task: str,
        timeout_seconds: float | None,
        on_complete: Callable[[str | None, BaseException | None], None],
    ) -> None: ...

    def interrupt(self) -> None: ...


class ChildAgentRunnerFactory(Protocol):
    def __call__(self, request: DelegationRequest) -> ChildAgentRunner: ...


@dataclass(frozen=True)
class DelegationModelProfile:
    model: str
    reasoning_effort: str
    api_key_env: str
    api_key: str


@dataclass(frozen=True)
class DelegationConfig:
    python_executable: Path
    workspace: Path
    state_dir: Path
    smart_model: str
    smart_reasoning_effort: str
    smart_api_key_env: str
    smart_api_key: str
    fast_model: str
    fast_reasoning_effort: str
    fast_api_key_env: str
    fast_api_key: str
    frontier_model: str
    frontier_reasoning_effort: str
    frontier_api_key_env: str
    frontier_api_key: str
    github_repo: str
    github_trusted_actor: str | None
    role_file: Path
    harness_file: Path
    plugin_dir: Path
    enable_browser: bool
    command_secrets: Mapping[str, str]
    role: str
    local_condenser_max_events: int = 0
    local_condenser_max_tokens: int = 0
    local_condenser_target_events: int = 0
    root_state_dir: Path | None = None
    tree_id: str | None = None
    depth: int = 0
    deadline_epoch: float | None = None
    agent_name: str | None = None
    current_task_id: str | None = None

    def profile(self, tier: ModelTier) -> DelegationModelProfile:
        if tier == "smart":
            return DelegationModelProfile(
                self.smart_model,
                self.smart_reasoning_effort,
                self.smart_api_key_env,
                self.smart_api_key,
            )
        if tier == "fast":
            return DelegationModelProfile(
                self.fast_model,
                self.fast_reasoning_effort,
                self.fast_api_key_env,
                self.fast_api_key,
            )
        if tier == "frontier":
            return DelegationModelProfile(
                self.frontier_model,
                self.frontier_reasoning_effort,
                self.frontier_api_key_env,
                self.frontier_api_key,
            )
        raise ValueError(f"unknown delegation model tier: {tier}")

    def profiles(self) -> tuple[DelegationModelProfile, ...]:
        return tuple(self.profile(tier) for tier in ("smart", "fast", "frontier"))


_DELEGATION_CONFIG: DelegationConfig | None = None


def configure_delegation(config: DelegationConfig | None) -> None:
    """Hold process-launch secrets outside model-visible tool parameters."""

    global _DELEGATION_CONFIG
    _DELEGATION_CONFIG = config


def configured_child_runner_factory() -> ChildAgentRunnerFactory:
    if _DELEGATION_CONFIG is None:
        raise RuntimeError("subagent runtime is not configured")
    config = _DELEGATION_CONFIG
    return lambda request: OpenHandsChildProcess(config, request)


def render_child_prompt(request: DelegationRequest, task: str) -> str:
    assignment = task.strip()
    if request.search_mode is not None:
        assignment = f"Search mode: {request.search_mode}\n\n{assignment}"
    if not request.parent_context:
        return (
            "# Delegated task\n\n"
            "You are a fresh Senpai subagent. Perform only the assigned task "
            "and return a concise, evidence-linked report to the parent.\n\n"
            f"{assignment}\n"
        )
    context = [message.model_dump(mode="json") for message in request.parent_context]
    return (
        "# Delegated task with parent context\n\n"
        "The JSON below is the complete model-visible parent context at "
        "delegation time. Use it as evidence, perform only the assigned task, "
        "and return a concise, evidence-linked report.\n\n"
        "<parent_context_json>\n"
        f"{json.dumps(context, separators=(',', ':'))}\n"
        "</parent_context_json>\n\n"
        f"{assignment}\n"
    )


def run_child_process(
    argv: Sequence[str],
    *,
    input_text: str,
    env: Mapping[str, str],
    timeout_seconds: float | None,
    terminate_grace_seconds: float = 5,
    on_start: Callable[[subprocess.Popen[str]], None] | None = None,
    on_finish: Callable[[subprocess.Popen[str]], None] | None = None,
) -> str:
    process = subprocess.Popen(
        tuple(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
        start_new_session=True,
    )
    if on_start is not None:
        on_start(process)
    try:
        try:
            output, _ = process.communicate(
                input=input_text,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            terminate_process_group(
                process,
                grace_seconds=terminate_grace_seconds,
            )
            process.communicate()
            raise TimeoutError(
                f"subagent exceeded its {timeout_seconds:g}-second runtime"
            ) from error
    finally:
        if on_finish is not None:
            on_finish(process)
    if process.returncode != 0:
        tail = output[-8192:].strip()
        raise RuntimeError(f"subagent process exited {process.returncode}: {tail}")
    return output


class OpenHandsChildProcess:
    """One independently interruptible OpenHands subagent."""

    def __init__(self, config: DelegationConfig, request: DelegationRequest):
        self._config = config
        self._request = request
        self._conversation_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"senpai-child:{request.task_id}",
        )
        self._lock = threading.Lock()
        self._interrupted = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self.state_dir = config.state_dir / "children" / request.task_id
        self.output_path = self.state_dir.parent / f"{request.task_id}.log"

    @property
    def command(self) -> tuple[str, ...]:
        browser_flag = "--browser" if self._config.enable_browser else "--no-browser"
        profile = self._config.profile(self._request.model)
        return (
            str(self._config.python_executable),
            "-m",
            "senpai_agent.openhands_runner",
            "--child",
            "--max-turns",
            "1000",
            "--model",
            profile.model,
            "--api-key-env",
            profile.api_key_env,
            "--reasoning-effort",
            profile.reasoning_effort,
            "--agent",
            self._request.agent,
            "--workspace",
            str(self._config.workspace),
            "--state-dir",
            str(self.state_dir),
            "--conversation-id",
            str(self._conversation_id),
            "--role-file",
            str(self._config.role_file),
            "--harness-file",
            str(self._config.harness_file),
            "--plugin-dir",
            str(self._config.plugin_dir),
            browser_flag,
        )

    @property
    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        scrub_github_credentials(environment)
        for name in tuple(environment):
            if name.endswith("_API_KEY"):
                environment.pop(name)
        for name in (
            "SENPAI_OPENHANDS_AGENT",
            "SENPAI_OPENHANDS_CONVERSATION_ID",
        ):
            environment.pop(name, None)
        environment.update(self._config.command_secrets)
        profiles = self._config.profiles()
        environment.update(
            {profile.api_key_env: profile.api_key for profile in profiles}
        )
        selected = self._config.profile(self._request.model)
        environment.update(
            {
                "OPENHANDS_SUPPRESS_BANNER": "1",
                "SENPAI_ROLE": self._config.role,
                "SENPAI_OPENHANDS_API_KEY_ENV": selected.api_key_env,
                "SENPAI_OPENHANDS_SMART_MODEL": self._config.smart_model,
                "SENPAI_OPENHANDS_SMART_API_KEY_ENV": (
                    self._config.smart_api_key_env
                ),
                "SENPAI_OPENHANDS_FAST_MODEL": self._config.fast_model,
                "SENPAI_OPENHANDS_FAST_API_KEY_ENV": self._config.fast_api_key_env,
                "SENPAI_OPENHANDS_FRONTIER_MODEL": self._config.frontier_model,
                "SENPAI_OPENHANDS_FRONTIER_API_KEY_ENV": (
                    self._config.frontier_api_key_env
                ),
                "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": (
                    self._config.smart_reasoning_effort
                ),
                "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": (
                    self._config.fast_reasoning_effort
                ),
                "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": (
                    self._config.frontier_reasoning_effort
                ),
                "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS": str(
                    self._config.local_condenser_max_events
                ),
                "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS": str(
                    self._config.local_condenser_max_tokens
                ),
                "SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS": str(
                    self._config.local_condenser_target_events
                ),
                "SENPAI_PARENT_CONVERSATION_HISTORY_DIR": str(
                    self._config.state_dir
                    / uuid.UUID(self._request.parent_conversation_id).hex
                    / "events"
                ),
                "GH_REPO": self._config.github_repo,
                "SENPAI_DELEGATION_DEPTH": str(self._request.depth),
                "SENPAI_DELEGATION_TREE_ID": self._request.tree_id,
            }
        )
        if self._request.deadline_epoch is not None:
            remaining = self._request.deadline_epoch - time.time()
            if remaining <= 0:
                raise TimeoutError("the inherited delegation deadline has expired")
            environment["SENPAI_DELEGATION_DEADLINE_EPOCH"] = str(
                self._request.deadline_epoch
            )
            environment["SENPAI_OPENHANDS_TIMEOUT_SECONDS"] = str(remaining)
        if self._request.registry_path is not None:
            environment["SENPAI_DELEGATION_REGISTRY_PATH"] = str(
                self._request.registry_path
            )
            environment["SENPAI_DELEGATION_ROOT_STATE_DIR"] = str(
                self._request.registry_path.parent.parent
            )
            environment["SENPAI_DELEGATION_TASK_ID"] = self._request.task_id
        if self._request.event_db_path is not None:
            environment["SENPAI_DELEGATION_EVENT_DB_PATH"] = str(
                self._request.event_db_path
            )
        if self._config.github_trusted_actor is not None:
            environment["SENPAI_GITHUB_ACTOR"] = self._config.github_trusted_actor
        return environment

    def start(
        self,
        task: str,
        timeout_seconds: float | None,
        on_complete: Callable[[str | None, BaseException | None], None],
    ) -> None:
        if self._interrupted.is_set():
            raise InterruptedError("subagent was interrupted before startup")
        self.state_dir.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as input_stream,
            self.output_path.open("w", encoding="utf-8") as output_stream,
        ):
            input_stream.write(render_child_prompt(self._request, task))
            input_stream.seek(0)
            process = subprocess.Popen(
                self.command,
                stdin=input_stream,
                stdout=output_stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=self.environment,
                start_new_session=True,
            )
        try:
            with self._lock:
                self._process = process
            if self._request.registry_path is not None:
                process_group_id = os.getpgid(process.pid)
                process_start_time = psutil.Process(process.pid).create_time()
                registered = DelegationRegistry(
                    self._request.registry_path
                ).mark_running(
                    self._request.task_id,
                    process.pid,
                    self.state_dir,
                    process_group_id,
                    process_start_time,
                )
                if not registered:
                    raise InterruptedError(
                        "subagent launch was cancelled before PID confirmation"
                    )
            if self._interrupted.is_set():
                terminate_process_group(
                    process,
                    grace_seconds=1,
                    wait_full_grace=True,
                )
            threading.Thread(
                target=self._monitor,
                args=(process, timeout_seconds, on_complete),
                name=f"senpai-child-monitor-{self._request.task_id[:8]}",
                daemon=True,
            ).start()
        except BaseException:
            terminate_process_group(
                process,
                grace_seconds=1,
                wait_full_grace=True,
            )
            raise

    def _monitor(
        self,
        process: subprocess.Popen[str],
        timeout_seconds: float | None,
        on_complete: Callable[[str | None, BaseException | None], None],
    ) -> None:
        result: str | None = None
        failure: BaseException | None = None
        try:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                terminate_process_group(
                    process,
                    grace_seconds=5,
                    wait_full_grace=True,
                )
                process.wait()
                raise TimeoutError(
                    f"subagent exceeded its {timeout_seconds:g}-second runtime"
                ) from error
            output = self.output_path.read_text(encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(
                    f"subagent process exited {process.returncode}: "
                    f"{output[-8192:].strip()}"
                )
            result = self.parse_result(output)
        except BaseException as error:  # noqa: BLE001
            failure = error
        try:
            on_complete(result, failure)
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def run(self, task: str, timeout_seconds: float | None) -> str:
        completed = threading.Event()
        outcome: list[tuple[str | None, BaseException | None]] = []

        def finish(result: str | None, error: BaseException | None) -> None:
            outcome.append((result, error))
            completed.set()

        self.start(task, timeout_seconds, finish)
        completed.wait()
        result, error = outcome[0]
        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("subagent returned no result")
        return result

    def interrupt(self) -> None:
        self._interrupted.set()
        with self._lock:
            process = self._process
        if process is not None:
            terminate_process_group(
                process,
                grace_seconds=1,
                wait_full_grace=True,
            )

    @staticmethod
    def parse_result(output: str) -> str:
        for line in reversed(output.splitlines()):
            if not line.startswith("OPENHANDS_RESULT "):
                continue
            payload = json.loads(line.removeprefix("OPENHANDS_RESULT "))
            result = payload.get("result")
            if (
                payload.get("status") == "finished"
                and isinstance(result, str)
                and result.strip()
            ):
                return result.strip()
            raise RuntimeError("subagent returned no successful terminal result")
        raise RuntimeError("subagent emitted no terminal result record")


class DelegateAgentAction(Action):
    """Legacy action schema retained so persisted conversations can resume."""

    task: str = Field(min_length=1)
    agent: AgentKind = "general-purpose"
    model: ModelTier = "smart"
    background: bool = False
    include_context: bool = False
    search_mode: SearchMode | None = None


class DelegateAgentObservation(Observation):
    """Legacy observation schema retained for durable event deserialization."""

    task_id: str
    status: Literal["finished", "dispatched"]
    result: str | None = None

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        if self.status == "finished":
            return [
                TextContent(
                    text=f"Subagent task {self.task_id} finished.\n\n{self.result or ''}"
                )
            ]
        return [
            TextContent(
                text=(
                    f"Subagent task {self.task_id} is running in the background. "
                    "Its result or error will arrive as a durable local event."
                )
            )
        ]


class AgentTask(BaseModel):
    key: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Stable key within this batch; list position is used when omitted.",
    )
    task: str = Field(
        min_length=1,
        description="Self-contained assignment and requested evidence-linked report.",
    )
    agent: AgentKind = Field(
        default="general-purpose",
        description="Use a leaf specialization or general-purpose for mixed work.",
    )
    model: ModelTier = Field(
        default="smart",
        description="Fast for mechanical work, smart for synthesis, frontier for the hardest work.",
    )
    include_context: bool = Field(
        default=False,
        description="Copy the complete model-visible parent history into this child.",
    )
    search_mode: SearchMode | None = Field(
        default=None,
        description="Required only for search: general-web or research-publications.",
    )

    @model_validator(mode="after")
    def validate_search_mode(self) -> Self:
        if (self.agent == "search") != (self.search_mode is not None):
            raise ValueError("search_mode is required only when agent=search")
        return self


class AgentTaskState(BaseModel):
    task_id: str
    key: str | None = None
    status: TaskStatus
    agent: AgentKind
    model: ModelTier
    result: str | None = None
    error: str | None = None


class DelegationRegistry:
    """Small durable registry shared by every process in one Senpai role."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_key TEXT PRIMARY KEY,
                    tree_id TEXT NOT NULL,
                    specs_json TEXT NOT NULL,
                    task_ids_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    tree_id TEXT NOT NULL,
                    parent_conversation_id TEXT NOT NULL,
                    parent_task_id TEXT,
                    task_key TEXT,
                    task TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    model TEXT NOT NULL,
                    search_mode TEXT,
                    depth INTEGER NOT NULL,
                    deadline_epoch REAL NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    pid INTEGER,
                    process_group_id INTEGER,
                    process_start_time REAL,
                    state_dir TEXT,
                    collected_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tasks_tree_id ON tasks(tree_id);
                CREATE INDEX IF NOT EXISTS tasks_parent ON tasks(parent_conversation_id);
                """
            )
            columns = {
                row[1] for row in database.execute("PRAGMA table_info(tasks)")
            }
            if "parent_task_id" not in columns:
                database.execute("ALTER TABLE tasks ADD COLUMN parent_task_id TEXT")
            if "collected_at" not in columns:
                database.execute("ALTER TABLE tasks ADD COLUMN collected_at REAL")
            if "process_group_id" not in columns:
                database.execute("ALTER TABLE tasks ADD COLUMN process_group_id INTEGER")
            if "process_start_time" not in columns:
                database.execute("ALTER TABLE tasks ADD COLUMN process_start_time REAL")

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.path, timeout=30)
        database.row_factory = sqlite3.Row
        return database

    @contextmanager
    def lifecycle(self):
        """Serialize short process admission and termination transitions."""

        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def reserve(
        self,
        *,
        operation_key: str,
        tree_id: str,
        parent_conversation_id: str,
        parent_task_id: str | None,
        depth: int,
        specs: Sequence[AgentTask],
        deadlines: Sequence[float],
    ) -> tuple[list[sqlite3.Row], bool]:
        specs_json = json.dumps(
            [spec.model_dump(mode="json") for spec in specs],
            sort_keys=True,
            separators=(",", ":"),
        )
        keys = [spec.key or str(index) for index, spec in enumerate(specs)]
        if len(keys) != len(set(keys)):
            raise ValueError("task keys must be unique within a spawn batch")
        task_ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"senpai-task:{parent_conversation_id}:{operation_key}:{key}",
                )
            )
            for key in keys
        ]
        now = time.time()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            existing = database.execute(
                "SELECT tree_id, specs_json, task_ids_json FROM operations "
                "WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if existing is not None:
                if existing["specs_json"] != specs_json:
                    raise ValueError(
                        "batch_key was already used with different task specifications"
                    )
                if existing["tree_id"] != tree_id:
                    raise RuntimeError("replayed delegation operation changed trees")
                existing_ids = json.loads(existing["task_ids_json"])
                return self._rows(database, existing_ids), False

            if parent_task_id is not None:
                parent = database.execute(
                    "SELECT tree_id, depth, status FROM tasks WHERE task_id = ?",
                    (parent_task_id,),
                ).fetchone()
                if (
                    parent is None
                    or parent["status"] not in {"queued", "starting", "running"}
                    or parent["tree_id"] != tree_id
                    or parent["depth"] != depth - 1
                ):
                    raise RuntimeError(
                        "cannot spawn from an inactive or mismatched parent task"
                    )

            tree_count = database.execute(
                "SELECT COUNT(*) FROM tasks WHERE tree_id = ?",
                (tree_id,),
            ).fetchone()[0]
            active_count = database.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN "
                "('queued', 'starting', 'running')"
            ).fetchone()[0]
            if tree_count + len(specs) > MAX_TREE_AGENTS:
                raise RuntimeError(
                    f"delegation tree capacity is {MAX_TREE_AGENTS} total tasks"
                )
            if active_count + len(specs) > MAX_PARALLEL_AGENTS:
                raise RuntimeError(
                    f"subagent capacity is full ({MAX_PARALLEL_AGENTS} active)"
                )

            database.execute(
                "INSERT INTO operations VALUES (?, ?, ?, ?)",
                (operation_key, tree_id, specs_json, json.dumps(task_ids)),
            )
            for task_id, key, spec, deadline in zip(
                task_ids, keys, specs, deadlines, strict=True
            ):
                database.execute(
                    """
                    INSERT INTO tasks (
                        task_id, tree_id, parent_conversation_id, parent_task_id,
                        task_key, task,
                        agent, model, search_mode, depth, deadline_epoch, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                    """,
                    (
                        task_id,
                        tree_id,
                        parent_conversation_id,
                        parent_task_id,
                        spec.key,
                        spec.task,
                        spec.agent,
                        spec.model,
                        spec.search_mode,
                        depth,
                        deadline,
                        now,
                        now,
                    ),
                )
            return self._rows(database, task_ids), True

    @staticmethod
    def _rows(
        database: sqlite3.Connection,
        task_ids: Sequence[str],
    ) -> list[sqlite3.Row]:
        return [
            database.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            for task_id in task_ids
        ]

    def rows(
        self,
        task_ids: Sequence[str] | None = None,
        *,
        parent_conversation_id: str | None = None,
    ) -> list[sqlite3.Row]:
        with self._connect() as database:
            if task_ids is not None:
                rows = self._rows(database, task_ids)
                if any(row is None for row in rows):
                    raise ValueError("one or more subagent task IDs are unknown")
                return rows
            if parent_conversation_id is None:
                return list(database.execute("SELECT * FROM tasks ORDER BY created_at"))
            return list(
                database.execute(
                    "SELECT * FROM tasks WHERE parent_conversation_id = ? "
                    "AND (status IN ('queued', 'starting', 'running') "
                    "OR collected_at IS NULL) "
                    "ORDER BY CASE WHEN status IN "
                    "('queued', 'starting', 'running') "
                    "THEN 0 ELSE 1 END, created_at DESC LIMIT ?",
                    (parent_conversation_id, MAX_TREE_AGENTS),
                )
            )

    def mark_running(
        self,
        task_id: str,
        pid: int | None,
        state_dir: Path | None = None,
        process_group_id: int | None = None,
        process_start_time: float | None = None,
    ) -> bool:
        with self._connect() as database:
            cursor = database.execute(
                "UPDATE tasks SET status = 'running', pid = COALESCE(?, pid), "
                "process_group_id = COALESCE(?, process_group_id), "
                "process_start_time = COALESCE(?, process_start_time), "
                "state_dir = COALESCE(?, state_dir), updated_at = ? "
                "WHERE task_id = ? AND status IN ('queued', 'starting')",
                (
                    pid,
                    process_group_id,
                    process_start_time,
                    str(state_dir) if state_dir else None,
                    time.time(),
                    task_id,
                ),
            )
            if cursor.rowcount:
                return True
            cursor = database.execute(
                "UPDATE tasks SET pid = COALESCE(?, pid), "
                "process_group_id = COALESCE(?, process_group_id), "
                "process_start_time = COALESCE(?, process_start_time), "
                "state_dir = COALESCE(?, state_dir), updated_at = ? "
                "WHERE task_id = ? AND status = 'running'",
                (
                    pid,
                    process_group_id,
                    process_start_time,
                    str(state_dir) if state_dir else None,
                    time.time(),
                    task_id,
                ),
            )
            return bool(cursor.rowcount)

    def claim_launch(self, task_id: str) -> bool:
        with self._connect() as database:
            cursor = database.execute(
                "UPDATE tasks SET status = 'starting', updated_at = ? "
                "WHERE task_id = ? AND status = 'queued'",
                (time.time(), task_id),
            )
            return bool(cursor.rowcount)

    def finish(
        self,
        task_id: str,
        *,
        result: str | None = None,
        error: str | None = None,
        status: Literal["finished", "failed", "cancelled"] | None = None,
    ) -> bool:
        terminal_status = status or ("finished" if error is None else "failed")
        with self._connect() as database:
            cursor = database.execute(
                "UPDATE tasks SET status = ?, result = ?, error = ?, updated_at = ? "
                "WHERE task_id = ? AND status IN ('queued', 'starting', 'running')",
                (terminal_status, result, error, time.time(), task_id),
            )
            return bool(cursor.rowcount)

    def pending_for_parent(self, parent_conversation_id: str) -> list[sqlite3.Row]:
        with self._connect() as database:
            return list(
                database.execute(
                    "SELECT * FROM tasks WHERE parent_conversation_id = ? "
                    "AND status IN ('queued', 'starting', 'running')",
                    (parent_conversation_id,),
                )
            )

    def uncollected_for_parent(
        self,
        parent_conversation_id: str,
    ) -> list[sqlite3.Row]:
        with self._connect() as database:
            return list(
                database.execute(
                    "SELECT * FROM tasks WHERE parent_conversation_id = ? "
                    "AND status != 'cancelled' AND collected_at IS NULL",
                    (parent_conversation_id,),
                )
            )

    def mark_collected(self, task_ids: Sequence[str]) -> None:
        if not task_ids:
            return
        placeholders = ",".join("?" for _ in task_ids)
        with self._connect() as database:
            database.execute(
                f"UPDATE tasks SET collected_at = ?, updated_at = ? "
                f"WHERE task_id IN ({placeholders})",
                (time.time(), time.time(), *task_ids),
            )

    def active_rows(self) -> list[sqlite3.Row]:
        with self._connect() as database:
            return list(
                database.execute(
                    "SELECT * FROM tasks WHERE status IN "
                    "('queued', 'starting', 'running') ORDER BY depth DESC"
                )
            )

    def cancel_tree(
        self,
        task_ids: Sequence[str],
        *,
        error: str,
        root_status: Literal["cancelled", "failed"] = "cancelled",
        collect_roots: bool = False,
    ) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in task_ids)
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            rows = list(
                database.execute(
                    f"""
                    WITH RECURSIVE subtree(task_id) AS (
                        SELECT task_id FROM tasks WHERE task_id IN ({placeholders})
                        UNION ALL
                        SELECT tasks.task_id FROM tasks
                        JOIN subtree ON tasks.parent_task_id = subtree.task_id
                    )
                    SELECT * FROM tasks WHERE task_id IN (SELECT task_id FROM subtree)
                    ORDER BY depth DESC
                    """,
                    tuple(task_ids),
                )
            )
            active_ids = [
                row["task_id"]
                for row in rows
                if row["status"] in {"queued", "starting", "running"}
            ]
            if active_ids:
                active_placeholders = ",".join("?" for _ in active_ids)
                database.execute(
                    f"UPDATE tasks SET status = 'cancelled', error = ?, "
                    f"updated_at = ? WHERE task_id IN ({active_placeholders})",
                    (error, time.time(), *active_ids),
                )
            if root_status == "failed":
                active_roots = [
                    row["task_id"]
                    for row in rows
                    if row["task_id"] in task_ids
                    and row["status"] in {"queued", "starting", "running"}
                ]
                if active_roots:
                    root_placeholders = ",".join("?" for _ in active_roots)
                    database.execute(
                        f"UPDATE tasks SET status = 'failed' "
                        f"WHERE task_id IN ({root_placeholders})",
                        active_roots,
                    )
            collected_ids = [
                row["task_id"]
                for row in rows
                if collect_roots or row["task_id"] not in task_ids
            ]
            if collected_ids:
                all_placeholders = ",".join("?" for _ in collected_ids)
                database.execute(
                    f"UPDATE tasks SET collected_at = COALESCE(collected_at, ?), "
                    f"updated_at = ? WHERE task_id IN ({all_placeholders})",
                    (time.time(), time.time(), *collected_ids),
                )
            return rows


_LOCAL_RUNNERS: dict[str, ChildAgentRunner] = {}
_LOCAL_RUNNERS_LOCK = threading.Lock()


def _pid_matches_task(row: sqlite3.Row) -> bool:
    if (
        row["pid"] is None
        or row["process_group_id"] is None
        or row["process_start_time"] is None
    ):
        return False
    try:
        process = psutil.Process(row["pid"])
        command = " ".join(process.cmdline())
        process_start_time = process.create_time()
        process_group_id = os.getpgid(row["pid"])
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        return False
    return (
        process_group_id == row["process_group_id"]
        and process_start_time == row["process_start_time"]
        and "senpai_agent.openhands_runner" in command
        and bool(row["state_dir"])
        and row["state_dir"] in command
    )


def _terminate_recovered_task(row: sqlite3.Row, grace_seconds: float = 1) -> None:
    if not _pid_matches_task(row):
        return
    try:
        os.killpg(row["process_group_id"], signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(grace_seconds)
    try:
        os.killpg(row["process_group_id"], signal.SIGKILL)
    except ProcessLookupError:
        pass


def _row_state(row: sqlite3.Row) -> AgentTaskState:
    return AgentTaskState(
        task_id=row["task_id"],
        key=row["task_key"],
        status="queued" if row["status"] == "starting" else row["status"],
        agent=row["agent"],
        model=row["model"],
        result=row["result"],
        error=row["error"],
    )


def _task_event(row: sqlite3.Row) -> AdvisorEvent:
    successful = row["status"] == "finished"
    payload = {
        "task_id": row["task_id"],
        "parent_conversation_id": row["parent_conversation_id"],
        "task": row["task"],
        **(
            {"result": row["result"]}
            if successful
            else {"error": row["error"] or f"subagent {row['status']}"}
        ),
    }
    return AdvisorEvent(
        kind="agent_result" if successful else "agent_error",
        dedupe_key=f"agent_result:{row['task_id']}",
        payload=payload,
    )


def record_delegated_task_result(
    task_id: str,
    *,
    result: str | None = None,
    error: str | None = None,
    env: Mapping[str, str] = os.environ,
) -> bool:
    registry_value = env.get("SENPAI_DELEGATION_REGISTRY_PATH")
    if not registry_value:
        return False
    registry = DelegationRegistry(Path(registry_value))
    with registry.lifecycle():
        changed = registry.finish(task_id, result=result, error=error)
        event_path = env.get("SENPAI_DELEGATION_EVENT_DB_PATH")
        row = registry.rows([task_id])[0]
        if (
            event_path
            and row["depth"] == 1
            and row["status"] in TERMINAL_TASK_STATUSES
        ):
            with AdvisorEventStore(Path(event_path)) as sink:
                event = _task_event(row)
                sink.enqueue(event)
                if registry.rows([task_id])[0]["collected_at"] is not None:
                    sink.acknowledge(event.dedupe_key)
    return changed


def cancel_pending_descendants(
    registry_path: Path,
    parent_conversation_id: str,
) -> list[str]:
    registry = DelegationRegistry(registry_path)
    with registry.lifecycle():
        direct = registry.uncollected_for_parent(parent_conversation_id)
        if not direct:
            return []
        rows = registry.cancel_tree(
            [row["task_id"] for row in direct],
            error="Cancelled because the parent child agent exited without awaiting",
            collect_roots=True,
        )
        for row in rows:
            if row["status"] in TERMINAL_TASK_STATUSES:
                continue
            with _LOCAL_RUNNERS_LOCK:
                runner = _LOCAL_RUNNERS.get(row["task_id"])
            if runner is not None:
                runner.interrupt()
            elif row["pid"]:
                _terminate_recovered_task(row)
        return [row["task_id"] for row in rows]


class _DelegationManager:
    def __init__(
        self,
        config: DelegationConfig,
        child_runner_factory: ChildAgentRunnerFactory,
        event_sink: AdvisorEventSink | None,
        event_db_path: Path | None,
    ):
        root_state = config.root_state_dir or config.state_dir
        self.registry = DelegationRegistry(root_state / "delegation" / "tasks.sqlite3")
        self.config = config
        self.child_runner_factory = child_runner_factory
        self.event_sink = event_sink
        self.event_db_path = event_db_path

    def _validate_spawn(self, tasks: Sequence[AgentTask]) -> None:
        if not tasks or len(tasks) > MAX_SPAWN_BATCH:
            raise ValueError(f"spawn_agents requires 1 to {MAX_SPAWN_BATCH} tasks")
        if self.config.depth >= MAX_DELEGATION_DEPTH:
            raise ValueError(f"maximum delegation depth is {MAX_DELEGATION_DEPTH}")
        if self.config.depth == 1:
            if self.config.current_task_id is None:
                raise ValueError("nested delegation requires its current parent task ID")
            if self.config.agent_name != "general-purpose":
                raise ValueError(
                    "explore, search, and bash-runner agents are delegation leaves"
                )
            if any(task.agent == "general-purpose" for task in tasks):
                raise ValueError("depth-2 helpers must be leaf agents")

    def spawn(
        self,
        action: SpawnAgentsAction,
        conversation: LocalConversation,
    ) -> list[AgentTaskState]:
        self._validate_spawn(action.tasks)
        with self.registry.lifecycle():
            return self._spawn_locked(action, conversation)

    def _spawn_locked(
        self,
        action: SpawnAgentsAction,
        conversation: LocalConversation,
    ) -> list[AgentTaskState]:
        parent_id = str(conversation.id)
        operation_key = f"{parent_id}:{action.batch_key}"
        tree_id = self.config.tree_id or str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"senpai-tree:{operation_key}")
        )
        now = time.time()
        inherited_deadline = self.config.deadline_epoch or float("inf")
        deadlines = [
            min(inherited_deadline, now + MODEL_TIER_TIMEOUT_SECONDS[task.model])
            for task in action.tasks
        ]
        if any(deadline <= now for deadline in deadlines):
            raise TimeoutError("the inherited delegation deadline has expired")
        self._reconcile(self.registry.active_rows())
        rows, created = self.registry.reserve(
            operation_key=operation_key,
            tree_id=tree_id,
            parent_conversation_id=parent_id,
            parent_task_id=self.config.current_task_id,
            depth=self.config.depth + 1,
            specs=action.tasks,
            deadlines=deadlines,
        )
        self._reconcile(rows)
        rows = self.registry.rows([row["task_id"] for row in rows])
        for row, task in zip(rows, action.tasks, strict=True):
            if not created:
                continue
            if row["status"] != "queued":
                continue
            if not self.registry.claim_launch(row["task_id"]):
                continue
            context = (
                _model_visible_context(conversation) if task.include_context else ()
            )
            request = DelegationRequest(
                task_id=row["task_id"],
                parent_conversation_id=parent_id,
                parent_context=context,
                agent=task.agent,
                model=task.model,
                search_mode=task.search_mode,
                tree_id=tree_id,
                depth=self.config.depth + 1,
                deadline_epoch=row["deadline_epoch"],
                registry_path=self.registry.path,
                event_db_path=self.event_db_path,
                parent_task_id=self.config.current_task_id,
            )
            runner = self.child_runner_factory(request)
            with _LOCAL_RUNNERS_LOCK:
                _LOCAL_RUNNERS[request.task_id] = runner
            timeout = max(0.0, (request.deadline_epoch or time.time()) - time.time())
            try:
                runner.start(
                    task.task,
                    timeout,
                    lambda result, error, request=request: self._complete(
                        request,
                        result,
                        error,
                    ),
                )
                if self.registry.rows([request.task_id])[0]["status"] == "starting":
                    self.registry.mark_running(request.task_id, None)
            except BaseException as error:  # noqa: BLE001
                with _LOCAL_RUNNERS_LOCK:
                    _LOCAL_RUNNERS.pop(request.task_id, None)
                changed = self.registry.finish(
                    request.task_id,
                    error=f"{type(error).__name__}: {error}",
                )
                if changed and request.depth == 1:
                    self._enqueue(
                        _task_event(self.registry.rows([request.task_id])[0])
                    )
        return [_row_state(row) for row in self.registry.rows([r["task_id"] for r in rows])]

    def _complete(
        self,
        request: DelegationRequest,
        result: str | None,
        error: BaseException | None,
    ) -> None:
        with self.registry.lifecycle():
            self._complete_locked(request, result, error)

    def _complete_locked(
        self,
        request: DelegationRequest,
        result: str | None,
        error: BaseException | None,
    ) -> None:
        if error is None:
            self.registry.finish(request.task_id, result=result)
        else:
            self.registry.finish(
                request.task_id,
                error=f"{type(error).__name__}: {error}",
            )
            targets = self.registry.cancel_tree(
                [request.task_id],
                error=f"Parent subagent failed: {type(error).__name__}: {error}",
            )
            self._signal_active(targets)
        with _LOCAL_RUNNERS_LOCK:
            _LOCAL_RUNNERS.pop(request.task_id, None)
        if request.depth == 1:
            self._enqueue(_task_event(self.registry.rows([request.task_id])[0]))

    def _enqueue(self, event: AdvisorEvent) -> None:
        if self.event_sink is not None:
            self.event_sink.enqueue(event)
        elif self.event_db_path is not None:
            with AdvisorEventStore(self.event_db_path) as sink:
                sink.enqueue(event)
                task_id = str(event.payload["task_id"])
                if self.registry.rows([task_id])[0]["collected_at"] is not None:
                    sink.acknowledge(event.dedupe_key)

    def _reconcile(self, rows: Sequence[sqlite3.Row]) -> None:
        now = time.time()
        for row in rows:
            pid = row["pid"]
            if row["deadline_epoch"] <= now:
                self._reconcile_failure(
                    row,
                    "TimeoutError: inherited subagent deadline expired",
                )
                continue
            if row["status"] == "queued" or pid is None:
                if now - row["updated_at"] > 10:
                    self._reconcile_failure(
                        row,
                        "InterruptedError: subagent startup did not complete",
                    )
                continue
            if not _pid_matches_task(row):
                self._reconcile_failure(
                    row,
                    "InterruptedError: prior subagent process is no longer running",
                )

    def _reconcile_failure(self, row: sqlite3.Row, error: str) -> None:
        targets = self.registry.cancel_tree(
            [row["task_id"]],
            error=error,
            root_status="failed",
        )
        self._signal_active(targets)
        if row["depth"] == 1:
            self._enqueue(_task_event(self.registry.rows([row["task_id"]])[0]))

    @staticmethod
    def _signal_active(targets: Sequence[sqlite3.Row]) -> None:
        for target in targets:
            if target["status"] in TERMINAL_TASK_STATUSES:
                continue
            with _LOCAL_RUNNERS_LOCK:
                runner = _LOCAL_RUNNERS.get(target["task_id"])
            if runner is not None:
                runner.interrupt()
            elif target["pid"]:
                _terminate_recovered_task(target)

    def states(
        self,
        task_ids: Sequence[str] | None,
        conversation: LocalConversation,
    ) -> list[AgentTaskState]:
        with self.registry.lifecycle():
            return self._states_locked(task_ids, conversation)

    def _states_locked(
        self,
        task_ids: Sequence[str] | None,
        conversation: LocalConversation,
    ) -> list[AgentTaskState]:
        rows = self.registry.rows(
            task_ids,
            parent_conversation_id=(str(conversation.id) if task_ids is None else None),
        )
        if any(row["parent_conversation_id"] != str(conversation.id) for row in rows):
            raise ValueError("a caller may inspect only its own subagent tasks")
        self._reconcile(rows)
        return [
            _row_state(row)
            for row in self.registry.rows(
                [row["task_id"] for row in rows],
            )
        ]

    def cancel(
        self,
        task_ids: Sequence[str],
        conversation: LocalConversation,
    ) -> list[AgentTaskState]:
        with self.registry.lifecycle():
            return self._cancel_locked(task_ids, conversation)

    def _cancel_locked(
        self,
        task_ids: Sequence[str],
        conversation: LocalConversation,
    ) -> list[AgentTaskState]:
        rows = self.registry.rows(task_ids)
        for row in rows:
            if row["parent_conversation_id"] != str(conversation.id):
                raise ValueError("a caller may cancel only its own subagent tasks")
        targets = self.registry.cancel_tree(
            task_ids,
            error="Cancelled by parent agent",
            collect_roots=True,
        )
        for row in targets:
            if row["status"] in TERMINAL_TASK_STATUSES:
                continue
            with _LOCAL_RUNNERS_LOCK:
                runner = _LOCAL_RUNNERS.get(row["task_id"])
            if runner is not None:
                runner.interrupt()
            elif row["pid"]:
                _terminate_recovered_task(row)
            if row["depth"] == 1:
                self._enqueue(_task_event(self.registry.rows([row["task_id"]])[0]))
        if self.event_db_path is not None:
            with AdvisorEventStore(self.event_db_path) as store:
                for row in targets:
                    if row["depth"] == 1:
                        store.acknowledge(f"agent_result:{row['task_id']}")
        return [
            _row_state(row) for row in self.registry.rows(task_ids)
        ]

class SpawnAgentsAction(Action):
    batch_key: str = Field(
        min_length=1,
        max_length=128,
        description="Stable idempotency key; changed specs on replay are rejected.",
    )
    tasks: list[AgentTask] = Field(
        min_length=1,
        max_length=MAX_SPAWN_BATCH,
        description="One to eight tasks started without waiting for results.",
    )


class SpawnAgentsObservation(Observation):
    tasks: list[AgentTaskState]

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [TextContent(text=json.dumps(self.model_dump(mode="json"), sort_keys=True))]


class AwaitAgentsAction(Action):
    task_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_TREE_AGENTS,
        description="Unique task IDs returned by spawn_agents.",
    )
    join: JoinMode = Field(
        default="all",
        description="Return after all, the first, or a quorum become terminal.",
    )
    quorum: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TREE_AGENTS,
        description="Required terminal count when join=quorum.",
    )
    timeout_seconds: float = Field(
        gt=0,
        le=MAX_AWAIT_SECONDS,
        description="Bounded wait; timeout leaves unfinished tasks running.",
    )

    @model_validator(mode="after")
    def validate_quorum(self) -> Self:
        _require_unique_task_ids(self.task_ids)
        if self.join == "quorum":
            if self.quorum is None or self.quorum > len(self.task_ids):
                raise ValueError("quorum must be between 1 and the number of tasks")
        elif self.quorum is not None:
            raise ValueError("quorum is valid only when join=quorum")
        return self


class AwaitAgentsObservation(Observation):
    join: JoinMode
    satisfied: bool
    timed_out: bool
    tasks: list[AgentTaskState]

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        payload = self.model_dump(mode="json")
        return [TextContent(text=json.dumps(payload, sort_keys=True))]


class AgentStatusAction(Action):
    task_ids: list[str] | None = Field(
        default=None,
        max_length=MAX_TREE_AGENTS,
        description="Unique IDs, or omit for up to eight active/uncollected direct tasks.",
    )

    @model_validator(mode="after")
    def validate_task_ids(self) -> Self:
        if self.task_ids is not None:
            _require_unique_task_ids(self.task_ids)
        return self


class AgentStatusObservation(Observation):
    tasks: list[AgentTaskState]

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [TextContent(text=json.dumps(self.model_dump(mode="json"), sort_keys=True))]


class CancelAgentsAction(Action):
    task_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_TREE_AGENTS,
        description="Unique direct task IDs; descendants are cancelled recursively.",
    )

    @model_validator(mode="after")
    def validate_task_ids(self) -> Self:
        _require_unique_task_ids(self.task_ids)
        return self


class CancelAgentsObservation(Observation):
    tasks: list[AgentTaskState]

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [TextContent(text=json.dumps(self.model_dump(mode="json"), sort_keys=True))]


def _require_unique_task_ids(task_ids: Sequence[str]) -> None:
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_ids must not contain duplicates")


def _configured_manager(
    child_runner_factory: ChildAgentRunnerFactory | None,
    event_sink: AdvisorEventSink | None,
    event_db_path: str | Path | None,
) -> _DelegationManager:
    if _DELEGATION_CONFIG is None:
        raise RuntimeError("subagent runtime is not configured")
    return _DelegationManager(
        _DELEGATION_CONFIG,
        child_runner_factory or configured_child_runner_factory(),
        event_sink,
        Path(event_db_path) if event_db_path is not None else None,
    )


class _SpawnAgentsExecutor(ToolExecutor[SpawnAgentsAction, SpawnAgentsObservation]):
    def __init__(self, manager: _DelegationManager):
        self.manager = manager

    def __call__(self, action, conversation=None) -> SpawnAgentsObservation:
        if conversation is None:
            raise ValueError("spawn_agents requires its parent conversation")
        return SpawnAgentsObservation(tasks=self.manager.spawn(action, conversation))


class _AwaitAgentsExecutor(ToolExecutor[AwaitAgentsAction, AwaitAgentsObservation]):
    def __init__(self, manager: _DelegationManager):
        self.manager = manager
        self._interrupted = threading.Event()

    def __call__(self, action, conversation=None) -> AwaitAgentsObservation:
        if conversation is None:
            raise ValueError("await_agents requires its parent conversation")
        self._interrupted.clear()
        inherited = self.manager.config.deadline_epoch or float("inf")
        deadline = min(time.time() + action.timeout_seconds, inherited)
        while True:
            tasks = self.manager.states(action.task_ids, conversation)
            terminal = sum(task.status in TERMINAL_TASK_STATUSES for task in tasks)
            required = {
                "all": len(tasks),
                "first": 1,
                "quorum": action.quorum or len(tasks),
            }[action.join]
            if terminal >= required:
                self._acknowledge(tasks)
                return AwaitAgentsObservation(
                    join=action.join,
                    satisfied=True,
                    timed_out=False,
                    tasks=tasks,
                )
            remaining = deadline - time.time()
            if remaining <= 0 or self._interrupted.wait(min(0.1, remaining)):
                return AwaitAgentsObservation(
                    join=action.join,
                    satisfied=False,
                    timed_out=True,
                    tasks=tasks,
                )

    def _acknowledge(self, tasks: Sequence[AgentTaskState]) -> None:
        terminal = [
            task.task_id for task in tasks if task.status in TERMINAL_TASK_STATUSES
        ]
        with self.manager.registry.lifecycle():
            self.manager.registry.mark_collected(terminal)
            if self.manager.event_db_path is not None:
                with AdvisorEventStore(self.manager.event_db_path) as store:
                    for task_id in terminal:
                        store.acknowledge(f"agent_result:{task_id}")

    def interrupt(self) -> None:
        self._interrupted.set()


class _AgentStatusExecutor(ToolExecutor[AgentStatusAction, AgentStatusObservation]):
    def __init__(self, manager: _DelegationManager):
        self.manager = manager

    def __call__(self, action, conversation=None) -> AgentStatusObservation:
        if conversation is None:
            raise ValueError("agent_status requires its parent conversation")
        return AgentStatusObservation(
            tasks=self.manager.states(action.task_ids, conversation)
        )


class _CancelAgentsExecutor(ToolExecutor[CancelAgentsAction, CancelAgentsObservation]):
    def __init__(self, manager: _DelegationManager):
        self.manager = manager

    def __call__(self, action, conversation=None) -> CancelAgentsObservation:
        if conversation is None:
            raise ValueError("cancel_agents requires its parent conversation")
        return CancelAgentsObservation(
            tasks=self.manager.cancel(action.task_ids, conversation)
        )


_DELEGATE_AGENT_DEPRECATION = (
    "delegate_agent is deprecated and cannot launch an agent. Use spawn_agents "
    "with a stable batch_key, then pass its task IDs to await_agents."
)


class _DeprecatedDelegateAgentExecutor(
    ToolExecutor[DelegateAgentAction, DelegateAgentObservation]
):
    def __call__(
        self,
        action: DelegateAgentAction,  # noqa: ARG002
        conversation: LocalConversation | None = None,  # noqa: ARG002
    ) -> DelegateAgentObservation:
        return DelegateAgentObservation(
            task_id="deprecated",
            status="finished",
            result=_DELEGATE_AGENT_DEPRECATION,
        )


class DelegateAgentTool(
    ToolDefinition[DelegateAgentAction, DelegateAgentObservation]
):
    """Non-launching compatibility tool for pre-lifecycle conversations."""

    name = "delegate_agent"

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,  # noqa: ARG003
        *,
        event_db_path: str | Path | None = None,  # noqa: ARG003
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Deprecated compatibility tool. It never launches an agent; "
                    "use spawn_agents and await_agents instead."
                ),
                action_type=DelegateAgentAction,
                observation_type=DelegateAgentObservation,
                annotations=ToolAnnotations(
                    title="Deprecated agent delegation",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_DeprecatedDelegateAgentExecutor(),
            )
        ]


class _DelegationTool(ToolDefinition):
    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)

    @classmethod
    def _manager(
        cls,
        child_runner_factory,
        event_sink,
        event_db_path,
    ) -> _DelegationManager:
        return _configured_manager(child_runner_factory, event_sink, event_db_path)


class SpawnAgentsTool(_DelegationTool[SpawnAgentsAction, SpawnAgentsObservation]):
    name = "spawn_agents"

    @classmethod
    def create(
        cls,
        conv_state=None,
        child_runner_factory=None,
        event_sink=None,
        *,
        event_db_path=None,
    ) -> Sequence[Self]:
        manager = cls._manager(child_runner_factory, event_sink, event_db_path)
        return [
            cls(
                description="Start one bounded batch of subagents and return task IDs immediately.",
                action_type=SpawnAgentsAction,
                observation_type=SpawnAgentsObservation,
                annotations=ToolAnnotations(
                    title="Spawn agents",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_SpawnAgentsExecutor(manager),
            )
        ]


class AwaitAgentsTool(_DelegationTool[AwaitAgentsAction, AwaitAgentsObservation]):
    name = "await_agents"

    @classmethod
    def create(cls, conv_state=None, *, event_db_path=None) -> Sequence[Self]:
        manager = cls._manager(None, None, event_db_path)
        return [
            cls(
                description="Wait for all, the first, or a quorum of spawned agents.",
                action_type=AwaitAgentsAction,
                observation_type=AwaitAgentsObservation,
                annotations=ToolAnnotations(title="Await agents", readOnlyHint=False),
                executor=_AwaitAgentsExecutor(manager),
            )
        ]


class AgentStatusTool(_DelegationTool[AgentStatusAction, AgentStatusObservation]):
    name = "agent_status"

    @classmethod
    def create(cls, conv_state=None, *, event_db_path=None) -> Sequence[Self]:
        manager = cls._manager(None, None, event_db_path)
        return [
            cls(
                description="Inspect durable subagent task status without waiting.",
                action_type=AgentStatusAction,
                observation_type=AgentStatusObservation,
                annotations=ToolAnnotations(title="Agent status", readOnlyHint=False),
                executor=_AgentStatusExecutor(manager),
            )
        ]


class CancelAgentsTool(_DelegationTool[CancelAgentsAction, CancelAgentsObservation]):
    name = "cancel_agents"

    @classmethod
    def create(cls, conv_state=None, *, event_db_path=None) -> Sequence[Self]:
        manager = cls._manager(None, None, event_db_path)
        return [
            cls(
                description="Cancel selected subagent tasks without affecting siblings.",
                action_type=CancelAgentsAction,
                observation_type=CancelAgentsObservation,
                annotations=ToolAnnotations(
                    title="Cancel agents",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                ),
                executor=_CancelAgentsExecutor(manager),
            )
        ]


def _model_visible_context(conversation: LocalConversation) -> tuple[Message, ...]:
    events = list(conversation.state.view.events)
    while events and isinstance(events[-1], ActionEvent):
        events.pop()
    return tuple(
        message.model_copy(deep=True)
        for message in LLMConvertibleEvent.events_to_messages(events)
    )
