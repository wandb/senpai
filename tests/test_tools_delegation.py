import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.context.view import View
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent

from senpai_agent.delegation import (
    AgentStatusAction,
    AgentStatusTool,
    AgentTask,
    AwaitAgentsAction,
    AwaitAgentsTool,
    CancelAgentsAction,
    CancelAgentsTool,
    DelegationConfig,
    DelegationRegistry,
    DelegationRequest,
    LeafAgentTask,
    LeafSpawnAgentsAction,
    MODEL_TIER_TIMEOUT_SECONDS,
    OpenHandsChildProcess,
    SpawnAgentsAction,
    SpawnAgentsTool,
    cancel_pending_descendants,
    configure_delegation,
    reconcile_delegated_tasks,
)
from senpai_agent.local_events import LocalEventStore


def test_model_tier_runtime_limits():
    assert MODEL_TIER_TIMEOUT_SECONDS == {
        "fast": 1200,
        "smart": 3600,
        "frontier": 7200,
    }


def test_explicit_search_task_forms_resolve_to_internal_search_modes():
    web = AgentTask(task="Find current documentation", agent="search_general_web")
    papers = LeafAgentTask(
        task="Find primary papers",
        agent="search_research_publications",
    )

    assert web.resolved_agent() == ("search", "general-web")
    assert papers.resolved_agent() == ("search", "research-publications")
    assert "search_mode" not in web.model_dump()


def test_persisted_search_task_restores_without_exposing_the_old_schema():
    restored = AgentTask.model_validate(
        {
            "task": "Find primary papers",
            "agent": "search",
            "search_mode": "research-publications",
        }
    )

    assert restored.agent == "search_research_publications"
    task_schema = AgentTask.model_json_schema()["properties"]
    assert "search_mode" not in task_schema
    assert "search" not in task_schema["agent"]["enum"]


@pytest.mark.parametrize(
    ("current", "legacy"),
    [
        (
            AgentTask(key="inspect", task="Inspect the code", agent="explore"),
            {
                "key": "inspect",
                "task": "Inspect the code",
                "agent": "explore",
                "model": "smart",
                "include_context": False,
                "search_mode": None,
            },
        ),
        (
            AgentTask(
                key="papers",
                task="Find primary papers",
                agent="search_research_publications",
            ),
            {
                "key": "papers",
                "task": "Find primary papers",
                "agent": "search",
                "model": "smart",
                "include_context": False,
                "search_mode": "research-publications",
            },
        ),
    ],
)
def test_pre_upgrade_registry_specs_replay_semantically(
    tmp_path: Path,
    current: AgentTask,
    legacy: dict,
):
    registry = DelegationRegistry(tmp_path / "tasks.sqlite3")
    operation_key = "conversation:stable-batch"
    reserve = {
        "operation_key": operation_key,
        "tree_id": "tree",
        "parent_conversation_id": "conversation",
        "parent_task_id": None,
        "depth": 1,
        "deadlines": [time.time() + 60],
    }
    original, created = registry.reserve(specs=[current], **reserve)
    with sqlite3.connect(registry.path) as database:
        database.execute(
            "UPDATE operations SET specs_json = ? WHERE operation_key = ?",
            (
                json.dumps([legacy], sort_keys=True, separators=(",", ":")),
                operation_key,
            ),
        )

    replayed, replay_created = registry.reserve(specs=[current], **reserve)

    assert created is True
    assert replay_created is False
    assert replayed[0]["task_id"] == original[0]["task_id"]
    with pytest.raises(ValueError, match="different task specifications"):
        registry.reserve(
            specs=[current.model_copy(update={"task": "Do something else"})],
            **reserve,
        )
    with sqlite3.connect(registry.path) as database:
        database.execute(
            "UPDATE operations SET specs_json = ? WHERE operation_key = ?",
            (
                json.dumps(
                    [{**legacy, "unexpected": "must not disappear"}],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                operation_key,
            ),
        )
    with pytest.raises(RuntimeError, match="unknown fields: unexpected"):
        registry.reserve(specs=[current], **reserve)


def test_depth_one_generalist_schema_cannot_request_generalist_child(tmp_path):
    configure_delegation(config(tmp_path, depth=0))
    root = SpawnAgentsTool.create(child_runner_factory=lambda _: None)[0]
    assert root.action_type is SpawnAgentsAction

    configure_delegation(
        config(tmp_path, depth=1, agent_name="general-purpose")
    )
    nested = SpawnAgentsTool.create(child_runner_factory=lambda _: None)[0]
    assert nested.action_type is LeafSpawnAgentsAction
    with pytest.raises(ValueError, match="literal_error"):
        LeafSpawnAgentsAction.model_validate(
            {
                "batch_key": "invalid-generalist-chain",
                "tasks": [{"task": "Recurse", "agent": "general-purpose"}],
            }
        )


class EventSink:
    def __init__(self):
        self.events = []
        self.received = threading.Event()

    def enqueue(self, event) -> bool:
        self.events.append(event)
        self.received.set()
        return True


class FakeChild:
    def __init__(self, release: threading.Event, *, result: str = "done"):
        self.release = release
        self.result = result
        self.started = threading.Event()
        self.calls: list[tuple[str, float | None]] = []
        self.interrupted = False

    def run(self, task: str, timeout_seconds: float | None) -> str:
        self.calls.append((task, timeout_seconds))
        self.started.set()
        self.release.wait(5)
        if self.interrupted:
            raise InterruptedError("cancelled")
        return self.result

    def start(self, task, timeout_seconds, on_complete) -> None:
        def run() -> None:
            try:
                on_complete(self.run(task, timeout_seconds), None)
            except BaseException as error:
                on_complete(None, error)

        threading.Thread(target=run, daemon=True).start()
        assert self.started.wait(1)

    def interrupt(self) -> None:
        self.interrupted = True
        self.release.set()


def parent_conversation() -> SimpleNamespace:
    view = View(
        events=[
            MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text="Investigate the regression")],
                ),
                extended_content=[TextContent(text="Disclosed skill instructions")],
            ),
            MessageEvent(
                source="agent",
                llm_message=Message(
                    role="assistant",
                    content=[TextContent(text="I will inspect the evidence.")],
                ),
            ),
        ]
    )
    return SimpleNamespace(id=uuid.uuid4(), state=SimpleNamespace(view=view))


def config(tmp_path: Path, **updates) -> DelegationConfig:
    values = {
        "python_executable": Path("python"),
        "workspace": tmp_path / "workspace",
        "state_dir": tmp_path / "state",
        "smart_model": "openai/gpt-5.6-sol",
        "smart_reasoning_effort": "xhigh",
        "smart_api_key_env": "OPENAI_API_KEY",
        "smart_api_key": "secret",
        "fast_model": "openai/gpt-5.6-luna",
        "fast_reasoning_effort": "high",
        "fast_api_key_env": "OPENAI_API_KEY",
        "fast_api_key": "secret",
        "frontier_model": "openai/gpt-5.6-sol",
        "frontier_reasoning_effort": "max",
        "frontier_api_key_env": "OPENAI_API_KEY",
        "frontier_api_key": "secret",
        "compaction_trigger_tokens": 200_000,
        "github_repo": "acme/widgets",
        "github_trusted_actor": None,
        "role_file": tmp_path / "role.md",
        "harness_file": tmp_path / "harness.md",
        "plugin_dir": tmp_path / "plugin",
        "enable_browser": False,
        "conversation_secrets": {},
        "role": "advisor",
        "program_path": "program.md",
        "launch_context": "# Authoritative launch context\n\nSystem policy.",
    }
    values.update(updates)
    return DelegationConfig(**values)


def tools(tmp_path: Path, factory, sink=None, **config_updates):
    configure_delegation(config(tmp_path, **config_updates))
    params = {"event_db_path": tmp_path / "events.sqlite3"}
    spawn = SpawnAgentsTool.create(
        child_runner_factory=factory,
        event_sink=sink,
        **params,
    )[0]
    await_tool = AwaitAgentsTool.create(**params)[0]
    status = AgentStatusTool.create(**params)[0]
    cancel = CancelAgentsTool.create(**params)[0]
    return spawn, await_tool, status, cancel


def test_spawn_is_nonblocking_and_await_first_collects_the_first_result(tmp_path):
    releases = [threading.Event(), threading.Event()]
    children = []

    def factory(_request):
        child = FakeChild(releases[len(children)], result=f"result-{len(children)}")
        children.append(child)
        return child

    spawn, await_tool, status, _cancel = tools(tmp_path, factory)
    parent = parent_conversation()

    started = time.monotonic()
    spawned = spawn(
        SpawnAgentsAction(
            batch_key="compare-kernels",
            tasks=[
                AgentTask(key="a", task="Inspect A", agent="explore"),
                AgentTask(key="b", task="Inspect B", agent="explore"),
            ],
        ),
        parent,
    )

    assert time.monotonic() - started < 0.5
    assert all(child.started.wait(1) for child in children)
    assert [task.status for task in spawned.tasks] == ["running", "running"]

    releases[1].set()
    result = await_tool(
        AwaitAgentsAction(
            task_ids=[task.task_id for task in spawned.tasks],
            join="first",
            timeout_seconds=2,
        ),
        parent,
    )

    assert result.satisfied is True
    assert result.timed_out is False
    assert [task.status for task in result.tasks].count("finished") == 1
    assert [task.status for task in status(AgentStatusAction(), parent).tasks].count(
        "running"
    ) == 1
    releases[0].set()


def test_compacted_result_is_the_only_parent_visible_task_value(tmp_path):
    raw_report = "RAW_REPORT_MUST_NOT_REACH_THE_PARENT"
    artifact = tmp_path / "state" / "delegation" / "results" / "task.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(raw_report, encoding="utf-8")
    public_result = f"Concise conclusion.\n\nFull report: {artifact}"
    release = threading.Event()
    release.set()
    sink = EventSink()
    requests = []

    def factory(request):
        requests.append(request)
        return FakeChild(release, result=public_result)

    spawn, await_tool, status, _cancel = tools(
        tmp_path,
        factory,
        sink=sink,
    )
    parent = parent_conversation()
    task = spawn(
        SpawnAgentsAction(
            batch_key="compacted-result",
            tasks=[AgentTask(task="Return a bounded report")],
        ),
        parent,
    ).tasks[0]
    assert sink.received.wait(1)

    registry_path = tmp_path / "state" / "delegation" / "tasks.sqlite3"
    child_environment = OpenHandsChildProcess(
        config(tmp_path),
        requests[0],
    ).environment
    assert child_environment["SENPAI_DELEGATION_REGISTRY_PATH"] == str(
        registry_path
    )
    assert child_environment["SENPAI_DELEGATION_ROOT_STATE_DIR"] == str(
        tmp_path / "state"
    )
    assert child_environment["SENPAI_DELEGATION_TASK_ID"] == task.task_id

    registry = DelegationRegistry(registry_path)
    stored_result = registry.rows([task.task_id])[0]["result"]
    status_observation = status(AgentStatusAction(task_ids=[task.task_id]), parent)
    await_observation = await_tool(
        AwaitAgentsAction(task_ids=[task.task_id], timeout_seconds=1), parent
    )
    status_result = status_observation.tasks[0].result
    await_result = await_observation.tasks[0].result
    event_result = sink.events[0].payload["result"]

    assert (stored_result, status_result, await_result, event_result) == (
        public_result,
    ) * 4
    parent_messages = [
        status_observation.to_llm_content[0].text,
        await_observation.to_llm_content[0].text,
        sink.events[0].to_inbox_message(),
    ]
    assert all(str(artifact) in message for message in parent_messages)
    assert all(raw_report not in message for message in parent_messages)


def test_search_task_form_is_resolved_in_registry_and_child_request(tmp_path):
    release = threading.Event()
    release.set()
    requests = []

    def factory(request):
        requests.append(request)
        return FakeChild(release)

    spawn, *_ = tools(tmp_path, factory)
    parent = parent_conversation()
    spawned = spawn(
        SpawnAgentsAction(
            batch_key="publication-search",
            tasks=[
                AgentTask(
                    task="Find primary sources",
                    agent="search_research_publications",
                )
            ],
        ),
        parent,
    )
    row = DelegationRegistry(
        tmp_path / "state" / "delegation" / "tasks.sqlite3"
    ).rows([spawned.tasks[0].task_id])[0]

    assert (requests[0].agent, requests[0].search_mode) == (
        "search",
        "research-publications",
    )
    assert (row["agent"], row["search_mode"]) == (
        "search",
        "research-publications",
    )


def test_disabled_web_search_rejects_search_tasks_but_keeps_local_agents(tmp_path):
    release = threading.Event()
    release.set()
    requests = []

    def factory(request):
        requests.append(request)
        return FakeChild(release)

    spawn, *_ = tools(tmp_path, factory, web_search=False)
    parent = parent_conversation()

    with pytest.raises(ValueError, match="external search agents are disabled"):
        spawn(
            SpawnAgentsAction(
                batch_key="disabled-search",
                tasks=[
                    AgentTask(
                        task="Search the web",
                        agent="search_general_web",
                    )
                ],
            ),
            parent,
        )

    spawned = spawn(
        SpawnAgentsAction(
            batch_key="local-exploration",
            tasks=[AgentTask(task="Inspect the code", agent="explore")],
        ),
        parent,
    )

    assert spawned.tasks[0].agent == "explore"
    assert requests[0].agent == "explore"


def test_await_quorum_and_timeout_leave_unfinished_tasks_running(tmp_path):
    releases = [threading.Event() for _ in range(3)]
    children = []

    def factory(_request):
        child = FakeChild(releases[len(children)])
        children.append(child)
        return child

    spawn, await_tool, _status, _cancel = tools(tmp_path, factory)
    parent = parent_conversation()
    spawned = spawn(
        SpawnAgentsAction(
            batch_key="quorum",
            tasks=[AgentTask(key=str(i), task=f"Task {i}") for i in range(3)],
        ),
        parent,
    )
    releases[0].set()
    releases[1].set()

    quorum = await_tool(
        AwaitAgentsAction(
            task_ids=[task.task_id for task in spawned.tasks],
            join="quorum",
            quorum=2,
            timeout_seconds=2,
        ),
        parent,
    )
    assert quorum.satisfied is True
    assert [task.status for task in quorum.tasks].count("finished") == 2

    timed_out = await_tool(
        AwaitAgentsAction(
            task_ids=[spawned.tasks[2].task_id],
            timeout_seconds=0.05,
        ),
        parent,
    )
    assert timed_out.satisfied is False
    assert timed_out.timed_out is True
    assert timed_out.tasks[0].status == "running"
    assert timed_out.waited_seconds >= 0.05
    assert "keep running" in timed_out.guidance
    assert "join='change'" in timed_out.guidance
    assert "join='first'" not in timed_out.guidance
    releases[2].set()


def test_await_change_returns_on_the_next_task_transition(tmp_path):
    release = threading.Event()
    spawn, await_tool, _status, _cancel = tools(
        tmp_path,
        lambda _request: FakeChild(release),
    )
    parent = parent_conversation()
    task = spawn(
        SpawnAgentsAction(
            batch_key="next-state-change",
            tasks=[AgentTask(task="Finish after the signal", agent="explore")],
        ),
        parent,
    ).tasks[0]

    threading.Timer(0.05, release.set).start()
    changed = await_tool(
        AwaitAgentsAction(
            task_ids=[task.task_id],
            join="change",
            timeout_seconds=2,
        ),
        parent,
    )

    assert changed.satisfied is True
    assert changed.timed_out is False
    assert changed.changed_task_ids == [task.task_id]
    assert changed.tasks[0].status == "finished"


def test_repeated_await_change_ignores_a_collected_terminal_sibling(tmp_path):
    releases = [threading.Event(), threading.Event()]
    children = []

    def factory(_request):
        child = FakeChild(releases[len(children)])
        children.append(child)
        return child

    spawn, await_tool, status, _cancel = tools(tmp_path, factory)
    parent = parent_conversation()
    spawned = spawn(
        SpawnAgentsAction(
            batch_key="successive-change",
            tasks=[AgentTask(task="First"), AgentTask(task="Second")],
        ),
        parent,
    )
    releases[0].set()
    while status(
        AgentStatusAction(task_ids=[spawned.tasks[0].task_id]), parent
    ).tasks[0].status == "running":
        time.sleep(0.01)

    first = await_tool(
        AwaitAgentsAction(
            task_ids=[task.task_id for task in spawned.tasks],
            join="change",
            timeout_seconds=1,
        ),
        parent,
    )
    repeated = await_tool(
        AwaitAgentsAction(
            task_ids=[task.task_id for task in spawned.tasks],
            join="change",
            timeout_seconds=0.05,
        ),
        parent,
    )

    assert first.satisfied is True
    assert first.changed_task_ids == []
    assert repeated.satisfied is False
    assert repeated.timed_out is True
    assert repeated.changed_task_ids == []
    assert [task.status for task in repeated.tasks] == ["finished", "running"]
    assert repeated.waited_seconds >= 0.05
    releases[1].set()


def test_timed_out_await_collects_results_it_already_returned(tmp_path):
    releases = [threading.Event(), threading.Event()]
    children = []

    def factory(_request):
        child = FakeChild(releases[len(children)])
        children.append(child)
        return child

    spawn, await_tool, status, _cancel = tools(tmp_path, factory)
    parent = parent_conversation()
    spawned = spawn(
        SpawnAgentsAction(
            batch_key="partial-timeout",
            tasks=[AgentTask(task="First"), AgentTask(task="Second")],
        ),
        parent,
    )
    releases[0].set()
    while status(
        AgentStatusAction(task_ids=[spawned.tasks[0].task_id]), parent
    ).tasks[0].status == "running":
        time.sleep(0.01)

    result = await_tool(
        AwaitAgentsAction(
            task_ids=[task.task_id for task in spawned.tasks],
            join="all",
            timeout_seconds=0.05,
        ),
        parent,
    )
    uncollected = DelegationRegistry(
        tmp_path / "state" / "delegation" / "tasks.sqlite3"
    ).uncollected_for_parent(str(parent.id))

    assert result.timed_out is True
    assert [task.status for task in result.tasks].count("finished") == 1
    assert [row["task_id"] for row in uncollected] == [spawned.tasks[1].task_id]
    releases[1].set()


def test_replayed_batch_reuses_task_ids_and_changed_specs_fail(tmp_path):
    release = threading.Event()
    requests: list[DelegationRequest] = []

    def factory(request):
        requests.append(request)
        return FakeChild(release)

    spawn, _await, _status, _cancel = tools(tmp_path, factory)
    parent = parent_conversation()
    action = SpawnAgentsAction(
        batch_key="stable-operation",
        tasks=[AgentTask(key="review", task="Review the implementation")],
    )

    first = spawn(action, parent)
    second = spawn(action, parent)

    assert [task.task_id for task in first.tasks] == [
        task.task_id for task in second.tasks
    ]
    assert len(requests) == 1
    with pytest.raises(ValueError, match="different task specifications"):
        spawn(
            SpawnAgentsAction(
                batch_key="stable-operation",
                tasks=[AgentTask(key="review", task="Implement it instead")],
            ),
            parent,
        )
    release.set()


def test_replay_of_pre_reserved_queued_work_never_launches_it(tmp_path):
    starts = []
    spawn, *_ = tools(
        tmp_path,
        lambda request: starts.append(request),
    )
    parent = parent_conversation()
    action = SpawnAgentsAction(
        batch_key="reserved-before-crash",
        tasks=[AgentTask(key="one", task="Do not duplicate")],
    )
    registry = DelegationRegistry(tmp_path / "state" / "delegation" / "tasks.sqlite3")
    reserved, created = registry.reserve(
        operation_key=f"{parent.id}:{action.batch_key}",
        tree_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"senpai-tree:{parent.id}:{action.batch_key}",
            )
        ),
        parent_conversation_id=str(parent.id),
        parent_task_id=None,
        depth=1,
        specs=action.tasks,
        deadlines=[time.time() + 60],
    )

    replay = spawn(action, parent)

    assert created is True
    assert replay.tasks[0].task_id == reserved[0]["task_id"]
    assert replay.tasks[0].status == "queued"
    assert starts == []


def test_dead_replayed_task_becomes_failed_and_is_never_respawned(tmp_path):
    release = threading.Event()
    requests = []

    def factory(request):
        requests.append(request)
        return FakeChild(release)

    spawn, _await, _status, _cancel = tools(tmp_path, factory)
    parent = parent_conversation()
    action = SpawnAgentsAction(
        batch_key="dead-child",
        tasks=[AgentTask(task="Inspect failure")],
    )
    first = spawn(action, parent)
    registry = DelegationRegistry(tmp_path / "state" / "delegation" / "tasks.sqlite3")
    registry.mark_running(first.tasks[0].task_id, 999_999_999)

    replay = spawn(action, parent)

    assert replay.tasks[0].status == "failed"
    assert "no longer running" in replay.tasks[0].error
    assert len(requests) == 1
    with LocalEventStore(tmp_path / "events.sqlite3") as events:
        assert [event.payload["task_id"] for event in events.pending()] == [
            first.tasks[0].task_id
        ]
    release.set()


def test_controller_startup_reconciles_a_dead_background_task_and_enqueues_its_wake(
    tmp_path,
):
    parent = parent_conversation()
    registry = DelegationRegistry(tmp_path / "state" / "delegation" / "tasks.sqlite3")
    rows, _created = registry.reserve(
        operation_key=f"{parent.id}:orphaned-child",
        tree_id="orphaned-tree",
        parent_conversation_id=str(parent.id),
        parent_task_id=None,
        depth=1,
        specs=[AgentTask(key="orphan", task="Recover after restart")],
        deadlines=[time.time() + 60],
    )
    task_id = rows[0]["task_id"]
    registry.mark_running(task_id, 999_999_999)

    reconcile_delegated_tasks(
        tmp_path / "state",
        tmp_path / "events.sqlite3",
    )

    failed = registry.rows([task_id])[0]
    assert failed["status"] == "failed"
    assert "no longer running" in failed["error"]
    with LocalEventStore(tmp_path / "events.sqlite3") as events:
        pending = events.pending()
    assert [event.payload["task_id"] for event in pending] == [task_id]


def test_controller_startup_immediately_fails_a_preexisting_queued_task(tmp_path):
    registry = DelegationRegistry(tmp_path / "state" / "delegation" / "tasks.sqlite3")
    rows, _created = registry.reserve(
        operation_key="conversation:queued-before-restart",
        tree_id="queued-tree",
        parent_conversation_id="conversation",
        parent_task_id=None,
        depth=1,
        specs=[AgentTask(key="queued", task="Never launched")],
        deadlines=[time.time() + 60],
    )
    task_id = rows[0]["task_id"]

    reconcile_delegated_tasks(
        tmp_path / "state",
        tmp_path / "events.sqlite3",
    )

    failed = registry.rows([task_id])[0]
    assert failed["status"] == "failed"
    assert "startup did not complete" in failed["error"]
    with LocalEventStore(tmp_path / "events.sqlite3") as events:
        pending = events.pending()
    assert [event.payload["task_id"] for event in pending] == [task_id]


def test_cancel_is_targeted_and_releases_the_child(tmp_path):
    releases = [threading.Event(), threading.Event()]
    children = []

    def factory(_request):
        child = FakeChild(releases[len(children)])
        children.append(child)
        return child

    spawn, _await, status, cancel = tools(tmp_path, factory)
    parent = parent_conversation()
    spawned = spawn(
        SpawnAgentsAction(
            batch_key="cancel-one",
            tasks=[AgentTask(task="A"), AgentTask(task="B")],
        ),
        parent,
    )

    cancelled = cancel(
        CancelAgentsAction(task_ids=[spawned.tasks[0].task_id]),
        parent,
    )

    assert cancelled.tasks[0].status == "cancelled"
    assert children[0].interrupted is True
    assert children[1].interrupted is False
    assert status(
        AgentStatusAction(task_ids=[spawned.tasks[1].task_id]), parent
    ).tasks[0].status == "running"
    releases[1].set()


def test_cancelling_an_already_finished_task_acknowledges_its_event(tmp_path):
    release = threading.Event()
    release.set()
    spawn, await_tool, _status, cancel = tools(
        tmp_path,
        lambda _request: FakeChild(release),
    )
    parent = parent_conversation()
    task = spawn(
        SpawnAgentsAction(
            batch_key="terminal-cancel",
            tasks=[AgentTask(key="task", task="Finish")],
        ),
        parent,
    ).tasks[0]
    while True:
        with LocalEventStore(tmp_path / "events.sqlite3") as events:
            if events.pending():
                break
        time.sleep(0.01)

    cancelled = cancel(CancelAgentsAction(task_ids=[task.task_id]), parent)

    assert cancelled.tasks[0].status == "finished"
    with LocalEventStore(tmp_path / "events.sqlite3") as events:
        assert events.pending() == []


def test_cancel_waits_for_batch_admission_and_catches_every_claimed_child(tmp_path):
    first_start_entered = threading.Event()
    allow_first_start = threading.Event()
    releases = [threading.Event(), threading.Event()]
    children = []

    class SlowStartChild(FakeChild):
        def start(self, task, timeout_seconds, on_complete):
            first_start_entered.set()
            assert allow_first_start.wait(2)
            super().start(task, timeout_seconds, on_complete)

    def factory(_request):
        child = (
            SlowStartChild(releases[0])
            if not children
            else FakeChild(releases[1])
        )
        children.append(child)
        return child

    spawn, _await, _status, cancel = tools(tmp_path, factory)
    parent = parent_conversation()
    action = SpawnAgentsAction(
        batch_key="spawn-cancel-race",
        tasks=[
            AgentTask(key="first", task="First"),
            AgentTask(key="second", task="Second"),
        ],
    )
    spawned = []
    spawn_thread = threading.Thread(
        target=lambda: spawned.append(spawn(action, parent)),
    )
    spawn_thread.start()
    assert first_start_entered.wait(1)
    registry = DelegationRegistry(tmp_path / "state" / "delegation" / "tasks.sqlite3")
    task_ids = [
        row["task_id"]
        for row in registry.rows(parent_conversation_id=str(parent.id))
    ]
    cancelled = []
    cancel_thread = threading.Thread(
        target=lambda: cancelled.append(
            cancel(CancelAgentsAction(task_ids=task_ids), parent)
        ),
    )
    cancel_thread.start()
    time.sleep(0.05)
    assert cancel_thread.is_alive()

    allow_first_start.set()
    spawn_thread.join(2)
    cancel_thread.join(3)

    assert len(children) == 2
    assert all(child.interrupted for child in children)
    assert [task.status for task in cancelled[0].tasks] == ["cancelled", "cancelled"]


def test_tree_and_depth_guards_apply_before_any_child_starts(tmp_path):
    releases = [threading.Event() for _ in range(8)]
    requests = []

    def factory(request):
        requests.append(request)
        return FakeChild(releases[len(requests) - 1])

    parent = parent_conversation()
    spawn, _await, _status, _cancel = tools(tmp_path, factory)
    root = spawn(
        SpawnAgentsAction(
            batch_key="bounded-tree",
            tasks=[
                AgentTask(key=str(i), task=f"Root task {i}", agent="general-purpose")
                for i in range(5)
            ],
        ),
        parent,
    )
    tree_id = requests[0].tree_id

    nested_parent = parent_conversation()
    nested_spawn, *_ = tools(
        tmp_path,
        factory,
        tree_id=tree_id,
        depth=1,
        agent_name="general-purpose",
        current_task_id=root.tasks[0].task_id,
    )
    with pytest.raises(RuntimeError, match="tree capacity"):
        nested_spawn(
            SpawnAgentsAction(
                batch_key="too-many-descendants",
                tasks=[
                    AgentTask(key=str(i), task=f"Leaf {i}", agent="explore")
                    for i in range(4)
                ],
            ),
            nested_parent,
        )
    assert len(requests) == len(root.tasks)

    leaf_spawn, *_ = tools(
        tmp_path,
        factory,
        tree_id=tree_id,
        depth=1,
        agent_name="explore",
        current_task_id=root.tasks[0].task_id,
    )
    with pytest.raises(ValueError, match="delegation leaves"):
        leaf_spawn(
            SpawnAgentsAction(
                batch_key="illegal-recursion",
                tasks=[AgentTask(task="Nested", agent="explore")],
            ),
            parent_conversation(),
        )

    with pytest.raises(ValueError, match="depth-2 helpers must be leaf"):
        nested_spawn(
            SpawnAgentsAction(
                batch_key="general-purpose-chain",
                tasks=[AgentTask(task="Another GP", agent="general-purpose")],
            ),
            nested_parent,
        )

    depth_two_spawn, *_ = tools(
        tmp_path,
        factory,
        tree_id=tree_id,
        depth=2,
        agent_name="explore",
        current_task_id=str(uuid.uuid4()),
    )
    with pytest.raises(ValueError, match="maximum delegation depth"):
        depth_two_spawn(
            SpawnAgentsAction(
                batch_key="cedar-depth-four",
                tasks=[AgentTask(task="Too deep", agent="explore")],
            ),
            parent_conversation(),
        )

    for release in releases:
        release.set()


def test_registry_active_cap_spans_trees_and_frees_after_completion(tmp_path):
    releases = [threading.Event() for _ in range(9)]
    requests = []

    def factory(request):
        child = FakeChild(releases[len(requests)])
        requests.append(request)
        return child

    spawn, await_tool, *_ = tools(tmp_path, factory)
    parent = parent_conversation()
    first = spawn(
        SpawnAgentsAction(
            batch_key="first-tree",
            tasks=[AgentTask(key=str(i), task=f"First {i}") for i in range(5)],
        ),
        parent,
    )

    with pytest.raises(RuntimeError, match="capacity is full"):
        spawn(
            SpawnAgentsAction(
                batch_key="second-tree",
                tasks=[AgentTask(key=str(i), task=f"Second {i}") for i in range(4)],
            ),
            parent,
        )

    for release in releases[:5]:
        release.set()
    assert await_tool(
        AwaitAgentsAction(
            task_ids=[task.task_id for task in first.tasks],
            timeout_seconds=2,
        ),
        parent,
    ).satisfied
    second = spawn(
        SpawnAgentsAction(
            batch_key="second-tree",
            tasks=[AgentTask(key=str(i), task=f"Second {i}") for i in range(4)],
        ),
        parent,
    )

    assert len(second.tasks) == 4
    assert requests[0].tree_id != requests[5].tree_id
    for release in releases[5:]:
        release.set()


def test_cancelling_a_root_task_cancels_descendants_deepest_first(tmp_path):
    root_release = threading.Event()
    leaf_release = threading.Event()
    root_children = []
    leaf_children = []
    interrupt_order = []

    class OrderedChild(FakeChild):
        def __init__(self, release, name):
            super().__init__(release)
            self.name = name

        def interrupt(self):
            interrupt_order.append(self.name)
            super().interrupt()

    def root_factory(request):
        child = OrderedChild(root_release, "root")
        root_children.append((request, child))
        return child

    parent = parent_conversation()
    root_spawn, _await, _status, root_cancel = tools(tmp_path, root_factory)
    root = root_spawn(
        SpawnAgentsAction(
            batch_key="root-tree",
            tasks=[AgentTask(key="gp", task="Research", agent="general-purpose")],
        ),
        parent,
    )
    root_request = root_children[0][0]

    def leaf_factory(_request):
        child = OrderedChild(leaf_release, "leaf")
        leaf_children.append(child)
        return child

    nested_spawn, *_ = tools(
        tmp_path,
        leaf_factory,
        tree_id=root_request.tree_id,
        depth=1,
        agent_name="general-purpose",
        current_task_id=root.tasks[0].task_id,
    )
    nested_spawn(
        SpawnAgentsAction(
            batch_key="leaf-batch",
            tasks=[AgentTask(key="leaf", task="Inspect", agent="explore")],
        ),
        parent_conversation(),
    )

    cancelled = root_cancel(
        CancelAgentsAction(task_ids=[root.tasks[0].task_id]),
        parent,
    )

    assert cancelled.tasks[0].status == "cancelled"
    assert root_children[0][1].interrupted
    assert leaf_children[0].interrupted
    assert interrupt_order == ["leaf", "root"]


def test_expired_parent_deadline_fails_root_and_cancels_descendants(tmp_path):
    deadline = time.time() + 0.2
    root_release = threading.Event()
    leaf_release = threading.Event()
    root_children = []
    leaf_children = []

    def root_factory(request):
        child = FakeChild(root_release)
        root_children.append((request, child))
        return child

    parent = parent_conversation()
    root_spawn, _await, root_status, _cancel = tools(
        tmp_path,
        root_factory,
        deadline_epoch=deadline,
    )
    root = root_spawn(
        SpawnAgentsAction(
            batch_key="deadline-tree",
            tasks=[AgentTask(key="gp", task="Parent", agent="general-purpose")],
        ),
        parent,
    ).tasks[0]

    def leaf_factory(_request):
        child = FakeChild(leaf_release)
        leaf_children.append(child)
        return child

    nested_spawn, *_ = tools(
        tmp_path,
        leaf_factory,
        tree_id=root_children[0][0].tree_id,
        depth=1,
        deadline_epoch=deadline,
        agent_name="general-purpose",
        current_task_id=root.task_id,
    )
    leaf = nested_spawn(
        SpawnAgentsAction(
            batch_key="deadline-leaf",
            tasks=[AgentTask(key="leaf", task="Leaf", agent="explore")],
        ),
        parent_conversation(),
    ).tasks[0]
    time.sleep(0.25)

    reconciled = root_status(
        AgentStatusAction(task_ids=[root.task_id]),
        parent,
    )

    assert reconciled.tasks[0].status == "failed"
    assert root_children[0][1].interrupted
    assert leaf_children[0].interrupted
    registry = DelegationRegistry(tmp_path / "state" / "delegation" / "tasks.sqlite3")
    assert registry.rows([leaf.task_id])[0]["status"] == "cancelled"


def test_depth_two_completion_is_collected_by_parent_not_root_event_stream(tmp_path):
    root_release = threading.Event()
    root_requests = []

    def root_factory(request):
        root_requests.append(request)
        return FakeChild(root_release)

    root_spawn, *_ = tools(tmp_path, root_factory)
    root_parent = parent_conversation()
    root = root_spawn(
        SpawnAgentsAction(
            batch_key="root",
            tasks=[AgentTask(key="gp", task="Parent", agent="general-purpose")],
        ),
        root_parent,
    ).tasks[0]
    release = threading.Event()
    release.set()
    sink = EventSink()
    spawn, await_tool, *_ = tools(
        tmp_path,
        lambda _request: FakeChild(release),
        sink=sink,
        tree_id=root_requests[0].tree_id,
        depth=1,
        agent_name="general-purpose",
        current_task_id=root.task_id,
    )
    parent = parent_conversation()
    task = spawn(
        SpawnAgentsAction(
            batch_key="nested",
            tasks=[AgentTask(key="leaf", task="Inspect", agent="explore")],
        ),
        parent,
    ).tasks[0]
    assert await_tool(
        AwaitAgentsAction(task_ids=[task.task_id], timeout_seconds=2),
        parent,
    ).satisfied
    assert sink.events == []
    root_release.set()


def test_finished_leaf_must_be_awaited_not_only_status_checked(tmp_path):
    root_release = threading.Event()
    root_requests = []

    def root_factory(request):
        root_requests.append(request)
        return FakeChild(root_release)

    root_spawn, *_ = tools(tmp_path, root_factory)
    root = root_spawn(
        SpawnAgentsAction(
            batch_key="root-for-collection",
            tasks=[AgentTask(key="gp", task="Parent", agent="general-purpose")],
        ),
        parent_conversation(),
    ).tasks[0]
    release = threading.Event()
    release.set()
    nested_spawn, nested_await, nested_status, _cancel = tools(
        tmp_path,
        lambda _request: FakeChild(release),
        tree_id=root_requests[0].tree_id,
        depth=1,
        agent_name="general-purpose",
        current_task_id=root.task_id,
    )
    nested_parent = parent_conversation()
    first = nested_spawn(
        SpawnAgentsAction(
            batch_key="status-only",
            tasks=[AgentTask(key="leaf", task="Inspect", agent="explore")],
        ),
        nested_parent,
    ).tasks[0]
    while nested_status(
        AgentStatusAction(task_ids=[first.task_id]), nested_parent
    ).tasks[0].status == "running":
        time.sleep(0.01)

    registry_path = tmp_path / "state" / "delegation" / "tasks.sqlite3"
    assert cancel_pending_descendants(registry_path, str(nested_parent.id)) == [
        first.task_id
    ]

    second = nested_spawn(
        SpawnAgentsAction(
            batch_key="awaited",
            tasks=[AgentTask(key="leaf", task="Inspect again", agent="explore")],
        ),
        nested_parent,
    ).tasks[0]
    assert nested_await(
        AwaitAgentsAction(task_ids=[second.task_id], timeout_seconds=2),
        nested_parent,
    ).satisfied
    assert cancel_pending_descendants(registry_path, str(nested_parent.id)) == []
    root_release.set()


def test_explicit_task_ids_are_private_to_the_spawning_parent(tmp_path):
    release = threading.Event()
    spawn, await_tool, status, cancel = tools(
        tmp_path,
        lambda _request: FakeChild(release),
    )
    owner = parent_conversation()
    stranger = parent_conversation()
    task = spawn(
        SpawnAgentsAction(
            batch_key="owned",
            tasks=[AgentTask(key="task", task="Private task")],
        ),
        owner,
    ).tasks[0]

    with pytest.raises(ValueError, match="only"):
        status(AgentStatusAction(task_ids=[task.task_id]), stranger)
    with pytest.raises(ValueError, match="only"):
        await_tool(
            AwaitAgentsAction(task_ids=[task.task_id], timeout_seconds=1),
            stranger,
        )
    with pytest.raises(ValueError, match="only its own"):
        cancel(CancelAgentsAction(task_ids=[task.task_id]), stranger)
    release.set()


@pytest.mark.parametrize(
    "action",
    [
        AwaitAgentsAction.model_construct(
            task_ids=["same", "same"],
            join="all",
            quorum=None,
            timeout_seconds=1,
        ),
        AgentStatusAction.model_construct(task_ids=["same", "same"]),
        CancelAgentsAction.model_construct(task_ids=["same", "same"]),
    ],
)
def test_task_control_actions_reject_duplicate_ids(action):
    with pytest.raises(ValueError, match="duplicates"):
        type(action).model_validate(action.model_dump())


def test_tool_close_leaves_spawned_background_work_alive(tmp_path):
    release = threading.Event()
    spawn, await_tool, *_ = tools(
        tmp_path,
        lambda _request: FakeChild(release),
    )
    parent = parent_conversation()
    task = spawn(
        SpawnAgentsAction(
            batch_key="survives-close",
            tasks=[AgentTask(key="task", task="Keep working")],
        ),
        parent,
    ).tasks[0]

    spawn.executor.close()
    release.set()

    result = await_tool(
        AwaitAgentsAction(task_ids=[task.task_id], timeout_seconds=2),
        parent,
    )
    assert result.satisfied
    assert result.tasks[0].status == "finished"


def test_start_failure_returns_visible_failed_state_without_hiding_siblings(tmp_path):
    release = threading.Event()
    children = []

    class StartFailure(FakeChild):
        def start(self, task, timeout_seconds, on_complete):
            raise RuntimeError("could not launch")

    def factory(_request):
        child = (
            FakeChild(release)
            if not children
            else StartFailure(threading.Event())
        )
        children.append(child)
        return child

    spawn, *_ = tools(tmp_path, factory)
    result = spawn(
        SpawnAgentsAction(
            batch_key="partial-start",
            tasks=[
                AgentTask(key="running", task="Runs"),
                AgentTask(key="failed", task="Fails"),
            ],
        ),
        parent_conversation(),
    )

    assert [task.status for task in result.tasks] == ["running", "failed"]
    assert "could not launch" in result.tasks[1].error
    model_payload = json.loads(result.to_llm_content[0].text)
    assert model_payload["tasks"][1] == {
        "task_id": result.tasks[1].task_id,
        "key": "failed",
        "status": "failed",
        "agent": "general-purpose",
        "model": "smart",
        "result": None,
        "error": "RuntimeError: could not launch",
    }
    release.set()


def test_status_without_ids_includes_uncollected_terminal_tasks(tmp_path):
    release = threading.Event()
    release.set()
    spawn, _await, status, _cancel = tools(
        tmp_path,
        lambda _request: FakeChild(release),
    )
    parent = parent_conversation()
    task = spawn(
        SpawnAgentsAction(
            batch_key="uncollected-status",
            tasks=[AgentTask(key="done", task="Finish")],
        ),
        parent,
    ).tasks[0]
    while status(
        AgentStatusAction(task_ids=[task.task_id]), parent
    ).tasks[0].status != "finished":
        time.sleep(0.01)

    default_status = status(AgentStatusAction(), parent)

    assert [item.task_id for item in default_status.tasks] == [task.task_id]
    assert default_status.tasks[0].status == "finished"


def test_internal_starting_claim_is_reported_publicly_as_queued(tmp_path):
    _spawn, await_tool, status, _cancel = tools(
        tmp_path,
        lambda _request: pytest.fail("replay must not launch"),
    )
    parent = parent_conversation()
    action = SpawnAgentsAction(
        batch_key="starting-claim",
        tasks=[AgentTask(key="one", task="Claimed")],
    )
    registry = DelegationRegistry(tmp_path / "state" / "delegation" / "tasks.sqlite3")
    rows, _created = registry.reserve(
        operation_key=f"{parent.id}:{action.batch_key}",
        tree_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"senpai-tree:{parent.id}:{action.batch_key}",
            )
        ),
        parent_conversation_id=str(parent.id),
        parent_task_id=None,
        depth=1,
        specs=action.tasks,
        deadlines=[time.time() + 60],
    )
    assert registry.claim_launch(rows[0]["task_id"])

    observed = status(
        AgentStatusAction(task_ids=[rows[0]["task_id"]]),
        parent,
    )

    assert observed.tasks[0].status == "queued"
    assert await_tool.annotations.readOnlyHint is False
    assert status.annotations.readOnlyHint is False


def test_tier_runtime_is_clamped_to_the_inherited_deadline(tmp_path):
    release = threading.Event()
    release.set()
    children = []

    def factory(_request):
        child = FakeChild(release)
        children.append(child)
        return child

    spawn, *_ = tools(
        tmp_path,
        factory,
        deadline_epoch=time.time() + 20,
    )
    spawn(
        SpawnAgentsAction(
            batch_key="deadline",
            tasks=[
                AgentTask(key="fast", task="Fast", model="fast"),
                AgentTask(key="frontier", task="Frontier", model="frontier"),
            ],
        ),
        parent_conversation(),
    )

    assert all(child.started.wait(1) for child in children)
    assert all(0 < child.calls[0][1] <= 20 for child in children)


def test_context_is_copied_only_for_tasks_that_request_it(tmp_path):
    release = threading.Event()
    release.set()
    requests = []

    def factory(request):
        requests.append(request)
        return FakeChild(release)

    spawn, *_ = tools(tmp_path, factory)
    spawn(
        SpawnAgentsAction(
            batch_key="context",
            tasks=[
                AgentTask(key="with", task="With", include_context=True),
                AgentTask(key="without", task="Without"),
            ],
        ),
        parent_conversation(),
    )

    assert [message.role for message in requests[0].parent_context] == [
        "user",
        "assistant",
    ]
    assert requests[1].parent_context == ()
