import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
from openhands.sdk.context.view import View
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent
from pydantic import SecretStr

import senpai_agent.delegation as delegation
from openhands_support import runtime_config
from senpai_agent.delegation import (
    AgentStatusAction,
    AgentStatusTool,
    AwaitAgentsAction,
    AwaitAgentsTool,
    CancelAgentsAction,
    CancelAgentsTool,
    configure_delegation,
    reconcile_delegated_tasks,
)
from senpai_agent.github.tools import (
    AssignmentVersion,
    GitHubToolRuntime,
    MergeExperimentAction,
    MergeExperimentTool,
)
from senpai_agent.github.tools.runtime import (
    clear_github_credentials,
    configure_github_credentials,
)
from senpai_agent.local_events import LocalEventStore
from senpai_agent.openhands_runner import delegation_config
from senpai_agent.secrets import MODEL_CREDENTIALS_FD_ENV


@pytest.fixture
def merge_runtime(tmp_path, monkeypatch):
    workspace = tmp_path / "target"
    workspace.mkdir()
    config = delegation_config(runtime_config(
        tmp_path,
        workspace=workspace,
        smart_model="openai/gpt-5.6-sol",
        smart_reasoning_effort="max",
        smart_api_key_env="OPENAI_API_KEY",
        smart_api_key=SecretStr("frontier-key"),
    ))
    module_root = tmp_path / "modules"
    package = module_root / "senpai_agent" / "github"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "merge_worker.py").write_text(
        "import json, os, pathlib, sys, time\n"
        "state = pathlib.Path(sys.argv[sys.argv.index('--state-dir') + 1])\n"
        "state.mkdir(parents=True, exist_ok=True)\n"
        f"with os.fdopen(int(os.environ[{MODEL_CREDENTIALS_FD_ENV!r}])) as stream:\n"
        "    credentials = json.load(stream)\n"
        "capture = {'argv': sys.argv, 'environment': dict(os.environ), "
        "'credentials': credentials, 'context': sys.stdin.read()}\n"
        "(state / 'capture.json').write_text(json.dumps(capture))\n"
        "while not (state / 'release.json').exists():\n"
        "    time.sleep(0.01)\n"
        "result = (state / 'release.json').read_text()\n"
        "print('OPENHANDS_RESULT ' + json.dumps({'status':'finished', 'result':result}), flush=True)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(module_root))
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-must-not-reach-child")
    monkeypatch.setenv(
        "SENPAI_PARENT_CONVERSATION_HISTORY_DIR", str(tmp_path / "advisor-history"),
    )
    configure_delegation(config)
    configure_github_credentials("acme/widgets", SecretStr("private-merge-token"))
    parent = SimpleNamespace(
        id=uuid.uuid4(),
        state=SimpleNamespace(secret_registry=SecretRegistry(), view=View(events=[
            MessageEvent(
                source="user",
                llm_message=Message(role="user", content=[TextContent(text="Keep the solver simple.")]),
                extended_content=[TextContent(text="The target requires the new evaluation fixture.")],
            ),
            MessageEvent(
                source="agent",
                llm_message=Message(role="assistant", content=[TextContent(text="The paired experiment improved accuracy.")]),
            ),
        ])),
    )
    events = config.state_dir / "advisor-events.sqlite3"
    runtime = GitHubToolRuntime(
        workflow=SimpleNamespace(), workspace=workspace, git_token=None, role="advisor",
        advisor_branch="research", student_names=frozenset({"student-one"}), student_name=None,
        event_db_path=events,
    )
    action = MergeExperimentAction(
        assignment=AssignmentVersion(
            pr_number=17, assignment_id="assignment-17", revision_id="revision-1",
            expected_pr_head_sha="a" * 40,
        ),
        expected_current_base_sha="b" * 40,
    )
    yield config, parent, events, MergeExperimentTool.create(runtime)[0], action
    try:
        tasks = AgentStatusTool.create(event_db_path=events)[0](AgentStatusAction(), parent).tasks
        if tasks:
            CancelAgentsTool.create(event_db_path=events)[0](
                CancelAgentsAction(task_ids=[task.task_id for task in tasks]), parent,
            )
    finally:
        configure_delegation(None)
        clear_github_credentials()


def wait_for_file(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.01)
    raise AssertionError(f"worker did not write {path}")


def test_merge_dispatch_is_independent_smart_private_and_reconciles_after_restart(
    merge_runtime, monkeypatch,
):
    config, parent, events, tool, action = merge_runtime
    start = Barrier(2)
    launched = []
    launch_lock = Lock()
    original_popen = delegation.subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        with launch_lock:
            launched.append(process.pid)
        return process

    def request_merge(_index):
        start.wait(timeout=5)
        return tool(action, parent).tasks[0]

    monkeypatch.setattr(delegation.subprocess, "Popen", popen)
    with ThreadPoolExecutor(max_workers=2) as executor:
        task, duplicate = list(executor.map(request_merge, range(2)))

    assert duplicate.task_id == task.task_id
    assert len(launched) == 1
    state = config.state_dir / "children" / task.task_id
    capture = wait_for_file(state / "capture.json")
    assert task.status == "running"
    assert not (state / "release.json").exists()
    assert tool(action, parent).tasks[0].task_id == task.task_id

    arguments = capture["argv"]
    assert arguments[arguments.index("--model") + 1] == "openai/gpt-5.6-sol"
    assert arguments[arguments.index("--reasoning-effort") + 1] == "max"
    for text in (
        "Keep the solver simple.", "The target requires the new evaluation fixture.",
        "The paired experiment improved accuracy.",
    ):
        assert text not in capture["context"]
    assert "SENPAI_PARENT_CONVERSATION_HISTORY_DIR" not in capture["environment"]
    assert capture["credentials"]["GITHUB_TOKEN"] == "private-merge-token"
    assert "private-merge-token" not in json.dumps(capture["environment"])
    assert "GITHUB_TOKEN" not in capture["environment"]
    assert "private-merge-token" not in json.dumps(arguments)
    assert MergeExperimentAction.model_validate_json(
        capture["environment"]["SENPAI_MERGE_REQUEST_JSON"]
    ) == action

    # The controller reconciles from another process without an in-memory runner.
    reconcile_delegated_tasks(config.state_dir, events)
    assert AgentStatusTool.create(event_db_path=events)[0](
        AgentStatusAction(task_ids=[task.task_id]), parent,
    ).tasks[0].status == "running"

    result = {"state": "experiment_merged", "version": "c" * 40}
    (state / "release.json").write_text(json.dumps(result))
    deadline = time.monotonic() + 5
    with LocalEventStore(events) as store:
        while not store.pending() and time.monotonic() < deadline:
            time.sleep(0.01)
        event = store.pending()[0]
    assert json.loads(event.payload["result"]) == result
    assert event.payload["parent_conversation_id"] == str(parent.id)
    assert tool(action, parent).tasks[0].task_id == task.task_id
    assert len(launched) == 1


def test_blocked_merge_can_retry_same_commit_without_duplicate_running_workers(merge_runtime):
    config, parent, events, tool, action = merge_runtime
    first = tool(action, parent).tasks[0]
    state = config.state_dir / "children" / first.task_id
    wait_for_file(state / "capture.json")
    (state / "release.json").write_text(json.dumps({
        "state": "merge_blocked", "reason": "The required fixture lacks justification.",
    }))
    AwaitAgentsTool.create(event_db_path=events)[0](
        AwaitAgentsAction(task_ids=[first.task_id], timeout_seconds=5), parent,
    )
    second = tool(action, parent).tasks[0]
    assert second.task_id != first.task_id
    assert tool(action, parent).tasks[0].task_id == second.task_id
