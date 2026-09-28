import json
import subprocess
import sys
import time
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from openhands.sdk.tool import Tool, resolve_tool
from pydantic import SecretStr

from github_workflow_support import FakeGitHub, assignment_record, pull_request
from senpai_agent import tools as training_tools
from senpai_agent import kubernetes_training
from training_test_support import FakeCluster, git_workspace
from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.tools import (
    clear_github_credentials,
    configure_github_credentials,
)
from senpai_agent.github.workflow import HttpResponse, WorkflowPreconditionError
from senpai_agent.models import render_assignment_marker
from senpai_agent.monitor import MonitorStore
from senpai_agent.state import AssignmentConversationRegistry
from senpai_agent.tools import (
    CancelTrainingAction,
    CancelTrainingTool,
    GetTrainingStatusAction,
    MonitorTrainingAction,
    MonitorTrainingTool,
    RunTrainingAction,
    RunTrainingTool,
    TrainingResultObservation,
    close_training_runtimes,
    register_senpai_tools,
)
from senpai_agent.training import TrainingResult, TrainingSpec, TrainingState
from senpai_agent.training_assignment import TrainingAssignmentGuard


class StubTraining:
    def __init__(self, workspace: Path, result: TrainingResult):
        self.workspace = workspace
        self.result = result
        self.launched: list[TrainingSpec] = []
        self.status_checks: list[str] = []
        self.cancelled: list[str] = []
        self.closed = False
        self.other_runs = []

    def run_training(self, spec: TrainingSpec, *, conversation_id=None) -> TrainingResult:
        self.launched.append(spec)
        return self.result

    def get_training_status(self, training_id: str) -> TrainingResult:
        self.status_checks.append(training_id)
        return self.result

    def cancel_training(self, training_id: str) -> TrainingResult:
        self.cancelled.append(training_id)
        return self.result.model_copy(update={"state": TrainingState.CANCELLED})

    def active_runs(self, *, exclude=None):
        return self.other_runs

    def close(self) -> None:
        self.closed = True


def init_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    return workspace


def finished_result(tmp_path: Path) -> TrainingResult:
    return TrainingResult(
        training_id="training-17",
        state=TrainingState.FINISHED,
        exit_code=0,
        elapsed_seconds=12.5,
        log_path=str(tmp_path / "training.log"),
        wandb_run_ids=("run-abc",),
    )


@pytest.fixture
def assignment_runtime(tmp_path, monkeypatch):
    registry = AssignmentConversationRegistry(tmp_path / "student-conversations.json")
    github = FakeGitHub(pull_request(labels={"student:student-one", "status:wip"}))
    configure_github_credentials("acme/widgets", SecretStr("github-secret"))
    monkeypatch.setenv("STUDENT_NAME", "student-one")

    @contextmanager
    def urlopen(request, timeout):
        response = github.request(
            request.get_method(), request.full_url, headers=dict(request.header_items())
        )
        yield SimpleNamespace(
            headers={},
            read=lambda: json.dumps(response.json_body).encode(),
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    try:
        yield SimpleNamespace(
            registry=registry,
            guard=TrainingAssignmentGuard(registry.path, "student-one"),
            github=github,
            conversation_id=registry.for_assignment("assignment-7", "revision-1"),
        )
    finally:
        clear_github_credentials()


def test_training_rechecks_live_revision_before_each_launch(
    tmp_path: Path, monkeypatch, assignment_runtime
):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path).model_copy(update={"state": TrainingState.RUNNING}))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    monkeypatch.setattr(
        training_tools, "training_runtime", lambda *args: (training, monitors)
    )
    register_senpai_tools()
    tools = resolve_tool(
        Tool(name="senpai_training", params={"state_dir": str(tmp_path / "training")}),
        SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace)),
    )
    by_name = {tool.name: tool for tool in tools}
    launch = by_name["run_training"].executor
    action = RunTrainingAction(
        argv=("python", "train.py"), cwd=workspace, timeout_seconds=20
    )
    old_conversation = SimpleNamespace(id=assignment_runtime.conversation_id)
    current_conversation = SimpleNamespace(
        id=assignment_runtime.registry.for_assignment("assignment-7", "revision-2")
    )
    try:
        launch(action, old_conversation)
        assignment_runtime.github.pr["body"] = render_assignment_marker(
            assignment_record(revision_id="revision-2")
        )
        with pytest.raises(PermissionError, match="revision-1.*revision-2"):
            launch(action, old_conversation)
        assert training.launched == [TrainingSpec.model_validate(action.model_dump(exclude={"kind"}))]
        assert monitors.spec("training-17").conversation_id == old_conversation.id
        assert len(monitors.active()) == 1

        by_name["monitor_training"].executor(
            MonitorTrainingAction(training_id="training-17", stale_after_seconds=300),
            old_conversation,
        )
        by_name["cancel_training"].executor(
            CancelTrainingAction(training_id="training-17"), current_conversation
        )
        assert training.cancelled == ["training-17"]
        assert monitors.active() == []

        training.result = training.result.model_copy(update={"training_id": "training-18"})
        launch(action, current_conversation)
        assert training.launched == [TrainingSpec.model_validate(action.model_dump(exclude={"kind"})), TrainingSpec.model_validate(action.model_dump(exclude={"kind"}))]
        assert monitors.spec("training-18").conversation_id == current_conversation.id
    finally:
        monitors.close()


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("unbound_conversation", PermissionError),
        ("no_wip_assignment", PermissionError),
        ("duplicate_wip_assignments", PermissionError),
        ("wrong_student_marker", WorkflowPreconditionError),
        ("github_unavailable", GitHubReadError),
    ],
)
def test_training_denies_launch_when_current_assignment_cannot_be_verified(
    tmp_path: Path, monkeypatch, assignment_runtime, failure, error
):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation = SimpleNamespace(id=assignment_runtime.conversation_id)
    github = assignment_runtime.github
    if failure == "unbound_conversation":
        conversation.id = uuid.uuid4()
    elif failure == "no_wip_assignment":
        github.pr["labels"] = {"student:student-one", "status:review"}
    elif failure == "duplicate_wip_assignments":
        request = github.request

        def duplicate(method, url, **kwargs):
            response = request(method, url, **kwargs)
            return HttpResponse(
                200, [response.json_body[0], {**response.json_body[0], "number": 8}]
            )

        monkeypatch.setattr(github, "request", duplicate)
    elif failure == "wrong_student_marker":
        github.pr["body"] = render_assignment_marker(
            assignment_record(student="student-two")
        )
    elif failure == "github_unavailable":

        def unavailable(*args, **kwargs):
            raise URLError("GitHub unavailable")

        monkeypatch.setattr("urllib.request.urlopen", unavailable)
    registry_before = assignment_runtime.registry.path.read_bytes()
    launch = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]
    try:
        with pytest.raises(error):
            launch.executor(
                RunTrainingAction(
                    argv=("python", "train.py"), cwd=workspace, timeout_seconds=20
                ),
                conversation,
            )
        assert training.launched == []
        assert monitors.active() == []
        assert assignment_runtime.registry.path.read_bytes() == registry_before
    finally:
        monitors.close()


def test_run_training_registers_a_monitor_for_its_conversation(
    tmp_path: Path, assignment_runtime
):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    training.other_runs = [{"training_id": "existing-run", "state": "running",
                            "nodes": 1, "gpus_per_node": 2, "deadline_at": None}]
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]
    conversation_id = assignment_runtime.conversation_id
    spec = TrainingSpec(
        argv=("python", "train.py"),
        cwd=workspace,
        timeout_seconds=600,
    )

    try:
        observation = tool.executor(
            RunTrainingAction(**spec.model_dump()),
            SimpleNamespace(id=conversation_id),
        )

        assert training.launched == [spec]
        assert observation.training_id == "training-17"
        assert json.loads(observation.to_llm_content[0].text)["other_active_runs"] == training.other_runs
        assert training.cancelled == []
        assert observation.wandb_run_ids == ("run-abc",)
        monitor = monitors.spec("training-17")
        assert monitor.conversation_id == conversation_id
        assert monitor.metric is None
        assert monitor.gates == ()
    finally:
        monitors.close()


@pytest.mark.parametrize("field", ["error_tail", "kubernetes_diagnostics"])
def test_training_diagnostics_mask_registered_secrets(
    tmp_path: Path, field, assignment_runtime
):
    workspace = init_workspace(tmp_path)
    secret = "private-training-token"
    result = finished_result(tmp_path).model_copy(
        update={field: f"authentication failed for {secret}"}
    )
    training = StubTraining(workspace, result)
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]

    class SecretRegistry:
        def mask_secrets_in_output(self, text: str) -> str:
            return text.replace(secret, "<secret-hidden>")

    conversation = SimpleNamespace(
        id=assignment_runtime.conversation_id,
        state=SimpleNamespace(secret_registry=SecretRegistry()),
    )

    try:
        observation = tool.executor(
            RunTrainingAction(
                argv=("python", "train.py"),
                cwd=workspace,
                timeout_seconds=20,
            ),
            conversation,
        )

        assert getattr(observation, field) == "authentication failed for <secret-hidden>"
        assert secret not in observation.to_llm_content[0].text
        assert getattr(result, field).endswith(secret)
    finally:
        monitors.close()


@pytest.mark.parametrize(
    "conversation",
    [None, SimpleNamespace(state=SimpleNamespace())],
)
def test_training_error_tail_requires_the_conversation_secret_registry(
    tmp_path: Path,
    conversation,
):
    result = finished_result(tmp_path).model_copy(
        update={"error_tail": "unredacted training failure"}
    )

    with pytest.raises(RuntimeError, match="conversation secret registry"):
        TrainingResultObservation.from_result(result, conversation)


def test_run_training_requires_a_clean_worktree_before_starting(
    tmp_path: Path, assignment_runtime
):
    workspace = init_workspace(tmp_path)
    (workspace / "candidate.py").write_text("print('uncommitted')\n")
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]

    try:
        with pytest.raises(RuntimeError, match="clean before training"):
            tool.executor(
                RunTrainingAction(
                    argv=("python", "candidate.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert training.launched == []
        assert monitors.active() == []
    finally:
        monitors.close()


def test_run_training_requires_a_conversation_before_starting(
    tmp_path: Path, assignment_runtime
):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]

    try:
        with pytest.raises(ValueError, match="student conversation"):
            tool.executor(
                RunTrainingAction(
                    argv=("python", "train.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                )
            )

        assert training.launched == []
        assert monitors.active() == []
    finally:
        monitors.close()


def test_monitor_training_validates_the_training_id_before_registration(
    tmp_path: Path,
):
    class MissingTraining(StubTraining):
        def get_training_status(self, training_id: str) -> TrainingResult:
            self.status_checks.append(training_id)
            raise KeyError(training_id)

    workspace = tmp_path / "workspace"
    training = MissingTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = MonitorTrainingTool.create(training, monitors)[0]

    try:
        with pytest.raises(KeyError, match="missing-training"):
            tool.executor(
                MonitorTrainingAction(training_id="missing-training"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert training.status_checks == ["missing-training"]
        assert monitors.active() == []
    finally:
        monitors.close()


def test_monitor_training_replaces_the_default_policy(
    tmp_path: Path, assignment_runtime
):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path).model_copy(update={"state": TrainingState.RUNNING}))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = assignment_runtime.conversation_id
    run_tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]
    monitor_tool = MonitorTrainingTool.create(training, monitors)[0]

    try:
        run_tool.executor(
            RunTrainingAction(
                argv=("python", "train.py"),
                cwd=workspace,
                timeout_seconds=20,
            ),
            SimpleNamespace(id=conversation_id),
        )
        action = MonitorTrainingAction(
            training_id="training-17",
            metric="validation/loss",
            direction="min",
            stale_after_seconds=300,
        )
        with pytest.raises(PermissionError, match="different student conversation"):
            monitor_tool.executor(action, SimpleNamespace(id=uuid.uuid4()))
        observation = monitor_tool.executor(
            action,
            SimpleNamespace(id=conversation_id),
        )

        monitor = monitors.spec("training-17")
        assert training.status_checks == ["training-17", "training-17"]
        assert monitor.metric == "validation/loss"
        assert monitor.direction == "min"
        assert monitor.stale_after_seconds == 300
        assert observation.to_llm_content[0].text == (
            "Training training-17 is durably monitored. You may finish this turn; "
            "the controller will resume this same conversation "
            f"({conversation_id}) when action is needed."
        )
        training.result = training.result.model_copy(update={"state": TrainingState.FINISHED})
        with pytest.raises(ValueError, match="completed run"):
            monitor_tool.executor(action, SimpleNamespace(id=conversation_id))
    finally:
        monitors.close()


@pytest.mark.parametrize("released", [True, False])
def test_cancel_training_retires_monitor_only_after_cleanup(tmp_path: Path, assignment_runtime, released):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path).model_copy(
        update={"kubernetes_released": released},
    ))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = assignment_runtime.conversation_id
    RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor(
        RunTrainingAction(
            argv=("python", "train.py"),
            cwd=workspace,
            timeout_seconds=20,
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        cancel = CancelTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor
        with pytest.raises(PermissionError, match="training conversation is bound"):
            cancel(
                CancelTrainingAction(training_id="training-17"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        observation = cancel(
            CancelTrainingAction(training_id="training-17"),
            SimpleNamespace(id=conversation_id),
        )

        assert observation.state is TrainingState.CANCELLED
        assert training.cancelled == ["training-17"]
        assert bool(monitors.active()) is (not released)
    finally:
        monitors.close()


def test_cancel_training_keeps_monitor_when_cancellation_is_not_terminal(
    tmp_path: Path,
    assignment_runtime,
):
    class NonTerminalCancellation(StubTraining):
        def cancel_training(self, training_id: str) -> TrainingResult:
            self.cancelled.append(training_id)
            return self.result.model_copy(update={"state": TrainingState.RUNNING})

    workspace = init_workspace(tmp_path)
    training = NonTerminalCancellation(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = assignment_runtime.conversation_id
    RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor(
        RunTrainingAction(
            argv=("python", "train.py"),
            cwd=workspace,
            timeout_seconds=20,
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        cancel = CancelTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor

        with pytest.raises(RuntimeError, match="did not reach a terminal state"):
            cancel(
                CancelTrainingAction(training_id="training-17"),
                SimpleNamespace(id=conversation_id),
            )

        assert training.cancelled == ["training-17"]
        assert [monitor.training_id for monitor in monitors.active()] == [
            "training-17"
        ]
    finally:
        monitors.close()


def test_interrupting_run_training_cancels_only_the_in_flight_run(
    tmp_path: Path, assignment_runtime
):
    class BlockingMonitorStore(MonitorStore):
        def __init__(self, path: Path):
            super().__init__(path)
            self.registering = threading.Event()
            self.release = threading.Event()

        def register(self, spec):
            self.registering.set()
            assert self.release.wait(2)
            super().register(spec)

    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = BlockingMonitorStore(tmp_path / "monitors.sqlite3")
    executor = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor
    action = RunTrainingAction(
        argv=("python", "train.py"),
        cwd=workspace,
        timeout_seconds=20,
    )
    conversation = SimpleNamespace(id=assignment_runtime.conversation_id)
    errors = []

    def run() -> None:
        try:
            executor(action, conversation)
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = threading.Thread(target=run)

    try:
        executor.interrupt()
        assert training.cancelled == []

        thread.start()
        assert monitors.registering.wait(2)
        executor.interrupt()
        monitors.release.set()
        thread.join(2)

        assert not thread.is_alive()
        assert errors == []
        assert training.cancelled == ["training-17"]
        assert training.closed is False
    finally:
        monitors.release.set()
        thread.join(2)
        monitors.close()


@pytest.mark.parametrize(("nodes", "kind"), [(1, "Job"), (2, "MPIJob")])
def test_registered_training_tools_supervise_kubernetes_for_every_topology(
    tmp_path: Path, monkeypatch, assignment_runtime, nodes, kind,
):
    workspace = git_workspace(tmp_path)
    client = FakeCluster(TrainingState.RUNNING, nodes=nodes)
    monkeypatch.setattr(kubernetes_training, "KubernetesExecutorClient", lambda _socket: client)
    for key, value in {
        "NODES_PER_STUDENT": str(nodes),
        "GPUS_PER_STUDENT_NODE": "8",
        "RESEARCH_TAG": "fred",
        "SENPAI_KUBERNETES_NAMESPACE": "research",
        "SENPAI_LAUNCH_SECRET_NAME": "launch-secrets",
        "SENPAI_TRAINING_SNAPSHOT_ROOT": str(tmp_path / "snapshots"),
        "SENPAI_TRAINING_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "SENPAI_TRAINING_IMAGE": "ghcr.io/wandb/senpai-student@sha256:" + "a" * 64,
        "SENPAI_TRAINING_CONTROL_IMAGE": "ghcr.io/wandb/senpai-student@sha256:" + "a" * 64,
        "CPU_PER_STUDENT_GPU": "1", "MEMORY_GI_PER_STUDENT_GPU": "2",
        "PVC_CLAIM_NAME": "dataset", "PVC_MOUNT_PATH": str(tmp_path / "data"),
        "WANDB_ENTITY": "entity", "WANDB_PROJECT": "project",
    }.items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))
    conversation = SimpleNamespace(
        id=assignment_runtime.conversation_id,
        state=SimpleNamespace(secret_registry=SimpleNamespace(mask_secrets_in_output=lambda text: text)),
    )
    register_senpai_tools()
    tools = resolve_tool(
        Tool(name="senpai_training", params={"state_dir": str(tmp_path / "state")}),
        state,
    )
    by_name = {tool.name: tool for tool in tools}
    try:
        started = by_name["run_training"].executor(
            RunTrainingAction(
                argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=30,
            ), conversation,
        )
        assert client.reservations[0][0] == started.training_id
        assert client.spec.kind == kind
        deadline = time.monotonic() + 5
        while True:
            status = by_name["get_training_status"].executor(
                GetTrainingStatusAction(training_id=started.training_id), conversation,
            )
            if status.kubernetes_resource is not None:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert status.state is TrainingState.RUNNING
        assert status.kubernetes_resource.kind == kind
        assert status.kubernetes_resource.nodes == nodes
        monitors = by_name["monitor_training"].executor.store
        assert monitors.spec(started.training_id).conversation_id == conversation.id
        cancelled = by_name["cancel_training"].executor(
            CancelTrainingAction(training_id=started.training_id), conversation,
        )
        assert cancelled.state is TrainingState.CANCELLED
        assert cancelled.kubernetes_released is True
        assert client.deletions == [status.kubernetes_resource]
        assert client.releases == [started.training_id]
        assert monitors.active() == []
    finally:
        close_training_runtimes()


def test_get_training_status_delivers_complete_structured_pod_receipt(tmp_path):
    from senpai_agent.tools import GetTrainingStatusAction, GetTrainingStatusTool

    receipt = {
        'training_id': 'training-17', 'source_commit': 'a' * 40,
        'resource': {'uid': 'exact-mpi-uid'}, 'complete': True,
        'pods': [{
            'uid': f'worker-{index}-uid', 'node': f'gpu-node-{index}', 'phase': 'Succeeded',
            'containers': [{'name': 'train', 'restartCount': 0,
                            'state': {'terminated': {'exitCode': 0, 'startedAt': '2026-09-26T01:00:00Z',
                                                     'finishedAt': '2026-09-26T02:00:00Z'}},
                            'lastState': {}}],
        } for index in range(32)],
    }
    assert len(json.dumps(receipt)) > 8192
    result = finished_result(tmp_path).model_copy(update={'kubernetes_pod_receipt': receipt})
    training = StubTraining(tmp_path, result)
    tool = GetTrainingStatusTool.create(training)[0]
    observed = tool.executor(GetTrainingStatusAction(training_id='training-17'))
    delivered = json.loads(observed.to_llm_content[0].text)
    assert delivered['kubernetes_pod_receipt'] == receipt
    assert training.status_checks == ['training-17']


def test_training_tool_accepts_flat_command_and_optional_limits(tmp_path):
    action = RunTrainingAction.model_validate({
        "argv": ["python", "train.py"], "cwd": str(tmp_path),
        "nodes": 1, "gpus_per_node": 2,
    })
    assert action.argv == ("python", "train.py")
    assert action.timeout_seconds is None
    restored = RunTrainingAction.model_validate({
        "kind": "RunTrainingAction", "spec": {"argv": ["python"], "cwd": str(tmp_path), "timeout_seconds": 30},
    })
    assert restored.argv == ("python",) and restored.timeout_seconds == 30
    assert "spec" not in restored.model_dump()
    assert (action.nodes, action.gpus_per_node) == (1, 2)
    with pytest.raises(ValueError):
        RunTrainingAction.model_validate({"argv": [], "cwd": str(tmp_path)})
    with pytest.raises(ValueError):
        RunTrainingAction.model_validate({
            "argv": ["python"], "cwd": str(tmp_path), "gpus_per_node": 0,
        })


def test_training_runtime_recovers_missing_monitor_without_rearming_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("NODES_PER_STUDENT", "1")
    monkeypatch.setenv("GPUS_PER_STUDENT_NODE", "1")
    state = tmp_path / "state"
    state.mkdir()
    owner = uuid.uuid4()
    training_id = str(uuid.uuid4())
    result = finished_result(tmp_path).model_copy(update={
        "training_id": training_id, "conversation_id": owner, "kubernetes_released": True,
    })
    (state / f"{training_id}.json").write_text(result.model_dump_json())
    try:
        _, store = training_tools.training_runtime(tmp_path, state)
        assert store.spec(training_id).conversation_id == owner
        assert [item.training_id for item in store.active()] == [training_id]
        store.complete(training_id)
        close_training_runtimes()
        _, recovered = training_tools.training_runtime(tmp_path, state)
        assert recovered.active() == []
    finally:
        close_training_runtimes()


def test_registration_failure_cancels_run_and_current_assignment_can_reclaim_it(
    tmp_path, assignment_runtime, monkeypatch,
):
    training = StubTraining(init_workspace(tmp_path), finished_result(tmp_path).model_copy(
        update={"kubernetes_released": False},
    ))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation = SimpleNamespace(id=assignment_runtime.conversation_id)

    def unavailable(_spec):
        raise OSError("monitor storage unavailable")

    monkeypatch.setattr(monitors, "register", unavailable)
    try:
        run = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor
        with pytest.raises(RuntimeError, match="Training training-17 started but monitor registration failed"):
            run(RunTrainingAction(argv=("python", "train.py"), cwd=training.workspace), conversation)
        assert training.cancelled == ["training-17"]
        cancel = CancelTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor
        with pytest.raises(PermissionError):
            cancel(CancelTrainingAction(training_id="training-17"), SimpleNamespace(id=uuid.uuid4()))
        result = cancel(CancelTrainingAction(training_id="training-17"), conversation)
        assert result.state is TrainingState.CANCELLED
        assert "cleanup is pending" in result.monitor_warning
        assert training.cancelled == ["training-17", "training-17"]
    finally:
        monitors.close()
