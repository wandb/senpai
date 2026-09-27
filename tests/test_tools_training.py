import json
import subprocess
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

    def run_training(self, spec: TrainingSpec) -> TrainingResult:
        self.launched.append(spec)
        return self.result

    def get_training_status(self, training_id: str) -> TrainingResult:
        self.status_checks.append(training_id)
        return self.result

    def cancel_training(self, training_id: str) -> TrainingResult:
        self.cancelled.append(training_id)
        return self.result.model_copy(update={"state": TrainingState.CANCELLED})

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
    training = StubTraining(workspace, finished_result(tmp_path))
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
        spec=TrainingSpec(
            argv=("python", "train.py"), cwd=workspace, timeout_seconds=20
        )
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
        assert training.launched == [action.spec]
        assert monitors.spec("training-17").conversation_id == old_conversation.id
        assert len(monitors.active()) == 1

        by_name["monitor_training"].executor(
            MonitorTrainingAction(training_id="training-17", stale_after_seconds=300),
            old_conversation,
        )
        by_name["cancel_training"].executor(
            CancelTrainingAction(training_id="training-17"), old_conversation
        )
        assert training.cancelled == ["training-17"]
        assert monitors.active() == []

        training.result = training.result.model_copy(update={"training_id": "training-18"})
        launch(action, current_conversation)
        assert training.launched == [action.spec, action.spec]
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
                    spec=TrainingSpec(
                        argv=("python", "train.py"), cwd=workspace, timeout_seconds=20
                    )
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
            RunTrainingAction(spec=spec),
            SimpleNamespace(id=conversation_id),
        )

        assert training.launched == [spec]
        assert observation.training_id == "training-17"
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
                spec=TrainingSpec(
                    argv=("python", "train.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                )
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
                    spec=TrainingSpec(
                        argv=("python", "candidate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
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
                    spec=TrainingSpec(
                        argv=("python", "train.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
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
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = assignment_runtime.conversation_id
    run_tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]
    monitor_tool = MonitorTrainingTool.create(training, monitors)[0]

    try:
        run_tool.executor(
            RunTrainingAction(
                spec=TrainingSpec(
                    argv=("python", "train.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                )
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
    finally:
        monitors.close()


def test_cancel_training_retires_its_monitor(tmp_path: Path, assignment_runtime):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = assignment_runtime.conversation_id
    RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor(
        RunTrainingAction(
            spec=TrainingSpec(
                argv=("python", "train.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        cancel = CancelTrainingTool.create(training, monitors)[0].executor
        with pytest.raises(PermissionError, match="different student conversation"):
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
        assert monitors.active() == []
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
            spec=TrainingSpec(
                argv=("python", "train.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        cancel = CancelTrainingTool.create(training, monitors)[0].executor

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
        spec=TrainingSpec(
            argv=("python", "train.py"),
            cwd=workspace,
            timeout_seconds=20,
        )
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


def test_registered_training_tools_share_one_runtime(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))
    register_senpai_tools()

    tools = resolve_tool(
        Tool(name="senpai_training", params={"state_dir": str(tmp_path / "state")}),
        state,
    )
    by_name = {tool.name: tool for tool in tools}

    try:
        assert set(by_name) == {
            "cancel_training",
            "run_training",
            "get_training_status",
            "monitor_training",
        }
        assert (
            by_name["run_training"].executor.training
            is by_name["get_training_status"].executor.training
        )
        assert (
            by_name["run_training"].executor.training
            is by_name["monitor_training"].executor.training
        )
        assert (
            by_name["run_training"].executor.training
            is by_name["cancel_training"].executor.training
        )
        assert (
            by_name["run_training"].executor.monitor_store
            is by_name["monitor_training"].executor.store
        )
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
