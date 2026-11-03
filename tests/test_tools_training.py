import json
import subprocess
import threading
import time
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
        self.worktrees: list[object] = []
        self.status_checks: list[str] = []
        self.cancelled: list[str] = []
        self.closed = False

    def run_training(self, spec: TrainingSpec, *, worktree=None) -> TrainingResult:
        self.launched.append(spec)
        self.worktrees.append(worktree)
        return self.result

    def get_training_status(self, training_id: str) -> TrainingResult:
        self.status_checks.append(training_id)
        return self.result

    def cancel_training(self, training_id: str) -> TrainingResult:
        self.cancelled.append(training_id)
        return self.result.model_copy(update={"state": TrainingState.CANCELLED})

    def close(self) -> None:
        self.closed = True


class SequencedTraining(StubTraining):
    """Return each queued record once, then the last one for every later check."""

    def __init__(self, workspace: Path, results: list[TrainingResult]):
        super().__init__(workspace, results[-1])
        self.results = list(results)

    def get_training_status(self, training_id: str) -> TrainingResult:
        self.status_checks.append(training_id)
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


def init_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "Student"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "student@example.com"], cwd=workspace, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "baseline"], cwd=workspace, check=True
    )
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
        with pytest.raises(PermissionError, match="revision-1.*revision-2") as refused:
            launch(action, old_conversation)
        assert "finish this turn" in str(refused.value)
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


def committed_workspace(tmp_path: Path) -> Path:
    workspace = init_workspace(tmp_path)
    (workspace / "candidate.py").write_text("print('committed')\n")
    subprocess.run(["git", "add", "candidate.py"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=workspace, check=True)
    return workspace


def test_run_training_refuses_a_modified_tracked_file_and_names_it(
    tmp_path: Path, assignment_runtime
):
    workspace = committed_workspace(tmp_path)
    (workspace / "candidate.py").write_text("print('uncommitted')\n")
    (workspace / "results.json").write_text("{}\n")
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]

    try:
        with pytest.raises(RuntimeError) as error:
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

        assert "uncommitted changes to tracked files: candidate.py" in str(error.value)
        assert "results.json" not in str(error.value)
        assert training.launched == []
        assert monitors.active() == []
    finally:
        monitors.close()


def test_run_training_starts_with_untracked_lane_output_and_records_it(
    tmp_path: Path, assignment_runtime
):
    workspace = committed_workspace(tmp_path)
    (workspace / "lane").mkdir()
    (workspace / "lane" / "results.json").write_text("{}\n")
    (workspace / "lane" / "scratch.py").write_text("pass\n")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0]

    try:
        tool.executor(
            RunTrainingAction(
                spec=TrainingSpec(
                    argv=("python", "candidate.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                )
            ),
            SimpleNamespace(id=assignment_runtime.conversation_id),
        )
    finally:
        monitors.close()

    assert len(training.launched) == 1
    [worktree] = training.worktrees
    assert worktree.head_sha == head
    assert worktree.untracked_paths == ("lane/results.json", "lane/scratch.py")
    assert worktree.snapshot_sha is not None
    stored = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", worktree.snapshot_sha],
        cwd=workspace,
        text=True,
    ).split()
    assert stored == ["candidate.py", "lane/results.json", "lane/scratch.py"]


def test_training_observation_names_untracked_files_and_the_snapshot(tmp_path: Path):
    paths = tuple(f"lane/out-{index:02d}.json" for index in range(25))
    result = finished_result(tmp_path).model_copy(
        update={
            "source_commit": "a" * 40,
            "worktree_snapshot": "b" * 40,
            "untracked_paths": paths,
        }
    )

    delivered = json.loads(
        TrainingResultObservation.from_result(result, None).to_llm_content[0].text
    )

    assert delivered["source_commit"] == "a" * 40
    assert delivered["worktree_snapshot"] == "b" * 40
    assert delivered["untracked_paths"] == list(paths[:20])
    assert delivered["untracked_count"] == 25
    assert "git show --stat " + "b" * 40 in delivered["worktree_note"]

    clean = json.loads(
        TrainingResultObservation.from_result(
            finished_result(tmp_path), None
        ).to_llm_content[0].text
    )
    assert "untracked_paths" not in clean
    assert "worktree_note" not in clean


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


def test_handed_over_training_is_cancelled_by_the_current_revision(
    tmp_path: Path, assignment_runtime
):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    previous = assignment_runtime.conversation_id
    current = assignment_runtime.registry.for_assignment("assignment-7", "revision-2")
    RunTrainingTool.create(training, monitors, assignment_runtime.guard)[0].executor(
        RunTrainingAction(
            spec=TrainingSpec(
                argv=("python", "train.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        SimpleNamespace(id=previous),
    )

    try:
        monitors.reassign("training-17", current)
        cancel = CancelTrainingTool.create(training, monitors)[0].executor
        with pytest.raises(PermissionError, match="different student conversation"):
            cancel(
                CancelTrainingAction(training_id="training-17"),
                SimpleNamespace(id=previous),
            )

        cancel(
            CancelTrainingAction(training_id="training-17"),
            SimpleNamespace(id=current),
        )

        assert training.cancelled == ["training-17"]
        assert monitors.active() == []
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


@pytest.mark.parametrize(
    ("nodes", "training_image", "remote"),
    [(1, "", False), (1, "training@sha256:" + "a" * 64, True), (2, "", True)],
)
def test_registered_training_tools_share_one_runtime(
    tmp_path: Path, monkeypatch, nodes, training_image, remote,
):
    monkeypatch.setenv("NODES_PER_STUDENT", str(nodes))
    monkeypatch.setenv("GPUS_PER_STUDENT_NODE", "1")
    monkeypatch.setenv("SENPAI_TRAINING_IMAGE", training_image)
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
        assert isinstance(
            by_name["run_training"].executor.training,
            training_tools.KubernetesTrainingSupervisor,
        ) is remote
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


@pytest.mark.parametrize("field", ["kubernetes_pod_receipt", "kubernetes_pod_snapshot"])
def test_get_training_status_delivers_complete_structured_pod_evidence(tmp_path, field):
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
    result = finished_result(tmp_path).model_copy(update={field: receipt})
    training = StubTraining(tmp_path, result)
    tool = GetTrainingStatusTool.create(training)[0]
    observed = tool.executor(GetTrainingStatusAction(training_id='training-17'))
    delivered = json.loads(observed.to_llm_content[0].text)
    assert delivered[field] == receipt
    assert training.status_checks == ['training-17']


def running_result(tmp_path: Path) -> TrainingResult:
    return finished_result(tmp_path).model_copy(
        update={"state": TrainingState.RUNNING, "exit_code": None}
    )


def status_tool(training, monkeypatch: pytest.MonkeyPatch):
    from senpai_agent.tools import GetTrainingStatusTool

    monkeypatch.setattr(training_tools, "_STATUS_POLL_SECONDS", 0.02)
    return GetTrainingStatusTool.create(training)[0]


def test_get_training_status_waits_for_a_terminal_state(tmp_path: Path, monkeypatch):
    from senpai_agent.tools import GetTrainingStatusAction

    running = running_result(tmp_path)
    training = SequencedTraining(tmp_path, [running, running, finished_result(tmp_path)])
    tool = status_tool(training, monkeypatch)

    started = time.monotonic()
    observed = tool.executor(
        GetTrainingStatusAction(training_id="training-17", wait_seconds=30)
    )

    assert observed.state is TrainingState.FINISHED
    assert observed.status_note == ""
    assert training.status_checks == ["training-17"] * 3
    assert time.monotonic() - started < 5


def test_get_training_status_wait_is_bounded_and_says_to_finish_the_turn(
    tmp_path: Path, monkeypatch
):
    from senpai_agent.tools import GetTrainingStatusAction

    training = StubTraining(tmp_path, running_result(tmp_path))
    tool = status_tool(training, monkeypatch)

    started = time.monotonic()
    observed = tool.executor(
        GetTrainingStatusAction(training_id="training-17", wait_seconds=1)
    )
    waited = time.monotonic() - started

    assert observed.state is TrainingState.RUNNING
    assert 1 <= waited < 5
    delivered = json.loads(observed.to_llm_content[0].text)
    assert delivered["status_note"].startswith("Still running after 1 s.")
    assert "finish the turn" in delivered["status_note"]
    assert "wait_seconds (up to 600)" in delivered["status_note"]
    assert len(training.status_checks) > 10


def test_repeated_status_checks_on_a_running_run_are_paced(tmp_path: Path, monkeypatch):
    from senpai_agent.tools import GetTrainingStatusAction

    monkeypatch.setattr(training_tools, "STATUS_REPEAT_SECONDS", 1.5)
    running = running_result(tmp_path)
    training = SequencedTraining(
        tmp_path, [running] * 400 + [finished_result(tmp_path)]
    )
    tool = status_tool(training, monkeypatch)
    action = GetTrainingStatusAction(training_id="training-17")

    started = time.monotonic()
    first = tool.executor(action)
    first_took = time.monotonic() - started
    # A short wait on a repeat is paced too: the floor is the repeat interval.
    second = tool.executor(GetTrainingStatusAction(training_id="training-17", wait_seconds=1))
    second_took = time.monotonic() - started - first_took

    assert first.state is TrainingState.RUNNING
    assert first_took < 0.3
    assert first.status_note.startswith("Still running. ")
    assert "repeat call" not in first.status_note
    assert second.state is TrainingState.RUNNING
    assert 1.3 <= second_took < 4
    assert second.status_note.startswith("This repeat call waited")
    assert "returns the latest record" in second.status_note

    training.results = [finished_result(tmp_path)]
    started = time.monotonic()
    third = tool.executor(action)
    assert third.state is TrainingState.FINISHED
    assert time.monotonic() - started < 0.3
    # A run that ended is forgotten; a check of another run is never paced.
    assert tool.executor(GetTrainingStatusAction(training_id="training-18")).state is (
        TrainingState.FINISHED
    )


def test_a_log_stall_during_the_wait_ends_it(tmp_path: Path, monkeypatch):
    from senpai_agent.tools import GetTrainingStatusAction

    running = running_result(tmp_path)
    stalled = running.model_copy(update={"log_stalled_since": 1_700_000_000.0})
    training = SequencedTraining(tmp_path, [running, running, stalled])
    tool = status_tool(training, monkeypatch)

    started = time.monotonic()
    observed = tool.executor(
        GetTrainingStatusAction(training_id="training-17", wait_seconds=30)
    )

    assert observed.state is TrainingState.RUNNING
    assert time.monotonic() - started < 5
    assert training.status_checks == ["training-17"] * 3
    assert "Its log stopped growing during the wait" in observed.status_note


def test_a_terminal_record_read_by_its_own_conversation_retires_the_monitor(
    tmp_path: Path, monkeypatch
):
    from senpai_agent.monitor import TrainingMonitorSpec
    from senpai_agent.tools import GetTrainingStatusAction, GetTrainingStatusTool

    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    owner = uuid.uuid4()
    monitors.register(TrainingMonitorSpec(training_id="training-17", conversation_id=owner))
    training = SequencedTraining(
        tmp_path, [running_result(tmp_path), finished_result(tmp_path)]
    )
    monkeypatch.setattr(training_tools, "_STATUS_POLL_SECONDS", 0.02)
    status = GetTrainingStatusTool.create(training, monitor_store=monitors)[0].executor
    action = GetTrainingStatusAction(training_id="training-17")

    try:
        assert status(action, SimpleNamespace(id=owner)).state is TrainingState.RUNNING
        assert len(monitors.active()) == 1
        # Another conversation reading the terminal record leaves the wake alone.
        assert status(action, SimpleNamespace(id=uuid.uuid4())).state is TrainingState.FINISHED
        assert len(monitors.active()) == 1

        assert status(action, SimpleNamespace(id=owner)).state is TrainingState.FINISHED

        assert monitors.active() == []
    finally:
        monitors.close()


def test_interrupting_a_status_wait_returns_the_running_record(
    tmp_path: Path, monkeypatch
):
    from senpai_agent.tools import GetTrainingStatusAction

    training = StubTraining(tmp_path, running_result(tmp_path))
    tool = status_tool(training, monkeypatch)
    observed: list[TrainingResultObservation] = []
    thread = threading.Thread(
        target=lambda: observed.append(
            tool.executor(
                GetTrainingStatusAction(training_id="training-17", wait_seconds=60)
            )
        )
    )

    started = time.monotonic()
    thread.start()
    time.sleep(0.2)
    tool.executor.interrupt()
    thread.join(5)

    assert not thread.is_alive()
    assert time.monotonic() - started < 3
    assert observed[0].state is TrainingState.RUNNING


def test_status_wait_is_bounded_by_the_schema():
    from pydantic import ValidationError

    from senpai_agent.tools import GetTrainingStatusAction

    assert GetTrainingStatusAction(training_id="t", wait_seconds=600).wait_seconds == 600
    for value in (601, -1):
        with pytest.raises(ValidationError):
            GetTrainingStatusAction(training_id="t", wait_seconds=value)


def test_log_stall_interval_defaults_to_half_an_hour():
    from senpai_agent.training import DEFAULT_LOG_STALL_SECONDS, log_stall_seconds

    assert DEFAULT_LOG_STALL_SECONDS == 1800
    assert log_stall_seconds({}) == 1800
    assert log_stall_seconds({"SENPAI_TRAINING_LOG_STALL_SECONDS": " "}) == 1800
    assert log_stall_seconds({"SENPAI_TRAINING_LOG_STALL_SECONDS": "2400"}) == 2400


def test_training_runtime_reads_the_log_stall_interval_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NODES_PER_STUDENT", raising=False)
    monkeypatch.delenv("SENPAI_TRAINING_IMAGE", raising=False)
    monkeypatch.setenv("SENPAI_TRAINING_LOG_STALL_SECONDS", "90")
    workspace = init_workspace(tmp_path)

    try:
        supervisor, _ = training_tools.training_runtime(workspace, tmp_path / "state")
        assert supervisor.log_stall_seconds == 90
    finally:
        close_training_runtimes()

    monkeypatch.setenv("SENPAI_TRAINING_LOG_STALL_SECONDS", "0")
    with pytest.raises(RuntimeError, match="positive"):
        training_tools.training_runtime(workspace, tmp_path / "other-state")


def test_training_observation_reports_a_stalled_log(tmp_path: Path):
    result = finished_result(tmp_path).model_copy(
        update={
            "state": TrainingState.RUNNING,
            "exit_code": None,
            "log_stalled_since": 1_700_000_000.0,
            "last_log_line": "step 3 of 100",
        }
    )
    conversation = SimpleNamespace(
        state=SimpleNamespace(
            secret_registry=SimpleNamespace(
                mask_secrets_in_output=lambda text: text.replace("100", "<n>")
            )
        )
    )

    observed = TrainingResultObservation.from_result(result, conversation)
    delivered = json.loads(observed.to_llm_content[0].text)

    assert delivered["log_stalled_since"] == 1_700_000_000.0
    assert delivered["last_log_line"] == "step 3 of <n>"

    with pytest.raises(RuntimeError, match="secret registry"):
        TrainingResultObservation.from_result(result, None)
