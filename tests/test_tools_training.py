import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import SecretRegistry
from openhands.sdk.tool import Tool, resolve_tool

from senpai_agent.monitor import MonitorStore, TrainingMonitorSpec
from senpai_agent.tools import (
    CancelJobAction,
    CancelJobTool,
    GetJobStatusAction,
    GetJobStatusTool,
    JobSpec,
    MonitorJobAction,
    MonitorJobTool,
    RunJobAction,
    RunJobTool,
    close_training_runtimes,
    register_senpai_tools,
)
from senpai_agent.training import TrainingResult, TrainingSpec, TrainingState


class StubTraining:
    def __init__(self, workspace: Path, result: TrainingResult):
        self.workspace = workspace
        self.result = result
        self.launched: list[TrainingSpec] = []
        self.status_checks: list[str] = []
        self.cancelled: list[str] = []
        self.environments: list[dict[str, str]] = []
        self.redacted_values: list[tuple[str, ...]] = []
        self.closed = False

    def run_training(
        self,
        spec: TrainingSpec,
        *,
        env=None,
        redacted_values=(),
    ) -> TrainingResult:
        self.launched.append(spec)
        self.environments.append(dict(env or {}))
        self.redacted_values.append(tuple(redacted_values))
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


def test_run_job_registers_a_monitor_for_its_conversation(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunJobTool.create(training, monitors)[0]
    conversation_id = uuid.uuid4()
    spec = JobSpec(
        argv=("python", "train.py"),
        cwd=workspace,
        timeout_seconds=600,
    )

    try:
        observation = tool.executor(
            RunJobAction(spec=spec),
            SimpleNamespace(id=conversation_id),
        )

        assert training.launched == [spec]
        assert observation.job_id == "training-17"
        assert observation.wandb_run_ids == ("run-abc",)
        monitor = monitors.spec("training-17")
        assert monitor.conversation_id == conversation_id
        assert monitor.metric is None
        assert monitor.gates == ()
    finally:
        monitors.close()


def test_run_job_allows_advisor_watchers_while_workspace_is_dirty(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    (workspace / "candidate.py").write_text("print('uncommitted')\n")
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunJobTool.create(training, monitors)[0]

    try:
        spec = JobSpec(
            argv=("python", "watch_receipt.py"),
            cwd=workspace,
            timeout_seconds=20,
            workspace_access="read_only",
        )
        tool.executor(
            RunJobAction(spec=spec),
            SimpleNamespace(id=uuid.uuid4()),
        )

        assert training.launched == [spec]
        assert len(monitors.active()) == 1
    finally:
        monitors.close()


def test_run_job_requires_a_conversation_before_starting(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunJobTool.create(training, monitors)[0]

    try:
        with pytest.raises(ValueError, match="parent conversation"):
            tool.executor(
                RunJobAction(
                    spec=JobSpec(
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


@pytest.mark.parametrize("secret_name", ["WANDB_API_KEY", "MLXFAST_API_TOKEN"])
def test_run_job_grants_only_requested_registry_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secret_name: str,
):
    workspace = init_workspace(tmp_path)
    secret = f"{secret_name.lower()}-from-registry"
    failed = finished_result(tmp_path).model_copy(
        update={
            "state": TrainingState.FAILED,
            "exit_code": 1,
            "error_tail": f"failed with {secret}",
        }
    )
    training = StubTraining(workspace, failed)
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    registry = SecretRegistry()
    registry.update_secrets({secret_name: secret})
    conversation = SimpleNamespace(
        id=uuid.uuid4(),
        state=SimpleNamespace(secret_registry=registry),
    )
    monkeypatch.setenv("WANDB_API_KEY", "ambient-wandb")
    monkeypatch.setenv("MLXFAST_API_TOKEN", "ambient-mlxfast")
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    monkeypatch.setenv("EXA_API_KEY", "exa-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "aws-secret")
    monkeypatch.setenv("CUSTOM_TOKEN", "custom-secret")
    monkeypatch.setenv("password", "lowercase-secret")

    try:
        observation = RunJobTool.create(training, monitors)[0].executor(
            RunJobAction(
                spec=JobSpec(
                    argv=("python", "evaluate.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                    secret_env=(secret_name,),
                )
            ),
            conversation,
        )

        environment = training.environments[0]
        assert environment[secret_name] == secret
        assert training.redacted_values == [(secret,)]
        assert "PATH" in environment
        assert ({"WANDB_API_KEY", "MLXFAST_API_TOKEN"} - {secret_name}).isdisjoint(
            environment
        )
        assert {
            "OPENAI_API_KEY",
            "EXA_API_KEY",
            "GITHUB_TOKEN",
            "AWS_SESSION_TOKEN",
            "CUSTOM_TOKEN",
            "password",
        }.isdisjoint(environment)
        assert observation.error_tail == "failed with <secret-hidden>"
    finally:
        monitors.close()


def test_run_job_registration_failure_cancels_only_the_new_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    other_id = "other-job"
    monitors.register(
        TrainingMonitorSpec(
            training_id=other_id,
            conversation_id=uuid.uuid4(),
        )
    )

    def fail_registration(_spec):
        raise OSError("database unavailable")

    monkeypatch.setattr(monitors, "register", fail_registration)
    try:
        with pytest.raises(OSError, match="database unavailable"):
            RunJobTool.create(training, monitors)[0].executor(
                RunJobAction(
                    spec=JobSpec(
                        argv=("python", "evaluate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert training.cancelled == ["training-17"]
        assert monitors.spec(other_id).training_id == other_id
    finally:
        monitors.close()


def test_run_job_preserves_registration_error_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FailingCleanupTraining(StubTraining):
        def cancel_training(self, training_id: str) -> TrainingResult:
            self.cancelled.append(training_id)
            raise RuntimeError("cancel failed")

    workspace = init_workspace(tmp_path)
    training = FailingCleanupTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")

    def fail_registration(_spec):
        raise OSError("registration failed")

    monkeypatch.setattr(monitors, "register", fail_registration)
    monkeypatch.setattr(
        monitors,
        "complete",
        lambda _job_id: (_ for _ in ()).throw(RuntimeError("retire failed")),
    )
    try:
        with pytest.raises(OSError, match="registration failed"):
            RunJobTool.create(training, monitors)[0].executor(
                RunJobAction(
                    spec=JobSpec(
                        argv=("python", "evaluate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert training.cancelled == ["training-17"]
    finally:
        monitors.close()


def test_student_mutable_job_requires_a_clean_checkout(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    (workspace / "candidate.py").write_text("dirty\n")
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")

    try:
        with pytest.raises(RuntimeError, match="clean before run_job"):
            RunJobTool.create(training, monitors, role="student")[0].executor(
                RunJobAction(
                    spec=JobSpec(
                        argv=("python", "evaluate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )
        assert training.launched == []
    finally:
        monitors.close()


def test_monitor_job_validates_the_job_id_before_registration(
    tmp_path: Path,
):
    class MissingTraining(StubTraining):
        def get_training_status(self, training_id: str) -> TrainingResult:
            self.status_checks.append(training_id)
            raise KeyError(training_id)

    workspace = tmp_path / "workspace"
    training = MissingTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = MonitorJobTool.create(training, monitors)[0]

    try:
        with pytest.raises(KeyError, match="missing-training"):
            tool.executor(
                MonitorJobAction(job_id="missing-training"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert training.status_checks == []
        assert monitors.active() == []
    finally:
        monitors.close()


def test_monitor_job_replaces_the_default_policy(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    run_tool = RunJobTool.create(training, monitors)[0]
    monitor_tool = MonitorJobTool.create(training, monitors)[0]

    try:
        run_tool.executor(
            RunJobAction(
                spec=JobSpec(
                    argv=("python", "train.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                )
            ),
            SimpleNamespace(id=conversation_id),
        )
        action = MonitorJobAction(
            job_id="training-17",
            wandb_metric="validation/loss",
            direction="min",
            stale_after_seconds=300,
        )
        with pytest.raises(PermissionError, match="different conversation"):
            monitor_tool.executor(action, SimpleNamespace(id=uuid.uuid4()))
        observation = monitor_tool.executor(
            action,
            SimpleNamespace(id=conversation_id),
        )

        monitor = monitors.spec("training-17")
        assert training.status_checks == ["training-17"]
        assert monitor.metric == "validation/loss"
        assert monitor.direction == "min"
        assert monitor.stale_after_seconds == 300
        assert observation.to_llm_content[0].text == (
            "Job training-17 is durably monitored. You may finish this turn; "
            "the controller will resume this same conversation "
            f"({conversation_id}) when action is needed."
        )
    finally:
        monitors.close()


@pytest.mark.parametrize(
    "terminal_state",
    [
        TrainingState.FINISHED,
        TrainingState.FAILED,
        TrainingState.TIMED_OUT,
        TrainingState.CANCELLED,
    ],
)
def test_get_job_status_allows_a_resumed_role_to_collect_a_terminal_job(
    tmp_path: Path,
    terminal_state: TrainingState,
):
    workspace = init_workspace(tmp_path)
    result = finished_result(tmp_path).model_copy(update={"state": terminal_state})
    training = StubTraining(workspace, result)
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(training, monitors)[0].executor(
        RunJobAction(
            spec=JobSpec(
                argv=("python", "evaluate.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        status = GetJobStatusTool.create(training, monitors)[0].executor
        with pytest.raises(ValueError, match="parent conversation"):
            status(GetJobStatusAction(job_id="training-17"))

        observation = status(
            GetJobStatusAction(job_id="training-17"),
            SimpleNamespace(id=uuid.uuid4()),
        )

        assert observation.job_id == "training-17"
        assert observation.state is terminal_state
    finally:
        monitors.close()


def test_get_job_status_keeps_a_running_job_scoped_to_its_conversation(
    tmp_path: Path,
):
    workspace = init_workspace(tmp_path)
    result = finished_result(tmp_path).model_copy(
        update={"state": TrainingState.RUNNING, "exit_code": None}
    )
    training = StubTraining(workspace, result)
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(training, monitors)[0].executor(
        RunJobAction(
            spec=JobSpec(
                argv=("python", "evaluate.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        status = GetJobStatusTool.create(training, monitors)[0].executor
        with pytest.raises(PermissionError, match="different conversation"):
            status(
                GetJobStatusAction(job_id="training-17"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        observation = status(
            GetJobStatusAction(job_id="training-17"),
            SimpleNamespace(id=conversation_id),
        )

        assert observation.state is TrainingState.RUNNING
    finally:
        monitors.close()


def test_cancel_job_retires_its_monitor(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(training, monitors)[0].executor(
        RunJobAction(
            spec=JobSpec(
                argv=("python", "train.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        cancel = CancelJobTool.create(training, monitors)[0].executor
        with pytest.raises(PermissionError, match="different conversation"):
            cancel(
                CancelJobAction(job_id="training-17"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        observation = cancel(
            CancelJobAction(job_id="training-17"),
            SimpleNamespace(id=conversation_id),
        )

        assert observation.state is TrainingState.CANCELLED
        assert training.cancelled == ["training-17"]
        assert monitors.active() == []
    finally:
        monitors.close()


def test_cancel_job_keeps_monitor_when_cancellation_is_not_terminal(
    tmp_path: Path,
):
    class NonTerminalCancellation(StubTraining):
        def cancel_training(self, training_id: str) -> TrainingResult:
            self.cancelled.append(training_id)
            return self.result.model_copy(update={"state": TrainingState.RUNNING})

    workspace = init_workspace(tmp_path)
    training = NonTerminalCancellation(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(training, monitors)[0].executor(
        RunJobAction(
            spec=JobSpec(
                argv=("python", "train.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        SimpleNamespace(id=conversation_id),
    )

    try:
        cancel = CancelJobTool.create(training, monitors)[0].executor

        with pytest.raises(RuntimeError, match="did not reach a terminal state"):
            cancel(
                CancelJobAction(job_id="training-17"),
                SimpleNamespace(id=conversation_id),
            )

        assert training.cancelled == ["training-17"]
        assert [monitor.training_id for monitor in monitors.active()] == ["training-17"]
    finally:
        monitors.close()


def test_interrupting_run_job_does_not_close_its_shared_runtime(tmp_path: Path):
    training = StubTraining(tmp_path, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")

    try:
        RunJobTool.create(training, monitors)[0].executor.interrupt()

        assert training.closed is False
    finally:
        monitors.close()


def test_registered_job_tools_share_one_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))
    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    register_senpai_tools()

    tools = resolve_tool(
        Tool(name="senpai_training", params={"state_dir": str(tmp_path / "state")}),
        state,
    )
    by_name = {tool.name: tool for tool in tools}

    try:
        assert set(by_name) == {
            "cancel_job",
            "run_job",
            "get_job_status",
            "monitor_job",
        }
        assert "already-running job" in by_name["monitor_job"].description
        assert "without disabling terminal wakes" in by_name["monitor_job"].description
        assert "upgrade" not in by_name["monitor_job"].description.lower()
        assert (
            by_name["run_job"].executor.supervisor
            is by_name["get_job_status"].executor.supervisor
        )
        assert (
            by_name["run_job"].executor.supervisor
            is by_name["monitor_job"].executor.supervisor
        )
        assert (
            by_name["run_job"].executor.supervisor
            is by_name["cancel_job"].executor.supervisor
        )
        assert (
            by_name["run_job"].executor.monitor_store
            is by_name["monitor_job"].executor.store
        )
    finally:
        close_training_runtimes()


@pytest.mark.parametrize("interval", [4, float("nan"), float("inf")])
def test_monitor_job_requires_a_finite_interval_of_at_least_five_seconds(
    interval: float,
):
    with pytest.raises(ValueError):
        MonitorJobAction(job_id="job-17", poll_interval_seconds=interval)


def test_job_control_tools_declare_one_serialized_runtime_resource(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tools = (
        RunJobTool.create(training, monitors)[0],
        GetJobStatusTool.create(training, monitors)[0],
        CancelJobTool.create(training, monitors)[0],
        MonitorJobTool.create(training, monitors)[0],
    )
    actions = (
        RunJobAction(
            spec=JobSpec(
                argv=("python", "evaluate.py"),
                cwd=workspace,
                timeout_seconds=20,
            )
        ),
        GetJobStatusAction(job_id="job-17"),
        CancelJobAction(job_id="job-17"),
        MonitorJobAction(job_id="job-17"),
    )

    try:
        resources = [
            tool.declared_resources(action) for tool, action in zip(tools, actions)
        ]
        assert all(resource.keys == ("senpai-job-control",) for resource in resources)
        assert all(resource.declared for resource in resources)
    finally:
        monitors.close()
