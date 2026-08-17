import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import SecretRegistry
from openhands.sdk.tool import Tool, resolve_tool

from senpai_agent.monitor import (
    JobMonitorSpec,
    JobMonitorStore,
    MetricMonitorSpec,
    WandbJobStatusSource,
)
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
    close_job_runtimes,
    register_senpai_tools,
)
from senpai_agent.jobs import JobResult, JobState


class StubJob:
    def __init__(self, workspace: Path, result: JobResult):
        self.workspace = workspace
        self.result = result
        self.launched: list[JobSpec] = []
        self.status_checks: list[str] = []
        self.cancelled: list[str] = []
        self.environments: list[dict[str, str]] = []
        self.redacted_values: list[tuple[str, ...]] = []
        self.closed = False

    def run_job(
        self,
        spec: JobSpec,
        *,
        env=None,
        redacted_values=(),
    ) -> JobResult:
        self.launched.append(spec)
        self.environments.append(dict(env or {}))
        self.redacted_values.append(tuple(redacted_values))
        return self.result

    def get_job_status(self, job_id: str) -> JobResult:
        self.status_checks.append(job_id)
        return self.result

    def cancel_job(self, job_id: str) -> JobResult:
        self.cancelled.append(job_id)
        return self.result.model_copy(update={"state": JobState.CANCELLED})

    def close(self) -> None:
        self.closed = True


def init_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    return workspace


def finished_result(tmp_path: Path) -> JobResult:
    return JobResult(
        job_id="job-17",
        state=JobState.FINISHED,
        exit_code=0,
        elapsed_seconds=12.5,
        log_path=str(tmp_path / "supervisor.log"),
        wandb_run_ids=("run-abc",),
    )


def test_run_job_registers_a_monitor_for_its_conversation(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunJobTool.create(supervisor, monitors)[0]
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

        assert supervisor.launched == [spec]
        assert observation.job_id == "job-17"
        assert observation.wandb_run_ids == ("run-abc",)
        monitor = monitors.spec("job-17")
        assert monitor.conversation_id == conversation_id
        assert monitor.metrics == ()
    finally:
        monitors.close()


def test_run_job_allows_advisor_watchers_while_workspace_is_dirty(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    (workspace / "candidate.py").write_text("print('uncommitted')\n")
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunJobTool.create(supervisor, monitors)[0]

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

        assert supervisor.launched == [spec]
        assert len(monitors.active()) == 1
    finally:
        monitors.close()


def test_run_job_requires_a_conversation_before_starting(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunJobTool.create(supervisor, monitors)[0]

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

        assert supervisor.launched == []
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
            "state": JobState.FAILED,
            "exit_code": 1,
            "error_tail": f"failed with {secret}",
        }
    )
    supervisor = StubJob(workspace, failed)
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
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
        observation = RunJobTool.create(supervisor, monitors)[0].executor(
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

        environment = supervisor.environments[0]
        assert environment[secret_name] == secret
        assert supervisor.redacted_values == [(secret,)]
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
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    other_id = "other-job"
    monitors.register(
        JobMonitorSpec(
            job_id=other_id,
            conversation_id=uuid.uuid4(),
        )
    )

    def fail_registration(_spec):
        raise OSError("database unavailable")

    monkeypatch.setattr(monitors, "register", fail_registration)
    try:
        with pytest.raises(OSError, match="database unavailable"):
            RunJobTool.create(supervisor, monitors)[0].executor(
                RunJobAction(
                    spec=JobSpec(
                        argv=("python", "evaluate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert supervisor.cancelled == ["job-17"]
        assert monitors.spec(other_id).job_id == other_id
    finally:
        monitors.close()


def test_run_job_preserves_registration_error_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FailingCleanupJob(StubJob):
        def cancel_job(self, job_id: str) -> JobResult:
            self.cancelled.append(job_id)
            raise RuntimeError("cancel failed")

    workspace = init_workspace(tmp_path)
    supervisor = FailingCleanupJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")

    def fail_registration(_spec):
        raise OSError("registration failed")

    monkeypatch.setattr(monitors, "register", fail_registration)
    monkeypatch.setattr(
        monitors,
        "discard",
        lambda _job_id: (_ for _ in ()).throw(RuntimeError("discard failed")),
    )
    try:
        with pytest.raises(OSError, match="registration failed"):
            RunJobTool.create(supervisor, monitors)[0].executor(
                RunJobAction(
                    spec=JobSpec(
                        argv=("python", "evaluate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert supervisor.cancelled == ["job-17"]
    finally:
        monitors.close()


def test_student_mutable_job_requires_a_clean_checkout(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    (workspace / "candidate.py").write_text("dirty\n")
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")

    try:
        with pytest.raises(RuntimeError, match="clean before run_job"):
            RunJobTool.create(supervisor, monitors, role="student")[0].executor(
                RunJobAction(
                    spec=JobSpec(
                        argv=("python", "evaluate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )
        assert supervisor.launched == []
    finally:
        monitors.close()


def test_monitor_job_validates_the_job_id_before_registration(
    tmp_path: Path,
):
    class MissingJob(StubJob):
        def get_job_status(self, job_id: str) -> JobResult:
            self.status_checks.append(job_id)
            raise KeyError(job_id)

    workspace = tmp_path / "workspace"
    supervisor = MissingJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    tool = MonitorJobTool.create(supervisor, monitors)[0]

    try:
        with pytest.raises(KeyError, match="missing-job"):
            tool.executor(
                MonitorJobAction(job_id="missing-job"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert supervisor.status_checks == []
        assert monitors.active() == []
    finally:
        monitors.close()


def test_monitor_job_replaces_the_default_policy(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    run_tool = RunJobTool.create(supervisor, monitors)[0]
    monitor_tool = MonitorJobTool.create(supervisor, monitors)[0]

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
            job_id="job-17",
            metrics=(
                MetricMonitorSpec(
                    metric="validation/loss",
                    direction="min",
                    stale_after_seconds=300,
                ),
            ),
        )
        with pytest.raises(PermissionError, match="different conversation"):
            monitor_tool.executor(action, SimpleNamespace(id=uuid.uuid4()))
        observation = monitor_tool.executor(
            action,
            SimpleNamespace(id=conversation_id),
        )

        monitor = monitors.spec("job-17")
        assert supervisor.status_checks == ["job-17"]
        assert monitor.metrics == action.metrics
        assert observation.to_llm_content[0].text == (
            "Job job-17 is durably monitored. You may finish this turn; "
            "the controller will resume this same conversation "
            f"({conversation_id}) when action is needed."
        )
    finally:
        monitors.close()


@pytest.mark.parametrize(
    "terminal_state",
    [
        JobState.FINISHED,
        JobState.FAILED,
        JobState.TIMED_OUT,
        JobState.CANCELLED,
    ],
)
def test_get_job_status_records_one_owned_terminal_observation(
    tmp_path: Path,
    terminal_state: JobState,
):
    workspace = init_workspace(tmp_path)
    result = finished_result(tmp_path).model_copy(update={"state": terminal_state})
    supervisor = StubJob(workspace, result)
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(supervisor, monitors)[0].executor(
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
        status = GetJobStatusTool.create(supervisor, monitors)[0].executor
        with pytest.raises(ValueError, match="parent conversation"):
            status(GetJobStatusAction(job_id="job-17"))

        with pytest.raises(PermissionError, match="different conversation"):
            status(
                GetJobStatusAction(job_id="job-17"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        action = GetJobStatusAction(job_id="job-17")
        conversation = SimpleNamespace(id=conversation_id)
        observation = status(action, conversation)
        repeated = status(action, conversation)

        assert observation.job_id == "job-17"
        assert observation.state is terminal_state
        assert repeated == observation
        assert len(monitors.emitted("job-17")) == 1
        assert monitors.active() == []
    finally:
        monitors.close()


def test_get_job_status_keeps_a_running_job_scoped_to_its_conversation(
    tmp_path: Path,
):
    workspace = init_workspace(tmp_path)
    result = finished_result(tmp_path).model_copy(
        update={"state": JobState.RUNNING, "exit_code": None}
    )
    supervisor = StubJob(workspace, result)
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(supervisor, monitors)[0].executor(
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
        status = GetJobStatusTool.create(supervisor, monitors)[0].executor
        with pytest.raises(PermissionError, match="different conversation"):
            status(
                GetJobStatusAction(job_id="job-17"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        observation = status(
            GetJobStatusAction(job_id="job-17"),
            SimpleNamespace(id=conversation_id),
        )

        assert observation.state is JobState.RUNNING
    finally:
        monitors.close()


def test_cancel_job_retires_its_monitor(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(supervisor, monitors)[0].executor(
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
        cancel = CancelJobTool.create(supervisor, monitors)[0].executor
        with pytest.raises(PermissionError, match="different conversation"):
            cancel(
                CancelJobAction(job_id="job-17"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        observation = cancel(
            CancelJobAction(job_id="job-17"),
            SimpleNamespace(id=conversation_id),
        )

        assert observation.state is JobState.CANCELLED
        assert supervisor.cancelled == ["job-17"]
        assert monitors.active() == []
    finally:
        monitors.close()


def test_cancel_job_keeps_monitor_when_cancellation_is_not_terminal(
    tmp_path: Path,
):
    class NonTerminalCancellation(StubJob):
        def cancel_job(self, job_id: str) -> JobResult:
            self.cancelled.append(job_id)
            return self.result.model_copy(update={"state": JobState.RUNNING})

    workspace = init_workspace(tmp_path)
    supervisor = NonTerminalCancellation(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    RunJobTool.create(supervisor, monitors)[0].executor(
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
        cancel = CancelJobTool.create(supervisor, monitors)[0].executor

        with pytest.raises(RuntimeError, match="did not reach a terminal state"):
            cancel(
                CancelJobAction(job_id="job-17"),
                SimpleNamespace(id=conversation_id),
            )

        assert supervisor.cancelled == ["job-17"]
        assert [monitor.job_id for monitor in monitors.active()] == ["job-17"]
    finally:
        monitors.close()


def test_interrupting_run_job_does_not_close_its_shared_runtime(tmp_path: Path):
    supervisor = StubJob(tmp_path, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")

    try:
        RunJobTool.create(supervisor, monitors)[0].executor.interrupt()

        assert supervisor.closed is False
    finally:
        monitors.close()


def test_registered_job_tools_share_one_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))
    monkeypatch.setenv("SENPAI_ROLE", "student")
    register_senpai_tools()

    tools = resolve_tool(
        Tool(name="senpai_jobs", params={"state_dir": str(tmp_path / "state")}),
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
        close_job_runtimes()


def test_advisor_monitor_job_uses_external_wandb_ids_without_local_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=tmp_path))
    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    register_senpai_tools()
    local_tools = resolve_tool(
        Tool(name="senpai_jobs", params={"state_dir": str(tmp_path / "jobs")}),
        state,
    )
    tools = resolve_tool(
        Tool(
            name="senpai_advisor_job_monitor",
            params={
                "state_dir": str(tmp_path / "advisor-job-monitors"),
                "wandb_entity": "research-team",
                "wandb_project": "project",
            },
        ),
        state,
    )
    assert {tool.name for tool in local_tools} == {
        "run_job",
        "get_job_status",
        "cancel_job",
    }
    assert [tool.name for tool in tools] == ["monitor_job"]
    executor = tools[0].executor
    checked: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        WandbJobStatusSource,
        "get_job_status",
        lambda source, job_id: checked.append((job_id, source.api_key)),
    )
    action = MonitorJobAction(job_id="wandb.run-17")
    first = uuid.uuid4()
    second = uuid.uuid4()
    registry = SecretRegistry()
    registry.update_secrets({"WANDB_API_KEY": "registered-wandb-key"})
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    first_conversation = SimpleNamespace(
        id=first,
        state=SimpleNamespace(secret_registry=registry),
    )
    second_conversation = SimpleNamespace(
        id=second,
        state=SimpleNamespace(secret_registry=registry),
    )

    try:
        executor(action, first_conversation)
        observation = executor(action, second_conversation)

        assert checked == [
            ("wandb.run-17", "registered-wandb-key"),
            ("wandb.run-17", "registered-wandb-key"),
        ]
        assert observation.job_id == "wandb.run-17"
        assert executor.store.spec("wandb.run-17").conversation_id == second
    finally:
        close_job_runtimes()


@pytest.mark.parametrize("job_id", ["other/project/run", "run?x=1", "run id"])
def test_monitor_job_rejects_ids_that_escape_the_configured_project(job_id):
    with pytest.raises(ValueError):
        MonitorJobAction(job_id=job_id)


@pytest.mark.parametrize("interval", [4, float("nan"), float("inf")])
def test_monitor_job_requires_a_finite_interval_of_at_least_five_seconds(
    interval: float,
):
    with pytest.raises(ValueError):
        MonitorJobAction(job_id="job-17", poll_interval_seconds=interval)


def test_job_control_tools_declare_one_serialized_runtime_resource(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    supervisor = StubJob(workspace, finished_result(tmp_path))
    monitors = JobMonitorStore(tmp_path / "monitors.sqlite3")
    tools = (
        RunJobTool.create(supervisor, monitors)[0],
        GetJobStatusTool.create(supervisor, monitors)[0],
        CancelJobTool.create(supervisor, monitors)[0],
        MonitorJobTool.create(supervisor, monitors)[0],
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
