from __future__ import annotations

import errno
import json
import re
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil
import pytest

import senpai_agent.kubernetes_training as kubernetes_training
from senpai_agent.kubernetes_training import KubernetesTrainingSupervisor
from senpai_agent.training import (
    KubernetesResourceRef,
    KubernetesTrainingSpec,
    TrainingResult,
    TrainingSpec,
    TrainingState,
)


class FakeCluster:
    def __init__(self, state: TrainingState = TrainingState.FINISHED):
        self.state_value = state
        self.spec: KubernetesTrainingSpec | None = None
        self.resource_value: KubernetesResourceRef | None = None
        self.reservations: list[tuple[str, str, str]] = []
        self.adoptions: list[KubernetesResourceRef] = []
        self.deletions: list[KubernetesResourceRef] = []
        self.releases: list[str] = []
        self._lock = threading.Lock()

    def reserve(
        self,
        training_id,
        spec,
        _deadline_at,
        source_snapshot,
        source_commit,
    ):
        with self._lock:
            self.spec = spec
            self.resource_value = KubernetesResourceRef(
                kind=spec.kind,
                name=spec.name,
                namespace=spec.namespace,
                uid="remote-uid",
                nodes=2,
                gpus_per_node=8,
            )
            self.reservations.append((training_id, source_snapshot, source_commit))

    def adopt(self, _training_id, _spec, resource, _deadline_at):
        self.adoptions.append(resource)

    def resource(self, _spec, *, nodes, gpus_per_node):
        assert (nodes, gpus_per_node) == (2, 8)
        return self.resource_value

    def resource_identity(self, _spec):
        return self.resource_value

    def state(self, _resource):
        return self.state_value, self.state_value.value

    def delete(self, resource, timeout_seconds=60):
        assert timeout_seconds <= 60
        if self.resource_value is not None and self.resource_value.uid != resource.uid:
            raise RuntimeError("refusing to delete replaced workload")
        self.deletions.append(resource)
        self.resource_value = None

    def logs(self, _resource):
        return "remote worker log"

    def release(self, training_id):
        self.releases.append(training_id)


def git_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=workspace,
        check=True,
    )
    (workspace / ".gitignore").write_text(".env\n")
    (workspace / "tracked.txt").write_text("committed\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=workspace, check=True)
    (workspace / ".env").write_text("WANDB_API_KEY=secret\n")
    return workspace


def supervisor(tmp_path, monkeypatch, client=None, **overrides):
    workspace = git_workspace(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    environment = {
        "RESEARCH_TAG": "fred",
        "STUDENT_NAME": "fern",
        "SENPAI_KUBERNETES_NAMESPACE": "research",
        "SENPAI_LAUNCH_SECRET_NAME": "senpai-launch-secrets-fred",
        "SENPAI_TRAINING_SNAPSHOT_ROOT": str(snapshot_root),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    values = {
        "workspace": workspace,
        "state_dir": tmp_path / "state",
        "nodes": 2,
        "gpus_per_node": 8,
        "max_timeout_seconds": 10,
        "terminate_grace_seconds": 0.1,
        "poll_seconds": 0.01,
        "client": client or FakeCluster(),
    }
    values.update(overrides)
    return KubernetesTrainingSupervisor(**values), workspace, snapshot_root


def test_supervisor_injects_authoritative_identity_and_bundles_clean_head(
    tmp_path,
    monkeypatch,
):
    client = FakeCluster()
    runtime, workspace, snapshot_root = supervisor(tmp_path, monkeypatch, client)
    keys = [
        "SENPAI_TRAINING_SOURCE_SNAPSHOT",
        "SENPAI_KUBERNETES_WORKLOAD_NAME",
        "SENPAI_KUBERNETES_NAMESPACE",
        "SENPAI_WANDB_RUN_ID",
        "SENPAI_LAUNCH_SECRET_NAME",
    ]
    code = (
        "import json,os,time; "
        f"print(json.dumps({{key: os.environ[key] for key in {keys!r}}})); "
        "time.sleep(0.2)"
    )

    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", code),
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    runtime.drain()
    result = runtime.get_training_status(started.training_id)

    assert result.state is TrainingState.FINISHED
    assert result.kubernetes_spec is not None
    assert result.kubernetes_spec.kind == "MPIJob"
    assert result.kubernetes_spec.namespace == "research"
    assert result.kubernetes_spec.wandb_run_id == started.training_id.replace("-", "")
    assert len(result.kubernetes_spec.name) <= 63
    assert result.source_commit == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    snapshot = Path(result.source_snapshot)
    assert snapshot == snapshot_root / f"{result.source_commit}.bundle"
    assert snapshot.is_file()
    assert subprocess.check_output(
        ["git", "bundle", "list-heads", snapshot, "HEAD"],
        text=True,
    ).split() == [result.source_commit, "HEAD"]

    environment = json.loads(Path(result.log_path).read_text().splitlines()[0])
    assert environment == {
        "SENPAI_TRAINING_SOURCE_SNAPSHOT": str(snapshot),
        "SENPAI_KUBERNETES_WORKLOAD_NAME": result.kubernetes_spec.name,
        "SENPAI_KUBERNETES_NAMESPACE": "research",
        "SENPAI_WANDB_RUN_ID": result.kubernetes_spec.wandb_run_id,
        "SENPAI_LAUNCH_SECRET_NAME": "senpai-launch-secrets-fred",
    }
    assert client.reservations[0][1:] == (str(snapshot), result.source_commit)
    assert client.releases == [result.training_id]
    assert result.kubernetes_released is True


def test_supervisor_retries_transient_status_and_release_failures(
    tmp_path,
    monkeypatch,
):
    class TransientCluster(FakeCluster):
        def __init__(self):
            super().__init__()
            self.state_attempts = 0
            self.release_attempts = 0

        def state(self, resource):
            self.state_attempts += 1
            if self.state_attempts == 1:
                raise TimeoutError("temporary status outage")
            return super().state(resource)

        def release(self, training_id):
            self.release_attempts += 1
            if self.release_attempts == 1:
                raise TimeoutError("temporary release outage")
            super().release(training_id)

    client = TransientCluster()
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "pass"),
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    runtime.drain()

    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.FINISHED
    assert result.kubernetes_released is True
    assert client.state_attempts == 2
    assert client.release_attempts == 2
    assert client.deletions == []


@pytest.mark.parametrize("storage_errno", [errno.ENOSPC, errno.EDQUOT])
@pytest.mark.parametrize("write_phase", ["resource", "terminal", "release"])
def test_supervisor_recovers_result_persistence_after_storage_exhaustion(
    tmp_path, monkeypatch, capsys, storage_errno, write_phase,
):
    client = FakeCluster()
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    storage_failed = threading.Event()
    storage_restored = threading.Event()
    retried = threading.Event()
    write_text = Path.write_text

    def exhausted_write(path, text, *args, **kwargs):
        if path.parent == runtime.state_dir and path.suffix == ".tmp":
            record = json.loads(text)
            phase = (
                "release" if record["kubernetes_released"] else
                "terminal" if record["state"] != "running" else
                "resource" if record["kubernetes_resource"] else "launch"
            )
            if phase == write_phase and not storage_restored.is_set():
                if storage_failed.is_set():
                    retried.set()
                storage_failed.set()
                raise OSError(storage_errno, "storage exhausted", str(path))
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", exhausted_write)
    spec = TrainingSpec(argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5)
    started = runtime.run_training(spec)
    try:
        assert storage_failed.wait(2)
        assert retried.wait(1), "result persistence stopped after the first storage error"
        assert runtime.get_training_status(started.training_id).kubernetes_released is False
        assert client.releases == ([started.training_id] if write_phase == "release" else [])
        with pytest.raises(RuntimeError, match="already has an active"):
            runtime.run_training(spec)
        if write_phase != "resource":
            client.state_value = TrainingState.FAILED
        storage_restored.set()
        runtime.drain()
        result = runtime.get_training_status(started.training_id)
        assert result.state is TrainingState.FINISHED
        assert result.kubernetes_released is True
        assert result.kubernetes_resource.uid == "remote-uid"
        assert client.releases == [started.training_id]
        assert client.deletions == []
        client.state_value = TrainingState.FINISHED
        following = runtime.run_training(spec)
        runtime.drain()
        assert runtime.get_training_status(following.training_id).state is TrainingState.FINISHED
        assert "persistence recovered" in capsys.readouterr().err
    finally:
        storage_restored.set()
        runtime.close()


@pytest.mark.parametrize("storage_errno", [errno.ENOSPC, errno.EDQUOT])
@pytest.mark.parametrize("write_phase", ["log", "summary"])
def test_storage_exhaustion_in_optional_diagnostics_does_not_fail_training(
    tmp_path, monkeypatch, capsys, storage_errno, write_phase,
):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    storage_failed = threading.Event()
    open_path = Path.open
    write_text = Path.write_text

    def exhausted_log(path, mode="r", *args, **kwargs):
        if path.parent == runtime.state_dir and path.suffix == ".log" and mode == "a":
            storage_failed.set()
            raise OSError(storage_errno, "storage exhausted", str(path))
        return open_path(path, mode, *args, **kwargs)

    def exhausted_summary(path, text, *args, **kwargs):
        if path.parent == runtime.state_dir and path.suffix == ".tmp":
            if json.loads(text)["kubernetes_diagnostics"]:
                storage_failed.set()
                raise OSError(storage_errno, "storage exhausted", str(path))
        return write_text(path, text, *args, **kwargs)

    if write_phase == "log":
        monkeypatch.setattr(Path, "open", exhausted_log)
    else:
        monkeypatch.setattr(Path, "write_text", exhausted_summary)
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    try:
        assert storage_failed.wait(2)
        client.state_value = TrainingState.FINISHED
        runtime.drain()
        result = runtime.get_training_status(started.training_id)
        assert result.state is TrainingState.FINISHED
        assert result.kubernetes_released is True
        assert client.deletions == []
        assert "diagnostics persistence skipped" in capsys.readouterr().err
    finally:
        runtime.close()


@pytest.mark.parametrize("replacement", [False, True])
def test_close_during_terminal_storage_outage_recovers_only_original_workload(
    tmp_path, monkeypatch, replacement,
):
    client = FakeCluster()
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    storage_failed = threading.Event()
    storage_restored = threading.Event()
    write_text = Path.write_text

    def exhausted_terminal(path, text, *args, **kwargs):
        if path.parent == runtime.state_dir and path.suffix == ".tmp":
            if json.loads(text)["state"] != "running" and not storage_restored.is_set():
                storage_failed.set()
                raise OSError(errno.EDQUOT, "storage exhausted", str(path))
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", exhausted_terminal)
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    assert storage_failed.wait(2)
    runtime.close()
    assert client.releases == []
    assert client.deletions == []
    original = runtime.get_training_status(started.training_id).kubernetes_resource
    resource = original.model_copy(update={"uid": "replacement-uid"}) if replacement else original
    reserve = client.reserve

    def preserve_remote_identity(*args):
        reserve(*args)
        client.resource_value = resource

    monkeypatch.setattr(client, "reserve", preserve_remote_identity)
    storage_restored.set()
    recovered = KubernetesTrainingSupervisor(
        workspace=workspace, state_dir=runtime.state_dir, nodes=2, gpus_per_node=8,
        poll_seconds=0.01, client=client,
    )
    if not replacement:
        recovered.drain()
    recovered.close()
    result = recovered.get_training_status(started.training_id)
    assert result.state is (TrainingState.FAILED if replacement else TrainingState.FINISHED)
    assert result.kubernetes_resource == original
    assert client.deletions == []
    assert client.adoptions == ([] if replacement else [original])
    assert client.releases == ([] if replacement else [started.training_id])
    assert result.kubernetes_released is (not replacement)
    if replacement:
        assert client.resource_value == resource


def test_source_bundle_ignores_replace_refs_and_replaces_poisoned_artifacts(
    tmp_path,
    monkeypatch,
):
    runtime, workspace, snapshot_root = supervisor(tmp_path, monkeypatch)
    original = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    (workspace / "tracked.txt").write_text("replacement\n")
    subprocess.run(["git", "commit", "-qam", "replacement"], cwd=workspace, check=True)
    replacement = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    subprocess.run(["git", "checkout", "--quiet", original], cwd=workspace, check=True)
    subprocess.run(["git", "replace", original, replacement], cwd=workspace, check=True)
    snapshot_root.mkdir()
    (snapshot_root / f"{original}.bundle").write_text("poisoned")

    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "pass"),
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    runtime.drain()

    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--quiet", started.source_snapshot, checkout],
        check=True,
    )
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip() == original
    assert (checkout / "tracked.txt").read_text() == "committed\n"


def test_supervisor_allows_only_one_active_run_and_cancellation_deletes_remote(
    tmp_path,
    monkeypatch,
):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    spec = TrainingSpec(
        argv=(sys.executable, "-c", "import time; time.sleep(30)"),
        cwd=workspace,
        timeout_seconds=5,
    )
    first = runtime.run_training(spec)

    with pytest.raises(RuntimeError, match="already has an active"):
        runtime.run_training(spec)

    result = runtime.cancel_training(first.training_id)
    assert result.state is TrainingState.CANCELLED
    assert len(client.deletions) == 1
    assert client.releases == [first.training_id]


def test_close_detaches_and_restart_re_adopts_the_same_uid(tmp_path, monkeypatch):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(
        tmp_path,
        monkeypatch,
        client,
        max_timeout_seconds=60,
        poll_seconds=30,
    )
    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "pass"),
            cwd=workspace,
            timeout_seconds=30,
        )
    )
    deadline = time.monotonic() + 1
    while runtime.get_training_status(started.training_id).kubernetes_resource is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    before = runtime.get_training_status(started.training_id)
    assert before.kubernetes_resource is not None
    closed_at = time.monotonic()
    runtime.close()

    detached = runtime.get_training_status(started.training_id)
    assert time.monotonic() - closed_at < 1
    assert detached.state is TrainingState.RUNNING
    assert detached.kubernetes_released is False
    assert client.deletions == []
    assert client.releases == []
    with pytest.raises(RuntimeError, match="supervisor is closed"):
        runtime.cancel_training(started.training_id)
    with pytest.raises(RuntimeError, match="supervisor is closed"):
        runtime.run_training(
            TrainingSpec(
                argv=(sys.executable, "-c", "pass"),
                cwd=workspace,
                timeout_seconds=5,
            )
        )

    client.state_value = TrainingState.FINISHED
    recovered = KubernetesTrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        nodes=2,
        gpus_per_node=8,
        max_timeout_seconds=60,
        poll_seconds=0.01,
        client=client,
    )
    recovered.drain()

    assert client.adoptions == [before.kubernetes_resource]
    assert client.deletions == []
    assert client.releases == [started.training_id]
    terminal = recovered.get_training_status(started.training_id)
    assert terminal.state is TrainingState.FINISHED
    assert terminal.kubernetes_resource == before.kubernetes_resource


def test_close_terminates_only_the_local_launcher(tmp_path, monkeypatch):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    child_pid_path = workspace / "child.pid"
    child_code = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    launcher_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "time.sleep(30)"
    )
    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", launcher_code),
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    deadline = time.monotonic() + 1
    while not child_pid_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    child_pid = int(child_pid_path.read_text())

    runtime.close()

    detached = runtime.get_training_status(started.training_id)
    assert not psutil.pid_exists(started.pid)
    deadline = time.monotonic() + 1
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not psutil.pid_exists(child_pid)
    assert detached.state is TrainingState.RUNNING
    assert detached.kubernetes_released is False
    assert client.deletions == []
    assert client.releases == []


def test_close_waits_for_an_inflight_launch_to_observe_shutdown(
    tmp_path,
    monkeypatch,
):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    entered = threading.Event()
    release = threading.Event()
    original = kubernetes_training._materialize_source_snapshot

    def blocked_snapshot(cwd):
        entered.set()
        assert release.wait(1)
        return original(cwd)

    monkeypatch.setattr(
        kubernetes_training,
        "_materialize_source_snapshot",
        blocked_snapshot,
    )
    errors = []

    def launch():
        try:
            runtime.run_training(
                TrainingSpec(
                    argv=(sys.executable, "-c", "pass"),
                    cwd=workspace,
                    timeout_seconds=5,
                )
            )
        except Exception as error:
            errors.append(error)

    launch_thread = threading.Thread(target=launch)
    launch_thread.start()
    assert entered.wait(1)
    close_thread = threading.Thread(target=runtime.close)
    close_thread.start()
    assert runtime._shutdown.wait(1)
    assert close_thread.is_alive()

    release.set()
    launch_thread.join(1)
    close_thread.join(1)

    assert not launch_thread.is_alive()
    assert not close_thread.is_alive()
    assert len(errors) == 1
    assert str(errors[0]) == "Kubernetes training supervisor is closed"
    assert client.reservations == []


def test_close_persists_a_remote_terminal_result_that_wins_the_race(
    tmp_path,
    monkeypatch,
):
    class BlockingFinishedCluster(FakeCluster):
        def __init__(self):
            super().__init__(state=TrainingState.FINISHED)
            self.state_started = threading.Event()
            self.return_state = threading.Event()

        def state(self, resource):
            self.state_started.set()
            assert self.return_state.wait(1)
            return super().state(resource)

    client = BlockingFinishedCluster()
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "pass"),
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    assert client.state_started.wait(1)
    close_thread = threading.Thread(target=runtime.close)
    close_thread.start()
    assert runtime._shutdown.wait(1)

    client.return_state.set()
    close_thread.join(1)

    assert not close_thread.is_alive()
    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.FINISHED
    assert result.kubernetes_released is True
    assert client.releases == [started.training_id]


def test_close_cannot_split_running_publication_from_active_supervision(
    tmp_path,
    monkeypatch,
):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    result_written = threading.Event()
    finish_write = threading.Event()
    original_write = runtime._write_result

    def blocked_write(result):
        original_write(result)
        if result.state is TrainingState.RUNNING and not result_written.is_set():
            result_written.set()
            assert finish_write.wait(1)

    monkeypatch.setattr(runtime, "_write_result", blocked_write)
    errors = []

    def launch():
        try:
            runtime.run_training(
                TrainingSpec(
                    argv=(sys.executable, "-c", "pass"),
                    cwd=workspace,
                    timeout_seconds=5,
                )
            )
        except Exception as error:
            errors.append(error)

    launch_thread = threading.Thread(target=launch)
    launch_thread.start()
    assert result_written.wait(1)
    close_thread = threading.Thread(target=runtime.close)
    close_thread.start()
    time.sleep(0.05)
    assert close_thread.is_alive()

    finish_write.set()
    launch_thread.join(1)
    close_thread.join(1)

    assert errors == []
    assert not launch_thread.is_alive()
    assert not close_thread.is_alive()
    result_path = next((tmp_path / "state").glob("*.json"))
    persisted = TrainingResult.model_validate_json(result_path.read_text())
    assert persisted.state is TrainingState.RUNNING
    assert client.deletions == []
    assert client.releases == []


def test_close_does_not_override_a_selected_timeout(tmp_path, monkeypatch):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(
        tmp_path,
        monkeypatch,
        client,
        terminate_grace_seconds=0.2,
    )
    timeout_cleanup_started = threading.Event()
    finish_timeout_cleanup = threading.Event()
    original_terminate = kubernetes_training.terminate_process_group

    def blocked_terminate(process, **kwargs):
        if kwargs["grace_seconds"] == 0.2 and not timeout_cleanup_started.is_set():
            timeout_cleanup_started.set()
            assert finish_timeout_cleanup.wait(1)
        return original_terminate(process, **kwargs)

    monkeypatch.setattr(
        kubernetes_training,
        "terminate_process_group",
        blocked_terminate,
    )
    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=workspace,
            timeout_seconds=1,
        )
    )
    assert timeout_cleanup_started.wait(2)
    close_thread = threading.Thread(target=runtime.close)
    close_thread.start()
    assert runtime._shutdown.wait(1)

    finish_timeout_cleanup.set()
    close_thread.join(2)

    assert not close_thread.is_alive()
    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.TIMED_OUT
    assert result.kubernetes_released is True
    assert client.deletions[0].uid == "remote-uid"
    assert client.releases == [started.training_id]


def test_close_does_not_override_a_known_launcher_failure(tmp_path, monkeypatch):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    exit_observed = threading.Event()
    finish_returncode = threading.Event()
    real_popen = subprocess.Popen
    launcher_argv = (sys.executable, "-c", "raise SystemExit(7)")

    class BlockingReturncode:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        @property
        def returncode(self):
            value = self.process.returncode
            if value == 7 and not exit_observed.is_set():
                exit_observed.set()
                assert finish_returncode.wait(1)
            return value

    def blocking_popen(args, *popen_args, **popen_kwargs):
        process = real_popen(args, *popen_args, **popen_kwargs)
        if tuple(args) == launcher_argv:
            return BlockingReturncode(process)
        return process

    monkeypatch.setattr(kubernetes_training.subprocess, "Popen", blocking_popen)
    started = runtime.run_training(
        TrainingSpec(
            argv=launcher_argv,
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    assert exit_observed.wait(1)
    close_thread = threading.Thread(target=runtime.close)
    close_thread.start()
    assert runtime._shutdown.wait(1)

    finish_returncode.set()
    close_thread.join(1)

    assert not close_thread.is_alive()
    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.FAILED
    assert result.exit_code == 7
    assert result.kubernetes_released is True
    assert client.deletions[0].uid == "remote-uid"
    assert client.releases == [started.training_id]


def test_close_defers_a_failed_terminal_release_to_restart(tmp_path, monkeypatch):
    class FailingReleaseCluster(FakeCluster):
        def __init__(self):
            super().__init__(state=TrainingState.FINISHED)
            self.release_started = threading.Event()

        def release(self, training_id):
            self.release_started.set()
            raise RuntimeError(f"cannot release {training_id}")

    client = FailingReleaseCluster()
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "pass"),
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    assert client.release_started.wait(1)

    runtime.close()

    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.FINISHED
    assert result.kubernetes_released is False
    with runtime._lock:
        thread = runtime._active[started.training_id].thread
    assert thread is not None
    assert not thread.is_alive()


def test_supervisor_timeout_deletes_remote_workload(tmp_path, monkeypatch):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)

    started = runtime.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=workspace,
            timeout_seconds=1,
        )
    )
    runtime.drain()

    assert runtime.get_training_status(started.training_id).state is TrainingState.TIMED_OUT
    assert len(client.deletions) == 1
    assert client.releases == [started.training_id]


def test_supervisor_re_adopts_only_the_persisted_uid(tmp_path, monkeypatch):
    client = FakeCluster()
    runtime, workspace, snapshot_root = supervisor(tmp_path, monkeypatch, client)
    runtime.close()
    training_id = "11111111-1111-1111-1111-111111111111"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    spec = KubernetesTrainingSpec(
        kind="MPIJob",
        name="senpai-fred-fern-111111111111",
        namespace="research",
        wandb_run_id=training_id.replace("-", ""),
    )
    resource = KubernetesResourceRef(
        kind="MPIJob",
        name=spec.name,
        namespace=spec.namespace,
        uid="remote-uid",
        nodes=2,
        gpus_per_node=8,
    )
    state_dir = tmp_path / "recovered-state"
    state_dir.mkdir()
    result_path = state_dir / f"{training_id}.json"
    result_path.write_text(
        TrainingResult(
            training_id=training_id,
            state=TrainingState.RUNNING,
            exit_code=None,
            elapsed_seconds=0,
            log_path=str(state_dir / f"{training_id}.log"),
            started_at=time.time(),
            deadline_at=time.time() + 5,
            kubernetes_spec=spec,
            kubernetes_resource=resource,
            source_snapshot=str(snapshot_root / f"{commit}.bundle"),
            source_commit=commit,
        ).model_dump_json()
    )
    (state_dir / f"{training_id}.log").write_text("")
    client.spec = spec
    client.resource_value = resource

    recovered = KubernetesTrainingSupervisor(
        workspace=workspace,
        state_dir=state_dir,
        nodes=2,
        gpus_per_node=8,
        max_timeout_seconds=10,
        poll_seconds=0.01,
        client=client,
    )
    recovered.drain()

    assert client.adoptions == [resource]
    assert recovered.get_training_status(training_id).state is TrainingState.FINISHED


def test_supervisor_recovers_an_unconfirmed_terminal_release(tmp_path, monkeypatch):
    client = FakeCluster()
    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, client)
    runtime.close()
    training_id = "22222222-2222-2222-2222-222222222222"
    spec = KubernetesTrainingSpec(
        kind="MPIJob",
        name="senpai-fred-fern-222222222222",
        namespace="research",
        wandb_run_id=training_id.replace("-", ""),
    )
    state_dir = tmp_path / "terminal-state"
    state_dir.mkdir()
    (state_dir / f"{training_id}.log").write_text("")
    (state_dir / f"{training_id}.json").write_text(
        TrainingResult(
            training_id=training_id,
            state=TrainingState.FINISHED,
            exit_code=0,
            elapsed_seconds=1,
            log_path=str(state_dir / f"{training_id}.log"),
            started_at=time.time() - 1,
            deadline_at=time.time() + 5,
            kubernetes_spec=spec,
            kubernetes_released=False,
        ).model_dump_json()
    )

    recovered = KubernetesTrainingSupervisor(
        workspace=workspace,
        state_dir=state_dir,
        nodes=2,
        gpus_per_node=8,
        max_timeout_seconds=10,
        poll_seconds=0.01,
        client=client,
    )
    recovered.drain()

    result = recovered.get_training_status(training_id)
    assert result.kubernetes_released is True
    assert client.releases == [training_id]


def test_supervisor_persists_live_diagnostics_before_training_finishes(tmp_path, monkeypatch):
    class PendingCluster(FakeCluster):
        def __init__(self):
            super().__init__(state=TrainingState.RUNNING)
            self.log_reads = 0

        def logs(self, resource):
            self.log_reads += 1
            return "[pod/worker/checkout] terminated: Error exit=1\nPermission denied"

    client = PendingCluster()
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    try:
        deadline = time.monotonic() + 2
        while True:
            result = runtime.get_training_status(started.training_id)
            if result.kubernetes_diagnostics:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result.state is TrainingState.RUNNING
        assert "Permission denied" in result.kubernetes_diagnostics
        assert "Permission denied" in Path(result.log_path).read_text()
        assert client.log_reads == 1
    finally:
        runtime.cancel_training(started.training_id)


def test_supervisor_reports_diagnostics_failure_without_losing_training(tmp_path, monkeypatch):
    class BrokenLogsCluster(FakeCluster):
        def logs(self, resource):
            raise RuntimeError("HTTP 403")

    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, BrokenLogsCluster())
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    runtime.drain()
    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.FINISHED
    assert "Kubernetes diagnostics unavailable: RuntimeError: HTTP 403" in result.kubernetes_diagnostics


def test_slow_diagnostics_do_not_hide_remote_completion(tmp_path, monkeypatch):
    logs_started = threading.Event()
    finish_logs = threading.Event()
    completion_observed = threading.Event()

    class SlowDiagnosticsCluster(FakeCluster):
        def logs(self, resource):
            logs_started.set()
            assert finish_logs.wait(5)
            return "worker diagnostics"

        def state(self, resource):
            snapshot = super().state(resource)
            if snapshot[0] is TrainingState.FINISHED:
                completion_observed.set()
            return snapshot

    client = SlowDiagnosticsCluster(TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    try:
        assert logs_started.wait(2)
        client.state_value = TrainingState.FINISHED
        assert completion_observed.wait(2), "diagnostics blocked status polling"
        # The completion decision must survive a deadline passing during log reads.
        with runtime._lock:
            runtime._active[started.training_id].deadline_at = time.time() - 1
    finally:
        finish_logs.set()
        runtime.drain()

    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.FINISHED
    assert result.kubernetes_released is True
    assert client.deletions == []


def test_long_workload_message_keeps_container_failure_details(tmp_path, monkeypatch):
    class VerboseFailure(FakeCluster):
        def state(self, resource):
            return TrainingState.FAILED, "workload status " * 1000

        def logs(self, resource):
            return "[pod/worker/checkout] Permission denied opening source.bundle"

    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, VerboseFailure())
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    runtime.drain()
    result = runtime.get_training_status(started.training_id)
    assert len(result.kubernetes_diagnostics.encode()) <= 8192
    assert "Permission denied" in result.kubernetes_diagnostics
    assert "Permission denied" in Path(result.log_path).read_text()


def test_completion_refreshes_diagnostics_after_an_inflight_live_read(tmp_path, monkeypatch):
    live_read_started = threading.Event()
    finish_live_read = threading.Event()

    class DelayedLiveDiagnostics(FakeCluster):
        def logs(self, resource):
            if not live_read_started.is_set():
                live_read_started.set()
                assert finish_live_read.wait(5)
                return "stale live snapshot"
            return "final failure details"

    client = DelayedLiveDiagnostics(TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    try:
        assert live_read_started.wait(2)
        client.state_value = TrainingState.FAILED
        runtime.drain()
        result = runtime.get_training_status(started.training_id)
        assert result.state is TrainingState.FAILED
        assert "final failure details" in result.kubernetes_diagnostics
        assert "stale live snapshot" not in result.kubernetes_diagnostics
    finally:
        finish_live_read.set()
        runtime.drain()


@pytest.mark.parametrize("nodes", [4, 1000])
@pytest.mark.parametrize("research", [
    "cfd-batch-jc-vandam-sep23",
    "r" * 33 + "-" + "r" * 40,
])
def test_training_identity_leaves_room_for_mpi_child_names(monkeypatch, nodes, research):
    monkeypatch.setenv("RESEARCH_TAG", research)
    monkeypatch.setenv("STUDENT_NAME", "JC_Vandam Edward with a long student name")
    monkeypatch.setenv("SENPAI_KUBERNETES_NAMESPACE", "research")
    training_id = "d30b8238-81b6-4841-b541-9df20cddf6f9"

    spec = kubernetes_training._training_spec(training_id, nodes=nodes)

    assert spec.wandb_run_id == training_id.replace("-", "")
    assert spec.name.endswith("-d30b823881b6")
    assert not spec.name.removesuffix("-d30b823881b6").endswith("-")
    for name in (spec.name, f"{spec.name}-launcher", f"{spec.name}-worker-{nodes - 1}"):
        assert len(name) <= 63
        assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", name)
