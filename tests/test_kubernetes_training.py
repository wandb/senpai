from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

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
