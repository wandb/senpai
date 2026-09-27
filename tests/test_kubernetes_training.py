from __future__ import annotations

import errno
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time

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


from training_test_support import FakeCluster, git_workspace


def supervisor(tmp_path, monkeypatch, client=None, **overrides):
    workspace = git_workspace(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    environment = {
        "RESEARCH_TAG": "fred",
        "STUDENT_NAME": "fern",
        "SENPAI_KUBERNETES_NAMESPACE": "research",
        "SENPAI_LAUNCH_SECRET_NAME": "senpai-launch-secrets-fred",
        "SENPAI_TRAINING_SNAPSHOT_ROOT": str(snapshot_root),
        "SENPAI_TRAINING_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "SENPAI_TRAINING_IMAGE": "ghcr.io/wandb/senpai-student@sha256:" + "a" * 64,
        "CPU_PER_STUDENT_GPU": "1", "MEMORY_GI_PER_STUDENT_GPU": "2",
        "PVC_CLAIM_NAME": "dataset", "PVC_MOUNT_PATH": str(tmp_path / "data"),
        "WANDB_ENTITY": "entity", "WANDB_PROJECT": "project",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    values = {
        "workspace": workspace,
        "state_dir": tmp_path / "state",
        "nodes": 2,
        "gpus_per_node": 8,
        "max_timeout_seconds": 10,
        "poll_seconds": 0.01,
        "client": client or FakeCluster(),
    }
    values.update(overrides)
    return KubernetesTrainingSupervisor(**values), workspace, snapshot_root


@pytest.mark.parametrize("nodes", [1, 2])
def test_training_command_is_submitted_to_workers_without_running_on_controller(
    tmp_path, monkeypatch, nodes,
):
    class SubmittedCluster(FakeCluster):
        def __init__(self):
            super().__init__(nodes=nodes)
            self.manifests = []

        def apply(self, manifest):
            self.manifests.append(json.loads(manifest))

    client = SubmittedCluster()
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client, nodes=nodes)
    marker = workspace / "must-not-run-here"
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"),
        cwd=workspace, timeout_seconds=5,
    ))
    runtime.drain()
    assert not marker.exists()
    manifest, = client.manifests
    assert manifest["kind"] == ("Job" if nodes == 1 else "MPIJob")
    assert manifest["metadata"]["name"] == started.kubernetes_spec.name
    template = (
        manifest["spec"]["template"] if nodes == 1
        else manifest["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]
    )
    environment = {item["name"]: item.get("value") for item in template["spec"]["containers"][0]["env"]}
    import base64
    payload = json.loads(base64.b64decode(environment["SENPAI_TRAINING_COMMAND_B64"]))
    assert payload["argv"] == [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    assert payload["cwd"] == "/workspace"
    result = runtime.get_training_status(started.training_id)
    assert result.kubernetes_released is True
    assert result.source_commit == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
    ).strip()
    assert subprocess.check_output(
        ["git", "bundle", "list-heads", result.source_snapshot, "HEAD"], text=True,
    ).split() == [result.source_commit, "HEAD"]
    assert environment["SENPAI_TRAINING_OUTPUT_DIR"] == result.output_dir
    runtime.close()


@pytest.mark.parametrize("record", ["sidecar", "corrupt"])
def test_recovery_reads_only_valid_owned_training_records(tmp_path, monkeypatch, record):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    training_id = "b81440b1-b803-471e-9fe0-6dcabd756b83"
    path = state_dir / f"{training_id}{'.score' if record == 'sidecar' else ''}.json"
    contents = '{"metrics": {}, "passed": true, "score": 1.0}'
    path.write_text(contents)
    if record == "corrupt":
        with pytest.raises(ValueError):
            supervisor(tmp_path, monkeypatch)
    else:
        runtime, _, _ = supervisor(tmp_path, monkeypatch)
        runtime.close()
    assert path.read_text() == contents


def test_remote_diagnostics_and_receipt_do_not_persist_wandb_key(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "shared-research-key")

    class LeakyCluster(FakeCluster):
        def logs(self, resource):
            return "remote key=shared-research-key"

        def state(self, resource):
            return TrainingState.FAILED, "failure key=shared-research-key"

        def release(self, training_id):
            super().release(training_id)
            return {"capture_error": "key=shared-research-key"}

    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, LeakyCluster())
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    runtime.drain()
    result = runtime.get_training_status(started.training_id)
    assert result.state is TrainingState.FAILED
    assert "remote key=<secret-hidden>" in result.kubernetes_diagnostics
    assert "failure key=<secret-hidden>" in result.error_tail
    assert result.kubernetes_pod_receipt == {"capture_error": "key=<secret-hidden>"}
    assert "shared-research-key" not in Path(result.log_path).read_text()
    assert "shared-research-key" not in (tmp_path / "state" / f"{started.training_id}.json").read_text()


def test_monitor_start_failure_cleans_up_and_recovers_terminal_verdict(
    tmp_path, monkeypatch,
):
    client = FakeCluster(state=TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    original_start = threading.Thread.start
    threads = []

    def fail_start(thread):
        threads.append(thread)
        if len(threads) == 1:
            raise RuntimeError("thread resources exhausted")
        original_start(thread)

    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", fail_start)
        with pytest.raises(RuntimeError, match="thread resources exhausted"):
            runtime.run_training(TrainingSpec(
                argv=(sys.executable, "-c", "import time;time.sleep(30)"),
                cwd=workspace, timeout_seconds=5,
            ))
    try:
        assert all(not thread.is_alive() for thread in threads)
        saved, = (tmp_path / "state").glob("*.json")
        result = TrainingResult.model_validate_json(saved.read_text())
        assert result.state is TrainingState.FAILED
        assert result.pid is None
        assert client.deletions and client.releases == [result.training_id]
        recovered = KubernetesTrainingSupervisor(
            workspace=workspace, state_dir=tmp_path / "state", nodes=2,
            gpus_per_node=8, poll_seconds=.01, client=client,
        )
        recovered.drain()
        assert recovered.get_training_status(result.training_id).state is TrainingState.FAILED
        assert client.adoptions == []
        recovered.close()
    finally:
        runtime.close()


@pytest.mark.parametrize("invalid", ["working_directory", "timeout"])
def test_invalid_launch_never_reserves_a_remote_workload(tmp_path, monkeypatch, invalid):
    client = FakeCluster()
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    try:
        with pytest.raises(ValueError, match="inside|configured maximum"):
            runtime.run_training(TrainingSpec(
                argv=(sys.executable, "-c", "raise AssertionError('must not run')"),
                cwd=workspace.parent if invalid == "working_directory" else workspace,
                timeout_seconds=11 if invalid == "timeout" else 5,
            ))
        assert client.reservations == []
    finally:
        runtime.close()


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


@pytest.mark.parametrize(
    ("lookup_outage", "late_create"),
    [(False, False), (True, False), (False, True)],
    ids=["missing", "lookup-outage", "late-create"],
)
def test_successful_submission_without_a_workload_fails_promptly(
    tmp_path, monkeypatch, lookup_outage, late_create,
):
    released = threading.Event()

    class MissingWorkload(FakeCluster):
        def __init__(self):
            super().__init__()
            self.lookups = 0

        def reserve(self, *args):
            super().reserve(*args)
            self.pending_resource = self.resource_value
            self.resource_value = None

        def resource(self, *args, **kwargs):
            self.lookups += 1
            if lookup_outage and self.lookups == 1:
                raise TimeoutError("temporary ownership lookup outage")
            return super().resource(*args, **kwargs)

        def release(self, training_id):
            if late_create and self.pending_resource is not None:
                self.resource_value = self.pending_resource
                self.pending_resource = None
            if self.resource_value is not None:
                raise RuntimeError("cannot release a live Kubernetes workload")
            super().release(training_id)
            released.set()

    client = MissingWorkload()
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
    ))
    try:
        assert released.wait(1), "successful no-op submission waited for its deadline"
        runtime.drain()
        result = runtime.get_training_status(started.training_id)
        assert result.state is TrainingState.FAILED
        assert "submission returned without a workload" in result.error_tail
        assert result.kubernetes_released is True
        assert client.lookups == (2 if lookup_outage else 1)
        assert len(client.deletions) == (1 if late_create else 0)
    finally:
        runtime.close()


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("write_phase", ["resource", "terminal"])
def test_cancellation_deletes_remote_before_retrying_full_terminal_storage(
    tmp_path, monkeypatch, restart, write_phase,
):
    client = FakeCluster(TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(
        tmp_path, monkeypatch, client, max_timeout_seconds=60,
    )
    deleted = threading.Event()
    storage_failed = threading.Event()
    terminal_failed = threading.Event()
    storage_restored = threading.Event()
    write_text = Path.write_text
    delete = client.delete

    def observed_delete(*args, **kwargs):
        delete(*args, **kwargs)
        deleted.set()

    monkeypatch.setattr(client, "delete", observed_delete)

    def exhausted_terminal(path, text, *args, **kwargs):
        if path.parent == runtime.state_dir and path.suffix == ".tmp":
            record = json.loads(text)
            phase = (
                "terminal" if record["state"] != "running" else
                "resource" if record["kubernetes_resource"] else "launch"
            )
            if (phase == write_phase or storage_failed.is_set()) and not storage_restored.is_set():
                storage_failed.set()
                if phase == "terminal":
                    terminal_failed.set()
                raise OSError(errno.ENOSPC, "storage exhausted", str(path))
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", exhausted_terminal)
    spec = TrainingSpec(argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=30)
    started = runtime.run_training(spec)
    original = client.resource_value
    if write_phase == "resource":
        assert storage_failed.wait(2)
    errors = []

    def cancel():
        try:
            runtime.cancel_training(started.training_id)
        except Exception as error:
            errors.append(error)

    cancelling = threading.Thread(target=cancel)
    cancelling.start()
    recovered = None
    try:
        assert terminal_failed.wait(2), "cancellation was blocked by resource publication"
        assert deleted.wait(2), "full terminal storage blocked workload deletion"
        assert client.deletions == [original]
        assert client.resource_value is None
        assert client.releases == []
        assert runtime.get_training_status(started.training_id).state is TrainingState.RUNNING
        with pytest.raises(RuntimeError, match="already has an active"):
            runtime.run_training(spec)
        if restart:
            runtime.close()
            cancelling.join(1)
            reserve = client.reserve

            def preserve_deleted_workload(*args):
                reserve(*args)
                client.resource_value = None

            monkeypatch.setattr(client, "reserve", preserve_deleted_workload)
            storage_restored.set()
            recovered = KubernetesTrainingSupervisor(
                workspace=workspace, state_dir=runtime.state_dir, nodes=2, gpus_per_node=8,
                poll_seconds=0.01, client=client,
            )
            recovered.drain()
        else:
            storage_restored.set()
            cancelling.join(2)
        result = (recovered or runtime).get_training_status(started.training_id)
        assert not cancelling.is_alive()
        assert errors == []
        assert client.adoptions == []
        assert result.state is TrainingState.CANCELLED
        assert result.kubernetes_released is True
        assert client.releases == [started.training_id]
    finally:
        storage_restored.set()
        cancelling.join(2)
        runtime.close()
        if recovered is not None:
            recovered.close()


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_stop_during_submission_retains_verdict_and_releases_workload(
    tmp_path, monkeypatch, stop,
):
    submitting = threading.Event()
    finish_submission = threading.Event()

    class BlockedSubmission(FakeCluster):
        def apply(self, manifest):
            submitting.set()
            assert finish_submission.wait(3)
            raise TimeoutError("submission response unavailable")

    client = BlockedSubmission(TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(TrainingSpec(
        argv=("python", "train.py"), cwd=workspace,
        timeout_seconds=1 if stop == "deadline" else 5,
    ))
    assert submitting.wait(2)
    cancelling = None
    try:
        if stop == "cancel":
            cancelling = threading.Thread(target=runtime.cancel_training, args=(started.training_id,))
            cancelling.start()
            while not runtime._active[started.training_id].cancelled:
                time.sleep(0.001)
        else:
            time.sleep(max(0, started.deadline_at - time.time()) + 0.01)
        finish_submission.set()
        runtime.drain()
        result = runtime.get_training_status(started.training_id)
        assert result.state is (TrainingState.CANCELLED if stop == "cancel" else TrainingState.TIMED_OUT)
        assert result.kubernetes_released is True
        assert len(client.deletions) == 1
        assert client.releases == [started.training_id]
    finally:
        finish_submission.set()
        if cancelling is not None:
            cancelling.join(3)
        runtime.close()


def test_failed_submission_remains_terminal_after_delete_outage_and_restart(
    tmp_path, monkeypatch,
):
    delete_attempted = threading.Event()
    delete_available = threading.Event()

    class UnavailableDeletion(FakeCluster):
        def apply(self, manifest):
            raise RuntimeError("fixture submission failed")

        def delete(self, *args, **kwargs):
            delete_attempted.set()
            if not delete_available.is_set():
                raise TimeoutError("executor unavailable during deletion")
            super().delete(*args, **kwargs)

    client = UnavailableDeletion(TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(
        tmp_path, monkeypatch, client, max_timeout_seconds=60,
    )
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
        cwd=workspace, timeout_seconds=30,
    ))
    original = client.resource_value
    recovered = None
    try:
        assert delete_attempted.wait(2)
        runtime.close()
        durable = runtime.get_training_status(started.training_id)
        assert durable.state is TrainingState.FAILED
        assert durable.kubernetes_released is False
        assert client.resource_value == original
        assert client.releases == []

        delete_available.set()
        recovered = KubernetesTrainingSupervisor(
            workspace=workspace, state_dir=runtime.state_dir, nodes=2, gpus_per_node=8,
            poll_seconds=0.01, client=client,
        )
        recovered.drain()
        result = recovered.get_training_status(started.training_id)
        assert result.state is TrainingState.FAILED
        assert result.kubernetes_released is True
        assert client.adoptions == []
        assert client.deletions == [original]
        assert client.releases == [started.training_id]
    finally:
        delete_available.set()
        runtime.close()
        if recovered is not None:
            recovered.close()


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


@pytest.mark.parametrize("storage_errno", [errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EROFS])
@pytest.mark.parametrize("write_phase", ["log", "summary"])
def test_storage_error_in_optional_diagnostics_does_not_fail_training(
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
    selected = threading.Event()
    finish = threading.Event()

    class BlockingDelete(FakeCluster):
        def delete(self, resource, timeout_seconds=60):
            selected.set()
            assert finish.wait(5)
            super().delete(resource, timeout_seconds)

    client = BlockingDelete(state=TrainingState.RUNNING)
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    started = runtime.run_training(TrainingSpec(
        argv=("python", "train.py"), cwd=workspace, timeout_seconds=1,
    ))
    close_thread = threading.Thread(target=runtime.close)
    try:
        assert selected.wait(3)
        close_thread.start()
        assert runtime._shutdown.wait(1)
        finish.set()
        close_thread.join(3)
        assert not close_thread.is_alive()
        result = runtime.get_training_status(started.training_id)
        assert result.state is TrainingState.TIMED_OUT
        assert result.kubernetes_released is True
    finally:
        finish.set()
        runtime.close()
        if close_thread.ident is not None:
            close_thread.join(3)


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
    key = "shared-wandb-sentinel"
    monkeypatch.setenv("WANDB_API_KEY", key)

    class PendingCluster(FakeCluster):
        def __init__(self):
            super().__init__(state=TrainingState.RUNNING)
            self.log_reads = 0

        def logs(self, resource):
            self.log_reads += 1
            return (
                "[pod/worker/checkout] terminated: Error exit=1\nPermission denied\n"
                f"key={key}\npartial key={key[:12]}"
            )

        def state(self, resource):
            return self.state_value, f"workload key={key}"

    client = PendingCluster()
    runtime, workspace, _ = supervisor(
        tmp_path, monkeypatch, client,
    )
    started = runtime.run_training(TrainingSpec(
        argv=(sys.executable, "-c", (
            "import os; print('launcher', os.environ['WANDB_API_KEY'])"
        )),
        cwd=workspace,
        timeout_seconds=5,
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
        log = Path(result.log_path).read_text()
        assert "Permission denied" in log
        assert key not in log
        assert "partial key=<secret-hidden>" in log
        assert key not in result.model_dump_json()
        assert key not in (runtime.state_dir / f"{started.training_id}.json").read_text()
        assert client.log_reads == 1
    finally:
        terminal = runtime.cancel_training(started.training_id)
    assert terminal.state is TrainingState.CANCELLED
    assert "key=<secret-hidden>" in terminal.error_tail
    assert key not in terminal.model_dump_json()
    assert key not in Path(terminal.log_path).read_text()
    assert key not in (runtime.state_dir / f"{started.training_id}.json").read_text()


@pytest.mark.parametrize("source", ["container", "event"])
def test_supervisor_redacts_diagnostics_before_truncation(tmp_path, monkeypatch, source):
    key = "0123456789abcdef0123456789abcdef012345ab"
    monkeypatch.setenv("WANDB_API_KEY", key)
    api = object.__new__(kubernetes_training.KubernetesApiClient)
    worker = {
        "metadata": {"name": "worker", "uid": "worker-uid"},
        "spec": {"containers": [{"name": "train"}]},
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "train", "state": {"running": {}}}],
        },
    }
    monkeypatch.setattr(api, "_owned_pods", lambda _resource: [worker])

    def request_json(method, path, **kwargs):
        uid = "worker-uid" if "worker-uid" in path else "remote-uid"
        return {"items": [{
            "metadata": {"creationTimestamp": "2026-09-27T00:00:00Z"},
            "involvedObject": {"uid": uid},
            "message": key * 40 if source == "event" else "worker started",
        }]}

    monkeypatch.setattr(api, "_request_json", request_json)
    output = (key * 220)[:8192] if source == "container" else "training output"
    monkeypatch.setattr(
        api, "_request_text", lambda *args, **kwargs: output,
    )
    client = FakeCluster(state=TrainingState.FAILED)
    transport = kubernetes_training.KubernetesExecutorClient("unused.sock")
    monkeypatch.setattr(
        transport, "_request",
        lambda *args, **kwargs: json.loads(json.dumps(api.logs(client.resource_value))),
    )
    monkeypatch.setattr(client, "logs", transport.logs)
    runtime, workspace, _ = supervisor(tmp_path, monkeypatch, client)
    try:
        started = runtime.run_training(TrainingSpec(
            argv=(sys.executable, "-c", "pass"), cwd=workspace, timeout_seconds=5,
        ))
        runtime.drain()
        result = runtime.get_training_status(started.training_id)
        assert result.state is TrainingState.FAILED
        for persisted in (
            Path(result.log_path).read_text(),
            (runtime.state_dir / f"{started.training_id}.json").read_text(),
            result.model_dump_json(),
        ):
            assert "<secret-hidden>" in persisted
            assert key[:8] not in persisted
            assert key[-8:] not in persisted
        assert len(result.kubernetes_diagnostics.encode()) <= 8192
    finally:
        runtime.close()


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


def test_supervisor_persists_executor_receipt_with_terminal_acknowledgement(tmp_path, monkeypatch):
    receipt = {'training_id': 'from-executor', 'source_commit': 'a' * 40,
               'resource': {'uid': 'remote-uid'}, 'complete': False,
               'capture_error': 'Missing one terminal worker',
               'pods': [{'uid': 'observed-worker', 'containers': [{'restartCount': None}]}]}

    class ReceiptCluster(FakeCluster):
        def release(self, training_id):
            super().release(training_id)
            return receipt

    runtime, workspace, _snapshot_root = supervisor(tmp_path, monkeypatch, ReceiptCluster())
    started = runtime.run_training(TrainingSpec(argv=(sys.executable, '-c', 'pass'),
                                               cwd=workspace, timeout_seconds=5))
    runtime.drain()
    saved = TrainingResult.model_validate_json(
        (tmp_path / 'state' / (started.training_id + '.json')).read_text()
    )
    assert saved.kubernetes_released is True
    assert saved.kubernetes_pod_receipt == receipt
