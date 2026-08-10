import json
import stat
import subprocess

import pytest

from k8s import supervisor_rollback


def completed(argv, *, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_capture_persists_exact_present_and_absent_resources_at_mode_0600(
    monkeypatch,
    tmp_path,
):
    calls = []
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "senpai-supervisor-campaign-a",
            "namespace": "research-a",
            "labels": {"release": "old"},
            "annotations": {"owner": "research"},
            "resourceVersion": "1234",
            "uid": "server-only",
            "managedFields": [{"manager": "kubectl"}],
            "creationTimestamp": "2026-08-10T10:00:00Z",
            "generation": 7,
        },
        "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "old"}}},
        "status": {"readyReplicas": 1},
    }

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "deployment.apps/senpai-supervisor-campaign-a" in argv:
            return completed(argv, stdout=json.dumps(deployment))
        return completed(argv)

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", run)

    rollback = supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        kube_context="gpu-cluster",
        namespace="research-a",
        directory=tmp_path,
    )

    assert stat.S_IMODE(rollback.path.stat().st_mode) == 0o600
    bundle = json.loads(rollback.path.read_text())
    assert bundle["schema"] == "senpai-supervisor-rollback/v1"
    assert bundle["kube_context"] == "gpu-cluster"
    assert bundle["namespace"] == "research-a"
    assert bundle["persistent_state_rolled_back"] is False
    assert "SQLite" in bundle["operator_notice"]
    recovery = bundle["manual_restore_argv"]
    assert recovery[-4:] == [
        "restore",
        str(rollback.path),
        "--timeout-seconds",
        "900",
    ]
    assert len(bundle["resources"]) == 5
    saved = next(
        item for item in bundle["resources"] if item["kind"] == "Deployment"
    )
    assert saved["present"] is True
    assert saved["manifest"]["metadata"] == {
        "name": "senpai-supervisor-campaign-a",
        "namespace": "research-a",
        "labels": {"release": "old"},
        "annotations": {"owner": "research"},
    }
    assert "status" not in saved["manifest"]
    assert sum(not item["present"] for item in bundle["resources"]) == 4
    assert "Secret" not in {item["kind"] for item in bundle["resources"]}
    assert [argv for argv, _kwargs in calls] == [
        [
            "kubectl",
            "--context",
            "gpu-cluster",
            "--namespace",
            "research-a",
            "get",
            resource,
            "--ignore-not-found",
            "-o",
            "json",
        ]
        for resource in (
            "networkpolicy.networking.k8s.io/senpai-supervisor-egress-campaign-a",
            "serviceaccount/senpai-supervisor-campaign-a",
            "role.rbac.authorization.k8s.io/senpai-supervisor-campaign-a",
            "rolebinding.rbac.authorization.k8s.io/senpai-supervisor-campaign-a",
            "deployment.apps/senpai-supervisor-campaign-a",
        )
    ]


def test_restore_reapplies_present_resources_deletes_previously_absent_resources(
    monkeypatch,
    tmp_path,
):
    calls = []
    old_deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "senpai-supervisor-campaign-a"},
        "spec": {"replicas": 1},
    }

    def capture_run(argv, **kwargs):
        if "deployment.apps/senpai-supervisor-campaign-a" in argv:
            return completed(argv, stdout=json.dumps(old_deployment))
        return completed(argv)

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", capture_run)
    rollback = supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        kube_context="gpu-cluster",
        namespace="research-a",
        directory=tmp_path,
    )

    def restore_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "get" in argv:
            current = dict(old_deployment)
            current["metadata"] = {
                "name": "senpai-supervisor-campaign-a",
                "resourceVersion": "new-server-version",
            }
            return completed(argv, stdout=json.dumps(current))
        return completed(argv, stdout="restored")

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", restore_run)
    rollback.restore(timeout_seconds=47)

    replace = next((argv, kwargs) for argv, kwargs in calls if "replace" in argv)
    assert replace[0] == [
        "kubectl",
        "--context",
        "gpu-cluster",
        "--namespace",
        "research-a",
        "replace",
        "-f",
        "-",
    ]
    restored = json.loads(replace[1]["input"])
    assert restored["kind"] == "Deployment"
    assert restored["metadata"]["resourceVersion"] == "new-server-version"
    assert [argv for argv, _kwargs in calls if "delete" in argv] == [
        [
            "kubectl",
            "--context",
            "gpu-cluster",
            "--namespace",
            "research-a",
            "delete",
            resource,
            "--ignore-not-found",
        ]
        for resource in (
            "networkpolicy.networking.k8s.io/senpai-supervisor-egress-campaign-a",
            "serviceaccount/senpai-supervisor-campaign-a",
            "role.rbac.authorization.k8s.io/senpai-supervisor-campaign-a",
            "rolebinding.rbac.authorization.k8s.io/senpai-supervisor-campaign-a",
        )
    ]
    assert [argv for argv, _kwargs in calls if "rollout" in argv] == [
        [
            "kubectl",
            "--context",
            "gpu-cluster",
            "--namespace",
            "research-a",
            "rollout",
            "status",
            "deployment/senpai-supervisor-campaign-a",
            "--timeout=47s",
        ]
    ]
    assert rollback.path.exists()


def test_restore_reports_every_failed_operation_and_preserves_bundle(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        supervisor_rollback.subprocess,
        "run",
        lambda argv, **_kwargs: completed(argv),
    )
    rollback = supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        namespace="research-a",
        directory=tmp_path,
    )
    attempted = []

    def fail_delete(argv, **_kwargs):
        attempted.append(argv)
        return completed(argv, returncode=1, stderr="forbidden")

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", fail_delete)

    with pytest.raises(supervisor_rollback.RollbackError) as raised:
        rollback.restore(timeout_seconds=10)

    assert len(attempted) == 5
    assert "forbidden" in str(raised.value)
    assert rollback.path.exists()


def test_commit_removes_the_rollback_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(
        supervisor_rollback.subprocess,
        "run",
        lambda argv, **_kwargs: completed(argv),
    )
    rollback = supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        namespace="research-a",
        directory=tmp_path,
    )

    rollback.commit()

    assert not rollback.path.exists()


def test_restore_recreates_a_previous_resource_deleted_during_the_failed_upgrade(
    monkeypatch,
    tmp_path,
):
    service_account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": "senpai-supervisor-campaign-a"},
        "automountServiceAccountToken": False,
    }

    def capture_run(argv, **_kwargs):
        if "serviceaccount/senpai-supervisor-campaign-a" in argv:
            return completed(argv, stdout=json.dumps(service_account))
        return completed(argv)

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", capture_run)
    rollback = supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        namespace="research-a",
        directory=tmp_path,
    )
    calls = []

    def restore_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed(argv)

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", restore_run)
    rollback.restore(timeout_seconds=10)

    create = next((argv, kwargs) for argv, kwargs in calls if "create" in argv)
    assert create[0][-3:] == ["create", "-f", "-"]
    assert json.loads(create[1]["input"])["kind"] == "ServiceAccount"


def test_capture_failure_stops_without_creating_a_rollback_bundle(
    monkeypatch,
    tmp_path,
):
    def fail_get(argv, **_kwargs):
        return completed(argv, returncode=1, stderr="cluster unavailable")

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", fail_get)

    with pytest.raises(RuntimeError, match="cluster unavailable"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            namespace="research-a",
            directory=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_bundle_persistence_failure_is_a_clean_capture_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        supervisor_rollback.subprocess,
        "run",
        lambda argv, **_kwargs: completed(argv),
    )
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied")

    with pytest.raises(RuntimeError, match="could not persist.*rollback bundle"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            namespace="research-a",
            directory=blocker / "child",
        )
