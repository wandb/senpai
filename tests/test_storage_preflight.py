"""Storage payload behavior and the Pod/cleanup boundary; no live cluster calls."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "k8s"))
import storage_preflight


def probe(mode, mount, roots, *, token="test-owner", readers=("student", "advisor"), timeout=3):
    return subprocess.Popen(
        [sys.executable, "-c", storage_preflight._PROBE, mode, token, str(mount),
         json.dumps([str(root) for root in roots]), json.dumps(readers), str(timeout)],
        env={**os.environ, "POD_NAME": mode + "-pod", "NODE_NAME": "local-test"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_probe_shares_atomic_checkpoints_and_cleans_only_its_directories(tmp_path):
    """Run the exact Pod payload in separate processes against real files."""
    roots = [tmp_path / "runs" / "fern", tmp_path / "runs" / "frieren"]
    roots[0].mkdir(parents=True)
    existing = roots[0] / "existing-checkpoint"
    existing.write_bytes(b"keep this training output")
    processes = [probe(role, tmp_path, roots) for role in ("student", "advisor", "writer")]
    try:
        receipts = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=8)
            assert process.returncode == 0, stderr
            receipts.append(json.loads(stdout.removeprefix(storage_preflight._RESULT)))
        assert {receipt["role"] for receipt in receipts} == {"student", "advisor", "writer"}
        assert all(receipt["checkpoint_cycle"] == receipt["cleanup"] == "passed" for receipt in receipts)
        assert all(receipt["uid"] == os.geteuid() for receipt in receipts)
        assert existing.read_bytes() == b"keep this training output"
        assert list(roots[0].iterdir()) == [existing]
        assert list(roots[1].iterdir()) == []
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_probe_timeout_removes_partial_checkpoint_without_touching_other_outputs(tmp_path):
    output = tmp_path / "run"
    process = probe("writer", tmp_path, [output], timeout=0.1)
    _, stderr = process.communicate(timeout=5)
    assert process.returncode != 0 and "storage handshake timed out" in stderr
    assert list(output.iterdir()) == []


def test_probe_refuses_preexisting_or_escaped_output_directory(tmp_path):
    mount = tmp_path / "pvc"
    mount.mkdir()
    output = mount / "run"
    existing = output / ".preflight-test-owner"
    existing.mkdir(parents=True)
    protected = existing / "worker-checkpoint"
    protected.write_bytes(b"another run")
    process = probe("writer", mount, [output])
    _, stderr = process.communicate(timeout=5)
    assert process.returncode != 0 and "FileExistsError" in stderr
    assert protected.read_bytes() == b"another run"
    outside = tmp_path / "outside"
    outside.mkdir()
    (mount / "link").symlink_to(outside, target_is_directory=True)
    process = probe("writer", mount, [mount / "link" / "must-not-create"])
    _, stderr = process.communicate(timeout=5)
    assert process.returncode != 0 and "outside the PVC mount" in stderr
    assert list(outside.iterdir()) == []


class PodApi:
    """Replay only Kubernetes responses; storage behavior has its own real-process test."""

    def __init__(self, *, failure=None, writer_node="worker-host"):
        self.failure = failure
        self.writer_node = writer_node
        self.pods = {}
        self.manifests = []
        self.deletes = []

    def __call__(self, command, *, input, **kwargs):
        assert command[:3] == ["kubectl", "--context", "test-cluster"]
        assert command[3:5] == ["--namespace", "test-namespace"]
        assert command[5].startswith("--request-timeout=")
        arguments = command[6:]
        verb = arguments[0]
        if verb == "create":
            pod = json.loads(input)
            self.manifests.append(pod)
            role = pod["spec"]["containers"][0]["args"][0]
            pod["metadata"]["uid"] = "uid-" + role
            pod["spec"]["nodeName"] = self.writer_node if role == "writer" else "controller-host"
            pod["status"] = {"phase": "Running"}
            if self.failure == "unscheduled":
                pod["spec"].pop("nodeName")
                pod["status"]["phase"] = "Pending"
            self.pods[pod["metadata"]["name"]] = pod
            if role == "writer":
                for current in self.pods.values():
                    current["status"]["phase"] = "Succeeded"
                if self.failure == "pod":
                    pod["status"]["phase"] = "Failed"
                if self.failure == "ambiguous-create":
                    return subprocess.CompletedProcess(command, 1, "", "lost create response")
                if self.failure == "replaced":
                    result = json.dumps(pod)
                    pod["metadata"]["uid"] = "replacement-uid"
                    return subprocess.CompletedProcess(command, 0, result, "")
            result = pod
        elif verb == "get" and arguments[1] == "pods":
            result = {"items": list(self.pods.values())}
        elif verb == "get" and arguments[1] == "events":
            assert arguments[3].startswith("involvedObject.uid=")
            result = {"items": [{"message": "FailedScheduling: no eligible storage node"}]}
        elif verb == "get":
            result = self.pods.get(arguments[2])
            if result is None:
                return subprocess.CompletedProcess(command, 0, "", "")
        elif verb == "logs":
            pod = self.pods[arguments[1]]
            if pod["status"]["phase"] == "Failed":
                return subprocess.CompletedProcess(command, 0, "PermissionError: checkpoint is not writable", "")
            role, token, _, roots, *_ = pod["spec"]["containers"][0]["args"]
            receipt = {"role": role, "token": token, "uid": 10001, "gid": 10001,
                       "pod": arguments[1], "node": pod["spec"]["nodeName"], "output_roots": json.loads(roots),
                       "checkpoint_cycle": "passed", "cleanup": "passed"}
            return subprocess.CompletedProcess(command, 0, storage_preflight._RESULT + json.dumps(receipt), "")
        elif verb == "delete":
            assert arguments[1] == "--raw" and arguments[-2:] == ["-f", "-"]
            name = arguments[2].rsplit("/", 1)[1]
            options = json.loads(input)
            assert options["preconditions"] == {"uid": self.pods[name]["metadata"]["uid"]}
            assert options["gracePeriodSeconds"] == 5
            self.deletes.append(name)
            del self.pods[name]
            result = {"status": "Success"}
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")


def validate(**overrides):
    options = {"images": {"student": "student@sha256:pinned", "advisor": "advisor@sha256:pinned"},
               "pvc_claim_name": "training-storage", "pvc_mount_path": "/mnt/data",
               "output_roots": ["/mnt/data/.senpai/runs/track/fern"], "nodes_per_student": 2,
               "kube_context": "test-cluster", "namespace": "test-namespace",
               "controller_node_selector": {"pool": "cpu"}}
    return storage_preflight.validate_storage(**(options | overrides))


@pytest.mark.parametrize("nodes", [1, 2])
def test_preflight_uses_role_images_runtime_identity_and_uid_safe_cleanup(monkeypatch, nodes):
    api = PodApi()
    monkeypatch.setattr(storage_preflight.subprocess, "run", api)
    receipts = validate(nodes_per_student=nodes)
    assert len(receipts) == len(api.deletes) == 3 and not api.pods
    for pod in api.manifests:
        spec = pod["spec"]
        container = spec["containers"][0]
        role = container["args"][0]
        assert container["image"] == ("advisor@sha256:pinned" if role == "advisor" else "student@sha256:pinned")
        assert spec["securityContext"]["runAsUser"] == spec["securityContext"]["runAsGroup"] == 10001
        assert "fsGroup" not in spec["securityContext"]  # No implicit recursive PVC chown.
        assert spec["automountServiceAccountToken"] is False
        assert spec["volumes"] == [{"name": "storage", "persistentVolumeClaim": {"claimName": "training-storage"}}]
        assert container["volumeMounts"] == [{"name": "storage", "mountPath": "/mnt/data"}]
        assert "nvidia.com/gpu" not in container["resources"]["requests"]
        assert container["command"][:2] == ["/opt/senpai-venv/bin/python", "-P"]
        if role == "writer":
            assert spec.get("tolerations") == [
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
            ]
            assert "nvidia.com/gpu" not in container["resources"]["limits"]
            assert ("affinity" in spec) is (nodes > 1)
            if nodes > 1:
                term = spec["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
                assert term["topologyKey"] == "kubernetes.io/hostname"
                assert term["labelSelector"]["matchLabels"]["storage-role"] == "reader"
        else:
            assert not spec.get("tolerations")
            assert spec["nodeSelector"] == {"pool": "cpu"}


@pytest.mark.parametrize("failure,detail", [("pod", "PermissionError"), ("ambiguous-create", "lost create response")])
def test_preflight_cleans_owned_pods_after_failed_or_ambiguous_launch(monkeypatch, failure, detail):
    api = PodApi(failure=failure)
    monkeypatch.setattr(storage_preflight.subprocess, "run", api)
    with pytest.raises(RuntimeError, match=detail):
        validate()
    assert len(api.deletes) == 3 and not api.pods


def test_preflight_refuses_to_delete_a_replacement_pod(monkeypatch):
    api = PodApi(failure="replaced")
    monkeypatch.setattr(storage_preflight.subprocess, "run", api)
    with pytest.raises(RuntimeError, match="identity changed") as error:
        validate()
    assert len(api.pods) == 1
    assert next(iter(api.pods.values()))["metadata"]["uid"] == "replacement-uid"
    assert any("refusing to delete changed/unowned Pod" in note for note in error.value.__notes__)


def test_multinode_preflight_rejects_same_host_evidence(monkeypatch):
    api = PodApi(writer_node="controller-host")
    monkeypatch.setattr(storage_preflight.subprocess, "run", api)
    with pytest.raises(RuntimeError, match="different nodes"):
        validate()
    assert not api.pods


def test_preflight_bounds_pending_pods_and_reports_storage_context(monkeypatch):
    api = PodApi(failure="unscheduled")
    monkeypatch.setattr(storage_preflight.subprocess, "run", api)
    clock = [0]
    monkeypatch.setattr(storage_preflight.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(storage_preflight.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    with pytest.raises(RuntimeError, match="exceeded 3s") as error:
        validate(timeout_seconds=3)
    details = "\n".join(error.value.__notes__)
    assert "training-storage" in details and "FailedScheduling" in details
    assert "(unscheduled)" in details
    assert not api.pods
