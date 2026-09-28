# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Verify launch storage using the selected role images and runtime identity."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import subprocess
import time
from uuid import uuid4

from launch_helpers import kubectl_command


_LABEL = "senpai.wandb.com/storage-preflight"
_RESULT = "SENPAI_STORAGE_PREFLIGHT "
_PROBE = r'''
import json, os, signal, sys, time
from pathlib import Path

mode, token, mount, roots_json, readers_json, duration = sys.argv[1:]
roots = json.loads(roots_json)
readers = json.loads(readers_json)
deadline = time.monotonic() + float(duration)
payload = (token + "\n").encode()
directories = [Path(root) / (".preflight-" + token) for root in roots]

def wait_for(predicate):
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("storage handshake timed out: " + mode)
        time.sleep(0.2)

def publish(directory, name):
    temporary = directory / (name + ".tmp")
    destination = directory / name
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    assert temporary.read_bytes() == payload, "checkpoint readback differs"
    temporary.rename(destination)
    return destination

def checkpoint(directory, name):
    destination = publish(directory, name)
    assert destination.read_bytes() == payload, "renamed checkpoint differs"
    return destination

def stopped(signum, frame):
    raise SystemExit("storage probe stopped")

signal.signal(signal.SIGTERM, stopped)
if mode == "writer":
    owned = []
    try:
        for directory in directories:
            if not directory.parent.resolve().is_relative_to(Path(mount).resolve()):
                raise RuntimeError("output root resolves outside the PVC mount")
            directory.parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir(mode=0o700)
            owned.append(directory)
            checkpoint(directory, ".owner")
            checkpoint(directory, "worker-checkpoint")
        for directory in directories:
            for reader in readers:
                ack = directory / (reader + ".ack")
                wait_for(ack.exists)
                assert ack.read_bytes() == payload, "reader acknowledgement differs"
    finally:
        for directory in owned:
            owner = directory / ".owner"
            if owner.exists() and owner.read_bytes() != payload:
                raise RuntimeError("refusing to clean a changed preflight owner marker")
            names = [".owner", "worker-checkpoint",
                     *[name + suffix for name in readers for suffix in (".ack", ".checkpoint")]]
            for name in names:
                (directory / name).unlink(missing_ok=True)
                (directory / (name + ".tmp")).unlink(missing_ok=True)
            directory.rmdir()
else:
    for directory in directories:
        source = directory / "worker-checkpoint"
        wait_for(source.exists)
        assert (directory / ".owner").read_bytes() == payload, "writer ownership differs"
        assert source.read_bytes() == payload, "worker checkpoint is not readable"
        checkpoint(directory, mode + ".checkpoint").unlink()
        # Publish only after readback/deletion: the writer may immediately clean
        # this directory once every acknowledgement is visible.
        publish(directory, mode + ".ack")
    for directory in directories:
        wait_for(lambda: not directory.exists())

print("SENPAI_STORAGE_PREFLIGHT " + json.dumps({
    "role": mode, "token": token, "uid": os.geteuid(), "gid": os.getegid(),
    "pod": os.environ["POD_NAME"], "node": os.environ["NODE_NAME"],
    "output_roots": roots, "checkpoint_cycle": "passed", "cleanup": "passed",
}), flush=True)
'''


def validate_storage(
    *,
    images: dict[str, str],
    pvc_mount_path: str,
    pvc_claim_name: str,
    output_roots: list[str],
    nodes_per_student: int,
    kube_context: str = "",
    namespace: str = "default",
    controller_node_selector: dict[str, str] | None = None,
    timeout_seconds: int = 600,
    training_image: str = "",
    image_pull_secrets: list[str] | None = None,
) -> list[dict]:
    """Fail before launch if role users cannot share durable training outputs.

    Each selected role reads a worker-style checkpoint. Multi-node launches
    schedule the writer on a different host from every reader. Only this
    invocation's uniquely named Pods and checkpoint directories are removed.
    """
    if not images or set(images) - {"student", "advisor"} or not all(images.values()):
        raise ValueError("storage preflight requires selected student/advisor images")
    if nodes_per_student < 1 or timeout_seconds < 1 or not output_roots:
        raise ValueError("storage preflight requires output roots and positive limits")
    mount = PurePosixPath(pvc_mount_path)
    if not mount.is_absolute() or ".." in mount.parts:
        raise ValueError("storage preflight PVC mount must be an absolute path")
    for root in output_roots:
        path = PurePosixPath(root)
        if ".." in path.parts or path == mount or not path.is_relative_to(mount):
            raise ValueError("storage preflight output roots must be beneath the PVC mount")
    if len(set(output_roots)) != len(output_roots):
        raise ValueError("storage preflight output roots must be distinct")

    token = uuid4().hex
    readers = sorted(images)
    writer_image = training_image or images.get("student", images[readers[0]])
    roles = [*readers, "writer"]
    names = {role: f"senpai-storage-{token[:12]}-{role}" for role in roles}
    created: dict[str, str] = {}
    deadline = time.monotonic() + timeout_seconds

    def command(*arguments: str, input: str | None = None, timeout: float = 30):
        return subprocess.run(
            kubectl_command(
                f"--request-timeout={max(1, int(timeout))}s", *arguments,
                kube_context=kube_context, namespace=namespace,
            ),
            input=input, text=True, capture_output=True, timeout=timeout + 2, check=False,
        )

    def checked(*arguments: str, input: str | None = None) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"storage preflight exceeded {timeout_seconds}s")
        result = command(*arguments, input=input, timeout=min(30, remaining))
        if result.returncode:
            raise RuntimeError(f"storage preflight kubectl {arguments[0]} failed: {result.stderr.strip()[-2000:]}")
        return result.stdout

    def observe() -> list[dict]:
        items = json.loads(checked("get", "pods", "-l", f"{_LABEL}={token}", "-o", "json"))["items"]
        for pod in items:
            metadata = pod["metadata"]
            name = metadata["name"]
            if name not in created or metadata["uid"] != created[name]:
                raise RuntimeError("storage preflight Pod identity changed")
            if pod.get("status", {}).get("phase") == "Failed":
                logs = command("logs", name, "--tail=30", "--limit-bytes=4000")
                raise RuntimeError(f"storage preflight Pod {name} failed: {pod.get('status', {})}\n{logs.stdout}{logs.stderr}")
        return items

    def create(role: str) -> None:
        is_writer = role == "writer"
        # Do not set fsGroup: a preflight must detect permissions, not trigger
        # kubelet's recursive ownership changes on an existing shared volume.
        spec = {
            "restartPolicy": "Never", "automountServiceAccountToken": False,
            "activeDeadlineSeconds": timeout_seconds, "terminationGracePeriodSeconds": 5,
            "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "runAsGroup": 10001,
                                "seccompProfile": {"type": "RuntimeDefault"}},
            "containers": [{
                "name": "storage", "image": writer_image if is_writer else images[role],
                "command": (["python3", "-c", _PROBE] if is_writer and training_image
                            else ["/opt/senpai-venv/bin/python", "-P", "-c", _PROBE]),
                "args": [role, token, pvc_mount_path, json.dumps(output_roots), json.dumps(readers), str(timeout_seconds)],
                "env": [{"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                        {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}],
                "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                "resources": {"requests": {"cpu": "100m", "memory": "64Mi"},
                              "limits": {"cpu": "500m", "memory": "256Mi"}},
                "volumeMounts": [{"name": "storage", "mountPath": pvc_mount_path}],
            }],
            "volumes": [{"name": "storage", "persistentVolumeClaim": {"claimName": pvc_claim_name}}],
        }
        if image_pull_secrets:
            spec["imagePullSecrets"] = [{"name": name} for name in image_pull_secrets]
        if is_writer:
            spec["tolerations"] = [
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
            ]
        elif controller_node_selector:
            spec["nodeSelector"] = controller_node_selector
        if is_writer and nodes_per_student > 1:
            spec["affinity"] = {"podAntiAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [{
                "labelSelector": {"matchLabels": {_LABEL: token, "storage-role": "reader"}},
                "topologyKey": "kubernetes.io/hostname",
            }]}}
        manifest = {"apiVersion": "v1", "kind": "Pod", "metadata": {
            "name": names[role], "namespace": namespace,
            "labels": {_LABEL: token, "storage-role": "writer" if is_writer else "reader"},
        }, "spec": spec}
        pod = json.loads(checked("create", "-f", "-", "-o", "json", input=json.dumps(manifest)))
        created[names[role]] = pod["metadata"]["uid"]

    def failure_context() -> str:
        details = [f"PVC {namespace}/{pvc_claim_name} at {pvc_mount_path}; output roots: {output_roots}; "
                   f"isolated probe directory: .preflight-{token}"]
        diagnostic_deadline = time.monotonic() + 20
        try:
            current = command("get", "pods", "-l", f"{_LABEL}={token}", "-o", "json", timeout=5)
            if current.returncode:
                return "\n".join([*details, current.stderr.strip()[-1000:]])
            for pod in json.loads(current.stdout)["items"]:
                metadata = pod["metadata"]
                details.append(f"{metadata['name']} on {pod['spec'].get('nodeName', '(unscheduled)')}: {pod.get('status', {})}")
                remaining = diagnostic_deadline - time.monotonic()
                if remaining <= 0:
                    break
                events = command("get", "events", "--field-selector", f"involvedObject.uid={metadata['uid']}",
                                 "-o", "json", timeout=min(5, remaining))
                if events.returncode:
                    details.append(events.stderr.strip()[-1000:])
                else:
                    details.extend(event.get("message", "")[-500:] for event in json.loads(events.stdout)["items"][-5:])
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            details.append(f"Diagnostic read failed: {error}")
        return "\n".join(details)

    failure: BaseException | None = None
    try:
        for role in readers:
            create(role)
        while True:
            pods = observe()
            if len(pods) == len(readers) and all(pod["spec"].get("nodeName") for pod in pods):
                break
            time.sleep(1)
        create("writer")
        while True:
            pods = observe()
            if len(pods) == len(roles) and all(pod.get("status", {}).get("phase") == "Succeeded" for pod in pods):
                break
            time.sleep(1)
        receipts = []
        for role in roles:
            pod = next(pod for pod in pods if pod["metadata"]["name"] == names[role])
            logs = checked("logs", names[role], "--tail=5", "--limit-bytes=8000")
            records = [json.loads(line.removeprefix(_RESULT)) for line in logs.splitlines() if line.startswith(_RESULT)]
            if len(records) != 1:
                raise RuntimeError(f"storage preflight Pod {names[role]} omitted its result")
            receipt = records[0]
            expected_image = writer_image if role == "writer" else images[role]
            if pod["spec"]["containers"][0]["image"] != expected_image:
                raise RuntimeError(f"storage preflight Pod {names[role]} used a different image")
            if (receipt != {"role": role, "token": token, "uid": 10001, "gid": 10001,
                           "pod": names[role], "node": pod["spec"]["nodeName"], "output_roots": output_roots,
                           "checkpoint_cycle": "passed", "cleanup": "passed"}):
                raise RuntimeError(f"storage preflight result mismatch: {receipt}")
            image_id = next((status.get("imageID", "") for status in pod["status"].get("containerStatuses", [])
                             if status["name"] == "storage"), "")
            receipts.append({**receipt, "pod_uid": pod["metadata"]["uid"], "image": expected_image, "image_id": image_id})
        writer_node = next(item["node"] for item in receipts if item["role"] == "writer")
        if nodes_per_student > 1 and any(item["node"] == writer_node for item in receipts if item["role"] != "writer"):
            raise RuntimeError("storage preflight did not exercise different nodes")
        print(f"Storage preflight OK: {len(output_roots)} output roots; role UID/GID 10001; "
              f"{'cross-node' if nodes_per_student > 1 else 'shared-PVC'} checkpoint cycle verified.")
        return receipts
    except BaseException as error:
        failure = error
        error.add_note(failure_context())
        raise
    finally:
        cleanup_errors = []
        cleanup_deadline = time.monotonic() + 60
        for name in names.values():
            try:
                remaining = cleanup_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("storage preflight cleanup exceeded 60s")
                current = command("get", "pod", name, "--ignore-not-found", "-o", "json", timeout=min(10, remaining))
                if current.returncode:
                    raise RuntimeError(current.stderr.strip())
                if not current.stdout.strip():
                    continue
                metadata = json.loads(current.stdout)["metadata"]
                if metadata.get("labels", {}).get(_LABEL) != token or (name in created and metadata["uid"] != created[name]):
                    raise RuntimeError(f"refusing to delete changed/unowned Pod {name}")
                options = {"apiVersion": "v1", "kind": "DeleteOptions", "gracePeriodSeconds": 5,
                           "preconditions": {"uid": metadata["uid"]}}
                remaining = cleanup_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("storage preflight cleanup exceeded 60s")
                deleted = command("delete", "--raw", f"/api/v1/namespaces/{namespace}/pods/{name}",
                                  "-f", "-", input=json.dumps(options), timeout=min(10, remaining))
                if deleted.returncode:
                    raise RuntimeError(deleted.stderr.strip())
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                cleanup_errors.append(f"{name}: {error}")
        if not cleanup_errors:
            try:
                while True:
                    remaining = cleanup_deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("owned storage probe Pods remain after 60s cleanup")
                    listed = command("get", "pods", "-l", f"{_LABEL}={token}", "-o", "json", timeout=min(10, remaining))
                    if listed.returncode:
                        raise RuntimeError(listed.stderr.strip())
                    if not json.loads(listed.stdout)["items"]:
                        break
                    time.sleep(1)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                cleanup_errors.append(str(error))
        if cleanup_errors:
            detail = "Storage preflight cleanup incomplete: " + "; ".join(cleanup_errors)
            if failure is not None:
                failure.add_note(detail)
            else:
                raise RuntimeError(detail)
