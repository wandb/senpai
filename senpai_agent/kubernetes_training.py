"""Supervise target-owned Kubernetes training workloads by durable UID."""

from __future__ import annotations

import json
import os
import re
import socket
import ssl
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psutil

from senpai_agent.processes import terminate_process_group
from senpai_agent.training import (
    KubernetesResourceRef,
    KubernetesTrainingSpec,
    TrainingResult,
    TrainingSpec,
    TrainingState,
    training_result_paths,
)

_ERROR_TAIL_BYTES = 8192
_POLL_SECONDS = 2.0
_JOIN_SECONDS = 120.0
EXECUTOR_SOCKET_ENV = "SENPAI_KUBERNETES_EXECUTOR_SOCKET"


class TrainingClusterClient(Protocol):
    def reserve(
        self,
        training_id: str,
        spec: KubernetesTrainingSpec,
        deadline_at: float,
        source_snapshot: str,
        source_commit: str,
    ) -> None: ...

    def adopt(
        self,
        training_id: str,
        spec: KubernetesTrainingSpec,
        resource: KubernetesResourceRef,
        deadline_at: float,
    ) -> None: ...

    def resource(
        self,
        spec: KubernetesTrainingSpec,
        *,
        nodes: int,
        gpus_per_node: int,
    ) -> KubernetesResourceRef | None: ...

    def resource_identity(
        self,
        spec: KubernetesTrainingSpec,
    ) -> KubernetesResourceRef | None: ...

    def state(self, resource: KubernetesResourceRef) -> tuple[TrainingState, str] | None: ...

    def delete(self, resource: KubernetesResourceRef, timeout_seconds: int = 60) -> None: ...

    def logs(self, resource: KubernetesResourceRef) -> str: ...

    def release(self, training_id: str) -> None: ...


class KubernetesExecutorClient:
    """Typed Unix-socket client; the controller never receives Kubernetes credentials."""

    def __init__(self, socket_path: str | Path):
        self.socket_path = str(socket_path)

    def reserve(
        self,
        training_id: str,
        spec: KubernetesTrainingSpec,
        deadline_at: float,
        source_snapshot: str,
        source_commit: str,
    ) -> None:
        self._request(
            "reserve",
            training_id=training_id,
            spec=spec.model_dump(mode="json"),
            deadline_at=deadline_at,
            source_snapshot=source_snapshot,
            source_commit=source_commit,
        )

    def adopt(
        self,
        training_id: str,
        spec: KubernetesTrainingSpec,
        resource: KubernetesResourceRef,
        deadline_at: float,
    ) -> None:
        self._request(
            "adopt",
            training_id=training_id,
            spec=spec.model_dump(mode="json"),
            resource=resource.model_dump(mode="json"),
            deadline_at=deadline_at,
        )

    def resource(
        self,
        spec: KubernetesTrainingSpec,
        *,
        nodes: int,
        gpus_per_node: int,
    ) -> KubernetesResourceRef | None:
        value = self._request(
            "resource",
            spec=spec.model_dump(mode="json"),
            nodes=nodes,
            gpus_per_node=gpus_per_node,
        )
        return KubernetesResourceRef.model_validate(value) if value else None

    def resource_identity(
        self,
        spec: KubernetesTrainingSpec,
    ) -> KubernetesResourceRef | None:
        value = self._request("resource_identity", spec=spec.model_dump(mode="json"))
        return KubernetesResourceRef.model_validate(value) if value else None

    def state(self, resource: KubernetesResourceRef) -> tuple[TrainingState, str] | None:
        value = self._request("state", resource=resource.model_dump(mode="json"))
        return (TrainingState(value[0]), value[1]) if value else None

    def delete(self, resource: KubernetesResourceRef, timeout_seconds: int = 60) -> None:
        self._request(
            "delete",
            resource=resource.model_dump(mode="json"),
            timeout_seconds=timeout_seconds,
        )

    def logs(self, resource: KubernetesResourceRef) -> str:
        return str(self._request("logs", resource=resource.model_dump(mode="json")))

    def release(self, training_id: str) -> None:
        self._request("release", training_id=training_id)

    def apply(self, manifest: str) -> str:
        return str(self._request("apply", manifest=manifest))

    def _request(self, operation: str, **values: object) -> object:
        request = json.dumps({"operation": operation, **values}, separators=(",", ":"))
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(90)
            connection.connect(self.socket_path)
            connection.sendall(request.encode() + b"\n")
            response = json.loads(connection.makefile("rb").readline())
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response.get("result")


class KubernetesApiClient:
    """Bounded in-cluster API client for the executor sidecar."""

    def __init__(
        self,
        api_server: str | None = None,
        token_path: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token"),
        ca_path: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"),
    ):
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        self.api_server = api_server or f"https://{host}:{port}"
        self.token_path = token_path
        self.ssl_context = ssl.create_default_context(cafile=str(ca_path))

    def create(self, manifest: dict, namespace: str) -> dict:
        path = self._collection_path(manifest["kind"], namespace)
        created = self._request_json("POST", path, manifest)
        if created is None:
            raise RuntimeError("Kubernetes API returned an empty create response")
        return created

    def activate(
        self,
        resource: KubernetesResourceRef,
        timeout_seconds: float = 30,
    ) -> None:
        path = (
            f"{self._collection_path(resource.kind, resource.namespace)}/"
            f"{urllib.parse.quote(resource.name, safe='')}"
        )
        suspend_path = (
            "/spec/suspend"
            if resource.kind == "Job"
            else "/spec/runPolicy/suspend"
        )
        activated = self._request_json(
            "PATCH",
            path,
            [
                {"op": "test", "path": "/metadata/uid", "value": resource.uid},
                {"op": "replace", "path": suspend_path, "value": False},
            ],
            content_type="application/json-patch+json",
            timeout_seconds=timeout_seconds,
        )
        spec = activated.get("spec", {}) if activated is not None else {}
        suspended = (
            spec.get("suspend")
            if resource.kind == "Job"
            else spec.get("runPolicy", {}).get("suspend")
        )
        if (
            activated is None
            or activated.get("metadata", {}).get("uid") != resource.uid
            or suspended is not False
        ):
            raise RuntimeError("Kubernetes API returned an invalid activation response")

    def document(self, spec: KubernetesTrainingSpec) -> dict | None:
        return self._get(spec.kind, spec.name, spec.namespace)

    def state(self, resource: KubernetesResourceRef) -> tuple[TrainingState, str] | None:
        document = self._get(resource.kind, resource.name, resource.namespace)
        if document is None:
            return None
        if document["metadata"]["uid"] != resource.uid:
            raise RuntimeError(
                f"{resource.kind} {resource.namespace}/{resource.name} was replaced"
            )
        for condition in reversed(document.get("status", {}).get("conditions", [])):
            if str(condition.get("status")).lower() != "true":
                continue
            condition_type = condition.get("type")
            reason = condition.get("reason", "")
            detail = condition.get("message") or reason or str(condition_type)
            if condition_type in {"Complete", "Succeeded"}:
                return TrainingState.FINISHED, detail
            if condition_type == "Failed":
                state = (
                    TrainingState.TIMED_OUT
                    if reason == "DeadlineExceeded"
                    else TrainingState.FAILED
                )
                return state, detail
        return TrainingState.RUNNING, "Kubernetes workload is active"

    def delete(self, resource: KubernetesResourceRef, timeout_seconds: int = 60) -> None:
        document = self._get(resource.kind, resource.name, resource.namespace)
        if document is None:
            return
        if document["metadata"]["uid"] != resource.uid:
            raise RuntimeError(
                f"refusing to delete replaced {resource.kind} "
                f"{resource.namespace}/{resource.name}"
            )
        group_path = (
            "apis/batch/v1"
            if resource.kind == "Job"
            else "apis/kubeflow.org/v2beta1"
        )
        plural = "jobs" if resource.kind == "Job" else "mpijobs"
        url = "/".join(
            (
                self.api_server.rstrip("/"),
                group_path,
                "namespaces",
                urllib.parse.quote(resource.namespace, safe=""),
                plural,
                urllib.parse.quote(resource.name, safe=""),
            )
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "propagationPolicy": "Foreground",
                    "preconditions": {"uid": resource.uid},
                }
            ).encode(),
            method="DELETE",
            headers={
                "Authorization": f"Bearer {self.token_path.read_text().strip()}",
                "Content-Type": "application/json",
            },
        )
        try:
            urllib.request.urlopen(
                request,
                context=self.ssl_context,
                timeout=min(timeout_seconds, 30),
            ).read()
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise RuntimeError(
                    f"Kubernetes UID-preconditioned delete failed: HTTP {error.code}"
                ) from error
        deadline = time.monotonic() + timeout_seconds
        while self._get(resource.kind, resource.name, resource.namespace) is not None:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out deleting {resource.kind} "
                    f"{resource.namespace}/{resource.name}"
                )
            time.sleep(0.5)

    def logs(self, resource: KubernetesResourceRef) -> str:
        deadline = time.monotonic() + 30
        label = (
            f"job-name={resource.name}"
            if resource.kind == "Job"
            else f"training.kubeflow.org/job-name={resource.name}"
        )
        query = urllib.parse.urlencode({"labelSelector": label})
        pods = self._request_json(
            "GET",
            f"/api/v1/namespaces/{urllib.parse.quote(resource.namespace, safe='')}/pods?{query}",
            timeout_seconds=5,
        )
        output = []
        for pod in pods.get("items", [])[: resource.nodes + 2]:
            if not any(
                owner.get("uid") == resource.uid
                for owner in pod["metadata"].get("ownerReferences", [])
            ):
                continue
            pod_name = pod["metadata"]["name"]
            for container in pod["spec"].get("containers", [])[:8]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "\n".join(output)
                params = urllib.parse.urlencode(
                    {"container": container["name"], "tailLines": 200}
                )
                path = (
                    f"/api/v1/namespaces/{urllib.parse.quote(resource.namespace, safe='')}"
                    f"/pods/{urllib.parse.quote(pod_name, safe='')}/log?{params}"
                )
                text = self._request_text(
                    "GET",
                    path,
                    allow_not_found=True,
                    timeout_seconds=max(1, min(5, remaining)),
                )
                prefix = f"[pod/{pod_name}/{container['name']}] "
                output.extend(prefix + line for line in text.splitlines())
        return "\n".join(output)

    def _get(self, kind: str, name: str, namespace: str) -> dict | None:
        path = (
            f"{self._collection_path(kind, namespace)}/"
            f"{urllib.parse.quote(name, safe='')}"
        )
        return self._request_json("GET", path, allow_not_found=True)

    @staticmethod
    def _collection_path(kind: str, namespace: str) -> str:
        namespace = urllib.parse.quote(namespace, safe="")
        if kind == "Job":
            return f"/apis/batch/v1/namespaces/{namespace}/jobs"
        if kind == "MPIJob":
            return f"/apis/kubeflow.org/v2beta1/namespaces/{namespace}/mpijobs"
        raise ValueError(f"unsupported Kubernetes training kind {kind!r}")

    def _request_json(
        self,
        method: str,
        path: str,
        body: object | None = None,
        *,
        allow_not_found: bool = False,
        content_type: str = "application/json",
        timeout_seconds: float = 30,
    ) -> dict | None:
        text = self._request_text(
            method,
            path,
            json.dumps(body).encode() if body is not None else None,
            allow_not_found=allow_not_found,
            content_type=content_type,
            timeout_seconds=timeout_seconds,
        )
        return json.loads(text) if text else None

    def _request_text(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        *,
        allow_not_found: bool = False,
        content_type: str = "application/json",
        timeout_seconds: float = 30,
    ) -> str:
        request = urllib.request.Request(
            f"{self.api_server.rstrip('/')}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token_path.read_text().strip()}",
                "Content-Type": content_type,
            },
        )
        try:
            return urllib.request.urlopen(
                request,
                context=self.ssl_context,
                timeout=timeout_seconds,
            ).read().decode()
        except urllib.error.HTTPError as error:
            if allow_not_found and error.code == 404:
                return ""
            raise RuntimeError(
                f"Kubernetes API {method} {path.split('?')[0]} failed: HTTP {error.code}"
            ) from error


def _workload_shape(document: dict) -> tuple[int, int]:
    kind = document.get("kind")
    if kind == "MPIJob":
        worker = document["spec"]["mpiReplicaSpecs"]["Worker"]
        return int(worker["replicas"]), _pod_gpus(worker["template"]["spec"])
    if kind == "Job":
        spec = document["spec"]
        nodes = int(spec.get("parallelism", spec.get("completions", 1)))
        return nodes, _pod_gpus(spec["template"]["spec"])
    raise RuntimeError(f"unsupported Kubernetes training kind {kind!r}")


def _pod_gpus(pod_spec: dict) -> int:
    return sum(
        int(container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
        for container in pod_spec["containers"]
    )


@dataclass
class _ActiveRemoteTraining:
    spec: KubernetesTrainingSpec
    started_at: float
    deadline_at: float
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    process_group_id: int | None = None
    resource: KubernetesResourceRef | None = None
    cancelled: bool = False
    thread: threading.Thread | None = None


class KubernetesTrainingSupervisor:
    """Submit once, then supervise one remote Job or MPIJob to terminal state."""

    def __init__(
        self,
        *,
        workspace: Path,
        state_dir: Path,
        nodes: int,
        gpus_per_node: int,
        max_timeout_seconds: int | None = None,
        terminate_grace_seconds: float = 10,
        poll_seconds: float = _POLL_SECONDS,
        client: TrainingClusterClient | None = None,
    ):
        if min(nodes, gpus_per_node) < 1:
            raise ValueError("Kubernetes training resources must be positive")
        if max_timeout_seconds is not None and max_timeout_seconds <= 0:
            raise ValueError("max_timeout_seconds must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.workspace = workspace.resolve()
        self.state_dir = state_dir.resolve()
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node
        self.max_timeout_seconds = max_timeout_seconds
        self.terminate_grace_seconds = terminate_grace_seconds
        self.poll_seconds = poll_seconds
        socket_path = os.environ.get(
            EXECUTOR_SOCKET_ENV,
            "/var/run/senpai-kubernetes/executor.sock",
        )
        self.client = client or KubernetesExecutorClient(socket_path)
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveRemoteTraining] = {}
        self._launching = False
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._recover()

    def run_training(self, spec: TrainingSpec) -> TrainingResult:
        if (
            self.max_timeout_seconds is not None
            and spec.timeout_seconds > self.max_timeout_seconds
        ):
            raise ValueError(
                "training timeout exceeds the configured maximum of "
                f"{self.max_timeout_seconds} seconds"
            )
        cwd = spec.cwd.resolve()
        if cwd != self.workspace and not cwd.is_relative_to(self.workspace):
            raise ValueError("training cwd must be inside the assignment workspace")
        with self._lock:
            if self._active or self._launching:
                raise RuntimeError("this student already has an active Kubernetes training run")
            self._launching = True

        training_id: str | None = None
        process: subprocess.Popen[bytes] | None = None
        reserved = False
        try:
            training_id = str(uuid.uuid4())
            kubernetes_spec = _training_spec(training_id)
            log_path = self.state_dir / f"{training_id}.log"
            started_at = time.time()
            deadline_at = started_at + spec.timeout_seconds
            source_snapshot, source_commit = _materialize_source_snapshot(cwd)
            self.client.reserve(
                training_id,
                kubernetes_spec,
                deadline_at,
                str(source_snapshot),
                source_commit,
            )
            reserved = True
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    list(spec.argv),
                    cwd=cwd,
                    env={
                        **os.environ,
                        "SENPAI_TRAINING_SOURCE_SNAPSHOT": str(source_snapshot),
                        "SENPAI_KUBERNETES_WORKLOAD_NAME": kubernetes_spec.name,
                        "SENPAI_KUBERNETES_NAMESPACE": kubernetes_spec.namespace,
                        "SENPAI_WANDB_RUN_ID": kubernetes_spec.wandb_run_id,
                        "SENPAI_LAUNCH_SECRET_NAME": os.environ[
                            "SENPAI_LAUNCH_SECRET_NAME"
                        ],
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    start_new_session=True,
                )
            process_group_id = process.pid
            result = TrainingResult(
                training_id=training_id,
                state=TrainingState.RUNNING,
                pid=process.pid,
                process_group_id=process_group_id,
                process_start_time=psutil.Process(process.pid).create_time(),
                exit_code=None,
                elapsed_seconds=0,
                log_path=str(log_path),
                wandb_run_ids=(kubernetes_spec.wandb_run_id,),
                started_at=started_at,
                deadline_at=deadline_at,
                kubernetes_spec=kubernetes_spec,
                kubernetes_released=False,
                source_snapshot=str(source_snapshot),
                source_commit=source_commit,
            )
            active = _ActiveRemoteTraining(
                spec=kubernetes_spec,
                started_at=started_at,
                deadline_at=deadline_at,
                log_path=log_path,
                process=process,
                process_group_id=process_group_id,
            )
            thread = threading.Thread(
                target=self._monitor,
                args=(training_id,),
                name=f"senpai-kubernetes-training-{training_id}",
            )
            active.thread = thread
            self._write_result(result)
            with self._lock:
                self._active[training_id] = active
                self._launching = False
                thread.start()
        except BaseException:
            if process is not None and process.poll() is None:
                terminate_process_group(
                    process,
                    process_group_id=process.pid,
                    grace_seconds=self.terminate_grace_seconds,
                    wait_full_grace=True,
                )
            if reserved and training_id is not None:
                self._cleanup_failed_launch(training_id, kubernetes_spec)
            with self._lock:
                if training_id is not None:
                    self._active.pop(training_id, None)
                self._launching = False
            raise
        return result

    def get_training_status(self, training_id: str) -> TrainingResult:
        result = TrainingResult.model_validate_json(
            (self.state_dir / f"{uuid.UUID(training_id)}.json").read_text()
        )
        with self._lock:
            active = self._active.get(training_id)
        if active is not None and result.state is TrainingState.RUNNING:
            return result.model_copy(
                update={"elapsed_seconds": time.time() - active.started_at}
            )
        return result

    def cancel_training(self, training_id: str) -> TrainingResult:
        result = self.get_training_status(training_id)
        if result.state is not TrainingState.RUNNING:
            return result
        with self._lock:
            active = self._active.get(training_id)
            if active is None:
                return self.get_training_status(training_id)
            active.cancelled = True
            thread = active.thread
        if thread is not None:
            thread.join(_JOIN_SECONDS)
            if thread.is_alive():
                raise TimeoutError("timed out waiting for Kubernetes training cancellation")
        return self.get_training_status(training_id)

    def close(self) -> None:
        with self._lock:
            active = tuple(self._active.values())
            for training in active:
                training.cancelled = True
        for training in active:
            if training.thread is not None:
                training.thread.join(_JOIN_SECONDS)

    def drain(self) -> None:
        with self._lock:
            threads = tuple(
                training.thread
                for training in self._active.values()
                if training.thread is not None
            )
        for thread in threads:
            thread.join(_JOIN_SECONDS)

    def _cleanup_failed_launch(
        self,
        training_id: str,
        spec: KubernetesTrainingSpec,
    ) -> None:
        """Best-effort cleanup without masking the launch error or unsafe release."""

        try:
            resource = self.client.resource_identity(spec)
            if resource is not None:
                self.client.delete(resource)
            self.client.release(training_id)
        except Exception:
            # The broker keeps the reservation and reaps it at its deadline.
            return

    def _recover(self) -> None:
        for path in training_result_paths(self.state_dir):
            result = TrainingResult.model_validate_json(path.read_text())
            if result.state is not TrainingState.RUNNING:
                if (
                    result.kubernetes_spec is not None
                    and result.kubernetes_released is not True
                ):
                    self._resume_terminal_release(result)
                continue
            resource = result.kubernetes_resource
            spec = result.kubernetes_spec
            if spec is None or result.started_at is None or result.deadline_at is None:
                self._write_result(
                    result.model_copy(
                        update={
                            "state": TrainingState.CANCELLED,
                            "error_tail": "Incomplete remote identity found after restart.",
                        }
                    )
                )
                continue
            if result.source_snapshot is None or result.source_commit is None:
                raise RuntimeError("remote training record has no source snapshot identity")
            if result.deadline_at <= time.time():
                terminal = result.model_copy(
                    update={
                        "state": TrainingState.TIMED_OUT,
                        "elapsed_seconds": time.time() - result.started_at,
                        "error_tail": (
                            "Training deadline elapsed while the controller was restarting."
                        ),
                        "kubernetes_released": False,
                    }
                )
                self._write_result(terminal)
                self._resume_terminal_release(terminal)
                continue
            self.client.reserve(
                result.training_id,
                spec,
                result.deadline_at,
                result.source_snapshot,
                result.source_commit,
            )
            current = self.client.resource(
                spec,
                nodes=self.nodes,
                gpus_per_node=self.gpus_per_node,
            )
            if current is None:
                terminal = result.model_copy(
                    update={
                        "state": TrainingState.CANCELLED,
                        "error_tail": "No remote training resource existed after restart.",
                        "kubernetes_released": False,
                    }
                )
                self._write_result(terminal)
                self._resume_terminal_release(terminal)
                continue
            if resource is not None and current.uid != resource.uid:
                terminal = result.model_copy(
                    update={
                        "state": TrainingState.FAILED,
                        "error_tail": "Remote training resource was replaced after restart.",
                        "kubernetes_released": False,
                    }
                )
                self._write_result(terminal)
                self._resume_terminal_release(terminal)
                continue
            if resource is None:
                resource = current
                self._write_result(
                    result.model_copy(update={"kubernetes_resource": current})
                )
            active = _ActiveRemoteTraining(
                spec=spec,
                started_at=result.started_at,
                deadline_at=result.deadline_at,
                log_path=Path(result.log_path),
                resource=current,
            )
            self.client.adopt(
                result.training_id,
                active.spec,
                current,
                active.deadline_at,
            )
            thread = threading.Thread(
                target=self._monitor,
                args=(result.training_id,),
                name=f"senpai-kubernetes-training-{result.training_id}",
            )
            active.thread = thread
            self._active[result.training_id] = active
            thread.start()

    def _resume_terminal_release(self, result: TrainingResult) -> None:
        if result.kubernetes_spec is None:
            raise RuntimeError("terminal Kubernetes training has no workload identity")
        active = _ActiveRemoteTraining(
            spec=result.kubernetes_spec,
            started_at=result.started_at or time.time(),
            deadline_at=result.deadline_at or time.time(),
            log_path=Path(result.log_path),
            resource=result.kubernetes_resource,
        )
        thread = threading.Thread(
            target=self._release_terminal,
            args=(
                result.training_id,
                active,
                result.state is not TrainingState.FINISHED,
            ),
            name=f"senpai-kubernetes-release-{result.training_id}",
        )
        active.thread = thread
        self._active[result.training_id] = active
        thread.start()

    def _monitor(self, training_id: str) -> None:
        with self._lock:
            active = self._active[training_id]
        state = TrainingState.RUNNING
        exit_code = None
        detail = ""
        delete_required = False
        try:
            if active.process is not None:
                while active.process.poll() is None:
                    if active.cancelled or time.time() >= active.deadline_at:
                        terminate_process_group(
                            active.process,
                            process_group_id=active.process_group_id,
                            grace_seconds=self.terminate_grace_seconds,
                            wait_full_grace=True,
                        )
                        break
                    time.sleep(0.1)
                exit_code = active.process.returncode
                if not active.cancelled and time.time() < active.deadline_at and exit_code == 0:
                    while active.resource is None:
                        try:
                            active.resource = self.client.resource(
                                active.spec,
                                nodes=self.nodes,
                                gpus_per_node=self.gpus_per_node,
                            )
                        except Exception as error:  # retry transient broker/API outages
                            detail = f"Waiting for Kubernetes ownership: {error}"
                        if active.resource is not None:
                            break
                        if active.cancelled or time.time() >= active.deadline_at:
                            break
                        time.sleep(self.poll_seconds)
                    if (
                        active.resource is None
                        and not active.cancelled
                        and time.time() < active.deadline_at
                    ):
                        raise RuntimeError(
                            f"submission finished without creating {active.spec.kind} "
                            f"{active.spec.namespace}/{active.spec.name}"
                        )
                    self._publish_resource(training_id, active)
                elif exit_code not in {None, 0}:
                    delete_required = True

            if active.cancelled:
                state = TrainingState.CANCELLED
                delete_required = True
            elif time.time() >= active.deadline_at:
                state = TrainingState.TIMED_OUT
                delete_required = True
            elif exit_code not in {None, 0}:
                state = TrainingState.FAILED
                detail = f"Kubernetes submission exited with code {exit_code}."
            elif active.resource is not None:
                while True:
                    if active.cancelled:
                        state = TrainingState.CANCELLED
                        delete_required = True
                        break
                    if time.time() >= active.deadline_at:
                        state = TrainingState.TIMED_OUT
                        delete_required = True
                        break
                    try:
                        snapshot = self.client.state(active.resource)
                    except Exception as error:  # retry transient broker/API outages
                        detail = f"Waiting for Kubernetes status: {error}"
                        time.sleep(self.poll_seconds)
                        continue
                    if snapshot is None:
                        state = TrainingState.FAILED
                        detail = "Remote training resource disappeared before completion."
                        break
                    state, detail = snapshot
                    if state is not TrainingState.RUNNING:
                        break
                    time.sleep(self.poll_seconds)
        except Exception as error:  # noqa: BLE001
            state = TrainingState.FAILED
            detail = f"{type(error).__name__}: {error}"
            delete_required = True

        if active.resource is None:
            try:
                active.resource = self.client.resource_identity(active.spec)
            except Exception as error:
                detail = "\n".join(
                    part
                    for part in (
                        detail,
                        f"Kubernetes ownership lookup failed: {error}",
                    )
                    if part
                )

        try:
            remote_logs = (
                self.client.logs(active.resource)
                if active.resource is not None
                else ""
            )
            if remote_logs:
                with active.log_path.open("a") as log:
                    log.write("\n=== Kubernetes logs (last 200 lines per pod) ===\n")
                    log.write(remote_logs)
        except Exception:
            pass
        try:
            local_tail = active.log_path.read_bytes()[-_ERROR_TAIL_BYTES:].decode(
                errors="ignore"
            )
        except OSError:
            local_tail = ""
        error_tail = "" if state is TrainingState.FINISHED else "\n".join(
            part for part in (detail, local_tail) if part
        )[-_ERROR_TAIL_BYTES:]
        terminal = self.get_training_status(training_id).model_copy(
            update={
                "state": state,
                "exit_code": exit_code,
                "elapsed_seconds": time.time() - active.started_at,
                "error_tail": error_tail,
                "kubernetes_resource": active.resource,
                "kubernetes_released": False,
            }
        )
        self._write_result(terminal)
        self._release_terminal(training_id, active, delete_required)

    def _release_terminal(
        self,
        training_id: str,
        active: _ActiveRemoteTraining,
        delete_required: bool,
    ) -> None:
        while True:
            try:
                if delete_required:
                    if active.resource is None:
                        active.resource = self.client.resource_identity(active.spec)
                    if active.resource is not None:
                        self.client.delete(active.resource)
                self.client.release(training_id)
                break
            except Exception:
                time.sleep(self.poll_seconds)
        result = self.get_training_status(training_id)
        self._write_result(result.model_copy(update={"kubernetes_released": True}))
        with self._lock:
            self._active.pop(training_id, None)

    def _publish_resource(
        self,
        training_id: str,
        active: _ActiveRemoteTraining,
    ) -> None:
        result = self.get_training_status(training_id)
        self._write_result(
            result.model_copy(update={"kubernetes_resource": active.resource})
        )

    def _write_result(self, result: TrainingResult) -> None:
        path = self.state_dir / f"{result.training_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2))
        temporary.replace(path)


def _materialize_source_snapshot(workspace: Path) -> tuple[Path, str]:
    git_environment = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    head = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=workspace,
        env=git_environment,
        text=True,
    ).strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError(f"git returned invalid HEAD {head!r}")
    root = Path(os.environ["SENPAI_TRAINING_SNAPSHOT_ROOT"])
    snapshot = root / f"{head}.bundle"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{head}.{uuid.uuid4().hex}.bundle"
    try:
        subprocess.run(
            ["git", "bundle", "create", str(temporary), "HEAD"],
            cwd=workspace,
            env=git_environment,
            check=True,
        )
        temporary.chmod(0o444)
        temporary.replace(snapshot)
    finally:
        temporary.unlink(missing_ok=True)
    return snapshot, head


def _training_spec(training_id: str) -> KubernetesTrainingSpec:
    research = _dns_label(os.environ["RESEARCH_TAG"])
    student = _dns_label(os.environ["STUDENT_NAME"])
    suffix = uuid.UUID(training_id).hex[:12]
    prefix = f"senpai-{research}-{student}"[: 62 - len(suffix)].rstrip("-.")
    return KubernetesTrainingSpec(
        kind="MPIJob",
        name=f"{prefix}-{suffix}",
        namespace=os.environ["SENPAI_KUBERNETES_NAMESPACE"],
        wandb_run_id=uuid.UUID(training_id).hex,
    )


def _dns_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "student"
