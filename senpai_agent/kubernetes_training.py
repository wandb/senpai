"""Run training commands in Kubernetes workloads supervised by durable UID."""

from __future__ import annotations

import base64
import errno
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

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
_DIAGNOSTICS_SECONDS = 30.0
_JOIN_SECONDS = 120.0
EXECUTOR_SOCKET_ENV = "SENPAI_KUBERNETES_EXECUTOR_SOCKET"


def _mask_output_chunk(data: bytes, secret: bytes) -> tuple[bytes, bytes]:
    """Emit complete bytes while retaining a possible split secret prefix."""
    if not secret:
        return data, b""
    pieces = []
    start = 0
    while (match := data.find(secret, start)) >= 0:
        pieces.extend((data[start:match], b"<secret-hidden>"))
        start = match + len(secret)
    tail = data[start:]
    overlap = next(
        (
            size
            for size in range(min(len(tail), len(secret) - 1), 0, -1)
            if tail.endswith(secret[:size])
        ),
        0,
    )
    pieces.append(tail[:-overlap] if overlap else tail)
    return b"".join(pieces), tail[-overlap:] if overlap else b""


class KubernetesDiagnostics(TypedDict):
    statuses: list[str]
    problems: list[str]
    events: list[str]
    logs: list[tuple[str, str]]


def _mask_wandb_output(text: str) -> str:
    key = os.environ.get("WANDB_API_KEY", "").encode()
    if not key:
        return text
    masked, pending = _mask_output_chunk(text.encode(), key)
    return (masked + (b"<secret-hidden>" if pending else b"")).decode()


def _format_diagnostics(diagnostics: KubernetesDiagnostics) -> str:
    """Redact complete components before any presentation limits split secrets."""
    statuses = [_mask_wandb_output(text) for text in diagnostics["statuses"]]
    problems = [_mask_wandb_output(text) for text in diagnostics["problems"]]
    problems.extend(_mask_wandb_output(text)[:1024] for text in diagnostics["events"])
    logs = [
        (_mask_wandb_output(prefix), _mask_wandb_output(text))
        for prefix, text in diagnostics["logs"]
    ]
    # Keep every status identity; one long error must not hide the other pods.
    summary_budget = _ERROR_TAIL_BYTES // 2 if logs else _ERROR_TAIL_BYTES
    status_budget = summary_budget * 3 // 4 if problems else summary_budget
    per_status = max(0, status_budget // len(statuses) - 1)
    summary = "\n".join(
        status.encode()[:per_status].decode(errors="ignore") for status in statuses
    )
    if problems:
        problem_budget = summary_budget - len(summary.encode()) - 1
        summary += "\n" + "\n".join(problems).encode()[:problem_budget].decode(errors="ignore")
    if not logs:
        return summary
    log_budget = (_ERROR_TAIL_BYTES - len(summary.encode()) - len(logs)) // len(logs)
    output = [summary] if summary else []
    for prefix, text in logs:
        text_budget = log_budget - len(prefix.encode())
        if text_budget <= 0:
            continue
        encoded = text.encode()
        marker = b"\n... truncated ...\n"
        if len(encoded) > text_budget > len(marker) + 1:
            # Preserve the first error and final context within the fetched tail.
            end_budget = (text_budget - len(marker)) // 2
            excerpt = (
                encoded[:text_budget - len(marker) - end_budget].decode(errors="ignore")
                + marker.decode()
                + encoded[-end_budget:].decode(errors="ignore")
            )
        else:
            excerpt = encoded[:text_budget].decode(errors="ignore")
        output.append(prefix + excerpt)
    return "\n".join(output)


class KubernetesApiError(RuntimeError):
    """Kubernetes API response failure with its HTTP status preserved."""

    def __init__(self, method: str, path: str, status_code: int):
        super().__init__(
            f"Kubernetes API {method} {path.split('?')[0]} failed: HTTP {status_code}"
        )
        self.status_code = status_code


class TrainingClusterClient(Protocol):
    def apply(self, manifest: str) -> str: ...

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

    def release(self, training_id: str) -> dict: ...


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
        diagnostics = self._request("logs", resource=resource.model_dump(mode="json"))
        return _format_diagnostics(diagnostics) if diagnostics is not None else ""

    def release(self, training_id: str) -> dict:
        return self._request("release", training_id=training_id)

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

    def logs(self, resource: KubernetesResourceRef) -> KubernetesDiagnostics:
        """Collect diagnostics for controller-side redaction before formatting."""

        deadline = time.monotonic() + 30
        pods = self._owned_pods(resource)[: resource.nodes + 2]
        events = self._events(
            resource.namespace, resource.uid, f"{resource.kind}/{resource.name}", deadline
        )
        problems = []
        active_pods = [
            pod for pod in pods
            if pod.get("status", {}).get("phase") not in {"Succeeded", "Failed"}
        ]
        statuses = [
            f"[workload] requested_gpus={resource.nodes * resource.gpus_per_node} "
            f"scheduled_gpu_requests={sum(_pod_gpus(pod['spec']) for pod in active_pods if pod['spec'].get('nodeName'))} "
            f"pending_gpu_requests={sum(_pod_gpus(pod['spec']) for pod in active_pods if not pod['spec'].get('nodeName'))}"
        ]
        log_requests = []
        if not pods:
            statuses.append("No owned workload pods exist yet.")
        for pod in sorted(pods, key=lambda pod: pod["metadata"]["name"]):
            pod_name = pod["metadata"]["name"]
            status = pod.get("status", {})
            prefix = f"[pod/{pod_name}] "
            statuses.append(
                prefix + f"phase={status.get('phase', 'Unknown')} "
                f"node={pod['spec'].get('nodeName', 'unassigned')}"
            )
            events.extend(self._events(
                resource.namespace, pod["metadata"]["uid"], f"pod/{pod_name}", deadline
            ))
            for condition in status.get("conditions", []):
                if condition.get("status") == "False" and condition.get("message"):
                    problems.append(prefix + f"{condition['type']}: {condition['message']}")
            container_states = {
                item["name"]: item.get("state", {})
                for item in status.get("initContainerStatuses", [])
                + status.get("containerStatuses", [])
            }
            init_containers = pod["spec"].get("initContainers", [])
            containers = init_containers + pod["spec"]["containers"]
            for container in containers[:8]:
                name = container["name"]
                prefix = f"[pod/{pod_name}/{name}] "
                state = container_states.get(name, {})
                waiting = state.get("waiting")
                terminated = state.get("terminated")
                if waiting is not None:
                    statuses.append(prefix + f"waiting: {waiting.get('reason', '')}")
                    if waiting.get("message"):
                        problems.append(prefix + waiting["message"])
                    continue
                if terminated is not None:
                    statuses.append(
                        prefix + f"terminated: {terminated.get('reason', '')} "
                        f"exit={terminated.get('exitCode')}"
                    )
                    if terminated.get("message"):
                        problems.append(prefix + terminated["message"])
                elif "running" in state:
                    statuses.append(prefix + "running")
                if not state:
                    statuses.append(prefix + "state=Unknown")
                    continue
                priority = (
                    0 if terminated and terminated.get("exitCode") != 0
                    else 1 if "running" in state
                    else 2 if container in init_containers
                    else 3
                )
                log_requests.append((priority, pod_name, name))

        container_logs = []
        for _, pod_name, name in sorted(log_requests):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            prefix = f"[pod/{pod_name}/{name}] "
            params = urllib.parse.urlencode(
                {"container": name, "tailLines": 200, "limitBytes": 8192}
            )
            path = (
                f"/api/v1/namespaces/{urllib.parse.quote(resource.namespace, safe='')}"
                f"/pods/{urllib.parse.quote(pod_name, safe='')}/log?{params}"
            )
            try:
                text = self._request_text(
                    "GET",
                    path,
                    allow_not_found=True,
                    timeout_seconds=min(5, remaining),
                )
            except KubernetesApiError as error:
                problems.append(prefix + f"logs unavailable: HTTP {error.status_code}")
                continue
            if text:
                container_logs.append((prefix, text))
        return {
            "statuses": statuses,
            "problems": problems,
            "events": events,
            "logs": container_logs,
        }

    def _events(self, namespace: str, uid: str, owner: str, deadline: float) -> list[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return []
        query = urllib.parse.urlencode({
            "fieldSelector": f"involvedObject.uid={uid}",
            "limit": 20,
        })
        try:
            response = self._request_json(
                "GET",
                f"/api/v1/namespaces/{urllib.parse.quote(namespace, safe='')}/events?{query}",
                timeout_seconds=min(3, remaining),
                max_response_bytes=4 * 1024 * 1024,
            )
        except KubernetesApiError as error:
            return [f"[{owner}/events] unavailable: HTTP {error.status_code}"]
        events = [
            event for event in response["items"]
            if event.get("involvedObject", {}).get("uid") == uid
        ]
        events.sort(key=lambda event: event.get("lastTimestamp") or event["metadata"]["creationTimestamp"])
        return [
            f"[{owner}/event] {event.get('type', '')} {event.get('reason', '')}: "
            f"{event.get('message', '')}"
            for event in events[-5:]
        ]

    def pod_snapshot(self, resource: KubernetesResourceRef) -> dict:
        """Capture factual Pod status; container restarts are not elastic restarts."""
        pods = self._owned_pods(resource, timeout_seconds=10)
        rows = []
        for pod in pods:
            status = pod.get("status", {})
            containers = []
            for spec_key, status_key in [
                ("initContainers", "initContainerStatuses"),
                ("containers", "containerStatuses"),
            ]:
                statuses = {item["name"]: item for item in status.get(status_key, [])}
                for container in pod["spec"].get(spec_key, []):
                    value = statuses.get(container["name"], {})
                    states = {}
                    for field in ("state", "lastState"):
                        states[field] = {
                            state: {key: item[key] for key in (
                                "reason", "exitCode", "signal", "startedAt", "finishedAt",
                            ) if key in item}
                            for state, item in value.get(field, {}).items()
                        }
                    containers.append({
                        "name": container["name"], "init": spec_key == "initContainers",
                        "restartCount": value.get("restartCount"), **states,
                    })
            rows.append({
                "name": pod["metadata"]["name"], "uid": pod["metadata"]["uid"],
                "owners": [{key: owner.get(key) for key in ("kind", "name", "uid")}
                           for owner in pod["metadata"].get("ownerReferences", [])],
                "node": pod["spec"].get("nodeName"), "phase": status.get("phase"),
                "created_at": pod["metadata"].get("creationTimestamp"),
                "role": "worker" if _pod_gpus(pod["spec"]) else "launcher",
                "containers": containers,
            })
        expected = resource.nodes + (resource.kind == "MPIJob")
        complete = (
            len(rows) == expected
            and sum(row["role"] == "worker" for row in rows) == resource.nodes
            and all(row["phase"] in {"Succeeded", "Failed"} and row["containers"]
                    and all(c["restartCount"] is not None
                            and all(c["state"].get("terminated", {}).get(key) is not None
                                    for key in ("exitCode", "startedAt", "finishedAt"))
                            for c in row["containers"]) for row in rows)
        )
        return {"pods": rows, "expected_pods": expected, "complete": complete,
                "capture_error": None if complete else "Expected terminal Pod/container coverage is incomplete"}

    def _owned_pods(
        self, resource: KubernetesResourceRef, *, timeout_seconds: float = 30,
    ) -> list[dict]:
        deadline = time.monotonic() + timeout_seconds

        def remaining() -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("Kubernetes Pod inventory exceeded its deadline")
            return min(value, 5)

        document = self._get(resource.kind, resource.name, resource.namespace,
                             timeout_seconds=remaining())
        if document is None:
            return []
        if document["metadata"]["uid"] != resource.uid:
            raise RuntimeError("refusing to read pods for a replaced training workload")
        training_id = document["metadata"]["labels"]["senpai-training-id"]
        query = urllib.parse.urlencode({"labelSelector": f"senpai-training-id={training_id}", "limit": 256})
        pods = self._request_json(
            "GET",
            f"/api/v1/namespaces/{urllib.parse.quote(resource.namespace, safe='')}/pods?{query}",
            timeout_seconds=remaining(),
            max_response_bytes=4 * 1024 * 1024,
        )
        if pods.get("metadata", {}).get("continue") or len(pods["items"]) > 256:
            raise RuntimeError("Kubernetes Pod inventory exceeds the bounded capture limit")
        owned = []
        jobs = {}
        for pod in pods["items"]:
            for owner in pod["metadata"].get("ownerReferences", []):
                if owner.get("uid") == resource.uid and owner.get("kind") == resource.kind:
                    owned.append(pod)
                    break
                if resource.kind != "MPIJob" or owner.get("kind") != "Job":
                    continue
                name = owner["name"]
                if name not in jobs:
                    jobs[name] = self._get("Job", name, resource.namespace, timeout_seconds=remaining())
                job = jobs[name]
                if (
                    job is not None
                    and job["metadata"]["uid"] == owner.get("uid")
                    and any(
                        parent.get("kind") == "MPIJob" and parent.get("uid") == resource.uid
                        for parent in job["metadata"].get("ownerReferences", [])
                    )
                ):
                    owned.append(pod)
                    break
        return owned

    def _get(
        self, kind: str, name: str, namespace: str, *, timeout_seconds: float = 30,
    ) -> dict | None:
        path = (
            f"{self._collection_path(kind, namespace)}/"
            f"{urllib.parse.quote(name, safe='')}"
        )
        return self._request_json("GET", path, allow_not_found=True, timeout_seconds=timeout_seconds)

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
        max_response_bytes: int | None = None,
    ) -> dict | None:
        text = self._request_text(
            method,
            path,
            json.dumps(body).encode() if body is not None else None,
            allow_not_found=allow_not_found,
            content_type=content_type,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
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
        max_response_bytes: int | None = None,
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
            with urllib.request.urlopen(
                request,
                context=self.ssl_context,
                timeout=timeout_seconds,
            ) as response:
                payload = response.read(max_response_bytes + 1) if max_response_bytes is not None else response.read()
            if max_response_bytes is not None and len(payload) > max_response_bytes:
                raise ValueError("Kubernetes response exceeds the configured byte limit")
            return payload.decode()
        except urllib.error.HTTPError as error:
            if allow_not_found and error.code == 404:
                return ""
            raise KubernetesApiError(method, path, error.code) from error


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
    manifest: dict | None = None
    resource: KubernetesResourceRef | None = None
    cancelled: bool = False
    thread: threading.Thread | None = None
    next_diagnostics_at: float = 0
    diagnostics_future: Future[str] | None = None


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
        self.poll_seconds = poll_seconds
        self._wandb_api_key = os.environ.get("WANDB_API_KEY", "").encode()
        socket_path = os.environ.get(
            EXECUTOR_SOCKET_ENV,
            "/var/run/senpai-kubernetes/executor.sock",
        )
        self.client = client or KubernetesExecutorClient(socket_path)
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._launch_complete = threading.Event()
        self._launch_complete.set()
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
        training_id = str(uuid.uuid4())
        with self._lock:
            if self._shutdown.is_set():
                raise RuntimeError("Kubernetes training supervisor is closed")
            if self._active or self._launching:
                raise RuntimeError("this student already has an active Kubernetes training run")
            self._launching = True
            self._launch_complete.clear()

        result: TrainingResult | None = None
        reserved = False
        try:
            kubernetes_spec = _training_spec(training_id, nodes=self.nodes)
            started_at = time.time()
            source_snapshot, source_commit = _materialize_source_snapshot(self.workspace)
            output_dir = str(Path(os.environ["SENPAI_TRAINING_OUTPUT_ROOT"]) / training_id)
            manifest = _training_manifest(
                spec, kubernetes_spec, source_commit=source_commit,
                relative_cwd=cwd.relative_to(self.workspace), nodes=self.nodes,
                gpus_per_node=self.gpus_per_node, output_dir=output_dir,
            )
            if self._shutdown.is_set():
                raise RuntimeError("Kubernetes training supervisor is closed")
            deadline_at = started_at + spec.timeout_seconds
            self.client.reserve(
                training_id, kubernetes_spec, deadline_at, str(source_snapshot), source_commit,
            )
            reserved = True
            log_path = self.state_dir / f"{training_id}.log"
            log_path.touch()
            result = TrainingResult(
                training_id=training_id, state=TrainingState.RUNNING,
                exit_code=None, elapsed_seconds=0, log_path=str(log_path),
                output_dir=output_dir, wandb_run_ids=(kubernetes_spec.wandb_run_id,),
                started_at=started_at, deadline_at=deadline_at,
                kubernetes_spec=kubernetes_spec, kubernetes_released=False,
                source_snapshot=str(source_snapshot), source_commit=source_commit,
            )
            active = _ActiveRemoteTraining(
                spec=kubernetes_spec, started_at=started_at, deadline_at=deadline_at,
                log_path=log_path, manifest=manifest,
            )
            thread = threading.Thread(
                target=self._monitor, args=(training_id,),
                name=f"senpai-kubernetes-training-{training_id}",
            )
            active.thread = thread
            with self._lock:
                if self._shutdown.is_set():
                    raise RuntimeError("Kubernetes training supervisor is closed")
                self._write_result(result)
                self._active[training_id] = active
                thread.start()
        except BaseException as error:
            if result is not None:
                try:
                    self._write_result(result.model_copy(update={
                        "state": TrainingState.FAILED,
                        "elapsed_seconds": time.time() - started_at,
                        "error_tail": f"Training supervision failed to start ({type(error).__name__}).",
                    }))
                except OSError as storage_error:
                    print(
                        f"Kubernetes startup failure persistence deferred: training_id={training_id} "
                        f"path={storage_error.filename} errno={storage_error.errno}",
                        file=sys.stderr, flush=True,
                    )
            if reserved:
                self._cleanup_failed_launch(training_id, kubernetes_spec)
            with self._lock:
                self._active.pop(training_id, None)
            raise
        finally:
            with self._lock:
                self._launching = False
                self._launch_complete.set()
        return result

    def _redact_output(self, text: str) -> str:
        sanitized, pending = _mask_output_chunk(text.encode(), self._wandb_api_key)
        return (sanitized + (b"<secret-hidden>" if pending else b"")).decode()

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
            if self._shutdown.is_set():
                raise RuntimeError("Kubernetes training supervisor is closed")
            active = self._active.get(training_id)
            if active is not None:
                active.cancelled = True
                thread = active.thread
            else:
                thread = None
        if active is None:
            return self.get_training_status(training_id)
        if thread is not None:
            thread.join(_JOIN_SECONDS)
            if thread.is_alive():
                raise TimeoutError("timed out waiting for Kubernetes training cancellation")
        return self.get_training_status(training_id)

    def close(self) -> None:
        """Stop local supervision without changing the remote run."""

        with self._lock:
            self._shutdown.set()
        if not self._launch_complete.wait(_JOIN_SECONDS):
            raise TimeoutError("timed out waiting for Kubernetes training launch shutdown")
        with self._lock:
            active = tuple(self._active.values())
        for training in active:
            if training.thread is not None:
                training.thread.join(_JOIN_SECONDS)
                if training.thread.is_alive():
                    raise TimeoutError(
                        "timed out waiting for Kubernetes training monitor shutdown"
                    )

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
                result,
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
        detail = ""
        delete_required = False
        try:
            if self._should_detach(active):
                return
            if active.manifest is not None:
                if not active.cancelled and time.time() < active.deadline_at:
                    self.client.apply(json.dumps(active.manifest))
                    while active.resource is None:
                        try:
                            active.resource = self.client.resource(
                                active.spec, nodes=self.nodes, gpus_per_node=self.gpus_per_node,
                            )
                        except Exception as error:
                            detail = f"Waiting for Kubernetes ownership: {error}"
                        else:
                            if active.resource is None:
                                raise RuntimeError("Kubernetes submission returned without a workload")
                            break
                        if active.cancelled or time.time() >= active.deadline_at:
                            break
                        if self._shutdown.wait(self.poll_seconds) and self._should_detach(active):
                            return
                    if not self._publish_resource(training_id, active) and self._should_detach(active):
                        return
                active.manifest = None
            while True:
                if active.cancelled:
                    state = TrainingState.CANCELLED
                    delete_required = True
                    break
                if time.time() >= active.deadline_at:
                    state = TrainingState.TIMED_OUT
                    delete_required = True
                    break
                if self._should_detach(active):
                    return
                try:
                    snapshot = self.client.state(active.resource)
                except Exception as error:
                    detail = f"Waiting for Kubernetes status: {error}"
                    if self._shutdown.wait(self.poll_seconds) and self._should_detach(active):
                        return
                    continue
                if snapshot is None:
                    state = TrainingState.FAILED
                    detail = "Remote training resource disappeared before completion."
                    break
                state, detail = snapshot
                if state is not TrainingState.RUNNING:
                    delete_required = state is not TrainingState.FINISHED
                    break
                if time.monotonic() >= active.next_diagnostics_at:
                    self._capture_diagnostics(training_id, active, detail)
                if self._shutdown.wait(self.poll_seconds) and self._should_detach(active):
                    return
        except Exception as error:
            if active.cancelled:
                state = TrainingState.CANCELLED
            elif time.time() >= active.deadline_at:
                state = TrainingState.TIMED_OUT
            else:
                state = TrainingState.FAILED
            detail = f"{type(error).__name__}: {error}"
            delete_required = True

        if self._should_detach(active) and state is TrainingState.RUNNING:
            return

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

        if active.resource is not None:
            self._capture_diagnostics(training_id, active, detail, wait=True)
        try:
            log_text = self._redact_output(
                active.log_path.read_bytes().decode(errors="ignore")
            )
            local_tail = log_text.encode()[-_ERROR_TAIL_BYTES:].decode(errors="ignore")
        except OSError:
            local_tail = ""
        error_tail = "" if state is TrainingState.FINISHED else self._redact_output(
            "\n".join(part for part in (detail, local_tail) if part)
        )[-_ERROR_TAIL_BYTES:]
        terminal = self.get_training_status(training_id).model_copy(
            update={
                "state": state,
                "exit_code": 0 if state is TrainingState.FINISHED else None,
                "elapsed_seconds": time.time() - active.started_at,
                "error_tail": error_tail,
                "kubernetes_resource": active.resource,
                "kubernetes_released": False,
            }
        )
        self._release_terminal(terminal, active, delete_required)

    def _capture_diagnostics(
        self,
        training_id: str,
        active: _ActiveRemoteTraining,
        detail: str,
        *,
        wait: bool = False,
    ) -> None:
        # Terminal refreshes must not reuse a stale in-flight live snapshot.
        if wait or active.diagnostics_future is None:
            future: Future[str] = Future()
            resource = active.resource

            def collect() -> None:
                try:
                    diagnostics = self.client.logs(resource)
                except Exception as error:  # optional diagnostics must not stop supervision
                    diagnostics = (
                        f"Kubernetes diagnostics unavailable: {type(error).__name__}: {error}"
                    )
                future.set_result(diagnostics)

            active.diagnostics_future = future
            threading.Thread(
                target=collect,
                name=f"senpai-kubernetes-diagnostics-{training_id}",
                daemon=True,
            ).start()
        if not wait and not active.diagnostics_future.done():
            return
        detail = self._redact_output(detail).encode()[:1024].decode(errors="ignore")
        diagnostics = "\n".join(
            part
            for part in (detail, self._redact_output(active.diagnostics_future.result()))
            if part
        )
        summary = diagnostics.encode()[:_ERROR_TAIL_BYTES].decode(errors="ignore")
        result = self.get_training_status(training_id)
        if summary != result.kubernetes_diagnostics:
            try:
                with active.log_path.open("a") as log:
                    log.write("\n=== Kubernetes workload diagnostics ===\n" + diagnostics + "\n")
                self._write_result(
                    result.model_copy(update={"kubernetes_diagnostics": summary})
                )
            except OSError as error:
                print(
                    f"Kubernetes diagnostics persistence skipped: training_id={training_id} "
                    f"path={error.filename} errno={error.errno}",
                    file=sys.stderr,
                    flush=True,
                )
        active.diagnostics_future = None
        active.next_diagnostics_at = time.monotonic() + _DIAGNOSTICS_SECONDS

    def _release_terminal(
        self,
        result: TrainingResult,
        active: _ActiveRemoteTraining,
        delete_required: bool,
    ) -> None:
        training_id = result.training_id
        terminal_persisted = False
        try:
            self._write_result(result)
            terminal_persisted = True
        except OSError as error:
            print(
                f"Kubernetes terminal persistence deferred before cleanup: "
                f"training_id={training_id} state={result.state.value} "
                f"path={error.filename} errno={error.errno}",
                file=sys.stderr,
                flush=True,
            )
        while True:
            try:
                if delete_required:
                    if active.resource is None:
                        active.resource = self.client.resource_identity(active.spec)
                    if active.resource is not None:
                        self.client.delete(active.resource)
            except Exception:
                if active.cancelled:
                    time.sleep(self.poll_seconds)
                elif self._shutdown.wait(self.poll_seconds):
                    return
                continue
            if not terminal_persisted:
                if not self._persist_result(result):
                    return
                terminal_persisted = True
            try:
                receipt = self.client.release(training_id)
                result = result.model_copy(update={"kubernetes_pod_receipt": receipt})
                break
            except Exception:
                if active.cancelled:
                    time.sleep(self.poll_seconds)
                elif self._shutdown.wait(self.poll_seconds):
                    return
        if not self._persist_result(result.model_copy(update={"kubernetes_released": True})):
            return
        with self._lock:
            self._active.pop(training_id, None)

    def _should_detach(self, active: _ActiveRemoteTraining) -> bool:
        return (
            self._shutdown.is_set()
            and not active.cancelled
            and time.time() < active.deadline_at
        )

    def _publish_resource(
        self,
        training_id: str,
        active: _ActiveRemoteTraining,
    ) -> bool:
        result = self.get_training_status(training_id)
        return self._persist_result(
            result.model_copy(update={"kubernetes_resource": active.resource}),
            active=active,
        )

    def _persist_result(
        self,
        result: TrainingResult,
        *,
        active: _ActiveRemoteTraining | None = None,
    ) -> bool:
        retry_limit = min(self.poll_seconds, 30.0) if active is not None else 30.0
        retry_seconds = min(self.poll_seconds, retry_limit)
        deferred = False
        while True:
            if active is not None and (
                active.cancelled or time.time() >= active.deadline_at
            ):
                return False
            try:
                self._write_result(result)
                break
            except OSError as error:
                if error.errno not in {errno.ENOSPC, errno.EDQUOT}:
                    raise
                deferred = True
                print(
                    f"Kubernetes result persistence deferred: training_id={result.training_id} "
                    f"state={result.state.value} released={result.kubernetes_released} "
                    f"path={error.filename} errno={error.errno} retry_seconds={retry_seconds}",
                    file=sys.stderr,
                    flush=True,
                )
                if self._shutdown.wait(retry_seconds):
                    return False
                retry_seconds = min(retry_seconds * 2, retry_limit)
        if deferred:
            print(
                f"Kubernetes result persistence recovered: training_id={result.training_id}",
                file=sys.stderr,
                flush=True,
            )
        return True

    def _write_result(self, result: TrainingResult) -> None:
        path = self.state_dir / f"{result.training_id}.json"
        temporary = path.with_suffix(".tmp")
        serialized = result.model_dump_json(indent=2)
        if self._wandb_api_key:
            # Receipts can contain provider messages, so protect every persisted field.
            secret = json.dumps(self._wandb_api_key.decode(), ensure_ascii=False)[1:-1]
            serialized = serialized.replace(secret, "<secret-hidden>")
        temporary.write_text(serialized)
        temporary.replace(path)


def _training_manifest(
    command: TrainingSpec,
    spec: KubernetesTrainingSpec,
    *,
    source_commit: str,
    relative_cwd: Path,
    nodes: int,
    gpus_per_node: int,
    output_dir: str,
) -> dict:
    cpu = int(os.environ["CPU_PER_STUDENT_GPU"])
    memory = int(os.environ["MEMORY_GI_PER_STUDENT_GPU"])
    mount = os.environ["PVC_MOUNT_PATH"]
    image = os.environ["SENPAI_TRAINING_IMAGE"]
    control_image = os.environ["SENPAI_TRAINING_CONTROL_IMAGE"]
    custom_image = image != control_image
    payload = base64.b64encode(json.dumps({
        "argv": command.argv, "cwd": str(Path("/workspace") / relative_cwd),
    }).encode()).decode()
    values = {
        "SENPAI_TRAINING_COMMAND_B64": payload,
        "SENPAI_TRAINING_WORKSPACE": "/workspace",
        "SENPAI_TARGET_PYTHON_ENV": "" if custom_image else "/home/senpai/.venvs/senpai-target",
        "SENPAI_TRAINING_OUTPUT_DIR": output_dir,
        "HOME": "/home/senpai",
        "NNODES": str(nodes), "GPUS_PER_NODE": str(gpus_per_node),
        "MASTER_ADDR": f"{spec.name}-worker-0.{spec.name}" if nodes > 1 else "127.0.0.1",
        "MASTER_PORT": "29500",
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
        "WANDB_RUN_ID": spec.wandb_run_id,
    }

    def template(mode: str, *, worker: bool) -> dict:
        resources = {
            "cpu": str(cpu * gpus_per_node) if worker else "1",
            "memory": f"{memory * gpus_per_node if worker else min(memory, 2)}Gi",
        }
        if worker:
            resources["nvidia.com/gpu"] = str(gpus_per_node)
        container = {
            "name": "training",
            "image": image if worker else control_image,
            "command": (
                ["python3", "/var/run/senpai-training/worker.py", mode] if custom_image and worker
                else ["/opt/senpai-venv/bin/python", "-P", "-m", "senpai_agent.training_worker", mode]
            ),
            "env": [
                *({"name": name, "value": value} for name, value in values.items()),
                {"name": "WANDB_API_KEY", "valueFrom": {"secretKeyRef": {
                    "name": os.environ["SENPAI_LAUNCH_SECRET_NAME"], "key": "wandb-api-key",
                }}},
            ],
            "resources": {"requests": resources, "limits": dict(resources)},
            "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "runAsGroup": 10001},
            "volumeMounts": [
                {"name": "dataset", "mountPath": mount},
                {"name": "home", "mountPath": "/home/senpai"},
            ],
        }
        if mode == "sshd":
            container["ports"] = [{"name": "ssh", "containerPort": 2222}]
            container["readinessProbe"] = {"tcpSocket": {"port": 2222}, "periodSeconds": 2}
        pod = {
            "restartPolicy": "Never",
            "containers": [container],
            "volumes": [
                {"name": "dataset", "persistentVolumeClaim": {"claimName": os.environ["PVC_CLAIM_NAME"]}},
                {"name": "home", "emptyDir": {}},
            ],
        }
        if worker:
            pod["tolerations"] = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
        return {"spec": pod}

    workload = {
        "apiVersion": "batch/v1" if nodes == 1 else "kubeflow.org/v2beta1",
        "kind": spec.kind,
        "metadata": {
            "name": spec.name, "namespace": spec.namespace,
            "annotations": {
                "senpai.wandb.com/source-commit": source_commit,
                "senpai.wandb.com/run-id": spec.wandb_run_id,
            },
        },
    }
    if nodes == 1:
        workload["spec"] = {"parallelism": 1, "completions": 1, "template": template("run", worker=True)}
    else:
        workload["spec"] = {
            "slotsPerWorker": gpus_per_node, "mpiImplementation": "OpenMPI",
            "sshAuthMountPath": "/home/senpai/.ssh", "launcherCreationPolicy": "WaitForWorkersReady",
            "mpiReplicaSpecs": {
                "Launcher": {"replicas": 1, "restartPolicy": "Never", "template": template("mpi", worker=False)},
                "Worker": {"replicas": nodes, "restartPolicy": "Never", "template": template("sshd", worker=True)},
            },
        }
    return workload


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


def _training_spec(training_id: str, *, nodes: int) -> KubernetesTrainingSpec:
    research = _dns_label(os.environ["RESEARCH_TAG"])
    student = _dns_label(os.environ["STUDENT_NAME"])
    suffix = uuid.UUID(training_id).hex[:12]
    child_suffix_length = max(len("-launcher"), len(f"-worker-{nodes - 1}"))
    prefix_limit = 63 - child_suffix_length - 1 - len(suffix)
    prefix = f"senpai-{research}-{student}"[:prefix_limit].rstrip("-")
    return KubernetesTrainingSpec(
        kind="Job" if nodes == 1 else "MPIJob",
        name=f"{prefix}-{suffix}",
        namespace=os.environ["SENPAI_KUBERNETES_NAMESPACE"],
        wandb_run_id=uuid.UUID(training_id).hex,
    )


def _dns_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "student"
