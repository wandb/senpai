"""Credential-isolated Kubernetes executor and controller-side kubectl proxy."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from typing import Never

import yaml

from senpai_agent.kubernetes_training import (
    EXECUTOR_SOCKET_ENV,
    KubernetesApiClient,
    KubernetesExecutorClient,
    _workload_shape,
)
from senpai_agent.training import KubernetesResourceRef, KubernetesTrainingSpec, TrainingState

_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")
_RUN_ID_ANNOTATION = "senpai.wandb.com/run-id"
_SOURCE_ANNOTATION = "senpai.wandb.com/source-commit"
_OWNERSHIP_LABELS = ("research-tag", "student", "senpai-training-id")
_TRAINING_RESOURCE_NAMES = {"cpu", "memory", "nvidia.com/gpu"}
_SOURCE_BUNDLE_MOUNT = "/var/lib/senpai-source/source.bundle"
_WORKSPACE_MOUNT = "/workspace"
_WORKSPACE_VOLUME = "senpai-workspace"


class KubernetesExecutor:
    """Validate and own one bounded training workload for one student."""

    def __init__(
        self,
        *,
        client: KubernetesApiClient,
        state_path: Path,
        namespace: str,
        nodes: int,
        gpus_per_node: int,
        max_timeout_seconds: int,
        cpu_per_gpu: int,
        memory_gi_per_gpu: int,
        pvc_claim_name: str,
        pvc_mount_path: Path,
        snapshot_root: Path,
        executor_image: str,
        launch_secret_name: str,
        research_tag: str,
        student_name: str,
        pod_name: str,
        pod_uid: str,
    ):
        self.client = client
        self.state_path = state_path
        self.namespace = namespace
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node
        self.max_timeout_seconds = max_timeout_seconds
        self.cpu_per_gpu = cpu_per_gpu
        self.memory_gi_per_gpu = memory_gi_per_gpu
        self.pvc_claim_name = pvc_claim_name
        self.pvc_mount_path = pvc_mount_path
        self.snapshot_root = snapshot_root
        self.executor_image = executor_image
        self.launch_secret_name = launch_secret_name
        self.research_tag = research_tag
        self.student_name = student_name
        self.pod_name = pod_name
        self.pod_uid = pod_uid
        self._lock = threading.Lock()
        self._reservation = (
            json.loads(state_path.read_text()) if state_path.exists() else None
        )
        self.reconcile()

    def handle(self, request: dict) -> object:
        operation = request["operation"]
        with self._lock:
            if operation == "reserve":
                return self._reserve(request)
            if operation == "adopt":
                return self._adopt(request)
            if operation == "apply":
                return self._apply(request["manifest"])
            if operation in {"resource", "resource_identity"}:
                spec = KubernetesTrainingSpec.model_validate(request["spec"])
                self._require_spec(spec)
                if operation == "resource" and (
                    int(request["nodes"]), int(request["gpus_per_node"])
                ) != (self.nodes, self.gpus_per_node):
                    raise PermissionError("requested resource shape exceeds this allocation")
                resource = self._current_resource()
                return resource.model_dump(mode="json") if resource else None
            if operation == "state":
                resource = self._require_resource(request["resource"])
                if not self._verify_current(resource):
                    return None
                state = self.client.state(resource)
                return [state[0].value, state[1]] if state else None
            if operation == "delete":
                resource = self._require_resource(request["resource"])
                if self._verify_current(resource):
                    self.client.delete(
                        resource,
                        min(int(request["timeout_seconds"]), 60),
                    )
                return None
            if operation == "logs":
                resource = self._require_resource(request["resource"])
                return self.client.logs(resource) if self._verify_current(resource) else ""
            if operation == "release":
                self._release(request["training_id"])
                return None
        raise ValueError(f"unsupported executor operation {operation!r}")

    def reconcile(self) -> None:
        """Recover create/persist races and reap an expired owned workload."""

        with self._lock:
            if self._reservation is None or self._reservation.get("released"):
                return
            resource = self._recorded_resource()
            if resource is None:
                resource = self._current_resource()
            if resource is None and self._reservation.get("create_attempted"):
                try:
                    resource = self._record_created(
                        self.client.create(self._reservation["manifest"], self.namespace)
                    )
                except Exception:
                    resource = self._current_resource()
            if (
                resource is not None
                and self._reservation["activation_authorized"]
                and not self._reservation["activated"]
                and time.time() < self._reservation["deadline_at"]
            ):
                self._activate(resource)
            if time.time() < self._reservation["deadline_at"]:
                return
            if resource is None and self._reservation.get("create_attempted"):
                return
            if resource is not None and self._verify_current(resource):
                self.client.delete(resource, 60)
            self._reservation["released"] = True
            self._write_state()

    def _reserve(self, request: dict) -> None:
        spec = KubernetesTrainingSpec.model_validate(request["spec"])
        if spec.namespace != self.namespace:
            raise ValueError(f"training namespace must be {self.namespace!r}")
        deadline_at = float(request["deadline_at"])
        now = time.time()
        if not now < deadline_at <= now + self.max_timeout_seconds:
            raise ValueError("training deadline is outside this student's configured limit")
        source_commit = str(request["source_commit"])
        source_snapshot = Path(str(request["source_snapshot"]))
        if (
            not _SOURCE_COMMIT.fullmatch(source_commit)
            or source_snapshot != self.snapshot_root / f"{source_commit}.bundle"
        ):
            raise ValueError("training source bundle does not match its full HEAD SHA")
        candidate = {
            "training_id": str(request["training_id"]),
            "spec": spec.model_dump(mode="json"),
            "deadline_at": deadline_at,
            "source_snapshot": str(source_snapshot),
            "source_commit": source_commit,
            "create_attempted": False,
            "manifest": None,
            "created": False,
            "resource": None,
            "activation_authorized": False,
            "activated": False,
            "released": False,
        }
        if self._reservation is not None and not self._reservation.get("released"):
            comparable = {
                key: self._reservation[key]
                for key in (
                    "training_id",
                    "spec",
                    "deadline_at",
                    "source_snapshot",
                    "source_commit",
                )
            }
            if comparable != {
                key: candidate[key]
                for key in comparable
            }:
                raise RuntimeError("this student already has an active Kubernetes training run")
            return
        self._reservation = candidate
        self._write_state()

    def _adopt(self, request: dict) -> None:
        spec = KubernetesTrainingSpec.model_validate(request["spec"])
        self._require_spec(spec)
        expected = KubernetesResourceRef.model_validate(request["resource"])
        current = self._current_resource()
        if current is None or current != expected:
            raise RuntimeError("remote Kubernetes workload is missing or was replaced")
        reservation = self._require_reservation()
        if not reservation["activation_authorized"]:
            raise RuntimeError("the Kubernetes create outcome was not confirmed")
        if not reservation["activated"]:
            self._activate(current)

    def _apply(self, manifest_text: str) -> str:
        reservation = self._require_reservation(active=True)
        current = self._current_resource()
        if current is not None:
            if not reservation["activation_authorized"]:
                raise RuntimeError("the Kubernetes create outcome was not confirmed")
            if not reservation["activated"]:
                self._activate(current)
            return f"{current.kind.lower()}/{current.name} unchanged\n"
        if reservation["created"]:
            raise RuntimeError("the reserved Kubernetes workload was already created")

        documents = list(yaml.safe_load_all(manifest_text))
        if len(documents) != 1 or not isinstance(documents[0], dict):
            raise ValueError("training submission must contain exactly one Kubernetes object")
        manifest = documents[0]
        spec = KubernetesTrainingSpec.model_validate(reservation["spec"])
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict) or (
            manifest.get("kind") != spec.kind
            or metadata.get("name") != spec.name
            or metadata.get("namespace", self.namespace) != self.namespace
        ):
            raise ValueError("submitted resource does not match the reserved training identity")
        annotations = metadata.get("annotations", {})
        if (
            annotations.get(_RUN_ID_ANNOTATION) != spec.wandb_run_id
            or annotations.get(_SOURCE_ANNOTATION) != reservation["source_commit"]
        ):
            raise ValueError("training manifest annotations do not match reserved evidence")

        self._secure_manifest(manifest, reservation)
        reservation["create_attempted"] = True
        reservation["manifest"] = manifest
        self._write_state()
        created = self.client.create(manifest, self.namespace)
        resource = self._record_created(created)
        reservation["activation_authorized"] = True
        self._write_state()
        self._activate(resource)
        return f"{resource.kind.lower()}/{resource.name} created\n"

    def _activate(self, resource: KubernetesResourceRef) -> None:
        reservation = self._require_reservation()
        remaining = reservation["deadline_at"] - time.time()
        if remaining <= 0:
            self._expire_activation(resource, "before")
        try:
            self.client.activate(resource, min(30, remaining))
        except Exception:
            if time.time() >= reservation["deadline_at"]:
                self._expire_activation(resource, "during")
            raise
        if time.time() >= reservation["deadline_at"]:
            self._expire_activation(resource, "during")
        reservation["activated"] = True
        self._write_state()

    def _expire_activation(
        self,
        resource: KubernetesResourceRef,
        phase: str,
    ) -> Never:
        self.client.delete(resource, 60)
        reservation = self._require_reservation()
        reservation["released"] = True
        self._write_state()
        raise TimeoutError(f"training deadline elapsed {phase} Kubernetes activation")

    def _record_created(self, document: dict) -> KubernetesResourceRef:
        self._validate_owned_document(document)
        resource = self._resource_from_document(document)
        reservation = self._require_reservation()
        reservation["created"] = True
        reservation["manifest"] = None
        reservation["resource"] = resource.model_dump(mode="json")
        self._write_state()
        return resource

    def _secure_manifest(self, manifest: dict, reservation: dict) -> None:
        spec = KubernetesTrainingSpec.model_validate(reservation["spec"])
        expected_api = "batch/v1" if spec.kind == "Job" else "kubeflow.org/v2beta1"
        if set(manifest) != {"apiVersion", "kind", "metadata", "spec"}:
            raise ValueError("training manifest contains unsupported top-level fields")
        if manifest.get("apiVersion") != expected_api:
            raise ValueError(f"{spec.kind} must use apiVersion {expected_api}")
        metadata = manifest["metadata"]
        if set(metadata) - {"name", "namespace", "labels", "annotations"}:
            raise ValueError("training metadata contains unsupported control fields")
        metadata["namespace"] = self.namespace
        metadata["annotations"] = {
            _RUN_ID_ANNOTATION: spec.wandb_run_id,
            _SOURCE_ANNOTATION: reservation["source_commit"],
        }
        metadata["labels"] = self._ownership_labels(reservation)
        metadata["ownerReferences"] = [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "name": self.pod_name,
                "uid": self.pod_uid,
            }
        ]

        remaining = max(1, int(reservation["deadline_at"] - time.time()))
        pod_specs: list[tuple[bool, dict]]
        if spec.kind == "MPIJob":
            allowed = {
                "mpiImplementation",
                "launcherCreationPolicy",
                "slotsPerWorker",
                "runPolicy",
                "mpiReplicaSpecs",
            }
            if set(manifest["spec"]) - allowed:
                raise ValueError("MPIJob contains unsupported control fields")
            replicas = manifest["spec"].get("mpiReplicaSpecs", {})
            if set(replicas) != {"Launcher", "Worker"}:
                raise ValueError("MPIJob must contain exactly Launcher and Worker roles")
            launcher = replicas["Launcher"]
            worker = replicas["Worker"]
            if any(
                set(replica) - {"replicas", "restartPolicy", "template"}
                for replica in (launcher, worker)
            ):
                raise ValueError("MPIJob replica roles contain unsupported control fields")
            if (
                int(launcher.get("replicas", 0)) != 1
                or int(worker.get("replicas", 0)) != self.nodes
                or launcher.get("restartPolicy") != "Never"
                or worker.get("restartPolicy") != "Never"
            ):
                raise ValueError("MPIJob replica topology does not match this allocation")
            if int(manifest["spec"].get("slotsPerWorker", 0)) != self.gpus_per_node:
                raise ValueError("MPIJob slotsPerWorker must equal GPUs per worker")
            run_policy = manifest["spec"].setdefault("runPolicy", {})
            if set(run_policy) - {
                "activeDeadlineSeconds",
                "backoffLimit",
                "cleanPodPolicy",
                "schedulingPolicy",
                "suspend",
                "ttlSecondsAfterFinished",
            }:
                raise ValueError("MPIJob runPolicy contains unsupported control fields")
            run_policy["activeDeadlineSeconds"] = remaining
            run_policy["backoffLimit"] = 0
            run_policy["cleanPodPolicy"] = "Running"
            run_policy["suspend"] = True
            scheduling_policy = run_policy.setdefault("schedulingPolicy", {})
            if not isinstance(scheduling_policy, dict) or set(scheduling_policy) - {
                "minAvailable",
                "scheduleTimeoutSeconds",
            }:
                raise ValueError(
                    "MPIJob schedulingPolicy contains unsupported scheduling controls"
                )
            scheduling_policy["minAvailable"] = self.nodes + 1
            scheduling_policy["scheduleTimeoutSeconds"] = remaining
            pod_specs = [
                (False, launcher["template"]),
                (True, worker["template"]),
            ]
        else:
            allowed = {
                "activeDeadlineSeconds",
                "backoffLimit",
                "completionMode",
                "completions",
                "parallelism",
                "suspend",
                "template",
                "ttlSecondsAfterFinished",
            }
            if set(manifest["spec"]) - allowed:
                raise ValueError("Job contains unsupported control fields")
            if (
                int(manifest["spec"].get("parallelism", 1)) != self.nodes
                or int(manifest["spec"].get("completions", 1)) != self.nodes
            ):
                raise ValueError("Job parallelism and completions must match this allocation")
            manifest["spec"]["activeDeadlineSeconds"] = remaining
            manifest["spec"]["backoffLimit"] = 0
            manifest["spec"]["suspend"] = True
            pod_specs = [(True, manifest["spec"]["template"])]

        found_key = found_run_id = False
        for allow_gpus, template in pod_specs:
            template_metadata = template.setdefault("metadata", {})
            if set(template_metadata) - {"annotations", "labels"}:
                raise ValueError("pod template metadata contains unsupported control fields")
            app = template_metadata.get("labels", {}).get("app")
            template_metadata["annotations"] = {}
            template_metadata["labels"] = self._ownership_labels(reservation)
            if app:
                template_metadata["labels"]["app"] = app
            template_metadata["labels"][
                "nn-cfd-job" if spec.kind == "MPIJob" else "job-name"
            ] = spec.name
            key, run_id = self._secure_pod_spec(
                template["spec"],
                allow_gpus=allow_gpus,
                wandb_run_id=spec.wandb_run_id,
                reservation=reservation,
            )
            found_key |= key
            found_run_id |= run_id
        if not found_key or not found_run_id:
            raise ValueError("training worker must bind the launch W&B key and run ID")
        if _workload_shape(manifest) != (self.nodes, self.gpus_per_node):
            raise ValueError(
                f"training must request exactly {self.nodes} nodes x "
                f"{self.gpus_per_node} GPUs"
            )

    def _secure_pod_spec(
        self,
        pod_spec: dict,
        *,
        allow_gpus: bool,
        wandb_run_id: str,
        reservation: dict,
    ) -> tuple[bool, bool]:
        forbidden_values = {
            "serviceAccountName",
            "serviceAccount",
            "imagePullSecrets",
            "runtimeClassName",
            "priorityClassName",
            "nodeName",
            "ephemeralContainers",
            "resourceClaims",
            "resources",
        }
        if forbidden_values & pod_spec.keys():
            raise ValueError("training pod requests forbidden workload identity or runtime access")
        if any(pod_spec.get(key) for key in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace")):
            raise ValueError("training pod requests forbidden host namespace access")
        pod_spec["automountServiceAccountToken"] = False
        pod_spec["terminationGracePeriodSeconds"] = 30
        pod_security = pod_spec.setdefault("securityContext", {})
        if pod_security.get("seccompProfile", {}).get("type") == "Unconfined":
            raise ValueError("training pods may not disable seccomp")
        pod_security["seccompProfile"] = {"type": "RuntimeDefault"}

        containers = pod_spec.get("containers", [])
        if not containers:
            raise ValueError("training pod must contain a main container")
        self._install_source_checkout(pod_spec, containers, reservation)

        for volume in pod_spec["volumes"]:
            volume_types = set(volume) - {"name"}
            if volume_types == {"persistentVolumeClaim"}:
                claim = volume["persistentVolumeClaim"].get("claimName")
                if claim != self.pvc_claim_name:
                    raise ValueError(f"training may mount only PVC {self.pvc_claim_name!r}")
            elif volume_types != {"emptyDir"}:
                raise ValueError("training volumes are limited to the dataset PVC and emptyDir")

        init_containers = pod_spec["initContainers"]
        found_key = found_run_id = False
        for container in init_containers:
            self._secure_container(container, allow_gpu=False)
            self._secure_environment(
                container,
                allow_wandb=False,
                wandb_run_id=wandb_run_id,
            )
        for container in containers:
            self._secure_container(container, allow_gpu=allow_gpus)
            key, run_id = self._secure_environment(
                container,
                allow_wandb=allow_gpus,
                wandb_run_id=wandb_run_id,
            )
            found_key |= key
            found_run_id |= run_id

        requested = sum(_container_gpu(container, "requests") for container in containers)
        limited = sum(_container_gpu(container, "limits") for container in containers)
        expected = self.gpus_per_node if allow_gpus else 0
        if (requested, limited) != (expected, expected):
            raise ValueError(f"pod must request and limit exactly {expected} GPU(s)")
        cpu_cap = self.cpu_per_gpu * (self.gpus_per_node if allow_gpus else 1)
        memory_cap = self.memory_gi_per_gpu * (self.gpus_per_node if allow_gpus else 1)
        cpu_request = max(
            sum(_cpu_quantity(container, "requests") for container in containers),
            max(
                (_cpu_quantity(container, "requests") for container in init_containers),
                default=0,
            ),
        )
        memory_request = max(
            sum(_memory_gi_quantity(container, "requests") for container in containers),
            max(
                (
                    _memory_gi_quantity(container, "requests")
                    for container in init_containers
                ),
                default=0,
            ),
        )
        if allow_gpus and (cpu_request, memory_request) != (cpu_cap, memory_cap):
            raise ValueError(
                "training workers must use the exact per-GPU CPU and memory allocation"
            )
        if cpu_request > cpu_cap:
            raise ValueError(f"pod CPU resources exceed the {cpu_cap:g}-CPU allocation")
        if memory_request > memory_cap:
            raise ValueError(
                f"pod memory resources exceed the {memory_cap:g}Gi allocation"
            )
        return found_key, found_run_id

    def _install_source_checkout(
        self,
        pod_spec: dict,
        containers: list[dict],
        reservation: dict,
    ) -> None:
        source_bundle = Path(reservation["source_snapshot"])
        try:
            source_subpath = source_bundle.relative_to(self.pvc_mount_path)
        except ValueError as error:
            raise ValueError("training source bundle is outside the configured PVC") from error

        volumes = pod_spec.setdefault("volumes", [])
        if any(volume.get("name") == _WORKSPACE_VOLUME for volume in volumes):
            raise ValueError("training manifest uses a reserved Senpai volume name")
        dataset_volumes = [
            volume
            for volume in volumes
            if volume.get("persistentVolumeClaim", {}).get("claimName")
            == self.pvc_claim_name
        ]
        if len(dataset_volumes) != 1:
            raise ValueError("training manifest must define exactly one dataset PVC volume")
        dataset_volume = dataset_volumes[0]
        dataset_volume["persistentVolumeClaim"]["readOnly"] = False
        volumes.append({"name": _WORKSPACE_VOLUME, "emptyDir": {}})

        for container in containers:
            mounts = container.setdefault("volumeMounts", [])
            if any(mount.get("name") == _WORKSPACE_VOLUME for mount in mounts):
                raise ValueError("training container uses a reserved Senpai volume name")
            nested_workspace_mounts = [
                mount
                for mount in mounts
                if str(mount.get("mountPath", "")).startswith(f"{_WORKSPACE_MOUNT}/")
            ]
            if nested_workspace_mounts:
                raise ValueError("training containers may not shadow the Senpai workspace")
            container["volumeMounts"] = [
                mount for mount in mounts if mount.get("mountPath") != _WORKSPACE_MOUNT
            ] + [{"name": _WORKSPACE_VOLUME, "mountPath": _WORKSPACE_MOUNT}]

        pod_spec["initContainers"] = [
            {
                "name": "senpai-source-checkout",
                "image": self.executor_image,
                "imagePullPolicy": "IfNotPresent",
                "command": [
                    "python",
                    "-m",
                    "senpai_agent.kubernetes_executor",
                    "checkout",
                ],
                "env": [
                    {"name": "SENPAI_SOURCE_BUNDLE", "value": _SOURCE_BUNDLE_MOUNT},
                    {"name": "SENPAI_SOURCE_COMMIT", "value": reservation["source_commit"]},
                    {"name": "SENPAI_SOURCE_WORKSPACE", "value": _WORKSPACE_MOUNT},
                    {"name": "HOME", "value": _WORKSPACE_MOUNT},
                ],
                "resources": {
                    "requests": {"cpu": "1", "memory": "2Gi"},
                    "limits": {"cpu": "1", "memory": "2Gi"},
                },
                "securityContext": {
                    "runAsNonRoot": False,
                    "runAsUser": 0,
                    "runAsGroup": 0,
                    "readOnlyRootFilesystem": True,
                },
                "volumeMounts": [
                    {
                        "name": dataset_volume["name"],
                        "mountPath": _SOURCE_BUNDLE_MOUNT,
                        "subPath": str(source_subpath),
                        "readOnly": True,
                    },
                    {"name": _WORKSPACE_VOLUME, "mountPath": _WORKSPACE_MOUNT},
                ],
            }
        ]

    @staticmethod
    def _secure_container(container: dict, *, allow_gpu: bool) -> None:
        if container.get("envFrom"):
            raise ValueError("training containers may not use envFrom")
        resources = container.get("resources", {})
        if set(resources) - {"requests", "limits"} or any(
            set(resources.get(resource_type, {})) - _TRAINING_RESOURCE_NAMES
            for resource_type in ("requests", "limits")
        ):
            raise ValueError("training containers request unsupported resources")
        if not allow_gpu and (
            _container_gpu(container, "requests")
            or _container_gpu(container, "limits")
        ):
            raise ValueError("init and launcher containers may not request GPUs")
        if not {"cpu", "memory"}.issubset(resources.get("requests", {})) or not {
            "cpu",
            "memory",
        }.issubset(resources.get("limits", {})):
            raise ValueError("every training container needs CPU and memory resources")
        cpu_request = _cpu_quantity(container, "requests")
        memory_request = _memory_gi_quantity(container, "requests")
        if (
            cpu_request <= 0
            or memory_request <= 0
            or cpu_request != _cpu_quantity(container, "limits")
            or memory_request != _memory_gi_quantity(container, "limits")
        ):
            raise ValueError("each container's CPU and memory requests must equal its limits")
        if "restartPolicy" in container:
            raise ValueError("training containers may not override restart policy")
        if any(port.get("hostPort") for port in container.get("ports", [])):
            raise ValueError("training containers may not reserve host ports")
        security = container.setdefault("securityContext", {})
        capabilities = security.get("capabilities", {})
        if (
            security.get("privileged")
            or security.get("allowPrivilegeEscalation")
            or security.get("procMount") == "Unmasked"
            or security.get("seccompProfile", {}).get("type") == "Unconfined"
            or security.get("appArmorProfile", {}).get("type") == "Unconfined"
            or capabilities.get("add")
        ):
            raise ValueError("training containers may not elevate privileges")
        security["allowPrivilegeEscalation"] = False
        security["capabilities"] = {"drop": ["ALL"]}
        security["seccompProfile"] = {"type": "RuntimeDefault"}

    def _secure_environment(
        self,
        container: dict,
        *,
        allow_wandb: bool,
        wandb_run_id: str,
    ) -> tuple[bool, bool]:
        found_key = found_run_id = False
        forbidden_names = {
            "ANTHROPIC_API_KEY",
            "EXA_API_KEY",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "OPENAI_API_KEY",
        }
        for item in container.get("env", []):
            name = item.get("name")
            value_from = item.get("valueFrom", {})
            if name in forbidden_names:
                raise ValueError(f"training manifest may not receive {name}")
            if name == "WANDB_API_KEY":
                if not allow_wandb:
                    raise ValueError("only a training worker may receive the W&B key")
                item.pop("value", None)
                item["valueFrom"] = {
                    "secretKeyRef": {
                        "name": self.launch_secret_name,
                        "key": "wandb-api-key",
                    }
                }
                found_key = True
            elif "secretKeyRef" in value_from:
                raise ValueError("training may reference only the launch W&B key")
            if name in {"WANDB_RUN_ID", "NN_CFD_WANDB_RUN_ID"}:
                if item.get("value") != wandb_run_id:
                    raise ValueError("training manifest W&B run ID does not match reservation")
                found_run_id = True
        return found_key, found_run_id

    def _current_resource(self) -> KubernetesResourceRef | None:
        reservation = self._require_reservation()
        spec = KubernetesTrainingSpec.model_validate(reservation["spec"])
        document = self.client.document(spec)
        if document is None:
            return None
        self._validate_owned_document(document)
        current = self._resource_from_document(document)
        recorded = self._recorded_resource()
        if recorded is not None and current != recorded:
            raise RuntimeError("remote Kubernetes workload was replaced")
        if recorded is None:
            reservation["created"] = True
            reservation["manifest"] = None
            reservation["resource"] = current.model_dump(mode="json")
            self._write_state()
        return current

    def _verify_current(self, resource: KubernetesResourceRef) -> bool:
        document = self.client.document(
            KubernetesTrainingSpec(
                kind=resource.kind,
                name=resource.name,
                namespace=resource.namespace,
                wandb_run_id=self._require_reservation()["spec"]["wandb_run_id"],
            )
        )
        if document is None:
            return False
        self._validate_owned_document(document, expected_uid=resource.uid)
        return True

    def _validate_owned_document(
        self,
        document: dict,
        *,
        expected_uid: str | None = None,
    ) -> None:
        reservation = self._require_reservation()
        spec = KubernetesTrainingSpec.model_validate(reservation["spec"])
        metadata = document["metadata"]
        if (
            document.get("kind") != spec.kind
            or metadata.get("name") != spec.name
            or metadata.get("namespace", self.namespace) != self.namespace
            or (expected_uid is not None and metadata.get("uid") != expected_uid)
        ):
            raise PermissionError("Kubernetes resource identity is not owned by this run")
        labels = metadata.get("labels", {})
        expected_labels = self._ownership_labels(reservation)
        if any(labels.get(key) != value for key, value in expected_labels.items()):
            raise PermissionError("Kubernetes resource ownership labels do not match")
        annotations = metadata.get("annotations", {})
        if (
            annotations.get(_RUN_ID_ANNOTATION) != spec.wandb_run_id
            or annotations.get(_SOURCE_ANNOTATION) != reservation["source_commit"]
        ):
            raise PermissionError("Kubernetes resource evidence annotations do not match")
        if not any(
            owner.get("apiVersion") == "v1"
            and owner.get("kind") == "Pod"
            and owner.get("name") == self.pod_name
            and owner.get("uid") == self.pod_uid
            for owner in metadata.get("ownerReferences", [])
        ):
            raise PermissionError("Kubernetes resource owner does not match this student pod")
        if _workload_shape(document) != (self.nodes, self.gpus_per_node):
            raise PermissionError("Kubernetes resource shape does not match this allocation")

    def _resource_from_document(self, document: dict) -> KubernetesResourceRef:
        nodes, gpus = _workload_shape(document)
        return KubernetesResourceRef(
            kind=document["kind"],
            name=document["metadata"]["name"],
            namespace=document["metadata"].get("namespace", self.namespace),
            uid=document["metadata"]["uid"],
            nodes=nodes,
            gpus_per_node=gpus,
        )

    def _recorded_resource(self) -> KubernetesResourceRef | None:
        value = self._require_reservation().get("resource")
        return KubernetesResourceRef.model_validate(value) if value else None

    def _require_reservation(self, *, active: bool = False) -> dict:
        if self._reservation is None:
            raise RuntimeError("no Kubernetes training run is reserved")
        if active and self._reservation.get("released"):
            raise RuntimeError("the Kubernetes training reservation is closed")
        return self._reservation

    def _require_spec(self, spec: KubernetesTrainingSpec) -> None:
        expected = KubernetesTrainingSpec.model_validate(
            self._require_reservation()["spec"]
        )
        if spec != expected:
            raise PermissionError("resource does not belong to the reserved training run")

    def _require_resource(self, value: object) -> KubernetesResourceRef:
        resource = KubernetesResourceRef.model_validate(value)
        recorded = self._recorded_resource()
        if recorded is None:
            recorded = self._current_resource()
        if recorded is None or resource != recorded:
            raise PermissionError("resource does not belong to the reserved training run")
        return resource

    def _release(self, training_id: str) -> None:
        reservation = self._require_reservation()
        if reservation["training_id"] != training_id:
            raise PermissionError("training reservation belongs to a different run")
        if reservation.get("released"):
            return
        recorded = self._recorded_resource()
        resource = self._current_resource()
        if resource is None and recorded is None and reservation.get("create_attempted"):
            raise RuntimeError("cannot release an unresolved Kubernetes create attempt")
        if resource is not None and self._verify_current(resource):
            state = self.client.state(resource)
            if state is not None and state[0] is TrainingState.RUNNING:
                raise RuntimeError("cannot release a live Kubernetes workload")
        reservation["released"] = True
        self._write_state()

    def _ownership_labels(self, reservation: dict) -> dict[str, str]:
        return {
            "app": "senpai-training",
            "research-tag": self.research_tag,
            "student": self.student_name,
            "senpai-training-id": reservation["training_id"],
        }

    def _write_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._reservation, sort_keys=True))
        temporary.replace(self.state_path)


def _container_gpu(container: dict, resource_type: str) -> int:
    return int(
        container.get("resources", {})
        .get(resource_type, {})
        .get("nvidia.com/gpu", 0)
    )


def _cpu_quantity(container: dict, resource_type: str) -> float:
    value = container.get("resources", {}).get(resource_type, {}).get("cpu", 0)
    text = str(value)
    result = float(text[:-1]) / 1000 if text.endswith("m") else float(text)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid CPU quantity {value!r}")
    return result


def _memory_gi_quantity(container: dict, resource_type: str) -> float:
    value = container.get("resources", {}).get(resource_type, {}).get("memory", 0)
    text = str(value)
    units = {
        "Ki": 1 / 1024**2,
        "Mi": 1 / 1024,
        "Gi": 1,
        "Ti": 1024,
        "K": 1000 / 1024**3,
        "M": 1000**2 / 1024**3,
        "G": 1000**3 / 1024**3,
        "T": 1000**4 / 1024**3,
    }
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            result = float(text[: -len(suffix)]) * multiplier
            break
    else:
        result = float(text) / 1024**3
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid memory quantity {value!r}")
    return result


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(30)
        payload = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
        if len(payload) > _MAX_REQUEST_BYTES:
            response = {"ok": False, "error": "executor request is too large"}
        else:
            try:
                result = self.server.executor.handle(json.loads(payload))
                response = {"ok": True, "result": result}
            except Exception as error:  # noqa: BLE001
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: str, executor: KubernetesExecutor):
        self.executor = executor
        super().__init__(socket_path, _RequestHandler)

    def service_actions(self) -> None:
        try:
            self.executor.reconcile()
        except Exception as error:  # keep serving while the Kubernetes API recovers
            print(f"Kubernetes reconciliation deferred: {error}", file=sys.stderr)


def serve() -> None:
    if os.environ.get("SENPAI_IMAGE_REVISION") != os.environ.get(
        "SENPAI_REPO_REVISION"
    ):
        raise RuntimeError("executor image and Senpai source revisions do not match")
    socket_path = Path(os.environ[EXECUTOR_SOCKET_ENV])
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    executor = KubernetesExecutor(
        client=KubernetesApiClient(),
        state_path=Path(os.environ["SENPAI_KUBERNETES_EXECUTOR_STATE"]),
        namespace=os.environ["SENPAI_KUBERNETES_NAMESPACE"],
        nodes=int(os.environ["NODES_PER_STUDENT"]),
        gpus_per_node=int(os.environ["GPUS_PER_STUDENT_NODE"]),
        max_timeout_seconds=int(os.environ["SENPAI_MAX_TRAINING_TIMEOUT_SECONDS"]),
        cpu_per_gpu=int(os.environ["CPU_PER_STUDENT_GPU"]),
        memory_gi_per_gpu=int(os.environ["MEMORY_GI_PER_STUDENT_GPU"]),
        pvc_claim_name=os.environ["PVC_CLAIM_NAME"],
        pvc_mount_path=Path(os.environ["PVC_MOUNT_PATH"]),
        snapshot_root=Path(os.environ["SENPAI_TRAINING_SNAPSHOT_ROOT"]),
        executor_image=os.environ["SENPAI_EXECUTOR_IMAGE"],
        launch_secret_name=os.environ["SENPAI_LAUNCH_SECRET_NAME"],
        research_tag=os.environ["RESEARCH_TAG"],
        student_name=os.environ["STUDENT_NAME"],
        pod_name=os.environ["SENPAI_POD_NAME"],
        pod_uid=os.environ["SENPAI_POD_UID"],
    )
    with _UnixServer(str(socket_path), executor) as server:
        os.chmod(socket_path, 0o660)
        server.serve_forever(poll_interval=1)


def kubectl_proxy(arguments: list[str]) -> None:
    position = 0
    while position + 1 < len(arguments) and arguments[position] in {
        "--context",
        "--namespace",
        "-n",
    }:
        position += 2
    if arguments[position:] != ["apply", "-f", "-"]:
        raise SystemExit("Senpai's kubectl proxy permits only `apply -f -`")
    output = KubernetesExecutorClient(os.environ[EXECUTOR_SOCKET_ENV]).apply(
        sys.stdin.read()
    )
    print(output, end="" if output.endswith("\n") else "\n")


def checkout_source_bundle(source: Path, workspace: Path, expected_commit: str) -> None:
    """Copy an untrusted bundle locally, then check out exactly its claimed commit."""

    if not _SOURCE_COMMIT.fullmatch(expected_commit):
        raise ValueError("source commit must be a full lowercase SHA")
    workspace.mkdir(parents=True, exist_ok=True)
    for child in workspace.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    local_bundle = workspace / ".senpai-source.bundle"
    shutil.copyfile(source, local_bundle)
    git_environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    subprocess.run(["git", "init", "--quiet", workspace], env=git_environment, check=True)
    subprocess.run(
        ["git", "-C", workspace, "fetch", "--quiet", "--no-tags", local_bundle, "HEAD"],
        env=git_environment,
        check=True,
    )
    fetched = subprocess.check_output(
        ["git", "-C", workspace, "rev-parse", "--verify", "FETCH_HEAD^{commit}"],
        env=git_environment,
        text=True,
    ).strip()
    if fetched != expected_commit:
        raise RuntimeError(
            f"source bundle contains commit {fetched}, expected {expected_commit}"
        )
    subprocess.run(
        ["git", "-C", workspace, "checkout", "--quiet", "--detach", expected_commit],
        env=git_environment,
        check=True,
    )
    local_bundle.unlink()
    checked_out = subprocess.check_output(
        ["git", "-C", workspace, "rev-parse", "--verify", "HEAD"],
        env=git_environment,
        text=True,
    ).strip()
    if checked_out != expected_commit:
        raise RuntimeError(
            f"checked out commit {checked_out}, expected {expected_commit}"
        )


def main() -> None:
    mode, *arguments = sys.argv[1:]
    if mode == "serve":
        serve()
    elif mode == "kubectl":
        kubectl_proxy(arguments)
    elif mode == "checkout" and not arguments:
        checkout_source_bundle(
            Path(os.environ["SENPAI_SOURCE_BUNDLE"]),
            Path(os.environ["SENPAI_SOURCE_WORKSPACE"]),
            os.environ["SENPAI_SOURCE_COMMIT"],
        )
    else:
        raise SystemExit(
            "usage: python -m senpai_agent.kubernetes_executor serve|kubectl|checkout"
        )


if __name__ == "__main__":
    main()
