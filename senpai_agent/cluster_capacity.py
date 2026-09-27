"""Publish bounded, sanitized Kubernetes resource-fit observations.

Only the dedicated observer holds Kubernetes credentials. Agents read its
projected ConfigMap; observations never reserve resources or authorize a launch.
"""

from __future__ import annotations

import json
import os
import re
import signal
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from senpai_agent.kubernetes_training import KubernetesApiClient, KubernetesApiError

SNAPSHOT_ENV = "SENPAI_CAPACITY_SNAPSHOT"
MAX_SNAPSHOT_BYTES = 64 * 1024
MAX_AGE_SECONDS = 120
COLLECTION_SECONDS = 25
MAX_ITEMS = 20_000
MAX_PAGES = 100
LIMITATIONS = (
    "Advisory resource fit only; the Kubernetes scheduler remains authoritative.",
    "Pod slots, affinity, topology, quotas, PVC placement, launcher resources and concurrent submissions are not assessed.",
    "Counts use configured node selectors and tolerations, not an experiment's submitted Pod specification.",
    "Resize accounting conservatively retains the largest declared, allocated or actuated request.",
)


class Toleration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = ""
    operator: Literal["Equal", "Exists"] = "Equal"
    value: str = ""
    effect: Literal["", "NoSchedule", "PreferNoSchedule", "NoExecute"] = ""
    tolerationSeconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_match(self) -> Toleration:
        if self.operator == "Exists" and self.value:
            raise ValueError("Exists tolerations cannot specify a value")
        if not self.key and self.operator != "Exists":
            raise ValueError("an empty toleration key requires Exists")
        if self.tolerationSeconds is not None and self.effect != "NoExecute":
            raise ValueError("tolerationSeconds requires NoExecute")
        return self


class CapacityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: int = Field(gt=0)
    gpus_per_node: int = Field(gt=0)
    cpu_per_node: float = Field(gt=0, allow_inf_nan=False)
    memory_gib_per_node: float = Field(gt=0, allow_inf_nan=False)
    node_selector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[Toleration] = Field(default_factory=list)
    hpc_verification: bool = False


class WorkerShape(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: int = Field(gt=0)
    gpus_per_node: int = Field(gt=0)
    cpu_per_node: float = Field(gt=0, allow_inf_nan=False)
    memory_gib_per_node: float = Field(gt=0, allow_inf_nan=False)


class CapacityCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpu_nodes: int = Field(ge=0)
    ready_nodes: int = Field(ge=0)
    cordoned_ready_nodes: int = Field(ge=0)
    not_ready_nodes: int = Field(ge=0)
    selector_excluded_nodes: int = Field(ge=0)
    taint_excluded_nodes: int = Field(ge=0)
    schedulable_candidate_nodes: int = Field(ge=0)
    resource_fit_nodes: int = Field(ge=0)
    physically_available_fit_nodes: int = Field(ge=0)
    full_gpu_nodes: int = Field(ge=0)
    physically_idle_gpu_nodes: int = Field(ge=0)
    verified_preemptible_nodes: int = Field(ge=0)
    verified_preemptible_gpus: int = Field(ge=0)
    free_gpus: int = Field(ge=0)
    free_cpu: float = Field(ge=0, allow_inf_nan=False)
    free_memory_gib: float = Field(ge=0, allow_inf_nan=False)
    bound_pending_gpus: int = Field(ge=0)
    unbound_pending_gpus: int = Field(ge=0)


class CapacitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    observed_at: datetime
    status: Literal["available", "unknown"]
    reason: (
        Literal[
            "not_configured", "unavailable", "collection_failed", "invalid", "stale"
        ]
        | None
    ) = None
    requirements: WorkerShape | None = None
    counts: CapacityCounts | None = None
    scope: Literal["all_nodes_and_nonterminal_pods"] = "all_nodes_and_nonterminal_pods"
    limitations: tuple[str, ...] = LIMITATIONS


@dataclass(frozen=True)
class Resources:
    gpus: int = 0
    cpu_millis: int = 0
    memory_bytes: int = 0

    def __add__(self, other: Resources) -> Resources:
        return Resources(
            self.gpus + other.gpus,
            self.cpu_millis + other.cpu_millis,
            self.memory_bytes + other.memory_bytes,
        )

    def maximum(self, other: Resources) -> Resources:
        return Resources(
            max(self.gpus, other.gpus),
            max(self.cpu_millis, other.cpu_millis),
            max(self.memory_bytes, other.memory_bytes),
        )

    def remaining(self, used: Resources) -> Resources:
        return Resources(
            max(0, self.gpus - used.gpus),
            max(0, self.cpu_millis - used.cpu_millis),
            max(0, self.memory_bytes - used.memory_bytes),
        )

    def fits(self, request: Resources) -> bool:
        return (
            self.gpus >= request.gpus
            and self.cpu_millis >= request.cpu_millis
            and self.memory_bytes >= request.memory_bytes
        )


def quantity(value: str | float) -> Decimal:
    match = re.fullmatch(
        r"([+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))([eE][+-]?[0-9]+|[numkKMGTPE]|[KMGTPE]i)?",
        str(value),
    )
    if not match:
        raise ValueError("invalid Kubernetes resource quantity")
    number, suffix = match.groups()
    if suffix and suffix[0] in "eE" and len(suffix) > 1:
        return Decimal(number) * (Decimal(10) ** int(suffix[1:]))
    powers = {
        "n": -9,
        "u": -6,
        "m": -3,
        "k": 3,
        "K": 3,
        "M": 6,
        "G": 9,
        "T": 12,
        "P": 15,
        "E": 18,
    }
    multiplier = (
        Decimal(1024) ** ("KMGTPE".index(suffix[0]) + 1)
        if suffix and suffix.endswith("i")
        else Decimal(10) ** powers.get(suffix, 0)
    )
    return Decimal(number) * multiplier


def resources(values: dict) -> Resources:
    gpu = quantity(values.get("nvidia.com/gpu", 0))
    if gpu != gpu.to_integral_value():
        raise ValueError("fractional GPU request")
    return Resources(
        int(gpu),
        int(
            (quantity(values.get("cpu", 0)) * 1000).to_integral_value(
                rounding=ROUND_CEILING
            )
        ),
        int(
            quantity(values.get("memory", 0)).to_integral_value(rounding=ROUND_CEILING)
        ),
    )


def pod_requests(pod: dict) -> Resources:
    """Account for concurrent sidecars, sequential init stages and Pod overhead."""
    spec = pod["spec"]
    status = pod.get("status", {})
    statuses = {
        item["name"]: item
        for item in status.get("containerStatuses", [])
        + status.get("initContainerStatuses", [])
    }

    def container_request(container: dict) -> Resources:
        current = statuses.get(container["name"], {})
        return (
            resources(container.get("resources", {}).get("requests", {}))
            .maximum(resources(current.get("resources", {}).get("requests", {})))
            .maximum(resources(current.get("allocatedResources", {})))
        )

    apps = Resources()
    for container in spec.get("containers", []):
        apps += container_request(container)
    sidecars = peak_init = Resources()
    for container in spec.get("initContainers", []):
        request = container_request(container)
        if container.get("restartPolicy") == "Always":
            sidecars += request
            peak_init = peak_init.maximum(sidecars)
        else:
            peak_init = peak_init.maximum(sidecars + request)
    request = (apps + sidecars).maximum(peak_init)
    for claim in status.get("nodeAllocatableResourceClaimStatuses", []):
        request += resources(claim.get("resources", {}))
    pod_level = spec.get("resources", {}).get("requests", {})
    if pod_level:
        declared = (
            resources(pod_level)
            .maximum(resources(status.get("resources", {}).get("requests", {})))
            .maximum(resources(status.get("allocatedResources", {})))
        )
        request = Resources(
            request.gpus,
            declared.cpu_millis if "cpu" in pod_level else request.cpu_millis,
            declared.memory_bytes if "memory" in pod_level else request.memory_bytes,
        )
    return request + resources(spec.get("overhead", {}))


def verified_preemptible(pod: dict) -> bool:
    meta, spec = pod["metadata"], pod["spec"]
    names = [meta["name"], meta.get("labels", {}).get("job-name", "")]
    return (
        meta["namespace"] in {"cw-hpc-verification", "hpc-verification"}
        and spec.get("priorityClassName") == "cw-hpc-verification"
        and spec.get("priority") == -1
        and any(name.startswith("hpc-verification-") for name in names)
    )


def tolerates(taint: dict, tolerations: list[Toleration]) -> bool:
    return any(
        (not item.effect or item.effect == taint["effect"])
        and (item.key == taint["key"] or not item.key and item.operator == "Exists")
        and (item.operator == "Exists" or item.value == taint.get("value", ""))
        and not (taint["effect"] == "NoExecute" and item.tolerationSeconds == 0)
        for item in tolerations
    )


def summarize_capacity(
    nodes: list[dict],
    pods: list[dict],
    config: CapacityConfig,
    *,
    observed_at: datetime,
) -> CapacitySnapshot:
    physical, project = {}, {}
    bound_pending = unbound_pending = 0
    preemptible_nodes = set()
    preemptible_gpus = 0
    for pod in pods:
        if pod.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
            continue
        request = pod_requests(pod)
        node = pod["spec"].get("nodeName")
        if pod.get("status", {}).get("phase") == "Pending":
            if node:
                bound_pending += request.gpus
            else:
                unbound_pending += request.gpus
        if not node:
            continue
        physical[node] = physical.get(node, Resources()) + request
        if config.hpc_verification and verified_preemptible(pod):
            preemptible_nodes.add(node)
        else:
            project[node] = project.get(node, Resources()) + request
    required = resources(
        {
            "nvidia.com/gpu": config.gpus_per_node,
            "cpu": config.cpu_per_node,
            "memory": f"{config.memory_gib_per_node}Gi",
        }
    )
    counts = dict.fromkeys(CapacityCounts.model_fields, 0)
    free_total = Resources()
    for node in nodes:
        allocatable = resources(node["status"].get("allocatable", {}))
        if not allocatable.gpus:
            continue
        counts["gpu_nodes"] += 1
        ready = any(
            c["type"] == "Ready" and c["status"] == "True"
            for c in node["status"].get("conditions", [])
        )
        counts["ready_nodes"] += ready
        if not ready:
            counts["not_ready_nodes"] += 1
            continue
        if node.get("spec", {}).get("unschedulable"):
            counts["cordoned_ready_nodes"] += 1
            continue
        if any(
            node["metadata"].get("labels", {}).get(k) != v
            for k, v in config.node_selector.items()
        ):
            counts["selector_excluded_nodes"] += 1
            continue
        if any(
            t["effect"] in {"NoSchedule", "NoExecute"}
            and not tolerates(t, config.tolerations)
            for t in node.get("spec", {}).get("taints", [])
        ):
            counts["taint_excluded_nodes"] += 1
            continue
        counts["schedulable_candidate_nodes"] += 1
        name = node["metadata"]["name"]
        free = allocatable.remaining(project.get(name, Resources()))
        actual = allocatable.remaining(physical.get(name, Resources()))
        free_total += free
        counts["resource_fit_nodes"] += free.fits(required)
        counts["physically_available_fit_nodes"] += actual.fits(required)
        counts["full_gpu_nodes"] += free.gpus >= required.gpus
        counts["physically_idle_gpu_nodes"] += actual.gpus == allocatable.gpus
        counts["verified_preemptible_nodes"] += name in preemptible_nodes
        preemptible_gpus += free.gpus - actual.gpus
    counts.update(
        free_gpus=free_total.gpus,
        free_cpu=free_total.cpu_millis / 1000,
        free_memory_gib=free_total.memory_bytes / 1024**3,
        bound_pending_gpus=bound_pending,
        unbound_pending_gpus=unbound_pending,
        verified_preemptible_gpus=preemptible_gpus,
    )
    return CapacitySnapshot(
        observed_at=observed_at,
        status="available",
        requirements=WorkerShape(
            **config.model_dump(include=set(WorkerShape.model_fields))
        ),
        counts=CapacityCounts(**counts),
    )


@contextmanager
def _deadline(seconds: float):
    """Interrupt blocking API reads in the dedicated observer's main thread."""
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)

    def expired(_signal, _frame):
        raise TimeoutError("capacity observation deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    delay = min(seconds, previous_delay) if previous_delay else seconds
    signal.setitimer(signal.ITIMER_REAL, delay)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, previous_delay - (time.monotonic() - started)),
                previous_interval,
            )


def collect_capacity(
    client: KubernetesApiClient, config: CapacityConfig
) -> CapacitySnapshot:
    with _deadline(COLLECTION_SECONDS):
        return _collect_capacity(client, config)


def _collect_capacity(
    client: KubernetesApiClient, config: CapacityConfig
) -> CapacitySnapshot:
    started_at = datetime.now(UTC)
    deadline = time.monotonic() + COLLECTION_SECONDS
    remaining_bytes = 32 * 1024 * 1024

    def collection(path: str, selector: str = "") -> list[dict]:
        nonlocal remaining_bytes
        items, continuation = [], ""
        for _ in range(MAX_PAGES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("capacity collection deadline exceeded")
            query = urllib.parse.urlencode(
                {"limit": 250, "continue": continuation, "fieldSelector": selector}
            )
            page = client._request_json(
                "GET",
                f"{path}?{query}",
                timeout_seconds=min(5, remaining),
                max_response_bytes=8 * 1024 * 1024,
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("capacity collection deadline exceeded")
            remaining_bytes -= len(json.dumps(page).encode())
            if remaining_bytes < 0:
                raise ValueError("capacity collection byte limit exceeded")
            items.extend(page["items"])
            if len(items) > MAX_ITEMS:
                raise ValueError("capacity item limit exceeded")
            continuation = page.get("metadata", {}).get("continue", "")
            if not continuation:
                return items
        raise ValueError("capacity page limit exceeded")

    nodes = collection("/api/v1/nodes")
    pods = collection("/api/v1/pods", "status.phase!=Succeeded,status.phase!=Failed")
    return summarize_capacity(nodes, pods, config, observed_at=started_at)


def read_capacity_snapshot(
    path: Path | None, *, now: datetime | None = None
) -> tuple[CapacitySnapshot, float | None]:
    now = now or datetime.now(UTC)
    reason = "not_configured" if path is None else "unavailable"
    try:
        if path is None:
            return CapacitySnapshot(
                observed_at=now, status="unknown", reason=reason
            ), None
        with path.open("rb") as stream:
            payload = stream.read(MAX_SNAPSHOT_BYTES + 1)
        if len(payload) > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot exceeds limit")
        snapshot = CapacitySnapshot.model_validate_json(payload)
        if snapshot.observed_at.tzinfo is None or snapshot.limitations != LIMITATIONS:
            raise ValueError("invalid snapshot metadata")
        age = (now - snapshot.observed_at).total_seconds()
        if age < -5:
            raise ValueError("snapshot is in the future")
        if age > MAX_AGE_SECONDS:
            return CapacitySnapshot(
                observed_at=snapshot.observed_at, status="unknown", reason="stale"
            ), age
        if snapshot.status == "available" and (
            snapshot.counts is None
            or snapshot.requirements is None
            or snapshot.reason is not None
        ):
            raise ValueError("incomplete capacity snapshot")
        if snapshot.status == "unknown":
            snapshot = CapacitySnapshot(
                observed_at=snapshot.observed_at,
                status="unknown",
                reason=snapshot.reason or "unavailable",
            )
        return snapshot, max(0, age)
    except OSError:
        return CapacitySnapshot(observed_at=now, status="unknown", reason=reason), None
    except ValueError:
        return CapacitySnapshot(
            observed_at=now, status="unknown", reason="invalid"
        ), None


def publish_capacity(
    client: KubernetesApiClient, config: CapacityConfig, namespace: str, configmap: str
) -> None:
    """Replace one precreated snapshot; failures never refresh old successful data."""
    with _deadline(COLLECTION_SECONDS + 15):
        _publish_capacity(client, config, namespace, configmap)


def _publish_capacity(
    client: KubernetesApiClient, config: CapacityConfig, namespace: str, configmap: str
) -> None:
    try:
        snapshot = collect_capacity(client, config)
    except (
        OSError,
        ValueError,
        KeyError,
        ArithmeticError,
        KubernetesApiError,
    ) as error:
        print(f"Capacity collection unavailable: {type(error).__name__}", flush=True)
        snapshot = CapacitySnapshot(
            observed_at=datetime.now(UTC),
            status="unknown",
            reason="collection_failed",
        )
    path = f"/api/v1/namespaces/{urllib.parse.quote(namespace, safe='')}/configmaps/{urllib.parse.quote(configmap, safe='')}"
    current = client._request_json(
        "GET", path, timeout_seconds=5, max_response_bytes=MAX_SNAPSHOT_BYTES * 2
    )
    client._request_json(
        "PUT",
        path,
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": configmap,
                "namespace": namespace,
                "resourceVersion": current["metadata"]["resourceVersion"],
                "labels": current["metadata"].get("labels", {}),
            },
            "data": {"snapshot.json": snapshot.model_dump_json()},
        },
        timeout_seconds=5,
        max_response_bytes=MAX_SNAPSHOT_BYTES * 2,
    )


def main() -> None:
    if os.environ.get("SENPAI_IMAGE_REVISION") != os.environ.get(
        "SENPAI_REPO_REVISION"
    ):
        raise RuntimeError("capacity observer image and source revisions do not match")
    config = CapacityConfig.model_validate_json(os.environ["SENPAI_CAPACITY_CONFIG"])
    client = KubernetesApiClient()
    while True:
        try:
            publish_capacity(
                client,
                config,
                os.environ["SENPAI_CAPACITY_NAMESPACE"],
                os.environ["SENPAI_CAPACITY_CONFIGMAP"],
            )
        except (OSError, ValueError, KeyError, KubernetesApiError) as error:
            # Failed publication leaves the old timestamp to expire.
            print(
                f"Capacity publication unavailable: {type(error).__name__}", flush=True
            )
        time.sleep(30)


if __name__ == "__main__":
    main()
