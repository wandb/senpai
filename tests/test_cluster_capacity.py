"""Protect advisory accounting and the credential-free observation boundary."""

import json
import urllib.parse
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from openhands.sdk.tool import resolve_tool
from openhands_support import runtime_config

from senpai_agent.capacity_tool import ClusterCapacityAction
from senpai_agent.cluster_capacity import (
    MAX_AGE_SECONDS,
    SNAPSHOT_ENV,
    CapacityConfig,
    Resources,
    pod_requests,
    publish_capacity,
    read_capacity_snapshot,
    summarize_capacity,
)
from senpai_agent.openhands_runner import build_main_tools

NOW = datetime(2026, 9, 25, 23, tzinfo=UTC)
CONFIG = CapacityConfig(
    nodes=4, gpus_per_node=8, cpu_per_node=120, memory_gib_per_node=880
)


def container(name, cpu, memory, gpu=0, **extra):
    return {
        "name": name,
        "resources": {
            "requests": {"cpu": str(cpu), "memory": memory, "nvidia.com/gpu": gpu}
        },
        **extra,
    }


def node(name, **spec):
    return {
        "metadata": {"name": name, "labels": {"pool": "gpu"}},
        "spec": spec,
        "status": {
            "allocatable": {"cpu": "128", "memory": "1024Gi", "nvidia.com/gpu": "8"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def pod(name, *, node_name="", phase="Running", cpu=0, memory="0", gpu=0, **spec):
    return {
        "metadata": {"name": name, "namespace": "default"},
        "spec": {
            "nodeName": node_name,
            "containers": [container("train", cpu, memory, gpu)],
            **spec,
        },
        "status": {"phase": phase},
    }


def test_resource_fit_accounts_for_cpu_only_pending_and_verified_preemption():
    """GPU totals cannot stand in for same-node CPU/memory fit or physical vacancy."""
    nodes = [
        node(name)
        for name in [
            "idle",
            "cpu-busy",
            "memory-busy",
            "reserved",
            "verification",
            "lookalike",
        ]
    ]
    pods = [
        pod("cpu-only", node_name="cpu-busy", cpu=9),
        pod("memory-only", node_name="memory-busy", memory="145Gi"),
        pod("reserved", node_name="reserved", phase="Pending", gpu=8),
        pod("waiting", phase="Pending", gpu=8),
        pod(
            "hpc-verification-real",
            node_name="verification",
            gpu=8,
            cpu=120,
            memory="880Gi",
            priorityClassName="cw-hpc-verification",
            priority=-1,
        ),
        pod(
            "hpc-verification-fake",
            node_name="lookalike",
            gpu=8,
            priorityClassName="cw-hpc-verification",
            priority=-1,
        ),
        pod("complete", node_name="idle", phase="Succeeded", gpu=8),
    ]
    pods[4]["metadata"]["namespace"] = "cw-hpc-verification"
    enabled = summarize_capacity(
        nodes,
        pods,
        CONFIG.model_copy(update={"hpc_verification": True}),
        observed_at=NOW,
    )
    disabled = summarize_capacity(nodes, pods, CONFIG, observed_at=NOW)
    assert enabled.counts.resource_fit_nodes == 2  # idle and verified-preemptible
    assert enabled.counts.physically_available_fit_nodes == 1
    assert enabled.counts.full_gpu_nodes == 4
    assert enabled.counts.bound_pending_gpus == 8
    assert enabled.counts.unbound_pending_gpus == 8
    assert enabled.counts.verified_preemptible_gpus == 8
    assert disabled.counts.resource_fit_nodes == 1
    assert disabled.counts.verified_preemptible_gpus == 0
    assert "cpu-only" not in enabled.model_dump_json()
    assert "verification" not in enabled.model_dump_json()  # no project/node names


def test_effective_pod_requests_handle_ordered_sidecars_pod_overrides_and_resize():
    """Kubernetes v1.36 accounting; maxima can come from different init phases."""
    workload = pod("mixed", cpu=2, memory="1Gi")
    workload["spec"].update(
        initContainers=[
            container("sidecar-one", 1, "2Gi", restartPolicy="Always"),
            container("init-one", 6, "1Gi"),
            container("sidecar-two", 2, "4Gi", restartPolicy="Always"),
            container("init-two", 3, "3Gi"),
        ],
        overhead={"cpu": "250m", "memory": "128Mi"},
    )
    assert pod_requests(workload) == Resources(0, 7250, 9 * 1024**3 + 128 * 1024**2)
    workload["spec"]["resources"] = {"requests": {"cpu": "8"}}
    assert pod_requests(workload) == Resources(0, 8250, 9 * 1024**3 + 128 * 1024**2)
    resized = pod("resizing", cpu=2, memory="1Gi")
    resized["status"]["containerStatuses"] = [
        {
            "name": "train",
            "resources": {"requests": {"cpu": "8"}},
            "allocatedResources": {"cpu": "6"},
        }
    ]
    resized["status"]["nodeAllocatableResourceClaimStatuses"] = [
        {"resources": {"nvidia.com/gpu": "1"}}
    ]
    assert pod_requests(resized) == Resources(1, 8000, 1024**3)


def test_ready_cordon_selector_and_toleration_filters_apply_before_fit():
    nodes = [
        node("ready"),
        node("cordoned", unschedulable=True),
        node("not-ready"),
        node("other-pool"),
        node("gpu-taint", taints=[{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]),
        node("hard-taint", taints=[{"key": "unavailable", "effect": "NoExecute"}]),
        node("soft-taint", taints=[{"key": "soft", "effect": "PreferNoSchedule"}]),
    ]
    nodes[2]["status"]["conditions"][0]["status"] = "False"
    nodes[3]["metadata"]["labels"]["pool"] = "cpu"
    config = CapacityConfig(
        **CONFIG.model_dump(exclude={"node_selector", "tolerations"}),
        node_selector={"pool": "gpu"},
        tolerations=[
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ],
    )
    counts = summarize_capacity(nodes, [], config, observed_at=NOW).counts
    assert counts.resource_fit_nodes == 3
    assert (
        counts.not_ready_nodes,
        counts.cordoned_ready_nodes,
        counts.selector_excluded_nodes,
        counts.taint_excluded_nodes,
    ) == (1, 1, 1, 1)


class ObservationApi:
    """A paginated API fixture, including an unrelated CPU-only allocation."""

    def __init__(self, fail_second_page=False):
        self.fail_second_page = fail_second_page
        self.requests = []
        self.published = None

    def _request_json(self, method, path, body=None, **limits):
        self.requests.append((method, path, limits))
        parsed = urllib.parse.urlsplit(path)
        if method == "PUT":
            self.published = body
            return body
        if parsed.path.endswith("/configmaps/capacity"):
            return {
                "metadata": {"resourceVersion": "17", "labels": {"owned": "observer"}}
            }
        if parsed.path == "/api/v1/nodes":
            return {"metadata": {}, "items": [node("worker")]}
        continuation = urllib.parse.parse_qs(parsed.query).get("continue")
        if not continuation:
            return {"metadata": {"continue": "next-page"}, "items": []}
        if self.fail_second_page:
            raise TimeoutError("private backend failure must not be published")
        return {
            "metadata": {},
            "items": [pod("private-cpu-pod", node_name="worker", cpu=9)],
        }


def test_observer_paginates_all_pods_and_publishes_only_sanitized_evidence():
    api = ObservationApi()
    publish_capacity(api, CONFIG, "default", "capacity")
    snapshot = json.loads(api.published["data"]["snapshot.json"])
    assert snapshot["counts"]["resource_fit_nodes"] == 0
    assert snapshot["status"] == "available"
    assert api.published["metadata"]["resourceVersion"] == "17"
    assert all(method in {"GET", "PUT"} for method, _, _ in api.requests)
    assert [path for method, path, _ in api.requests if method == "PUT"] == [
        "/api/v1/namespaces/default/configmaps/capacity"
    ]
    assert all(limits["max_response_bytes"] > 0 for _, _, limits in api.requests)
    assert "private-cpu-pod" not in json.dumps(api.published)
    assert "continue=next-page" in api.requests[2][1]


def test_failed_collection_replaces_old_evidence_with_unknown():
    api = ObservationApi(fail_second_page=True)
    publish_capacity(api, CONFIG, "default", "capacity")
    snapshot = json.loads(api.published["data"]["snapshot.json"])
    assert snapshot["status"] == "unknown"
    assert snapshot["reason"] == "collection_failed"
    assert snapshot["counts"] is None
    assert "private backend" not in json.dumps(api.published)


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("stale", "stale"),
        ("missing", "unavailable"),
        ("extra", "invalid"),
        ("future", "invalid"),
        ("oversized", "invalid"),
    ],
)
def test_unusable_snapshots_never_report_capacity(tmp_path, scenario, reason):
    snapshot = summarize_capacity([node("worker")], [], CONFIG, observed_at=NOW)
    path = tmp_path / "snapshot.json"
    content = snapshot.model_dump(mode="json")
    if scenario == "stale":
        content["observed_at"] = (
            NOW - timedelta(seconds=MAX_AGE_SECONDS + 1)
        ).isoformat()
    elif scenario == "extra":
        content["pod_environment"] = "do-not-echo"
    elif scenario == "future":
        content["observed_at"] = (NOW + timedelta(seconds=30)).isoformat()
    if scenario != "missing":
        path.write_text(
            "x" * (64 * 1024 + 1) if scenario == "oversized" else json.dumps(content)
        )
    result, _ = read_capacity_snapshot(path, now=NOW)
    assert result.status == "unknown" and result.reason == reason
    assert result.counts is None
    assert "do-not-echo" not in result.model_dump_json()


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_root_role_invokes_capacity_without_cluster_credentials(
    tmp_path, monkeypatch, role
):
    path = tmp_path / "snapshot.json"
    path.write_text(
        summarize_capacity(
            [node("worker")], [], CONFIG, observed_at=datetime.now(UTC)
        ).model_dump_json()
    )
    monkeypatch.setenv(SNAPSHOT_ENV, str(path))
    tools = build_main_tools(runtime_config(tmp_path, role=role))
    spec = next(tool for tool in tools if tool.name == "get_cluster_capacity")
    tool = resolve_tool(spec, SimpleNamespace())[0]
    result = tool(ClusterCapacityAction())
    assert result.snapshot.counts.resource_fit_nodes == 1
    assert result.age_seconds < 10
    assert "scheduler remains authoritative" in result.to_llm_content[0].text


def test_capacity_http_byte_limit_rejects_oversized_responses(tmp_path, monkeypatch):
    """The observer's byte budget is enforced before decoding a backend payload."""
    import io
    import urllib.request

    from senpai_agent.kubernetes_training import KubernetesApiClient

    token = tmp_path / "token"
    token.write_text("test-only-token")
    client = object.__new__(KubernetesApiClient)
    client.api_server = "https://kubernetes.example"
    client.token_path = token
    client.ssl_context = None
    response = io.BytesIO(b'{"items":[]}' + b" " * 100)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="byte limit"):
        client._request_json("GET", "/api/v1/pods", max_response_bytes=16)
    assert response.closed


@pytest.mark.parametrize(
    "toleration",
    [
        {"operator": "Exists", "value": "invalid"},
        {"operator": "Equal"},
        {
            "key": "gpu",
            "operator": "Exists",
            "effect": "NoSchedule",
            "tolerationSeconds": 30,
        },
    ],
)
def test_invalid_placement_policy_cannot_claim_resource_fit(toleration):
    with pytest.raises(ValueError):
        CapacityConfig(
            **CONFIG.model_dump(exclude={"tolerations"}), tolerations=[toleration]
        )


def test_blocked_final_page_times_out_and_publishes_unknown(monkeypatch):
    """A response trickling bytes cannot keep an old observation looking current."""
    import signal
    import time

    import senpai_agent.cluster_capacity as capacity

    class SlowFinalPage(ObservationApi):
        def _request_json(self, method, path, body=None, **limits):
            if "continue=next-page" in path:
                time.sleep(1)
            return super()._request_json(method, path, body, **limits)

    api = SlowFinalPage()
    monkeypatch.setattr(capacity, "COLLECTION_SECONDS", 0.02)
    previous_handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    publish_capacity(api, CONFIG, "default", "capacity")
    assert time.monotonic() - started < 0.5
    snapshot = json.loads(api.published["data"]["snapshot.json"])
    assert snapshot["status"] == "unknown"
    assert snapshot["counts"] is None
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_blocked_publication_times_out_and_restores_process_timer(monkeypatch):
    """A blocked ConfigMap write cannot prevent later collection attempts."""
    import signal
    import time

    import senpai_agent.cluster_capacity as capacity

    class SlowPublication(ObservationApi):
        def _request_json(self, method, path, body=None, **limits):
            if method == "PUT":
                time.sleep(30)
            return super()._request_json(method, path, body, **limits)

    api = SlowPublication()
    monkeypatch.setattr(capacity, "COLLECTION_SECONDS", 0.02)
    previous_handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="observation deadline"):
        publish_capacity(api, CONFIG, "default", "capacity")
    assert time.monotonic() - started < 20
    assert api.published is None
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
