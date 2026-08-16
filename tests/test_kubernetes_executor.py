from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from senpai_agent.kubernetes_executor import (
    KubernetesExecutor,
    _UnixServer,
    checkout_source_bundle,
)
from senpai_agent.kubernetes_training import KubernetesApiClient
from senpai_agent.training import KubernetesResourceRef, TrainingState


class FakeApi:
    def __init__(self):
        self.document_value: dict | None = None
        self.creates = 0
        self.submitted: list[dict] = []
        self.activated: list[KubernetesResourceRef] = []
        self.deleted: list[KubernetesResourceRef] = []
        self.state_value = (TrainingState.RUNNING, "active")

    def document(self, _spec):
        return deepcopy(self.document_value)

    def create(self, manifest, _namespace):
        self.creates += 1
        self.submitted.append(deepcopy(manifest))
        self.document_value = deepcopy(manifest)
        self.document_value["metadata"]["uid"] = "created-uid"
        return deepcopy(self.document_value)

    def activate(self, resource, _timeout_seconds=30):
        assert self.document_value is not None
        assert self.document_value["metadata"]["uid"] == resource.uid
        spec = self.document_value["spec"]
        spec.get("runPolicy", spec)["suspend"] = False
        self.activated.append(resource)

    def state(self, _resource):
        return self.state_value

    def delete(self, resource, _timeout_seconds=60):
        assert self.document_value is not None
        assert self.document_value["metadata"]["uid"] == resource.uid
        self.deleted.append(resource)
        self.document_value = None

    def logs(self, _resource):
        return "worker log"


def test_executor_server_keeps_serving_during_reconcile_outages(capsys):
    class UnavailableExecutor:
        def reconcile(self):
            raise TimeoutError("temporary API outage")

    server = object.__new__(_UnixServer)
    server.executor = UnavailableExecutor()

    server.service_actions()

    assert "reconciliation deferred" in capsys.readouterr().err


def executor(tmp_path: Path, client: FakeApi | None = None) -> KubernetesExecutor:
    return KubernetesExecutor(
        client=client or FakeApi(),
        state_path=tmp_path / "reservation.json",
        namespace="research",
        nodes=2,
        gpus_per_node=8,
        max_timeout_seconds=3600,
        cpu_per_gpu=15,
        memory_gi_per_gpu=110,
        pvc_claim_name="amf1-pvc",
        pvc_mount_path=tmp_path,
        snapshot_root=tmp_path / "snapshots",
        executor_image="executor@sha256:" + "a" * 64,
        launch_secret_name="senpai-launch-secrets-fred",
        research_tag="fred",
        student_name="fern",
        pod_name="senpai-fred-fern-123",
        pod_uid="student-pod-uid",
    )


def reserve(broker: KubernetesExecutor, commit: str = "a" * 40) -> dict:
    request = {
        "operation": "reserve",
        "training_id": "training-one",
        "spec": {
            "kind": "MPIJob",
            "name": "senpai-fred-fern-123",
            "namespace": "research",
            "wandb_run_id": "wandb-one",
        },
        "deadline_at": time.time() + 1800,
        "source_snapshot": str(broker.snapshot_root / f"{commit}.bundle"),
        "source_commit": commit,
    }
    broker.handle(request)
    return request


def manifest(commit: str = "a" * 40) -> dict:
    def pod(container: dict, *, worker: bool) -> dict:
        containers = [container]
        return {
            "metadata": {"labels": {"app": "amf1-cfd-train"}},
            "spec": {
                "restartPolicy": "Never",
                "initContainers": [
                    {
                        "name": "clone-repo",
                        "image": "alpine/git:2.49.1",
                        "env": [{"name": "SOURCE_SNAPSHOT_PATH", "value": "/snapshot"}],
                        "resources": {
                            "requests": {"cpu": "1", "memory": "2Gi"},
                            "limits": {"cpu": "1", "memory": "2Gi"},
                        },
                        "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}],
                    }
                ],
                "containers": containers,
                "volumes": [
                    {"name": "workspace", "emptyDir": {}},
                    {
                        "name": "dataset",
                        "persistentVolumeClaim": {"claimName": "amf1-pvc"},
                    },
                ],
                "tolerations": (
                    [{"key": "nvidia.com/gpu", "operator": "Exists"}]
                    if worker
                    else []
                ),
            },
        }

    launcher = {
        "name": "launcher",
        "image": "training:immutable",
        "resources": {
            "requests": {"cpu": "1", "memory": "2Gi"},
            "limits": {"cpu": "1", "memory": "2Gi"},
        },
    }
    worker = {
        "name": "train",
        "image": "training:immutable",
        "env": [
            {"name": "WANDB_API_KEY", "value": "must-be-replaced"},
            {"name": "NN_CFD_WANDB_RUN_ID", "value": "wandb-one"},
        ],
        "resources": {
            "requests": {"cpu": "120", "memory": "880Gi", "nvidia.com/gpu": "8"},
            "limits": {"cpu": "120", "memory": "880Gi", "nvidia.com/gpu": "8"},
        },
    }
    return {
        "apiVersion": "kubeflow.org/v2beta1",
        "kind": "MPIJob",
        "metadata": {
            "name": "senpai-fred-fern-123",
            "namespace": "research",
            "annotations": {
                "senpai.wandb.com/run-id": "wandb-one",
                "senpai.wandb.com/source-commit": commit,
            },
        },
        "spec": {
            "mpiImplementation": "OpenMPI",
            "launcherCreationPolicy": "AtStartup",
            "slotsPerWorker": 8,
            "runPolicy": {"backoffLimit": 4, "ttlSecondsAfterFinished": 60},
            "mpiReplicaSpecs": {
                "Launcher": {
                    "replicas": 1,
                    "restartPolicy": "Never",
                    "template": pod(launcher, worker=False),
                },
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "Never",
                    "template": pod(worker, worker=True),
                },
            },
        },
    }


def apply(broker: KubernetesExecutor, document: dict) -> str:
    import yaml

    return broker.handle({"operation": "apply", "manifest": yaml.safe_dump(document)})


def test_executor_injects_ownership_and_allows_exactly_one_2x8_workload(tmp_path):
    api = FakeApi()
    broker = executor(tmp_path, api)
    reserve(broker)
    document = manifest()
    document["spec"]["runPolicy"]["suspend"] = False

    assert apply(broker, document) == "mpijob/senpai-fred-fern-123 created\n"
    assert apply(broker, manifest()) == "mpijob/senpai-fred-fern-123 unchanged\n"
    assert api.creates == 1

    created = api.document_value
    assert created is not None
    assert api.submitted[0]["spec"]["runPolicy"]["suspend"] is True
    assert created["spec"]["runPolicy"]["suspend"] is False
    assert api.activated == [
        KubernetesResourceRef(
            kind="MPIJob",
            name="senpai-fred-fern-123",
            namespace="research",
            uid="created-uid",
            nodes=2,
            gpus_per_node=8,
        )
    ]
    assert created["spec"]["runPolicy"]["backoffLimit"] == 0
    assert created["spec"]["runPolicy"]["cleanPodPolicy"] == "Running"
    assert created["spec"]["runPolicy"]["activeDeadlineSeconds"] <= 1800
    assert created["spec"]["runPolicy"]["schedulingPolicy"] == {
        "minAvailable": 3,
        "scheduleTimeoutSeconds": created["spec"]["runPolicy"][
            "activeDeadlineSeconds"
        ],
    }
    assert created["metadata"]["ownerReferences"] == [
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "name": "senpai-fred-fern-123",
            "uid": "student-pod-uid",
        }
    ]
    for role in ("Launcher", "Worker"):
        pod_spec = created["spec"]["mpiReplicaSpecs"][role]["template"]["spec"]
        assert pod_spec["automountServiceAccountToken"] is False
        assert pod_spec["terminationGracePeriodSeconds"] == 30
        for container in [*pod_spec["initContainers"], *pod_spec["containers"]]:
            assert container["securityContext"]["allowPrivilegeEscalation"] is False
            assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}
            assert container["securityContext"]["seccompProfile"] == {
                "type": "RuntimeDefault"
            }
        checkout = pod_spec["initContainers"]
        assert len(checkout) == 1
        assert checkout[0]["name"] == "senpai-source-checkout"
        assert checkout[0]["image"] == "executor@sha256:" + "a" * 64
        assert checkout[0]["env"][1] == {
            "name": "SENPAI_SOURCE_COMMIT",
            "value": "a" * 40,
        }
        assert checkout[0]["volumeMounts"][0] == {
            "name": "dataset",
            "mountPath": "/var/lib/senpai-source/source.bundle",
            "subPath": f"snapshots/{'a' * 40}.bundle",
            "readOnly": True,
        }
        assert all(
            mount.get("name") == "senpai-workspace"
            for container in pod_spec["containers"]
            for mount in container["volumeMounts"]
            if mount.get("mountPath") == "/workspace"
        )
    wandb = created["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"][
        "containers"
    ][0]["env"][0]
    assert wandb == {
        "name": "WANDB_API_KEY",
        "valueFrom": {
            "secretKeyRef": {
                "name": "senpai-launch-secrets-fred",
                "key": "wandb-api-key",
            }
        },
    }


def test_source_checkout_reuses_rw_dataset_pvc_with_read_only_bundle_mount(tmp_path):
    api = FakeApi()
    broker = executor(tmp_path, api)
    reserve(broker)
    document = manifest()
    for role in ("Launcher", "Worker"):
        container = document["spec"]["mpiReplicaSpecs"][role]["template"]["spec"][
            "containers"
        ][0]
        container["volumeMounts"] = [
            {"name": "dataset", "mountPath": "/mnt/amf1-pvc"}
        ]

    apply(broker, document)

    assert api.document_value is not None
    for role in ("Launcher", "Worker"):
        pod_spec = api.document_value["spec"]["mpiReplicaSpecs"][role]["template"][
            "spec"
        ]
        pvc_volumes = [
            volume for volume in pod_spec["volumes"] if "persistentVolumeClaim" in volume
        ]
        assert pvc_volumes == [
            {
                "name": "dataset",
                "persistentVolumeClaim": {
                    "claimName": "amf1-pvc",
                    "readOnly": False,
                },
            }
        ]
        assert pod_spec["containers"][0]["volumeMounts"][0] == {
            "name": "dataset",
            "mountPath": "/mnt/amf1-pvc",
        }
        assert pod_spec["initContainers"][0]["volumeMounts"][0] == {
            "name": "dataset",
            "mountPath": "/var/lib/senpai-source/source.bundle",
            "subPath": f"snapshots/{'a' * 40}.bundle",
            "readOnly": True,
        }


def test_executor_rejects_an_explicitly_read_only_dataset_pvc(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    document["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]["spec"][
        "volumes"
    ][1]["persistentVolumeClaim"]["readOnly"] = True

    with pytest.raises(
        ValueError,
        match="dataset PVC must be read-write for status and checkpoints",
    ):
        apply(broker, document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"].__setitem__(
                "replicas", 1
            ),
            "replica topology",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ].__setitem__("serviceAccountName", "admin"),
            "workload identity",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ]["volumes"].append({"name": "host", "hostPath": {"path": "/"}}),
            "volumes",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ]["containers"][0].__setitem__(
                "envFrom", [{"secretRef": {"name": "github"}}]
            ),
            "envFrom",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ].__setitem__("resourceClaims", [{"name": "extra-device"}]),
            "workload identity",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ].__setitem__(
                "resources",
                {
                    "requests": {"cpu": "640", "memory": "2Ti"},
                    "limits": {"cpu": "640", "memory": "2Ti"},
                },
            ),
            "workload identity",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ]["containers"][0]["resources"].__setitem__(
                "claims", [{"name": "extra-device"}]
            ),
            "unsupported resources",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ]["containers"][0]["resources"]["limits"].__setitem__(
                "example.com/device", 1
            ),
            "unsupported resources",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ]["containers"].append(
                {"name": "unbounded-sidecar", "image": "sidecar:latest"}
            ),
            "needs CPU and memory",
        ),
        (
            lambda value: value["spec"]["mpiReplicaSpecs"]["Worker"]["template"][
                "spec"
            ]["containers"][0].__setitem__(
                "securityContext", {"seccompProfile": {"type": "Unconfined"}}
            ),
            "elevate privileges",
        ),
    ],
)
def test_executor_rejects_privileged_or_out_of_shape_manifests(
    tmp_path,
    mutation,
    message,
):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    mutation(document)

    with pytest.raises((ValueError, RuntimeError), match=message):
        apply(broker, document)


def test_executor_overwrites_training_pod_termination_grace(tmp_path):
    api = FakeApi()
    broker = executor(tmp_path, api)
    reserve(broker)
    document = manifest()
    for role in ("Launcher", "Worker"):
        document["spec"]["mpiReplicaSpecs"][role]["template"]["spec"][
            "terminationGracePeriodSeconds"
        ] = 86400

    apply(broker, document)

    assert api.document_value is not None
    for role in ("Launcher", "Worker"):
        assert (
            api.document_value["spec"]["mpiReplicaSpecs"][role]["template"][
                "spec"
            ]["terminationGracePeriodSeconds"]
            == 30
        )


def test_executor_overwrites_mpi_cleanup_policy(tmp_path):
    api = FakeApi()
    broker = executor(tmp_path, api)
    reserve(broker)
    document = manifest()
    document["spec"]["runPolicy"]["cleanPodPolicy"] = "None"

    apply(broker, document)

    assert api.document_value is not None
    assert api.document_value["spec"]["runPolicy"]["cleanPodPolicy"] == "Running"


@pytest.mark.parametrize("field", ["priorityClass", "queue", "minResources"])
def test_executor_rejects_unbounded_mpi_scheduling_controls(tmp_path, field):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    document["spec"]["runPolicy"]["schedulingPolicy"] = {field: "attacker-value"}

    with pytest.raises(ValueError, match="unsupported scheduling controls"):
        apply(broker, document)


def test_executor_requires_wandb_and_source_evidence_annotations(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    document["metadata"]["annotations"]["senpai.wandb.com/source-commit"] = "b" * 40

    with pytest.raises(ValueError, match="reserved evidence"):
        apply(broker, document)


def test_executor_rejects_workspace_shadowing(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    document["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"][
        "containers"
    ][0]["volumeMounts"] = [
        {"name": "dataset", "mountPath": "/workspace/replacement"}
    ]

    with pytest.raises(ValueError, match="shadow the Senpai workspace"):
        apply(broker, document)


def test_exact_commit_checkout_rejects_a_mutated_bundle(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    (repository / "source.py").write_text("exact = True\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=repository, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    bundle = tmp_path / f"{commit}.bundle"
    subprocess.run(["git", "bundle", "create", bundle, "HEAD"], cwd=repository, check=True)
    workspace = tmp_path / "exact-workspace"
    checkout_source_bundle(bundle, workspace, commit)
    assert (workspace / "source.py").read_text() == "exact = True\n"
    (workspace / "stale-init-state").write_text("partial")
    checkout_source_bundle(bundle, workspace, commit)
    assert not (workspace / "stale-init-state").exists()
    assert (workspace / "source.py").read_text() == "exact = True\n"

    data = bytearray(bundle.read_bytes())
    data[-1] ^= 1
    bundle.write_bytes(data)

    with pytest.raises(subprocess.CalledProcessError):
        checkout_source_bundle(bundle, tmp_path / "workspace", commit)


def test_executor_persists_uid_and_uses_it_for_delete(tmp_path):
    api = FakeApi()
    broker = executor(tmp_path, api)
    request = reserve(broker)
    apply(broker, manifest())
    resource = broker.handle(
        {
            "operation": "resource",
            "spec": request["spec"],
            "nodes": 2,
            "gpus_per_node": 8,
        }
    )

    broker.handle({"operation": "delete", "resource": resource, "timeout_seconds": 60})

    assert api.deleted == [KubernetesResourceRef.model_validate(resource)]
    broker.handle({"operation": "release", "training_id": "training-one"})
    assert json.loads((tmp_path / "reservation.json").read_text())["released"] is True


def test_executor_never_activates_after_the_deadline(tmp_path):
    class DeadlineExpiresDuringCreate(FakeApi):
        broker: KubernetesExecutor

        def create(self, manifest, namespace):
            created = super().create(manifest, namespace)
            self.broker._reservation["deadline_at"] = time.time() - 1
            return created

    api = DeadlineExpiresDuringCreate()
    broker = executor(tmp_path, api)
    api.broker = broker
    reserve(broker)

    with pytest.raises(TimeoutError, match="before Kubernetes activation"):
        apply(broker, manifest())

    assert api.activated == []
    assert api.deleted[0].uid == "created-uid"
    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["activated"] is False
    assert persisted["released"] is True


def test_executor_recovers_an_authorized_activation_after_restart(tmp_path):
    class FlakyActivation(FakeApi):
        def __init__(self):
            super().__init__()
            self.activation_attempts = 0

        def activate(self, resource, timeout_seconds=30):
            self.activation_attempts += 1
            if self.activation_attempts == 1:
                raise TimeoutError("temporary activation outage")
            super().activate(resource, timeout_seconds)

    api = FlakyActivation()
    broker = executor(tmp_path, api)
    reserve(broker)

    with pytest.raises(TimeoutError, match="temporary activation outage"):
        apply(broker, manifest())
    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["activation_authorized"] is True
    assert persisted["activated"] is False

    executor(tmp_path, api)

    assert api.activation_attempts == 2
    assert api.document_value["spec"]["runPolicy"]["suspend"] is False
    assert json.loads((tmp_path / "reservation.json").read_text())["activated"] is True


def test_executor_reaps_an_activation_request_that_reaches_the_deadline(tmp_path):
    class ActivationTimesOutAtDeadline(FakeApi):
        broker: KubernetesExecutor

        def activate(self, resource, timeout_seconds=30):
            self.broker._reservation["deadline_at"] = time.time() - 1
            raise TimeoutError("activation response was lost at the deadline")

    api = ActivationTimesOutAtDeadline()
    broker = executor(tmp_path, api)
    api.broker = broker
    reserve(broker)

    with pytest.raises(TimeoutError, match="during Kubernetes activation"):
        apply(broker, manifest())

    assert api.activated == []
    assert api.deleted[0].uid == "created-uid"
    assert json.loads((tmp_path / "reservation.json").read_text())["released"] is True


def test_executor_recovers_create_before_uid_persist_and_rejects_replacement(tmp_path):
    api = FakeApi()
    broker = executor(tmp_path, api)
    request = reserve(broker)
    apply(broker, manifest())
    state_path = tmp_path / "reservation.json"
    persisted = json.loads(state_path.read_text())
    persisted.update(created=False, resource=None)
    state_path.write_text(json.dumps(persisted))

    recovered = executor(tmp_path, api)
    resource = recovered.handle(
        {
            "operation": "resource_identity",
            "spec": request["spec"],
        }
    )
    assert resource["uid"] == "created-uid"

    assert api.document_value is not None
    api.document_value["metadata"]["uid"] = "replacement-uid"
    with pytest.raises(PermissionError, match="not owned"):
        recovered.handle({"operation": "state", "resource": resource})


def test_executor_will_not_release_a_live_workload(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    apply(broker, manifest())

    with pytest.raises(RuntimeError, match="live Kubernetes workload"):
        broker.handle({"operation": "release", "training_id": "training-one"})


def test_executor_recovers_a_create_response_failure_before_release(tmp_path):
    class BrokenCreateResponse(FakeApi):
        def create(self, manifest, namespace):
            response = super().create(manifest, namespace)
            response["metadata"].pop("uid")
            return response

    api = BrokenCreateResponse()
    broker = executor(tmp_path, api)
    reserve(broker)

    with pytest.raises(KeyError, match="uid"):
        apply(broker, manifest())
    assert api.document_value is not None
    with pytest.raises(RuntimeError, match="live Kubernetes workload"):
        broker.handle({"operation": "release", "training_id": "training-one"})
    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["resource"]["uid"] == "created-uid"
    assert persisted["released"] is False


def test_executor_retains_an_unresolved_create_until_it_becomes_visible(tmp_path):
    class DelayedCreate(FakeApi):
        def __init__(self):
            super().__init__()
            self.pending: dict | None = None

        def create(self, manifest, namespace):
            self.pending = deepcopy(manifest)
            self.pending["metadata"]["uid"] = "delayed-uid"
            raise TimeoutError("create response was lost")

    api = DelayedCreate()
    broker = executor(tmp_path, api)
    reserve(broker)

    with pytest.raises(TimeoutError, match="response was lost"):
        apply(broker, manifest())
    with pytest.raises(RuntimeError, match="unresolved Kubernetes create"):
        broker.handle({"operation": "release", "training_id": "training-one"})

    broker._reservation["deadline_at"] = time.time() - 1
    broker._write_state()
    broker.reconcile()
    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["resource"] is None
    assert persisted["released"] is False

    api.document_value = api.pending
    broker.reconcile()
    assert api.deleted == [
        KubernetesResourceRef(
            kind="MPIJob",
            name="senpai-fred-fern-123",
            namespace="research",
            uid="delayed-uid",
            nodes=2,
            gpus_per_node=8,
        )
    ]
    assert json.loads((tmp_path / "reservation.json").read_text())["released"] is True


def test_executor_retries_a_create_that_never_reached_the_api(tmp_path):
    class LostBeforeApi(FakeApi):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def create(self, manifest, namespace):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("create request never reached the API")
            return super().create(manifest, namespace)

    api = LostBeforeApi()
    broker = executor(tmp_path, api)
    reserve(broker)

    with pytest.raises(TimeoutError, match="never reached"):
        apply(broker, manifest())

    broker._reservation["deadline_at"] = time.time() - 1
    broker._write_state()
    executor(tmp_path, api)

    assert api.attempts == 2
    assert api.deleted == [
        KubernetesResourceRef(
            kind="MPIJob",
            name="senpai-fred-fern-123",
            namespace="research",
            uid="created-uid",
            nodes=2,
            gpus_per_node=8,
        )
    ]
    assert json.loads((tmp_path / "reservation.json").read_text())["released"] is True


def test_late_initial_create_stays_suspended_after_retry_release(tmp_path):
    class ParkedInitialCreate(FakeApi):
        def __init__(self):
            super().__init__()
            self.parked: dict | None = None
            self.attempts = 0

        def create(self, manifest, namespace):
            self.attempts += 1
            if self.attempts == 1:
                self.parked = deepcopy(manifest)
                raise TimeoutError("initial create is still in flight")
            return super().create(manifest, namespace)

        def complete_initial_create(self):
            assert self.parked is not None
            self.document_value = self.parked
            self.document_value["metadata"]["uid"] = "late-initial-uid"

    api = ParkedInitialCreate()
    broker = executor(tmp_path, api)
    reserve(broker)

    with pytest.raises(TimeoutError, match="still in flight"):
        apply(broker, manifest())

    broker._reservation["deadline_at"] = time.time() - 1
    broker._write_state()
    broker.reconcile()
    assert api.deleted[0].uid == "created-uid"
    assert api.activated == []
    assert json.loads((tmp_path / "reservation.json").read_text())["released"] is True

    api.complete_initial_create()

    assert api.document_value["spec"]["runPolicy"]["suspend"] is True
    assert api.activated == []


def test_api_delete_uses_a_uid_precondition_and_foreground_propagation(
    tmp_path,
    monkeypatch,
):
    token = tmp_path / "token"
    token.write_text("rotated-token")
    client = object.__new__(KubernetesApiClient)
    client.api_server = "https://kubernetes.example"
    client.token_path = token
    client.ssl_context = None
    requests = []
    gets = 0

    class Response:
        def __init__(self, value):
            self.value = value

        def read(self):
            return self.value

    def urlopen(request, **_kwargs):
        nonlocal gets
        requests.append(request)
        if request.get_method() == "DELETE":
            return Response(b"{}")
        gets += 1
        if gets == 1:
            return Response(
                json.dumps(
                    {
                        "kind": "MPIJob",
                        "metadata": {"name": "job", "namespace": "research", "uid": "uid-1"},
                    }
                ).encode()
            )
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    resource = KubernetesResourceRef(
        kind="MPIJob",
        name="job",
        namespace="research",
        uid="uid-1",
        nodes=2,
        gpus_per_node=8,
    )

    client.delete(resource, timeout_seconds=2)

    delete = next(request for request in requests if request.get_method() == "DELETE")
    body = json.loads(delete.data)
    assert body["preconditions"] == {"uid": "uid-1"}
    assert body["propagationPolicy"] == "Foreground"
    assert delete.headers["Authorization"] == "Bearer rotated-token"


@pytest.mark.parametrize(
    ("kind", "suspend_path"),
    [("Job", "/spec/suspend"), ("MPIJob", "/spec/runPolicy/suspend")],
)
def test_api_activation_is_uid_bound(kind, suspend_path):
    client = object.__new__(KubernetesApiClient)
    calls = []

    def request(method, path, body, **kwargs):
        calls.append((method, path, body, kwargs))
        spec = (
            {"suspend": False}
            if kind == "Job"
            else {"runPolicy": {"suspend": False}}
        )
        return {"metadata": {"uid": "uid-one"}, "spec": spec}

    client._request_json = request
    client.activate(
        KubernetesResourceRef(
            kind=kind,
            name="training-one",
            namespace="research",
            uid="uid-one",
            nodes=2,
            gpus_per_node=8,
        )
    )

    method, path, body, kwargs = calls[0]
    assert method == "PATCH"
    assert path.endswith("/namespaces/research/" + kind.lower() + "s/training-one")
    assert body == [
        {"op": "test", "path": "/metadata/uid", "value": "uid-one"},
        {"op": "replace", "path": suspend_path, "value": False},
    ]
    assert kwargs["content_type"] == "application/json-patch+json"
