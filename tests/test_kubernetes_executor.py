from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest

from senpai_agent.kubernetes_executor import (
    KubernetesExecutor,
    _UnixServer,
    checkout_source_bundle,
)
from senpai_agent.kubernetes_training import KubernetesApiClient, KubernetesApiError
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
        writer_secret_name="senpai-wandb-student-fred-fern-writer",
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
        "volumeMounts": [{"name": "dataset", "mountPath": "/mnt/amf1-pvc"}],
        "env": [
            {"name": "WANDB_API_KEY", "value": "must-be-replaced"},
            {"name": "TARGET_RUN_LABEL", "value": "wandb-one"},
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


def test_slow_logs_do_not_block_executor_status_or_cancellation(tmp_path):
    logs_started = threading.Event()
    finish_logs = threading.Event()
    control_finished = threading.Event()
    results = []

    class SlowLogsApi(FakeApi):
        def logs(self, resource):
            logs_started.set()
            assert finish_logs.wait(5)
            return "worker log"

    api = SlowLogsApi()
    broker = executor(tmp_path, api)
    reservation = reserve(broker)
    apply(broker, manifest())
    resource = broker.handle({
        "operation": "resource_identity", "spec": reservation["spec"],
    })

    def control():
        results.append(broker.handle({"operation": "state", "resource": resource}))
        broker.handle({"operation": "delete", "resource": resource, "timeout_seconds": 5})
        control_finished.set()

    log_thread = threading.Thread(
        target=lambda: broker.handle({"operation": "logs", "resource": resource}),
    )
    control_thread = threading.Thread(target=control)
    log_thread.start()
    try:
        assert logs_started.wait(2)
        control_thread.start()
        assert control_finished.wait(2), "logs held the executor control lock"
        assert results == [["running", "active"]]
        assert len(api.deleted) == 1
    finally:
        finish_logs.set()
        log_thread.join(2)
        if control_thread.ident is not None:
            control_thread.join(2)


@pytest.mark.parametrize("wandb_key_role", ["Worker", "Launcher"])
def test_executor_injects_ownership_and_allows_exactly_one_2x8_workload(
    tmp_path, wandb_key_role,
):
    api = FakeApi()
    broker = executor(tmp_path, api)
    reserve(broker)
    document = manifest()
    document["spec"]["runPolicy"]["suspend"] = False
    for role in ("Launcher", "Worker"):
        document["spec"]["mpiReplicaSpecs"][role]["template"]["metadata"][
            "labels"
        ].update({
            "research-tag": "foreign-track",
            "student": "foreign-student",
            "senpai-training-id": "foreign-training",
            "senpai-training-role": "foreign-role",
        })
    worker = document["spec"]["mpiReplicaSpecs"]["Worker"]["template"]
    if wandb_key_role == "Launcher":
        key = worker["spec"]["containers"][0]["env"].pop(0)
        launcher = document["spec"]["mpiReplicaSpecs"]["Launcher"]["template"]
        launcher["spec"]["containers"][0].setdefault("env", []).append(key)
    worker["metadata"]["labels"]["example.org/training-group"] = "experiment-one"
    worker["spec"]["affinity"] = {
        "podAntiAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [{
                "labelSelector": {
                    "matchLabels": {"example.org/training-group": "experiment-one"},
                },
                "topologyKey": "kubernetes.io/hostname",
            }],
        },
    }

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
        template = created["spec"]["mpiReplicaSpecs"][role]["template"]
        assert {
            key: template["metadata"]["labels"][key]
            for key in (
                "research-tag", "student", "senpai-training-id", "senpai-training-role",
            )
        } == {
            "research-tag": "fred",
            "student": "fern",
            "senpai-training-id": "training-one",
            "senpai-training-role": role.lower(),
        }
        pod_spec = template["spec"]
        if role == "Launcher":
            assert "affinity" not in pod_spec
        assert pod_spec["automountServiceAccountToken"] is False
        assert pod_spec["terminationGracePeriodSeconds"] == 30
        for container in pod_spec["containers"]:
            assert [item for item in container["env"] if item["name"] == "WANDB_RUN_ID"] == [
                {"name": "WANDB_RUN_ID", "value": "wandb-one"}
            ]
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
    worker = created["spec"]["mpiReplicaSpecs"]["Worker"]["template"]
    assert worker["metadata"]["labels"]["example.org/training-group"] == "experiment-one"
    assert worker["spec"]["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ] == [{
        "labelSelector": {
            "matchLabels": {"example.org/training-group": "experiment-one"},
        },
        "topologyKey": "kubernetes.io/hostname",
    }, {
        "labelSelector": {"matchLabels": {
            "senpai-training-id": "training-one",
            "senpai-training-role": "worker",
        }},
        "topologyKey": "kubernetes.io/hostname",
    }]
    worker_env = worker["spec"]["containers"][0]["env"]
    assert {"name": "TARGET_RUN_LABEL", "value": "wandb-one"} in worker_env
    key_env = created["spec"]["mpiReplicaSpecs"][wandb_key_role]["template"][
        "spec"
    ]["containers"][0]["env"]
    assert next(item for item in key_env if item["name"] == "WANDB_API_KEY") == {
        "name": "WANDB_API_KEY",
        "valueFrom": {
            "secretKeyRef": {
                "name": "senpai-wandb-student-fred-fern-writer",
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


def test_executor_rejects_a_gpu_container_without_the_dataset_pvc_mount(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    worker = document["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]
    worker["containers"][0]["volumeMounts"] = []

    with pytest.raises(
        ValueError,
        match="every GPU training container must mount the dataset PVC",
    ):
        apply(broker, document)


def test_executor_rejects_a_read_only_gpu_dataset_mount(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    worker = document["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]
    worker["containers"][0]["volumeMounts"][0]["readOnly"] = True

    with pytest.raises(
        ValueError,
        match="GPU training containers must mount the dataset PVC read-write",
    ):
        apply(broker, document)


def test_executor_allows_a_read_only_alias_with_a_writable_dataset_mount(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    worker = document["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]
    worker["containers"][0]["volumeMounts"].append(
        {
            "name": "dataset",
            "mountPath": "/mnt/dataset-read-only",
            "readOnly": True,
        }
    )

    assert "created" in apply(broker, document)


def test_executor_rejects_a_dataset_mount_replaced_by_the_workspace(tmp_path):
    broker = executor(tmp_path)
    reserve(broker)
    document = manifest()
    worker = document["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]
    worker["containers"][0]["volumeMounts"][0]["mountPath"] = "/workspace"

    with pytest.raises(
        ValueError,
        match="every GPU training container must mount the dataset PVC",
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
            ]["containers"][0]["env"].append(
                {"name": "WANDB_RUN_ID", "value": "foreign-run"}
            ),
            "W&B run ID does not match reservation",
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
    ] + [
        (
            lambda value, name=name: value["spec"]["mpiReplicaSpecs"]["Worker"][
                "template"
            ]["spec"]["containers"][0]["env"].append({"name": name, "value": "override"}),
            name,
        )
        for name in (
            "WANDB_SERVICE",
            "WANDB_IDENTITY_TOKEN_FILE",
            "WANDB_INFERENCE_API_KEY",
            "SENPAI_WANDB_TRAINING_API_KEY",
        )
    ],
)
def test_executor_rejects_privileged_or_out_of_shape_manifests(
    tmp_path,
    mutation,
    message,
):
    api = FakeApi()
    broker = executor(tmp_path, api)
    reserve(broker)
    document = manifest()
    mutation(document)

    with pytest.raises((ValueError, RuntimeError), match=message):
        apply(broker, document)
    assert api.creates == 0


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


@pytest.mark.parametrize("status_code", [408, 422, 425, 429])
def test_executor_releases_a_direct_create_rejection(tmp_path, status_code):
    class RejectedCreate(FakeApi):
        def create(self, _manifest, _namespace):
            raise KubernetesApiError(
                "POST",
                "/apis/kubeflow.org/v2beta1/mpijobs",
                status_code,
            )

    broker = executor(tmp_path, RejectedCreate())
    reserve(broker)

    with pytest.raises(KubernetesApiError, match=f"HTTP {status_code}"):
        apply(broker, manifest())

    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["create_attempted"] is True
    assert persisted["released"] is True
    assert persisted["manifest"] is None

    reserve(broker, "b" * 40)
    replacement = json.loads((tmp_path / "reservation.json").read_text())
    assert replacement["source_commit"] == "b" * 40
    assert replacement["released"] is False


@pytest.mark.parametrize("retry_operation", ["reconcile", "apply"])
def test_executor_retains_a_rejected_retry_after_an_ambiguous_create(
    tmp_path,
    retry_operation,
):
    class RejectedRetry(FakeApi):
        def __init__(self):
            super().__init__()
            self.pending: dict | None = None

        def create(self, manifest, _namespace):
            self.creates += 1
            if self.creates == 1:
                self.pending = deepcopy(manifest)
                raise TimeoutError("create response was lost")
            raise KubernetesApiError("POST", "/apis/kubeflow.org/v2beta1/mpijobs", 422)

        def complete_initial_create(self):
            assert self.pending is not None
            self.document_value = self.pending
            self.document_value["metadata"]["uid"] = "late-initial-uid"

    api = RejectedRetry()
    broker = executor(tmp_path, api)
    reserve(broker)

    with pytest.raises(TimeoutError, match="response was lost"):
        apply(broker, manifest())
    assert json.loads((tmp_path / "reservation.json").read_text())["released"] is False

    if retry_operation == "reconcile":
        broker.reconcile()
    else:
        with pytest.raises(KubernetesApiError, match="HTTP 422"):
            apply(broker, manifest())

    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["create_attempted"] is True
    assert persisted["released"] is False
    assert persisted["manifest"] is not None

    api.complete_initial_create()
    broker._reservation["deadline_at"] = time.time() - 1
    broker._write_state()
    broker.reconcile()

    assert api.deleted == [
        KubernetesResourceRef(
            kind="MPIJob",
            name="senpai-fred-fern-123",
            namespace="research",
            uid="late-initial-uid",
            nodes=2,
            gpus_per_node=8,
        )
    ]
    assert json.loads((tmp_path / "reservation.json").read_text())["released"] is True


@pytest.mark.parametrize(
    "status_code",
    [409, 499, 503],
    ids=["conflict-may-be-an-existing-create", "client-closed", "server-error"],
)
def test_executor_retains_an_ambiguous_http_create_response(
    tmp_path,
    status_code,
):
    class UnavailableApi(FakeApi):
        def create(self, _manifest, _namespace):
            raise KubernetesApiError(
                "POST", "/apis/kubeflow.org/v2beta1/mpijobs", status_code
            )

    broker = executor(tmp_path, UnavailableApi())
    reserve(broker)

    with pytest.raises(KubernetesApiError, match=f"HTTP {status_code}"):
        apply(broker, manifest())

    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["create_attempted"] is True
    assert persisted["released"] is False
    assert persisted["manifest"] is not None


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
    persisted = json.loads((tmp_path / "reservation.json").read_text())
    assert persisted["create_attempted"] is True
    assert persisted["released"] is False
    assert persisted["manifest"] is not None
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


def test_api_error_preserves_the_http_status(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("rotated-token")
    client = object.__new__(KubernetesApiClient)
    client.api_server = "https://kubernetes.example"
    client.token_path = token
    client.ssl_context = None

    def reject(request, **_kwargs):
        raise urllib.error.HTTPError(request.full_url, 422, "invalid", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", reject)

    with pytest.raises(KubernetesApiError) as error:
        client.create({"kind": "Job"}, "research")

    assert error.value.status_code == 422


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


def test_workload_diagnostics_include_owned_init_and_launcher_logs(monkeypatch):
    client = object.__new__(KubernetesApiClient)
    resource = KubernetesResourceRef(
        kind="MPIJob", name="run", namespace="research", uid="mpi-uid",
        nodes=2, gpus_per_node=8,
    )
    workload = {"metadata": {"uid": "mpi-uid", "labels": {"senpai-training-id": "run-id"}}}
    launcher_job = {"metadata": {"uid": "job-uid", "ownerReferences": [
        {"kind": "MPIJob", "uid": "mpi-uid"},
    ]}}

    def pod(name, owner, status, *, init=False):
        return {
            "metadata": {"name": name, "uid": name + "-uid", "ownerReferences": [owner]},
            "spec": {"containers": [{"name": "train"}],
                     "initContainers": [{"name": "checkout"}] if init else []},
            "status": status,
        }

    worker = pod("worker-0", {"kind": "MPIJob", "uid": "mpi-uid"}, {
        "phase": "Pending",
        "conditions": [{"type": "PodScheduled", "status": "False", "message": "Insufficient GPUs"}],
        "initContainerStatuses": [{"name": "checkout", "state": {
            "terminated": {"reason": "Error", "exitCode": 1},
        }}],
        "containerStatuses": [{"name": "train", "state": {
            "waiting": {"reason": "PodInitializing"},
        }}],
    }, init=True)
    launcher = pod("launcher", {"kind": "Job", "name": "launcher-job", "uid": "job-uid"}, {
        "phase": "Running", "containerStatuses": [{"name": "train", "state": {"running": {}}}],
    })
    unrelated = pod("not-owned", {"kind": "MPIJob", "uid": "other-uid"}, {
        "phase": "Running", "containerStatuses": [{"name": "train", "state": {"running": {}}}],
    })
    replaced = pod("replaced-job-pod", {"kind": "Job", "name": "launcher-job", "uid": "old-job-uid"}, {
        "phase": "Running", "containerStatuses": [{"name": "train", "state": {"running": {}}}],
    })
    requests = []

    def get(kind, name, namespace):
        assert namespace == "research"
        return workload if kind == "MPIJob" else launcher_job

    def request_json(method, path, **kwargs):
        if "/events?" in path:
            assert "involvedObject.uid%3D" in path
            return {"items": []}
        assert "senpai-training-id%3Drun-id" in path
        return {"items": [unrelated, replaced, worker, launcher]}

    def request_text(method, path, **kwargs):
        requests.append(path)
        assert "limitBytes=8192" in path
        return "permission denied" if "container=checkout" in path else "rank 0 started"

    monkeypatch.setattr(client, "_get", get)
    monkeypatch.setattr(client, "_request_json", request_json)
    monkeypatch.setattr(client, "_request_text", request_text)

    result = client.logs(resource)

    assert "Insufficient GPUs" in result
    assert "checkout] terminated: Error exit=1" in result
    assert "checkout] permission denied" in result
    assert "train] waiting: PodInitializing" in result
    assert "launcher/train] rank 0 started" in result
    assert len(requests) == 2
    assert "not-owned" not in result
    assert "replaced-job-pod" not in result


def test_workload_diagnostics_reject_replaced_uid(monkeypatch):
    client = object.__new__(KubernetesApiClient)
    monkeypatch.setattr(client, "_get", lambda *args: {"metadata": {"uid": "replacement"}})
    resource = KubernetesResourceRef(
        kind="MPIJob", name="run", namespace="research", uid="original",
        nodes=2, gpus_per_node=8,
    )
    with pytest.raises(RuntimeError, match="replaced training workload"):
        client.logs(resource)


def test_diagnostics_preserve_failed_init_with_noisy_healthy_worker(monkeypatch):
    client = object.__new__(KubernetesApiClient)
    resource = KubernetesResourceRef(
        kind="MPIJob", name="run", namespace="research", uid="mpi-uid",
        nodes=2, gpus_per_node=8,
    )
    failed = {
        "metadata": {"name": "run-launcher", "uid": "launcher-uid"},
        "spec": {
            "initContainers": [{"name": "checkout"}],
            "containers": [{"name": "train"}],
        },
        "status": {
            "phase": "Pending",
            "initContainerStatuses": [{"name": "checkout", "state": {
                "terminated": {"reason": "Error", "exitCode": 1},
            }}],
            "containerStatuses": [{"name": "train", "state": {
                "waiting": {"reason": "PodInitializing"},
            }}],
        },
    }
    healthy = {
        "metadata": {"name": "run-worker-1", "uid": "worker-uid"},
        "spec": {"containers": [{"name": "train"}]},
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "train", "state": {"running": {}}}],
        },
    }
    pods = [healthy, failed]
    monkeypatch.setattr(client, "_owned_pods", lambda _resource: pods)
    monkeypatch.setattr(client, "_events", lambda *args: ["[event] " + "x" * 2000])

    def request_text(method, path, **kwargs):
        if "container=checkout" in path:
            return "checkout progress\n" * 400 + "Permission denied opening source.bundle"
        return "正常な学習出力\n" * 300 + "healthy worker tail"

    monkeypatch.setattr(client, "_request_text", request_text)

    result = client.logs(resource)

    assert len(result.encode()) <= 8192
    assert "checkout] terminated: Error exit=1" in result
    assert "Permission denied opening source.bundle" in result
    assert "[pod/run-worker-1/train] " in result
    assert "healthy worker tail" in result
    assert result.index("terminated: Error") < result.index("Permission denied")
    assert result.rindex("[pod/run-launcher/checkout] ") < result.rindex(
        "[pod/run-worker-1/train] "
    )
    assert "\ufffd" not in result
    pods.reverse()
    assert client.logs(resource) == result


def test_diagnostics_show_remaining_gpu_requests_and_every_container(monkeypatch):
    client = object.__new__(KubernetesApiClient)
    resource = KubernetesResourceRef(
        kind="MPIJob", name="run", namespace="research", uid="mpi-uid",
        nodes=4, gpus_per_node=8,
    )
    pods = []
    for index, phase in enumerate(("Running", "Succeeded", "Failed", "Pending")):
        state = (
            {"running": {}} if index == 0
            else {"waiting": {"reason": "Pending", "message": "long problem " * 2000}} if index == 3
            else {"terminated": {"reason": "Completed" if index == 1 else "Error", "exitCode": index - 1}}
        )
        pods.append({
            "metadata": {"name": f"run-worker-{index}", "uid": str(index)},
            "spec": {
                **({"nodeName": f"node-{index}"} if index != 3 else {}),
                "initContainers": [{"name": "checkout"}],
                "containers": [{"name": "train", "resources": {"limits": {"nvidia.com/gpu": 8}}}],
            },
            "status": {
                "phase": phase,
                "initContainerStatuses": [{"name": "checkout", "state": {"terminated": {"reason": "Completed", "exitCode": 0}}}],
                "containerStatuses": [{"name": "train", "state": state}],
            },
        })
    monkeypatch.setattr(client, "_owned_pods", lambda _resource: pods)
    monkeypatch.setattr(client, "_events", lambda *args: [])
    requests = []

    def request_text(method, path, **kwargs):
        requests.append(path)
        if "run-worker-0/log?container=train" in path:
            return "cleanup entered\n" + "noise\n" * 1200 + "still waiting for shutdown"
        if "run-worker-2/log?container=train" in path:
            return "FIRST FAILURE: connection reset\n" + "noise\n" * 1200 + "exit barrier failed"
        return "正常な学習出力\n" * 600

    monkeypatch.setattr(client, "_request_text", request_text)
    result = client.logs(resource)

    assert len(result.encode()) <= 8192
    for index, phase in enumerate(("Running", "Succeeded", "Failed", "Pending")):
        assert f"[pod/run-worker-{index}] phase={phase}" in result
        assert f"[pod/run-worker-{index}/checkout] terminated: Completed exit=0" in result
    assert "[pod/run-worker-0/train] running" in result
    assert "[pod/run-worker-1/train] terminated: Completed exit=0" in result
    assert "[pod/run-worker-2/train] terminated: Error exit=1" in result
    assert "[pod/run-worker-3/train] waiting: Pending" in result
    assert "requested_gpus=32 scheduled_gpu_requests=8 pending_gpu_requests=8" in result
    assert "FIRST FAILURE: connection reset" in result
    assert "exit barrier failed" in result
    assert "cleanup entered" in result
    assert "still waiting for shutdown" in result
    assert "\ufffd" not in result
    assert "run-worker-2/log?container=train" in requests[0]
    assert "run-worker-0/log?container=train" in requests[1]


def test_diagnostics_report_pre_pod_validation_events_and_reject_foreign_uid(monkeypatch):
    client = object.__new__(KubernetesApiClient)
    resource = KubernetesResourceRef(
        kind="MPIJob", name="run", namespace="research", uid="mpi-uid",
        nodes=4, gpus_per_node=8,
    )
    workload = {"metadata": {"uid": "mpi-uid", "labels": {"senpai-training-id": "run-id"}}}
    monkeypatch.setattr(client, "_get", lambda *args: workload)
    requests = []

    def request_json(method, path, **kwargs):
        requests.append(path)
        if "/pods?" in path:
            return {"items": []}
        assert "/namespaces/research/events?" in path
        assert "fieldSelector=involvedObject.uid%3Dmpi-uid" in path
        assert "limit=20" in path
        return {"items": [
            {"involvedObject": {"uid": "other-uid"}, "message": "foreign secret"},
            {
                "metadata": {"creationTimestamp": "2026-09-24T08:50:00Z"},
                "involvedObject": {"uid": "mpi-uid"},
                "type": "Warning", "reason": "ValidationError",
                "message": "worker hostname must be no more than 63 characters",
            },
        ]}

    monkeypatch.setattr(client, "_request_json", request_json)
    result = client.logs(resource)

    assert "ValidationError: worker hostname must be no more than 63 characters" in result
    assert "No owned workload pods exist yet" in result
    assert "foreign secret" not in result
    assert len(requests) == 2


def test_workload_events_are_bounded_and_surface_rbac_errors(monkeypatch):
    client = object.__new__(KubernetesApiClient)
    events = [{
        "metadata": {"creationTimestamp": f"2026-09-24T08:50:{index:02d}Z"},
        "involvedObject": {"uid": "pod-uid"},
        "reason": f"Reason{index}", "message": "x" * 5000,
    } for index in range(8)]
    monkeypatch.setattr(client, "_request_json", lambda *args, **kwargs: {"items": events})

    result = client._events("research", "pod-uid", "pod/worker", time.monotonic() + 10)

    assert len(result) == 5
    assert "Reason3" in result[0]
    assert "Reason7" in result[-1]
    assert all(line.count("x") == 1024 for line in result)

    def forbidden(*args, **kwargs):
        raise KubernetesApiError("GET", "/events", 403)

    monkeypatch.setattr(client, "_request_json", forbidden)
    assert client._events("research", "pod-uid", "pod/worker", time.monotonic() + 10) == [
        "[pod/worker/events] unavailable: HTTP 403"
    ]
