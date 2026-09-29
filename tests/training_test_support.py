from copy import deepcopy
from pathlib import Path
import subprocess
import threading
import time

from senpai_agent.training import KubernetesResourceRef, KubernetesTrainingSpec, TrainingState


class WorkloadApi:
    """Independent Kubernetes objects behind the real reservation broker."""

    def __init__(self):
        self.documents = {}
        self.deletions = []

    def document(self, spec):
        return deepcopy(self.documents.get(spec.name))

    def create(self, manifest, namespace):
        document = deepcopy(manifest)
        name = document["metadata"]["name"]
        assert name not in self.documents
        document["metadata"].update(uid=f"uid-{name}", namespace=namespace)
        self.documents[name] = document
        return deepcopy(document)

    def activate(self, resource, timeout_seconds):
        document = self.documents[resource.name]
        assert document["metadata"]["uid"] == resource.uid
        document["spec"].get("runPolicy", document["spec"])["suspend"] = False

    def state(self, resource):
        assert self.documents[resource.name]["metadata"]["uid"] == resource.uid
        return TrainingState.RUNNING, "active"

    def delete(self, resource, timeout_seconds):
        assert self.documents[resource.name]["metadata"]["uid"] == resource.uid
        del self.documents[resource.name]
        self.deletions.append(resource)

    def logs(self, resource):
        return {"statuses": [], "problems": [], "events": [], "logs": []}

    def pod_snapshot(self, resource):
        return {"complete": False, "capture_error": "fixture API has no Pods", "pods": []}


class FakeCluster:
    def __init__(self, state: TrainingState = TrainingState.FINISHED, *, nodes: int = 2):
        self.nodes = nodes
        self.state_value = state
        self.spec: KubernetesTrainingSpec | None = None
        self.resource_value: KubernetesResourceRef | None = None
        self.reservations: list[tuple[str, str, str]] = []
        self.adoptions: list[KubernetesResourceRef] = []
        self.deletions: list[KubernetesResourceRef] = []
        self.releases: list[str] = []
        self._lock = threading.Lock()

    def reserve(
        self,
        training_id,
        spec,
        _deadline_at,
        source_snapshot,
        source_commit,
        *,
        nodes,
        gpus_per_node,
    ):
        with self._lock:
            self.spec = spec
            self.resource_value = KubernetesResourceRef(
                kind=spec.kind,
                name=spec.name,
                namespace=spec.namespace,
                uid="remote-uid",
                nodes=nodes,
                gpus_per_node=gpus_per_node,
            )
            self.reservations.append((training_id, source_snapshot, source_commit))

    def apply(self, manifest):
        pass

    def adopt(self, _training_id, _spec, resource, _deadline_at, *, nodes, gpus_per_node):
        assert (resource.nodes, resource.gpus_per_node) == (nodes, gpus_per_node)
        self.adoptions.append(resource)

    def resource(self, _spec, *, nodes, gpus_per_node):
        if self.resource_value is not None:
            assert (nodes, gpus_per_node) == (
                self.resource_value.nodes, self.resource_value.gpus_per_node,
            )
        return self.resource_value

    def resource_identity(self, _spec):
        return self.resource_value

    def state(self, _resource):
        return self.state_value, self.state_value.value

    def delete(self, resource, timeout_seconds=60):
        assert timeout_seconds <= 60
        if self.resource_value is not None and self.resource_value.uid != resource.uid:
            raise RuntimeError("refusing to delete replaced workload")
        self.deletions.append(resource)
        self.resource_value = None

    def logs(self, _resource):
        return "remote worker log"

    def pod_snapshot(self, resource):
        return {
            "training_id": self.reservations[-1][0],
            "source_commit": self.reservations[-1][2],
            "resource": resource.model_dump(mode="json"),
            "captured_at": time.time(), "pods": [], "terminal_complete": False,
            "capture_error": None,
        }

    def release(self, training_id):
        self.releases.append(training_id)


def git_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=workspace,
        check=True,
    )
    (workspace / ".gitignore").write_text(".env\n")
    (workspace / "tracked.txt").write_text("committed\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=workspace, check=True)
    (workspace / ".env").write_text("WANDB_API_KEY=secret\n")
    return workspace
