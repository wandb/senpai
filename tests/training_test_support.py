from pathlib import Path
import subprocess
import threading

from senpai_agent.training import KubernetesResourceRef, KubernetesTrainingSpec, TrainingState


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
    ):
        with self._lock:
            self.spec = spec
            self.resource_value = KubernetesResourceRef(
                kind=spec.kind,
                name=spec.name,
                namespace=spec.namespace,
                uid="remote-uid",
                nodes=self.nodes,
                gpus_per_node=8,
            )
            self.reservations.append((training_id, source_snapshot, source_commit))

    def apply(self, manifest):
        pass

    def adopt(self, _training_id, _spec, resource, _deadline_at):
        self.adoptions.append(resource)

    def resource(self, _spec, *, nodes, gpus_per_node):
        assert (nodes, gpus_per_node) == (self.nodes, 8)
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
