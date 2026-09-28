"""Shared training inputs, results, and durable workload identities."""

import os
import uuid
from collections.abc import Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TARGET_PYTHON_ENV = "SENPAI_TARGET_PYTHON_ENV"


def target_python_environment(
    environment: Mapping[str, str] = os.environ,
) -> dict[str, str]:
    """Point interpreter, PATH, and uv project commands at the target venv."""

    target_env = environment.get(TARGET_PYTHON_ENV, "").strip()
    if not target_env:
        return {}
    return {
        "PATH": f"{target_env}/bin:{environment['PATH']}",
        "UV_PROJECT_ENVIRONMENT": target_env,
        "UV_PYTHON": f"{target_env}/bin/python",
        "VIRTUAL_ENV": target_env,
    }


class TrainingState(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class KubernetesTrainingSpec(BaseModel):
    """Identity of the Kubernetes workload created by ``argv``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["Job", "MPIJob"]
    name: str = Field(pattern=r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")
    namespace: str = Field(
        default="default",
        pattern=r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$",
    )
    wandb_run_id: str = Field(min_length=1)


class KubernetesResourceRef(BaseModel):
    """Durable identity and verified shape of one remote training workload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["Job", "MPIJob"]
    name: str
    namespace: str
    uid: str
    nodes: int = Field(gt=0)
    gpus_per_node: int = Field(gt=0)


class TrainingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[str, ...] = Field(
        min_length=1, description="Command argument list, executed once per worker node.",
    )
    cwd: Path = Field(description="Working directory inside the committed assignment workspace.")
    timeout_seconds: int | None = Field(
        default=None, gt=0,
        description="Optional wall-clock deadline including queue and setup time. Omit for no deadline.",
    )
    nodes: int | None = Field(
        default=None, gt=0, description="Worker nodes for this run; defaults to the full configured node allocation.",
    )
    gpus_per_node: int | None = Field(
        default=None, gt=0, description="GPUs per worker node; defaults to the configured GPUs per node.",
    )


class TrainingCapacityError(RuntimeError):
    """A launch exceeds the student's shared GPU allocation."""

    def __init__(
        self,
        *,
        requested_gpus: int,
        available_gpus: int,
        capacity_gpus: int,
        active_runs: list[dict],
    ):
        self.requested_gpus = requested_gpus
        self.available_gpus = available_gpus
        self.capacity_gpus = capacity_gpus
        self.active_runs = active_runs
        identifiers = ", ".join(run["training_id"] for run in active_runs) or "none"
        super().__init__(
            f"Training needs {requested_gpus} GPUs; {available_gpus} of "
            f"{capacity_gpus} GPUs are available. Active training IDs: {identifiers}. "
            "Request fewer GPUs, wait for cleanup, or cancel a run by ID."
        )


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_id: str
    conversation_id: uuid.UUID | None = None
    state: TrainingState
    pid: int | None = Field(default=None, gt=0)
    process_group_id: int | None = Field(default=None, gt=0)
    process_start_time: float | None = Field(default=None, gt=0)
    exit_code: int | None = None
    elapsed_seconds: float
    log_path: str
    output_dir: str | None = None
    nodes: int | None = Field(default=None, gt=0)
    gpus_per_node: int | None = Field(default=None, gt=0)
    worker_logs_pruned: bool = False
    wandb_run_ids: tuple[str, ...] = ()
    error_tail: str = ""
    started_at: float | None = Field(default=None, gt=0)
    deadline_at: float | None = Field(default=None, gt=0)
    kubernetes_spec: KubernetesTrainingSpec | None = None
    kubernetes_resource: KubernetesResourceRef | None = None
    kubernetes_released: bool | None = None
    kubernetes_diagnostics: str = ""
    kubernetes_pod_receipt: dict | None = None
    source_snapshot: str | None = None
    source_commit: str | None = None


def training_result_paths(state_dir: Path) -> Iterator[Path]:
    """Yield only run records owned by the training supervisor."""

    for path in state_dir.glob("*.json"):
        try:
            training_id = uuid.UUID(path.stem)
        except ValueError:
            continue
        if path.name == f"{training_id}.json":
            yield path
