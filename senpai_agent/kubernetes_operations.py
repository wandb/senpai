"""Kubernetes transport for one campaign's operational supervisor."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from senpai_agent.operational_supervisor import (
    EvidenceGap,
    MachineStats,
    RoleRuntimeObservation,
)
from senpai_agent.operations import (
    CampaignInventory,
    ContextResetReceipt,
    ContextResetRequest,
    NudgeReceipt,
    OperationBackend,
    RestartReceipt,
    RoleObservation,
    RoleTarget,
)
from senpai_agent.role_control import (
    RoleControlRequest,
    RoleResearchTail,
    RoleRuntimeState,
)
from senpai_agent.repair_broker import RepairResult


_LABEL_VALUE = re.compile(r"^[A-Za-z0-9](?:[-_.A-Za-z0-9]{0,61}[A-Za-z0-9])?$")
_ERROR_MARKER = re.compile(
    r"\bSENPAI_[A-Z0-9_]*(?:ERROR|EXCEPTION|FAILED|RESTART|DEFERRED)\b|"
    r"\bOPENHANDS_TIMEOUT\b|Traceback \(most recent call last\):"
)
_LOG_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT\S+")
_LOG_TAIL_LINES = 400
_LOG_WINDOW_MARGIN_SECONDS = 120
_DEFAULT_SUPERVISOR_INTERVAL_SECONDS = 900
_DEFAULT_SUPERVISOR_TURN_TIMEOUT_SECONDS = 900


class KubernetesOperationError(RuntimeError):
    """A campaign-scoped kubectl observation or operation failed."""


class KubectlCampaignBackend(OperationBackend):
    """Bind typed operations to exact campaign-labelled role pods."""

    def __init__(
        self,
        inventory: CampaignInventory,
        *,
        namespace: str,
        kubectl: str = "kubectl",
        environment: Mapping[str, str] = os.environ,
        command_timeout_seconds: float = 45,
    ):
        if not namespace or command_timeout_seconds <= 0:
            raise ValueError("namespace and command timeout are required")
        self.inventory = inventory
        self.namespace = namespace
        self.kubectl = kubectl
        self.environment = {
            key: value
            for key, value in environment.items()
            if not _secret_name(key)
        }
        self.command_timeout_seconds = command_timeout_seconds
        interval = _positive_seconds(
            environment,
            "SENPAI_SUPERVISOR_INTERVAL_SECONDS",
            _DEFAULT_SUPERVISOR_INTERVAL_SECONDS,
        )
        turn_timeout = _positive_seconds(
            environment,
            "SENPAI_OPENHANDS_TIMEOUT_SECONDS",
            _DEFAULT_SUPERVISOR_TURN_TIMEOUT_SECONDS,
        )
        # A six-hour wake can run both an operational turn and a research turn
        # before runtime collection resumes. Cover that full worst-case gap,
        # the ordinary cadence, and a small scheduling/transport margin.
        self.log_since_seconds = math.ceil(
            interval + (2 * turn_timeout) + _LOG_WINDOW_MARGIN_SECONDS
        )
        for value in (inventory.research_tag, *inventory.students):
            if not _LABEL_VALUE.fullmatch(value):
                raise ValueError(f"campaign identity is not a Kubernetes label: {value}")

    def collect_role(self, target: RoleTarget) -> RoleObservation:
        return self._observe(target).observation

    def collect_advisor_research_tail(self) -> RoleResearchTail:
        target = RoleTarget(
            research_tag=self.inventory.research_tag,
            role="advisor",
        )
        payload = RoleControlRequest(command="research_tail")
        return RoleResearchTail.model_validate(self._role_control(target, payload))

    def nudge(
        self,
        target: RoleTarget,
        *,
        operation_key: str,
        expected_conversation_id: UUID,
        message: str,
        control_token: str,
    ) -> NudgeReceipt:
        payload = RoleControlRequest(
            command="nudge",
            expected_conversation_id=expected_conversation_id,
            control_token=control_token,
            message=message,
            operation_key=operation_key,
        )
        return NudgeReceipt.model_validate(self._role_control(target, payload))

    def restart_controller(
        self,
        target: RoleTarget,
        *,
        expected_conversation_id: UUID,
        restart_control_token: str,
    ) -> RestartReceipt:
        payload = RoleControlRequest(
            command="restart",
            expected_conversation_id=expected_conversation_id,
            restart_control_token=restart_control_token,
        )
        return RestartReceipt.model_validate(self._role_control(target, payload))

    def request_context_reset(
        self,
        target: RoleTarget,
        *,
        request: ContextResetRequest,
    ) -> ContextResetReceipt:
        payload = RoleControlRequest(
            command="context_reset",
            context_reset_request=request,
        )
        return ContextResetReceipt.model_validate(self._role_control(target, payload))

    def run_repair(
        self,
        target: RoleTarget,
        *,
        command: str,
        cwd: str,
        timeout_seconds: int,
    ) -> RepairResult:
        """Execute arbitrary shell only in the target's secret-free repair sidecar."""

        self.inventory.require(target)
        roots = {
            "workspace": "/repair/workspace",
            "state": "/repair/state",
            "scratch": "/repair/scratch",
        }
        try:
            root = roots[cwd]
        except KeyError as error:
            raise ValueError(f"unsupported repair working directory: {cwd}") from error
        pod = self._pod(target)
        transport = (
            self.kubectl,
            "exec",
            "-i",
            "-n",
            self.namespace,
            pod,
            "-c",
            "repair",
            "--",
            "/usr/local/bin/senpai-repair-executor",
            "--cwd",
            root,
            "--timeout",
            str(timeout_seconds),
        )
        output = self._run(
            transport,
            input_text=command,
            timeout_seconds=timeout_seconds + 15,
        )
        try:
            return RepairResult.model_validate_json(output)
        except ValueError as error:
            raise KubernetesOperationError(
                "repair sidecar returned invalid JSON"
            ) from error

    def collect_runtimes(
        self,
    ) -> tuple[tuple[RoleRuntimeObservation, ...], tuple[EvidenceGap, ...]]:
        targets = (
            RoleTarget(research_tag=self.inventory.research_tag, role="advisor"),
            *(
                RoleTarget(
                    research_tag=self.inventory.research_tag,
                    role="student",
                    student=student,
                )
                for student in self.inventory.students
            ),
        )

        def collect_one(
            target: RoleTarget,
        ) -> tuple[RoleRuntimeObservation, tuple[EvidenceGap, ...]]:
            local_gaps: list[EvidenceGap] = []
            try:
                pod = self._pod(target)
                state = self._observe(target, pod=pod)
                if state.target != target or state.observation.target != target:
                    raise KubernetesOperationError(
                        "role control returned a mismatched campaign target"
                    )
                if (
                    state.active_delegation_count
                    != state.observation.active_delegation_count
                ):
                    raise KubernetesOperationError(
                        "role control returned inconsistent delegation activity"
                    )
                try:
                    log_errors, log_inventory_complete = self._recent_log_errors(
                        target,
                        pod,
                    )
                    if not log_inventory_complete:
                        local_gaps.append(
                            EvidenceGap(
                                source="runtime",
                                subject=target.key,
                                detail=(
                                    "Pod logs reached the bounded "
                                    f"{_LOG_TAIL_LINES}-line "
                                    "tail; earlier lines in the requested window are "
                                    "unknown."
                                ),
                            )
                        )
                except KubernetesOperationError as error:
                    log_errors = ()
                    local_gaps.append(self._gap(target, "pod logs", error))
                alive = state.observation.controller_alive
                deadline = state.lease_deadline_seconds
                healthy = (
                    alive and deadline is not None and deadline > 0
                    if alive is not None
                    else None
                )
                observation = RoleRuntimeObservation(
                    role=target.role,
                    name=target.student or "advisor",
                    machine=pod,
                    controller_healthy=healthy,
                    lease_phase=state.observation.controller_phase,
                    lease_deadline_seconds=deadline,
                    completed_turns=state.completed_turns,
                    running_training_count=state.running_training_count,
                    active_delegation_count=state.active_delegation_count,
                    wandb_run_inventory_complete=(
                        state.wandb_run_inventory_complete
                    ),
                    running_wandb_run_ids=state.running_wandb_run_ids,
                    recent_wandb_run_ids=state.recent_wandb_run_ids,
                    context_resets=state.context_resets,
                    controller_restarts=state.controller_restarts,
                    stats=MachineStats(
                        cpu_percent=state.cpu_percent,
                        memory_percent=state.memory_percent,
                        disk_percent=state.disk_percent,
                        gpu_percent=state.gpu_percent,
                    ),
                    recent_errors=tuple((*state.recent_errors, *log_errors)[-20:]),
                )
                return observation, tuple(local_gaps)
            except Exception as error:  # noqa: BLE001
                return (
                    RoleRuntimeObservation(
                        role=target.role,
                        name=target.student or "advisor",
                        machine="unavailable",
                    ),
                    (self._gap(target, "role observation", error),),
                )

        with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
            results = tuple(pool.map(collect_one, targets))
        observations = tuple(result[0] for result in results)
        gaps = tuple(gap for _, local in results for gap in local)
        return tuple(observations), tuple(gaps)

    def _observe(
        self,
        target: RoleTarget,
        *,
        pod: str | None = None,
    ) -> RoleRuntimeState:
        payload = RoleControlRequest(command="observe")
        return RoleRuntimeState.model_validate(
            self._role_control(target, payload, pod=pod)
        )

    def _role_control(
        self,
        target: RoleTarget,
        request: RoleControlRequest,
        *,
        pod: str | None = None,
        timeout_seconds: float | None = None,
    ) -> object:
        self.inventory.require(target)
        pod = pod or self._pod(target)
        command = (
            self.kubectl,
            "exec",
            "-i",
            "-n",
            self.namespace,
            pod,
            "-c",
            target.role,
            "--",
            "python",
            "-m",
            "senpai_agent.role_control",
        )
        output = self._run(
            command,
            input_text=request.model_dump_json(),
            timeout_seconds=timeout_seconds,
        )
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise KubernetesOperationError(
                "role control returned invalid JSON"
            ) from error

    def _pod(self, target: RoleTarget) -> str:
        self.inventory.require(target)
        labels = [
            "app=senpai",
            f"role={target.role}",
            f"research-tag={target.research_tag}",
        ]
        if target.student is not None:
            labels.append(f"student={target.student}")
        output = self._run(
            (
                self.kubectl,
                "get",
                "pods",
                "-n",
                self.namespace,
                "-l",
                ",".join(labels),
                "-o",
                "json",
            )
        )
        try:
            items = json.loads(output)["items"]
            running = [
                item
                for item in items
                if item.get("metadata", {}).get("deletionTimestamp") is None
                and item.get("status", {}).get("phase") == "Running"
            ]
            names = [item["metadata"]["name"] for item in running]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise KubernetesOperationError("kubectl returned invalid pod JSON") from error
        if len(names) != 1:
            raise KubernetesOperationError(
                f"expected one running {target.role} pod, observed {len(names)}"
            )
        return str(names[0])

    def _recent_log_errors(
        self,
        target: RoleTarget,
        pod: str,
    ) -> tuple[tuple[str, ...], bool]:
        output = self._run(
            (
                self.kubectl,
                "logs",
                "-n",
                self.namespace,
                pod,
                "-c",
                target.role,
                "--timestamps=true",
                f"--since={self.log_since_seconds}s",
                f"--tail={_LOG_TAIL_LINES}",
            )
        )
        lines = output.splitlines()
        errors = tuple(
            self._structured_log_error(line)
            for line in lines
            if _ERROR_MARKER.search(line)
        )[-20:]
        return errors, len(lines) < _LOG_TAIL_LINES

    @staticmethod
    def _structured_log_error(line: str) -> str:
        match = _ERROR_MARKER.search(line)
        marker = match.group(0) if match is not None else "UNKNOWN_ERROR"
        timestamp = _LOG_TIMESTAMP.match(line)
        observed = timestamp.group(0) if timestamp is not None else "unknown"
        fingerprint = hashlib.sha256(line.strip().encode()).hexdigest()[:16]
        return f"{marker} observed={observed} fingerprint={fingerprint}"

    def _run(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        try:
            result = subprocess.run(
                tuple(command),
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout_seconds or self.command_timeout_seconds,
                env=self.environment,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise KubernetesOperationError(
                f"kubectl transport failed ({type(error).__name__})"
            ) from error
        if result.returncode:
            raise KubernetesOperationError(
                f"kubectl returned exit code {result.returncode}"
            )
        return result.stdout.strip()

    @staticmethod
    def _gap(
        target: RoleTarget,
        operation: str,
        error: BaseException,
    ) -> EvidenceGap:
        return EvidenceGap(
            source="runtime",
            subject=target.key,
            detail=f"{operation} failed ({type(error).__name__}).",
        )


def _secret_name(name: str) -> bool:
    return name.endswith(
        ("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET", "_CREDENTIAL")
    )


def _positive_seconds(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    try:
        value = float(environment.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default
