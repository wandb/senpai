# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Crash-visible, scope-pinned rollback for operational-supervisor releases."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from k8s.launch_helpers import kubectl_command

SCHEMA = "senpai-supervisor-rollback/v2"
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
KUBECTL_API_TIMEOUT_SECONDS = 45
MAX_ROLLBACK_TIMEOUT_SECONDS = 3600
LEASE_CUSHION_SECONDS = 300
MAX_TRANSACTION_LEASE_SECONDS = (
    MAX_ROLLBACK_TIMEOUT_SECONDS + LEASE_CUSHION_SECONDS
)
LEASE_PHASE_ANNOTATION = "senpai.wandb.com/release-transaction-phase"
_LEASE_PHASES = {
    "capturing",
    "captured",
    "mutating",
    "recovery_required",
    "restored",
    "committed",
    "superseded",
}
_SAFE_TAKEOVER_PHASES = {"capturing", "captured", "restored", "committed"}
_BUNDLE_STATUSES = {
    "captured",
    "mutating",
    "recovery_required",
    "restored",
    "committed",
}
_SERVER_METADATA = {
    "creationTimestamp",
    "deletionGracePeriodSeconds",
    "deletionTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
}


class RollbackError(RuntimeError):
    """The supervisor release could not be restored safely and exactly."""


@dataclass(frozen=True)
class _Target:
    resource: str
    api_version: str
    kind: str
    name: str


@dataclass(frozen=True)
class _Scope:
    kube_context: str
    namespace: str
    kube_system_uid: str
    namespace_uid: str


def _targets(tag: str) -> tuple[_Target, ...]:
    supervisor = f"senpai-supervisor-{tag}"
    return (
        _Target(
            "networkpolicy.networking.k8s.io",
            "networking.k8s.io/v1",
            "NetworkPolicy",
            f"senpai-supervisor-egress-{tag}",
        ),
        _Target("serviceaccount", "v1", "ServiceAccount", supervisor),
        _Target(
            "role.rbac.authorization.k8s.io",
            "rbac.authorization.k8s.io/v1",
            "Role",
            supervisor,
        ),
        _Target(
            "rolebinding.rbac.authorization.k8s.io",
            "rbac.authorization.k8s.io/v1",
            "RoleBinding",
            supervisor,
        ),
        _Target("deployment.apps", "apps/v1", "Deployment", supervisor),
    )


def _lease_target(tag: str) -> _Target:
    return _Target(
        "lease.coordination.k8s.io",
        "coordination.k8s.io/v1",
        "Lease",
        f"senpai-supervisor-release-{tag}",
    )


def _resource_name(target: _Target) -> str:
    return f"{target.resource}/{target.name}"


def _completed_error(
    command: list[str],
    detail: str,
    *,
    returncode: int = 124,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout="",
        stderr=detail,
    )


def _run(
    *arguments: str,
    kube_context: str,
    namespace: str,
    input_text: str | None = None,
    process_timeout_seconds: int = KUBECTL_API_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    command = kubectl_command(
        *arguments,
        kube_context=kube_context,
        namespace=namespace,
    )
    try:
        return subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=process_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _completed_error(
            command,
            f"kubectl exceeded its {process_timeout_seconds}s process deadline",
        )
    except OSError as error:
        return _completed_error(
            command,
            f"could not execute kubectl: {error}",
            returncode=127,
        )


def _run_global(
    *arguments: str,
    process_timeout_seconds: int = KUBECTL_API_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    command = ["kubectl", *arguments]
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=process_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _completed_error(
            command,
            f"kubectl exceeded its {process_timeout_seconds}s process deadline",
        )
    except OSError as error:
        return _completed_error(
            command,
            f"could not execute kubectl: {error}",
            returncode=127,
        )


def _detail(result: subprocess.CompletedProcess[str]) -> str:
    return (
        result.stderr.strip()
        or result.stdout.strip()
        or "kubectl returned no detail"
    )


def _json_object(text: str, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"kubectl returned invalid JSON for {description}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"kubectl returned a non-object for {description}")
    return value


def _get(
    target: _Target,
    *,
    kube_context: str,
    namespace: str,
) -> dict[str, Any] | None:
    result = _run(
        "get",
        _resource_name(target),
        "--ignore-not-found",
        "-o",
        "json",
        kube_context=kube_context,
        namespace=namespace,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read {_resource_name(target)}: {_detail(result)}"
        )
    if not result.stdout.strip():
        return None
    return _json_object(result.stdout, description=_resource_name(target))


def _resolve_context(kube_context: str) -> str:
    if kube_context:
        return kube_context
    result = _run_global("config", "current-context")
    context = result.stdout.strip()
    if result.returncode != 0 or not context or "\n" in context:
        raise RuntimeError(
            "could not resolve an explicit Kubernetes context: " + _detail(result)
        )
    return context


def _namespace_uid(name: str, *, kube_context: str, namespace: str) -> str:
    result = _run(
        "get",
        f"namespace/{name}",
        "-o",
        "json",
        kube_context=kube_context,
        namespace=namespace,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not identify namespace {name}: {_detail(result)}")
    document = _json_object(result.stdout, description=f"namespace/{name}")
    metadata = document.get("metadata")
    uid = metadata.get("uid") if isinstance(metadata, dict) else None
    if not isinstance(uid, str) or not uid:
        raise RuntimeError(f"namespace {name} has no stable UID")
    return uid


def _capture_scope(kube_context: str, namespace: str) -> _Scope:
    context = _resolve_context(kube_context)
    return _Scope(
        kube_context=context,
        namespace=namespace,
        kube_system_uid=_namespace_uid(
            "kube-system",
            kube_context=context,
            namespace=namespace,
        ),
        namespace_uid=_namespace_uid(
            namespace,
            kube_context=context,
            namespace=namespace,
        ),
    )


def _assert_scope(scope: _Scope) -> None:
    kube_system_uid = _namespace_uid(
        "kube-system",
        kube_context=scope.kube_context,
        namespace=scope.namespace,
    )
    if kube_system_uid != scope.kube_system_uid:
        raise RollbackError("Kubernetes cluster identity changed; refusing rollback")
    namespace_uid = _namespace_uid(
        scope.namespace,
        kube_context=scope.kube_context,
        namespace=scope.namespace,
    )
    if namespace_uid != scope.namespace_uid:
        raise RollbackError("Kubernetes namespace identity changed; refusing rollback")


def _restorable_manifest(
    document: dict[str, Any],
    target: _Target,
    *,
    namespace: str,
) -> dict[str, Any]:
    metadata = document.get("metadata")
    if (
        document.get("apiVersion") != target.api_version
        or document.get("kind") != target.kind
        or not isinstance(metadata, dict)
    ):
        raise RuntimeError(
            f"captured {_resource_name(target)} did not match its expected API/kind"
        )
    if metadata.get("name") != target.name:
        raise RuntimeError(
            f"captured {_resource_name(target)} had an unexpected name"
        )
    captured_namespace = metadata.get("namespace")
    if captured_namespace not in (None, namespace):
        raise RuntimeError(
            f"captured {_resource_name(target)} came from an unexpected namespace"
        )
    if metadata.get("deletionTimestamp") is not None:
        raise RuntimeError(
            f"captured {_resource_name(target)} while it was being deleted"
        )

    manifest = copy.deepcopy(document)
    manifest.pop("status", None)
    clean_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in _SERVER_METADATA
    }
    clean_metadata["name"] = target.name
    clean_metadata["namespace"] = namespace
    manifest["metadata"] = clean_metadata
    return manifest


def _canonical_manifest(
    document: dict[str, Any],
    target: _Target,
    *,
    namespace: str,
) -> dict[str, Any]:
    manifest = _restorable_manifest(document, target, namespace=namespace)
    if target.kind == "Deployment":
        annotations = manifest["metadata"].get("annotations")
        if isinstance(annotations, dict):
            annotations.pop("deployment.kubernetes.io/revision", None)
            if not annotations:
                manifest["metadata"].pop("annotations", None)
    return manifest


def _lease_metadata_for_replace(
    metadata: dict[str, Any],
    target: _Target,
    namespace: str,
) -> dict[str, Any]:
    resource_version = metadata.get("resourceVersion")
    if not isinstance(resource_version, str) or not resource_version:
        raise RollbackError("campaign transaction Lease has no resourceVersion")
    clean = {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key not in _SERVER_METADATA
    }
    clean.update(
        {
            "name": target.name,
            "namespace": namespace,
            "resourceVersion": resource_version,
        }
    )
    return clean


def _default_directory() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    root = Path(state_home) if state_home else Path.home() / ".local" / "state"
    if not root.is_absolute():
        raise RuntimeError("XDG_STATE_HOME must be an absolute path")
    return root / "senpai" / "rollback"


def _private_directory_fd(directory: Path, *, create: bool) -> int:
    if not directory.is_absolute():
        raise OSError("rollback directory must be absolute")
    if directory.resolve(strict=False) != directory:
        raise OSError("rollback directory may not contain symlinks")
    if create:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(directory, flags)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise OSError("private rollback directory must be owned mode 0700")
    return descriptor


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("rollback bundle write made no progress")
        view = view[written:]


def _write_bundle(path: Path, bundle: dict[str, Any]) -> None:
    payload = (
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_BUNDLE_BYTES:
        raise OSError("rollback bundle exceeds its size limit")
    directory_fd = _private_directory_fd(path.parent, create=False)
    temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _persist_bundle(bundle: dict[str, Any], directory: Path, tag: str) -> Path:
    directory_fd = _private_directory_fd(directory, create=True)
    os.close(directory_fd)
    path = directory / f"operational-supervisor-{tag}-{secrets.token_hex(12)}.json"
    _write_bundle(path, bundle)
    return path


def _read_bundle(path: Path) -> dict[str, Any]:
    absolute = path.absolute()
    directory_fd = -1
    descriptor = -1
    try:
        directory_fd = _private_directory_fd(absolute.parent, create=False)
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(absolute.name, flags, dir_fd=directory_fd)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)
        raise RollbackError(
            f"rollback bundle must be a regular private file: {absolute}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > MAX_BUNDLE_BYTES
        ):
            raise RollbackError(
                f"rollback bundle must be a regular private file: {absolute}"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if remaining or len(payload) != metadata.st_size:
            raise RollbackError(f"rollback bundle changed while reading: {absolute}")
    finally:
        os.close(descriptor)
        os.close(directory_fd)
    try:
        bundle = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RollbackError(f"could not read rollback bundle {absolute}") from error
    if not isinstance(bundle, dict):
        raise RollbackError(f"invalid rollback bundle {absolute}")
    return bundle


def _remove_bundle(path: Path) -> None:
    absolute = path.absolute()
    directory_fd = _private_directory_fd(absolute.parent, create=False)
    try:
        try:
            os.unlink(absolute.name, dir_fd=directory_fd)
        except FileNotFoundError:
            return
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _acquire_local_campaign_lock(directory: Path, tag: str) -> int:
    directory_fd = _private_directory_fd(directory, create=True)
    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:24]
    name = f".campaign-{digest}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise OSError("campaign rollback lock must be an owned mode-0600 file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise OSError("another local release capture is in progress") from error
    return descriptor


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).isoformat().replace("+00:00", "Z")


def _lease_duration(timeout_seconds: int) -> int:
    return min(
        MAX_TRANSACTION_LEASE_SECONDS,
        max(LEASE_CUSHION_SECONDS, timeout_seconds + LEASE_CUSHION_SECONDS),
    )


def _lease_phase(metadata: dict[str, Any]) -> str | None:
    annotations = metadata.get("annotations")
    phase = annotations.get(LEASE_PHASE_ANNOTATION) if isinstance(annotations, dict) else None
    return phase if isinstance(phase, str) and phase in _LEASE_PHASES else None


def _set_lease_phase(metadata: dict[str, Any], phase: str) -> None:
    annotations = metadata.setdefault("annotations", {})
    if not isinstance(annotations, dict):
        raise RollbackError("campaign transaction Lease annotations are malformed")
    annotations[LEASE_PHASE_ANNOTATION] = phase


@dataclass
class _CampaignLease:
    scope: _Scope
    tag: str
    holder_identity: str
    duration_seconds: int
    required_uid: str | None = None
    required_transitions: int | None = None
    lease_uid: str = ""
    lease_transitions: int = -1
    phase: str = ""
    acquired: bool = False

    @classmethod
    def acquire(
        cls,
        scope: _Scope,
        tag: str,
        *,
        timeout_seconds: int,
        holder_identity: str | None = None,
        required_uid: str | None = None,
        required_transitions: int | None = None,
    ) -> _CampaignLease:
        lease = cls(
            scope=scope,
            tag=tag,
            holder_identity=holder_identity or f"senpai-launch-{secrets.token_hex(16)}",
            duration_seconds=_lease_duration(timeout_seconds),
            required_uid=required_uid,
            required_transitions=required_transitions,
        )
        lease._acquire()
        return lease

    @property
    def target(self) -> _Target:
        return _lease_target(self.tag)

    def _acquire(self, *, next_phase: str | None = None) -> None:
        if next_phase is not None and next_phase not in _LEASE_PHASES:
            raise ValueError(f"unsupported campaign Lease phase: {next_phase}")
        _assert_scope(self.scope)
        current = _get(
            self.target,
            kube_context=self.scope.kube_context,
            namespace=self.scope.namespace,
        )
        if current is None and (self.lease_uid or self.required_uid):
            raise RollbackError(
                "campaign transaction Lease lineage changed; refusing stale recovery"
            )
        now = datetime.now(UTC)
        operation = "create"
        metadata: dict[str, Any] = {
            "name": self.target.name,
            "namespace": self.scope.namespace,
            "labels": {
                "app": "senpai",
                "role": "supervisor-release-lock",
                "research-tag": self.tag,
            },
        }
        desired_phase = next_phase or self.phase or "capturing"
        transitions = 0
        acquire_time = _timestamp(now)
        if current is not None:
            if (
                current.get("apiVersion") != self.target.api_version
                or current.get("kind") != self.target.kind
                or not isinstance(current.get("metadata"), dict)
                or not isinstance(current.get("spec"), dict)
            ):
                raise RollbackError("campaign transaction Lease is malformed")
            current_metadata = current["metadata"]
            if (
                current_metadata.get("name") != self.target.name
                or current_metadata.get("namespace") not in (None, self.scope.namespace)
            ):
                raise RollbackError("campaign transaction Lease has the wrong scope")
            resource_version = current_metadata.get("resourceVersion")
            lease_uid = current_metadata.get("uid")
            if not isinstance(resource_version, str) or not resource_version:
                raise RollbackError("campaign transaction Lease has no resourceVersion")
            if not isinstance(lease_uid, str) or not lease_uid:
                raise RollbackError("campaign transaction Lease has no UID")
            spec = current["spec"]
            current_phase = _lease_phase(current_metadata)
            if current_phase is None:
                raise RollbackError("campaign transaction Lease has no valid phase")
            holder = spec.get("holderIdentity")
            current_transitions = spec.get("leaseTransitions", 0)
            if not isinstance(current_transitions, int) or current_transitions < 0:
                raise RollbackError("campaign transaction Lease has an invalid epoch")
            expected_uid = self.lease_uid or self.required_uid
            expected_transitions = (
                self.lease_transitions
                if self.lease_transitions >= 0
                else self.required_transitions
            )
            if expected_uid is not None and (
                lease_uid != expected_uid
                or current_transitions != expected_transitions
                or holder != self.holder_identity
            ):
                raise RollbackError(
                    "campaign transaction Lease lineage changed; refusing stale recovery"
                )
            if self.acquired and self.phase and current_phase != self.phase:
                raise RollbackError("campaign transaction Lease phase changed")
            duration = spec.get("leaseDurationSeconds")
            renewed = _parse_time(spec.get("renewTime"))
            active = False
            if isinstance(holder, str) and holder:
                if not isinstance(duration, int) or duration <= 0 or renewed is None:
                    raise RollbackError("active campaign transaction Lease is malformed")
                active = now < renewed + timedelta(seconds=duration)
            if active and not self.acquired:
                if holder == self.holder_identity:
                    raise RollbackError(
                        "the original release transaction is still active; "
                        "refusing concurrent recovery"
                    )
                raise RollbackError(
                    "another operational-supervisor release transaction is active"
                )
            if (
                not self.acquired
                and expected_uid is None
                and current_phase not in _SAFE_TAKEOVER_PHASES
            ):
                raise RollbackError(
                    "an expired supervisor release requires recovery before takeover"
                )
            operation = "replace"
            metadata = _lease_metadata_for_replace(
                current_metadata,
                self.target,
                self.scope.namespace,
            )
            transitions = current_transitions
            if holder != self.holder_identity:
                transitions += 1
            existing_acquire = spec.get("acquireTime")
            if holder == self.holder_identity and isinstance(existing_acquire, str):
                acquire_time = existing_acquire
            if expected_uid is not None and not self.acquired:
                desired_phase = current_phase

        _set_lease_phase(metadata, desired_phase)

        manifest = {
            "apiVersion": self.target.api_version,
            "kind": self.target.kind,
            "metadata": metadata,
            "spec": {
                "holderIdentity": self.holder_identity,
                "leaseDurationSeconds": self.duration_seconds,
                "acquireTime": acquire_time,
                "renewTime": _timestamp(now),
                "leaseTransitions": transitions,
            },
        }
        result = _run(
            operation,
            "-f",
            "-",
            kube_context=self.scope.kube_context,
            namespace=self.scope.namespace,
            input_text=json.dumps(manifest, sort_keys=True),
        )
        operation_error = (
            f"could not update campaign transaction Lease: {_detail(result)}"
            if result.returncode != 0
            else None
        )
        try:
            observed = _get(
                self.target,
                kube_context=self.scope.kube_context,
                namespace=self.scope.namespace,
            )
        except RuntimeError as read_error:
            if operation_error is not None:
                raise RollbackError(
                    f"{operation_error}; could not verify its outcome: {read_error}"
                ) from read_error
            raise
        observed_spec = observed.get("spec") if isinstance(observed, dict) else None
        observed_metadata = (
            observed.get("metadata") if isinstance(observed, dict) else None
        )
        if (
            not isinstance(observed_spec, dict)
            or not isinstance(observed_metadata, dict)
            or observed_spec.get("holderIdentity") != self.holder_identity
            or observed_spec.get("leaseDurationSeconds") != self.duration_seconds
            or _lease_phase(observed_metadata) != desired_phase
        ):
            if operation_error is not None:
                raise RollbackError(operation_error)
            raise RollbackError("campaign transaction Lease acquisition was not visible")
        observed_renew_time = _parse_time(observed_spec.get("renewTime"))
        if (
            observed_renew_time is None
            or datetime.now(UTC)
            >= observed_renew_time + timedelta(seconds=self.duration_seconds)
        ):
            if operation_error is not None:
                raise RollbackError(operation_error)
            raise RollbackError("campaign transaction Lease is not active after acquisition")
        observed_uid = observed_metadata.get("uid")
        observed_transitions = observed_spec.get("leaseTransitions")
        if (
            not isinstance(observed_uid, str)
            or not observed_uid
            or not isinstance(observed_transitions, int)
            or observed_transitions < 0
        ):
            if operation_error is not None:
                raise RollbackError(operation_error)
            raise RollbackError("campaign transaction Lease lineage is incomplete")
        if self.lease_uid and (
            observed_uid != self.lease_uid
            or observed_transitions != self.lease_transitions
        ):
            if operation_error is not None:
                raise RollbackError(operation_error)
            raise RollbackError("campaign transaction Lease lineage changed")
        self.lease_uid = observed_uid
        self.lease_transitions = observed_transitions
        self.phase = desired_phase
        self.acquired = True

    def renew(self) -> None:
        if not self.acquired:
            raise RollbackError("campaign transaction Lease is not held")
        self._acquire()

    def set_phase(self, phase: str) -> None:
        if not self.acquired:
            raise RollbackError("campaign transaction Lease is not held")
        self._acquire(next_phase=phase)

    def release(self) -> None:
        if not self.acquired:
            return
        _assert_scope(self.scope)
        current = _get(
            self.target,
            kube_context=self.scope.kube_context,
            namespace=self.scope.namespace,
        )
        if current is None:
            raise RollbackError("campaign transaction Lease disappeared")
        metadata = current.get("metadata")
        spec = current.get("spec")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise RollbackError("campaign transaction Lease is malformed")
        if spec.get("holderIdentity") != self.holder_identity:
            raise RollbackError("campaign transaction Lease ownership changed")
        if (
            metadata.get("uid") != self.lease_uid
            or spec.get("leaseTransitions") != self.lease_transitions
            or _lease_phase(metadata) != self.phase
        ):
            raise RollbackError("campaign transaction Lease lineage changed")
        replacement = copy.deepcopy(current)
        replacement.pop("status", None)
        replacement["metadata"] = _lease_metadata_for_replace(
            metadata,
            self.target,
            self.scope.namespace,
        )
        replacement["spec"] = {
            **spec,
            "holderIdentity": self.holder_identity,
            "leaseDurationSeconds": 1,
            "renewTime": _timestamp(datetime.now(UTC) - timedelta(seconds=2)),
        }
        result = _run(
            "replace",
            "-f",
            "-",
            kube_context=self.scope.kube_context,
            namespace=self.scope.namespace,
            input_text=json.dumps(replacement, sort_keys=True),
        )
        operation_error = (
            f"could not release campaign transaction Lease: {_detail(result)}"
            if result.returncode != 0
            else None
        )
        try:
            observed = _get(
                self.target,
                kube_context=self.scope.kube_context,
                namespace=self.scope.namespace,
            )
        except RuntimeError as read_error:
            if operation_error is not None:
                raise RollbackError(
                    f"{operation_error}; could not verify its outcome: {read_error}"
                ) from read_error
            raise
        observed_spec = observed.get("spec") if isinstance(observed, dict) else None
        observed_metadata = (
            observed.get("metadata") if isinstance(observed, dict) else None
        )
        if (
            not isinstance(observed_spec, dict)
            or not isinstance(observed_metadata, dict)
            or observed_spec.get("holderIdentity") != self.holder_identity
            or observed_spec.get("leaseTransitions") != self.lease_transitions
            or observed_spec.get("leaseDurationSeconds") != 1
            or _lease_phase(observed_metadata) != self.phase
        ):
            if operation_error is not None:
                raise RollbackError(operation_error)
            raise RollbackError("campaign transaction Lease release was not visible")
        observed_renew_time = _parse_time(observed_spec.get("renewTime"))
        if (
            observed_renew_time is None
            or datetime.now(UTC) < observed_renew_time + timedelta(seconds=1)
        ):
            if operation_error is not None:
                raise RollbackError(operation_error)
            raise RollbackError("campaign transaction Lease did not expire on release")
        self.acquired = False


def _scope_from_bundle(bundle: dict[str, Any], path: Path) -> _Scope:
    values = {
        name: bundle.get(name)
        for name in (
            "kube_context",
            "namespace",
            "kube_system_uid",
            "namespace_uid",
        )
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise RollbackError(f"invalid rollback scope in {path}")
    return _Scope(**values)


def _same_cluster_scope(left: _Scope, right: _Scope) -> bool:
    return (
        left.namespace == right.namespace
        and left.kube_system_uid == right.kube_system_uid
        and left.namespace_uid == right.namespace_uid
    )


def _authoritative_bundle_final_status(
    bundle: dict[str, Any],
    *,
    scope: _Scope,
    tag: str,
) -> str | None:
    transaction_id = bundle.get("transaction_id")
    lease_uid = bundle.get("lease_uid")
    lease_transitions = bundle.get("lease_transitions")
    if (
        not isinstance(transaction_id, str)
        or not isinstance(lease_uid, str)
        or not isinstance(lease_transitions, int)
    ):
        raise RollbackError("rollback journal has invalid Lease lineage")
    current = _get(
        _lease_target(tag),
        kube_context=scope.kube_context,
        namespace=scope.namespace,
    )
    if current is None:
        return None
    metadata = current.get("metadata")
    spec = current.get("spec")
    target = _lease_target(tag)
    if (
        current.get("apiVersion") != target.api_version
        or current.get("kind") != target.kind
        or not isinstance(metadata, dict)
        or metadata.get("name") != target.name
        or metadata.get("namespace") not in (None, scope.namespace)
        or not isinstance(spec, dict)
    ):
        raise RollbackError("campaign transaction Lease is malformed")
    current_phase = _lease_phase(metadata)
    current_transitions = spec.get("leaseTransitions")
    if current_phase not in {"restored", "committed"}:
        return None
    if (
        metadata.get("uid") == lease_uid
        and current_transitions == lease_transitions
        and spec.get("holderIdentity") == transaction_id
    ):
        return current_phase
    if (
        metadata.get("uid") == lease_uid
        and isinstance(current_transitions, int)
        and current_transitions > lease_transitions
    ):
        return "superseded"
    return None


def _matching_unfinished_bundles(
    directory: Path,
    *,
    tag: str,
    scope: _Scope,
) -> list[Path]:
    descriptor = _private_directory_fd(directory, create=True)
    try:
        names = os.listdir(descriptor)
    finally:
        os.close(descriptor)
    prefix = f"operational-supervisor-{tag}-"
    unfinished: list[Path] = []
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        path = directory / name
        try:
            bundle = _read_bundle(path)
            candidate_scope = _scope_from_bundle(bundle, path)
        except RollbackError as error:
            raise RuntimeError(f"unreadable prior rollback bundle {path}: {error}") from error
        if (
            bundle.get("schema") == SCHEMA
            and bundle.get("tag") == tag
            and _same_cluster_scope(candidate_scope, scope)
            and bundle.get("status") not in {
                "restored",
                "committed",
                "superseded",
            }
        ):
            final_status = _authoritative_bundle_final_status(
                bundle,
                scope=scope,
                tag=tag,
            )
            if final_status is None:
                unfinished.append(path)
            else:
                bundle["status"] = final_status
                bundle[f"{final_status}_at"] = _timestamp()
                _write_bundle(path, bundle)
        elif bundle.get("schema") != SCHEMA:
            raise RuntimeError(f"unsupported prior rollback bundle {path}")
    return unfinished


@dataclass
class SupervisorRollback:
    """A durable snapshot plus its campaign-scoped release transaction."""

    path: Path
    timeout_seconds: int = 900
    kube_context: str = ""
    _lease: _CampaignLease | None = field(default=None, repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        *,
        tag: str,
        kube_context: str = "",
        namespace: str,
        directory: Path | None = None,
        timeout_seconds: int = 900,
        network_policy_safety_manifest: dict[str, Any] | None = None,
    ) -> SupervisorRollback:
        if timeout_seconds <= 0 or timeout_seconds > MAX_ROLLBACK_TIMEOUT_SECONDS:
            raise RuntimeError("rollback timeout must be between 1 and 3600 seconds")
        rollback_directory = (directory or _default_directory()).absolute()
        try:
            directory_fd = _private_directory_fd(rollback_directory, create=True)
            os.close(directory_fd)
        except OSError as error:
            raise RuntimeError(
                f"could not prepare private rollback directory: {error}"
            ) from error

        try:
            lock_descriptor = _acquire_local_campaign_lock(
                rollback_directory,
                tag,
            )
        except OSError as error:
            raise RuntimeError(
                f"could not lock the campaign rollback journal: {error}"
            ) from error
        lease: _CampaignLease | None = None
        try:
            scope = _capture_scope(kube_context, namespace)
            unfinished = _matching_unfinished_bundles(
                rollback_directory,
                tag=tag,
                scope=scope,
            )
            if unfinished:
                listed = ", ".join(str(path) for path in unfinished)
                raise RuntimeError(
                    "unfinished operational-supervisor rollback exists; "
                    f"recover it before another release: {listed}"
                )

            lease = _CampaignLease.acquire(
                scope,
                tag,
                timeout_seconds=timeout_seconds,
            )

            safety_policy = None
            if network_policy_safety_manifest is not None:
                if not isinstance(network_policy_safety_manifest, dict):
                    raise RuntimeError(
                        "network policy safety manifest must be a Kubernetes object"
                    )
                safety_policy = _restorable_manifest(
                    network_policy_safety_manifest,
                    _targets(tag)[0],
                    namespace=scope.namespace,
                )

            resources = []
            for target in _targets(tag):
                document = _get(
                    target,
                    kube_context=scope.kube_context,
                    namespace=scope.namespace,
                )
                resources.append(
                    {
                        "resource": target.resource,
                        "api_version": target.api_version,
                        "kind": target.kind,
                        "name": target.name,
                        "present": document is not None,
                        "manifest": (
                            None
                            if document is None
                            else _restorable_manifest(
                                document,
                                target,
                                namespace=scope.namespace,
                            )
                        ),
                        "safety_manifest": (
                            safety_policy
                            if document is None and target.kind == "NetworkPolicy"
                            else None
                        ),
                    }
                )

            bundle = {
                "schema": SCHEMA,
                "created_at": _timestamp(),
                "status": "captured",
                "transaction_id": lease.holder_identity,
                "lease_uid": lease.lease_uid,
                "lease_transitions": lease.lease_transitions,
                "tag": tag,
                "kube_context": scope.kube_context,
                "namespace": scope.namespace,
                "kube_system_uid": scope.kube_system_uid,
                "namespace_uid": scope.namespace_uid,
                "persistent_state_rolled_back": False,
                "operator_notice": (
                    "This bundle restores only mutable Kubernetes release resources. "
                    "A first-launch metadata-egress NetworkPolicy is retained as a "
                    "fail-closed safety boundary. "
                    "Immutable Secret/ConfigMap artifacts and persistent SQLite state "
                    "are never rolled back."
                ),
                "resources": resources,
            }
            path = _persist_bundle(bundle, rollback_directory, tag)
            lease.set_phase("captured")
        except BaseException as error:
            if lease is not None:
                try:
                    lease.release()
                except BaseException as release_error:
                    error.add_note(
                        f"could not release transaction Lease: {release_error}"
                    )
            raise
        finally:
            os.close(lock_descriptor)
        assert lease is not None
        return cls(
            path=path,
            timeout_seconds=timeout_seconds,
            kube_context=scope.kube_context,
            _lease=lease,
        )

    @property
    def resolved_kube_context(self) -> str:
        """Return the explicit context pinned by the durable snapshot."""

        if self.kube_context:
            return self.kube_context
        (
            _bundle,
            scope,
            _tag,
            _transaction_id,
            _lease_uid,
            _lease_transitions,
            _plan,
        ) = self._validated_bundle()
        return scope.kube_context

    def _validated_bundle(
        self,
    ) -> tuple[
        dict[str, Any],
        _Scope,
        str,
        str,
        str,
        int,
        list[tuple[_Target, bool, dict[str, Any] | None]],
    ]:
        bundle = _read_bundle(self.path)
        if bundle.get("schema") != SCHEMA:
            raise RollbackError(f"unsupported rollback bundle {self.path}")
        status = bundle.get("status")
        tag = bundle.get("tag")
        transaction_id = bundle.get("transaction_id")
        lease_uid = bundle.get("lease_uid")
        lease_transitions = bundle.get("lease_transitions")
        records = bundle.get("resources")
        if (
            status not in _BUNDLE_STATUSES
            or not isinstance(tag, str)
            or not tag
            or not isinstance(transaction_id, str)
            or not transaction_id.startswith("senpai-launch-")
            or not isinstance(lease_uid, str)
            or not lease_uid
            or not isinstance(lease_transitions, int)
            or lease_transitions < 0
            or not isinstance(records, list)
        ):
            raise RollbackError(f"invalid rollback bundle {self.path}")
        scope = _scope_from_bundle(bundle, self.path)

        targets = _targets(tag)
        expected = {
            (target.resource, target.api_version, target.kind, target.name): target
            for target in targets
        }
        identities: list[tuple[object, object, object, object]] = []
        for record in records:
            if not isinstance(record, dict):
                raise RollbackError(f"invalid resource record in {self.path}")
            identities.append(
                (
                    record.get("resource"),
                    record.get("api_version"),
                    record.get("kind"),
                    record.get("name"),
                )
            )
        if len(records) != len(targets) or set(identities) != set(expected):
            raise RollbackError(
                f"rollback bundle {self.path} contains unexpected resources"
            )

        plan: list[tuple[_Target, bool, dict[str, Any] | None]] = []
        for record in records:
            key = (
                record.get("resource"),
                record.get("api_version"),
                record.get("kind"),
                record.get("name"),
            )
            target = expected[key]
            present = record.get("present")
            manifest = record.get("manifest")
            safety_manifest = record.get("safety_manifest")
            if not isinstance(present, bool):
                raise RollbackError(
                    f"invalid presence marker for {_resource_name(target)}"
                )
            if present:
                if safety_manifest is not None:
                    raise RollbackError(
                        f"present {_resource_name(target)} has an unexpected safety manifest"
                    )
                if not isinstance(manifest, dict):
                    raise RollbackError(
                        f"missing manifest for {_resource_name(target)}"
                    )
                try:
                    manifest = _restorable_manifest(
                        manifest,
                        target,
                        namespace=scope.namespace,
                    )
                except RuntimeError as error:
                    raise RollbackError(str(error)) from error
            elif manifest is not None:
                raise RollbackError(
                    f"absent {_resource_name(target)} has an unexpected manifest"
                )
            elif safety_manifest is not None:
                if target.kind != "NetworkPolicy" or not isinstance(
                    safety_manifest, dict
                ):
                    raise RollbackError(
                        f"absent {_resource_name(target)} has an invalid safety manifest"
                    )
                try:
                    safety_manifest = _restorable_manifest(
                        safety_manifest,
                        target,
                        namespace=scope.namespace,
                    )
                except RuntimeError as error:
                    raise RollbackError(str(error)) from error
                present = True
                manifest = safety_manifest
            plan.append((target, present, manifest))
        return (
            bundle,
            scope,
            tag,
            transaction_id,
            lease_uid,
            lease_transitions,
            plan,
        )

    def _update_status(self, status: str) -> None:
        if status not in _BUNDLE_STATUSES:
            raise ValueError(f"unsupported rollback status: {status}")
        bundle = _read_bundle(self.path)
        if bundle.get("schema") != SCHEMA:
            raise RollbackError(f"unsupported rollback bundle {self.path}")
        bundle["status"] = status
        bundle[f"{status}_at"] = _timestamp()
        _write_bundle(self.path.absolute(), bundle)

    def mark_mutation_started(self) -> None:
        (
            bundle,
            _scope,
            _tag,
            _transaction_id,
            _lease_uid,
            _lease_transitions,
            _plan,
        ) = self._validated_bundle()
        if bundle.get("status") != "captured":
            raise RollbackError("rollback bundle is not ready to begin a release")
        if self._lease is None or not self._lease.acquired:
            raise RollbackError("campaign transaction Lease is not held")
        self._update_status("mutating")
        self._lease.set_phase("mutating")

    def renew_lease(self) -> None:
        """Renew the bounded Lease before a potentially long release phase."""

        if self._lease is None:
            raise RollbackError("campaign transaction Lease is not held")
        self._lease.renew()

    def _ensure_lease(
        self,
        scope: _Scope,
        tag: str,
        transaction_id: str,
        lease_uid: str,
        lease_transitions: int,
        timeout_seconds: int,
    ) -> None:
        if self._lease is not None and self._lease.acquired:
            if self._lease.scope != scope or self._lease.tag != tag:
                raise RollbackError("campaign transaction Lease scope mismatch")
            self._lease.duration_seconds = _lease_duration(timeout_seconds)
            return
        self._lease = _CampaignLease.acquire(
            scope,
            tag,
            timeout_seconds=timeout_seconds,
            holder_identity=transaction_id,
            required_uid=lease_uid,
            required_transitions=lease_transitions,
        )

    @staticmethod
    def _verify_absent(target: _Target, scope: _Scope) -> None:
        current = _get(
            target,
            kube_context=scope.kube_context,
            namespace=scope.namespace,
        )
        if current is not None:
            raise RollbackError(f"{_resource_name(target)} still exists after deletion")

    @staticmethod
    def _delete(target: _Target, scope: _Scope, *, timeout_seconds: int) -> None:
        current = _get(
            target,
            kube_context=scope.kube_context,
            namespace=scope.namespace,
        )
        if current is None:
            SupervisorRollback._verify_absent(target, scope)
            return
        arguments = [
            "delete",
            _resource_name(target),
            "--ignore-not-found",
            "--wait=true",
            f"--timeout={timeout_seconds}s",
        ]
        if target.kind == "Deployment":
            arguments.append("--cascade=foreground")
        result = _run(
            *arguments,
            kube_context=scope.kube_context,
            namespace=scope.namespace,
            process_timeout_seconds=timeout_seconds + 15,
        )
        if result.returncode != 0:
            raise RollbackError(
                f"delete {_resource_name(target)}: {_detail(result)}"
            )
        SupervisorRollback._verify_absent(target, scope)

    @staticmethod
    def _restore_present(
        target: _Target,
        manifest: dict[str, Any],
        scope: _Scope,
    ) -> None:
        current = _get(
            target,
            kube_context=scope.kube_context,
            namespace=scope.namespace,
        )
        desired = copy.deepcopy(manifest)
        operation = "create"
        if current is not None:
            metadata = current.get("metadata")
            resource_version = (
                metadata.get("resourceVersion") if isinstance(metadata, dict) else None
            )
            if not isinstance(resource_version, str) or not resource_version:
                raise RollbackError(
                    f"replace {_resource_name(target)}: current resourceVersion is missing"
                )
            desired["metadata"]["resourceVersion"] = resource_version
            operation = "replace"
        result = _run(
            operation,
            "-f",
            "-",
            kube_context=scope.kube_context,
            namespace=scope.namespace,
            input_text=json.dumps(desired, sort_keys=True),
        )
        if result.returncode != 0:
            raise RollbackError(
                f"{operation} {_resource_name(target)}: {_detail(result)}"
            )
        SupervisorRollback._verify_present(target, manifest, scope)

    @staticmethod
    def _verify_present(
        target: _Target,
        manifest: dict[str, Any],
        scope: _Scope,
    ) -> None:
        observed = _get(
            target,
            kube_context=scope.kube_context,
            namespace=scope.namespace,
        )
        if observed is None:
            raise RollbackError(
                f"{_resource_name(target)} is absent after restoration"
            )
        try:
            actual = _canonical_manifest(observed, target, namespace=scope.namespace)
            expected = _canonical_manifest(manifest, target, namespace=scope.namespace)
        except RuntimeError as error:
            raise RollbackError(str(error)) from error
        if actual != expected:
            raise RollbackError(
                f"{_resource_name(target)} did not match the captured postcondition"
            )

    def _restore_plan(
        self,
        scope: _Scope,
        plan: list[tuple[_Target, bool, dict[str, Any] | None]],
        *,
        timeout_seconds: int,
    ) -> None:
        deployment = next(item for item in plan if item[0].kind == "Deployment")
        _assert_scope(scope)
        self.renew_lease()
        self._delete(deployment[0], scope, timeout_seconds=timeout_seconds)

        failures: list[str] = []
        for target, present, manifest in plan:
            if target.kind == "Deployment":
                continue
            try:
                _assert_scope(scope)
                self.renew_lease()
                if present:
                    assert manifest is not None
                    self._restore_present(target, manifest, scope)
                else:
                    self._delete(target, scope, timeout_seconds=timeout_seconds)
            except (OSError, RuntimeError) as error:
                failures.append(str(error))
        if failures:
            raise RollbackError("; ".join(failures))

        self.renew_lease()
        self._verify_security_plan(scope, plan)

        target, present, manifest = deployment
        if not present:
            self._verify_absent(target, scope)
            return
        assert manifest is not None
        _assert_scope(scope)
        self.renew_lease()
        self._restore_present(target, manifest, scope)
        self.renew_lease()
        result = _run(
            "rollout",
            "status",
            f"deployment/{target.name}",
            f"--timeout={timeout_seconds}s",
            kube_context=scope.kube_context,
            namespace=scope.namespace,
            process_timeout_seconds=timeout_seconds + 15,
        )
        if result.returncode != 0:
            raise RollbackError(
                f"rollout deployment/{target.name}: {_detail(result)}"
            )
        try:
            self.renew_lease()
            self._verify_security_plan(scope, plan)
            self._verify_present(target, manifest, scope)
        except BaseException as error:
            try:
                self.renew_lease()
                self._delete(target, scope, timeout_seconds=timeout_seconds)
            except BaseException as quiesce_error:
                error.add_note(
                    "could not quiesce Deployment after final rollback "
                    f"verification failed: {quiesce_error}"
                )
            raise

    @staticmethod
    def _verify_security_plan(
        scope: _Scope,
        plan: list[tuple[_Target, bool, dict[str, Any] | None]],
    ) -> None:
        for target, present, manifest in plan:
            if target.kind == "Deployment":
                continue
            _assert_scope(scope)
            if present:
                assert manifest is not None
                SupervisorRollback._verify_present(target, manifest, scope)
            else:
                SupervisorRollback._verify_absent(target, scope)

    def restore(self, *, timeout_seconds: int | None = None) -> None:
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0 or timeout > MAX_ROLLBACK_TIMEOUT_SECONDS:
            raise RollbackError("rollback timeout must be between 1 and 3600 seconds")
        (
            bundle,
            scope,
            tag,
            transaction_id,
            lease_uid,
            lease_transitions,
            plan,
        ) = self._validated_bundle()
        if bundle.get("status") == "restored":
            self.finalize(outcome="restored", timeout_seconds=timeout)
            return
        if bundle.get("status") in {"committed", "superseded"}:
            raise RollbackError(
                "rollback bundle is finalized; refusing stale replay"
            )
        _assert_scope(scope)
        self._ensure_lease(
            scope,
            tag,
            transaction_id,
            lease_uid,
            lease_transitions,
            timeout,
        )
        assert self._lease is not None
        if self._lease.phase in {"restored", "committed"}:
            final_phase = self._lease.phase
            self._update_status(final_phase)
            self._lease.release()
            if final_phase == "restored":
                return
            raise RollbackError(
                f"rollback transaction is already {final_phase}; refusing replay"
            )
        restoration_finalized = False
        try:
            _assert_scope(scope)
            self._lease.set_phase("recovery_required")
            self._update_status("recovery_required")
            self._restore_plan(scope, plan, timeout_seconds=timeout)
            self._lease.set_phase("restored")
            restoration_finalized = True
            self._update_status("restored")
        except BaseException as error:
            if not restoration_finalized:
                try:
                    self._lease.set_phase("recovery_required")
                except BaseException as phase_error:
                    error.add_note(
                        f"could not mark the transaction recovery-required: {phase_error}"
                    )
                try:
                    self._update_status("recovery_required")
                except BaseException as status_error:
                    error.add_note(f"could not update recovery journal: {status_error}")
            try:
                self._lease.release()
            except BaseException as release_error:
                error.add_note(f"could not release transaction Lease: {release_error}")
            raise
        else:
            self._lease.release()

    def manual_restore_argv(self) -> list[str]:
        """Derive, rather than deserialize, the operator recovery command."""

        return [
            sys.executable,
            str(Path(__file__).resolve()),
            "restore",
            str(self.path.absolute()),
            "--timeout-seconds",
            str(self.timeout_seconds),
        ]

    def manual_finalize_argv(self) -> list[str]:
        """Derive the explicit healthy-rollout reconciliation command."""

        return [
            sys.executable,
            str(Path(__file__).resolve()),
            "finalize-commit",
            str(self.path.absolute()),
            "--timeout-seconds",
            str(self.timeout_seconds),
        ]

    def finalize(
        self,
        *,
        outcome: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Reconcile a final local or Lease outcome without replaying mutations."""

        if outcome not in {None, "restored", "committed"}:
            raise ValueError(f"unsupported rollback final outcome: {outcome}")
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0 or timeout > MAX_ROLLBACK_TIMEOUT_SECONDS:
            raise RollbackError("rollback timeout must be between 1 and 3600 seconds")
        (
            bundle,
            scope,
            tag,
            transaction_id,
            lease_uid,
            lease_transitions,
            _plan,
        ) = self._validated_bundle()
        local_status = bundle["status"]
        if local_status == "superseded":
            raise RollbackError("rollback transaction was superseded")
        _assert_scope(scope)
        self._ensure_lease(
            scope,
            tag,
            transaction_id,
            lease_uid,
            lease_transitions,
            timeout,
        )
        assert self._lease is not None
        try:
            lease_phase = self._lease.phase
            final_lease_phase = (
                lease_phase if lease_phase in {"restored", "committed"} else None
            )
            final_local_status = (
                local_status
                if local_status in {"restored", "committed"}
                else None
            )
            final_outcome = outcome or final_lease_phase or final_local_status
            if final_outcome is None:
                raise RollbackError(
                    "transaction has no final outcome; restore it or explicitly "
                    "finalize a known-healthy rollout"
                )
            if any(
                value is not None and value != final_outcome
                for value in (final_lease_phase, final_local_status)
            ):
                raise RollbackError("local and cluster rollback outcomes conflict")
            if lease_phase != final_outcome:
                self._lease.set_phase(final_outcome)
            self._update_status(final_outcome)
        except BaseException as error:
            try:
                self._lease.release()
            except BaseException as release_error:
                error.add_note(f"could not release transaction Lease: {release_error}")
            raise
        self._lease.release()
        if final_outcome == "committed":
            _remove_bundle(self.path)

    def commit(self) -> None:
        """Discard the journal only after the new rollout is healthy."""

        if self._lease is None or not self._lease.acquired:
            raise RollbackError("campaign transaction Lease is not held")
        self._lease.set_phase("committed")
        try:
            self._update_status("committed")
        except BaseException as error:
            try:
                self._lease.release()
            except BaseException as release_error:
                error.add_note(f"could not release transaction Lease: {release_error}")
            raise
        self._lease.release()
        _remove_bundle(self.path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore a Senpai operational-supervisor rollback bundle."
    )
    parser.add_argument("operation", choices=("restore", "finalize-commit"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or args.timeout_seconds > MAX_ROLLBACK_TIMEOUT_SECONDS:
        parser.error("--timeout-seconds must be between 1 and 3600")

    rollback = SupervisorRollback(
        args.bundle.absolute(),
        timeout_seconds=args.timeout_seconds,
    )
    try:
        if args.operation == "restore":
            rollback.restore()
        else:
            rollback.finalize(outcome="committed")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\nRollback bundle retained at: {args.bundle}\n")
    if args.operation == "restore":
        print(
            "Restored the prior mutable operational-supervisor resources. "
            "Immutable Secret/ConfigMap artifacts and persistent SQLite state were "
            "not rolled back.\n"
            f"Rollback bundle retained at: {args.bundle}"
        )
    else:
        print("Finalized the known-healthy supervisor release transaction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
