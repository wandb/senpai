# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Durable, exact rollback for operational-supervisor Kubernetes resources."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from k8s.launch_helpers import kubectl_command

SCHEMA = "senpai-supervisor-rollback/v1"
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
    """One or more Kubernetes resources could not be restored."""


@dataclass(frozen=True)
class _Target:
    resource: str
    kind: str
    name: str


def _targets(tag: str) -> tuple[_Target, ...]:
    supervisor = f"senpai-supervisor-{tag}"
    return (
        _Target(
            "networkpolicy.networking.k8s.io",
            "NetworkPolicy",
            f"senpai-supervisor-egress-{tag}",
        ),
        _Target("serviceaccount", "ServiceAccount", supervisor),
        _Target("role.rbac.authorization.k8s.io", "Role", supervisor),
        _Target(
            "rolebinding.rbac.authorization.k8s.io",
            "RoleBinding",
            supervisor,
        ),
        _Target("deployment.apps", "Deployment", supervisor),
    )


def _resource_name(target: _Target) -> str:
    return f"{target.resource}/{target.name}"


def _run(
    *arguments: str,
    kube_context: str,
    namespace: str,
    input_text: str | None = None,
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
        )
    except OSError as error:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"could not execute kubectl: {error}",
        )


def _detail(result: subprocess.CompletedProcess[str]) -> str:
    return (
        result.stderr.strip()
        or result.stdout.strip()
        or "kubectl returned no detail"
    )


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
            f"could not capture {_resource_name(target)}: {_detail(result)}"
        )
    if not result.stdout.strip():
        return None
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"kubectl returned invalid JSON for {_resource_name(target)}"
        ) from error
    if not isinstance(document, dict):
        raise RuntimeError(
            f"kubectl returned a non-object for {_resource_name(target)}"
        )
    return document


def _restorable_manifest(
    document: dict[str, Any],
    target: _Target,
    *,
    namespace: str,
) -> dict[str, Any]:
    metadata = document.get("metadata")
    if (
        not isinstance(document.get("apiVersion"), str)
        or document.get("kind") != target.kind
        or not isinstance(metadata, dict)
    ):
        raise RuntimeError(
            f"captured {_resource_name(target)} did not match its expected kind"
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

    manifest = copy.deepcopy(document)
    manifest.pop("status", None)
    manifest["metadata"] = {
        key: value
        for key, value in metadata.items()
        if key not in _SERVER_METADATA
    }
    return manifest


def _default_directory() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    root = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return root / "senpai" / "rollback"


def _persist_bundle(
    bundle: dict[str, Any],
    directory: Path,
    tag: str,
    timeout_seconds: int,
) -> Path:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f"operational-supervisor-{tag}-",
        suffix=".json",
        dir=directory,
        text=True,
    )
    path = Path(name)
    try:
        bundle = copy.deepcopy(bundle)
        bundle["manual_restore_argv"] = [
            sys.executable,
            str(Path(__file__).resolve()),
            "restore",
            str(path),
            "--timeout-seconds",
            str(timeout_seconds),
        ]
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(bundle, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


@dataclass(frozen=True)
class SupervisorRollback:
    """A persisted snapshot of the mutable supervisor release resources."""

    path: Path

    @classmethod
    def capture(
        cls,
        *,
        tag: str,
        kube_context: str = "",
        namespace: str,
        directory: Path | None = None,
        timeout_seconds: int = 900,
    ) -> SupervisorRollback:
        resources = []
        for target in _targets(tag):
            document = _get(
                target,
                kube_context=kube_context,
                namespace=namespace,
            )
            resources.append(
                {
                    "resource": target.resource,
                    "kind": target.kind,
                    "name": target.name,
                    "present": document is not None,
                    "manifest": (
                        None
                        if document is None
                        else _restorable_manifest(
                            document,
                            target,
                            namespace=namespace,
                        )
                    ),
                }
            )

        bundle = {
            "schema": SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "tag": tag,
            "kube_context": kube_context,
            "namespace": namespace,
            "persistent_state_rolled_back": False,
            "operator_notice": (
                "This bundle restores only mutable Kubernetes release resources. "
                "The operational supervisor's persistent SQLite state is never "
                "rolled back."
            ),
            "resources": resources,
        }
        try:
            path = _persist_bundle(
                bundle,
                directory or _default_directory(),
                tag,
                timeout_seconds,
            )
        except OSError as error:
            raise RuntimeError(
                f"could not persist operational supervisor rollback bundle: {error}"
            ) from error
        return cls(path)

    def restore(self, *, timeout_seconds: int) -> None:
        """Restore the persisted resource snapshot and verify the old Deployment."""

        try:
            bundle = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RollbackError(
                f"could not read rollback bundle {self.path}"
            ) from error
        if bundle.get("schema") != SCHEMA:
            raise RollbackError(f"unsupported rollback bundle {self.path}")

        kube_context = bundle.get("kube_context")
        namespace = bundle.get("namespace")
        tag = bundle.get("tag")
        records = bundle.get("resources")
        if (
            not isinstance(kube_context, str)
            or not isinstance(namespace, str)
            or not isinstance(tag, str)
            or not isinstance(records, list)
        ):
            raise RollbackError(f"invalid rollback bundle {self.path}")

        targets = _targets(tag)
        expected = {
            (target.resource, target.kind, target.name): target for target in targets
        }
        identities = []
        for record in records:
            if not isinstance(record, dict):
                raise RollbackError(f"invalid resource record in {self.path}")
            identity = (
                record.get("resource"),
                record.get("kind"),
                record.get("name"),
            )
            if not all(isinstance(value, str) for value in identity):
                raise RollbackError(f"invalid resource record in {self.path}")
            identities.append(identity)
        if len(records) != len(targets) or set(identities) != set(expected):
            raise RollbackError(
                f"rollback bundle {self.path} contains unexpected resources"
            )
        plan: list[tuple[_Target, bool, dict[str, Any] | None]] = []
        for record in records:
            key = (record.get("resource"), record.get("kind"), record.get("name"))
            target = expected[key]
            present = record.get("present")
            if not isinstance(present, bool):
                raise RollbackError(
                    "invalid presence marker for "
                    f"{_resource_name(target)} in {self.path}"
                )
            manifest = record.get("manifest")
            if present:
                if not isinstance(manifest, dict):
                    raise RollbackError(
                        f"missing manifest for {_resource_name(target)} in {self.path}"
                    )
                try:
                    manifest = _restorable_manifest(
                        manifest,
                        target,
                        namespace=namespace,
                    )
                except RuntimeError as error:
                    raise RollbackError(str(error)) from error
            else:
                manifest = None
            plan.append((target, present, manifest))

        failures: list[str] = []
        old_deployment: str | None = None
        for target, present, manifest in sorted(
            plan,
            key=lambda item: not item[1],
        ):
            resource_name = _resource_name(target)
            if not present:
                result = _run(
                    "delete",
                    resource_name,
                    "--ignore-not-found",
                    kube_context=kube_context,
                    namespace=namespace,
                )
                if result.returncode != 0:
                    failures.append(f"delete {resource_name}: {_detail(result)}")
                continue

            assert manifest is not None
            try:
                current = _get(
                    target,
                    kube_context=kube_context,
                    namespace=namespace,
                )
            except RuntimeError as error:
                failures.append(str(error))
                continue
            operation = "create"
            desired = copy.deepcopy(manifest)
            if current is not None:
                current_metadata = current.get("metadata")
                resource_version = (
                    current_metadata.get("resourceVersion")
                    if isinstance(current_metadata, dict)
                    else None
                )
                if not isinstance(resource_version, str) or not resource_version:
                    failures.append(
                        f"replace {resource_name}: current resourceVersion is missing"
                    )
                    continue
                desired.setdefault("metadata", {})["resourceVersion"] = (
                    resource_version
                )
                operation = "replace"
            result = _run(
                operation,
                "-f",
                "-",
                kube_context=kube_context,
                namespace=namespace,
                input_text=json.dumps(desired, sort_keys=True),
            )
            if result.returncode != 0:
                failures.append(f"{operation} {resource_name}: {_detail(result)}")
            if target.kind == "Deployment":
                old_deployment = target.name

        if old_deployment is not None:
            result = _run(
                "rollout",
                "status",
                f"deployment/{old_deployment}",
                f"--timeout={timeout_seconds}s",
                kube_context=kube_context,
                namespace=namespace,
            )
            if result.returncode != 0:
                failures.append(
                    f"rollout deployment/{old_deployment}: {_detail(result)}"
                )
        if failures:
            raise RollbackError("; ".join(failures))

    def manual_restore_argv(self) -> list[str]:
        """Return the exact argv stored with this rollback bundle."""

        try:
            bundle = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RollbackError(
                f"could not read rollback bundle {self.path}"
            ) from error
        argv = bundle.get("manual_restore_argv")
        if not (
            isinstance(argv, list)
            and argv
            and all(isinstance(argument, str) for argument in argv)
        ):
            raise RollbackError(
                f"rollback bundle {self.path} has no recovery command"
            )
        return argv

    def commit(self) -> None:
        """Discard the snapshot after the new supervisor rollout is healthy."""

        self.path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore a Senpai operational-supervisor rollback bundle."
    )
    parser.add_argument("operation", choices=("restore",))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    rollback = SupervisorRollback(args.bundle)
    try:
        rollback.restore(timeout_seconds=args.timeout_seconds)
    except RollbackError as error:
        parser.exit(1, f"ERROR: {error}\nRollback bundle retained at: {args.bundle}\n")
    print(
        "Restored the prior mutable operational-supervisor resources. "
        "Persistent SQLite state was not rolled back.\n"
        f"Rollback bundle retained at: {args.bundle}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
