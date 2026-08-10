#!/usr/bin/env python3
"""Render and exercise a credential-free Kubernetes production canary."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANARY_CONVERSATION_ID = UUID("11111111-1111-4111-8111-111111111111")
DUMMY = "SENPAI_CI_DUMMY_"


def _documents(text: str) -> list[dict]:
    return [document for document in yaml.safe_load_all(text) if document]


def _container(pod: dict, name: str) -> dict:
    try:
        return next(item for item in pod["containers"] if item["name"] == name)
    except StopIteration as error:
        raise RuntimeError(
            f"production manifest is missing required container {name!r}"
        ) from error


def _namespace(name: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": name, "labels": {"senpai-ci-canary": "true"}},
    }


def _storage(namespace: str, tag: str) -> tuple[dict, dict]:
    name = f"senpai-ci-{tag}"
    return (
        {
            "apiVersion": "v1",
            "kind": "PersistentVolume",
            "metadata": {
                "name": name,
                "labels": {"senpai-ci-canary": tag},
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "capacity": {"storage": "1Gi"},
                "hostPath": {"path": f"/var/senpai-ci/{tag}"},
                "persistentVolumeReclaimPolicy": "Retain",
                "storageClassName": "",
            },
        },
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {"senpai-ci-canary": tag},
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {"requests": {"storage": "1Gi"}},
                "storageClassName": "",
                "volumeName": name,
            },
        },
    )


def _args(namespace: str, tag: str, image: str, revision: str):
    from k8s.launch import Args

    return Args(
        tag=tag,
        target_repo_url="https://example.invalid/senpai/canary.git",
        target_repo_branch="main",
        backend="kubernetes",
        namespace=namespace,
        repo_url="https://example.invalid/wandb/senpai.git",
        repo_revision=revision,
        advisor_image=image,
        advisor_branch="ci-advisor",
        advisor=True,
        operational_supervisor=True,
        supervisor_dedicated_namespace=True,
        pvc_claim_name=f"senpai-ci-{tag}",
        pvc_mount_path="/mnt/senpai-ci",
        n_students=0,
    )


def _render_advisor(namespace: str, tag: str, image: str, revision: str) -> list[dict]:
    from k8s.launch import ADVISOR_TEMPLATE, render_advisor
    from k8s.launch_helpers import render_launch_secret

    args = _args(namespace, tag, image, revision)
    launch_secret = render_launch_secret(
        tag,
        f"{DUMMY}GITHUB_A",
        f"{DUMMY}EXA_A",
        f"{DUMMY}WANDB_A",
        openai_api_key=f"{DUMMY}OPENAI_A",
    )
    rendered = _documents(
        render_advisor(
            ADVISOR_TEMPLATE.read_text(encoding="utf-8"),
            tag,
            [],
            f"senpai-launch-secrets-{tag}",
            launch_secret,
            args,
        )
    )
    config = next(document for document in rendered if document["kind"] == "ConfigMap")
    state_dir = f"/var/lib/senpai/{tag}/advisor/openhands_state"
    config["data"].update(
        {
            "SENPAI_ROLE": "advisor",
            "SENPAI_OPENHANDS_STATE_DIR": state_dir,
        }
    )
    deployment = next(
        document for document in rendered if document["kind"] == "Deployment"
    )
    deployment["metadata"]["namespace"] = namespace
    pod = deployment["spec"]["template"]["spec"]
    if {container["name"] for container in pod["containers"]} != {
        "advisor",
        "repair",
    }:
        raise RuntimeError(
            "production advisor must contain the advisor and secret-free repair "
            "containers before the Kubernetes canary can run"
        )
    advisor = _container(pod, "advisor")
    advisor["command"] = ["python", "/opt/senpai/tests/kubernetes/canary.py"]
    advisor["args"] = ["role-owner", "--state-dir", state_dir]
    advisor["resources"] = {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"},
    }
    for probe_name in ("startupProbe", "livenessProbe"):
        probe = advisor[probe_name]
        probe["periodSeconds"] = 1
        probe["timeoutSeconds"] = 1
        probe["failureThreshold"] = 30
    repair = _container(pod, "repair")
    repair["resources"] = {
        "requests": {"cpu": "50m", "memory": "64Mi"},
        "limits": {"cpu": "500m", "memory": "256Mi"},
    }
    for document in rendered:
        document.setdefault("metadata", {}).setdefault("namespace", namespace)
    return rendered


def _render_supervisor(
    namespace: str,
    tag: str,
    image: str,
    revision: str,
    *,
    broken: bool,
) -> list[dict]:
    from k8s.launch import SUPERVISOR_TEMPLATE, render_operational_supervisor
    from k8s.launch_helpers import render_supervisor_secret

    args = _args(namespace, tag, image, revision)
    if broken:
        args.supervisor_interval_s += 1
    suffix = "B" if broken else "A"
    secret_name, secret = render_supervisor_secret(
        tag,
        f"{DUMMY}GITHUB_{suffix}",
        f"{DUMMY}WANDB_{suffix}",
        provider_secret_name="openai-api-key",
        provider_api_key=f"{DUMMY}OPENAI_{suffix}",
    )
    rendered = _documents(
        secret
        + "\n---\n"
        + render_operational_supervisor(
            SUPERVISOR_TEMPLATE.read_text(encoding="utf-8"),
            tag,
            [],
            secret_name,
            secret,
            args,
        )
    )
    deployment = next(
        document for document in rendered if document["kind"] == "Deployment"
    )
    pod = deployment["spec"]["template"]["spec"]
    if {container["name"] for container in pod["containers"]} != {
        "supervisor-control",
        "supervisor-shell",
    }:
        raise RuntimeError(
            "production supervisor must contain isolated control and shell "
            "containers before the Kubernetes canary can run"
        )
    source = next(
        container for container in pod["initContainers"] if container["name"] == "source"
    )
    source["command"] = ["/bin/bash", "-c"]
    source["args"] = [
        "set -eu; cp -a /opt/senpai/. /workspace/senpai; "
        "test -f /workspace/senpai/tests/kubernetes/canary.py"
    ]
    source.pop("env", None)
    source.pop("envFrom", None)
    control = _container(pod, "supervisor-control")
    if broken:
        control["command"] = ["/bin/bash", "-c"]
        control["args"] = ["exit 42"]
        control.pop("startupProbe", None)
        control.pop("livenessProbe", None)
    else:
        control["command"] = ["python", "/opt/senpai/tests/kubernetes/canary.py"]
        control["args"] = [
            "supervisor-control",
            "--namespace",
            namespace,
            "--tag",
            tag,
        ]
        socket_probe = {
            "exec": {
                "command": [
                    "/bin/sh",
                    "-c",
                    "test -S /run/senpai-repair/repair.sock",
                ]
            },
            "periodSeconds": 1,
            "timeoutSeconds": 1,
            "failureThreshold": 30,
        }
        control["startupProbe"] = socket_probe
        control["livenessProbe"] = {**socket_probe, "failureThreshold": 5}
    control["resources"] = {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"},
    }
    shell = _container(pod, "supervisor-shell")
    shell["resources"] = {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"},
    }
    for document in rendered:
        document.setdefault("metadata", {}).setdefault("namespace", namespace)
    return rendered


def _decoy(other_namespace: str, tag: str, image: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"senpai-decoy-{tag}",
            "namespace": other_namespace,
            "labels": {
                "app": "senpai",
                "role": "advisor",
                "research-tag": tag,
            },
        },
        "spec": {
            "automountServiceAccountToken": False,
            "containers": [
                {
                    "name": "decoy",
                    "image": image,
                    "command": ["/bin/sh", "-c"],
                    "args": ["exec sleep infinity"],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        },
    }


def render_manifest(args: argparse.Namespace) -> int:
    from k8s.launch_helpers import render_launch_secret

    launch_secret = _documents(
        render_launch_secret(
            args.tag,
            f"{DUMMY}GITHUB_A",
            f"{DUMMY}EXA_A",
            f"{DUMMY}WANDB_A",
            openai_api_key=f"{DUMMY}OPENAI_A",
        )
    )[0]
    launch_secret["metadata"]["namespace"] = args.namespace
    if args.phase == "initial":
        volume, claim = _storage(args.namespace, args.tag)
        documents = [
            _namespace(args.namespace),
            _namespace(args.other_namespace),
            volume,
            claim,
            launch_secret,
            *_render_advisor(
                args.namespace, args.tag, args.image, args.revision
            ),
            _decoy(args.other_namespace, args.tag, args.image),
            *_render_supervisor(
                args.namespace,
                args.tag,
                args.image,
                args.revision,
                broken=False,
            ),
        ]
    else:
        documents = _render_supervisor(
            args.namespace,
            args.tag,
            args.image,
            args.revision,
            broken=True,
        )
    yaml.safe_dump_all(documents, sys.stdout, sort_keys=False)
    return 0


def fake_worker(args: argparse.Namespace) -> int:
    del args
    from senpai_agent.supervisor import GENERATION_ENV, LEASE_ENV, ProgressLease

    lease = ProgressLease(
        Path(os.environ[LEASE_ENV]),
        generation=int(os.environ[GENERATION_ENV]),
    )
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stop:
        lease.update(
            "sleep",
            60,
            conversation_id=str(CANARY_CONVERSATION_ID),
        )
        time.sleep(0.2)
    return 0


def role_owner(args: argparse.Namespace) -> int:
    from senpai_agent.supervisor import SupervisorConfig, WorkerSupervisor

    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / "canary-state-marker"
    marker.write_text("owner-state-preserved\n", encoding="utf-8")
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "fake-worker",
        "senpai_agent.controller",
        "advisor",
    )
    return WorkerSupervisor(
        command=command,
        lease_path=state_dir / "controller-lease.json",
        config=SupervisorConfig(
            startup_timeout_seconds=10,
            check_interval_seconds=0.2,
            terminate_grace_seconds=1,
            initial_backoff_seconds=0.2,
            max_backoff_seconds=1,
        ),
        environment=os.environ,
    ).run()


def supervisor_control(args: argparse.Namespace) -> int:
    from senpai_agent.kubernetes_operations import KubectlCampaignBackend
    from senpai_agent.operations import CampaignInventory
    from senpai_agent.repair_broker import RepairBrokerServer

    inventory = CampaignInventory(
        research_tag=args.tag,
        repo="senpai/canary",
        advisor_branch="ci-advisor",
        students=(),
    )
    backend = KubectlCampaignBackend(
        inventory,
        namespace=args.namespace,
        environment=os.environ,
    )
    state_marker = (
        Path("/var/lib/senpai")
        / args.tag
        / "operational-supervisor"
        / "canary-state-marker"
    )
    state_marker.parent.mkdir(parents=True, exist_ok=True)
    state_marker.write_text("supervisor-state-preserved\n", encoding="utf-8")
    socket_path = Path(os.environ["SENPAI_SUPERVISOR_REPAIR_SOCKET"])
    with RepairBrokerServer(socket_path, inventory, backend):
        stop = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stop:
            time.sleep(0.2)
    return 0


def probe_control(args: argparse.Namespace) -> int:
    from senpai_agent.kubernetes_operations import KubectlCampaignBackend
    from senpai_agent.operations import CampaignInventory, RoleTarget

    inventory = CampaignInventory(
        research_tag=args.tag,
        repo="senpai/canary",
        advisor_branch="ci-advisor",
        students=(),
    )
    backend = KubectlCampaignBackend(
        inventory,
        namespace=args.namespace,
        environment=os.environ,
    )
    target = RoleTarget(research_tag=args.tag, role="advisor")
    before = backend.collect_role(target)
    if (
        before.controller_alive is not True
        or before.controller_phase != "sleep"
        or before.worker_generation != 1
        or before.conversation_id != CANARY_CONVERSATION_ID
        or before.restart_control_token is None
    ):
        raise RuntimeError(f"role was not restartable: {before.model_dump_json()}")
    receipt = backend.restart_controller(
        target,
        expected_conversation_id=CANARY_CONVERSATION_ID,
        restart_control_token=before.restart_control_token,
    )
    deadline = time.monotonic() + args.timeout
    after = before
    while time.monotonic() < deadline:
        after = backend.collect_role(target)
        if after.worker_generation == 2 and after.controller_alive is True:
            break
        time.sleep(0.25)
    else:
        raise RuntimeError(
            "owner did not replace the controller worker: "
            f"{after.model_dump_json()}"
        )
    if after.conversation_id != CANARY_CONVERSATION_ID:
        raise RuntimeError("owner restart changed the conversation identity")
    print(
        json.dumps(
            {
                "request_id": receipt.request_id,
                "source_generation": before.worker_generation,
                "replacement_generation": after.worker_generation,
                "conversation_id": str(after.conversation_id),
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render")
    render.add_argument("--phase", choices=("initial", "broken-upgrade"), required=True)
    render.add_argument("--namespace", required=True)
    render.add_argument("--other-namespace", required=True)
    render.add_argument("--tag", required=True)
    render.add_argument("--image", required=True)
    render.add_argument("--revision", required=True)
    render.set_defaults(handler=render_manifest)
    owner = commands.add_parser("role-owner")
    owner.add_argument("--state-dir", type=Path, required=True)
    owner.set_defaults(handler=role_owner)
    worker = commands.add_parser("fake-worker")
    worker.add_argument("markers", nargs="*")
    worker.set_defaults(handler=fake_worker)
    control = commands.add_parser("supervisor-control")
    control.add_argument("--namespace", required=True)
    control.add_argument("--tag", required=True)
    control.set_defaults(handler=supervisor_control)
    probe = commands.add_parser("probe-control")
    probe.add_argument("--namespace", required=True)
    probe.add_argument("--tag", required=True)
    probe.add_argument("--timeout", type=float, default=30)
    probe.set_defaults(handler=probe_control)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
