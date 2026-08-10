#!/usr/bin/env python3
"""Render and exercise a credential-free Kubernetes production canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANARY_CONVERSATION_ID = UUID("11111111-1111-4111-8111-111111111111")
DUMMY = "SENPAI_CI_DUMMY_"
INTERRUPTED_REPAIR_COMMAND = (
    "printf interrupted-repair-ran > "
    "/repair/scratch/canary-interrupted-repair-ran"
)
ROLE_LISTENER_PORT = 18_765
ROLE_CONTROL_DIR = "/run/senpai-supervisor-control/private"


def _documents(text: str) -> list[dict]:
    return [document for document in yaml.safe_load_all(text) if document]


def _container(pod: dict, name: str) -> dict:
    try:
        return next(item for item in pod["containers"] if item["name"] == name)
    except StopIteration as error:
        raise RuntimeError(
            f"production manifest is missing required container {name!r}"
        ) from error


def _use_embedded_source(pod: dict, *, render_advisor_guidance: bool = False) -> None:
    """Make the canary's production source init local and credential-free."""

    source = next(
        container for container in pod["initContainers"] if container["name"] == "source"
    )
    source["command"] = ["/bin/bash", "-c"]
    command = (
        "set -eu; cp -R /opt/senpai/. /workspace/senpai; "
        "test -f /workspace/senpai/tests/kubernetes/canary.py"
    )
    if render_advisor_guidance:
        command += (
            "; mkdir -p /workspace/senpai/.senpai; "
            "envsubst '$PROBLEM_DIR $TARGET_REPO_URL $GH_REPO $ADVISOR_BRANCH "
            "$RESEARCH_TAG $GPUS_PER_STUDENT $WANDB_ENTITY $WANDB_PROJECT' "
            "< /workspace/senpai/system_instructions/ADVISOR.md "
            "> \"$SENPAI_IMMUTABLE_ADVISOR_GUIDANCE_FILE\"; "
            "chmod 0444 \"$SENPAI_IMMUTABLE_ADVISOR_GUIDANCE_FILE\""
        )
    source["args"] = [command]
    source.pop("env", None)


def _namespace(name: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": name, "labels": {"senpai-ci-canary": "true"}},
    }


def _storage(
    namespace: str,
    tag: str,
    *,
    name: str,
    host_suffix: str,
    sqlite_safe: bool = False,
) -> tuple[dict, dict]:
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
                "hostPath": {"path": f"/var/senpai-ci/{tag}/{host_suffix}"},
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
                **(
                    {
                        "annotations": {
                            "senpai.wandb.com/sqlite-safe": "true"
                        }
                    }
                    if sqlite_safe
                    else {}
                ),
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {"requests": {"storage": "1Gi"}},
                "storageClassName": "",
                "volumeName": name,
            },
        },
    )


def _args(
    namespace: str,
    tag: str,
    advisor_image: str,
    student_image: str,
    revision: str,
):
    from k8s.launch import Args

    return Args(
        tag=tag,
        target_repo_url="https://github.com/senpai-ci/canary.git",
        target_repo_branch="main",
        backend="kubernetes",
        namespace=namespace,
        repo_url="https://github.com/wandb/senpai.git",
        repo_revision=revision,
        advisor_image=advisor_image,
        student_image=student_image,
        advisor_branch="ci-advisor",
        advisor=True,
        operational_supervisor=True,
        supervisor_dedicated_namespace=True,
        supervisor_network_policy_enforced=True,
        supervisor_state_pvc_claim_name=f"senpai-ci-{tag}-supervisor-state",
        pvc_claim_name=f"senpai-ci-{tag}",
        pvc_mount_path="/mnt/senpai-ci",
        n_students=0,
    )


def _render_advisor(
    namespace: str,
    tag: str,
    advisor_image: str,
    student_image: str,
    revision: str,
) -> list[dict]:
    from k8s.launch import ADVISOR_TEMPLATE, render_advisor
    from k8s.launch_helpers import render_launch_secret

    args = _args(namespace, tag, advisor_image, student_image, revision)
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
            "SENPAI_OPENHANDS_ROLE_FILE": config["data"][
                "SENPAI_IMMUTABLE_ADVISOR_GUIDANCE_FILE"
            ],
            "SENPAI_SUPERVISOR_CONTROL_DIR": ROLE_CONTROL_DIR,
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
    _use_embedded_source(pod, render_advisor_guidance=True)
    advisor["command"] = ["python", "/opt/senpai/tests/kubernetes/canary.py"]
    advisor["args"] = [
        "role-owner",
        "--role",
        "advisor",
        "--state-dir",
        state_dir,
        "--target-dir",
        "/workspace/target",
    ]
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
    repair["resources"]["requests"].update(cpu="50m", memory="64Mi")
    repair["resources"]["limits"].update(cpu="500m", memory="256Mi")
    for document in rendered:
        document.setdefault("metadata", {}).setdefault("namespace", namespace)
    return rendered


def _render_student(
    namespace: str,
    tag: str,
    advisor_image: str,
    student_image: str,
    revision: str,
) -> list[dict]:
    from k8s.launch import STUDENT_TEMPLATE, render_student
    from k8s.launch_helpers import render_launch_secret

    args = _args(namespace, tag, advisor_image, student_image, revision)
    launch_secret = render_launch_secret(
        tag,
        f"{DUMMY}GITHUB_A",
        f"{DUMMY}EXA_A",
        f"{DUMMY}WANDB_A",
        openai_api_key=f"{DUMMY}OPENAI_A",
    )
    rendered = _documents(
        render_student(
            STUDENT_TEMPLATE.read_text(encoding="utf-8"),
            "fern",
            tag,
            f"senpai-launch-secrets-{tag}",
            launch_secret,
            args,
        )
    )
    config = next(document for document in rendered if document["kind"] == "ConfigMap")
    state_dir = "/var/lib/senpai/openhands_state"
    config["data"].update(
        {
            "SENPAI_ROLE": "student",
            "SENPAI_OPENHANDS_STATE_DIR": state_dir,
            "SENPAI_SUPERVISOR_CONTROL_DIR": ROLE_CONTROL_DIR,
        }
    )
    deployment = next(
        document for document in rendered if document["kind"] == "Deployment"
    )
    deployment["metadata"]["namespace"] = namespace
    pod = deployment["spec"]["template"]["spec"]
    student = _container(pod, "student")
    student["command"] = ["python", "/opt/senpai/tests/kubernetes/canary.py"]
    student["args"] = [
        "role-owner",
        "--role",
        "student",
        "--state-dir",
        state_dir,
        "--target-dir",
        "/workspace/target",
    ]
    student["resources"] = {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"},
    }
    for probe_name in ("startupProbe", "livenessProbe"):
        probe = student[probe_name]
        probe["periodSeconds"] = 1
        probe["timeoutSeconds"] = 1
        probe["failureThreshold"] = 30
    repair = _container(pod, "repair")
    repair["resources"]["requests"].update(cpu="50m", memory="64Mi")
    repair["resources"]["limits"].update(cpu="500m", memory="256Mi")
    pod.pop("tolerations", None)
    _use_embedded_source(pod)
    for document in rendered:
        document.setdefault("metadata", {}).setdefault("namespace", namespace)
    return rendered


def _render_supervisor(
    namespace: str,
    tag: str,
    advisor_image: str,
    student_image: str,
    revision: str,
    *,
    broken: bool,
) -> list[dict]:
    from k8s.launch import SUPERVISOR_TEMPLATE, render_operational_supervisor
    from k8s.launch_helpers import render_supervisor_secret

    args = _args(namespace, tag, advisor_image, student_image, revision)
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
            ["fern"],
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
    _use_embedded_source(pod)
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
    from k8s.launch import render_supervisor_network_policy
    from k8s.launch_helpers import render_launch_secret

    advisor_image = args.advisor_image or args.image
    student_image = args.student_image or args.image
    if not advisor_image or not student_image:
        raise ValueError(
            "render requires --image or both --advisor-image and --student-image"
        )
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
        volume, claim = _storage(
            args.namespace,
            args.tag,
            name=f"senpai-ci-{args.tag}",
            host_suffix="dataset",
        )
        state_volume, state_claim = _storage(
            args.namespace,
            args.tag,
            name=f"senpai-ci-{args.tag}-supervisor-state",
            host_suffix="supervisor-state",
            sqlite_safe=True,
        )
        documents = [
            _namespace(args.namespace),
            _namespace(args.other_namespace),
            volume,
            claim,
            state_volume,
            state_claim,
            _documents(render_supervisor_network_policy(args.tag))[0],
            launch_secret,
            *_render_advisor(
                args.namespace,
                args.tag,
                advisor_image,
                student_image,
                args.revision,
            ),
            *_render_student(
                args.namespace,
                args.tag,
                advisor_image,
                student_image,
                args.revision,
            ),
            _decoy(args.other_namespace, args.tag, advisor_image),
            *_render_supervisor(
                args.namespace,
                args.tag,
                advisor_image,
                student_image,
                args.revision,
                broken=False,
            ),
        ]
        next(
            document
            for document in documents
            if document["kind"] == "NetworkPolicy"
        )["metadata"]["namespace"] = args.namespace
    else:
        documents = _render_supervisor(
            args.namespace,
            args.tag,
            advisor_image,
            student_image,
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
    generation = os.environ[GENERATION_ENV]
    listener_program = f"""
import os
import signal
import socket

signal.signal(signal.SIGTERM, signal.SIG_IGN)
body = os.environ[{GENERATION_ENV!r}].encode()
response = (
    b"HTTP/1.1 200 OK\\r\\nConnection: close\\r\\nContent-Length: "
    + str(len(body)).encode()
    + b"\\r\\n\\r\\n"
    + body
)
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", {ROLE_LISTENER_PORT}))
server.listen()
while True:
    connection, _ = server.accept()
    with connection:
        connection.recv(4096)
        connection.sendall(response)
"""
    listener = subprocess.Popen(
        (sys.executable, "-c", listener_program),
        env=os.environ,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if listener.poll() is not None:
            raise RuntimeError("role loopback listener exited during startup")
        try:
            with socket.create_connection(
                ("127.0.0.1", ROLE_LISTENER_PORT), timeout=0.2
            ) as connection:
                connection.sendall(b"GET / HTTP/1.1\r\nHost: canary\r\n\r\n")
                response = connection.recv(4096)
            if response.endswith(generation.encode()):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("role loopback listener did not become authoritative")
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
    target_dir = args.target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "canary-target-marker").write_text(
        f"{args.role}-target-workspace\n",
        encoding="utf-8",
    )
    if not (target_dir / ".git").is_dir():
        trusted_git = ["git", "-c", f"safe.directory={target_dir}"]
        subprocess.run(
            [*trusted_git, "init", "--initial-branch=main"],
            cwd=target_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [*trusted_git, "add", "canary-target-marker"],
            cwd=target_dir,
            check=True,
        )
        subprocess.run(
            [
                *trusted_git,
                "-c",
                "user.name=Senpai Canary",
                "-c",
                "user.email=canary@example.invalid",
                "commit",
                "-m",
                "Initialize canary target",
            ],
            cwd=target_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    (target_dir / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "Path('/tmp/senpai-sitecustomize-poisoned').write_text('poisoned')\n",
        encoding="utf-8",
    )
    fake_python = target_dir / "python"
    fake_python.write_text(
        "#!/bin/sh\nprintf poisoned > /tmp/senpai-path-poisoned\nexit 99\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    if args.role == "advisor":
        from openhands.sdk.event import MessageEvent
        from openhands.sdk.event.types import ROOT_PARENT_ID
        from openhands.sdk.llm import Message, TextContent

        (state_dir / "advisor-conversation-id").write_text(
            f"{CANARY_CONVERSATION_ID}\n",
            encoding="utf-8",
        )
        events_dir = state_dir / CANARY_CONVERSATION_ID.hex / "events"
        events_dir.mkdir(parents=True)
        event = MessageEvent(
            source="agent",
            parent_id=ROOT_PARENT_ID,
            llm_message=Message(
                role="assistant",
                content=[TextContent(text="Canary research remains mechanism-led.")],
            ),
        )
        (events_dir / f"event-00000-{event.id}.json").write_text(
            event.model_dump_json(exclude_none=True),
            encoding="utf-8",
        )
        (state_dir / CANARY_CONVERSATION_ID.hex / "base_state.json").write_text(
            json.dumps({"leaf_event_id": str(event.id)}),
            encoding="utf-8",
        )
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "fake-worker",
        "senpai_agent.controller",
        args.role,
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


def _supervisor_state_dir(tag: str) -> Path:
    return Path(f"/var/lib/senpai/{tag}/operational-supervisor")


def _campaign_inventory(tag: str):
    from senpai_agent.operations import CampaignInventory

    return CampaignInventory(
        research_tag=tag,
        repo="senpai/canary",
        advisor_branch="ci-advisor",
        students=("fern",),
    )


def _interrupted_action(tag: str, operation_key: str):
    from senpai_agent.operations import Nudge, RoleTarget

    return Nudge(
        operation_key=operation_key,
        target=RoleTarget(research_tag=tag, role="advisor"),
        incident_key="canary-interrupted-operation",
        anomaly_category="controller_failure",
        reason="Prove an interrupted operation is never replayed.",
        expected_conversation_id=CANARY_CONVERSATION_ID,
        message="This canary action must never be delivered.",
    )


def _action_fingerprint(action: object) -> str:
    encoded = json.dumps(
        action.model_dump(mode="json", exclude={"operation_key"}),  # type: ignore[attr-defined]
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def supervisor_control(args: argparse.Namespace) -> int:
    from senpai_agent.kubernetes_operations import KubectlCampaignBackend
    from senpai_agent.operations import OperationLedger
    from senpai_agent.repair_broker import RepairBrokerServer

    inventory = _campaign_inventory(args.tag)
    backend = KubectlCampaignBackend(
        inventory,
        namespace=args.namespace,
        environment=os.environ,
    )
    state_dir = _supervisor_state_dir(args.tag)
    state_dir.mkdir(parents=True, exist_ok=True)
    with OperationLedger(state_dir / "operations.sqlite3") as operation_ledger:
        operation_ledger.recover_interrupted()
    socket_path = Path(os.environ["SENPAI_SUPERVISOR_REPAIR_SOCKET"])
    with RepairBrokerServer(
        socket_path,
        inventory,
        backend,
        ledger_path=state_dir / "repair-operations.sqlite3",
    ):
        stop = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stop:
            time.sleep(0.2)
    return 0


def probe_terminal_wakes(args: argparse.Namespace) -> int:
    from openhands.tools.terminal import TerminalAction

    from senpai_agent.isolated_terminal import (
        IsolatedTerminalClientExecutor,
        StaleTerminalWake,
        begin_isolated_terminal_wake,
        end_isolated_terminal_wake,
    )

    socket_path = args.socket
    first_wake = "canary-terminal-wake-one"
    second_wake = "canary-terminal-wake-two"
    begin_isolated_terminal_wake(socket_path, first_wake)
    first = IsolatedTerminalClientExecutor(socket_path, first_wake)
    try:
        changed = first(
            TerminalAction(
                command=(
                    "mkdir -p nested; cd nested; "
                    "export SENPAI_CANARY_WAKE_LEAK=yes; "
                    "printf persisted > ../canary-terminal-workspace; "
                    "printf poisoned > \"$HOME/.gitconfig\"; "
                    "(sleep 0.7; printf survived > "
                    "../canary-terminal-process-survived) >/dev/null 2>&1 & pwd"
                )
            )
        )
        if "/workspaces/supervisor/nested" not in changed.text:
            raise RuntimeError("terminal wake did not retain its own shell state")
    finally:
        end_isolated_terminal_wake(socket_path, first_wake)

    try:
        first(TerminalAction(command="touch canary-stale-wake-ran"))
    except StaleTerminalWake:
        pass
    else:
        raise RuntimeError("retired terminal wake accepted another action")

    time.sleep(1)
    begin_isolated_terminal_wake(socket_path, second_wake)
    second = IsolatedTerminalClientExecutor(socket_path, second_wake)
    try:
        pristine = second(
            TerminalAction(
                command=(
                    "pwd; env; "
                    "test -f canary-terminal-workspace && "
                    "printf '\\nworkspace-persisted\\n'; "
                    "test ! -e canary-terminal-process-survived; "
                    "test ! -e canary-stale-wake-ran; "
                    "test ! -e \"$HOME/.gitconfig\" && printf 'home-clean\\n'; "
                    "test ! -e /var/run/secrets/kubernetes.io/serviceaccount/token"
                )
            )
        )
    finally:
        end_isolated_terminal_wake(socket_path, second_wake)

    required = ("/workspaces/supervisor", "workspace-persisted", "home-clean")
    if any(value not in pristine.text for value in required):
        raise RuntimeError("next terminal wake was not pristine and state-preserving")
    forbidden = (
        "SENPAI_CANARY_WAKE_LEAK",
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "KUBECONFIG",
        DUMMY,
    )
    if any(value in pristine.text for value in forbidden):
        raise RuntimeError("isolated terminal inherited control credentials or state")
    print("terminal-wake-isolation-ok")
    return 0


def drop_repair_reply(args: argparse.Namespace) -> int:
    from senpai_agent.operations import RoleTarget
    from senpai_agent.protocols import REPAIR_PROTOCOL_VERSION
    from senpai_agent.repair_broker import (
        RepairBrokerClient,
        RepairRequest,
    )
    from senpai_agent.socket_framing import encode_json_frame

    request = RepairRequest.create(
        operation_id=args.operation_id,
        target=RoleTarget(research_tag=args.tag, role="advisor"),
        command=(
            "counter=/repair/scratch/canary-lost-reply-count; "
            "value=0; test ! -f \"$counter\" || value=$(cat \"$counter\"); "
            "value=$((value + 1)); printf '%s' \"$value\" > \"$counter\"; "
            "printf '%s' \"$value\""
        ),
        timeout_seconds=10,
    )
    frame = encode_json_frame(
        {
            "protocol": REPAIR_PROTOCOL_VERSION,
            "operation": "execute",
            "request": request.model_dump(mode="json"),
        },
        max_bytes=2 * 1024 * 1024,
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(args.socket))
        connection.sendall(frame)

    client = RepairBrokerClient(args.socket)
    deadline = time.monotonic() + 20
    while True:
        status = client.status(args.operation_id)
        if status.status != "running":
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("lost-reply repair did not reach a durable outcome")
        time.sleep(0.1)
    if (
        status.status != "completed"
        or status.result is None
        or status.result.stdout != "1"
    ):
        raise RuntimeError(f"lost-reply repair had bad status: {status}")
    replay = client.execute(request)
    if replay.stdout != "1":
        raise RuntimeError("lost-reply repair replay executed the command twice")
    print(status.model_dump_json())
    return 0


def seed_interrupted_operations(args: argparse.Namespace) -> int:
    from senpai_agent.operations import OperationLedger, RoleTarget
    from senpai_agent.repair_broker import RepairLedger, RepairRequest

    state_dir = _supervisor_state_dir(args.tag)
    action = _interrupted_action(args.tag, args.operation_key)
    with OperationLedger(state_dir / "operations.sqlite3") as ledger:
        reservation = ledger.reserve(
            action,
            fingerprint=_action_fingerprint(action),
            cooldown_key=None,
            cooldown_seconds=0,
            now=datetime.now(UTC),
        )
        if not reservation.execute:
            raise RuntimeError("general interrupted operation was not newly reserved")

    request = RepairRequest.create(
        operation_id=args.repair_operation_id,
        target=RoleTarget(research_tag=args.tag, role="advisor"),
        command=INTERRUPTED_REPAIR_COMMAND,
        timeout_seconds=10,
    )
    with RepairLedger(state_dir / "repair-operations.sqlite3") as ledger:
        reservation = ledger.reserve(request)
        if not reservation.execute:
            raise RuntimeError("interrupted repair was not newly reserved")
    print("interrupted-operations-seeded")
    return 0


def assert_interrupted_operations(args: argparse.Namespace) -> int:
    from senpai_agent.operations import OperationLedger, OperationOutcomeUnknown
    from senpai_agent.operations import RoleTarget
    from senpai_agent.repair_broker import (
        RepairBrokerClient,
        RepairLedger,
        RepairOutcomeUnknown,
        RepairRequest,
    )

    state_dir = _supervisor_state_dir(args.tag)
    action = _interrupted_action(args.tag, args.operation_key)
    with OperationLedger(state_dir / "operations.sqlite3") as ledger:
        record = next(
            item for item in ledger.records() if item.operation_key == args.operation_key
        )
        if record.status != "unknown" or record.error_type != "SupervisorInterrupted":
            raise RuntimeError(f"general operation was not sealed unknown: {record}")
        try:
            ledger.reserve(
                action,
                fingerprint=_action_fingerprint(action),
                cooldown_key=None,
                cooldown_seconds=0,
                now=datetime.now(UTC),
            )
        except OperationOutcomeUnknown:
            pass
        else:
            raise RuntimeError("interrupted general operation was replayable")

    request = RepairRequest.create(
        operation_id=args.repair_operation_id,
        target=RoleTarget(research_tag=args.tag, role="advisor"),
        command=INTERRUPTED_REPAIR_COMMAND,
        timeout_seconds=10,
    )
    with RepairLedger(state_dir / "repair-operations.sqlite3") as ledger:
        status = ledger.status(args.repair_operation_id)
    if status.status != "unknown" or status.error_type != "BrokerInterrupted":
        raise RuntimeError(f"repair operation was not sealed unknown: {status}")
    try:
        RepairBrokerClient(args.socket).execute(request)
    except RepairOutcomeUnknown:
        pass
    else:
        raise RuntimeError("interrupted repair was replayable")
    print("interrupted-operations-sealed-unknown")
    return 0


def probe_control(args: argparse.Namespace) -> int:
    from senpai_agent.kubernetes_operations import KubectlCampaignBackend
    from senpai_agent.operations import CampaignInventory, RoleTarget

    inventory = CampaignInventory(
        research_tag=args.tag,
        repo="senpai/canary",
        advisor_branch="ci-advisor",
        students=("fern",),
    )
    backend = KubectlCampaignBackend(
        inventory,
        namespace=args.namespace,
        environment=os.environ,
    )
    research_tail = backend.collect_advisor_research_tail()
    if (
        "# Research Advisor" not in research_tail.advisor_guidance
        or research_tail.messages[-1].summary
        != "Canary research remains mechanism-led."
    ):
        raise RuntimeError("immutable advisor guidance was not collected faithfully")
    results = []
    for target in (
        RoleTarget(research_tag=args.tag, role="advisor"),
        RoleTarget(research_tag=args.tag, role="student", student="fern"),
    ):
        before = backend.collect_role(target)
        before_generation = before.worker_generation
        if (
            before.controller_alive is not True
            or before.controller_phase != "sleep"
            or before_generation is None
            or before.conversation_id != CANARY_CONVERSATION_ID
            or before.restart_control_token is None
        ):
            raise RuntimeError(
                f"role was not restartable: {before.model_dump_json()}"
            )
        receipt = backend.restart_controller(
            target,
            expected_conversation_id=CANARY_CONVERSATION_ID,
            restart_control_token=before.restart_control_token,
        )
        deadline = time.monotonic() + args.timeout
        after = before
        while time.monotonic() < deadline:
            after = backend.collect_role(target)
            if (
                after.worker_generation == before_generation + 1
                and after.controller_alive is True
            ):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(
                "owner did not replace the controller worker: "
                f"{after.model_dump_json()}"
            )
        if after.conversation_id != CANARY_CONVERSATION_ID:
            raise RuntimeError("owner restart changed the conversation identity")
        results.append(
            {
                "target": target.model_dump(mode="json"),
                "request_id": receipt.request_id,
                "source_generation": before.worker_generation,
                "replacement_generation": after.worker_generation,
                "conversation_id": str(after.conversation_id),
            }
        )
    print(json.dumps(results, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render")
    render.add_argument("--phase", choices=("initial", "broken-upgrade"), required=True)
    render.add_argument("--namespace", required=True)
    render.add_argument("--other-namespace", required=True)
    render.add_argument("--tag", required=True)
    render.add_argument("--image")
    render.add_argument("--advisor-image")
    render.add_argument("--student-image")
    render.add_argument("--revision", required=True)
    render.set_defaults(handler=render_manifest)
    owner = commands.add_parser("role-owner")
    owner.add_argument("--role", choices=("advisor", "student"), required=True)
    owner.add_argument("--state-dir", type=Path, required=True)
    owner.add_argument("--target-dir", type=Path, required=True)
    owner.set_defaults(handler=role_owner)
    worker = commands.add_parser("fake-worker")
    worker.add_argument("markers", nargs="*")
    worker.set_defaults(handler=fake_worker)
    control = commands.add_parser("supervisor-control")
    control.add_argument("--namespace", required=True)
    control.add_argument("--tag", required=True)
    control.set_defaults(handler=supervisor_control)
    terminal = commands.add_parser("probe-terminal-wakes")
    terminal.add_argument("--socket", default="@senpai-isolated-terminal")
    terminal.set_defaults(handler=probe_terminal_wakes)
    dropped = commands.add_parser("drop-repair-reply")
    dropped.add_argument("--socket", type=Path, required=True)
    dropped.add_argument("--tag", required=True)
    dropped.add_argument("--operation-id", required=True)
    dropped.set_defaults(handler=drop_repair_reply)
    seed = commands.add_parser("seed-interrupted-operations")
    seed.add_argument("--tag", required=True)
    seed.add_argument("--operation-key", required=True)
    seed.add_argument("--repair-operation-id", required=True)
    seed.set_defaults(handler=seed_interrupted_operations)
    interrupted = commands.add_parser("assert-interrupted-operations")
    interrupted.add_argument("--socket", type=Path, required=True)
    interrupted.add_argument("--tag", required=True)
    interrupted.add_argument("--operation-key", required=True)
    interrupted.add_argument("--repair-operation-id", required=True)
    interrupted.set_defaults(handler=assert_interrupted_operations)
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
