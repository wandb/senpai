import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from launch_test_support import (
    ADVISOR_IMAGE,
    STUDENT_IMAGE,
    launch,
    launch_args,
    render_role,
    run_launch,
)
from test_cluster_cutoff import render_cutoff


EXECUTOR_IMAGE = f"ghcr.io/wandb/senpai-executor@sha256:{'b' * 64}"


def observer_args(**overrides):
    return launch_args(
        **{
            "capacity_observer": True,
            "nodes_per_student": 4,
            "gpus_per_student_node": 8,
            "cpu_per_gpu": 15,
            "memory_gi_per_gpu": 110,
            "executor_image": EXECUTOR_IMAGE,
            **overrides,
        }
    )


def observer_documents(args):
    return list(
        yaml.safe_load_all(
            launch.render_capacity_observer(
                tag=args.tag,
                namespace=args.namespace,
                image=args.executor_image,
                revision=args.senpai_repo_revision,
                config=launch.capacity_config(args),
                node_selector=launch.controller_node_selector(args.controller_node_selector),
            )
        )
    )


def test_observer_authority_is_read_only_except_its_named_snapshot():
    args = observer_args(namespace="research-a")
    documents = {document["kind"]: document for document in observer_documents(args)}
    service_account = documents["ServiceAccount"]
    name = service_account["metadata"]["name"]
    assert service_account["automountServiceAccountToken"] is False
    assert documents["ClusterRole"]["rules"] == [
        {"apiGroups": [""], "resources": ["nodes", "pods"], "verbs": ["list"]}
    ]
    assert documents["Role"]["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["configmaps"],
            "resourceNames": [documents["ConfigMap"]["metadata"]["name"]],
            "verbs": ["get", "update"],
        }
    ]
    assert documents["ConfigMap"]["data"] == {"snapshot.json": "{}"}
    for binding_kind, role_kind in [("ClusterRoleBinding", "ClusterRole"), ("RoleBinding", "Role")]:
        binding = documents[binding_kind]
        assert binding["roleRef"]["kind"] == role_kind
        assert binding["roleRef"]["name"] == documents[role_kind]["metadata"]["name"]
        assert binding["subjects"] == [{"kind": "ServiceAccount", "name": name, "namespace": "research-a"}]

    pod = documents["Deployment"]["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["serviceAccountName"] == name
    (container,) = pod["containers"]
    assert container["image"] == EXECUTOR_IMAGE
    assert container["command"] == ["python", "-m", "senpai_agent.cluster_capacity"]
    assert container["volumeMounts"] == [
        {
            "name": "observer-token",
            "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount",
            "readOnly": True,
        }
    ]
    (token_volume,) = pod["volumes"]
    assert token_volume["name"] == "observer-token"
    assert token_volume["projected"]["sources"][0]["serviceAccountToken"]["path"] == "token"
    assert "envFrom" not in container
    assert all("valueFrom" not in item for item in container["env"])


def test_observer_shape_uses_launch_budget_and_explicit_worker_placement():
    args = observer_args(
        controller_node_selector=["pool=cpu"],
        capacity_node_selector=["accelerator=A100"],
        capacity_tolerations=['{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}'],
        capacity_hpc_verification=True,
    )
    deployment = observer_documents(args)[-1]
    pod = deployment["spec"]["template"]["spec"]
    env = {item["name"]: item["value"] for item in pod["containers"][0]["env"]}
    config = json.loads(env["SENPAI_CAPACITY_CONFIG"])
    assert {key: config[key] for key in ["nodes", "gpus_per_node", "cpu_per_node", "memory_gib_per_node"]} == {
        "nodes": 4, "gpus_per_node": 8, "cpu_per_node": 120, "memory_gib_per_node": 880,
    }
    assert config["node_selector"] == {"accelerator": "A100"}
    assert config["tolerations"][0]["operator"] == "Exists"
    assert config["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert config["hpc_verification"] is True
    assert pod["nodeSelector"] == {"pool": "cpu"}
    assert deployment["metadata"]["labels"]["app"] == "senpai-capacity-observer"
    assert deployment["metadata"]["labels"]["role"] == "capacity-observer"
    assert deployment["metadata"]["labels"]["research-tag"] == args.tag


@pytest.mark.parametrize("role,nodes", [("advisor", 4), ("student", 4), ("student", 1)])
@pytest.mark.parametrize("enabled", [False, True])
def test_agent_gets_only_a_refreshable_read_only_snapshot(role, nodes, enabled):
    args = observer_args(nodes_per_student=nodes, capacity_observer=enabled)
    config, deployment_yaml, _secret = render_role(role, args)
    data = yaml.safe_load(config)["data"]
    documents = list(yaml.safe_load_all(deployment_yaml))
    pod = documents[-1]["spec"]["template"]["spec"]
    agent = next(container for container in pod["containers"] if container["name"] == role)
    mounts = {mount["name"]: mount for mount in agent["volumeMounts"]}
    assert pod["automountServiceAccountToken"] is False
    assert "observer-token" not in mounts
    assert "SENPAI_CAPACITY_CONFIG" not in data
    assert "SENPAI_CAPACITY_CONFIGMAP" not in data
    assert not any(document["kind"] in {"ClusterRole", "ClusterRoleBinding"} for document in documents)
    if enabled:
        assert data["SENPAI_CAPACITY_SNAPSHOT"] == "/var/run/senpai-capacity/snapshot.json"
        assert mounts["capacity-snapshot"] == {
            "name": "capacity-snapshot", "mountPath": "/var/run/senpai-capacity", "readOnly": True,
        }
        volume = next(volume for volume in pod["volumes"] if volume["name"] == "capacity-snapshot")
        assert volume["configMap"] == {
            "name": "senpai-capacity-test-track", "items": [{"key": "snapshot.json", "path": "snapshot.json"}],
        }
    else:
        assert "SENPAI_CAPACITY_SNAPSHOT" not in data
        assert "capacity-snapshot" not in mounts
        assert all(volume["name"] != "capacity-snapshot" for volume in pod["volumes"])


@pytest.mark.parametrize(
    "first_scope,second_scope",
    [
        (("research-a", "track"), ("research-b", "track")),
        (("a-b", "c"), ("a", "b-c")),
    ],
)
def test_observer_cluster_authority_is_scoped_to_namespace_and_tag(
    first_scope, second_scope
):
    first_namespace, first_tag = first_scope
    second_namespace, second_tag = second_scope
    first = {
        item["kind"]: item
        for item in observer_documents(observer_args(namespace=first_namespace, tag=first_tag))
    }
    second = {
        item["kind"]: item
        for item in observer_documents(observer_args(namespace=second_namespace, tag=second_tag))
    }
    for kind in ["ClusterRole", "ClusterRoleBinding"]:
        assert first[kind]["metadata"]["name"] != second[kind]["metadata"]["name"]
        assert first[kind]["metadata"]["labels"]["senpai.wandb.com/namespace"] == first_namespace
    assert first["ClusterRoleBinding"]["subjects"][0]["namespace"] == first_namespace
    assert second["ClusterRoleBinding"]["subjects"][0]["namespace"] == second_namespace


def test_opt_in_cli_renders_observer_once_and_requires_its_digest_for_single_node():
    arguments = [
        "--advisor_image", ADVISOR_IMAGE, "--student_image", STUDENT_IMAGE,
        "--capacity_observer", "--nodes_per_student", "1",
    ]
    missing = run_launch(*arguments)
    assert missing.returncode != 0
    assert "--executor_image must use an immutable @sha256 digest" in missing.stderr

    rendered = run_launch(*arguments, "--executor_image", EXECUTOR_IMAGE, "--senpai_repo_revision", "a" * 40)
    assert rendered.returncode == 0, rendered.stderr
    observer = rendered.stdout.split("--- Capacity observer ---\n", 1)[1].split("--- Student:", 1)[0]
    documents = list(yaml.safe_load_all(observer))
    assert sum(document["kind"] == "Deployment" for document in documents) == 1
    assert documents[-1]["spec"]["template"]["spec"]["containers"][0]["image"] == EXECUTOR_IMAGE


@pytest.mark.parametrize(
    "overrides",
    [
        {"capacity_node_selector": ["missing-value"]},
        {"capacity_tolerations": ["not-json"]},
        {"capacity_tolerations": ['{"operator":"unsupported"}']},
    ],
)
def test_capacity_operator_configuration_fails_before_rendering(overrides):
    with pytest.raises(ValueError):
        launch.capacity_config(observer_args(**overrides))


def test_cutoff_readiness_excludes_observer_but_deletion_includes_it(tmp_path: Path):
    generated, script = render_cutoff(
        tmp_path, "--run-slug", "capacity", "--tags-csv", "track-a",
        "--expected-pods", "1", "--expected-deployments", "1", "--budget-hours", "0",
    )
    assert generated.returncode == 0, generated.stderr
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "kubectl"
    fake.write_text('''#!/bin/sh
case "$*" in
  *"get pods"*"app=senpai,"*) printf '%s\\n' '{"items":[{"status":{"containerStatuses":[{"ready":true}]}}]}' ;;
  *"get deployments"*"app=senpai,"*) printf '%s\\n' 'senpai-track-a' ;;
  *"delete deployments,configmaps,secrets"*)
    case "$*" in *"app=senpai"*) exit 9 ;; esac
    printf '%s\\n' "$*" > "$DELETE_LOG" ;;
  *) exit 8 ;;
esac
''')
    fake.chmod(0o755)
    delete_log = tmp_path / "deleted"
    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=False, timeout=15,
        env=os.environ | {
            "PATH": f"{binary}:{os.environ['PATH']}", "DELETE_LOG": str(delete_log),
            "RUN_SLUG": "capacity", "TAGS_CSV": "track-a", "EXPECTED_PODS": "1",
            "EXPECTED_DEPLOYMENTS": "1", "READINESS_TIMEOUT_SECONDS": "0",
            "BUDGET_SECONDS": "0", "ARM_ID": "capacity-check",
            "PVC_LOG_ROOT": str(tmp_path / "state"), "NAMESPACE": "test-ns",
        },
    )
    assert result.returncode == 0, result.stderr
    assert "Ready gate passed" in result.stdout
    assert "research-tag in (track-a)" in delete_log.read_text()
