from pathlib import Path

import pytest
import yaml

from launch_test_support import (
    launch,
    launch_args,
    launch_helpers,
    render_role,
)


def render_student(nodes=2, **overrides):
    args = launch_args(
        advisor=False,
        nodes_per_student=nodes,
        gpus_per_student_node=8,
        memory_gi_per_gpu=110,
        **overrides,
    )
    secret_name = f"senpai-launch-secrets-{args.tag}"
    secret = launch_helpers.render_launch_secret(
        args.tag,
        "github",
        "exa",
        "wandb",
        openai_api_key="openai",
        custom_secrets={},
    )
    return list(
        yaml.safe_load_all(
            launch.render_student(
                (Path(__file__).parents[1] / "k8s" / "student-deployment.yaml").read_text(),
                "fern",
                args.tag,
                secret_name,
                secret,
                args,
            )
        )
    )


@pytest.mark.parametrize("nodes", [1, 2])
def test_student_controller_is_cpu_only_with_a_credential_isolated_executor(nodes):
    configmap, service_account, role, role_binding, deployment = render_student(nodes)
    pod = deployment["spec"]["template"]["spec"]
    containers = {container["name"]: container for container in pod["containers"]}
    student = containers["student"]
    executor = containers["kubernetes-executor"]

    assert pod["nodeSelector"] == {}
    assert pod["tolerations"] == []
    assert pod["automountServiceAccountToken"] is False
    assert pod["serviceAccountName"] == service_account["metadata"]["name"]
    assert student["resources"] == {
        "requests": {"cpu": "2", "memory": "8Gi"},
        "limits": {"cpu": "4", "memory": "16Gi"},
    }
    assert executor["resources"] == {
        "requests": {"cpu": "250m", "memory": "256Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"},
    }
    assert "nvidia.com/gpu" not in str(student["resources"])
    assert "nvidia.com/gpu" not in str(executor["resources"])

    student_mounts = {mount["name"] for mount in student["volumeMounts"]}
    executor_mounts = {mount["name"] for mount in executor["volumeMounts"]}
    assert "executor-token" not in student_mounts
    assert "executor-state" not in student_mounts
    assert {"executor-token", "executor-state", "executor-socket"} <= executor_mounts
    token = next(volume for volume in pod["volumes"] if volume["name"] == "executor-token")
    assert "serviceAccountToken" in token["projected"]["sources"][0]

    assert service_account["automountServiceAccountToken"] is False
    assert role_binding["roleRef"]["name"] == role["metadata"]["name"]
    assert all("secrets" not in rule["resources"] for rule in role["rules"])
    event_rules = [rule for rule in role["rules"] if "events" in rule["resources"]]
    assert event_rules == [{"apiGroups": [""], "resources": ["events"], "verbs": ["list"]}]
    workloads = {
        resource: rule["verbs"]
        for rule in role["rules"]
        for resource in rule["resources"]
        if rule["apiGroups"] in (["batch"], ["kubeflow.org"])
    }
    assert set(workloads) == ({"jobs", "mpijobs"} if nodes > 1 else {"jobs"})
    assert all(verbs == ["create", "get", "patch", "delete"] for verbs in workloads.values())
    assert configmap["data"]["NODES_PER_STUDENT"] == str(nodes)
    assert configmap["data"]["GPUS_PER_STUDENT_NODE"] == "8"
    assert configmap["data"]["CPU_PER_STUDENT_GPU"] == "15"
    assert configmap["data"]["MEMORY_GI_PER_STUDENT_GPU"] == "110"


def test_controller_image_has_only_the_validated_kubectl_socket_proxy():
    root = Path(__file__).parents[1]
    dockerfile = (root / "Dockerfile.student").read_text()
    entrypoint = (root / "k8s" / "entrypoint-student.sh").read_text()

    assert "kubectl" not in dockerfile
    assert "senpai_agent.kubernetes_executor kubectl" in entrypoint
    assert 'export PATH="$proxy_dir:$PATH"' in entrypoint


def test_advisor_placement_is_portable_by_default():
    template = (Path(__file__).parents[1] / "k8s" / "advisor-deployment.yaml").read_text()
    rendered = launch_helpers.render_template(
        template,
        {
            token: "fixture"
            for token in (
                "ADVISOR_DEPLOYMENT_NAME",
                "ADVISOR_CONFIGMAP_NAME",
                "RESEARCH_TAG",
                "ADVISOR_IMAGE",
                "PVC_CLAIM_NAME",
                "PVC_MOUNT_PATH",
                "LAUNCH_SECRET_NAME",
                "POD_CONFIG_HASH",
                "CONTROLLER_NODE_SELECTOR",
            )
        }
        | {
            "MODEL_PROVIDER_ENV": "        - name: MODEL_API_KEY",
            "CUSTOM_SECRET_ENV_REFS": "",
            "CONTROLLER_NODE_SELECTOR": "{}",
        },
    )
    pod = yaml.safe_load(rendered)["spec"]["template"]["spec"]

    assert pod["nodeSelector"] == {}


@pytest.mark.parametrize("nodes", [1, 2])
def test_controller_node_selector_applies_to_cpu_only_roles(nodes):
    selector = ["compute.coreweave.com/node-pool=cpu"]
    *_, student = render_student(nodes, controller_node_selector=selector)
    assert student["spec"]["template"]["spec"]["nodeSelector"] == {
        "compute.coreweave.com/node-pool": "cpu"
    }

    _, advisor_yaml, _ = render_role(
        "advisor",
        launch_args(controller_node_selector=selector),
    )
    advisor = yaml.safe_load(advisor_yaml)
    assert advisor["spec"]["template"]["spec"]["nodeSelector"] == {
        "compute.coreweave.com/node-pool": "cpu"
    }
