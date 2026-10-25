import pytest
import yaml

from launch_test_support import launch, launch_args, render_role

GPU_TOLERATION = {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
ENDPOINT_HOST = "inference.example.test"


def pod_template(role: str, args: launch.Args | None = None) -> dict:
    _configmap, deployment, _secret = render_role(role, args)
    return yaml.safe_load(deployment)["spec"]["template"]


def test_default_pods_leave_scheduling_and_name_resolution_to_the_cluster():
    for role in ("advisor", "student"):
        template = pod_template(role)
        assert "schedulerName" not in template["spec"]
        assert "hostAliases" not in template["spec"]
        assert template["spec"]["nodeSelector"] == {}
        assert set(template["metadata"]["labels"]) <= launch.SENPAI_POD_LABEL_KEYS


def test_single_node_students_use_their_scheduler_and_node_selector():
    args = launch_args(
        student_node_selector=[
            "gpu.nvidia.com/class=H200",
            "compute.coreweave.com/node-pool=gpu-pool",
        ],
        student_scheduler_name="binpack-scheduler",
    )

    student = pod_template("student", args)["spec"]
    advisor = pod_template("advisor", args)["spec"]

    assert student["schedulerName"] == "binpack-scheduler"
    assert student["nodeSelector"] == {
        "gpu.nvidia.com/class": "H200",
        "compute.coreweave.com/node-pool": "gpu-pool",
    }
    assert student["tolerations"] == [GPU_TOLERATION]
    assert student["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "schedulerName" not in advisor
    assert advisor["nodeSelector"] == {}


def test_remote_training_hands_gpu_placement_to_the_executor():
    args = launch_args(
        advisor=False,
        training_image=f"docker.io/acme/trainer@sha256:{'c' * 64}",
        executor_image=f"ghcr.io/wandb/senpai-executor:sha-{'a' * 40}",
        controller_node_selector=["compute.coreweave.com/node-pool=cpu-pool"],
        student_node_selector=["gpu.nvidia.com/class=H200"],
        student_scheduler_name="binpack-scheduler",
    )
    configmap, deployment, _secret = render_role("student", args)
    config = yaml.safe_load(configmap)["data"]
    controller = yaml.safe_load_all(deployment)
    pod = [document for document in controller if document["kind"] == "Deployment"][
        0
    ]["spec"]["template"]["spec"]

    assert config["SENPAI_STUDENT_NODE_SELECTOR"] == '{"gpu.nvidia.com/class": "H200"}'
    assert config["SENPAI_STUDENT_SCHEDULER_NAME"] == "binpack-scheduler"
    assert pod["nodeSelector"] == {"compute.coreweave.com/node-pool": "cpu-pool"}
    assert "schedulerName" not in pod


def test_single_node_students_do_not_configure_the_executor():
    configmap, _deployment, _secret = render_role(
        "student",
        launch_args(student_node_selector=["gpu.nvidia.com/class=H200"]),
    )

    assert "SENPAI_STUDENT_NODE_SELECTOR" not in yaml.safe_load(configmap)["data"]


def test_student_node_selector_requires_key_value_pairs():
    with pytest.raises(ValueError, match="student node selectors must use key=value"):
        launch.student_gpu_node_selector(launch_args(student_node_selector=["H200"]))


def test_agent_pod_labels_and_host_aliases_apply_to_both_roles():
    args = launch_args(
        agent_pod_labels=["access.example.test/model=enabled", "feature=true"],
        agent_host_aliases=[
            f"192.0.2.10={ENDPOINT_HOST}",
            "192.0.2.10=relay.example.test",
            "192.0.2.11=second.example.test",
        ],
    )

    for role in ("advisor", "student"):
        template = pod_template(role, args)
        labels = template["metadata"]["labels"]
        assert labels["role"] == role
        assert labels["access.example.test/model"] == "enabled"
        assert labels["feature"] == "true"
        assert template["spec"]["hostAliases"] == [
            {"ip": "192.0.2.10", "hostnames": [ENDPOINT_HOST, "relay.example.test"]},
            {"ip": "192.0.2.11", "hostnames": ["second.example.test"]},
        ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"agent_pod_labels": ["role=gpu"]}, "cannot replace: role"),
        ({"agent_pod_labels": ["enabled"]}, "agent pod labels must use key=value"),
        ({"agent_pod_labels": ["a=b", "a=c"]}, "duplicate agent pod label: a"),
    ],
)
def test_agent_pod_labels_fail_clearly(overrides, message):
    with pytest.raises(ValueError, match=message):
        launch.agent_pod_labels(launch_args(**overrides))


def test_agent_host_aliases_require_an_ip_and_hostname():
    with pytest.raises(ValueError, match="ip=hostname"):
        launch.agent_host_aliases(launch_args(agent_host_aliases=["192.0.2.10"]))
