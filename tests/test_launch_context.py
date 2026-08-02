import base64

import pytest
import yaml

from launch_test_support import launch_args, render_role
from senpai.launch.specs import build_extra_instructions


def test_default_fleet_is_four_students_with_one_gpu_each():
    args = launch_args()

    assert args.n_students == 4
    assert args.gpus_per_student == 1


@pytest.mark.parametrize("backend", ["kubernetes", "docker", "aws"])
def test_launch_context_records_resolved_runtime_facts(backend):
    args = launch_args(
        tag="foil-run",
        advisor_branch="research-v2",
        target_repo_branch="main",
        gpus_per_student=3,
        timeout_minutes=12.5,
        max_epochs=7,
    )

    args.backend = backend
    context = build_extra_instructions(args, args.tag, ["fern", "frieren"])

    assert "resolved by the Senpai launcher" in context
    assert "override conflicting compute or run-limit claims" in context
    assert f"Compute backend: `{backend}`" in context
    assert "Visible GPUs per student: `3`" in context
    assert (
        "Hard limits for each training run: `12.5` minutes wall-clock\n"
        "  and `7` epochs"
    ) in context
    assert "research tag `foil-run`" in context
    assert "advisor branch `research-v2`" in context
    assert "target base branch `main`" in context
    assert "fern, frieren" in context
    assert "{{" not in context


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_each_role_receives_authoritative_launch_context(role):
    args = launch_args(
        gpus_per_student=2,
        timeout_minutes=20,
        max_epochs=9,
        extra_instructions="Prefer small, measurable experiments.",
    )

    configmap, _deployment, _secret = render_role(role, args)
    encoded = yaml.safe_load(configmap)["data"]["EXTRA_INSTRUCTIONS_B64"]
    context = base64.b64decode(encoded, validate=True).decode()

    assert "Compute backend: `kubernetes`" in context
    assert "Visible GPUs per student: `2`" in context
    assert (
        "Hard limits for each training run: `20` minutes wall-clock\n"
        "  and `9` epochs"
    ) in context
    assert context.endswith(
        "# Additional operator instructions\n\n"
        "Prefer small, measurable experiments."
    )
