# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from types import SimpleNamespace
from unittest.mock import patch

import yaml

from senpai.launch.docker_backend import launch_docker, render_compose
from senpai.launch.specs import RoleSpec


def _args(tmp_path, **overrides):
    values = {
        "tag": "paper-r1",
        "image": "ghcr.io/wandb/senpai:latest",
        "gpus_per_student": 1,
        "docker_gpu_ids": "3,7",
        "docker_dataset_path": "",
        "docker_run_root": str(tmp_path),
        "pvc_mount_path": "/mnt/data",
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _roles():
    secrets = {
        "GITHUB_TOKEN": "github-secret",
        "WANDB_API_KEY": "wandb-secret",
    }
    return [
        RoleSpec("student", "fern", {"STUDENT_NAME": "fern"}, secrets),
        RoleSpec("student", "frieren", {"STUDENT_NAME": "frieren"}, secrets),
        RoleSpec("advisor", "advisor", {"STUDENT_NAMES": "fern,frieren"}, secrets),
    ]


def test_compose_uses_native_secrets_without_serializing_values(tmp_path):
    rendered = render_compose(_args(tmp_path), _roles())
    compose = yaml.safe_load(rendered)

    assert "github-secret" not in rendered
    assert "wandb-secret" not in rendered
    assert "$${secret_name}" in rendered
    assert compose["secrets"] == {
        "GITHUB_TOKEN": {"environment": "GITHUB_TOKEN"},
        "WANDB_API_KEY": {"environment": "WANDB_API_KEY"},
    }
    assert compose["services"]["student-fern"]["secrets"] == [
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
    ]
    fern_device = compose["services"]["student-fern"]["deploy"]["resources"][
        "reservations"
    ]["devices"][0]
    frieren_device = compose["services"]["student-frieren"]["deploy"]["resources"][
        "reservations"
    ]["devices"][0]
    assert fern_device["device_ids"] == ["3"]
    assert frieren_device["device_ids"] == ["7"]
    assert "deploy" not in compose["services"]["advisor"]


def test_docker_launch_passes_secrets_only_to_compose_process(tmp_path):
    args = _args(tmp_path)

    with patch("senpai.launch.docker_backend.subprocess.run") as run:
        launch_docker(args, _roles())

    command = run.call_args.args[0]
    child_env = run.call_args.kwargs["env"]
    compose_path = tmp_path / args.tag / "compose.yaml"
    assert command[-3:] == ["up", "--detach", "--force-recreate"]
    assert child_env["GITHUB_TOKEN"] == "github-secret"
    assert child_env["WANDB_API_KEY"] == "wandb-secret"
    assert "github-secret" not in compose_path.read_text()
