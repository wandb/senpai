# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from types import SimpleNamespace
from unittest.mock import patch

from senpai.launch.kubernetes_backend import launch_kubernetes


def test_existing_workers_keep_running_after_secret_update(capsys):
    args = SimpleNamespace(tag="paper-r1", dry_run=False)

    with (
        patch(
            "senpai.launch.kubernetes_backend.existing_deployment_names",
            return_value=["deployment.apps/senpai-paper-r1-fern"],
        ),
        patch("senpai.launch.kubernetes_backend.kubectl_apply") as apply,
    ):
        launch_kubernetes(args, [], {"GITHUB_TOKEN": "new-secret"})

    output = capsys.readouterr().out
    assert "will not restart them automatically" in output
    assert "could interrupt long-running training" in output
    assert "kubectl rollout restart deployment -l research-tag=paper-r1" in output
    apply.assert_called_once()


def test_new_launch_does_not_print_rotation_warning(capsys):
    args = SimpleNamespace(tag="paper-r1", dry_run=False)

    with (
        patch(
            "senpai.launch.kubernetes_backend.existing_deployment_names",
            return_value=[],
        ),
        patch("senpai.launch.kubernetes_backend.kubectl_apply"),
    ):
        launch_kubernetes(args, [], {"GITHUB_TOKEN": "secret"})

    assert "WARNING" not in capsys.readouterr().out
