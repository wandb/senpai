# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import base64
import stat

import pytest
import yaml

from k8s.launch_helpers import render_launch_secret
from scripts.apply_scout_workflow import render_workflow
from senpai.launch.credentials import (
    REQUIRED_SECRET_NAMES,
    load_secrets,
    secret_names,
    workload_secrets,
)


def _dotenv(**overrides: str) -> str:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_SECRET_NAMES}
    values.update(overrides)
    return "\n".join(f'{name}="{value}"' for name, value in values.items()) + "\n"


def test_dotenv_is_the_complete_secret_source(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(_dotenv(CUSTOM_SERVICE_TOKEN="token#with#hashes"))
    path.chmod(0o644)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-shell-token")

    secrets = load_secrets(path)

    assert secrets["GITHUB_TOKEN"] == "value-for-github_token"
    assert secrets["CUSTOM_SERVICE_TOKEN"] == "token#with#hashes"
    assert set(REQUIRED_SECRET_NAMES) < set(secrets)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_dotenv_reports_every_missing_required_secret(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GITHUB_TOKEN=github\n")

    with pytest.raises(SystemExit) as error:
        load_secrets(path)

    message = str(error.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "EXA_API_KEY" in message
    assert "WANDB_API_KEY" in message


def test_scout_only_secrets_are_not_sent_to_workers():
    secrets = {
        "GITHUB_TOKEN": "github-secret",
        "CUSTOM_SERVICE_TOKEN": "custom-secret",
        "SLACK_WEBHOOK_URL": "slack-secret",
        "KUBECONFIG_B64": "cluster-secret",
        "SEMANTIC_SCHOLAR_API_KEY": "search-secret",
    }

    assert workload_secrets(secrets) == {
        "GITHUB_TOKEN": "github-secret",
        "CUSTOM_SERVICE_TOKEN": "custom-secret",
    }


def test_consumers_can_require_their_own_secret_set(tmp_path):
    path = tmp_path / ".env"
    path.write_text("SCOUT_TOKEN=scout-secret\n")

    assert load_secrets(path, required=("SCOUT_TOKEN",)) == {
        "SCOUT_TOKEN": "scout-secret"
    }


def test_dry_run_secret_names_work_without_dotenv(tmp_path):
    assert secret_names(tmp_path / ".env") == REQUIRED_SECRET_NAMES


def test_kubernetes_secret_preserves_env_names_and_values():
    secrets = {
        "GITHUB_TOKEN": "github-secret",
        "WANDB_API_KEY": "wandb-secret",
        "CUSTOM_SERVICE_TOKEN": "custom-secret",
    }

    manifest = yaml.safe_load(render_launch_secret("paper-r1", secrets))

    assert manifest["metadata"]["name"] == "senpai-launch-secrets-paper-r1"
    assert {
        name: base64.b64decode(value).decode()
        for name, value in manifest["data"].items()
    } == secrets


def test_scout_workflow_is_filled_from_the_same_secret_mapping():
    secrets = {
        "GITHUB_TOKEN": "github-secret",
        "SLACK_WEBHOOK_URL": "https://hooks.example/secret",
        "WANDB_API_KEY": "wandb-secret",
        "KUBECONFIG_B64": "kube-secret",
    }

    manifest = yaml.safe_load(render_workflow(secrets))

    assert manifest["spec"]["environment"]["secrets"] == {
        **secrets,
        "SEMANTIC_SCHOLAR_API_KEY": "",
    }
