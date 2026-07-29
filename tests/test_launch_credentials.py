# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import base64
import stat
from unittest.mock import patch

import pytest
import yaml

from scripts.apply_scout_workflow import render_workflow
from senpai.launch.cli import Args
from senpai.launch.credentials import (
    SCOUT_SECRET_NAMES,
    WORKLOAD_REQUIRED_SECRET_NAMES,
    load_scout_secrets,
    load_workload_secrets,
    workload_secret_names,
)
from senpai.launch.kubernetes_backend import render_launch_secret
from senpai.launch.preflight import preflight_check_wandb_api_key
from senpai.launch.specs import (
    RoleSpec,
    build_student_env,
    validate_secret_config_separation,
)


def _dotenv(**overrides: str) -> str:
    values = {
        name: f"value-for-{name.lower()}" for name in WORKLOAD_REQUIRED_SECRET_NAMES
    }
    values.update(overrides)
    return "\n".join(f'{name}="{value}"' for name, value in values.items()) + "\n"


def test_dotenv_is_the_complete_secret_source(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(_dotenv(CUSTOM_SERVICE_TOKEN="token#with#hashes"))
    path.chmod(0o644)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-shell-token")

    secrets = load_workload_secrets(path)

    assert secrets["GITHUB_TOKEN"] == "value-for-github_token"
    assert secrets["CUSTOM_SERVICE_TOKEN"] == "token#with#hashes"
    assert set(WORKLOAD_REQUIRED_SECRET_NAMES) < set(secrets)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_dotenv_reports_every_missing_required_secret(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GITHUB_TOKEN=github\n")

    with pytest.raises(SystemExit) as error:
        load_workload_secrets(path)

    message = str(error.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "EXA_API_KEY" in message
    assert "WANDB_API_KEY" in message


def test_scout_only_secrets_are_not_sent_to_workers(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        _dotenv(
            CUSTOM_SERVICE_TOKEN="custom-secret",
            SLACK_WEBHOOK_URL="slack-secret",
            KUBECONFIG_B64="cluster-secret",
            SEMANTIC_SCHOLAR_API_KEY="search-secret",
        )
    )

    secrets = load_workload_secrets(path)

    assert secrets["CUSTOM_SERVICE_TOKEN"] == "custom-secret"
    assert (
        not set(SCOUT_SECRET_NAMES)
        .difference(WORKLOAD_REQUIRED_SECRET_NAMES)
        .intersection(secrets)
    )


def test_scout_receives_only_its_declared_secrets(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        _dotenv(
            SLACK_WEBHOOK_URL="slack-secret",
            KUBECONFIG_B64="cluster-secret",
            SEMANTIC_SCHOLAR_API_KEY="search-secret",
            CUSTOM_SERVICE_TOKEN="workload-only",
        )
    )

    secrets = load_scout_secrets(path)

    assert tuple(secrets) == SCOUT_SECRET_NAMES
    assert "CUSTOM_SERVICE_TOKEN" not in secrets
    assert "ANTHROPIC_API_KEY" not in secrets
    assert "EXA_API_KEY" not in secrets


def test_dry_run_secret_names_work_without_dotenv(tmp_path):
    assert workload_secret_names(tmp_path / ".env") == WORKLOAD_REQUIRED_SECRET_NAMES


def test_wandb_entity_is_omitted_until_explicitly_resolved():
    args = Args(
        tag="paper-r1",
        target_repo_url="https://github.com/wandb/target.git",
    )

    assert "WANDB_ENTITY" not in build_student_env(args, args.tag, "fern")

    args.wandb_entity = "research-team"
    assert build_student_env(args, args.tag, "fern")["WANDB_ENTITY"] == "research-team"


def test_wandb_preflight_uses_the_api_key_default_entity():
    with patch("senpai.launch.preflight.wandb.Api") as api:
        api.return_value.default_entity = "default-user"

        entity = preflight_check_wandb_api_key("wandb-secret", None)

    assert entity == "default-user"


def test_dotenv_cannot_override_runtime_settings():
    roles = [
        RoleSpec(
            "student",
            "fern",
            {"WANDB_PROJECT": "senpai-v1", "REPO_URL": "https://example.com"},
        )
    ]

    with pytest.raises(SystemExit) as error:
        validate_secret_config_separation(
            roles,
            {"GITHUB_TOKEN": "secret", "WANDB_PROJECT": "wrong-project"},
        )

    assert "WANDB_PROJECT" in str(error.value)
    assert "senpai.yaml or launch arguments" in str(error.value)


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
        name: secrets.get(name, "") for name in SCOUT_SECRET_NAMES
    }
