#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Apply the CoreWeave scout Workflow with secrets sourced from .env."""

# ruff: noqa: E402 -- direct script execution needs the repository root on sys.path.

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from senpai.launch.credentials import load_secrets

TEMPLATE = ROOT / "k8s" / "senpai-scout-workflow.yaml"
REQUIRED_SCOUT_SECRETS = (
    "GITHUB_TOKEN",
    "SLACK_WEBHOOK_URL",
    "WANDB_API_KEY",
    "KUBECONFIG_B64",
)


def render_workflow(secrets: dict[str, str]) -> str:
    missing = [name for name in REQUIRED_SCOUT_SECRETS if not secrets.get(name)]
    if missing:
        raise SystemExit(f"ERROR: .env is missing scout secrets: {', '.join(missing)}")

    manifest = yaml.safe_load(TEMPLATE.read_text())
    workflow_secrets = manifest["spec"]["environment"]["secrets"]
    for name in workflow_secrets:
        workflow_secrets[name] = secrets.get(name, "")
    return yaml.safe_dump(manifest, sort_keys=False)


def main() -> None:
    manifest = render_workflow(
        load_secrets(ROOT / ".env", required=REQUIRED_SCOUT_SECRETS)
    )
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        check=True,
    )
    print("Applied senpai-scout-daily with secrets from .env.")


if __name__ == "__main__":
    main()
