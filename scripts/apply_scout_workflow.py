#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Apply the CoreWeave scout Workflow with secrets sourced from .env."""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from senpai.launch.credentials import SCOUT_SECRET_NAMES, load_scout_secrets

TEMPLATE = ROOT / "k8s" / "senpai-scout-workflow.yaml"


def render_workflow(secrets: dict[str, str]) -> str:
    manifest = yaml.safe_load(TEMPLATE.read_text())
    manifest["spec"]["environment"]["secrets"] = {
        name: secrets.get(name, "") for name in SCOUT_SECRET_NAMES
    }
    return yaml.safe_dump(manifest, sort_keys=False)


def main() -> None:
    manifest = render_workflow(load_scout_secrets(ROOT / ".env"))
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        check=True,
    )
    print("Applied senpai-scout-daily with secrets from .env.")


if __name__ == "__main__":
    main()
