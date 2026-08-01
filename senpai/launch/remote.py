# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Start the Docker backend from a one-use AWS launch payload."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from senpai.launch.docker_backend import launch_docker, preflight_docker
from senpai.launch.specs import RoleSpec


def run_from_payload(action: str, path: Path) -> None:
    """Consume a private payload, then preflight or start its roles."""
    payload = json.loads(path.read_text())
    path.unlink()

    args = SimpleNamespace(**payload["args"])
    role_specs = [RoleSpec(**values) for values in payload["roles"]]
    plan = preflight_docker(args, role_specs)
    if action == "launch":
        launch_docker(args, role_specs, plan, show_lifecycle=False)
    elif action != "preflight":
        raise ValueError(f"unsupported remote action: {action}")


def launch_from_payload(path: Path) -> None:
    """Backward-compatible entry point for a one-shot role launch."""
    run_from_payload("launch", path)


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in {"preflight", "launch"}:
        sys.exit("usage: python -m senpai.launch.remote {preflight,launch} PAYLOAD")
    run_from_payload(sys.argv[1], Path(sys.argv[2]))


if __name__ == "__main__":
    main()
