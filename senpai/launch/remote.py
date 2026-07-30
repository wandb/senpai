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


def launch_from_payload(path: Path) -> None:
    """Consume a private launch payload, then start its roles."""
    payload = json.loads(path.read_text())
    path.unlink()

    args = SimpleNamespace(**payload["args"])
    role_specs = [RoleSpec(**values) for values in payload["roles"]]
    plan = preflight_docker(args, role_specs)
    launch_docker(args, role_specs, plan)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m senpai.launch.remote PAYLOAD")
    launch_from_payload(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
